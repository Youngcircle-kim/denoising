"""Unofficial paper-based MSTMN; no official weights or code are implied.

Paper-derived components and reconstruction assumptions are documented in
research/MSTMN_implementation_spec.md. In particular, patch-token retrieval,
Gaussian coefficients, sampling/warping conventions and activations are explicit
implementation choices. Flow must be provided; zero-flow is never a default.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GaussianPyramid(nn.Module):
    """Four 4-channel Gaussian levels and three Laplacian levels (Eq.11/12)."""

    def __init__(self) -> None:
        super().__init__()
        coefficients = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16.0
        self.register_buffer("kernel", torch.outer(coefficients, coefficients)[None, None])

    def blur(self, image: Tensor) -> Tensor:
        if min(image.shape[-2:]) <= 2:
            raise ValueError("Gaussian filtering requires spatial dimensions above 2.")
        kernel = self.kernel.to(dtype=image.dtype).expand(image.shape[1], 1, 5, 5)
        return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"),
                        kernel, groups=image.shape[1])

    def expand(self, image: Tensor, size: tuple[int, int]) -> Tensor:
        return self.blur(F.interpolate(image, size=size, mode="bilinear", align_corners=False))

    def forward(self, image: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        gaussians = [image]
        for _ in range(3):
            gaussians.append(self.blur(gaussians[-1])[..., ::2, ::2])
        laplacians = [gaussians[level] - self.expand(gaussians[level + 1], gaussians[level].shape[-2:])
                      for level in range(3)]
        return gaussians, laplacians

    def collapse(self, laplacians: list[Tensor], base: Tensor) -> Tensor:
        if len(laplacians) != 3:
            raise ValueError("MSTMN collapse requires three Laplacian levels.")
        result = base
        for laplacian in reversed(laplacians):
            result = self.expand(result, laplacian.shape[-2:]) + laplacian
        return result


class PatchMemory(nn.Module):
    """Full sigmoid patch-feature attention at the two smaller scales.

    This is an explicit dimensional repair of the paper's ambiguous Eq.14,
    preserving its stated merge of memory frames and channels: Q=[B,N,D],
    K/V=[B,N,T*D], A=sigmoid(Q.T@K), retrieved=(A@V.T).T. D=16*p*p.
    The outer product order is repaired, not verified against official code.
    No softmax, sqrt(D), top-k or row-sum normalization is added. Chunking is
    along D and keeps the same attention. Inverse-fold overlap division only
    corrects duplicate spatial coverage, not the attention map.
    """

    def __init__(self, patch_size: int = 4, stride: int = 4,
                 query_chunk_size: int = 128) -> None:
        super().__init__()
        for name, value in (("patch_size", patch_size), ("stride", stride),
                            ("query_chunk_size", query_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if stride > patch_size:
            raise ValueError("Patch stride cannot exceed patch size: that leaves uncovered pixels.")
        self.patch_size = patch_size
        self.stride = stride
        self.query_chunk_size = query_chunk_size

    def _padded_size(self, height: int, width: int) -> tuple[int, int]:
        def padded(length: int) -> int:
            return max(self.patch_size,
                       self.patch_size + math.ceil(max(length - self.patch_size, 0) / self.stride) * self.stride)
        return padded(height), padded(width)

    def patchify(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        padded_size = self._padded_size(*image.shape[-2:])
        padded = F.pad(image, (0, padded_size[1] - image.shape[-1],
                               0, padded_size[0] - image.shape[-2]), mode="replicate")
        return F.unfold(padded, self.patch_size, stride=self.stride).transpose(1, 2), padded_size

    def unpatchify(self, tokens: Tensor, padded_size: tuple[int, int],
                   image_size: tuple[int, int]) -> Tensor:
        columns = tokens.transpose(1, 2)
        reconstructed = F.fold(columns, padded_size, self.patch_size, stride=self.stride)
        # Divisor depends only on spatial coverage, not image channels/batch or
        # attention. Building it on-device also supports arbitrary overlaps.
        divisor_columns = torch.ones((1, self.patch_size ** 2, tokens.shape[1]),
                                     device=tokens.device, dtype=tokens.dtype)
        divisor = F.fold(divisor_columns, padded_size, self.patch_size, stride=self.stride)
        result = reconstructed / divisor.clamp_min(1)
        return result[..., :image_size[0], :image_size[1]]

    def forward(self, query_key: Tensor, memory_keys: Tensor, memory_values: Tensor) -> Tensor:
        if memory_keys.ndim != 5 or memory_values.shape != memory_keys.shape:
            raise ValueError("Memory keys/values must have the same [B,T,C,H,W] shape.")
        batch, frames, channels, height, width = memory_keys.shape
        if frames < 1 or query_key.shape != (batch, channels, height, width):
            raise ValueError("Memory and query channel/spatial shapes must match.")
        query, padded_size = self.patchify(query_key)
        keys, _ = self.patchify(memory_keys.reshape(batch * frames, channels, height, width))
        values, _ = self.patchify(memory_values.reshape(batch * frames, channels, height, width))
        patch_count, features = keys.shape[1:]
        # tau merges T with patch-feature channels D; it does not merge T
        # with spatial patch count N. Thus the affinity matrix is D x (T*D),
        # independent of image patch count, while the dot product sums over N.
        key_features = keys.reshape(batch, frames, patch_count, features)
        key_features = key_features.permute(0, 2, 1, 3).reshape(batch, patch_count, frames * features)
        value_features = values.reshape(batch, frames, patch_count, features)
        value_features = value_features.permute(0, 2, 1, 3).reshape(batch, patch_count, frames * features)
        query_features = query.transpose(1, 2)
        chunks = []
        for start in range(0, features, self.query_chunk_size):
            affinity = torch.sigmoid(torch.bmm(query_features[:, start:start + self.query_chunk_size],
                                               key_features))
            chunks.append(torch.bmm(affinity, value_features.transpose(1, 2)))
        retrieved = torch.cat(chunks, dim=1).transpose(1, 2)
        return self.unpatchify(retrieved, padded_size, (height, width))


class DepthwiseSeparable(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=False),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class WeightPredictor(nn.Module):
    """1x1 -> DS3x3 -> DS3x3 -> 1x1; independent sigmoid W or V."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, 1), nn.ReLU(inplace=False),
            DepthwiseSeparable(), DepthwiseSeparable(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class ScaleModules(nn.Module):
    def __init__(self, has_coarser_weights: bool) -> None:
        super().__init__()
        self.memory_fusion = nn.Conv2d(32, 16, 3, padding=1)
        # The paper describes 3 conv/ReLU stages with 16 outputs each. The final
        # ReLU is retained literally and recorded as an unresolved choice.
        self.denoiser = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=False),
        )
        self.noise_weight = WeightPredictor(33 if has_coarser_weights else 32)
        self.alignment_weight = WeightPredictor(33 if has_coarser_weights else 32)


def resize_flow(flow: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize flow in pixel units, scaling x/y displacement with the grid."""
    height, width = flow.shape[-2:]
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    factors = flow.new_tensor([size[1] / width, size[0] / height]).view(1, 2, 1, 1)
    return resized * factors


def warp_previous(previous: Tensor, backward_flow: Tensor) -> Tensor:
    """Read previous at current+(dx,dy); pixel-center, bilinear, border padding."""
    if backward_flow.shape != (previous.shape[0], 2, *previous.shape[-2:]):
        raise ValueError("Flow must be [B,2,H,W] on the previous image grid.")
    height, width = previous.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=previous.device, dtype=previous.dtype),
                            torch.arange(width, device=previous.device, dtype=previous.dtype),
                            indexing="ij")
    x = xx[None] + backward_flow[:, 0]
    y = yy[None] + backward_flow[:, 1]
    x = 2 * x / max(width - 1, 1) - 1
    y = 2 * y / max(height - 1, 1) - 1
    grid = torch.stack((x, y), dim=-1)
    return F.grid_sample(previous, grid, mode="bilinear", padding_mode="border", align_corners=True)


def weighted_fusion(enhanced: Tensor, previous: Tensor, denoised: Tensor,
                    noise_weight: Tensor, alignment_weight: Tensor) -> Tensor:
    """Eq.(16c), including a possibly negative (1-W-V) coefficient."""
    return (noise_weight * enhanced + alignment_weight * previous
            + (1 - noise_weight - alignment_weight) * denoised)


class MSTMN(nn.Module):
    """Unofficial three-scale recurrent model with an explicitly injected flow.

    ``shot`` is a/R and ``read`` is b/R**2, both variance-model coefficients
    for normalized packed RAW. ``read`` is already a variance, not its square
    root. The caller initializes previous/history and controls BPTT boundaries.
    """

    def __init__(self, flow_model: nn.Module, memory_frames: int = 3,
                 patch_size: int = 4, patch_stride: int = 4,
                 query_chunk_size: int = 128) -> None:
        super().__init__()
        if not isinstance(flow_model, nn.Module):
            raise ValueError("An explicit optical-flow nn.Module is required; no identity fallback exists.")
        if isinstance(memory_frames, bool) or not isinstance(memory_frames, int) or memory_frames < 1:
            raise ValueError("memory_frames must be a positive integer.")
        self.flow_model = flow_model
        self.memory_frames = memory_frames
        self.pyramid = GaussianPyramid()
        self.memory = PatchMemory(patch_size, patch_stride, query_chunk_size)
        # Indices 0,1,2 are l=1,2,3, so l=3 has no incoming coarse weights.
        self.scales = nn.ModuleList([ScaleModules(True), ScaleModules(True), ScaleModules(False)])
        self.model_config: dict[str, Any] = {
            "name": "unofficial-paper-based-mstmn",
            "memory_frames": memory_frames, "patch_size": patch_size,
            "patch_stride": patch_stride, "query_chunk_size": query_chunk_size,
            "pyramid": "packed4-pyramid-then-pixel-unshuffle2",
            "gaussian_kernel": "binomial-5x5", "sampling_phase": "even-index",
            "gaussian_padding": "reflect", "input_padding": "replicate-to-packed-multiple16",
            "upsample": "bilinear-align_corners_false-then-gaussian",
            "memory_attention": "sigmoid-patch-feature-full-unscaled-Eq14-dimensional-repair",
            "memory_tensor_layout": "Q=B,N,D;K/V=B,N,T*D;A=Q.T@K;R=(A@V.T).T",
            "memory_chunk_axis": "patch-feature-D",
            "memory_concat": "conv3x3-32to16-linear",
            "denoiser": "three-conv3x3-16ch-relu-all-stages",
            "scale_parameters": "independent",
            "variance": "clamp_min(shot*Gaussian+read,0);read-already-variance",
            "weight_maps": "independent-sigmoid;no-sum-renormalization",
            "warp": "bilinear-border-align_corners_true-backward-pixel-units",
        }

    @staticmethod
    def _coefficient(value: Tensor, batch: int, reference: Tensor, name: str) -> Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.numel() != batch:
            raise ValueError(f"{name} must contain one coefficient per batch item.")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"{name} coefficients must be finite and nonnegative.")
        return value.reshape(batch, 1, 1, 1)

    def forward(self, noisy: Tensor, previous: Tensor, history: Tensor,
                shot: Tensor, read: Tensor) -> Tensor:
        if noisy.ndim != 4 or noisy.shape[1] != 4 or not noisy.is_floating_point():
            raise ValueError("noisy must be floating packed RAW [B,4,H,W].")
        if previous.shape != noisy.shape or previous.dtype != noisy.dtype or previous.device != noisy.device:
            raise ValueError("previous must have the same shape, dtype and device as noisy.")
        batch, _, height, width = noisy.shape
        if batch < 1 or min(height, width) < 1:
            raise ValueError("Input batch and spatial dimensions must be nonempty.")
        if (history.shape != (batch, self.memory_frames, 4, height, width)
                or history.dtype != noisy.dtype or history.device != noisy.device):
            raise ValueError("history must be matching [B,memory_frames,4,H,W] noisy frames.")
        shot = self._coefficient(shot, batch, noisy, "shot")
        read = self._coefficient(read, batch, noisy, "read")
        pad_h, pad_w = (-height) % 16, (-width) % 16
        padded_height, padded_width = height + pad_h, width + pad_w
        # At least packed16 is required by reflect filtering at all four levels.
        if min(padded_height, padded_width) < 16:
            raise ValueError("Padded packed input must have spatial dimensions at least 16.")
        noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode="replicate")
        previous = F.pad(previous, (0, pad_w, 0, pad_h), mode="replicate")
        memory_images = history.reshape(batch * self.memory_frames, 4, height, width)
        memory_images = F.pad(memory_images, (0, pad_w, 0, pad_h), mode="replicate")
        gaussians, laplacians = self.pyramid(noisy)
        previous_g, previous_l = self.pyramid(previous)
        memory_g, memory_l = self.pyramid(memory_images)
        g16 = [F.pixel_unshuffle(item, 2) for item in gaussians[:3]]
        l16 = [F.pixel_unshuffle(item, 2) for item in laplacians]
        previous_g16 = [F.pixel_unshuffle(item, 2) for item in previous_g[:3]]
        previous_l16 = [F.pixel_unshuffle(item, 2) for item in previous_l]
        flow = self.flow_model(g16[2], previous_g16[2])
        if not isinstance(flow, Tensor) or flow.shape != (batch, 2, *g16[2].shape[-2:]):
            raise ValueError("flow_model must return [B,2,Hcoarse,Wcoarse] backward pixel flow.")
        if flow.device != noisy.device or not bool(torch.isfinite(flow).all()):
            raise ValueError("flow_model must return finite flow on the input device.")
        flow = flow.to(dtype=noisy.dtype)
        restored: list[Tensor | None] = [None, None, None]
        coarse_w = coarse_v = retrieved = None
        for level in (2, 1, 0):
            gaussian, laplacian = g16[level], l16[level]
            level_flow = flow if level == 2 else resize_flow(flow, gaussian.shape[-2:])
            aligned_g = warp_previous(previous_g16[level], level_flow)
            aligned_l = warp_previous(previous_l16[level], level_flow)
            if level:
                key = F.pixel_unshuffle(memory_g[level], 2)
                value = F.pixel_unshuffle(memory_l[level], 2)
                key = key.reshape(batch, self.memory_frames, *key.shape[1:])
                value = value.reshape(batch, self.memory_frames, *value.shape[1:])
                retrieved = self.memory(gaussian, key, value)
            else:
                # Eq.15 only at l=1; full retrieval was performed at l=2.
                retrieved = F.interpolate(retrieved, size=gaussian.shape[-2:],
                                          mode="bilinear", align_corners=False)
            modules = self.scales[level]
            enhanced = modules.memory_fusion(torch.cat((laplacian, retrieved), dim=1))
            variance = F.pixel_unshuffle((shot * gaussians[level] + read).clamp_min(0), 2)
            denoised = modules.denoiser(torch.cat((enhanced, variance), dim=1))
            w_inputs = [denoised, enhanced]
            v_inputs = [gaussian, aligned_g]
            if coarse_w is not None:
                w_inputs.append(F.interpolate(coarse_w, size=gaussian.shape[-2:],
                                               mode="bilinear", align_corners=False))
                v_inputs.append(F.interpolate(coarse_v, size=gaussian.shape[-2:],
                                               mode="bilinear", align_corners=False))
            weight_w = modules.noise_weight(torch.cat(w_inputs, dim=1))
            weight_v = modules.alignment_weight(torch.cat(v_inputs, dim=1))
            result = weighted_fusion(enhanced, aligned_l, denoised, weight_w, weight_v)
            restored[level] = F.pixel_shuffle(result, 2)
            coarse_w, coarse_v = weight_w, weight_v
        output = self.pyramid.collapse(restored, gaussians[3])
        return output[..., :height, :width]
