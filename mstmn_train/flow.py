"""SPyNet compatibility implementation and an explicit RAW-to-RGB bridge.

Architecture/weight conventions adapted from XPixelGroup/BasicSR's
basicsr/archs/spynet_arch.py (Apache-2.0, see third_party/BasicSR-LICENSE.txt).
Modified: safe local loading, standalone warp, frozen inference, RAW bridge.
The selected Sintel checkpoint and RGB conversion are reproduction choices;
the MSTMN authors' flow checkpoint/preprocessing are unknown.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def warp(image, flow):
    b, _, h, w = image.shape
    y, x = torch.meshgrid(torch.arange(h, device=image.device, dtype=image.dtype),
                          torch.arange(w, device=image.device, dtype=image.dtype), indexing='ij')
    grid = torch.stack((x, y), dim=-1)[None] + flow.permute(0, 2, 3, 1)
    normalized = torch.stack((2 * grid[..., 0] / max(w - 1, 1) - 1,
                              2 * grid[..., 1] / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(image, normalized, mode='bilinear', padding_mode='border', align_corners=True)


class _FlowStage(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [8, 32, 64, 32, 16, 2]
        layers = []
        for index, (left, right) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.Conv2d(left, right, 7, padding=3))
            if index < 4:
                layers.append(nn.ReLU())
        self.basic_module = nn.Sequential(*layers)

    def forward(self, tensor):
        return self.basic_module(tensor)


class FrozenSPyNet(nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        self.basic_module = nn.ModuleList([_FlowStage() for _ in range(6)])
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        parameters = state['params']
        # BasicSR loads learned weights before registering normalization buffers.
        self.load_state_dict(parameters, strict=True)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, reference, support):
        h, w = reference.shape[-2:]
        resized = (max(32, (h + 31) // 32 * 32), max(32, (w + 31) // 32 * 32))
        ref = [(F.interpolate(reference, size=resized, mode='bilinear', align_corners=False) - self.mean) / self.std]
        supp = [(F.interpolate(support, size=resized, mode='bilinear', align_corners=False) - self.mean) / self.std]
        for _ in range(5):
            ref.insert(0, F.avg_pool2d(ref[0], 2))
            supp.insert(0, F.avg_pool2d(supp[0], 2))
        flow = ref[0].new_zeros(reference.shape[0], 2, *ref[0].shape[-2:])
        for level, stage in enumerate(self.basic_module):
            if level:
                previous_h, previous_w = flow.shape[-2:]
                flow = F.interpolate(flow, size=ref[level].shape[-2:], mode='bilinear', align_corners=True)
                flow = flow * flow.new_tensor([ref[level].shape[-1] / previous_w,
                                              ref[level].shape[-2] / previous_h])[None, :, None, None]
            flow = flow + stage(torch.cat((ref[level], warp(supp[level], flow), flow), dim=1))
        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
        return flow * flow.new_tensor([w / resized[1], h / resized[0]])[None, :, None, None]


class RawSPyNet(nn.Module):
    """16-channel unshuffled Gaussian RAW -> displacement in the same grid."""

    def __init__(self, checkpoint):
        super().__init__()
        self.network = FrozenSPyNet(checkpoint)

    @staticmethod
    def rgb(tensor):
        packed = F.pixel_shuffle(tensor, 2)
        return torch.stack((packed[:, 0], (packed[:, 1] + packed[:, 3]) * 0.5, packed[:, 2]), dim=1).clamp(0, 1)

    @torch.no_grad()
    def forward(self, reference, support):
        flow = self.network(self.rgb(reference), self.rgb(support))
        return F.interpolate(flow, size=reference.shape[-2:], mode='bilinear', align_corners=False) / 2
