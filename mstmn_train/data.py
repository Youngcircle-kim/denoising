"""Deterministic CRVD training samples for an explicitly assumed recipe.

MSTMN states 25-frame sequences and packed 128x128 crops but its official
sampling code is unavailable. This module uses random-phase/direction ping-pong
extension of the seven captured frames, without duplicating turning endpoints.
That extension and the synthetic-first choice are reproduction assumptions.

Sensor/noise evidence (CRVD author code, not MSTMN code):
https://github.com/cao-cong/RViDeNet/blob/master/synthesize_noise.py#L8-L23
https://github.com/cao-cong/RViDeNet/blob/master/synthesize_noise.py#L35-L39
The coefficient a is in DN, b is a variance in DN squared. Synthetic inputs use
scaled Poisson plus Gaussian noise, clipping in original DN, and optional uint16
truncation matching that script's saved files. Negative GT-black values are
clamped only for the Poisson rate; floating point targets remain unchanged.

No torch import is needed: torch DataLoader supports this map-style protocol.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from denoising_eval.data import ISO_VALUES, normalize_raw, pack_gbrg, read_raw


TRAIN_SCENES = (1, 2, 3, 4, 5, 6)
NOISE_COEFFICIENTS_DN = {
    1600: (3.513262, 11.917691),
    3200: (6.955588, 38.117816),
    6400: (13.486051, 130.818508),
    12800: (26.585953, 484.539790),
    25600: (52.032536, 1819.818657),
}


def _integers(values: Iterable[int], allowed: Iterable[int], name: str) -> tuple[int, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty explicit list of integers")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty explicit list of integers") from error
    if not items or any(type(value) is not int for value in items):
        raise ValueError(f"{name} must be a nonempty explicit list of integers")
    if len(items) != len(set(items)) or not set(items).issubset(allowed):
        raise ValueError(f"Invalid/duplicate {name}: {items}; allowed: {tuple(allowed)}")
    return tuple(sorted(items))


def _nonnegative_integer(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def pingpong_frames(length: int, phase: int = 0, direction: int = 1) -> np.ndarray:
    """Return 1-based IDs from the period 1,2,3,4,5,6,7,6,5,4,3,2.

    This is a declared extension policy, not a recovered paper setting.
    """
    _nonnegative_integer(length, "length", 1)
    if type(phase) is not int or not 0 <= phase < 12 or direction not in (-1, 1):
        raise ValueError("phase must be 0..11 and direction must be -1 or 1")
    positions = (phase + direction * np.arange(length, dtype=np.int64)) % 12
    return np.where(positions <= 6, positions + 1, 13 - positions)


def synthesize_noise(
    target: np.ndarray,
    iso: int,
    rng: np.random.Generator,
    *,
    black_level: float = 240.0,
    white_level: float = 4095.0,
    clip_inputs: bool = True,
    quantize: bool = True,
) -> np.ndarray:
    """Create noisy normalized RGBG from normalized clean RGBG.

    ``clip_inputs`` means DN clipping to [0, white_level], not normalized [0,1].
    Therefore values below black level are retained as negative normalized DN.
    ``quantize=True`` truncates clipped DN as the official CRVD writer does.
    Quantization without clipping is refused to avoid integer wraparound.
    """
    if iso not in NOISE_COEFFICIENTS_DN:
        raise ValueError(f"Unsupported CRVD ISO: {iso}")
    if not np.isfinite(black_level) or not np.isfinite(white_level) or not 0 <= black_level < white_level:
        raise ValueError("Require finite 0 <= black_level < white_level")
    if type(clip_inputs) is not bool or type(quantize) is not bool:
        raise ValueError("clip_inputs and quantize must be booleans")
    if quantize and (not clip_inputs or white_level > np.iinfo(np.uint16).max):
        raise ValueError("uint16 quantization requires DN clipping and white_level <= 65535")
    clean = np.asarray(target)
    if clean.dtype.kind not in "uif" or not clean.size or not np.isfinite(clean).all():
        raise ValueError("Noise synthesis requires nonempty finite real target values")
    scale = white_level - black_level
    a, b = NOISE_COEFFICIENTS_DN[iso]
    rate = np.maximum(clean.astype(np.float64), 0.0) * (scale / a)
    noisy_dn = rng.poisson(rate) * a + rng.normal(0.0, np.sqrt(b), clean.shape) + black_level
    if clip_inputs:
        noisy_dn = np.clip(noisy_dn, 0.0, white_level)
    if quantize:
        noisy_dn = noisy_dn.astype(np.uint16)
    return normalize_raw(noisy_dn, black_level, white_level, clip=False)


@dataclass(frozen=True)
class TrainingSequence:
    scene: int
    iso: int
    gt: tuple[Path, ...]
    noisy: dict[int, tuple[Path, ...]]


def _matching_files(directory: Path, pattern: str) -> dict[tuple[int, ...], Path]:
    files: dict[tuple[int, ...], Path] = {}
    if not directory.is_dir():
        return files
    regex = re.compile(pattern, re.IGNORECASE)
    for path in directory.iterdir():
        match = regex.fullmatch(path.name)
        if match is not None and path.is_file():
            identity = tuple(int(value) for value in match.groups())
            if identity in files:
                raise ValueError(f"Ambiguous file IDs: {files[identity]} and {path}")
            files[identity] = path
    return files


class CRVDTrainingDataset:
    """Map sample IDs to independent deterministic scene-1..6 training crops.

    Every sample's RNG depends solely on ``seed`` and ``start_index + index``;
    worker count, prefetch, cache state and call ordering cannot change a sample.
    For resume, use DataLoader ``shuffle=False`` and set ``start_index`` to the
    number of training samples already consumed (not prefetched). This dataset
    has no mutable epoch or global NumPy RNG state. ``samples`` is its length.

    A packed crop has shape 128x128 by default (256x256 original Bayer). Crop
    origins are even in packed coordinates, i.e. multiples of four in the
    original sensor, preserving CFA and pixel-unshuffle phase. No spatial
    augmentation or temporal interpolation is applied. Real mode selects one
    measured realization ID for the whole sequence; repeated frames reuse it.
    Synthetic mode draws independent noise at each time step, including repeats.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        scenes: Iterable[int] = TRAIN_SCENES,
        isos: Iterable[int] = ISO_VALUES,
        mode: str = "synthetic",
        sequence_length: int = 25,
        patch_size: int = 128,
        samples: int = 1_920_000,
        seed: int = 0,
        start_index: int = 0,
        noisy_indices: Iterable[int] = tuple(range(10)),
        clip_inputs: bool = True,
        quantize_synthetic: bool = True,
        black_level: float = 240.0,
        white_level: float = 4095.0,
        cache_sequences: int = 2,
    ):
        self.scenes = _integers(scenes, TRAIN_SCENES, "training scenes (test scenes forbidden)")
        self.isos = _integers(isos, ISO_VALUES, "isos")
        self.noisy_indices = _integers(noisy_indices, range(10), "noisy_indices")
        if mode not in ("synthetic", "real"):
            raise ValueError("mode must be 'synthetic' or 'real'")
        self.mode = mode
        self.sequence_length = _nonnegative_integer(sequence_length, "sequence_length", 1)
        self.patch_size = _nonnegative_integer(patch_size, "patch_size", 2)
        if patch_size % 2:
            raise ValueError("Packed patch_size must be even for pixel-unshuffle")
        self.samples = _nonnegative_integer(samples, "samples", 1)
        self.seed = _nonnegative_integer(seed, "seed")
        self.start_index = _nonnegative_integer(start_index, "start_index")
        self.cache_sequences = _nonnegative_integer(cache_sequences, "cache_sequences")
        if type(clip_inputs) is not bool or type(quantize_synthetic) is not bool:
            raise ValueError("clip_inputs and quantize_synthetic must be booleans")
        if quantize_synthetic and not clip_inputs and mode == "synthetic":
            raise ValueError("Synthetic uint16 quantization requires clip_inputs=True")
        if not np.isfinite(black_level) or not np.isfinite(white_level) or not 0 <= black_level < white_level:
            raise ValueError("Require finite 0 <= black_level < white_level")
        if mode == "synthetic" and quantize_synthetic and white_level > 65535:
            raise ValueError("Synthetic uint16 quantization requires white_level <= 65535")
        self.clip_inputs = clip_inputs
        self.quantize_synthetic = quantize_synthetic
        self.black_level, self.white_level = float(black_level), float(white_level)
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"CRVD root does not exist: {self.root}")
        self.sequences = self._manifest()
        self._cache: OrderedDict[tuple[int, int], tuple[np.ndarray, ...]] = OrderedDict()

    def _manifest(self) -> tuple[TrainingSequence, ...]:
        records: list[TrainingSequence] = []
        missing: list[Path] = []
        for scene in self.scenes:
            for iso in self.isos:
                suffix = Path(f"scene{scene}") / f"ISO{iso}"
                gt_dir = self.root / "indoor_raw_gt" / suffix
                gt_lookup = _matching_files(gt_dir, r"frame([0-9]+)_clean_and_slightly_denoised\.tiff")
                gt_paths = []
                for frame in range(1, 8):
                    path = gt_lookup.get((frame,), gt_dir / f"frame{frame}_clean_and_slightly_denoised.tiff")
                    gt_paths.append(path)
                    if (frame,) not in gt_lookup:
                        missing.append(path)
                noisy_paths: dict[int, tuple[Path, ...]] = {}
                if self.mode == "real":
                    noisy_dir = self.root / "indoor_raw_noisy" / suffix
                    noisy_lookup = _matching_files(noisy_dir, r"frame([0-9]+)_noisy([0-9]+)\.tiff")
                    for index in self.noisy_indices:
                        paths = []
                        for frame in range(1, 8):
                            path = noisy_lookup.get((frame, index), noisy_dir / f"frame{frame}_noisy{index}.tiff")
                            paths.append(path)
                            if (frame, index) not in noisy_lookup:
                                missing.append(path)
                        noisy_paths[index] = tuple(paths)
                records.append(TrainingSequence(scene, iso, tuple(gt_paths), noisy_paths))
        if missing:
            preview = "\n".join(path.relative_to(self.root).as_posix() for path in missing[:20])
            raise FileNotFoundError(f"Incomplete training selection: {len(missing)} missing files\n{preview}")
        return tuple(records)

    def __len__(self) -> int:
        return self.samples

    def _clean_frames(self, sequence: TrainingSequence) -> tuple[np.ndarray, ...]:
        key = (sequence.scene, sequence.iso)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        frames = tuple(pack_gbrg(normalize_raw(read_raw(path), self.black_level, self.white_level, clip=False)) for path in sequence.gt)
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames):
            raise ValueError(f"Inconsistent GT frame sizes: scene{sequence.scene}/ISO{sequence.iso}")
        if min(shape[:2]) < self.patch_size:
            raise ValueError(f"Packed crop {self.patch_size} exceeds GT shape {shape}")
        if self.cache_sequences:
            self._cache[key] = frames
            while len(self._cache) > self.cache_sequences:
                self._cache.popitem(last=False)
        return frames

    def __getitem__(self, index: int) -> dict[str, object]:
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)) or not 0 <= index < self.samples:
            raise IndexError(index)
        sample_index = self.start_index + int(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, sample_index]))
        sequence = self.sequences[int(rng.integers(len(self.sequences)))]
        clean_frames = self._clean_frames(sequence)
        height, width = clean_frames[0].shape[:2]
        top = 2 * int(rng.integers((height - self.patch_size) // 2 + 1))
        left = 2 * int(rng.integers((width - self.patch_size) // 2 + 1))
        frame_numbers = pingpong_frames(self.sequence_length, int(rng.integers(12)), 1 if rng.integers(2) else -1)
        crop = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))
        target = np.stack([clean_frames[int(frame) - 1][crop] for frame in frame_numbers])
        noisy_index = -1
        if self.mode == "synthetic":
            noisy = synthesize_noise(target, sequence.iso, rng, black_level=self.black_level,
                                     white_level=self.white_level, clip_inputs=self.clip_inputs,
                                     quantize=self.quantize_synthetic)
        else:
            noisy_index = self.noisy_indices[int(rng.integers(len(self.noisy_indices)))]
            # Decode each unique captured frame once even in a 25-step window.
            measured: dict[int, np.ndarray] = {}
            for frame in set(int(value) for value in frame_numbers):
                raw = read_raw(sequence.noisy[noisy_index][frame - 1])
                if raw.shape != (height * 2, width * 2):
                    raise ValueError(f"Noisy/GT shape mismatch: {sequence.noisy[noisy_index][frame - 1]}")
                if self.clip_inputs:
                    raw = np.clip(raw, 0.0, self.white_level)
                measured[frame] = pack_gbrg(normalize_raw(raw, self.black_level, self.white_level, clip=False))[crop]
            noisy = np.stack([measured[int(frame)] for frame in frame_numbers])
        scale = self.white_level - self.black_level
        a, b = NOISE_COEFFICIENTS_DN[sequence.iso]
        return {
            "input": np.ascontiguousarray(np.moveaxis(noisy, -1, 1), dtype=np.float32),
            "target": np.ascontiguousarray(np.moveaxis(target, -1, 1), dtype=np.float32),
            "shot": np.float32(a / scale), "read": np.float32(b / scale**2),
            "scene": sequence.scene, "iso": sequence.iso, "sample_index": sample_index,
            "frame_numbers": frame_numbers, "crop_yx": np.array([top, left], dtype=np.int64),
            "noisy_index": noisy_index,
        }
