"""Strict CRVD test selection and explicit RAW conversions.

These are documented evaluation choices, not recovered MSTMN source code.
CRVD is GBRG; the packed array has shape (H/2, W/2, 4), ordered R,G,B,G.
The first green comes from the top-left sensor position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import tifffile


TEST_SCENES = (7, 8, 9, 10, 11)
ISO_VALUES = (1600, 3200, 6400, 12800, 25600)
FRAME_NUMBERS = tuple(range(1, 8))
NOISY_INDICES = tuple(range(10))


@dataclass(frozen=True, order=True)
class SequenceKey:
    scene: int
    iso: int
    noisy_index: int


@dataclass(frozen=True)
class FramePair:
    frame: int
    noisy: Path
    gt: Path


@dataclass(frozen=True)
class SequenceRecord:
    key: SequenceKey
    frames: tuple[FramePair, ...]


def _selection(values: Iterable[int], allowed: tuple[int, ...], name: str) -> tuple[int, ...]:
    """Require an explicit, unambiguous selection and canonical numeric order."""
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an explicit nonempty list of integers")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an explicit nonempty list of integers") from error
    if not items or any(type(value) is not int for value in items):
        raise ValueError(f"{name} must be an explicit nonempty list of integers")
    if len(items) != len(set(items)):
        raise ValueError(f"{name} contains duplicate values")
    unexpected = sorted(set(items).difference(allowed))
    if unexpected:
        raise ValueError(f"Unsupported {name}: {unexpected}; allowed: {list(allowed)}")
    return tuple(sorted(items))


def _index_files(directory: Path, pattern: re.Pattern[str]) -> dict[tuple[int, ...], Path]:
    """Parse numeric IDs rather than relying on filesystem/glob ordering."""
    result: dict[tuple[int, ...], Path] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.iterdir(), key=lambda entry: entry.name.casefold()):
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        identity = tuple(int(part) for part in match.groups())
        if identity in result:
            raise ValueError(
                f"Ambiguous CRVD file IDs in {directory}: "
                f"{result[identity].name!r} and {path.name!r}"
            )
        result[identity] = path
    return result


def build_manifest(
    data_root: str | Path,
    scenes: Iterable[int],
    isos: Iterable[int],
    noisy_indices: Iterable[int],
) -> list[SequenceRecord]:
    """Select complete seven-frame CRVD test sequences or fail before inference.

    ``data_root`` contains ``indoor_raw_gt`` and ``indoor_raw_noisy``. Noisy
    realization IDs must be supplied explicitly; the available release uses
    IDs 0 through 9. ``frameN_clean.tiff`` in the noisy collection is never
    selected as an input or as ground truth. Extra unselected files are ignored.
    Every returned path is absolute; serialize relative paths against data_root
    when writing a portable manifest.
    """
    selected_scenes = _selection(scenes, TEST_SCENES, "scenes")
    selected_isos = _selection(isos, ISO_VALUES, "isos")
    selected_indices = _selection(noisy_indices, NOISY_INDICES, "noisy_indices")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CRVD data root does not exist: {root}")
    gt_pattern = re.compile(r"frame([0-9]+)_clean_and_slightly_denoised\.tiff", re.IGNORECASE)
    noisy_pattern = re.compile(r"frame([0-9]+)_noisy([0-9]+)\.tiff", re.IGNORECASE)
    missing: list[Path] = []
    sequences: list[SequenceRecord] = []
    for scene in selected_scenes:
        for iso in selected_isos:
            suffix = Path(f"scene{scene}") / f"ISO{iso}"
            gt_dir = root / "indoor_raw_gt" / suffix
            noisy_dir = root / "indoor_raw_noisy" / suffix
            gt_files = _index_files(gt_dir, gt_pattern)
            noisy_files = _index_files(noisy_dir, noisy_pattern)
            for frame in FRAME_NUMBERS:
                if (frame,) not in gt_files:
                    missing.append(gt_dir / f"frame{frame}_clean_and_slightly_denoised.tiff")
            for noisy_index in selected_indices:
                pairs: list[FramePair] = []
                for frame in FRAME_NUMBERS:
                    noisy = noisy_files.get((frame, noisy_index))
                    gt = gt_files.get((frame,))
                    if noisy is None:
                        missing.append(noisy_dir / f"frame{frame}_noisy{noisy_index}.tiff")
                    if noisy is not None and gt is not None:
                        pairs.append(FramePair(frame, noisy, gt))
                sequences.append(SequenceRecord(SequenceKey(scene, iso, noisy_index), tuple(pairs)))
    if missing:
        preview = "\n".join(f"  {path.relative_to(root).as_posix()}" for path in missing[:20])
        remaining = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Incomplete CRVD selection: {len(missing)} missing file(s)\n{preview}{remaining}")
    return sequences


def _real_array(values: np.ndarray, description: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "uif":
        raise ValueError(f"{description} requires real integer or floating point values, got {array.dtype}")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{description} must be nonempty and finite")
    return array


def read_raw(path: str | Path) -> np.ndarray:
    """Read one 2D real TIFF in original sensor DN units as float32.

    Averaged CRVD ground truth can be a floating point TIFF. No normalization,
    clipping, quantization, demosaicing, or cropping is performed here.
    """
    raw = _real_array(tifffile.imread(path), f"RAW TIFF {path}")
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D Bayer TIFF, got shape {raw.shape}: {path}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"RAW TIFF values cannot be represented as finite float32: {path}")
    return result


def normalize_raw(
    raw: np.ndarray,
    black_level: float = 240.0,
    white_level: float = 4095.0,
    clip: bool = False,
) -> np.ndarray:
    """Apply (DN-black)/(white-black); clipping is an explicit protocol choice."""
    if not np.isfinite(black_level) or not np.isfinite(white_level) or white_level <= black_level:
        raise ValueError("Require finite white_level > black_level")
    if not isinstance(clip, bool):
        raise ValueError("clip must be a boolean")
    array = _real_array(raw, "RAW normalization")
    with np.errstate(over="ignore", invalid="ignore"):
        result = (array.astype(np.float32) - black_level) / (white_level - black_level)
    if not np.isfinite(result).all():
        raise ValueError("Normalized RAW values must be finite float32")
    result = result.astype(np.float32, copy=False)
    return np.clip(result, 0.0, 1.0) if clip else result


def pack_gbrg(raw: np.ndarray) -> np.ndarray:
    """Pack a GBRG mosaic into HWC RGBG without interpolation or value changes."""
    array = _real_array(raw, "GBRG packing")
    if array.ndim != 2 or any(dimension % 2 for dimension in array.shape):
        raise ValueError(f"GBRG packing requires a 2D array with even H/W, got {array.shape}")
    return np.stack(
        (array[1::2, 0::2], array[0::2, 0::2], array[0::2, 1::2], array[1::2, 1::2]),
        axis=-1,
    )


def unpack_gbrg(packed: np.ndarray) -> np.ndarray:
    """Invert :func:`pack_gbrg` exactly, preserving dtype and both green planes."""
    array = _real_array(packed, "GBRG unpacking")
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"GBRG unpacking requires (H,W,4), got {array.shape}")
    height, width = array.shape[:2]
    raw = np.empty((height * 2, width * 2), dtype=array.dtype)
    raw[1::2, 0::2] = array[..., 0]
    raw[0::2, 0::2] = array[..., 1]
    raw[0::2, 1::2] = array[..., 2]
    raw[1::2, 1::2] = array[..., 3]
    return raw
