"""Explicit RAW-domain metrics and aggregation for the evaluation harness.

These are harness choices, not a verified MSTMN evaluation protocol. Inputs are
two-dimensional Bayer mosaics already normalized using the selected black/white
levels. No clipping, demosaicing, border removal or quantization is performed
here. The metric range is fixed at 1 even when noisy/predicted values exceed it.

SSIM follows the Gaussian/population-covariance settings documented at
https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
with sigma=1.5 (an effective 11-pixel window), K1=0.01 and K2=0.03. This does
not establish that the paper used those settings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from numbers import Integral
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


METRIC_NAMES = ("mse", "psnr", "ssim")


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compare same-shaped finite 2D mosaics using a normalized range of 1.

    MSE and SSIM use float64 inputs. PSNR is ``-10*log10(MSE)``; a perfect
    prediction returns positive infinity. A JSON writer must encode that value
    explicitly instead of silently producing non-standard JSON or dropping it.
    Each side must contain at least 11 pixels for the fixed SSIM window.
    """
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have identical shapes.")
    if prediction.ndim != 2:
        raise ValueError("Metrics require a 2D RAW mosaic, not packed channels or RGB.")
    if min(prediction.shape) < 11:
        raise ValueError("Each image dimension must be at least 11 for Gaussian SSIM.")
    if (not np.issubdtype(prediction.dtype, np.number)
            or not np.issubdtype(ground_truth.dtype, np.number)
            or np.iscomplexobj(prediction) or np.iscomplexobj(ground_truth)):
        raise ValueError("Metric inputs must be real numeric arrays.")
    prediction = prediction.astype(np.float64, copy=False)
    ground_truth = ground_truth.astype(np.float64, copy=False)
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("Metric inputs must contain only finite values.")
    difference = prediction - ground_truth
    mse = float(np.mean(difference * difference, dtype=np.float64))
    if not math.isfinite(mse):
        raise ValueError("MSE overflowed; check the RAW normalization and input values.")
    psnr = math.inf if mse == 0 else -10.0 * math.log10(mse)
    ssim = float(structural_similarity(
        ground_truth,
        prediction,
        data_range=1.0,
        channel_axis=None,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        K1=0.01,
        K2=0.03,
    ))
    if not math.isfinite(ssim):
        raise ValueError("SSIM produced a non-finite value; check the input normalization.")
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def _mean(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {name: float(np.mean([record[name] for record in records], dtype=np.float64))
            for name in METRIC_NAMES}


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sequences: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        sequences[record["sequence_id"]].append(record)
    means = [_mean(group) for group in sequences.values()]
    return {
        "frame_count": len(records),
        "sequence_count": len(sequences),
        "frame_mean": _mean(records),
        "sequence_mean": _mean(means),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report explicit frame-equal and sequence-equal arithmetic means.

    Required keys are ``scene``, ``iso``, ``sequence_id``, ``frame`` and the
    three metric names. A sequence is one scene/ISO/noisy-realization temporal
    pass, as identified by its sequence_id. Frames within a sequence are first
    averaged, then sequence_mean gives each sequence equal weight. frame_mean
    instead gives every observed frame equal weight. Neither is declared to
    match the paper's unconfirmed aggregation procedure. PSNR values are
    averaged directly; PSNR is not recomputed from the aggregate MSE.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot summarize an empty evaluation.")
    identities: set[tuple[str, int]] = set()
    sequence_metadata: dict[str, tuple[int, int, Any]] = {}
    for record in records:
        for key in ("scene", "iso", "frame"):
            if (key not in record or isinstance(record[key], bool)
                    or not isinstance(record[key], Integral) or record[key] < 1):
                raise ValueError(f"Every record must have a positive integer {key}.")
        sequence_id = record.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("Every record must have a nonempty sequence_id string.")
        identity = (sequence_id, int(record["frame"]))
        if identity in identities:
            raise ValueError(f"Duplicate evaluated frame: {sequence_id}, frame {record['frame']}.")
        identities.add(identity)
        metadata = (int(record["scene"]), int(record["iso"]), record.get("noisy_index"))
        if sequence_id in sequence_metadata and sequence_metadata[sequence_id] != metadata:
            raise ValueError(f"Inconsistent scene/ISO/noisy_index for sequence {sequence_id}.")
        sequence_metadata[sequence_id] = metadata
        for key in METRIC_NAMES:
            try:
                value = float(record[key])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Every record must have a numeric {key}.") from exc
            if math.isnan(value) or value == -math.inf or (key != "psnr" and not math.isfinite(value)):
                raise ValueError(f"Invalid {key} value in sequence {sequence_id}.")
            if key == "mse" and value < 0:
                raise ValueError("MSE cannot be negative.")

    sequences: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    isos: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    scenes: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        sequences[record["sequence_id"]].append(record)
        isos[int(record["iso"])].append(record)
        scenes[int(record["scene"])].append(record)
    per_sequence = []
    for sequence_id, group in sorted(sequences.items()):
        scene, iso, noisy_index = sequence_metadata[sequence_id]
        item = {"sequence_id": sequence_id, "scene": scene, "iso": iso,
                "frame_count": len(group), "mean": _mean(group)}
        if noisy_index is not None:
            item["noisy_index"] = noisy_index
        per_sequence.append(item)
    return {
        **_group_summary(records),
        "aggregation": {
            "frame_mean": "arithmetic mean over all evaluated frames",
            "sequence_mean": "arithmetic mean over within-sequence frame means",
            "psnr": "mean of per-frame PSNR, not PSNR of mean MSE",
            "paper_protocol_verified": False,
        },
        "per_sequence": per_sequence,
        "per_iso": [{"iso": iso, **_group_summary(group)}
                    for iso, group in sorted(isos.items())],
        "per_scene": [{"scene": scene, **_group_summary(group)}
                      for scene, group in sorted(scenes.items())],
    }
