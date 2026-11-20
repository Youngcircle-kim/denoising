"""Model adapter contract; importing this module never loads PyTorch or weights."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np


class Predictor(Protocol):
    """HWC float32 packed RGBG, normalized DN; never receives ground truth.

    reset receives scene, iso, noisy_index and sequence_id. Implementations own
    previous restoration, past noisy frames, noise conditioning and padding.
    predict receives a one-based frame index in chronological order.
    """

    def reset(self, sequence: Mapping) -> None: ...

    def predict(self, noisy: np.ndarray, *, frame_index: int) -> np.ndarray: ...


class IdentityBaseline:
    """Input-copy diagnostic baseline. This is NOT MSTMN inference."""

    def reset(self, sequence: Mapping) -> None:
        pass

    def predict(self, noisy: np.ndarray, *, frame_index: int) -> np.ndarray:
        return noisy.copy()


def load_adapter(spec: str, checkpoint: Path, device: str, protocol: dict) -> Predictor:
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint}')
    if spec.count(':') != 1:
        raise ValueError('Adapter must be module_name:factory_name')
    module_name, factory_name = spec.split(':')
    factory = getattr(importlib.import_module(module_name), factory_name)
    predictor = factory(checkpoint=checkpoint.resolve(), device=device, protocol=protocol)
    if not callable(getattr(predictor, 'reset', None)) or not callable(
        getattr(predictor, 'predict', None)
    ):
        raise TypeError('Adapter must implement reset(sequence) and predict(noisy, *, frame_index)')
    return predictor
