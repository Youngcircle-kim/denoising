"""Connect our explicitly unofficial retrained checkpoints to denoising_eval."""

import os
from pathlib import Path

import numpy as np
import torch

from denoising_eval.runner import sha256_file
from .data import NOISE_COEFFICIENTS_DN
from .flow import RawSPyNet
from .model import MSTMN


class RetrainedPredictor:
    def __init__(self, *, checkpoint: Path, device: str, protocol: dict):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('artifact_type') != 'mstmn-unofficial-retrain' or saved.get('schema_version') != 1:
            raise ValueError('Expected our unofficial MSTMN training checkpoint; official checkpoints need their own bridge')
        config = saved['config']
        for key in ('black_level', 'white_level'):
            expected = config['dataset'].get(key, 240.0 if key == 'black_level' else 4095.0)
            if protocol[key] != expected:
                raise ValueError(f'Evaluation/training {key} mismatch')
        flow_path = Path(os.environ.get('MSTMN_FLOW_CHECKPOINT', saved['flow']['path']))
        if not flow_path.is_file() or sha256_file(flow_path) != saved['flow']['sha256']:
            raise ValueError('Missing/mismatched SPyNet file; set MSTMN_FLOW_CHECKPOINT to the recorded upstream checkpoint')
        self.device = torch.device(device)
        self.model = MSTMN(flow_model=RawSPyNet(flow_path), **config['model'])
        self.model.load_state_dict(saved['model_state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.memory_frames = config['model']['memory_frames']
        self.range = protocol['white_level'] - protocol['black_level']
        self.previous = None
        self.history = None
        self.frame_index = 0

    def reset(self, sequence):
        iso = sequence['iso']
        if iso not in NOISE_COEFFICIENTS_DN:
            raise ValueError(f'Unknown ISO: {iso}')
        a, b = NOISE_COEFFICIENTS_DN[iso]
        self.shot = torch.tensor([a / self.range], device=self.device)
        self.read = torch.tensor([b / self.range**2], device=self.device)
        self.previous = None
        self.history = None
        self.frame_index = 0

    @torch.inference_mode()
    def predict(self, noisy, *, frame_index):
        if frame_index != self.frame_index + 1:
            raise ValueError('Retrained predictor requires consecutive frames starting at 1 after reset')
        current = torch.from_numpy(np.ascontiguousarray(np.moveaxis(noisy, -1, 0))).unsqueeze(0).to(self.device)
        if self.previous is None:
            self.previous = current
            self.history = current[:, None].repeat(1, self.memory_frames, 1, 1, 1)
        restored = self.model(current, self.previous, self.history, self.shot, self.read)
        self.previous = restored
        self.history = torch.cat((self.history[:, 1:], current[:, None]), dim=1)
        self.frame_index = frame_index
        return np.moveaxis(restored[0].cpu().numpy(), 0, -1)


def create_predictor(*, checkpoint, device, protocol):
    return RetrainedPredictor(checkpoint=checkpoint, device=device, protocol=protocol)
