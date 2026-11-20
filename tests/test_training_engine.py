"""Schedule/resume checks and optional CPU tests of true temporal gradients."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

from mstmn_train.train import learning_rate_for_step, sample_window, train_effective_batch, validate_config, validate_resume


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/mstmn_retrain.json"


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_recipe_validation_and_one_based_learning_rate_boundaries(self):
        validate_config(self.config)
        for step, rate in ((1, .001), (14999, .001), (15000, .0005), (39999, .0005), (40000, .0001), (120000, .0001)):
            with self.subTest(step=step):
                self.assertEqual(learning_rate_for_step(step, self.config['optimizer']), rate)

    def test_resume_sample_window_has_no_repeated_or_skipped_samples(self):
        self.assertEqual(sample_window(100, 120000, 16), (1600, 1918400))
        self.assertEqual(sample_window(100, 100, 16), (1600, 0))
        with self.assertRaises(ValueError):
            sample_window(101, 100, 16)

    def test_resume_rejects_recipe_flow_or_implementation_drift(self):
        checkpoint = {'schema_version': 1, 'artifact_type': 'mstmn-unofficial-retrain',
                      'config': self.config, 'flow': {'sha256': 'abc'},
                      'source_hashes': {'model.py': 'def'}, 'completed_steps': 100}
        self.assertEqual(validate_resume(checkpoint, self.config, 'abc', {'model.py': 'def'}), 100)
        for key in ('config', 'flow', 'source_hashes', 'artifact_type'):
            changed = copy.deepcopy(checkpoint)
            if key == 'config':
                changed[key]['dataset']['sequence_length'] = 7
            elif key == 'flow':
                changed[key]['sha256'] = 'changed'
            elif key == 'source_hashes':
                changed[key]['model.py'] = 'changed'
            else:
                changed[key] = 'official-mstmn'
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_resume(changed, self.config, 'abc', {'model.py': 'def'})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed in this environment')
class GradientTests(unittest.TestCase):
    def batch(self, count=4):
        import torch
        generator = torch.Generator().manual_seed(11)
        return {'input': torch.rand(count, 5, 4, 2, 2, generator=generator),
                'target': torch.zeros(count, 5, 4, 2, 2),
                'shot': torch.ones(count) * .001, 'read': torch.ones(count) * .000001}

    def model(self):
        import torch

        class TinyRecurrent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
                self.seen_history = []

            def forward(self, noisy, previous, history, shot, read):
                self.seen_history.append(history.detach().clone())
                return self.weight * noisy + .2 * previous

        return TinyRecurrent()

    def test_microbatch_accumulation_matches_effective_batch_update(self):
        import torch
        full, micro = self.model(), self.model()
        full_optimizer = torch.optim.SGD(full.parameters(), lr=.01)
        micro_optimizer = torch.optim.SGD(micro.parameters(), lr=.01)
        batch = self.batch()
        full_loss = train_effective_batch(full, full_optimizer, batch, microbatch=4, memory_frames=3, device='cpu')
        micro_loss = train_effective_batch(micro, micro_optimizer, batch, microbatch=3, memory_frames=3, device='cpu')
        self.assertAlmostEqual(full_loss, micro_loss, places=7)
        torch.testing.assert_close(full.weight, micro.weight, rtol=1e-6, atol=1e-8)

    def test_temporal_gradient_is_full_bptt_and_initial_memory_is_first_frame(self):
        import torch
        model = self.model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        batch = self.batch(1)
        loss = train_effective_batch(model, optimizer, batch, microbatch=1, memory_frames=3, device='cpu')
        actual_gradient = model.weight.grad.clone()
        torch.testing.assert_close(model.seen_history[0], batch['input'][:, 0, None].expand(-1, 3, -1, -1, -1))
        torch.testing.assert_close(model.seen_history[1], model.seen_history[0])
        torch.testing.assert_close(model.seen_history[4], batch['input'][:, 1:4])
        reference = self.model()
        previous = batch['input'][:, 0]
        reference_loss = 0
        for frame in range(5):
            previous = reference.weight * batch['input'][:, frame] + .2 * previous
            reference_loss = reference_loss + previous.abs().mean() / 5
        reference_loss.backward()
        self.assertAlmostEqual(loss, float(reference_loss), places=7)
        torch.testing.assert_close(actual_gradient, reference.weight.grad)

    def test_nonfinite_loss_does_not_update_parameters(self):
        import torch
        model = self.model()
        original = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        batch = self.batch()
        batch['input'][0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            train_effective_batch(model, optimizer, batch, microbatch=2, memory_frames=3, device='cpu')
        torch.testing.assert_close(model.weight, original)


if __name__ == '__main__':
    unittest.main()
