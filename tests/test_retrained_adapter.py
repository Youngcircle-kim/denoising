import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required for checkpoint adapter tests')
class AdapterTests(unittest.TestCase):
    def setUp(self):
        import torch
        from mstmn_train.model import MSTMN
        from denoising_eval.runner import sha256_file
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.flow_file = self.folder / 'test-flow.bin'
        self.flow_file.write_bytes(b'synthetic test flow, not upstream weights')
        class ExplicitTestFlow(torch.nn.Module):
            def forward(self, current, previous):
                return current.new_zeros(current.shape[0], 2, *current.shape[-2:])
        self.flow_type = ExplicitTestFlow
        self.config = {'dataset': {'black_level': 240, 'white_level': 4095},
                       'model': {'memory_frames': 3, 'patch_size': 4, 'patch_stride': 4, 'query_chunk_size': 128}}
        model = MSTMN(flow_model=self.flow_type(), **self.config['model'])
        self.checkpoint = self.folder / 'test.pth'
        self.saved = {'schema_version': 1, 'artifact_type': 'mstmn-unofficial-retrain',
                      'config': self.config, 'model_state_dict': model.state_dict(),
                      'flow': {'path': str(self.flow_file), 'sha256': sha256_file(self.flow_file)}}
        torch.save(self.saved, self.checkpoint)
        self.protocol = {'black_level': 240, 'white_level': 4095}

    def load(self):
        from mstmn_train.adapter import create_predictor
        with patch('mstmn_train.adapter.RawSPyNet', side_effect=lambda _: self.flow_type()):
            return create_predictor(checkpoint=self.checkpoint, device='cpu', protocol=self.protocol)

    def test_checkpoint_runs_and_reset_clears_previous_sequence(self):
        predictor = self.load()
        observation = np.full((16, 16, 4), .2, np.float32)
        predictor.reset({'iso': 1600})
        first = predictor.predict(observation, frame_index=1)
        predictor.predict(observation, frame_index=2)
        predictor.reset({'iso': 1600})
        replay = predictor.predict(observation, frame_index=1)
        self.assertEqual(first.shape, observation.shape)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, replay, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            predictor.predict(observation, frame_index=4)

    def test_wrong_provenance_or_sensor_range_is_refused(self):
        import torch
        self.protocol['white_level'] = 1023
        with self.assertRaises(ValueError):
            self.load()
        self.protocol['white_level'] = 4095
        self.saved['artifact_type'] = 'official-mstmn'
        torch.save(self.saved, self.checkpoint)
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
