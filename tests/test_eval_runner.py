"""Exercise complete TIFF-to-report evaluation and recurrent adapter boundaries."""

import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from denoising_eval.__main__ import main
from denoising_eval.backends import load_adapter
from denoising_eval.runner import evaluate, load_config, prepare


class StatefulPredictor:
    def __init__(self):
        self.resets = []
        self.counts = []

    def reset(self, sequence):
        self.resets.append(dict(sequence))
        self.count = 0

    def predict(self, noisy, *, frame_index):
        self.count += 1
        self.counts.append(self.count)
        assert frame_index == self.count
        return noisy.copy()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.config = load_config(Path(__file__).resolve().parents[1] / 'configs/crvd_raw_test.json')
        self.config.update(scenes=[7, 8], iso_values=[1600], noisy_indices=[0], clip_prediction=False)
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        for scene in (7, 8):
            for frame in range(1, 8):
                for collection, suffix, dtype in [
                    ('indoor_raw_gt', 'clean_and_slightly_denoised', np.float32),
                    ('indoor_raw_noisy', 'noisy0', np.uint16),
                ]:
                    path = self.data / collection / f'scene{scene}' / 'ISO1600' / f'frame{frame}_{suffix}.tiff'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(path, np.full((16, 16), 1000, dtype=dtype))

    def test_complete_pipeline_resets_state_and_serializes_perfect_psnr(self):
        output = self.root / 'out'
        records = prepare(self.data, self.config, output)
        predictor = StatefulPredictor()
        result = evaluate(records, self.config, predictor, output, model={'kind': 'test'}, save_predictions=True)
        self.assertEqual(predictor.counts, list(range(1, 8)) * 2)
        self.assertEqual(len(predictor.resets), 2)
        self.assertTrue(all(set(x) == {'scene', 'iso', 'noisy_index', 'sequence_id'} for x in predictor.resets))
        self.assertEqual(result['evaluated_frames'], 14)
        self.assertEqual(result['metrics']['frame_mean']['ssim'], 1)
        self.assertFalse(result['official_reproduction'])
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['metrics']['frame_mean']['psnr'], 'Infinity')
        self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'completed')
        self.assertEqual(len(list((output / 'predictions').rglob('*.npy'))), 14)

    def test_warmup_updates_state_and_smoke_run_is_marked_partial(self):
        config = copy.deepcopy(self.config)
        config['warmup_frames'] = 3
        output = self.root / 'warmup'
        records = prepare(self.data, config, output)
        predictor = StatefulPredictor()
        result = evaluate(records, config, predictor, output, model={'kind': 'test'}, max_sequences=1)
        self.assertEqual(predictor.counts, list(range(1, 8)))
        self.assertEqual(result['evaluated_frames'], 4)
        self.assertTrue(result['partial_run'])
        self.assertFalse(result['selection_is_full_release_test'])

    def test_relative_data_root_and_existing_output(self):
        output = self.root / 'relative'
        records = prepare(Path(os.path.relpath(self.data)), self.config, output)
        self.assertEqual(len(records), 2)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertFalse(Path(manifest[0]['frames'][0]['noisy']).is_absolute())
        with self.assertRaises(FileExistsError):
            prepare(self.data, self.config, output)
        with self.assertRaises(ValueError):
            prepare(self.data, self.config, self.data / 'results')

    def test_missing_checkpoint_cannot_fall_back_to_diagnostic(self):
        output = self.root / 'missing'
        code = main(['evaluate', '--config', str(self.config_path), '--data-root', str(self.data),
                     '--output', str(output), '--adapter', 'no_model:create',
                     '--checkpoint', str(self.root / 'absent.pth'), '--model-name', 'MSTMN'])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())

    def test_adapter_factory_receives_checkpoint_device_and_protocol(self):
        checkpoint = self.root / 'weights.bin'
        checkpoint.write_bytes(b'test checkpoint, not a model')
        module = types.ModuleType('_test_adapter')
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return StatefulPredictor()
        module.create = create
        with patch.dict('sys.modules', {'_test_adapter': module}):
            predictor = load_adapter('_test_adapter:create', checkpoint, 'cpu', self.config)
        self.assertIsInstance(predictor, StatefulPredictor)
        self.assertEqual(calls[0], {'checkpoint': checkpoint.resolve(), 'device': 'cpu', 'protocol': self.config})

    def test_invalid_output_is_failed_without_completed_summary(self):
        class BadPredictor(StatefulPredictor):
            def predict(self, noisy, *, frame_index):
                return np.full_like(noisy, np.nan)
        output = self.root / 'bad'
        with patch('denoising_eval.__main__.IdentityBaseline', return_value=BadPredictor()):
            code = main(['evaluate', '--config', str(self.config_path), '--data-root', str(self.data),
                         '--output', str(output), '--baseline', 'identity'])
        self.assertEqual(code, 1)
        self.assertFalse((output / 'summary.json').exists())
        self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')
        self.assertEqual(json.loads((output / 'failure.json').read_text())['status'], 'failed')

    def test_cli_inspects_all_selected_headers(self):
        output = self.root / 'inspect'
        code = main(['inspect', '--config', str(self.config_path), '--data-root', str(self.data), '--output', str(output)])
        self.assertEqual(code, 0)
        report = json.loads((output / 'inspection.json').read_text())
        self.assertEqual(report['frame_pairs'], 14)
        self.assertEqual(report['unique_selected_tiffs'], 28)
        self.assertEqual(report['headers']['gt'][0]['dtype'], 'float32')

    def test_invalid_protocol_options_are_rejected(self):
        for field, value in [('warmup_frames', 7), ('clip_input', 'false'), ('crop_border', -1),
                             ('noisy_indices', [0, 0]), ('scenes', [1]), ('white_level', 240)]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                self.config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(self.config_path)


if __name__ == '__main__':
    unittest.main()
