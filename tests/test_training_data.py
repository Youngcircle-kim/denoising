"""Small synthetic TIFF fixtures verify training split and sampling semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from mstmn_train.data import CRVDTrainingDataset, NOISE_COEFFICIENTS_DN, pingpong_frames, synthesize_noise


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gt_dir = self.root / "indoor_raw_gt/scene1/ISO1600"
        self.noisy_dir = self.root / "indoor_raw_noisy/scene1/ISO1600"
        self.gt_dir.mkdir(parents=True)
        self.noisy_dir.mkdir(parents=True)
        for frame in range(1, 8):
            # Float GT preserves sub-DN averages and below-black observations.
            raw = (np.arange(16 * 20, dtype=np.float32).reshape(16, 20) + 230.5 + frame)
            tifffile.imwrite(self.gt_dir / f"frame{frame}_clean_and_slightly_denoised.tiff", raw)
            tifffile.imwrite(self.noisy_dir / f"frame{frame}_noisy0.tiff", raw.astype(np.uint16))
            tifffile.imwrite(self.noisy_dir / f"frame{frame}_clean.tiff", np.full_like(raw, 4095))

    def dataset(self, **kwargs):
        return CRVDTrainingDataset(self.root, scenes=[1], isos=[1600], patch_size=4,
                                   samples=16, noisy_indices=[0], **kwargs)

    def test_no_test_scene_can_enter_training(self):
        for scenes in ([7], [1, 7], [6, 11]):
            with self.subTest(scenes=scenes):
                with self.assertRaisesRegex(ValueError, "test scenes forbidden"):
                    CRVDTrainingDataset(self.root, scenes=scenes, isos=[1600])

    def test_float_targets_shape_crop_phase_and_coefficients(self):
        sample = self.dataset(seed=42)[0]
        self.assertEqual(sample['input'].shape, (25, 4, 4, 4))
        self.assertEqual(sample['target'].shape, (25, 4, 4, 4))
        self.assertEqual(sample['input'].dtype, np.float32)
        self.assertEqual(sample['target'].dtype, np.float32)
        self.assertTrue(sample['input'].flags.c_contiguous)
        self.assertTrue(np.all(sample['crop_yx'] % 2 == 0))
        self.assertTrue(np.isfinite(sample['input']).all())
        a, b = NOISE_COEFFICIENTS_DN[1600]
        self.assertAlmostEqual(float(sample['shot']), a / 3855, places=9)
        self.assertAlmostEqual(float(sample['read']), b / 3855**2, places=12)
        # Fractional DN in GT is retained, rather than casting the GT to uint16.
        dn = sample['target'] * 3855 + 240
        np.testing.assert_allclose(dn % 1, 0.5, atol=0.0001)

    def test_reproducible_sample_ids_independent_of_order_cache_and_resume(self):
        dataset = self.dataset(seed=123)
        original = dataset[8]
        _ = dataset[3]
        resumed = self.dataset(seed=123, start_index=8, cache_sequences=0)[0]
        for key in original:
            with self.subTest(key=key):
                np.testing.assert_array_equal(original[key], dataset[8][key])
                np.testing.assert_array_equal(original[key], resumed[key])
        self.assertFalse(np.array_equal(dataset[7]['input'], original['input']))

    def test_real_mode_uses_one_noisy_id_and_never_clean_file(self):
        sample = self.dataset(mode='real', seed=7)[0]
        self.assertEqual(sample['noisy_index'], 0)
        self.assertLess(float(sample['input'].max()), 0.2)
        frame_numbers = sample['frame_numbers']
        for frame in range(1, 8):
            repeated = sample['input'][frame_numbers == frame]
            self.assertGreaterEqual(len(repeated), 2)
            np.testing.assert_array_equal(repeated, np.broadcast_to(repeated[0], repeated.shape))
        # Measured uint16 input differs from fractional floating point GT.
        np.testing.assert_allclose(sample['target'] - sample['input'], 0.5 / 3855, atol=1e-8)

    def test_synthetic_does_not_require_measured_noisy_files(self):
        for path in self.noisy_dir.iterdir():
            path.unlink()
        self.assertEqual(self.dataset()[0]['noisy_index'], -1)
        with self.assertRaisesRegex(FileNotFoundError, '7 missing files'):
            self.dataset(mode='real')

    def test_missing_gt_and_inconsistent_dimensions_fail(self):
        path = self.gt_dir / 'frame7_clean_and_slightly_denoised.tiff'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset()
        tifffile.imwrite(path, np.zeros((18, 20), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'Inconsistent GT frame sizes'):
            self.dataset()[0]

    def test_bounds_and_invalid_training_configuration(self):
        dataset = self.dataset()
        for index in (-1, 16, True, 1.5):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    dataset[index]
        for kwargs in ({'sequence_length': 0}, {'seed': -1}, {'start_index': -1},
                       {'clip_inputs': False}, {'mode': 'unknown'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.dataset(**kwargs)


class TemporalAndNoiseTests(unittest.TestCase):
    def test_pingpong_25_and_all_phases_preserve_adjacent_motion(self):
        expected = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
        np.testing.assert_array_equal(pingpong_frames(25), expected)
        for phase in range(12):
            for direction in (-1, 1):
                frames = pingpong_frames(25, phase, direction)
                self.assertEqual(set(frames), set(range(1, 8)))
                np.testing.assert_array_equal(np.abs(np.diff(frames)), np.ones(24))

    def test_noise_mean_variance_match_normalized_dn_coefficients(self):
        # Unclipped/unquantized samples isolate the mathematical noise model.
        target = np.full(120000, 0.4, dtype=np.float32)
        for iso in (1600, 25600):
            with self.subTest(iso=iso):
                noisy = synthesize_noise(target, iso, np.random.default_rng(42), clip_inputs=False, quantize=False)
                error = noisy - target
                a, b = NOISE_COEFFICIENTS_DN[iso]
                variance = 0.4 * a / 3855 + b / 3855**2
                self.assertLess(abs(float(error.mean())), 4 * np.sqrt(variance / target.size))
                self.assertAlmostEqual(float(error.var()) / variance, 1.0, delta=0.02)

    def test_dn_clipping_retains_negative_normalized_noise_and_target(self):
        target = np.full(10000, -0.01, dtype=np.float32)
        original = target.copy()
        noisy = synthesize_noise(target, 25600, np.random.default_rng(12))
        self.assertLess(float(noisy.min()), 0)
        self.assertGreaterEqual(float(noisy.min()), -240 / 3855 - 1e-8)
        self.assertLessEqual(float(noisy.max()), 1)
        np.testing.assert_array_equal(target, original)
        dn = noisy * 3855 + 240
        np.testing.assert_allclose(dn, np.rint(dn), atol=0.0001)


if __name__ == '__main__':
    unittest.main()
