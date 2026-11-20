"""Synthetic numerical checks; these do not establish MSTMN reproduction."""

import math
import unittest

import numpy as np
from skimage.metrics import structural_similarity

from denoising_eval.metrics import compute_metrics, summarize_records


class MetricsTests(unittest.TestCase):
    def test_identity_has_zero_mse_infinite_psnr_and_unit_ssim(self):
        image = np.random.default_rng(4).uniform(size=(24, 32)).astype(np.float32)
        result = compute_metrics(image, image)
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["psnr"], math.inf)
        self.assertAlmostEqual(result["ssim"], 1.0)

    def test_constant_offset_has_known_mse_and_psnr(self):
        target = np.full((16, 16), 0.25)
        result = compute_metrics(target + 0.125, target)
        self.assertEqual(result["mse"], 0.015625)
        self.assertAlmostEqual(result["psnr"], 18.06179973983887)

    def test_ssim_matches_explicit_documented_settings(self):
        rng = np.random.default_rng(17)
        target = rng.uniform(size=(30, 24))
        prediction = target + rng.normal(0, 0.04, target.shape)
        expected = structural_similarity(
            target, prediction, data_range=1.0, channel_axis=None,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
            K1=0.01, K2=0.03,
        )
        self.assertAlmostEqual(compute_metrics(prediction, target)["ssim"], expected, places=14)

    def test_outside_range_values_are_not_silently_clipped(self):
        target = np.ones((11, 11))
        result = compute_metrics(target + 0.5, target)
        self.assertEqual(result["mse"], 0.25)
        self.assertAlmostEqual(result["psnr"], 6.020599913279624)

    def test_unsigned_subtraction_does_not_wrap(self):
        result = compute_metrics(np.zeros((11, 11), dtype=np.uint16),
                                 np.ones((11, 11), dtype=np.uint16))
        self.assertEqual(result["mse"], 1.0)
        self.assertEqual(result["psnr"], 0.0)

    def test_invalid_shapes_small_images_and_nonfinite_values_fail(self):
        good = np.zeros((16, 16))
        cases = [np.zeros((16, 15)), np.zeros((16, 16, 4)),
                 np.zeros((10, 16)), np.full((16, 16), np.nan),
                 np.full((16, 16), np.inf), np.zeros((16, 16), dtype=complex),
                 np.full((16, 16), "not numeric")]
        for prediction in cases:
            target = np.zeros_like(prediction) if prediction.shape != good.shape else good
            if prediction.shape == (16, 15):
                target = good
            with self.subTest(shape=prediction.shape, dtype=prediction.dtype):
                with self.assertRaises(ValueError):
                    compute_metrics(prediction, target)


def record(sequence, frame, mse, psnr, ssim, scene=7, iso=1600):
    return {"sequence_id": sequence, "frame": frame, "scene": scene,
            "iso": iso, "noisy_index": 0, "mse": mse, "psnr": psnr, "ssim": ssim}


class AggregationTests(unittest.TestCase):
    def test_unequal_sequence_lengths_keep_both_reductions_explicit(self):
        records = [record("a", 1, 0.01, 20.0, 0.8),
                   record("a", 2, 0.01, 20.0, 0.8),
                   record("b", 1, 0.1, 10.0, 0.5, scene=8, iso=3200)]
        result = summarize_records(records)
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["sequence_count"], 2)
        self.assertAlmostEqual(result["frame_mean"]["psnr"], 50 / 3)
        self.assertEqual(result["sequence_mean"]["psnr"], 15.0)
        self.assertAlmostEqual(result["sequence_mean"]["mse"], 0.055)
        self.assertEqual([row["iso"] for row in result["per_iso"]], [1600, 3200])
        self.assertEqual(result["per_sequence"][0]["frame_count"], 2)
        self.assertFalse(result["aggregation"]["paper_protocol_verified"])
        self.assertNotAlmostEqual(result["frame_mean"]["psnr"],
                                  -10 * math.log10(result["frame_mean"]["mse"]))

    def test_per_iso_sequence_mean_is_sequence_equal(self):
        records = [record("a", 1, 0.01, 20, 0.8), record("a", 2, 0.01, 20, 0.8),
                   record("b", 1, 0.1, 10, 0.5, scene=8)]
        result = summarize_records(records)
        self.assertEqual(result["per_iso"][0]["sequence_mean"]["psnr"], 15.0)
        self.assertAlmostEqual(result["per_iso"][0]["frame_mean"]["psnr"], 50 / 3)

    def test_perfect_frame_psnr_remains_infinite(self):
        result = summarize_records([record("a", 1, 0.0, math.inf, 1.0),
                                    record("a", 2, 0.1, 10, 0.5)])
        self.assertEqual(result["frame_mean"]["psnr"], math.inf)
        self.assertEqual(result["sequence_mean"]["psnr"], math.inf)

    def test_invalid_records_and_duplicate_frames_fail(self):
        original = record("a", 1, 0.1, 10, 0.5)
        bad_cases = [[], [original, dict(original)],
                     [dict(original, scene=0)], [dict(original, sequence_id="")],
                     [dict(original, mse=-1)], [dict(original, psnr=math.nan)],
                     [dict(original, ssim=math.inf)],
                     [original, dict(original, frame=2, scene=8)]]
        for records in bad_cases:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    summarize_records(records)


if __name__ == "__main__":
    unittest.main()
