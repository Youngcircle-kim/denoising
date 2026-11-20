"""Synthetic file fixtures test protocol selection without downloading data."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from denoising_eval.data import build_manifest, normalize_raw, pack_gbrg, read_raw, unpack_gbrg


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gt_dir = self.root / "indoor_raw_gt/scene7/ISO1600"
        self.noisy_dir = self.root / "indoor_raw_noisy/scene7/ISO1600"
        self.gt_dir.mkdir(parents=True)
        self.noisy_dir.mkdir(parents=True)
        # Deliberately insert names out of temporal order and include leaked GT
        # in the noisy collection: neither may influence the selected inputs.
        for frame in (7, 2, 6, 1, 5, 3, 4):
            (self.gt_dir / f"frame{frame}_clean_and_slightly_denoised.tiff").touch()
            (self.noisy_dir / f"frame{frame}_clean.tiff").touch()
            for index in (0, 9):
                (self.noisy_dir / f"frame{frame}_noisy{index}.tiff").touch()

    def test_explicit_selection_temporal_order_and_no_clean_input(self):
        records = build_manifest(self.root, [7], [1600], [9, 0])
        self.assertEqual([record.key.noisy_index for record in records], [0, 9])
        self.assertEqual([pair.frame for pair in records[0].frames], list(range(1, 8)))
        for record in records:
            self.assertEqual(len(record.frames), 7)
            for pair in record.frames:
                self.assertEqual(pair.noisy.name, f"frame{pair.frame}_noisy{record.key.noisy_index}.tiff")
                self.assertEqual(pair.gt.parent, self.gt_dir.resolve())
                self.assertTrue(pair.noisy.is_absolute())

    def test_missing_selected_noisy_cannot_fall_back_to_clean(self):
        (self.noisy_dir / "frame4_noisy0.tiff").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "frame4_noisy0.tiff"):
            build_manifest(self.root, [7], [1600], [0])
        # Missing unselected realization zero does not prevent selecting nine.
        self.assertEqual(len(build_manifest(self.root, [7], [1600], [9])), 1)

    def test_missing_gt_and_missing_sequence_report_before_inference(self):
        (self.gt_dir / "frame7_clean_and_slightly_denoised.tiff").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "15 missing file"):
            build_manifest(self.root, [7], [1600, 3200], [0])

    def test_ambiguous_numeric_file_ids_are_rejected(self):
        (self.noisy_dir / "frame01_noisy0.tiff").touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous CRVD file IDs"):
            build_manifest(self.root, [7], [1600], [0])

    def test_protocol_selection_must_be_explicit_valid_and_unique(self):
        for scenes, isos, indices in (
            ([1], [1600], [0]), ([7], [800], [0]), ([7], [1600], None),
            ([7], [1600], []), ([7], [1600], [10]), ([7], [1600], [0, 0]),
            ([True], [1600], [0]), ([7], [1600], "0"),
        ):
            with self.subTest(selection=(scenes, isos, indices)):
                with self.assertRaises(ValueError):
                    build_manifest(self.root, scenes, isos, indices)


class RawTests(unittest.TestCase):
    def test_read_preserves_float_gt_and_integer_noisy_in_original_units(self):
        with tempfile.TemporaryDirectory() as directory:
            for dtype in (np.uint16, np.float32, np.float64):
                with self.subTest(dtype=dtype):
                    raw = np.array([[240, 241], [4094, 4095]], dtype=dtype)
                    if np.issubdtype(dtype, np.floating):
                        raw[0, 1] = 241.5
                    path = Path(directory) / "raw.tiff"
                    tifffile.imwrite(path, raw)
                    result = read_raw(path)
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_array_equal(result, raw)

    def test_read_rejects_multichannel_nonfinite_and_complex(self):
        with tempfile.TemporaryDirectory() as directory:
            for raw in (
                np.zeros((2, 2, 3), dtype=np.uint16),
                np.array([[np.nan]], dtype=np.float32),
                np.array([[np.inf]], dtype=np.float32),
                np.ones((2, 2), dtype=np.complex64),
            ):
                with self.subTest(shape=raw.shape, dtype=raw.dtype):
                    path = Path(directory) / "invalid.tiff"
                    tifffile.imwrite(path, raw)
                    with self.assertRaises(ValueError):
                        read_raw(path)

    def test_normalization_is_float_and_clipping_requires_explicit_choice(self):
        raw = np.array([[0, 240, 4095, 5000]], dtype=np.uint16)
        result = normalize_raw(raw)
        self.assertEqual(result.dtype, np.float32)
        self.assertLess(result[0, 0], 0)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[0, 2], 1)
        self.assertGreater(result[0, 3], 1)
        np.testing.assert_array_equal(normalize_raw(raw, clip=True), [[0, 0, 1, 1]])
        np.testing.assert_array_equal(normalize_raw(np.array([10, 20]), 10, 20), [0, 1])

    def test_normalization_rejects_invalid_levels_and_nonfinite_data(self):
        for black, white in ((240, 240), (4095, 240), (np.nan, 4095), (240, np.inf)):
            with self.subTest(levels=(black, white)):
                with self.assertRaises(ValueError):
                    normalize_raw(np.zeros((2, 2)), black, white)
        with self.assertRaises(ValueError):
            normalize_raw(np.array([np.inf]))
        with self.assertRaises(ValueError):
            normalize_raw(np.array([240]), clip="false")

    def test_gbrg_rgbg_channel_positions_and_lossless_inverse(self):
        raw = np.arange(24, dtype=np.uint16).reshape(4, 6)
        packed = pack_gbrg(raw)
        self.assertEqual(packed.shape, (2, 3, 4))
        np.testing.assert_array_equal(packed[0, 0], [6, 0, 1, 7])
        np.testing.assert_array_equal(packed[1, 2], [22, 16, 17, 23])
        np.testing.assert_array_equal(unpack_gbrg(packed), raw)
        self.assertEqual(unpack_gbrg(packed).dtype, raw.dtype)

    def test_packing_rejects_odd_dimensions_and_wrong_channel_count(self):
        for raw in (np.zeros((3, 4)), np.zeros((4, 3)), np.zeros((4, 4, 1))):
            with self.subTest(shape=raw.shape):
                with self.assertRaises(ValueError):
                    pack_gbrg(raw)
        for packed in (np.zeros((2, 2)), np.zeros((2, 2, 3))):
            with self.subTest(shape=packed.shape):
                with self.assertRaises(ValueError):
                    unpack_gbrg(packed)


if __name__ == "__main__":
    unittest.main()
