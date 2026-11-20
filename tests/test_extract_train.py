"""Offline extraction tests use tiny ZIPs, never real data or network access."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from scripts import extract_crvd_train as extraction


def archive_with_names(names):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name in names:
            archive.writestr(name, b'raw')
    stream.seek(0)
    return zipfile.ZipFile(stream)


class ZipLayoutTests(unittest.TestCase):
    def test_exact_gt_and_noisy_layouts(self):
        for collection, count, per_iso in (('indoor_raw_gt', 35, 7), ('indoor_raw_noisy', 385, 77)):
            with self.subTest(collection=collection), archive_with_names(
                    sorted(extraction.expected_files(collection, 1))) as archive:
                stats = extraction.validate_entries(archive.infolist(), collection, 1)
                self.assertEqual(stats['files'], count)
                self.assertEqual(set(stats['iso_counts'].values()), {per_iso})

    def test_unsafe_or_wrong_dataset_names_fail(self):
        invalid = ('../escape.tiff', '/absolute.tiff', 'C:/drive.tiff',
                   'scene1/ISO1600/frame1_noisy0.tiff:ads',
                   r'scene1\ISO1600\frame1_noisy0.tiff',
                   'scene7/ISO1600/frame1_noisy0.tiff',
                   'scene1/ISO800/frame1_noisy0.tiff',
                   'scene1/ISO1600/frame8_noisy0.tiff')
        valid = sorted(extraction.expected_files('indoor_raw_noisy', 1))
        for name in invalid:
            with self.subTest(name=name), archive_with_names([*valid, name]) as archive:
                with self.assertRaises(ValueError):
                    extraction.validate_entries(archive.infolist(), 'indoor_raw_noisy', 1)

    def test_missing_duplicate_and_special_entries_fail(self):
        valid = sorted(extraction.expected_files('indoor_raw_gt', 1))
        with archive_with_names(valid[:-1]) as archive:
            with self.assertRaises(ValueError):
                extraction.validate_entries(archive.infolist(), 'indoor_raw_gt', 1)
        with archive_with_names(valid) as archive:
            entries = archive.infolist()
            with self.assertRaises(ValueError):
                extraction.validate_entries([*entries, entries[0]], 'indoor_raw_gt', 1)
            entries[0].external_attr = (stat.S_IFLNK | 0o777) << 16
            with self.assertRaises(ValueError):
                extraction.validate_entries(entries, 'indoor_raw_gt', 1)


class ExtractionTests(unittest.TestCase):
    def make_release(self, root):
        plan = extraction.build_plan(extraction.load_manifest(), 'crvd', 'train', None, root)
        for item in plan['files']:
            archive, _, receipt = extraction.archive_paths(item, root)
            archive.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
                for name in sorted(extraction.expected_files(item['collection'], item['scene'])):
                    zipped.writestr(name, b'synthetic RAW bytes')
            receipt.parent.mkdir(parents=True, exist_ok=True)
            size, digest, _ = extraction.digest_file(archive)
            count = 35 if item['collection'] == 'indoor_raw_gt' else 385
            receipt.write_text(json.dumps({
                'source_id': item['id'], 'name': item['name'], 'sha256': digest,
                'size_bytes': size, 'zip_entries': count,
                'zip_uncompressed_bytes': count * len(b'synthetic RAW bytes'),
            }), encoding='utf-8')
        return plan

    def test_complete_release_preserves_test_data_and_reread_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.make_release(root)
            sentinel = root / 'crvd/indoor_raw_gt/scene7/keep.txt'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'existing test data')
            with contextlib.redirect_stdout(io.StringIO()):
                report = extraction.extract(root)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['file_count'], 2520)
            self.assertEqual(len(report['published_directories']), 12)
            self.assertEqual(sentinel.read_bytes(), b'existing test data')
            self.assertEqual(len(list((root / 'archives/crvd').rglob('*.zip'))), 12)
            sample = report['files'][0]
            payload = (root / 'crvd' / sample['path']).read_bytes()
            self.assertEqual(sample['sha256'], hashlib.sha256(payload).hexdigest())
            self.assertEqual(json.loads((root / 'receipts/crvd-train-extraction.json').read_text())['status'],
                             'completed')
            with self.assertRaises(FileExistsError):
                extraction.extract(root)

    def test_missing_last_archive_prevents_any_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = self.make_release(root)
            extraction.archive_paths(plan['files'][-1], root)[0].unlink()
            with self.assertRaises(FileNotFoundError):
                extraction.extract(root)
            self.assertFalse((root / '.staging').exists())
            self.assertFalse((root / 'crvd').exists())

    def test_wrong_source_and_archive_hash_are_refused(self):
        for field, replacement in (('source_id', 'another-source'), ('sha256', '0' * 64)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                plan = self.make_release(root)
                receipt = extraction.archive_paths(plan['files'][0], root)[2]
                data = json.loads(receipt.read_text())
                data[field] = replacement
                receipt.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    extraction.extract(root)
                self.assertFalse((root / '.staging').exists())

    def test_existing_training_scene_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            sentinel = root / 'crvd/indoor_raw_noisy/scene3/keep.txt'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                extraction.extract(root)
            self.assertEqual(sentinel.read_bytes(), b'keep')
            self.assertFalse((root / '.staging').exists())


if __name__ == '__main__':
    unittest.main()
