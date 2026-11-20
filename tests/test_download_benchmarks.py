"""Offline tests: tiny synthetic archives and injected download functions only."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_benchmarks", REPO / "scripts" / "download_benchmarks.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def make_zip(path, payload=b"synthetic RAW bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("scene7/ISO1600/frame1.tiff", payload)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest = bench.load_manifest()

    def plan(self, split="test", scenes=None):
        return bench.build_plan(self.manifest, "crvd", split, scenes, Path("unused-data"))

    def test_default_test_plan_has_five_paired_scenes(self):
        plan = self.plan()
        self.assertEqual(plan["scenes"], [7, 8, 9, 10, 11])
        self.assertEqual(plan["archive_count"], 10)
        self.assertEqual(plan["iso_values"], [1600, 3200, 6400, 12800, 25600])
        self.assertEqual(
            {(item["scene"], item["collection"]) for item in plan["files"]},
            {(scene, collection) for scene in range(7, 12)
             for collection in ("indoor_raw_gt", "indoor_raw_noisy")},
        )
        self.assertEqual(len({item["id"] for item in plan["files"]}), 10)

    def test_wrong_split_and_duplicate_scene_selections_fail(self):
        for split, scenes in (("test", [6]), ("train", [7]),
                              ("test", [7, 7]), ("test", [])):
            with self.subTest(split=split, scenes=scenes):
                with self.assertRaises(bench.BenchmarkError):
                    self.plan(split, scenes)

    def test_manifest_duplicate_identity_is_rejected(self):
        for key in ("id", "name"):
            manifest = copy.deepcopy(self.manifest)
            files = manifest["datasets"]["crvd"]["files"]
            if key == "id":
                files[1][key] = files[0][key]
            else:
                files.append(dict(files[0], id="another-valid-drive-id"))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "sources.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(duplicate=key):
                    with self.assertRaises(bench.BenchmarkError):
                        bench.load_manifest(path)

    def test_missing_selected_manifest_pair_is_rejected(self):
        files = self.manifest["datasets"]["crvd"]["files"]
        files[:] = [item for item in files
                    if (item["scene"], item["collection"]) != (7, "indoor_raw_gt")]
        with self.assertRaises(bench.BenchmarkError):
            self.plan()

    def test_default_action_is_offline_plan_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "must-not-exist"
            with mock.patch.object(bench, "get_downloader", side_effect=AssertionError("dependency access")), \
                    mock.patch.object(bench.importlib.metadata, "version", side_effect=AssertionError("dependency lookup")), \
                    mock.patch("builtins.__import__", wraps=__import__) as import_spy, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                status = bench.main(["--data-root", str(root), "--json"])
            self.assertEqual(status, 0)
            self.assertFalse(root.exists())
            self.assertEqual(json.loads(output.getvalue())["archive_count"], 10)
            self.assertFalse(any(call.args[0] == "gdown" for call in import_spy.call_args_list))

    def test_real_rawvsr_plan_is_manual_and_not_an_automatic_download(self):
        plan = bench.build_plan(self.manifest, "real-rawvsr", "test", None, Path("unused"))
        self.assertEqual(plan["mode"], "manual")
        self.assertIn("pan.baidu.com", plan["download_url"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "not-created"
            with mock.patch.object(bench, "get_downloader", side_effect=AssertionError("unexpected downloader")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = bench.main(["download", "--dataset", "real-rawvsr", "--data-root", str(root)])
            self.assertEqual(status, 2)
            self.assertFalse(root.exists())


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        self.item = bench.build_plan(bench.load_manifest(), "crvd", "test", [7], self.root)["files"][0]

    def paths(self):
        return bench.archive_paths(self.item, self.root)

    def successful_downloader(self, **kwargs):
        self.assertTrue(kwargs["resume"])
        self.assertFalse(kwargs["use_cookies"])
        self.assertEqual(kwargs["id"], self.item["id"])
        make_zip(Path(kwargs["output"]))
        return kwargs["output"]

    def download(self, downloader=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return bench.download_file(self.item, self.root, downloader or self.successful_downloader)

    def test_interrupted_partial_is_not_published_and_next_call_resumes(self):
        def interrupted(**kwargs):
            Path(kwargs["output"] + ".part").write_bytes(b"partial transport bytes")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.download(interrupted)
        target, staging, receipt = self.paths()
        self.assertFalse(target.exists())
        self.assertFalse(receipt.exists())
        self.assertTrue(Path(str(staging) + ".part").exists())

        def resumed(**kwargs):
            self.assertTrue(kwargs["resume"])
            partial = Path(kwargs["output"] + ".part")
            self.assertTrue(partial.exists())
            make_zip(Path(kwargs["output"]))
            partial.unlink()
            return kwargs["output"]

        self.download(resumed)
        self.assertTrue(target.is_file())
        self.assertTrue(receipt.is_file())
        self.assertFalse(staging.exists())
        self.assertEqual(bench.verify_file(self.item, self.root)["source_id"], self.item["id"])

    def test_none_result_without_complete_file_is_not_success(self):
        with self.assertRaises(bench.BenchmarkError):
            self.download(lambda **kwargs: None)
        target, _, receipt = self.paths()
        self.assertFalse(target.exists())
        self.assertFalse(receipt.exists())

    def test_html_and_corrupt_zip_are_not_published(self):
        for payload in (b"<html>Login required</html>", b"PK\x03\x04truncated ZIP"):
            with self.subTest(payload=payload):
                def invalid(**kwargs):
                    Path(kwargs["output"]).write_bytes(payload)
                    return kwargs["output"]
                with self.assertRaises(bench.BenchmarkError):
                    self.download(invalid)
                target, staging, receipt = self.paths()
                self.assertFalse(target.exists())
                self.assertFalse(receipt.exists())
                staging.unlink(missing_ok=True)

    def test_crc_failure_is_rejected(self):
        archive = Path(self.tmp.name) / "crc.zip"
        payload = b"known synthetic payload to corrupt"
        make_zip(archive, payload)
        data = archive.read_bytes()
        self.assertIn(payload, data)
        archive.write_bytes(data.replace(payload, b"X" + payload[1:], 1))
        with self.assertRaises(bench.BenchmarkError):
            bench.inspect_archive(archive, self.item)

    def test_corrupt_deflate_is_reported_as_benchmark_error(self):
        archive = Path(self.tmp.name) / "deflate.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("a", b"A" * 1000)
        data = bytearray(archive.read_bytes())
        name_length = int.from_bytes(data[26:28], "little")
        extra_length = int.from_bytes(data[28:30], "little")
        compressed_start = 30 + name_length + extra_length
        # Invalid DEFLATE block type: ZipFile.testzip raises zlib.error here.
        data[compressed_start] = 0xFF
        archive.write_bytes(data)
        with self.assertRaises(bench.BenchmarkError):
            bench.inspect_archive(archive, self.item)

    def test_atomic_receipt_write_does_not_follow_predictable_temp_symlink(self):
        self.root.mkdir()
        receipt = self.root / "receipt.json"
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("must remain unchanged", encoding="utf-8")
        predictable_temp = receipt.with_suffix(receipt.suffix + ".tmp")
        try:
            predictable_temp.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        bench.write_json_atomic(receipt, {"source_id": self.item["id"]})
        self.assertEqual(outside.read_text(encoding="utf-8"), "must remain unchanged")
        self.assertFalse(receipt.is_symlink())
        self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")),
                         {"source_id": self.item["id"]})

    def test_valid_zip_changed_since_receipt_is_detected(self):
        self.download()
        target, _, _ = self.paths()
        make_zip(target, b"different valid data")
        with self.assertRaisesRegex(bench.BenchmarkError, "changed since download"):
            bench.verify_file(self.item, self.root)

    def test_verified_existing_archive_skips_downloader(self):
        first = self.download()
        downloader = mock.Mock(side_effect=AssertionError("must not download twice"))
        self.assertEqual(self.download(downloader), first)
        downloader.assert_not_called()

    def test_existing_archive_without_receipt_is_not_silently_adopted(self):
        target, _, _ = self.paths()
        make_zip(target)
        downloader = mock.Mock(side_effect=AssertionError("must not overwrite existing archive"))
        with self.assertRaisesRegex(bench.BenchmarkError, "Missing provenance receipt"):
            self.download(downloader)
        downloader.assert_not_called()

    def test_source_revision_uses_different_staging_path(self):
        _, first_staging, _ = self.paths()
        revised = dict(self.item, id="revised-valid-drive-id")
        _, second_staging, _ = bench.archive_paths(revised, self.root)
        self.assertNotEqual(first_staging, second_staging)

    def test_symlink_parent_cannot_escape_data_root(self):
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        self.root.mkdir()
        try:
            (self.root / "archives").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        downloader = mock.Mock(side_effect=AssertionError("unsafe download"))
        with self.assertRaisesRegex(bench.BenchmarkError, "escapes the data root"):
            self.download(downloader)
        downloader.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
