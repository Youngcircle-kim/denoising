#!/usr/bin/env python3
"""Plan or download the official MSTMN benchmarks; no download by default."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "benchmarks" / "sources.json"
GDOWN_VERSION = "6.5.2"
SPLITS = {"test": range(7, 12), "train": range(1, 7),
          "indoor": range(1, 12), "outdoor": range(1, 11)}


class BenchmarkError(Exception):
    """An actionable failure, including incomplete or corrupt downloads."""


def load_manifest(path: Path = MANIFEST) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise BenchmarkError("Unsupported benchmark manifest schema")
    seen_ids, seen_names = set(), set()
    collections = {"indoor_raw_gt", "indoor_raw_noisy", "outdoor_raw_noisy"}
    for item in manifest["datasets"]["crvd"]["files"]:
        if item["collection"] not in collections:
            raise BenchmarkError("Unknown CRVD collection in manifest")
        if type(item["scene"]) is not int or item["scene"] < 1:
            raise BenchmarkError("Invalid scene in manifest")
        expected = f"{item['collection']}_scene{item['scene']}.zip"
        if item["name"] != expected or not re.fullmatch(r"[\w-]{10,}", item["id"], re.ASCII):
            raise BenchmarkError("Unsafe filename or invalid Drive ID in manifest")
        if item["id"] in seen_ids or item["name"] in seen_names:
            raise BenchmarkError("Duplicate archive name or Drive ID")
        seen_ids.add(item["id"])
        seen_names.add(item["name"])
    return manifest


def build_plan(manifest: dict, dataset: str, split: str, scenes: list[int] | None,
               data_root: Path) -> dict:
    source = manifest["datasets"][dataset]
    base = {"dataset": dataset, "data_root": str(data_root.resolve()),
            "homepage": source["homepage"], "license": source["license"],
            "checked_on": manifest["checked_on"]}
    if dataset == "real-rawvsr":
        if scenes or split != "test":
            raise BenchmarkError("Real-RawVSR supports a manual test-data plan only")
        return {**base, "mode": "manual", "download_url": source["download_url"],
                "access_code": source["access_code"], "note": source["note"],
                "destination": str(data_root.resolve() / "manual" / dataset)}
    selected_scenes = sorted(scenes if scenes is not None else SPLITS[split])
    if not selected_scenes or len(set(selected_scenes)) != len(selected_scenes):
        raise BenchmarkError("Scene selection is empty or contains duplicates")
    if not set(selected_scenes).issubset(SPLITS[split]):
        raise BenchmarkError(f"Scenes must belong to split {split}: {list(SPLITS[split])}")
    collections = (["outdoor_raw_noisy"] if split == "outdoor"
                   else ["indoor_raw_gt", "indoor_raw_noisy"])
    files = []
    for scene in selected_scenes:
        for collection in collections:
            matches = [f for f in source["files"]
                       if f["scene"] == scene and f["collection"] == collection]
            if len(matches) != 1:
                raise BenchmarkError(f"Manifest missing/duplicating {collection}, scene {scene}")
            item = dict(matches[0])
            item["url"] = f"https://drive.google.com/uc?id={item['id']}"
            item["relative_path"] = str(Path("archives") / "crvd" / collection / item["name"])
            files.append(item)
    return {**base, "mode": "google-drive", "split": split, "scenes": selected_scenes,
            "iso_values": source["iso_values"], "archive_count": len(files), "files": files}


def print_plan(plan: dict) -> None:
    print(f"Dataset: {plan['dataset']} | License: {plan['license']}")
    print(f"Official source: {plan['homepage']}")
    print(f"Data root: {plan['data_root']}")
    if plan["mode"] == "manual":
        print(f"Manual download: {plan['download_url']}")
        print(f"Baidu access code: {plan['access_code']}")
        print(f"Keep the downloaded original files under: {plan['destination']}")
        print(plan["note"])
        return
    print(f"Split: {plan['split']} | Scenes: {plan['scenes']} | Archives: {plan['archive_count']}")
    print(f"ISO conditions (within archives; not filtered): {plan['iso_values']}")
    for item in plan["files"]:
        print(f"  {item['relative_path']} | Drive ID: {item['id']}")
    print("Publisher byte sizes and SHA-256 hashes are unknown. Total size is not estimated.")
    print("Archives are preserved; no extraction, preprocessing, training or evaluation is performed.")


def _inside(path: Path, root: Path) -> Path:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BenchmarkError(f"Path escapes the data root: {path}") from exc
    if path.is_symlink():
        raise BenchmarkError(f"Refusing a symlink: {path}")
    return path


def archive_paths(item: dict, root: Path) -> tuple[Path, Path, Path]:
    target = root / "archives" / "crvd" / item["collection"] / item["name"]
    # Include the source ID so a later manifest revision cannot reuse another source's partial file.
    staging = root / ".staging" / "crvd" / item["id"] / item["name"]
    receipt = root / "receipts" / "crvd" / item["collection"] / (item["name"] + ".json")
    return tuple(_inside(p, root) for p in (target, staging, receipt))


def inspect_archive(path: Path, item: dict) -> dict:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkError(f"Missing or non-regular archive: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [info for info in archive.infolist() if not info.is_dir()]
            if not entries:
                raise BenchmarkError(f"Empty ZIP archive: {path}")
            bad = archive.testzip()
            if bad is not None:
                raise BenchmarkError(f"ZIP CRC failure in {path}: {bad}")
            uncompressed = sum(info.file_size for info in entries)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
        raise BenchmarkError(f"Invalid/unreadable ZIP {path}: {exc}") from exc
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    size, sha256 = path.stat().st_size, digest.hexdigest()
    if item.get("size_bytes") is not None and size != item["size_bytes"]:
        raise BenchmarkError(f"Publisher size mismatch: {path}")
    if item.get("sha256") is not None and sha256 != item["sha256"]:
        raise BenchmarkError(f"Publisher SHA-256 mismatch: {path}")
    return {"size_bytes": size, "sha256": sha256, "zip_entries": len(entries),
            "zip_uncompressed_bytes": uncompressed}


def verify_file(item: dict, root: Path) -> dict:
    target, _, receipt_path = archive_paths(item, root)
    if not receipt_path.is_file():
        raise BenchmarkError(f"Missing provenance receipt: {receipt_path}; not silently adopting an existing ZIP")
    with receipt_path.open(encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("source_id") != item["id"] or receipt.get("name") != item["name"]:
        raise BenchmarkError(f"Source identity mismatch in receipt: {receipt_path}")
    observed = inspect_archive(target, item)
    if any(observed[key] != receipt.get(key) for key in ("size_bytes", "sha256")):
        raise BenchmarkError(f"Archive changed since download: {target}")
    return receipt


def write_json_atomic(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    tmp = Path(name)
    try:
        # Exclusive creation avoids following an existing temporary-file symlink.
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def download_file(item: dict, root: Path, downloader) -> dict:
    target, staging, receipt_path = archive_paths(item, root)
    if target.exists():
        receipt = verify_file(item, root)
        print(f"Verified existing: {item['name']}")
        return receipt
    for parent in (target.parent, staging.parent, receipt_path.parent):
        _inside(parent, root).mkdir(parents=True, exist_ok=True)
    if not staging.exists():
        print(f"Downloading: {item['name']}")
        result = downloader(id=item["id"], output=str(staging), resume=True,
                            use_cookies=False, retries=3, timeout=(15, 60))
        if result is None or not staging.is_file():
            raise BenchmarkError(f"Download incomplete: {item['name']}; re-run to resume partial files")
    # A completed staging file may remain after an interruption. Always validate it before publication.
    try:
        observed = inspect_archive(staging, item)
    except BenchmarkError as exc:
        raise BenchmarkError(f"{exc}. Preserve or remove invalid staging file {staging} "
                             "before retrying; partial .part files are kept for resume.") from exc
    receipt = {"schema_version": 1, "name": item["name"], "source_id": item["id"],
               "source_url": item["url"], **observed,
               "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
               "publisher_sha256": item.get("sha256"),
               "validation": "ZIP CRC plus local SHA-256; local hash is not a publisher checksum"}
    # Write provenance first: after a crash, a published final archive always has its receipt.
    write_json_atomic(receipt_path, receipt)
    if target.exists():
        raise BenchmarkError(f"Destination appeared during download; refusing overwrite: {target}")
    staging.rename(target)
    print(f"Validated: {item['name']} | {observed['size_bytes']} bytes | SHA256 {observed['sha256']}")
    return receipt


def get_downloader():
    try:
        version = importlib.metadata.version("gdown")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BenchmarkError("gdown is not installed. Use a Python >=3.10 environment and run: "
                             "python -m pip install -r requirements-download.txt") from exc
    if version != GDOWN_VERSION:
        raise BenchmarkError(f"Expected gdown=={GDOWN_VERSION}, found {version}. "
                             "Use requirements-download.txt in a separate download environment.")
    import gdown  # Optional dependency: only loaded for an explicit download action.
    return gdown.download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="plan", choices=["plan", "download", "verify"])
    parser.add_argument("--dataset", choices=["crvd", "real-rawvsr"], default="crvd")
    parser.add_argument("--split", choices=list(SPLITS), default="test")
    parser.add_argument("--scenes", nargs="+", type=int)
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--json", action="store_true", help="Offline plan as JSON")
    args = parser.parse_args(argv)
    try:
        if args.json and args.action != "plan":
            raise BenchmarkError("--json is supported only with plan")
        plan = build_plan(load_manifest(), args.dataset, args.split, args.scenes, args.data_root)
        if args.action == "plan":
            if args.json:
                print(json.dumps(plan, indent=2))
            else:
                print_plan(plan)
            return 0
        if plan["mode"] == "manual":
            print_plan(plan)
            raise BenchmarkError("Real-RawVSR requires the official Baidu download flow. "
                                 "This script does not automate Baidu authentication or claim a completed download.")
        root = Path(plan["data_root"])
        if args.action == "download":
            downloader = get_downloader()  # Check dependency before creating output directories.
            root.mkdir(parents=True, exist_ok=True)
            print(f"Free space now: {shutil.disk_usage(root).free / 2**30:.2f} GiB; required total is unknown")
            for item in plan["files"]:
                download_file(item, root, downloader)
        else:
            for item in plan["files"]:
                verify_file(item, root)
                print(f"Verified: {item['name']}")
        print(f"Validated {len(plan['files'])} selected ZIP archives. "
              "Internal frame/ISO coverage and benchmark readiness are not yet verified.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed archives are retained; re-run download to resume partial files.", file=sys.stderr)
        return 130
    except (BenchmarkError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
