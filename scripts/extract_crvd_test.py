#!/usr/bin/env python3
"""Extract the downloaded CRVD test ZIPs without replacing existing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path


def digest_file(path: Path) -> tuple[int, str, int]:
    size, crc = 0, 0
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            size += len(chunk)
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, digest.hexdigest(), crc & 0xffffffff


def reject_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            attrs = part.lstat()
            if stat.S_ISLNK(attrs.st_mode) or (
                getattr(attrs, 'st_file_attributes', 0) & 0x400
            ):
                raise ValueError(f'Refusing symlink/reparse path: {part}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.absolute()
    reject_links(root)
    inventory_path = root / 'receipts/crvd-test-inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if (inventory['dataset'], inventory['split'], inventory['scenes']) != (
        'crvd', 'test', [7, 8, 9, 10, 11]
    ):
        raise ValueError('Expected the CRVD scene 7-11 download inventory')
    expected_names = {
        f'{collection}_scene{scene}.zip'
        for collection in ('indoor_raw_gt', 'indoor_raw_noisy')
        for scene in range(7, 12)
    }
    if len(inventory['archives']) != 10 or {
        a['name'] for a in inventory['archives']
    } != expected_names:
        raise ValueError('Incomplete/duplicate archive selection')
    destination = root / 'crvd'
    receipt = root / 'receipts/crvd-test-extraction.json'
    if destination.exists() or destination.is_symlink() or receipt.exists():
        raise FileExistsError('Extraction destination/receipt exists; refusing overwrite')
    reject_links(receipt.parent)
    total_bytes = 0
    plans = []
    # Inspect all entries before creating any extracted files. Restrict paths to
    # this dataset layout, also excluding Windows drive names and ADS syntax.
    for item in inventory['archives']:
        collection, scene = re.fullmatch(
            r'(indoor_raw_gt|indoor_raw_noisy)_scene(7|8|9|10|11)\.zip', item['name']
        ).groups()
        archive = root / 'archives/crvd' / collection / item['name']
        reject_links(archive)
        size, sha256, _ = digest_file(archive)
        if size != item['size_bytes'] or sha256 != item['sha256']:
            raise ValueError(f'Archive changed since download: {archive}')
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            seen, iso_counts, files = set(), {}, 0
            for info in entries:
                pattern = rf'scene{scene}/(?:ISO(1600|3200|6400|12800|25600)/(?:frame[0-9]+_[A-Za-z0-9_]+\.tiff)?)?'
                match = re.fullmatch(pattern, info.filename)
                if not match or (not info.is_dir() and match.group(1) is None):
                    raise ValueError(f'Unexpected ZIP path: {info.filename}')
                key = info.filename.rstrip('/').casefold()
                if key in seen:
                    raise ValueError(f'Duplicate ZIP path: {info.filename}')
                seen.add(key)
                mode = stat.S_IFMT(info.external_attr >> 16)
                if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError(f'Special ZIP entry: {info.filename}')
                if not info.is_dir():
                    files += 1
                    iso = match.group(1)
                    iso_counts[iso] = iso_counts.get(iso, 0) + 1
                    total_bytes += info.file_size
            if files != item['files'] or iso_counts != item['iso_counts']:
                raise ValueError(f'File/ISO count mismatch: {archive}')
        plans.append((item, collection, archive))
    if shutil.disk_usage(root).free < total_bytes + 1024**3:
        raise OSError('Insufficient free space including 1 GiB safety margin')
    staging_parent = root / '.staging'
    reject_links(staging_parent)
    staging_parent.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='crvd-test-extract-', dir=staging_parent))
    report = {
        'schema_version': 1, 'dataset': 'crvd', 'split': 'test',
        'scenes': [7, 8, 9, 10, 11], 'iso_values': inventory['iso_values'],
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': str(root), 'extracted_root': str(destination),
        'original_archives_preserved': True,
        'validation': 'Archive local SHA-256, ZIP stream CRC; extracted files reread for size, CRC and SHA-256; exact file list',
        'archives': [], 'files': [],
    }
    print(f'Preflight passed: 10 ZIPs, {total_bytes} uncompressed bytes', flush=True)
    print(f'Staging: {staging}', flush=True)
    # On failure keep the unique staging tree for inspection, without publishing it.
    for item, collection, archive in plans:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                target = staging / collection / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with source.open(info) as reader, target.open('xb') as writer:
                    for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                        writer.write(chunk)
                        digest.update(chunk)
                size, sha256, crc = digest_file(target)
                if (size, sha256, crc) != (info.file_size, digest.hexdigest(), info.CRC):
                    raise ValueError(f'Extracted file verification failed: {target}')
                report['files'].append({
                    'path': target.relative_to(staging).as_posix(),
                    'size_bytes': size, 'sha256': sha256, 'crc32': f'{crc:08x}',
                    'source_archive': item['name'],
                })
        report['archives'].append({
            'name': item['name'], 'source_id': item['source_id'],
            'sha256': item['sha256'], 'files': item['files'],
            'iso_counts': item['iso_counts'],
        })
        print(f"Extracted and verified: {item['name']} ({item['files']} files)", flush=True)
    observed = {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file()}
    if observed != {f['path'] for f in report['files']}:
        raise ValueError('Extracted file list mismatch')
    report['file_count'] = len(report['files'])
    report['total_extracted_bytes'] = sum(f['size_bytes'] for f in report['files'])
    if report['file_count'] != 2100 or report['total_extracted_bytes'] != total_bytes:
        raise ValueError('Extracted totals mismatch')
    # A directory rename on the same drive publishes only the complete dataset.
    # All paths were exclusive/new, and the existing final directory is refused.
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    with receipt.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(f'COMPLETE: {report["file_count"]} files, {total_bytes} bytes -> {destination}', flush=True)
    print(f'Receipt: {receipt}', flush=True)


if __name__ == '__main__':
    main()
