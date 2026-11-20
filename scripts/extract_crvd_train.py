#!/usr/bin/env python3
"""Verify and extract CRVD training scenes 1-6 beside existing test data.

Requires all twelve downloaded archives and their provenance receipts. Existing
training scene directories or an extraction receipt are always refused. On an
interruption inspect retained staging/publication records before attempting any
recovery; this script never adopts or overwrites a partial training extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .download_benchmarks import archive_paths, build_plan, load_manifest
    from .extract_crvd_test import digest_file, reject_links
else:
    from download_benchmarks import archive_paths, build_plan, load_manifest
    from extract_crvd_test import digest_file, reject_links


ISOS = (1600, 3200, 6400, 12800, 25600)
COLLECTIONS = ('indoor_raw_gt', 'indoor_raw_noisy')


def expected_files(collection: str, scene: int) -> set[str]:
    if collection not in COLLECTIONS or scene not in range(1, 7):
        raise ValueError('Only paired CRVD training scenes 1-6 are supported')
    suffixes = (['clean_and_slightly_denoised'] if collection == 'indoor_raw_gt'
                else ['clean', *(f'noisy{i}' for i in range(10))])
    return {f'scene{scene}/ISO{iso}/frame{frame}_{suffix}.tiff'
            for iso in ISOS for frame in range(1, 8) for suffix in suffixes}


def validate_entries(entries: list[zipfile.ZipInfo], collection: str, scene: int) -> dict:
    """Allow only exact released RAW names, ISO conditions and normal entries."""
    allowed_files = expected_files(collection, scene)
    allowed_dirs = {f'scene{scene}/', *(f'scene{scene}/ISO{iso}/' for iso in ISOS)}
    seen, found = set(), set()
    iso_counts = {str(iso): 0 for iso in ISOS}
    total_bytes = 0
    for info in entries:
        name = info.filename
        key = name.rstrip('/').casefold()
        if key in seen:
            raise ValueError(f'Duplicate ZIP path: {name}')
        seen.add(key)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f'Special ZIP entry: {name}')
        if info.flag_bits & 1:
            raise ValueError(f'Encrypted ZIP entry: {name}')
        if info.is_dir():
            if name not in allowed_dirs or mode == stat.S_IFREG:
                raise ValueError(f'Unexpected ZIP directory: {name}')
            continue
        if name not in allowed_files or mode == stat.S_IFDIR:
            raise ValueError(f'Unexpected ZIP path: {name}')
        if info.file_size < 1:
            raise ValueError(f'Empty RAW entry: {name}')
        found.add(name)
        iso_counts[name.split('/')[1][3:]] += 1
        total_bytes += info.file_size
    if found != allowed_files:
        raise ValueError(f'Incomplete {collection} scene{scene}: '
                         f'{len(found)} files, expected {len(allowed_files)}')
    per_iso = 7 if collection == 'indoor_raw_gt' else 77
    if set(iso_counts.values()) != {per_iso}:
        raise ValueError(f'ISO counts mismatch: {collection} scene{scene}')
    return {'files': len(found), 'iso_counts': iso_counts,
            'zip_uncompressed_bytes': total_bytes}


def preflight(root: Path) -> tuple[list[dict], int]:
    """Validate the entire selection before creating staging/extracted files."""
    reject_links(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Data root does not exist: {root}')
    destination = root / 'crvd'
    receipt_path = root / 'receipts/crvd-train-extraction.json'
    reject_links(destination)
    reject_links(receipt_path)
    if receipt_path.exists():
        raise FileExistsError(f'Extraction receipt exists; inspect before recovery: {receipt_path}')
    for collection in COLLECTIONS:
        for scene in range(1, 7):
            final = destination / collection / f'scene{scene}'
            reject_links(final)
            if final.exists():
                raise FileExistsError(f'Training destination exists; refusing overwrite: {final}')
    plan = build_plan(load_manifest(), 'crvd', 'train', None, root)
    expected = {(collection, scene) for collection in COLLECTIONS for scene in range(1, 7)}
    if len(plan['files']) != 12 or {(i['collection'], i['scene']) for i in plan['files']} != expected:
        raise ValueError('Expected the complete twelve-archive training selection')
    plans, total_bytes = [], 0
    for item in plan['files']:
        archive, _, provenance_path = archive_paths(item, root)
        reject_links(archive)
        reject_links(provenance_path)
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if provenance.get('source_id') != item['id'] or provenance.get('name') != item['name']:
            raise ValueError(f'Wrong archive provenance: {provenance_path}')
        size, sha256, _ = digest_file(archive)
        if size != provenance.get('size_bytes') or sha256 != provenance.get('sha256'):
            raise ValueError(f'Archive changed since download: {archive}')
        with zipfile.ZipFile(archive) as source:
            stats = validate_entries(source.infolist(), item['collection'], item['scene'])
        if (provenance.get('zip_entries') != stats['files']
                or provenance.get('zip_uncompressed_bytes') != stats['zip_uncompressed_bytes']):
            raise ValueError(f'Archive metadata differs from download receipt: {archive}')
        plans.append({'item': item, 'archive': archive, 'sha256': sha256,
                      'size_bytes': size, **stats})
        total_bytes += stats['zip_uncompressed_bytes']
    if shutil.disk_usage(root).free < total_bytes + 1024**3:
        raise OSError('Insufficient free space including 1 GiB safety margin')
    return plans, total_bytes


def extract(root: Path) -> dict:
    root = root.expanduser().absolute()
    plans, total_bytes = preflight(root)
    staging_parent = root / '.staging'
    reject_links(staging_parent)
    staging_parent.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='crvd-train-extract-', dir=staging_parent))
    destination = root / 'crvd'
    receipt_path = root / 'receipts/crvd-train-extraction.json'
    report = {
        'schema_version': 1, 'dataset': 'crvd', 'split': 'train',
        'scenes': list(range(1, 7)), 'iso_values': list(ISOS),
        'status': 'staging', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': str(root), 'extracted_root': str(destination), 'staging': str(staging),
        'original_archives_preserved': True, 'existing_test_scenes_preserved': True,
        'validation': 'Source ID and local archive SHA-256 matched download receipt; '
                      'ZIP stream CRC; extracted files reread for size, CRC and SHA-256; '
                      'exact scene/ISO/frame file lists. Local hashes are not publisher checksums.',
        'archives': [], 'files': [], 'published_directories': [],
    }
    print(f'Preflight passed: 12 ZIPs, {total_bytes} uncompressed bytes', flush=True)
    print(f'Staging: {staging}', flush=True)
    # All files remain in unique staging until the twelve archives pass.
    for plan in plans:
        item, archive = plan['item'], plan['archive']
        collection = item['collection']
        with zipfile.ZipFile(archive) as source:
            stats = validate_entries(source.infolist(), collection, item['scene'])
            if stats != {k: plan[k] for k in ('files', 'iso_counts', 'zip_uncompressed_bytes')}:
                raise ValueError(f'ZIP metadata changed after preflight: {archive}')
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
            'name': item['name'], 'collection': collection, 'scene': item['scene'],
            'source_id': item['id'], 'source_url': item['url'],
            'sha256': plan['sha256'], 'size_bytes': plan['size_bytes'],
            **{k: plan[k] for k in ('files', 'iso_counts', 'zip_uncompressed_bytes')},
        })
        print(f"Extracted and verified: {item['name']} ({plan['files']} files)", flush=True)
    expected = {f['path'] for f in report['files']}
    observed = {p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file()}
    report['file_count'] = len(report['files'])
    report['total_extracted_bytes'] = sum(f['size_bytes'] for f in report['files'])
    if observed != expected or report['file_count'] != 2520 or report['total_extracted_bytes'] != total_bytes:
        raise ValueError('Extracted file list/totals mismatch')
    finals = [destination / c / f'scene{s}' for c in COLLECTIONS for s in range(1, 7)]
    for final in finals:
        reject_links(final)
        if final.exists():
            raise FileExistsError(f'Destination appeared during extraction: {final}')
        final.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new receipt before publishing. A crash/rename failure leaves an
    # explicit non-completed record and blocks reruns from overwriting anything.
    reject_links(receipt_path)
    with receipt_path.open('x', encoding='utf-8') as receipt:
        def record_status() -> None:
            receipt.seek(0)
            json.dump(report, receipt, indent=2)
            receipt.write('\n')
            receipt.truncate()
            receipt.flush()
        report['status'] = 'publishing'
        record_status()
        try:
            for final in finals:
                reject_links(final)
                if final.exists():
                    raise FileExistsError(f'Refusing overwrite during publication: {final}')
                (staging / final.relative_to(destination)).rename(final)
                report['published_directories'].append(final.relative_to(destination).as_posix())
                record_status()
        except Exception as exc:
            report.update(status='publication_failed', error=f'{type(exc).__name__}: {exc}')
            record_status()
            raise
        report.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
        record_status()
    # Only empty directories remain after the twelve same-volume renames.
    for collection in COLLECTIONS:
        (staging / collection).rmdir()
    staging.rmdir()
    print(f'COMPLETE: {report["file_count"]} files, {total_bytes} bytes -> {destination}', flush=True)
    print(f'Receipt: {receipt_path}', flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    extract(args.data_root)


if __name__ == '__main__':
    main()
