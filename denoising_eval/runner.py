"""Streaming RAW evaluation with explicit recurrent sequence boundaries."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from .backends import Predictor
from .data import build_manifest, normalize_raw, pack_gbrg, read_raw, unpack_gbrg
from .metrics import compute_metrics, summarize_records


CONFIG_KEYS = {
    'schema_version', 'scenes', 'iso_values', 'noisy_indices', 'black_level',
    'white_level', 'clip_input', 'clip_prediction', 'clip_target', 'crop_border',
    'warmup_frames', 'protocol_status', 'protocol_notes',
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding='utf-8'))
    if set(config) != CONFIG_KEYS or config['schema_version'] != 1:
        raise ValueError('Config must contain exactly the documented schema v1 keys')
    for key in ('scenes', 'iso_values', 'noisy_indices'):
        values = config[key]
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError(f'{key} must be a nonempty integer list')
        if len(values) != len(set(values)):
            raise ValueError(f'{key} contains duplicates')
    if not set(config['scenes']) <= set(range(7, 12)):
        raise ValueError('Only test scenes 7-11 are supported')
    if not set(config['iso_values']) <= {1600, 3200, 6400, 12800, 25600}:
        raise ValueError('Unknown CRVD ISO')
    if not set(config['noisy_indices']) <= set(range(10)):
        raise ValueError('CRVD noisy indices must be 0-9')
    for key in ('clip_input', 'clip_prediction', 'clip_target'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    for key in ('black_level', 'white_level'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f'{key} must be finite')
    if config['white_level'] <= config['black_level']:
        raise ValueError('white_level must exceed black_level')
    for key in ('crop_border', 'warmup_frames'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if config['warmup_frames'] >= 7:
        raise ValueError('warmup_frames must leave at least one scored frame')
    if not isinstance(config['protocol_status'], str) or not config['protocol_status']:
        raise ValueError('protocol_status must be a nonempty string')
    if not isinstance(config['protocol_notes'], list) or not all(
        isinstance(note, str) for note in config['protocol_notes']
    ):
        raise ValueError('protocol_notes must be a string list')
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Keep JSON standard-compliant, including perfect-reconstruction PSNR."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return 'Infinity' if value == math.inf else '-Infinity' if value == -math.inf else 'NaN'
    return value


def write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def update_json(path: Path, value) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(data_root: Path, config: dict, output: Path) -> list:
    data_root = data_root.expanduser().resolve()
    records = build_manifest(data_root, config['scenes'], config['iso_values'], config['noisy_indices'])
    # Refuse overwrites and writing results into the dataset itself.
    if output.resolve().is_relative_to(data_root.resolve()):
        raise ValueError('Output must be outside the dataset root')
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for record in records:
        key = record.key
        manifest.append({
            'scene': key.scene, 'iso': key.iso, 'noisy_index': key.noisy_index,
            'frames': [{'frame': pair.frame,
                        'noisy': pair.noisy.relative_to(data_root).as_posix(),
                        'gt': pair.gt.relative_to(data_root).as_posix()} for pair in record.frames],
        })
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    return records


def inspect_dataset(records: list, config: dict) -> dict:
    """Inspect every selected TIFF header; decode one representative pair."""
    metadata, unique = {}, set()
    for record in records:
        for pair in record.frames:
            shapes = []
            for label, path in (('noisy', pair.noisy), ('gt', pair.gt)):
                with tifffile.TiffFile(path) as tif:
                    if len(tif.pages) != 1:
                        raise ValueError(f'Expected single-page RAW TIFF: {path}')
                    page = tif.pages[0]
                    shape, dtype = tuple(page.shape), np.dtype(page.dtype)
                if len(shape) != 2 or any(v % 2 for v in shape) or dtype.kind not in 'uif':
                    raise ValueError(f'Unsupported RAW TIFF header: {path} {shape} {dtype}')
                if min(shape) - 2 * config['crop_border'] < 11:
                    raise ValueError(f'Image too small for configured crop and SSIM: {path}')
                shapes.append(shape)
                unique.add(path)
                entry = metadata.setdefault(label, set())
                entry.add((shape, str(dtype)))
            if shapes[0] != shapes[1]:
                raise ValueError(f'GT/noisy shape mismatch: {pair.noisy}')
    sample = records[0].frames[0]
    stats = {}
    for label, path in (('noisy', sample.noisy), ('gt', sample.gt)):
        image = read_raw(path)
        normalized = normalize_raw(image, config['black_level'], config['white_level'], clip=False)
        stats[label] = {'dtype_decoded': str(image.dtype), 'dn_min': float(image.min()),
                        'dn_max': float(image.max()), 'normalized_min': float(normalized.min()),
                        'normalized_max': float(normalized.max()), 'path': str(path)}
    return {'sequence_count': len(records), 'frame_pairs': sum(len(r.frames) for r in records),
            'unique_selected_tiffs': len(unique),
            'headers': {key: [{'shape': shape, 'dtype': dtype} for shape, dtype in sorted(values)]
                        for key, values in metadata.items()},
            'decoded_sample': stats, 'pixel_validation_scope': 'first pair only; all selected TIFF headers inspected'}


def runtime_info() -> dict:
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name)
                         for name in ('numpy', 'tifffile', 'scikit-image')},
            'evaluator_source_sha256': {p.name: sha256_file(p)
                                        for p in sorted(Path(__file__).parent.glob('*.py'))}}


def evaluate(records: list, config: dict, predictor: Predictor, output: Path, *,
             model: dict, max_sequences: int | None = None, save_predictions: bool = False) -> dict:
    if max_sequences is not None and max_sequences < 1:
        raise ValueError('max_sequences must be positive')
    selected = records if max_sequences is None else records[:max_sequences]
    full_selection = (set(config['scenes']) == set(range(7, 12)) and
                      set(config['iso_values']) == {1600, 3200, 6400, 12800, 25600} and
                      set(config['noisy_indices']) == set(range(10)))
    is_partial = len(selected) != len(records) or not full_selection
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'model': model, 'runtime': runtime_info(), 'protocol': config,
             'official_reproduction': False, 'partial_run': is_partial,
             'selection_is_full_release_test': full_selection,
             'scope': 'RAW Bayer metrics only; official code/weights and protocol equivalence are unverified'}
    write_json(output / 'run.json', state)
    rows = []
    columns = ['sequence_id', 'scene', 'iso', 'noisy_index', 'frame', 'mse', 'psnr', 'ssim']
    with (output / 'frames.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in selected:
            key = record.key
            sid = f'scene{key.scene}/ISO{key.iso}/noisy{key.noisy_index}'
            predictor.reset({'scene': key.scene, 'iso': key.iso, 'noisy_index': key.noisy_index,
                             'sequence_id': sid})
            for offset, pair in enumerate(record.frames):
                noisy = normalize_raw(read_raw(pair.noisy), config['black_level'],
                                      config['white_level'], clip=config['clip_input'])
                target = normalize_raw(read_raw(pair.gt), config['black_level'],
                                       config['white_level'], clip=config['clip_target'])
                if target.shape != noisy.shape:
                    raise ValueError(f'GT/noisy shape mismatch: {pair.noisy}')
                packed = pack_gbrg(noisy)
                # Adapter sees only the current observation and sequence metadata.
                prediction = np.asarray(predictor.predict(packed, frame_index=pair.frame))
                if prediction.shape != packed.shape or prediction.dtype.kind not in 'uif':
                    raise ValueError(f'Adapter output must be packed HWC {packed.shape}, got {prediction.shape}')
                if not np.isfinite(prediction).all():
                    raise ValueError(f'Nonfinite adapter output: {sid}, frame {pair.frame}')
                restored = unpack_gbrg(prediction)
                if config['clip_prediction']:
                    restored = np.clip(restored, 0.0, 1.0)
                if save_predictions:
                    path = output / 'predictions' / sid / f'frame{pair.frame}.npy'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, restored.astype(np.float32), allow_pickle=False)
                # Warm-up observations still update model state, but are not scored.
                if offset < config['warmup_frames']:
                    continue
                border = config['crop_border']
                if border:
                    restored, target = restored[border:-border, border:-border], target[border:-border, border:-border]
                metrics = compute_metrics(restored, target)
                row = {'sequence_id': sid, 'scene': key.scene, 'iso': key.iso,
                       'noisy_index': key.noisy_index, 'frame': pair.frame, **metrics}
                rows.append(row)
                writer.writerow(row)
            handle.flush()
            print(f'Evaluated {sid}: {len(record.frames)} input frames', flush=True)
    state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                 evaluated_sequences=len(selected), evaluated_frames=len(rows))
    summary = {**state, 'metrics': summarize_records(rows),
               'paper_reference_only': {'table': 'III', 'model': 'MSTMN',
                                        'raw_psnr_db': 42.92, 'raw_ssim': 0.9838,
                                        'comparison_validated': False}}
    write_json(output / 'summary.json', summary)
    update_json(output / 'run.json', state)
    return summary
