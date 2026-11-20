"""python -m denoising_eval: inspect CRVD or evaluate a supplied model adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .backends import IdentityBaseline, load_adapter
from .runner import evaluate, inspect_dataset, load_config, prepare, runtime_info, sha256_file, write_json, update_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'evaluate'])
    parser.add_argument('--config', type=Path, default=Path('configs/crvd_raw_test.json'))
    parser.add_argument('--data-root', type=Path, required=True, help='Contains indoor_raw_gt and indoor_raw_noisy')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are refused')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--adapter', help='Trusted Python module:factory implementing the documented API')
    group.add_argument('--baseline', choices=['identity'], help='Input-copy diagnostic, not MSTMN')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-name', help='Descriptive model label, recorded verbatim; does not verify provenance')
    parser.add_argument('--device', default='cpu', help='Passed to adapter, e.g. cuda:0; baseline uses CPU')
    parser.add_argument('--max-sequences', type=int, help='Smoke test only; marked partial if truncated')
    parser.add_argument('--save-predictions', action='store_true', help='Save normalized float32 Bayer mosaics as NPY')
    args = parser.parse_args(argv)
    if args.action == 'evaluate' and not (args.adapter or args.baseline):
        parser.error('evaluate requires --adapter or the explicit --baseline identity diagnostic')
    if args.adapter and (not args.checkpoint or not args.model_name):
        parser.error('--adapter requires --checkpoint and --model-name')
    if args.baseline and args.checkpoint:
        parser.error('The identity diagnostic does not use a checkpoint')
    if args.action == 'inspect' and (args.adapter or args.baseline or args.checkpoint or args.max_sequences or args.save_predictions):
        parser.error('Model and smoke-test options apply only to evaluate')
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error('--max-sequences must be positive')
    prepared = False
    try:
        config = load_config(args.config)
        if args.adapter and not args.checkpoint.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {args.checkpoint}')
        records = prepare(args.data_root, config, args.output)
        prepared = True
        inspection = inspect_dataset(records, config)
        write_json(args.output / 'inspection.json', {**inspection, 'runtime': runtime_info()})
        if args.action == 'inspect':
            print(f"Inspected {inspection['frame_pairs']} pairs / {inspection['sequence_count']} sequences")
            return 0
        if args.adapter:
            checkpoint_hash = sha256_file(args.checkpoint)
            predictor = load_adapter(args.adapter, args.checkpoint, args.device, config)
            model = {'kind': 'custom_adapter', 'name': args.model_name, 'adapter': args.adapter,
                     'checkpoint': str(args.checkpoint.resolve()), 'sha256': checkpoint_hash,
                     'device': args.device}
        else:
            predictor = IdentityBaseline()
            model = {'kind': 'diagnostic_baseline', 'name': 'identity_input_copy', 'device': 'cpu'}
        result = evaluate(records, config, predictor, args.output, model=model,
                          max_sequences=args.max_sequences, save_predictions=args.save_predictions)
        print(f"Completed {result['evaluated_frames']} scored frames; output: {args.output}")
        print('Official MSTMN reproduction: NOT VERIFIED. See summary.json protocol notes.')
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        if prepared:
            failure = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                       'failed_utc': datetime.now(timezone.utc).isoformat()}
            update_json(args.output / 'failure.json', failure)
            run_path = args.output / 'run.json'
            if run_path.is_file():
                state = json.loads(run_path.read_text(encoding='utf-8'))
                update_json(run_path, {**state, **failure})
        # Preserve partial artifacts for diagnosis; never fall back to identity.
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
