#!/usr/bin/env python3
"""Launch the Windows training process independently of its SSH session."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/mstmn_retrain.json'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--flow-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--microbatch', type=int)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This detached launcher targets Windows; elsewhere run python -m mstmn_train.train directly')
    output = args.output.resolve()
    if (output / '.train.lock').exists():
        parser.error('A training lock exists; inspect its PID before starting another process')
    if output.exists() and args.resume is None:
        parser.error('Output exists: use an explicit compatible --resume or a new output')
    for path in (args.config, args.flow_checkpoint):
        if not path.is_file():
            parser.error(f'Missing input file: {path}')
    command = [sys.executable, '-u', '-m', 'mstmn_train.train',
               '--config', str(args.config.resolve()), '--data-root', str(args.data_root.resolve()),
               '--flow-checkpoint', str(args.flow_checkpoint.resolve()), '--output', str(output)]
    for option, value in (('--resume', args.resume), ('--max-steps', args.max_steps), ('--microbatch', args.microbatch)):
        if value is not None:
            command += [option, str(value.resolve() if isinstance(value, Path) else value)]
    logs = output.parent / 'launches'
    logs.mkdir(parents=True, exist_ok=True)
    run_id = output.name + '-' + uuid.uuid4().hex[:12]
    stdout, stderr = logs / (run_id + '.stdout.log'), logs / (run_id + '.stderr.log')
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    environment['PYTHONUTF8'] = '1'
    environment.setdefault('OMP_NUM_THREADS', '8')
    environment.setdefault('MKL_NUM_THREADS', '8')
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=environment,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 close_fds=True, creationflags=flags)
    receipt = {'state': 'spawned_not_yet_verified_running', 'pid': child.pid,
               'at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
               'output': str(output), 'stdout': str(stdout), 'stderr': str(stderr),
               'status': str(output / 'status.json'), 'ssh_independent': True}
    with (logs / (run_id + '.launch.json')).open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
