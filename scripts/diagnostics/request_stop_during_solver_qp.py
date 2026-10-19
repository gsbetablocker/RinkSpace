"""Send one explicit diagnostic stop when this run enters its next QP backend.

Run only after the caller has finished the intended complete-cycle checks.
This is a one-shot control test, not a production iteration or time limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--save-current-linearization-cache', action='store_true',
                        help='Use the original fresh-normal handoff before exiting')
    args = parser.parse_args()
    run = args.run.resolve()
    bootstrap = Path(str(run) + '.bootstrap')
    launch = json.loads((bootstrap / 'launch.json').read_text(encoding='utf-8-sig'))
    if Path(launch['output']).resolve() != run:
        raise ValueError('The explicit run must match its launch receipt')
    control_path = run / 'solver' / 'control.json'
    if control_path.exists():
        raise ValueError('An existing control instruction must not be replaced')
    report_path = bootstrap / 'actual_QP_stop_request.json'
    if report_path.exists():
        raise ValueError('A previous stop test must not be overwritten')
    with (bootstrap / 'stdout.log').open(encoding='utf8') as stream:
        stream.seek(0, 2)
        pending = ''
        while True:
            part = stream.readline()
            if not part:
                if (run / 'solver' / 'solver_result.json').exists():
                    raise RuntimeError('The solver completed before the next QP test')
                time.sleep(.05)
                continue
            pending += part
            if not pending.endswith('\n'):
                continue
            line, pending = pending, ''
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('stage') != 'QP_backend_begin':
                continue
            checkpoint = run / 'solver' / 'latest_accepted_state.npz'
            with np.load(checkpoint, allow_pickle=False) as saved:
                before = dict(q_sha256=hashlib.sha256(saved['q'].tobytes()).hexdigest(),
                    damping=float(saved['damping']),
                    checkpoint_file_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            cycles = json.loads((run / 'actual_accepted_cycle_performance.json').read_text(encoding='utf8'))['cycles']
            request = dict(stop_branch=True,
                reason='Complete-cycle stability verified; test explicit stop inside the next real QP',
                save_current_linearization_cache=args.save_current_linearization_cache)
            before_write = time.time_ns()
            temporary = control_path.with_name('control.json.stop_test.tmp')
            temporary.write_text(json.dumps(request, ensure_ascii=False) + '\n', encoding='utf8')
            if control_path.exists():
                raise ValueError('Another explicit control instruction arrived; retain it')
            temporary.replace(control_path)
            after_write = time.time_ns()
            report = dict(schema='rinkspace.actual-QP-explicit-stop-request.v1',
                run=str(run), backend_begin=event,
                request_before_write_wall_time_ns=before_write,
                request_after_write_wall_time_ns=after_write,
                last_saved_accepted_state=before, completed_accepted_cycles=len(cycles),
                request=request, production_timeout_or_iteration_limit_added=False)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print(json.dumps(report), flush=True)
            return


if __name__ == '__main__':
    main()
