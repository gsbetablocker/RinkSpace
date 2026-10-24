"""Compare explicit native graph references at one complete numerical state."""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(before_file, after_file):
    before_file, after_file = Path(before_file), Path(after_file)
    a, b = (json.loads(p.read_text('utf8')) for p in (before_file, after_file))
    same = ('schema', 'objective_id', 'raw_rows', 'active_rows', 'active_raw_ids_sha256',
            'actual_residual_blocks', 'ordered_block_sizes_sha256', 'residual_order',
            'numerical_parameter_blocks', 'numerical_parameter_scalars',
            'numerical_parameters_sha256', 'all_raw_station_sha256',
            'all_raw_row_sha256', 'all_raw_alpha_sha256', 'all_raw_path_i32_sha256')
    failures = [key for key in same if a[key] != b[key]]
    arrays = {}
    for key in ('target_residuals', 'constraint_g', 'constraint_h'):
        x, y = a[key], b[key]
        xp, yp = before_file.parent / x['path'], after_file.parent / y['path']
        if digest(xp) != x['sha256'] or digest(yp) != y['sha256']:
            raise ValueError('Explicit reference bytes changed: ' + key)
        if xp.stat().st_size != x['count'] * 8 or yp.stat().st_size != y['count'] * 8:
            raise ValueError('Explicit reference array size differs: ' + key)
        if x['count'] != y['count']:
            failures.append(key + '.count')
            continue
        maximum = 0.0
        violations = 0
        u = np.memmap(xp, dtype='<f8', mode='r') if x['count'] else np.empty(0)
        v = np.memmap(yp, dtype='<f8', mode='r') if y['count'] else np.empty(0)
        for start in range(0, len(u), 1 << 18):
            ux, vx = u[start:start + (1 << 18)], v[start:start + (1 << 18)]
            if not np.isfinite(ux).all() or not np.isfinite(vx).all():
                raise ValueError('Nonfinite explicit reference: ' + key)
            difference = np.abs(ux - vx)
            maximum = max(maximum, float(difference.max(initial=0)))
            violations += int(np.count_nonzero(difference > 1e-12 + 1e-12 * np.maximum(np.abs(ux), np.abs(vx))))
        arrays[key] = dict(count=len(u), byte_equal=x['sha256'] == y['sha256'], max_abs_difference=maximum, tolerance_violations=violations)
        if violations:
            failures.append(key + '.values')
    costs = [a['actual_cost'], b['actual_cost']]
    cost_difference = abs(costs[0] - costs[1])
    if not all(math.isfinite(x) for x in costs) or cost_difference > 1e-9 + 1e-12 * max(map(abs, costs)):
        failures.append('actual_cost')
    for key in ('global_constraints', 'global_phr_state'):
        if a[key] != b[key]:
            failures.append(key)
    return dict(schema='rinkfit.current_state_numerical_comparison.v1',
                before=str(before_file.resolve()), after=str(after_file.resolve()),
                before_sha256=digest(before_file), after_sha256=digest(after_file),
                passed=not failures, failures=failures, arrays=arrays,
                before_cost=costs[0], after_cost=costs[1], absolute_cost_difference=cost_difference,
                full_raw_numerical_state_equal=not any(x.startswith(('all_raw_', 'numerical_')) for x in failures),
                state_identity_equal=a['complete_state_token'] == b['complete_state_token'],
                tolerance=dict(absolute=1e-12, relative=1e-12),
                full_arrays_compared=True, quality_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', 'utf8')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
