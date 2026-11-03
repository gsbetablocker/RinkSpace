"""Replay compact opening parameters against saved exports, without fitting.

Preserved from verify_saved.py. Expected identities come from the existing
programme binding rather than a hardcoded local drive or frozen hash literal.
Only the explicitly selected new report is written.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from rinkspace.path_policy import guard_project_write
from .opening_repair import setup, OpeningWindow
from .rebuild_prepare import load, digest


def run(root, output, *, report_path, project=None, parameters=None, binding_path=None):
    project = Path(project or Path(__file__).resolve().parents[3]).resolve()
    root, output = Path(root).resolve(), Path(output).resolve()
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    parameters = Path(parameters or programme/'opening_repair_1102.npz').resolve()
    binding_path = Path(binding_path or parameters.with_suffix('.json')).resolve()
    binding = json.loads(binding_path.read_text(encoding='utf8'))
    report_path = guard_project_write(report_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    # Prefer the current exact path; retain a unique historical source key
    # after relocating the Data root, without rewriting the original binding.
    hashes = binding['input_contract']['reference_hashes']
    input_key = str(root/'input.npz')
    if input_key in hashes:
        expected_input_hash = hashes[input_key]
    else:
        inputs = [value for key, value in hashes.items()
                  if key.replace('\\', '/').endswith('/body_s01_20261029/input.npz')]
        if len(inputs) != 1:
            raise ValueError('The binding must uniquely identify body_s01_20261029/input.npz')
        expected_input_hash = inputs[0]

    compact = load(parameters)
    reference, _, info, _ = setup(project, root, output)
    window = OpeningWindow(reference, (22.12, 25.), 'full')
    for actual, expected in (
        (window.knots, compact['knots']), (window.active, compact['active']),
        (reference.anchor, compact['initial_anchor']), (reference.heading, compact['heading'])
    ):
        np.testing.assert_array_equal(actual, expected)
    reference.refinements.append((window, compact['x']))

    packet = load(output/'fitted/result.npz')
    parts = {key: [] for key in ('P', 'Q', 'local_R', 'native', 'gear')}
    for start in range(0, len(reference.times), 32):
        state = reference.world(reference.at(reference.times[start:start+32]))
        for key in parts:
            parts[key].append(state[key])
    ratio = info['length_export_ratio']
    errors = {}
    for key, target, scale in (
        ('P', 'P_world_mid', ratio), ('Q', 'Q_world', 1),
        ('local_R', 'local_R_mid', 1), ('native', 'native22_world_mid', ratio),
        ('gear', 'gear_world_mid', ratio)
    ):
        value = np.concatenate(parts[key])*scale
        errors[key] = float(np.max(abs(value-packet[target])))
        np.testing.assert_allclose(value, packet[target], rtol=0, atol=2e-12)

    report = json.loads((output/'opening_report.json').read_text(encoding='utf8'))
    camera_hash = digest(info['state_path'])
    parent_hash = digest(root/'boot_integration_1102/fitted/result.npz')
    input_hash = digest(root/'input.npz')
    record = dict(
        schema='rinkspace.opening-compact-verification.v1',
        all_report_invariants_passed=report['invariants']['all_passed'],
        one_drive_parameter_replay_max_absolute=errors,
        compact_parameter_sha256=digest(parameters), camera_state_sha256=camera_hash,
        camera_matches_frozen_1102=camera_hash == binding['camera']['state_sha256'],
        parent_boot_result_sha256=parent_hash,
        boot_parent_unchanged=parent_hash == binding['input_contract']['parent_result_sha256'],
        input_sha256=input_hash, body_input_unchanged=input_hash == expected_input_hash,
        source_pts_preserved=bool(np.array_equal(packet['pts'], reference.p['pts'])),
        derivative_checks=json.loads((output/'derivative_checks.json').read_text(encoding='utf8')),
        binding=dict(path=str(binding_path), sha256=digest(binding_path)))
    historical = output/'verify_saved.py'
    if historical.is_file():
        record['historical_script'] = dict(path=str(historical), sha256=digest(historical))
    record['passed'] = all(record[key] for key in (
        'all_report_invariants_passed', 'camera_matches_frozen_1102',
        'boot_parent_unchanged', 'body_input_unchanged', 'source_pts_preserved'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project')
    parser.add_argument('--root', required=True, help='Existing body_s01 run directory')
    parser.add_argument('--output', required=True, help='Existing opening repair directory')
    parser.add_argument('--parameters', help='Compact parameter NPZ; defaults to programme binding')
    parser.add_argument('--binding', help='Identity binding JSON; defaults to parameters with .json suffix')
    parser.add_argument('--report', required=True, help='New report file; existing files are refused')
    args = parser.parse_args(argv)
    record = run(args.root, args.output, report_path=args.report, project=args.project,
                 parameters=args.parameters, binding_path=args.binding)
    print(json.dumps(dict(report=str(Path(args.report).resolve()), passed=record['passed'],
                         replay_max_absolute=record['one_drive_parameter_replay_max_absolute']), ensure_ascii=False))
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
