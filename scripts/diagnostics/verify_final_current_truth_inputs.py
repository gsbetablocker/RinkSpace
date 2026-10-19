"""Evaluate every public truth factor from the actual composed current config."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_current_user_ice_truth as audit
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = audit.read(args.config)
    options = config['source_options']
    saved = Path(options['--camera-seed']).parent
    state, q, identity, camera_error = audit.load_state(saved,
        Path(options['--world-init']), audit.read(saved.parent.parent / 'input_manifest.json'))
    assert camera_error == 0.
    identity['_accepted_directory'] = str(saved)
    original_q = q.tobytes()
    board = audit.NativeBoardEdgeFactor(state, options['--additional-board-source'], len(q))
    ice = audit.NativeIceCurveFactor(state, options['--additional-ice-source'], len(q))
    human = audit.bind_human(state, board, ice, options, len(q))
    wide = audit.wide_factor(state, identity, options, len(q), options['--practical-human-location-envelope'])
    old_options = audit.read(audit.ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle, material, placement = audit.existing_LOW_source_owners(state, q, saved,
        old_options, audit.read(saved / 'scene_UNACCEPTED.json'))
    confirmed = audit.UserConfirmedSourceTruth(state,
        options['--user-confirmed-source-truth-authority'], board, ice, parameter_count=len(q),
        circle_factor=circle, material_factor=material, placement_source=placement)
    actual = audit.values(human, wide, confirmed, q)
    scope = audit.read(options['--truth-restoration-reviewed-manual-append-audit'])
    reference = scope['exact_caps_values']
    assert audit.sha(reference['path']) == reference['sha256']
    expected = audit.read(reference['path'])
    families = {}
    for family, field in [('native', 'primary'), ('wide', 'wide'), ('confirmed', 'confirmed_details')]:
        ids = [entry['id'] for entry in actual[field]]
        target = expected['families'][family]
        values = actual['buffers'][family]
        assert ids == target['group_ids'], family
        assert np.allclose(values, target['violations_px'], rtol=0., atol=1e-8), family
        families[family] = dict(group_ids=ids, actual_violation_px=values.tolist(),
            expected_value_max_difference=float(np.max(abs(values - target['violations_px']))))
    assert sum(len(entry['group_ids']) for entry in families.values()) == 85
    assert q.tobytes() == original_q
    result = dict(schema='rinkspace.actual-final-config-public-truth-construction.v1', passed=True,
        input_config=audit.ref(args.config), complete_state=audit.ref(saved / 'complete_state_UNACCEPTED.npz'),
        complete_q_sha256=identity['q_sha256'], camera_max_difference=camera_error,
        actual_ice_packet=audit.ref(options['--additional-ice-source']),
        actual_confirmed_authority=audit.ref(options['--user-confirmed-source-truth-authority']),
        actual_ice_active_scalar_rows=ice.row_count, original_fit_groups=len(ice.rows), held_groups=len(ice.held),
        families=families, source_values_are_actual_public_query_results=True,
        source_values_are_not_recovery_caps=True, full_motion_graph_or_optimization_run=False,
        accuracy_accepted=False, seconds=time.perf_counter()-started)
    audit.save(args.output, result)
    print({key: result[key] for key in ('passed', 'actual_ice_active_scalar_rows', 'seconds')})


if __name__ == '__main__':
    main()
