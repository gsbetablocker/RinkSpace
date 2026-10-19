"""Summarize the completed fixed-W experiment; no source/model evaluation."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np

H = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
SOURCE = H / 'shot5_fixed_common_rotation_original_source_response_20261011.json'
EXPECTED = '1b34581b02abaab5260c112312e6fc2ed0d2cb65a25c6ed4cabf1957382dbd2b'
OUT = H / 'shot5_common_rotation_causal_interpretation_20261011.json'
SIX = {229953536, 229961216, 230306816, 230314496, 230352896, 230360576}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(rows):
    before = np.asarray([r['original_cost'] for r in rows])
    after = np.asarray([r['trial_cost'] for r in rows])
    return dict(operators=len(rows), actual_exposures=len({r['actual_pts'] for r in rows}),
                scalar_rows=sum(r['scalar_rows'] for r in rows),
                original_cost=float(before.sum()), counterfactual_cost=float(after.sum()),
                cost_increased=int((after > before).sum()),
                cost_decreased=int((after < before).sum()), cost_unchanged=int((after == before).sum()),
                operators_are_not_independent_landmark_counts=True)


def main():
    assert sha(SOURCE) == EXPECTED
    d = json.loads(SOURCE.read_text(encoding='utf8'))
    for ref in d['references'] + [d['arrays']]:
        assert sha(ref['path']) == ref['sha256'], ref['path']
    rows = d['absolute_source_groups']
    assert len(rows) == 187 and sum(r['scalar_rows'] for r in rows) == 3744
    by_family = defaultdict(list)
    for r in rows:
        by_family[r['family']].append(r)
    with np.load(d['arrays']['path'], allow_pickle=False) as z:
        assert not any(k in z for k in ('q', 'dq', 'J'))
        W = z['W_world_rotation']
        np.testing.assert_allclose(W.T @ W, np.eye(3), rtol=0, atol=2e-15)
        assert abs(np.linalg.det(W) - 1) < 2e-15
        assert len(z['motion_pairs']) == 1507 and z['motion_pair_counts'].sum() == 264138
        np.testing.assert_array_equal(z['absolute_original_cost'], [r['original_cost'] for r in rows])
        np.testing.assert_array_equal(z['absolute_trial_cost'], [r['trial_cost'] for r in rows])
        total = next(r for r in d['all_shot_motion']['branches'] if r['branch'] == 'all_shot5_motion')
        assert abs(float((z['motion_pair_trial_cost'] - z['motion_pair_original_cost']).sum()) - total['cost_change']) < 1e-10
    violations = [r for r in d['trial_all86_guard_values']['rows'] if r['exceeds_original_q8_limit']]
    assert len(violations) == 36
    tail = d['camera_dependent_original_tail_values']
    tracks = next(r for r in tail['families'] if r['family'] == 'additional_native_static_board_tracks')
    result = dict(schema='rinkspace.fixed-whole-shot-causal-interpretation.v1',
        source=dict(path=str(SOURCE), sha256=EXPECTED), summarizer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        source_interval='04:09:30.433-04:09:57.983 (exclusive)', frozen_q_sha256=d['frozen_complete_q_sha256'],
        scope=dict(actual_exposures=1381, original_motion_points=264138, original_pairs=1507),
        all_absolute=summary(rows), LOW6_absolute=summary([r for r in rows if r['actual_pts'] in SIX]),
        other_absolute=summary([r for r in rows if r['actual_pts'] not in SIX]),
        absolute_by_family={k: summary(v) for k, v in by_family.items()},
        LOW6_original_operators=[r for r in rows if r['actual_pts'] in SIX],
        original_motion_branches=d['all_shot_motion']['branches'],
        unknown_numeric=d['all_shot_motion']['unknown_depth_original_numeric_covariance'],
        static_board_track_operator_kinds=tracks['by_operator_basis'],
        original_guard_violations=dict(count=len(violations), native=sum(r['family']=='native' for r in violations),
            confirmed=sum(r['family']=='confirmed' for r in violations), maximum_excess_px=max(r['trial_violation_px']-r['original_q8_limit_px'] for r in violations)),
        domain=d['domain'], tail_domain=d['independently_checked_tail_domain'],
        findings=[
            'The isolated-frame motion jump is removed by a common whole-shot rotation; its old cost increase was not evidence of a physical incompatibility.',
            'This fixed common rotation fails across the real whole-shot absolute sources and original caps; the 47px exposure cannot stand for the shot.',
            'Adjacent motion and current-ray transfers weakly constrain this absolute orientation mode at this frozen installation; temporal smoothness does not certify world registration.',
            'Frozen-template target/anchor costs are conditional on the earlier model-inferred print shape, not independent measured world XYZ.',
            'The failed fixed W does not prove every common rotation fails, that geometry alone is wrong, or that another physically admissible trajectory cannot fit.',
        ],
        unchanged_model=True, new_source_images=0, new_videos=0, accuracy_accepted=False,
        root_reference_array_and_aggregation_checks=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(OUT), sha256=sha(OUT), all_absolute=result['all_absolute'],
                          guards=result['original_guard_violations'])))


if __name__ == '__main__':
    main()
