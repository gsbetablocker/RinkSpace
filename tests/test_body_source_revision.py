"""Exact source PTS edits and immutable protected quality populations."""
import copy
import json

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.body_source_revision import (
    revise_body_observations,
    revised_quality_population,
)
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment


S08_PTS = tuple(range(1802752, 1804032+1, 256))
LEG_PTS = (2072320, 2081792)
WRIST_PTS = (2081536, 2081792)
PROTECTED = [5, 6, 11, 12, 13, 14, 15, 16]
PROVENANCE = {'review': 'exact original native exposure identity'}
LEG_EDITS = [dict(source_pts=pts, joint_pairs=[[13, 14], [15, 16]]) for pts in LEG_PTS]
WRIST_EDITS = [dict(source_pts=pts, joint_id=10, reason='duplicate right-wrist detection')
               for pts in WRIST_PTS]


@pytest.fixture
def observations():
    pts = np.array([*S08_PTS, 2072064, 2072320, 2072576, 2081280, 2081536, 2081792, 2082048])
    n = len(pts)
    ids = np.tile(np.arange(17), (n, 1))
    ids[:len(S08_PTS), 13:17] = [14, 13, 16, 15]
    use = np.ones((n, 17), dtype=bool)
    use[0, 3] = False
    previous = dict(kind='source_observation_correspondence', source_pts=list(S08_PTS),
                    source_measurements_unchanged=True, review='prior S08 bilateral repair')
    prior_admission = dict(kind='source_reviewed_admission',
                           excluded=[dict(source_pts=int(pts[0]), joint_id=3, reason='prior source review')])
    return dict(
        pts=pts, model_joint_ids=ids, use=use,
        uv=np.arange(n*17*2, dtype=np.float32).reshape(n, 17, 2)*.25,
        sigma_px=2.+np.arange(n*17).reshape(n, 17)/70,
        confidence=np.linspace(.18, .99, n*17, dtype=np.float32).reshape(n, 17),
        body_joint_assignment_provenance_json=np.array(json.dumps(previous)),
        body_admission_parent_use=np.ones_like(use),
        body_admission_revision_json=np.array(json.dumps(prior_admission)),
    )


def revise(observations):
    return revise_body_observations(observations, LEG_EDITS, WRIST_EDITS, PROVENANCE)


def test_revision_keeps_s08_and_changes_only_exact_leg_correspondence_and_right_wrist_use(observations):
    original = copy.deepcopy(observations)
    revised, record = revise(observations)
    leg_rows = np.searchsorted(original['pts'], LEG_PTS)
    wrist_rows = np.searchsorted(original['pts'], WRIST_PTS)
    expected_ids = original['model_joint_ids'].copy()
    expected_ids[leg_rows, 13:17] = [14, 13, 16, 15]
    expected_use = original['use'].copy()
    expected_use[wrist_rows, 10] = False

    np.testing.assert_array_equal(body_joint_assignment(revised), expected_ids)
    np.testing.assert_array_equal(revised['model_joint_ids'][:len(S08_PTS)],
                                  original['model_joint_ids'][:len(S08_PTS)])
    np.testing.assert_array_equal(revised['model_joint_ids'][:, 9:13], original['model_joint_ids'][:, 9:13])
    np.testing.assert_array_equal(revised['use'], expected_use)
    np.testing.assert_array_equal(revised['use'][:, 9], original['use'][:, 9])
    for key, value in original.items():
        np.testing.assert_array_equal(observations[key], value)
    for key in ('pts', 'uv', 'sigma_px', 'confidence'):
        np.testing.assert_array_equal(revised[key], original[key])
        assert not np.shares_memory(revised[key], observations[key])

    saved = json.loads(str(revised['body_joint_assignment_provenance_json']))
    assert saved['source_pts'] == [*S08_PTS, *LEG_PTS]
    assert saved['previous_correspondence'] == json.loads(str(original['body_joint_assignment_provenance_json']))
    assert saved['new_assignments'] == [
        dict(source_pts=pts, joint_pair=pair, before=pair, after=pair[::-1])
        for pts in LEG_PTS for pair in ([13, 14], [15, 16])
    ]
    expected_exclusions = [dict(row, raw_uv_retained=True) for row in WRIST_EDITS]
    assert saved['derived_use_exclusions'] == expected_exclusions
    assert record['changed_correspondence_entries'] == 8
    assert record['changed_use_entries'] == 2
    np.testing.assert_array_equal(revised['body_admission_parent_use'], original['use'])
    admission = json.loads(str(revised['body_admission_revision_json']))
    assert admission['excluded'] == expected_exclusions
    assert admission['parent_admission'] == str(original['body_admission_revision_json'])


@pytest.mark.parametrize(('edits', 'reason'), [
    ([dict(source_pts=LEG_PTS[0], joint_pairs=[[13, 14], [13, 14]])], 'Duplicate'),
    ([dict(source_pts=S08_PTS[0], joint_pairs=[[13, 14]])], 'already revised'),
    ([dict(source_pts=LEG_PTS[0], joint_pairs=[[13, 16]])], 'same bilateral joint kind'),
    ([dict(source_pts=LEG_PTS[0]+1, joint_pairs=[[13, 14]])], 'outside native exposures'),
    ([dict(source_pts=float(LEG_PTS[0]), joint_pairs=[[13, 14]])], 'Exact integer PTS'),
])
def test_invalid_correspondence_revisions_are_rejected_without_mutating_input(observations, edits, reason):
    original = copy.deepcopy(observations)
    with pytest.raises(ValueError, match=reason):
        revise_body_observations(observations, edits, [], PROVENANCE)
    for key, value in original.items():
        np.testing.assert_array_equal(observations[key], value)


@pytest.mark.parametrize(('exclusions', 'reason'), [
    ([WRIST_EDITS[0], WRIST_EDITS[0]], 'Duplicate'),
    ([dict(WRIST_EDITS[0], source_pts=WRIST_PTS[0]+1)], 'outside native exposures'),
    ([dict(source_pts=S08_PTS[0], joint_id=3, reason='already excluded')], 'inactive'),
])
def test_exclusions_cannot_repeat_existing_rows_or_address_non_native_pts(observations, exclusions, reason):
    original = observations['use'].copy()
    with pytest.raises(ValueError, match=reason):
        revise_body_observations(observations, [], exclusions, PROVENANCE)
    np.testing.assert_array_equal(observations['use'], original)


@pytest.fixture
def quality_baseline():
    return dict(
        source={'runner': [2.4, 4.]},
        torso=dict(observations=211, pixel_p95=17.125),
        turn_episodes={
            'LSp/torso': dict(observations=45, pixel_p95=18.5),
            'LSp/lower_limbs': dict(observations=37, pixel_p95=28.75),
        },
        body_observation_reference=dict(path='previous.npz', sha256='previous-source-digest'),
        stage_reset_allowed=False,
    )


def test_revised_quality_population_retains_protected_masks_and_every_nonlinear_limit(observations, quality_baseline):
    revised, record = revise(observations)
    old_baseline = copy.deepcopy(quality_baseline)
    reference = dict(path='revised.npz', sha256='revised-source-digest')
    updated = revised_quality_population(quality_baseline, observations, revised, reference, record)

    np.testing.assert_array_equal(revised['use'][:, PROTECTED], observations['use'][:, PROTECTED])
    for key in ('source', 'torso', 'turn_episodes', 'stage_reset_allowed'):
        assert updated[key] == old_baseline[key]
    assert quality_baseline == old_baseline
    assert updated['body_observation_reference'] == reference
    history = updated['body_source_revision']
    assert history['previous_quality_population'] == old_baseline['body_observation_reference']
    assert history['source_revision'] == record
    assert history['excluded_rows'] == 2
    assert history['protected_torso_lower_population_identical']
    assert history['all_existing_source_torso_turn_limits_unchanged']


@pytest.mark.parametrize(('joint', 'declare', 'reason'), [
    (13, False, 'protected quality population'),
    (14, True, 'protected quality population'),
    (9, False, 'undeclared evidence'),
])
def test_quality_population_refuses_knees_and_undeclared_upper_limb_exits(
        observations, quality_baseline, joint, declare, reason):
    revised, record = revise(observations)
    pts = LEG_PTS[0]
    revised['use'][np.searchsorted(observations['pts'], pts), joint] = False
    if declare:
        record['derived_use_exclusions'].append(dict(
            source_pts=pts, joint_id=joint, reason='injected protected-row exclusion',
        ))
    with pytest.raises(ValueError, match=reason):
        revised_quality_population(quality_baseline, observations, revised, {}, record)
