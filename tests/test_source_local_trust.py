"""Source-local nonlinear feedback must preserve unrelated coupled controls."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import programme_source_step as source_step
from calibration.program_tracking.body_tracking.weighted_blocks import block


WIDTH = 80
SELECTION = np.array([18, 23, 41, 58, 74])


def image_ball(source_id, foot, columns):
    local_rows = np.arange(len(columns)) % 2
    J = csr_matrix((np.ones(len(columns)), (local_rows, columns)), shape=(2, WIDTH))
    return dict(
        source_id=source_id, model_foot=foot, jacobian=J,
        offset=np.zeros((1, 2)), radius=np.ones(1), labels=[source_id+'/point_0'],
    )


def margins(*balls):
    return source_step.SourceMargins(csr_matrix((0, WIDTH)), np.empty(0), [], balls)


def pixel_check(after, *, before=0., radius=4., foot=1):
    return dict(rejected=bool(after > before+.25*radius), before=before, after=after,
                original_radius_px=radius, model_foot=foot)


def test_each_failed_row_contracts_only_its_raw_columns_and_overlap_uses_minimum():
    # The first ball includes a frozen shared column and an inactive body column.
    # The second ball overlaps one active column but has a stricter nonlinear row.
    limits = np.array([.4, .8, .6, 1.2, .9])
    direction = limits*np.array([1, -1, 1, 1, -1])
    guards = margins(
        image_ball('front_stanchion', 1, [3, 18, 21, 23]),
        image_ball('release_sole', 0, [23, 41]),
    )

    def check(actual_direction):
        a, b, c, _, _ = actual_direction/limits
        return {
            'front_stanchion': pixel_check(.3 + 1.6*(a*a+b*b), before=.3),
            'release_sole': pixel_check(24.*(b*b+c*c), foot=0),
        }

    new, receipt = source_step.contract_source_trust(guards, SELECTION, limits, direction, check)
    assert receipt['status'] == 'contracted'
    assert set(receipt['changed_global_columns']) == {18, 23, 41}
    assert receipt['failed_source_ids'] == ['front_stanchion', 'release_sole']
    rows = {row['source_id']: row for row in receipt['rows']}
    assert rows['front_stanchion']['passing_fraction'] == .5
    assert rows['release_sole']['passing_fraction'] == .125
    assert set(rows['front_stanchion']['global_columns']) == {18, 23}
    assert set(rows['release_sole']['global_columns']) == {23, 41}
    assert rows['front_stanchion']['before'] == .3
    assert rows['release_sole']['original_radius_px'] == 4.
    assert receipt['original_nonlinear_limits_unchanged']
    np.testing.assert_array_equal(new, limits*np.array([.5, .125, .125, 1., 1.]))
    np.testing.assert_array_equal(limits, [.4, .8, .6, 1.2, .9])
    np.testing.assert_array_equal(direction, [.4, -.8, .6, 1.2, -.9])
    assert check(direction*.5)['front_stanchion']['rejected'] is False
    assert check(direction*.25)['release_sole']['rejected'] is True
    assert check(direction*.125)['release_sole']['rejected'] is False


def test_inside_band_source_uses_raw_pixels_even_without_weighted_rows():
    source_id = 'inside_band_sole'
    raw = image_ball(source_id, 1, [23, 41])['jacobian']
    contour = SimpleNamespace(
        rows=[dict(id=source_id, foot=1)], point_rows=np.array([0]),
        source_uv=np.zeros((1, 2)), proposal_pixel_prediction=np.zeros((1, 2)),
        proposal_pixel_jacobian=raw,
    )
    value = dict(
        action_metrics=dict(visible_runner_contours=[
            dict(id=source_id, foot=1, outside_visible_band_px=[0.], model_closest_uv=[[0., 0.]]),
        ]),
        weighted_blocks=[block('full_source_skate_structure', np.zeros(2), csr_matrix((2, WIDTH)))],
    )
    guards = source_step.contour_margins(value, contour, {source_id: (0., 4.)})
    assert guards[0].shape == (0, WIDTH)
    ball, = guards.image_balls
    assert ball['source_id'] == source_id and ball['model_foot'] == 1
    limits = np.ones(len(SELECTION))

    def check(direction):
        return {source_id: pixel_check(3.*direction[1]**2)}

    new, receipt = source_step.contract_source_trust(guards, SELECTION, limits, limits, check)
    assert receipt['status'] == 'contracted'
    assert set(receipt['changed_global_columns']) == {23, 41}
    np.testing.assert_array_equal(new, [1., .5, .5, 1., 1.])


def test_already_passing_source_proposal_leaves_all_controls_unchanged():
    limits = np.array([.1, .25, .4, .5, .6])
    guards = margins(image_ball('source', 1, [18, 23]))
    new, receipt = source_step.contract_source_trust(
        guards, SELECTION, limits, limits*.8, lambda direction: {'source': pixel_check(.5)},
    )
    assert receipt['status'] == 'passed'
    np.testing.assert_array_equal(new, limits)


@pytest.mark.parametrize('failure', ['domain_at_full', 'domain_at_probe', 'unknown_id', 'changed_branch'])
def test_unattributed_failures_do_not_shrink_source_or_remote_controls(failure):
    limits = np.array([.2, .35, .4, .5, .6])
    direction = limits.copy()
    guards = margins(image_ball('source', 1, [18, 23]))

    def check(point):
        at_full = point[0] > .75*direction[0]
        if failure == 'domain_at_full' or (failure == 'domain_at_probe' and not at_full):
            return None
        name = 'unregistered' if failure == 'unknown_id' else 'source'
        foot = 0 if failure == 'changed_branch' and not at_full else 1
        return {name: pixel_check(2. if at_full else .5, foot=foot)}

    new, receipt = source_step.contract_source_trust(guards, SELECTION, limits, direction, check)
    assert receipt['status'] == 'domain_unattributed'
    np.testing.assert_array_equal(new, limits)


def test_re_solved_coupled_quadratic_retains_independent_remote_direction():
    selected = np.array([18, 23, 74])
    limits = np.array([.8, 1., 1.])
    preferred = np.array([.8, .2, .7])
    hessian = np.array([[2., 1., 0.], [1., 2., 0.], [0., 0., 1.]])

    class CoupledNormal:
        scale_inv = np.ones(len(preferred))

        def solve_scaled(self, rhs, scale, *, damp):
            np.testing.assert_array_equal(rhs, [-2.])
            np.testing.assert_array_equal(scale, np.ones(len(preferred)))
            assert damp == .07
            return preferred.copy(), {}

        def scaled_normal_objective(self, rhs, scale, *, damp):
            return objective

    def objective(point):
        offset = point-preferred
        gradient = hessian@offset
        return float(.5*offset@gradient - .5*preferred@hessian@preferred), gradient

    def check(point):
        return {'source': pixel_check((point[0]/.22)**2)}

    normal = CoupledNormal()
    value = {'residual': np.array([2.])}
    bounds = (-np.ones(3), np.ones(3))
    guards = margins(image_ball('source', 1, [18]))
    first, _ = source_step.source_conditioned_direction(
        normal, value, bounds, guards, selected, control_step_limit=limits,
    )
    assert check(first)['source']['rejected']
    trust_state = {}
    updated, receipt = source_step.source_conditioned_direction(
        normal, value, bounds, guards, selected, control_step_limit=limits,
        nonlinear_source_check=check, trust_state=trust_state,
    )
    feedback = receipt['local_source_feedback']
    assert feedback[-1]['status'] == 'passed'
    assert any(row['status'] == 'contracted' and row['changed_global_columns'] == [18] for row in feedback)
    new = trust_state['control_step_limit']
    np.testing.assert_array_equal(new, [.2, 1., 1.])
    np.testing.assert_array_equal(limits, [.8, 1., 1.])
    np.testing.assert_allclose(updated, [.2, .5, .7], atol=1e-8, rtol=0)
    assert updated[2] == first[2]
    assert not check(updated)['source']['rejected']
    # The coupled neighbour moves to the quadratic optimum even though its
    # trust limit stays fixed. The remote variable remains at the old optimum.
    clipped = np.clip(first, -new, new)
    assert objective(updated)[0] < objective(clipped)[0]


def test_source_trial_rows_opt_in_checks_all_footwear_source_families():
    from calibration.program_tracking.body_tracking.sequence_refit import SequenceActions

    x = np.array([.12, -.04])
    assigned_row = dict(id='assigned_runner', foot=0)
    episode_row = dict(id='fixed_episode_sole', foot=1, foot_is_model_association=True)
    additional_row = dict(id='additional_episode_sole', foot=0, foot_is_model_association=True)
    checked = set()

    def factor(metric, row):
        def evaluate(point, jacobian):
            np.testing.assert_array_equal(point, x)
            assert jacobian is False
            checked.add(row['id'])
            return dict(metrics={metric: [row]})
        return SimpleNamespace(evaluate=evaluate)

    actions = SimpleNamespace(
        contour=factor('visible_runner_contours', assigned_row),
        unassigned=factor('visible_unassigned_skate_contours', episode_row),
        extra_unassigned_episodes=[
            ('additional_episode', factor('visible_unassigned_skate_contours', additional_row)),
        ],
    )
    actions.unassigned.episode = {'episode_id': 'fixed_episode'}
    actions.unassigned.episode_choice = 6
    assert SequenceActions.source_trial_rows(actions, x) == [assigned_row]
    assert checked == {assigned_row['id']}
    checked.clear()
    assert SequenceActions.source_trial_rows(actions, x, include_unassigned=True) == [
        assigned_row, episode_row, additional_row,
    ]
    assert checked == {row['id'] for row in (assigned_row, episode_row, additional_row)}
