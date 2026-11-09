"""Resume the real action factors from a synthetic programme checkpoint."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import action_objective_context as context
from calibration.program_tracking.body_tracking.absolute_motion_curve import (
    AbsoluteMotionCurve,
    _exp,
    clamped_cubic_knots,
)
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
from calibration.program_tracking.body_tracking.complete_runner_extent import CompleteRunnerExtentFactors
from calibration.program_tracking.body_tracking.episode_factor_adapter import EpisodeScopedFactor
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT
from calibration.program_tracking.body_tracking.programme_action_factors import (
    ProgrammeRootTrust,
    SourceTurnPath,
)
from calibration.program_tracking.body_tracking.sequence_dynamics import TemporalContactFactors
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from calibration.program_tracking.body_tracking.skate_binding import NEW_B_TO_OLD, SkateBinding
from calibration.program_tracking.body_tracking.skate_template import SkateParameters
from calibration.program_tracking.body_tracking.source_contour_contract import polyline_measure_weights
from calibration.program_tracking.body_tracking.source_skate_structure_factors import UnassignedSkateSetFactors
from calibration.program_tracking.body_tracking.weighted_blocks import pack_weighted, signature


def _sequence_problem():
    rng = np.random.default_rng(1107)
    times = np.arange(65) * 0.02
    knots = clamped_cubic_knots(np.linspace(-0.04, times[-1] + 0.04, 7))
    controls = len(knots) - 4
    rotations = _exp(rng.normal(0, 0.12, (controls, 22, 3)))
    rotations[:, 0] = _exp(np.array([np.pi / 2, 0, 0])) @ rotations[:, 0]
    curve = AbsoluteMotionCurve(knots, rng.normal(0, 0.03, (controls, 3)), rotations)
    shoe_knots = clamped_cubic_knots(np.linspace(-0.04, times[-1] + 0.04, 5))
    shoe_controls = len(shoe_knots) - 4
    shoe_curve = AbsoluteMotionCurve(
        shoe_knots, np.zeros((shoe_controls, 3)), _exp(rng.normal(0, 0.08, (shoe_controls, 2, 3)))
    )

    binding = SkateBinding.__new__(SkateBinding)
    binding.layout = OBSERVATION_LAYOUT
    binding.ratio = 0.91
    binding.base_parameters = SkateParameters()
    binding.x = np.r_[rng.normal(0, 0.025, 12), np.log([1.04, 0.95, 1.08, 0.94, 1.06])]
    binding.R0 = np.broadcast_to(NEW_B_TO_OLD, (2, 3, 3)).copy()
    binding.t0 = np.array([[0.01, -0.03, 0.02], [-0.02, -0.04, 0.01]])
    binding._parameter_key = binding._packet = None
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body = SimpleNamespace(J=rng.normal(0, 0.15, (22, 3)), parents=parents, scale=0.95)

    def at(samples):
        state = evaluate_curve(curve, samples)
        return dict(P=state['P'], Q=state['Q'], local_R=state['local_R'],
                    skate_relative_R=evaluate_curve(shoe_curve, samples)['rotations'])

    ref = SimpleNamespace(
        absolute_curve=curve, relative_skate_curve=shoe_curve, times=times, binding=binding,
        body=body, p=dict(pts=450560 + np.arange(len(times), dtype=np.int64) * 256,
                          time_base=np.array([1, 12800])), at=at,
    )
    binding.reference = ref
    state = SequenceState(ref, (35.2, 35.2 + times[-1]))
    problem = SimpleNamespace(reference=ref, body=SimpleNamespace(state=state, size=state.size),
                              binding=binding, size=17 + state.size, checkpoint_extra={})
    return problem, rng.normal(0, 0.06, state.size)


@pytest.fixture
def action_case(tmp_path):
    prototype, start = _sequence_problem()
    state = prototype.body.state
    controls = start[:state.body_size].reshape(-1, 69)
    controls[:, 3:6] = 0.0
    controls[:, 4] = np.linspace(-1.2, 1.2, state.m)
    current = start.copy()
    current_controls = current[:state.body_size].reshape(-1, 69)
    current_controls[:, 0] += 0.2
    current_controls[:, 4] *= -1.0

    pts = prototype.reference.p['pts']
    turn = dict(episode_id='turn', kind='turn_candidate', turn_direction_sign=0,
                start_pts=int(pts[0]), end_pts_inclusive=int(pts[-1]))
    support = dict(episode_id='support', kind='support_candidate', foot=1,
                   start_pts=int(pts[8]), end_pts_inclusive=int(pts[56]))
    document = dict(generic_context=True,
                    programme_action_physics=dict(stage='synthetic', structure_enabled=False),
                    source_episode_review=dict(episodes=[turn, support]))
    source = tmp_path / 'source_evidence.json'
    source.write_text(json.dumps(document), encoding='utf-8')

    def construct(warm):
        problem, _ = _sequence_problem()
        problem.reference.source_document = document
        direction = SourceTurnPath(problem, turn, warm)
        primary = SimpleNamespace(factors=[ProgrammeRootTrust(problem, warm),
                                          EpisodeScopedFactor(direction, turn['episode_id'])],
                                  turn_factors=[direction])
        problem.action_factors = SimpleNamespace(modes=TemporalContactFactors(problem, warm))
        return problem, primary

    original, primary = construct(start)
    context.bind_action_objective_context(original, primary, source)
    return SimpleNamespace(construct=construct, original=original, primary=primary, start=start,
                           current=current, source=source)


def _packet(problem, x):
    modes = problem.action_factors.modes
    return dict(problem.checkpoint_extra, **problem.body.state.packet(x),
                x=np.r_[problem.binding.x, x], prepared_input_sha256=np.array('1' * 64),
                prepared_observations_sha256=np.array('2' * 64),
                sequence_contact_modes=modes.modes.copy(), sequence_contact_status=modes.status.copy(),
                sequence_contact_mode_names=np.asarray(modes.record['mode_names']),
                sequence_contact_association_json=np.array(json.dumps(modes.record, sort_keys=True)))


def _weighted(problem, primary, x):
    factors = [*primary.factors, problem.action_factors.modes]
    blocks = [block for factor in factors for block in factor.evaluate(x, True)['weighted_blocks']]
    weights = {'programme_parent_root_trust': 0.24, 'episode/turn/rotation_direction_path': 12.0,
               'temporal_contact_mode': 0.55}
    residual, jacobian, _ = pack_weighted(blocks, weights, True)
    return residual, jacobian


def _assert_same_objective(case, problem, primary):
    expected, expected_jac = _weighted(case.original, case.primary, case.current)
    residual, jacobian = _weighted(problem, primary, case.current)
    np.testing.assert_array_equal(residual, expected)
    assert jacobian.shape == expected_jac.shape
    assert (jacobian != expected_jac).nnz == 0

    np.testing.assert_array_equal(primary.factors[0].parent, case.primary.factors[0].parent)
    turn, original_turn = primary.turn_factors[0], case.primary.turn_factors[0]
    assert (turn.sign, turn.sign_scores, turn.sign_provenance) == (
        original_turn.sign, original_turn.sign_scores, original_turn.sign_provenance)
    modes, original_modes = problem.action_factors.modes, case.original.action_factors.modes
    np.testing.assert_array_equal(modes.modes, original_modes.modes)
    np.testing.assert_array_equal(modes.status, original_modes.status)
    assert modes.record == original_modes.record
    assert modes.association_signature == original_modes.association_signature


def test_exact_resume_restores_fixed_action_factors(action_case):
    case = action_case
    problem, primary = case.construct(case.current)
    assert np.any(case.primary.factors[0].evaluate(case.current)['residual'])
    assert primary.turn_factors[0].sign != case.primary.turn_factors[0].sign
    # Force distinct new contact inference before restoring the saved fields.
    modes = problem.action_factors.modes
    modes.modes.fill(6)
    modes.status[:] = 'unknown'
    modes.record['method'] = 'synthetic replacement association'
    modes.association_signature = signature(modes.modes, modes.status)
    saved = _packet(case.original, case.current)

    restored = context.bind_action_objective_context(problem, primary, case.source, saved=saved)

    assert restored['restoration'] == 'checkpoint_context'
    _assert_same_objective(case, problem, primary)


@pytest.mark.parametrize(('change', 'message'), [
    ('source', 'source_evidence SHA256'),
    ('clock', 'source clock, seed or control layout'),
    ('anchor', 'root anchor hash mismatch'),
    ('modes', 'contact context mismatch'),
])
def test_exact_resume_rejects_changed_context(action_case, change, message):
    case = action_case
    problem, primary = case.construct(case.current)
    saved = _packet(case.original, case.current)
    if change == 'source':
        case.source.write_text(case.source.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    elif change == 'clock':
        problem.reference.p['pts'] = problem.reference.p['pts'] + 1
    elif change == 'anchor':
        saved[context.ANCHOR] = saved[context.ANCHOR] + 0.1
    elif change == 'modes':
        modes = saved['sequence_contact_modes']
        modes[0, 0] = (modes[0, 0] + 1) % len(saved['sequence_contact_mode_names'])

    with pytest.raises(ValueError, match=message):
        context.bind_action_objective_context(problem, primary, case.source, saved=saved)


def _legacy_packet(problem, x):
    omitted = {context.CONTEXT, context.ANCHOR, context.ROOT_PTS, context.SOURCE_PTS}
    return {key: value for key, value in _packet(problem, x).items() if key not in omitted}


@pytest.fixture
def legacy_stage(action_case, tmp_path):
    case = action_case
    origin = tmp_path / 'direct_origin.npz'
    np.savez_compressed(origin, **_legacy_packet(case.original, case.start))
    stage = tmp_path / 'legacy_stage'
    stage.mkdir()
    metrics = case.primary.turn_factors[0].evaluate(case.start)['metrics']
    (stage / 'initial_action_factors.json').write_text(
        json.dumps({'source_and_physics': {'episode/turn/metrics': metrics}}), encoding='utf-8')
    receipt = dict(parameter=str(origin), parameter_sha256=hashlib.sha256(origin.read_bytes()).hexdigest())
    return stage, origin, receipt


@pytest.mark.parametrize('receipt_name', ['warm_start.json', 'resume.json'])
def test_legacy_resume_migrates_once_from_the_immediate_stage(action_case, legacy_stage, receipt_name):
    case = action_case
    stage, origin, receipt = legacy_stage
    receipt_path = stage / receipt_name
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    problem, primary = case.construct(case.current)

    migrated = context.bind_action_objective_context(
        problem, primary, case.source, saved=_legacy_packet(case.original, case.current), legacy_stage=stage)

    assert migrated['restoration'] == 'legacy_migration'
    _assert_same_objective(case, problem, primary)
    saved = _packet(problem, case.current)
    receipt_path.unlink()
    (stage / 'initial_action_factors.json').unlink()
    origin.unlink()
    stage.rmdir()
    resumed, resumed_primary = case.construct(case.current)
    restored = context.bind_action_objective_context(resumed, resumed_primary, case.source, saved=saved)

    assert restored['restoration'] == 'checkpoint_context'
    _assert_same_objective(case, resumed, resumed_primary)


@pytest.mark.parametrize(('missing', 'error', 'message'), [
    ('receipt', ValueError, 'does not prove a zero root anchor'),
    ('parameter_sha', ValueError, 'parameter SHA256 mismatch'),
    ('turn_record', FileNotFoundError, 'original ambiguous-turn records'),
])
def test_legacy_resume_requires_direct_evidence(action_case, legacy_stage, missing, error, message):
    case = action_case
    stage, _, receipt = legacy_stage
    if missing == 'parameter_sha':
        receipt['parameter_sha256'] = '0' * 64
    if missing != 'receipt':
        (stage / 'resume.json').write_text(json.dumps(receipt), encoding='utf-8')
    if missing == 'turn_record':
        (stage / 'initial_action_factors.json').unlink()
    problem, primary = case.construct(case.current)

    with pytest.raises(error, match=message):
        context.bind_action_objective_context(
            problem, primary, case.source, saved=_legacy_packet(case.original, case.current), legacy_stage=stage)


def test_revised_objective_warm_start_anchors_to_current_state(action_case):
    case = action_case
    problem, primary = case.construct(case.current)

    fresh = context.bind_action_objective_context(problem, primary, case.source)

    assert fresh['restoration'] == 'new_context'
    np.testing.assert_array_equal(primary.factors[0].evaluate(case.current)['residual'], 0.0)
    assert primary.turn_factors[0].sign != case.primary.turn_factors[0].sign


def test_exact_resume_uses_the_saved_source_choice_instead_of_an_override():
    choices = {'conditional_episode': 1}
    saved = {'source_association_extra_conditional_episode_choice': np.array(1)}

    assert context.resume_source_choices(choices, None) == choices
    assert context.resume_source_choices(choices, saved) == {}
    assert choices == {'conditional_episode': 1}
    with pytest.raises(ValueError, match='missing or different source episode choice'):
        context.resume_source_choices({'conditional_episode': 0}, saved)
    with pytest.raises(ValueError, match='missing or different source episode choice'):
        context.resume_source_choices(choices, {})


def _source_contour(*, frozen):
    return SimpleNamespace(
        association_policy={'freeze_inner': frozen},
        fixed_choice=np.array([1, 2]) if frozen else None,
        source_uv=np.zeros((2, 2)), starts=np.arange(5), point_rows=np.array([0, 1]),
        rows=[{'structure_feature': 'runner'}, {'structure_feature': 'sole'}],
        segment_geometry={'feature_segments': {'runner': [1, 3], 'sole': [2, 4]}},
    )


def test_exact_resume_restores_saved_contour_segments():
    contour = _source_contour(frozen=True)
    prefix = 'source_association_contour_'
    saved = np.array([3, 4])

    context.restore_source_masks(contour, {prefix + 'fixed_choice': saved}, prefix)

    np.testing.assert_array_equal(contour.fixed_choice, [3, 4])
    assert not np.shares_memory(contour.fixed_choice, saved)
    with pytest.raises(ValueError, match='changed source feature'):
        context.restore_source_masks(contour, {prefix + 'fixed_choice': [2, 4]}, prefix)


@pytest.mark.parametrize('frozen', [True, False], ids=['frozen', 'dynamic'])
def test_exact_resume_requires_a_mask_only_for_frozen_correspondence(frozen):
    contour = _source_contour(frozen=frozen)
    prefix = 'source_association_contour_'
    if frozen:
        with pytest.raises(ValueError, match='requires saved frozen source mask'):
            context.restore_source_masks(contour, {}, prefix)
    else:
        context.restore_source_masks(contour, {}, prefix)
        assert contour.fixed_choice is None


def test_exact_resume_does_not_require_masks_for_factors_without_admitted_rows():
    contour = _source_contour(frozen=True)
    contour.rows = []
    contour.source_uv = np.empty((0, 2))
    contour.fixed_choice = None
    runner = CompleteRunnerExtentFactors.__new__(CompleteRunnerExtentFactors)
    runner.association_policy = {'model_visibility': 'near_visible_runner_branch'}
    runner.rows = []
    runner.count = 162
    runner.visible_mask = None

    context.restore_source_masks(contour, {}, 'source_association_contour_')
    context.restore_source_masks(runner, {}, 'source_association_runner_extent_')

    assert contour.fixed_choice is None
    assert runner.visible_mask is None
    empty_mask = np.empty((0, runner.count), dtype=bool)
    runner.restore_visible_mask(empty_mask)
    np.testing.assert_array_equal(runner.visible_mask, empty_mask)
    assert runner.visibility_record['model_side'] == []


@pytest.mark.parametrize('legacy', [True, False], ids=['legacy-base-episode', 'frozen-extra-episode'])
def test_exact_resume_evaluates_the_saved_episode_branch(legacy):
    other = _source_contour(frozen=not legacy)
    selected = _source_contour(frozen=not legacy)
    episode = UnassignedSkateSetFactors.__new__(UnassignedSkateSetFactors)
    episode.episode = {'candidates': [{'candidate_id': 'other'}, {'candidate_id': 'saved'}]}
    episode.episode_factors = [other, selected]
    episode.episode_choice = 0
    episode.episode_costs = [0.1, 2.0]

    def wrong_branch(x, jacobian):
        pytest.fail('Exact resume evaluated a branch other than the saved choice')

    def saved_branch(x, jacobian):
        if not legacy:
            np.testing.assert_array_equal(selected.fixed_choice, [3, 4])
        return dict(residual=np.array([2.0]), jac=None, image_cost=2.0, metrics={})

    other.evaluate = wrong_branch
    selected.evaluate = saved_branch
    selected.association_signature = 'saved-segments'
    if legacy:
        prefix = 'source_association_unassigned_'
        choice_key = 'source_association_episode_choice'
        saved = {choice_key: np.array(1)}
        expected_costs = []
    else:
        prefix = 'source_association_extra_conditional_episode_'
        choice_key = None
        expected_costs = [0.1, 2.0]
        saved = {prefix + 'choice': np.array(1), prefix + 'candidate_costs': np.array(expected_costs),
                 prefix + 'fixed_choice': np.array([3, 4])}
        assert context.resume_source_choices({'conditional_episode': 1}, saved) == {}
        with pytest.raises(ValueError, match='requires saved frozen source mask'):
            context.restore_source_episode(episode, {prefix + 'choice': np.array(1)}, prefix)
        assert episode.episode_choice == 0

    context.restore_source_episode(episode, saved, prefix, choice_key=choice_key)
    value = episode.evaluate(np.empty(0), False)

    assert value['image_cost'] == 2.0
    result = value['metrics']['source_unassigned_episode']
    assert result['candidate_id'] == 'saved'
    assert result['candidate_costs'] == expected_costs


def test_exact_resume_restores_visible_runner_weights_from_the_saved_side():
    runner = CompleteRunnerExtentFactors.__new__(CompleteRunnerExtentFactors)
    runner.association_policy = {'model_visibility': 'near_visible_runner_branch'}
    runner.count = 162
    t = np.linspace(0.0, 1.0, 81)
    runner.curves = np.stack([np.c_[t, t * 0.0, t ** 2], np.c_[t ** 2, t * 0.0, t]])
    runner.rows = [dict(contour_frame_budget=0.8, feature_budget_fraction=0.25),
                   dict(contour_frame_budget=0.6, feature_budget_fraction=0.5)]
    runner.factor_rows = [{'objective_weight': 1.0} for _ in range(2 * runner.count)]
    sides = np.arange(runner.count)[None] // 81
    runner.visible_mask = sides == np.array([[0], [1]])
    saved = sides == np.array([[1], [0]])
    prefix = 'source_association_runner_extent_'

    context.restore_source_masks(runner, {prefix + 'visible_mask': saved}, prefix)

    np.testing.assert_array_equal(runner.visible_mask, saved)
    assert runner.visibility_record['model_side'] == [1, 0]
    actual = np.array([row['objective_weight'] for row in runner.factor_rows]).reshape(2, -1)
    expected = np.zeros_like(actual)
    expected[0, 81:] = (0.8 * 0.25) * polyline_measure_weights(runner.curves[1][:, [0, 2]])
    expected[1, :81] = (0.6 * 0.5) * polyline_measure_weights(runner.curves[0][:, [0, 2]])
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='requires saved runner visibility'):
        context.restore_source_masks(runner, {}, prefix)
    saved[0, 81] = False
    with pytest.raises(ValueError, match='select one complete 81-point side'):
        context.restore_source_masks(runner, {prefix + 'visible_mask': saved}, prefix)


def test_exact_resume_compares_active_indices_instead_of_only_the_count(tmp_path):
    problem = SimpleNamespace(size=22, active_parameter_indices=np.array([17, 19, 21]))
    solve = tmp_path / 'solve'
    solve.mkdir()
    receipt = solve / 'parameter_selection.json'
    receipt.write_text(json.dumps(dict(full_layout=22, selection=[17, 19, 21])), encoding='utf-8')
    context.validate_resume_selection(problem, tmp_path)

    receipt.write_text(json.dumps(dict(full_layout=22, selection=[17, 20, 21])), encoding='utf-8')
    with pytest.raises(ValueError, match='changed the complete active parameter selection'):
        context.validate_resume_selection(problem, tmp_path)
