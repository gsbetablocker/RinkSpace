import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver
from structured_programme_source_candidate.lagged_jacobian_source import LaggedJacobianSource
from z190_streaming_source_normal import SourceRowBlock
from calibration.program_tracking.first_shot_prediction_refresh import (
    compile_quality_solver, install_prediction_refresh, refreshed_fraction)


def test_only_the_numerical_search_seed_is_added_to_frozen_solver():
    source = inspect.getsource(solver.solve_joint_source)
    updated = compile_quality_solver(solver.solve_joint_source)
    assert updated.__globals__['_prediction_refresh_fraction'] is refreshed_fraction
    assert inspect.signature(updated) == inspect.signature(solver.solve_joint_source)
    assert inspect.getsource(solver.solve_joint_source) == source


LOG = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac/transverse_v23_native_f2.stdout.log')


@pytest.mark.skipif(not LOG.exists(), reason='Real stopped v23 events unavailable')
def test_real_accepted_predictions_refresh_without_rejecting_the_accepted_state():
    checked = []
    for line in LOG.read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('stage') != 'joint_full_source_iteration' or not row.get('accepted'):
            continue
        report = row['source_jacobian_reuse']
        if report['exact_current_J']:
            continue
        predicted, actual = report['predicted_source_reduction'], report['actual_source_reduction']
        # Replay the saved actual nonlinear cost and its saved directional
        # quadratic prediction through the real public derivative model.
        model = LaggedJacobianSource(.25)
        model.force_fresh = False
        model.accepted(row['cost'], row['next_cost'], np.ones(1), csr_matrix((1, 1)), np.array([-predicted]))
        assert model.current_report['actual_source_reduction'] == actual
        fraction = row['adaptive_source_backtracking']['next_line_search_fraction']
        reset = refreshed_fraction(model, fraction, True)
        if actual / predicted < .25:
            assert model.force_fresh and reset == 1.
            checked.append(row['iteration'])
        else:
            assert not model.force_fresh and reset == fraction
    assert {11, 12, 13, 14, 15, 16, 19, 24, 25, 34}.issubset(checked)


def test_disabled_adaptive_search_and_healthy_model_retain_fraction():
    model = LaggedJacobianSource(.25)
    model.force_fresh = True
    assert refreshed_fraction(model, .125, False) == .125
    model.force_fresh = False
    assert refreshed_fraction(model, .125, True) == .125


def test_new_source_initial_seed_reset_preserves_canonical_state_and_damping():
    saved_q = np.arange(4, dtype=float)
    class Owner:
        _initial_complete_J_reuse = False
        source_producers = []
        recovery_references = []
        source_jacobian_reuse_policy = {'existing_default_prediction_refresh_ratio_unchanged': True}
        start = saved_q.copy()
        def checkpoint_identity(self):
            return {'saved': 'unchanged'}
        def solver_kwargs(self):
            return dict(reuse_source_jacobian=True, adaptive_source_backtracking=True,
                initial_line_search_fraction=1./4096, initial_damping=5.830856904508203e-7,
                checkpoint_identity=self.checkpoint_identity())
    owner = Owner()
    driver = SimpleNamespace(solve_joint_source=solver.solve_joint_source, solver_runtime=solver)
    install_prediction_refresh(owner, driver=driver, reset_initial_fresh_source_fraction=True)
    args = owner.solver_kwargs()
    assert args['source_jacobian_refresh_ratio'] == .25
    assert args['initial_line_search_fraction'] == 1.
    assert args['initial_damping'] == 5.830856904508203e-7
    assert np.array_equal(owner.start, saved_q)
    reset = args['checkpoint_identity']['fresh_source_initial_search_seed_reset']
    assert reset['saved_fraction'] == 1./4096
    assert reset['saved_canonical_modified'] is False
    assert driver.solve_joint_source is not solver.solve_joint_source
    assert owner.checkpoint_identity()['source_J_execution_policy']['existing_default_prediction_refresh_ratio_unchanged'] is False


def test_public_loop_refreshes_and_tries_new_direction_from_full_seed(tmp_path):
    fresh, accepted = [], []
    def blocks(q, derivative):
        x, y = q
        if derivative:
            fresh.append(x)
        yield SourceRowBlock(np.array([1.-x+7.5*x*x, y]),
            csr_matrix([[-1.+15.*x, 0.], [0., 1.]]) if derivative else None,
            np.ones(2, bool), np.ones(2),
            dict(family='nonlinear_prediction_test', original_start=0, original_stop=2))
    def checkpoint(q, event):
        if event['checkpoint_kind'] == 'accepted':
            accepted.append(q.copy())
            if len(accepted) == 2:
                return dict(stop_branch=True, reason='Two public numerical transitions checked')
    def unused(*args):
        raise AssertionError('Streaming source required')
    implementation = compile_quality_solver(solver.solve_joint_source)
    result = implementation(unused, np.zeros(2), np.ones(2, bool), np.ones(2), tmp_path,
        lambda q: True, source_block_factory=blocks, cache_native_source_replay=True,
        reuse_source_jacobian=True, source_jacobian_refresh_ratio=.25,
        global_schur_count=1, schur_backtracking=True, adaptive_source_backtracking=True,
        initial_line_search_fraction=.125, checkpoint_callback=checkpoint, verbose=False)
    first, second = result.history
    assert first['source_jacobian_reuse']['actual_to_predicted_reduction'] < .25
    assert first['adaptive_source_backtracking']['next_line_search_fraction'] == 1.
    assert second['adaptive_source_backtracking']['initial_line_search_fraction'] == 1.
    assert second['source_jacobian_reuse']['exact_current_J']
    assert fresh[1] == accepted[0][0]
    assert second['next_cost'] < first['next_cost'] < first['cost']


@pytest.mark.skipif(not (LOG.parent / 'native_pp_continuation.py').is_file(),
    reason='Programme continuation runtime unavailable')
@pytest.mark.parametrize('explicit_initializer', [False, True])
def test_identity_entry_installs_refresh_after_authenticated_fresh_source_start(tmp_path, monkeypatch, explicit_initializer):
    # Exercise the real replay/install chain with a tiny immutable inventory.
    # Only the already-tested complete checkpoint loader and worker backend
    # are inert: no source worker, projection, new J or fit is started here.
    from calibration.program_tracking import first_shot_fused_execution as fused
    from calibration.program_tracking import first_shot_raw_execution as raw
    from calibration.program_tracking import first_shot_resume_cache as cache
    from calibration.program_tracking import first_shot_source_transition as transition
    program = Path(__file__).resolve().parents[1] / 'events/ankara_2026/programmes/okada_mei_sp'
    monkeypatch.syspath_prepend(str(program))
    entry = importlib.import_module('fit_post_native_identity')
    frames = [dict(index=1, native_pts=256256, source_point_count=1,
        rows=[dict(source_point_ids=['ordinary-point'], uv=np.array([[12., 34.]]),
            target=dict(kind='ice_curve', curve_id='B1'))])]
    inventory, _ = cache.source_inventory(SimpleNamespace(frames=frames))
    run = tmp_path / 'native_run'
    run.mkdir()
    inventory_path = run / 'inventory.npz'
    np.savez(inventory_path, **inventory)
    manifest = dict(base_binding={'path': 'old_binding', 'sha256': 'old'},
        base_association={'path': 'old_association', 'sha256': 'old'},
        revised_binding={'path': 'new_binding', 'sha256': 'new'},
        revised_association={'path': 'new_association', 'sha256': 'new'})
    def document(path, data):
        path.write_text(json.dumps(data), encoding='utf8')
        return transition.reference(path)
    document(run / 'run_identity.json', dict(binding=manifest['base_binding'],
        association=manifest['base_association'], source_inventory=transition.reference(inventory_path)))
    prior_path = run / 'current.json'
    document(prior_path, dict(snapshot={'path': str(run / 'snapshots/state.npz')},
        source_objective_cost=123.))
    manifest_path = tmp_path / 'transition.json'
    manifest_ref = document(manifest_path, manifest)
    monkeypatch.setattr(entry, 'manifest_path', manifest_path)
    monkeypatch.setattr(entry, 'initializer_path', None)
    monkeypatch.setattr(entry, 'fixed_scene', False)
    monkeypatch.setattr(entry, 'frame_export_directory', None)
    monkeypatch.setattr(entry.delta, 'read_transition', lambda path: (manifest_ref, manifest))
    def loaded_previous(replay, adapter, **kwargs):
        assert adapter.association_reference == manifest['base_association']
        if explicit_initializer:
            np.testing.assert_array_equal(adapter.start, np.arange(4, dtype=float)-1.)
            assert kwargs['checkpoint'] == 'accepted-parent-canonical'
        replay.adapter = adapter
        replay.q = np.empty(0)
        replay.damping, replay.fraction = 5.830856904508203e-7, 1./4096
        replay.receipt = dict(source_rows=2, source_points=1)
    monkeypatch.setattr(entry.native.BaseFreshCurrentReplay, '__init__', loaded_previous)
    monkeypatch.setattr(raw, 'install', lambda owner: None)
    monkeypatch.setattr(fused, 'install_worker_execution', lambda owner, **kwargs: {})
    monkeypatch.setattr(entry.native, 'pivot_roundoff', False)
    monkeypatch.setattr(entry.native, 'live_review', False)
    driver = entry.reuse.mount.driver
    monkeypatch.setattr(driver, 'solve_joint_source', driver.solver_runtime.solve_joint_source)
    class Owner:
        def __init__(self):
            self.start = np.arange(4, dtype=float)
            self.scales = np.ones(2)
            self.source_producers, self.recovery_references = [], []
            self.seed_receipt = dict(initial_damping=5.830856904508203e-7,
                initial_line_search_fraction=1./4096)
            self._unmeasured_mount_domain = {'released': True}
            self.association_reference = manifest['revised_association']
            self.bundle = SimpleNamespace(frames=frames,
                state=SimpleNamespace(binding_reference=manifest['revised_binding']),
                _post_identity_previous_frames=frames,
                _post_identity_previous_state=SimpleNamespace(),
                _post_identity_previous_scales=np.ones(2),
                _post_identity_previous_mask=np.zeros(2, bool),
                post_native_identity_receipt={'source_scope': {'verified_target_changes': 1}})
        def checkpoint_identity(self):
            return {}
        def checkpoint_arrays(self):
            return {'native_pts': np.array([256256])}
        def solver_kwargs(self):
            return dict(cache_native_source_replay=True, source_block_factory=lambda q, j: (),
                adaptive_source_backtracking=True, initial_line_search_fraction=1./4096,
                initial_damping=5.830856904508203e-7, checkpoint_identity=self.checkpoint_identity())
        def close(self):
            raise AssertionError('Inert installation should succeed')
    owner = Owner()
    saved_q = owner.start.copy()
    monkeypatch.setattr(entry.reuse, 'output_directory', tmp_path / 'candidate')
    options = dict(pool_transition=tmp_path, prior_report=prior_path)
    if explicit_initializer:
        from calibration.program_tracking import first_shot_parameter_initializer as initial
        seed = tmp_path / 'explicit.npz'
        np.savez(seed, q=saved_q)
        recipe_path = tmp_path / 'explicit.json'
        document(recipe_path, {'explicit': True})
        monkeypatch.setattr(entry, 'initializer_path', recipe_path)
        monkeypatch.setattr(initial, 'read_parameter_initializer', lambda *a, **kw: (
            transition.reference(recipe_path), dict(initializer=transition.reference(seed),
                accepted_parent_report=transition.reference(prior_path),
                accepted_parent_checkpoint={'path': 'accepted-parent-canonical'}),
            dict(q=saved_q, aux__native_pts=np.array([256256])), saved_q-1.,
            dict(parameter_q_sha256='explicit-rescue-token')))
        options['checkpoint'] = seed
    replay = entry.IdentityReplay(owner, **options)
    assert replay._new_source_ref == manifest_ref
    assert replay.install() is replay
    args = owner.solver_kwargs()
    assert owner._initial_complete_J_reuse is False
    assert args['source_jacobian_refresh_ratio'] == .25
    assert args['initial_line_search_fraction'] == 1.
    assert args['initial_damping'] == replay.damping
    assert np.array_equal(owner.start, saved_q)
    assert replay.receipt['latest_accepted_q_aux_damping_and_fraction_preserved'] is (not explicit_initializer)
    if explicit_initializer:
        assert replay.receipt['accepted_old_source_resume_claimed'] is False
        assert replay.receipt['complete_rescue_q_and_all_accepted_aux_damping_fraction_preserved']
    assert replay.receipt['prediction_quality_source_J_refresh'] == owner.prediction_refresh_receipt
    assert replay.receipt['source_J_execution_policy'] == owner.source_jacobian_reuse_policy
    assert args['checkpoint_identity']['initial_exact_complete_current_state_full_J_reuse'] is False
    assert args['checkpoint_identity']['fresh_source_initial_search_seed_reset']['saved_fraction'] == replay.fraction
