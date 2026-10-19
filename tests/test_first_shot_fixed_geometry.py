from types import SimpleNamespace
from pathlib import Path
import importlib.util

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime import install
install()
import z190_joint_source_solver as solver
from z190_streaming_source_normal import SourceRowBlock
from calibration.program_tracking.first_shot_fixed_geometry import (
    install_fixed_geometry, scene_camera_parameter_groups)


class ExistingDomainOwner:
    def __init__(self):
        self.start = np.array([2., .5, 0., 0., 0., 0.])
        self.state = SimpleNamespace(world_parameter_indices=np.array([0, 1]),
            count=6, frame_count=1, pp_columns=lambda knot: np.array([4, 5]))
        self.shared = np.array([0, 1, 2])
        self.fixed = np.array([4, 5])  # Historical zero PP gauge retains its meaning.
        self.source_producers, self.recovery_references = [], []
        self.calls, self.recoveries = [], []
        self.corrupt_recovery = False
        self.lower = None
    def checkpoint_identity(self):
        return {'original_zero_gauge': self.fixed.tolist()}
    def linear_constraints(self, q, selected):
        np.testing.assert_array_equal(selected, self.shared)
        # Fixed geometry still contributes 2*q0 to the actual current margin.
        self.lower = np.array([5. - (2.*q[0] + q[1] + q[2])])
        self.calls.append(q.copy())
        return dict(matrix=np.array([[2., 1., 1.]]), lower=self.lower,
            retained_payload=self.calls, report={'original_course_face': True})
    def guard(self, q):
        assert np.all(q[self.fixed] == 0.)
        return True
    def recover(self, parent, proposed):
        result = proposed.copy()
        if self.corrupt_recovery:
            result[0] += 1.
        self.recoveries.append(result.copy())
        return dict(q=result, report={'original_operator_called': True})
    def solver_kwargs(self):
        return dict(global_schur_count=3, global_schur_indices=self.shared,
            fixed_parameter_indices=self.fixed, shared_linear_constraints=self.linear_constraints,
            trial_feasibility_recovery=self.recover, checkpoint_identity=self.checkpoint_identity(),
            schur_backtracking=True, adaptive_source_backtracking=True,
            initial_damping=.001, initial_line_search_fraction=1.)


def restrict(owner):
    return install_fixed_geometry(owner, fixed_parameter_indices=[0],
        fixed_parameter_values=[2.], native_pp_values=np.zeros((1, 2)))


def test_original_margin_is_preserved_and_recovery_cannot_reopen_geometry():
    owner = ExistingDomainOwner()
    original = owner.start.copy()
    receipt = restrict(owner)
    result = owner.linear_constraints(owner.start, np.array([1, 2]))
    np.testing.assert_array_equal(result['matrix'], [[1., 1.]])
    assert result['lower'] is owner.lower and result['lower'][0] == .5
    assert result['retained_payload'] is owner.calls
    assert result['report']['original_course_face']
    np.testing.assert_array_equal(owner.fixed, [4, 5])
    np.testing.assert_array_equal(owner.solver_kwargs()['fixed_parameter_indices'], [4, 5, 0])
    np.testing.assert_array_equal(owner.start, original)
    assert receipt['native_principal_offsets_restored_to_zero']
    owner.corrupt_recovery = True
    with pytest.raises(ValueError, match='fixed geometry changed in recovery result'):
        owner.recover(owner.start, owner.start.copy())
    changed = owner.start.copy(); changed[4] = 1.
    with pytest.raises(ValueError, match='fixed geometry changed in physical guard'):
        owner.guard(changed)


def test_public_solver_keeps_scene_and_native_PP_fixed_while_print_and_camera_fit(tmp_path):
    owner = ExistingDomainOwner()
    restrict(owner)
    J = csr_matrix([[1., 0., 0., 0., 0., 0.],
        [0., 1., .5, .1, 0., 0.], [0., .2, 1., -.2, 0., 0.],
        [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])
    target = np.array([9., 2., -.2, 100., -100.])
    mask, scales = np.ones(5, bool), np.ones(5)
    def blocks(q, derivative):
        yield SourceRowBlock(J@q-target, J if derivative else None, mask, scales,
            dict(family='fixed_scene_original_source', original_start=0, original_stop=5))
    def stop(q, event):
        if event['checkpoint_kind'] == 'accepted':
            return {'stop_branch': True, 'reason': 'Small public solver interface verified'}
    def unused(*args):
        raise AssertionError('Original streaming source must be used')
    result = solver.solve_joint_source(unused, owner.start, mask, scales, tmp_path,
        owner.guard, source_block_factory=blocks, cache_native_source_replay=True,
        checkpoint_callback=stop, verbose=False, **owner.solver_kwargs())
    assert result.status == 'stopped_by_source_review' and result.history[0]['accepted']
    np.testing.assert_array_equal(result.q[[0, 4, 5]], owner.start[[0, 4, 5]])
    assert result.q[1] != owner.start[1] and result.q[2] != owner.start[2]
    assert result.q[1] + result.q[2] >= 1. - 1e-12
    assert owner.calls and owner.recoveries
    assert result.history[0]['global_schur_linear_report']['declared_fixed_shared_elimination'][
        'algebraically_eliminated_fixed_shared_indices'] == [0]


METRIC = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac/mount_source_metric_direction.py')


@pytest.mark.skipif(not METRIC.is_file(), reason='Original programme conditional metric unavailable')
def test_unused_released_mount_metric_can_inspect_original_chart_after_J_yield(tmp_path):
    # Reproduce the real first-J failure boundary: the source producer yields
    # every block and only then constructs the historical auxiliary metric.
    spec = importlib.util.spec_from_file_location('original_conditional_mount_metric_test', METRIC)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    owner = ExistingDomainOwner()
    owner._unmeasured_mount_domain = dict(
        conditional_source_metric_may_still_be_built_but_never_projects_to_photo_box=True)
    owner.state.shared_parameter_bounds = {}
    restrict(owner)
    metrics = []
    J = csr_matrix([[1., 0., 0., 0., 0., 0.],
        [0., 1., .5, .1, 0., 0.], [0., .2, 1., -.2, 0., 0.],
        [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])
    target = np.array([9., 2., -.2, 100., -100.])
    mask, scales = np.ones(5, bool), np.ones(5)
    def blocks(q, derivative):
        residual = J@q-target
        yield SourceRowBlock(residual, J if derivative else None, mask, scales,
            dict(family='actual_auxiliary_metric_protocol', original_start=0, original_stop=5))
        if derivative:
            metrics.append(module.ConditionalMountDirection(owner, q, (J.T@J).tocsc(),
                np.asarray(J.T@residual).ravel(), .001, columns=[2],
                mount_support=lambda *a: None, mount_values=lambda state, value: value[[2]]))
    def stop(q, event):
        if event['checkpoint_kind'] == 'accepted':
            return {'stop_branch': True, 'reason': 'Real original full-chart metric built after J'}
    result = solver.solve_joint_source(lambda *a: None, owner.start, mask, scales, tmp_path,
        owner.guard, source_block_factory=blocks, cache_native_source_replay=True,
        checkpoint_callback=stop, verbose=False, **owner.solver_kwargs())
    assert metrics and result.history[0]['accepted']
    np.testing.assert_array_equal(metrics[0].selected, owner.shared)
    np.testing.assert_array_equal(result.q[[0, 4, 5]], owner.start[[0, 4, 5]])
    assert result.q[1] != owner.start[1] and result.q[2] != owner.start[2]
    assert owner.linear_constraints(owner.start, owner.shared)['report'][
        'fixed_geometry_tangent_pullback']['unused_legacy_metric_full_chart_inspection']
    assert owner.linear_constraints(owner.start, np.array([1, 2]))['report'][
        'fixed_geometry_tangent_pullback']['actual_public_solver_fixed_elimination']


def test_groups_derive_scene_print_deviation_and_PP_from_actual_chart():
    original = SimpleNamespace(world_parameter_indices=np.array([0, 1, 7, 8]),
        print_layout_global_indices_by_ad={2: np.array([7, 8])})
    state = SimpleNamespace(original=original, world_parameter_indices=np.array([0, 1, 7, 8, 10]),
        deviation_indices=np.array([10]), count=11, frame_count=2,
        pp_columns=lambda knot: np.array([3 + 2*knot, 4 + 2*knot]))
    groups = scene_camera_parameter_groups(state)
    np.testing.assert_array_equal(groups['base_scene_geometry'], [0, 1])
    np.testing.assert_array_equal(groups['print_layout'], [7, 8])
    np.testing.assert_array_equal(groups['added_geometry_deviations'], [10])
    np.testing.assert_array_equal(groups['native_principal_offsets'], [3, 4, 5, 6])


def test_fixed_values_must_match_explicit_prepared_state():
    owner = ExistingDomainOwner()
    with pytest.raises(ValueError, match='fixed geometry changed in prepared start'):
        install_fixed_geometry(owner, fixed_parameter_indices=[0], fixed_parameter_values=[3.])
    assert not hasattr(owner, 'fixed_geometry_receipt')
    with pytest.raises(ValueError, match='all-zero native principal offsets'):
        install_fixed_geometry(owner, fixed_parameter_indices=[0], fixed_parameter_values=[2.],
            native_pp_values=np.ones((1, 2)))
