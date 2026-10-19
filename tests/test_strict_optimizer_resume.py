from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json

import numpy as np
import pytest

from calibration.solver_runtime import strict_optimizer_resume as runtime


V7_MANIFEST = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/'
    'solver_acceleration_20261003/okada_strict_v6_8a28_fast_v7_20261013/input_manifest.json')


@pytest.fixture
def actual_target():
    if not V7_MANIFEST.exists():
        pytest.skip('External real strict V7 manifest absent')
    old = json.loads(V7_MANIFEST.read_text(encoding='utf8'))
    current = deepcopy(old['strict_source_dataset'])
    current['strict_dataset_execution']['test_idle_receipt'] = 'changed execution state'
    finite = current['finite_dataset_contract']
    finite['authenticated_FIT_consumer_seconds'] += 50
    finite['complete_factor_preparation_seconds'] += 50
    finite['strict_source_read_receipt']['seconds'] += 50
    for pack in finite['strict_source_read_receipt']['source_packs']:
        pack['seconds'] += 50
    protected = {key: deepcopy(old[key]) for key in (
        'full_parameters', 'complete_parameter_layout', 'geometry_world_indices',
        'physical_geometry_initializer', 'latest9a8c_reference18_preserved',
        'correlated_roll_prior', 'principal_point_engineering_prior',
        'camera_mount_working_constraints', 'camera_pivot_exterior_constraints',
        'roll_prior_component_sigma_degrees', 'user_trace_sha256')}
    return old, current, protected


def _same(target, **kw):
    old, current, protected = target
    values = dict(old_quadratic=np.array([False, True]), quadratic=np.array([False, True]),
        old_scales=np.array([2., 1.]), scales=np.array([2., 1.]),
        protected_fields=protected, full_truth_resume_required=True)
    values.update(kw)
    return runtime.same_strict_optimizer_target(old, current, **values)


def test_actual_v7_identity_ignores_only_explicit_execution_fields(actual_target):
    assert _same(actual_target)
    assert len(actual_target[1]['finite_dataset_contract']['strict_source_read_receipt']['source_packs']) == 31
    assert actual_target[1]['finite_dataset_contract']['original_fit_points'] == 1282102


@pytest.mark.parametrize('mutation', ['dataset_sha', 'loss', 'binding', 'layout', 'pack_hash',
    'motion_sha', 'operator', 'camera_mapping', 'principal_point'])
def test_real_target_identity_changes_never_inherit(actual_target, mutation):
    _, current, protected = actual_target
    finite = current['finite_dataset_contract']
    if mutation == 'dataset_sha': finite['dataset_reference']['sha256'] = '0' * 64
    elif mutation == 'loss': finite['source_loss_policy']['normalized_component_huber_delta'] = 3.
    elif mutation == 'binding': finite['confirmed_original_target_label_bindings']['effective_scope'] = 'external_border'
    elif mutation == 'layout': finite['logical_world_parameter_indices'][0] += 1
    elif mutation == 'pack_hash': finite['strict_source_read_receipt']['source_packs'][0]['sha256'] = '0' * 64
    elif mutation == 'motion_sha': current['strict_motion_reference']['sha256'] = '0' * 64
    elif mutation == 'operator': finite['operator'] += ' changed'
    elif mutation == 'camera_mapping': current['strict_dataset_execution']['candidate_camera_mapping_enabled'] = True
    elif mutation == 'principal_point': protected['principal_point_engineering_prior']['engineering_sigma_native_px'] = 99.
    assert not _same(actual_target)


def test_complete_arrays_and_protected_restore_required(actual_target):
    assert not _same(actual_target, scales=np.array([2., 1., 1., 1., 1., 1., 1., 1.]))
    assert not _same(actual_target, quadratic=np.array([True, True]))
    assert not _same(actual_target, full_truth_resume_required=False)
    assert not _same(actual_target, physical_initialization_changed=True)


def test_zero_motion_only_and_exact_clock(monkeypatch):
    arrays = dict(a=np.empty((0, 2)), b=np.empty((0, 2)), stamps=np.array([10, 20]))
    ref = dict(path='proof', sha256='hash')
    block = SimpleNamespace(name='low', bundle=SimpleNamespace(stamps=np.array([10, 20])))
    calls = []
    def load(path, *, expected_sha256):
        calls.append((path, expected_sha256))
        return dict(low=arrays), dict(review=dict(path='review', sha256='reviewhash'))
    monkeypatch.setattr(runtime, 'load_strict_programme_motion_inputs', load)
    result = runtime.retired_motion_compaction_policy(ref, [block])
    assert calls == [('proof', 'hash')]
    assert result['verify_value_J'] is False
    assert result['archived_motion_numerical_check'] == 'NOT_APPLICABLE_RETIRED_PIXEL_OBJECTIVE'
    arrays['a'] = np.array([[1., 2.]])
    with pytest.raises(ValueError, match='zero admitted'): runtime.retired_motion_compaction_policy(ref, [block])
    arrays['a'] = np.empty((0, 2)); arrays['stamps'] = np.array([10, 21])
    with pytest.raises(ValueError, match='clocks'): runtime.retired_motion_compaction_policy(ref, [block])


def test_driver_prior_arrays_before_inheritance_and_retired_compactor_switch():
    path = Path('calibration/solver_runtime/legacy/run_current_named66_full_programme.py')
    source = path.read_text(encoding='utf8'); ast.parse(source)
    assert source.index('source=wrap_current_principal_point_working_prior') < source.index('initial_damping=.001;')
    assert source.index('retired_motion_compaction_policy(') < source.index('motion_input_compaction=compact_original_motion_inputs(')
    assert 'verify_value_J=retired_motion_compaction is None' in source
    assert 'full_truth_resume_required=bool(opt.joint_source_truth_restoration and opt.truth_restoration_state is not None)' in source
    assert source.index('restored_truth_state=load_for_strict_revision(') < source.index('strict_preflight=strict_dataset_factor.preflight(')
