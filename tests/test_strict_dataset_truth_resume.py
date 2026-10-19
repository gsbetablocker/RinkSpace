from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import install
from calibration.solver_runtime import strict_dataset_truth_resume as resume
from calibration.solver_runtime.fixed_seed_truth_envelope import make_fixed_seed_envelope


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
    return dict(path=str(path), sha256=resume.file_sha(path))


@pytest.fixture
def case(tmp_path):
    install()
    import run_current_named66_full_programme as driver
    q = np.array([.125, .25], np.float64)
    qsha = hashlib.sha256(q.tobytes()).hexdigest()
    state = tmp_path / 'complete_state.npz'
    np.savez(state, q=q)
    authority = write_json(tmp_path / 'authority.json', dict(source='actual_original_ice'))
    ids = ['zero_material', 'positional_stroke']
    limits, envelope = make_fixed_seed_envelope([0., .01801501778965431], [0., 4.],
        seed_q_sha256='1' * 64, group_ids=ids)
    envelope['original_recorded_engineering_uncertainties_px'] = [0., .5]
    factors = dict(native=SimpleNamespace(contract=dict(authority=authority),
        records=[dict(row=dict(id=identity)) for identity in ids]))
    old = {key: dict(preserved=key) for key in resume.PRESERVED_MANIFEST_FIELDS}
    old.update(full_parameters=2, source_rows=20,
        source_contract=dict(schema='archived_complete_pixel_source'),
        principal_point_engineering_prior=dict(rows=6, target=[960., 540.]),
        current_geometry=dict(coordinate_chart=dict(parameter_count=2), programme_blocks=1,
            world_geometry=dict(schema='physical_world', parameter_count=2,
                parameter_names=['course_x', 'pivot_z'], original_world66_prefix_count=2)),
        joint_source_truth_restoration={key: dict(preserved=key)
                                      for key in resume.PRESERVED_RESTORATION_FIELDS})
    old['joint_source_truth_restoration'].update(progress_policy='fixed_seed_envelope',
                                                seed_engineering_scale_source=None)
    old_ref = write_json(tmp_path / 'old_manifest.json', old)
    packet = dict(schema='rinkspace.joint-source-truth-restoration-state.v1',
        complete_q_sha256=qsha, complete_state_reference=dict(path=str(state), sha256=resume.file_sha(state)),
        source_input_manifest_reference=old_ref, source_target=driver.truth_restoration_source_target(old),
        source_role_operator_target=driver.role_target_from_restored_manifest(old),
        families=dict(native=dict(authority=authority, group_ids=ids,
                                  limits=limits.tolist(), uncertainties=[0., .5])),
        progress_policy='fixed_seed_envelope', fixed_seed_envelopes=dict(native=envelope),
        seed_engineering_scale_source=None, historical_group_minimum_limits_preserved=False)
    truth = tmp_path / 'truth.json'
    write_json(truth, packet)
    frozen = write_json(tmp_path / 'state_identity.json', dict(q_sha256=qsha))
    source = dict(video_sha256='a' * 64, time_base='1/15360', pixel_domain='native1920x1080')
    dataset = write_json(tmp_path / 'dataset.json', dict(schema=resume.DATASET_SCHEMA,
        fit_points=3, held_points=1, source=source, frozen_model_reference=frozen))
    motion = write_json(tmp_path / 'motion.json',
        dict(schema='rinkspace.strict-model-scope-original-motion-inputs.v1',
             unknown_or_unreviewed_default='excluded', source=source, blocks=[dict(after_points=0)]))
    finite = dict(schema='rinkspace.strict-reviewed-finite-dataset-factor.v1',
        dataset_reference=dataset, source_video=source, original_fit_points=3,
        held_points=1, scalar_rows=6, parameter_count=2, held_used_in_fit=False)
    strict = dict(schema='rinkspace.strict-reviewed-finite-programme-source.v1',
        archived_pixel_factors_in_FIT=False, archived_pixel_guards_evaluated=False,
        archived_motion_profiles_evaluated=False, all_pixel_families_routed_to_strict_input=True,
        admitted_motion_points=0, finite_dataset_contract=finite,
        strict_motion_reference=motion, source_identity=source,
        parameter_count=2, source_rows=9, engineering_prior_rows=3)
    current = deepcopy(old)
    current.update(source_rows=15, strict_source_dataset=strict, legacy_pixel_objective_replaced=True,
        source_contract=dict(schema='current_canonical_source', evaluation_source_contract=strict),
        original_board_curve_integration=None, original_ice_curve_integration=None,
        original_wide_curve_integration=None, native_base_curve_integration=None)
    old.update(original_board_curve_integration=dict(old_loss=True),
        original_ice_curve_integration=dict(old_loss=True), original_wide_curve_integration=dict(old_loss=True),
        native_base_curve_integration=dict(old_loss=True))
    old_ref = write_json(tmp_path / 'old_manifest.json', old)
    packet['source_input_manifest_reference'] = old_ref
    write_json(truth, packet)
    return SimpleNamespace(path=truth, q=q, current=current, old=old, packet=packet,
                           factors=factors, driver=driver, state=state)


def invoke(c, **kwargs):
    return resume.load_for_strict_revision(c.path, c.q, c.current, c.factors,
        archived_loader=c.driver.load_truth_restoration_state,
        source_target=c.driver.truth_restoration_source_target,
        role_target=c.driver.role_target_from_restored_manifest, **kwargs)


def test_new_dataset_loss_retires_old_loss_but_preserves_every_cap_and_seed_byte(case):
    source_bytes = case.path.read_bytes()
    state_bytes = case.state.read_bytes()
    actual = invoke(case)
    for key in resume.PROTECTED_PACKET_FIELDS:
        assert actual.get(key) == case.packet.get(key)
    assert resume._protected_digest(actual) == resume._protected_digest(case.packet)
    assert actual['source_target'] != case.packet['source_target']
    assert actual['source_target']['source_rows'] == 15
    receipt = actual['authenticated_strict_dataset_source_revision']
    assert receipt['source_loss_objective_revised']
    assert receipt['old_pixel_cost_preserved'] is False
    assert receipt['old_source_J_profiles_normals_reuse_authorized'] is False
    assert case.path.read_bytes() == source_bytes and case.state.read_bytes() == state_bytes
    reread = json.loads(json.dumps(actual, allow_nan=False))
    assert resume._protected_digest(reread) == resume._protected_digest(case.packet)


def test_old_normal_loader_rejects_new_pixel_target_without_explicit_revision(case):
    with pytest.raises(ValueError, match='mathematical source target differs'):
        case.driver.load_truth_restoration_state(case.path, case.q, case.current, case.factors)


@pytest.mark.parametrize('key', ['highest_direct_human_hard_truth', 'human_truth_support_operator',
    'camera_pivot_exterior_constraints', 'complete_parameter_layout', 'physical_geometry_initializer',
    'principal_point_engineering_prior', 'latest9a8c_reference18_preserved'])
def test_retiring_old_pixel_loss_cannot_change_manual_or_physical_contract(case, key):
    case.current[key] = dict(changed=True)
    with pytest.raises(ValueError, match='protected manual/physical contract'):
        invoke(case)


def test_q_value_equal_but_bytes_changed_and_state_file_change_are_rejected(case):
    case.q = case.q.astype(np.float32)
    with pytest.raises(ValueError, match='exact complete saved q bytes'):
        invoke(case)
    case.q = case.q.astype(np.float64)
    with case.state.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        invoke(case)


def test_revised_authority_and_group_order_are_rejected_by_original_authentication(case):
    case.factors['native'].records.reverse()
    with pytest.raises(ValueError, match='group identity/order differs'):
        invoke(case)


def test_original_zero_groups_and_fixed_seed_limits_cannot_be_expanded(case):
    packet = json.loads(case.path.read_text(encoding='utf8'))
    packet['families']['native']['limits'][0] = 1.
    packet['fixed_seed_envelopes']['native']['limits_px'][0] = 1.
    write_json(case.path, packet)
    with pytest.raises(ValueError, match='Saved envelope differs'):
        invoke(case)


def test_strict_reference_sha_and_held_separation_are_required(case):
    ref = case.current['strict_source_dataset']['finite_dataset_contract']['dataset_reference']
    with open(ref['path'], 'a', encoding='utf8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='bytes changed'):
        invoke(case)


def test_retired_pixels_cannot_reenter_through_the_new_complete_source_contract(case):
    case.current['strict_source_dataset']['archived_pixel_factors_in_FIT'] = True
    with pytest.raises(ValueError, match='only the strict finite dataset route'):
        invoke(case)
