from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime.strict_source_dataset_adapter import StrictDatasetFactor
from calibration.solver_runtime.strict_dataset_camera_runtime import (
    compare_strict_dataset_camera_mapping, strict_dataset_camera_mapping,
    install_strict_dataset_camera_execution)


def physical_camera_factor(pivot_height):
    from calibration.solver_runtime import install
    install()
    from z190_physical_camera import SharedCalibration
    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle
    from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeMountOpticsChart
    from structured_programme_source_candidate.paired_programme_blocks import StructuredProgrammeBlock

    calibration = SharedCalibration('qualified-test-installation', (23., 47., 5.),
        lever_xyz=(.31, -.17, .09), q_reference=-.18,
        aspect_fx_over_fy=1.007, principal_x_fraction=(.493,), principal_y_fraction=(.511,),
        k1=(-.03,), k2=(-.002, .0001), tangential_p1_p2=(.0002, -.0001), axial=(0., .001))
    chart = PairedProgrammeMountOpticsChart(calibration.installation_id,
        (34204,34205,34206), 34204, mount_indices=(34195,34196,34197),
        additional_world_parameter_count=34)
    core_count = chart.coordinate_chart.core_count
    bundle = FixedLeverXYZSourceBundle.__new__(FixedLeverXYZSourceBundle)
    bundle.calibration = calibration; bundle.index = {100:0,200:1}
    block = StructuredProgrammeBlock.__new__(StructuredProgrammeBlock)
    block.name = 'real_chart_fixture'; block.bundle = bundle
    block.xyz_slots=np.array([5,6,7]); block.start=14
    block.optics_chart=chart; block.optics_base_calibration=calibration
    block.q0=np.r_[np.asarray(calibration.pivot_xyz)/100.,0.,np.zeros(8)]
    block.geometry_parameter_indices=np.asarray(chart.coordinate_chart.world_indices)
    def validate(world):
        assert world.shape==(100,) and np.isfinite(world).all()
    block.geometry=SimpleNamespace(validate_parameters=validate)
    q=np.zeros(core_count+31); q[block.xyz_slots]=[7.2,14.3,pivot_height]
    q[np.asarray(chart.indices)]=[3.4,-2.7,8.2]
    q[np.asarray(chart.mount_indices)]=[.006,-.003,.005]
    q[14:18]=[.32,-.21,.17,np.log(1520.)]
    q[18:22]=[-.17,.31,-.26,np.log(2140.)]
    factor=StrictDatasetFactor.__new__(StrictDatasetFactor)
    factor.parameter_count=len(q); factor.core_count=core_count
    factor.state=SimpleNamespace(bypts={100:block,200:block})
    factor._evaluation_q=factor._world_cache_q=factor._camera_cache_q=None
    factor._world_cache={};factor._derivative_cache={};factor._camera_cache={}
    return factor,q,block


@pytest.mark.parametrize('pivot_height',[3.1,1e-8])
def test_original_complete_camera_FD_and_near_zero_inward_stencil_match_analytical_map(pivot_height):
    factor,q,block=physical_camera_factor(pivot_height)
    before=q.tobytes()
    receipt=compare_strict_dataset_camera_mapping(factor,q,[100,200])
    assert receipt['passed'] and not receipt['factor_method_installed']
    assert q.tobytes()==before
    assert all(c['nominal_camera25_bytes_identical'] for c in receipt['checks'])
    assert all(len(c['global_columns'])==13 for c in receipt['checks'])
    # Mutating every real camera coordinate still gives exact nominal camera
    # bytes; no cached calibration or view can survive a changed current q.
    changed=q.copy()
    changed[block.xyz_slots[:2]]+=[.013,-.024]
    changed[np.asarray(block.optics_chart.indices)]+=[.21,-.16,.18]
    changed[np.asarray(block.optics_chart.mount_indices)]+=[.002,.003,-.001]
    changed[14:18]+=[.004,-.003,.002,.007]
    updated=compare_strict_dataset_camera_mapping(factor,changed,[100,200])
    assert updated['passed']


def test_prepared_camera_uses_only_one_native_local_chart_access():
    factor,q,block=physical_camera_factor(3.1)
    calls=[]; original=block.camera_local
    def observed(value):
        calls.append(1)
        return original(value)
    block.camera_local=observed
    strict_dataset_camera_mapping(block,q,100,core_count=factor.core_count)
    assert len(calls)==1
    calls.clear(); factor._prepare_evaluation(q)
    factor._camera(q,100,True)
    assert len(calls)==28


def test_unsupported_lens_polynomial_is_refused():
    from dataclasses import replace
    factor,q,block=physical_camera_factor(3.1)
    block.optics_base_calibration=replace(block.optics_base_calibration,k1=(-.03,.0001))
    with pytest.raises(ValueError):
        strict_dataset_camera_mapping(block,q,100,core_count=factor.core_count)


def test_installer_qualifies_original_FD_before_complete_preflight_and_retains_nominal_on_new_q():
    factor,q,block=physical_camera_factor(3.1)
    factor.family_groups={'real_chart_fixture':[dict(actual_pts=100),dict(actual_pts=200)]}
    observed=[]
    def original_preflight(value):
        observed.append(factor._camera(value,100,True))
        return dict(passed=True)
    factor.preflight=original_preflight
    receipt=install_strict_dataset_camera_execution(factor)
    assert not receipt['prepared_mapping_installed']
    result=factor.preflight(q)
    assert result['passed'] and receipt['prepared_mapping_installed']
    assert len(receipt['qualification']['checks'])==2
    changed=q.copy();changed[14:18]+=[.001,-.002,.003,.004]
    factor._prepare_evaluation(changed)
    calls=[];original_local=block.camera_local
    def seen(value):
        calls.append(1)
        return original_local(value)
    block.camera_local=seen
    actual=factor._camera(changed,100,True)
    assert len(calls)==2 # original nominal plus analytical mapping
    original_camera=StrictDatasetFactor._camera(factor,changed,100,False)[0]
    assert actual[0].tobytes()==original_camera.tobytes()
    assert actual[1].shape==(13,)
    again=factor._camera(changed,100,True)
    assert again is actual and len(calls)==2


def test_qualification_failure_never_installs_prepared_camera_or_runs_preflight(monkeypatch):
    factor,q,block=physical_camera_factor(3.1)
    factor.family_groups={'real_chart_fixture':[dict(actual_pts=100),dict(actual_pts=200)]}
    factor.preflight=lambda value:pytest.fail('Original preflight must not run after failed qualification')
    original_camera=factor._camera
    from calibration.solver_runtime import strict_dataset_camera_runtime as runtime
    original_mapping=runtime.strict_dataset_camera_mapping
    def wrong(*args,**kwargs):
        camera,columns,chain=original_mapping(*args,**kwargs)
        chain=chain.copy();chain[0,0]+=.01
        return camera,columns,chain
    monkeypatch.setattr(runtime,'strict_dataset_camera_mapping',wrong)
    receipt=install_strict_dataset_camera_execution(factor)
    with pytest.raises(ValueError,match='complete-column FD qualification'):
        factor.preflight(q)
    assert factor._camera==original_camera
    assert not receipt['prepared_mapping_installed'] and receipt['qualification'] is None
