from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from calibration.program_tracking.first_shot_mount_revision import initialize_restored_mount, read_mount_revision
from calibration.program_tracking.official_static_surface_dataset import reference
from calibration.program_tracking.z190_physical_camera import SharedCalibration, ViewState


def test_restore_changes_one_shared_mount_preserving_mean_native_C_and_other_coordinates():
    base = SharedCalibration(installation_id="test", pivot_xyz=(0., -50., 16.), lever_xyz=(0.,10.,.3))
    q = np.zeros(140)
    q[127:139] = [1.6,-.3,.3,1., 1.6,0.,0.,.8, 1.6,.3,-.3,1.2]
    state = SimpleNamespace(mount_revision_proof={"initial_lever_working_m":[0.,-.2,.1],
                            "lower_lever_working_m":[-.05,-.3,-.05],"upper_lever_working_m":[.05,-.1,.25]},
                            scene_unit_working_m=.3, frame_count=3)
    state.view_columns = lambda n: 127+4*(n-1)+np.arange(4)
    state.calibration = lambda x: replace(base, pivot_xyz=tuple(np.asarray(base.pivot_xyz)+x[118:121]/.3),
                                             lever_xyz=tuple(np.asarray(base.lever_xyz)+x[121:124]/.3))
    before = np.array([state.calibration(q).camera25(ViewState.from_state4(q[state.view_columns(n)]))[:3] for n in (1,2,3)])
    updated, receipt = initialize_restored_mount(state,q)
    after = np.array([state.calibration(updated).camera25(ViewState.from_state4(updated[state.view_columns(n)]))[:3] for n in (1,2,3)])
    np.testing.assert_allclose(after.mean(0),before.mean(0),atol=2e-14)
    np.testing.assert_allclose(np.array(state.calibration(updated).lever_xyz)*.3,[0.,-.2,.1],atol=1e-15)
    mask=np.ones(len(q),bool);mask[118:124]=False
    assert np.array_equal(q[mask],updated[mask])
    assert not np.allclose(after-before,0.)  # one P cannot preserve every C
    assert receipt['clipped_parameter_values'] is False
    assert receipt['native_exposures']==3
    assert np.array_equal(q[118:124],np.zeros(6))
    retained, second = initialize_restored_mount(state,updated)
    assert retained is updated
    assert second['already_within_restored_range'] and not second['mount_reinitialized']


def test_no_revision_preserves_the_historical_parameter_object():
    q=np.arange(20,dtype=float)
    result,proof=initialize_restored_mount(SimpleNamespace(),q)
    assert result is q and proof is None


def test_restore_requires_exact_facts_reference_and_working_scale(tmp_path):
    facts={"metres_per_original_scene_unit":.3,"rigid_mount_offset":dict(
        bounds_are_user_working_search_ranges_not_measurements=True,
        shared_across_same_installation_exposures=True,per_frame_free_optical_centre_enabled=False,
        lower_camera_xyz_m=[-.05,-.3,-.05],initial_camera_xyz_m=[0.,-.2,.1],upper_camera_xyz_m=[.05,-.1,.25])}
    f=tmp_path/'facts.json';f.write_text(json.dumps(facts));fr=reference(f)
    policy=dict(schema="rinkspace.first-shot-restored-user-mount.v1",facts=fr,
        range_policy="restore_user_adopted_v4_working_search_range",initializer="user_lever_and_mean_native_optical_centres",
        working_metres_are_measured=False)
    p=tmp_path/'policy.json';p.write_text(json.dumps(policy))
    assert read_mount_revision(reference(p),fr,.3)['initial_lever_working_m']==[0.,-.2,.1]
    with pytest.raises(ValueError,match='working units'):read_mount_revision(reference(p),fr,.4)
    wrong=dict(fr,sha256='0'*64)
    with pytest.raises(ValueError,match='authenticated'):read_mount_revision(reference(p),wrong,.3)
