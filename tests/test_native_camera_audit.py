"""Complete registered exposure audit with sparse synthetic observations only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import audit_native_cameras as audit


def manifest(rows=2):
    return dict(shots=[dict(id='s07',camera_index=0,row_begin=0,row_stop=rows,
        first_pts=512,last_pts=1024,stop_pts_exclusive=1280,reference_pts=512,
        time_base_num=1,time_base_den=12800,control_begin=0,control_stop=8,
        first_control_time=-.02)],control_dt=.02,first_control_time=0.)


def test_registered_middle_exposure_survives_missing_observations():
    observed=np.array([512,1024],dtype=np.int64);before=observed.copy()
    (pts,coverage),=audit.source_exposure_plan(manifest(),observed)
    np.testing.assert_array_equal(pts,[512,768,1024])
    assert coverage==dict(native_pts_step=256,source_exposures=3,frames_with_observations=2,
        frames_without_observations=1,observation_rows=2,observations_interpolated=False)
    np.testing.assert_array_equal(observed,before)


def test_entire_shot_without_observations_still_has_all_exposures():
    (pts,coverage),=audit.source_exposure_plan(manifest(0),np.empty(0,dtype=np.int64))
    np.testing.assert_array_equal(pts,[512,768,1024])
    assert coverage['frames_with_observations']==0 and coverage['frames_without_observations']==3


@pytest.mark.parametrize('location',['manifest','source_clock','shot'])
def test_native_step_comes_from_registered_metadata_not_sparse_point_spacing(location):
    m=manifest()
    if location=='manifest':m['native_pts_step']=128
    elif location=='source_clock':m['source_clock']=dict(native_pts_step=128)
    else:m['shots'][0]['native_pts_step']=128
    (pts,coverage),=audit.source_exposure_plan(m,np.array([512,1024],dtype=np.int64))
    np.testing.assert_array_equal(pts,[512,640,768,896,1024])
    assert coverage['frames_without_observations']==3


@pytest.mark.parametrize('observed',[
    np.array([512,640],np.int64),  # In range, but off the registered clock.
    np.array([512,1280],np.int64),
    np.array([512.,1024.]),
])
def test_sparse_observations_must_belong_to_registered_shot_exposures(observed):
    with pytest.raises(ValueError):audit.source_exposure_plan(manifest(),observed)


def test_duplicate_measurements_do_not_add_source_frames_and_outside_rows_fail():
    observed=np.array([512,512,1024],np.int64)
    (_,coverage),=audit.source_exposure_plan(manifest(3),observed)
    assert coverage['observation_rows']==3 and coverage['frames_with_observations']==2
    with pytest.raises(ValueError,match='outside registered shot rows'):
        audit.source_exposure_plan(manifest(2),observed)
    m=manifest(3);m['shots'][0]['row_begin']=1
    with pytest.raises(ValueError,match='row partition'):audit.source_exposure_plan(m,observed)


@pytest.mark.parametrize('override',[False,True])
def test_main_sends_missing_middle_frame_to_motion_and_lens_native_interfaces(tmp_path,monkeypatch,override):
    root=tmp_path/'prepared';root.mkdir()
    library=tmp_path/'runtime.dll';library.write_bytes(b'No native runtime is loaded by this test')
    bound=tmp_path/'bound.dll';bound.write_bytes(b'Actual state runtime, mocked by this test')
    selected=bound if override else library
    m=manifest();m.update(native_library=str(library),native_library_sha256=audit.digest(library))
    (root/'joint_manifest.json').write_text(json.dumps(m),encoding='utf8')
    np.savez(root/'point_observations.npz',pts=np.array([512,1024],np.int64))
    state=dict(controls=np.zeros((8,3)),rig=np.zeros((1,43)),
        native_library_provenance_sha256=np.array(audit.digest(selected)),
        joint_numeric_dependencies_json=np.array(json.dumps(dict(control_dt=m['control_dt'],
            first_control_time=m['first_control_time'],shots=copy.deepcopy(m['shots'])))))
    baseline,candidate=tmp_path/'baseline.npz',tmp_path/'candidate.npz'
    np.savez(baseline,**state);np.savez(candidate,**state)
    protected={path:audit.digest(path) for path in [baseline,candidate,root/'point_observations.npz',
                                                  root/'joint_manifest.json',library,bound]}
    read=audit.read
    monkeypatch.setattr(audit,'read',lambda path:dict(converted_span_xy_m=[60.,30.])
        if Path(path).name=='ankara_2026_metric_scale_assumed_v1.json' else read(path))
    core=object()
    def load_core(path):
        assert Path(path).resolve()==selected.resolve()
        return core
    monkeypatch.setattr(audit,'NativeCore',load_core)
    motion_calls=[];lens_calls=[]
    def motion(actual_core,rig,controls,first,dt,times,seconds,scale):
        assert actual_core is core
        np.testing.assert_array_equal(controls,state['controls'])
        motion_calls.append((times.copy(),seconds.copy()))
        report=dict(channels=dict(pan=dict(speed=dict(continuous_peak=0.),acceleration=dict(continuous_peak=0.))))
        return report,dict(motion_samples=times.copy()),object()
    def lens(actual_core,rig,controls,first,dt,times,seconds,camera,width,height):
        assert actual_core is core and (width,height)==(1920,1080)
        lens_calls.append(times.copy())
        return dict(native_forward_preserved=True),dict(lens_samples=times.copy())
    monkeypatch.setattr(audit,'inspect_shot_motion',motion)
    monkeypatch.setattr(audit,'inspect_lens_forward',lens)
    output=tmp_path/'audit.json'
    audit.main(SimpleNamespace(run_root=root,baseline=baseline,candidate=candidate,output=output,
                               native_library=bound if override else None))
    expected_times=np.array([.02,.04,.06]);expected_seconds=np.array([512,768,1024])/12800
    assert len(motion_calls)==len(lens_calls)==2
    for (times,seconds),lens_times in zip(motion_calls,lens_calls):
        np.testing.assert_allclose(times,expected_times,rtol=0,atol=1e-14)
        np.testing.assert_array_equal(seconds,expected_seconds)
        np.testing.assert_array_equal(times,lens_times)
    report=json.loads(output.read_text('utf8'))
    assert report['source_exposures']==3
    assert report['frames_with_observations']==2 and report['frames_without_observations']==1
    assert report['state_mutated'] is report['optimizer_started'] is report['observations_interpolated'] is False
    assert report['prepared_native_library']==str(library)
    assert report['prepared_native_library_sha256']==audit.digest(library)
    assert Path(report['selected_native_library']).resolve()==selected.resolve()
    assert report['native_library_sha256']==audit.digest(selected)
    assert report['native_library_override_explicit']is override
    for name in ['baseline','candidate']:
        coverage=report['states'][name]['shots']['s07']['source_coverage']
        assert coverage['frames_without_observations']==1
        assert report['states'][name]['native_library_provenance_sha256']==audit.digest(selected)
    with np.load(output.with_suffix('.npz'),allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['candidate_s07_pts'],[512,768,1024])
    assert all(audit.digest(path)==sha for path,sha in protected.items())


@pytest.mark.parametrize('defect',['missing_override','wrong_override','missing_provenance','mixed_state_libraries'])
def test_audit_rejects_unbound_libraries_before_native_or_output(tmp_path,monkeypatch,defect):
    root=tmp_path/'prepared';root.mkdir()
    old,bound,wrong=[tmp_path/name for name in ['old.dll','bound.dll','wrong.dll']]
    for path in [old,bound,wrong]:path.write_bytes(path.name.encode())
    (root/'joint_manifest.json').write_text(json.dumps(dict(
        native_library=str(old),native_library_sha256=audit.digest(old))),encoding='utf8')
    baseline,candidate=tmp_path/'baseline.npz',tmp_path/'candidate.npz'
    for path in [baseline,candidate]:
        np.savez(path,native_library_provenance_sha256=np.array(audit.digest(bound)))
    if defect=='missing_provenance':np.savez(baseline,controls=np.zeros((1,3)))
    if defect=='mixed_state_libraries':
        np.savez(baseline,native_library_provenance_sha256=np.array(audit.digest(old)))
    selected=None if defect=='missing_override'else wrong if defect=='wrong_override'else bound
    monkeypatch.setattr(audit,'NativeCore',lambda *a:pytest.fail('Rejected binding must not load native'))
    output=tmp_path/'audit.json'
    with pytest.raises(ValueError,match='native.library|native_library_provenance'):
        audit.main(SimpleNamespace(run_root=root,baseline=baseline,candidate=candidate,output=output,
                                   native_library=selected))
    assert not output.exists()and not output.with_suffix('.npz').exists()
