"""Read-only still DLL selection with synthetic arrays and mocked native/media."""
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import render_native_review_stills as stills


def case(tmp_path,monkeypatch,new_state):
    root=tmp_path/'prepared';root.mkdir();here=tmp_path/'programme';here.mkdir()
    old,bound,scene,video=[tmp_path/name for name in ['old.dll','bound.dll','scene.json','source.mp4']]
    old.write_bytes(b'Prepared DLL');bound.write_bytes(b'Actual state DLL')
    scene.write_text('{}',encoding='utf8');video.write_bytes(b'No media is decoded by this test')
    points=dict(pts=np.array([512,1024],np.int64),uv=np.zeros((2,2)),path_indices=np.zeros(2,np.int32))
    np.savez(root/'point_observations.npz',**points)
    processing=dict(first_pts=512,last_pts=1024,stop_pts_exclusive=1280,native_frame_count=3)
    shot=dict(id='s09',row_begin=0,row_stop=2,camera_index=0,reference_pts=512,
        time_base_num=1,time_base_den=12800,control_begin=0,control_stop=8,first_control_time=-.02,**processing)
    manifest=dict(shots=[shot],control_dt=.02,first_control_time=0.,max_readout=.02,
        native_library=str(old),native_library_sha256=stills.prepared.digest(old),scene_path=str(scene),
        scene_file_sha256=stills.prepared.digest(scene),point_input_sha256=stills.prepared.digest(root/'point_observations.npz'))
    scope=dict(required_shot_ids=['s09'],shots=[dict(id='s09',processing=processing)],
        source=dict(registered_path=str(video),sha256=stills.prepared.digest(video),width=1920,height=1080),
        source_clock=dict(time_base_num=1,time_base_den=12800,native_pts_step=256,native_fps_num=50,native_fps_den=1))
    for path,value in [(root/'joint_manifest.json',manifest),(root/'mapping.json',dict(path_ids=['pathA'])),
                       (here/'rebuild_scope.json',scope)]:stills.prepared.write(path,value)
    selected=bound if new_state else old
    state=dict(rig=np.zeros((1,43)),controls=np.zeros((8,3)),scene_values=np.zeros(0),
        native_library_provenance_sha256=np.array(stills.prepared.digest(selected)))
    state_path=tmp_path/'state.npz';np.savez(state_path,**state)
    monkeypatch.setattr(stills,'HERE',here);monkeypatch.setattr(stills,'configured_data_root',lambda p:tmp_path)
    return root,state_path,old,bound,manifest,state


@pytest.mark.parametrize('override',[False,True])
def test_still_uses_explicit_bound_dll_and_records_both_identities(tmp_path,monkeypatch,override):
    root,state_path,old,bound,manifest,state=case(tmp_path,monkeypatch,override)
    selected=bound if override else old
    protected={p:stills.prepared.digest(p)for p in [root/'joint_manifest.json',state_path,old,bound]}
    def core(path):
        assert Path(path)==selected.resolve()
        return object()
    monkeypatch.setattr(stills,'NativeCore',core)
    model=SimpleNamespace(initial_state=SimpleNamespace(layout='mock layout'))
    monkeypatch.setattr(stills,'SceneParameterModel',lambda *a,**kw:model)
    checked=[];monkeypatch.setattr(stills.prepared,'validate_snapshot',lambda *a:checked.append(a[3]))
    monkeypatch.setattr(stills,'ParameterState',lambda *a:object())
    monkeypatch.setattr(stills,'scene_content_identity',lambda *a:'unchanged scene content')
    native=SimpleNamespace(scene=dict(board=np.zeros(6),shared_layout=np.zeros(0),shared_geometry_options=0))
    monkeypatch.setattr(stills.prepared,'NativeSceneGPU',lambda *a,**kw:native)
    monkeypatch.setattr(stills,'sample_native_review_scene',lambda *a,**kw:dict(
        native_paths_resolved=True,xyz=np.empty((0,3)),cuts=[]))
    resident=SimpleNamespace(configure_geometry=lambda *a:None,configure_projection_time_support=lambda **kw:None,
        evaluate=lambda *a:dict(values=np.empty((0,5)),status=np.empty(0,dtype=int)))
    monkeypatch.setattr(stills,'ResidentResidualProblem',lambda *a,**kw:contextlib.nullcontext(resident))
    monkeypatch.setattr(stills,'source_frames',lambda *a:[(512,np.zeros((1080,1920,3),np.uint8))])
    def image_sink(path,canvas):
        Path(path).write_bytes(b'Mocked image sink; no source image or native render')
        return True
    monkeypatch.setattr(stills.cv2,'imwrite',image_sink)
    receipt=stills.render(root,state_path,tmp_path/'review',{'s09':[512]},native_library=bound if override else None)
    assert checked[0]['native_library_sha256']==stills.prepared.digest(selected)
    assert receipt['prepared_native_library']==str(old)
    assert receipt['prepared_native_library_sha256']==stills.prepared.digest(old)
    assert receipt['selected_native_library']==str(selected.resolve())
    assert receipt['native_library']['sha256']==stills.prepared.digest(selected)
    assert receipt['native_library_override_explicit']is override
    assert receipt['state_native_library_provenance_sha256']==str(state['native_library_provenance_sha256'])
    assert receipt['prepared_manifest']['sha256']==protected[root/'joint_manifest.json']
    assert all(stills.prepared.digest(p)==sha for p,sha in protected.items())


@pytest.mark.parametrize('defect',['missing_override','wrong_override','missing_provenance'])
def test_still_rejects_unbound_dll_before_native_or_output(tmp_path,monkeypatch,defect):
    root,state_path,old,bound,manifest,state=case(tmp_path,monkeypatch,True)
    if defect=='missing_provenance':
        state.pop('native_library_provenance_sha256');np.savez(state_path,**state)
    selected=None if defect=='missing_override'else old if defect=='wrong_override'else bound
    monkeypatch.setattr(stills,'NativeCore',lambda *a:pytest.fail('Rejected binding must not load native'))
    out=tmp_path/'review'
    with pytest.raises(ValueError,match='native.library|native_library_provenance'):
        stills.render(root,state_path,out,{'s09':[512]},native_library=selected)
    assert not out.exists()


def test_cli_passes_only_explicit_override(monkeypatch):
    calls=[];monkeypatch.setattr(stills,'render',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(sys,'argv',['render_native_review_stills.py','--run-root','prepared','--state','state.npz',
        '--output','review','--shot-pts','s09:512','--native-library','bound.dll'])
    stills.main()
    assert calls==[((Path('prepared'),Path('state.npz'),Path('review'),{'s09':[512]}),
                    dict(native_library=Path('bound.dll')))]
