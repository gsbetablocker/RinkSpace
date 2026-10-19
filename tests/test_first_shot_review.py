"""Same-state review assembly, exact native clocks and lossless small exports."""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from calibration.program_tracking import first_shot_review as review


def test_missing_ink_bridge_preserves_three_point_anchors_and_existing_scene(monkeypatch):
    original={'source_conditioned_board_prints':{'patterns':[{'advertisement_id':8,'external_contours':[]}]}}
    before=deepcopy(original)
    contours=[{'kind':'ad_print','advertisement_id':12,'contour_key':'a'},
              {'kind':'ad_print','advertisement_id':12,'contour_key':'b'}]
    paths=(np.arange(9,dtype=float).reshape(3,3),np.arange(12,dtype=float).reshape(4,3))
    calls=[]
    def worlds(q, targets):
        calls.append((q.copy(),targets));return paths
    state=SimpleNamespace(review_scene=lambda q:original,world_paths=worlds)
    bundle=SimpleNamespace(state=state,target_descriptors=contours)
    anchors={'number':12,'primitive':'points','pattern_kind':'shield','xyz':np.zeros((3,3))}
    monkeypatch.setattr(review,'advertisement_review_paths',lambda scene:[anchors])
    q=np.array([.1,.2])
    scene=review.complete_review_scene(bundle,q)
    assert original==before
    assert len(scene['source_conditioned_board_prints']['patterns'])==2
    added=scene['source_conditioned_board_prints']['patterns'][1]
    assert added['advertisement_id']==12
    assert [p['id'] for p in added['external_contours']]==['a','b']
    for curve,path in zip(added['external_contours'],paths,strict=True):
        np.testing.assert_array_equal(curve['xyz_scene_units'],path)
    assert anchors['primitive']=='points' and len(anchors['xyz'])==3
    np.testing.assert_array_equal(calls[0][0],q)


def short_view():
    obj=review.FirstShotReview.__new__(review.FirstShotReview)
    obj.q=np.arange(60,dtype=np.float64)/100
    obj.q_sha256=hashlib.sha256(obj.q.tobytes()).hexdigest()
    state=SimpleNamespace(count=60,view_stop=12,
        global_shared_indices=np.r_[0:4,12:60],world_parameter_indices=np.r_[0:3,12:60],
        view_columns=lambda index:np.arange(4*index,4*index+4),
        camera25=lambda q,index:np.arange(25,dtype=float)+q[4*index],
        binding_reference={'path':'source.json','sha256':'source'},
        source_clock={'time_base':'1/12800','fps':'50/1'},
        print_layout_advertisement_ids=(2,3,4,5,6,7),scene_unit_working_m=.3135)
    obj.bundle=SimpleNamespace(state=state,frames=[{'index':1,'native_pts':256256},{'index':2,'native_pts':256512}])
    obj.scene={'same_q_scene':True}
    obj.checkpoint_receipt={'archive':{'path':'full.npz','sha256':'archive'}}
    obj.video_reference={'path':'native.mp4','sha256':'video'}
    obj.ad12_proof={'actual_curves':2,'independent_anchor_points':3,'anchors_connected_as_triangle':False}
    return obj


def test_small_RF4_export_recovers_every_physical_and_layout_coordinate():
    obj=short_view();record=obj.parameters();q=np.empty(60)
    q[record['global_shared_columns']]=record['global_shared_values']
    for exposure in record['exposures']:q[exposure['RF4_columns']]=exposure['RF4']
    assert q.tobytes()==obj.q.tobytes()
    assert record['every_coordinate_reconstructed_byte_exact']
    assert record['print_layout_advertisement_ids']==[2,3,4,5,6,7]
    assert record['accuracy_accepted'] is False


def test_parameter_export_refuses_missing_RF4_columns():
    obj=short_view()
    obj.bundle.frames=obj.bundle.frames[:1]
    # Omitted coordinates must not silently become an exported complete state.
    with pytest.raises(ValueError,match='omitted'):
        obj.parameters()


def test_model_exports_are_authenticated_and_preserve_existing_files(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'ROOT',tmp_path)
    obj=short_view();folder=tmp_path/'models'/'candidate'
    entry=obj.export_model(folder)
    assert entry==review.reference(folder/'model_entry.json')
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(FileExistsError):obj.export_model(folder)
    assert {p.name:p.read_bytes() for p in folder.iterdir()}==before


@pytest.mark.parametrize('mutation',['fps','time_base','pts','bytes'])
def test_original_native_clock_and_source_identity_are_strict(tmp_path,mutation):
    source_path=tmp_path/'source.mp4';source_path.write_bytes(b'authenticated native source')
    source={'native_fps':'50/1','time_base':'1/12800',**review.reference(source_path)}
    pts=np.arange(256256,575489,256,dtype=np.int64)
    if mutation=='fps':source['native_fps']='60/1'
    elif mutation=='time_base':source['time_base']='1/15360'
    elif mutation=='pts':pts[2]+=1
    else:source_path.write_bytes(b'wrong source')
    with pytest.raises(ValueError):review.verify_source_clock(source,pts,verify_video=False)


def test_all_1248_native_presentations_are_accepted_only_with_exact_clock(tmp_path):
    source_path=tmp_path/'source.mp4';source_path.write_bytes(b'authenticated native source')
    source={'native_fps':'50/1','time_base':'1/12800',**review.reference(source_path)}
    assert review.verify_source_clock(source,np.arange(256256,575489,256,dtype=np.int64),verify_video=False)==review.reference(source_path)


@pytest.mark.parametrize('rate,pts,count',[(Fraction(60),[0,256,512],3),(Fraction(50),[0,256,513],3),(Fraction(50),[0,256],3)])
def test_encoded_review_rejects_cadence_or_coverage_change(monkeypatch,rate,pts,count):
    stream=SimpleNamespace(width=2580,height=1080,average_rate=rate,time_base=Fraction(1,12800))
    class Video:
        streams=SimpleNamespace(video=[stream])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,_):return iter([SimpleNamespace(pts=p) for p in pts])
    monkeypatch.setattr(review.av,'open',lambda _:Video())
    with pytest.raises(ValueError):review.verify_video('unused.mp4',count)


def test_review_canvas_keeps_full_native_frame_size():
    assert review.SIZE==(1920+660,1080)


def test_revision_review_retains_quarantined_source_pixels():
    frame={'rows':[{'uv':np.array([[4.,5.],[7.,8.]])}],
           'inactive_source_correspondences':[{'UV':[11.,12.]}]}
    np.testing.assert_array_equal(review.review_source_uv(frame),[[4.,5.],[7.,8.],[11.,12.]])
