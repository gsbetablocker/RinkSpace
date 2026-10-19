from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace import camera_timeline as timeline


def make_scope(pts, target, ignored=(), pending=()):
    step=int(pts[1]-pts[0])
    def intervals(ranges):
        return [{'start_pts':int(pts[a]),
                 'end_pts_exclusive':int(pts[b]) if b<len(pts) else int(pts[-1]+step)}
                for a,b in ranges]
    return {'source_sha256':'test-sha','time_base':'1/15360',
            'target_intervals':intervals(target),'ignored_intervals':intervals(ignored),
            'pending_intervals':intervals(pending)}


def test_scope_rejects_wrong_track_and_overlapping_decisions():
    pts=np.arange(10,dtype=np.int64)*512
    track={'source_sha256':'test-sha','time_base':'1/15360'}
    scope=make_scope(pts,[(2,5)],[(0,2)])
    runs,masks=timeline.scope_frame_runs(pts,scope,track)
    assert runs==[(2,5)]
    assert masks['pending'].sum()==5
    with pytest.raises(ValueError,match='source track'):
        timeline.scope_frame_runs(pts,scope,{**track,'source_sha256':'other'})
    with pytest.raises(ValueError,match='overlap'):
        timeline.scope_frame_runs(pts,make_scope(pts,[(2,5)],[(4,6)]),track)


def test_independent_runs_keep_source_indices_without_cross_gap_cut(tmp_path,monkeypatch):
    pts=np.arange(70,dtype=np.int64)*512
    grid=np.full((70,6,8,3),60,dtype=np.uint8)
    grid[35:]=190
    features={'spatial_yuv':grid,'histogram_yuv':np.zeros((70,48),dtype=np.uint8)}
    scores=np.zeros(70,dtype=np.float32);scores[35]=.9
    calls=[]
    actual=timeline.detect_boundaries
    def capture(run_pts,run_scores,run_grid,*args):
        calls.append((run_pts.copy(),run_scores.copy()))
        return actual(run_pts,run_scores,run_grid,*args)
    monkeypatch.setattr(timeline,'detect_boundaries',capture)
    segments,boundaries,_,_=timeline.scoped_partition(
        {'pts':pts,'legacy_cuts':[]},features,scores,Fraction(1,15360),
        [(5,25),(35,55),(65,66)],'test',tmp_path)
    assert [list(p) for p,_ in calls]==[list(pts[5:25]),list(pts[35:55]),list(pts[65:66])]
    assert all(s[0]==0 for _,s in calls)
    assert boundaries==[]
    assert [(s['start_frame_index'],s['end_frame_index_exclusive']) for s in segments]==[(5,25),(35,55),(65,66)]
    assert [s['start_pts'] for s in segments]==[int(pts[i]) for i in [5,35,65]]
    assert segments[-1]['end_pts_exclusive']==int(pts[66])


def test_reviews_outside_run_are_not_injected(tmp_path):
    review_path=tmp_path/'outputs/camera_classification/timelines/development_reviews/boundary_reviews.json'
    timeline.dump(review_path,{'reviews':[
        {'source_id':'test','source_interval_seconds':[0,.1],'source_interval_pts':[0,512],
         'kind':'hard_cut','reason':'outside target','evidence_paths':[]},
        {'source_id':'test','source_interval_seconds':[90,91],'source_interval_pts':[1382400,1397760],
         'kind':'dissolve','reason':'outside target','evidence_paths':[]}]})
    pts=np.arange(60,100,dtype=np.int64)*512
    boundaries,rows,_=timeline.detect_boundaries(pts,np.zeros(40),np.full((40,6,8,3),80,dtype=np.uint8),
        np.zeros((40,48),dtype=np.uint8),Fraction(1,15360),[],'test',tmp_path)
    assert boundaries==rows==[]


def test_uncertain_candidate_indices_are_restored_to_full_source(tmp_path):
    pts=np.arange(90,dtype=np.int64)*512
    grid=np.full((90,6,8,3),60,dtype=np.uint8);grid[60]=210
    scores=np.full(90,.005,dtype=np.float32);scores[60:62]=.5
    segments,boundaries,_,_=timeline.scoped_partition({'pts':pts,'legacy_cuts':[]},
        {'spatial_yuv':grid,'histogram_yuv':np.zeros((90,48),dtype=np.uint8)},
        scores,Fraction(1,15360),[(40,85)],'test',tmp_path)
    assert [b['frame_index'] for b in boundaries]==[60,61]
    event=next(s for s in segments if s['interval_type']=='boundary_unresolved')
    assert [p['frame_index'] for p in event['candidate_points']]==[60,61]
    assert event['start_pts']==int(pts[event['start_frame_index']])


def test_build_requires_scope_before_loading_media(tmp_path,monkeypatch):
    monkeypatch.setattr(timeline,'load_inputs',lambda *_:pytest.fail('scope must precede media loading'))
    with pytest.raises(FileNotFoundError,match='scope'):
        timeline.build_source(tmp_path,'test',tmp_path/'outputs/camera_classification/rink_timelines')


def test_build_source_passes_only_target_runs_to_feature_stage(tmp_path,monkeypatch):
    pts=np.arange(10,dtype=np.int64)*512
    scope=make_scope(pts,[(2,4),(6,8)],[(0,2),(4,6)],[(8,10)])
    output=tmp_path/'outputs/camera_classification'
    timeline.dump(output/'rink_scope.json',{'sources':{'test':scope}})
    inputs={'pts':pts,'track':{'source_sha256':'test-sha','time_base':'1/15360','source_path':'source.mp4'},
        'summary':{'duration_seconds':1/3,'title':'test'},'legacy_cuts':[],'legacy_score':np.zeros(10),
        'chunks':[],'candidates':[],'candidate_pts':np.array([],dtype=np.int64),
        'second_features':[],'feature_pts':np.array([],dtype=np.int64)}
    monkeypatch.setattr(timeline,'load_inputs',lambda *_:inputs)
    calls=[]
    def cached(root,sid,loaded,out,runs):
        calls.append(runs)
        return {'spatial_yuv':np.zeros((10,6,8,3),dtype=np.uint8),
                'histogram_yuv':np.zeros((10,48),dtype=np.uint8)}, {'selected_feature_pts_match_original':True}
    monkeypatch.setattr(timeline,'feature_cache',cached)
    monkeypatch.setattr(timeline,'seam_luma_cache',lambda *_:(np.zeros(10),{'seams':[]}))
    def extract(root,track,requested,out):
        for paths in requested.values():
            for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'evidence')
    monkeypatch.setattr(timeline,'extract_images',extract)
    result=timeline.build_source(tmp_path,'test',output/'rink_timelines')
    assert calls==[[(2,4),(6,8)]]
    assert result['retained_frame_count']==4 and result['ignored_frame_count']==4
    record=timeline.read_json(output/'rink_timelines/test.json')
    assert record['timeline_validation']['no_ignored_or_pending_frame_segmented']
    assert [(s['start_frame_index'],s['end_frame_index_exclusive']) for s in record['segments']]==[(2,4),(6,8)]
    assert all(2<=o['frame_index']<4 or 6<=o['frame_index']<8
               for s in record['segments'] for o in s['observations'])


def test_missing_features_compute_only_retained_frames(tmp_path,monkeypatch):
    pts=np.arange(10,dtype=np.int64)*512
    stream=SimpleNamespace(time_base=Fraction(1,15360),width=10,height=10,average_rate=30,
        codec_context=SimpleNamespace(name='mock'),index=0,duration=5120,start_time=0)
    class Container:
        streams=SimpleNamespace(video=[stream])
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def seek(self,p,**_):self.start=max(0,p//512-1)
        def decode(self,_):return iter(SimpleNamespace(pts=int(p)) for p in pts[self.start:])
    calls=[]
    monkeypatch.setattr(timeline.av,'open',lambda _:Container())
    def feature(frame):
        calls.append(frame.pts)
        return np.zeros((6,8,3),dtype=np.uint8),np.zeros(48,dtype=np.uint8)
    monkeypatch.setattr(timeline,'spatial_colour_feature',feature)
    out=tmp_path/'outputs/camera_classification/rink_timelines';out.mkdir(parents=True)
    inputs={'pts':pts,'track':{'source_sha256':'test-sha','time_base':'1/15360','source_path':'source.mp4'}}
    features,meta=timeline.feature_cache(tmp_path,'test',inputs,out,[(1,3),(6,9)])
    assert calls==list(pts[[1,2,6,7,8]])
    assert meta['new_feature_frame_count']==5
    assert np.flatnonzero(features['valid_frame_mask']).tolist()==[1,2,6,7,8]


def test_historical_feature_cache_is_reused_without_writes(tmp_path,monkeypatch):
    pts=np.arange(10,dtype=np.int64)*512
    historical=tmp_path/'outputs/camera_classification/timelines';historical.mkdir(parents=True)
    archive=historical/'test_features.npz'
    np.savez_compressed(archive,source_pts=pts,spatial_yuv=np.zeros((10,6,8,3),dtype=np.uint8),
                        histogram_yuv=np.zeros((10,48),dtype=np.uint8))
    timeline.dump(historical/'test_features.json',{'feature_version':timeline.FEATURE_VERSION,
        'source_sha256':'test-sha','all_pts_equal_legacy_index':True})
    previous=archive.read_bytes()
    monkeypatch.setattr(timeline.av,'open',lambda _:pytest.fail('historical cache must avoid decoding'))
    out=tmp_path/'outputs/camera_classification/rink_timelines';out.mkdir()
    _,meta=timeline.feature_cache(tmp_path,'test',{'pts':pts,'track':{'source_sha256':'test-sha'}},out,[(1,3)])
    assert meta['cache_reused_read_only'] and meta['selected_frame_count']==2
    assert archive.read_bytes()==previous
    assert list(out.iterdir())==[]


def test_scope_audit_accepts_expected_gaps_and_rejects_segmenting_them(tmp_path):
    pts=np.arange(10,dtype=np.int64)*512;tb=Fraction(1,15360)
    out=tmp_path/'rink_timelines';out.mkdir()
    archive=out/'test_features.npz';np.savez(archive,source_pts=pts)
    scope=make_scope(pts,[(1,4),(6,8)],[(0,1),(4,6)],[(8,10)])
    def segment(a,b):
        return {'shot_id':f'test:{a}','interval_type':'shot','start_frame_index':a,
            'end_frame_index_exclusive':b,'start_pts':int(pts[a]),'end_pts_exclusive':int(pts[b]),
            'start_seconds':float(pts[a]*tb),'end_seconds':float(pts[b]*tb),'frame_count':b-a,'observations':[]}
    record={'source_id':'test','track':{'time_base':str(tb),'source_sha256':'test-sha'},
        'duration_seconds':float(5120*tb),'rink_scope':scope,'segments':[segment(1,4),segment(6,8)],
        'boundaries':[],'shot_count':2,'interval_type_counts':{'shot':2},'chunk_audit':[],
        'chunk_seam_features':{'seams':[]},'feature_cache':{'feature_archive':'rink_timelines/test_features.npz',
        'selected_feature_pts_match_original':True}}
    timeline.dump(out/'test.json',record)
    audit=timeline.audit_timelines(tmp_path,out,['test'])
    assert audit['passed'] and audit['total_retained_frames']==5
    assert audit['total_ignored_frames']==3 and audit['total_pending_frames']==2
    record['segments']=[segment(1,8)]
    timeline.dump(out/'test.json',record)
    assert not timeline.audit_timelines(tmp_path,out,['test'])['passed']
