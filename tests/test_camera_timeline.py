from fractions import Fraction
import numpy as np
from rinkspace.camera_timeline import frame_partition, detect_boundaries, change_signals


def test_partition_preserves_half_open_pts_and_each_frame_once():
    pts=np.arange(256,256+20*512,512,dtype=np.int64)
    boundaries=[{'frame_index':3},{'frame_index':4},{'frame_index':19}]
    rows=frame_partition(pts,boundaries,Fraction(1,15360))
    assert sum(r['frame_count'] for r in rows)==len(pts)
    assert [r['frame_count'] for r in rows]==[3,1,15,1]
    assert rows[0]['start_pts']==256
    assert rows[-1]['end_pts_exclusive']==pts[-1]+512
    assert all(a['end_pts_exclusive']==b['start_pts'] for a,b in zip(rows,rows[1:]))
    allocated=[i for r in rows for i in range(r['start_frame_index'],r['end_frame_index_exclusive'])]
    assert allocated==list(range(len(pts)))


def test_faint_old_luma_cut_can_be_recovered_by_spatial_colour(tmp_path):
    pts=np.arange(40,dtype=np.int64)*512
    grid=np.full((40,6,8,3),70,dtype=np.uint8)
    grid[20:]=170
    hist=np.zeros((40,48),dtype=np.uint8)
    hist[:20,4::16]=255
    hist[20:,11::16]=255
    old=np.full(40,.006,dtype=np.float32);old[20]=.082
    boundaries,rows,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert len(boundaries)==1
    assert boundaries[0]['frame_index']==20
    assert boundaries[0]['status']=='machine_supported_cut'
    assert boundaries[0]['legacy_candidate'] is None


def test_continuous_camera_motion_does_not_become_repeated_hard_cuts(tmp_path):
    pts=np.arange(50,dtype=np.int64)*512
    grid=np.broadcast_to(np.arange(50,dtype=np.uint8)[:,None,None,None]+60,(50,6,8,3)).copy()
    hist=np.zeros((50,48),dtype=np.uint8)
    old=np.full(50,.04,dtype=np.float32)
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert boundaries==[]


def test_chunk_first_zero_luma_does_not_hide_colour_discontinuity(tmp_path):
    pts=np.arange(40,dtype=np.int64)*512
    grid=np.full((40,6,8,3),70,dtype=np.uint8);grid[20:]=170
    hist=np.zeros((40,48),dtype=np.uint8);hist[:20,3::16]=255;hist[20:,10::16]=255
    old=np.full(40,.006,dtype=np.float32)
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert any(r['frame_index']==20 for r in boundaries)
    # Such a boundary stays unresolved without the original luma spike; no invented certainty.
    assert boundaries[0]['status']=='unresolved_transition'


def test_feature_difference_does_not_wrap_uint8():
    grid=np.zeros((2,6,8,3),dtype=np.uint8);grid[1]=255
    hist=np.zeros((2,48),dtype=np.uint8)
    bands,colour=change_signals(grid,hist)
    assert np.allclose(bands[1],1)


def test_one_frame_insert_is_visible_as_unresolved_short_event(tmp_path):
    pts=np.arange(45,dtype=np.int64)*512
    grid=np.full((45,6,8,3),60,dtype=np.uint8);grid[20]=210
    hist=np.zeros((45,48),dtype=np.uint8)
    old=np.full(45,.005,dtype=np.float32);old[20:22]=.5
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert [r['frame_index'] for r in boundaries]==[20,21]
    assert all(r['status']=='unresolved_short_event' for r in boundaries)
    segments=frame_partition(pts,boundaries,Fraction(1,15360))
    unresolved=[s for s in segments if s['interval_type']=='boundary_unresolved']
    assert len(unresolved)==1
    assert [r['frame_index'] for r in unresolved[0]['candidate_points']]==[20,21]
    assert sum(r['frame_count'] for r in segments)==45
    assert not any(r['interval_type']=='shot' and r['frame_count']==1 for r in segments)


def test_slow_dissolve_has_a_range_without_invented_exact_cut(tmp_path):
    pts=np.arange(90,dtype=np.int64)*512
    values=np.concatenate([np.full(25,60),np.linspace(60,160,16),np.full(49,160)]).astype(np.uint8)
    grid=np.broadcast_to(values[:,None,None,None],(90,6,8,3)).copy()
    hist=np.zeros((90,48),dtype=np.uint8)
    old=np.full(90,.004,dtype=np.float32);old[25:42]=.025
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert any(r['status']=='unresolved_gradual_transition' for r in boundaries)
    row=next(r for r in boundaries if r['status']=='unresolved_gradual_transition')
    assert row['provisional_split_not_exact_cut']
    assert row['uncertainty_interval_seconds'][0]<row['source_seconds']<row['uncertainty_interval_seconds'][1]


def test_overlapping_uncertainty_ranges_union_and_preserve_points():
    pts=np.arange(60,dtype=np.int64)*512
    tb=Fraction(1,15360)
    boundaries=[]
    for i,left,right in [(20,18,23),(24,22,27),(40,38,42)]:
        boundaries.append({'boundary_id':f'b{i}','frame_index':i,'source_pts':int(pts[i]),
                           'source_seconds':float(pts[i]*tb),'status':'unresolved_short_event',
                           'uncertainty_interval_seconds':[float(pts[left]*tb),float(pts[right]*tb)]})
    rows=frame_partition(pts,boundaries,tb)
    events=[r for r in rows if r['interval_type']=='boundary_unresolved']
    assert len(events)==2
    assert [r['frame_index'] for r in events[0]['candidate_points']]==[20,24]
    assert events[0]['start_frame_index']==18
    assert events[0]['end_frame_index_exclusive']==28
    assert sum(r['frame_count'] for r in rows)==60
    assert all(a['end_frame_index_exclusive']==b['start_frame_index'] for a,b in zip(rows,rows[1:]))


def test_low_contrast_ice_dissolve_is_retained_as_uncertain_range(tmp_path):
    pts=np.arange(100,dtype=np.int64)*512
    values=np.concatenate([np.full(40,100),np.linspace(100,113,7),np.full(53,113)]).astype(np.uint8)
    grid=np.broadcast_to(values[:,None,None,None],(100,6,8,3)).copy()
    hist=np.zeros((100,48),dtype=np.uint8)
    old=np.full(100,.01,dtype=np.float32);old[40:47]=.035
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    events=[r for r in frame_partition(pts,boundaries,Fraction(1,15360)) if r['interval_type']!='shot']
    assert any(r['start_frame_index']<=43<r['end_frame_index_exclusive'] for r in events)


def test_temporal_windows_normalize_speed_and_do_not_flag_steady_pan(tmp_path):
    pts=np.arange(90,dtype=np.int64)*512
    grid=np.full((90,6,8,3),100,dtype=np.uint8)
    grid[:,:,:,0]=np.arange(90,dtype=np.uint8)[:,None,None]*2+40
    hist=np.zeros((90,48),dtype=np.uint8)
    old=np.full(90,.03,dtype=np.float32)
    boundaries,_,_=detect_boundaries(pts,old,grid,hist,Fraction(1,15360),[],'test',tmp_path)
    assert not boundaries
