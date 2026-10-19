import hashlib
import json

import numpy as np
import pytest

from rinkspace.programme_source_measurement_cache import (
    ProgrammeSourceMeasurementCache,recheck_source_candidate_exclusions)


def test_chunk_slicing_preserves_frame_local_correspondences(tmp_path):
    path=tmp_path/'source_pixels_0000.npz'
    np.savez_compressed(path,actual_pts=[100,200],native_pts=[100,200],
        BGR_sha256=['b'*64,'c'*64],RGB_sha256=['d'*64,'e'*64],shot_id=['shot','shot'],
        query_offsets=[0,2,3],query_curve=[0,1,0],query_uv=[[1,1],[2,2],[3,3]],
        query_normal=[[0,1]]*3,query_interval=[[-128,128]]*3,query_status=[1]*3,
        candidate_offsets=[0,2,3],candidate_uv=[[1,1],[2,2],[3,3]],
        candidate_sides=np.ones((3,2,2)),candidate_query=[0,1,0],candidate_flags=[8]*3,
        curve_status=np.ones((2,2),np.uint8))
    manifest=tmp_path/'source_measurements.json'
    reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        first_actual_pts=100,last_actual_pts=200)
    packet=dict(source_sha256='a'*64,time_base='1/15360',source_path='original.mp4',
        inventory=[dict(curve_id='C:segment0')],chunks=[reference])
    manifest.write_text(json.dumps(packet),encoding='utf8')
    cache=ProgrammeSourceMeasurementCache(manifest,source_sha256='a'*64,
        time_base='1/15360',expected_actual_pts=[100,200])
    second=cache.frame(200)
    assert second['candidate_query'].tolist()==[0]
    assert second['query_uv'].tolist()==[[3,3]]
    assert second['source_binding']['decoded_native_BGR_sha256']=='c'*64
    binding=dict(second['source_binding'],decoded_native_BGR_sha256='x'*64)
    with pytest.raises(ValueError,match='exact source differ'):cache.frame(200,source_binding=binding)


def test_old_scope_bit_is_remeasured_and_ambiguity_survives():
    measurement=dict(candidate_uv=np.array([[30.,30.],[60.,60.],[1.,1.]]),
        candidate_sides=np.array([[[28.,30.],[32.,30.]],[[58.,60.],[62.,60.]],[[1.,1.],[3.,1.]]]),
        query_normal=np.array([[1.,0.]]*3),candidate_query=np.array([0,1,2]),
        candidate_flags=np.array([14,8,9],np.uint8))
    mask=np.zeros((100,100),bool);mask[60,60]=True
    result=recheck_source_candidate_exclusions(measurement,mask)
    assert result['candidate_flags'].tolist()==[12,10,11]
    assert result['current_exact_source_scope_rechecked'] is True


def test_producer_cannot_replace_cached_prefix(tmp_path):
    path=tmp_path/'source_measurements.json'
    packet=dict(source_sha256='a'*64,time_base='1/15360',source_path='original.mp4',
        inventory=[],chunks=[dict(first_actual_pts=100,last_actual_pts=100,path='old',sha256='b'*64)])
    path.write_text(json.dumps(packet),encoding='utf8')
    cache=ProgrammeSourceMeasurementCache(path,source_sha256='a'*64,
        time_base='1/15360',expected_actual_pts=[100],follow=True)
    packet['chunks'][0]['sha256']='c'*64
    path.write_text(json.dumps(packet),encoding='utf8')
    with pytest.raises(ValueError,match='changed.*prefix'):cache.refresh()
