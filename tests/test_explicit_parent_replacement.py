"""Selective parent IDs preserve both sides of one continuous board entity."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
import prepare_native_source_update as update


@pytest.fixture
def parent_packet(tmp_path):
    path='board.continuous/top'
    old=dict(pts=np.array([1000,1000,1256,1256,2000],np.int64),
        observation_ids=np.array(['near-0','far-0','near-1','far-1','other-shot'],dtype='U96'),
        uv=np.array([[50.,700.],[50.,100.],[60.,710.],[60.,110.],[100.,100.]]),
        kind=np.ones(5,np.int64),stations=np.full(5,.5),source_rows=np.arange(5,dtype=np.int64),
        path_indices=np.zeros(5,np.int32))
    parent=tmp_path/'parent.npz';np.savez(parent,**old)
    manifest=dict(point_input_sha256=update.prepared.digest(parent),source_video_sha256='a'*64,
        shots=[dict(id='s07',row_begin=0,row_stop=4,first_pts=1000,last_pts=1256,stop_pts_exclusive=1512),
               dict(id='s08',row_begin=4,row_stop=5,first_pts=2000,last_pts=2000,stop_pts_exclusive=2256)])
    mapping=dict(path_ids=[path],source_video_sha256='a'*64)
    state=dict(stations=np.array([.13,.93,.23,.83,.77]))
    incoming={key:value[[0,2]].copy()for key,value in old.items()}
    incoming.update(observation_ids=np.array(['new-near-0','new-near-1'],dtype='U96'),
        uv=np.array([[55.,705.],[65.,715.]]),source_rows=np.array([900,901],np.int64))
    folder=tmp_path/'packet';folder.mkdir()
    np.savez(folder/'point_observations.npz',**incoming)
    capture=folder/'source_rows.jsonl';raw=b'';locations=[]
    for i,pts in enumerate(incoming['pts']):
        row=dict(path_id=path,uv=incoming['uv'][i].tolist(),kind=1,fit_eligible=True,
                 measurement_operator='actual_RGB_boundary')
        line=(json.dumps(dict(pts=int(pts),source_BGR_sha256='b'*64,fit_eligible=True,rows=[row]))+'\n').encode()
        raw+=line
        locations.append(dict(packet_row=i,observation_id=str(incoming['observation_ids'][i]),
            source_row=int(incoming['source_rows'][i]),pts=int(pts),path_id=path,uv=row['uv'],
            source_BGR_sha256='b'*64,source_jsonl=str(capture),source_line=i+1,
            source_line_sha256=hashlib.sha256(line).hexdigest(),row_index=0))
    capture.write_bytes(raw)
    for location in locations:location['source_jsonl_sha256']=update.prepared.digest(capture)
    row_manifest=folder/'source_row_manifest.jsonl'
    row_manifest.write_text(''.join(json.dumps(row)+'\n'for row in locations),encoding='utf8')
    declaration=dict(policy='explicit_parent_observation_ids',parent_point_input_sha256=manifest['point_input_sha256'],
        groups=[dict(pts=1000,path_id=path,parent_observation_ids=['near-0']),
                dict(pts=1256,path_id=path,parent_observation_ids=['near-1'])])
    packet_mapping=dict(path_ids=[path],source_video_sha256='a'*64,independent_source_capture=True,
        fit_eligible=True,pixel_domain='native_distorted',image_size=[1920,1080],
        point_input_sha256=update.prepared.digest(folder/'point_observations.npz'),
        source_capture_jsonl=str(capture),source_capture_sha256=update.prepared.digest(capture),
        source_row_manifest=row_manifest.name,source_row_manifest_sha256=update.prepared.digest(row_manifest),
        source_replacement=declaration)
    update.prepared.write(folder/'mapping.json',packet_mapping)
    return folder,old,state,manifest,mapping,packet_mapping,parent


def authenticate(case):
    folder,old,_,manifest,mapping,packet_mapping,_=case
    update.prepared.write(folder/'mapping.json',packet_mapping)
    return update.authenticate_packet(folder,manifest,mapping,old)


def assert_parent_exact(merged,old,state,parents):
    for parent in parents:
        rows=np.flatnonzero(merged['parent_rows']==parent)
        assert len(rows)==1
        row=rows[0]
        for key in old:
            assert merged['points'][key][row].tobytes()==old[key][parent].tobytes()
        assert merged['numeric'][row].tobytes()==state['stations'][parent].tobytes()


def test_explicit_near_side_replacement_keeps_far_side_of_same_top_group_bit_exact(parent_packet):
    _,old,state,manifest,mapping,packet_mapping,_=parent_packet
    packet=authenticate(parent_packet)
    assert packet['replacement_spans']=={}
    assert packet['replacement_parent_observation_ids']=={
        (1000,'board.continuous/top'):['near-0'],(1256,'board.continuous/top'):['near-1']}
    merged=update.replace_source_groups(old,state,manifest,mapping,[packet])
    np.testing.assert_array_equal(merged['replaced'],[0,2])
    assert_parent_exact(merged,old,state,[1,3,4])
    for change,group in zip(merged['changes'],packet_mapping['source_replacement']['groups']):
        assert change['parent_observation_ids']==group['parent_observation_ids']
        assert change['parent_point_input_sha256']==manifest['point_input_sha256']
        assert change['action']=='replace_explicit_parent_observation_ids'


def test_empty_explicit_ids_append_without_deleting_existing_same_group_points(parent_packet):
    _,old,state,manifest,mapping,packet_mapping,_=parent_packet
    for group in packet_mapping['source_replacement']['groups']:group['parent_observation_ids']=[]
    packet=authenticate(parent_packet)
    merged=update.replace_source_groups(old,state,manifest,mapping,[packet])
    assert not len(merged['replaced']) and len(merged['points']['pts'])==7
    assert_parent_exact(merged,old,state,range(5))
    assert all(change['action']=='append_explicit_parent_observation_ids'and change['parent_observation_ids']==[]
               for change in merged['changes'])


@pytest.mark.parametrize('invalid,message',[
    ('bad_sha','parent point input SHA256'),('unknown_id','exactly one current parent'),
    ('foreign_shot','another shot'),('foreign_pts','PTS/path'),('foreign_path','PTS/path'),
    ('foreign_kind','kind differs'),('missing_group','cover every incoming'),
    ('duplicate_id','Duplicate explicit'),('duplicate_group','uniquely identify'),
    ('ambiguous_parent','exactly one current parent'),
])
def test_explicit_policy_rejects_foreign_or_incomplete_authority(parent_packet,invalid,message):
    _,old,_,manifest,mapping,packet_mapping,parent=parent_packet
    declaration=packet_mapping['source_replacement'];groups=declaration['groups']
    if invalid=='bad_sha':declaration['parent_point_input_sha256']='f'*64
    elif invalid=='unknown_id':groups[0]['parent_observation_ids']=['missing']
    elif invalid=='foreign_shot':groups[0]['parent_observation_ids']=['other-shot']
    elif invalid=='foreign_pts':groups[0]['parent_observation_ids']=['near-1'];groups[1]['parent_observation_ids']=[]
    elif invalid=='missing_group':groups.pop()
    elif invalid=='duplicate_id':groups[0]['parent_observation_ids']*=2
    elif invalid=='duplicate_group':groups.append(copy.deepcopy(groups[0]))
    else:
        if invalid=='foreign_path':mapping['path_ids'].append('board.other');old['path_indices'][0]=1
        elif invalid=='foreign_kind':old['kind'][0]=0
        else:old['observation_ids'][1]='near-0'
        np.savez(parent,**old)
        manifest['point_input_sha256']=update.prepared.digest(parent)
        declaration['parent_point_input_sha256']=manifest['point_input_sha256']
    with pytest.raises(ValueError,match=message):authenticate(parent_packet)


def test_default_policy_keeps_existing_complete_group_replacement_semantics(parent_packet):
    _,old,state,manifest,mapping,packet_mapping,_=parent_packet
    packet_mapping.pop('source_replacement')
    packet=authenticate(parent_packet)
    assert 'replacement_parent_observation_ids'not in packet
    merged=update.replace_source_groups(old,state,manifest,mapping,[packet])
    np.testing.assert_array_equal(merged['replaced'],[0,1,2,3])
    assert_parent_exact(merged,old,state,[4])
