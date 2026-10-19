"""Reject incompatible W aliases and evidence assigned to the wrong original pixel."""
import copy
import hashlib
import json
import numpy as np
import pytest
from rinkfit_start.source_audit_revision import _verify_flattened_statistics, _verify_native_source_key, _source_registry_rows


def _ref(path):
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')
    return _ref(path)


def _statistics(tmp_path):
    W=np.array([[1.,2.]])
    f=tmp_path/'W.f64';W.tofile(f)
    proof={'original_source_rows':[0,1],'source_rows':[1]}
    p=_json(tmp_path/'proof.json',proof)
    r={'group_id':'group','source_rows':[1],'rank':1,'W':dict(_ref(f),dtype='<f8',shape=[1,2]),'proof':p}
    canonical={'group':{'source_rows':np.array([1]),'rank':1,'W':W,'proof':proof}}
    return {'replacements':[r]},canonical,[{'group_id':'group','matrix_sha256':'original'}]


def test_flattened_W_is_an_exact_alias_of_validated_source_covariance(tmp_path):
    stats,canonical,groups=_statistics(tmp_path)
    _verify_flattened_statistics(tmp_path,stats,canonical,groups)
    W=np.array([[2.,4.]])
    f=tmp_path/'other_W.f64';W.tofile(f)
    stats['replacements'][0]['W']=dict(_ref(f),dtype='<f8',shape=[1,2])
    with pytest.raises(ValueError,match='Flattened W differs'):
        _verify_flattened_statistics(tmp_path,stats,canonical,groups)


@pytest.mark.parametrize('change',['missing','duplicate','extra','wrong_rows'])
def test_flattened_W_membership_cannot_diverge(tmp_path,change):
    stats,canonical,groups=_statistics(tmp_path)
    if change=='missing':stats['replacements']=[]
    elif change=='duplicate':stats['replacements']*=2
    elif change=='extra':stats['replacements'].append(dict(stats['replacements'][0],group_id='other'))
    else:stats['replacements'][0]['source_rows']=[0]
    with pytest.raises(ValueError):_verify_flattened_statistics(tmp_path,stats,canonical,groups)


def _native(tmp_path):
    packet={'source_pts':123,'programme_frame_index_1based':4,
        'source':{'native_pts':123,'actual_pts':123,'png_sha256':'original_png','native_hw':[100,200]},
        'features':[{'id':'Ad.top','segments':[{'uv':[[21.,34.]]}]}]}
    ps=_json(tmp_path/'packet.json',packet)
    canonical={'source_row':2,'source_packet':ps,'snapshot_frame':4,'source_feature_id':'Ad.top',
        'source_segment_index':0,'original_vertex_index':0,'native_query_id':None}
    rs=_json(tmp_path/'record.json',canonical)
    record={'source_row':2,'native_pts':123,'source_point_id':'Ad.top/segment/0/vertex/0',
        'original_source_label':'Ad.top','source_uv':[21.,34.],'source_record':rs}
    evidence={'source_registry_record':rs,'source_packet':ps,
        'exact_native_frame':{'native_pts':123,'sha256':'original_png','width':200,'height':100}}
    return record,evidence,canonical


def test_original_native_vertex_key_is_verified(tmp_path):
    r,e,c=_native(tmp_path)
    _verify_native_source_key(tmp_path,r,e,c)


@pytest.mark.parametrize('field,value',[('native_pts',0),('source_uv',[0.,0.]),('source_point_id','wrong'),('original_source_label','wrong')])
def test_matching_duplicated_audit_fields_do_not_override_actual_source(tmp_path,field,value):
    r,e,c=_native(tmp_path)
    r[field]=value;e[field]=value
    with pytest.raises(ValueError,match='differs from original source'):
        _verify_native_source_key(tmp_path,r,e,c)


def test_frame_native_PTS_must_match_packet(tmp_path):
    r,e,c=_native(tmp_path);e['exact_native_frame']['native_pts']=0
    with pytest.raises(ValueError):_verify_native_source_key(tmp_path,r,e,c)


def test_changed_registry_copy_is_rejected(tmp_path):
    r,e,c=_native(tmp_path);bad=copy.deepcopy(c);bad['source_row']=1
    rs=_json(tmp_path/'different_record.json',bad);r['source_record']=rs;e['source_registry_record']=rs
    with pytest.raises(ValueError,match='immutable original registry'):_verify_native_source_key(tmp_path,r,e,c)


def test_registry_row_order_and_completeness_are_verified(tmp_path):
    f=tmp_path/'rows.jsonl';f.write_text('\n'.join(json.dumps({'source_row':i})for i in range(3))+'\n')
    assert _source_registry_rows(tmp_path,_ref(f),{2},3)=={2:{'source_row':2}}
    f.write_text('\n'.join(json.dumps({'source_row':i})for i in [0,2,1])+'\n')
    with pytest.raises(ValueError,match='row order'):_source_registry_rows(tmp_path,_ref(f),{2},3)
