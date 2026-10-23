import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_joint.observed_primal_restart import reference
from rinkfit_joint.fixed_local_anchor_audit import audit


def fixture(root, distances, *, status=(7,7), baseline=None):
    root.mkdir()
    def put(name,value):
        p=root/name;p.write_text(json.dumps(value),encoding='utf8');return reference(p)
    def col(name,value,dtype):
        arr=np.asarray(value,dtype=dtype);p=root/(name+'.bin');arr.tofile(p)
        return dict(reference(p),dtype=arr.dtype.str,shape=list(arr.shape))
    model=put('model.json',dict(materials=[{'paths':[{'kind':2,'physical_point':True}]},{'paths':[{'kind':2,'physical_point':True,'geometry_block':0}]}]))
    packet=put('packet.json',dict(source_row_ids=['corner','texture'],shot_ids=['s01'],arrays=dict(material_index=col('material',[0,1],'<u4'),shot_index=col('shot',[0,0],'<u4'),E=col('E',[2.,2.],'<f8'))))
    cache=root/'distances.npz';np.savez(cache,packet_rows=np.asarray([0,1],dtype='<u4'),true_RS_distance=np.asarray(distances),status=np.asarray(status,dtype='u1'))
    r=dict(model=model,packet=packet,distances=reference(cache),rows=2,cohort_row_ID_sha256=hashlib.sha256(b'0\tcorner\n1\ttexture\n').hexdigest())
    if baseline:r['baseline']=reference(baseline)
    return Path(put('report.json',r)['file'])


def test_lower_aggregate_does_not_hide_known_local_anchor_regression(tmp_path):
    old=fixture(tmp_path/'before',[1.,20.]);now=fixture(tmp_path/'after',[10.,1.],baseline=old)
    r=audit(now,tmp_path/'out.json');c=r['same_source_comparison']
    assert c['known_local_ad_points']['baseline']['rms_px']==1.
    assert c['known_local_ad_points']['candidate']['rms_px']==10.
    assert c['known_local_ad_points']['worsened_beyond_original_E']==1
    assert c['free_or_initializer_surface_points']['candidate']['rms_px']==1.
    assert r['world_coordinates_calibrated'] is False


def test_invalid_anchor_remains_in_denominator(tmp_path):
    report=fixture(tmp_path/'data',[np.inf,1.],status=(0,7))
    s=audit(report,tmp_path/'out.json')['summary']['known_local_ad_points']
    assert s['N']==1 and s['undefined_error_rows']==1 and s['rms_px']=='+Infinity'


def test_changed_cache_is_rejected(tmp_path):
    report=fixture(tmp_path/'data',[1.,2.])
    with (report.parent/'distances.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed absolute pinned file'):audit(report,tmp_path/'out.json')


def test_row_reordering_is_rejected_even_with_current_cache_hash(tmp_path):
    report=fixture(tmp_path/'data',[1.,2.]);r=json.loads(report.read_text())
    cache=report.parent/'distances.npz';np.savez(cache,packet_rows=np.asarray([1,0],dtype='<u4'),true_RS_distance=[1.,2.],status=[7,7])
    r['distances']=reference(cache);report.write_text(json.dumps(r),encoding='utf8')
    with pytest.raises(ValueError,match='row identities'):audit(report,tmp_path/'out.json')
