import json
import sqlite3

import numpy as np
import pytest

from rinkspace.source_state_spill import ExactSourceStateSpill


def test_exact_overlay_dtypes_signed_zero_nan_and_shot_archive(tmp_path):
    store=ExactSourceStateSpill(tmp_path/'state')
    values=dict(quantiles=np.array([np.nan,-0.,1.23456789012345]),
        flags=np.array([0,4,255],np.uint8),queries={'curve':np.array([2,5],np.int16)},
        proof={'source':'own RGB','actual_pts':100})
    store[100]=values
    loaded=store[100]
    for name in ('quantiles','flags'):
        assert loaded[name].dtype==values[name].dtype
        assert loaded[name].tobytes()==values[name].tobytes()
    assert loaded['queries']['curve'].tobytes()==values['queries']['curve'].tobytes()
    assert list(store)==[100] and len(store)==1
    loaded['flags'][0]=99
    assert store[100]['flags'][0]==0
    store.clear()
    assert len(store)==0
    store[200]={'frame':200}
    ref=store.seal()
    receipt=json.loads(open(ref['path'],encoding='utf8').read())
    assert receipt['correction_exposures']==2
    assert receipt['source_points_removed']==0
    connection=sqlite3.connect(receipt['correction_database']['path'])
    assert connection.execute('SELECT pts FROM corrections ORDER BY pts').fetchall()==[(100,),(200,)]
    connection.close()


def test_numeric_graph_is_readonly_bitexact_and_labels_remain_live(tmp_path):
    store=ExactSourceStateSpill(tmp_path/'state')
    raw=np.array([[1.,-0.],[np.nan,3.]])
    frame=dict(pts=100,ids=np.array([10,12],np.int32),query_ids=np.array([4,7],np.int32),
        uv=raw[:,::-1],descriptor=np.arange(12,dtype=np.float16).reshape(2,6),
        label={0:{'physical_material_id':'B1'}},parents={},children={},emitted={0})
    original={k:v.tobytes() for k,v in frame.items() if isinstance(v,np.ndarray)}
    labels=frame['label'];store.spill_graph_frame(frame)
    for name,value in original.items():
        assert frame[name].tobytes()==value
        assert not frame[name].flags.writeable
    assert frame['label'] is labels
    frame['children'][0]=[(1,1,{'own_source_link':True})]
    assert frame['children'][0][0][:2]==(1,1)
    with pytest.raises(ValueError):
        frame['uv'][0,0]=99.
    receipt=json.loads(open(store.seal()['path'],encoding='utf8').read())
    assert len(receipt['graph_numeric_frames'])==1
    assert not receipt['temporal_graph_links_or_labels_removed']


def test_overlay_or_mapped_state_tamper_is_rejected(tmp_path):
    store=ExactSourceStateSpill(tmp_path/'state')
    store[100]={'value':np.array([1.])}
    store.connection.execute("UPDATE corrections SET payload=x'0000' WHERE pts=100")
    store.connection.commit()
    with pytest.raises(ValueError,match='bytes changed'):
        _=store[100]
    with pytest.raises(ValueError,match='bytes changed'):
        store.seal()
    other=ExactSourceStateSpill(tmp_path/'other')
    result=other.spill_graph_frame(dict(pts=200,uv=np.array([[1.,2.]])))
    with open(result['path'],'r+b') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        other.seal()


def test_empty_graph_and_prior_spill_are_preserved(tmp_path):
    store=ExactSourceStateSpill(tmp_path/'state')
    frame=dict(pts=100,uv=np.empty((0,2)),ids=np.empty(0,np.int32))
    store.spill_graph_frame(frame)
    assert frame['uv'].shape==(0,2) and not frame['uv'].flags.writeable
    store.seal()
    with pytest.raises(FileExistsError,match='Preserve'):
        ExactSourceStateSpill(tmp_path/'state')
