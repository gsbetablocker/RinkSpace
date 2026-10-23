import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rinkfit_joint.packet_compile import append_admitted_sources
from rinkfit_joint.io_utils import file_sha256,object_sha256


def test_two_appends_preserve_paths_queries_domains_and_source_order(tmp_path):
    def write(name,obj):
        path=tmp_path/name;path.write_text(json.dumps(obj),encoding='utf8');return path
    def binary(name,a):
        path=tmp_path/(name+'.bin');a.tofile(path)
        return dict(file=str(path),sha256=file_sha256(path),dtype=a.dtype.str,shape=list(a.shape))
    values=dict(uv=([[100.,100.]],'<f8'),native_pts=([621312],'<i8'),shot_index=([1],'<u2'),kind=([0],'|u1'),material_index=([0],'<u4'),path=([1],'<u4'),admission=([1],'|u1'),fit_role=([1],'|u1'),E=([2.],'<f8'),station=([.5],'<f8'),row=([100.],'<f8'),alpha=([0.],'<f8'),source_sigma=([1.],'<f8'),profile_normal=([[0.,1.]],'<f8'),query_uv=([[100.,100.]],'<f8'),alpha_bounds=([[-3.,3.]],'<f8'))
    arrays={k:binary(k,np.asarray(a,dtype))for k,(a,dtype)in values.items()}
    W=binary('W',np.eye(2,dtype='<f8'))
    query=dict(row_index=0,alpha=[-3.,3.],native_uv=[[100.,97.],[100.,103.]],normal=[0.,1.])
    domains=dict(schema='rinkfit.source_row_station_domains.v1',source_refs=[],rows=[dict(row_index=0,source_row_id='old',path_index=1,allowed_station_intervals=[[.25,.75]])])
    original=dict(row_count=1,arrays=arrays,groups=[dict(id='old',rows=[0],rank=2,W=W,statistic_kind='initialization_surrogate')],source_row_ids=['old'],source_dataset_ids=['old_dataset'],source_manifests=[],current_original_W_rank=0,joint_statistics_rank=2,active_rows=1,new_source_start=0,allowed_paths={'0':[1]},shot_ids=['s01','s02'],queries=[query],source_station_domains=domains)
    parent=write('parent.json',original)
    mapping=write('map.json',dict(s01_material_indices_preserved=True,semantic_to_material={'material':dict(geometry_ready=True,material_index=0)}))
    model=write('model.json',dict(materials=[dict(paths=[dict(id='p0',physical_point=False),dict(id='p1',physical_point=False)])]))
    for batch in range(2):
        row=dict(source_row_id='new'+str(batch),source_uv=[200.,200.],current_uv=[200.,200.],native_pts=621568+256*batch,shot_id='s02',joint_material_id='material',joint_material_index=0,measurement_operator_resolved='curve',solver_eligible=True,allowed_path_indices=[batch])
        ledger=tmp_path/('rows'+str(batch)+'.jsonl');ledger.write_text(json.dumps(row)+'\n',encoding='utf8')
        mean=object_sha256(dict(native_pts=row['native_pts'],current_uv=row['current_uv'],source_row_id=row['source_row_id'],measurement_operator='curve'))
        stats=write('stats'+str(batch)+'.json',{'g'+str(batch):dict(row_ids=[row['source_row_id']],rank=2,W=W,current_measurements_sha256={row['source_row_id']:mean},statistic_kind='initialization_surrogate')})
        out=tmp_path/('append'+str(batch))
        append_admitted_sources(parent,ledger,stats,mapping,model,out)
        parent=out/'packet.json';packet=json.loads(parent.read_text())
        assert packet['source_row_ids']==['old']+['new'+str(i)for i in range(batch+1)]
        assert packet['queries']==[query]
        assert packet['source_station_domains']==domains
        assert packet['allowed_paths']=={'0':[1],**{str(i+1):[i]for i in range(batch+1)}}
        assert packet['recapture_rows']==[batch+1]
        assert np.fromfile(out/'uv.bin','<f8').reshape(-1,2)[0].tolist()==[100.,100.]
    qualified=copy.deepcopy(original);qualified['board_material_source_qualification']={'file':'not-read','sha256':'0'*64}
    with pytest.raises(ValueError,match='qualified'):
        append_admitted_sources(write('qualified.json',qualified),ledger,stats,mapping,model,tmp_path/'rejected')
