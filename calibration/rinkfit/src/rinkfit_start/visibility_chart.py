"""Finite image-row guard for a temporary fixed-station proposal chart.
The source station remains a physical nuisance variable; this numerical chart
is not a new observation, physical fact, residual, or permanent constraint.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import sparse

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb")as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def verify(ref):
 p=Path(ref["path"]);assert sha(p)==ref["sha256"],str(p);return p
def array(base,desc):
 p=Path(desc["file"]);p=p if p.is_absolute()else Path(base)/p
 assert sha(p)==desc["sha256"],str(p)
 z=np.fromfile(p,dtype=desc["dtype"]).reshape(desc["shape"])
 assert z.dtype.kind!="f"or np.isfinite(z).all(),str(p)
 return z

def load_visibility_chart_guards(qbar_manifest,wholeUV_manifest,offset,scale,free,source_rows):
 Q=Path(qbar_manifest);q=read(Q);T=Path(wholeUV_manifest);t=read(T)
 assert t["schema"]=="rinkfit.whole_original_W_native_UV_transport.v1"
 assert t["raw_rows"]==182017 and t["active_rows"]==182016 and t["excluded_rows"]==[25537]
 for key in ["model","packet","semantic"]:
  assert t[key]["sha256"]==q[key]["sha256"],"Different current state authority: "+key
  verify(t[key]);verify(q[key])
 C=verify(t["checkpoint"]);cp=read(C);assert cp["schema"]=="rinkfit.complete_continuous_reduced_SLP_checkpoint.v1"
 assert cp["model"]["sha256"]==t["model"]["sha256"];verify(cp["model"])
 I=verify(q["input"]);inp=read(I);selected=array(Q.parent,q["source_rows"]).astype(np.int64)
 assert np.all(selected>=0)and np.all(selected<182017)and len(np.unique(selected))==len(selected)
 authority={}
 for key,ckey in [("station","station"),("path","branch"),("row","row"),("alpha","alpha")]:
  expected=array(I.parent,inp["arrays"][key]).copy()
  if key in ["station","row"]:expected[selected]=array(Q.parent,q["linearization_"+key])
  actual=array(C.parent,cp["row_state_files"][ckey])
  assert expected.shape==actual.shape==(182017,)and expected.dtype==actual.dtype
  assert hashlib.sha256(expected.tobytes()).hexdigest()==cp["row_state_files"][ckey]["sha256"],"Different complete nuisance state: "+key
  assert np.array_equal(expected,actual),"Different complete nuisance bytes: "+key
  authority[ckey]=actual
 rows=np.asarray(source_rows,dtype=np.int64);assert rows.ndim==1 and len(rows)>0 and len(np.unique(rows))==len(rows)
 assert np.all(rows>=0)and np.all(rows<182017)and 25537 not in rows
 packet=read(verify(t["packet"]));height=int(packet["native_image_wh"][1]);assert height==1080
 v=authority["row"][rows];assert np.all(v>=0.)and np.all(v<=height-1)
 offset=np.asarray(offset,dtype=np.int64);scale=np.asarray(scale);free=np.asarray(free,dtype=np.int64)
 assert len(scale)==offset[-1]and len(np.unique(free))==len(free)and np.all(scale>0)
 mapping=np.full(offset[-1],-1,np.int64);mapping[free]=np.arange(len(free))
 c=t["physical_UV_CSR"];ptr=array(T.parent,c["row_offsets"]).astype(np.int64)
 bid=array(T.parent,c["block_id"]).astype(np.int64);com=array(T.parent,c["component"]).astype(np.int64);values=array(T.parent,c["values"])
 assert len(ptr)==2*182017+1 and ptr[-1]==len(values)==len(bid)==len(com)
 row_ids=[];col_ids=[];data=[]
 for k,row in enumerate(rows):
  a,b=ptr[2*row+1:2*row+3];ambient=offset[bid[a:b]]+com[a:b]
  keep=mapping[ambient]>=0;row_ids.append(np.full(int(keep.sum()),k));col_ids.append(mapping[ambient[keep]]);data.append(values[a:b][keep]*scale[ambient[keep]])
 Jv=sparse.csr_matrix((np.concatenate(data),(np.concatenate(row_ids),np.concatenate(col_ids))),shape=(len(rows),len(free)))
 H=sparse.vstack([Jv,-Jv],format="csr");rhs=np.r_[height-1-v,v]
 detail=dict(schema="rinkfit.temporary_fixed_station_visibility_chart.v1",transport=dict(path=str(T),sha256=sha(T)),checkpoint=t["checkpoint"],qbar=dict(path=str(Q),sha256=sha(Q)),source_rows=rows.tolist(),count=len(rhs),ordering=[dict(kind="row_upper",count=len(rows),scale=1.),dict(kind="row_lower",count=len(rows),scale=1.)],native_row_domain=[0.,float(height-1)],saved_rows=v.tolist(),complete_nuisance_sha256={k:cp["row_state_files"][k]["sha256"]for k in ["station","branch","row","alpha"]},temporary_algorithm_chart_not_physical_fact=True,original_station_remains_free=True,only_this_proposal_uses_fixed_station_witness=True,source_W_E_unchanged=True,extra_chart_duals_are_not_original_scene_or_source_q_duals=True,full_original_source_physics_gate_still_required=True)
 return H,rhs,detail


def select_visibility_chart_risks(qbar_manifest,wholeUV_manifest,offset,scale,free,trust_scaled):
 """Conservative current box-touch selection, not a physical certificate.
 Every selected row still needs load_visibility_chart_guards and the original
 full-source nonlinear gate. No source obligation is removed by selection.
 """
 T=Path(wholeUV_manifest);t=read(T);S=verify(t["semantic"]);sem=read(S)
 active=array(S.parent,sem["active_source_mask"]).astype(bool);kind=array(S.parent,sem["arrays"]["kind"])
 ordinary=active&(kind==0);assert active.sum()==182016 and not active[25537] and ordinary.any()
 # Reuse the complete state/authority verification before selecting any rows.
 anchor=int(np.flatnonzero(ordinary)[0]);_,_,authority_mapping=load_visibility_chart_guards(qbar_manifest,T,offset,scale,free,[anchor])
 C=verify(t["checkpoint"]);cp=read(C);model=read(verify(t["model"]));offset=np.asarray(offset,dtype=np.int64);scale=np.asarray(scale);free=np.asarray(free,dtype=np.int64)
 expected_offset=np.r_[0,np.cumsum([len(b["values"])for b in model["blocks"]])];assert np.array_equal(offset,expected_offset)
 value=np.concatenate([b["values"]for b in model["blocks"]]);lower=np.concatenate([b["lower"]for b in model["blocks"]]);upper=np.concatenate([b["upper"]for b in model["blocks"]]);assert np.array_equal(scale,np.concatenate([b["scales"]for b in model["blocks"]]))
 assert np.isfinite(trust_scaled)and trust_scaled>0
 delta_lower=np.maximum(-float(trust_scaled)*scale[free],lower[free]-value[free]);delta_upper=np.minimum(float(trust_scaled)*scale[free],upper[free]-value[free]);assert np.all(delta_lower<=0)and np.all(delta_upper>=0)
 radius=np.zeros(len(value));radius[free]=np.maximum(abs(delta_lower),abs(delta_upper))
 c=t["physical_UV_CSR"];ptr=array(T.parent,c["row_offsets"]).astype(np.int64);bid=array(T.parent,c["block_id"]).astype(np.int64);com=array(T.parent,c["component"]).astype(np.int64);val=array(T.parent,c["values"])
 capacity=np.zeros(182017)
 # Bounded working chunks affect only summation scheduling, never coverage.
 for begin in range(0,182017,4096):
  end=min(182017,begin+4096);a,b=int(ptr[2*begin]),int(ptr[2*end]);prefix=np.r_[0.,np.cumsum(abs(val[a:b])*radius[offset[bid[a:b]]+com[a:b]])];ids=np.arange(begin,end);capacity[ids]=prefix[ptr[2*ids+2]-a]-prefix[ptr[2*ids+1]-a]
 row=array(C.parent,cp["row_state_files"]["row"]);margin=np.minimum(row,1079.-row);risk=ordinary&(margin<=capacity);rows=np.flatnonzero(risk)
 assert np.isfinite(capacity).all()and np.all(capacity>=0)
 detail=dict(schema="rinkfit.temporary_visibility_chart_box_touch_selection.v1",transport=dict(path=str(T),sha256=sha(T)),qbar=authority_mapping["qbar"],checkpoint=t["checkpoint"],complete_nuisance_sha256=authority_mapping["complete_nuisance_sha256"],trust_scaled=float(trust_scaled),ordinary_rows=int(ordinary.sum()),source_rows=rows.tolist(),count=len(rows),criterion="ordinary current saved-V margin <= sum(abs(fixed-S trueRS Jv) * max(abs(original-box intersected scaled-TR physical delta)))",conservative_possible_touch_only_not_violation_or_feasible_direction=True,temporary_algorithm_chart_not_physical_fact=True,selection_does_not_remove_any_source_obligation=True,profile_and_entity_not_reclassified=True,records=[dict(source_row=int(i),saved_row=float(row[i]),margin_px=float(margin[i]),physical_box_Jv_capacity_px=float(capacity[i]))for i in rows])
 return rows,detail
