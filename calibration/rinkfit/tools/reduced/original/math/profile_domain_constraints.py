"""Exact original profile finite-domain linearization; no new physical prior."""
from pathlib import Path
import hashlib,json,numpy as np
from scipy import sparse
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def array(base,d):
 p=Path(base)/d["file"];assert sha(p)==d["sha256"];z=np.fromfile(p,d["dtype"]).reshape(d["shape"]);assert np.isfinite(z).all();return z
def load_profile_domains(qbar_path,domain_path,offset,scale,free):
 Q=Path(qbar_path);qm=read(Q);P=Path(domain_path);d=read(P)
 assert d["schema"]=="rinkfit.original_finite_profile_implicit_transport.v1"
 for key in["model","semantic","packet"]:assert d[key]["sha256"]==qm[key]["sha256"] and sha(d[key]["path"])==d[key]["sha256"]
 packet=read(d['packet']['path']);semantic_path=Path(d['semantic']['path']);semantic=read(semantic_path)
 N=int(packet['observations']);assert d['raw_rows']==N and semantic['observations']==N
 active=array(semantic_path.parent,semantic['active_source_mask']).astype(bool)
 kind=array(semantic_path.parent,semantic['arrays']['kind'])
 assert active.shape==kind.shape==(N,)
 expected_rows=np.flatnonzero(active&(kind==2));profiles=len(expected_rows)
 assert d['profiles']==profiles
 cp=Path(d["checkpoint"]["path"]);assert sha(cp)==d["checkpoint"]["sha256"];c=read(cp);inp=Path(qm["input"]["path"]);assert sha(inp)==qm["input"]["sha256"];idef=read(inp);selected=array(Q.parent,qm["source_rows"]).astype(int)
 for key,ckey in[("station","station"),("path","branch"),("row","row"),("alpha","alpha")]:
  expected=array(inp.parent,idef["arrays"][key]).copy()
  if key in["station","row"]:expected[selected]=array(Q.parent,qm["linearization_"+key])
  cd=c["row_state_files"][ckey];assert hashlib.sha256(expected.tobytes()).hexdigest()==cd["sha256"],"Profile domain and qbar use different complete nuisance states: "+key
 rows=array(P.parent,d["profile_rows"]).astype(int);assert rows.shape==(profiles,) and np.array_equal(rows,expected_rows)
 mapping=np.full(offset[-1],-1,int);mapping[free]=np.arange(len(free));n=len(free)
 def csr(key):
  z=d[key];ptr=array(P.parent,z["row_offsets"]).astype(int);bid=array(P.parent,z["block_id"]).astype(int);com=array(P.parent,z["component"]).astype(int);v=array(P.parent,z["values"])
  assert ptr.shape==(N+1,) and ptr[0]==0 and ptr[-1]==len(v) and np.all(np.diff(ptr)>=0)
  assert bid.shape==com.shape==v.shape and np.all((bid>=0)&(bid<len(offset)-1)) and np.all((com>=0)&(com<np.diff(offset)[bid]))
  col=offset[bid]+com;keep=mapping[col]>=0;rr=np.repeat(np.arange(N),np.diff(ptr));return sparse.csr_matrix((v[keep]*scale[col[keep]],(rr[keep],mapping[col[keep]])),shape=(N,n))[rows]
 S=csr("dS");V=csr("dV");AL=csr("dalpha");v=array(cp.parent,c["row_state_files"]["row"])[rows]
 lowerS=array(P.parent,d["delta_S_lower"]);upperS=array(P.parent,d["delta_S_upper"]);lowerA=array(P.parent,d["delta_alpha_lower"]);upperA=array(P.parent,d["delta_alpha_upper"])
 # Every bound is the source/model's original finite domain, with scale=1.
 H=sparse.vstack([S,-S,V,-V,AL,-AL],format="csr");rhs=np.r_[upperS,-lowerS,1079.-v,v,upperA,-lowerA]
 assert np.all(rhs>=0),"Current complete profile state is outside its original domain"
 mapping_doc=dict(schema="rinkfit.original_profile_domain_linearized_rows.v1",transport=dict(path=str(P),sha256=sha(P)),checkpoint=d["checkpoint"],count=len(rhs),profiles=profiles,source_rows=rows.tolist(),ordering=[dict(kind=k,count=profiles,scale=1.)for k in["station_upper","station_lower","row_upper","row_lower","alpha_upper","alpha_lower"]],extra_LP_duals_are_not_scene415_or_source_q_duals=True,original_bounds_unchanged=True)
 return H,rhs,mapping_doc
