"""Data-only original correlated-W conditional physical tie-break.
The primary minimax LP optimum/constraints remain unchanged. Fixed ordinary S
is a conditional proposal chart, never Phase-II joint station optimality.
"""
from pathlib import Path
import argparse,hashlib,json,time,sys,numpy as np
from scipy import sparse
import clarabel
sys.stdout.reconfigure(encoding="utf-8")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def ref(path):return dict(path=str(path),sha256=sha(path))
def verify(d):
 p=Path(d["path"]);assert sha(p)==d["sha256"];return p
def array(d,base,allow_nonfinite=False):
 p=Path(d["file"]);p=p if p.is_absolute()else base/p
 assert sha(p)==d["sha256"],str(p)
 a=np.fromfile(p,d["dtype"]).reshape(d["shape"])
 if a.dtype.kind=="f"and not allow_nonfinite:assert np.isfinite(a).all(),str(p)
 return a

def primary_program(report_path):
 P=Path(report_path);r=read(P);assert (r["proposed"]and r["lexicographic_l1_succeeded"])or(r.get("primary_master_solved")and r.get("master_ready_for_W_secondary")and not r.get("physical_proposal_authorized",True)and r["requires_original_W_secondary_and_primal_repair"])
 L=verify(r["linearization"]);d=read(L);m=read(verify(d["model"]));blocks=m["blocks"]
 sizes=np.asarray([len(b["values"])for b in blocks]);offset=np.r_[0,np.cumsum(sizes)]
 z=np.load(P.parent/"lp_duals.npz");free=z["free_ambient_indices"];n=len(free)
 mapping=np.full(offset[-1],-1,int);mapping[free]=np.arange(n)
 scale=np.concatenate([b["scales"]for b in blocks]);value=np.concatenate([b["values"]for b in blocks])
 lo=np.concatenate([b["lower"]for b in blocks]);hi=np.concatenate([b["upper"]for b in blocks])
 def csr(c,rows,base=L.parent):
  ptr=array(c["row_offsets"],base).astype(np.int64);bi=array(c["block_id"],base).astype(np.int64);co=array(c["component"],base).astype(np.int64);v=array(c["values"],base)
  ambient=offset[bi]+co;keep=mapping[ambient]>=0;rr=np.repeat(np.arange(rows),np.diff(ptr))
  return sparse.csr_matrix((v[keep]*scale[ambient[keep]],(rr[keep],mapping[ambient[keep]])),shape=(rows,n))
 rows=array(d["source_rows"],L.parent).astype(np.int64);q=array(d["q"],L.parent);status=array(d["status"],L.parent);sel=np.flatnonzero(status==0)
 assert np.array_equal(rows[sel],z["master_source_rows"])
 Q=csr(d["qbar_CSR"],len(rows));G=csr(d["scene_CSR"],len(d["scene_specs"]));g=array(d["scene_values"],L.parent)
 A=sparse.vstack([sparse.hstack([Q[sel],-np.ones((len(sel),1))]),sparse.hstack([G,sparse.csr_matrix((len(g),1))])],format="csr");rhs=np.r_[-q[sel],-g]
 if r.get("profile_domain_constraints"):
  from profile_domain_constraints import load_profile_domains
  dm=r["profile_domain_constraints"];H,domain_rhs,actual_mapping=load_profile_domains(L,verify(dm["transport"]),offset,scale,free);assert actual_mapping==dm
  A=sparse.vstack([A,sparse.hstack([H,sparse.csr_matrix((len(domain_rhs),1))])],format="csr");rhs=np.r_[rhs,domain_rhs]
 if r.get("temporary_visibility_chart_constraints"):
  from temporary_visibility_chart import load_visibility_chart_guards
  cm=r["temporary_visibility_chart_constraints"];Hc,chart_rhs,actual_chart=load_visibility_chart_guards(L,verify(cm["transport"]),offset,scale,free,cm["source_rows"]);actual_chart["count"]=len(chart_rhs);assert actual_chart==cm
  A=sparse.vstack([A,sparse.hstack([Hc,sparse.csr_matrix((len(chart_rhs),1))])],format="csr");rhs=np.r_[rhs,chart_rhs]
 if r["constraint_generation"]:
  A=sparse.vstack([A,sparse.csr_matrix(([-1.],([0],[n])),shape=(1,n+1))],format="csr");rhs=np.r_[rhs,-r["unmodeled_q_constant_floor"]]
 trust=r["trust_scaled_infinity"];lower=np.r_[np.maximum(-trust,(lo[free]-value[free])/scale[free]),0.];upper=np.r_[np.minimum(trust,(hi[free]-value[free])/scale[free]),np.inf]
 assert len(z["master_primal"])==n+1 and len(z["master_inequality_dual"])==len(rhs)
 roundoff=64*np.finfo(float).eps*(1+abs(rhs)+abs(A)@abs(z["master_primal"]))
 assert np.all(A@z["master_primal"]-rhs<=roundoff)
 return dict(report=r,doc=d,model=m,model_path=verify(d["model"]),A=A,rhs=rhs,lower=lower,upper=upper,primary=z,free=free,n=n,offset=offset,scale=scale,value=value,csr=csr)

def load_W_transport(path,p):
 T=Path(path);t=read(T);assert t["schema"]=="rinkfit.whole_original_W_native_UV_transport.v1"
 assert t["model"]["sha256"]==sha(p["model_path"])
 from rinkfit_start.native_norms import validate_semantic_scope
 scope_contract=p.get('source_scope_contract')
 if scope_contract is None:
  from fixed_entity_norm_constraints import SCOPE_CONTRACT
  scope_contract=SCOPE_CONTRACT
 scope,selected_active,_=validate_semantic_scope(t,read(verify(t['semantic'])),read(verify(t['packet'])),scope_contract,Path(t['semantic']['path']).parent)
 for k in["packet","semantic"]:assert t[k]["sha256"]==p["doc"][k]["sha256"]
 # A matching theta alone is insufficient: W depends on the entire nuisance state.
 cp=read(verify(t["checkpoint"]));assert cp["model"]["sha256"]==t["model"]["sha256"]
 inp=verify(p["doc"]["input"]);idef=read(inp);Q=verify(p["report"]["linearization"]);selected=array(p["doc"]["source_rows"],Q.parent).astype(int)
 for key,ckey in[("station","station"),("path","branch"),("row","row"),("alpha","alpha")]:
  state=array(idef["arrays"][key],inp.parent).copy()
  if key in["station","row"]:state[selected]=array(p["doc"]["linearization_"+key],Q.parent)
  desc=cp["row_state_files"][ckey];assert hashlib.sha256(state.tobytes()).hexdigest()==desc["sha256"],"Whole-W transport uses a different complete nuisance state: "+key
  assert sha(Path(t["checkpoint"]["path"]).parent/desc["file"])==desc["sha256"]
 nraw=t["raw_rows"];uv=p["csr"](t["physical_UV_CSR"],2*nraw,T.parent)
 e=array(t["arrays"]["error_uv"],T.parent,allow_nonfinite=True);excluded=np.asarray(t["excluded_rows"],dtype=int);assert np.array_equal(excluded,np.flatnonzero(~selected_active));excluded_columns=np.ravel(np.column_stack([2*excluded,2*excluded+1]));active_columns=np.ones(2*nraw,bool);active_columns[excluded_columns]=False;assert np.isfinite(e[active_columns]).all();e=e.copy();e[excluded_columns]=0.;r=array(t["arrays"]["W_residual"],T.parent);group_rows=array(t["arrays"]["group_rows"],T.parent).astype(np.int64);weights=array(t["arrays"]["group_W"],T.parent)
 rowparts=[];colparts=[];dataparts=[];at=0
 for g in t["groups"]:
  members=group_rows[g["row_offset"]:g["row_offset"]+g["members"]];rank=g["rank"];width=2*len(members);v=weights[g["weight_offset"]:g["weight_offset"]+rank*width].reshape(rank,width);rr,cc=np.nonzero(v)
  rowparts.append(rr+at);colparts.append(2*members[cc//2]+cc%2);dataparts.append(v[rr,cc]);at+=rank
 assert at==scope['full_W_rank']
 W=sparse.csr_matrix((np.concatenate(dataparts),(np.concatenate(rowparts),np.concatenate(colparts))),shape=(at,2*nraw)) if dataparts else sparse.csr_matrix((at,2*nraw))
 assert W[:,excluded_columns].nnz==0
 assert np.max(abs(W@e-r),initial=0.)<1e-10
 J=(W@uv).tocsc();assert J.shape==(scope['full_W_rank'],p["n"])
 return t,J,r,dict(UV_nnz=int(uv.nnz),W_nnz=int(W.nnz),W_J_nnz=int(J.nnz),W_residual_reconstruction_max=float(np.max(abs(W@e-r),initial=0.)),baseline_W_half_cost=float(.5*r@r),ordinary_S_conditional=True,PhaseII_station_optimum=False)

