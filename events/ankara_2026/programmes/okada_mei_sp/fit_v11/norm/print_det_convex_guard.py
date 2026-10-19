"""Conservative convex local guard for the ORIGINAL print determinant g.
Data-only operator; no model/source/E/W change, no nonlinear acceptance or KKT.
Physical columns use the caller's original scaled-free coordinates.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json,numpy as np
from scipy import sparse
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf8"))
def verify(d,base=None):
 p=Path(d.get("path",d.get("file")));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=d["sha256"]:raise ValueError("Changed determinant-guard authority "+str(p))
 return p
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
@dataclass
class PrintDetConvexGuard:
 g0:np.ndarray
 G:sparse.csr_matrix
 B:sparse.csr_matrix
 A0:np.ndarray
 original_g_scale:np.ndarray
 mapping:list
 authority:dict
 def values(self,physical_scaled):
  d=np.asarray(physical_scaled,float)
  if d.shape!=(self.G.shape[1],)or not np.isfinite(d).all():raise ValueError("Complete finite original physical coordinates")
  D=(self.B@d).reshape(-1,4);v=self.A0+D;linear=self.g0+self.G@d
  exact=(1e-6-(v[:,0]*v[:,3]-v[:,1]*v[:,2]))/self.original_g_scale
  quadratic=.5*np.einsum("ij,ij->i",D,D)/self.original_g_scale
  upper=linear+quadratic
  return dict(exact_original_g=exact,convex_upper=upper,linear=linear,D=D,convex_quadratic_term=quadratic)
 def audit(self,delta):
  v=self.values(delta);g=v["exact_original_g"];u=v["convex_upper"]
  return dict(original_print_constraints=len(g),original_exact_g_max=float(g.max()),convex_guard_max=float(u.max()),original_g_violating_ids=[self.mapping[i]["id"]for i in np.flatnonzero(g>0)],convex_guard_violating_ids=[self.mapping[i]["id"]for i in np.flatnonzero(u>0)],exact_original_print_g_pass=bool(np.all(g<=0)),convex_print_guard_pass=bool(np.all(u<=0)),convex_upper_minus_exact_min=float(np.min(u-g)),no_new_physical_constraint_or_DOF=True,nonlinear_complete_scene_still_required=True)
 def oa_support(self,delta,indices=None):
  """Return C d <= rhs tangent supports; ALL guard values require final audit."""
  d=np.asarray(delta,float);v=self.values(d);ids=np.arange(len(self.g0))if indices is None else np.asarray(indices,dtype=int)
  if ids.ndim!=1 or np.any(ids<0)or np.any(ids>=len(self.g0))or len(np.unique(ids))!=len(ids):raise ValueError("Guard index mapping")
  weights=(v["D"]/self.original_g_scale[:,None]).reshape(-1)
  contract=sparse.csr_matrix((weights,(np.repeat(np.arange(len(self.g0)),4),np.arange(4*len(self.g0)))),shape=(len(self.g0),4*len(self.g0)))
  C=(self.G+contract@self.B)[ids].tocsr();rhs=np.asarray(C@d).ravel()-v["convex_upper"][ids]
  return C,rhs,ids
 def clarabel_soc6(self,center,coordinate_scale,station_columns=0):
  """s=rhs-A*z, physical delta=center+coordinate_scale*z[:n].
  Returns exactly 93 SOC6 blocks (or the model's actual number).
  Added station columns are zero; caller retains original A/box/profile/W.
  """
  n=self.G.shape[1];c=np.asarray(center,float);sc=np.asarray(coordinate_scale,float);k=int(station_columns)
  if c.shape!=(n,)or sc.shape!=(n,)or np.any(sc<=0)or not np.isfinite(np.r_[c,sc]).all()or k<0:raise ValueError("Finite physical chart scale")
  m=len(self.g0);gn=(self.G@sparse.diags(sc)).tocsr();bn=(sparse.diags(np.repeat(np.sqrt(2/self.original_g_scale),4))@self.B@sparse.diags(sc)).tocsr()
  gc=self.g0+self.G@c;dc=np.sqrt(2/self.original_g_scale)[:,None]*(self.B@c).reshape(-1,4)
  A=sparse.vstack([gn,-bn,gn],format="csr");perm=np.column_stack([np.arange(m),m+4*np.arange(m),m+4*np.arange(m)+1,m+4*np.arange(m)+2,m+4*np.arange(m)+3,5*m+np.arange(m)]).ravel()
  A=A[perm];A=sparse.hstack([A,sparse.csr_matrix((6*m,k))],format="csc");rhs=np.column_stack([1-gc,dc,-1-gc]).ravel()
  return A,rhs,[6]*m
def from_full_norm_operator(norms,free,scales):
 """Reuse the independently validated full-source operator's same-state closure."""
 meta=norms.metadata;Q=verify(meta["qbar"]);T=verify(meta["transport"]);q=read(Q);tr=read(T)
 for key in ["checkpoint","model","packet","semantic"]:
  if meta[key]["sha256"]!=tr[key]["sha256"]:raise ValueError("Mixed full-norm determinant authority "+key)
  verify(tr[key])
 CP=verify(tr["checkpoint"]);cp=read(CP);MP=verify(tr["model"]);model=read(MP)
 if cp["model"]["sha256"]!=tr["model"]["sha256"]:raise ValueError("Mixed checkpoint model")
 aux=meta["complete_nuisance_sha256"]
 for k in ["station","branch","row","alpha"]:
  d=cp["row_state_files"][k];verify(d,CP.parent)
  if d["sha256"]!=aux[k]:raise ValueError("Mixed complete nuisance "+k)
 blocks=model["blocks"];off=np.r_[0,np.cumsum([len(b["values"])for b in blocks])]
 scale=np.concatenate([np.array(b["scales"])for b in blocks]);fixed=np.zeros(off[-1],bool)
 for i,b in enumerate(blocks):
  if b.get("constant",False):fixed[off[i]:off[i+1]]=True
  for j in b.get("fixed_indices",[]):fixed[off[i]+j]=True
 free=np.asarray(free,dtype=int);scales=np.asarray(scales,float)
 if not np.array_equal(free,np.flatnonzero(~fixed))or not np.array_equal(scales,scale):raise ValueError("Original free layout/gauge/scales")
 mapping=np.full(off[-1],-1,int);mapping[free]=np.arange(len(free));spec=[(i,s)for i,s in enumerate(q["scene_specs"])if s["type"]==11]
 vals=[];gg=[];ss=[];maps=[];rr=[];cc=[];vv=[]
 for j,(scene_i,s)in enumerate(spec):
  bi=s["block_ids"][-1];b=blocks[bi];a=np.array(b["values"]);amb=off[bi]+np.array([0,1,3,4])
  if len(a)!=6 or not b["name"].startswith("print/"):raise ValueError("Actual print affine layout")
  vals.append(a[[0,1,3,4]]);ss.append(float(s["scale"]));maps.append(dict(id=s["id"],scene_index=scene_i,block_id=bi,block_name=b["name"]))
  for row,col in enumerate(amb):
   if mapping[col]>=0:rr.append(4*j+row);cc.append(mapping[col]);vv.append(scale[col])
 A0=np.array(vals);ss=np.array(ss);B=sparse.csr_matrix((vv,(rr,cc)),shape=(4*len(spec),len(free)))
 coef=np.column_stack([-A0[:,3],A0[:,2],A0[:,1],-A0[:,0]])/ss[:,None]
 contraction=sparse.csr_matrix((coef.ravel(),(np.repeat(np.arange(len(spec)),4),np.arange(4*len(spec)))),shape=(len(spec),4*len(spec)))
 G=(contraction@B).tocsr();g0=(1e-6-(A0[:,0]*A0[:,3]-A0[:,1]*A0[:,2]))/ss
 sg=np.fromfile(verify(q["scene_values"],Q.parent),q["scene_values"]["dtype"])
 if not np.array_equal(g0,sg[[i for i,s in spec]]):raise ValueError("Original production determinant g identity")
 authority={k:tr[k]for k in ["checkpoint","model","packet","semantic"]};authority.update(transport=ref(T),qbar=ref(Q),complete_auxiliary_sha256=aux,source=ref(__file__),formula="g_upper=g0+dg*delta+0.5*||DeltaA_2x2||F^2/original_scene_scale; dominates exact original 1e-6-det(A+DeltaA)")
 return PrintDetConvexGuard(g0,G,B,A0,ss,maps,authority)

