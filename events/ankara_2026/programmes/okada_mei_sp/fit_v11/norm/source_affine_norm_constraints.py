"""All-source proposal-chart affine native norms, without adding a DOF.
Ordinary S is a temporary witness in this proposal, never a physical fact
or a Phase-II station optimum. Profiles retain their original normal and
the already validated exact finite-Q implicit UV derivative.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from fixed_entity_norm_constraints import read,verify,ref,array,FixedEntityNorms,load_fixed_entity_norms

@dataclass
class AllSourceAffineNorms:
 rows:np.ndarray
 kind:np.ndarray
 residual:np.ndarray
 J:sparse.csr_matrix
 metadata:dict
 def values(self,delta):
  r=self.residual+(self.J@delta).reshape(-1,2)
  if not np.isfinite(r).all():raise ValueError('Nonfinite full-source affine residual')
  return np.einsum('ij,ij->i',r,r),r
 def subset(self,source_rows):
  source_rows=np.asarray(source_rows,dtype=np.int64);idx=np.searchsorted(self.rows,source_rows)
  if np.any(idx>=len(self.rows))or not np.array_equal(self.rows[idx],source_rows)or len(np.unique(idx))!=len(idx):raise ValueError('Invalid/duplicate active norm membership')
  uv=np.ravel(np.column_stack([2*idx,2*idx+1]));meta=dict(self.metadata)
  meta.update(selected_norm_rows=len(idx),selected_kind_counts={str(k):int(np.count_nonzero(self.kind[idx]==k))for k in [0,1,2]})
  return FixedEntityNorms(source_rows.astype('<u8'),self.residual[idx],self.J[uv].tocsr(),meta)
 def audit(self,delta,b):
  q,r=self.values(delta);bad=np.flatnonzero(q>b);worst=int(np.argmax(q-b))
  return dict(active_rows=len(self.rows),ordinary_count=int(np.count_nonzero(self.kind==0)),entity_count=int(np.count_nonzero(self.kind==1)),
   profile_count=int(np.count_nonzero(self.kind==2)),maximum_affine_q=float(q.max()),maximum_affine_q_minus_b=float(np.max(q-b)),
   worst_source_row=int(self.rows[worst]),violating_source_count=len(bad),violating_source_rows=self.rows[bad].tolist(),
   violating_kind_counts={str(k):int(np.count_nonzero(self.kind[bad]==k))for k in [0,1,2]},all_source_affine_norms_within_b=bool(not len(bad)),
   ordinary_fixed_S_is_temporary_numerical_witness=True,profile_original_normal_1D=True,nonlinear_physical_acceptance=False)

def load_all_source_affine_norms(qbar,transport,offset,scale,free):
 # Reuse the completed strict theta, authority, all4aux and layout validator.
 entities=load_fixed_entity_norms(qbar,transport,offset,scale,free)
 T=Path(transport);t=read(T);S=verify(t['semantic']);sem=read(S);P=verify(t['packet']);pk=read(P)
 active=array(sem['active_source_mask'],S.parent).astype(bool);kind=array(sem['arrays']['kind'],S.parent)
 if not np.array_equal(np.flatnonzero(~active),[25537])or np.any(kind>2):raise ValueError('Active/raw source scope')
 rows=np.flatnonzero(active).astype('<u8');E=array(pk['arrays']['engineering_budget_px'],P.parent)
 normal=array(sem['arrays']['profile_normal'],S.parent);profile=np.flatnonzero(active&(kind==2))
 if len(profile)!=961 or not np.isfinite(normal[profile]).all():raise ValueError('Original profile normal coverage')
 offset=np.asarray(offset);scale=np.asarray(scale);free=np.asarray(free);mapping=np.full(offset[-1],-1,np.int64);mapping[free]=np.arange(len(free))
 c=t['physical_UV_CSR'];ptr=array(c['row_offsets'],T.parent);bi=array(c['block_id'],T.parent);co=array(c['component'],T.parent);v=array(c['values'],T.parent)
 ambient=offset[bi]+co;keep=mapping[ambient]>=0;ri=np.repeat(np.arange(364034),np.diff(ptr).astype(np.int64))
 UV=sparse.csr_matrix((v[keep]*scale[ambient[keep]],(ri[keep],mapping[ambient[keep]])),shape=(364034,len(free)))
 ordinary=np.flatnonzero(active&(kind!=2));diag=np.ravel(np.column_stack([2*ordinary,2*ordinary+1]));weights=np.repeat(1/E[ordinary],2)
 lr=np.r_[diag,2*profile,2*profile];lc=np.r_[diag,2*profile,2*profile+1];lv=np.r_[weights,normal[profile,0]/E[profile],normal[profile,1]/E[profile]]
 transform=sparse.csr_matrix((lv,(lr,lc)),shape=(364034,364034));J=(transform@UV).tocsr();J.eliminate_zeros()
 error=array(t['arrays']['error_uv'],T.parent).reshape(182017,2).copy()
 if not np.isfinite(error[active]).all()or np.any(E[active]<=0):raise ValueError('All-source native primal arithmetic')
 r=np.zeros((182017,2));r[ordinary]=error[ordinary]/E[ordinary,None]
 r[profile,0]=np.einsum('ij,ij->i',normal[profile],error[profile])/E[profile]
 pair=np.ravel(np.column_stack([2*rows,2*rows+1]));J=J[pair].tocsr()
 if not np.isfinite(J.data).all():raise ValueError('All-source normalized Jacobian arithmetic')
 meta=dict(entities.metadata);meta.update(schema='rinkfit.all_source_affine_native_norm_proposal_chart.v1',active_rows=182016,
  ordinary_entity_native_2D_count=len(ordinary),genuine_profile_normal_1D_count=961,entities_always_included=1692,
  ordinary_station_fixed_only_for_this_numerical_proposal=True,final_PhaseII_joint_station_claim=False,unselected_source_not_deleted=True,
  all_source_affine_check_is_not_nonlinear_acceptance=True)
 return AllSourceAffineNorms(rows,kind[rows],r[rows],J,meta)
