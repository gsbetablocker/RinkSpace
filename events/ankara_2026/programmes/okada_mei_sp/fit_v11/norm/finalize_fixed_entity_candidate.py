from pathlib import Path
import argparse,json,time,sys,numpy as np
from scipy import sparse
from fixed_entity_norm_constraints import read,verify,ref,array,FixedEntityNorms
p=argparse.ArgumentParser();p.add_argument('--repair-report',required=True);p.add_argument('--core-directory',required=True);a=p.parse_args()
start=time.monotonic();R=Path(a.repair_report).resolve();rr=read(R)
if not rr['proposed']:raise ValueError('No strict proposal')
P=verify(rr['primary_report']);pr=read(P);A=sparse.load_npz(verify(pr['original_A']));bd=np.load(verify(pr['original_rhs_bounds']));z=np.load(verify(rr['proposal']))
L=verify(rr['linearization']);d=read(L);modelpath=verify(d['model']);m=read(modelpath)
offset=np.r_[0,np.cumsum([len(b['values'])for b in m['blocks']])];free=z['free_ambient_indices'];scale=z['scales'];n=len(free);mapping=np.full(offset[-1],-1,np.int64);mapping[free]=np.arange(n)
def csr(c,rows,base=L.parent):
 ptr=array(c['row_offsets'],base).astype(np.int64);bi=array(c['block_id'],base).astype(np.int64);co=array(c['component'],base).astype(np.int64);v=array(c['values'],base)
 ambient=offset[bi]+co;keep=mapping[ambient]>=0;ri=np.repeat(np.arange(rows),np.diff(ptr))
 return sparse.csr_matrix((v[keep]*scale[ambient[keep]],(ri[keep],mapping[ambient[keep]])),shape=(rows,n))
ctx=dict(report={'linearization':pr['linearization']},doc=d,model=m,model_path=modelpath,free=free,n=n,csr=csr)
sys.path.insert(0,str(Path(a.core_directory).resolve()));from whole_W_qp_core import load_W_transport
t,J,r,Waudit=load_W_transport(verify(rr['transport']),ctx);J.eliminate_zeros();Az=A[:,:n].tocsc();Az.eliminate_zeros()
ec=np.load(verify(pr['entity_chart_arrays']));en=FixedEntityNorms(ec['source_rows'],ec['residual'],sparse.load_npz(verify(pr['entity_J'])),pr['entity_chart']);Ez=en.J.tocsc();Ez.eliminate_zeros()
null=np.flatnonzero((np.diff(Az.indptr)==0)&(np.diff(J.indptr)==0)&(np.diff(Ez.indptr)==0))
delta=z['step_scaled'];canonical=z['canonical_primal'];bstar=float(rr['linearized_epigraph']);assert canonical[-1].tobytes()==np.float64(bstar).tobytes()
v=A@canonical-bd['rhs'];error=64*np.finfo(float).eps*(1+abs(bd['rhs'])+abs(A)@abs(canonical));norm=en.audit(delta,bstar)
assert np.all(v<=error)and np.all(canonical>=bd['lower'])and np.all(canonical<=bd['upper'])and norm['all_entity_affine_norms_within_b']
assert not np.any(delta[null])
diff=J@delta;baseline=float(.5*r@r);linear=float(r@diff);quadratic=float(.5*diff@diff);direct=float(.5*(r+diff)@(r+diff))
report=dict(rr);report.update(baseline_W_half_cost=baseline,predicted_W_linear_delta=linear,predicted_W_quadratic_delta=quadratic,
 predicted_W_half_cost=direct,W_quadratic_expansion_roundoff=float(direct-(baseline+linear+quadratic)),
 exact_common_A_WJ_entityJ_zero_columns=null.tolist(),null_column_drift_count=int(np.count_nonzero(delta[null])),
 every_entity_audited_AFTER_final_repair=True,entity_norm_audit=norm,postrepair_all_original_unit_checks_passed=True,
 postrepair_W_coefficients_same_delta=True,repair_report=ref(R),transport_audit=Waudit,
 postrepair_finalization_source=ref(__file__),postrepair_finalization_core=ref(sys.modules['whole_W_qp_core'].__file__),
 postrepair_finalization_seconds=time.monotonic()-start)
out=R.with_name('final_report.json');assert not out.exists();out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(report=ref(out),
 baseline=baseline,linear=linear,quadratic=quadratic,predicted=direct,null_columns=len(null),all_gates=True),indent=2),flush=True)
