"""Normalize a finite-visible coupled-W result for actual nonlinear trial.
No numerical gate, budget or state is repaired by this adapter.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(r,base=None):
 p=Path(r.get('path',r.get('file','')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=r['sha256']:raise ValueError('Changed coupled-W authority '+str(p))
 return p.resolve()
def array(d,base):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])

def run(secondary_report,before_report,output):
 start=time.monotonic();P=Path(secondary_report).resolve();p=read(P);f=p.get('final')or{}
 if not p.get('proposed')or not f.get('proposed'):raise ValueError('Secondary has no strict full-source proposal')
 au=f['all_source_affine_audit']
 if au['original_A_beyond_IEEE']or not all(au[k]for k in ['strict_original_box_and_finite_station_cell','all_active_sensor_domains','all_source_affine_norms_within_b']):raise ValueError('Missing original full-source/finite/sensor/A/box gate')
 authority=p['authority'];CP=verify(authority['checkpoint']);cp=read(CP);T=verify(authority['transport']);t=read(T);Q=verify(authority['linearization']);q=read(Q);MP=verify(authority['model']);SM=verify(authority['semantic']);PK=verify(authority['packet']);sem=read(SM);pk=read(PK);F=verify(f['proposal']);z=dict(np.load(F));B=Path(before_report).resolve();before=read(B)
 for name,key in [('station.f64','station'),('path.u32','branch'),('initial_row.f64','row'),('initial_alpha.f64','alpha')]:
  if sha(B.parent/name)!=cp['row_state_files'][key]['sha256']:raise ValueError('Mixed before complete auxiliary '+key)
 if sha(B.parent/'model.json')!=sha(MP):raise ValueError('Mixed before physical model')
 before_cpp=float(before['best']['max_q'])
 if before_cpp!=float(cp['b']):raise ValueError('Before CPP and authority checkpoint must match exactly')
 active=array(sem['active_source_mask'],SM.parent).astype(bool);kind=array(sem['arrays']['kind'],SM.parent);E=array(pk['arrays']['engineering_budget_px'],PK.parent);error=array(t['arrays']['error_uv'],T.parent).reshape(182017,2);normal=array(sem['arrays']['profile_normal'],SM.parent);res=np.zeros((182017,2));ordinary=active&(kind!=2);profile=active&(kind==2);res[ordinary]=error[ordinary]/E[ordinary,None];res[profile,0]=np.einsum('ij,ij->i',normal[profile],error[profile])/E[profile];q0=np.einsum('ij,ij->i',res[active],res[active]);before_affine=float(q0.max());b=float(p['primary_working_b'])
 if not b<min(before_cpp,before_affine):raise ValueError('Actual Phase-I working budget must descend under both bound evaluators')
 total=z['station_delta_full']
 if total.shape!=(182017,)or not np.isfinite(total).all()or np.any(total[(~active)|(kind!=0)]!=0):raise ValueError('Full original-source station seed scope')
 if 'station_TR_application' in p:
  sr=p['station_TR_application'];C=verify(sr['curvature_manifest']);cd=read(C);radius=array(cd['arrays'][sr['array_key']],C.parent)
  if np.any(abs(total)>radius):raise ValueError('Final total seed+eta outside current numerical curvature trust')
 else:raise ValueError('New nonlinear trial requires explicit same-state station curvature trust')
 coeff=[float(f[k])for k in ['predicted_W_baseline','predicted_W_linear_coefficient','predicted_W_quadratic_coefficient']]
 out=dict(schema='rinkfit.finite_visible_coupled_original_W_working_proposal.v1',proposed=True,model=authority['model'],packet=authority['packet'],semantic=authority['semantic'],linearization=authority['linearization'],transport=authority['transport'],checkpoint=authority['checkpoint'],authority=authority,proposal=f['proposal'],original_secondary_report=ref(P),primary_report=p['primary_report'],before_value_report=ref(B),before_native_cpp_q=before_cpp,before_original_affine_zero_q=before_affine,before_values_arithmetic_not_forced_equal=True,linearized_epigraph=b,working_target_b=b,primary_epigraph_coordinate=p['primary_epigraph_coordinate'],exact_primary_epigraph_value=p['fixed_primary_epigraph_value'],canonical_primal_npz_is_squared_budget_coordinate=True,all_source_affine_audit=au,ordinary_source_count=int(np.count_nonzero(active&(kind==0))),entity_source_count=int(np.count_nonzero(active&(kind==1))),original_normal_profile_count=int(np.count_nonzero(profile)),raw_rows=182017,active_rows=182016,original_W_rank=214112,total_original_station_delta_descriptor=dict(proposal=f['proposal'],key='station_delta_full',shape=[182017],units='Original material station S'),station_TR_application=p['station_TR_application'],original_A=p['original_A'],original_rhs_bounds=p['original_rhs_bounds'],predicted_W_baseline=coeff[0],predicted_W_linear_coefficient=coeff[1],predicted_W_quadratic_coefficient=coeff[2],predicted_W_half_cost=f['predicted_W_half_cost'],fractional_W_origin_is_original_unshifted_rW=True,primary_working_target_declared_inexact=True,W_secondary_declared_inexact=True,numeric_solver_status_not_overridden=True,raw_solver_status=f['raw_solver_status'],original_W_E_and_all_source_obligations_unchanged=True,PHR_multiplier_update_permitted=False,physical_nonlinear_actual_line_required=True,not_PhaseII_station_optimality_or_true_KKT=True,final_fit_accepted=False,source=ref(__file__),operator_recipe_refs=p['operator_recipe_refs'],elapsed_seconds=time.monotonic()-start)
 O=Path(output).resolve()
 if O.exists():raise ValueError('Never replace a reported proposal')
 O.mkdir();(O/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(proposal=ref(O/'report.json'),working_b=b,before_cpp=before_cpp,before_affine=before_affine)),flush=True);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--secondary-report',required=True);p.add_argument('--before-value-report',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.secondary_report,a.before_value_report,a.output)
