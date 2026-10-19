"""Full lawful original-S / all-source conditional original-W secondary.
Same frozen core assembly and numerical solver. Selection expands original latent
station columns only; all source SOCs present before Solve. No physical line.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy import sparse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(r):
 p=Path(r['path']).resolve()
 if sha(p)!=r['sha256']:raise ValueError('Changed frozen reference '+str(p))
 return p

def run(input_path,output,stop_file=None):
 start=time.monotonic();IP=Path(input_path).resolve();spec=read(IP)
 if spec.get('schema')!='rinkfit.finite_visible_original_W_secondary_input.v1':raise ValueError('Typed input schema')
 PR=verify(spec['primary_report']);pr=read(PR)
 if not(pr.get('ready_for_conditional_W') or (pr.get('final')or{}).get('ready_for_conditional_W')):raise ValueError('Primary has no strictly audited full-source working target')
 Q=verify(spec['linearization']);T=verify(spec['transport']);PP=verify(spec['primary_proposal']);SEED=verify(spec['station_seed']);SELECT=verify(spec['station_selection']);AP=verify(spec['original_A']);BP=verify(spec['rhs_bounds'])
 bundle=verify(spec['group_W_recipe']);bd=read(bundle);group_source=verify(spec['group_W_source']);norm_source=verify(spec['full_norm_source']);secondary_source=verify(spec['secondary_source'])
 if sha(secondary_source)!='557d4325ff2499d5e0c98d6956f590c6e0a169edf8b5fc4efcfdeffb9351b041':raise ValueError('Exact frozen conditional W math policy changed')
 for name,actual in [('linearization',Q),('qbar',Q),('transport',T),('proposal',PP),('station_seed',SEED),('A',AP),('rhs_bounds',BP)]:
  claim=pr.get(name)or(pr.get('final')or{}).get(name)
  if claim is not None and claim.get('sha256')!=sha(actual):raise ValueError('Primary report mismatch '+name)
 sys.path.insert(0,str(norm_source.parent));sys.path.insert(0,str(group_source.parent));sys.path.insert(0,str(secondary_source.parent))
 from group_W_station import assemble,save
 from source_affine_norm_constraints import load_all_source_affine_norms
 from finite_visible_W_secondary import solve
 D=Path(output).resolve();D.mkdir(exist_ok=False)
 tr=read(T);cp=verify(tr['checkpoint']);allrows=np.arange(int(tr['raw_rows']),dtype='<u8');ar=D/'all_requested_source_rows.u64';ar.write_bytes(allrows.tobytes())
 allsel=dict(schema='rinkfit.station_column_selection.v1',transport=ref(T),checkpoint=ref(cp),source_rows=dict(file=ar.name,dtype='<u8',shape=[len(allrows)],sha256=sha(ar)),scope='Every raw source requested; frozen original eligibility filter releases only lawful smooth ordinary S; no new physical DOF.')
 SEL=D/'full_lawful_station_selection.json';SEL.write_text(json.dumps(allsel,indent=2),encoding='utf8')
 chart=assemble(Q,T,selection=SEL,station_seed=SEED,station_trust=spec.get('station_trust'))
 curvature=verify(spec['station_TR_manifest']);claimed=pr.get('station_TR_manifest')or(pr.get('final')or{}).get('station_TR_manifest')
 if claimed is None or claimed['sha256']!=sha(curvature):raise ValueError('Primary must be recovered/re-solved under the identical total-S curvature trust')
 limiter=verify(spec['station_TR_limiter_source']);sys.path.insert(0,str(limiter.parent));from station_curvature_TR_chart import apply_station_curvature_TR
 chart,station_TR_application=apply_station_curvature_TR(chart,curvature,spec.get('station_TR_array_key','radius'))
 A=sparse.load_npz(AP).tocsr();bs=dict(np.load(BP));rhs=bs['rhs'];primary=dict(np.load(PP));delta=primary['step_scaled'];n=len(chart.free)
 if not np.array_equal(primary['parent_values'],chart.parent_values)or not np.array_equal(primary['scales'],chart.scale)or not np.array_equal(primary['free_ambient_indices'],chart.free):raise ValueError('Mixed primary physical state or scales')
 sizes=np.array([len(b['values'])for b in chart.model['blocks']]);offset=np.r_[0,np.cumsum(sizes)];norms=load_all_source_affine_norms(Q,T,offset,chart.scale,chart.free)
 packet=verify(chart.authority['packet']);pk=read(packet);ed=pk['arrays']['engineering_budget_px'];ep=Path(ed.get('path',ed.get('file')));ep=ep if ep.is_absolute()else packet.parent/ep
 if sha(ep)!=ed['sha256']:raise ValueError('Changed original engineering budget')
 E=np.fromfile(ep,ed['dtype']).reshape(ed['shape'])
 guard_source=verify(spec['print_det_guard_source']);sys.path.insert(0,str(guard_source.parent));from print_det_convex_guard import from_full_norm_operator
 guard=from_full_norm_operator(norms,chart.free,chart.scale)
 save(chart,D/'group_W_chart')
 report=solve(chart,norms,E,delta,float(spec['working_b']),A,rhs,float(spec['trust_scaled']),D/'secondary',seed_membership=norms.rows.tolist(),stop_file=stop_file,assembly_only=bool(spec.get('assembly_only',False)),primary_epigraph_coordinate=spec['primary_epigraph_coordinate'],primary_epigraph_value=spec.get('primary_epigraph_value'),scene_guard=guard)
 out=dict(report,all_source_SOC_constraints_present_before_Solve=True,full_source_active_cones=len(norms.rows),all_lawful_original_latent_S_columns_requested=True,expanded_station_selection=ref(SEL),original_limited_station_selection=ref(SELECT),full_S_scope_is_numerical_proposal_not_new_physical_DOF=True,proposed=bool(report.get('proposed',False)),station_TR_application=station_TR_application,input=ref(IP),primary_report=ref(PR),operator_recipe_refs=dict(group_W=ref(bundle),group_W_source=ref(group_source),full_norm_source=ref(norm_source),secondary_source=ref(secondary_source),entry=ref(__file__),station_TR_limiter_source=ref(limiter),station_TR_manifest=ref(curvature),print_det_guard_source=ref(guard_source),primary_report=ref(PR)),elapsed_seconds=time.monotonic()-start,original_A=ref(AP),original_rhs_bounds=ref(BP),linearization=ref(Q),transport=ref(T),model=chart.authority['model'],packet=chart.authority['packet'],semantic=chart.authority['semantic'],group_W_chart=ref(D/'group_W_chart/manifest.json'))
 if report.get('final'):
  f=report['final'];out.update(proposal=f['proposal'],linearized_epigraph=f['working_target_b'],predicted_W_baseline=f['predicted_W_baseline'],predicted_W_linear_coefficient=f['predicted_W_linear_coefficient'],predicted_W_quadratic_coefficient=f['predicted_W_quadratic_coefficient'],predicted_W_half_cost=f['predicted_W_half_cost'])
 (D/'report.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(dict(output=ref(D/'report.json'),proposed=out['proposed'],elapsed_seconds=out['elapsed_seconds'])),flush=True);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--stop-file');a=p.parse_args();r=run(a.input,a.output,a.stop_file);raise SystemExit(0 if r.get('proposed')or r.get('no_Solve')else 2)
