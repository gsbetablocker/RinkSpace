"""Export a complete strictly lawful but explicitly unselected full native trial.
This is a fresh-linearization probe, regardless of whether q is above/below its
parent. No projection, Solve, acceptance rule, main selection or KKT is changed.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,shutil,sys,time
import numpy as np
N=182017

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(x,base=None):
 p=Path(x.get('path',x.get('file')))
 if not p.is_absolute():p=Path(base)/p
 if sha(p)!=x['sha256']:raise ValueError('Changed immutable reference '+str(p))
 return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');os.replace(t,p)
def asset(base,d):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])

def run(ip,out):
 started=time.time();ip=Path(ip).resolve();out=Path(out).resolve();z=read(ip)
 if z.get('schema')!='rinkfit.legal_unselected_native_full_probe_export_input.v1':raise ValueError('Explicit legal-unselected full probe scope required')
 if not isinstance(z.get('retained_for_correction_reason'),str)or not z['retained_for_correction_reason'].strip():raise ValueError('Explicit reason for an unselected correction probe required')
 C=verify(z['parent_checkpoint']);parent=read(C);body=dict(parent)
 if body.pop('state_token')!=identity(body):raise ValueError('Parent complete state token')
 MAIN=verify(z['current_main_checkpoint']);main_before=ref(MAIN)
 L=verify(z['actual_line_report']);line=read(L);F=verify(z['native_report']);state=read(F);seed=F.parent;P=verify(z['original_proposal']);proposal=read(P)
 if not(line['raw_rows']==N and line['active_rows']==182016 and line['source_W_E_unchanged']and line['physical_model_DOF_unchanged']):raise ValueError('Original complete source/model line scope')
 if line['proposal']!=z['original_proposal']or proposal['checkpoint']!=z['parent_checkpoint']or not proposal['proposed']:raise ValueError('Original proposal/line/parent mismatch')
 trial=line['history'][int(z['trial_index'])]
 if trial['fraction']!=1. or trial['report']!=z['native_report']or trial['returncode']!=0:raise ValueError('Only the exact completed full physical trial can be a full probe')
 best=state['best']
 if trial['best']!=best or not trial['strict_source_scene_query_numerical_gates_passed']:raise ValueError('The original strict native trial gate did not pass')
 if not(best['valid']and best['active_RS_valid_rows']==182016 and best['query_numerical_roots']==961 and np.isfinite(best['max_query_equality_px'])and best['max_query_equality_px']<=1e-6):raise ValueError('Original RS and finite-Q native gates')
 if not(state['raw_rows']==N and state['active_rows']==182016 and state['frames']==1248 and state['raw_material_labels']==171 and state['statistics_rank']==214112 and state['partition_exactly_once']and state['source_W_E_unchanged']):raise ValueError('Original full-source W/coverage ledger')
 if not(state['whole_static_scene']['passed']and state['whole_static_scene']['all_material_paths_checked']==257):raise ValueError('Original whole static finite-material scene gate')
 if line['semantic']!=state['semantic']or state['semantic']!=proposal['semantic']:raise ValueError('Source semantic identity changed')
 for n,d in state['files'].items():
  if Path(n).name!=n or sha(seed/n)!=d:raise ValueError('Native output file changed '+n)
 model=read(seed/'model.json');oldmodel=read(verify(parent['model']))
 if len(model['blocks'])!=1376 or len(model['blocks'])!=len(oldmodel['blocks'])or model['materials']!=oldmodel['materials']or model['camera']!=oldmodel['camera']:raise ValueError('Physical model/camera/finite material template changed')
 for b,a in zip(model['blocks'],oldmodel['blocks']):
  if {k:v for k,v in b.items()if k!='values'}!={k:v for k,v in a.items()if k!='values'}:raise ValueError('Original block/bounds/scales/gauge changed')
 raw=np.load(verify(proposal['proposal']),allow_pickle=False);old=np.concatenate([b['values']for b in oldmodel['blocks']]);current=np.concatenate([b['values']for b in model['blocks']]);scales=np.concatenate([b['scales']for b in oldmodel['blocks']])
 if not np.array_equal(raw['parent_values'],old)or not np.array_equal(raw['scales'],scales)or not np.array_equal(old+raw['physical_delta'],current):raise ValueError('Full physical trial differs from exact proposal/parent values')
 MP=verify(state['model_parent'])
 if sha(MP)!=trial['physical_model_sha256']:raise ValueError('Physical input model differs from original full line')
 pm=read(MP)
 if pm['camera']!=model['camera']or pm['materials']!=model['materials']or pm['blocks']!=model['blocks']:raise ValueError('Native state differs from physical trial model')
 predicted=np.load(seed/'candidate_predictions.npz',allow_pickle=False)
 if not np.array_equal(predicted['source_rows'],np.arange(N))or np.flatnonzero(~predicted['active_source_mask']).tolist()!=[25537]or not np.all(predicted['RS_valid'][predicted['active_source_mask']]):raise ValueError('Complete native source ledger or strict RS failed')
 for key,n,dt in [('stations','station.f64','<f8'),('branches','path.u32','<u4'),('root_rows','initial_row.f64','<f8'),('profile_alpha','initial_alpha.f64','<f8')]:
  values=np.fromfile(seed/n,dt)
  if values.shape!=(N,)or predicted[key].tobytes()!=values.tobytes():raise ValueError('Native predictions and full auxiliary bytes disagree '+n)
 B=float(proposal['working_target_b']);before=float(parent['b']);before_affine=float(proposal['before_original_affine_zero_q']);before_native=float(proposal['before_native_cpp_q']);actual=float(best['max_q'])
 if not np.isfinite([B,before,before_affine,before_native,actual]).all()or min(B,before,before_affine,before_native,actual)<0:raise ValueError('Finite original budget values required')
 if float(proposal['linearized_epigraph'])!=B:raise ValueError('Working target must use explicit current squared budget coordinate')
 before_report=verify(proposal['before_value_report']);baseline=read(before_report)
 if line['baseline']['sha256']!=sha(before_report)or float(line['baseline']['max_q'])!=before_native or float(baseline['best']['max_q'])!=before_native:raise ValueError('Recorded same-evaluator beforeq lineage differs')
 frozen=read(verify(parent['frozen_run']));base_spec=copy.deepcopy(frozen['immutable_target'])
 for key in ['packet','semantic']:
  if proposal[key]!=base_spec[key]:raise ValueError('Original target '+key+' changed')
 driver=next(verify(c['frozen'])for c in reversed(frozen['code'])if Path(c['frozen']['path']).name=='reduced_slp_driver.py')
 if out.exists():raise FileExistsError('A probe output may not overwrite any previous state')
 out.mkdir(parents=True);(out/'code').mkdir();shutil.copyfile(driver,out/'code/reused_state_writer.py');shutil.copyfile(__file__,out/'code/export_unselected_full_probe.py')
 predicted_drop=before_native-B;ratio=(before_native-actual)/predicted_drop if predicted_drop>0 else None
 receipt=dict(schema='rinkfit.legal_unselected_complete_native_full_probe_lineage.v1',input=ref(ip),reference_checkpoint=ref(C),current_main_not_modified=main_before,native_report=ref(F),line_report=ref(L),original_proposal=ref(P),before_value_report=ref(before_report),trial=trial,source=ref(out/'code/export_unselected_full_probe.py'),selected_main=False,accepted_physical_iteration_not_incremented=True,native_physical_trial_legal=True,recorded_line_historical_candidate_flag=bool(trial.get('accepted_as_PhaseI_initialization_candidate',False)),export_is_not_adoption=True,actual_q=actual,original_parent_b=before,original_same_evaluator_before_q=before_native,original_affine_zero_q=before_affine,actual_minus_parent_b=actual-before,actual_minus_same_evaluator_before_q=actual-before_native,original_working_b=B,actual_minus_working_b=actual-B,actual_over_working_prediction_ratio=ratio,retained_for_correction_reason=z['retained_for_correction_reason'],no_budget_acceptance_comparison_used_for_probe_eligibility=True,no_projection=True,no_Solve=True,no_acceptance_gate_or_TR_changed=True)
 atomic(out/'probe_lineage.json',receipt);frozen['parent_frozen_run']=parent['frozen_run'];frozen['unaccepted_native_probe_lineage']=ref(out/'probe_lineage.json');frozen['target_id']=identity(dict(parent_target_id=parent['target_id'],legal_unselected_full_probe=ref(out/'probe_lineage.json')));frozen['code'].append(dict(original=ref(__file__),frozen=ref(out/'code/export_unselected_full_probe.py'),role='legal_unselected_native_full_probe_export'));atomic(out/'frozen_run.json',frozen)
 control=copy.deepcopy(parent['control']);namespace=dict(Path=Path,json=json,np=np,copy=copy,time=time,shutil=shutil,os=os,sys=sys,hashlib=hashlib,N=N,D=out,sha=sha,ref=ref,identity=identity,atomic=atomic,verify=verify,spec=base_spec,control=control,target_id=frozen['target_id'],frozen_path=out/'frozen_run.json',parent_cp=ref(C),current_cp=None,started=started,elapsed_base=float(parent['cumulative_elapsed_seconds']));namespace['log']=lambda *a,**kw:None
 names={'asset_array','same_linearization_state','current_chart_state','copy_chart_state','save_state'};tree=ast.parse(driver.read_text(encoding='utf8'));func=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
 if {n.name for n in func}!=names:raise ValueError('Original five complete-state writers missing')
 sys.path.insert(0,frozen['python_frozen_dir']);exec(compile(ast.Module(body=func,type_ignores=[]),str(driver),'exec'),namespace)
 doc=namespace['save_state'](seed,'legal_but_unselected_full_native_probe_for_fresh_SOC_operators',None,None,None);cp=verify(namespace['current_cp']);cpdir=cp.parent
 for n in ['report.json',*state['files']]:shutil.copyfile(seed/n,cpdir/n)
 for n,d in state['files'].items():
  if sha(cpdir/n)!=d:raise ValueError('Complete copied native state changed '+n)
 if doc['control']['iteration']!=parent['control']['iteration']or doc['control']['trust']!=parent['control']['trust']:raise ValueError('Unselected probe must retain parent iteration/TR')
 doc.update(algorithm_id='legal_unselected_full_native_probe_for_same_objective_fresh_SOC_restoration',selected_main=False,accepted_physical_step=False,unaccepted_probe=True,native_physical_trial_legal=True,probe_lineage=ref(out/'probe_lineage.json'),native_value_report=ref(cpdir/'report.json'),native_predictions=ref(cpdir/'candidate_predictions.npz'),canonical_old_driver_resume_supported=False,next_execution_requires_algorithm_continue_from=True,source_W_E_query_and_physical_parameterization_unchanged=True,fresh_derivatives_required=True,PHR_multiplier_update_permitted=False,elapsed_accounting_complete_for_all_external_branches=False,actual_minus_parent_b=actual-before,actual_minus_same_evaluator_before_q=actual-before_native,original_working_b=B,actual_minus_working_b=actual-B,retained_for_correction_reason=z['retained_for_correction_reason'],scope='Strictly lawful full native trial retained only as an explicit unselected fresh-linearization origin, independent of whether q improves parent; original current main, acceptance gates, physical iteration and TR unchanged; no optimum, PhaseII, KKT or final fit claim')
 init=dict(schema='rinkfit.continuous_SLP_complete_seed.v1',model=doc['model'],semantic_manifest=base_spec['semantic'],files={n:ref(cpdir/n)for n in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']});atomic(cpdir/'initializer_manifest.json',init);doc['complete_seed_initializer']=ref(cpdir/'initializer_manifest.json');doc.pop('state_token',None);doc['state_token']=identity(doc);atomic(cp,doc);atomic(out/'current.json',ref(cp))
 if ref(MAIN)!=main_before:raise ValueError('Current main checkpoint changed during export')
 atomic(out/'result.json',dict(schema='rinkfit.legal_unselected_complete_native_full_probe_export.v1',current_complete_checkpoint=ref(cp),selected_main=False,no_projection=True,no_Solve=True,accepted_physical_iteration_unchanged=control['iteration'],whole_fit_complete=False,control=control,cumulative_elapsed_seconds=doc['cumulative_elapsed_seconds'],frozen_run=ref(out/'frozen_run.json'),probe_lineage=ref(out/'probe_lineage.json'),current_main_unchanged=main_before));print(json.dumps(ref(cp)));return ref(cp)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.input).resolve(),Path(a.output).resolve())
