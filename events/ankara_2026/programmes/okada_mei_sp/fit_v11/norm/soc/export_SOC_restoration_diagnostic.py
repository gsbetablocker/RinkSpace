"""Complete unselected Phase-I SOC-restoration candidate using the original five AST state writers."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,shutil,sys,time
import numpy as np
N=182017

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(x,base=None):
 p=Path(x.get('path',x.get('file')));p=p if p.is_absolute()else Path(base)/p
 if sha(p)!=x['sha256']:raise ValueError('Changed immutable restoration reference '+str(p))
 return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');os.replace(q,p)
def asset(base,d):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])

def run(ip,out):
 started=time.time();z=read(ip)
 if z['schema']!='rinkfit.SOC_restoration_unselected_diagnostic_export_input.v1':raise ValueError('Explicit unselected restoration scope')
 paths={k:verify(z[k])for k in ['reference_checkpoint','unaccepted_probe_checkpoint','fresh_candidate_prepared','restoration_report','actual_native_report','model_metadata_adapter','original_proposal','original_actual_line']}
 C=paths['reference_checkpoint'];parent=read(C);body=dict(parent)
 if body.pop('state_token')!=identity(body):raise ValueError('Parent state token')
 rest=read(paths['restoration_report']);context=read(verify(rest['context']));prep=read(paths['fresh_candidate_prepared']);probe=read(paths['unaccepted_probe_checkpoint']);state=read(paths['actual_native_report']);seed=paths['actual_native_report'].parent;adapter=read(paths['model_metadata_adapter']);pr=read(paths['original_proposal'])
 if rest['schema']!='rinkfit.fresh_candidate_total_step_SOC_restoration_proposal.v1'or not rest['proposed_for_actual_nonlinear_trial']:raise ValueError('No strict restoration proposal')
 if rest['reference_checkpoint']!=z['reference_checkpoint']or rest['candidate_checkpoint']!=prep['checkpoint']:raise ValueError('Mixed restoration authorities')
 if prep['parent_complete_checkpoint']!=z['unaccepted_probe_checkpoint']or not probe['unaccepted_probe']or probe['selected_main']:raise ValueError('Not the explicit unaccepted candidate origin')
 for key in ['linearization','transport','profile_domains']:
  if rest['fresh_'+key]!=prep[key]:raise ValueError('Fresh derivative mismatch '+key)
  verify(prep[key])
 if context['original_proposal']!=z['original_proposal']:raise ValueError('Mixed original proposal')
 if adapter['restoration_report']!=z['restoration_report']or not adapter['all_physical_blocks_values_bounds_scales_camera_materials_bitexact']:raise ValueError('Metadata-only model adapter')
 verify(adapter['original_model']);verify(adapter['normalized_model'])
 if state['model_parent']['sha256']!=adapter['normalized_model']['sha256']:raise ValueError('Native replay used a different corrected physical model')
 best=state['best'];valid=best['valid']and best['active_RS_valid_rows']==182016 and best['query_numerical_roots']==961 and best['max_query_equality_px']<=1e-6 and state['whole_static_scene']['passed']and state['whole_static_scene']['all_material_paths_checked']==257
 if not valid or not best['max_q']<parent['b']:raise ValueError('Not a legal true whole-source Phase-I descending candidate')
 for name,digest in state['files'].items():
  if Path(name).name!=name or sha(seed/name)!=digest:raise ValueError('Native file changed '+name)
 predicted=np.load(seed/'candidate_predictions.npz',allow_pickle=False)
 if not np.array_equal(predicted['source_rows'],np.arange(N))or np.flatnonzero(~predicted['active_source_mask']).tolist()!=[25537]or not np.all(predicted['RS_valid'][predicted['active_source_mask']]):raise ValueError('Raw/active source coverage')
 for k,n,dt in [('stations','station.f64','<f8'),('branches','path.u32','<u4'),('root_rows','initial_row.f64','<f8'),('profile_alpha','initial_alpha.f64','<f8')]:
  if predicted[k].tobytes()!=np.fromfile(seed/n,dt).tobytes():raise ValueError('Predictions and full row state disagree '+n)
 model=read(seed/'model.json');old=read(verify(parent['model']));a=np.concatenate([b['values']for b in model['blocks']]);oldv=np.concatenate([b['values']for b in old['blocks']]);q=dict(np.load(verify(rest['final']['proposal'])));sc=q['scales'];free=q['free_ambient_indices'];total=(a[free]-oldv[free])/sc[free];TR=float(parent['control']['trust'])
 if model['materials']!=old['materials']or model['camera']!=old['camera']or len(model['blocks'])!=1376:raise ValueError('Physical parameterization changed')
 for x,y in zip(model['blocks'],old['blocks']):
  if {k:v for k,v in x.items()if k!='values'}!={k:v for k,v in y.items()if k!='values'}:raise ValueError('Original physical bounds/scales/gauge changed')
 if not np.array_equal(a,q['candidate_values']+q['correction_physical'])or not np.array_equal(a-oldv,q['total_physical_delta']):raise ValueError('Actual physical theta does not match audited restoration')
 lo=np.concatenate([b['lower']for b in model['blocks']]);hi=np.concatenate([b['upper']for b in model['blocks']])
 if np.any(a<lo)or np.any(a>hi)or np.any(abs(total)>TR)or np.any(a[np.setdiff1d(np.arange(len(a)),free)]!=oldv[np.setdiff1d(np.arange(len(a)),free)]):raise ValueError('Original total physical box/TR/gauge violated')
 assert not out.exists();out.mkdir(parents=True);(out/'code').mkdir();frozen=read(verify(parent['frozen_run']));base_spec=copy.deepcopy(frozen['immutable_target']);driver=next(verify(c['frozen'])for c in reversed(frozen['code'])if Path(c['frozen']['path']).name=='reduced_slp_driver.py');shutil.copyfile(driver,out/'code/reused_state_writer.py');shutil.copyfile(__file__,out/'code/export_SOC_restoration.py')
 working=float(rest['working_b']);ratio=(parent['b']-best['max_q'])/(parent['b']-working);lineage=dict(schema='rinkfit.SOC_restoration_complete_candidate_lineage.v1',input=ref(ip),reference_checkpoint=ref(C),unaccepted_probe=z['unaccepted_probe_checkpoint'],fresh_candidate_prepared=z['fresh_candidate_prepared'],restoration_report=z['restoration_report'],actual_native_report=z['actual_native_report'],original_proposal=z['original_proposal'],original_actual_line=z['original_actual_line'],model_metadata_adapter=z['model_metadata_adapter'],selected_main=False,valid_PhaseI_candidate=True,physical_branch_iteration=int(parent['control']['iteration'])+1,original_total_TR=TR,correction_scaled_inf=rest['final']['correction_scaled_inf'],working_b=working,actual_b=float(best['max_q']),actual_minus_working_b=float(best['max_q']-working),exact_working_b_attained=False,actual_to_declared_budget_chord_ratio=ratio,declared_solver_status_not_changed=True,source=ref(out/'code/export_SOC_restoration.py'),source_W_E_and_physical_model_unchanged=True,no_main_checkpoint_overwritten=True,no_Solve=True)
 atomic(out/'restoration_lineage.json',lineage);frozen['parent_frozen_run']=parent['frozen_run'];frozen['SOC_restoration_lineage']=ref(out/'restoration_lineage.json');frozen['target_id']=identity(dict(parent_target_id=parent['target_id'],SOC_restoration=ref(out/'restoration_lineage.json')));frozen['code'].append(dict(original=ref(__file__),frozen=ref(out/'code/export_SOC_restoration.py'),role='unselected_SOC_restoration_diagnostic_export'));atomic(out/'frozen_run.json',frozen)
 control=copy.deepcopy(parent['control']);control['iteration']+=1;control['trust']=TR;control['history'].append(dict(iteration=control['iteration'],fraction=1.,ratio=ratio,parent_checkpoint=ref(C),restoration_lineage=ref(out/'restoration_lineage.json'),selected_main=False,actual_full_native_budget=float(best['max_q']),working_b=working,exact_same_working_b_claimed=False))
 namespace=dict(Path=Path,json=json,np=np,copy=copy,time=time,shutil=shutil,os=os,sys=sys,hashlib=hashlib,N=N,D=out,sha=sha,ref=ref,identity=identity,atomic=atomic,verify=verify,spec=base_spec,control=control,target_id=frozen['target_id'],frozen_path=out/'frozen_run.json',parent_cp=ref(C),current_cp=None,started=started,elapsed_base=float(parent['cumulative_elapsed_seconds']));namespace['log']=lambda*a,**kw:None
 names={'asset_array','same_linearization_state','current_chart_state','copy_chart_state','save_state'};func=[n for n in ast.parse(driver.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)and n.name in names]
 if {n.name for n in func}!=names:raise ValueError('Original five writer functions absent')
 sys.path.insert(0,frozen['python_frozen_dir']);exec(compile(ast.Module(body=func,type_ignores=[]),str(driver),'exec'),namespace);doc=namespace['save_state'](seed,'valid_unselected_SOC_restoration_PhaseI_diagnostic',None,None,None);cp=verify(namespace['current_cp']);dest=cp.parent
 for n in ['report.json',*state['files']]:shutil.copyfile(seed/n,dest/n)
 for n,digest in state['files'].items():
  if sha(dest/n)!=digest:raise ValueError('Checkpoint copy differs '+n)
 doc.update(algorithm_id='fresh_candidate_total_step_SOC_restoration_unselected_PhaseI_diagnostic',selected_main=False,valid_PhaseI_candidate=True,accepted_physical_step_on_diagnostic_branch=True,unaccepted_probe=False,restoration_lineage=ref(out/'restoration_lineage.json'),native_value_report=ref(dest/'report.json'),native_predictions=ref(dest/'candidate_predictions.npz'),external_accepted_native_report=ref(dest/'report.json'),external_accepted_predictions=ref(dest/'candidate_predictions.npz'),canonical_old_driver_resume_supported=False,next_execution_requires_algorithm_continue_from=True,source_W_E_query_and_physical_parameterization_unchanged=True,fresh_derivatives_required=True,PHR_multiplier_update_permitted=False,elapsed_accounting_complete_for_all_external_branches=False,working_target_b=working,actual_budget_exceeds_working_target=float(best['max_q']-working),working_target_exactly_attained=False,numerical_optimality_or_final_KKT_certified=False,scope='Valid unselected budget-descent candidate only. Main i20 unchanged. W/median/P95/over2 worsened; no overall fit or exact working-budget success claimed.')
 init=dict(schema='rinkfit.continuous_SLP_complete_seed.v1',model=doc['model'],semantic_manifest=base_spec['semantic'],files={n:ref(dest/n)for n in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']});atomic(dest/'initializer_manifest.json',init);doc['complete_seed_initializer']=ref(dest/'initializer_manifest.json');doc.pop('state_token',None);doc['state_token']=identity(doc);atomic(cp,doc);atomic(out/'current.json',ref(cp));atomic(out/'result.json',dict(schema='rinkfit.complete_SOC_restoration_diagnostic_export.v1',current_complete_checkpoint=ref(cp),selected_main=False,valid_PhaseI_candidate=True,no_Solve=True,control=control,cumulative_elapsed_seconds=doc['cumulative_elapsed_seconds'],frozen_run=ref(out/'frozen_run.json'),whole_fit_complete=False));print(json.dumps(dict(checkpoint=ref(cp),actual=best,ratio=ratio)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.input).resolve(),Path(a.output).resolve())
