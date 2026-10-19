"""Adopt a complete same-theta full-ordinary initializer, without a physical step."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,shutil,sys,time
import numpy as np
N=182017

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(x,base=None):
 p=Path(x.get('path',x.get('file')))
 if not p.is_absolute():p=Path(base)/p
 assert sha(p)==x['sha256'],'Changed reference '+str(p)
 return p

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def asset(base,d):return np.fromfile(verify(d,base),d['dtype']).reshape(d['shape'])

def context(ip):
 z=read(ip);assert z['schema']=='rinkfit.same_theta_full_ordinary_capture_adoption_input.v1'
 cpfile=verify(z['parent_checkpoint']);cp=read(cpfile);body=dict(cp);assert body.pop('state_token')==identity(body)
 candidatefile=verify(z['candidate']);c=read(candidatefile);assert c['schema']=='rinkfit.complete_same_theta_capture_state_candidate.v1'
 assert c['source_checkpoint']==z['parent_checkpoint'] and c['model']==cp['model'] and c['control']==cp['control'],'Same exact parent/model/control required'
 frozen=read(verify(cp['frozen_run']));target=frozen['immutable_target']
 for k in ['packet','semantic']:assert c[k]==target[k];verify(c[k])
 model=read(verify(c['model']));assert len(model['blocks'])==1376
 initfile=verify(c['initializer']);ini=read(initfile);diagfile=verify(ini['all_source_diagnostic']);diag=read(diagfile)
 assert ini['source_checkpoint']==z['parent_checkpoint'] and ini['model']==c['model'] and ini['theta_IEEE_unchanged']
 assert ini['entity_profile_excluded_all4_aux_IEEE_unchanged'] and ini['old_candidate_IEEE_retained_per_source_in_numpy_and_native']
 assert diag['checkpoint']==z['parent_checkpoint'] and diag['model']==c['model']
 assert diag['all_source_counts']==dict(raw=N,active=182016,ordinary=179363,entity=1692,profile=961,only_exclusion=25537)
 assert diag['per_source_no_worse']['native_all_ordinary'] and diag['per_source_no_worse']['numpy_all_ordinary']
 assert diag['theta_model_IEEE_unchanged'] and diag['entity_profile_exclusion_all4_aux_IEEE_unchanged']
 assert diag['whole_W']['ledger']['total_rank']==214112 and diag['whole_W']['ledger']['partition_exactly_once']
 V=verify(c['native_values']);v=read(V);B=verify(diag['baseline_values']);bv=read(B)
 for doc in [v,bv]:
  for k in ['model','packet','semantic']:assert doc[k]==c[k]
  assert doc['raw_rows']==N and doc['active_rows']==182016 and doc['frames']==1248 and doc['raw_material_labels']==171
  assert doc['physical_blocks']==1376 and doc['profiles']==961 and doc['full_W_rank']==214112
  assert doc['all_active_strict_valid'] and doc['invalid_active_rows']==[] and doc['excluded_rows']==[25537]
  assert doc['maximum_query_h']<=1e-6 and doc['maximum_RS_h']<=1e-6 and doc['scene_max_g']<=0 and all(a<=0 for a in doc['domain_max_g'])
 S=verify(c['semantic']);sm=read(S);active=asset(S.parent,sm['active_source_mask']).astype(bool);kind=asset(S.parent,sm['arrays']['kind']);ordinary=active&(kind==0)
 assert np.flatnonzero(~active).tolist()==[25537] and ordinary.sum()==179363
 assert np.array_equal(asset(V.parent,v['arrays']['active_mask']).astype(bool),active)
 assert np.array_equal(asset(B.parent,bv['arrays']['active_mask']).astype(bool),active)
 assert np.all(asset(V.parent,v['arrays']['strict_valid'])[active])
 q=asset(V.parent,v['arrays']['q']);oldq=asset(B.parent,bv['arrays']['q']);assert np.all(q[ordinary]<=oldq[ordinary]);assert q[active].max()==v['budget_sq']<=bv['budget_sq']
 assert v['original_W_half_cost']<bv['original_W_half_cost'],'This selected initializer must improve original W; not a permanent model gate'
 arrays={};names={'station':'station.f64','branch':'path.u32','row':'initial_row.f64','alpha':'initial_alpha.f64'}
 for k,n in names.items():
  vk='path' if k=='branch' else k;d=c['row_state_files'][k];assert d['shape']==[N] and d==ini['row_state_files'][k]
  arrays[k]=asset(candidatefile.parent,d);original=asset(cpfile.parent,cp['row_state_files'][k]);assert arrays[k][~ordinary].tobytes()==original[~ordinary].tobytes()
  assert sha(verify(d,candidatefile.parent))==v['arrays'][vk]['sha256']
  assert sha(verify(cp['row_state_files'][k],cpfile.parent))==bv['arrays'][vk]['sha256']
  assert arrays[k].tobytes()==asset(V.parent,v['arrays'][vk]).tobytes()
 assert q[~ordinary].tobytes()==oldq[~ordinary].tobytes()
 assert c['b']==cp['b'] and c['physical_iteration_not_incremented'] and c['fresh_all_operators_required']
 return dict(z=z,parent=cp,parent_file=cpfile,candidate=c,candidate_file=candidatefile,initializer=ini,initializer_file=initfile,diagnostic=diag,diagnostic_file=diagfile,values=v,values_file=V,baseline=bv,baseline_file=B,arrays=arrays,active=active,kind=kind,q=q,frozen=frozen,target=target,ordinary=ordinary)

def adopt(ip,out):
 started=time.time();a=context(ip);p=a['parent'];c=a['candidate'];v=a['values'];V=a['values_file'];assert not out.exists();out.mkdir(parents=True);(out/'code').mkdir();seed=out/'native_seed';seed.mkdir()
 shutil.copyfile(verify(c['model']),seed/'model.json')
 names={'station':'station.f64','branch':'path.u32','row':'initial_row.f64','alpha':'initial_alpha.f64'}
 for k,n in names.items():shutil.copyfile(verify(c['row_state_files'][k],a['candidate_file'].parent),seed/n)
 pkfile=verify(c['packet']);packet=read(pkfile);E=asset(pkfile.parent,packet['arrays']['engineering_budget_px']);observed=asset(pkfile.parent,packet['arrays']['native_uv'])
 err=asset(V.parent,v['arrays']['error_uv']).reshape(N,2);coverage=E*np.sqrt(np.maximum(a['q'],0));active=a['active'];pred=dict(source_rows=asset(pkfile.parent,packet['arrays']['source_rows']).astype(np.int64),native_pts=asset(pkfile.parent,packet['arrays']['native_pts']),active_source_mask=active)
 assert np.array_equal(pred['source_rows'],np.arange(N)) and np.array_equal(pred['active_source_mask'],active)
 pred.update(stations=a['arrays']['station'],branches=a['arrays']['branch'],root_rows=a['arrays']['row'],profile_alpha=a['arrays']['alpha'],error_native_uv=err,coverage_error_px=coverage,uv=observed+err,RS_valid=asset(V.parent,v['arrays']['strict_valid']).astype(bool));np.savez_compressed(seed/'candidate_predictions.npz',**pred)
 metrics=dict(label='0',delta=[0.,0.],valid=True,active_RS_valid_rows=int(active.sum()),max_q=v['budget_sq'],maximum_error_px=float(coverage[active].max()),quantiles_px=np.quantile(coverage[active],[.5,.95,.99,1.]).tolist(),over2=int(np.count_nonzero(coverage[active]>E[active])),whole_original_W_half_cost=v['original_W_half_cost'],max_query_equality_px=v['maximum_query_h'],query_numerical_roots=961,profile_global_interval_certified=False,locally_captured_ordinary_candidates=179363,inactive_under_current_worst_bound_rows=0,maximum_source_row=v['maximum_source_row'],elapsed_seconds=a['initializer']['elapsed_seconds'])
 state=dict(schema='rinkfit.same_theta_full_ordinary_complete_native_seed.v1',best=metrics,whole_static_scene=p['physical_scene'],raw_rows=N,active_rows=182016,frames=1248,raw_material_labels=171,statistics_rank=214112,partition_exactly_once=True,source_W_E_unchanged=True,model_parent=c['model'],semantic=c['semantic'],initialization_only=True,final_physical_fit_accepted=False,exact_value_state=ref(V),same_theta_initializer=c['initializer'])
 state['files']={n:sha(seed/n) for n in ['model.json',*names.values(),'candidate_predictions.npz']};atomic(seed/'report.json',state)
 frozen=copy.deepcopy(a['frozen']);driver=next(verify(x['frozen']) for x in reversed(frozen['code']) if Path(x['frozen']['path']).name=='reduced_slp_driver.py');shutil.copyfile(driver,out/'code/reused_state_writer.py');shutil.copyfile(__file__,out/'code/adopt_capture.py')
 receipts=dict(schema='rinkfit.same_theta_capture_adoption_receipt.v1',input=ref(ip),parent_checkpoint=ref(a['parent_file']),candidate=ref(a['candidate_file']),initializer=ref(a['initializer_file']),native_values=ref(V),diagnostic=ref(a['diagnostic_file']),source=ref(out/'code/adopt_capture.py'),complete_native_seed=ref(seed/'report.json'),no_physical_step=True,physical_iteration_not_incremented=True,selected_main=bool(a['z']['select_main']),no_Solve=True,resource_ledger=[dict(report=ref(a['initializer_file']),seconds_field='elapsed_seconds',seconds=a['initializer']['elapsed_seconds'],role='same_theta_full_ordinary_capture_and_full_W_statistics_wall')])
 atomic(out/'operator_receipts.json',receipts);frozen['parent_frozen_run']=p['frozen_run'];frozen['same_theta_full_capture_adoption']=ref(out/'operator_receipts.json');frozen['code'].append(dict(original=ref(__file__),frozen=ref(out/'code/adopt_capture.py'),role='same_theta_capture_adopter'));frozen['target_id']=identity(dict(parent_target_id=p['target_id'],capture_operator=ref(out/'operator_receipts.json')));atomic(out/'frozen_run.json',frozen)
 control=copy.deepcopy(p['control']);control['compute_elapsed_seconds']+=a['initializer']['elapsed_seconds'];control.setdefault('auxiliary_refinement_history',[]).append(dict(iteration=control['iteration'],operator=ref(out/'operator_receipts.json'),physical_step=False,metrics_before=p['metrics'],metrics_after=metrics))
 namespace=dict(Path=Path,json=json,np=np,copy=copy,time=time,shutil=shutil,os=os,sys=sys,hashlib=hashlib,N=N,D=out,sha=sha,ref=ref,identity=identity,atomic=atomic,verify=verify,spec=a['target'],control=control,target_id=frozen['target_id'],frozen_path=out/'frozen_run.json',parent_cp=ref(a['parent_file']),current_cp=None,started=started,elapsed_base=float(p['cumulative_elapsed_seconds'])+a['initializer']['elapsed_seconds'])
 namespace['log']=lambda *args,**kw:None;chosen={'asset_array','same_linearization_state','current_chart_state','copy_chart_state','save_state'};tree=ast.parse(driver.read_text(encoding='utf-8'));func=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in chosen];assert {x.name for x in func}==chosen;sys.path.insert(0,frozen['python_frozen_dir']);exec(compile(ast.Module(body=func,type_ignores=[]),str(driver),'exec'),namespace)
 doc=namespace['save_state'](seed,'same_theta_all_ordinary_capture_new_auxiliaries_fresh_derivatives_pending',None,None,None);CP=verify(namespace['current_cp']);d=CP.parent
 for n in ['report.json',*state['files']]:shutil.copyfile(seed/n,d/n)
 doc.update(algorithm_id='same_theta_full_ordinary_finite_RS_capture_initialization_refinement',operator_receipts=ref(out/'operator_receipts.json'),native_value_report=ref(d/'report.json'),native_predictions=ref(d/'candidate_predictions.npz'),selected_main=bool(a['z']['select_main']),no_physical_step=True,physical_iteration_not_incremented=True,canonical_old_driver_resume_supported=False,next_execution_requires_algorithm_continue_from=True,source_W_E_query_and_physical_parameterization_unchanged=True,fresh_derivatives_required=True,PHR_multiplier_update_permitted=False,elapsed_accounting_complete_for_all_external_branches=False,elapsed_accounting='Parent known boundary plus frozen capture wall; historical incomplete accounting flag retained',full_ordinary_capture_is_not_original_groupW_joint_optimum=True)
 init=dict(schema='rinkfit.continuous_SLP_complete_seed.v1',model=doc['model'],semantic_manifest=c['semantic'],files={n:ref(d/n) for n in names.values()});atomic(d/'initializer_manifest.json',init);doc['complete_seed_initializer']=ref(d/'initializer_manifest.json');doc.pop('state_token',None);doc['state_token']=identity(doc);atomic(CP,doc);atomic(out/'current.json',ref(CP));atomic(out/'result.json',dict(schema='rinkfit.same_theta_full_capture_adopted.v1',status='COMPLETE_SAME_THETA_STATION_REFINEMENT_FRESH_OPERATORS_REQUIRED',current_complete_checkpoint=ref(CP),frozen_run=ref(out/'frozen_run.json'),control=control,cumulative_elapsed_seconds=doc['cumulative_elapsed_seconds'],selected_main=bool(a['z']['select_main']),no_physical_step=True,whole_fit_complete=False));print(json.dumps(ref(CP)));return doc

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output');p.add_argument('--check-context-only',action='store_true');a=p.parse_args();ip=Path(a.input).resolve()
 if a.check_context_only:context(ip);print('SAME_THETA_CAPTURE_CONTEXT_VALID')
 else:
  if not a.output:p.error('--output required')
  adopt(ip,Path(a.output).resolve())
