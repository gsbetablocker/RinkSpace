"""Verify a recomputed native state before emitting a reviewable receipt.

A receipt binds the evaluation report's frozen output hashes, selected source
operator, proposal and its actual direct parent. It never grants fit acceptance.
"""
from pathlib import Path
import argparse,hashlib,json,numpy as np

AUX={'station.f64':('stations','<f8'),'path.u32':('branches','<u4'),'initial_row.f64':('root_rows','<f8'),'initial_alpha.f64':('profile_alpha','<f8')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def resolved(base,descriptor):
 p=Path(descriptor.get('file',descriptor.get('path','')));p=p if p.is_absolute()else Path(base)/p
 if not descriptor.get('sha256')or sha(p)!=descriptor['sha256']:raise ValueError('Referenced content SHA mismatch: '+str(p))
 return p

def require(condition,message):
 if not condition:raise ValueError(message)

def typed(base,d):return np.fromfile(resolved(base,d),d['dtype']).reshape(d['shape'])

def _validate_outputs(D,rep,pk,sem,PK,SP):
 D=Path(D);PK=Path(PK);SP=Path(SP);n=int(pk['observations'])
 from .state_integrity import validate_report_files
 validate_report_files(D,rep)
 raw={name:typed(PK.parent,d)for name,d in pk['arrays'].items()};raw['string_tables_ref']=ref(resolved(PK.parent,pk['string_tables']))
 require(raw['native_uv'].shape==(n,2),'Raw UV shape differs');require(raw['native_pts'].shape==(n,),'Raw PTS shape differs');require(np.array_equal(raw['source_rows'],np.arange(n)),'Raw source row order differs')
 active=typed(SP.parent,sem['active_source_mask']).astype(bool);kind=typed(SP.parent,sem['arrays']['kind']);normal=typed(SP.parent,sem['arrays']['profile_normal']);require(active.shape==kind.shape==(n,),'Semantic row shape differs');require(normal.shape==(n,2),'Profile normal shape differs')
 ordinary=int(np.count_nonzero(active&(kind==0)));profile=int(np.count_nonzero(active&(kind==2)));best=rep['best']
 require(rep.get('raw_rows')==n and rep.get('active_rows')==int(active.sum()),'Report source count differs')
 require(rep.get('source_W_E_unchanged')is True and rep.get('partition_exactly_once')is True,'Report source operator is not complete and unchanged')
 require(best.get('valid')is True and best.get('active_RS_valid_rows')==int(active.sum()),'Report active RS coverage differs')
 require(best.get('locally_captured_ordinary_candidates')==ordinary and best.get('inactive_under_current_worst_bound_rows')==0,'Not every ordinary station was actually recomputed')
 require(best.get('query_numerical_roots')==profile,'Profile root count differs')
 require(rep.get('whole_static_scene',{}).get('passed')is True,'Physical scene guard did not pass')
 required={'source_rows':(n,),'native_pts':(n,),'active_source_mask':(n,),'uv':(n,2),'stations':(n,),'branches':(n,),'root_rows':(n,),'profile_alpha':(n,),'error_native_uv':(n,2),'coverage_error_px':(n,),'RS_valid':(n,)}
 with np.load(D/'candidate_predictions.npz',allow_pickle=False)as z:
  for key,shape in required.items():require(key in z.files and z[key].shape==shape,'Saved prediction key/shape differs: '+key)
  require(np.array_equal(z['source_rows'],raw['source_rows']),'Saved source rows differ');require(np.array_equal(z['native_pts'],raw['native_pts']),'Saved PTS differ');require(np.array_equal(z['active_source_mask'],active),'Saved active scope differs')
  require(z['RS_valid'].dtype==np.dtype(bool)and bool(z['RS_valid'][active].all()),'Active true-RS validity is incomplete')
  for name,(key,dtype)in AUX.items():
   a=np.fromfile(D/name,dtype);require(a.shape==(n,),'Auxiliary length differs: '+name);require(z[key].dtype==np.dtype(dtype)and np.array_equal(a,z[key],equal_nan=True),'Auxiliary contents differ from saved predictions: '+name);require(np.isfinite(a[active]).all(),'Nonfinite active auxiliary: '+name)
  require(np.isfinite(z['uv'][active]).all(),'Nonfinite active projection');require(np.all((z['stations'][active]>=0)&(z['stations'][active]<=1)),'Station outside finite support')
  require(np.allclose(z['root_rows'][active],z['uv'][active,1],atol=2e-8,rtol=0),'Saved row root differs from projected v')
  error=z['uv']-raw['native_uv'];coverage=np.linalg.norm(error,axis=1);prof=kind==2;coverage[prof]=abs(np.einsum('ij,ij->i',normal[prof],error[prof]))
  require(np.allclose(error[active],z['error_native_uv'][active],atol=1e-12,rtol=0),'Saved original-UV residual differs');require(np.allclose(coverage[active],z['coverage_error_px'][active],atol=1e-12,rtol=0),'Saved engineering error differs')
  budget=raw['engineering_budget_px'];require(budget.shape==(n,)and np.isfinite(budget).all()and(budget>0).all(),'Engineering budget differs');q=(coverage/budget)**2
  values={'max_q':float(q[active].max()),'maximum_error_px':float(coverage[active].max())}
  for k,v in values.items():require(np.isclose(best.get(k,np.nan),v,atol=1e-10,rtol=1e-11),'Report metric differs from predictions: '+k)
  require(np.allclose(best.get('quantiles_px',[]),np.quantile(coverage[active],[.5,.95,.99,1]),atol=1e-10,rtol=1e-11),'Report quantiles differ');require(best.get('over2')==int(np.count_nonzero(active&(coverage>2))),'Report over2 count differs')
 return raw,active,kind,ordinary,profile

def _validate_lineage(D,proposal,parent,PK,SP,rep,pk):
 D=Path(D);PP=Path(proposal);CP=Path(parent);PK=Path(PK);SP=Path(SP);m=read(D/'model.json');pd=read(PP);cd=read(CP)
 require(isinstance(m.get('blocks'),list)and len(m['blocks'])>0 and isinstance(m.get('materials'),list)and len(m['materials'])>0 and isinstance(m.get('camera'),dict)and len(m['camera'])>0,'Complete physical model structure is absent')
 for b in m['blocks']:require('name'in b and np.isfinite(np.asarray(b.get('values',[]),float)).all(),'Invalid complete physical parameter block')
 require(str(pd.get('schema','')).startswith('rinkfit.')and 'model'in pd and 'parent_checkpoint'in pd and 'row_state_files'in pd,'Proposal is not a physical complete-state trial seed')
 require(str(cd.get('schema','')).startswith('rinkfit.complete_')and 'model'in cd and 'row_state_files'in cd,'Parent is not a complete checkpoint')
 direct=resolved(PP.parent,pd['parent_checkpoint']);require(sha(direct)==sha(CP),'Supplied parent is not the proposal direct parent')
 source=pk.get('immutable_source_hashes',{}).get('observations.npz');require(source is not None and m.get('source_identity')==source,'Model source identity differs from immutable packet')
 pm_path=resolved(PP.parent,pd['model']);pm=read(pm_path);cm_path=resolved(CP.parent,cd['model']);cm=read(cm_path)
 require(pm.get('source_identity')==source==cm.get('source_identity'),'Proposal/parent physical source identities differ')
 require(rep.get('semantic',{}).get('sha256')==sha(SP)and resolved(D,rep['semantic'])==SP.resolve(),'Report semantic authority differs')
 require(rep.get('packet',{}).get('sha256')==sha(PK)and resolved(D,rep['packet'])==PK.resolve(),'Report packet differs')
 source_model=resolved(D,rep.get('model_parent',{}));require(sha(source_model)==sha(pm_path),'Evaluated input model is not the proposal model')
 provenance=m.get('reduced_initializer_provenance',{});require(provenance.get('parent_sha256')==sha(pm_path)and provenance.get('semantic_sha256')==sha(SP),'Output model has a different evaluation parent')
 for key in ['blocks','camera','materials','source_identity']:require(m.get(key)==pm.get(key),'Complete evaluation unexpectedly changed physical '+key)
 for role,name in [('station','station.f64'),('branch','path.u32'),('row','initial_row.f64'),('alpha','initial_alpha.f64')]:
  require(role in pd['row_state_files']and(role in cd['row_state_files']or name in cd['row_state_files']),'Proposal/parent lacks complete seed auxiliary '+role)
  resolved(PP.parent,pd['row_state_files'][role]);resolved(CP.parent,cd['row_state_files'].get(role,cd['row_state_files'].get(name)))
 return m,pd,cd

def build_receipt(state,proposal,parent,packet,semantic,output=None,baseline=None):
 D=Path(state);PK=Path(packet).resolve();SP=Path(semantic).resolve();pk=read(PK);sem=read(SP);rep=read(D/'report.json');best=rep['best']
 from .state_integrity import validate_candidate_state
 verified=validate_candidate_state(D,PK,SP)
 raw,active,kind,ordinary,profile=_validate_outputs(D,rep,pk,sem,PK,SP);m,pd,cd=_validate_lineage(D,proposal,parent,PK,SP,rep,pk)
 # Reuse the independently maintained original full-W evaluator, including
 # validated source-audit marginal blocks; no fresh scalar weights are invented.
 from .regression import NativeRegression
 require('run'in rep,'Evaluation report lacks its source run identity');reg=NativeRegression(rep['run'],SP);require(np.array_equal(reg.uv,raw['native_uv'])and np.array_equal(reg.pts,raw['native_pts']),'Run and packet source arrays differ');check,_,_,_=reg.evaluate(D)
 require(np.isclose(check['whole_original_W_half_cost'],best['whole_original_W_half_cost'],rtol=1e-11,atol=1e-9),'Report original W differs from independent reconstruction');require(check['statistics_rank']==rep['statistics_rank'],'Report W rank differs')
 rawrefs={name:ref(resolved(PK.parent,d))for name,d in list(pk['arrays'].items())+[('string_tables',pk['string_tables'])]}
 manifest=dict(schema='rinkfit.complete_recomputed_native_candidate.v1',integrity_protocol='report_bound_complete_native_state.v2',model=ref(D/'model.json'),evaluation_report=ref(D/'report.json'),parent_complete_checkpoint=ref(parent),proposal=ref(proposal),packet=ref(PK),semantic=ref(SP),source_identity=m['source_identity'],raw_source_arrays_verified_exact=rawrefs,source_rows=len(active),active_rows=int(active.sum()),all_frames=int(pk['frame_count']),existing_excluded_rows=np.flatnonzero(~active).tolist(),engineering_budget_and_original_W_unchanged=True,source_operator_unchanged_relative_to_selected_authority=True,row_state_files={name:ref(D/name)for name in AUX},predictions=ref(D/'candidate_predictions.npz'),all_ordinary_stations_recomputed=ordinary,true_RS_valid_rows=best['active_RS_valid_rows'],profile_new_numerical_roots=profile,metrics=best,physical_guard=rep['whole_static_scene'],physical_fit_accepted=False,complete_state_for_fresh_relinearization=True,old_gradient_or_duals_not_reused=True,profile_global_optimality_certified=False,whole_shot_visual_validation_completed=False,independent_W_reconstruction=dict(half_cost=check['whole_original_W_half_cost'],rank=check['statistics_rank']))
 if baseline is not None:manifest['baseline_reference']=ref(baseline)
 if(D/'independent_numpy_replay.json').exists():
  replay=read(D/'independent_numpy_replay.json');require(replay.get('model_sha256')==sha(D/'model.json')and replay.get('all_source_predictions_pass')is True and replay.get('NumPy_active_RS_valid')==int(active.sum()),'Independent RS replay belongs to a different or invalid state');manifest['independent_NumPy_full_RS']=ref(D/'independent_numpy_replay.json')
 O=Path(output)if output else D/'complete_state_manifest.json'
 if O.exists():raise ValueError('Refuse overwriting a complete receipt')
 O.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');return O

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--proposal',required=True);p.add_argument('--parent',required=True);p.add_argument('--packet',required=True);p.add_argument('--semantic',required=True);p.add_argument('--output');p.add_argument('--baseline');a=p.parse_args(argv)
 print(build_receipt(a.state,a.proposal,a.parent,a.packet,a.semantic,a.output,a.baseline))
if __name__=='__main__':main()
