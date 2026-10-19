"""Evaluate or reuse an exact full-source native SOC-restoration trial; no optimizer."""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def verify(d):
 p=Path(d['path']).resolve()
 if sha(p)!=d['sha256']:raise ValueError('Changed immutable trial input '+str(p))
 return p

def run(ip,out):
 start=time.perf_counter();IP=Path(ip).resolve();s=read(IP)
 if s['schema']!='rinkfit.SOC_restoration_actual_trial_input.v1':raise ValueError('Typed SOC-restoration actual scope')
 R=verify(s['restoration_report']);r=read(R);PR=verify(s['reference_prepared']);pr=read(PR);PC=verify(s['candidate_prepared']);pc=read(PC)
 if not r['proposed_for_actual_nonlinear_trial']or r['reference_checkpoint']!=pr['checkpoint']or r['candidate_checkpoint']!=pc['checkpoint']:raise ValueError('No matching strict fresh-candidate correction')
 for k in ['linearization','transport','profile_domains']:
  if r['fresh_'+k]!=pc[k]:raise ValueError('Mixed fresh operator '+k)
  verify(pc[k])
 modelpath=verify(r['final']['model']);mm=read(modelpath);mp=copy.deepcopy(mm);proof=mp.pop('second_order_restoration_provenance');mp['reduced_physical_proposal']=dict(parent_physical_proposal=mp.get('reduced_physical_proposal'),fresh_candidate_SOC_restoration=proof,physical_values_not_changed_by_metadata_adapter=True)
 D=Path(out).resolve();D.mkdir(exist_ok=False);MP=D/'physical_model.json';MP.write_text(json.dumps(mp,indent=2),encoding='utf8');adapter=dict(schema='rinkfit.metadata_only_physical_proposal_normalization.v1',original_model=ref(modelpath),normalized_model=ref(MP),restoration_report=ref(R),all_physical_blocks_values_bounds_scales_camera_materials_bitexact=True,producer_metadata_uses_existing_reduced_physical_proposal_field=True,source=ref(__file__));(D/'model_metadata_adapter.json').write_text(json.dumps(adapter,indent=2),encoding='utf8')
 T=read(verify(pc['transport']));SM=verify(T['semantic']);base=Path(pc['base_directory']);EVALUATOR=verify(s['evaluator']);verify(s['evaluator_bundle'])
 execution=dict(source=ref(__file__),input=ref(IP),restoration_report=ref(R),candidate_prepared=ref(PC),reference_prepared=ref(PR),model=ref(MP),semantic=ref(SM),full_initial_arrays={n:ref(base/n)for n in ['station.f64','path.u32','initial_row.f64','initial_alpha.f64']})
 if s.get('reuse_native_report'):
  NATIVE=verify(s['reuse_native_report']);native=read(NATIVE);code=0;execution.update(no_new_evaluation=True,reused_native_report=ref(NATIVE),no_cold_command_claimed=True)
 else:
  cmd=[sys.executable,'-X','utf8',str(EVALUATOR),'--model',str(MP),'--initial-dir',str(base),'--semantic-manifest',str(SM),'--output',str(D/'inner')];env={k.upper():v for k,v in os.environ.items()};env.update(PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  with(D/'stdout.log').open('x',encoding='utf8')as f:code=subprocess.run(cmd,cwd=D,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  NATIVE=D/'inner/report.json';native=read(NATIVE);execution.update(command=cmd,returncode=code)
 if native['model_parent']['sha256']!=sha(MP):raise ValueError('Native values evaluated a different corrected physical model')
 snap=read(verify(native['compiled_camera_snapshot']));bindings=snap['bindings']
 if bindings['model']['sha256']!=sha(MP)or bindings['semantic']['sha256']!=sha(SM):raise ValueError('Mixed actual camera/source snapshot')
 for n,d in execution['full_initial_arrays'].items():
  if bindings['initial_state'][n]['sha256']!=d['sha256']:raise ValueError('Actual snapshot has different full row initialization '+n)
 for n,h in native['files'].items():
  if Path(n).name!=n or sha(NATIVE.parent/n)!=h:raise ValueError('Native output changed '+n)
 pred=np.load(NATIVE.parent/'candidate_predictions.npz',allow_pickle=False)
 if not np.array_equal(pred['source_rows'],np.arange(182017))or np.flatnonzero(~pred['active_source_mask']).tolist()!=[25537]or not np.all(pred['RS_valid'][pred['active_source_mask']]):raise ValueError('Full source coverage or strict RS failed')
 before=read(verify(pr['checkpoint']));b=native['best'];valid=code==0 and b['valid']and b['active_RS_valid_rows']==182016 and b['query_numerical_roots']==961 and b['max_query_equality_px']<=1e-6 and native['whole_static_scene']['passed']and native['statistics_rank']==214112 and native['partition_exactly_once']
 good=bool(valid and b['max_q']<before['b']);work=float(r['working_b']);result=dict(schema='rinkfit.SOC_restoration_actual_full_source_trial.v1',restoration_report=ref(R),reference_checkpoint=pr['checkpoint'],candidate_checkpoint=pc['checkpoint'],reference_prepared=ref(PR),candidate_prepared=ref(PC),actual_native_report=ref(NATIVE),actual_predictions=ref(NATIVE.parent/'candidate_predictions.npz'),model_metadata_adapter=ref(D/'model_metadata_adapter.json'),execution=execution,strict_whole_source_RS_Q_scene_passed=bool(valid),valid_PhaseI_candidate=good,selected_main=False,before_q=float(before['b']),working_b=work,actual_q=float(b['max_q']),actual_minus_working_b=float(b['max_q']-work),exact_working_budget_achieved=bool(b['max_q']<=work),actual_to_declared_budget_chord_ratio=float((before['b']-b['max_q'])/(before['b']-work)),metrics=b,original_total_box_TR_preserved=r['original_total_box_TR_not_relaxed'],source_W_E_and_parameterization_unchanged=True,ordinary_S_only_temporary_witness_and_actual_recovered_freely=True,PhaseII_original_W_optimum=False,PHR_multiplier_update_permitted=False,true_KKT_or_final_fit=False,elapsed_seconds=time.perf_counter()-start)
 (D/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf8');(D/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(dict(report=ref(D/'report.json'),valid_PhaseI_candidate=good,actual_q=b['max_q'],working_b=work)));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.output)
