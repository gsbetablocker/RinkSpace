"""Accepted-model source review; requests a recoverable batch stop on evidence.

This does not alter pixels, qualification, weights, or certify a physical model.
The original E is an engineering comparison scale, never a statistical sigma.
"""
from pathlib import Path
import hashlib,json,math
from .io_utils import read_json,write_json,file_sha256

def reference(path):
 p=Path(path).resolve();return {'file':str(p),'sha256':file_sha256(p)}

def decide_anchor_review(report, intermediate_policy="strict"):
 if intermediate_policy not in ("strict", "diagnostic"):
  raise ValueError("Unknown intermediate review policy")
 reasons=[]
 for sid,c in report['same_source_comparison']['by_shot'].items():
  if c['newly_invalid']:
   reasons.append({'shot_id':sid,'reason':'original_fixed_source_became_invalid','rows':c['newly_invalid']});continue
  b=c['baseline_true_RS'];v=c['candidate_true_RS']
  if not all(b.get(k)is not None and v.get(k)is not None for k in ('rms_px','p95_px')):continue
  witnesses=[r for r in report['worst_worsened_subject_by_entity']if r['shot_id']==sid and r['candidate_error_px']is not None and r['baseline_error_px']is not None and r['candidate_error_px']-r['baseline_error_px']>r['original_E']]
  if v['rms_px']>b['rms_px']and v['p95_px']>b['p95_px']and c['sum_squared_error_delta_on_common_valid']>0 and witnesses:
   reasons.append({'shot_id':sid,'reason':'fixed_cohort_RMS_and_p95_worsened_with_original_E_exceeding_source_witness','baseline_RMS_px':b['rms_px'],'candidate_RMS_px':v['rms_px'],'baseline_p95_px':b['p95_px'],'candidate_p95_px':v['p95_px'],'source_witnesses':witnesses})
 for entry in report['same_source_comparison'].get('by_shot_entity',[]):
  sid=entry['shot_id'];entity=entry['feature_id'];c=entry['comparison']
  if any(r.get('shot_id')==sid and r.get('scope')!='shot_entity' for r in reasons):continue
  if c['newly_invalid']:
   reasons.append(dict(scope='shot_entity',shot_id=sid,feature_id=entity,reason='original_fixed_entity_source_became_invalid',rows=c['newly_invalid']));continue
  b=c['baseline_true_RS'];v=c['candidate_true_RS']
  if not all(b.get(k)is not None and v.get(k)is not None for k in ('rms_px','p95_px')):continue
  witnesses=[r for r in report['worst_worsened_subject_by_entity']if r['shot_id']==sid and r['feature_id']==entity and r['candidate_error_px']is not None and r['baseline_error_px']is not None and r['candidate_error_px']-r['baseline_error_px']>r['original_E']]
  if v['rms_px']>b['rms_px']and v['p95_px']>b['p95_px']and c['sum_squared_error_delta_on_common_valid']>0 and witnesses:
   reasons.append(dict(scope='shot_entity',shot_id=sid,feature_id=entity,reason='entity_source_RMS_and_p95_worsened_with_original_E_exceeding_witness',baseline_RMS_px=b['rms_px'],candidate_RMS_px=v['rms_px'],baseline_p95_px=b['p95_px'],candidate_p95_px=v['p95_px'],source_witnesses=witnesses))
 return {'schema':'rinkfit.accepted_fixed_anchor_review_decision.v1','requires_source_or_geometry_review':bool(reasons),'request_stop':bool(reasons) and intermediate_policy=='strict','intermediate_policy':intermediate_policy,'final_quality_gate_unchanged':True,'reasons':reasons,'source_UV_or_qualification_modified':False,'quality_accepted':False,'original_E_is_engineering_scale_not_sigma':True,'additional_arbitrary_error_threshold_issued':False}

class AcceptedSourceReview:
 def __init__(self,output,packet,configuration):
  self.output=Path(output).resolve();self.packet=Path(packet).resolve();self.last_sha=None;self.records=[];self.stopped=False;self.failed=False
  cfg=read_json(configuration['file'])
  if file_sha256(configuration['file'])!=configuration['sha256']:raise ValueError('Changed accepted review configuration')
  if cfg.get('schema')!='rinkfit.dense6.accepted_source_anchor_review.v1':raise ValueError('Unknown accepted review schema')
  self.baseline=Path(cfg['baseline']['file']).resolve()
  if file_sha256(self.baseline)!=cfg['baseline']['sha256']:raise ValueError('Changed complete anchor baseline')
  self.config=configuration
  self.intermediate_policy=cfg.get("intermediate_policy", "strict")
  if self.intermediate_policy not in ("strict", "diagnostic"):raise ValueError("Unknown intermediate review policy")
 def poll(self):
  if self.stopped or self.failed:return None
  folder=self.output/'working_physical'
  try:
   rb=(folder/'receipt.json').read_bytes();r=json.loads(rb)
   if r.get('state_kind')!='physical_working_candidate'or r.get('complete_checkpoint')is not False:raise ValueError('Unexpected working physical state')
   if int(r['global_iteration'])<=0 or r['model_sha256']==self.last_sha:return None
   payload=(folder/'model.json').read_bytes()
   if rb!=(folder/'receipt.json').read_bytes()or hashlib.sha256(payload).hexdigest()!=r['model_sha256']:return None
  except FileNotFoundError:return None
  snapshot=json.loads(payload)
  if not all(math.isfinite(float(v)) for block in snapshot['blocks'] for v in block['values']):
   raise ValueError('Accepted physical snapshot contains nonfinite parameters')
  target=self.output/'accepted_source_review'/str(int(r['global_iteration']))
  target.mkdir(parents=True,exist_ok=False);model=target/'model.json';model.write_bytes(payload);(target/'receipt.json').write_bytes(rb)
  from .dense_fixed_anchor_guard import evaluate_fixed_anchors
  report=evaluate_fixed_anchors(model,self.packet,target/'anchors',self.baseline);decision=decide_anchor_review(report, self.intermediate_policy)
  from .fixed_local_anchor_audit import audit
  strata=audit(target/'anchors/report.json',target/'anchor_strata.json')
  decision['fixed_local_anchor_strata']=reference(target/'anchor_strata.json')
  decision['fixed_source_entity_does_not_mean_known_local_coordinate']=True
  known=strata['same_source_comparison'].get('known_local_ad_points')
  decision['known_local_anchor_regression_observed']=bool(known and (known['newly_invalid'] or known['sum_squared_error_delta_on_common_valid']>0))
  write_json(target/'decision.json',decision)
  if decision['request_stop']:
   try:
    with (self.output/'STOP_REQUESTED').open('x',encoding='utf8')as f:f.write('Actual full fixed-source regression; preserve complete candidate at batch boundary. See '+str(target/'decision.json')+'\n')
   except FileExistsError:pass
   self.stopped=True
  self.last_sha=r['model_sha256'];record={'global_iteration':r['global_iteration'],'model':reference(model),'guard':reference(target/'anchors/report.json'),'decision':reference(target/'decision.json'),'fixed_local_anchor_strata':reference(target/'anchor_strata.json'),'recoverable_batch_stop_requested':decision['request_stop']};self.records.append(record)
  write_json(self.output/'accepted_source_review/latest.json',{'configuration':self.config,'last_review':record,'reviews':self.records,'new_source_rows_or_RGB_re_read':False,'physical_quality_accepted':False})
  return record
 def fail_closed(self,error):
  self.failed=True
  target=self.output/'accepted_source_review';target.mkdir(parents=True,exist_ok=True)
  write_json(target/'error.json',{'schema':'rinkfit.accepted_source_review_failure.v1','error':repr(error),'quality_accepted':False,'source_UV_modified':False})
  try:
   with (self.output/'STOP_REQUESTED').open('x',encoding='utf8')as f:f.write('Accepted source review failed; preserve candidate, inspect accepted_source_review/error.json\n')
  except FileExistsError:pass
