"""Exact frozen native R-inner material reader; station remains unknown."""
from pathlib import Path
from collections import OrderedDict
import numpy as np
from .io_utils import read_json,write_json,object_sha256
from .packet_bound_source_review import reference
from . import source_recovery_review as legacy
from . import r_inner_prefix_v1 as source
VERSION='exact_frozen_R_inner_source_review_v1'
PREFIX=source.PREFIX
_CONTEXT={}
def _policy(cfg):return cfg.get('R_inner_source_policy')
def _dependencies(cfg):
 code=[reference(__file__),reference(source.__file__),reference(source.SOURCE/'source_operator.py'),reference(source.SOURCE/'qualify.py')]
 return dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(_policy(cfg))),code

def _context(cfg):
 qref=cfg['R_inner_source_qualification'];key=object_sha256(qref)
 if key in _CONTEXT:return _CONTEXT[key]
 if qref['sha256']!=source.FROZEN['strict_source_qualification.json']:raise ValueError('Different frozen R source qualification')
 q=source.source_proofs();c=dict(rows={},proofs={},frames={},dependencies=[qref,reference(source.SOURCE/'generation/manifest.json')])
 for p in q['actual_source_proofs']:
  f=source.checked(p['source_packet']);frame=c['frames'].get(p['native_pts'])
  if frame is None:frame=read_json(f);c['frames'][p['native_pts']]=frame;c['dependencies'].append(p['source_packet'])
  r=next(x for x in frame['qualified_local_material_rows']if x['source_row_id']==p['source_row_id'])
  if r['current_uv']!=p['current_uv']or r['native_pts']!=p['native_pts']or r['joint_material_index']!=73:raise ValueError('R source identity/proof differs')
  c['rows'][p['source_row_id']]=r;c['proofs'][p['source_row_id']]=p
  for ref in [frame['current_mask'],dict(path=r['native_png'],sha256=r['native_png_sha256'])]:source.checked(ref);c['dependencies'].append(ref)
 c['operator']=source.source_module('source_operator');_CONTEXT[key]=c;return c

def _inputs(s,packet,arrays,index,cfg):
 c=_context(cfg);rid=packet['source_row_ids'][index];r=c['rows'].get(rid);p=c['proofs'].get(rid)
 if r is None:raise ValueError('No exact R source row')
 if (int(s['pts'])!=r['native_pts']or s['time_base']!='1/12800'or s['video_sha256']!=r['source_video_sha256']or arrays['uv'][index].tolist()!=r['current_uv']or int(arrays['native_pts'][index])!=r['native_pts']or int(arrays['material_index'][index])!=73 or int(arrays['kind'][index])!=0):raise ValueError('Current R packet UV/ID/PTS/material differs')
 return dict(schema='rinkfit.exact_frozen_R_inner_feedback_inputs.v1',source=s,observation=legacy.observation_content(packet,arrays,index),source_row=r,source_proof=p,qualification=cfg['R_inner_source_qualification'],frame_inputs=cfg.get('frames',{}).get(str(r['native_pts']),{}),programme_scope=cfg['programme_scope'],frame_roles=cfg['frame_roles'],reader_policy=_policy(cfg))

def _cache(s,packet,index,code):
 from rinkspace.storage import AssetStore
 k=object_sha256(dict(source=s,dataset=packet['source_dataset_ids'][index],row_id=packet['source_row_ids'][index]));return AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/review_cache'/('ri_'+code[:16])/(k+'.json')
def cached_dependencies(s,packet,arrays,index,*,review_config=None):
 if review_config is None:return None
 try:
  cfg=read_json(legacy.checked(review_config)if isinstance(review_config,dict)else review_config);base,_=_dependencies(cfg);p=_cache(s,packet,index,base['reader_code_sha256'])
  if not p.is_file():return None
  cache=read_json(p);inputs=_inputs(s,packet,arrays,index,cfg);deps=cache['review_dependencies']
  if any(deps.get(k)!=v for k,v in base.items())or deps.get('actual_source_inputs_sha256')!=object_sha256(inputs):return None
  for r in cache['evidence_refs']:legacy.checked(r)
  if read_json(legacy.checked(cache['input_binding']))!=inputs:return None
  return deps
 except (OSError,ValueError,KeyError):return None

class RInnerSourceReview:
 def __init__(self,base,*,requested_row_indices):
  self.base=base;self.packet=base.packet;self.arrays=base.arrays;self.cfg=base.config;self.requested=set(requested_row_indices);self.context=_context(self.cfg);self.frames=OrderedDict();self.depbase,self.code=_dependencies(self.cfg)
  if _policy(self.cfg) is None or _policy(self.cfg).get('version')!=VERSION:raise ValueError('Explicit R source policy required')
  for i in self.requested:
   row=self.context['rows'][self.packet['source_row_ids'][i]];s=dict(video_sha256=row['source_video_sha256'],time_base='1/12800',coordinate_domain='native_pixels',pts=int(self.arrays['native_pts'][i]));_inputs(s,self.packet,self.arrays,i,self.cfg)
 def _frame(self,pts):
  if pts in self.frames:return self.frames[pts]
  from .native_source import iter_scope_native_rgb
  from PIL import Image
  _,rgb,decode=next(iter_scope_native_rgb(self.base.scope,[int(pts)],roles_path=self.base.roles));f=self.context['frames'][pts];mask=np.asarray(Image.open(source.checked(f['current_mask'])).convert('L'))!=0;cfgmask=self.cfg.get('frames',{}).get(str(pts),{}).get('mask')
  if cfgmask is not None:mask=mask|(np.asarray(Image.open(legacy.checked(cfgmask)).convert('L'))!=0)
  value=dict(rgb=rgb,decode=decode,mask=mask,digest=self.context['operator'].sha_bytes(rgb));self.frames[pts]=value
  while len(self.frames)>2:self.frames.popitem(last=False)
  return value
 def read_frame(self,pts,tasks,output):
  out=Path(output).resolve()
  if out.exists():raise FileExistsError('Immutable R inner source review exists')
  for r in self.code+self.context['dependencies']:legacy.checked(r)
  frame=self._frame(int(pts));out.mkdir(parents=True);coeff=self.context['operator'].prepare_cubic_channels(frame['rgb']);outcomes=[];records=[]
  for task in tasks:
   i=int(task['source_row_index']);inputs=_inputs(task['context']['source'],self.packet,self.arrays,i,self.cfg);row=inputs['source_row'];p=inputs['source_proof'];cert=row['source_object_and_material_identity_certificate'];frozen=row['source_measurement_audit']['source_material_receipt']['frozen_point_operator'];raw_normal=[1.,0.]if cert['actual_branch_source_navigation']=='inner_stem'else[1.,-self.context['frames'][int(pts)]['source_only_navigation_definition']['inner_leg_dx_per_dy']];reading=self.context['operator'].read(frame['rgb'],frame['mask'],frozen['navigation_seed_uv'],raw_normal,cert,coefficients=coeff)
   good=bool(reading.get('current_uv')==p['current_uv']and reading.get('normal')==p['normal']and reading.get('source_material_receipt')==row['source_measurement_audit']['source_material_receipt']and frame['digest']==row['native_RGB_sha256'])
   outcomes.append(dict(source_row_index=i,source_row_id=p['source_row_id'],status='exact_actual_R_inner_material_reproduced'if good else'unknown_current_R_inner_material_support',actual_reading=reading,original_UV_retained=p['current_uv'],current_RGB_sha256=frame['digest'],source_station_unknown=True,source_station_intervals=[[.25,.75]],physical_capture_pass_claimed=False,noise_precision_issued=False,source_UV_changed=False));records.append((i,task,inputs,good))
  write_json(out/'review.json',dict(schema='rinkfit.actual_R_inner_frozen_source_review.v1',native_pts=int(pts),decode=frame['decode'],outcomes=outcomes,source_UV_changed=False));review=reference(out/'review.json');resolutions=[];entries=[]
  from rinkspace.feedback_runtime import feedback_store
  from rink_feedback.cycle import FeedbackCycle
  for i,task,inputs,good in records:
   s=inputs['source'];deps=dict(self.depbase,actual_source_inputs_sha256=object_sha256(inputs),observation_snapshot_sha256=object_sha256(inputs['observation']),source_binding_sha256=object_sha256(dict(source=s,RGB_sha256=frame['digest'],mask=self.context['frames'][int(pts)]['current_mask'])),navigation_state_sha256='not_used',effective_facts_sha256=object_sha256(dict(qualification=inputs['qualification'],source_proof=inputs['source_proof'])))
   ip=out/('input_'+str(i)+'.json');write_json(ip,inputs);native=dict(path=inputs['source_row']['native_png'],sha256=inputs['source_row']['native_png_sha256']);refs=[native,review,reference(ip),*self.code,*self.context['dependencies']]
   cache=_cache(s,self.packet,i,deps['reader_code_sha256']);cache.parent.mkdir(parents=True,exist_ok=True);write_json(cache,dict(review_dependencies=deps,evidence_refs=refs,input_binding=reference(ip),review=review,material_label='Ad4.print.word_ANKARA_R_arc1',shot_id='s09'))
   with feedback_store(legacy.PROJECT,'ankara_2026')as store:
    finding=dict(source=s,event_id='ankara_2026',programme_id='okada_mei_sp',camera_id='main',entity_ids=['Ad4.print.word_ANKARA_R_arc1'],observation_ids=[self.packet['source_dataset_ids'][i]+'::'+self.packet['source_row_ids'][i]],signal_type='source_material_review',problem_key='current_frozen_R_inner_source',context_tags=['native_source_recheck','R_inner_source'],evidence_refs=[native,review],review_dependencies=deps)
    cycle=FeedbackCycle(store);job=cycle.checkpoint(out.name,[finding]);checks={k:dict(status='verified'if good else'unknown',evidence_refs=[native['sha256']])for k in ('semantic_class','physical_instance','measurement_operator','visibility','position')}
    if good:checks['visibility']['visible']=True;checks['position']['value']='actual finite R inner source material reproduced; station remains unknown; no physical closure or noise precision'
    learned=cycle.learn(dict(finding_id=job['tasks'][0]['finding_id'],source=s,resolution_type='model_mismatch'if good else'unresolved',checks=checks,evidence_refs=[dict(native,kind='native_pixels',source=s,independent_of_model=True,current_frame=True),review,reference(ip)],review_dependencies=deps,request_recheck=True),lesson=None);resolutions.append(learned['resolution'])
   entries.append(dict(source_row_index=i,review_dependencies=deps))
  result=dict(schema='rinkfit.actual_R_inner_source_feedback_receipt.v1',native_pts=int(pts),source_packet=reference(self.base.packet_path),review=review,counts=dict(subjects=len(records),R_inner_source=len(records),source_supported=sum(x[3]for x in records),pending=sum(not x[3]for x in records)),resolutions=resolutions,review_dependencies=entries,source_UV_changed=False,noise_precision_issued=False,new_observation_installed=False,physical_capture_pass_claimed=False);write_json(out/'receipt.json',result);return result
