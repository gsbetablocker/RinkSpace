"""Re-read the actual two paint interfaces; never assert a known world point.

Frozen native acquisition profiles and exact material qualification select a
source candidate. Camera, width nuisance, fitted UV and screen interpolation
are absent from the source mean and its cache identity.
"""
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json as _write_json,object_sha256
from .packet_bound_source_review import reference
from . import source_recovery_review as legacy


def _jsonable(value):
    if isinstance(value,np.ndarray):return _jsonable(value.tolist())
    if isinstance(value,np.generic):return _jsonable(value.item())
    if isinstance(value,float)and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):_jsonable(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_jsonable(v)for v in value]
    return value

def write_json(path,value):
    # Failed/off-sensor profile samples remain explicit null; never zero UV.
    return _write_json(path,_jsonable(value))

VERSION='exact_frozen_original_C_interfaces_review_v1'
PREFIX='native_C_operator_supersession_v1/'
_CONTEXT={}

def _policy(cfg):return cfg.get('original_C_pair_source_policy')

def _dependencies(cfg):
    root=Path(__file__).parent;project=root.parents[3]
    code=[reference(root/name)for name in ('original_paint_pair_feedback_review_v1.py','paint_midpoint_supersession.py','physical_source_operator.py','native_paint_wide.py','native_source.py','source_texture_tracklets.py')]
    code += [reference(project/'src/rinkspace/vision_programme_ice_review.py')]
    return dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(_policy(cfg))),code

def _context(cfg):
    refs=cfg.get('original_C_pair_source_bundles',[]);key=object_sha256(refs)
    if key in _CONTEXT:return _CONTEXT[key]
    c=dict(rows={},receipts={},actors={},dependencies=[])
    for ref in refs:
        bp=legacy.checked(ref);b=read_json(bp);c['dependencies'].append(ref)
        if b.get('schema')!='rinkfit.original_native_C_operator_transition_bundle.v1':raise ValueError('Unknown original paint-interface operator bundle')
        def checked_ref(r):c['dependencies'].append(r);return legacy.checked(r)
        sp=checked_ref(b['original_source_authority']);a=read_json(sp);rows=list(read_jsonl(checked_ref(a['source_rows_ref'])));proofs=list(read_jsonl(checked_ref(a['actual_replay'])));q=read_json(checked_ref(a['same_current_C_material_qualification']));gen=read_json(checked_ref(q['source_generation']))
        if a.get('qualified_midpoint_supersessions')!=13 or len(rows)!=26 or len(proofs)!=13 or a.get('old_raw_UV_and_ID_preserved')is not True:raise ValueError('Original paint interfaces lost source derivation authority')
        c['source_video']=gen['source_video'];c['qualification']=q;c['qualification_ref']=a['same_current_C_material_qualification'];wanted={r['native_pts']for r in rows}
        for ar in gen['actor_ledgers']:
            c['dependencies'].append(ar)
            for actor in read_jsonl(legacy.checked(ar)):
                if actor['native_pts']in wanted:c['actors'][int(actor['native_pts'])]=actor
        for j,row in enumerate(rows):
            rid=row['source_row_id'];proof=proofs[j//2];d=proof['existing_source_derivation'];old=d['original_source_row'];side=j%2
            if proof.get('qualified_for_operator_supersession')is not True or row['current_uv']!=d['source_flanks_uv'][side]or row['source_uv']!=row['current_uv']or row['old_correct_midpoint_UV']!=d['correct_original_packet_UV']or row['derived_midpoint_source_row_id']!=d['source_row_id']or row['source_side_index']!=side or row['joint_material_index']!=145 or row['point_correspondence']is not False or row['pixels_from_model']is not False:raise ValueError('Original paint review redirects actual source interface')
            checked_ref(d['journal']);checked_ref(proof['frozen_replay_module'])
            c['rows'][rid]=row;c['receipts'][rid]=dict(original_source_row=old,original_derivation=d,replay_proof=proof,chosen_uv=row['current_uv'],frozen_original_material_receipt=old['source_operator_result']['source_material_receipt'],qualification_ref=b['original_source_authority'])
    _CONTEXT[key]=c;return c

def _inputs(source,packet,arrays,index,cfg):
    c=_context(cfg);rid=packet['source_row_ids'][index];row=c['rows'].get(rid)
    if row is None:raise ValueError('No actual qualified paint source for current packet row')
    pts=int(arrays['native_pts'][index]);uv=arrays['uv'][index].tolist()
    if row['native_pts']!=pts or int(source['pts'])!=pts or source['time_base']!='1/12800'or source['video_sha256']!=row['source_video_sha256']or uv!=row['current_uv']or int(arrays['kind'][index])!=0 or int(arrays['material_index'][index])!=145:raise ValueError('Current packet paint UV/material/native clock differs')
    return dict(schema='rinkfit.exact_frozen_original_paint_interface_feedback_inputs.v1',source=source,observation=legacy.observation_content(packet,arrays,index),source_row=row,qualification_receipt=c['receipts'][rid],material_qualification_ref=c['qualification_ref'],material_component_policy=c['qualification']['semantic_component_policy'],current_actor_ledger=c['actors'][pts],frame_inputs=cfg.get('frames',{}).get(str(pts),{}),programme_scope=cfg['programme_scope'],frame_roles=cfg['frame_roles'],reader_policy=_policy(cfg))

def _cache(source,packet,index,code):
    from rinkspace.storage import AssetStore
    key=object_sha256(dict(source=source,dataset=packet['source_dataset_ids'][index],row_id=packet['source_row_ids'][index]))
    return AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/review_cache'/('po_'+code[:16])/(key+'.json')

def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    if review_config is None:return None
    try:
        cfg=read_json(legacy.checked(review_config)if isinstance(review_config,dict)else review_config);base,_=_dependencies(cfg);path=_cache(source,packet,index,base['reader_code_sha256'])
        if not path.is_file():return None
        cache=read_json(path);inputs=_inputs(source,packet,arrays,index,cfg);deps=cache['review_dependencies']
        if any(deps.get(k)!=v for k,v in base.items())or deps.get('actual_source_inputs_sha256')!=object_sha256(inputs):return None
        for ref in cache['evidence_refs']:legacy.checked(ref)
        if read_json(legacy.checked(cache['input_binding']))!=inputs:return None
        return deps
    except (ValueError,KeyError,OSError):return None

class OriginalPaintPairSourceReview:
    def __init__(self,base,*,requested_row_indices):
        self.base=base;self.packet=base.packet;self.arrays=base.arrays;self.cfg=base.config;self.requested=set(requested_row_indices);self.context=_context(self.cfg);self.frames=OrderedDict();self.depbase,self.code=_dependencies(self.cfg)
        if _policy(self.cfg)is None or _policy(self.cfg).get('version')!=VERSION:raise ValueError('Paint pair reread requires an explicit supported strategy')
        for i in self.requested:
            source=dict(video_sha256=self.context['source_video']['sha256'],time_base='1/12800',coordinate_domain='native_pixels',pts=int(self.arrays['native_pts'][i]));_inputs(source,self.packet,self.arrays,i,self.cfg)
    def _frame(self,pts):
        if pts in self.frames:return self.frames[pts]
        from .native_source import iter_scope_native_rgb
        from .source_texture_tracklets import _source_mask
        from PIL import Image
        _,rgb,decode=next(iter_scope_native_rgb(self.base.scope,[int(pts)],roles_path=self.base.roles));excluded,digest=_source_mask(rgb,self.context['actors'][pts],self.context['source_video'],pts,'s09');frame=self.cfg.get('frames',{}).get(str(pts),{});maskref=frame.get('mask')
        if maskref is not None:
            a=np.asarray(Image.open(legacy.checked(maskref)).convert('L'))
            if a.shape!=rgb.shape[:2]:raise ValueError('Paint review mask has another native domain')
            excluded=excluded.copy();excluded[a!=0]=255
        value=dict(rgb=rgb,decode=decode,mask=maskref,digest=digest,excluded=excluded);self.frames[pts]=value
        while len(self.frames)>2:self.frames.popitem(last=False)
        return value
    def read_frame(self,pts,tasks,output):
        from PIL import Image
        from rinkspace.vision_source_shapes import read_source_chromatic_stroke_centers
        from rinkspace.vision_programme_ice_review import _candidate_context_excluded
        out=Path(output).resolve()
        if out.exists():raise FileExistsError('Immutable actual paint source review exists')
        for ref in self.code+self.context['dependencies']:legacy.checked(ref)
        frame=self._frame(int(pts));out.mkdir(parents=True);Image.fromarray(frame['rgb']).save(out/'native.png');native=reference(out/'native.png');outcomes=[];records=[];readings={}
        binding=dict(source_sha256=self.context['source_video']['sha256'],source_pts=int(pts),time_base='1/12800',native_hw=[1080,1920],shot_id='s09',decoded_native_rgb_sha256=frame['digest'])
        for task in tasks:
            i=int(task['source_row_index']);inputs=_inputs(task['context']['source'],self.packet,self.arrays,i,self.cfg);row=inputs['source_row'];rid=row['derived_midpoint_source_row_id'];receipt=inputs['qualification_receipt'];query=row['source_acquisition_query']
            if rid not in readings:
                from .paint_midpoint_supersession import replay_original_flanks
                readings[rid]=replay_original_flanks(frame['rgb'],receipt['frozen_original_material_receipt'])
            reading=readings[rid];expected=np.asarray(receipt['original_derivation']['source_flanks_uv']);actual=np.asarray(reading['flanks']);mean=np.asarray(reading['current_uv']);err=float(np.max(abs(actual-expected)))
            excluded=any(_candidate_context_excluded(frame['excluded'],point,row['normal'])for point in [mean,*actual]);matches=[]
            good=bool(frame['digest']==row['current_rgb_sha256']and err<=1e-10 and reading['profile_max_abs_error']<=1e-10 and reading['window_boundary_count']==0 and not excluded)
            if good:matches.append(dict(replay_max_abs_difference_px=err,actual_original_flank_UV=actual.tolist(),actual_five_profile_max_error=reading['profile_max_abs_error'],frozen_channel_polarity_windows_unchanged=True))
            outcome=dict(source_row_index=i,source_row_id=row['source_row_id'],reader=VERSION,status='exact_actual_two_paint_interfaces_reproduced'if good else'pending_current_paint_material_support',original_UV_retained=row['current_uv'],source_side_index=row['source_side_index'],actual_interface_matches=matches,current_RGB_sha256=frame['digest'],world_station_unknown=True,world_width_unknown=True,own_RS_operator_required=True,source_width_is_not_sigma_or_tolerance=True,source_UV_changed=False)
            outcomes.append(outcome);records.append((i,task,inputs,good))
        write_json(out/'source_profiles.json',readings);write_json(out/'review.json',dict(schema='rinkfit.actual_two_paint_interface_source_review.v1',native_pts=int(pts),decode=frame['decode'],outcomes=outcomes,source_UV_changed=False));review=reference(out/'review.json');profile=reference(out/'source_profiles.json');resolutions=[];entries=[]
        for i,task,inputs,good in records:
            source=inputs['source'];deps=dict(self.depbase,actual_source_inputs_sha256=object_sha256(inputs),observation_snapshot_sha256=object_sha256(inputs['observation']),source_binding_sha256=object_sha256(dict(source=source,RGB_sha256=frame['digest'],mask_sha256=None if frame['mask']is None else frame['mask']['sha256'])),navigation_state_sha256='not_used',effective_facts_sha256=object_sha256(dict(qualification=inputs['material_qualification_ref'],receipt=inputs['qualification_receipt'])))
            f=out/('input_'+str(i)+'.json');write_json(f,inputs);refs=[native,review,profile,reference(f),*self.code,*self.context['dependencies']]
            for ref in inputs['frame_inputs'].values():
                if isinstance(ref,dict)and 'sha256'in ref:legacy.checked(ref);refs.append(ref)
            cache=_cache(source,self.packet,i,deps['reader_code_sha256']);cache.parent.mkdir(parents=True,exist_ok=True);write_json(cache,dict(review_dependencies=deps,evidence_refs=refs,input_binding=reference(f),review=review,material_label='ice.C:segment0',shot_id='s09'))
            from rinkspace.feedback_runtime import feedback_store
            from rink_feedback.cycle import FeedbackCycle
            with feedback_store(legacy.PROJECT,'ankara_2026')as store:
                finding=dict(source=source,event_id='ankara_2026',programme_id='okada_mei_sp',camera_id='main',entity_ids=['ice.C:segment0'],observation_ids=[self.packet['source_dataset_ids'][i]+'::'+self.packet['source_row_ids'][i]],signal_type='source_material_review',problem_key='current_frozen_original_C_interfaces',context_tags=['native_source_recheck','original_paint_pair'],evidence_refs=[native,review],review_dependencies=deps)
                cycle=FeedbackCycle(store);job=cycle.checkpoint(out.name,[finding]);checks={key:dict(status='verified'if good else'unknown',evidence_refs=[native['sha256']])for key in ('semantic_class','physical_instance','measurement_operator','visibility','position')}
                if good:checks['visibility']['visible']=True;checks['position']['value']='actual material interfaces reproduced; physical station and local width remain unknown; no noise precision'
                resolved=cycle.learn(dict(finding_id=job['tasks'][0]['finding_id'],source=source,resolution_type='model_mismatch'if good else'unresolved',checks=checks,evidence_refs=[dict(native,kind='native_pixels',source=source,independent_of_model=True,current_frame=True),review,reference(f)],review_dependencies=deps,request_recheck=True),lesson=None);resolutions.append(resolved['resolution'])
            entries.append(dict(source_row_index=i,review_dependencies=deps))
        result=dict(schema='rinkfit.actual_paint_pair_feedback_receipt.v1',native_pts=int(pts),source_packet=reference(self.base.packet_path),native_image=native,review=review,counts=dict(subjects=len(records),paint_interface=len(records),source_supported=sum(x[3]for x in records),pending=sum(not x[3]for x in records)),resolutions=resolutions,review_dependencies=entries,source_UV_changed=False,noise_precision_issued=False,new_observation_installed=False)
        write_json(out/'receipt.json',result);return result
