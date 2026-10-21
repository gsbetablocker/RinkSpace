"""Re-read qualified unknown board-print texture points from native RGB.

No corner operator, ice homography, model UV, world-coordinate assertion or
observation mutation is used. Acquisition and current visibility stay separate.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib,json
import cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from .packet_bound_source_review import reference
from . import source_recovery_review as legacy

VERSION='exact_native_static_board_texture_track_review_v1'
PREFIX='static_board_texture/'
_CONTEXT={}


def _policy(cfg):return cfg.get('static_board_texture_policy')


def _dependencies(cfg):
    root=Path(__file__).parent
    code=[reference(root/name)for name in ('static_board_feedback_review.py','native_source.py','source_texture_tracklets.py','source_worker.py','native_entity_tracklets.py','static_board_tracklet_append.py')]
    return dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(_policy(cfg))),code


def _context(cfg):
    refs=cfg.get('static_board_texture_sources',[]);key=object_sha256(refs)
    if key in _CONTEXT:return _CONTEXT[key]
    result=dict(rows={},receipts={},owners={},numeric={},RGB={},actors={},dependencies=[],previous={})
    for ref in refs:
        bp=legacy.checked(ref);b=read_json(bp);result['dependencies'].append(ref)
        if b.get('schema')!='rinkfit.native_static_board_tracklet_qualified_bundle.v1':raise ValueError('unknown board texture source bundle')
        def checked(name):
            r=b[name];p=legacy.checked(r);result['dependencies'].append(r);return p
        rows=list(read_jsonl(checked('source_rows_ref')));receipts={r['source_row_id']:r for r in read_jsonl(checked('qualification_receipts'))};q=read_json(checked('static_surface_qualification'));gen=read_json(checked('source_generation'))
        if q.get('observations_from_model_or_ice_H')is not False or q.get('UV_moved_or_interpolated')is not False:raise ValueError('board texture source contains model/interpolated UV')
        owners={r['track_id']:r for r in q['per_track']if r.get('qualified_for_unknown_shared_board_surface_point_initialization')is True}
        numeric=read_json(legacy.checked(q['numeric_replay']));rgb=read_json(legacy.checked(q['current_RGB_receipt']));result['dependencies'] += [q['numeric_replay'],q['current_RGB_receipt'],gen['source_seed_review']]
        result['numeric'].update({(r['track_id'],int(r['native_pts'])):r for r in numeric['rows_evidence']});result['RGB'].update({(r['track_id'],int(r['native_pts'])):r for r in rgb['rows']})
        result['owners'].update(owners);result['criteria']=numeric['existing_navigation_criteria'];result['source_video']=gen['source_video']
        wanted={int(r['native_pts'])for r in rows}
        for r in gen['independent_actor_ledgers']:
            p=legacy.checked(r);result['dependencies'].append(r)
            for actor in read_jsonl(p):
                if int(actor['native_pts'])in wanted:result['actors'][int(actor['native_pts'])]=actor
        bytrack={}
        for row in rows:
            rid=row['source_row_id'];tid=row['track_id'];pts=int(row['native_pts']);receipt=receipts[rid]
            if rid in result['rows']and result['rows'][rid]!=row:raise ValueError('ambiguous actual texture measurement version')
            if (tid not in owners or receipt.get('schema')!='rinkfit.static_board_tracklet_point_receipt.v1'or receipt['source_uv']!=row['source_uv']or row['current_uv']!=row['source_uv']or receipt.get('known_world_corner_correspondence')is not False or row.get('world_position_is_shared_track_latent')is not True):raise ValueError('board texture source owner/UV role differs')
            result['rows'][rid]=row;result['receipts'][rid]=receipt;bytrack.setdefault(tid,[]).append(row)
        for tid,rr in bytrack.items():
            rr.sort(key=lambda r:r['native_pts'])
            for j,row in enumerate(rr):result['previous'][row['source_row_id']]=None if j==0 else rr[j-1]
    _CONTEXT[key]=result;return result


def _inputs(source,packet,arrays,index,cfg):
    c=_context(cfg);rid=packet['source_row_ids'][index];row=c['rows'].get(rid)
    if row is None:raise ValueError('current texture row is absent from exact qualified source')
    pts=int(arrays['native_pts'][index]);tid=row['track_id'];owner=c['owners'][tid];receipt=c['receipts'][rid];uv=arrays['uv'][index].tolist()
    if (int(source['pts'])!=pts or source.get('time_base')!='1/12800'or row['native_pts']!=pts or row['current_uv']!=uv or row['joint_material_index']!=int(arrays['material_index'][index])or int(arrays['kind'][index])!=1 or row['source_video_sha256']!=source['video_sha256']or owner['board_instance_id']!=row['surface_instance_id']):raise ValueError('current packet texture mean/owner/clock differs')
    previous=c['previous'][rid]
    return dict(schema='rinkfit.exact_native_board_texture_feedback_inputs.v1',source=source,observation=legacy.observation_content(packet,arrays,index),source_row=row,qualification_receipt=receipt,track_owner=owner,acquisition_numeric_evidence=c['numeric'][(tid,pts)],acquisition_RGB_evidence=c['RGB'][(tid,pts)],current_actor_ledger=c['actors'][pts],previous_source_row=previous,previous_actor_ledger=None if previous is None else c['actors'][previous['native_pts']],frame_inputs=cfg.get('frames',{}).get(str(pts),{}),previous_frame_inputs=None if previous is None else cfg.get('frames',{}).get(str(previous['native_pts']),{}),programme_scope=cfg['programme_scope'],frame_roles=cfg['frame_roles'],reader_policy=_policy(cfg))


def _cache(source,packet,index,code_sha):
    from rinkspace.storage import AssetStore
    key=object_sha256(dict(source=source,dataset=packet['source_dataset_ids'][index],row_id=packet['source_row_ids'][index]))
    return AssetStore(legacy.PROJECT).data/'events/ankara_2026/feedback/review_cache'/('bt_'+code_sha[:16])/(key+'.json')


def cached_dependencies(source,packet,arrays,index,*,review_config=None):
    if review_config is None:return None
    try:
        cfg=read_json(legacy.checked(review_config)if isinstance(review_config,dict)else review_config);base,_=_dependencies(cfg);p=_cache(source,packet,index,base['reader_code_sha256'])
        if not p.is_file():return None
        cache=read_json(p);inputs=_inputs(source,packet,arrays,index,cfg);deps=cache['review_dependencies']
        if any(deps.get(k)!=v for k,v in base.items())or deps.get('actual_source_inputs_sha256')!=object_sha256(inputs):return None
        for ref in cache['evidence_refs']:legacy.checked(ref)
        if read_json(legacy.checked(cache['input_binding']))!=inputs:return None
        return deps
    except (ValueError,KeyError,OSError):return None


class StaticBoardTextureReview:
    def __init__(self,base,*,requested_row_indices):
        self.base=base;self.packet=base.packet;self.arrays=base.arrays;self.cfg=base.config;self.requested=set(requested_row_indices);self.context=_context(self.cfg);self.frames=OrderedDict()
        if _policy(self.cfg) is None or _policy(self.cfg).get('version')!=VERSION:raise ValueError('unrecognized current board texture strategy; reread needs an explicit supported policy')
        self.depbase,self.code=_dependencies(self.cfg)
        for i in self.requested:
            pts=int(self.arrays['native_pts'][i]);source=dict(video_sha256=self.context['source_video']['sha256'],time_base='1/12800',coordinate_domain='native_pixels',pts=pts);_inputs(source,self.packet,self.arrays,i,self.cfg)
    def _frame(self,pts):
        if pts in self.frames:return self.frames[pts]
        from .native_source import iter_scope_native_rgb
        from .source_texture_tracklets import _source_mask
        from PIL import Image
        _,rgb,decode=next(iter_scope_native_rgb(self.base.scope,[int(pts)],roles_path=self.base.roles));actor=self.context['actors'][pts]
        excluded,digest=_source_mask(rgb,actor,self.context['source_video'],pts,'s02');cfg=self.cfg.get('frames',{}).get(str(pts),{});maskref=cfg.get('mask')
        allowed=np.ones(rgb.shape[:2],bool)if maskref is None else np.asarray(Image.open(legacy.checked(maskref)).convert('L'))==0
        if allowed.shape!=rgb.shape[:2]:raise ValueError('review mask has another native pixel domain')
        result=dict(rgb=rgb,gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),decode=decode,digest=digest,excluded=excluded,allowed=allowed,mask=maskref)
        self.frames[pts]=result
        while len(self.frames)>2:self.frames.popitem(last=False)
        return result
    def read_frame(self,pts,tasks,output):
        from PIL import Image
        from .source_worker import _channels,_remeasure
        from .native_entity_tracklets import _ncc
        out=Path(output).resolve()
        if out.exists():raise FileExistsError('immutable native texture review exists')
        for ref in self.code+self.context['dependencies']:legacy.checked(ref)
        frame=self._frame(int(pts));out.mkdir(parents=True);cfg=self.cfg.get('frames',{}).get(str(pts),{});native=cfg.get('native_image')
        frame_refs=[]
        for frame_inputs in (self.cfg.get('frames',{}).get(str(p),{})for p in {int(pts),*[int(self.context['previous'][self.packet['source_row_ids'][int(t['source_row_index'])]]['native_pts'])for t in tasks if self.context['previous'][self.packet['source_row_ids'][int(t['source_row_index'])]]is not None]}):
            for ref in frame_inputs.values():
                if isinstance(ref,dict)and 'sha256'in ref and ('path'in ref or'file'in ref):
                    legacy.checked(ref);frame_refs.append(ref)
        if native is not None:
            p=legacy.checked(native)
            if not np.array_equal(np.asarray(Image.open(p).convert('RGB')),frame['rgb']):raise ValueError('current source native image does not equal exact decode')
            native=reference(p)
        else:Image.fromarray(frame['rgb']).save(out/'native.png');native=reference(out/'native.png')
        channels=_channels(frame['rgb']);outcomes=[];records=[]
        params=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
        for task in tasks:
            i=int(task['source_row_index']);rid=self.packet['source_row_ids'][i];source=task['context']['source']
            if int(self.arrays['native_pts'][i])!=int(pts):raise ValueError('texture task belongs to another exact native PTS')
            inputs=_inputs(source,self.packet,self.arrays,i,self.cfg);row=inputs['source_row'];uv=np.asarray(row['source_uv']);xy=np.rint(uv).astype(int);diag=_remeasure(uv,[0,0],channels,'texture',corner=True);previous=inputs['previous_source_row'];support=frame['allowed'][xy[1]-16:xy[1]+17,xy[0]-16:xy[0]+17];maskclear=bool(support.shape==(33,33)and support.all()and not frame['excluded'][xy[1],xy[0]]);motion=None
            if previous is not None:
                if int(previous['native_pts'])+256!=int(pts):raise ValueError('texture motion crosses source gap/cut')
                prior=self._frame(int(previous['native_pts']));p=np.asarray(previous['source_uv'],np.float32).reshape(1,1,2);q,ok,_=cv2.calcOpticalFlowPyrLK(prior['gray'],frame['gray'],p,None,**params);back,bok,_=cv2.calcOpticalFlowPyrLK(frame['gray'],prior['gray'],q,None,**params);a=cv2.getRectSubPix(prior['gray'].astype(np.float32),(33,33),tuple(previous['source_uv']));b=cv2.getRectSubPix(frame['gray'].astype(np.float32),(33,33),tuple(row['source_uv']));fb=float(np.linalg.norm(back[0,0]-p[0,0]));ncc=_ncc(a,b);motion=dict(previous_native_pts=int(previous['native_pts']),previous_current_RGB_sha256=prior['digest'],LK_current_UV=q[0,0].astype(float).tolist(),LK_to_stored_UV_delta_px=float(np.linalg.norm(q[0,0]-uv)),forward_backward_px=fb,NCC=ncc,forward_backward_status=bool(ok[0,0]and bok[0,0]),acquisition_FB_px=row['forward_backward_px'],acquisition_NCC=row['patch_ncc'])
            good=bool(frame['digest']==row['current_rgb_sha256']and maskclear and diag.get('status')=='entity_corner_two_direction_support'and(motion is None or motion['forward_backward_status']and motion['forward_backward_px']<=self.context['criteria']['reciprocal_max_px']and motion['NCC']>=self.context['criteria']['patch_ncc_min']and motion['LK_to_stored_UV_delta_px']==0.))
            outcome=dict(source_row_id=rid,source_row_index=i,track_id=row['track_id'],surface_class='board',source_ad_instance=row['surface_instance_id'],reader=VERSION,status='source_texture_UV_reproduced'if good else'pending_current_texture_support',source_UV_retained=row['source_uv'],current_RGB_sha256=frame['digest'],source_RGB_sha256_matches=frame['digest']==row['current_rgb_sha256'],source_actor_exclusion_clear=not bool(frame['excluded'][xy[1],xy[0]]),current_review_mask_clear=maskclear,two_direction_texture=diag,motion=motion,track_owner_qualification=inputs['track_owner'],world_position_unknown=True,known_world_corner=False,corner_operator_called=False,ice_H_used=False,observation_installed=False)
            outcomes.append(outcome);records.append((i,task,inputs,good))
        review=out/'review.json';write_json(review,dict(schema='rinkfit.actual_native_board_texture_feedback.v1',native_pts=int(pts),decode=frame['decode'],outcomes=outcomes,source_UV_changed=False,statistical_precision_issued=False));resolutions=[];cache_entries=[]
        for i,task,inputs,good in records:
            source=inputs['source'];deps=dict(self.depbase,actual_source_inputs_sha256=object_sha256(inputs),observation_snapshot_sha256=object_sha256(inputs['observation']),source_binding_sha256=object_sha256(dict(source=source,RGB_sha256=frame['digest'],mask_sha256=None if frame['mask']is None else frame['mask']['sha256'])),navigation_state_sha256='not_used',effective_facts_sha256=object_sha256(dict(track_owner=inputs['track_owner'],qualification_receipt=inputs['qualification_receipt'])))
            binding=out/('input_'+str(i)+'.json');write_json(binding,inputs);refs=[native,reference(review),reference(binding),*self.code,*self.context['dependencies'],*frame_refs];cache=_cache(source,self.packet,i,deps['reader_code_sha256']);cache.parent.mkdir(parents=True,exist_ok=True);write_json(cache,dict(review_dependencies=deps,evidence_refs=refs,input_binding=reference(binding),review=reference(review),source_row_id=self.packet['source_row_ids'][i],material_label=task['context']['entity_ids'][0],shot_id='s02'));cache_entries.append(dict(source_row_index=i,review_dependencies=deps))
            from rinkspace.feedback_runtime import feedback_store
            from rink_feedback.cycle import FeedbackCycle
            with feedback_store(legacy.PROJECT,'ankara_2026')as store:
                try:old=store.get_finding(task['finding_id'])
                except Exception:old={}
                finding=dict(source=source,event_id='ankara_2026',programme_id='okada_mei_sp',camera_id=old.get('camera_id'),entity_ids=task['context']['entity_ids'],observation_ids=[self.packet['source_dataset_ids'][i]+'::'+self.packet['source_row_ids'][i]],signal_type=old.get('signal_type','source_material_review'),problem_key=old.get('problem_key','current_native_source_material_review'),context_tags=old.get('context_tags',['native_source_recheck','board_print_texture']),evidence_refs=[native,reference(review)],review_dependencies=deps)
                cycle=FeedbackCycle(store);job=cycle.checkpoint(out.name,[finding]);fid=job['tasks'][0]['finding_id'];checks={name:dict(status='verified'if good else'unknown',evidence_refs=[native['sha256']])for name in('semantic_class','physical_instance','measurement_operator','visibility','position')}
                if good:checks['visibility']['visible']=True;checks['position']['value']='exact current native texture UV reproduced; shared board coordinate remains unknown; no known-world-corner or noise precision asserted'
                resolved=cycle.learn(dict(finding_id=fid,source=source,resolution_type='model_mismatch'if good else'unresolved',checks=checks,evidence_refs=[dict(native,kind='native_pixels',source=source,independent_of_model=True,current_frame=True),reference(review),reference(binding)],review_dependencies=deps,request_recheck=True),lesson=None);resolutions.append(resolved['resolution'])
        result=dict(schema='rinkfit.actual_native_board_texture_feedback_receipt.v1',native_pts=int(pts),decode=frame['decode'],source_packet=reference(self.base.packet_path),native_image=native,review=reference(review),counts=dict(subjects=len(records),board_texture=len(records),source_supported=sum(x[3]for x in records),pending=sum(not x[3]for x in records)),resolutions=resolutions,review_dependencies=cache_entries,known_world_corner=False,source_UV_changed=False,new_observation_installed=False,visual_accuracy_improvement_claimed=False)
        write_json(out/'receipt.json',result);return result
