"""Requalify unchanged native board readouts through observed material roles.

Old colour candidates, logits, model navigation and old certificates are not
physical authority. This reader produces a new per-exposure, per-readout ledger.
It never remeasures the supplied mean, changes a Query, or creates a noise W.
"""
from functools import lru_cache
from pathlib import Path
import hashlib,json
import numpy as np
from .io_utils import file_sha256,object_sha256

SCHEMA='rinkfit.actual_board_material_role_requalification.v1'
VERSION='ordered_same_board_material_roles_v1'
CONFIG=dict(version=VERSION,profile_radius_px=128.,profile_spacing_px=8.,
    blue_entry_and_exit_are_separate_roles=True,old_source_certificate_is_authority=False,
    model_or_logit_identity_authority=False,mean_or_window_changed=False,
    body_neutral_chroma_range=0.20,body_luminance_min=0.28,body_white_flanks_required=True,global_body_white_fraction_is_diagnostic_only=True)

def _np(v):
    return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)

def _plain(v):
    if hasattr(v,'detach'):v=v.detach().cpu().numpy()
    if isinstance(v,np.ndarray):return _plain(v.tolist())
    if isinstance(v,np.generic):return _plain(v.item())
    if isinstance(v,dict):return {str(k):_plain(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [_plain(x)for x in v]
    if isinstance(v,float)and not np.isfinite(v):return None
    return v

def _mean(r):
    a=r.get('source_measurement_audit',{})
    return np.asarray(r.get('current_uv',a.get('current_uv',r.get('source_uv'))),np.float64)

def _role(r):
    s=str(r.get('material_role',r.get('target_id',r.get('joint_material_id',''))))
    return 'board.contact' if 'contact' in s else 'board.top' if 'top' in s else None

def measurement_sha256(r,pts):
    return object_sha256(dict(native_pts=int(pts),current_uv=_mean(r).tolist(),
        source_row_id=r['source_row_id'],measurement_operator=r.get('measurement_operator_resolved',r.get('measurement_operator'))))

def _support_contains(mu,endpoints,normal,xradius=8.):
    e=np.asarray(endpoints,float)
    if e.shape!=(2,2)or not np.isfinite(e).all():return False
    n=np.asarray(normal,float);n=n/np.linalg.norm(n);t=np.array([n[1],-n[0]])
    d=(np.asarray(mu)-e)@n
    return bool(d[0]*d[1]<=1e-9 and abs((np.asarray(mu)-e.mean(0))@t)<=xradius)

class BoardMaterialRoleRequalifier:
    """One frozen reader per process; every frame is independently reset."""
    def __init__(self,project=None,*,device='cuda',use_frozen_visual_model=False):
        self.project=Path(project or Path(__file__).resolve().parents[4]);self.device=device
        self.vision=None;self.refs=[]
        for relative in ('src/rinkspace/vision_source_topology.py','src/rinkspace/vision_geometry.py',
            'calibration/rinkfit/src/rinkfit_joint/board_outer_source.py',
            'events/ankara_2026/model_constraints/ice_contact_material_roles_user_confirmed_20261006.json',
            'events/ankara_2026/model_constraints/user_board_edge_semantics_20261010.json',
            'src/rinkspace/feedback_runtime.py','src/rinkspace/feedback_outbox.py',
            'src/rink_feedback/core.py','src/rink_feedback/cycle.py','src/rink_feedback/adapters.py',
            'calibration/rinkfit/src/rinkfit_joint/feedback_material_guard.py'):
            p=self.project/relative;self.refs.append(dict(path=str(p),sha256=file_sha256(p)))
        if use_frozen_visual_model:
            from rinkspace.vision import SourceRinkReader
            self.vision=SourceRinkReader(self.project/'configs/rink_vision.json')
            self.refs.append(dict(path=str(self.vision.vision.model_path),sha256=self.vision.vision.model_sha256,
                use='frozen source context and actual RGB edge navigation; no physical identity from logits'))
        self.refs.append(dict(path=str(Path(__file__).resolve()),sha256=file_sha256(Path(__file__))))
        self.recipe_sha256=object_sha256(dict(schema=SCHEMA,config=CONFIG,refs=self.refs))

    def evaluate(self,rgb,mask,rows,context):
        import torch,cv2
        from rinkspace.vision_source_topology import (read_board_foot_material_profiles,
            board_foot_material_authority,_runs,read_source_board_rail_material_profiles,_material_parents)
        image=np.asarray(rgb);mask=np.asarray(mask,bool);rows=list(rows);h,w=mask.shape
        if image.shape!=(h,w,3)or image.dtype!=np.uint8:raise ValueError('Unresampled uint8 original RGB/mask required')
        receipt=context.get('native_video_decode_receipt')or context.get('original_source_frame_proof',{}).get('native_video_decode_receipt')
        if not receipt:raise ValueError('Actual official native video decode receipt required')
        pts=int(receipt['native_pts']);sid=str(receipt['shot_id']);rgbsha=hashlib.sha256(image.tobytes()).hexdigest()
        expected=receipt.get('decoded_native_RGB_sha256')
        if expected!=rgbsha or receipt.get('native_RGB_resampled',False):raise ValueError('Original native RGB differs from source receipt')
        if receipt.get('time_base')!='1/12800':raise ValueError('This event programme requires its official 50fps native PTS clock')
        if receipt.get('source_video_sha256')!='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d':raise ValueError('Wrong programme processing video')
        binding=dict(source_sha256=receipt['source_video_sha256'],source_pts=pts,time_base=receipt['time_base'],shot_id=sid,native_hw=[h,w])
        from rinkspace.feedback_runtime import SourceFeedbackRead
        feedback = SourceFeedbackRead(dict(binding, decoded_native_RGB_sha256=rgbsha),
            project=self.project, entity_ids=['board.contact', 'board.bottom_strip', 'board.top'],
            tags=['ice_contact', 'board_foot', 'material_boundary'],
            run_id=context.get('feedback_run_id'))

        maskref=context.get('current_mask_ref')or context.get('original_source_frame_proof',{}).get('current_mask')
        if maskref:
            if file_sha256(maskref.get('path',maskref.get('file'))) !=maskref['sha256']:raise ValueError('Source exclusion mask changed')
            if not np.array_equal(cv2.imread(str(maskref.get('path',maskref.get('file'))),0)>0,mask):raise ValueError('Exclusion mask pixels differ')
        for r in rows:
            if int(r.get('native_pts',pts))!=pts or str(r.get('shot_id',sid))!=sid:raise ValueError('Readout belongs to another exposure')
            if _mean(r).shape!=(2,)or not np.isfinite(_mean(r)).all():raise ValueError('Finite unchanged current mean required')
        before=[measurement_sha256(r,pts)for r in rows]
        contact=np.array([_mean(r)for r in rows if _role(r)=='board.contact'],float).reshape(-1,2)
        top=np.array([_mean(r)for r in rows if _role(r)=='board.top'],float).reshape(-1,2)
        vision_usage=None;semantic=None;components=[]
        if self.vision is not None:
            result=self.vision.read(image,binding,reset=True)
            vision_usage=result['model_usage'];components=result['source']['components']
            semantic=_np(result['source']['semantic_argmax'])
        print_materials=_material_parents(image,[dict(kind=c['typed_class_candidate'],points=_np(c['source_support_uv']).astype(int))
            for c in components if c.get('typed_class_candidate')in('ad_boundary','ice_paint')])
        instances=[dict(object_id='current_source_exclusion',kind='occluder',region_mask=torch.as_tensor(mask),source_binding=binding)]
        authority=board_foot_material_authority(self.project/'events/ankara_2026/model_constraints/ice_contact_material_roles_user_confirmed_20261006.json')
        # Reuse the user's event material fact, retaining the old annotation
        # source clock. No old annotation coordinates or video pixels are used.
        authority=dict(authority,original_confirmation_source_sha256=authority['source_sha256'],
            source_sha256=binding['source_sha256'],same_event_material_fact_application=True,
            original_annotation_pixels_transferred=False)
        authority=feedback.material_authority(authority)
        foot=None;rail=None
        explicit_queries=None
        if not len(contact):
            # Search every observed warm run rather than choose the lowest
            # coloured edge. These are navigation pixels only; full paired
            # RGB/family/material role checks below still decide qualification.
            from rinkspace.vision_source_topology import _split_native_material_runs
            warm=(image[:,:,0].astype(float)+image[:,:,1])/2-image[:,:,2]
            actual=(warm>=.012*255)&(image.mean(2)>=.28*255)
            points=[]
            for x in range(2,w-2,8):
                for run in _split_native_material_runs(actual[:,x]):
                    if len(run)>=2:points.append([x,int(run[len(run)//2])])
            explicit_queries=np.asarray(points,float).reshape(-1,2)if points else None
        if len(contact)or explicit_queries is not None:
            topology=dict(source_binding=binding,selected_pair=dict(contact_chain=0)if len(contact)else None,chains=[dict(
                outer_contact_source_pixel_runs_y_x_length=_runs(np.rint(contact).astype(int)))])
            foot=read_board_foot_material_profiles(image,dict(source_binding=binding,instances=instances),topology,
                material_role_authority=authority,device=self.device,search_radius_px=CONFIG['profile_radius_px'],
                sample_spacing_px=CONFIG['profile_spacing_px'],source_query_points_uv=explicit_queries)
        if len(top):
            # Historical top samples navigate the blue-band profile only.
            # Every original mean is checked against entering BLUE, never its
            # blue-to-white exit; the supplied top is not declared physical.
            nav=dict(source_binding=binding,RGB_sha256=rgbsha,actual_source_top_uv=top)
            rail=read_source_board_rail_material_profiles(image,nav,source_instances=instances,sample_spacing_px=1)
        fuv=np.empty((0,2));fmask=np.empty(0,bool);fend=np.empty((0,2,2));fn=np.empty((0,2));entry=np.empty((0,2));fev=[]
        if foot and foot.get('readings'):
            fuv=_np(foot['readings'][0]['candidate_source_uv']);fmask=_np(foot['readings'][0]['primary_role_samples']).astype(bool)
            fend=_np(foot['source_material_transition_support']['observed_native_warm_colour_fall_support_endpoints_uv'])
            fn=_np(foot['original_RGB_profiles']['normals_unit'])
            pp=foot['original_RGB_profiles'];ids=_np(foot['same_material_band_pair_evidence']['entry_index']).astype(int)
            entry=_np(pp['sampled_native_uv'])[np.arange(len(ids)),ids]
        blue_pixels=(image[:,:,2].astype(float)-image[:,:,0]>4)&(image[:,:,2].astype(float)-image[:,:,1]>4)
        _,blue_labels,blue_stats,_=cv2.connectedComponentsWithStats(blue_pixels.astype(np.uint8),8)
        top_evidence=list(rail['source_material_role_evidence'])if rail else []
        # Missing historical top samples are not evidence of invisibility.
        # Fresh frozen-type RGB edge components can independently navigate
        # additional blue bands; all of their pixels are re-read below.
        for c in components:
            if c.get('typed_class_candidate')!='board_top':continue
            hint=_np(c.get('source_support_uv',())).astype(float).reshape(-1,2)
            if not len(hint):continue
            fresh=read_source_board_rail_material_profiles(image,dict(source_binding=binding,RGB_sha256=rgbsha,
                actual_source_top_uv=hint),source_instances=instances,sample_spacing_px=8)
            top_evidence.extend(fresh['source_material_role_evidence'])
        # Actual connected blue pixels provide search seeds where an old
        # issuer omitted one side. They are NEVER physical role authority.
        for label,st in enumerate(blue_stats):
            if label==0 or st[2]<48 or st[2]<4*st[3]:continue
            left,upper,width,height=map(int,st[:4]);hint=[]
            for x in range(left,left+width,8):
                yy=np.flatnonzero(blue_labels[:,x]==label)
                if len(yy):hint.append([x,int(yy[len(yy)//2])])
            if not hint:continue
            fresh=read_source_board_rail_material_profiles(image,dict(source_binding=binding,RGB_sha256=rgbsha,
                actual_source_top_uv=np.asarray(hint,float)),source_instances=instances,sample_spacing_px=1)
            top_evidence.extend(fresh['source_material_role_evidence'])
        for e in top_evidence:
            a=np.asarray(e['actual_source_material_entry_uv']);b=np.asarray(e['actual_source_material_exit_uv'])
            core=np.rint((a+b)/2).astype(int);label=int(blue_labels[np.clip(core[1],0,h-1),np.clip(core[0],0,w-1)])
            st=blue_stats[label];band_width=float(np.linalg.norm(b-a))
            coherent=bool(label and st[2]>=48 and st[2]>=4*st[3] and st[3]<=max(48.,2.5*band_width))
            e['actual_blue_connected_material_context']=dict(label=label,bbox_xywh=st[:4].tolist(),
                observed_blue_normal_width_px=band_width,coherent_thin_material_component=coherent,
                component_is_context_not_semantic_identity=True)
        def body_pair(mu):
            # The pair is local actual RGB support, not a fitted y-extremum or
            # a model silhouette. A missing band stays unresolved.
            options=[];rejected={}
            for ti,e in enumerate(top_evidence):
                if e['competing_current_source_person_support'] or e['selected_optical_peak_at_profile_search_boundary']or not e['actual_blue_connected_material_context']['coherent_thin_material_component']:continue
                tu=np.asarray(e['actual_source_material_entry_uv']);tx=np.asarray(e['actual_source_material_exit_uv'])
                if abs(tu[0]-mu[0])>12:continue
                ids=np.flatnonzero(fmask&np.isfinite(fuv).all(1)&(abs(fuv[:,0]-tu[0])<=8))
                for fi in ids:
                    if not tx[1]+4<entry[fi,1]<fuv[fi,1]:continue
                    x=int(round(tu[0]));a=int(np.ceil(max(tx[1],np.asarray(e['exit_observed_same_signed_material_transition']['actual_source_optical_interval_UV'])[:,1].max())+2));b=int(np.floor(entry[fi,1]-2))
                    if b-a<10:continue
                    sample=image[a:b,max(0,x-1):min(w,x+2)].astype(float)/255
                    excluded=mask[a:b,max(0,x-1):min(w,x+2)]
                    neutral=(np.ptp(sample,axis=2)<=CONFIG['body_neutral_chroma_range'])&(sample.mean(2)>=CONFIG['body_luminance_min'])
                    print_support=np.zeros_like(excluded)
                    used_print=[]
                    for pm in print_materials:
                        x0,y0,x1,y1=pm['bounds']
                        if x0<=x<x1 and a+3<y0<y1<b-3:
                            print_support[max(0,int(y0)-a):min(b-a,int(y1)-a)]=True
                            used_print.append(pm['object_id'])
                    visible=~print_support
                    fraction=float(neutral[visible].mean())if visible.any()else 0.
                    upper_white=float(neutral[:min(12,len(neutral))].mean())
                    lower_white=float(neutral[-min(12,len(neutral)):].mean())
                    if min(upper_white,lower_white)<0.50:
                        rejected['missing_white_body_flanks_around_actual_print']=rejected.get('missing_white_body_flanks_around_actual_print',0)+1;continue
                    # A second observed blue plateau inside the alleged white
                    # body is competing top support (stands/rail cannot jump
                    # over the real board cap). Keep it unresolved.
                    inside_labels=np.unique(blue_labels[a:b,max(0,x-1):min(w,x+2)])
                    own=e['actual_blue_connected_material_context']['label'];competing=[]
                    for lab in inside_labels:
                        if not lab or lab==own:continue
                        st=blue_stats[lab]
                        if st[2]>=48 and st[2]>=4*st[3] and st[3]<=max(48.,2.5*e['actual_source_material_normal_width_px']):competing.append(int(lab))
                    if competing:
                        rejected['intervening_coherent_blue_material_band']=rejected.get('intervening_coherent_blue_material_band',0)+1;continue
                    options.append(((-e['actual_blue_connected_material_context']['bbox_xywh'][2],abs(tu[0]-mu[0])+abs(fuv[fi,0]-mu[0])),ti,int(fi),dict(
                        top_entry_uv=tu,top_exit_uv=tx,yellow_entry_uv=entry[fi],first_iceward_fall_uv=fuv[fi],
                        observed_white_body_fraction=fraction,body_source_y_interval=[a,b],
                        source_exclusion_fraction=float(excluded.mean()),competing_blue_platform=False,actual_print_context_ids=used_print,white_body_above_print_fraction=upper_white,white_body_below_print_fraction=lower_white)))
            return (min(options,key=lambda x:x[0])if options else None),rejected
        membership=[]
        for r,digest in zip(rows,before):
            mu=_mean(r);role=_role(r);status='unresolved';reasons=[];pair,body_rejections=body_pair(mu);ev=dict(body_pair_rejection_counts=body_rejections)
            xi,yi=np.rint(mu).astype(int);excluded=not(0<=xi<w and 0<=yi<h)or bool(mask[yi,xi])
            if excluded:reasons.append('source_candidate_exclusion_or_native_domain_conflict_not_proof_of_invisibility')
            elif pair is None:reasons.append('missing_unambiguous_same_board_blue_white_yellow_ice_support')
            else:
                _,ti,fi,relation=pair;e=top_evidence[ti];ev.update(same_board_material_relation=relation,
                    foot_profile_index=fi,top_profile_index=ti,actual_foot_family_ids=[f['family_id']for f in foot['source_foot_physical_role_families']['families']if fi in f['original_RGB_query_indices']],
                    top_entry_transition=e['entry_observed_same_signed_material_transition'],actual_blue_material_component=e['actual_blue_connected_material_context'],
                    top_exit_transition=e['exit_observed_same_signed_material_transition'],
                    foot_fall_support_endpoints_uv=fend[fi],foot_profile_normal=fn[fi],physical_material_direction='positive original normal: warm-yellow to ice; B-(R+G)/2 increasing',
                    foot_profile_entry_index=int(_np(foot['same_material_band_pair_evidence']['entry_index'])[fi]),
                    foot_profile_exit_index=int(_np(foot['same_material_band_pair_evidence']['exit_index'])[fi]))
                if role=='board.contact':
                    if _support_contains(mu,fend[fi],fn[fi]):
                        status='pass';reasons.append('unchanged_mean_in_same_strip_first_iceward_material_fall_support')
                    elif relation['top_exit_uv'][1]<mu[1]<relation['yellow_entry_uv'][1]-2:
                        status='wrong_role';reasons.append('unchanged_mean_inside_board_body_or_print_above_observed_yellow_strip')
                    elif mu[1]>max(fend[fi,:,1])+2:
                        status='wrong_role';reasons.append('unchanged_mean_below_observed_first_iceward_material_fall')
                    else:reasons.append('mean_not_in_observed_lower_material_transition_support')
                elif role=='board.top':
                    en=e['entry_observed_same_signed_material_transition']['actual_source_optical_interval_UV']
                    ex=e['exit_observed_same_signed_material_transition']['actual_source_optical_interval_UV']
                    normal=np.array([0.,1.])
                    if e['actual_distinct_entry_material_transition_competing']:
                        reasons.append('distinct_blue_entry_transitions_unresolved_for_upper_edge')
                    elif _support_contains(mu,en,normal,xradius=12):
                        status='pass';reasons.append('unchanged_mean_in_observed_entering_blue_upper_transition')
                    elif _support_contains(mu,ex,normal,xradius=12):
                        status='wrong_role';reasons.append('blue_to_white_internal_exit_is_not_user_blue_upper_edge')
                    else:reasons.append('mean_not_in_observed_entering_blue_upper_transition')
            if semantic is not None and 0<=xi<w and 0<=yi<h:
                ev['frozen_model_class_at_mean_diagnostic_only']=int(np.squeeze(semantic)[yi,xi])
            item=dict(schema=SCHEMA,source_row_id=r['source_row_id'],native_pts=pts,shot_id=sid,
                current_uv=mu.tolist(),current_measurement_sha256=digest,status=status,material_role=role,
                reason_codes=reasons,actual_material_evidence=_plain(ev),source_recipe_sha256=self.recipe_sha256,
                source_pixels_unmodified=True,old_flags_or_model_navigation_used_as_identity=False,
                unknown_physical_role_admitted=False,evidence_refs=[])
            membership.append(item)
        from .feedback_material_guard import check_retrieved_material_order
        regression_check = check_retrieved_material_order(membership, authority, _support_contains)
        if before!=[measurement_sha256(r,pts)for r in rows]:raise RuntimeError('Qualification changed the original source')
        summary=dict(schema=SCHEMA,version=VERSION,source_binding=binding,actual_native_RGB_sha256=rgbsha,
            original_mask_ref=maskref,source_recipe_sha256=self.recipe_sha256,configuration=CONFIG,
            frozen_module_refs=self.refs,frozen_model_usage=vision_usage,material_role_authority=authority,
            feedback_material_order_regression=regression_check,
            physical_membership_rows=membership,counts={s:sum(r['status']==s for r in membership)for s in ('pass','wrong_role','unresolved')},
            annotations_or_world_projection_used_as_measured_pixels=False,old_source_certificates_reused=False)
        evidence_dir=context.get('evidence_directory')
        if evidence_dir:
            out=Path(evidence_dir);out.mkdir(parents=True,exist_ok=True);path=out/(str(pts)+'_material_profiles.npz')
            if path.exists():raise ValueError('Immutable frame evidence already exists')
            arrays={}
            if foot:
                pp=foot['original_RGB_profiles'];arrays.update(foot_RGB_profile=_np(pp['rgb_profile']),
                    foot_sampled_native_uv=_np(pp['sampled_native_uv']),foot_valid=_np(pp['valid_sample_mask'])if 'valid_sample_mask'in pp else np.ones(_np(pp['rgb_profile']).shape[:2],bool),
                    foot_primary=fmask,foot_entry_uv=entry,foot_first_fall_uv=fuv,foot_first_fall_support=fend)
            if rail:
                for k,v in rail['actual_original_RGB_profiles'].items():arrays['top_'+k]=np.asarray(v)
            np.savez_compressed(path,**arrays);ref=dict(path=str(path),sha256=file_sha256(path),role='one_actual_RGB_material_profile_evidence_per_exposure')
            for r in membership:r['evidence_refs']=[ref]
            summary['frame_profile_evidence']=ref
        evidence_refs = [summary['frame_profile_evidence']] if summary.get('frame_profile_evidence') else []
        summary['source_feedback'] = feedback.finish(image, dict(counts=summary['counts'],
            source_recipe_sha256=self.recipe_sha256, configuration=CONFIG,
            executed_regression=regression_check,
            mean_or_window_changed=False, physical_membership_status_by_row={r['source_row_id']:r['status'] for r in membership},
            original_decode_receipt=receipt), tool='BoardMaterialRoleRequalifier.evaluate',
            current_evidence=evidence_refs, observation_ids=[r['source_row_id'] for r in rows])
        return summary

@lru_cache(maxsize=1)
def _engine(project,device,use_frozen_visual_model):
    return BoardMaterialRoleRequalifier(project,device=device,use_frozen_visual_model=use_frozen_visual_model)

def reevaluate_current_board_source_rows(rgb,mask,rows,context,*,project=None,device='cuda',use_frozen_visual_model=True):
    return _engine(str(project or Path(__file__).resolve().parents[4]),device,use_frozen_visual_model).evaluate(rgb,mask,rows,context)
