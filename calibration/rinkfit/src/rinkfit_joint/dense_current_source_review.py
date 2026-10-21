"""Recheck actual dense findings with original source profiles and current domains.

The numeric packet normal is intentionally not a source measurement. This
consumer recovers the original pixel operator, reads only requested native
exposures, and keeps source UV and all qualification changes at a later batch
boundary. Reproducing a material profile alone does not certify world identity.
"""
from pathlib import Path
from collections import defaultdict
import gzip,json,hashlib
import numpy as np
from .io_utils import read_json,read_jsonl,object_sha256,file_sha256
from .dense_fit_feedback import _one_frame,reference,_checked,_new_json
from .dense_joint_packet import _array

ROOT=Path(__file__).resolve().parents[4]

def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v)for v in value]
    return value


def _original_records(subjects):
    """Read exact JSON pointers, indexed gzip members or one shared gzip pass."""
    result={};line_files=defaultdict(list);json_files={};members={};evidence={}
    for index,row in subjects.items():
        ref=row['source_pixel_or_annotation_ref'];path=Path(ref['file']).resolve()
        if 'jsonl_line_0based'in ref:line_files[path].append((index,int(ref['jsonl_line_0based'])));continue
        if 'json_pointer'in ref:
            if path not in json_files:json_files[path]=read_json(path);evidence[str(path)]=reference(path)
            item=json_files[path]
            for key in ref['json_pointer'].strip('/').split('/'):
                item=item[int(key)]if isinstance(item,list)else item[key.replace('~1','/').replace('~0','~')]
            result[index]=item;continue
        if 'gzip_member_offset'in ref:
            key=str(path),int(ref['gzip_member_offset']),int(ref['gzip_member_bytes']),ref['gzip_member_sha256']
            if key not in members:
                with path.open('rb')as stream:stream.seek(key[1]);blob=stream.read(key[2])
                if hashlib.sha256(blob).hexdigest()!=key[3]:raise ValueError('Original indexed gzip source member changed')
                members[key]=gzip.decompress(blob).decode('utf8').splitlines()
                evidence['member:'+str(key)]=dict(kind='exact_gzip_member',file=str(path),offset_bytes=key[1],bytes=key[2],sha256=key[3])
            result[index]=json.loads(members[key][int(ref['member_jsonl_line_0based'])]);continue
        raise ValueError('No actual original profile locator for dense source '+str(index))
    for path,wanted in line_files.items():
        byline=defaultdict(list)
        for index,line in wanted:byline[line].append(index)
        with gzip.open(path,'rt',encoding='utf8')as stream:
            for line,text in enumerate(stream):
                if line in byline:
                    value=json.loads(text)
                    for index in byline.pop(line):result[index]=value
                    if not byline:break
        if byline:raise ValueError('Original canonical source line is missing')
        evidence[str(path)]=reference(path)
    for index,row in subjects.items():
        original=result[index]
        if original.get('uv')!=row['uv']or int(original.get('native_pts',-1))!=int(row['native_pts'])or original.get('shot_id')!=row['shot_id']or original.get('feature_id')!=row['feature_id']:
            raise ValueError('Original profile locator does not reproduce the current original UV/ID/exposure')
        original_id=original.get('source_row_id',original.get('point_id'))
        if original_id!=row['source_point_id']:raise ValueError('Original source point identity differs')
    return result,list(evidence.values())


def _current_inputs(config,subjects):
    wanted={(r['shot_id'],int(r['native_pts']))for r in subjects.values()};out={};bindings=read_json(_checked(config['storage_bindings']))
    for sid in ('s05','s06'):
        if not any(s==sid for s,p in wanted):continue
        ref=bindings['generations'][sid]['consumer_bindings']
        for frame in read_jsonl(ref):
            key=sid,int(frame['native_pts'])
            if key in wanted:out[key]=dict(frame,consumer_binding_ref=ref)
    first={k for k in wanted if k[0]in('s02','s03','s04')}
    if first:
        for frame in read_jsonl(_checked(config['s02_s04_mask_index'])):
            key=frame['shot_id'],int(frame['native_pts'])
            if key in first:out[key]=dict(frame,mask=dict(file=frame['mask_file'],sha256=frame['mask_sha256']),consumer_binding_ref=config['s02_s04_mask_index'])
    if set(out)!=wanted:raise ValueError('Actual current per-exposure source-domain binding is missing')
    return out


def _mask_inputs(frame,shape):
    import cv2
    masks={};refs=[]
    for key in ('mask','graphics_mask','source_unknown_exclusion_mask_ref'):
        ref=frame.get(key)
        if ref is None:continue
        path=_checked(ref);mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=shape:raise ValueError('Current source mask raster differs')
        masks[key]=mask>0;refs.append(ref)
    if 'mask'not in masks:raise ValueError('Actual current person mask is mandatory')
    excluded=np.zeros(shape,bool)
    for key,mask in masks.items():
        if key=='graphics_mask':excluded|=mask
        else:excluded|=cv2.dilate(mask.astype('u1'),np.ones((17,17),'u1'))>0
    return masks,excluded,refs


def _normal(original):
    normal=original.get('pixel_support',{}).get('normal',original.get('pixel_support',{}).get('profile_normal'))
    if normal is None:
        interval=original.get('pixel_support',{}).get('position_interval_endpoints_uv')
        if interval is not None:normal=np.asarray(interval[1])-interval[0]
    normal=np.asarray(normal,float)
    if normal.shape!=(2,)or not np.isfinite(normal).all()or np.linalg.norm(normal)<=0:raise ValueError('Original source profile normal is missing; numeric packet normal is not a substitute')
    return normal/np.linalg.norm(normal)


def _original_profile_policy(subject,original):
    name=subject['feature_id'].split(':')[0].removeprefix('ice.');kind=subject['source_kind'];nav=original.get('navigation',{})
    if name in('B1','B2','C'):return dict(radius_px=190.,family='confirmed_wide_paint_profile',branch_coherence_required='complete_band' in kind)
    if subject['shot_id']in('s05','s06'):
        return dict(radius_px=64. if name in('A1','A2','A3','A4')else 36.,family='dense56_exact_or_final_source_domain',branch_coherence_required='complete_band' in kind)
    if 'current_grayscale_paired_RGB_edges'in kind:return dict(radius_px=64.,family='AB_gray_v2',branch_coherence_required=False)
    if 'weak_current_paired_RGB_edges'in kind:return dict(radius_px=28.,family='weak_C0_v1',branch_coherence_required=False)
    if nav.get('physical_navigation_origin')in('cross_shot_source_curve_navigation_only','8a28_named_scene_geometry_navigation_only','same_shot_source_curve'):
        return dict(radius_px=48.,family='read_missing_8a_ice_v2',branch_coherence_required=False)
    return dict(radius_px=28.,family='read_dense_ice_v4_or_goal_material_v1',branch_coherence_required=False)


def _actual_material_read(rgb,original,subject,normal):
    from rinkspace.vision_source_shapes import _MaterialProfileFrame,read_connected_source_material_profiles
    from rinkspace.vision_programme_ice_review import _batched_actual_profiles,_actual_profile_candidates
    support=original.get('pixel_support',{});uv=np.asarray(subject['uv'],float)
    if subject['physical_surface']=='ice':
        signal=support.get('material_signal');transition=support.get('material_transition')
        if signal not in('dark','red','blue'):raise ValueError('Original material signal/operator is unavailable')
        definition=subject['measurement_definition'];operator='paint_center'if definition=='width_midpoint_between_both_material_edges'else'material_interface'
        if operator=='material_interface'and transition not in('dark_to_light','light_to_dark'):raise ValueError('Original ring/material transition direction is unavailable')
        query=uv-float(support.get('native_shift_px',0.))*normal
        policy=_original_profile_policy(subject,original);radius=policy['radius_px']
        binding=dict(source_sha256=subject['source_video_sha256'],source_pts=subject['native_pts'],time_base='1/12800',native_hw=list(rgb.shape[:2]),shot_id=subject['shot_id'])
        frame=_MaterialProfileFrame(rgb,binding);off,prof,smooth,valid=_batched_actual_profiles(frame,query[None],normal[None],np.array([[-radius,radius]]),np.array([signal]));count=int(valid[0].sum())
        choices=_actual_profile_candidates(query,normal,off[0,:count],prof[0,:count],smooth[0,:count],operator=operator,signal=signal,transition=transition,with_quantiles=True)
        candidates=[dict(source_uv=q,paired_material_sides=sides,source_profile_disposition=bad,interface_quartiles=quartiles,original_source_distance_px=float(np.linalg.norm(q-uv)))for q,sides,bad,quartiles in choices]
        return dict(reader='vision_programme_ice_review._batched_actual_profiles+_actual_profile_candidates',original_profile_policy=policy,original_estimator_replay_claimed=not policy['branch_coherence_required'],original_operator=operator,original_material_signal=signal,original_transition=transition,actual_source_query_uv=query,actual_source_profile_normal=normal,actual_source_profile_interval_px=[-radius,radius],candidates=candidates,original_source_UV_retained=True,source_candidate_installed=False)
    binding=dict(source_sha256=subject['source_video_sha256'],source_pts=subject['native_pts'],time_base='1/12800',native_hw=list(rgb.shape[:2]),shot_id=subject['shot_id'])
    # Board/print transitions remain normal-only source materials. They are
    # never sent to a fixed-world-corner estimator or promoted to 2D anchors.
    radius=3. if subject['source_kind']=='board_ink_material_half_transition'else 28.
    readings={direction:read_connected_source_material_profiles(rgb,binding=binding,search_points_uv=[uv],normals=[normal],search_intervals_px=[[-radius,radius]],transition=direction,keep_profiles=False)for direction in('dark_to_light','light_to_dark')}
    return dict(reader='vision_source_shapes.read_connected_source_material_profiles',operator_role='actual_normal_material_candidates_diagnostic',original_estimator_replay_claimed=False,original_source_definition=subject['measurement_definition'],actual_source_profile_normal=normal,readings=readings,normal_only_source_not_world_corner=True,original_source_UV_retained=True,source_candidate_installed=False)


def review_dense_findings(*,packet,findings_document,config,output,cache_directory):
    """Execute only real bound findings; residuals trigger reads, never decisions."""
    import av,cv2,sys
    runtime=dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__,pyav=av.__version__)
    cfg=read_json(config)if not isinstance(config,dict)else config
    doc=read_json(findings_document);findings=doc.get('findings',[])
    if not findings:raise ValueError('An actual nonempty dense finding document is required')
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True);cache=Path(cache_directory).resolve();cache.mkdir(exist_ok=True,parents=True)
    packet_path=Path(packet).resolve();packet_doc=read_json(packet_path);qualification=read_json(_checked(packet_doc['dense_qualification_binding']));current_parts=qualification['qualified_source_parts']
    subjects={};indices={};frames={} 
    for finding in findings:
        if finding.get('residual_is_truth_label')is not False:raise ValueError('Source review cannot consume a truth-labelled model residual')
        if not any(finding['source_rows']==part['reference']for part in current_parts):raise ValueError('Finding is bound to an older semantic source stream than the current packet')
        s=finding['subject'];key=s['shot_id'],int(s['native_pts'])
        if key not in frames:frames[key]=_one_frame(finding['source_rows'],*key)[0]
        current=frames[key][s['selected_row_id']]
        if current!=finding['qualification_snapshot']:raise ValueError('Finding is not bound to its actual current source generation')
        rid=s['selected_row_id'];subjects[rid]=dict(current,source_video_sha256=finding['source']['video_sha256']);indices[rid]=finding
    source=_checked(cfg['source_video']);originals,original_evidence=_original_records(subjects);inputs=_current_inputs(cfg,subjects)
    code=[reference(__file__),reference(ROOT/'src/rinkspace/vision_programme_ice_review.py'),reference(ROOT/'src/rinkspace/vision_source_shapes.py')]+list(cfg.get('original_reader_policy_sources',[]))
    for ref in code:_checked(ref)
    facts=list(cfg.get('measurement_facts',[cfg['measurement_roles']]))
    for ref in facts+([cfg['storage_bindings'],cfg['s02_s04_mask_index']]):_checked(ref)
    selected_ids=set(subjects);packet_indices={ident:i for i,ident in enumerate(packet_doc['source_row_ids'])if ident in selected_ids}
    if set(packet_indices)!=selected_ids:raise ValueError('Actual finding source ID is absent from the physical packet')
    packet_uv=_array(packet_doc['arrays']['uv'],packet_path.parent);packet_pts=_array(packet_doc['arrays']['native_pts'],packet_path.parent)
    for rid,row in subjects.items():
        i=packet_indices[rid]
        if list(packet_uv[i])!=row['uv']or int(packet_pts[i])!=int(row['native_pts']):raise ValueError('Finding original UV/PTS differ from actual physical packet')
    groups=defaultdict(list);dependencies={};targets={};reused=[]
    for rid,row in subjects.items():
        key=row['shot_id'],int(row['native_pts']);input_binding=inputs[key]
        maskrefs=[input_binding[k]for k in('mask','graphics_mask','source_unknown_exclusion_mask_ref')if input_binding.get(k)is not None]
        for ref in maskrefs:_checked(ref)
        snapshot={k:v for k,v in row.items()if k not in('source_fit_eligible','physical_instance_qualification','qualification_reason_code','qualification_review_id')}
        deps=dict(reader_code_sha256=object_sha256(code),reader_config_sha256=object_sha256(dict(operator='original_source_normal_and_definition',world_corner_for_dense_material=False,actual_runtime=runtime)),source_binding_sha256=object_sha256(dict(shot_id=key[0],native_pts=key[1],RGB=input_binding['source_RGB_sha256'],BGR=input_binding['source_BGR_sha256'],masks=maskrefs,actual_source_domain=input_binding)),observation_snapshot_sha256=object_sha256(dict(source=snapshot,original=originals[rid])),effective_facts_sha256=object_sha256(facts),navigation_state_sha256='not_used_current_model')
        digest=object_sha256(dict(source_point_id=row['source_point_id'],dependencies=deps));target=cache/(digest+'.json');dependencies[rid]=deps;targets[rid]=target
        if target.is_file():
            saved=read_json(target)
            if saved['review_dependencies']!=deps:raise ValueError('Exact dense reader cache identity collision')
            reused.append(dict(selected_row_id=rid,review=reference(target),exact_same_source_operator_UV_RGB_mask_definition_verified=True,source_pixels_decoded_this_review=0));continue
        groups[key].append(rid)
    reviews=[];wanted={p for s,p in groups};seen=set()
    if not wanted:
        receipt=dict(schema='rinkfit.current_dense_source_review_dispatch.v1',packet=reference(packet),real_findings_document=reference(findings_document),consumer=reference(__file__),reviews=[],reused=reused,actual_native_exposures=0,source_UV_qualification_or_packet_mutated=False,source_only_review=True,world_entity_accuracy_accepted=False)
        _new_json(out/'receipt.json',receipt);return receipt
    with av.open(str(source))as container:
        stream=container.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Actual native source clock differs')
        container.seek(min(wanted),stream=stream,backward=True)
        for frame in container.decode(stream):
            pts=int(frame.pts)
            if pts>max(wanted):break
            if pts not in wanted:continue
            key=next(k for k in groups if k[1]==pts);input_binding=inputs[key];rgb=frame.to_ndarray(format='rgb24');bgr=frame.to_ndarray(format='bgr24')
            rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();bgrsha=hashlib.sha256(bgr.tobytes()).hexdigest()
            if rgbsha!=input_binding['source_RGB_sha256']or bgrsha!=input_binding['source_BGR_sha256']:raise ValueError('Actual pixels differ from current accepted per-exposure source inputs')
            masks,excluded,maskrefs=_mask_inputs(input_binding,rgb.shape[:2]);seen.add(key)
            for rid in groups[key]:
                row=subjects[rid];original=originals[rid]
                deps=dependencies[rid];target=targets[rid]
                pixel=np.rint(row['uv']).astype(int);visibility_excluded=bool(excluded[pixel[1],pixel[0]])
                try:
                    normal=_normal(original);reading=_actual_material_read(rgb,original,row,normal)
                except (ValueError,KeyError,IndexError)as error:
                    reading=dict(status='pending_exact_original_source_operator',error_type=type(error).__name__,reason=str(error),original_source_UV_retained=True,source_candidate_installed=False)
                checks={name:dict(status='unknown',reason='Actual material replay alone does not certify this dimension')for name in('semantic_class','physical_instance','measurement_operator','visibility','position')}
                checks['measurement_operator']=dict(status='original_operator_and_definition_recovered',definition=row['measurement_definition'],source_normal_from_original_pixels=True,numeric_packet_normal_used=False)
                checks['position']=dict(status='original_source_UV_retained',source_UV=row['uv'],replacement_or_precision_issued=False)
                checks['visibility']=dict(status='actual_current_source_masks_read',source_center_excluded=visibility_excluded,new_source_visibility_qualification_issued=False)
                result=dict(schema='rinkfit.current_dense_actual_material_review.v1',selected_row_id=rid,source_point_id=row['source_point_id'],shot_id=key[0],native_pts=pts,source_current_semantic_ref=indices[rid]['source_rows'],review_dependencies=deps,actual_material_read=reading,checks=checks,original_source_evidence=original_evidence,current_input_binding=input_binding,actual_RGB_sha256=rgbsha,actual_BGR_sha256=bgrsha,source_UV_changed=False,qualification_changed=False,material_candidate_or_world_identity_installed=False,source_pixels_decoded_this_review=1,model_residual_used_for_qualification=False,source_reader_version='original_source_profile_v1')
                _new_json(target,_plain(result));reviews.append(dict(selected_row_id=rid,review=reference(target)))
    if seen!=set(groups):raise ValueError('Not every actual finding exposure was decoded')
    receipt=dict(schema='rinkfit.current_dense_source_review_dispatch.v1',packet=reference(packet),real_findings_document=reference(findings_document),consumer=reference(__file__),reviews=reviews,reused=reused,actual_native_exposures=len(seen),source_UV_qualification_or_packet_mutated=False,source_only_review=True,world_entity_accuracy_accepted=False)
    _new_json(out/'receipt.json',receipt);return receipt


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet","findings_document","config","output","cache_directory"):parser.add_argument(name)
    args=parser.parse_args();result=review_dense_findings(**vars(args))
    print(json.dumps(dict(actual_native_exposures=result["actual_native_exposures"],reviews=len(result["reviews"]),reused=len(result["reused"]),source_UV_qualification_or_packet_mutated=False),ensure_ascii=False))

if __name__=="__main__":main()
