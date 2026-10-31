"""Full S07 Ad9 source candidates, always pending, with exact profile telemetry.

Direct RootSIFT is attempted first. Existing bidirectional source-graph transfer
is only a fallback and every current frame must pass the existing whole-pattern
check. Affine/similarity comparisons are diagnostics, never source UV or native
updates. No component default, qualification policy or fitted state is changed.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np

import capture_independent_badge as badge
import capture_independent_prints as prints
import capture_independent_materials as materials
from prepare_independent_navigation import frames
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from rinkspace.geometry import project
from rinkspace.session.native_shape_guidance import register_native_shapes
from rinkspace.path_policy import guard_project_write


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(materials.plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def transfer_candidates(pts,anchors,last_verified,graph_index,cumulative,inverse,bad_prefix):
    before=[p for p in anchors if p<=pts];after=[p for p in anchors if p>pts]
    selected=([max(before)]if before else[])+([min(after)]if after else[])
    parents={p:(anchors[p],'bidirectional_source_graph_anchor')for p in selected}
    if last_verified is not None:
        p,h=last_verified
        parents[p]=(h,'current_adjacent_verified_pattern'if abs(graph_index[pts]-graph_index[p])==1 else'previous_verified_pattern')
    i=graph_index[pts];result=[]
    for p,(hh,basis)in parents.items():
        j=graph_index[p];lo,hi=sorted([i,j])
        if bad_prefix[hi+1]-bad_prefix[lo+1]:continue
        h=cumulative[i]@inverse[j]@hh
        if not np.isfinite(h).all()or h[2,2]==0:continue
        h=h/h[2,2]
        if np.linalg.det(h[:2,:2])<=0:continue
        result.append((p,basis,h))
    return result


def read_profile_proof(rgb,search,exclusion,binding,*,material_signal='red'):
    """Observe original profiles/components without altering the existing reader."""
    module=badge.panel_reader
    original_batch=module._batched_actual_profiles;original_candidates=module._actual_profile_candidates
    previous_profiler=sys.getprofile();target=module.read_panel_boundary.__code__
    arrays={};all_rgb=[];candidate_query=[];call_index=0
    def batched(frame,q,n,interval,signals):
        result=original_batch(frame,q,n,interval,signals)
        arrays.update(query_uv=np.array(q,copy=True),query_normal=np.array(n,copy=True),query_interval=np.array(interval,copy=True),
            profile_offsets=np.array(result[0],copy=True),actual_luminance_profiles=np.array(result[1],copy=True),
            smoothed_profiles=np.array(result[2],copy=True),profile_valid_offsets=np.array(result[3],copy=True))
        return result
    def candidates(*args,**kwargs):
        nonlocal call_index
        result=original_candidates(*args,**kwargs)
        for value in result:all_rgb.append(value);candidate_query.append(call_index)
        call_index+=1
        return result
    def observe_return(frame,event,value):
        if event=='return'and frame.f_code is target:
            local=frame.f_locals
            if 'lo'in local and'hi'in local:
                lo,hi=np.array(local['lo']),np.array(local['hi'])
                arrays.update(source_roi_lo=lo,source_roi_hi=hi,source_rgb_roi=np.array(rgb[lo[1]:hi[1],lo[0]:hi[0]],copy=True),
                    source_exclusion_roi=np.array(exclusion[lo[1]:hi[1],lo[0]:hi[0]],copy=True))
                for name in('component','material_component','available','binary','color_seed'):
                    if name in local and local[name]is not None:arrays['reader_'+name+'_roi']=np.array(local[name],copy=True)
                if 'threshold'in local:arrays['reader_dark_threshold']=np.array(local['threshold'])
    module._batched_actual_profiles=batched;module._actual_profile_candidates=candidates;sys.setprofile(observe_return)
    try:public,record,raw=badge.read_with_pending_exterior(rgb,search,exclusion,binding,material_signal=material_signal)
    finally:
        sys.setprofile(previous_profiler);module._batched_actual_profiles=original_batch;module._actual_profile_candidates=original_candidates
    arrays.update(source_exclusion_packbits=np.packbits(exclusion),source_exclusion_shape=np.array(exclusion.shape),search_uv=np.array(search,copy=True))
    if all_rgb:
        arrays.update(all_RGB_candidate_uv=np.array([r[0]for r in all_rgb]),all_RGB_candidate_sides=np.array([r[1]for r in all_rgb]),
            all_RGB_candidate_bad=np.array([r[2]for r in all_rgb]),all_RGB_candidate_quantiles=np.array([r[3]for r in all_rgb]),
            all_RGB_candidate_query=np.array(candidate_query,np.int64))
    if raw:
        arrays.update(exterior_candidate_uv=np.array([r['uv']for r in raw]),exterior_candidate_quantiles=np.array([r['actual_source_quantiles']for r in raw]),
            exterior_candidate_sides=np.array([r['reader_candidate_sides']for r in raw]),exterior_candidate_flags=np.array([r['candidate_flags']for r in raw]))
    return public,record,raw,arrays


def shape_diagnostics(raw,search):
    result=dict(role='diagnostic source-shape organization only',all_source_candidates_remain_pending=True,
        source_UV_or_native_changed=False,low_residual_is_not_correctness_or_identity=True)
    if len(raw)<3:return dict(result,status='insufficient_actual_source_candidates')
    observed=np.array([r['uv']for r in raw]);before=observed.copy();curve=np.array(search,copy=True)
    for family in('similarity','affine'):
        try:
            fit=register_native_shapes(observed,np.full(len(observed),'badge_outer'),{'badge_outer':curve},transform_model=family)
            singular=np.linalg.svd(fit.affine[:2,:2],compute_uv=False)
            result[family]=dict(rms_distance_px=fit.rms_distance_px,initial_rms_distance_px=fit.initial_rms_distance_px,
                source_search_affine=fit.affine.tolist(),distance_px=fit.matches.distance_px.tolist(),
                singular_values=singular.tolist(),axis_ratio=float(singular[0]/singular[-1]),iterations=fit.iterations,stop_reason=fit.stop_reason,
                closest_UV_is_diagnostic_not_measurement=True)
        except ValueError as error:result[family]=dict(status='source_shape_diagnostic_unresolved',reason=str(error))
    if not np.array_equal(observed,before)or not np.array_equal(curve,search):raise ValueError('Shape diagnostic changed source or search UV')
    return result


def main(args):
    cv2.setNumThreads(2);root,out=Path(args.root).resolve(),guard_project_write(args.output)
    if out.exists():raise FileExistsError('Preserve existing full badge capture: '+str(out))
    if out.is_relative_to(badge.PROJECT.resolve()):raise ValueError('Source capture artifacts belong outside OneDrive')
    pilot=args.reference_root or root/'badge_pilot';pilot_binding=read(pilot/'binding.json');meta=read(pilot/'reference.json')
    entity=meta.get('ad_id','Ad9');material_signal=meta.get('material_signal','red');qualify=bool(getattr(args,'qualify_approved_exterior',False))
    source_outline=bool(getattr(args,'source_outline_navigation',False))
    if not source_outline and sha(pilot/'frozen_state.npz')!=pilot_binding['fixed_state_sha256']:raise ValueError('Fixed native reference snapshot changed')
    index=read(root/'index.json');shot=next(s for s in index['shots']if s['shot_id']=='s07')
    source=Path(index['source']['resolved_path'])
    if sha(source)!=index['source']['actual_sha256']:raise ValueError('Actual original source video changed')
    ref_image=cv2.imread(meta['source_image']);ref_mask=cv2.imread(str(pilot/'reference_mask.png'),0)>0
    if hashlib.sha256(ref_image.tobytes()).hexdigest()!=meta['source_BGR_sha256']:raise ValueError('Independent reference RGB changed')
    features=np.load(pilot/'reference_rootsift.npz',allow_pickle=False);reference_features=(features['feature_uv'],features['descriptors'])
    reference=dict(source_image=ref_image,reference_mask=ref_mask)
    geometry=np.load(pilot/'reference_geometry.npz',allow_pickle=False);reference_search=geometry['reference_search_uv'];source_reference_arrays=None
    if source_outline:
        ref_exclusion,_=materials.source_exclusion(root,'s07',meta['pts'])
        _,_,_,source_reference_arrays=read_profile_proof(cv2.cvtColor(ref_image,cv2.COLOR_BGR2RGB),reference_search,ref_exclusion,dict(source_sha256=index['source']['actual_sha256'],source_pts=meta['pts'],time_base='1/12800',shot_id='s07'),material_signal=material_signal)
        component=np.asarray(source_reference_arrays['reader_component_roi'],'uint8');contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours:raise ValueError('Approved source reference has no public material exterior')
        contour=max(contours,key=cv2.contourArea)[:,0]+source_reference_arrays['source_roi_lo'];reference_search=np.vstack((contour,contour[0])).astype(float)
    context_margin=int(getattr(args,'descriptor_context_margin',0))
    if context_margin:
        if context_margin<0:raise ValueError('Context margin cannot be negative')
        ref_exclusion,_=materials.source_exclusion(root,'s07',meta['pts']);x0,y0,x1,y1=meta['roi_xyxy_inclusive'];context_mask=np.zeros((1080,1920),'uint8');context_mask[max(0,y0-context_margin):min(1080,y1+context_margin+1),max(0,x0-context_margin):min(1920,x1+context_margin+1)]=255;context_mask[ref_exclusion]=0;reference_features=descriptors(ref_image,context_mask)
    graph_path=root/'s07/navigation/source_graph.npz';graph=dict(np.load(graph_path,allow_pickle=False))
    expected=np.arange(shot['processing']['first_pts'],shot['processing']['stop_pts_exclusive'],256)
    if not np.array_equal(graph['pts'],expected):raise ValueError('Current source graph native PTS differs')
    cumulative=np.empty((len(expected),3,3));cumulative[0]=np.eye(3)
    for i,step in enumerate(graph['adjacent_h'],1):cumulative[i]=step@cumulative[i-1];cumulative[i]/=cumulative[i,2,2]
    inverse=np.linalg.inv(cumulative);graph_index={int(p):i for i,p in enumerate(expected)};bad_prefix=np.r_[0,np.cumsum(~graph['supported'])]
    first_pts=int(expected[0])if getattr(args,'first_pts',None)is None else int(args.first_pts);last_pts=int(expected[-1])if getattr(args,'last_pts',None)is None else int(args.last_pts)
    if first_pts not in graph_index or last_pts not in graph_index or first_pts>last_pts:raise ValueError('Invalid requested current native source interval')
    selected_expected=expected[(expected>=first_pts)&(expected<=last_pts)]
    anchors={}
    for item in pilot_binding['summaries']:
        proof=read(pilot/f"{item['pts']}_proof.json")
        if proof['source_pattern_check']is not None and proof['source_pattern_check']['supported']:
            anchors[item['pts']]=np.array(proof['H_current_from_reference'])
    if not anchors:raise ValueError('No independently supported badge source anchors')
    source_files={
        'full_adapter':Path(__file__),'badge_adapter':Path(badge.__file__),'current_print_adapter':Path(prints.__file__),
        'fine_source_exclusion_adapter':Path(materials.__file__),'panel_reader':Path(badge.read_panel_boundary.__code__.co_filename),
        'profile_reader':Path(badge.panel_reader._batched_actual_profiles.__code__.co_filename),
        'RootSIFT_and_matcher':Path(descriptors.__code__.co_filename),'source_pattern_checker':Path(prints.check_source_pattern.__code__.co_filename),
        'native_shape_registration':Path(register_native_shapes.__code__.co_filename)}
    script_hashes={name:dict(path=str(p),sha256=sha(p))for name,p in source_files.items()}
    out.mkdir(parents=True);(out/'profiles').mkdir();(out/'proofs').mkdir()
    write(out/'script_hashes.json',script_hashes)
    if source_reference_arrays is not None:np.savez_compressed(out/'actual_source_reference_navigation.npz',reference_search_uv=reference_search,**source_reference_arrays)
    if context_margin:np.savez_compressed(out/'reference_context_RootSIFT.npz',feature_uv=reference_features[0],descriptors=reference_features[1]);cv2.imwrite(str(out/'reference_context_navigation_mask.png'),context_mask)
    binding=dict(source_sha256=index['source']['actual_sha256'],reference=str(pilot/'reference.json'),reference_sha256=sha(pilot/'reference.json'),
        fixed_native_reference_snapshot=str(pilot/'frozen_state.npz'),fixed_native_reference_snapshot_sha256=pilot_binding['fixed_state_sha256'],
        reference_geometry=str(pilot/'reference_geometry.npz'),reference_geometry_sha256=sha(pilot/'reference_geometry.npz'),
        source_graph=str(graph_path),source_graph_sha256=sha(graph_path),script_hashes=script_hashes,native_path_id=meta['native_path_id'],
        fit_eligible=qualify,complete=False,fit_executed=False,native_or_source_pixels_changed=False,
        requested_native_interval=dict(first_pts=first_pts,last_pts=last_pts,frames=len(selected_expected)),
        source_reference_outline_navigation=source_outline,source_reference_component_vertices_are_not_observations=True,
        fixed_native_reference_snapshot_not_read=source_outline,descriptor_context_margin_px=context_margin,
        descriptor_context_is_navigation_only=True,whole_pattern_reference_mask_unchanged=True,
        panel_instance=entity,material_signal=material_signal,only_existing_public_reader_organized_rows_qualified=qualify,
        primary_source_identity_approved_for_requested_instance=qualify,
        source_mask_definition='Exact current fine source_exclusion; full applied mask bitpacked in each profile proof',
        all_actual_candidates_preserved=True,internal_white_letters_and_instance_identity_require_independent_review=True)
    write(out/'binding.json',binding)
    totals=Counter();statuses=Counter();gaps=[];seen=[];last_verified=None;started=time.monotonic();affine_pairs=[]
    preview_pts=set(badge.PILOT_PTS+[int(expected[0]),int(expected[len(expected)//2]),int(expected[-1])]+[int(q['pts'])for q in pilot_binding['summaries']]+list(getattr(args,'proof_pts',None)or[]))
    config=ReuseConfig(min_matches=6,ransac_px=3.)
    with(out/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)as rows_out,(out/'registration.jsonl').open('w',encoding='utf8',buffering=1)as registration,(out/'progress.jsonl').open('w',encoding='utf8',buffering=1)as progress:
      for fi,(pts,image)in enumerate(frames(index,shot)):
        if pts!=int(expected[fi]):raise ValueError('Actual full source exposure sequence differs')
        if pts<first_pts:continue
        if pts>last_pts:break
        seen.append(pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);exclusion,mask_path=materials.source_exclusion(root,'s07',pts)
        image_hash=hashlib.sha256(image.tobytes()).hexdigest();mask_hash=sha(mask_path)
        current=descriptors(image,(~exclusion).astype(np.uint8)*255)
        direct=match_features(reference_features,current,config);h=None;patch=None;basis=None;parent=None;direct_note=dict(matches=0)
        if direct is not None:
            left,right,hh=direct;direct_note=dict(matches=len(left),reference_keypoints=left.tolist(),current_keypoints=right.tolist())
            if np.isfinite(hh).all()and np.linalg.det(hh[:2,:2])>0:
                check=prints.source_patch_check(image,exclusion,meta,reference,hh);direct_note['source_pattern_check']=check
                if check['supported']:h=hh;patch=check;basis='direct_RootSIFT';anchors[pts]=hh
        transfers=[]
        if h is None:
            for source_pts,parent_basis,hh in transfer_candidates(pts,anchors,last_verified,graph_index,cumulative,inverse,bad_prefix):
                check=prints.source_patch_check(image,exclusion,meta,reference,hh)
                transfers.append(dict(parent_pts=source_pts,basis=parent_basis,source_pattern_check=check))
                if check['supported']and(patch is None or check['whole_visible_pattern_correlation']>patch['whole_visible_pattern_correlation']):
                    h=hh;patch=check;basis=parent_basis;parent=source_pts
        pending=[];raw=[];public=[];readout=None;diagnostic=None;profile_path=None;spaced=[]
        status='pending_no_supported_source_pattern'
        if h is not None:
            last_verified=(pts,h.copy());totals['pattern_frames']+=1;totals[basis]+=1
            search=project(h,reference_search)
            if np.isfinite(search).all()and max(np.ptp(search,axis=0))<=10000:
                source_binding=dict(source_sha256=index['source']['actual_sha256'],source_pts=pts,time_base='1/12800',shot_id='s07',native_hw=[1080,1920],decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                public,readout,raw,arrays=read_profile_proof(rgb,search,exclusion,source_binding,material_signal=material_signal)
                profile_path=out/'profiles'/f'{pts}.npz';np.savez_compressed(profile_path,**arrays)
                diagnostic=shape_diagnostics(raw,search)
                if 'similarity'in diagnostic and'affine'in diagnostic and'rms_distance_px'in diagnostic['affine']and'rms_distance_px'in diagnostic['similarity']:
                    affine_pairs.append((pts,diagnostic['similarity']['rms_distance_px'],diagnostic['affine']['rms_distance_px'],diagnostic['affine']['axis_ratio']))
                public_uv={tuple(r['uv'])for r in public}
                for candidate_order,row in enumerate(raw):
                    if row['uv']!=row['actual_source_quantiles'][0][1]:raise ValueError('Actual source q50 changed')
                    pending.append(dict(row,pts=pts,path_id=meta['native_path_id'],entity=entity,candidate_order=candidate_order,
                        source_reference_pts=meta['pts'],source_BGR_sha256=image_hash,source_mask_path=str(mask_path),source_mask_sha256=mask_hash,
                        existing_closed_outline_organized=tuple(row['uv'])in public_uv,
                        actual_outer_boundary_or_internal_letter_review_pending=True,physical_instance_identity_review_pending=True,
                        fit_eligible=False,review_state='pending_independent_source_material_and_instance_review',
                        source_profile_proof=str(profile_path),source_UV_synthesized=False))
                spaced=materials.spaced_indices([r['uv']for r in pending]).tolist()
                status='pending_actual_exterior_candidates'if pending else'pending_no_reader_exterior_candidates'
                totals['profile_frames']+=1;totals['public_reader_rows']+=len(public)
            else:status='pending_unsupported_search_extent'
        qualified_rows=[]
        if qualify and public:
            for at in materials.spaced_indices([r['uv']for r in public]):
                row=public[at];candidate=next(p for p in pending if p['uv']==row['uv'])
                if source_outline:
                    point=np.asarray(row['uv']);normal=np.asarray(row['normal']);qq=np.asarray(row['actual_source_quantiles']);offsets=(qq[:,[0,2]].reshape(-1,2)-point)@normal;lo,hi=offsets.min()-3.,offsets.max()+3.;probes=point+np.linspace(lo,hi,max(2,int(np.ceil((hi-lo)*4))+1))[:,None]*normal
                    clear=bool((probes>=0).all()and(probes<=[1919,1079]).all())
                    if clear:
                        lower,upper=np.floor(probes).astype(int),np.ceil(probes).astype(int);clear=not any(exclusion[y[:,1],x[:,0]].any()for x in(lower,upper)for y in(lower,upper))
                    candidate['actual_full_quantile_guard_plus3_all_pixel_contributors_clear']=clear
                    if not clear:candidate['pending_reason']='actual_quantile_guard_crosses_source_image_or_fine_exclusion';continue
                qualified_rows.append(dict(candidate,fit_eligible=True,physical_instance_identity_review_pending=False,
                    actual_outer_boundary_or_internal_letter_review_pending=False,
                    review_state='approved_pilot_instance_and_public_original_RGB_exterior_qualified',
                    qualification=dict(known_source_instance=entity,source_pattern_check=patch,
                        existing_public_closed_outline_qualified=True,original_RGB_UV_unchanged=True,
                        native_residual_used=False,shape_or_H_UV_used_as_observation=False,
                        source_reference=str(pilot/'reference.json'),source_reference_sha256=sha(pilot/'reference.json'))))
        if qualified_rows:totals['qualified_frames']+=1
        totals['qualified_rows']+=len(qualified_rows)
        if pending:totals['candidate_frames']+=1
        else:gaps.append(pts)
        totals['raw_pending_rows']+=len(pending);totals['spaced_pending_rows']+=len(spaced);statuses[status]+=1
        row_record=dict(pts=pts,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=image_hash,
            mask_path=str(mask_path),mask_sha256=mask_hash,rows=qualified_rows,pending_rows=pending,spaced_pending_indices=spaced,
            fit_eligible=bool(qualified_rows),review_state='approved_pilot_instance_and_public_original_RGB_exterior_qualified'if qualified_rows else'pending_independent_source_material_and_instance_review',status=status)
        rows_out.write(json.dumps(materials.plain(row_record),ensure_ascii=False,allow_nan=False)+'\n')
        note=dict(pts=pts,status=status,direct_RootSIFT=direct_note,source_graph_attempts=transfers,source_transfer_basis=basis,
            source_graph_parent_pts=parent,H_current_from_reference=None if h is None else h.tolist(),source_pattern_check=patch,
            reader_record=readout,all_exterior_candidate_count=len(raw),existing_public_reader_row_count=len(public),
            affine_vs_similarity_diagnostic=diagnostic,profile_proof=None if profile_path is None else str(profile_path),
            profile_proof_sha256=None if profile_path is None else sha(profile_path),actual_UV_or_native_changed=False,
            shape_residual_does_not_qualify_identity_or_material=True,fit_eligible=False)
        registration.write(json.dumps(materials.plain(note),ensure_ascii=False,allow_nan=False)+'\n')
        if pts in preview_pts:
            preview=image.copy()
            if h is not None:
                xy=np.rint(project(h,reference_search)).astype(np.int32);cv2.polylines(preview,[xy],True,(255,180,0),1,cv2.LINE_AA)
            for row in pending:cv2.circle(preview,tuple(np.rint(row['uv']).astype(int)),2,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(preview,f'{entity} | S07 PTS {pts} | qualified {len(qualified_rows)} | raw candidates {len(pending)}',(25,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2,cv2.LINE_AA)
            cv2.imwrite(str(out/'proofs'/f'{pts}_source_candidates.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
        if fi%25==0 or fi==len(expected)-1:
            message=dict(frames=fi+1,total=len(expected),pattern_frames=totals['pattern_frames'],candidate_frames=totals['candidate_frames'],pending_rows=totals['raw_pending_rows'],seconds=time.monotonic()-started)
            progress.write(json.dumps(message)+'\n');print(json.dumps(message),flush=True)
    if not np.array_equal(seen,selected_expected):raise ValueError('Incomplete requested S07 source sequence')
    missing_runs=[]
    for pt in gaps:
        if missing_runs and pt==missing_runs[-1]['last_pts']+256:missing_runs[-1]['last_pts']=pt;missing_runs[-1]['frames']+=1
        else:missing_runs.append(dict(first_pts=pt,last_pts=pt,frames=1))
    summary=dict(schema='rinkspace.pending_full_badge_source_capture.v1',shot_id='s07',frames=len(selected_expected),requested_native_interval=dict(first_pts=first_pts,last_pts=last_pts),full_registered_shot_frames=len(expected),counts=dict(totals),statuses=dict(statuses),
        no_candidate_intervals=missing_runs,affine_similarity_diagnostic_frames=len(affine_pairs),
        affine_similarity_diagnostic_pairs=[dict(pts=p,similarity_rms=s,affine_rms=a,affine_axis_ratio=r)for p,s,a,r in affine_pairs],
        fit_eligible=qualify,fit_executed=False,all_actual_candidates_preserved=True,source_graph_is_navigation_only=True,
        panel_instance=entity,material_signal=material_signal,
        affine_is_diagnostic_only=True,low_shape_residual_is_not_correctness=True,existing_reference_bank_modified=False,
        source_rows=str(out/'source_rows.jsonl'),registration=str(out/'registration.jsonl'))
    write(out/'summary.json',summary);binding.update(complete=True,summary=str(out/'summary.json'),source_rows_sha256=sha(out/'source_rows.jsonl'),
        registration_sha256=sha(out/'registration.jsonl'),script_hashes_after={name:sha(p)for name,p in source_files.items()})
    write(out/'binding.json',binding)
    if qualify:
        import shutil
        from build_independent_shot_packets import build
        qualified_capture=out/'qualified';qfolder=qualified_capture/'s07';qfolder.mkdir(parents=True)
        shutil.copyfile(out/'source_rows.jsonl',qfolder/'source_rows.jsonl')
        pose_source=root/'s07/navigation/poses_reanchored.npz';shutil.copyfile(pose_source,qfolder/'poses.npz')
        write(qualified_capture/'binding.json',dict(binding,source_capture=str(out),source_capture_binding_sha256=sha(out/'binding.json'),
            qualification_method='Approved pilot physical instance + unchanged whole-pattern check + blue/red actual source component exterior + unchanged public closed-outline organization; raw candidates remain pending',
            human_full_sequence_accuracy_claim=False,source_UV_unchanged=True))
        build(root,qualified_capture,out/'source_packets',shot_ids=['s07'])
    print(json.dumps(dict(complete=True,frames=len(selected_expected),counts=dict(totals),summary=str(out/'summary.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path)
    parser.add_argument('--first-pts',type=int)
    parser.add_argument('--last-pts',type=int)
    parser.add_argument('--source-outline-navigation',action='store_true',help='Use public selected material exterior of the independently reviewed original reference as navigation; no camera state read or source UV generated from it')
    parser.add_argument('--descriptor-context-margin',type=int,default=0,help='Original source context used only to navigate matching; whole-pattern ROI unchanged')
    parser.add_argument('--qualify-approved-exterior',action='store_true',help='Only explicitly approved instance public RGB exterior rows become fit eligible; raw candidates stay pending')
    parser.add_argument('--proof-pts',nargs='*',type=int,help='Additional native source proof frames')
    main(parser.parse_args())
