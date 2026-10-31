"""Pending Ad9 exterior candidates from a separate independent source reference.

This pilot does not qualify observations, modify the print bank, or run a fit.
The native contour and bbox affine are search geometry only. Every candidate UV
comes from the existing current-RGB red-component exterior reader.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import datetime
import hashlib
import json
import shutil

import av
import cv2
import numpy as np

from capture_native_panels import single_panel, panel_material_metadata
from capture_independent_materials import source_exclusion, spaced_indices, plain
from capture_independent_prints import source_patch_check
from rinkspace.camera_reuse_localization import descriptors, match_features, ReuseConfig
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.geometry import project
from rinkspace.path_policy import guard_project_write
from rinkspace.review.scene import sample_scene
from rinkspace.session.panel_boundaries import read_panel_boundary
import rinkspace.session.panel_boundaries as panel_reader

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
REFERENCE_PTS=1433856
REFERENCE_ROI=[1010,334,1205,468]
PILOT_PTS=[1433856,1430784,1435648,1440512,1446912]
AD='Ad9'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,data):
    Path(path).write_text(json.dumps(plain(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def exact_frames(source,targets):
    targets=set(map(int,targets));found={};first,last=min(targets),max(targets)
    with av.open(str(source))as container:
        stream=container.streams.video[0]
        if stream.time_base!=Fraction(1,12800)or(stream.width,stream.height)!=(1920,1080):
            raise ValueError('Original source native dimensions/time base differ')
        container.seek(first,stream=stream,backward=True,any_frame=False);expected=first
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Original source exposure has no PTS')
            if frame.pts<first:continue
            if frame.pts>last:break
            if frame.pts!=expected or frame.time_base!=Fraction(1,12800):raise ValueError('Actual source PTS discontinuity')
            if frame.pts in targets:found[int(frame.pts)]=frame.to_ndarray(format='bgr24')
            expected+=256
    if set(found)!=targets:raise ValueError('Requested actual source frames missing')
    return found


def material_sides(rgb,row):
    uv=np.asarray(row['uv']);normal=np.asarray(row['normal']);qq=np.asarray(row['actual_source_quantiles'])
    # Retain the same actual q50 sides returned by the interface operator.
    sides=qq[:,1].copy()
    radius=max(4.,float(np.linalg.norm(qq[0,2]-qq[0,0]))+3.)
    positions=uv+np.array([-radius,radius])[:,None]*normal
    sampled=cv2.remap(rgb.astype(np.float32),positions[:,0,None].astype(np.float32),
        positions[:,1,None].astype(np.float32),cv2.INTER_LINEAR).reshape(2,3)
    return dict(actual_q50_side_uv=sides.tolist(),inside_outside_probe_uv=positions.tolist(),
        inside_outside_source_RGB=sampled.tolist(),probe_radius_px=radius,
        support_role='diagnostic original RGB only; no extra admission threshold')


def read_with_pending_exterior(rgb, search, exclusion, binding, *, material_signal='red'):
    """Serial telemetry only; the existing closed-outline decision is unchanged.

    The public reader does not return pre-organization exterior candidates.
    Observe that internal measurement at its existing qualification call, then
    return the exact original decision. Nothing rejected becomes qualified.
    """
    captured=[]
    original=panel_reader._qualify_closed_outline
    def observe(measurement,curve,path,mask):
        query=np.asarray(measurement['candidate_query'])
        for i in np.flatnonzero(measurement['query_curve'][query]==curve):
            qi=int(query[i]);qq=np.asarray(measurement['candidate_side_quantiles_uv'][i])
            row=dict(curve=int(curve),uv=np.asarray(measurement['candidate_uv'][i]).tolist(),
                normal=np.asarray(measurement['query_normal'][qi]).tolist(),actual_source_quantiles=qq.tolist(),
                model_UV_synthesized=False,measurement_operator='source_outer_panel_material_interface',
                reader_prequalification_candidate_id=int(i),candidate_flags=int(measurement['candidate_flags'][i]),
                source_closed_outline_qualification='pending',fit_eligible=False)
            row['reader_candidate_sides']=np.asarray(measurement['candidate_sides'][i]).tolist()
            row['actual_source_sides']=material_sides(rgb,row)
            captured.append(row)
        return original(measurement,curve,path,mask)
    panel_reader._qualify_closed_outline=observe
    try:
        rows,record=read_panel_boundary(rgb,[search],exclusion,binding=binding,
            material_signal=material_signal,closed_paths=[True])
    finally:
        panel_reader._qualify_closed_outline=original
    return rows,record,captured


def prepare(args):
    cv2.setNumThreads(2)
    ad=getattr(args,'panel_instance',AD)
    graph_transfer=bool(getattr(args,'source_graph_transfer',False))
    full_profiles=bool(getattr(args,'full_profile_proof',False))
    reference_pts=args.reference_pts;reference_roi=args.reference_roi;pilot_pts=sorted(set([reference_pts,*args.pts]))
    root,run,out=[Path(p).resolve()for p in(args.root,args.run_root,args.output)]
    out=guard_project_write(out)
    if out.exists():raise FileExistsError('Preserve existing badge pilot: '+str(out))
    if out.is_relative_to(PROJECT.resolve()):raise ValueError('Badge image outputs belong outside OneDrive')
    index=read(root/'index.json');prepared=Path(getattr(args,'prepared_root',None) or run/'prepared_full').resolve();manifest=read(prepared/'joint_manifest.json')
    source=Path(index['source']['resolved_path']).resolve(strict=True)
    if sha(source)!=index['source']['actual_sha256']:raise ValueError('Original source SHA256 changed')
    live=Path(getattr(args,'state',None) or run/'joint_full/accepted_pixels_latest.npz').resolve()
    library=Path(manifest['native_library']).resolve(strict=True)
    out.mkdir(parents=True,exist_ok=False)
    # The solver atomically publishes accepted snapshots. Copy exactly once,
    # then consume only the private fixed copy while the live solve continues.
    frozen=out/'frozen_state.npz';shutil.copyfile(live,frozen)
    with np.load(frozen,allow_pickle=False)as archive:
        state={key:archive[key]for key in archive.files}
    if(str(state['native_library_provenance_sha256'])!=sha(library)
        or sha(library)!=manifest['native_library_sha256']):raise ValueError('Frozen state/native library provenance differs')
    if str(state['point_input_sha256'])!=manifest['point_input_sha256']:raise ValueError('Frozen state/current PreparedProblem differs')
    scene_path=Path(manifest['scene_path']).resolve(strict=True);scene=read(scene_path)
    if sha(scene_path)!=manifest['scene_file_sha256']:raise ValueError('Current scene changed')
    model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    if model.initial_state.layout['scene_content_revision']!=manifest['scene_revision']:raise ValueError('Current scene content revision differs')
    core=NativeCore(library)
    samples=sample_scene(scene,core,scene_path,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
    panel=single_panel(ad,'outer_ring')
    if len(panel['path_ids'])!=1:raise ValueError('Badge pilot requires one native exterior contour')
    material_signal=panel['material_signal']
    if ad=='Ad9' and panel['path_ids']!=['print_instance/Ad9::Ad9.print.red_field.boundary.0000']:raise ValueError('Ad9 exterior native path differs')
    path_id=panel['path_ids'][0]
    cut=next(c for c in samples['cuts']if c['id']==path_id)
    if not cut['closed']:raise ValueError('Current badge outer contour must be a closed native path')
    world=samples['xyz'][cut['begin']:cut['end']]
    metadata=panel_material_metadata(scene,scene_path)[path_id]
    shot=next(s for s in manifest['shots']if s['id']=='s07')
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    frame_time=(reference_pts-shot['reference_pts'])/12800+timeline.offset(shot)
    native=core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,frame_time,world)
    shape_reference=None
    if np.any(native['status'])or not np.isfinite(native['uv']).all():
        if args.shape_reference is None:
            raise ValueError('Current native badge reference contour is not fully projectable; preserve pilot for diagnosis')
        # A previously projected catalogue contour can still locate an RGB
        # search through an explicit source bbox. It does not validate the
        # current camera or create any point correspondence.
        shape_reference=dict(path=str(args.shape_reference),sha256=sha(args.shape_reference),navigation_only=True)
        with np.load(args.shape_reference,allow_pickle=False)as template:uv=template['native_uv'].copy()
        if not np.isfinite(uv).all():raise ValueError('Native catalogue search outline is not finite')
    else:uv=native['uv']
    lo,hi=uv.min(0),uv.max(0)
    if np.any(hi-lo<=0):raise ValueError('Current native badge contour has no usable search extent')
    roi=np.array(reference_roi,float);scale=(roi[2:]-roi[:2])/(hi-lo)
    affine=np.eye(3);affine[0,0],affine[1,1]=scale;affine[:2,2]=roi[:2]-scale*lo
    reference_search=project(affine,uv)
    np.savez_compressed(out/'reference_geometry.npz',world_xyz=world,native_uv=uv,native_status=native['status'],
        bbox_search_affine=affine,reference_search_uv=reference_search)
    images=exact_frames(source,pilot_pts)
    originals=out/'originals';originals.mkdir()
    for pts,image in images.items():cv2.imwrite(str(originals/f's07_pts{pts}.png'),image)
    provided_reference=getattr(args,'reference_image',None)
    if provided_reference is not None:
        supplied=cv2.imread(str(provided_reference))
        if supplied is None or supplied.shape!=(1080,1920,3) or not np.array_equal(supplied,images[reference_pts]):
            raise ValueError('Approved original reference PNG differs from exact decoded native PTS')
    ref_image=images[reference_pts];ref_exclusion,ref_mask_path=source_exclusion(root,'s07',reference_pts)
    ref_mask=np.zeros((1080,1920),np.uint8);x0,y0,x1,y1=reference_roi;ref_mask[y0:y1+1,x0:x1+1]=255
    ref_mask[ref_exclusion]=0
    context_margin=int(getattr(args,'descriptor_context_margin',0))
    if context_margin<0:raise ValueError('Descriptor context margin must be nonnegative')
    feature_mask=ref_mask.copy()
    if context_margin:
        feature_mask[:]=0;fx0,fy0=np.maximum(np.array(reference_roi[:2])-context_margin,[0,0]);fx1,fy1=np.minimum(np.array(reference_roi[2:])+context_margin,[1919,1079])
        feature_mask[fy0:fy1+1,fx0:fx1+1]=255;feature_mask[ref_exclusion]=0
        cv2.imwrite(str(out/'reference_feature_context_mask.png'),feature_mask)
    feature_uv,description=descriptors(ref_image,feature_mask)
    if description is None:description=np.empty((0,128),np.float32)
    np.savez_compressed(out/'reference_rootsift.npz',feature_uv=feature_uv,descriptors=description)
    cv2.imwrite(str(out/'reference_mask.png'),ref_mask)
    meta=dict(ad_id=ad,shot_id='s07',pts=reference_pts,roi_xyxy_inclusive=reference_roi,material_signal=material_signal)
    reference=dict(source_image=ref_image,reference_mask=ref_mask>0)
    reference_record=dict(meta,source_image=str(originals/f's07_pts{reference_pts}.png'),
        source_BGR_sha256=hashlib.sha256(ref_image.tobytes()).hexdigest(),source_image_sha256=sha(originals/f's07_pts{reference_pts}.png'),
        person_mask=str(ref_mask_path),person_mask_sha256=sha(ref_mask_path),descriptor_kind='RootSIFT',keypoints=len(feature_uv),
        native_path_id=path_id,closed=True,path_material_metadata=metadata,material_signal=material_signal,
        descriptor_context_margin_px=context_margin,descriptor_context_is_navigation_only=True,
        whole_pattern_mask_remains_original_reference_ROI=True,
        approved_reference_image=None if provided_reference is None else str(Path(provided_reference).resolve()),
        approved_reference_image_sha256=None if provided_reference is None else sha(provided_reference),
        bbox_affine=affine.tolist(),bbox_affine_is_search_geometry_only=True,
        source_shape_template=shape_reference,current_native_projection_complete=not bool(np.any(native['status'])),
        reference_roi_authority='Main-session independent source-image bbox; not reviewed by this adapter',
        fit_eligible=False,source_observations_generated_from_bbox_or_native=False)
    write(out/'reference.json',reference_record)
    config=ReuseConfig(min_matches=6,ransac_px=3.)
    graph=None;graph_note=None
    if graph_transfer:
        from capture_independent_ice import load_navigation_graph,navigation_between
        graph,graph_note=load_navigation_graph(root,next(q for q in index['shots']if q['shot_id']=='s07'))
    if full_profiles:(out/'profiles').mkdir()
    summaries=[]
    with(out/'source_rows.jsonl').open('w',encoding='utf8',buffering=1)as stream:
      for pts in pilot_pts:
        image=images[pts];rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);exclusion,mask_path=source_exclusion(root,'s07',pts)
        current=descriptors(image,(~exclusion).astype(np.uint8)*255)
        matched=match_features((feature_uv,description),current,config)
        direct_match_found=matched is not None;transfer_basis='direct_RootSIFT'if direct_match_found else None;transfer_reason=None
        if graph_transfer and pts==reference_pts:
            matched=(np.empty((0,2)),np.empty((0,2)),np.eye(3));transfer_basis='same_exact_source_RGB_anchor';transfer_reason='same_source_BGR_sha256_verified'
        elif graph_transfer and matched is None:
            hh,transfer_reason=navigation_between(graph,reference_pts,pts)
            if hh is not None:matched=(np.empty((0,2)),np.empty((0,2)),hh);transfer_basis='supported_source_graph_navigation_then_whole_pattern_check'
        rows=[];pending_rows=[];spaced_pending=[];raw_exterior=[];all_reader_rows=[];readout=None;pattern=None;h=None;profile_path=None;left=right=None;status='pending_no_reliable_source_match'
        if matched is not None:
            left,right,h=matched
            if not np.isfinite(h).all()or np.linalg.det(h[:2,:2])<=0:
                status='pending_invalid_source_transfer'
            else:
                pattern=source_patch_check(image,exclusion,meta,reference,h)
                if pattern['supported']:
                    search=project(h,reference_search)
                    binding=dict(source_sha256=index['source']['actual_sha256'],source_pts=pts,time_base='1/12800',
                        shot_id='s07',native_hw=[1080,1920],decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    if full_profiles:
                        from capture_independent_badge_full import read_profile_proof
                        all_reader_rows,readout,raw_exterior,profile_arrays=read_profile_proof(rgb,search,exclusion,binding,material_signal=material_signal)
                        profile_path=out/'profiles'/f'{pts}.npz';np.savez_compressed(profile_path,**profile_arrays)
                    else:
                        all_reader_rows,readout,raw_exterior=read_with_pending_exterior(rgb,search,exclusion,binding,material_signal=material_signal)
                    keep=spaced_indices([r['uv']for r in all_reader_rows])
                    for i in keep:
                        row=all_reader_rows[i]
                        if row['uv']!=row['actual_source_quantiles'][0][1]:raise ValueError('Badge output differs from actual source q50')
                        rows.append(dict(row,pts=pts,path_id=path_id,entity=ad,source_reference_pts=reference_pts,
                            source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),evidence_image=str(originals/f's07_pts{pts}.png'),
                            capture_family=f'independent_registered_{material_signal}_badge_exterior',actual_source_sides=material_sides(rgb,row),
                            fit_eligible=False,review_state='pending_main_or_independent_image_review',
                            internal_white_letter_edges_are_not_target=True))
                    qualified_uv={tuple(row['uv'])for row in all_reader_rows}
                    unorganized=[row for row in raw_exterior if tuple(row['uv'])not in qualified_uv]
                    for row in unorganized:
                        pending_rows.append(dict(row,pts=pts,path_id=path_id,entity=ad,source_reference_pts=reference_pts,
                            source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),evidence_image=str(originals/f's07_pts{pts}.png'),
                            capture_family=f'independent_registered_{material_signal}_badge_exterior',actual_source_sides=material_sides(rgb,row),
                            fit_eligible=False,review_state='pending_closed_shape_and_main_or_independent_image_review',
                            reason=f'Existing reader {material_signal} exterior RGB candidate; closed-shape organization did not qualify it',
                            internal_white_letter_edges_are_not_target=True))
                    spaced_pending=[pending_rows[i]for i in spaced_indices([row['uv']for row in pending_rows])]
                    status=('pending_source_badge_candidates'if rows else
                        'pending_prequalification_exterior_candidates'if pending_rows else'pending_no_source_exterior')
                else:status='pending_whole_pattern_not_supported'
        proof=dict(pts=pts,status=status,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            mask_path=str(mask_path),mask_sha256=sha(mask_path),source_identity_match_found=matched is not None,direct_RootSIFT_match_found=direct_match_found,
            source_transfer_basis=transfer_basis,source_transfer_navigation_status=transfer_reason,
            source_profile_proof=None if profile_path is None else str(profile_path),
            source_profile_proof_sha256=None if profile_path is None else sha(profile_path),material_signal=material_signal,
            source_keypoints_reference=[]if left is None else left.tolist(),source_keypoints_current=[]if right is None else right.tolist(),
            H_current_from_reference=None if h is None else h.tolist(),source_pattern_check=pattern,
            native_or_bbox_or_H_pixels_are_not_observations=True,reader_record=readout,
            all_unthinned_reader_rows=all_reader_rows,spaced_candidate_rows=rows,
            all_prequalification_exterior_candidates=raw_exterior,all_pending_exterior_rows=pending_rows,spaced_pending_rows=spaced_pending,
            fit_eligible=False,review_state='pending_main_or_independent_image_review')
        write(out/f'{pts}_proof.json',proof)
        record=dict(pts=pts,source_video_sha256=index['source']['actual_sha256'],source_BGR_sha256=proof['source_BGR_sha256'],
            mask_path=str(mask_path),mask_sha256=proof['mask_sha256'],rows=rows,pending_rows=pending_rows,spaced_pending_rows=spaced_pending,
            fit_eligible=False,review_state=proof['review_state'],status=status,proof=str(out/f'{pts}_proof.json'))
        stream.write(json.dumps(plain(record),ensure_ascii=False,allow_nan=False)+'\n')
        preview=image.copy();cv2.rectangle(preview,(x0,y0),(x1,y1),(0,255,255),1)if pts==reference_pts else None
        if h is not None and pattern is not None and pattern['supported']:
            curve=np.rint(project(h,reference_search)).astype(np.int32);cv2.polylines(preview,[curve],True,(255,180,0),1,cv2.LINE_AA)
        for row in rows+pending_rows:cv2.circle(preview,tuple(np.rint(row['uv']).astype(int)),3,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(preview,f'{ad} pending | S07 PTS {pts} | pending RGB candidates {len(rows)+len(pending_rows)}',(25,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(out/f'{pts}_source_candidates.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
        if graph_transfer and h is not None and pattern is not None and pattern['supported']:
            bounds=project(h,reference_search);crop_lo=np.maximum(np.floor(bounds.min(0)-35).astype(int),[0,0]);crop_hi=np.minimum(np.ceil(bounds.max(0)+36).astype(int),[1920,1080])
            if np.all(crop_hi>crop_lo):cv2.imwrite(str(out/f'{pts}_source_candidates_crop.png'),preview[crop_lo[1]:crop_hi[1],crop_lo[0]:crop_hi[0]])
        summaries.append(dict(pts=pts,status=status,matches=0 if left is None else len(left),rows=len(rows),pending_rows=len(pending_rows),spaced_pending_rows=len(spaced_pending),prequalification_exterior_candidates=len(raw_exterior),
            unthinned_source_rows=len(all_reader_rows),pattern_check=pattern,reader_status=None if readout is None else readout['status'],fit_eligible=False))
        print(json.dumps(plain(summaries[-1]),ensure_ascii=False,allow_nan=False),flush=True)
    write(out/'binding.json',dict(source_sha256=index['source']['actual_sha256'],fixed_state=str(frozen),fixed_state_sha256=sha(frozen),
        live_state_copied_once=str(live),copied_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prepared_manifest=str(prepared/'joint_manifest.json'),prepared_manifest_sha256=sha(prepared/'joint_manifest.json'),
        native_library=str(library),native_library_sha256=sha(library),scene_path=str(scene_path),scene_file_sha256=sha(scene_path),
        native_path_id=path_id,panel_instance=ad,material_signal=material_signal,source_graph=graph_note,reference=str(out/'reference.json'),summaries=summaries,
        fit_eligible=False,user_visual_acceptance=False,fit_executed=False,existing_print_bank_modified=False,
        reader_file_modified=False,serial_private_measurement_telemetry_used=True,adapter_sha256=sha(__file__),
        source_rows=str(out/'source_rows.jsonl'),component_interface_limits=[
            'single_panel currently requires badge_part; Ad9 ignores that argument',
            'match_features retains its existing min spatial feature spread and six-match conditions',
            'read_panel_boundary public API omits pre-organization exterior candidates; serial telemetry preserves those actual RGB measurements only as pending and leaves the original qualification decision unchanged',
            ('Full actual RGB candidates, profiles and component bitmaps saved by serial diagnostic telemetry' if full_profiles else 'No every-profile rejected-transition dump or filled-component bitmap is exposed by the existing reader'),
            'pattern appearance alone does not distinguish repeated Ad9/Ad22 artwork; physical-instance review remains pending']))
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('root','run-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-pts',type=int,default=REFERENCE_PTS)
    parser.add_argument('--reference-roi',nargs=4,type=int,default=REFERENCE_ROI)
    parser.add_argument('--pts',nargs='+',type=int,default=PILOT_PTS)
    parser.add_argument('--shape-reference',type=Path,help='Previously projected native catalogue outline, bbox search only when current projection is incomplete')
    parser.add_argument('--panel-instance',choices=['Ad9','Ad22','Ad12','Ad25'],default=AD)
    parser.add_argument('--prepared-root',type=Path,help='Current PreparedProblem; default preserves Ad9 prepared_full')
    parser.add_argument('--state',type=Path,help='Current native state; default preserves Ad9 joint_full')
    parser.add_argument('--reference-image',type=Path,help='Approved native source PNG; must equal exact decoded reference PTS')
    parser.add_argument('--source-graph-transfer',action='store_true',help='Explicit same-source anchor or supported graph navigation, always whole-pattern checked')
    parser.add_argument('--full-profile-proof',action='store_true',help='Save unchanged public RGB profile/component telemetry')
    parser.add_argument('--descriptor-context-margin',type=int,default=0,help='Additional original-source context for navigation descriptors only; whole-pattern ROI is unchanged')
    prepare(parser.parse_args())
