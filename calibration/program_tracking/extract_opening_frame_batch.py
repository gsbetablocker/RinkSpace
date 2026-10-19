"""Measure consecutive official50 images using an approved native-image example.

Background flow and the historical model only navigate. All later edge readouts
come from the current official image; masks exclude people, and sparse straight
line representatives explicitly share their source estimate. No optimizer runs.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT/'src'), str(Path(__file__).resolve().parent)])
from rinkspace.packed_assets import materialize_project_asset, read_project_json
from calibration.program_tracking.single_frame_source_review import (
    scan_curve, print_contours, compact_review_points, straight_source_boundary,
    space_boundary_samples, remove_temporary_readouts, compact_run)
from calibration.program_tracking.render_single_frame_source_review import native_overlay, source_segments
from calibration.program_tracking.build_framewise_model import make_binding

PARENT = ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
OUT = PARENT/'opening_100_50fps'
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/opening_100_50fps')
MODEL = ROOT/'events/ankara_2026/models/framewise_8a28_20261013'

def read(path): return read_project_json(path, project_root=ROOT)
def sha(path):
    with materialize_project_asset(path, project_root=ROOT).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

def transform(points, matrix):
    return cv2.transform(np.asarray(points,np.float64)[None],matrix)[0]

class BackgroundNavigation:
    def __init__(self, image, person):
        self.gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        allowed=(~person).astype('uint8')*255
        allowed[:180]=0; allowed[1040:]=0
        self.points=cv2.goodFeaturesToTrack(self.gray,600,.008,8,mask=allowed,blockSize=7)
        if len(self.points)<20: raise ValueError('Insufficient static board source features')
        from calibration.program_tracking.framewise_model import feature_world
        from calibration.program_tracking.projection import project_fast
        scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json')
        corners=[feature_world(scene,registry['targets']['Ad2.'+k])[0]for k in ('left_top','right_top','left_bottom','right_bottom')]
        world=np.array([corners[i]*(1-t)+corners[i+1]*t for i in (0,2) for t in np.linspace(0,.45,12)])
        cams=np.load(DATA/'camera_seeds.npz')['camera25']
        first_uv=project_fast(cams[0],world)[0]
        self.seeds=[cv2.estimateAffine2D(first_uv,project_fast(c,world)[0],ransacReprojThreshold=3.)[0]for c in cams]

    def current(self,image,person,index):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        seed=self.seeds[index-1]
        predicted=transform(self.points[:,0],seed).astype('float32')[:,None]
        next_points,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,predicted.copy(),
            winSize=(41,41),maxLevel=3,criteria=(3,40,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,next_points,self.points.copy(),
            winSize=(41,41),maxLevel=3,criteria=(3,40,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        p=self.points[:,0];q=next_points[:,0]
        valid=status[:,0].astype(bool)&reverse[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p,axis=1)<1.5)
        valid &= np.linalg.norm(q-predicted[:,0],axis=1)<15
        x,y=np.rint(q).astype(int).T
        inside=(x>=0)&(x<1920)&(y>=0)&(y<1080)
        valid &= inside
        valid[inside] &= ~person[y[inside],x[inside]]
        matrix,inliers=cv2.estimateAffine2D(p[valid],q[valid],method=cv2.RANSAC,
            ransacReprojThreshold=2.0,maxIters=3000,confidence=.999,refineIters=10)
        if matrix is None or np.count_nonzero(inliers)<30: raise ValueError('Static board navigation failed')
        errors=np.linalg.norm(transform(p[valid],matrix)-q[valid],axis=1)
        selected=errors[inliers[:,0].astype(bool)]
        return matrix,dict(method='same_shot_static_board_LK_forward_backward_RANSAC_affine',
            matrix=matrix.tolist(),source_frame_index=1,retained_inliers=len(selected),
            median_navigation_error_px=float(np.median(selected)),
            max_navigation_error_px=float(selected.max()),observation_coordinates_from_flow=False)

def current_scan(raw,mask,matrix,anchors,signal,axis=0,radius=24):
    points=transform(anchors,matrix)
    # Endpoints on the source image boundary move outside the frame on zoom.
    # Extend horizontal material edges all the way to the current image edge.
    if axis==0 and anchors[-1][0]==1919:
        a,b=points[-2:];points[-1]=[1919,a[1]+(1919-a[0])*(b[1]-a[1])/(b[0]-a[0])]
    if axis==0 and anchors[0][0]==0:
        a,b=points[:2];points[0]=[0,a[1]-a[0]*(b[1]-a[1])/(b[0]-a[0])]
    points[:,axis]=np.clip(points[:,axis],0,raw.shape[1-axis]-1)
    segments=scan_curve(raw,mask,dict(navigation_xy=points.tolist(),axis=axis,
                                    normal_radius_px=radius,signal=signal))
    return [s for s in segments if np.linalg.norm(np.array(s['uv'][-1])-s['uv'][0])>=12]

def fit_axis_line(segments,axis):
    p=np.concatenate([s['_native_uv'] for s in segments])
    from scipy.optimize import least_squares
    a=p[:,axis];b=p[:,1-axis]
    initial=np.polyfit(a,b,1)
    fit=least_squares(lambda c: c[0]*a+c[1]-b,initial,loss='soft_l1',f_scale=3.)
    return fit.x

def coherent_ad_edge(segments):
    """Reject isolated colour changes inconsistent with the same visible ad edge."""
    line=fit_axis_line(segments,0);kept=[];rejected=[]
    for segment in segments:
        uv=np.asarray(segment['_native_uv']);width=np.asarray(segment['_native_iqr'])
        good=np.abs(uv[:,1]-(line[0]*uv[:,0]+line[1]))<=np.clip(width*.5,3.,6.)
        rejected.extend(uv[~good].tolist())
        indices=np.flatnonzero(good)
        for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if len(run)<2 or np.linalg.norm(uv[run[-1]]-uv[run[0]])<12:continue
            result=compact_run(uv[run],width[run])
            result['_native_uv']=uv[run].tolist();result['_native_iqr']=width[run].tolist()
            kept.append(result)
    return kept,dict(source_line_coefficients=line.tolist(),rejected_native_colour_candidates=rejected,
                      reason='inconsistent_with_the_same_current_visible_ad_border')

def rectangle(features):
    by={f['id']:f for f in features}
    upper=by['Ad2.top_edge']['segments'][0]
    lower=by['Ad2.bottom_edge']['segments'][0]
    side=by['Ad2.left_vertical']['segments'][0]
    t=fit_axis_line([upper],0);b=fit_axis_line([lower],0);l=fit_axis_line([side],1)
    def corner(h):
        y=(h[0]*l[1]+h[1])/(1-h[0]*l[0]);return [l[0]*y+l[1],y]
    uv=[upper['uv'][-1],corner(t),corner(b),lower['uv'][-1]]
    by['Ad2.top_edge']['segments'].pop(0);by['Ad2.bottom_edge']['segments'].pop(0)
    features.remove(by['Ad2.left_vertical'])
    features.append(dict(id='Ad2.visible_left_outline',label_zh='Ad2矩形左侧与两个角点',
        physical_surface='board',world_curve='Ad2.external_border',allowed_for_fit=False,
        segments=[dict(uv=uv,closed=False,vertex_count=4,protected_vertex_indices=[0,1,2,3],
            edge_roles=['top_edge','left_vertical','bottom_edge'],
            corner_vertex_indices={'top_left':1,'bottom_left':2},
            measurement_origin='current_source_robust_line_intersection',
            source_line_coefficients={'top':t.tolist(),'bottom':b.tolist(),'left':l.tolist()},
            _edge_readouts=[upper,side,lower],model_projection_used_for_corner_coordinates=False)]))

def top_profile_evidence(raw,mask,matrix,first,first_mask):
    """Keep the approved physical edge definition through paired RGB profiles.

    The first frame supplies the location within its blurred transition. Each
    later frame independently aligns the actual native colour profile near
    source-flow navigation; no camera projection supplies the measured offset.
    """
    blue=lambda image: image[:,:,0].astype('float32')-(image[:,:,1].astype('float32')+image[:,:,2])/2
    before=blue(first);current=blue(raw);points=[];sigmas=[];correlations=[]
    offsets=np.linspace(-12,12,97);rejected=[]
    for x in np.arange(40,1881,80):
        y=120.14428384099743-.01731026368283015*x
        p=np.array([x,y]);q=transform([p],matrix)[0]
        if not 28<q[0]<1892 or not 30<q[1]<1050:
            rejected.append(dict(first_x=int(x),reason='profile_support_outside_current_image'));continue
        ix,iy=np.rint(q).astype(int)
        if mask[iy-25:iy+26,ix-4:ix+5].any() or first_mask[int(y)-25:int(y)+26,int(x)-4:int(x)+5].any():
            rejected.append(dict(first_x=int(x),reason='foreground_intersects_current_or_reference_profile_support'));continue
        coords=p+[[-2,t]for t in np.linspace(-23,23,47)]
        old_values=[];new_values=[]
        for dx in range(5):
            old=coords+[dx,0];mapped=transform(old,matrix)
            old_values.append(cv2.remap(before,old[:,0].astype('float32')[:,None],old[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0])
            xx=np.broadcast_to(mapped[:,0],(len(offsets),47)).astype('float32')
            yy=(mapped[:,1][None]+offsets[:,None]).astype('float32')
            new_values.append(cv2.remap(current,xx,yy,cv2.INTER_LINEAR))
        template=np.median(old_values,axis=0);values=np.median(new_values,axis=0)
        template-=template.mean();values-=values.mean(axis=1)[:,None]
        norm=np.linalg.norm(template)
        if norm<30:
            rejected.append(dict(first_x=int(x),reason='insufficient_reference_colour_transition'));continue
        corr=values@template/(np.maximum(np.linalg.norm(values,axis=1),1e-8)*norm)
        k=int(np.argmax(corr))
        if not 0<k<len(offsets)-1 or corr[k]<.88:
            rejected.append(dict(first_x=int(x),reason='ambiguous_current_profile_alignment',best_ncc=float(corr[k])));continue
        delta=float(offsets[k]);points.append([float(q[0]),float(q[1]+delta)])
        sigmas.append(3.0);correlations.append(float(corr[k]))
    if len(points)<8:raise ValueError('Too few current board-top profiles agreeing with approved edge definition')
    return [dict(uv=points,localization_sigma_px=sigmas,profile_ncc=correlations,rejected_profiles=rejected,
                 measurement_origin='current_native_profile_registration_to_approved_edge_definition')]

def measured_features(raw,mask,matrix,first,first_mask):
    features=[]
    defs=[('board_top','围挡上沿','board_surface.top_edge_xyz','blue',[[0,120],[940,104],[1100,101],[1919,87]],0),
          ('board_contact','围挡接冰下沿','ice_boundary.points_xyz','yellow',[[0,1014],[600,998],[1300,980],[1919,967]],0),
          ('Ad2.top_edge','Ad2上边框','Ad2.top_edge','magenta',[[553,235],[1919,209]],0),
          ('Ad2.left_vertical','Ad2左边框','Ad2.left_vertical','magenta',[[544,248],[548,847]],1),
          ('Ad2.bottom_edge','Ad2下边框','Ad2.bottom_edge','magenta',[[561,868],[1919,827]],0)]
    for fid,label,world,signal,anchors,axis in defs:
        segments=current_scan(raw,mask,matrix,anchors,signal,axis,28)
        if not segments:raise ValueError('Visible required source feature missing: '+fid)
        features.append(dict(id=fid,label_zh=label,world_curve=world,physical_surface='board',
                             allowed_for_fit=False,segments=segments))
        if fid in ('Ad2.top_edge','Ad2.bottom_edge'):
            clean,diagnostic=coherent_ad_edge(segments)
            if not clean:raise ValueError('No coherent visible ad edge '+fid)
            features[-1].update(segments=clean,source_consistency_filter=diagnostic)
    top=features[0]
    evidence=top_profile_evidence(raw,mask,matrix,first,first_mask)
    # Joint line is fitted to this frame's colour observations. Visibility is
    # evaluated on that line against the per-frame foreground candidate mask.
    straight_source_boundary(top,evidence,[[0,1919]])
    est=top['source_straight_line_estimate'];xx=np.arange(1920)
    yy=np.rint(est['y_intercept_px']+est['y_per_x']*xx).astype(int)
    visible=np.ones(1920,bool)
    for d in range(-3,4):visible &= ~mask[np.clip(yy+d,0,1079),xx]
    indices=np.flatnonzero(visible)
    runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    intervals=[[int(r[0]),int(r[-1])]for r in runs if len(r)>12]
    straight_source_boundary(top,evidence,intervals)
    est=top['source_straight_line_estimate']
    est['evidence_source']='current_official50_native_colour_profiles_registered_to_approved_first_edge_definition'
    est['profile_registration']=evidence
    est['prior_first_frame_edge_position_is_user_accepted']=True
    est['current_profile_offsets_measured_independently']=True
    # Keep one compact evidence record, not dense optimizer points.
    rectangle(features)
    roi=transform([[1190,345],[1920,345],[1920,685],[1190,685]],matrix)
    bounds=[max(0,int(roi[:,0].min())-20),max(0,int(roi[:,1].min())-20),
            1920,min(1080,int(roi[:,1].max())+20)]
    # All visible capital glyphs share the source-confirmed print row. Bright
    # costume trim below that row is never a candidate simply because it is
    # outside a foreground mask. This is navigation support, not a UV template.
    print_support=np.zeros(raw.shape[:2],np.uint8)
    polygon=transform([[1100,378],[2350,378],[2350,680],[1100,680]],matrix)
    cv2.fillConvexPoly(print_support,np.rint(polygon).astype('int32'),1)
    glyph_mask=mask | (print_support==0)
    # The approved defocused print has 8.6--15.1 px edge IQR. Much sharper
    # foreground trim is not the same material even when it is bright white.
    minimum_blur=5.*float(np.sqrt(abs(np.linalg.det(matrix[:,:2]))))
    glyphs=print_contours(raw,glyph_mask,bounds,minimum_transition_iqr_px=minimum_blur,
                          require_magenta_white_pair=True)
    inverse=cv2.invertAffineTransform(matrix)
    for glyph in glyphs:
        points=np.concatenate([s['uv']for s in glyph['segments']])
        centre=transform([np.mean(points,axis=0)],inverse)[0]
        x=centre[0]
        label=('U' if 1340<x<1565 else 'hyphen' if 1565<=x<1680 else 'S' if 1680<=x<1845
               else 'right_cropped_letter' if 1845<=x<2055 else 'A' if x>=2055
               else 'left_S_visible_fragment')
        if label=='A' and glyph.get('source_contour_internal_hole'):label='A_inner'
        glyph.update(id='Ad2.print.'+label,label=label,
            source_material_identity_checks=dict(confirmed_print_row=True,
                native_magenta_background_to_neutral_white_transition_required=True,
                minimum_colour_transition_iqr_px=minimum_blur,
                basis='approved_defocused_print_material_and_current_image_scale'))
        # Native cv2 contours can include clipping-created short pieces. Only
        # genuine remeasured boundary runs survive; contour normals reject the cut.
    features.extend(glyphs)
    occurrences={};counts={}
    for f in features:counts[f['id']]=counts.get(f['id'],0)+1
    for f in features:
        key=f['id'];occurrences[key]=occurrences.get(key,0)+1
        if counts[key]>1:f['id']+=f'.fragment{occurrences[key]}'
    # Source-blur simplification first, then protect long-edge coverage. Use
    # more vertices when additional K/A contours become visible after zoom.
    extra_k=sum(len(s['uv'])for f in glyphs if f['label']=='right_cropped_letter'for s in f['segments'])
    extra_a=sum(len(s['uv'])for f in glyphs if f['label'].startswith('A')for s in f['segments'])
    target=100+min(15,max(0,extra_k-7))+min(16,extra_a)
    compact_review_points(features,target)
    space_boundary_samples(features,250.)
    compaction=compact_review_points(features,target)
    compaction['algorithm_selected_sparse_budget']=compaction.pop('maximum_points_requested_by_user')
    compaction.update(user_hard_ceiling=500,user_density_reference_approved_count=99,
                      budget_reason='similar_density_plus_newly_visible_K_A_contours')
    remove_temporary_readouts(features)
    return features,compaction

def binding_for(packet,path,registry):
    # Base helper binds the five original glyphs. Additional named templates
    # have the same finite material-contour semantics and are bound explicitly.
    extra=[f for f in packet['features'] if f.get('label') in ('A','A_inner')]
    base=deepcopy(packet);base['features']=[f for f in base['features']if f.get('label') not in ('A','A_inner')]
    base['point_count']=sum(len(s['uv'])for f in base['features']for s in f['segments'])
    binding=make_binding(base,path,registry)
    for f in extra:
        for si,s in enumerate(f['segments']):
            target='Ad2.print.SKATING_A'+('_inner' if f['label']=='A_inner' else '')
            if target not in registry['targets']:raise ValueError('Missing A material entry')
            n=len(s['uv'])
            binding['observations'].append(dict(id=f"{f['id']}/segment/{si}/{target}",
                source_feature_id=f['id'],source_segment_index=si,source_vertex_indices=list(range(n)),
                source_point_ids=[f"{f['id']}/segment/{si}/vertex/{i}"for i in range(n)],
                target_id=target,point_correspondence=False,dependence_group=target,
                observation_uncertainty='retained_in_current_source_packet',source_coordinates_copied_or_changed=False))
    binding.update(point_count=packet['point_count'],next_frame_processed=False,
                   status='all_points_have_typed_model_entries_pending_batch_acceptance')
    return binding

def run(indices):
    cv2.setNumThreads(2)
    manifest=read(DATA/'frame_manifest.json');nav=read(OUT/'navigation_mapping.json')
    approved_path=PARENT/'frame_000001_points.json'; approved=read(approved_path)
    approved_sha=sha(approved_path); registry=read(MODEL/'feature_registry.json')
    first=cv2.imread(str(DATA/'frames/000001.png'))
    mask1=cv2.imread(str(DATA/'masks/000001.png'),0)>0
    motion=BackgroundNavigation(first,mask1)
    records=manifest['frames']
    for index in indices:
        path=DATA/f'frames/{index:06d}.png';raw=cv2.imread(str(path))
        mask_path=DATA/f'masks_current/{index:06d}.png'
        if not mask_path.exists():raise FileNotFoundError('Current official50 foreground exclusion required: '+str(mask_path))
        mask=cv2.imread(str(mask_path),0)>0
        record=records[index-1];navig=nav['frames'][index-1]
        matrix,tracking=motion.current(raw,mask,index)
        pts=256256+256*(index-1)
        source=dict(native_frame_index_1based=pts//256+1,programme_frame_index_1based=index,native_pts=pts,actual_pts=pts,
            source=manifest['source']['path'],source_video_sha256=manifest['source']['sha256'],
            time_base='1/12800',source_time=f'00:00:{pts/12800:06.3f}',
            decoded_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),image_path=str(path),
            mask_status='current_official50_source_person_exclusion',mask_path=str(mask_path),mask_sha256=sha(mask_path),
            legacy60_navigation_pts=navig['legacy60_pts'],navigation_mapping_status=navig['mapping_status'])
        if index==1:
            packet=deepcopy(approved);packet['source']=source
            packet['approval_provenance']=dict(original_packet=dict(path=str(approved_path),sha256=approved_sha),
                uv_inherited_unchanged=True,coordinate_origin='approved_legacy60_same_exposure',
                different_encoding=True,official50_reannotation_accepted=False)
            packet['status']='historically_approved_points_on_confirmed_same_exposure_official50'
        else:
            features,compaction=measured_features(raw,mask,matrix,first,mask1)
            packet=dict(schema='rinkspace.single-frame-source-review.v1',source=source,
                status='awaiting_user_acceptance',user_accepted=False,features=features,
                frame_count=1,next_frame_processed=False,solver_inputs_modified=False,
                point_count=sum(len(s['uv'])for f in features for s in f['segments']),
                user_requested_compaction=compaction,
                source_measurement='current_native_colour_transitions_source_intersections_and_shared_straight_top',
                excluded_features_by_user=approved['excluded_features_by_user'],
                observation_coordinates_from_projection=False,independent_accuracy_claim=False)
        packet['batch_sequence_index_1based']=index;packet['background_navigation']=tracking
        packet['batch_review_status']='first_frame_historical_approval_only_rest_awaiting_user'
        if packet['point_count']>500:raise ValueError('User hard ceiling exceeded')
        out=OUT/f'frames/{index:06d}.json';write(out,packet)
        binding=binding_for(packet,out,registry);write(OUT/f'bindings/{index:06d}.json',binding)
        overlay=native_overlay(raw,list(source_segments(packet)))
        (DATA/'overlays').mkdir(exist_ok=True)
        cv2.imwrite(str(DATA/f'overlays/{index:06d}.png'),overlay,[cv2.IMWRITE_PNG_COMPRESSION,1])
        print(json.dumps(dict(index=index,pts=pts,points=packet['point_count'],nav_error=tracking['median_navigation_error_px']),ensure_ascii=False),flush=True)
    assert sha(approved_path)==approved_sha

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--indices',default='1:100');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    indices=list(range(int(args.indices.split(':')[0]),int(args.indices.split(':')[1])+1)) if ':'in args.indices else list(map(int,args.indices.split(',')))
    if args.workers==1:run(indices)
    else:
        from concurrent.futures import ProcessPoolExecutor
        partitions=[list(map(int,part))for part in np.array_split(indices,args.workers)if len(part)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run,partitions))
