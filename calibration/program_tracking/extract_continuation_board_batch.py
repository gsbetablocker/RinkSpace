"""Reread visible physical board and Ad2 materials in an official50 batch.

The selected native example and 8a28 cameras only navigate current searches.
Every material readout is measured in its own frame. Off-screen ad corners,
foreground silhouettes, and the excluded strip/seam edges are never emitted.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT/'src'),str(Path(__file__).resolve().parent)])
from .extract_opening_frame_batch import read,write,sha,fit_axis_line,transform,MODEL
from .single_frame_source_review import (scan_curve,compact_run,transition,
    compact_review_points,straight_source_boundary,space_boundary_samples,
    remove_temporary_readouts)
from .extract_selected_frame_review import visible_intervals
from .render_single_frame_source_review import native_overlay,source_segments

PARENT=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/continuation_101_200_50fps')
REFERENCE=PARENT/'frame_000200_50fps'
REFDATA=DATA.parent/'frame_000200_50fps'


class BoardNavigation:
    """Model-seeded native static-material LK, used for search only."""
    def __init__(self,data=DATA):
        self.raw=cv2.imread(str(REFDATA/'source.png'))
        self.mask=cv2.imread(str(REFDATA/'person_mask.png'),0)>0
        self.gray=cv2.cvtColor(self.raw,cv2.COLOR_BGR2GRAY)
        allowed=np.zeros(self.gray.shape,np.uint8)
        # Restrict the reference tracks to the physical board face and top.
        # Audience, floor, broadcast overlays, and the skater never seed LK.
        spec=read(REFERENCE/'specification.json')
        searches={s['id']:s for s in spec['edge_searches']}
        top=np.asarray(searches['board_top']['navigation_xy'],float)
        bottom=np.asarray(searches['board_contact']['navigation_xy'],float)
        polygon=np.vstack([top+[[0,2],[0,2]],bottom[::-1]-[[0,4],[0,4]]])
        cv2.fillConvexPoly(allowed,np.rint(polygon).astype(np.int32),255)
        allowed[self.mask]=0
        self.points=cv2.goodFeaturesToTrack(self.gray,500,.005,7,mask=allowed,blockSize=7)
        from .framewise_model import feature_world
        from .projection import project_fast
        scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json')
        corners=[feature_world(scene,registry['targets']['Ad2.'+k])[0]
                 for k in ('left_top','right_top','left_bottom','right_bottom')]
        world=np.array([corners[i]*(1-t)+corners[i+1]*t
                        for i in (0,2) for t in np.linspace(0,1,16)])
        archive=np.load(data/'camera_seeds.npz')
        self.indices=archive['programme_indices'].astype(int).tolist()
        cameras=archive['camera25'];reference_uv=project_fast(cameras[self.indices.index(200)],world)[0]
        self.seeds={int(i):cv2.estimateAffine2D(reference_uv,project_fast(c,world)[0],
                     ransacReprojThreshold=3.)[0]for i,c in zip(self.indices,cameras)}

    def current(self,raw,mask,index):
        gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY);seed=self.seeds[index]
        predicted=transform(self.points[:,0],seed).astype('float32')[:,None]
        interior=(predicted[:,0,0]>=10)&(predicted[:,0,0]<1910)&(predicted[:,0,1]>=10)&(predicted[:,0,1]<1070)
        p=self.points[:,0][interior];predicted=predicted[interior]
        # Match native texture at the same apparent scale before local LK.
        # This is navigation, and supplies no material observation vertex.
        warped=cv2.warpAffine(self.gray,seed,(1920,1080),flags=cv2.INTER_LINEAR)
        next_points,status,_=cv2.calcOpticalFlowPyrLK(warped,gray,predicted,predicted.copy(),
            winSize=(41,41),maxLevel=4,criteria=(3,50,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,warped,next_points,predicted.copy(),
            winSize=(41,41),maxLevel=4,criteria=(3,50,.01),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        q=next_points[:,0]
        valid=status[:,0].astype(bool)&reverse[:,0].astype(bool)
        valid &= np.linalg.norm(back[:,0]-predicted[:,0],axis=1)<1.5
        valid &= np.linalg.norm(q-predicted[:,0],axis=1)<25
        x,y=np.rint(q).astype(int).T;inside=(x>=3)&(x<1917)&(y>=3)&(y<1077)
        valid &= inside
        safe=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
        valid[inside] &= ~safe[y[inside],x[inside]]
        matrix,inliers=cv2.estimateAffinePartial2D(p[valid],q[valid],method=cv2.RANSAC,
            ransacReprojThreshold=2.,maxIters=5000,confidence=.999,refineIters=20)
        if matrix is None or np.count_nonzero(inliers)<8:
            raise ValueError('Current static board LK failed '+str(index)+
                ' candidates='+str(len(p))+' verified='+str(int(valid.sum()))+
                ' inliers='+str(int(np.count_nonzero(inliers)) if inliers is not None else 0))
        errors=np.linalg.norm(transform(p[valid],matrix)-q[valid],axis=1)
        selected=errors[inliers[:,0].astype(bool)]
        return matrix,dict(method='reference200_native_static_board_LK_forward_backward_similarity',
            reference_programme_frame=200,retained_inliers=len(selected),
            median_navigation_error_px=float(np.median(selected)),
            max_navigation_error_px=float(selected.max()),matrix=matrix.tolist(),
            model_projection_supplies_search_seed_only=True,observation_coordinates_from_flow=False,
            source_static_point_pairs=[dict(reference_uv=a.tolist(),current_uv=b.tolist())
                for a,b in zip(p[valid][inliers[:,0].astype(bool)],q[valid][inliers[:,0].astype(bool)])])


def transformed_scan(item,matrix):
    item=deepcopy(item);original_radius=item['normal_radius_px'];anchors=transform(item['navigation_xy'],matrix)
    axis=item['axis'];other=1-axis
    # Interpolate before clipping so zoomed corners never become fake corners.
    lo=max(0.,anchors[0,axis]);hi=min(float(1919 if axis==0 else 1079),anchors[-1,axis])
    if hi-lo<12:return None
    if item['id'].startswith('board_'):
        lo,hi=0.,1919.
    a,b=anchors[[0,-1]];slope=(b[other]-a[other])/(b[axis]-a[axis])
    points=np.array([[0.,0.],[0.,0.]])
    points[:,axis]=[lo,hi];points[:,other]=a[other]+(points[:,axis]-a[axis])*slope
    if points[:,other].min()<0 or points[:,other].max()>float(1079 if axis==0 else 1919):return None
    scale=float(np.sqrt(abs(np.linalg.det(matrix[:,:2]))))
    radius=int(np.clip(round(original_radius*scale),8,36))
    # A right border that has just entered the frame still has real white
    # material context. Shorten the search support to the available pixels;
    # do not drop that whole visible side merely because its original normal
    # window extends beyond the image. Below eight pixels the pair is unclear.
    if axis==1:
        available=int(np.floor(min(points[:,other].min(),1919-points[:,other].max())))
        if available<8:return None
        radius=min(radius,available)
    item.update(navigation_xy=points.tolist(),normal_radius_px=radius)
    return item


def sparse_native_segment(segment,spacing):
    native=np.asarray(segment['_native_uv']);width=np.asarray(segment['_native_iqr'])
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(native,axis=0),axis=1))]
    stations=np.linspace(0,distance[-1],max(2,int(np.ceil(distance[-1]/spacing))+1))
    keep=np.unique([int(np.argmin(abs(distance-p)))for p in stations])
    segment.update(uv=native[keep].tolist(),vertex_count=len(keep),
        transition_iqr_px=width[keep].tolist(),localization_sigma_px=(width[keep]/1.349).tolist(),
        measurement_origin='distributed_current_native_material_readouts_on_straight_physical_edge',
        texture_wiggles_not_extra_physical_features=True)


def board_features(raw,mask,spec,matrix):
    features=[];scans={};diagnostics={'nonvisible_edges':[],'nonvisible_corners':[]}
    for original in spec['edge_searches']:
        item=transformed_scan(original,matrix)
        if item is None:
            diagnostics['nonvisible_edges'].append(original['id']);continue
        segments=scan_curve(raw,mask,item)
        segments=[s for s in segments if np.linalg.norm(np.array(s['_native_uv'][-1])-s['_native_uv'][0])>=10]
        if not segments:
            if original['id'].startswith('board_'):raise ValueError('Missing board material '+original['id'])
            diagnostics['nonvisible_edges'].append(original['id']);continue
        # Ad borders are each a single straight material boundary. Remove
        # isolated chroma proposals far from that current-source line.
        if original['id'].startswith('Ad2.'):
            line=fit_axis_line(segments,item['axis']);clean=[]
            for segment in segments:
                native=np.asarray(segment['_native_uv']);width=np.asarray(segment['_native_iqr'])
                good=np.abs(native[:,1-item['axis']]-(line[0]*native[:,item['axis']]+line[1]))<4.5
                idx=np.flatnonzero(good)
                for run in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1):
                    if len(run)<6:continue
                    result=compact_run(native[run],width[run])
                    result['_native_uv']=native[run].tolist();result['_native_iqr']=width[run].tolist();clean.append(result)
            segments=clean
            if not segments:
                diagnostics['nonvisible_edges'].append(original['id']);continue
        for segment in segments:sparse_native_segment(segment,240)
        scans[original['id']]=segments
        features.append(dict(id=original['id'],label_zh=original['label_zh'],
            physical_surface='board',target_id=original['target_id'],world_curve=original['target_id'],
            segments=segments,allowed_for_fit=False))
    by={f['id']:f for f in features};top=by['board_top'];evidence=deepcopy(top['segments'])
    straight_source_boundary(top,evidence,[[0,1919]])
    estimate=top['source_straight_line_estimate']
    intervals=visible_intervals(mask,estimate['y_intercept_px'],estimate['y_per_x'],0,1919)
    straight_source_boundary(top,evidence,intervals)
    lines={fid:fit_axis_line(segments,1 if 'vertical' in fid else 0)
           for fid,segments in scans.items() if fid.startswith('Ad2.')}
    corners={}
    for side in ('left','right'):
        vertical=lines.get('Ad2.'+side+'_vertical')
        for level in ('top','bottom'):
            key=side+'_'+level;horizontal=lines.get('Ad2.'+level+'_edge')
            if vertical is None or horizontal is None:
                diagnostics['nonvisible_corners'].append(key);continue
            y=(horizontal[0]*vertical[1]+horizontal[1])/(1-horizontal[0]*vertical[0])
            uv=np.array([vertical[0]*y+vertical[1],y]);x,y0=np.rint(uv).astype(int)
            if not(4<=x<1916 and 4<=y0<1076) or mask[y0-3:y0+4,x-3:x+4].any():
                diagnostics['nonvisible_corners'].append(key);continue
            # The rectangle vertex is a same-source edge-line intersection.
            # It is only emitted when both adjoining measured materials exist.
            corners[key]=uv.tolist()
            horizontal_feature=by['Ad2.'+level+'_edge']
            segment=horizontal_feature['segments'][0 if side=='left' else -1]
            pos=0 if side=='left' else -1
            segment['uv'][pos]=uv.tolist()
    space_boundary_samples(features,240)
    for key,point in corners.items():
        side,level=key.split('_');segments=by['Ad2.'+level+'_edge']['segments']
        segment=segments[0 if side=='left' else -1];index=0 if side=='left' else len(segment['uv'])-1
        segment['point_bindings'][index].update(kind='corner',physical_target='Ad2.'+key,
            required_residual='point_to_corresponding_model_corner_2d',
            measurement_origin='current_source_edge_line_intersection')
        segment.setdefault('corner_vertex_indices',{})[key]=index
        segment['source_corner_line_equations']={k:v.tolist()for k,v in lines.items()}
    return features,diagnostics


def current_print_features(raw,mask,spec,matrix):
    """Current white/magenta proposal and native normal profiles, including tiny arcs."""
    from rinkspace.vision_instances import source_print_boundary_contours
    x0,y0,x1,y1=spec['print_roi']
    polygon=transform([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],matrix)
    support=np.zeros(raw.shape[:2],np.uint8)
    cv2.fillConvexPoly(support,np.rint(polygon).astype(np.int32),255)
    green=raw[:,:,1];values=green[(support>0)&~mask]
    if not len(values):return [],dict(status='print_fully_occluded_or_outside_image')
    threshold,_=cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary=((green>threshold)&(support>0)&~mask).astype(np.uint8)*255
    contours=source_print_boundary_contours({'ink_mask':binary},stride=1)
    inverse=cv2.invertAffineTransform(matrix);result=[];seen={};rejected=[]
    registered=read(MODEL/'feature_registry.json')['targets']
    scale=float(np.sqrt(abs(np.linalg.det(matrix[:,:2]))));radius=int(np.clip(round(10*scale),8,18))
    clearance=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
    for contour in contours:
        path=np.asarray(contour['uv'],float)
        if len(path)<6:continue
        ref=transform(path,inverse);x=float(np.median(ref[:,0]))
        matches=[r for r in spec['printed_letter_search_intervals']if r['x_interval'][0]<=x<r['x_interval'][1]]
        identity_override=None
        extent=np.ptp(ref,axis=0)
        if 1030<=x<1083 and extent[0]>1.25*max(1.,extent[1]) and extent[1]<35:
            # This short horizontal white block is the visible hyphen. Its
            # reference200 search interval ends inside a blurred hidden span,
            # so current-source shape and print order resolve that identity.
            matches=[r for r in spec['printed_letter_search_intervals']if r['contour_key']=='hyphen']
            identity_override='current_source_horizontal_white_hyphen_in_labelled_print_order'
        if len(matches)!=1:continue
        name=matches[0]['contour_key']+('_inner'if contour['internal_hole'] else '')
        if 'Ad2.print.'+name not in registered:continue
        rows=[];votes=[];closed=bool(contour['closed'])
        for i,point in enumerate(path):
            before=(i-4)%len(path)if closed else max(0,i-4)
            after=(i+4)%len(path)if closed else min(len(path)-1,i+4)
            tangent=path[after]-path[before];norm=np.linalg.norm(tangent)
            if norm<.1:rows.append(None);continue
            normal=np.array([-tangent[1],tangent[0]])/norm
            coords=point+np.arange(-radius,radius+1)[:,None]*normal
            if coords[:,0].min()<0 or coords[:,0].max()>1919 or coords[:,1].min()<0 or coords[:,1].max()>1079:
                rows.append(None);continue
            ix,iy=np.rint(point).astype(int)
            if clearance[iy,ix]:rows.append(None);continue
            colors=cv2.remap(raw,coords[:,0].astype('float32')[:,None],coords[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float)
            sides=np.array([np.median(colors[:5],axis=0),np.median(colors[-5:],axis=0)])
            white=sides[np.argmax(sides[:,1])];purple=sides[np.argmin(sides[:,1])]
            paired=bool(white[1]>=135 and white[1]-purple[1]>=35 and np.ptp(white)<65 and (purple[0]+purple[2])/2-purple[1]>=23)
            votes.append(paired)
            measured=transition(colors[:,1],radius)
            if measured is None:rows.append(None);continue
            qs,strength=measured
            uv=point+(qs[1]-radius)*normal;mx,my=np.rint(uv).astype(int)
            if abs(qs[1]-radius)>radius*.5 or qs[2]-qs[0]<1.0 or clearance[my,mx]:rows.append(None);continue
            # Short components need paired material proof at each retained row;
            # larger glyph corners are authenticated by their same component.
            rows.append((uv,qs[2]-qs[0],paired))
        if not votes or np.mean(votes)<.4:
            rejected.append(dict(name=name,proposal_count=len(path),pair_fraction=float(np.mean(votes))if votes else 0));continue
        if len(path)<20:rows=[r if r is not None and r[2] else None for r in rows]
        if all(r is not None for r in rows):
            segments=[compact_run([r[0]for r in rows],[r[1]for r in rows],closed=closed)]
        else:
            ordered=rows+[None]
            if closed:
                begin=next(i for i,r in enumerate(rows)if r is None)
                ordered=rows[begin+1:]+rows[:begin+1]
            runs=[];run=[]
            for row in ordered:
                if row is None:
                    if len(run)>=3:runs.append(run)
                    run=[]
                else:run.append(row)
            segments=[compact_run([r[0]for r in run],[r[1]for r in run])for run in runs]
            segments=[s for s in segments if len(s['uv'])>=2 and np.linalg.norm(np.array(s['uv'][-1])-s['uv'][0])>=1.5]
        if not segments:continue
        fid='Ad2.print.'+name;seen[fid]=seen.get(fid,0)+1
        result.append(dict(id=fid+'.fragment'+str(seen[fid]),target_id=fid,label=name,
            physical_surface='board',world_correspondence_status='named_independent_material_template',
            source_contour_internal_hole=bool(contour['internal_hole']),segments=segments,
            source_magenta_white_pair_fraction=float(np.mean(votes)),source_print_threshold=float(threshold),
            source_print_identity=dict(method='current_source_reading_with_reference200_search_intervals',
                reference_x_interval=matches[0]['x_interval'],coordinates_generated_by_search_interval=False,
                current_source_shape_identity_override=identity_override),
            allowed_for_fit=False))
    # A partly exposed U bottom arc can be below Otsu's component proposal
    # threshold. These labelled search stations still read the current image;
    # foreground exclusion and material-pair checks decide their visibility.
    for extra in spec.get('short_print_profile_searches',[]):
        target=extra['target_id']
        existing_paths=[np.asarray(s['uv'])for f in result if f['target_id']==target for s in f['segments']]
        anchors=transform(extra['search_anchors'],matrix);uv=[];widths=[];evidence=[]
        for ax,ay in anchors:
            x=int(round(ax));centre=int(round(ay));span=int(np.clip(round(extra['radius_px']*scale),8,24))
            ys=np.arange(centre-span,centre+span+1)
            if not(0<=x<1920)or ys[0]<0 or ys[-1]>=1080:continue
            colors=raw[ys,x].astype(float);found=transition(colors[:,1],span)
            if found is None:continue
            q,strength=found;y=float(ys[0]+q[1]);lo=int(np.floor(ys[0]+q[0]));hi=int(np.ceil(ys[0]+q[2]))
            if abs(y-ay)>span*.6 or mask[max(0,lo):min(1080,hi+1),x].any():continue
            sides=np.array([np.median(colors[:4],axis=0),np.median(colors[-4:],axis=0)])
            white=sides[np.argmax(sides[:,1])];purple=sides[np.argmin(sides[:,1])]
            if white[1]<130 or white[1]-purple[1]<30 or np.ptp(white)>70 or (purple[0]+purple[2])/2-purple[1]<20:continue
            point=np.array([float(x),y]);distance=np.inf
            for path in existing_paths:
                for a,b in zip(path[:-1],path[1:]):
                    d=b-a;t=np.clip((point-a)@d/max(float(d@d),1e-12),0.,1.)
                    distance=min(distance,float(np.linalg.norm(point-(a+t*d))))
            if distance<=max(2.,scale*.75):continue
            uv.append([float(x),y]);widths.append(q[2]-q[0]);evidence.append(dict(native_x=x,native_y_range=[int(ys[0]),int(ys[-1])],
                native_BGR_profile=colors.tolist(),quartile_indices=q,green_contrast=strength,
                source_visual_materials_verified=['white_Ad2_print','magenta_Ad2_background']))
        if len(uv)<2:continue
        result.append(dict(id=target+'.short_visible_arc',target_id=target,label=target.split('.')[-1],
            physical_surface='board',allowed_for_fit=False,
            segments=[dict(uv=uv,vertex_count=len(uv),closed=False,transition_iqr_px=widths,
                localization_sigma_px=(np.array(widths)/1.349).tolist(),protected_vertex_indices=list(range(len(uv))),
                source_evidence=evidence,measurement_origin='native_colour_half_transition_in_labelled_short_arc_search',
                pixels_measured_in_this_frame=True)]))
        seen[target]=1
    return result,dict(proposal_count=len(contours),rejected_material_proposals=rejected,
        visible_letter_targets=sorted(seen),short_contours_reread=True,profile_radius_px=radius)


def print_navigation_from_ad_edges(features,reference_packet,fallback):
    """Refine letter search identity from the current native rectangular border.

    Intersections behind the foreground are only search geometry here. Their
    presence never emits an observed source corner or a letter vertex.
    """
    by={f['id']:f for f in features};reference={}
    for f in reference_packet['features']:
        for s in f['segments']:
            for uv,binding in zip(s['uv'],s.get('point_bindings',[])):
                if binding.get('kind')=='corner':reference[binding['physical_target']]=np.asarray(uv)
    required=['Ad2.left_vertical','Ad2.top_edge','Ad2.bottom_edge']
    if any(k not in by for k in required):return fallback,dict(status='source_ad_edges_incomplete_LK_search_retained')
    vertical=fit_axis_line(by['Ad2.left_vertical']['segments'],1);points=[]
    for level in ('top','bottom'):
        horizontal=fit_axis_line(by['Ad2.'+level+'_edge']['segments'],0)
        y=(horizontal[0]*vertical[1]+horizontal[1])/(1-horizontal[0]*vertical[0]);points.append(np.array([vertical[0]*y+vertical[1],y]))
    a,b=[reference['Ad2.left_'+level]for level in ('top','bottom')];c,d=points
    old=b-a;new=d-c;den=float(old@old)
    measured_scale=float(np.linalg.norm(new)/np.linalg.norm(old))
    fallback_scale=float(np.sqrt(abs(np.linalg.det(fallback[:,:2]))))
    # A briefly exposed vertical edge has little leverage for rotation. Its
    # local blurred chroma slope must not rotate the print ROI onto the bottom
    # white border. The robust static-board tracks determine search rotation;
    # the independently measured ad height and left corner determine scale
    # and translation. None of these values becomes a source letter vertex.
    linear=fallback[:,:2]/fallback_scale*measured_scale
    matrix=np.column_stack([linear,c-linear@a])
    ratio=float(np.sqrt(abs(np.linalg.det(matrix[:,:2])/np.linalg.det(fallback[:,:2]))))
    if not .90<ratio<1.10:return fallback,dict(status='ad_source_scale_inconsistent_LK_search_retained',source_scale_ratio=ratio)
    return matrix,dict(status='source_ad_left_edge_height_and_rectangular_intersections_search_only',
        matrix=matrix.tolist(),source_scale_ratio=ratio,source_letter_vertices_generated_by_navigation=False)


def recover_open_print_hole_identities(features,matrix,reference_packet):
    """A foreground cut changes proposal topology, not physical O/A inner arcs.

    Reference source paths navigate identity only. The retained current native
    readouts, their visibility breaks, and their material quartiles stay intact.
    """
    paths={key:[] for key in ('Ad2.print.COM_O','Ad2.print.COM_O_inner','Ad2.print.SKATING_A','Ad2.print.SKATING_A_inner')}
    for f in reference_packet['features']:
        if f.get('target_id')not in paths:continue
        for s in f['segments']:
            path=np.asarray(s['uv'],float)
            if s.get('closed'):path=np.vstack([path,path[0]])
            paths[f['target_id']].append(path)
    # A is covered in the selected200 example; its already approved native100
    # inner/outer source paths supply search identity, never new observations.
    old=read(PARENT/'opening_100_50fps/frames/000100.json')
    def corner(packet,key):
        return next(np.asarray(uv)for f in packet['features']for s in f['segments']
                    for binding,uv in zip(s.get('point_bindings',[]),s['uv'])if binding['physical_target']==key)
    a,b=corner(old,'Ad2.top_left'),corner(old,'Ad2.bottom_left')
    c,d=corner(reference_packet,'Ad2.left_top'),corner(reference_packet,'Ad2.left_bottom')
    old_delta=b-a;new_delta=d-c;den=float(old_delta@old_delta)
    u=float(new_delta@old_delta)/den;v=float(new_delta[1]*old_delta[0]-new_delta[0]*old_delta[1])/den
    linear=np.array([[u,-v],[v,u]]);old_to_reference=np.column_stack([linear,c-linear@a])
    for f in old['features']:
        if f.get('label')not in ('A','A_inner'):continue
        target='Ad2.print.SKATING_A'+('_inner'if f['label']=='A_inner'else '')
        for s in f['segments']:
            path=transform(s['uv'],old_to_reference)
            if s.get('closed'):path=np.vstack([path,path[0]])
            paths[target].append(path)
    def distances(points,target):
        result=np.full(len(points),np.inf)
        for path in paths[target]:
            for a,b in zip(path[:-1],path[1:]):
                d=b-a;t=np.clip(((points-a)@d)/max(float(d@d),1e-12),0,1)
                result=np.minimum(result,np.linalg.norm(points-(a+t[:,None]*d),axis=1))
        return result
    inverse=cv2.invertAffineTransform(matrix);result=[];changed=[]
    for feature in features:
        family=feature.get('target_id')
        if family not in ('Ad2.print.COM_O','Ad2.print.SKATING_A'):result.append(feature);continue
        groups={}
        for si,segment in enumerate(feature['segments']):
            points=transform(segment['uv'],inverse)
            outside=distances(points,family);inside=distances(points,family+'_inner')
            bound,gap=(3.5,3.)if family.endswith('SKATING_A')else (5.,2.)
            inner=(inside<=bound)&(inside+gap<outside)
            if segment.get('closed'):inner[:]=False
            if inner.any():
                changed.append(dict(source_feature=feature['id'],source_segment=si,
                    median_reference_outer_distance_px=float(np.median(outside)),
                    median_reference_inner_distance_px=float(np.median(inside)),
                    recovered_inner_source_vertex_indices=np.flatnonzero(inner).tolist(),
                    reason='foreground_opened_white_component_hole_while_current_inner_material_arc_remained_visible'))
            runs=np.split(np.arange(len(points)),np.flatnonzero(inner[1:]!=inner[:-1])+1)
            for run in runs:
                target=family+'_inner'if inner[run[0]]else family
                selected=deepcopy(segment);n=len(segment['uv'])
                for key in ('uv','transition_iqr_px','localization_sigma_px','source_y_interval_px','retained_pre_compaction_vertex_indices'):
                    if key in selected and len(selected[key])==n:selected[key]=[selected[key][i]for i in run]
                if len(runs)>1:
                    selected.update(closed=False,vertex_count=len(run),hidden_span_filled=False,
                        protected_vertex_indices=sorted({0,len(run)-1}),
                        source_identity_split_vertex_indices=run.tolist(),
                        measurement_origin='retained_current_native_material_readouts_split_at_physical_inner_outer_identity_change')
                groups.setdefault(target,[]).append(selected)
        for target,segments in groups.items():
            f=deepcopy(feature);f.update(target_id=target,label=target.split('.')[-1],segments=segments)
            if target.endswith('_inner'):
                f['source_proposal_contour_internal_hole']=f.get('source_contour_internal_hole')
                f['source_contour_internal_hole']=True
                f['source_print_identity']['physical_inner_arc_retained_despite_mask_proposal_opening']=True
                f['source_print_identity']['identity_navigation_supplies_no_observation_coordinates']=True
            result.append(f)
    seen={}
    for f in result:
        if '.print.'not in f['id']:continue
        if f['id'].endswith('.short_visible_arc'):continue
        target=f['target_id'];seen[target]=seen.get(target,0)+1;f['id']=target+'.fragment'+str(seen[target])
    return result,changed


def compact_current_curvature(features,minimum_budget,reference_errors,scale):
    """Same measured-vertex compaction with a per-material curvature bound.

    Cache unaffected contour choices. The sparse count can rise when additional
    visible curvature requires it; no source vertex is moved or synthesized.
    """
    records=[]
    for feature in features:
        for segment in feature['segments']:
            p=np.asarray(segment['uv'],float);n=len(p);protected=set(segment.get('protected_vertex_indices',[]))
            closed=bool(segment.get('closed'))
            if closed:protected.update(int(i)for i in [np.argmin(p[:,0]),np.argmax(p[:,0]),np.argmin(p[:,1]),np.argmax(p[:,1])])
            else:protected.update([0,n-1])
            bound=max(2.5,reference_errors.get(feature['target_id'],2.5))*scale
            records.append(dict(segment=segment,p=p,keep=list(range(n)),protected=protected,bound=bound,closed=closed))
    def error(row,keep):
        p=row['p'];poly=p[keep]
        if len(p)==1:return 0.
        if row['closed']:poly=np.vstack([poly,poly[0]])
        distances=np.full(len(p),np.inf)
        for a,b in zip(poly[:-1],poly[1:]):
            d=b-a;t=np.clip(((p-a)@d)/max(float(d@d),1e-12),0,1)
            distances=np.minimum(distances,np.linalg.norm(p-(a+t[:,None]*d),axis=1))
        return float(np.max(distances))
    def choice(ri):
        row=records[ri];keep=row['keep']
        if len(keep)<=(6 if row['closed']else 2):return None
        options=[]
        for index in keep:
            if index in row['protected']:continue
            selected=[i for i in keep if i!=index];deviation=error(row,selected)
            options.append((deviation/row['bound'],deviation,ri,index,selected))
        return min(options)if options else None
    before=sum(len(r['keep'])for r in records);count=before
    cache=[choice(ri)for ri in range(len(records))]
    while count>minimum_budget:
        available=[c for c in cache if c is not None]
        if not available:break
        ratio,_,ri,_,selected=min(available)
        if ratio>1.:break
        records[ri]['keep']=selected;count-=1;cache[ri]=choice(ri)
    for row in records:
        segment=row['segment'];keep=row['keep'];n=len(row['p'])
        for key in ('uv','transition_iqr_px','localization_sigma_px','source_y_interval_px'):
            if key in segment and len(segment[key])==n:segment[key]=[segment[key][i]for i in keep]
        segment.update(vertex_count=len(keep),retained_pre_compaction_vertex_indices=keep,
            pre_compaction_vertex_count=n,sparse_representation_max_deviation_px=round(error(row,keep),3),
            current_curvature_bound_px=row['bound'])
        if 'reconstruction_max_px'in segment:
            segment['pre_compaction_reconstruction_max_px']=segment.pop('reconstruction_max_px')
            segment['pre_compaction_simplification_tolerance_px']=segment.pop('simplification_tolerance_px')
        if 'protected_vertex_indices'in segment:segment['protected_vertex_indices']=[keep.index(i)for i in row['protected']]
    return dict(algorithm_selected_sparse_budget=count,baseline_density_count=minimum_budget,
        points_before=before,points_after=count,all_segments_and_closure_retained=True,
        original_retained_coordinates_unchanged=True,retained_curvature_compared_with_selected200=True,
        image_accuracy_accepted=False)


def contact_sheets(data,counts):
    records=sorted(counts,key=lambda r:r['index'])
    for begin in range(0,len(records),25):
        part=records[begin:begin+25];sheet=np.zeros((270*5,480*5,3),np.uint8)
        for offset,row in enumerate(part):
            raw=cv2.imread(str(data/f"board_overlays/{row['index']:06d}.png"))
            tile=cv2.resize(raw,(480,270),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,f"{row['index']} / {row['points']} board+print",(6,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            y,x=divmod(offset,5);sheet[y*270:(y+1)*270,x*480:(x+1)*480]=tile
        cv2.imwrite(str(data/f"board_contact_sheet_{part[0]['index']:06d}_{part[-1]['index']:06d}.png"),sheet)


def board_unresolved_regions(index):
    if index!=175:return []
    return [dict(id='Ad2.print.ISU_I.mask_adjacent_short_fragments',physical_target='Ad2.print.ISU_I',
        status='short_source_fragments_excluded_by_current_foreground_band',
        source_regions_xyxy=[[998,330,1025,350],[998,417,1025,435]],
        reason_zh='ISU_I贴黑袖的极短白片在当前前景排除带内，未强行读取；不表示该字符整段不存在。',
        whole_character_declared_absent=False,inferred_points_emitted=False)]


def count_row(packet):
    features=packet['features'];nav=packet['background_navigation']
    return dict(index=packet['programme_frame_index_1based'],points=packet['point_count'],
        board_points=sum(len(s['uv'])for f in features if '.print.'not in f['id']for s in f['segments']),
        glyph_points=sum(len(s['uv'])for f in features if '.print.'in f['id']for s in f['segments']),
        letter_targets=sorted({f['target_id']for f in features if '.print.'in f['id']}),
        navigation_median_px=nav['median_navigation_error_px'],
        nonvisible_corners=packet['diagnostics']['board']['nonvisible_corners'])


def run(indices,data=DATA):
    cv2.setNumThreads(2);spec=read(REFERENCE/'specification.json');motion=BoardNavigation(data)
    manifest=read(data/'frame_manifest.json');records={int(r['index']):r for r in manifest['frames']}
    start_count=read(PARENT/'opening_100_50fps/frames/000100.json')['point_count']
    endpoint_packet=read(REFERENCE/'frame_000200_points.json')
    end_count=sum(len(s['uv'])for f in endpoint_packet['features']if f.get('physical_surface')=='board'for s in f['segments'])
    reference_errors={}
    for f in endpoint_packet['features']:
        if '.print.' not in f['id']:continue
        reference_errors[f['target_id']]=max(reference_errors.get(f['target_id'],0.),
            max(s.get('sparse_representation_max_deviation_px',0.)for s in f['segments']))
    start_scale=float(np.sqrt(abs(np.linalg.det(motion.seeds[101][:,:2]))))
    counts=[]
    for index in indices:
        record=records[index];raw=cv2.imread(str(record['image_path']));mask=cv2.imread(str(record['mask_path']),0)>0
        matrix,nav=motion.current(raw,mask,index)
        features,board_diagnostic=board_features(raw,mask,spec,matrix)
        print_matrix,print_nav=print_navigation_from_ad_edges(features,endpoint_packet,matrix)
        glyphs,print_diagnostic=current_print_features(raw,mask,spec,print_matrix);features+=glyphs
        print_diagnostic['navigation']=print_nav
        features,identity_changes=recover_open_print_hole_identities(features,print_matrix,endpoint_packet)
        print_diagnostic['opened_inner_arc_identity_recoveries']=identity_changes
        print_diagnostic['visible_letter_targets']=sorted({f['target_id']for f in features if '.print.'in f['id']})
        before=sum(len(s['uv'])for f in features for s in f['segments'])
        # Same geometric-error ordering as the selected-frame interface. It
        # keeps all contours, their visible ends, and the long-edge stations.
        scale=float(np.sqrt(abs(np.linalg.det(matrix[:,:2]))))
        fraction=float(np.clip((scale-1.)/(start_scale-1.),0.,1.))
        budget=int(round(start_count*fraction+end_count*(1.-fraction)))
        minimum=0
        for feature in features:
            for segment in feature['segments']:
                p=np.asarray(segment['uv']);protected=set(segment.get('protected_vertex_indices',[]))
                if segment.get('closed'):
                    protected.update(int(i)for i in [np.argmin(p[:,0]),np.argmax(p[:,0]),np.argmin(p[:,1]),np.argmax(p[:,1])])
                    minimum+=max(len(protected),min(len(p),6))
                else:
                    protected.update([0,len(p)-1]);minimum+=len(protected)
        budget=max(budget,minimum)
        compaction=compact_current_curvature(features,min(before,budget),reference_errors,scale)
        compaction['density_navigation']=dict(method='current_visible_curvature_and_source_apparent_scale_compared_with_selected200',
            apparent_scale_from_200=scale,reference100_point_count=start_count,reference200_board_print_count=end_count,
            current_geometry_can_raise_point_count=True,reference200_per_target_sparse_deviation_px=reference_errors,
            short_or_missing_reference_contour_error_scale_px=2.5,pixels_moved_by_density_policy=False)
        remove_temporary_readouts(features)
        count=sum(len(s['uv'])for f in features for s in f['segments'])
        for feature in features:
            for segment in feature['segments']:
                xy=np.rint(segment['uv']).astype(int)
                if mask[xy[:,1],xy[:,0]].any():raise ValueError('Current foreground point '+str(index)+' '+feature['id'])
        source=deepcopy(record['source'])
        source.update(mask_path=str(record['mask_path']),mask_sha256=sha(record['mask_path']),
            source_points_extracted=True,optimization_performed=False)
        packet=dict(schema='rinkspace.single-frame-source-review.v1',source=source,
            source_pts=int(source['native_pts']),programme_frame_index_1based=index,
            batch_sequence_index_1based=index,frame_count=1,features=features,point_count=count,
            status='board_and_print_candidate_waiting_for_batch_review',user_accepted=False,
            sparse_compaction=compaction,background_navigation=nav,diagnostics=dict(board=board_diagnostic,print=print_diagnostic),
            unresolved_source_regions=board_unresolved_regions(index),
            user_hard_point_ceiling=500,observation_coordinates_from_projection=False,
            optimization_performed=False,independent_accuracy_claim=False,
            excluded_features_by_user=['blue_strip_lower_edge','yellow_strip_upper_edge','board_panel_seam'])
        write(data/f'board_candidates/{index:06d}.json',packet)
        overlay=native_overlay(raw,list(source_segments(packet)))
        (data/'board_overlays').mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),overlay,[cv2.IMWRITE_PNG_COMPRESSION,1])
        row=count_row(packet)
        counts.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    write(data/f'board_counts_{indices[0]:06d}_{indices[-1]:06d}.json',counts)
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--indices',default='101:199');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();indices=list(range(int(args.indices.split(':')[0]),int(args.indices.split(':')[1])+1))if ':'in args.indices else list(map(int,args.indices.split(',')))
    if args.workers==1:counts=run(indices)
    else:
        from concurrent.futures import ProcessPoolExecutor
        partitions=[list(map(int,p))for p in np.array_split(indices,args.workers)if len(p)]
        with ProcessPoolExecutor(max_workers=args.workers)as executor:counts=[row for part in executor.map(run,partitions)for row in part]
    all_counts=[count_row(read(p))for p in sorted((DATA/'board_candidates').glob('0*.json'))if 101<=int(p.stem)<=199]
    write(DATA/'board_counts.json',all_counts)
    write(DATA/f"board_counts_{all_counts[0]['index']:06d}_{all_counts[-1]['index']:06d}.json",all_counts)
    contact_sheets(DATA,all_counts)
