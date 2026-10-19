"""Repair selected-shot board readouts from whole current native materials.

The historical failed delivery is an input only.  This CLI writes solely to a
separate data/model snapshot.  Source joint curves cover complete visible rail
intervals; an advertisement contour is read as one current connected material,
never as independently displaced tiny profile fragments.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .extract_first_shot_board_source import (
    read, write, sha, profile_read, visible_projected_runs, native_overlay,
    source_segments, refined_ad_mapping, transform,
)
from .extract_shot2_board import extract_frame
from .framewise_model import feature_world
from .projection import project_fast
from .extract_shot2_board import build_current_rectangle_charts
from .repair_first_shot_board_patterns import circle_feature


def runs(indices):
    indices=list(indices)
    if not indices:return []
    return [a.tolist() for a in np.split(np.asarray(indices),
        np.flatnonzero(np.diff(indices)>1)+1) if len(a)>=2]


def point_segment(uv, target, *, closed=False, evidence=None, joint=False):
    uv=np.asarray(uv,float)
    return dict(uv=uv.tolist(),closed=bool(closed),vertex_count=len(uv),
        measurement_origin='current_native_whole_material_joint_curve' if joint
                           else 'current_native_whole_connected_material_boundary',
        pixels_measured_in_this_frame=not joint,
        source_evidence_measured_in_this_frame=True,
        representative_points_share_line_estimate=joint,
        source_native_readout_uv=np.asarray(evidence if evidence is not None else uv,float).tolist(),
        point_bindings=[dict(kind='print_contour_sample' if '.print.' in target else 'edge_sample',
            physical_target=target,required_residual='point_to_named_finite_print_contour' if '.print.' in target
            else 'point_to_named_finite_model_edge',
            measurement_origin='current_native_whole_material_source_readout') for _ in uv],
        localization_sigma_px=[1.5]*len(uv),transition_iqr_px=[3.]*len(uv),
        source_clear_component_identity_verified_by_current_shape=not joint)


def robust_curve(points, degree=3):
    p=np.asarray(points,float)
    degree=min(degree,len(p)-1)
    keep=np.ones(len(p),bool)
    for _ in range(4):
        fit=np.polynomial.Polynomial.fit(p[keep,0],p[keep,1],degree)
        error=np.abs(p[:,1]-fit(p[:,0]))
        limit=max(1.5,3*np.median(error[keep]))
        updated=error<limit
        if updated.sum()<8:break
        keep=updated
    return fit,p[keep]


def complete_board_edges(raw, mask, features, camera, scene, registry):
    """Same first-shot colour half-transitions, globally shared source curve.

    Yellow bottom profiles are searched only around the already source-read
    body edge.  The faint yellow reflection farther down is never a seed.
    Weak-colour intervals share the continuous current source curve estimate;
    representative vertices are explicitly correlated fitted observations.
    """
    result=[];receipts=[]
    for role,prefix,weights,direction in (
        ('top','board_top',[1,-.5,-.5],[0,1]),
        ('contact','board_contact',[-1,.5,.5],[0,-1]),
    ):
        old=[f for f in features if f['id'].startswith(prefix+'.')]
        support=[uv for f in old for s in f['segments']
                 for uv in s.get('source_native_readout_uv',s['uv'])]
        if len(support)<8:raise ValueError('Missing current body-edge support: '+role)
        initial,evidence=robust_curve(support)
        x=np.arange(0.,1920.,3.)
        query=np.column_stack((x,initial(x)))
        # Preserve the accepted material/sign definition.  A small radius
        # prevents the yellow-body bottom from jumping to its reflection.
        rows,_=profile_read(raw,mask,query,weights,9.,maximum_blur=14.,
            minimum_amplitude=7.,navigation_tolerance=6.,gradient_direction=direction)
        measured=np.asarray([row[0] for row in rows if row is not None])
        if len(measured)<30:raise ValueError('Insufficient whole-width native edge readout: '+role)
        fit,evidence=robust_curve(measured)
        xy=np.column_stack((x,fit(x)))
        inside=(xy[:,1]>4)&(xy[:,1]<1075)
        xi,yi=np.rint(np.clip(xy,[0,0],[1919,1079])).astype(int).T
        # Only actual current person pixels cut a visible physical interval.
        visible=inside&~mask[yi,xi]
        seeds=[]
        for tid,target in registry['targets'].items():
            if target.get('kind')!='board_edge' or target.get('role')!=role or not target.get('part'):continue
            for p in visible_projected_runs(camera,feature_world(scene,target)):
                if len(p)>=2 and np.min(p[:,1])<900:seeds.append((tid,p))
        if not seeds:raise ValueError('Missing finite board parts: '+role)
        distances=[]
        for _,p in seeds:
            # Assignment navigates finite physical identity only; every output
            # coordinate is evaluated from the current native source fit.
            distances.append(np.min(np.linalg.norm(xy[:,None]-p[None],axis=2),axis=1))
        labels=np.argmin(np.asarray(distances),axis=0)
        for label,(tid,_) in enumerate(seeds):
            segments=[]
            for keep in runs(np.flatnonzero(visible&(labels==label))):
                first,last=keep[0],keep[-1]
                stations=np.linspace(x[first],x[last],max(2,int(np.ceil((x[last]-x[first])/150))+1))
                uv=np.column_stack((stations,fit(stations)))
                current=evidence[(evidence[:,0]>=x[first])&(evidence[:,0]<=x[last])]
                segment=point_segment(uv,tid,evidence=current,joint=True)
                segment.update(native_material_definition='blue_cap_upper_body_edge' if role=='top'
                    else 'yellow_kickplate_body_lower_edge_reflection_excluded',
                    source_curve_fit_support_count=len(evidence),
                    visible_interval_not_cut_by_profile_rejection=True,
                    current_mask_only_occlusion_split=True,
                    current_interval_local_colour_support_count=len(current),
                    continuous_low_contrast_source_interval=(len(current)<len(keep)*.5))
                segments.append(segment)
            if segments:result.append(dict(id=prefix+'.'+tid.rsplit('.',1)[1],
                label_zh=tid,physical_surface='board',target_id=tid,world_curve=tid,
                review_layer=prefix,segments=segments,allowed_for_fit=False))
        receipts.append(dict(role=role,current_native_support_count=len(evidence),
            visible_source_intervals=[[float(x[k[0]]),float(x[k[-1]])]for k in runs(np.flatnonzero(visible))],
            median_residual_px=float(np.median(np.abs(evidence[:,1]-fit(evidence[:,0])))),
            body_definition='blue upper body edge' if role=='top' else 'yellow body lower edge, reflection excluded'))
    return result,receipts


def current_component(raw, mask, seed, *, hole=False, bright=True, epsilon=2.2, colour=False, semantic=False, phone=False, red=False):
    """Read and simplify a complete same-frame connected entity together."""
    seed=np.asarray(seed,float)
    if len(seed)<3 or not np.isfinite(seed).all():return []
    padding=4 if semantic else 12
    lo=np.floor(seed.min(0)-padding).astype(int);hi=np.ceil(seed.max(0)+padding).astype(int)
    x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[1920,1080])
    if min(x1-x0,y1-y0)<5:return []
    crop=raw[y0:y1,x0:x1]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    if phone:
        smooth=cv2.GaussianBlur(gray,(0,0),2.)
        interior=np.zeros(gray.shape,np.uint8);cv2.fillPoly(interior,[np.rint(seed-[x0,y0]).astype('int32')],1)
        inner=cv2.erode(interior,np.ones((9,9),np.uint8))>0
        outer=(cv2.dilate(interior,np.ones((13,13),np.uint8))>0)&~(interior>0)
        if not inner.any() or not outer.any():return []
        level=(float(np.median(smooth[inner]))+float(np.median(smooth[outer])))/2
        binary=(smooth>level).astype('uint8')*255
    elif colour:
        # Pale-blue printed ribbon vs magenta ad backing.  This is the actual
        # material signal, jointly smoothed before a whole contour is traced.
        b,g,r=crop.astype(float).transpose(2,0,1)
        signal=cv2.GaussianBlur((b-r).astype('float32'),(0,0),1.2)
        binary=(signal>15).astype('uint8')*255
    elif red:
        b,g,r=crop.astype(float).transpose(2,0,1)
        signal=cv2.GaussianBlur((r-(b+g)/2).astype('float32'),(0,0),.8)
        level=max(10.,float(np.percentile(signal,90))*.5)
        binary=(signal>level).astype('uint8')*255
    else:
        _,binary=cv2.threshold(gray,0,255,(cv2.THRESH_BINARY if bright else cv2.THRESH_BINARY_INV)+cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:return []
    sb=seed.max(0)-seed.min(0);center=(seed.max(0)+seed.min(0))/2
    choices=[]
    for j,c in enumerate(contours):
        p=c[:,0].astype(float)+[x0,y0];extent=np.ptp(p,axis=0)
        if len(p)<8 or np.min(extent)<2:continue
        parent=int(hierarchy[0,j,3]);depth=0
        while parent>=0:depth+=1;parent=int(hierarchy[0,parent,3])
        is_hole=bool(depth%2)
        if bool(hole)!=bool(is_hole):continue
        delta=np.abs((p.min(0)+p.max(0))/2-center)
        if (delta>np.maximum(10.,sb*.35)).any():continue
        ratio=extent/np.maximum(sb,2.)
        if (ratio<.45).any() or (ratio>(1.8 if semantic else 1.7)).any():continue
        # Comparing a whole component prevents normals from jumping between
        # neighbouring glyph sides even when one side is momentarily weak.
        d=np.linalg.norm(p[:,None]-seed[None],axis=2)
        score=float(np.median(d.min(0))+np.median(d.min(1))+np.linalg.norm(delta))
        choices.append((score,p))
    if not choices:return []
    _,path=min(choices,key=lambda item:item[0])
    q=np.rint(path).astype(int)
    valid=(q[:,0]>x0+1)&(q[:,0]<x1-2)&(q[:,1]>y0+1)&(q[:,1]<y1-2)
    valid&=~mask[q[:,1],q[:,0]]
    # The silhouette next to a person or ROI boundary is not printed material.
    safe=cv2.dilate(mask.astype('uint8'),np.ones((5,5),np.uint8))>0
    valid&=~safe[q[:,1],q[:,0]]
    if phone or colour:
        # UI cells inside the screen cannot become the phone's outer shell.
        # Keep only jointly located perimeter evidence near this finite body;
        # internal zigzags selected by a luma threshold are explicitly rejected.
        distance=np.min(np.linalg.norm(path[:,None]-seed[None],axis=2),axis=1)
        valid&=distance<(7. if phone else 5.)
    output=[]
    if valid.all():
        compact=cv2.approxPolyDP(path.astype('float32')[:,None],epsilon,True)[:,0]
        if len(compact)>=4:output.append((compact,True,path))
    else:
        cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(path)),np.arange(cut+1)]
        for ids in runs(np.flatnonzero(valid[order])):
            p=path[order[ids]]
            if np.linalg.norm(np.ptp(p,axis=0))<8:continue
            compact=cv2.approxPolyDP(p.astype('float32')[:,None],epsilon,False)[:,0]
            if len(compact)>=2:output.append((compact,False,p))
    return output


def ribbon_side_segments(current,tid):
    """Read each real blue-band side as a low-density smooth source curve."""
    output=[]
    for _,closed,evidence in current:
        evidence=np.asarray(evidence,float)
        if len(evidence)<8:continue
        pieces=[evidence]
        if closed:
            a=int(np.argmin(evidence[:,0]));b=int(np.argmax(evidence[:,0]))
            if a>b:a,b=b,a
            pieces=[evidence[a:b+1],np.vstack((evidence[b:],evidence[:a+1]))]
        for p in pieces:
            if np.ptp(p[:,0])<35:continue
            # A crop wall or glyph boundary is not a longitudinal ribbon side.
            tangent=np.diff(p,axis=0);horizontal=np.abs(tangent[:,0]).sum()
            vertical=np.abs(tangent[:,1]).sum()
            if horizontal<vertical*.9:continue
            fit,support=robust_curve(p,degree=2)
            if np.median(np.abs(support[:,1]-fit(support[:,0])))>1.8:continue
            xs=np.linspace(float(support[:,0].min()),float(support[:,0].max()),
                max(3,int(np.ceil(np.ptp(support[:,0])/85))+1))
            segment=point_segment(np.column_stack((xs,fit(xs))),tid,evidence=support,joint=True)
            segment.update(measurement_origin='current_native_joint_smooth_blue_ribbon_longitudinal_side',
                current_source_side_no_roi_or_glyph_connector=True,finite_actual_print_boundary=True)
            output.append(segment)
    return output


def current_hole_in_named_parent(raw,mask,parent):
    """Read the actual white void inside an already identified current glyph.

    The finite parent glyph fixes identity.  Its current native silhouette fixes
    the search region, so a seed camera/chart displacement cannot suppress a
    perfectly visible hole. No source coordinate comes from the hole template.
    """
    outline=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in parent['segments']])
    lo=np.maximum(np.floor(outline.min(0)-12).astype(int),[0,0]);hi=np.minimum(np.ceil(outline.max(0)+12).astype(int),[1920,1080])
    x0,y0=lo;x1,y1=hi
    if min(hi-lo)<10:return []
    gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    otsu,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    candidates=[]
    for threshold in (otsu, min(170.,otsu+15.),min(175.,otsu+30.)):
        binary=(gray<threshold).astype('uint8')*255
        cs,h=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if h is None:continue
        for j,c in enumerate(cs):
            father=int(h[0,j,3]);depth=0
            while father>=0:depth+=1;father=int(h[0,father,3])
            if depth%2!=1:continue
            path=c[:,0].astype(float)+lo;area=abs(cv2.contourArea(c));extent=np.ptp(path,axis=0)
            center=(path.min(0)+path.max(0))/2
            if area<35 or np.min(extent)<6:continue
            if (center<outline.min(0)).any() or (center>outline.max(0)).any():continue
            if ((path[:,0]<=x0+1)|(path[:,0]>=x1-2)|(path[:,1]<=y0+1)|(path[:,1]>=y1-2)).any():continue
            # One O/R parent has one substantial enclosed white void.
            candidates.append((area-abs(float(threshold)-otsu)*2.,path))
    if not candidates:return []
    _,path=max(candidates,key=lambda item:item[0])
    q=np.rint(path).astype(int);safe=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
    valid=~safe[q[:,1],q[:,0]]
    if valid.all():
        p=cv2.approxPolyDP(path.astype('float32')[:,None],1.6,True)[:,0]
        return [(p,True,path)] if len(p)>=4 else []
    cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(path)),np.arange(cut+1)]
    result=[]
    for ids in runs(np.flatnonzero(valid[order])):
        evidence=path[order[ids]]
        if np.linalg.norm(np.ptp(evidence,axis=0))<10:continue
        p=cv2.approxPolyDP(evidence.astype('float32')[:,None],1.6,False)[:,0]
        if len(p)>=2:result.append((p,False,evidence))
    return result


def repair_prints(raw,mask,features,camera,scene,registry,templates):
    result=[];receipts=[]
    mappings={n:refined_ad_mapping(features,camera,scene,registry,n) for n in (1,2,3,4,5,6,7)}
    shifts={}
    for n in (1,2,7,8,22,24,25):
        paths=[]
        for tid,target in registry['targets'].items():
            if target.get('kind')!='ad_print' or int(target.get('advertisement_id',0))!=n or not target.get('semantic_glyph'):continue
            if n==8 and 'Wyndham_icon' in tid:continue
            if target.get('geometry_source')=='scene_existing_external_contour':
                seed,depth=project_fast(camera,feature_world(scene,target,templates))
                if (depth<=0).any():continue
            else:
                contour=templates.get('advertisements',{}).get(str(n),{}).get('contours',{}).get(target.get('contour_key'))
                if contour is None or mappings.get(n) is None:continue
                seed=transform(contour['material_uv'],mappings[n])
            if seed[:,0].max()<0 or seed[:,0].min()>1919:continue
            paths.append((seed,'inner' in tid))
        if paths:
            allp=np.vstack([p for p,_ in paths]);lo=np.maximum(np.floor(allp.min(0)-2).astype(int),[0,0]);hi=np.minimum(np.ceil(allp.max(0)+2).astype(int),[1920,1080])
            if min(hi-lo)>5:
                prototype=np.zeros(tuple((hi-lo)[::-1]),np.uint8)
                for p,hole in sorted(paths,key=lambda pair:pair[1]):cv2.fillPoly(prototype,[np.rint(p-lo).astype('int32')],0 if hole else 255)
                prototype=cv2.GaussianBlur(prototype,(0,0),3.5,1.5)
                margin=45;low=np.maximum(lo-margin,[0,0]);high=np.minimum(hi+margin,[1920,1080])
                current=cv2.cvtColor(raw[low[1]:high[1],low[0]:high[0]],cv2.COLOR_BGR2GRAY)
                if n in (8,25):current=255-current
                if current.shape[0]>=prototype.shape[0] and current.shape[1]>=prototype.shape[1]:
                    scores=cv2.matchTemplate(current,prototype,cv2.TM_CCOEFF_NORMED)
                    _,maximum,_,location=cv2.minMaxLoc(scores)
                    delta=np.asarray(location)+low-lo
                    if maximum>.13 and np.linalg.norm(delta)<60:shifts[n]=delta
    for tid,target in registry['targets'].items():
        if target.get('kind')!='ad_print':continue
        n=int(target['advertisement_id'])
        if target.get('geometry_source') in ('scene_existing_external_contour','scene_existing_board_pattern','scene_shield_anchor_material_chart'):
            seed,depth=project_fast(camera,feature_world(scene,target,templates))
            if (depth<=0).any():continue
            # Existing word/graphic colour roles are physically fixed.
            bright=n==22 and 'component0' not in str(target.get('contour_key'))
            if n==24:bright='inner' in tid
            if n==25:bright='inner' in tid
        else:
            t=templates.get('advertisements',{}).get(str(n),templates)
            record=t.get('contours',{}).get(target.get('contour_key'))
            mapping=mappings.get(n)
            if record is None or mapping is None:continue
            seed=transform(record['material_uv'],mapping)
            bright=n in (1,2,3,6,7)
        if np.max(seed[:,0])<0 or np.min(seed[:,0])>1919 or np.min(seed[:,1])>1079:continue
        parent_tid=target.get('parent_external_target')
        if parent_tid:
            current_parent=next((f for f in result if f['target_id']==parent_tid),None)
            if current_parent is not None:
                parent_uv,parent_depth=project_fast(camera,feature_world(scene,registry['targets'][parent_tid],templates))
                current_uv=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in current_parent['segments']])
                before_lo=parent_uv.min(0);before_span=np.maximum(np.ptp(parent_uv,axis=0),1.)
                after_lo=current_uv.min(0);after_span=np.ptp(current_uv,axis=0)
                seed=after_lo+(seed-before_lo)*np.clip(after_span/before_span,.6,1.6)
        if n in shifts and target.get('semantic_glyph'):seed=seed+shifts[n]
        if target.get('contour_key') in ('outermost_visible_circular_print_stroke','outer_circle_print_stroke'):
            feature,diagnostic=circle_feature(raw,mask,seed,features)
            if feature:
                feature.update(id=tid,target_id=tid,label_zh=tid)
                result.append(feature)
            continue
        bright=target.get('source_bright',bright)
        hole='inner' in tid or target.get('contour_role')=='visible_printed_hole_boundary'
        if parent_tid and current_parent is not None:
            current=current_hole_in_named_parent(raw,mask,current_parent)
        else:
            current=current_component(raw,mask,seed,hole=hole,bright=bright,
                colour=target.get('contour_role')=='visible_blue_ribbon_material_boundary',
                semantic=bool(target.get('semantic_glyph')),
                phone=target.get('contour_role')=='visible_phone_outer_outline',
                red=target.get('contour_role')=='visible_red_shield_material_boundary')
        if not current:continue
        if target.get('contour_role')=='visible_blue_ribbon_material_boundary':
            segments=ribbon_side_segments(current,tid)
        else:segments=[point_segment(p,tid,closed=closed,evidence=evidence)for p,closed,evidence in current]
        if not segments:continue
        result.append(dict(id=tid,target_id=tid,label_zh=tid,physical_surface='board',review_layer='ad2_print',
            world_correspondence_status='named_same_physical_whole_print_component',segments=segments,
            allowed_for_fit=False,current_source_only_readout=True,model_or_template_generated_observation=False,
            source_identity_recovery='whole_current_connected_material_not_independent_normal_fragments'))
        receipts.append(dict(target_id=tid,contours=len(segments),points=sum(len(s['uv'])for s in segments)))
    receipts.append(dict(joint_source_glyph_translations={str(n):d.tolist()for n,d in shifts.items()},
        common_shape_translation_from_current_native_whole_graphic=True))
    return result,receipts


def world_on_body(pixels,camera,scene,reference_world):
    """Inverse projection on confirmed finite board-body triangles only.

    Used only for source print *model initializers*, never for observations.
    Existing ice/top vertices stay unchanged.  Curved perimeter pieces retain
    their actual common 3D support instead of a fabricated planar ad surface.
    """
    pixels=np.asarray(pixels,float)
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    bottom=np.asarray(scene['ice_boundary']['points_xyz'],float)
    reference=np.asarray(reference_world,float);center=reference.mean(0)
    candidates=[]
    for i in range(len(top)):
        j=(i+1)%len(top)
        if np.linalg.norm((top[i,:2]+top[j,:2])/2-center[:2])>max(8.,np.linalg.norm(np.ptp(reference[:,:2],axis=0))*1.3):continue
        for triangle in (np.array([top[i],bottom[i],bottom[j]]),np.array([top[i],bottom[j],top[j]])):
            uv,depth=project_fast(camera,triangle)
            if (depth<=0).any():continue
            basis=np.column_stack((uv[1]-uv[0],uv[2]-uv[0]))
            if abs(np.linalg.det(basis))<.001:continue
            ab=(pixels-uv[0])@np.linalg.inv(basis).T
            bary=np.column_stack((1-ab.sum(1),ab))
            valid=(bary>=-.002).all(1)&(bary<=1.002).all(1)
            weights=bary/depth[None];weights/=weights.sum(1)[:,None]
            xyz=weights@triangle
            candidates.append((valid,xyz,np.linalg.norm(xyz-center,axis=1)))
    if not candidates:raise ValueError('Source print has no confirmed board body support')
    output=[]
    for k in range(len(pixels)):
        choices=[(distance[k],xyz[k])for valid,xyz,distance in candidates if valid[k]]
        if not choices:raise ValueError('Source print pixel outside confirmed board-body triangles')
        output.append(min(choices,key=lambda item:item[0])[1])
    return np.asarray(output)


def build_complete_source_charts(data,model,records,cameras,scene,registry,templates):
    approved=read(model.parent/'framewise_8a28_first_shot_20261014'/'repair_20261014'/'scene.json')
    # The r1 model had relocated Ad25's shared coordinates without updating the
    # shared invariant. Restore the actual approved common descriptor verbatim;
    # the r2 opposite printed badge uses explicit source-conditioned contours.
    scene['shared_board_patterns']=deepcopy(approved['shared_board_patterns'])
    scene['r2_shared_descriptor_repair']=dict(
        source=str(model.parent/'framewise_8a28_first_shot_20261014'/'repair_20261014'/'scene.json'),
        restored_shared_pattern_contract=True,actual_Ad25_source_prints_are_independent_explicit_contours=True,
        original_shared_anchors_and_xyz_unchanged=True)
    # Independent named Ad2 glyphs retain the approved first-shot shape system.
    # A blurred current threshold component never replaces several letters.
    template2=deepcopy(templates)
    templates.setdefault('advertisements',{})['2']['contours']={
        **deepcopy(template2['contours']),
        **{k:v for k,v in templates['advertisements']['2']['contours'].items()
           if v.get('role')=='visible_blue_ribbon_material_boundary'}}
    registry['targets']={k:v for k,v in registry['targets'].items() if not k.startswith('Ad2.print.')}
    for key,contour in templates['advertisements']['2']['contours'].items():
        contour['advertisement_id']=2
        registry['targets']['Ad2.print.'+key]=dict(kind='ad_print',advertisement_id=2,contour_key=key,
            residual='point_to_named_finite_print_contour',template='ad2_material_template.json',
            contour_role=contour.get('role'),semantic_glyph=key in template2['contours'],
            geometry_status='approved_first_shot_named_glyph_source_chart' if key in template2['contours'] else 'current_source_blue_ribbon_initializer')
    # These boxes choose semantic source entities, not observation vertices.
    # Every retained contour below is read from the referenced current RGB.
    groups={
        7:(1249,[('ISU',(1044,365,1218,439),True),('INTERNATIONAL',(1220,372,1495,400),True),
            ('SKATING_UNION',(1220,400,1495,430),True)]),
        8:(1249,[('Wyndham_icon',(245,353,312,415),False),('WYNDHAM',(319,361,635,412),False),
            ('Ankara',(328,421,455,455),False)]),
        22:(1469,[('oz',(1470,552,1526,621),True),('can',(1528,555,1646,622),True),
            ('pismanyiye',(1487,620,1637,664),True),('1957',(1534,662,1585,686),True)]),
        24:(1518,[('red_shield',(144,507,224,661),False),('crescent',(148,526,183,581),True),
            ('star',(179,532,201,557),True),('skate',(164,578,215,644),False)]),
        25:(1469,[('badge_body',(58,483,102,589),False),('tower',(61,487,98,540),True),
            ('badge_star',(67,552,93,586),True),('TC',(99,492,123,513),False),
            ('ANKARA',(99,512,146,534),False),('BUYUKSEHIR',(99,532,170,555),False),
            ('BELEDIYESI',(99,551,166,576),False)]),
        1:(1660,[('qr_finder_left_top',(143,458,210,515),False),
            ('qr_finder_right_top',(279,455,334,513),False),
            ('qr_finder_left_bottom',(143,593,214,650),False),
            ('qr_centre_unique',(218,520,276,578),False)]),
    }
    by={int(r['index']):r for r in records}
    receipts=[]
    for n,(index,boxes) in groups.items():
        record=by[index];raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)>0
        references=[feature_world(scene,t,templates)for t in registry['targets'].values()
            if t.get('kind')=='ad_print' and int(t.get('advertisement_id',0))==n]
        if not references:continue
        reference_world=np.vstack(references)
        # Keep the named whole outer graphic/circle.  Replace fragmented glyph
        # charts in this isolated model snapshot, retaining old scene history.
        registry['targets']={k:v for k,v in registry['targets'].items()
            if not (k.startswith(f'Ad{n}.print.') and v.get('contour_key') not in
                ('Ad22:component0','outermost_visible_circular_print_stroke','outer_circle_print_stroke') and n!=1)}
        ad=next((a for a in scene['advertisements']if int(a['user_number'])==n),None)
        if ad is None:
            ad=dict(user_number=n,external_contours=[],print_pattern_only=True,geometry_accuracy_accepted=False)
            scene['advertisements'].append(ad)
        ad['external_contours']=[c for c in ad.get('external_contours',[])if not c['id'].startswith('r2_')]
        preview=raw.copy();added=0
        for label,box,bright in boxes:
            x0,y0,x1,y1=box;gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
            _,binary=cv2.threshold(gray,0,255,(cv2.THRESH_BINARY if bright else cv2.THRESH_BINARY_INV)+cv2.THRESH_OTSU)
            contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if hierarchy is None:continue
            inventory=[]
            for j,c in enumerate(contours):
                p=c[:,0].astype(float)+[x0,y0];extent=np.ptp(p,axis=0)
                if abs(cv2.contourArea(c))<4 or np.min(extent)<2:continue
                if ((p[:,0]<=x0+1)|(p[:,0]>=x1-2)|(p[:,1]<=y0+1)|(p[:,1]>=y1-2)).any():continue
                if mask[np.rint(p[:,1]).astype(int),np.rint(p[:,0]).astype(int)].any():continue
                compact=cv2.approxPolyDP(p.astype('float32')[:,None],1.2,True)[:,0].astype(float)
                if len(compact)<4:continue
                parent=int(hierarchy[0,j,3]);depth=0
                while parent>=0:depth+=1;parent=int(hierarchy[0,parent,3])
                inventory.append((float(p[:,0].min()),float(p[:,1].min()),bool(depth%2),compact))
            inventory.sort(key=lambda item:(item[0],item[1],item[2]))
            for j,(_,_,hole,p) in enumerate(inventory):
                key=f'r2_{label}_{j:02d}'+('_inner' if hole else '')
                try:xyz=world_on_body(p,cameras[index],scene,reference_world)
                except ValueError:continue
                xyz=np.vstack((xyz,xyz[:1]))
                ad['external_contours'].append(dict(id=key,world_xyz_scene_units=xyz.tolist(),closed=True,
                    source_native_uv=p.tolist(),source_frame_index=index,source_path=record['image_path'],
                    source_png_sha256=sha(record['image_path']),geometry_initialization_only=True,
                    source_material='confirmed_continuous_board_body',model_accuracy_accepted=False))
                registry['targets'][f'Ad{n}.print.{key}']=dict(kind='ad_print',advertisement_id=n,
                    contour_key=key,geometry_source='scene_existing_external_contour',
                    residual='point_to_named_finite_print_contour',source_bright=bright,semantic_glyph=True,
                    contour_closed=True,source_entity_group=label,geometry_status='whole_current_native_source_board_body_initializer')
                cv2.polylines(preview,[np.rint(p).astype('int32')],True,(0,0,255),1);added+=1
        cv2.imwrite(str(data/f'Ad{n}_whole_source_chart.png'),preview)
        descriptor=scene.setdefault('source_conditioned_board_prints',dict(
            schema='rinkspace.current-source-conditioned-board-print-contours.v1',patterns=[]))
        descriptor['patterns']=[p for p in descriptor['patterns']if int(p['advertisement_id'])!=n]
        if n not in (1,7):
            descriptor['patterns'].append(dict(advertisement_id=n,geometry_initialization_only=True,
                accuracy_accepted=False,external_contours=[dict(role=c['id'],xyz_scene_units=c['world_xyz_scene_units'])
                    for c in ad['external_contours']]))
        receipts.append(dict(advertisement=n,source_index=index,whole_named_components=added))
    write(model/'scene.json',scene);write(model/'feature_registry.json',registry);write(model/'ad2_material_template.json',templates)
    manifest=read(model/'model.json');manifest['scene']['sha256']=sha(model/'scene.json')
    manifest['feature_registry']['sha256']=sha(model/'feature_registry.json');manifest['material_templates']['sha256']=sha(model/'ad2_material_template.json')
    manifest['r2_whole_print_source_charts']=receipts;write(model/'model.json',manifest)
    write(data/'whole_print_chart_audit.json',receipts)


def run(data,model):
    if data.name!='r2' or '_r2_' not in model.name:raise ValueError('Refuse to overwrite a historical delivery/model')
    cv2.setNumThreads(2)
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json');templates=read(model/'ad2_material_template.json')
    records=read(data/'frame_manifest.json')['frames']
    archive=np.load(data/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    build_complete_source_charts(data,model,records,cameras,scene,registry,templates)
    for name in ('board_candidates','board_overlays'):(data/name).mkdir(parents=True,exist_ok=True)
    audits=[]
    for record in records:
        index=int(record['index']);camera=cameras[index]
        extract_frame(index,record,camera,scene,registry,templates,data=data,overlay=False)
        packet=read(data/f'board_candidates/{index:06d}.json')
        raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)>0
        mask=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
        edges,edge_audit=complete_board_edges(raw,mask,packet['features'],camera,scene,registry)
        borders=[f for f in packet['features'] if f['target_id'].startswith('Ad') and '.print.'not in f['target_id']]
        prints,print_audit=repair_prints(raw,mask,edges+borders,camera,scene,registry,templates)
        packet['features']=edges+borders+prints
        packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
        packet['status']='r2_current_native_whole_material_candidate_requires_visual_review'
        packet['diagnostics'].update(board_point_count=sum(len(s['uv'])for f in edges+borders for s in f['segments']),
            printed_point_count=sum(len(s['uv'])for f in prints for s in f['segments']),
            whole_native_board_body_edge_audit=edge_audit,whole_current_print_audit=print_audit)
        packet['source_correlated_groups']=[dict(id=f['target_id']+'_whole_current_source_curve',feature_id=f['id'],
            source='same_current_native_whole_visible_body_curve',representative_points_share_line_estimate=True)for f in edges]
        write(data/f'board_candidates/{index:06d}.json',packet)
        cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))))
        row=dict(index=index,points=packet['point_count'],body_edges=edge_audit,prints=print_audit)
        audits.append(row);print(json.dumps(dict(index=index,points=packet['point_count']),ensure_ascii=False),flush=True)
    write(data/'board_r2_audit.json',dict(schema='rinkspace.whole-current-native-board-r2.v1',frames=audits,
        historical_failed_delivery_unchanged=True,user_accepted=False))


def add_clear_holes_and_tin(data,model):
    """Narrow final-QA repair: actual Ad23 white holes and Ad1 T/I/N ink."""
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json');templates=read(model/'ad2_material_template.json')
    records={int(r['index']):r for r in read(data/'frame_manifest.json')['frames']}
    programme_id=next(iter(records.values()))['source']['programme_id']
    guide_path=model.parents[1]/'programmes'/programme_id/'source_datasets'/'shot2_r2_20261014'/'board_source_guides.json'
    guides=read(guide_path)
    archive=np.load(data/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    added=[]
    for n,index in ((23,int(guides['sport_holes']['source_frame'])),
        (1,int(guides['all_skating_tin']['source_frame'])),
        (24,int(guides['red_shield']['source_frame']))):
        record=records[index];raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)>0
        ad=next(a for a in scene['advertisements']if int(a['user_number'])==n)
        targets=[(k,v)for k,v in registry['targets'].items()if v.get('kind')=='ad_print'
            and int(v.get('advertisement_id',0))==n
            and not (n==24 and k.startswith('Ad24.print.r2_red_shield_'))]
        reference_world=np.vstack([feature_world(scene,v,templates)for k,v in targets])
        entries=[]
        if n==23:
            packet=read(data/f'board_candidates/{index:06d}.json')
            for f in packet['features']:
                if not f['target_id'].startswith(guides['sport_holes']['parent_target_prefix']):continue
                source=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in f['segments']])
                padding=int(guides['sport_holes']['source_roi_padding_px'])
                lo=np.maximum(np.floor(source.min(0)-padding).astype(int),[0,0]);hi=np.minimum(np.ceil(source.max(0)+padding).astype(int),[1920,1080])
                x0,y0=lo;x1,y1=hi
                gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
                _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                cs,h=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if h is None:continue
                holes=[]
                for j,c in enumerate(cs):
                    parent=int(h[0,j,3]);depth=0
                    while parent>=0:depth+=1;parent=int(h[0,parent,3])
                    if depth%2!=1:continue
                    p=c[:,0].astype(float)+lo;extent=np.ptp(p,axis=0)
                    if abs(cv2.contourArea(c))<15 or np.min(extent)<5:continue
                    if ((p[:,0]<=x0+1)|(p[:,0]>=x1-2)|(p[:,1]<=y0+1)|(p[:,1]>=y1-2)).any():continue
                    if mask[np.rint(p[:,1]).astype(int),np.rint(p[:,0]).astype(int)].any():continue
                    p=cv2.approxPolyDP(p.astype('float32')[:,None],1.4,True)[:,0].astype(float)
                    holes.append(p)
                for j,p in enumerate(holes):entries.append((f'r2_{f["target_id"].split(":")[-1]}_hole{j:02d}_inner',p,True,False))
        elif n==1:
            # The old 1670 chart was occluded here; 1660 shows these same three
            # large letters completely on confirmed board body.
            for glyph in guides['all_skating_tin']['glyph_rois']:
                label= glyph['glyph'];box=glyph['source_roi_xyxy']
                x0,y0,x1,y1=box;gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
                _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if not cs:continue
                c=max(cs,key=cv2.contourArea);p=c[:,0].astype(float)+[x0,y0]
                good=(p[:,0]>x0+1)&(p[:,0]<x1-2)&(p[:,1]>y0+1)&(p[:,1]<y1-2)
                # No ROI clipping boundary is registered as printed ink.
                if good.all():pieces=[(p,True)]
                else:
                    cut=int(np.flatnonzero(~good)[0]);order=np.r_[np.arange(cut+1,len(p)),np.arange(cut+1)]
                    pieces=[(p[order[ids]],False)for ids in runs(np.flatnonzero(good[order]))]
                for j,(piece,closed) in enumerate(pieces):
                    if np.linalg.norm(np.ptp(piece,axis=0))<10:continue
                    compact=cv2.approxPolyDP(piece.astype('float32')[:,None],1.5,closed)[:,0].astype(float)
                    if len(compact)>=3:entries.append((f'r2_ALL_SKATING_{label}_{j:02d}',compact,closed,True))
            for old in guides['all_skating_tin']['superseded_fragment_targets']:
                registry['targets'].pop(old,None)
        else:
            x0,y0,x1,y1=guides['red_shield']['source_roi_xyxy']
            b,g,r=raw[y0:y1,x0:x1].astype(float).transpose(2,0,1)
            signal=cv2.GaussianBlur((r-(b+g)/2).astype('float32'),(0,0),.8)
            binary=(signal>max(10.,float(np.percentile(signal,90))*.5)).astype('uint8')*255
            cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if cs:
                c=max(cs,key=cv2.contourArea);p=c[:,0].astype(float)+[x0,y0]
                p=cv2.approxPolyDP(p.astype('float32')[:,None],1.3,True)[:,0].astype(float)
                entries.append(('r2_red_shield_body',p,True,False))
            registry['targets']={k:v for k,v in registry['targets'].items()if not k.startswith('Ad24.print.r2_red_shield_')}
        ad['external_contours']=[c for c in ad.get('external_contours',[])if not (
            c['id'].startswith('r2_ALL_SKATING_') or ('r2_component' in c['id']and '_hole' in c['id'])
            or (n==24 and c['id'].startswith('r2_red_shield_')))]
        for key,p,closed,bright in entries:
            ad['external_contours']=[c for c in ad['external_contours']if c['id']!=key]
            xyz=world_on_body(p,cameras[index],scene,reference_world)
            if closed:xyz=np.vstack((xyz,xyz[:1]))
            ad['external_contours'].append(dict(id=key,world_xyz_scene_units=xyz.tolist(),closed=closed,
                source_native_uv=p.tolist(),source_frame_index=index,source_path=record['image_path'],
                source_png_sha256=sha(record['image_path']),geometry_initialization_only=True,
                source_material='confirmed_continuous_board_body',model_accuracy_accepted=False))
            tid=f'Ad{n}.print.{key}'
            registry['targets'][tid]=dict(kind='ad_print',advertisement_id=n,contour_key=key,
                geometry_source='scene_existing_external_contour',residual='point_to_named_finite_print_contour',
                source_bright=bright,semantic_glyph=True,contour_closed=closed,
                source_entity_group='clear_SPORTOTO_white_hole' if n==23 else 'ALL_SKATING_'+key.split('_')[-2],
                geometry_status='clear_current_native_source_body_print_initializer')
            if n==24:registry['targets'][tid].update(contour_role='visible_red_shield_material_boundary',
                source_entity_group='red_shield_actual_outer_material_boundary',semantic_glyph=False)
            if n==23:registry['targets'][tid]['parent_external_target']='Ad23.print.Ad10:'+key.split('_')[1]
            added.append(dict(target_id=tid,source_index=index,source_contour_points=len(p)))
        if n in (23,24):
            descriptor=scene['source_conditioned_board_prints']
            for item in descriptor['patterns']:
                if int(item['advertisement_id'])==n:item['external_contours']=[dict(role=c['id'],xyz_scene_units=c['world_xyz_scene_units'])for c in ad['external_contours']]
    write(model/'scene.json',scene);write(model/'feature_registry.json',registry)
    manifest=read(model/'model.json');manifest['scene']['sha256']=sha(model/'scene.json');manifest['feature_registry']['sha256']=sha(model/'feature_registry.json')
    manifest['final_clear_holes_and_tin']=added
    manifest['board_source_guides']=dict(path=str(guide_path),sha256=sha(guide_path));write(model/'model.json',manifest)
    active=guides['candidate_update_frames']
    counts=[]
    for index in active:
        record=records[index];camera=cameras[index];raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)>0
        mask=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
        packet=read(data/f'board_candidates/{index:06d}.json')
        source_edges=[f for f in packet['features']if '.print.'not in f['target_id']]
        prints,audit=repair_prints(raw,mask,source_edges,camera,scene,registry,templates)
        packet['features']=source_edges+prints;packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
        packet['diagnostics']['printed_point_count']=sum(len(s['uv'])for f in prints for s in f['segments'])
        packet['diagnostics']['whole_current_print_audit']=audit
        packet['unresolved_source_regions']=[dict(frame=index,**entry)
            for entry in guides['unresolved_source_regions'].get(str(index),[])]
        write(data/f'board_candidates/{index:06d}.json',packet)
        cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))))
        counts.append(dict(index=index,points=packet['point_count']));print(json.dumps(counts[-1]),flush=True)
    write(data/'final_clear_holes_and_tin_audit.json',dict(added=added,frames=counts,source_only=True,user_accepted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--holes-and-tin-only',action='store_true')
    args=parser.parse_args()
    if args.holes_and_tin_only:add_clear_holes_and_tin(args.data,args.model)
    else:run(args.data,args.model)
