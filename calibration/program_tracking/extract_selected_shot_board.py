"""Read one independently prepared selected shot from its actual native PNGs.

The reviewed material charts navigate named entities.  This entry point never
rebuilds a prior shot's initializers and never writes to its review dataset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from .runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import cv2
import numpy as np

from .extract_first_shot_board_source import read, write, sha, native_overlay, source_segments, profile_read, segment_records, path_stations, refined_ad_mapping, transform, print_inside_current_board_band
from .extract_shot2_board import extract_frame
from .repair_shot2_board import complete_board_edges, repair_prints, point_segment, world_on_body, runs
from .framewise_model import feature_world
from .projection import project_fast


def verify_current_source_binding(record):
    """Verify the exact native clock, pixels and current mask before reading."""
    source=record['source'];reference=record.get('source_binding')
    if not reference:raise ValueError('Selected native source has no exact source-binding reference')
    path=Path(reference['path'])
    if sha(path)!=reference['sha256']:raise ValueError('Current exact source-binding file changed')
    binding=read(path)
    for key in ('native_pts','time_base','programme_frame_index_1based','source_video_sha256','png_sha256','decoded_BGR_sha256'):
        if binding.get(key)!=source.get(key):raise ValueError(f'Exact native board source-binding {key} mismatch')
    mask_path=Path(record['mask_path']);mask_sha=sha(mask_path)
    if mask_sha!=source.get('mask_sha256') or mask_sha!=binding.get('mask_sha256'):
        raise ValueError('Exact native source-binding current mask hash mismatch')
    if Path(binding['mask_path']).resolve()!=mask_path.resolve():
        raise ValueError('Exact native source-binding current mask path mismatch')
    return deepcopy(reference)


def roi_material_contours(raw,mask,box,*,bright=True,hole=False,epsilon=2.5,signal=None,all_components=False,minimum_component_area=None,minimum_contour_span=7):
    """Read a named source entity, removing every ROI/crop/person boundary."""
    lo=np.maximum(np.floor(box[:2]).astype(int),[0,0]);hi=np.minimum(np.ceil(box[2:]).astype(int),[1920,1080])
    if min(hi-lo)<5:return []
    x0,y0=lo;x1,y1=hi
    gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    if signal in ('cyan','blue','pink','chroma'):
        b,g,r=raw[y0:y1,x0:x1].astype(float).transpose(2,0,1)
        contrast=np.ptp(raw[y0:y1,x0:x1].astype(float),axis=2) if signal=='chroma' else (b+g)*.5-r if signal=='cyan' else b-r if signal=='blue' else r-(b+g)*.5
        values=cv2.GaussianBlur(np.clip(contrast,0,255).astype('uint8'),(0,0),1.)
        _,binary=cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        otsu,binary=cv2.threshold(gray,0,255,(cv2.THRESH_BINARY if bright else cv2.THRESH_BINARY_INV)+cv2.THRESH_OTSU)
    binaries=[binary]
    if hole and bright and signal is None:
        # The glyph's tiny dark void has much less area than the white stroke.
        # A whole-ROI Otsu level can fill a visible low-contrast void. Preserve
        # only an actually enclosed current dark component at local levels.
        binaries.extend((gray>level).astype('uint8')*255 for level in
            (min(190.,otsu+20),min(190.,otsu+35),min(190.,otsu+45)))
    choices=[];center=(lo+hi)/2
    for binary in binaries:
        cs,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:continue
        for j,c in enumerate(cs):
            p=c[:,0].astype(float)+lo
            father=int(hierarchy[0,j,3]);depth=0
            while father>=0:depth+=1;father=int(hierarchy[0,father,3])
            minimum_area=minimum_component_area if minimum_component_area is not None else (6 if hole else 20)
            if bool(depth%2)!=hole or abs(cv2.contourArea(c))<minimum_area or np.min(np.ptp(p,axis=0))<(1 if all_components else 2 if hole else 3):continue
            distance=np.linalg.norm((p.min(0)+p.max(0))/2-center)
            choices.append((abs(cv2.contourArea(c))-distance*5,p))
    if not choices:return []
    output=[]
    selected=sorted(choices,key=lambda item:tuple(item[1].min(0))) if all_components else [max(choices,key=lambda item:item[0])]
    for _,p in selected:
        ip=np.rint(p).astype(int)
        valid=(ip[:,0]>x0+1)&(ip[:,0]<x1-2)&(ip[:,1]>y0+1)&(ip[:,1]<y1-2)
        valid&=~mask[ip[:,1],ip[:,0]]
        if valid.all():pieces=[(p,True)]
        else:
            cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(p)),np.arange(cut+1)]
            pieces=[(p[order[ids]],False)for ids in runs(np.flatnonzero(valid[order]))]
        for piece,closed in pieces:
            if np.linalg.norm(np.ptp(piece,axis=0))<minimum_contour_span:continue
            compact=cv2.approxPolyDP(piece.astype('float32')[:,None],epsilon,closed)[:,0].astype(float)
            if len(compact)>=(3 if closed else 2):output.append((compact,closed,piece))
    return output


def named_roi_options(item):
    return dict(bright=item.get('bright',True),hole=item.get('hole',False),signal=item.get('signal'),
        epsilon=item.get('source_contour_tolerance_px',2.5),all_components=item.get('read_all_material_components',False),
        minimum_component_area=item.get('minimum_component_area'),minimum_contour_span=item.get('minimum_contour_span',7))


def finite_polyline_distance(points,curve,closed=False):
    """Distance to the actual finite navigation path, not its sparse anchors."""
    p=np.asarray(points,float);c=np.asarray(curve,float)
    if closed and len(c)>1 and np.linalg.norm(c[0]-c[-1])>1e-8:c=np.vstack((c,c[0]))
    if len(c)<2:return np.linalg.norm(p[:,None]-c[None],axis=2).min(1)
    a=c[:-1];v=np.diff(c,axis=0);vv=np.maximum((v*v).sum(1),1e-10)
    t=np.clip(((p[:,None]-a[None])*v[None]).sum(2)/vv[None],0,1)
    return np.linalg.norm(p[:,None]-(a[None]+t[:,:,None]*v[None]),axis=2).min(1)


def verify_ordered_contour_pixels(evidence,closed=False):
    """A CHAIN_APPROX_NONE native contour may not bridge a rejected gap."""
    p=np.asarray(evidence,float)
    delta=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if closed and len(p)>1:delta=np.r_[delta,np.linalg.norm(p[0]-p[-1])]
    if len(delta) and delta.max()>np.sqrt(2)+1e-6:
        raise ValueError('Ordered current native pixel contour bridged a physical break or ROI crop gap')


def initialize_named_current_entities(data,model,records,cameras,scene,registry,guides):
    """Add source-conditioned print identity charts on actual board support."""
    by={int(r['index']):r for r in records};receipts=[]
    for item in guides.get('named_current_source_entities',[]):
        tid=item['target_id'];index=item['source_frame'];adnum=item['advertisement_number']
        record=by[index]
        verify_current_source_binding(record)
        if sha(record['image_path'])!=record['source']['png_sha256'] or sha(record['mask_path'])!=record['source']['mask_sha256']:
            raise ValueError('Current initializer source PNG or mask no longer matches the frozen manifest')
        raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)>0
        mask=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8))>0
        source=roi_material_contours(raw,mask,item['source_roi_xyxy'],**named_roi_options(item))
        if not source:continue
        ad=next(a for a in scene['advertisements']if int(a['user_number'])==adnum)
        if ad.get('corners_xyz_scene_units'):
            reference=np.asarray(list(ad['corners_xyz_scene_units'].values()))
        else:
            existing=[]
            for target in registry['targets'].values():
                if target.get('kind')!='ad_print' or int(target.get('advertisement_id',0))!=adnum:continue
                try:existing.append(feature_world(scene,target,read(model/'ad2_material_template.json')))
                except (KeyError,ValueError):continue
            if not existing:
                receipts.append(dict(target_id=tid,source_index=index,initializer_skipped=True,
                    reason='No confirmed same-ad board-body reference; source identity cannot create an unknown-depth fit entity'))
                continue
            reference=np.vstack(existing)
        contours=[]
        for k,(uv,closed,evidence) in enumerate(source):
            try:xyz=world_on_body(uv,cameras[index],scene,reference)
            except ValueError:continue
            if closed:xyz=np.vstack((xyz,xyz[:1]))
            contours.append(dict(id=tid.split('.print.')[1]+f'__part{k}',world_xyz_scene_units=xyz.tolist(),closed=closed))
        if not contours:continue
        if item.get('named_square_material_profile'):
            native=np.vstack([p for p,_,_ in source])
            item['source_material_box_xyxy']=np.r_[native.min(0),native.max(0)].tolist()
        # The model initializer contains actual material edges only. Its search
        # rectangle is separate metadata and cannot enter source observations.
        prefix=tid.split('.print.')[1]
        item['registered_targets']=[]
        inherited_geometry=bool(item.get('reuse_existing_target_geometry') and tid in registry['targets'])
        if inherited_geometry:
            item['registered_targets']=[tid]
        else:
            registry['targets']={k:v for k,v in registry['targets'].items()if k!=tid and not k.startswith(tid+'__part')}
            ad['external_contours']=[c for c in ad.get('external_contours',[])if c['id']!=prefix and not c['id'].startswith(prefix+'__part')]
        for k,contour in enumerate([] if inherited_geometry else contours):
            path=prefix+(f'__part{k}' if len(contours)>1 else '')
            target=f'Ad{adnum}.print.{path}'
            ad['external_contours'].append(dict(id=path,world_xyz_scene_units=contour['world_xyz_scene_units'],closed=contour['closed'],
                source_frame_index=index,source_path=record['image_path'],source_png_sha256=sha(record['image_path']),
                geometry_initialization_only=True,source_material='confirmed_continuous_board_body',model_accuracy_accepted=False))
            registry['targets'][target]=dict(kind='ad_print',advertisement_id=adnum,contour_key=path,
                geometry_source='scene_existing_external_contour',residual='point_to_named_finite_print_contour',
                contour_role='visible_printed_hole_boundary' if item.get('hole')else 'visible_printed_external_boundary',
                semantic_glyph=True,source_bright=item.get('bright',True),
                geometry_status='current_native_source_board_body_initializer',contour_closed=contour['closed'])
            item['registered_targets'].append(target)
        box=np.asarray(item['source_roi_xyxy']);quad=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
        try:item['search_roi_world_xyz']=world_on_body(quad,cameras[index],scene,reference).tolist()
        except ValueError:item.pop('search_roi_world_xyz',None)
        receipts.append(dict(target_id=tid,source_index=index,source_parts=len(contours),native_points=sum(len(p) for p,_,_ in source)))
    write(model/'scene.json',scene);write(model/'feature_registry.json',registry)
    descriptor=read(model/'model.json');descriptor['scene']['sha256']=sha(model/'scene.json');descriptor['feature_registry']['sha256']=sha(model/'feature_registry.json')
    descriptor['new_current_source_print_initializers']=receipts
    descriptor['new_print_initializers_accuracy_accepted']=False
    write(model/'model.json',descriptor)
    return receipts


def current_named_entities(raw,mask,camera,scene,registry,templates,guides,source_features,current_frame_index=None,current_source=None):
    features=[]
    for item in guides.get('named_current_source_entities',[]):
        if item.get('selected_source_indices') and current_frame_index not in item['selected_source_indices']:continue
        tids=item.get('registered_targets',[])
        if not tids:continue
        override=item.get('current_source_search_overrides',{}).get(str(current_frame_index))
        if override:
            if not current_source or any(override[k]!=current_source[k]for k in ('native_pts','png_sha256')):
                raise ValueError('Current explicit named search ROI source clock/hash does not match')
            box=np.asarray(override['source_roi_xyxy'],float)
        elif item['source_frame']==current_frame_index:
            # The same explicitly inspected native frame needs no inverse
            # model ROI. A crop wall may have no complete finite world quad.
            box=np.asarray(item['source_roi_xyxy'],float)
        else:
            if not item.get('search_roi_world_xyz'):continue
            quad,depth=project_fast(camera,np.asarray(item['search_roi_world_xyz']))
            if (depth<=0).any() or not np.isfinite(quad).all():continue
            box=np.r_[quad.min(0),quad.max(0)]
        material_box=None
        if item.get('source_reference_roi_xyxy'):
            reference=np.asarray(item['source_reference_roi_xyxy'],float)
            source_box=np.asarray(item['source_roi_xyxy'],float)
            paper=next((f for f in source_features if f['id']==item.get('current_reference_target')),None)
            if paper is not None:
                current_uv=np.vstack([s.get('source_native_readout_uv',s['uv']) for s in paper['segments']])
                after=np.r_[current_uv.min(0),current_uv.max(0)]
                before_span=reference[2:]-reference[:2];after_span=after[2:]-after[:2]
                box=np.r_[after[:2]+(source_box[:2]-reference[:2])/before_span*after_span,
                    after[:2]+(source_box[2:]-reference[:2])/before_span*after_span]
                if item.get('source_material_box_xyxy'):
                    sb=np.asarray(item['source_material_box_xyxy'],float)
                    material_box=np.r_[after[:2]+(sb[:2]-reference[:2])/before_span*after_span,
                        after[:2]+(sb[2:]-reference[:2])/before_span*after_span]
        if item.get('whole_source_identity_requires_unoccluded_ROI'):
            lo=np.maximum(np.floor(box[:2]).astype(int),[0,0]);hi=np.minimum(np.ceil(box[2:]).astype(int),[1920,1080])
            if min(hi-lo)<5 or mask[lo[1]:hi[1],lo[0]:hi[0]].any():continue
        if item.get('named_square_material_profile') and material_box is not None:
            x0,y0,x1,y1=material_box
            paths=[([[x0,y0],[x1,y0]],[0,1]if item.get('hole')else[0,-1]),
                ([[x0,y1],[x1,y1]],[0,-1]if item.get('hole')else[0,1]),
                ([[x0,y0],[x0,y1]],[1,0]if item.get('hole')else[-1,0]),
                ([[x1,y0],[x1,y1]],[-1,0]if item.get('hole')else[1,0])]
            tid=tids[0];segments=[]
            for path,direction in paths:
                rows,_=profile_read(raw,mask,path_stations(path,3),[1/3]*3,5,
                    maximum_blur=14,minimum_amplitude=6,navigation_tolerance=4,gradient_direction=direction)
                segment=segment_records(rows,spacing=100,straight=True,target_id=tid,minimum_span=5)
                for s in segment:
                    s['point_bindings']=[dict(kind='print_contour_sample',physical_target=tid,required_residual='point_to_named_finite_print_contour',measurement_origin='current_native_named_finder_material_interface') for _ in s['uv']]
                    s.update(outer_and_inner_finder_interfaces_not_mixed=True,source_roi_wall_removed=True)
                segments.extend(segment)
            if segments:
                features.append(dict(id=tid,target_id=tid,label_zh=item.get('label_zh',tid),physical_surface='board',review_layer='ad2_print',allowed_for_fit=False,
                    current_source_only_readout=True,model_or_template_generated_observation=False,segments=segments,
                    world_correspondence_status='same_named_QR_finder_physical_interface_current_source'))
            continue
        current=roi_material_contours(raw,mask,box,**named_roi_options(item))
        if not current:continue
        groups={tid:[] for tid in tids}
        seeds={tid:project_fast(camera,feature_world(scene,registry['targets'][tid],templates))[0] for tid in tids}
        identity_limit=(override or {}).get('maximum_identity_boundary_distance_px',item.get('maximum_identity_boundary_distance_px'))
        if identity_limit:
            kept=[];limit=float(identity_limit)
            for p,closed,evidence in current:
                evidence=np.asarray(evidence,float)
                valid=np.min(np.stack([finite_polyline_distance(evidence,seed,
                    closed=registry['targets'][tid].get('contour_closed',False))for tid,seed in seeds.items()]),axis=0)<limit
                if valid.all():kept.append((p,closed,evidence));continue
                if closed:
                    cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(evidence)),np.arange(cut+1)]
                else:
                    # An open source path has two physical ends. Wrapping its
                    # last and first retained fragments fabricates a connector.
                    order=np.arange(len(evidence))
                for ids in runs(np.flatnonzero(valid[order])):
                    part=evidence[order[ids]]
                    if len(part)<4 or np.linalg.norm(np.ptp(part,axis=0))<7:continue
                    compact=cv2.approxPolyDP(part.astype('float32')[:,None],2.5,False)[:,0].astype(float)
                    if len(compact)>=2:kept.append((compact,False,part))
            current=kept
        for p,closed,evidence in current:
            verify_ordered_contour_pixels(evidence,closed)
            tid=min(tids,key=lambda t:float(np.median(finite_polyline_distance(p,seeds[t],
                closed=registry['targets'][t].get('contour_closed',False)))))
            segment=point_segment(p,tid,closed=closed,evidence=evidence)
            segment.update(source_roi_wall_removed=True,source_named_entity_verified_by_current_shape=True,
                source_native_trace_topology='ordered_unit_pixel_contour',raw_trace_neighbors_contiguous=True,
                draw_current_native_material_readout=True)
            if item.get('sigma_px'):segment['localization_sigma_px']=[float(item['sigma_px'])]*len(p)
            groups[tid].append(segment)
        for tid,segments in groups.items():
            if not segments:continue
            features.append(dict(id=tid,target_id=tid,label_zh=item.get('label_zh',tid),physical_surface='board',review_layer='ad2_print',
                allowed_for_fit=False,current_source_only_readout=True,model_or_template_generated_observation=False,
                segments=segments,world_correspondence_status='same_named_static_print_material_current_ROI_read'))
    return features


def sparse_print_representatives(features,tolerance=3.5):
    """Keep each actual entity and hole, retaining its full native readout.

    Vertices remain real current boundary pixels.  The approximation only
    reduces redundant contour vertices; it does not alter their coordinates.
    Shared side-line estimates and typed advertisement corners stay intact.
    """
    receipts=[]
    for feature in features:
        if '.print.'not in feature['id']:continue
        for segment in feature['segments']:
            if segment.get('representative_points_share_line_estimate'):continue
            source=np.asarray(segment.get('source_native_readout_uv',[]),float)
            old=np.asarray(segment['uv'],float)
            if len(source)<4 or len(old)<4:continue
            points=cv2.approxPolyDP(source.astype('float32')[:,None],tolerance,segment.get('closed',False))[:,0].astype(float)
            if len(points)<3 or len(points)>=len(old):continue
            # Match precision/binding metadata to the nearest same-entity
            # original representative; every binding names the same contour.
            nearest=np.argmin(np.linalg.norm(points[:,None]-old[None],axis=2),axis=1)
            for key in ('point_bindings','localization_sigma_px','transition_iqr_px','sigma_px','source_evidence','point_ids'):
                if isinstance(segment.get(key),list)and len(segment[key])==len(old):segment[key]=[deepcopy(segment[key][j])for j in nearest]
            segment.update(uv=points.tolist(),vertex_count=len(points),source_native_contour_retained_complete=True,
                source_contour_representation_tolerance_px=tolerance,vertices_are_current_native_boundary_pixels=True)
            receipts.append(dict(target_id=feature['target_id'],before=len(old),after=len(points),tolerance_px=tolerance))
    return receipts


def scoped_current_print_reads(raw,mask,board_features,camera,scene,registry,templates):
    """Constrain template search to the current board and named finite panel.

    This is an exclusion guard, never proof of advertisement identity. Each
    resulting named entity still requires native shape and temporal review.
    A displaced old chart cannot acquire spectators or a neighbouring panel.
    """
    height,width=raw.shape[:2];xx=np.arange(width);bounds={}
    for role in ('top','contact'):
        support=[np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            for f in board_features if f['target_id'].startswith('board.'+role+'.') for s in f['segments']
            if len(s.get('source_native_readout_uv',s['uv']))]
        if not support:raise ValueError('No actual current source board material band support: '+role)
        native=np.vstack(support)
        order=np.argsort(native[:,0]);native=native[order]
        axis,inv=np.unique(native[:,0],return_inverse=True)
        yy=np.array([np.median(native[inv==j,1])for j in range(len(axis))])
        bounds[role]=np.interp(xx,axis,yy)
    rows=np.arange(height)[:,None]
    board_excluded=(rows<=bounds['top'][None]+8)|(rows>=bounds['contact'][None]-6)
    prints=[];audits=[];rejected=[]
    numbers=sorted({int(t['advertisement_id'])for t in registry['targets'].values()if t.get('kind')=='ad_print'})
    for n in numbers:
        scoped=mask|board_excluded;quad=None;scope_kind='source_board_body_guard_only_identity_still_requires_native_review'
        corners=[f'Ad{n}.{k}'for k in ('left_top','right_top','right_bottom','left_bottom')]
        if all(k in registry['targets']for k in corners):
            mapping=refined_ad_mapping(board_features,camera,scene,registry,n)
            if mapping is not None:
                quad=transform([[0,0],[1,0],[1,1],[0,1]],mapping)
                scope_kind='current_source_finite_advertisement_sides_exclusion_guard'
            else:
                world=np.concatenate([feature_world(scene,registry['targets'][k])for k in corners])
                q,depth=project_fast(camera,world)
                if (depth>0).all()and np.isfinite(q).all():quad=q;scope_kind='same_named_finite_model_panel_navigation_exclusion_only'
        if quad is not None:
            domain=np.zeros((height,width),np.uint8)
            clipped=np.clip(quad,[-10000,-10000],[10000,10000])
            cv2.fillPoly(domain,[np.rint(clipped).astype('int32')],1)
            scoped=scoped|~(domain>0)
        sub=deepcopy(registry)
        sub['targets']={k:v for k,v in sub['targets'].items()if v.get('kind')!='ad_print' or
            (int(v['advertisement_id'])==n and not v.get('retired_source_read_navigation'))}
        current,audit=repair_prints(raw,scoped,board_features,camera,scene,sub,templates)
        for feature in current:
            uv=np.vstack([np.asarray(s.get('source_native_readout_uv',s['uv']),float)for s in feature['segments']])
            if not print_inside_current_board_band(board_features,uv).all():
                rejected.append(dict(target_id=feature['target_id'],reason='Whole named print rejected: current evidence crossed the actual source board material band',current_native_uv=uv.tolist()))
                continue
            feature['current_source_print_scope_guard']=scope_kind
            feature['scope_mask_is_not_identity_authority']=True
            prints.append(feature)
        audits.extend(audit)
    return prints,audits,rejected


def reread_explicit_current_edges(raw,mask,record,features,guides):
    """Recover a source-confirmed finite side where the old camera drifted."""
    result=list(features);receipts=[]
    for guide in guides.get('current_source_edge_guides',[]):
        if int(guide['source_frame'])!=int(record['index']):continue
        if guide['native_pts']!=record['source']['native_pts'] or guide['source_png_sha256']!=record['source']['png_sha256']:
            raise ValueError('Explicit current side guide source clock/hash differs')
        tid=guide['target_id']
        path=path_stations(np.asarray(guide['uv'],float),3)
        rows,_=profile_read(raw,mask,path,guide['weights_BGR'],guide.get('radius_px',12),
            maximum_blur=guide.get('maximum_blur_px',18),minimum_amplitude=guide.get('minimum_amplitude',8),
            navigation_tolerance=guide.get('navigation_tolerance_px',10),gradient_direction=guide['gradient_direction'],signal_mode='chroma')
        segments=segment_records(rows,spacing=160,straight=True,target_id=tid,minimum_span=3)
        if not segments:continue
        feature=dict(id=tid,target_id=tid,label_zh=tid,physical_surface='board',review_layer='ad2_border',world_curve=tid,
            segments=segments,allowed_for_fit=False,source_manual_visual_search_is_not_observation=True)
        result=[f for f in result if f['target_id']!=tid]+[feature]
        receipts.append(dict(target_id=tid,current_source_readouts=sum(len(s.get('source_native_readout_uv',[]))for s in segments),
            recovery_reason=guide.get('reason_zh'),actual_pixels_measured_in_this_frame=True))
    return result,receipts


def reread_visible_ad_side_ends(raw,mask,features):
    """Search source-visible side ends using the same ad's measured other sides.

    Historical camera sampling can stop a fixed distance short of a visible
    material corner. Current side intersections only navigate new profiles;
    returned endpoints still come from actual current material transitions.
    """
    by={f['target_id']:f for f in features};result=[];receipts=[]
    for feature in features:
        tid=feature['target_id']
        if not tid.startswith('Ad') or not tid.endswith('_vertical'):
            result.append(feature);continue
        ad,role=tid.split('.',1);side=role.removesuffix('_vertical')
        current=np.vstack([np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            for s in feature['segments']if len(s.get('source_native_readout_uv',s['uv']))])
        if len(current)<2:result.append(feature);continue
        line=cv2.fitLine(current.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
        if abs(line[1])<.2:result.append(feature);continue
        lo,hi=float(current[:,1].min()),float(current[:,1].max());expanded=False
        for level in ('top','bottom'):
            horizontal=by.get(ad+'.'+level+'_edge')
            if horizontal is None:continue
            endpoints=np.vstack([np.asarray(s['uv'])[[0,-1]]for s in horizontal['segments']])
            xline=line[2]+(endpoints[:,1]-line[3])*line[0]/line[1]
            j=int(np.argmin(abs(endpoints[:,0]-xline)));corner=endpoints[j]
            if abs(corner[0]-xline[j])>12:continue
            target_y=float(corner[1])
            gap=lo-target_y if level=='top'else target_y-hi
            if not 4<gap<100:continue
            if level=='top':lo=target_y-2
            else:hi=target_y+2
            expanded=True
        if not expanded:result.append(feature);continue
        radius=40
        ys=np.arange(lo,hi+.01,3);xs=line[2]+(ys-line[3])*line[0]/line[1]
        normal_inside=1 if side=='left'else -1
        samples=[]
        for sign in (-1,1):
            q=np.column_stack((xs+normal_inside*sign*32,ys));pix=np.rint(q).astype(int)
            good=(pix[:,0]>=0)&(pix[:,0]<raw.shape[1])&(pix[:,1]>=0)&(pix[:,1]<raw.shape[0])
            good[good]&=~mask[pix[good,1],pix[good,0]]
            if not good.any():samples.append(None);continue
            colours=cv2.remap(raw,q[good,0].astype('float32')[:,None],q[good,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float)
            samples.append(np.median(colours,axis=0))
        if any(p is None for p in samples):result.append(feature);continue
        delta=samples[1]-samples[0]
        if np.linalg.norm(delta)<15:result.append(feature);continue
        weights=delta/np.linalg.norm(delta)
        rows,_=profile_read(raw,mask,np.column_stack((xs,ys)),[1,-.5,-.5],radius,
            maximum_blur=42,minimum_amplitude=6,navigation_tolerance=24,
            gradient_direction=[1,0]if side=='left'else[-1,0],signal_mode='chroma',
            require_clear_end_plateaus=True,minimum_gradient=.6)
        # This pass adds missing source-visible ends. Previously measured
        # valid intervals must not disappear when a broader new query has
        # weaker contrast or crosses another printed feature inside the ad.
        covered=[(float(np.min(p[:,1]))-1.5,float(np.max(p[:,1]))+1.5)
            for s in feature['segments']for p in [np.asarray(s.get('source_native_readout_uv',s['uv']),float)]if len(p)]
        rows=[None if row is not None and any(a<=row[0][1]<=b for a,b in covered)else row for row in rows]
        segments=segment_records(rows,spacing=160,straight=True,target_id=tid,minimum_span=3)
        if not segments:result.append(feature);continue
        for segment in segments:
            segment.update(current_material_signal='actual_current_BGR_chroma_half_transition_with_clear_end_plateaus',
                current_source_outside_inside_plateau_BGR=np.asarray(samples).tolist(),source_visible_endpoint_addition=True)
            segment['localization_sigma_px']=[max(1.5,float(np.median(segment.get('source_native_transition_iqr_px',[3])))/1.349)]*len(segment['uv'])
        updated=deepcopy(feature);updated['segments']+=segments
        updated['visible_side_end_recovery']='Actual current profiles navigated by the same advertisement current measured sides; no endpoint pulled to a corner'
        result.append(updated)
        receipts.append(dict(target_id=tid,source_query_y_range=[lo,hi],actual_readout_points=sum(len(s.get('source_native_readout_uv',s['uv']))for s in segments),
            fixed_camera_endpoint_margin_not_preserved=True,corner_coordinates_not_used_as_observations=True))
    return result,receipts


def actual_strip_interface_exclusion(auxiliary,uv,*,identity,above):
    """Only local adjacent actual profile samples support this exclusion."""
    p=np.asarray(uv,float);excluded=np.zeros(len(p),bool)
    for f in auxiliary:
        if f['id']!='unbound.board.'+identity:continue
        for s in f['segments']:
            trace=np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            if len(trace)<2:continue
            if np.all(np.diff(trace[:,0])<0):trace=trace[::-1]
            if not np.all(np.diff(trace[:,0])>0):raise ValueError('Auxiliary material profile x-order is not monotone')
            ids=np.clip(np.searchsorted(trace[:,0],p[:,0],side='right')-1,0,len(trace)-2)
            a,b=trace[ids],trace[ids+1];dx=b[:,0]-a[:,0]
            local=(p[:,0]>=a[:,0])&(p[:,0]<=b[:,0])&(dx<=8)&(dx>.01)&(np.abs(b[:,1]-a[:,1])<=6)
            y=a[:,1]+(p[:,0]-a[:,0])/dx*(b[:,1]-a[:,1])
            excluded|=local&((p[:,1]<y-2)if above else(p[:,1]>y+2))
    return excluded


def reread_ad_edges_inside_actual_strips(raw,mask,features,auxiliary):
    """Reject cap/strip evidence before estimating a source ad edge again.

    Actual strip interfaces only exclude impossible ad material. They neither
    certify physical Ad identity nor generate advertisement observations.
    """
    result=[];rejected=[]
    for f in features:
        tid=f['target_id'];level='top'if tid.endswith('.top_edge')else'bottom'if tid.endswith('.bottom_edge')else None
        if level is None:result.append(f);continue
        kept=[]
        for j,s in enumerate(f['segments']):
            actual=np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            if not len(actual):kept.append(s);continue
            bad=actual_strip_interface_exclusion(auxiliary,actual,
                identity='blue_cap_lower_body_interface'if level=='top'else'yellow_strip_upper_body_interface',above=level=='top')
            if not bad.any():kept.append(s);continue
            rejected.append(dict(target_id=tid,segment_index=j,reason='Actual locally visible strip material interface excludes old cap/kickplate evidence; entire old joint segment discarded before current reread',
                excluded_source_native_uv=actual[bad].tolist(),old_source_native_uv=actual.tolist(),old_observations=s['uv']))
            support=actual[~bad]
            if len(support)<3 or np.ptp(support[:,0])<6:continue
            coefficients=np.polynomial.Polynomial.fit(support[:,0],support[:,1],min(2,len(support)-1))
            xs=np.arange(float(actual[:,0].min()),float(actual[:,0].max())+.01,3)
            path=np.column_stack((xs,coefficients(xs)))
            rows,_=profile_read(raw,mask,path,[1,-.5,-.5],6,maximum_blur=18,
                minimum_amplitude=6,navigation_tolerance=5,gradient_direction=[0,1]if level=='top'else[0,-1],signal_mode='chroma')
            rows=[None if row is not None and actual_strip_interface_exclusion(auxiliary,[row[0]],
                identity='blue_cap_lower_body_interface'if level=='top'else'yellow_strip_upper_body_interface',above=level=='top')[0]else row for row in rows]
            new=segment_records(rows,spacing=160,straight=False,smooth_curve=True,target_id=tid,minimum_span=6)
            for segment in new:segment['old_cap_or_strip_evidence_excluded_before_new_current_source_fit']=True
            kept.extend(new)
        if kept:
            updated=deepcopy(f);updated['segments']=kept;result.append(updated)
    return result,rejected


def label_current_observed_ad_corners(features):
    """Label an unchanged observed end only if the other real side reaches it."""
    by={f['target_id']:f for f in features}
    for f in features:
        tid=f['target_id'];level='top'if tid.endswith('.top_edge')else'bottom'if tid.endswith('.bottom_edge')else None
        if level is None:continue
        ad=tid.split('.')[0]
        for side in ('left','right'):
            corner=side+'_'+level
            if any(corner in s.get('corner_vertex_indices',{})for s in f['segments']):continue
            vertical=by.get(ad+'.'+side+'_vertical')
            if vertical is None:continue
            v=np.vstack([np.asarray(s['uv'])[[0,-1]]for s in vertical['segments']])
            choices=[(float(np.min(np.linalg.norm(v-np.asarray(s['uv'][j]),axis=1))),s,j)for s in f['segments']for j in (0,len(s['uv'])-1)]
            distance,s,j=min(choices,key=lambda row:row[0])
            if distance>3:continue
            s.setdefault('corner_vertex_indices',{})[corner]=j
            s['point_bindings'][j]=dict(kind='corner',physical_target=ad+'.'+corner,
                required_residual='point_to_corresponding_model_corner_2d',
                measurement_origin='unchanged_current_source_observed_endpoint_with_measured_adjacent_side_identity',
                no_coordinate_snapping_or_model_corner_projection=True)
    return features


def reread_unbound_current_materials(raw,mask,record,board_features,guides):
    """Source-visible static body traces with no invented 3D correspondence."""
    result=[]
    for guide in guides.get('unbound_current_trace_guides',[]):
        if int(guide['source_frame'])!=int(record['index']):continue
        if guide['native_pts']!=record['source']['native_pts'] or guide['source_png_sha256']!=record['source']['png_sha256']:
            raise ValueError('Unbound body trace guide clock/hash differs')
        if guide.get('read_narrow_dark_trace_both_sides'):
            segments=[]
            for path in guide.get('visible_paths',[guide['uv']]):
                stations=path_stations(path,2);offsets=np.arange(-8,8.1,.5)
                q=stations[:,None]+np.stack((np.broadcast_to(offsets,(len(stations),len(offsets))),np.zeros((len(stations),len(offsets)))),-1)
                gray=cv2.remap(raw,q[:,:,0].astype('float32'),q[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float).mean(2)
                gray=cv2.GaussianBlur(gray,(5,1),.5)
                baseline=np.stack([np.linspace(np.median(r[:4]),np.median(r[-4:]),len(offsets))for r in gray])
                dips=baseline-gray;side_rows=[[],[]]
                for j,dip in enumerate(dips):
                    candidates=np.flatnonzero(abs(offsets)<=1.5);k=int(candidates[np.argmax(dip[candidates])])
                    peak=float(dip[k]);background_delta=abs(float(np.median(gray[j,:4])-np.median(gray[j,-4:])))
                    left=np.flatnonzero(dip[:k]<peak*.5);right=np.flatnonzero(dip[k+1:]<peak*.5)
                    if peak<guide.get('minimum_dip_contrast',1.5) or background_delta>20 or not len(left) or not len(right):
                        side_rows[0].append(None);side_rows[1].append(None);continue
                    a=int(left[-1]);b=int(k+1+right[0]);level=peak*.5
                    lx=offsets[a]+.5*(level-dip[a])/max(dip[a+1]-dip[a],1e-8)
                    rx=offsets[b-1]+.5*(level-dip[b-1])/min(dip[b]-dip[b-1],-1e-8)
                    qi=np.rint(q[j,a:b+1]).astype(int)
                    if rx-lx>5 or mask[qi[:,1],qi[:,0]].any():
                        side_rows[0].append(None);side_rows[1].append(None);continue
                    for side,position in enumerate((lx,rx)):
                        side_rows[side].append((stations[j]+[position,0],rx-lx,peak))
                for side,rows in enumerate(side_rows):
                    current=segment_records(rows,spacing=65,straight=True,minimum_span=8)
                    for segment in current:segment.update(trace_material_interface='left_half_darkness' if side==0 else 'right_half_darkness',half_level_relative_to_actual_current_local_background=True)
                    segments.extend(current)
        else:
            rows,_=profile_read(raw,mask,path_stations(guide['uv'],2),guide.get('weights_BGR',[1/3]*3),3,
                maximum_blur=8,minimum_amplitude=3,navigation_tolerance=2)
            segments=segment_records(rows,spacing=65,straight=True,minimum_span=8)
        kept=[]
        for segment in segments:
            actual=np.asarray(segment.get('source_native_readout_uv',segment['uv']))
            if not print_inside_current_board_band(board_features,actual).all():continue
            segment.pop('point_bindings',None)
            segment.update(source_evidence_measured_in_this_frame=True,source_pixel_coordinates_only=True,
                known_depth_assumed=False,allowed_for_fit=False)
            kept.append(segment)
        if kept:
            result.append(dict(id=guide['id'],label_zh=guide['label_zh'],physical_surface='board',
                review_layer='unbound_materials',allowed_for_fit=False,user_confirmed=False,
                binding_status='static_board_material_current_pixel_readout_without_3D_correspondence',
                source_identity_evidence=guide['static_material_evidence'],boundary_definition=guide['boundary_definition'],
                segments=kept,point_count=sum(len(s['uv'])for s in kept)))
    for guide in guides.get('unbound_current_roi_guides',[]):
        if int(guide['source_frame'])!=int(record['index']):continue
        if guide['native_pts']!=record['source']['native_pts'] or guide['source_png_sha256']!=record['source']['png_sha256']:
            raise ValueError('Unbound native material ROI clock/hash differs')
        current=roi_material_contours(raw,mask,guide['source_roi_xyxy'],**named_roi_options(guide))
        supported=[np.asarray(s.get('source_native_readout_uv',s['uv']),float)for f in board_features
            if f['target_id'].startswith(guide.get('exclude_bound_target_prefix','__none__'))for s in f['segments']]
        pieces=[]
        for p,closed,evidence in current:
            if not print_inside_current_board_band(board_features,evidence).all():continue
            if any(np.mean(np.min(np.linalg.norm(evidence[:,None]-old[None],axis=2),axis=1)<2)>.95 for old in supported):continue
            verify_ordered_contour_pixels(evidence,closed)
            s=dict(uv=p.tolist(),closed=closed,vertex_count=len(p),source_native_readout_uv=np.asarray(evidence).tolist(),
                measurement_origin='actual_current_native_material_contour_without_3D_correspondence',
                pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True,
                source_native_trace_topology='ordered_unit_pixel_contour',raw_trace_neighbors_contiguous=True,
                draw_current_native_material_readout=True,source_roi_wall_removed=True,
                source_pixel_coordinates_only=True,known_depth_assumed=False,allowed_for_fit=False,
                localization_sigma_px=[guide.get('sigma_px',4)]*len(p))
            pieces.append(s)
        if pieces:
            result.append(dict(id=guide['id'],label_zh=guide['label_zh'],physical_surface='board',review_layer='unbound_materials',
                allowed_for_fit=False,user_confirmed=False,binding_status='confirmed_current_static_material_without_3D_geometry',
                source_identity_evidence=guide['static_material_evidence'],boundary_definition=guide['boundary_definition'],
                segments=pieces,point_count=sum(len(s['uv'])for s in pieces)))
    return result


def reread_auxiliary_board_interfaces(raw,mask,board_features,guides):
    """Read the other two actual strip interfaces without borrowing a world edge."""
    enabled=set(guides.get('read_auxiliary_board_interfaces',[]));output=[]
    specifications=[('blue_cap_lower_body_interface','top',list(range(6,46,4)),[1,-.5,-.5],[0,-1],
        '蓝色压顶下沿：当前材料接口，三维未绑定'),
        ('yellow_strip_upper_body_interface','contact',list(range(-6,-50,-4)),[-1,.5,.5],[0,1],
        '黄色带上沿：当前材料接口，三维未绑定')]
    for identity,role,offsets,weights,direction,label in specifications:
        if identity not in enabled:continue
        segments=[]
        for feature in board_features:
            if not feature['target_id'].startswith('board.'+role+'.'):continue
            for reference in feature['segments']:
                base=path_stations(np.asarray(reference['uv'],float),4)
                # The nearest strip interface gets its own short profile so a
                # stronger distant advertising edge cannot win that query.
                alternatives=[profile_read(raw,mask,base+[0,offset],weights,6,maximum_blur=14,
                    minimum_amplitude=6,navigation_tolerance=5,gradient_direction=direction)[0]for offset in offsets]
                ordered=[]
                for j,p in enumerate(base):
                    choices=[]
                    for rows in alternatives:
                        row=rows[j]
                        if row is None:continue
                        uv,width,strength=row;distance=float(uv[1]-p[1])
                        physical_side=(2<distance<48) if role=='top' else (-50<distance<-2)
                        if not physical_side:continue
                        cy=p[1]+np.arange(2,62,2) if role=='top' else p[1]-np.arange(2,72,2)
                        cq=np.column_stack((np.full(len(cy),uv[0]),cy));cxy=np.rint(cq).astype(int)
                        if (cxy<[0,0]).any() or (cxy>[1919,1079]).any():continue
                        rgb=cv2.remap(raw,cq[:,0].astype('float32')[:,None],cq[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float)
                        white=(np.ptp(rgb,axis=1)<35)&(np.mean(rgb,axis=1)>135)&~mask[cxy[:,1],cxy[:,0]]
                        if white.sum()<3:continue
                        white_level=float(np.median(rgb[white]@np.asarray(weights)))
                    # Verify the coloured strip itself between the mandatory
                    # own edge and this interface. A nearby blue ad, white gap
                    # or yellow ice reflection cannot supply this evidence.
                        ys=np.arange(p[1]+2,uv[1]-2,1) if role=='top' else np.arange(uv[1]+2,p[1]-2,1)
                        if len(ys)<2:continue
                        outside=uv[1]+(1 if role=='top'else -1)*np.array([3,5,7,9])
                        q=np.column_stack((np.full(len(ys)+4,uv[0]),np.r_[ys,outside]));xy=np.rint(q).astype(int)
                        if (xy<[0,0]).any() or (xy>[1919,1079]).any() or mask[xy[:,1],xy[:,0]].any():continue
                        colours=cv2.remap(raw,q[:,0].astype('float32')[:,None],q[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float)
                        # A sparse handful of interior samples can jump a white
                        # board gap and land in a similarly coloured advertisement.
                        # Verify the whole source strip interval, allowing only a
                        # short native highlight rather than a white material gap.
                        neutral=np.ptp(colours[:len(ys)],axis=1)<25
                        if len(neutral)>=3 and np.convolve(neutral.astype(int),np.ones(3,dtype=int),'valid').max()>=3:continue
                        if np.sum(np.ptp(colours[len(ys):],axis=1)<35)<3:continue
                        values=colours@np.asarray(weights)
                        values-=white_level
                        inside=float(np.median(values[:len(ys)]));after=float(np.median(values[len(ys):]))
                        if inside<8 or np.mean(values[:len(ys)]>5)<.7 or after>max(6,inside*.35):continue
                        choices.append(row)
                    ordered.append(max(choices,key=lambda row:float(row[2]/max(row[1],1)))if choices else None)
                current=segment_records(ordered,spacing=160,smooth_curve=True,minimum_span=14)
                for segment in current:
                    actual=np.asarray(segment.get('source_native_readout_uv',segment['uv']),float)
                    selected=np.asarray(segment['uv'],float)
                    # A source-only interface has no model edge. Retain real
                    # measured half-transition coordinates, including its two
                    # visible ends, rather than an extrapolated fitted endpoint.
                    ids=sorted({0,len(actual)-1,*np.argmin(np.linalg.norm(selected[:,None]-actual[None],axis=2),axis=1).tolist()})
                    segment.update(uv=actual[ids].tolist(),vertex_count=len(ids),
                        measurement_origin='actual_current_native_strip_material_half_transition',
                        pixels_measured_in_this_frame=True,representative_points_share_line_estimate=False,
                        source_curve_fit_used_only_to_select_spaced_real_source_observations=True)
                    segment.update(source_native_trace_topology='ordered_material_profile_samples',
                        material_interface_definition=identity,model_world_target_defined=False,
                        known_depth_assumed=False,allowed_for_fit=False,
                        source_material_signal_BGR_weights=weights,
                        actual_local_whiteboard_baseline_removed_from_strip_colour_signal=True,
                        native_strip_interior_checked_at_1px_spacing=True,
                        neutral_material_gap_of_3_consecutive_native_pixels_rejected=True,
                        mandatory_other_strip_edge_used_only_for_source_search_and_strip_identity=True,
                        printed_or_reflected_colour_separated_by_actual_strip_interior_support=True)
                    count=len(segment['uv']);iqr=segment.get('source_native_transition_iqr_px',segment.get('transition_iqr_px',[3]))
                    segment['localization_sigma_px']=[max(1,float(np.median(iqr))/1.349)]*count
                segments.extend(current)
        if segments:
            output.append(dict(id='unbound.board.'+identity,label_zh=label,physical_surface='board',
                review_layer='unbound_materials',static=True,allowed_for_fit=False,user_confirmed=False,
                model_binding_confirmed=False,binding_status='actual_static_board_material_interface_without_defined_world_target',
                boundary_definition='Actual strip material interface; neither a width midpoint nor a substitute for the mandatory blue upper/yellow lower edge',
                source_identity_evidence='Same current native material strip; measured normal half-transitions and native strip-interior colour support; no unknown depth used',
                segments=segments,point_count=sum(len(s['uv'])for s in segments)))
    return output


def reread_white_cap_materials(raw,mask,board_features,auxiliary,guides):
    """Actual discernible bounded white materials inside the source cap band."""
    if not guides.get('read_static_white_cap_materials'):return [],[]
    evidence=guides.get('static_white_cap_material_source_evidence',[])
    if not evidence:raise ValueError('White cap material opt-in needs native temporal identity evidence')
    h,w=raw.shape[:2];domain=np.zeros((h,w),bool);yy=np.arange(h)[:,None]
    uppers=[np.asarray(s['uv'],float)for f in board_features if f['target_id'].startswith('board.top.')for s in f['segments']]
    for f in auxiliary:
        if f['id']!='unbound.board.blue_cap_lower_body_interface':continue
        for s in f['segments']:
            lower=np.asarray(s.get('source_native_readout_uv',s['uv']),float)
            if len(lower)<2:continue
            if lower[0,0]>lower[-1,0]:lower=lower[::-1]
            for upper in uppers:
                if upper[0,0]>upper[-1,0]:upper=upper[::-1]
                a=max(0,int(np.ceil(max(lower[0,0],upper[0,0]))));b=min(w-1,int(np.floor(min(lower[-1,0],upper[-1,0]))))
                if b<a:continue
                xx=np.arange(a,b+1);ids=np.clip(np.searchsorted(lower[:,0],xx,side='right')-1,0,len(lower)-2)
                valid=(lower[ids+1,0]-lower[ids,0]<=8)&(np.abs(lower[ids+1,1]-lower[ids,1])<=6)
                bottom=np.interp(xx,lower[:,0],lower[:,1]);top=np.interp(xx,upper[:,0],upper[:,1])
                domain[:,xx]|=(yy>=top[None]+1)&(yy<=bottom[None]-1)&valid[None]
    domain&=~mask
    safe=cv2.erode(domain.astype('uint8'),np.ones((3,3),np.uint8))>0
    from scipy.ndimage import median_filter
    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(float)
    # A long cap highlight can be brighter than small real white marks. Remove
    # its current horizontal body background before seeking bounded marks.
    background=median_filter(gray,size=(1,51),mode='nearest');signal=gray-background;values=signal[safe]
    if len(values)<30:return [],[dict(reason='No sufficient current visible cap interior')]
    dark=float(np.percentile(values,45));bright=float(signal[domain].max())
    if bright-dark<25:return [],[dict(reason='No distinct current white cap material contrast',native_cap_local_gray_contrast_plateaus=[dark,bright])]
    binary=(domain&(signal>12)).astype('uint8')*255
    coarse,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);features=[];limits=[];cs=[]
    for candidate in coarse:
        x,y,cw,ch=cv2.boundingRect(candidate)
        if cw>30 or ch>20:
            limits.append(dict(native_bbox_xywh=[x,y,cw,ch],reason='Extended cap highlight/blur does not identify an independent bounded white-mark material'));continue
        component=np.zeros((ch,cw),np.uint8);cv2.fillPoly(component,[candidate-[x,y]],1)
        peak=float(signal[y:y+ch,x:x+cw][component>0].max())
        if peak<25:continue
        level=max(12.,peak*.5);x0=max(0,x-3);y0=max(0,y-3);x1=min(w,x+cw+3);y1=min(h,y+ch+3)
        local_binary=(domain[y0:y1,x0:x1]&(signal[y0:y1,x0:x1]>level)).astype('uint8')*255
        local_cs,_=cv2.findContours(local_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in local_cs:
            c=c+np.array([x0,y0],dtype='int32');cx,cy,ww,hh=cv2.boundingRect(c)
            point=tuple(map(float,c[0,0]))
            if cv2.pointPolygonTest(candidate,point,False)>=0:cs.append((c,level))
    cs=sorted(cs,key=lambda item:(cv2.boundingRect(item[0])[0],cv2.boundingRect(item[0])[1]))
    for c,level in cs:
        x,y,cw,ch=cv2.boundingRect(c);area=abs(cv2.contourArea(c));trace=c[:,0].astype(float);pix=trace.astype(int)
        if area<.5 or min(cw,ch)<2 or len(trace)<3:continue
        local=safe[max(0,y-3):min(h,y+ch+3),max(0,x-3):min(w,x+cw+3)]
        local_gray=gray[max(0,y-3):min(h,y+ch+3),max(0,x-3):min(w,x+cw+3)]
        if local.sum()<4 or float(signal[pix[:,1],pix[:,0]].max())<25:continue
        if cw>30 or ch>20:
            limits.append(dict(native_bbox_xywh=[x,y,cw,ch],reason='Bright cap region lacks an independently bounded stable native outline; strip/mask boundary, extended highlight or excessive smear excluded'));continue
        # A white mark can meet the real cap edge while its left/right source
        # interfaces remain visible. Retain any boundary pixel with a real
        # dark-cap neighbour in the current cap domain; an artificial domain
        # closure across uniform white has no such material-transition pair.
        valid=np.zeros(len(pix),bool)
        for shift in ([1,0],[-1,0],[0,1],[0,-1]):
            q=pix+shift;inside=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
            ids=np.flatnonzero(inside)
            valid[ids]|=domain[q[ids,1],q[ids,0]]&(signal[q[ids,1],q[ids,0]]<=level)
        pieces=[]
        if valid.all():pieces=[(trace,True)]
        elif valid.any():
            cut=int(np.flatnonzero(~valid)[0]);order=np.r_[np.arange(cut+1,len(trace)),np.arange(cut+1)]
            pieces=[(trace[order[ids]],False)for ids in runs(np.flatnonzero(valid[order]))]
        segments=[]
        for actual,closed in pieces:
            if len(actual)<(3 if closed else 2):continue
            p=cv2.approxPolyDP(actual.astype('float32')[:,None],.35 if len(actual)<=5 else .8,closed)[:,0].astype(float)
            if len(p)<(3 if closed else 2):continue
            verify_ordered_contour_pixels(actual,closed)
            segments.append(dict(uv=p.tolist(),closed=closed,vertex_count=len(p),source_native_readout_uv=actual.tolist(),
                source_native_trace_topology='ordered_unit_pixel_contour',raw_trace_neighbors_contiguous=True,
                measurement_origin='actual_current_native_white_cap_material_outer_pixel_boundary',pixels_measured_in_this_frame=True,
                source_evidence_measured_in_this_frame=True,localization_sigma_px=[1.5]*len(p),
                current_native_cap_white_local_background_gray_half_level=level,
                actual_cap_local_horizontal_background_removed=True,world_target_defined=False,allowed_for_fit=False,
                source_cap_domain_closure_removed=not closed,actual_source_material_visible_piece_only=True,
                draw_current_native_material_readout=True,native_material_display_is_not_extra_observations=True))
        if not segments:
            limits.append(dict(native_bbox_xywh=[x,y,cw,ch],reason='White cap material has no reliable native boundary pair after removing actual strip-domain/person clipping edges'));continue
        features.append(dict(id=f'unbound.board.white_cap_material.{len(features):03d}',label_zh='帽带内部静止白色材料外廓（三维未绑定）',
            physical_surface='board',review_layer='unbound_materials',static=True,allowed_for_fit=False,user_confirmed=False,
            model_binding_confirmed=False,binding_status='current_static_white_cap_material_without_defined_world_geometry',
            material_identity='bounded white cap material; hardware/printing classification and depth not asserted',
            source_identity_evidence=evidence,boundary_definition='Actual current outer material silhouette, never a centre/bolt/unknown XYZ',
            segments=segments,point_count=sum(len(s['uv'])for s in segments)))
    if not features and not limits:limits.append(dict(reason='Current white marks lack a stable native 2D boundary pair after weak/subpixel contrast and actual strip-domain edges are excluded'))
    return features,limits


def run(data: Path, model: Path, dataset: Path, *, body_only=False):
    if (dataset/'user_acceptance.json').exists() or any(
        read(p).get('manual_editor') for p in (dataset/'frames').glob('*.json')
    ):
        raise ValueError('Refuse to regenerate a dataset already under manual review')
    manifest=read(data/'frame_manifest.json')
    shot=int(manifest['shot_scope']['shot_index'])
    if shot <= 2:
        raise ValueError('This entry point handles newly prepared subsequent shots only')
    expected=f'shot{shot}'
    if not data.name.startswith(expected) or not model.name.startswith(expected) or not dataset.name.startswith(expected):
        raise ValueError('The new source/model/dataset must belong to the same separate shot')
    scene=read(model/'scene.json');registry=read(model/'feature_registry.json')
    templates=read(model/'ad2_material_template.json')
    source_guides=dataset/'board_source_guides.json'
    guides=read(source_guides) if source_guides.exists() else {}
    retired=set(guides.get('forbidden_print_targets',[]))|set(guides.get('superseded_print_targets',[]))
    for tid in retired:
        if tid not in registry['targets']:continue
        registry['targets'][tid].update(retired_source_read_navigation=True,
            retirement_reason='Current native semantic ownership rejection or replacement by separately named current physical contours; preserved as historical model only',
            retirement_source_shot=shot,automatic_source_read_allowed=False)
    records=manifest['frames']
    archive=np.load(data/'camera_seeds.npz',allow_pickle=False)
    cameras=dict(zip(archive['programme_indices'].tolist(),archive['camera25']))
    if set(cameras)!={int(r['index']) for r in records}:
        raise ValueError('Selected camera/source indices differ')
    for name in ('board_candidates','board_overlays'):
        (data/name).mkdir(parents=True,exist_ok=True)
    initializer_receipts=[] if body_only else initialize_named_current_entities(data,model,records,cameras,scene,registry,guides)
    if source_guides.exists():
        if initializer_receipts:write(source_guides,guides)
        descriptor=read(model/'model.json')
        if descriptor.get('board_source_guides',{}).get('path')!=str(source_guides):
            descriptor.setdefault('inherited_board_source_guide_context',descriptor.get('board_source_guides'))
        descriptor['board_source_guides']=dict(path=str(source_guides),sha256=sha(source_guides))
        write(model/'model.json',descriptor)
    audits=[]
    for record in records:
        index=int(record['index']);camera=cameras[index]
        source=record['source']
        if not manifest['shot_scope']['first_pts']<=source['native_pts']<manifest['shot_scope']['stop_pts_exclusive']:
            raise ValueError('Current source PTS falls outside this shot')
        if source['time_base']!='1/12800' or source.get('shot_id',shot)!=shot:
            raise ValueError('Current source clock or shot identity differs')
        raw=cv2.imread(record['image_path'])
        if raw is None or raw.shape!=(1080,1920,3) or sha(record['image_path'])!=source['png_sha256']:
            raise ValueError('Current native source is unavailable or changed')
        if hashlib.sha256(raw.tobytes()).hexdigest()!=source['decoded_BGR_sha256']:
            raise ValueError('Current decoded native source pixels differ from the exact source binding')
        if sha(record['mask_path'])!=source['mask_sha256']:
            raise ValueError('Current source mask changed after manifest snapshot; refresh its source binding and reread')
        source_binding=verify_current_source_binding(record)
        # Body construction is exactly the same in both stages. Printing is
        # read independently below, never by the historical extractor first.
        base_registry=deepcopy(registry)
        base_registry['targets']={k:v for k,v in registry['targets'].items()if v.get('kind')!='ad_print'}
        stage=data/'board_extract_work'
        (stage/'board_candidates').mkdir(parents=True,exist_ok=True)
        extract_frame(index,record,camera,scene,base_registry,templates,data=stage,overlay=False)
        packet=read(stage/f'board_candidates/{index:06d}.json')
        # Historical extractor exclusions are not the current user's scope.
        packet.pop('excluded_features_by_user',None)
        packet['source_binding']=deepcopy(source_binding)
        packet['source']['source_binding']=deepcopy(source_binding)
        packet['current_exact_source_binding_and_mask_verified']=True
        mask=cv2.imread(record['mask_path'],0)
        if mask is None:raise ValueError('Current person exclusion is unavailable')
        mask=cv2.dilate((mask>0).astype('uint8'),np.ones((3,3),np.uint8))>0
        mask[:120,:570]=True;mask[945:,1660:]=True
        edges,edge_audit=complete_board_edges(raw,mask,packet['features'],camera,scene,registry)
        borders=[f for f in packet['features'] if f['target_id'].startswith('Ad') and '.print.' not in f['target_id']]
        borders,explicit_side_audit=reread_explicit_current_edges(raw,mask,record,borders,guides)
        auxiliary=[]if body_only else reread_auxiliary_board_interfaces(raw,mask,edges,guides)
        borders,ad_strip_rejections=(borders,[])if body_only else reread_ad_edges_inside_actual_strips(raw,mask,borders,auxiliary)
        borders,side_end_audit=reread_visible_ad_side_ends(raw,mask,borders)
        borders=label_current_observed_ad_corners(borders)
        prints,print_audit,scope_rejections=([],[],[]) if body_only else scoped_current_print_reads(raw,mask,edges+borders,camera,scene,registry,templates)
        replacements=[] if body_only else current_named_entities(raw,mask,camera,scene,registry,templates,guides,prints,current_frame_index=index,current_source=source)
        new_targets={tid for item in guides.get('named_current_source_entities',[])for tid in item.get('registered_targets',[])}
        superseded=set(guides.get('superseded_print_targets',[]))
        prints=[f for f in prints if f['target_id']not in new_targets|superseded]+replacements
        forbidden=set(guides.get('forbidden_print_targets',[]))|set(guides.get('forbidden_print_targets_by_frame',{}).get(str(index),[]))
        explicitly_rejected=[f for f in prints if f['target_id']in forbidden]
        prints=[f for f in prints if f['target_id']not in forbidden]
        scope_rejections.extend(dict(target_id=f['target_id'],reason='Native source semantic review rejects this inherited chart as another material; no partial reuse or relabel',current_native_uv=[s.get('source_native_readout_uv',s['uv'])for s in f['segments']])for f in explicitly_rejected)
        sparse_receipts=sparse_print_representatives(prints)
        packet['features']=edges+borders+prints
        packet.update(point_count=sum(len(s['uv']) for f in packet['features'] for s in f['segments']),
            status='selected_current_native_body_only_print_readout_pending' if body_only else 'selected_current_native_material_candidate_requires_visual_review',
            current_source_only_readout=True,observation_coordinates_from_projection=False)
        packet['printed_material_reread_pending']=bool(body_only)
        packet['diagnostics'].update(
            board_point_count=sum(len(s['uv']) for f in edges+borders for s in f['segments']),
            printed_point_count=sum(len(s['uv']) for f in prints for s in f['segments']),
            whole_native_board_body_edge_audit=edge_audit,whole_current_print_audit=print_audit,
            source_print_representative_simplification=sparse_receipts,explicit_current_source_side_recovery=explicit_side_audit,
            source_visible_ad_side_end_recovery=side_end_audit)
        packet['source_material_identity_rejections']=scope_rejections
        packet['source_ad_edge_strip_ownership_rejections']=ad_strip_rejections
        packet['unbound_static_materials']=[] if body_only else reread_unbound_current_materials(raw,mask,record,packet['features'],guides)
        if not body_only:packet['unbound_static_materials']+=auxiliary
        cap_materials,cap_limits=([],[])if body_only else reread_white_cap_materials(raw,mask,edges,auxiliary,guides)
        packet['unbound_static_materials']+=cap_materials
        packet['current_white_cap_material_read_limits']=cap_limits
        packet['source_correlated_groups']=[dict(id=f['target_id']+'_whole_current_source_curve',
            feature_id=f['id'],source='same_current_native_whole_visible_body_curve',
            representative_points_share_line_estimate=True)for f in edges]
        for feature in prints:
            for j,segment in enumerate(feature['segments']):
                if not segment.get('representative_points_share_line_estimate'):continue
                gid=feature['target_id']+f'_current_material_side_{j}'
                segment['source_evidence_group_id']=gid
                packet['source_correlated_groups'].append(dict(id=gid,feature_id=feature['id'],segment_index=j,
                    source='same_current_native_named_finite_material_side',representative_points_share_line_estimate=True))
        packet['unresolved_source_regions']=deepcopy(guides.get('unresolved_source_regions',{}).get(str(index),[]))
        write(data/f'board_candidates/{index:06d}.json',packet)
        cv2.imwrite(str(data/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet)),packet=packet))
        row=dict(index=index,points=packet['point_count'],body_edges=edge_audit,prints=print_audit)
        audits.append(row)
        print(json.dumps(dict(index=index,points=packet['point_count']),ensure_ascii=False),flush=True)
    write(data/'board_source_audit.json',dict(schema='rinkspace.selected-current-native-board.v1',
        shot_index=shot,frames=audits,previous_shot_review_data_untouched=True,user_accepted=False,
        printed_material_reread_pending=bool(body_only),
        model_is_identity_navigation_only=True,source_guides=dict(path=str(source_guides),sha256=sha(source_guides)) if source_guides.exists() else None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--body-only',action='store_true',help='Read complete current board body and finite sides first; printed material remains explicitly pending')
    args=parser.parse_args();cv2.setNumThreads(2)
    run(args.data.resolve(),args.model.resolve(),args.dataset.resolve(),body_only=args.body_only)
