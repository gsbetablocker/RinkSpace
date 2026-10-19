"""Drop unresolved print fragments, retaining source pixels and baseline archive."""
from __future__ import annotations
from copy import deepcopy
import cv2
import numpy as np
from .repair_first_shot_board_patterns import DATA,MODEL,REPAIR,read,write,sha,current_mask_path,feature_world,project_fast
from .extract_first_shot_board_source import native_overlay,source_segments


def fragment_rejection(feature,segment,expected_source_extent=None):
    """Do not promote a tiny isolated transition to a named glyph contour."""
    if '.print.'not in feature['target_id']:return None
    if feature.get('physical_shape','').startswith('circle_')or feature['target_id']in(
            'Ad11.print.outermost_visible_circular_print_stroke','Ad13.print.outer_circle_print_stroke'):return None
    if segment.get('source_clear_component_identity_verified_by_current_shape'):
        return None
    p=np.asarray(segment['uv'],float)
    extent=np.ptp(p,axis=0);span=float(np.linalg.norm(extent))
    all_uv=np.asarray([uv for s in feature['segments']for uv in s['uv']],float)
    if len(all_uv)<4 and np.max(np.ptp(all_uv,axis=0))<80:
        return 'isolated_component_remnant_without_enough_shape_to_confirm_its_print_identity'
    width=np.asarray(segment.get('transition_iqr_px',[3.]*len(p)),float)
    closed=bool(segment.get('closed'))
    # A fully seen sharp small component can still identify its own outline.
    # An isolated two-point speck cannot establish which letter side it is.
    complete_sharp=closed and len(p)>=4 and np.min(extent)>=3 and np.median(width)<=2.3
    if span<18 and not complete_sharp:
        return 'short_isolated_print_fragment_has_insufficient_current_source_shape_to_identify_named_letter_side'
    if (expected_source_extent is not None and np.max(expected_source_extent)<65
            and not complete_sharp):
        # Horizontal motion blur can leave a sharp top/bottom transition even
        # when adjacent letters merge.  A small normal transition IQR alone
        # therefore cannot establish the identity of a partial small glyph.
        return 'partial_small_glyph_without_separately_identifiable_letter_outline'
    if 'inner'in feature['target_id'] and (np.min(extent)<5 or np.max(extent)<12):
        return 'tiny_partial_glyph_hole_not_separately_resolved_from_outer_ink'
    if closed and np.min(extent)<4 and np.median(width)>2.3:
        return 'blurred_component_narrower_than_distinct_sided_material_support'
    return None


def prune():
    destination=REPAIR/'board_candidates';destination.mkdir(exist_ok=True)
    manifest={r['index']:r for r in read(DATA/'frame_manifest.json')['frames']}
    registry=read(MODEL/'feature_registry.json');scene=read(MODEL/'scene.json');templates=read(MODEL/'ad2_material_template.json')
    worlds={k:feature_world(scene,d,templates)for k,d in registry['targets'].items()if d['kind']=='ad_print'}
    archive=np.load(DATA/'camera_seeds.npz');cameras=dict(zip(archive['programme_indices'],archive['camera25']))
    changes=[];counts=[]
    for index in range(201,1249):
        file=destination/f'{index:06d}.json'
        packet=read(file if file.is_file()else DATA/f'board_candidates/{index:06d}.json')
        old=deepcopy(packet);removed=[]
        for feature in packet['features']:
            extent=None
            if feature['target_id']in worlds:
                projected,depth=project_fast(cameras[index],worlds[feature['target_id']])
                if np.isfinite(projected).all()and(depth>0).all():extent=np.ptp(projected,axis=0)
            segments=[]
            for segment in feature['segments']:
                reason=fragment_rejection(feature,segment,extent)
                if reason:
                    removed.append(dict(target_id=feature['target_id'],reason=reason,uv=segment['uv'],points=len(segment['uv'])))
                else:segments.append(segment)
            feature['segments']=segments
        packet['features']=[f for f in packet['features']if f['segments']]
        packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
        packet['diagnostics']['board_point_count']=sum(len(s['uv'])for f in packet['features']if'.print.'not in f['id']for s in f['segments'])
        packet['diagnostics']['printed_point_count']=packet['point_count']-packet['diagnostics']['board_point_count']
        mask_path=current_mask_path(index,manifest[index]);packet['source'].update(mask_path=str(mask_path),mask_sha256=sha(mask_path))
        feature_ids={f['id']for f in packet['features']};packet['source_correlated_groups']=[g for g in packet.get('source_correlated_groups',[])if g.get('feature_id')in feature_ids]
        if removed:
            packet.setdefault('source_unknown_print_rejections',[]).extend(removed)
            changes.append(dict(index=index,removed_points=sum(r['points']for r in removed),removals=removed))
        packet.setdefault('repair_baseline_candidate_sha256',sha(DATA/f'board_candidates/{index:06d}.json'))
        packet['source_confirmed_static_materials_only']=True
        write(file,packet);counts.append(dict(index=index,points=packet['point_count']))
        if index in (201,300,400,465,500,517,600,700,775,800,830,834,850,852,870,900,1000,1100,1200,1248):
            raw=cv2.imread(manifest[index]['image_path']);cv2.imwrite(str(REPAIR/f'board_diagnostics/board_{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
    previous=read(REPAIR/'board_unknown_print_rejections.json')if(REPAIR/'board_unknown_print_rejections.json').is_file()else{}
    previous_rows=previous.get('rows',[])
    combined=previous_rows+changes
    write(REPAIR/'board_unknown_print_rejections.json',dict(schema='rinkspace.first-shot-source-unresolved-print-fragment-rejections.v1',
        original_delivery_unchanged=True,points_moved=False,unknown_depth_used_for_fit=False,rows=combined,frames=1048,
        removed_points=sum(r['removed_points']for r in combined),remaining_board_print_points=sum(r['points']for r in counts),
        maximum_remaining_board_print_points=max(r['points']for r in counts),counts=counts,
        limitation='This shape-support rejection does not certify every retained glyph pixel; source visual review remains required.'))
    print('Pruned unresolved print:',sum(r['removed_points']for r in changes),'points in',len(changes),'frames; max',max(r['points']for r in counts),flush=True)


if __name__=='__main__':prune()
