"""Read one explicitly selected native frame, keeping typed physical targets.

Programme-local search anchors and letter identities are navigation only.
Every edge/letter UV is reread from that frame; the four corners are source
line intersections. No neighboring frames, optimizer or acceptance is run.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from .extract_opening_frame_batch import read, write, sha, fit_axis_line, MODEL, ROOT
from .single_frame_source_review import (scan_curve, print_contours, straight_source_boundary,
    space_boundary_samples, remove_temporary_readouts, compact_review_points, transition)
from .render_single_frame_source_review import native_overlay, source_segments, review_html


from rinkspace.storage import AssetStore

def visible_intervals(mask, intercept, slope, lo, hi):
    xx=np.arange(lo,hi+1); yy=np.rint(intercept+slope*xx).astype(int)
    good=np.ones(len(xx),bool)
    for offset in (-2,-1,0,1,2):good &= ~mask[np.clip(yy+offset,0,1079),xx]
    indices=np.flatnonzero(good)
    return [[int(xx[r[0]]),int(xx[r[-1]])] for r in
            np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(r)>12]


def board_features(raw, mask, spec):
    features=[]; scans={}
    for item in spec['edge_searches']:
        segments=scan_curve(raw,mask,item)
        if not segments:raise ValueError('Missing required visible edge '+item['id'])
        # These are the same long straight physical materials as the approved
        # example. Tiny scan-to-scan texture changes need no extra solver rows.
        # Select distributed actual readouts, keeping every visibility break.
        if item['id']!='board_top':
            for segment in segments:
                native=np.asarray(segment['_native_uv'])
                widths=np.asarray(segment['_native_iqr'])
                axis=item['axis']
                extent=float(native[-1,axis]-native[0,axis])
                count=max(2,int(np.ceil(abs(extent)/spec['maximum_straight_spacing_px']))+1)
                stations=np.linspace(native[0,axis],native[-1,axis],count)
                keep=np.unique([int(np.argmin(abs(native[:,axis]-p)))for p in stations])
                segment.update(uv=native[keep].tolist(),vertex_count=len(keep),
                    transition_iqr_px=widths[keep].tolist(),localization_sigma_px=(widths[keep]/1.349).tolist(),
                    measurement_origin='distributed_current_native_material_readouts_on_straight_physical_edge',
                    texture_wiggles_not_extra_physical_features=True)
        scans[item['id']]=segments
        features.append(dict(id=item['id'],label_zh=item['label_zh'],
            physical_surface='board',target_id=item['target_id'],world_curve=item['target_id'],
            segments=segments,allowed_for_fit=False))
    by={f['id']:f for f in features}
    top=by['board_top'];evidence=deepcopy(top['segments'])
    straight_source_boundary(top,evidence,[[0,1919]])
    est=top['source_straight_line_estimate']
    intervals=visible_intervals(mask,est['y_intercept_px'],est['y_per_x'],0,1919)
    straight_source_boundary(top,evidence,intervals)
    lines={fid:fit_axis_line(scans[fid],1 if 'vertical' in fid else 0)
           for fid in scans if fid.startswith('Ad2.')}
    corners={}
    for side in ('left','right'):
        vertical=lines['Ad2.'+side+'_vertical']
        for level in ('top','bottom'):
            horizontal=lines['Ad2.'+level+'_edge']
            y=(horizontal[0]*vertical[1]+horizontal[1])/(1-horizontal[0]*vertical[0])
            point=np.array([vertical[0]*y+vertical[1],y])
            x,y0=np.rint(point).astype(int)
            if not (0<=x<1920 and 0<=y0<1080) or mask[y0,x]:
                raise ValueError('Required corner is not visible: '+side+'_'+level)
            corners[side+'_'+level]=point.tolist()
    # Corner points occur once, as endpoints of the top/bottom paths. Their
    # identities are separate from the sampled material edges.
    for level in ('top','bottom'):
        feature=by['Ad2.'+level+'_edge'];segments=feature['segments']
        segments[0]['uv'][0]=corners['left_'+level]
        segments[-1]['uv'][-1]=corners['right_'+level]
    space_boundary_samples(features,spec['maximum_straight_spacing_px'])
    for level in ('top','bottom'):
        segments=by['Ad2.'+level+'_edge']['segments']
        for side,segment,index in [('left',segments[0],0),('right',segments[-1],len(segments[-1]['uv'])-1)]:
            binding=segment['point_bindings'][index]
            binding.update(kind='corner',physical_target='Ad2.'+side+'_'+level,
                           required_residual='point_to_corresponding_model_corner_2d',
                           measurement_origin='current_source_edge_line_intersection')
            segment.setdefault('corner_vertex_indices',{})[side+'_'+level]=index
            segment['source_corner_line_equations']={k:v.tolist() for k,v in lines.items()}
    return features


def glyph_features(raw,mask,spec):
    glyphs=print_contours(raw,mask,spec['print_roi'],minimum_transition_iqr_px=spec['print_minimum_iqr_px'],
        require_magenta_white_pair=True,profile_radius_px=spec['print_profile_radius_px'],foreground_clearance_px=3)
    result=[];seen={}
    for glyph in glyphs:
        all_uv=np.concatenate([s['uv'] for s in glyph['segments']])
        x=float(np.median(all_uv[:,0]))
        matches=[r for r in spec['printed_letter_search_intervals'] if r['x_interval'][0]<=x<r['x_interval'][1]]
        if len(matches)!=1:continue
        name=matches[0]['contour_key']
        if glyph.get('source_contour_internal_hole'):name+='_inner'
        fid='Ad2.print.'+name
        seen[fid]=seen.get(fid,0)+1
        glyph.update(id=fid+'.fragment'+str(seen[fid]),label=name,target_id=fid,
            world_correspondence_status='named_independent_material_template',
            source_print_identity=dict(method='current_source_reading_of_ISU_SKATING_COM_with_labelled_search_intervals',
                search_x_interval=matches[0]['x_interval'],coordinates_generated_by_search_interval=False))
        result.append(glyph)
    # A short visible letter arc may fail the whole-contour proposal near the
    # person. Explicitly reviewed source search windows still reread native
    # material transitions; anchors themselves are never observations.
    for extra in spec.get('short_print_profile_searches',[]):
        uv=[];widths=[];evidence=[]
        for x,anchor_y in extra['search_anchors']:
            radius=extra['radius_px']; ys=np.arange(anchor_y-radius,anchor_y+radius+1)
            colors=raw[ys,x].astype(float)
            found=transition(colors[:,1],radius)
            if found is None:raise ValueError('Short print profile has no material transition')
            q,strength=found; y=float(ys[0]+q[1])
            if mask[int(round(y)),x]:raise ValueError('Short print point overlaps person')
            lo=int(np.floor(ys[0]+q[0]));hi=int(np.ceil(ys[0]+q[2]))
            if mask[lo:hi+1,x].any():raise ValueError('Short print interface overlaps person')
            uv.append([float(x),y]);widths.append(q[2]-q[0])
            evidence.append(dict(native_x=x,native_y_range=[int(ys[0]),int(ys[-1])],
                native_BGR_profile=colors.tolist(),quartile_indices=q,green_contrast=strength,
                source_visual_materials_verified=['white_Ad2_print','magenta_Ad2_background']))
        target=extra['target_id']
        result.append(dict(id=target+'.short_visible_arc',target_id=target,label=target,
            physical_surface='board',allowed_for_fit=False,
            segments=[dict(uv=uv,vertex_count=len(uv),closed=False,
                transition_iqr_px=widths,localization_sigma_px=(np.array(widths)/1.349).tolist(),
                protected_vertex_indices=list(range(len(uv))),source_evidence=evidence,
                measurement_origin='native_colour_half_transition_in_visually_reviewed_short_arc',
                pixels_measured_in_this_frame=True)]))
    return result


def make_binding(packet,packet_path,registry,*,apply_feedback_withdrawals=True):
    observations=[];count=0
    for feature in packet['features']:
        for si,segment in enumerate(feature['segments']):
            targets={}
            for vi,_ in enumerate(segment['uv']):
                target=(segment['point_bindings'][vi]['physical_target']
                        if segment.get('point_bindings') else feature['target_id'])
                if target not in registry['targets']:raise ValueError('Missing physical target '+target)
                targets.setdefault(target,[]).append(vi)
            for target,indices in targets.items():
                observations.append(dict(id=f"{feature['id']}/{si}/{target}",
                    source_feature_id=feature['id'],source_segment_index=si,source_vertex_indices=indices,
                    source_point_ids=[segment['point_ids'][v] if segment.get('point_ids') else
                                      f"{feature['id']}/segment/{si}/vertex/{v}" for v in indices],
                    target_id=target,point_correspondence=registry['targets'][target]['kind']=='ad_corner',
                    dependence_group=feature.get('source_evidence_group_id',
                        'board_top_shared_source_line' if feature['id']=='board_top' else target)))
                count+=len(indices)
    assert count==packet['point_count']
    result=dict(schema='rinkspace.framewise-source-model-binding.v1',model_id=registry['model_id'],
        source_packet=dict(path=str(packet_path),sha256=sha(packet_path)),
        source_pts=packet['source']['native_pts'],source_time_base=packet['source']['time_base'],
        frame_count=1,source_user_accepted=False,point_count=count,observations=observations,
        status='all_points_have_typed_model_entries',optimization_performed=False)
    if apply_feedback_withdrawals:
        from rinkspace.feedback_withdrawal import enforce_packet_withdrawals
        checked=enforce_packet_withdrawals(packet,result,
            project=Path(__file__).resolve().parents[2],packet_path=packet_path,gate='model_binding')
        result['human_withdrawal_admission']={key:checked[key] for key in
            ('runtime_version','status','stable_identity_count')}
    return result


def run(spec_path, *, board_only=False):
    spec=read(spec_path);dest=spec_path.parent;media=Path(spec['media_directory'])
    raw=cv2.imread(str(media/'source.png'));mask=cv2.imread(str(media/'person_mask.png'),0)>0
    source=read(media/'source_binding.json')
    if source['native_pts']!=spec['native_pts'] or sha(media/'source.png')!=source['png_sha256']:
        raise ValueError('Selected frame source identity changed')
    features=board_features(raw,mask,spec)+glyph_features(raw,mask,spec)
    # One common geometric-error ordering removes redundant glyph samples;
    # explicit corners and every protected long-edge sample stay intact.
    count_before=sum(len(s['uv']) for f in features for s in f['segments'])
    compaction=compact_review_points(features,min(count_before,spec['board_and_print_sparse_budget']))
    compaction['algorithm_selected_sparse_budget']=compaction.pop('maximum_points_requested_by_user')
    remove_temporary_readouts(features)
    ice=None
    if not board_only:
        ice=read(dest/'ice_features.json')
        if ice['source_pts']!=spec['native_pts']:raise ValueError('Ice source clock mismatch')
        if ice['source']['mask_sha256']!=sha(media/'person_mask.png'):
            raise ValueError('Ice foreground mask identity changed')
        features+=ice['features']
    mapping=read(dest/'navigation_mapping.json')
    source.update(source_points_extracted=True,
        mask_path=str(media/'person_mask.png'),mask_sha256=sha(media/'person_mask.png'),
        model_q_sha256=mapping['source_q_sha256'])
    packet=dict(schema='rinkspace.single-frame-source-review.v1',source=source,
        frame_count=1,batch_sequence_index_1based=spec['programme_frame_index_1based'],
        status='candidate_waiting_for_user_review',user_accepted=False,next_frame_processed=False,
        features=features,point_count=sum(len(s['uv']) for f in features for s in f['segments']),
        sparse_compaction=compaction,observation_coordinates_from_projection=False,
        user_hard_point_ceiling=500,programme_frames_101_to_199_processed=False,optimization_performed=False,
        excluded_features_by_user=['blue_strip_lower_edge','yellow_strip_upper_edge','board_panel_seam'],
        unresolved_source_regions=spec.get('unresolved_source_regions',[]))
    if ice:
        packet['inferred_segments']=ice.get('inferred_segments',[])
        packet['source_correlated_groups']=ice.get('source_correlated_groups',[])
        packet['unresolved_source_regions']+=ice['unresolved_source_regions']
        packet['ice_source_readout_packet']=dict(path=str(dest/'ice_features.json'),
            sha256=sha(dest/'ice_features.json'))
        packet['ice_physical_aliases']=ice['physical_aliases']
        packet['source_shape_reasoning']=dict(
            circles_checked_as_projected_physical_circles=True,
            lines_checked_as_projected_physical_straight_lines=True,
            inferred_search_guides_are_not_source_observations=True,
            ice_sampling_policy=ice['source_sampling_policy'])
        correction=dest/'user_correction_01.json'
        if correction.exists():
            packet['user_correction']=dict(path=str(correction),sha256=sha(correction))
    assert packet['point_count']<=500
    for f in features:
        for s in f['segments']:
            xy=np.rint(s['uv']).astype(int)
            if mask[xy[:,1],xy[:,0]].any():raise ValueError('Foreground point '+f['id'])
    packet_path=dest/f"frame_{spec['programme_frame_index_1based']:06d}_points.json"
    if board_only:packet_path=dest/'board_and_print_candidate.json'
    write(packet_path,packet)
    segments=list(source_segments(packet))
    cv2.imwrite(str(media/('board_and_print_review.png' if board_only else 'points_review.png')),
        native_overlay(raw,segments,packet.get('inferred_segments')))
    if not board_only:
        binding=make_binding(packet,packet_path,read(MODEL/'feature_registry.json'))
        write(dest/f"frame_{spec['programme_frame_index_1based']:06d}_model_binding.json",binding)
        navigation=cv2.imread(str(media/'navigation_raw.png'));navigation[mask]=raw[mask]
        cv2.imwrite(str(media/'navigation.png'),navigation)
        page = AssetStore(ROOT).review_page(dest)
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(review_html(packet,segments,media/'source.png',media/'navigation.png',binding),encoding='utf8')
    return dict(point_count=packet['point_count'],feature_counts={f['id']:sum(len(s['uv'])for s in f['segments'])for f in features})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification',type=Path);parser.add_argument('--board-only',action='store_true')
    args=parser.parse_args()
    import json
    print(json.dumps(run(args.specification,board_only=args.board_only),ensure_ascii=False))
