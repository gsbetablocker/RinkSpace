"""Source-only top-deck evidence and exclusions; the native board stays zero-thickness."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
from .vision_near_board_material import extract_candidate, _transition_interval


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def exterior_mask(points, iceward_normals, support, native_hw):
    """Conservative source search exclusion; interpolation is never a measurement.

    Only the measured scan span is covered. No line is extrapolated past its
    visible endpoints. Material intervals keep their iceward limit excluded.
    """
    uv, normal, interval = map(np.asarray, (points, iceward_normals, support))
    mask = np.zeros(native_hw, bool)
    if len(uv) < 2:
        return mask
    normal = normal/np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    axis = int(np.argmax(np.ptp(uv, axis=0)))
    cross = 1-axis
    if not (np.all(normal[:, cross] > .25) or np.all(normal[:, cross] < -.25)):
        return mask
    order = np.argsort(uv[:, axis]); uv, normal, interval = uv[order], normal[order], interval[order]
    if np.any(np.diff(uv[:, axis]) <= 0):
        return mask
    edge = uv+normal*interval[:, 1, None]
    first, last = int(np.ceil(uv[0, axis])), int(np.floor(uv[-1, axis]))
    length = native_hw[1-axis]
    scans = np.arange(max(0,first), min(length-1,last)+1)
    boundary = np.interp(scans, uv[:,axis], edge[:,cross])
    positive = np.median(normal[:,cross]) > 0
    for scan, bound in zip(scans,boundary):
        if axis == 1:
            if positive: mask[scan,:int(np.ceil(bound))+1] = True
            else: mask[scan,max(0,int(np.floor(bound))):] = True
        else:
            if positive: mask[:int(np.ceil(bound))+1,scan] = True
            else: mask[max(0,int(np.floor(bound))):,scan] = True
    return mask


@lru_cache(maxsize=4)
def _grey_history(report_path, report_sha):
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_sha:
        raise ValueError('Historical near-board source report changed')
    packet = json.loads(raw)
    return packet, {r['native_pts']:r for r in packet['frames']}


@lru_cache(maxsize=1)
def configured_near_evidence():
    path=Path(__file__).resolve().parents[2]/'configs/rink_vision.json'
    if not path.is_file():return {}
    return json.loads(path.read_text(encoding='utf-8-sig')).get('near_board_source_evidence',{})


def read_near_board(rgb, binding, *, gray_history=None, guides=(), person_mask=None, qualified_generation=None):
    """Read current pixels using only explicitly selected historical evidence.

    The generic reader must not open another configuration's event or Data
    paths. SourceRinkReader passes its selected evidence; missing explicit
    references continue to fail their existing identity/hash checks.
    """
    image = np.ascontiguousarray(rgb[...,::-1])
    from .vision_near_board_qualified import reread_qualified_near_board
    qualified = reread_qualified_near_board(image, binding,
        qualified_generation, person_mask=person_mask)
    if qualified is not None:
        return dict(qualified, source_binding=dict(binding),
            decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            exclusion_pixel_count=int(qualified['ad_exclusion_mask'].sum()),
            vertical_face_admission_mask=qualified['qualified_vertical_face_mask'],
            model_board_thickness=0,world_geometry_modified=False,
            exterior_top_deck_cameras_judges_people_are_advertisements=False,
            mask_is_search_exclusion_not_measured_geometry=True,
            unsampled_scan_spans_extrapolated=False,
            unsampled_exclusion_spans_extrapolated=True)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    blue = (extract_candidate(image, binding['source_pts'], person_mask=person_mask)
            if image.shape == (1080,1920,3) else [])
    readings, mask = [], np.zeros(rgb.shape[:2],bool)
    for row in blue:
        uv, normals, interval = (row[k] for k in ('observed_uv','outward_normal','normal_interval_px'))
        mask |= exterior_mask(uv,normals,interval,rgb.shape[:2])
        readings.append(dict(kind='board_top',source_board_view='top_face_only',
            material='blue_cap_to_ice',actual_source_uv=uv,iceward_normals=normals,
            normal_interval_px=interval, source_context=row['source_context'],
            native_path_id='board.top', physical_side_identity='unresolved_without_source_binding',
            human_confirmed_current_pixels=False, source_point_count=len(uv)))
    # Grey material has reviewed source-window evidence; colour alone is not its identity.
    grey_guides = [g for g in guides if g.get('source_board_view') == 'top_face_only'
                   and g.get('near_board_material') == 'grey_to_ice' and g.get('identity_authority')]
    seeds, history_ref = [], None
    if gray_history:
        packet, rows = _grey_history(gray_history['path'],gray_history['sha256'])
        origin = packet['provenance']
        if (origin['source_video_sha256']==binding['source_sha256']
                and origin['native_time_base']==binding['time_base']
                and binding['source_pts'] in rows):
            row=rows[binding['source_pts']]
            if row['source_pixel_sha256'] != digest:
                raise ValueError('Near-board report does not belong to current native RGB')
            seeds=[(m['reviewed_seed_x'],m['y']) for m in row['measurements']+row['unknown_rows']]
            history_ref=gray_history
    for guide in grey_guides:
        points=_array(guide['points_uv'])
        valid=_array(guide.get('temporal_flow_supported',np.ones(len(points),bool)))
        seeds.extend(points[valid].tolist())
    if seeds:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        intervals, unknown = [], []
        for x,y in seeds:
            bounds,status,contrast=_transition_interval(gray,int(round(y)),int(round(x)))
            if bounds is None:
                unknown.append(dict(query_uv=[x,y],status=status)); continue
            intervals.append(dict(y=int(round(y)),x_interval_px=bounds,contrast=contrast))
        if intervals:
            # These midpoints represent intervals for navigation/display only.
            # The original grey reader deliberately does not claim precise UV.
            uv=np.asarray([[np.mean(r['x_interval_px']),r['y']] for r in intervals],np.float32)
            half=np.asarray([np.diff(r['x_interval_px'])[0]/2 for r in intervals])
            normal=np.tile([1.,0.],(len(uv),1))
            support=np.c_[-half,half]
            mask |= exterior_mask(uv,normal,support,rgb.shape[:2])
            readings.append(dict(kind='board_top',source_board_view='top_face_only',
                material='grey_to_ice',actual_source_uv=np.empty((0,2),np.float32),
                actual_source_intervals=intervals,interval_display_uv=uv,iceward_normals=normal,
                normal_interval_px=support,unknown_rows=unknown,source_reference=history_ref,
                native_path_id='board.top',physical_side_identity='lower_long_side',
                source_point_count=0,source_interval_count=len(intervals),
                interval_midpoints_are_exact_source_points=False))
    return dict(source_binding=dict(binding),decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        readings=readings,ad_exclusion_mask=mask,exclusion_pixel_count=int(mask.sum()),
        model_board_thickness=0,world_geometry_modified=False,
        exterior_top_deck_cameras_judges_people_are_advertisements=False,
        mask_is_search_exclusion_not_measured_geometry=True,
        unsampled_scan_spans_extrapolated=False, source_reader='retained_20260927_20260928_material_readers')


def exclude_top_face_prints(instance_packet, near):
    """Prevent any advertisement path from re-admitting top deck/outside pixels."""
    from .feedback_runtime import native_source
    first,second=instance_packet['source_binding'],near['source_binding']
    a,b=native_source(first),native_source(second)
    if (any(a[k]!=b[k] for k in ('video_sha256','pts','time_base','coordinate_domain'))
            or (first.get('shot_id') is not None and second.get('shot_id') is not None
                and first['shot_id']!=second['shot_id'])):
        raise ValueError('Near-board mask and advertisement belong to different source frames')
    blocked=near['ad_exclusion_mask'].copy()
    if near.get('vertical_face_admission_mask') is not None:
        blocked |= ~_array(near['vertical_face_admission_mask'])
    for item in instance_packet.get('instances',()):
        if item.get('kind') != 'advertisement_panel': continue
        region=item.get('region_mask')
        if region is not None:
            keep=~blocked
            if hasattr(region,'detach'):
                import torch
                keep=torch.as_tensor(keep,device=region.device)
            item['region_mask']=region & keep
        for key in ('boundary_readings','source_print_role_conflict_readings','source_print_raw_boundary_readings'):
            for row in item.get(key,()):
                uv=_array(row['actual_source_uv']); ij=np.rint(uv).astype(int)
                valid=(ij[:,0]>=0)&(ij[:,0]<blocked.shape[1])&(ij[:,1]>=0)&(ij[:,1]<blocked.shape[0])
                rejected=~valid; rejected[valid]=blocked[ij[valid,1],ij[valid,0]]
                if not rejected.any(): continue
                row['top_face_excluded_source_uv']=uv[rejected]
                value=row['actual_source_uv']
                keep=~rejected
                if hasattr(value,'detach'):
                    import torch
                    keep=torch.as_tensor(keep,device=value.device)
                for name in ('actual_source_uv','source_indices','segment_ids'):
                    if name in row and len(row[name])==len(uv):row[name]=row[name][keep]
                # Rebuild only consecutive retained source runs; never bridge an excluded gap.
                row['pre_top_face_exclusion_runs']=row.pop('measured_runs',[])
                retained=_array(row['actual_source_uv'])
                indices=_array(row.get('source_indices',np.flatnonzero(~rejected)))
                segments=_array(row.get('segment_ids',np.zeros(len(retained),int)))
                runs=[];start=0
                for stop in range(1,len(retained)+1):
                    if stop==len(retained) or indices[stop]!=indices[stop-1]+1 or segments[stop]!=segments[stop-1]:
                        runs.append(dict(actual_source_uv=row['actual_source_uv'][start:stop],
                            source_indices=indices[start:stop],segment_id=int(segments[start])))
                        start=stop
                row['measured_runs']=runs
                if 'coherent_source_uv' in row:
                    row.update(coherent_source_uv=row['actual_source_uv'],coherent_source_indices=indices,
                        coherent_measured_runs=runs)
                quality=row.get('source_quality_mask')
                if quality is not None:
                    quality[...] = False
                    quality[indices] = True
                row['source_peak_candidates']=None
                row.setdefault('metadata',{}).update(actual_measured_count=int((~rejected).sum()),top_deck_excluded=True)
        item['top_face_exclusion_applied']=True
    instance_packet['near_board_ad_exclusion']=dict(source_binding=near['source_binding'],
        exclusion_pixel_count=near['exclusion_pixel_count'],source_reader=near['source_reader'])
    return instance_packet
