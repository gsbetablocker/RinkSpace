"""Actual alias pixels inherit the verified canonical grid-context identity.

This independent delta API does not alter the frozen legacy alias helper.
It validates the canonical source context against its own exact BGR binding,
preserves that parent lineage, and creates no additional FIT exposure.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.source_identity_navigation import admissible_source_identity_context
from calibration.program_tracking.strict_model_scope import authenticate_source_reference
from calibration.program_tracking.projection import safe_project
from .vision_board_material_point_profiles import MATERIAL_ROLES, read_printed_material_corners
from .vision_board_material_point_native_alias import _context, _literal
from .vision_board_model_source_profiles import read_circle_material_interface


def review_same_model_board_native_alias(*, canonical_raw_bgr, canonical_board_rows,
        raw_bgr, exclusion_mask, canonical_exclusion_mask, source_binding):
    """JSON-serializable current native readouts; missing roles remain pending."""
    canonical_digest=hashlib.sha256(canonical_raw_bgr.tobytes()).hexdigest()
    native_digest=hashlib.sha256(raw_bgr.tobytes()).hexdigest()
    if (raw_bgr.shape!=(1080,1920,3) or canonical_raw_bgr.shape!=raw_bgr.shape
            or raw_bgr.dtype!=np.uint8 or canonical_raw_bgr.dtype!=np.uint8
            or native_digest!=source_binding.get('decoded_native_BGR_sha256',source_binding.get('decoded_BGR_sha256'))):
        raise ValueError('The two original native source images/bindings are required')
    native=int(source_binding.get('native_pts',source_binding.get('source_pts')))
    results=[];pending=False;frame_image=raw_bgr.astype(np.float32,copy=False)
    cameras={};scene={}
    for row_number,row in enumerate(canonical_board_rows):
        uv=np.asarray(row['uv'],float).reshape(-1,2)
        expected=dict(source_binding,native_pts=int(row['native_pts']),actual_pts=int(row['actual_pts']),
            decoded_native_BGR_sha256=canonical_digest,decoded_BGR_sha256=canonical_digest,shot_id=row['shot_id'])
        allowed=(row.get('static_material_confirmed') is True and row.get('pixel_location_confirmed') is True
            and row.get('actual_source_RGB_measured') is True and row.get('model_projection_used_as_navigation_and_identity_context') is True
            and row['actual_pts']==int(source_binding['actual_pts']) and row['shot_id']==source_binding['shot_id']
            and int(row['native_pts'])<native and admissible_source_identity_context(row,expected=expected))
        result=dict(canonical_row_index=row_number,physical_material_id=row['physical_material_id'],
            native_pts=native,actual_pts=int(source_binding['actual_pts']),uv=[],normal=[],
            canonical_point_count=len(uv),new_FIT_observations_created=False,
            canonical_UV_used_as_output=False,missing_or_competing_canonical_point_indices=list(range(len(uv))))
        if not allowed:
            result['status']='pending_canonical_same_model_physical_identity_or_exact_source_binding';pending=True;results.append(result);continue
        kind=row['curve']['kind'];reads=[]
        if kind=='shared_board_pattern_material_points':
            role=row['material_roles'][0]
            evidence=row['source_identity_navigation_evidence'];ref=evidence['source_model_navigation_reference']
            key=ref['sha256']
            if key not in cameras:
                ip,_=authenticate_source_reference(ref);identity=json.loads(Path(ip).read_text(encoding='utf8'))
                cp,_=authenticate_source_reference(dict(path=identity['camera_path'],sha256=identity['camera_sha256']))
                sp=Path(ip).with_name('scene_UNACCEPTED.json');authenticate_source_reference(dict(path=str(sp),sha256=identity['scene_sha256']))
                with np.load(cp,allow_pickle=False) as saved:cameras[key]=dict(zip(map(int,saved['pts']),saved['camera25']))
                scene[key]=json.loads(sp.read_text(encoding='utf8'))
            material=scene[key]['shared_board_patterns']['shield_fixed_material_points_xyz_scene_units'][str(row['advertisement_number'])]
            xyz=np.asarray([material[r] for r in MATERIAL_ROLES]);guides,_,good=safe_project(cameras[key][row['native_pts']],xyz)
            if good.all():
                index=MATERIAL_ROLES.index(role)
                # Shift model NAV by this actual observed corner; only current
                # RGB two-border intersections can become alias output.
                guides+=uv[0]-guides[index]
                read=read_printed_material_corners(frame_image,exclusion_mask,guides).get(index)
                if read is not None:reads=[(0,read['uv'],read)]
        elif kind=='shared_board_pattern_circle_curve' and len(uv):
            normals=np.asarray(row['normal'],float)
            if normals.shape==uv.shape:
                profile=read_circle_material_interface(frame_image,exclusion_mask,uv,outward_normal=normals)
                reads=[(r['navigation_index'],r['uv'],r) for r in profile['readings']]
        else:
            result['status']='pending_source_family_or_short_native_circle_segment';pending=True;results.append(result);continue
        kept=[];evidence=[]
        for index,own,e in reads:
            first=_context(canonical_raw_bgr,canonical_exclusion_mask,uv[index])
            second=_context(raw_bgr,exclusion_mask,own)
            if first is None or second is None:continue
            correlation=float(first[0]@second[0])
            if correlation<.90:continue
            kept.append((index,own));evidence.append(dict(e,actual_original_source_material_context_correlation=correlation))
        missing=sorted(set(range(len(uv)))-{i for i,_ in kept})
        pending|=bool(missing)
        result.update(status='actual_native_alias_visible_same_model_board_material_read' if kept else 'pending_missing_or_competing_native_material_interface',
            uv=[p for _,p in kept],canonical_point_indices=[i for i,_ in kept],
            actual_source_RGB_measured=bool(kept),actual_native_source_material_readouts=evidence,
            canonical_grid_identity_parentlineage=deepcopy(row['source_identity_navigation_evidence']),
            model_projection_used_as_navigation_and_identity_context=True,projected_UV_used_as_source_pixel_truth=False,
            source_identity_not_confirmed_from_projection_alone=True,fit_residual_used_for_identity=False,
            model_projection_or_fit_residual_used_for_identity=True,
            full_current_person_union_exclusion_retained=True,
            missing_or_competing_canonical_point_indices=missing)
        results.append(result)
    return _literal(dict(board_material_check_pending=pending,
        canonical_board_row_count=len(canonical_board_rows),native_rows=results,
        new_FIT_observations_created=False))
