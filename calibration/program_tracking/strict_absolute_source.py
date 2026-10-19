"""Compact, exact-source ice/board absolute observations for every pixel family.

Source readers remain responsible for actual RGB measurements and entity
continuity. This consumer never upgrades a candidate, fills a missing frame,
or accepts an old family merely because it used to be fitted.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import os

import numpy as np

from .compact_source_json import ContentExactEvidenceCanonicalizer, pack_source_json, unpack_source_json
from .source_identity_navigation import admissible_source_identity_context, source_identity_context_fields

from .strict_model_scope import (
    ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference,
    approved_source_pixel_indices, file_sha, source_pixel_key,
    verify_source_identities_unchanged,
)


PACK_SCHEMA = 'rinkspace.strict-source-absolute-pixel-pack.v1'
ADMITTED_STATUSES = frozenset((
    'admitted_source_material_center_temporal_identity',
    'admitted_source_physical_boundary_temporal_identity',
    'admitted_source_advertisement_temporal_identity',
))
SEMANTICS = frozenset(('paint_centerline', 'paint_layer_boundary',
    'board_physical_boundary', 'advertisement_boundary',
    'advertisement_corner', 'board_print_feature'))


def _literal(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()
        if isinstance(x, np.generic) else str(x)))


def _target_key(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('Explicit current 3D model target descriptor required')
    return json.dumps(_literal(value), sort_keys=True, separators=(',', ':'))


def _temporal_identity(row):
    evidence = row.get('temporal_evidence')
    if (not isinstance(evidence, dict)
            or evidence.get('same_physical_entity_confirmed') is not True
            or evidence.get('current_frame_RGB_measured') is not True
            or not admissible_source_identity_context(evidence,expected=row)
            or evidence.get('physical_material_id') != row['physical_material_id']
            or evidence.get('shot_id') != row['shot_id']
            or evidence.get('current_actual_pts') != row['actual_pts']):
        return False
    # At a shot entry, an explicitly source-confirmed seed still needs a
    # real RGB observation of this frame; projected or interpolated seeds fail.
    if evidence.get('kind') == 'confirmed_original_source_identity_anchor':
        seed = evidence.get('source_identity_authority')
        if not isinstance(seed, dict) or not seed.get('path'):
            return False
        authenticate_source_reference(seed)
        return evidence.get('source_identity_confirmed') is True
    grid_anchor=evidence.get('kind')=='actual_RGB_material_field_order_and_continuity_with_model_navigation'
    if grid_anchor:
        if (evidence.get('model_projection_used_as_navigation_and_identity_context') is not True
                or evidence.get('source_identity_confirmed') is not True
                or evidence.get('source_identity_authority')!=
                    evidence['source_identity_navigation_evidence']['source_identity_authority']):
            return False
    elif evidence.get('kind') != 'same_shot_actual_RGB_temporal_identity':
        return False
    direction=evidence.get('direction','forward')
    previous=evidence.get('previous_actual_pts')
    valid_order=isinstance(previous,int) and (previous<row['actual_pts'] if direction=='forward'
        else previous>row['actual_pts'] if direction=='backward' else False)
    if (evidence.get('previous_shot_id') != row['shot_id']
            or not valid_order
            or evidence.get('previous_frame_RGB_measured') is not True
            or evidence.get('source_RGB_tracking_confirmed') is not True
            or evidence.get('current_feature_semantics_remeasured') is not True
            or len(evidence.get('previous_decoded_BGR_sha256', '')) != 64):
        return False
    return True


def _row_admission_problem(row):
    if row.get('admission_status') not in ADMITTED_STATUSES:
        return 'candidate_or_unconfirmed_identity'
    if (row.get('scope') not in ('ice', 'board')
            or row.get('static_material_confirmed') is not True
            or row.get('actual_source_RGB_measured') is not True
            or not admissible_source_identity_context(row)
            or row.get('pixel_location_confirmed') is not True):
        return 'unconfirmed_static_source_scope_or_measurement'
    if row.get('observation_semantics') not in SEMANTICS:
        return 'unknown_center_or_boundary_semantics'
    if row.get('feature_id') in ('C', 'B1', 'B2') and row['observation_semantics'] != 'paint_centerline':
        return 'ordinary_wide_paint_requires_centerline'
    if (not str(row.get('physical_material_id', '')).strip()
            or not str(row.get('shot_id', '')).strip()
            or not isinstance(row.get('actual_pts'), int)):
        return 'missing_exact_entity_or_exposure'
    if not _temporal_identity(row):
        return 'missing_same_entity_source_temporal_evidence'
    _target_key(row['model_target'])
    return None


def _center_construction_mask(row, uv, sides):
    if row.get('observation_semantics') != 'paint_centerline':
        return np.ones(len(uv), bool)
    if row.get('center_definition') != 'midpoint_of_actual_two_visible_material_edges':
        return np.zeros(len(uv), bool)
    valid = np.isfinite(sides).all(axis=(1, 2))
    # Normalization recomputes this from actual float32 side readouts in
    # float64. No approximate acceptance radius hides edge/center confusion.
    valid &= np.all(uv == sides.mean(axis=1), axis=1)
    valid &= np.linalg.norm(sides[:, 1]-sides[:, 0], axis=1) > 0
    return valid


def _exact_binding(source_binding):
    binding = deepcopy(source_binding)
    binding.setdefault('video_sha256', binding.get('source_sha256'))
    binding.setdefault('path', binding.get('source_path'))
    binding.setdefault('decoded_BGR_sha256', binding.get('decoded_native_BGR_sha256'))
    if (len(binding.get('video_sha256', '')) != 64 or not binding.get('time_base')
            or not binding.get('path') or len(binding.get('decoded_BGR_sha256', '')) != 64
            or not isinstance(binding.get('actual_pts'), (int, np.integer))
            or not isinstance(binding.get('native_pts'), (int, np.integer))):
        raise ValueError('Exact original video/native PTS/actual PTS/BGR binding required')
    binding['actual_pts'] = int(binding['actual_pts'])
    binding['native_pts'] = int(binding['native_pts'])
    binding['pixel_domain'] = 'native1920x1080'
    authenticate_source_reference(dict(path=binding['path'], sha256=binding['video_sha256']))
    return binding


def _selected_review_rows(rows, binding, reviewer):
    source = {k: binding[k] for k in ('video_sha256', 'time_base', 'pixel_domain')}
    endpoints = {}
    for row in rows:
        for uv in row['uv']:
            key = source_pixel_key(source['video_sha256'], source['time_base'], row['actual_pts'], uv)
            proof = dict(actual_pts=row['actual_pts'], native_pts=binding['native_pts'],
                uv=uv.tolist(), scope=row['scope'], physical_material_id=row['physical_material_id'],
                decision='allow', static_material_confirmed=True, pixel_location_confirmed=True,
                source_feature_physical_identity_confirmed=True, actual_source_RGB_measured=True,
                original_source_image_actually_reviewed=False,
                model_projection_or_fit_residual_used_for_identity=False,
                review_authority='source_measurement_with_reviewed_identity', reviewer=reviewer,
                review_note='Current RGB measurement with source-confirmed physical identity and same-shot continuity',
                decoded_BGR_sha256=binding['decoded_BGR_sha256'],
                source_image_evidence=dict(kind='original_video_exact_native_frame', path=binding['path'],
                    sha256=binding['video_sha256'], native_pts=binding['native_pts'],
                    decoded_BGR_sha256=binding['decoded_BGR_sha256']))
            context=source_identity_context_fields(row)
            if context:
                # Do not rewrite grid-assisted identities as historical
                # projection-independent facts.
                proof.pop('model_projection_or_fit_residual_used_for_identity')
                proof.update(context,shot_id=row['shot_id'])
            previous = endpoints.get(key)
            if previous is not None and previous != proof:
                raise ValueError('One exact source pixel has competing physical identities')
            endpoints[key] = proof
    return ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA, source=source,
        reviewed_endpoints=list(endpoints.values())))


def _measured_ice_readout_support(row,measurement,indices):
    previous=row.get('source_readout_uncertainty',{})
    if previous.get('kind')=='actual_source_readout_blur_and_quantization':
        # Selector support may have been referenced to the raw center and
        # sliced before q50 normalization. Rebuild it from this measurement,
        # or leave the weight missing when current normal/blur is unavailable.
        row.pop('source_readout_uncertainty',None)
        row.pop('sigma_px',None)
        row.pop('sigma_semantics',None)
    quantiles=measurement.get('candidate_side_quantiles_uv')
    if quantiles is None:return
    quantiles=np.asarray(quantiles,np.float64)[indices]
    if quantiles.shape!=(len(indices),2,3,2):
        return
    # The final source gate checks these actual q50 interfaces. Retain that
    # same readout for a house boundary as for a paint center; missing blur or
    # adjacency support must not silently move it back to the raw f32 pixel.
    row['source_side_quantiles_uv']=quantiles
    measured=np.isfinite(quantiles[:,:,1]).all(axis=(1,2))
    if measured.any():
        row['source_sides_uv']=np.asarray(row['source_sides_uv'],np.float64).copy()
        row['uv']=np.asarray(row['uv'],np.float64).copy()
        row['source_sides_uv'][measured]=quantiles[measured,:,1]
        row['uv'][measured]=row['source_sides_uv'][measured].mean(axis=1)
        row['source_readout_precision']='actual_gradient_q50_float64_when_available'
    if not np.isfinite(quantiles).all():return
    step=measurement.get('source_readout_quantization_step_px')
    grid=measurement.get('source_interpolation_grid_px')
    if not isinstance(step,(int,float)) or not isinstance(grid,(int,float)) or step<=0 or grid<0:
        return
    # q25/q50/q75 are measured within each edge's signed RGB transition.
    # Their center support is not the distance between the two paint edges.
    if row['observation_semantics']=='paint_centerline':
        row['source_sides_uv']=quantiles[:,:,1]
        row['uv']=row['source_sides_uv'].mean(1)
        vector=row['source_sides_uv'][:,1]-row['source_sides_uv'][:,0]
        length=np.linalg.norm(vector,axis=1)
        if np.any(length<=0):return
        normal=vector/length[:,None]
    elif len(row['uv'])>1 and np.all(np.diff(np.asarray(measurement['candidate_query'])[indices])==1):
        tangent=np.gradient(row['uv'],axis=0);normal=np.c_[-tangent[:,1],tangent[:,0]]
        length=np.linalg.norm(normal,axis=1)
        if np.any(length<=0):return
        normal/=length[:,None]
        row['source_neighbour_adjacency_confirmed']=True
    else:
        return
    projected=np.einsum('nqj,nj->nq',quantiles.mean(axis=1)-row['uv'][:,None],normal)
    blur=np.c_[np.minimum(projected[:,0],projected[:,2]),np.maximum(projected[:,0],projected[:,2])]
    if np.any(blur[:,0]>0) or np.any(blur[:,1]<0):return
    quantization=.5*step+.5*grid*np.abs(normal).sum(1)
    support=blur+np.c_[-quantization,quantization]
    row['source_readout_uncertainty']=dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True,paint_width_used_as_center_tolerance=False,
        measurement_method='current_signed_RGB_interface_q25_q50_q75_readout',
        quantization_step_px=float(step),source_interpolation_grid_px=float(grid),
        actual_readout_blur_interval_px=blur,normal_interval_px=support,
        quantization_margin_px=quantization,is_calibrated_uncertainty=False)
    row['sigma_px']=np.max(np.abs(support),axis=1)
    row['sigma_semantics']='actual_source_readout_support_half_extent_not_calibrated_Gaussian_covariance'


def _measured_board_readout_support(row):
    """Working support from current visible contour readout, never paint width."""
    uncertainty=row.get('source_readout_uncertainty',{})
    if (uncertainty.get('actual_source_RGB_measured') is not True
            or uncertainty.get('paint_width_used_as_center_tolerance') is not False
            or row.get('normal_from_actual_source_neighbour_pixels') is not True
            or row.get('source_neighbour_adjacency_confirmed') is not True):return
    uv=np.asarray(row['uv'],float).reshape(-1,2)
    normal=np.asarray(row.get('normal',[]),float)
    blur=np.asarray(uncertainty.get('actual_readout_blur_interval_px',[]),float)
    step=uncertainty.get('source_native_pixel_quantization_step_px')
    grid=uncertainty.get('source_interpolation_grid_px')
    if (len(uv)<2 or normal.shape!=uv.shape or blur.shape!=(len(uv),2)
            or not np.isfinite(normal).all() or not np.isfinite(blur).all()
            or np.any(blur[:,0]>0) or np.any(blur[:,1]<0)
            or not isinstance(step,(int,float)) or step<=0
            or not isinstance(grid,(int,float)) or grid<0):return
    distances=np.linalg.norm(np.diff(uv,axis=0),axis=1)
    if np.any(distances<=0) or np.any(distances>6.):return
    tangent=np.gradient(uv,axis=0);derived=np.c_[-tangent[:,1],tangent[:,0]]
    length=np.linalg.norm(derived,axis=1)
    if np.any(length<=0):return
    derived/=length[:,None]
    # The reader has already projected its measured transition onto this
    # actual adjacent-pixel normal. A transported/template normal cannot stand
    # in for that direction, even if its label claims current-source support.
    if not np.allclose(np.abs(np.sum(derived*normal,axis=1)),1.,atol=1e-8,rtol=0):return
    quantization=.5*float(step)+.5*float(grid)*np.abs(normal).sum(1)
    support=blur+np.c_[-quantization,quantization]
    row['source_readout_uncertainty']=dict(uncertainty,normal_interval_px=support,
        quantization_margin_px=quantization,is_calibrated_uncertainty=False)
    row['sigma_px']=np.max(np.abs(support),axis=1)
    row['sigma_semantics']='actual_source_readout_support_half_extent_not_calibrated_Gaussian_covariance'


def _normalize_selected_rows(selected_rows, *, source_binding, family, split, reviewer):
    binding = _exact_binding(source_binding)
    if split not in ('fit', 'held'):
        raise ValueError('New source observations must keep FIT/HELD explicit')
    rows, rejected = [], []
    for index, original in enumerate(selected_rows):
        row = deepcopy(original)
        row.setdefault('actual_pts', binding['actual_pts'])
        row.setdefault('family', family)
        row.setdefault('split', split)
        if row['actual_pts'] != binding['actual_pts'] or row['split'] != split:
            raise ValueError('Selected row belongs to another exposure or FIT/HELD role')
        row['actual_pts'] = int(row['actual_pts'])
        if row.get('model_projection_used_as_navigation_and_identity_context') is True:
            current=dict(binding,shot_id=row.get('shot_id'))
            if not admissible_source_identity_context(row,expected=current):
                rejected.append(dict(selected_row_index=index,
                    reason='grid_context_differs_from_actual_measured_source_binding',
                    physical_material_id=row.get('physical_material_id')))
                continue
            row.setdefault('source_binding',current)
        problem = _row_admission_problem(row)
        if problem:
            rejected.append(dict(selected_row_index=index, reason=problem,
                physical_material_id=row.get('physical_material_id')))
            continue
        uv = np.asarray(row['uv'], np.float64).reshape(-1, 2)
        if not len(uv):
            continue
        row['uv'] = uv
        row['native_pts'] = binding['native_pts']
        row['decoded_BGR_sha256'] = binding['decoded_BGR_sha256']
        row.setdefault('original_locator', dict(kind='current_source_selector', selected_row_index=index))
        rows.append(row)
    review = _selected_review_rows(rows, binding, reviewer)
    return dict(rows=rows, review=review, rejected_rows=rejected)


def normalize_selected_ice_rows(selected_rows, measurement, *, source_binding,
        curve_inventory, family='source_remeasured_ice', split='fit', reviewer='ice_source_identity_reader'):
    """Normalize only selector-admitted material observations from this frame.

    The selector supplies the identity/temporal evidence and candidate indices.
    The current measured side arrays supply the center; the packed candidate UV
    remains a raw readout, never an acceptance or projection-derived value.
    """
    selected = selected_rows.get('rows', ()) if isinstance(selected_rows, dict) else selected_rows
    inventory = {entry['curve_id']: entry for entry in curve_inventory}
    raw_uv = np.asarray(measurement['candidate_uv'])
    raw_sides = np.asarray(measurement['candidate_sides'])
    candidate_query = np.asarray(measurement['candidate_query'])
    query_curve = np.asarray(measurement['query_curve'])
    binding = _exact_binding(source_binding)
    mb = measurement['source_binding']
    if (mb.get('source_sha256', mb.get('video_sha256')) != binding['video_sha256']
            or mb['time_base'] != binding['time_base']
            or mb.get('native_pts',mb.get('source_pts'))!=binding['native_pts']
            or mb.get('actual_pts',binding['actual_pts'])!=binding['actual_pts']):
        raise ValueError('Ice reader and native source binding differ')
    measured_BGR=mb.get('decoded_native_BGR_sha256',mb.get('decoded_BGR_sha256'))
    if measured_BGR is not None and measured_BGR!=binding['decoded_BGR_sha256']:
        raise ValueError('Ice measurement and current native source BGR differ')
    normalized = []
    for original in selected:
        row = deepcopy(original)
        if row.get('admission_status') not in ADMITTED_STATUSES:
            normalized.append(row)
            continue
        if measured_BGR is None and (row.get('decoded_native_BGR_sha256',row.get('decoded_BGR_sha256'))!=binding['decoded_BGR_sha256']
                or not measurement.get('decoded_native_RGB_sha256')
                or row.get('decoded_native_RGB_sha256')!=measurement['decoded_native_RGB_sha256']):
            raise ValueError('Ice selected identity and measured current RGB binding differ')
        ids = np.asarray(row.get('original_candidate_indices', row.get('source_candidate_indices',
            row.get('candidate_indices', []))), np.int64)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(raw_uv)):
            raise ValueError('Selected ice row needs exact current source candidate indices')
        curve = inventory[row['curve_id']]
        expected = list(inventory).index(row['curve_id'])
        if not np.all(query_curve[candidate_query[ids]] == expected):
            raise ValueError('Selected ice pixels do not belong to this exact modeled curve')
        if np.any(np.asarray(measurement['candidate_flags'])[ids] & 3):
            raise ValueError('Truncated/unresolved/excluded source candidate cannot be admitted')
        semantics = 'paint_centerline' if curve['operator'] == 'paint_center' else 'paint_layer_boundary'
        if curve['operator'] not in ('paint_center', 'material_interface'):
            raise ValueError('Unresolved ice material semantics cannot be admitted')
        row.update(scope='ice', feature_id=curve['feature_id'], observation_semantics=semantics,
            model_target=dict(kind='current_scene_ice_curve', curve_id=curve['curve_id'],
                feature_id=curve['feature_id'], world_curve_role=curve['world_curve_role']),
            source_sides_uv=raw_sides[ids].astype(np.float64), raw_candidate_uv=raw_uv[ids].copy(),
            source_readout_precision=str(raw_uv.dtype), original_point_indices=ids,
            original_locator=dict(kind='current_ice_source_candidates', curve_id=row['curve_id']))
        local=measurement.get('candidate_local_source_clearance',{})
        local_proofs=[local[int(i)] for i in ids if int(i) in local]
        if local_proofs:row['candidate_local_source_clearance']=local_proofs
        if row['curve_id'] in measurement.get('source_role_override_evidence',{}):
            row['original_locator'].update(candidate_indices_domain='current_actual_RGB_remeasurement_overlay',
                source_role_override_evidence=measurement['source_role_override_evidence'][row['curve_id']])
        if semantics == 'paint_centerline':
            row['uv'] = row['source_sides_uv'].mean(axis=1)
            row['center_definition'] = 'midpoint_of_actual_two_visible_material_edges'
        else:
            row['uv'] = raw_uv[ids].astype(np.float64)
        _measured_ice_readout_support(row,measurement,ids)
        normalized.append(row)
    return _normalize_selected_rows(normalized, source_binding=binding,
        family=family, split=split, reviewer=reviewer)


def board_print_target_inventory(scene):
    """Exact existing frozen print components; this list grants no source identity."""
    prints=scene.get('source_conditioned_board_prints',{})
    if prints.get('schema')!='rinkspace.current-source-conditioned-board-print-contours.v1':return ()
    q_sha=prints.get('q_sha256')
    if not isinstance(q_sha,str) or len(q_sha)!=64:raise ValueError('Frozen board print model identity required')
    result=[]
    for pattern in prints.get('patterns',()):
        for contour in pattern.get('external_contours',()):
            xyz=np.asarray(contour['xyz_scene_units'],float)
            if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():
                raise ValueError('Existing physical board print component has no finite model path')
            result.append(dict(kind='current_scene_board_print_contour',
                advertisement_id=int(pattern['advertisement_id']),contour_id=str(contour['id']),
                source_anchor_actual_pts=int(pattern['anchor_actual_pts']),scene_q_sha256=q_sha))
    if len({_target_key(r) for r in result})!=len(result):raise ValueError('Duplicate frozen print component target')
    return tuple(result)


def board_material_point_target_inventory(scene):
    """Existing discrete model correspondences; grants no current pixel identity."""
    result=[]
    points=scene.get('shared_board_patterns',{}).get('shield_fixed_material_points_xyz_scene_units',{})
    roles=('upper_left_printed_outline_corner','upper_right_printed_outline_corner','bottom_printed_outline_tip')
    for number,material in points.items():
        if int(number) not in (12,25):
            raise ValueError('Unknown discrete printed material model')
        for role in roles:
            if role not in material:continue
            xyz=np.asarray(material[role],float)
            if xyz.shape!=(3,) or not np.isfinite(xyz).all():
                raise ValueError('Existing discrete model point has no finite coordinate')
            result.append(dict(kind='shared_board_pattern_material_points',
                advertisement_id=int(number),material_roles=[role]))
    return tuple(result)


def _discrete_board_material_target(row, targets):
    curve=row['curve'];roles=row.get('material_roles',[])
    target=dict(curve,material_roles=roles)
    if (len(roles)!=1 or row.get('point_correspondence') is not True
            or _target_key(target) not in {_target_key(t) for t in targets}
            or int(row['advertisement_number'])!=int(curve['advertisement_id'])
            or row.get('observation_semantics')!='board_print_feature'):
        raise ValueError('Discrete source correspondence is absent from this exact frozen model')
    uv=np.asarray(row['uv'],float)
    readout=row.get('source_material_point_readout',{})
    bounds=np.asarray(readout.get('actual_2d_readout_bounds_xy_px',[]),float)
    borders=readout.get('visible_border_readings',[])
    if (uv.shape!=(1,2) or np.asarray(row.get('normal',[])).size
            or row.get('source_neighbour_adjacency_confirmed') is not False
            or row.get('physical_material_id')!=f"Ad{curve['advertisement_id']}.material_point.{roles[0]}"
            or readout.get('actual_source_RGB_measured') is not True
            or readout.get('source_guides_used_as_output') is not False
            or readout.get('fitted_scene_or_camera_used_for_readout') is not False
            or readout.get('measurement_method')!='actual_original_RGB_two_visible_printed_border_intersection'
            or not np.array_equal(np.asarray(readout.get('uv',[]),float),uv[0])
            or bounds.shape!=(2,2) or not np.isfinite(bounds).all()
            or np.any(bounds[0]>uv[0]) or np.any(bounds[1]<uv[0])
            or len(borders)!=2 or any(len(edge)<4 for edge in borders)):
        raise ValueError('Discrete source corner requires its own actual two-border readout and 2D support')
    if not np.array_equal(np.asarray(row.get('actual_2d_readout_bounds_xy_px',[]),float),bounds[None]):
        raise ValueError('Discrete source support differs from its actual readout')
    for edge in borders:
        for point in edge:
            if (np.asarray(point.get('uv',[])).shape!=(2,)
                    or np.asarray(point.get('blur',[])).shape!=(2,)
                    or not np.isfinite(np.r_[point['uv'],point['blur']]).all()):
                raise ValueError('Actual source border profiles are missing')
    row.update(model_target=target,scope='board',feature_id=f"ad{curve['advertisement_id']}",
        vision_semantic_kind='ad_boundary',sigma_px=np.nan,
        sigma_semantics='actual_2d_source_readout_bounds_no_calibrated_scalar_sigma')
    row.setdefault('original_point_indices',row['anchor_point_indices'])
    row.setdefault('original_locator',dict(kind='current_source_discrete_printed_material_point',
        source_anchor_id=row.get('source_anchor_id'),material_role=roles[0],
        source_annotation_reference=row.get('source_annotation_reference')))


def normalize_selected_board_rows(selected_rows, measurement, *, source_binding,
        family='source_remeasured_board', split='fit', reviewer='board_source_identity_reader',
        board_print_targets=(), board_material_point_targets=()):
    """Keep exact board/ad source UV and explicit selector identity evidence."""
    binding = _exact_binding(source_binding)
    mb = measurement.get('source', measurement.get('source_binding', {}))
    if (mb.get('video_sha256', mb.get('source_sha256')) != binding['video_sha256']
            or mb.get('native_pts', mb.get('source_pts')) != binding['native_pts']
            or mb.get('actual_pts') != binding['actual_pts']
            or mb.get('decoded_BGR_sha256', mb.get('decoded_native_BGR_sha256')) != binding['decoded_BGR_sha256']):
        raise ValueError('Board reader and exact native source binding differ')
    selected = selected_rows.get('rows', selected_rows.get('advertisement_rows', ())) if isinstance(selected_rows, dict) else selected_rows
    normalized=[]
    for original in selected:
        row=deepcopy(original)
        if row.get('admission_status') in ADMITTED_STATUSES:
            if measurement.get('advertisement_rows') is not None:
                matches=[candidate for candidate in measurement['advertisement_rows']
                    if candidate['physical_material_id']==row['physical_material_id']]
                if not matches:
                    raise ValueError('Selected source advertisement identity is missing')
                # One physical contour can have several currently visible,
                # disconnected runs. They keep one identity and separate rows.
                known={tuple(point) for candidate in matches
                    for point in np.asarray(candidate['uv'],float).reshape(-1,2)}
                if any(tuple(point) not in known for point in np.asarray(row['uv'],float).reshape(-1,2)):
                    raise ValueError('Selected board UV is not a current actual RGB measured pixel')
            curve=row.get('curve')
            target=row.get('model_target',{})
            if curve is not None and curve.get('kind')=='shared_board_pattern_material_points':
                _discrete_board_material_target(row,board_material_point_targets)
            elif target.get('kind')=='current_scene_board_print_contour':
                allowed={_target_key(r) for r in board_print_targets}
                if _target_key(target) not in allowed:
                    raise ValueError('Source print component is absent from this exact frozen model')
                number=int(target['advertisement_id'])
                if int(row['advertisement_number'])!=number:
                    raise ValueError('Actual source print instance and frozen component target differ')
                if row.get('observation_semantics')!='board_print_feature':
                    raise ValueError('Printed material contour cannot become an advertisement rectangle')
                row.setdefault('scope','board');row.setdefault('feature_id',f'ad{number}')
                row.setdefault('vision_semantic_kind','ad_boundary')
                row.setdefault('original_point_indices',row.get('anchor_point_indices',np.arange(len(row['uv']))))
                row.setdefault('original_locator',dict(kind='current_source_printed_component_contour',
                    source_anchor_id=row.get('source_anchor_id'),source_annotation_reference=row.get('source_annotation_reference')))
            elif curve is not None:
                if curve.get('kind') not in ('advertisement_boundary','board_circle','shared_board_pattern_circle_curve'):
                    raise ValueError('Unknown source board curve cannot be normalized into the model')
                number=int(row['advertisement_number'])
                row.setdefault('scope','board');row.setdefault('feature_id',f'ad{number}')
                row.setdefault('model_target',deepcopy(curve))
                row.setdefault('observation_semantics','advertisement_boundary'
                    if curve['kind']=='advertisement_boundary' else 'board_print_feature')
                row.setdefault('vision_semantic_kind','ad_boundary')
                row.setdefault('original_point_indices',row.get('anchor_point_indices',np.arange(len(row['uv']))))
                row.setdefault('original_locator',dict(kind='current_source_advertisement_boundary',
                    source_anchor_id=row.get('source_anchor_id'),source_annotation_reference=row.get('source_annotation_reference')))
            row.setdefault('raw_candidate_uv',np.asarray(row['uv'],float).copy())
            _measured_board_readout_support(row)
        normalized.append(row)
    return _normalize_selected_rows(normalized, source_binding=binding,
        family=family, split=split, reviewer=reviewer)


class StrictAbsolutePackWriter:
    """Append admitted source rows from the shared per-exposure reader."""
    def __init__(self, path, *, source, expected_actual_pts, original_references=(), legacy_pixel_families=(), recovery_journal=None):
        self.path = Path(path).resolve()
        if self.path.exists():
            raise ValueError('Do not overwrite an earlier source observation pack')
        self.source = deepcopy(source)
        self.expected_pts = np.asarray(expected_actual_pts, np.int64)
        if not np.array_equal(self.expected_pts, np.unique(self.expected_pts)):
            raise ValueError('Exact sorted exposure inventory required')
        self.allowed_pts = set(map(int, self.expected_pts))
        self.original_references = list(deepcopy(original_references))
        self.evidence_identities = {}
        for reference in self.original_references:
            path,identity=authenticate_source_reference(reference)
            self.evidence_identities[str(path)]=identity
        self.legacy_pixel_families = sorted(set(legacy_pixel_families))
        self.rows, self.uv, self.sigma, self.sides, self.original_indices = [], [], [], [], []
        self.raw_uv, self.proof_indices = [], []
        self.side_quantiles = []
        self.proof_templates, self.proof_lookup = [], {}
        self.exclusions = []
        self.processed_pts = set()
        self.frame_bindings = {}
        self.finished = False
        self._header_interner = ContentExactEvidenceCanonicalizer()
        self.recovery_journal = None
        if recovery_journal is not None:
            from .strict_source_recovery import SourceAdmissionRecoveryJournal
            self.recovery_journal = SourceAdmissionRecoveryJournal(recovery_journal, self)

    def _register_frame(self,actual_pts,native_pts,decoded_BGR_sha256,shot_id=None):
        value=dict(native_pts=int(native_pts),decoded_BGR_sha256=decoded_BGR_sha256,
            shot_id=None if shot_id is None else str(shot_id))
        if len(decoded_BGR_sha256)!=64:raise ValueError('Exact source BGR hash required')
        previous=self.frame_bindings.get(int(actual_pts))
        if previous is not None:
            if any(previous[k]!=value[k] for k in ('native_pts','decoded_BGR_sha256')):
                raise ValueError('Source frame native clock/hash changed during processing')
            if previous['shot_id'] is not None and value['shot_id'] is not None and previous['shot_id']!=value['shot_id']:
                raise ValueError('One source exposure has competing shot identities')
            if value['shot_id'] is None:value['shot_id']=previous['shot_id']
        self.frame_bindings[int(actual_pts)]=value

    def mark_exposure_processed(self, actual_pts, *, source_binding=None):
        if self.finished:
            raise ValueError('Source pack was already finished')
        if int(actual_pts) not in self.allowed_pts or int(actual_pts) in self.processed_pts:
            raise ValueError('A source exposure must be processed exactly once')
        if source_binding is not None:
            binding=_exact_binding(source_binding)
            if ({k:binding[k] for k in ('video_sha256','time_base','pixel_domain')}!=self.source
                    or binding['actual_pts']!=int(actual_pts)):
                raise ValueError('Processed source exposure identity differs')
            self._register_frame(actual_pts,binding['native_pts'],binding['decoded_BGR_sha256'],binding.get('shot_id'))
            path,identity=authenticate_source_reference(dict(path=binding['path'],sha256=binding['video_sha256']))
            self.evidence_identities[str(path)]=identity
        self.processed_pts.add(int(actual_pts))
        if self.recovery_journal is not None:self.recovery_journal.checkpoint(self)

    def add_row(self, row, review):
        if self.finished:
            raise ValueError('Source pack was already finished')
        row = deepcopy(row)
        if review.source != self.source:
            raise ValueError('Source review and source pack identity differ')
        pts = int(row['actual_pts'])
        if pts not in self.allowed_pts:
            raise ValueError('Source row belongs to an undeclared actual exposure')
        if 'native_pts' in row and 'decoded_BGR_sha256' in row:
            self._register_frame(pts,row['native_pts'],row['decoded_BGR_sha256'],row.get('shot_id'))
        split = row.get('split')
        if split not in ('fit', 'held'):
            raise ValueError('FIT/HELD must stay explicit; pending rows are not FIT')
        uv = np.asarray(row['uv'], np.float64).reshape(-1, 2)
        source_indices = np.asarray(row.get('original_point_indices', np.arange(len(uv))), np.int64)
        if source_indices.shape != (len(uv),):
            raise ValueError('Every source pixel needs its exact original point index')
        problem = _row_admission_problem(row)
        if not problem and row['temporal_evidence']['kind']=='confirmed_original_source_identity_anchor':
            path,identity=authenticate_source_reference(row['temporal_evidence']['source_identity_authority'])
            self.evidence_identities[str(path)]=identity
        keep = np.empty(0, np.int64) if problem else approved_source_pixel_indices(
            pts, uv, review, physical_material_id=row['physical_material_id'])
        sides = np.asarray(row.get('source_sides_uv', np.full((len(uv), 2, 2), np.nan)), float)
        if sides.shape != (len(uv), 2, 2):
            raise ValueError('Paired source-side evidence shape differs from source UV')
        if len(keep):
            keep = keep[_center_construction_mask(row, uv[keep], sides[keep])]
        excluded = np.setdiff1d(np.arange(len(uv)), keep, assume_unique=True)
        if len(excluded):
            self.exclusions.append(dict(family=row['family'], actual_pts=pts,
                feature_id=row['feature_id'], original_locator=row['original_locator'],
                original_point_indices=source_indices[excluded].tolist(),
                reason=problem or 'exact_source_pixel_or_center_construction_not_approved'))
        if not len(keep):
            return 0
        # Source identity and accurate pixels may be retained before a source
        # uncertainty readout exists. Missing weights stay missing; this pack
        # cannot silently become a solver-ready observation graph.
        sigma = np.asarray(row.get('sigma_px',np.nan), float)
        sigma = np.full(len(uv),float(sigma)) if sigma.ndim == 0 else sigma
        if (sigma.shape != (len(uv),) or np.isinf(sigma).any()
                or np.any(sigma[np.isfinite(sigma)] <= 0)):
            raise ValueError('Explicit positive source working sigma required')
        proof_ids = []
        for point in uv[keep]:
            proof = deepcopy(review.endpoint(pts, point))
            if (proof['scope'] != row['scope'] or proof['native_pts'] != row['native_pts']
                    or proof['decoded_BGR_sha256'] != row['decoded_BGR_sha256']):
                raise ValueError('Source row and pixel proof refer to differing source material/frame')
            proof.pop('uv')  # The exact UV is held once in the compact array.
            key = json.dumps(proof, sort_keys=True, separators=(',', ':'))
            if key not in self.proof_lookup:
                self.proof_lookup[key] = len(self.proof_templates)
                self.proof_templates.append(proof)
            proof_ids.append(self.proof_lookup[key])
        raw = np.asarray(row.get('raw_candidate_uv', uv), np.float64).reshape(-1, 2)
        if raw.shape != uv.shape or not np.isfinite(raw).all():
            raise ValueError('Raw measured pixel readout shape differs from normalized UV')
        quantiles=np.asarray(row.get('source_side_quantiles_uv',np.full((len(uv),2,3,2),np.nan)),np.float64)
        if quantiles.shape!=(len(uv),2,3,2):raise ValueError('Source side quantile shape differs')
        header={k:_literal(v) for k,v in row.items() if k not in
            ('uv','sigma_px','source_sides_uv','original_point_indices','raw_candidate_uv','source_side_quantiles_uv')}
        # A partially approved row cannot retain unsliced normals, candidate
        # indices, or readout intervals referring to the removed source UV.
        for key in ('normal','source_candidate_indices','original_candidate_indices',
                    'anchor_point_indices','source_query_indices','candidate_sides','actual_source_normal_uv'):
            if key in header:
                value=np.asarray(header[key])
                if value.ndim and len(value)==len(uv):header[key]=_literal(value[keep])
        uncertainty=header.get('source_readout_uncertainty')
        if isinstance(uncertainty,dict):
            for key in ('actual_readout_blur_interval_px','normal_interval_px','quantization_margin_px'):
                if key not in uncertainty:continue
                value=np.asarray(uncertainty[key])
                if value.ndim>=2 and len(value)==len(uv):uncertainty[key]=_literal(value[keep])
                elif key=='quantization_margin_px' and value.shape==(len(uv),):
                    uncertainty[key]=_literal(value[keep])
        restored_scope=header.get('source_scope_restored_from_authenticated_actual_graph_nodes')
        if restored_scope:
            for key in ('original_candidate_indices','frozen_original_scope_flags'):
                value=np.asarray(restored_scope[key])
                if len(value)!=len(uv):raise ValueError('Recovered source scope locator count differs')
                restored_scope[key]=_literal(value[keep])
        header.update(point_count=len(keep),working_sigma_status='explicit_source_working_sigma'
            if np.isfinite(sigma[keep]).all() else 'missing_or_partial_source_working_sigma_not_fit_ready')
        self.rows.append(self._header_interner.canonicalize(header));self.uv.append(uv[keep]);self.sigma.append(sigma[keep])
        self.sides.append(sides[keep]);self.original_indices.append(source_indices[keep])
        self.raw_uv.append(raw[keep]);self.proof_indices.append(np.asarray(proof_ids, np.int32))
        self.side_quantiles.append(quantiles[keep])
        review.verify_evidence_unchanged()
        self.evidence_identities.update(review.evidence_identities)
        return len(keep)

    def finish(self):
        if self.finished or self.path.exists():
            raise ValueError('Do not overwrite an earlier source observation pack')
        if self.recovery_journal is not None:self.recovery_journal.checkpoint(self)
        verify_source_identities_unchanged(self.evidence_identities)
        for row in self.rows:
            temporal=row['temporal_evidence']
            if temporal['kind'] in ('same_shot_actual_RGB_temporal_identity',
                    'actual_RGB_material_field_order_and_continuity_with_model_navigation'):
                previous=self.frame_bindings.get(temporal['previous_actual_pts'])
                if (previous is not None and previous['decoded_BGR_sha256']!=temporal['previous_decoded_BGR_sha256']):
                    raise ValueError('Temporal previous BGR hash differs from actual processed source frame')
        offsets=np.r_[0,np.cumsum([len(uv) for uv in self.uv])].astype(np.int64)
        metadata=dict(schema=PACK_SCHEMA,source=self.source,original_references=self.original_references,
            legacy_pixel_families=self.legacy_pixel_families,legacy_family_fallback='forbidden',
            expected_actual_pts=self.expected_pts.tolist(),processed_actual_pts=sorted(self.processed_pts),
            all_exposures_processed=self.processed_pts==self.allowed_pts,
            source_measurement_processing_is_not_manual_visual_completion=True,
            all_exposures_exact_native_BGR_bound=self.allowed_pts.issubset(self.frame_bindings),
            source_uncertainty_ready_for_fit=any(r['split']=='fit' for r in self.rows) and all(np.isfinite(s).all()
                for r,s in zip(self.rows,self.sigma) if r['split']=='fit'),
            physically_compacted_approved_pixels_only=True,solver_started=False)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        staging=self.path.with_name(self.path.stem+'.writing.npz')
        if staging.exists():raise ValueError('Preserve an earlier incomplete source pack write')
        np.savez_compressed(staging,uv=np.concatenate(self.uv) if self.uv else np.empty((0,2)),
            sigma_px=np.concatenate(self.sigma) if self.sigma else np.empty(0),
            source_sides_uv=np.concatenate(self.sides) if self.sides else np.empty((0,2,2)),
            raw_candidate_uv=np.concatenate(self.raw_uv) if self.raw_uv else np.empty((0,2)),
            source_side_quantiles_uv=np.concatenate(self.side_quantiles) if self.side_quantiles else np.empty((0,2,3,2)),
            source_proof_template_index=np.concatenate(self.proof_indices) if self.proof_indices else np.empty(0,np.int32),
            source_frame_actual_pts=self.expected_pts,
            source_frame_native_pts=np.asarray([self.frame_bindings.get(int(p),{}).get('native_pts',-1) for p in self.expected_pts],np.int64),
            source_frame_BGR_sha256=np.asarray([self.frame_bindings.get(int(p),{}).get('decoded_BGR_sha256','') for p in self.expected_pts],dtype='S64'),
            original_point_indices=np.concatenate(self.original_indices) if self.original_indices else np.empty(0,np.int64),
            row_offsets=offsets,rows_json=pack_source_json(self.rows),
            source_proof_templates_json=pack_source_json(self.proof_templates),
            exclusions_json=pack_source_json(self.exclusions),
            metadata_json=json.dumps(metadata,ensure_ascii=False,separators=(',',':')))
        verify_source_identities_unchanged(self.evidence_identities)
        if self.path.exists():raise ValueError('Do not replace an earlier source pack')
        os.rename(staging,self.path)
        self.finished = True
        if self.recovery_journal is not None:self.recovery_journal.close()
        return dict(path=str(self.path),sha256=file_sha(self.path),points=int(offsets[-1]),rows=len(self.rows))


class StrictAbsoluteSourcePack:
    """The actual consumer exposes approved FIT and HELD arrays separately."""
    def __init__(self,path,*,expected_sha256):
        if file_sha(path)!=expected_sha256:raise ValueError('Strict absolute source pack changed')
        with np.load(path,allow_pickle=False) as z:self.values={k:z[k].copy() for k in z.files}
        self.metadata=json.loads(str(self.values.pop('metadata_json').item()))
        self.rows=unpack_source_json(self.values.pop('rows_json'))
        self.exclusions=unpack_source_json(self.values.pop('exclusions_json'))
        self.proof_templates=unpack_source_json(self.values.pop('source_proof_templates_json'))
        if (self.metadata.get('schema')!=PACK_SCHEMA or self.metadata.get('legacy_family_fallback')!='forbidden'
                or self.metadata.get('physically_compacted_approved_pixels_only')is not True):
            raise ValueError('Strict compact source pack is required')
        for reference in self.metadata['original_references']:authenticate_source_reference(reference)
        self.allowed={};self.fit_rows=[];self.held_rows=[]
        offsets=self.values['row_offsets'];uv=self.values['uv']
        if offsets.shape!=(len(self.rows)+1,)or offsets[0]!=0 or offsets[-1]!=len(uv)or np.any(np.diff(offsets)<0):
            raise ValueError('Exact compact absolute row offsets required')
        for name, shape in (('uv',(len(uv),2)), ('raw_candidate_uv',(len(uv),2)),
                ('sigma_px',(len(uv),)), ('source_sides_uv',(len(uv),2,2)),
                ('source_side_quantiles_uv',(len(uv),2,3,2)),
                ('source_proof_template_index',(len(uv),)), ('original_point_indices',(len(uv),))):
            if self.values[name].shape != shape:
                raise ValueError('Strict source array shape differs: '+name)
        sigma=self.values['sigma_px']
        if (not np.isfinite(uv).all() or np.isinf(sigma).any()
                or np.any(sigma[np.isfinite(sigma)] <= 0)):
            raise ValueError('Strict absolute storage has invalid UV/sigma')
        proof_ids=self.values['source_proof_template_index']
        if proof_ids.dtype.kind not in 'iu' or np.any(proof_ids<0) or np.any(proof_ids>=len(self.proof_templates)):
            raise ValueError('Exact source proof template indices required')
        expected=set(self.metadata['expected_actual_pts'])
        frame_pts=self.values['source_frame_actual_pts'];frame_native=self.values['source_frame_native_pts']
        frame_sha=self.values['source_frame_BGR_sha256']
        if (not np.array_equal(frame_pts,self.metadata['expected_actual_pts'])
                or frame_native.shape!=frame_pts.shape or frame_sha.shape!=frame_pts.shape):
            raise ValueError('Exact native source exposure catalogue differs')
        self.frame_bindings={int(p):dict(native_pts=int(n),decoded_BGR_sha256=sha.decode())
            for p,n,sha in zip(frame_pts,frame_native,frame_sha) if n>=0 and sha}
        if self.metadata['all_exposures_exact_native_BGR_bound']!=(set(self.frame_bindings)==expected):
            raise ValueError('Source frame binding coverage differs')
        processed=self.metadata['processed_actual_pts']
        if (len(processed)!=len(set(processed)) or not set(processed).issubset(expected)
                or self.metadata['all_exposures_processed'] != (set(processed)==expected)):
            raise ValueError('Source exposure processing inventory differs')
        for i,row in enumerate(self.rows):
            start,stop=map(int,offsets[i:i+2]);points=uv[start:stop]
            if (row['split'] not in ('fit','held') or row['actual_pts'] not in expected
                    or row['point_count'] != len(points) or _row_admission_problem(row)):
                raise ValueError('Unconfirmed row entered strict absolute storage')
            frame=self.frame_bindings.get(row['actual_pts'])
            if frame is None or frame['native_pts']!=row['native_pts'] or frame['decoded_BGR_sha256']!=row['decoded_BGR_sha256']:
                raise ValueError('Strict source row differs from its native exposure catalogue')
            ready=np.isfinite(sigma[start:stop]).all()
            if (row['working_sigma_status']=='explicit_source_working_sigma') != ready:
                raise ValueError('Missing source uncertainty cannot become an explicit source weight')
            temporal=row['temporal_evidence']
            if temporal['kind'] in ('same_shot_actual_RGB_temporal_identity',
                    'actual_RGB_material_field_order_and_continuity_with_model_navigation'):
                previous=self.frame_bindings.get(temporal['previous_actual_pts'])
                if previous is not None and previous['decoded_BGR_sha256']!=temporal['previous_decoded_BGR_sha256']:
                    raise ValueError('Temporal source BGR binding differs from actual processed frame')
            proofs=[dict(self.proof_templates[int(code)],uv=point.tolist())
                for code,point in zip(proof_ids[start:stop],points)]
            for proof in proofs:
                if (proof['scope']!=row['scope'] or proof['actual_pts']!=row['actual_pts']
                        or proof['native_pts']!=row['native_pts']
                        or proof['decoded_BGR_sha256']!=row['decoded_BGR_sha256']):
                    raise ValueError('Strict pixel evidence belongs to another scope/exposure')
            if not _center_construction_mask(row,points,self.values['source_sides_uv'][start:stop]).all():
                raise ValueError('Strict paint centerline differs from actual source sides')
            review=ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=self.metadata['source'],reviewed_endpoints=proofs))
            if len(approved_source_pixel_indices(row['actual_pts'],points,review,physical_material_id=row['physical_material_id']))!=len(points):
                raise ValueError('Strict absolute storage contains an unapproved exact source pixel')
            current=dict(row,uv=points,sigma_px=self.values['sigma_px'][start:stop],
                source_sides_uv=self.values['source_sides_uv'][start:stop],
                raw_candidate_uv=self.values['raw_candidate_uv'][start:stop],
                source_side_quantiles_uv=self.values['source_side_quantiles_uv'][start:stop],
                original_point_indices=self.values['original_point_indices'][start:stop])
            (self.fit_rows if row['split']=='fit'else self.held_rows).append(current)
            if row['split']=='fit':
                target=_target_key(row['model_target'])
                for point in points:
                    key=source_pixel_key(self.metadata['source']['video_sha256'],self.metadata['source']['time_base'],row['actual_pts'],point)
                    self.allowed.setdefault((row['family'],target),set()).add(key)
        ready=bool(self.fit_rows) and all(np.isfinite(row['sigma_px']).all() for row in self.fit_rows)
        if ready!=self.metadata['source_uncertainty_ready_for_fit']:
            raise ValueError('Source weight readiness differs from physically stored FIT rows')

    def filter_legacy_row(self,family,row,*,actual_pts,model_target,source_split=None):
        """No old absolute, ad, wide, manual-centre or alias family can bypass."""
        split=row.get('split',source_split)
        if (split!='fit' or row.get('source_role') in ('held','holdout','future','qualification','pending')
                or family not in {key[0] for key in self.allowed} or not model_target):
            return None
        uv=np.asarray(row['uv'],np.float64).reshape(-1,2)
        allowed=self.allowed.get((family,_target_key(model_target)),set())
        source=self.metadata['source']
        keep=np.asarray([i for i,p in enumerate(uv) if source_pixel_key(source['video_sha256'],source['time_base'],actual_pts,p)in allowed],np.int64)
        if not len(keep):return None
        result=deepcopy(row);result['uv']=uv[keep].copy();result['strict_source_original_point_indices']=keep.tolist()
        for key in ('sigma','sigma_px','normal','normal_interval','normal_interval_px'):
            if key in row:
                value=np.asarray(row[key])
                if value.ndim and len(value)==len(uv):result[key]=value[keep].copy()
        return result

    def compact_legacy_absolute_inputs(self, entries):
        """Physical consumer for all legacy UV families, including unknowns.

        Entry fields are family, row, actual_pts, model_target, split and an
        optional original_locator. No unchecked row is returned to a caller.
        The original row arrays and source records are never modified.
        """
        active, retired = [], []
        for index, entry in enumerate(entries):
            row=entry['row']
            before=len(np.asarray(row['uv']).reshape(-1,2))
            value=self.filter_legacy_row(entry['family'],row,
                actual_pts=entry['actual_pts'],model_target=entry['model_target'],
                source_split=entry.get('split'))
            retained=[] if value is None else value['strict_source_original_point_indices']
            if value is not None:
                active.append(dict(entry,row=value))
            excluded=np.setdiff1d(np.arange(before),retained,assume_unique=True)
            if len(excluded):
                retired.append(dict(family=entry['family'],actual_pts=entry['actual_pts'],
                    original_locator=entry.get('original_locator',dict(entry_index=index)),
                    original_point_indices=excluded))
        return active,retired

    def require_complete_legacy_family_gate(self,current_family_receipts):
        required={r['family'] for r in current_family_receipts if r.get('pixel_input_count',0)>0 or r.get('outside_programme_pixel_count',0)>0}
        motion={r for r in required if r.endswith(':motion')}
        if not (required-motion).issubset(self.metadata['legacy_pixel_families']):
            raise ValueError('Strict source gate omitted an old absolute pixel family')
        return True
