"""Read explicit source-qualified fixed pan-head support, without projection.

Visible rotating housings stay report-only. Native support remains a range;
clipped one-sided bounds stay open, rather than becoming image-edge points.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import cv2,numpy as np
from z190_main_zoom_k1_source_information import read,ref
from z190_continuous_source_bundle import SOURCE_SHA

ROLES=frozenset(('long_side_main','long_side_secondary','low_end'))


def load_fixed_panhead_sources(path,aliases,installation_ids):
    path=Path(path);packet=read(path);authority=ref(path)
    if packet.get('schema')!='rinkspace.real-intercamera-fixed-panhead-source.v1':
        raise ValueError('Explicit fixed pan-head source schema required')
    if packet.get('source_sha256')!=SOURCE_SHA or packet.get('source_time_base')!='1/15360':
        raise ValueError('Direct camera observations must use the verified native programme clock')
    if len(installation_ids)!=3:raise ValueError('Exactly three identified physical installations required')
    accepted=[];excluded=[];cache=set();identities={};held=set()
    for row in packet['rows']:
        p=int(aliases[int(row['source_native_pts'])])
        if row['split']=='held':held.add(p)
    held.update(int(aliases[p])for p in(230011136,230011392,230422016)if p in aliases)
    for row in packet['rows']:
        component=str(row.get('observed_component',''))
        whole_head=component in('fixed_panhead_possible_axis_whole_head_support',
            'fixed_panhead_visible_pan_tilt_head_and_collar_kinematic_volume_support')
        origin_uncertainty=row.get('physical_origin_uncertainty')
        if (not row.get('same_fixed_panhead_identity')or
                row.get('fixed_pivot_source_qualified')is False or
                not(component.startswith('fixed_panhead_visible_')or whole_head)):
            excluded.append(dict(id=row.get('id'),reason='Visible housing or occluded axis is report-only'));continue
        if whole_head:
            enclosing=row.get('visible_pan_tilt_head_assembly_enclosing_bbox_native',
                row.get('visible_pan_tilt_head_and_lower_support_bbox_native'))
            if component=='fixed_panhead_possible_axis_whole_head_support':
                qualified=(row.get('source_engineering_axis_support_does_not_equal_collar_centre')is True and
                    bool(origin_uncertainty)and Path(row.get('whole_source_axis_volume_review','')).is_file())
            else:
                volume_authority=row.get('source_kinematic_support_requalification_authority',{})
                origin_uncertainty=volume_authority.get('relation')
                qualified=(row.get('physical_axis_is_continuous_mount_volume_inference_not_measured_pixel')is True and
                    volume_authority.get('source_only_bbox_manually_read_from_native_coordinate_grid')is True and
                    volume_authority.get('no_projection_or_residual_used_to_expand_support')is True and
                    volume_authority.get('optical_axis_or_housing_centre_not_a_pivot_measurement')is True and bool(origin_uncertainty))
            if (not qualified or enclosing is None or len(enclosing)!=4 or not np.isfinite(enclosing).all()):
                raise ValueError('Whole pan/tilt head support needs source-reviewed mechanical-origin uncertainty')
        if (row.get('independent_source_material_manual_review')is not True or
                row.get('model_projection_used_to_generate_or_select_uv')is not False or
                row.get('frozen_before_fit')is not True or row.get('nominal_uv_is_not_measured_exact_axis')is not True):
            raise ValueError('Manually qualified source interval without exact-axis pixel fabrication required')
        role=row['split']
        if role not in('fit','held'):raise ValueError('Fixed source admission role must be explicit')
        native=int(row['source_native_pts']);actual=int(aliases[native])
        if actual!=int(row['source_actual_pts']):raise ValueError('Direct camera native/actual identity changed')
        if role=='fit'and actual in held:raise ValueError('A held physical exposure or native alias entered fit')
        observer,target=row['observer_installation'],row['target_installation']
        if observer not in ROLES or target not in ROLES or observer==target:
            raise ValueError('Direct observation requires two distinct named physical cameras')
        raw=row['source_native_reference'];png=Path(raw['raw_png'])
        key=(str(png),raw['native_bgr_sha256'])
        if key not in cache:
            if hashlib.sha256(png.read_bytes()).hexdigest()!=raw['raw_png_sha256']:
                raise ValueError('Direct camera source native PNG authority changed')
            image=cv2.imread(str(png))
            if image is None or image.shape!=(1080,1920,3)or hashlib.sha256(image.tobytes()).hexdigest()!=raw['native_bgr_sha256']:
                raise ValueError('Direct camera source BGR pixels changed')
            cache.add(key)
        lower=row['physical_axis_broad_support_xy_min'];upper=row['physical_axis_broad_support_xy_max']
        if len(lower)!=2 or len(upper)!=2:raise ValueError('Native support X/Y range required')
        bounds=np.array([[float('-inf')if a is None else float(a),float('inf')if b is None else float(b)]for a,b in zip(lower,upper)])
        if np.isnan(bounds).any()or np.any(bounds[:,0]>bounds[:,1])or not np.isfinite(bounds).any():
            raise ValueError('A source-qualified interval must constrain at least one physical pixel component')
        if whole_head and not(np.all(bounds[:,0]<=np.asarray(enclosing[:2]))and np.all(bounds[:,1]>=np.asarray(enclosing[2:]))):
            raise ValueError('Possible mechanical origin support must contain its source-reviewed whole head volume')
        if any(a is None or b is None for a,b in zip(lower,upper))and not row.get('source_clipped_or_occluded_axis_support_authority'):
            raise ValueError('One-sided support requires actual clipping/occlusion authority')
        sigma=np.broadcast_to(np.asarray(row['source_sigma_engineering_native_px'],float),(2,)).copy()
        if np.any(sigma<=0)or not np.isfinite(sigma).all()or not row.get('sigma_semantics'):
            raise ValueError('Positive declared source engineering sigma required')
        normalized=dict(id=row['id'],source_native_pts=native,actual_pts=actual,split=role,
            observer_installation=observer,target_installation=target,source_row=row,
            source_authority=authority,factor_source=dict(
                source_visible=True,source_support_manually_qualified=True,
                model_projection_used_to_generate_or_select_uv=False,
                physical_target_kind='fixed_pan_head_pivot',fixed_pivot_source_qualified=True,
                physical_target_authority=dict(packet=authority,observed_component=row['observed_component'],
                    broad_axis_support_not_exact_pixel=True,source_identity=row.get('physical_device_id'),
                    whole_head_mechanical_origin_support=whole_head,
                    physical_origin_uncertainty=origin_uncertainty),
                native_support_bounds_xy_px=bounds.tolist(),sigma_native_px=sigma.tolist(),sigma_semantics=row['sigma_semantics']))
        ident=(observer,target,actual,row['observed_component'])
        if ident in identities:
            previous=identities[ident]
            if previous['split']!=role or previous['factor_source']!=normalized['factor_source']:
                raise ValueError('Conflicting support or role on the same physical target exposure')
            excluded.append(dict(id=row['id'],reason='Duplicate native alias; physical exposure weighted once'));continue
        identities[ident]=normalized;accepted.append(normalized)
    if not accepted:raise ValueError('No qualified fixed camera support in source ledger')
    return accepted,dict(packet=authority,qualified_fixed_panhead_rows=len(accepted),
        fit=sum(r['split']=='fit'for r in accepted),held=sum(r['split']=='held'for r in accepted),
        original_native_source_UV_support_sigma_roles_preserved=True,
        source_image_hashes_verified=True,model_projection_not_used=True,
        source_interval_not_exact_pivot_pixel=True,excluded=excluded,held_actual_pts=sorted(held),accuracy_accepted=False)
