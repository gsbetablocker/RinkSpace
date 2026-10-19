"""Add source-backed low-end exposures to the existing movable-centre rig.

This module only creates reference nodes. Source UV and world-curve bindings
come from the separately reviewed loader. Reference camera vectors initialize
R, zoom and centre residuals; they never become observation/prior targets.
"""
from __future__ import annotations

import numpy as np

from continuous_camera_model import FrameBinding
from continuous_rig_camera import ContinuousRigCameraModel
from continuous_reference_calibration import (
    ReferenceCalibration, StaticReferenceBasis, seed_reference_initial)


def prepare_low_end_references(data, programme_model, programme_prior, inventory):
    if data['camera_id'] != 'low_end':
        raise ValueError('Low-end source references cannot be assigned to another rig')
    if inventory['source_video_sha256'] != data['diag']['source_sha256']:
        raise ValueError('Reference video differs from the programme source')
    if (inventory.get('schema') != 'rinkspace.low_end_reference_source_inventory.v1'
            or inventory.get('camera_id') != 'low_end'
            or inventory.get('counts', {}).get('reference_frames') != 3
            or inventory['counts'].get('curve_groups') != 23
            or inventory['counts'].get('source_uv') != 148):
        raise ValueError('Reviewed low-end source inventory schema or counts changed')
    if data['scene']['coordinate_system']['verified_metre_scale'] is not False:
        raise ValueError('This calibration uses conditional scene units')
    rows = []
    for incoming in inventory['references']:
        row = dict(incoming)
        if (row.get('camera_id') != 'low_end'
                or row.get('direct_ad3_uv') is not None
                or row.get('derived_ad3_excluded') is not False):
            raise ValueError('Low-end source row changed camera or added unreviewed ad data')
        row['camera25_initializer'] = np.asarray(row['camera25_initializer'], float)
        if (row['camera25_initializer'].shape != (25,)
                or not np.isfinite(row['camera25_initializer']).all()
                or not row['curves']):
            raise ValueError('Every source exposure needs curves and a finite camera25 start')
        row.setdefault('direct_ad3_uv', None)
        row.setdefault('ad3_world_xyz', np.empty((0, 3)))
        row.setdefault('ad3_sigma_px', 1.)
        row.setdefault('derived_ad3_excluded', False)
        for curve in row['curves']:
            uv = np.asarray(curve['source_uv'], np.float64)
            xyz = np.asarray(curve['world_xyz'], np.float64)
            if (uv.ndim != 2 or uv.shape[1] != 2 or not len(uv)
                    or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
                    or not np.isfinite(uv).all() or not np.isfinite(xyz).all()
                    or not np.isfinite(float(curve['sigma_px']))
                    or float(curve['sigma_px']) <= 0):
                raise ValueError('Low-end source curve changed UV/world/sigma schema')
        rows.append(row)
    pts = [int(row['source_pts']) for row in rows]
    if len(set(pts)) != len(pts) or set(pts) & set(programme_model.frames):
        raise ValueError('Independent reference exposures are duplicated or in the programme')
    if set(pts) != {14177280, 205992960, 226160640}:
        raise ValueError('Unexpected low-end source set; review source identities explicitly')
    reserved = set(programme_prior) | set(data['captured']['guesses']) | set(pts)
    key = max(reserved) + 1
    initial = {int(k): np.asarray(v, float).copy() for k, v in programme_prior.items()}
    bindings, splines = {}, {}
    for ordinal, row in enumerate(rows):
        stamp = int(row['source_pts'])
        frame_key, control_key = key, key + 1
        key += 2
        shot_id = 2000 + ordinal
        splines[shot_id] = StaticReferenceBasis(shot_id, stamp, (control_key,))
        bindings[stamp] = FrameBinding(stamp, frame_key, shot_id, 'low_end',
                                        row['camera25_initializer'].copy())
        row.update(frame_key=frame_key, control_key=control_key)
        initial[frame_key] = np.zeros(11)
        initial[control_key] = np.zeros(11)
    model = ContinuousRigCameraModel(
        splines, bindings, programme_model.lens_base_keys,
        programme_model.lens_slope_keys, programme_model.lens_zoom_reference,
        mount_keys=programme_model.mount_keys, offset_keys=programme_model.offset_keys,
        offset_slope_keys=programme_model.offset_slope_keys,
        brown_keys=programme_model.brown_keys,
        brown_slope_keys=programme_model.brown_slope_keys)
    seed_reference_initial(model, initial, rows)
    if any(not np.array_equal(v, initial[k]) for k, v in programme_prior.items()):
        raise RuntimeError('Adding references changed programme prior targets')
    metadata = dict(
        camera_id='low_end', reference_frame_count=len(rows), reference_pts=pts,
        source_video_sha256=data['diag']['source_sha256'],
        current_programme_scene_sha256=data['diag']['model_sha256'],
        source_curve_group_count=sum(len(r['curves']) for r in rows),
        source_curve_pixel_count=sum(len(c['source_uv']) for r in rows for c in r['curves']),
        reference_image_sha256={str(r['source_pts']): r['image_sha256'] for r in rows},
        reference_sources={str(r['source_pts']): dict(
            image_path=r['image_path'], image_sha256=r['image_sha256'],
            camera_assignment_authority=r.get('camera_assignment_authority'),
            observation_file=r.get('source_observation_file'),
            observation_sha256=r.get('source_observation_sha256'),
            source_measurement_files=r.get('source_measurement_files', []),
            curve_targets=[c['target'] for c in r['curves']],
            curve_ids=[c['id'] for c in r['curves']],
            curve_identity_statuses=[c['identity_status'] for c in r['curves']],
            curve_measurement_sources=[dict(
                id=c['id'], source_file=c['source_file'],
                source_file_sha256=c['source_file_sha256'],
                source_collection=c['source_collection'],
                source_measurement_role=c['source_measurement_role'],
                source_correlation_group=c['source_correlation_group'],
                source_course_region=c['source_course_region'],
                physical_group_id=c['physical_group_id'],
                source_uv_count=len(c['source_uv']),
                sigma_px=float(c['sigma_px'])) for c in r['curves']])
            for r in rows},
        new_graph_nodes={str(r['source_pts']): dict(frame_key=r['frame_key'],
            c_residual_key=r['control_key']) for r in rows},
        source_inventory_provenance={k: v for k, v in inventory.items()
            if k != 'references'},
        source_uv_modified=False, temporal_reference_links=0,
        reference_pose_prior='none; source camera25 values only initialize new nodes',
        reference_c_residual_prior_sigma_scene=.25,
        camera_model='Same adjustable P/offset/lens/Brown nodes as the complete programme rig',
        first_camera_fixed=False, optical_center_fixed=False,
        source_validation_status='Historical source images; not a new blind validation set')
    return ReferenceCalibration(model, initial, rows, metadata)
