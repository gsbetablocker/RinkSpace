"""Map native source pixels to the frozen rink plane without fitting anything.

An ice-ray footprint is conditional on the selected camera, scene and Z=0.
It is neither an independently measured 3-D blade nor proof of contact. Only
an explicitly annotated ``ice_interface`` is exported as a contact candidate;
all other features remain ray footprints with their source interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .action_clip_input import camera25_from_snapshot
from .camera_bridge import rays
from .rebuild_exposure_body import moving_projection


SCHEMA = 'rinkspace.source-skate-plane.v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class SourceSkatePlane:
    """Read-only pixel/plane bridge using each pixel's own native row camera.

    ``camera`` and ``info`` are the frozen ``camera_binding`` outputs. No body
    pose, shoe geometry, material-point prediction or optimization is queried.
    """

    def __init__(self, camera, info):
        self.camera, self.info = camera, dict(info)
        self.scale = float(info['metres_per_scene_unit'])
        self.solver_scale = float(camera.scale)
        if not np.isfinite([self.scale, self.solver_scale]).all() or min(self.scale, self.solver_scale) <= 0:
            raise ValueError('Positive finite frozen scales required')
        native_pts = np.asarray(camera.p['pts'])
        if len(set(map(int, native_pts))) != len(native_pts):
            raise ValueError('Duplicate native source PTS')
        self.frames = {int(pts): i for i, pts in enumerate(native_pts)}

    def _indices(self, pts, count):
        values = np.broadcast_to(np.asarray(pts), (count,))
        if not np.isfinite(values).all() or np.any(values != values.astype(np.int64)):
            raise ValueError('Source PTS must be finite integers')
        try:
            return np.array([self.frames[int(v)] for v in values], dtype=int)
        except KeyError as exc:
            raise ValueError('PTS absent from frozen native exposures') from exc

    def inverse_pixels(self, pts, points_uv):
        """Return per-point native row time, conditional XYZ and validity.

        Invalid Brown branches, outside-source pixels, parallel/backward rays
        remain invalid. Nothing is clamped or replaced by a body prediction.
        """
        uv = np.asarray(points_uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError('Expected native points_uv (N,2)')
        frames = self._indices(pts, len(uv))
        valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) &
                 (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1079))
        scene = np.full((len(uv), 3), np.nan)
        row_time = np.full(len(uv), np.nan)
        direction = np.full_like(scene, np.nan)
        for i in np.flatnonzero(valid):
            row_time[i] = (float(self.camera.p['motion_times'][frames[i]]) +
                           float(self.camera.p['rig'][37]) * (uv[i, 1] / 1079. - .5))
        use = np.flatnonzero(valid)
        if len(use):
            p = self.camera.p
            # Equal native row times have the same frozen camera. Batch those
            # Brown inversions; this changes neither rows nor optical model.
            times, camera_index = np.unique(row_time[use], return_inverse=True)
            pose = self.camera.core.evaluate_motion_camera(
                p['rig'], p['controls'], float(p['first_control_time']),
                float(p['control_dt']), times)
            cameras = camera25_from_snapshot(pose)
            for j, cam25 in enumerate(cameras):
                indices = use[camera_index == j]
                ray, good = rays(uv[indices], cam25)
                centre = cameras[j, :3]
                direction[indices] = ray
                good &= np.isfinite(ray).all(axis=1) & (np.abs(ray[:, 2]) >= 1e-12)
                distance = np.full(len(indices), np.nan)
                distance[good] = -centre[2] / ray[good, 2]
                good &= np.isfinite(distance) & (distance > 0)
                valid[indices] = good
                scene[indices[good]] = centre + distance[good, None] * ray[good]
                scene[indices[good], 2] = 0.
        return dict(points_uv=uv.copy(), frames=frames, valid=valid,
                    camera_row_time_seconds=row_time, ray_world=direction,
                    xyz_scene=scene, xyz_conditional_m=scene * self.scale)

    def project_scene(self, pts, xyz_scene):
        """Native implicit-row projection through repeated static five nodes."""
        xyz = np.asarray(xyz_scene, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Expected finite frozen-scene XYZ (N,3)')
        frames = self._indices(pts, len(xyz))
        # FixedNativeProjector consumes solver length units and divides by
        # camera.scale. This is distinct from the conditional export scale.
        nodes = np.broadcast_to((xyz * self.solver_scale)[:, None, None, :],
                                (len(xyz), 5, 1, 3)).copy()
        return moving_projection(self.camera, nodes, frames)['uv'][:, 0]

    def map_points(self, pts, points_uv, *, pixel_radius=None,
                   jacobian_step_px=.1, envelope_samples=64):
        """Map pixels, verify native closure and propagate source pixel error.

        Jacobians are d(X,Y)/d(u,v), including perturbed row timing. A supplied
        radius denotes an isotropic pixel disk; the nonlinear boundary is a
        finite sampled envelope, not a proven bound. Calibration/scale error
        has no covariance here and is explicitly excluded.
        """
        if not np.isfinite(jacobian_step_px) or jacobian_step_px <= 0:
            raise ValueError('Positive finite Jacobian step required')
        if int(envelope_samples) != envelope_samples or envelope_samples < 8:
            raise ValueError('At least eight uncertainty boundary samples required')
        result = self.inverse_pixels(pts, points_uv)
        n = len(result['valid'])
        point_pts = np.broadcast_to(np.asarray(pts), (n,))
        closure_uv = np.full((n, 2), np.nan)
        good = result['valid']
        if good.any():
            closure_uv[good] = self.project_scene(point_pts[good], result['xyz_scene'][good])
        result['native_five_node_reprojection_uv'] = closure_uv
        result['native_five_node_closure_error_px'] = np.linalg.norm(
            closure_uv - result['points_uv'], axis=1)
        jac = np.full((n, 2, 2), np.nan)
        for axis in range(2):
            delta = np.zeros((n, 2)); delta[:, axis] = jacobian_step_px
            plus = self.inverse_pixels(point_pts, result['points_uv'] + delta)
            minus = self.inverse_pixels(point_pts, result['points_uv'] - delta)
            use = good & plus['valid'] & minus['valid']
            jac[use, :, axis] = (plus['xyz_conditional_m'][use, :2] -
                                minus['xyz_conditional_m'][use, :2]) / (2 * jacobian_step_px)
        result['xy_jacobian_conditional_m_per_px'] = jac
        result['jacobian_step_px'] = jacobian_step_px
        if pixel_radius is not None:
            radius = np.broadcast_to(np.asarray(pixel_radius, float), (n,)).copy()
            if not np.isfinite(radius).all() or np.any(radius < 0):
                raise ValueError('Pixel uncertainty radius must be finite and nonnegative')
            angle = 2 * np.pi * np.arange(envelope_samples) / envelope_samples
            offsets = np.column_stack((np.cos(angle), np.sin(angle)))
            boundary_uv = result['points_uv'][:, None] + radius[:, None, None] * offsets
            boundary = self.inverse_pixels(np.repeat(point_pts, envelope_samples),
                                           boundary_uv.reshape(-1, 2))
            boundary_xy = boundary['xyz_conditional_m'][:, :2].reshape(n, envelope_samples, 2)
            boundary_valid = boundary['valid'].reshape(n, envelope_samples)
            aabb = np.full((n, 2, 2), np.nan)
            distance_max = np.full(n, np.nan)
            for i in range(n):
                if good[i] and boundary_valid[i].all():
                    aabb[i] = [boundary_xy[i].min(axis=0), boundary_xy[i].max(axis=0)]
                    distance_max[i] = np.linalg.norm(boundary_xy[i] - result['xyz_conditional_m'][i, :2], axis=1).max()
            result.update(pixel_uncertainty_radius_px=radius,
                sampled_pixel_disk_boundary_xy_conditional_m=boundary_xy,
                sampled_pixel_disk_boundary_valid=boundary_valid,
                sampled_xy_aabb_conditional_m=aabb,
                sampled_xy_max_offset_conditional_m=distance_max,
                linearized_xy_ellipse_semiaxes_conditional_m=
                    np.array([np.linalg.svd(j, compute_uv=False) * r if np.isfinite(j).all()
                              else [np.nan, np.nan] for j, r in zip(jac, radius)]))
        return result

    def grid_uv(self, pts, centre_xy_conditional_m, *, half_extent_m=1.,
                spacing_m=.25, samples_per_line=65):
        """Auxiliary local coordinate grid, never claimed to be printed ice lines."""
        centre = np.asarray(centre_xy_conditional_m, float)
        if centre.shape != (2,) or not np.isfinite(centre).all():
            raise ValueError('Expected finite auxiliary grid centre XY')
        if min(half_extent_m, spacing_m) <= 0 or samples_per_line < 2:
            raise ValueError('Positive auxiliary grid extent/spacing required')
        values = np.arange(-np.floor(half_extent_m / spacing_m),
                            np.floor(half_extent_m / spacing_m) + 1) * spacing_m
        span = np.linspace(-half_extent_m, half_extent_m, samples_per_line)
        lines = []
        for axis in (0, 1):
            for offset in values:
                xy = np.empty((samples_per_line, 2))
                xy[:, axis] = centre[axis] + offset
                xy[:, 1-axis] = centre[1-axis] + span
                xyz = np.c_[xy / self.scale, np.zeros(samples_per_line)]
                uv = self.project_scene(pts, xyz)
                lines.append(dict(constant_axis='XY'[axis],
                    constant_coordinate_conditional_m=float(centre[axis]+offset),
                    xyz_scene=xyz, points_uv=uv,
                    inside_native_image=((uv[:, 0] >= 0) & (uv[:, 0] < 1920) &
                                         (uv[:, 1] >= 0) & (uv[:, 1] <= 1079))))
        return dict(kind='auxiliary_scene_coordinate_grid_not_printed_ice_markings',
                    centre_xy_conditional_m=centre, spacing_conditional_m=spacing_m,
                    lines=lines)


def _features(document, diagnostic_footprint):
    """Adapt explicit new features or old contours; old data stay diagnostic."""
    if 'frames' in document:
        for frame in document['frames']:
            raw_pts = frame['pts']
            if not np.isfinite(raw_pts) or raw_pts != int(raw_pts):
                raise ValueError('Annotation PTS must be a source-clock integer')
            for name, feature in frame.get('features', {}).items():
                if not isinstance(feature, dict):
                    raise ValueError('Each named feature must be an object')
                yield frame, name, feature
    elif diagnostic_footprint and 'rows' in document:
        for row in document['rows']:
            yield row, 'diagnostic_visible_contour', dict(row,
                points_uv=row['polyline_uv'], status=row.get('evidence_status', 'unknown'))
    else:
        raise ValueError('Expected frames/features, or explicitly enabled diagnostic rows')


def map_document(document, bridge, *, diagnostic_footprint=False, pixel_radius=None,
                 include_grid=True, feature_names=('runner_lower_edge', 'ice_interface')):
    """Preserve feature status and source provenance; do not promote footprints."""
    if document.get('time_base') != [1, 12800]:
        raise ValueError('Expected native source time_base [1,12800]')
    frames, arrays = {}, {}
    errors = []
    for frame, name, feature in _features(document, diagnostic_footprint):
        raw_pts = frame['pts']
        if not np.isfinite(raw_pts) or raw_pts != int(raw_pts):
            raise ValueError('Annotation PTS must be a source-clock integer')
        pts = int(raw_pts)
        if ('source_seconds' in frame and
                abs(float(frame['source_seconds']) - pts/12800.) > 1e-9):
            raise ValueError('Annotation seconds do not agree with native PTS')
        foot = frame.get('foot', feature.get('foot'))
        if foot not in (0, 1):
            raise ValueError('Physical foot 0/1 is required')
        key = (pts, foot)
        output_frame = frames.setdefault(key, dict(pts=pts, foot=foot,
            source_seconds=pts/12800., features={}))
        if name in output_frame['features']:
            raise ValueError('Duplicate feature at the same native PTS and foot')
        status = feature.get('status', feature.get('evidence_status', 'unknown'))
        provenance = {k: v for k, v in frame.items() if k not in ('features', 'polyline_uv')}
        if not diagnostic_footprint and name not in feature_names:
            output_frame['features'][name] = dict(source_annotation=feature,
                source_frame=provenance, source_status=status,
                mapping_role='source_structure_only_no_plane_mapping')
            continue
        uv = np.asarray(feature.get('points_uv', []), float)
        if uv.size == 0:
            uv = np.empty((0, 2))
        radius = feature.get('pixel_uncertainty_radius_px',
                    frame.get('pixel_uncertainty_radius_px',
                    document.get('pixel_uncertainty_radius_px', pixel_radius)))
        mapped = bridge.map_points(pts, uv, pixel_radius=radius)
        is_interface = name == 'ice_interface' and not diagnostic_footprint
        result = dict(source_annotation=feature, source_frame=provenance,
            source_status=status,
            mapping_role=('source_ice_interface_candidate' if is_interface else 'source_ray_ice_footprint'),
            actual_contact_confirmed=False,
            physical_interpretation=('annotated interface, conditional plane mapping; source status retained'
                if is_interface else 'ray/Z0 intersection only; no claim this blade feature lies on ice'),
            mapping=mapped)
        if name in ('runner_lower_edge', 'ice_interface') and len(uv):
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
            if length[-1] > 0:
                midpoint = np.array([np.interp(length[-1]/2, length, uv[:, axis]) for axis in range(2)])
                representative_radius = (None if radius is None else float(np.interp(
                    length[-1]/2, length, np.broadcast_to(np.asarray(radius, float), (len(uv),)))))
            else:
                midpoint = uv[0]
                representative_radius = None if radius is None else float(np.broadcast_to(radius, (len(uv),))[0])
            result['auxiliary_representative'] = dict(
                kind='source_polyline_arclength_midpoint_not_a_material_correspondence',
                mapping=bridge.map_points(pts, midpoint[None], pixel_radius=representative_radius))
        output_frame['features'][name] = result
        prefix = f'pts_{pts}_foot_{foot}_{name}'
        for field, value in mapped.items():
            if isinstance(value, np.ndarray):
                arrays[prefix+'__'+field] = value
        errors.extend(mapped['native_five_node_closure_error_px'][mapped['valid']].tolist())
    if include_grid:
        for frame in frames.values():
            candidates = [f for f in frame['features'].values() if 'mapping' in f]
            candidates.sort(key=lambda x: x['mapping_role'] != 'source_ice_interface_candidate')
            for feature in candidates:
                mapped = feature['mapping']
                if mapped['valid'].any():
                    centre = np.mean(mapped['xyz_conditional_m'][mapped['valid'], :2], axis=0)
                    frame['auxiliary_grid'] = bridge.grid_uv(frame['pts'], centre)
                    break
    output = dict(schema=SCHEMA, time_base=[1, 12800],
        source_id=document.get('source_id'), source_sha256=document.get('source_sha256'),
        coordinate_domain='frozen_scene_xyz_and_conditional_metres',
        metres_per_scene_unit=bridge.scale, camera_binding=bridge.info,
        frames=list(frames.values()),
        mapped_feature_names=(['diagnostic_visible_contour'] if diagnostic_footprint else list(feature_names)),
        diagnostic_footprint_only=bool(diagnostic_footprint),
        no_fit_performed=True, no_body_pose_used=True,
        native_five_node_closure=dict(valid_points=len(errors),
            max_px=max(errors) if errors else None,
            median_px=float(np.median(errors)) if errors else None),
        limitations=['conditional on frozen camera, Z=0 and scene scale; not independent measured 3-D',
            'source ice_interface is separate from runner lower edge; footprint does not prove contact',
            'pixel disk boundary and AABB are sampled nonlinear envelopes, not proven bounds',
            'pixel uncertainty propagation excludes camera, scene and scale covariance (not supplied)',
            'auxiliary coordinate grid is not a real printed ice marking'])
    arrays['metadata_json'] = np.array(json.dumps(_jsonable({k:v for k,v in output.items() if k!='frames'}), ensure_ascii=False))
    return output, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--parameter', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, default=None,
                        help='Programme primary_source_selection.json (defaults beside camera record)')
    parser.add_argument('--output', type=Path, required=True, help='JSON output; companion .npz is written beside it')
    parser.add_argument('--diagnostic-footprint', action='store_true')
    parser.add_argument('--pixel-radius', type=float, default=None,
                        help='Explicit fallback disk radius in source px; missing radius stays unknown')
    parser.add_argument('--no-grid', action='store_true')
    parser.add_argument('--plane-features', nargs='+', default=['runner_lower_edge', 'ice_interface'],
                        help='Source features to ray-intersect with Z0; other source structures stay unchanged')
    args = parser.parse_args()
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = str(args.native_library.resolve())
    from .blade_projection import SavedBladeProjection
    saved = SavedBladeProjection(args.project, args.stage, args.parameter)
    bridge = SourceSkatePlane(saved.camera, saved.camera_info)
    annotation_bytes = args.annotations.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    document = json.loads(annotation_bytes.decode('utf-8-sig'))
    source_record = (args.source_record or
        Path(saved.camera_info['record']).parents[1] / 'primary_source_selection.json')
    selected_source = json.loads(source_record.read_text(encoding='utf-8-sig'))['primary_source']
    if document.get('source_sha256') != selected_source['sha256']:
        raise ValueError('Annotation source SHA differs from selected programme source')
    if ((selected_source['width'], selected_source['height']) != (1920, 1080) or
            selected_source['time_base'] != '1/12800'):
        raise ValueError('Source pixels/clock differ from current frozen-camera contract')
    output, arrays = map_document(document, bridge,
        diagnostic_footprint=args.diagnostic_footprint, pixel_radius=args.pixel_radius,
        include_grid=not args.no_grid, feature_names=args.plane_features)
    if _sha(args.annotations) != annotation_sha256:
        raise ValueError('Annotation file changed during mapping; preserve a stable source input')
    output.update(annotation_path=str(args.annotations.resolve()), annotation_sha256=annotation_sha256,
                  selected_parameter_path=str(args.parameter.resolve()), selected_parameter_sha256=_sha(args.parameter),
                  source_record_path=str(source_record.resolve()), source_record_sha256=_sha(source_record),
                  annotation_matches_selected_source_sha256=True,
                  implementation_sha256=_sha(__file__))
    arrays['document_json'] = np.array(json.dumps(_jsonable(output), ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(output), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    print(json.dumps(dict(output=str(args.output), frames=len(output['frames']),
        diagnostic_footprint=output['diagnostic_footprint_only'],
        closure=output['native_five_node_closure']), ensure_ascii=False))


if __name__ == '__main__':
    main()
