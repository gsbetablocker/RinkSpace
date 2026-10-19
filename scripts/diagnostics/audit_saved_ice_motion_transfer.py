"""Query RGB-qualified original ice pairs on one saved camera, without fitting.

The infinite-depth limit is a geometric comparison, not a replay of the
production profiled-depth objective. Small adjacent-pair errors do not certify
absolute grid alignment or the shared rink geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]
import numpy as np
from calibration.program_tracking.continuous_precise_inverse_optin import precise_world_rays
from calibration.program_tracking.projection import project_central_valid


def reference(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-identity', type=Path, required=True)
    parser.add_argument('--point-package', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads(args.state_identity.read_text(encoding='utf8'))
    package = json.loads(args.point_package.read_text(encoding='utf8'))
    selection = json.loads(args.selection.read_text(encoding='utf8'))
    if (selection['point_package'] != reference(args.point_package)
            or selection['selection_uses_model_error'] is not False
            or selection['source_pixels_sigma_or_correspondence_changed'] is not False):
        raise ValueError('Require unchanged original queries selected by RGB material evidence')
    for evidence in selection['material_review_references']:
        if reference(evidence['path']) != evidence:
            raise ValueError('Material evidence changed')
    if reference(package['point_arrays']['path']) != package['point_arrays']:
        raise ValueError('Original source array archive changed')
    camera_ref = reference(identity['camera_path'])
    if camera_ref['sha256'] != identity['camera_sha256']:
        raise ValueError('Saved camera changed')
    pairs = {row['array_prefix']: row for row in package['pairs']}
    rows = []
    seen = set()
    with np.load(camera_ref['path'], allow_pickle=False) as cameras, np.load(
            package['point_arrays']['path'], allow_pickle=False) as points:
        metadata = json.loads(cameras['metadata_json'].item())
        if (metadata['source_sha256'] != package['source_video']['sha256']
                or metadata['source_time_base'] != package['time_base']):
            raise ValueError('Camera/source identity mismatch')
        indices = {int(pts): index for index, pts in enumerate(cameras['pts'])}
        camera_vectors = cameras['camera25']
        aliases = cameras['exposure_representative_pts']
        for chosen in selection['records']:
            prefix, ordinal = chosen['array_prefix'], int(chosen['ordinal'])
            key = (prefix, ordinal)
            if key in seen or chosen['surface_id'] != 'ice_z0':
                raise ValueError('Require distinct RGB-qualified ice-plane rows')
            seen.add(key)
            pair = pairs[prefix]
            def point(name):
                return points[prefix + '_' + name][ordinal]
            a, b = np.asarray(point('source_uv'), float), np.asarray(point('target_uv'), float)
            ta, tb = int(point('source_pts')), int(point('target_pts'))
            if (ta, tb) != (pair['source_pts'], pair['target_pts']):
                raise ValueError('Original pair direction changed')
            ia, ib = indices[ta], indices[tb]
            if (int(aliases[ia]), int(aliases[ib])) != (ta, tb):
                raise ValueError('Queries must use actual exposure representatives')
            ca, cb = camera_vectors[ia], camera_vectors[ib]
            ray = precise_world_rays(ca, a[None])[0]
            distance = -ca[2] / ray[2]
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError('Qualified ice ray has no forward ice intersection')
            xyz = ca[:3] + distance * ray
            uv, depth, central = project_central_valid(cb, xyz[None])
            limit_camera = cb.copy()
            limit_camera[:3] = ca[:3]
            limit, limit_depth, limit_central = project_central_valid(
                limit_camera, (ca[:3] + 100. * ray)[None])
            if not (central[0] and limit_central[0] and depth[0] > 0 and limit_depth[0] > 0):
                raise ValueError('Transfer leaves the original forward central optical domain')
            rows.append(dict(array_prefix=prefix, ordinal=ordinal,
                original_admission_row_index=int(point('original_admission_row_index')),
                current_compact_bundle_index=int(point('current_compact_bundle_index')),
                source_pts=ta, target_pts=tb, source_uv=a.tolist(), target_uv=b.tolist(),
                source_sigma_px=np.asarray(point('source_sigma_px')).tolist(),
                ice_prediction_uv=uv[0].tolist(), ice_error_xy_px=(uv[0]-b).tolist(),
                ice_error_px=float(np.linalg.norm(uv[0]-b)),
                infinite_depth_limit_error_px=float(np.linalg.norm(limit[0]-b)),
                ice_to_infinite_limit_difference_px=float(np.linalg.norm(uv[0]-limit[0])),
                camera_centre_shift_scene_units=float(np.linalg.norm(cb[:3]-ca[:3]))))
    result = dict(schema='rinkspace.saved-camera-qualified-ice-pair-transfer.v1',
        state_identity=reference(args.state_identity), camera=camera_ref,
        q_sha256=identity['q_sha256'], point_package=reference(args.point_package),
        material_selection=reference(args.selection), records=rows,
        no_solver_or_J_or_profile_update=True, production_rows_changed=False,
        infinite_limit_is_not_production_profile_replay=True,
        absolute_grid_or_shared_geometry_accuracy_proven=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(args.output), points=len(rows),
        ice_error_px=[row['ice_error_px'] for row in rows],
        ice_to_infinite_limit_difference_px=[row['ice_to_infinite_limit_difference_px'] for row in rows])))


if __name__ == '__main__':
    main()
