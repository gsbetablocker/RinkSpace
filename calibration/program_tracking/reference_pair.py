"""Real reference-to-programme pixels on one identified physical plane.

Both cameras can move during calibration and retain their own Brown optics.
The plane supplies surface identity; no estimated world point becomes a
fixed measurement. Homography predictions must never be supplied as pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from projection import from_state, project_fast
from track_sequence import PairConstraint
from track_sequence import fast_world_rays


@dataclass
class ReferencePlanePair:
    xy0: np.ndarray
    xy1: np.ndarray
    plane_point: np.ndarray
    plane_normal: np.ndarray
    camera0_template: np.ndarray
    camera1_template: np.ndarray

    def __post_init__(self):
        self.xy0 = np.asarray(self.xy0, float).reshape(-1, 2).copy()
        self.xy1 = np.asarray(self.xy1, float).reshape(-1, 2).copy()
        if self.xy0.shape != self.xy1.shape or not len(self.xy0):
            raise ValueError('Both actual source pixel arrays must have the same length')
        if not np.isfinite(self.xy0).all() or not np.isfinite(self.xy1).all():
            raise ValueError('Actual source pixels must be finite')
        n = len(self.xy0)
        self.plane_point = np.broadcast_to(np.asarray(self.plane_point, float), (n, 3)).copy()
        self.plane_normal = np.broadcast_to(np.asarray(self.plane_normal, float), (n, 3)).copy()
        if not np.isfinite(self.plane_point).all() or not np.isfinite(self.plane_normal).all():
            raise ValueError('Source surface must be a finite plane')
        length = np.linalg.norm(self.plane_normal, axis=1)
        if np.any(length < 1e-10):
            raise ValueError('Source plane must have a nonzero normal')
        self.plane_normal /= length[:, None]
        for name in ('camera0_template', 'camera1_template'):
            camera = np.asarray(getattr(self, name), float).copy()
            if camera.shape != (25,) or not np.isfinite(camera).all():
                raise ValueError('Each reference camera needs its own finite 25-vector')
            setattr(self, name, camera)
        self._source = PairConstraint(self.xy0, self.xy1, self.plane_point,
            self.plane_normal, np.full(n, 2, np.uint8), self.camera0_template, False)

    def projected(self, states, with_jacobian=False):
        xyz = self._source._world_points(states[0])
        target = from_state(states[1], self.camera1_template)
        return project_fast(target, xyz, with_jacobian=with_jacobian)

    def residual(self, states):
        uv = self.projected(states)[0]
        return np.nan_to_num((uv-self.xy1).ravel(), nan=1e3, posinf=1e3, neginf=-1e3)

    def source_pixel_jacobians(self, states):
        """Target UV sensitivity to one native source pixel, in pixels/pixel."""
        source = from_state(states[0], self.camera0_template)
        target = from_state(states[1], self.camera1_template)

        def transfer(pixels):
            rays = fast_world_rays(source, pixels)
            denominator = np.sum(rays*self.plane_normal, axis=1)
            if np.any(np.abs(denominator) < 1e-9):
                raise ValueError('Reference ray is tangent to its identified board plane')
            distance = np.sum((self.plane_point-source[:3])*self.plane_normal, axis=1)/denominator
            if np.any(distance <= 0):
                raise ValueError('Reference board plane lies behind the source camera')
            uv, depth = project_fast(target, source[:3]+distance[:, None]*rays)
            if np.any(depth <= 0) or not np.isfinite(uv).all():
                raise ValueError('Reference point is not visible at noise linearization')
            return uv

        step = .01
        return np.stack([(transfer(self.xy0+step*np.eye(2)[axis])-
                          transfer(self.xy0-step*np.eye(2)[axis]))/(2*step)
                         for axis in range(2)], axis=-1)

    def jacobians(self, states):
        source, target = states
        # Differentiates Brown inverse and the changing plane intersection.
        # Only a small number of cross-reference pairs uses this fallback.
        source_jac = np.empty((2*len(self.xy0), 11))
        for column in range(11):
            step = 1e-6
            high, low = source.copy(), source.copy()
            high[column] += step; low[column] -= step
            source_jac[:, column] = (self.residual([high, target])-
                                     self.residual([low, target]))/(2*step)
        target_jac = self.projected(states, True)[2]
        return [source_jac, target_jac]


def load_reference_pairs(specifications, scene, cameras, reference_cameras,
                         source_hash, base_hash, input_hashes):
    """Load reviewed native pixels on a scene-existing flat printed panel."""
    records = []
    ads = {int(a['user_number']): a for a in scene['advertisements']}
    for spec in specifications:
        path = Path(spec['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if spec.get('sha256') != digest:
            raise ValueError('Reviewed reference matching packet hash changed')
        input_hashes[str(path.resolve())] = digest
        with np.load(path, allow_pickle=False) as data:
            a, b = int(data['source_pts']), int(data['target_pts'])
            if (str(data['target_source_video_sha256']) != source_hash or
                    str(data['base_scene_sha256']) != base_hash):
                raise ValueError('Reference pair source/base scene differs')
            if a not in reference_cameras or b not in cameras:
                raise ValueError('Both reference pair cameras must be optimized graph nodes')
            source_file = Path(str(data['source_image_path']))
            if hashlib.sha256(source_file.read_bytes()).hexdigest() != str(data['source_image_sha256']):
                raise ValueError('Actual reference source image changed')
            xy0 = np.asarray(data['source_uv_inliers'], float)
            xy1 = np.asarray(data['target_uv_inliers'], float)
            physical_identity = str(data['physical_identity'])
        combined = np.column_stack((xy0, xy1))
        _, unique = np.unique(combined, axis=0, return_index=True)
        unique.sort(); xy0, xy1 = xy0[unique], xy1[unique]
        if len(xy0) < 8 or np.any(combined < 0) or np.any(combined >= (1920,1080,1920,1080)):
            raise ValueError('Insufficient or non-native reference source pixels')
        number = int(spec['user_ad_number'])
        ad = ads[number]
        corners = ad['corners_xyz_scene_units']
        origin = np.asarray(corners['left_bottom'], float)
        along = np.asarray(corners['right_bottom'], float)-origin
        up = np.asarray(corners['left_top'], float)-origin
        normal = np.cross(along, up); normal /= np.linalg.norm(normal)
        physical_curve = np.asarray(ad['fitted_boundary_curves_xyz']['top_edge_xyz'], float)
        if np.max(np.abs((physical_curve-origin) @ normal)) > 1e-4:
            raise ValueError('A curved board advertisement cannot use one plane')
        sigma = float(spec.get('sigma_px', 3.))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Reference pixel sigma must be finite and positive')
        pair = ReferencePlanePair(xy0, xy1, origin, normal, reference_cameras[a], cameras[b])
        records.append((a, b, pair, sigma, {'path': str(path.resolve()),
            'sha256': digest, 'source_pts': a, 'target_pts': b, 'user_ad_number': number,
            'physical_identity': physical_identity, 'actual_pixel_pairs': len(xy0),
            'duplicates_removed': len(combined)-len(xy0), 'sigma_px': sigma,
            'plane_point_scene': origin.tolist(), 'plane_normal': normal.tolist(),
            'separate_source_target_Brown': True, 'synthetic_pixels': False}))
    return records
