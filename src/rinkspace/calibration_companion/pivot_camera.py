"""A shared virtual tripod pivot with a rotating camera-centre offset.

``TripodPivotJointProblem`` accepts the ordinary ``JointSceneProblem`` input,
with an additional configuration on each affected installation::

    "tripod_pivot": {
        "model": "rigid_offset_in_camera_coordinates",
        "initial_offset_camera_xyz": [0.0, 0.0, 1.0],
        "fit_offset_axes": [true, false, true],
        "offset_bounds_camera_xyz": [[-5.0, 0.0, -5.0],
                                     [ 5.0, 0.0,  5.0]],
        "offset_prior_camera_xyz": {  # optional, only fitted axes enter
            "mean": [0.0, 0.0, 0.0], "sigma": [2.0, 1.0, 2.0]
        },
        "zoom_axial_shift": {  # optional, one shared fitted slope
            "initial_slope_scene_units_per_log_focal": 0.0,
            "bounds": [-2.0, 2.0],
            "reference_focal_px": 4000.0
        }
    }

For these installations, the existing ``initial_center_xyz`` and
``center_bounds_xyz`` fields specify the shared *pivot* P, rather than an
optical centre. ``fit_center`` and ``center_prior`` likewise apply to P. The
per-frame optical centre is C_i = P + R_i.T @ d, where R_i is the existing
OpenCV world-to-camera rotation and d is fixed in camera coordinates. The
camera-y component of d may be fitted when explicitly requested; near-horizontal
pans couple it strongly with pivot height, so it is fixed to zero by default.
Each frame retains its own
rotation and focal length; scene geometry and other optics are unchanged.
When ``zoom_axial_shift`` is supplied, the camera-z offset additionally changes
by ``slope * log(focal_px / reference_focal_px)``. This is one shared empirical
term, not a measured entrance-pupil calibration.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .joint_scene import JointSceneProblem
from .scene_camera import Camera, Scene


def _tripod_config(installation: dict[str, Any]) -> dict[str, Any]:
    """Validate a declared camera-frame lever arm without inventing bounds."""
    iid = installation['id']
    config = copy.deepcopy(installation['tripod_pivot'])
    required = {'initial_offset_camera_xyz', 'fit_offset_axes',
                'offset_bounds_camera_xyz'}
    if not required <= config.keys():
        raise ValueError(f'{iid}: tripod_pivot requires {sorted(required)}')
    allowed = required | {'model', 'offset_prior_camera_xyz', 'zoom_axial_shift'}
    if config.keys() - allowed:
        raise ValueError(f'{iid}: unknown tripod_pivot fields {sorted(config.keys() - allowed)}')
    if config.get('model', 'rigid_offset_in_camera_coordinates') != 'rigid_offset_in_camera_coordinates':
        raise ValueError(f'{iid}: unsupported tripod_pivot model')
    initial = np.asarray(config['initial_offset_camera_xyz'], dtype=float)
    bounds = np.asarray(config['offset_bounds_camera_xyz'], dtype=float)
    axes = config['fit_offset_axes']
    if (initial.shape != (3,) or bounds.shape != (2, 3)
            or not np.isfinite(initial).all() or not np.isfinite(bounds).all()):
        raise ValueError(f'{iid}: tripod offset and bounds must be finite xyz arrays')
    if not isinstance(axes, (list, tuple)) or len(axes) != 3 or any(type(v) is not bool for v in axes):
        raise ValueError(f'{iid}: fit_offset_axes must contain three booleans')
    for axis in range(3):
        lo, hi = bounds[:, axis]
        if axes[axis]:
            if not lo < hi or not lo <= initial[axis] <= hi:
                raise ValueError(f'{iid}: fitted offset axis {axis} needs explicit valid bounds')
        elif lo != initial[axis] or hi != initial[axis]:
            raise ValueError(f'{iid}: fixed offset axis {axis} bounds must equal its initial value')
    prior = config.get('offset_prior_camera_xyz')
    if prior is not None:
        if set(prior) != {'mean', 'sigma'}:
            raise ValueError(f'{iid}: offset prior needs mean and sigma xyz arrays')
        mean = np.asarray(prior['mean'], dtype=float)
        sigma = np.asarray(prior['sigma'], dtype=float)
        if (mean.shape != (3,) or sigma.shape != (3,)
                or not np.isfinite(mean).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError(f'{iid}: offset prior needs finite mean and positive sigma')
    zoom = config.get('zoom_axial_shift')
    if zoom is not None:
        expected = {'initial_slope_scene_units_per_log_focal', 'bounds',
                    'reference_focal_px'}
        if not isinstance(zoom, dict) or set(zoom) != expected:
            raise ValueError(f'{iid}: zoom_axial_shift requires {sorted(expected)}')
        slope = float(zoom['initial_slope_scene_units_per_log_focal'])
        slope_bounds = np.asarray(zoom['bounds'], dtype=float)
        reference = float(zoom['reference_focal_px'])
        if (slope_bounds.shape != (2,) or
                not np.isfinite([slope, reference]).all() or
                not np.isfinite(slope_bounds).all() or reference <= 0 or
                not slope_bounds[0] < slope_bounds[1] or
                not slope_bounds[0] <= slope <= slope_bounds[1]):
            raise ValueError(f'{iid}: invalid zoom axial slope, bounds, or reference focal length')
    return config


class TripodPivotJointProblem(JointSceneProblem):
    """Joint scene problem with a shared pivot and orientation-dependent centre."""

    def __init__(self, scene: Scene, problem: dict[str, Any]):
        self.supporting_line_observation_ids = set(
            problem.get('supporting_line_observation_ids', []))
        if len(self.supporting_line_observation_ids) != len(
                problem.get('supporting_line_observation_ids', [])):
            raise ValueError('duplicate supporting line observation ID')
        observations_by_id = {o['id']: o for o in problem['observations']}
        for oid in self.supporting_line_observation_ids:
            if oid not in observations_by_id or observations_by_id[oid]['target']['kind'] != 'board_side':
                raise ValueError(f'{oid}: supporting line override needs a board_side observation')
        self.pivot_specs = {
            install['id']: _tripod_config(install)
            for install in problem['installations'] if 'tripod_pivot' in install
        }
        if not self.pivot_specs:
            raise ValueError('TripodPivotJointProblem requires a tripod_pivot installation')
        # JointSceneProblem calls camera() and _make_sparsity() in its own
        # constructor. Before the extra parameters exist, camera() uses the
        # declared initial offset and the sparsity extension is empty.
        self.offset_indices: dict[str, dict[int, int]] = {}
        self.zoom_slope_indices: dict[str, int] = {}
        super().__init__(scene, problem)
        self.pivot_slices = {iid: self.center_slices[iid]
                             for iid in self.pivot_specs if iid in self.center_slices}
        for iid, where in self.pivot_slices.items():
            for axis, index in enumerate(range(where.start, where.stop)):
                self.parameter_names[index] = f'installation:{iid}:pivot_{axis}'

        values = self.x0.tolist()
        lower, upper = (part.tolist() for part in self.bounds)
        for iid, config in self.pivot_specs.items():
            initial = config['initial_offset_camera_xyz']
            bounds = config['offset_bounds_camera_xyz']
            axes = config['fit_offset_axes']
            prior = config.get('offset_prior_camera_xyz')
            indices = self.offset_indices[iid] = {}
            for axis in range(3):
                if not axes[axis]:
                    continue
                index = len(values)
                indices[axis] = index
                values.append(float(initial[axis]))
                lower.append(float(bounds[0][axis]))
                upper.append(float(bounds[1][axis]))
                self.parameter_names.append(f'installation:{iid}:offset_camera_{axis}')
                if prior is not None:
                    self.parameter_priors.append((index, {
                        'mean': float(prior['mean'][axis]),
                        'sigma': float(prior['sigma'][axis]),
                    }))
            zoom = config.get('zoom_axial_shift')
            if zoom is not None:
                index = len(values)
                self.zoom_slope_indices[iid] = index
                values.append(float(zoom['initial_slope_scene_units_per_log_focal']))
                lower.append(float(zoom['bounds'][0]))
                upper.append(float(zoom['bounds'][1]))
                self.parameter_names.append(
                    f'installation:{iid}:zoom_axial_slope_scene_units_per_log_focal')
        self.x0 = np.asarray(values, dtype=float)
        self.bounds = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self._sparsity = self._make_sparsity()
        if len(self.residual(self.x0)) < len(self.x0):
            raise ValueError('fewer residuals than pivot-model unknowns')

    def offset_camera_xyz(self, x: np.ndarray, iid: str) -> np.ndarray:
        config = self.pivot_specs[iid]
        offset = np.asarray(config['initial_offset_camera_xyz'], dtype=float).copy()
        for axis, index in self.offset_indices.get(iid, {}).items():
            offset[axis] = x[index]
        return offset

    def effective_offset_camera_xyz(self, x: np.ndarray, fid: str,
                                    focal_px: float) -> np.ndarray:
        iid = self.frames[fid]['installation_id']
        offset = self.offset_camera_xyz(x, iid)
        zoom = self.pivot_specs[iid].get('zoom_axial_shift')
        if zoom is not None:
            index = self.zoom_slope_indices.get(iid)
            slope = (zoom['initial_slope_scene_units_per_log_focal']
                     if index is None else x[index])
            offset[2] += float(slope) * np.log(
                focal_px / float(zoom['reference_focal_px']))
        return offset

    def pivot_xyz(self, x: np.ndarray, iid: str) -> np.ndarray:
        installation = self.installations[iid]
        if iid in self.pivot_slices:
            return np.asarray(x[self.pivot_slices[iid]], dtype=float)
        return np.asarray(installation['initial_center_xyz'], dtype=float)

    def camera(self, x: np.ndarray, fid: str) -> Camera:
        base = super().camera(x, fid)
        iid = self.frames[fid]['installation_id']
        if iid not in self.pivot_specs:
            return base
        offset = self.effective_offset_camera_xyz(x, fid, float(base.K_px[0, 0]))
        optical_center = base.center_xyz + base.R.T @ offset
        return Camera(optical_center, base.rvec_world_to_camera,
                      base.K_px, base.distortion_opencv)

    def _error(self, x, observation, uv, geometry, cameras):
        """Measure a lane supporting line as an image line in pinhole views.

        The legacy nearest-polyline residual samples the entire rink-width
        extension. Near a camera-plane crossing this can create a spurious
        projected branch and a nearly zero minimum depth. A supporting-line
        observation intentionally has no endpoint evidence, so two well-in-
        front points define its image line without extrapolating a visible
        segment through the camera plane. Distorted views keep the sampled
        curve implementation because straight world lines can image as curves.
        """
        supporting = (observation['target']['kind'] == 'polygon_supporting_line' or
                      observation['id'] in self.supporting_line_observation_ids)
        if not supporting:
            return super()._error(x, observation, uv, geometry, cameras)
        cam = cameras[observation['frame_id']]
        if np.any(cam.distortion_opencv):
            return super()._error(x, observation, uv, geometry, cameras)
        xyz = geometry.points(observation['target'], observation['type'], self.samples)
        depth = ((xyz - cam.center_xyz) @ cam.R.T)[:, 2]
        safe = np.flatnonzero(depth > max(1e-3, .02 * float(np.max(depth))))
        if len(safe) < 2:
            return super()._error(x, observation, uv, geometry, cameras)
        a, b = safe[len(safe) // 4], safe[3 * len(safe) // 4]
        if a == b:
            a, b = safe[0], safe[-1]
        projected, selected_depth = cam.project(xyz[[a, b]])
        direction = projected[1] - projected[0]
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-8:
            return super()._error(x, observation, uv, geometry, cameras)
        normal = np.array([-direction[1], direction[0]]) / norm
        signed = (uv - projected[0]) @ normal
        delta = -signed[:, None] * normal[None, :]
        return delta, selected_depth, 1.0

    def _make_sparsity(self):
        matrix = super()._make_sparsity().tolil()
        cursor = 0
        for observation, uv in self.observations:
            if observation['use'] != 'fit':
                continue
            end = cursor + 2 * len(uv) + 2
            iid = self.frames[observation['frame_id']]['installation_id']
            for index in self.offset_indices.get(iid, {}).values():
                matrix[cursor:end, index] = 1
            if iid in self.zoom_slope_indices:
                matrix[cursor:end, self.zoom_slope_indices[iid]] = 1
            cursor = end
        return matrix.tocsr()

    def solve(self, max_nfev=None, *, verbose=0, callback=None, x0=None):
        fit = super().solve(max_nfev, verbose=verbose, callback=callback, x0=x0)
        x = np.asarray(fit['parameter_values'], dtype=float)
        for report in fit['installations']:
            iid = report['id']
            if iid not in self.pivot_specs:
                continue
            installation = self.installations[iid]
            report.pop('center_xyz', None)
            report.pop('center_bounds_xyz', None)
            report.update({
                'pivot_xyz': self.pivot_xyz(x, iid).tolist(),
                'pivot_bounds_xyz': installation.get('center_bounds_xyz'),
                'pivot_prior': installation.get('center_prior'),
                'offset_camera_xyz': self.offset_camera_xyz(x, iid).tolist(),
                'offset_bounds_camera_xyz': self.pivot_specs[iid]['offset_bounds_camera_xyz'],
                'fit_offset_axes': self.pivot_specs[iid]['fit_offset_axes'],
                'offset_prior_camera_xyz': self.pivot_specs[iid].get('offset_prior_camera_xyz'),
                'frame_optical_centers_xyz': {
                    fid: self.camera(x, fid).center_xyz.tolist()
                    for fid, frame in self.frames.items() if frame['installation_id'] == iid
                },
                'position_status': 'conditional_virtual_pivot_and_offset_NOT_measured_tripod',
            })
            zoom = self.pivot_specs[iid].get('zoom_axial_shift')
            if zoom is not None:
                report['zoom_axial_shift'] = {
                    'slope_scene_units_per_log_focal': float(x[self.zoom_slope_indices[iid]]),
                    'initial_slope_scene_units_per_log_focal': zoom['initial_slope_scene_units_per_log_focal'],
                    'bounds': zoom['bounds'],
                    'reference_focal_px': zoom['reference_focal_px'],
                    'interpretation': 'Conditional shared axial optical-centre shift versus log focal; not a measured entrance pupil.',
                }
                report['frame_effective_offset_camera_xyz'] = {
                    fid: self.effective_offset_camera_xyz(
                        x, fid, float(self.camera(x, fid).K_px[0, 0])).tolist()
                    for fid, frame in self.frames.items() if frame['installation_id'] == iid
                }
        has_zoom = any('zoom_axial_shift' in config for config in self.pivot_specs.values())
        if has_zoom:
            for frame_report in fit['frames']:
                fid = frame_report['id']
                if 'zoom_axial_shift' not in self.pivot_specs.get(frame_report['installation_id'], {}):
                    continue
                focal = float(frame_report['camera']['K_px'][0][0])
                frame_report['effective_offset_camera_xyz'] = (
                    self.effective_offset_camera_xyz(x, fid, focal).tolist())
        fit['schema_version'] = 'rinkspace.joint-scene-tripod-pivot-fit.v1'
        fit['camera_kinematics'] = {
            'model': ('shared_virtual_pivot_with_zoom_axial_shift' if has_zoom else
                      'shared_virtual_pivot_with_camera_frame_offset'),
            'equation': ('C_i = P + R_world_to_camera_i.T @ [d_x, d_y, d_z + slope*log(f_i/f_ref)]'
                         if has_zoom else
                         'C_i = P + R_world_to_camera_i.T @ d_camera'),
            'interpretation': 'Equivalent camera-centre orbit; pivot and lever arm are conditional estimates, not physical measurements.',
            'frame_optical_centers_vary_with_pose': True,
        }
        fit['supporting_line_residual'] = (
            'pinhole: image line from two front-facing world points; '
            'no unobserved endpoint or near-plane branch')
        fit['supporting_line_observation_ids'] = sorted(
            self.supporting_line_observation_ids)
        fit['limitations'] = [
            text for text in fit['limitations']
            if text != 'Shared optical centre is an approximation, not a surveyed tripod location.'
        ]
        vertical_fitted = any(config['fit_offset_axes'][1]
                              for config in self.pivot_specs.values())
        vertical_note = ('camera-y offset is strongly coupled to pivot height'
                         if vertical_fitted else 'camera-y offset is fixed to zero')
        fit['limitations'].append(
            ('One virtual pivot with a shared log-focal axial shift approximates the rig and zoom optics; '
             + vertical_note + '; the fitted slope is not a measured pupil displacement.'
             if has_zoom else
             'One virtual pivot and a rigid camera-frame offset approximate the rig; '
             + vertical_note + '; zoom-dependent pupil motion is not modelled.'))
        return fit
