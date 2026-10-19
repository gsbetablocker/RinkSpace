"""Carry the joint fit's real source landmarks into native-frame tracking.

Camera estimates are not pixel observations. This module reuses the original
ice curves and advertisement corners, with the joint candidate's shared ad
geometry frozen. It never loads validation points or manufactures observations
by projecting a fitted camera. Native measured board edges are added separately
by the tracker, so the older keyframe board-edge set is deliberately excluded.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .ice_refine import _curve_error, _world_curves
    from .projection import from_state, project_fast
    from .robust_observations import pointwise_huber
except ImportError:
    from ice_refine import _curve_error, _world_curves
    from projection import from_state, project_fast
    from robust_observations import pointwise_huber


@dataclass(frozen=True)
class CurvePixels:
    identity: str
    world: np.ndarray
    uv: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class CornerPixels:
    identities: tuple[tuple[int, str], ...]
    xyz: np.ndarray
    uv: np.ndarray
    sigma: np.ndarray


class NativeLandmarks:
    def __init__(self, curves, corners, provenance):
        self.curves: dict[int, list[CurvePixels]] = curves
        self.corners: dict[int, CornerPixels] = corners
        self.provenance = provenance

    @classmethod
    def load(cls, config_path: Path, joint_dir: Path, scene: dict,
             shot_index: int, *, source_sha256: str, model_sha256: str):
        config_path, joint_dir = Path(config_path), Path(joint_dir)
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        diagnostic_bytes = (joint_dir / 'diagnostics.json').read_bytes()
        diagnostic = json.loads(diagnostic_bytes)
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        if (diagnostic['configuration_sha256'] != config_hash
                or diagnostic['model_sha256'] != model_sha256
                or diagnostic['source_sha256'] != source_sha256):
            raise ValueError('Joint source/configuration/model provenance differs')
        items = [item for item in config['shots'] if int(item['shot_index']) == shot_index]
        if len(items) != 1:
            raise ValueError(f'Expected one joint configuration for shot {shot_index}')
        with np.load(joint_dir / f'shot{shot_index:02d}_keyframes.npz', allow_pickle=False) as data:
            if (str(data['model_sha256']) != model_sha256
                    or str(data['source_sha256']) != source_sha256):
                raise ValueError('Joint keyframes belong to a different source/model')
            stamps = set(map(int, data['pts']))
        original_hashes = {str(Path(path).resolve()): digest for path, digest
                           in diagnostic['observation_input_sha256'].items()}
        consumed = {}

        def read_original(path):
            path = Path(path).resolve()
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if original_hashes.get(str(path)) != digest:
                raise ValueError(f'Landmark bytes differ from the original joint input: {path}')
            consumed[str(path)] = digest
            return payload

        world = _world_curves(scene)
        grouped, uncertainties = {}, {}
        for spec in items[0].get('curves', []):
            allowed = set(spec['ids']) - {'board_top', 'ice_board_contact'}
            if not allowed:
                continue
            for stamp in sorted(stamps):
                path = Path(spec.get('overrides', {}).get(str(stamp),
                            str(Path(spec['directory']) / f'{stamp}.npz')))
                if not path.exists():
                    # A file used by the joint solve may not silently disappear.
                    if str(path.resolve()) in original_hashes:
                        raise FileNotFoundError(path)
                    continue
                with np.load(io.BytesIO(read_original(path)), allow_pickle=False) as data:
                    names = data['physical_id'].astype(str)
                    uv_all = np.asarray(data['observed_uv'], float)
                    sigma_all = np.asarray(data['point_sigma_px'], float) if 'point_sigma_px' in data else \
                        np.full(len(uv_all), float(config.get('marking_sigma_px', 2.)))
                    if (uv_all.shape != (len(names), 2) or sigma_all.shape != (len(names),)
                            or not np.isfinite(uv_all).all() or not np.isfinite(sigma_all).all()
                            or np.any(sigma_all <= 0)):
                        raise ValueError(f'Invalid original curve pixels/uncertainty: {path}')
                    for name in sorted(allowed & set(names)):
                        selected = names == name
                        if selected.sum() < 6:
                            continue  # Same source-group inclusion as the joint solve.
                        if name not in world:
                            raise ValueError(f'Unknown physical ice marking: {name}')
                        grouped.setdefault((stamp, name), []).append(uv_all[selected].copy())
                        uncertainties.setdefault((stamp, name), []).append(sigma_all[selected].copy())
        curves = {}
        for (stamp, name), blocks in grouped.items():
            uv, indices = np.unique(np.concatenate(blocks), axis=0, return_index=True)
            sigma = np.concatenate(uncertainties[stamp, name])[indices]
            curves.setdefault(stamp, []).append(CurvePixels(name, world[name], uv, sigma))

        unique_corners = {}
        for filename in config.get('ad_corner_observations', []):
            manifest = json.loads(read_original(filename))
            if manifest.get('source_sha256') != source_sha256:
                raise ValueError('Advertisement pixels belong to a different source video')
            if manifest.get('model_sha256', model_sha256) != model_sha256:
                raise ValueError('Advertisement identity belongs to a different rink model')
            for row in manifest['observations']:
                if row.get('status') != 'measured' or int(row['pts']) not in stamps:
                    continue
                key = (int(row['pts']), int(row['user_number']), row['role'])
                unique_corners[key] = row  # Same explicit override order as joint solve.
        by_stamp = {}
        geometry = diagnostic['ad_geometry_candidates']
        for (stamp, number, role), row in sorted(unique_corners.items()):
            uv = np.asarray(row['uv'], float)
            xyz = np.asarray(geometry[str(number)]['corners_xyz_scene_units'][role], float)
            sigma = float(row.get('sigma_px', 3.))
            if (uv.shape != (2,) or xyz.shape != (3,) or not np.isfinite(uv).all()
                    or not np.isfinite(xyz).all() or np.any(uv < 0) or np.any(uv >= [1920, 1080])
                    or not np.isfinite(sigma) or sigma <= 0):
                raise ValueError(f'Invalid measured advertisement corner: {(stamp, number, role)}')
            by_stamp.setdefault(stamp, []).append(((number, role), xyz, uv, sigma))
        corners = {stamp: CornerPixels(tuple(row[0] for row in rows),
                                       np.stack([row[1] for row in rows]),
                                       np.stack([row[2] for row in rows]),
                                       np.asarray([row[3] for row in rows]))
                   for stamp, rows in by_stamp.items()}
        return cls(curves, corners, {
            'configuration_sha256': config_hash,
            'joint_diagnostics_sha256': hashlib.sha256(diagnostic_bytes).hexdigest(),
            'source_sha256': source_sha256, 'model_sha256': model_sha256,
            'shot_index': shot_index, 'observation_input_sha256': consumed,
            'advertisement_geometry': 'frozen_joint_candidate_scene_units_not_new_measurements',
            'source_pixels': 'original_joint_training_pixels_reused_not_independent_validation',
        })

    def has_observations(self, pts: int) -> bool:
        return bool(self.curves.get(pts)) or pts in self.corners

    def metadata(self) -> dict:
        return {**self.provenance,
                'ice_frames': len(self.curves),
                'ice_points': sum(len(group.uv) for groups in self.curves.values() for group in groups),
                'ad_frames': len(self.corners),
                'ad_corners': sum(len(group.uv) for group in self.corners.values())}

    def add_factors(self, graph, pts: int, template: np.ndarray,
                    fixed_mount: bool = True) -> None:
        template = np.asarray(template, float).copy()

        def camera(state):
            result = from_state(state, template)
            if fixed_mount:
                result[:3] = template[:3]
            return result

        for group in self.curves.get(pts, []):
            def residual(q, group=group):
                return _curve_error(camera(q[0]), group.world, group.uv) / group.sigma

            def jacobian(q, group=group):
                derivative = _curve_error(camera(q[0]), group.world, group.uv, True)[1]
                if fixed_mount:
                    derivative[:, :3] = 0.
                return [derivative / group.sigma[:, None]]

            robust, derivative = pointwise_huber(residual, jacobian, group_size=1, delta=2.)
            graph.add_observation([pts], robust, residual_dim=len(group.uv), sigma=1., jac_fn=derivative)
        group = self.corners.get(pts)
        if group is not None:
            def residual(q, group=group):
                uv, _ = project_fast(camera(q[0]), group.xyz)
                return ((uv - group.uv) / group.sigma[:, None]).ravel()

            def jacobian(q, group=group):
                derivative = project_fast(camera(q[0]), group.xyz, with_jacobian=True)[2]
                if fixed_mount:
                    derivative[:, :3] = 0.
                return [derivative / np.repeat(group.sigma, 2)[:, None]]

            robust, derivative = pointwise_huber(residual, jacobian, group_size=2, delta=3.)
            graph.add_observation([pts], robust, residual_dim=2 * len(group.uv), sigma=1., jac_fn=derivative)
