"""Frozen source-image quality for graph factors; never inferred from fit error."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


class ObservationQuality:
    def __init__(self, specification, source_hash, input_hashes):
        self.manifest = None
        self.groups = {}
        self.by_frame = {}
        self.provenance = None
        if specification:
            from image_quality import load_quality_manifest
            path = Path(specification['path'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != specification['sha256']:
                raise ValueError('Source quality manifest hash changed')
            self.manifest = load_quality_manifest(path, source_sha256=source_hash)
            input_hashes[str(path.resolve())] = digest
            self.provenance = {'path': str(path.resolve()), 'sha256': digest,
                'source_sha256': source_hash,
                'policy': 'source_pixels_only; inflate_existing_sigma; no_residual_feedback',
                'unknown_is_not_clear': True,
                'pair_covariance': 'D_quality * existing_covariance * D_quality in the linear branch',
                'robust_application_order': 'quality scale, pointwise Huber, existing correlated whitener',
                'acceptance': 'quality_is_not_localization_validation'}

    def _record(self, kind, multiplier):
        values = np.asarray(multiplier, float).reshape(-1)
        if not np.isfinite(values).all() or np.any(values < 1.):
            raise ValueError('Image quality may only inflate source uncertainty')
        row = self.groups.setdefault(kind, {'observations': 0, 'inflated': 0,
                                           'sum_multiplier': 0., 'max_multiplier': 1.})
        row['observations'] += len(values)
        row['inflated'] += int(np.count_nonzero(values > 1.001))
        row['sum_multiplier'] += float(values.sum())
        row['max_multiplier'] = max(row['max_multiplier'], float(values.max(initial=1.)))
        return values

    def require_pts(self, pts):
        if self.manifest is not None:
            missing = set(map(int, pts))-set(map(int, self.manifest.pts))
            if missing:
                raise ValueError(f'Image quality missing graph source PTS: {sorted(missing)[:8]}')

    def scales(self, pts, uv, kind):
        uv = np.asarray(uv, float).reshape(-1, 2)
        if self.manifest is None:
            return np.ones(len(uv))
        values = np.broadcast_to(self.manifest.sigma_multiplier(int(pts), uv), (len(uv),))
        self._frame_record(pts, uv, values, is_motion=False)
        return self._record(kind, values)

    def _frame_record(self, pts, uv, values, *, is_motion):
        states = self.manifest.status(int(pts), np.asarray(uv))
        row = self.by_frame.setdefault(int(pts), {'direct_count': 0, 'motion_count': 0,
            'soft_count': 0, 'unknown_count': 0, 'max_multiplier': 1.})
        row['motion_count' if is_motion else 'direct_count'] += len(values)
        row['soft_count'] += int(np.count_nonzero(states == 'soft'))
        row['unknown_count'] += int(np.count_nonzero(states == 'unknown'))
        row['max_multiplier'] = max(row['max_multiplier'], float(np.max(values, initial=1.)))

    def pair_scales(self, a, b, xy0, xy1, kind):
        if self.manifest is None:
            return np.ones(len(xy0))
        values = self.manifest.pair_sigma_multiplier(int(a), int(b), xy0, xy1)
        self._frame_record(a, xy0, self.manifest.sigma_multiplier(int(a), xy0), is_motion=True)
        self._frame_record(b, xy1, self.manifest.sigma_multiplier(int(b), xy1), is_motion=True)
        return self._record(kind, np.broadcast_to(values, (len(xy0),)))

    @staticmethod
    def wrap(residual, jacobian, multipliers, group_size=2):
        scale = np.repeat(np.asarray(multipliers, float), group_size)
        if not np.isfinite(scale).all() or np.any(scale < 1.):
            raise ValueError('Invalid source uncertainty multiplier')
        if np.all(scale == 1.):
            return residual, jacobian
        def value(states):
            return residual(states)/scale
        def derivative(states):
            return [matrix/scale[:, None] for matrix in jacobian(states)]
        return value, derivative

    def frame_fields(self, pts):
        if self.manifest is None:
            return {}
        local = [self.by_frame.get(int(p), {}) for p in pts]
        return {
            'image_quality_sigma_multiplier': np.asarray([
                self.manifest.sigma_multiplier(int(p)) for p in pts], np.float32),
            'image_quality_status': np.asarray([self.manifest.status(int(p)) for p in pts]),
            'image_quality_manifest_sha256': np.asarray(self.provenance['sha256']),
            'local_quality_max_sigma_multiplier': np.asarray([
                row.get('max_multiplier', 1.) for row in local], np.float32),
            'direct_source_factor_pixel_count': np.asarray([
                row.get('direct_count', 0) for row in local], np.int32),
            'motion_source_factor_pixel_count': np.asarray([
                row.get('motion_count', 0) for row in local], np.int32),
            'local_quality_soft_pixel_count': np.asarray([
                row.get('soft_count', 0) for row in local], np.int32),
            'local_quality_unknown_pixel_count': np.asarray([
                row.get('unknown_count', 0) for row in local], np.int32),
            'source_factor_counts_semantics': np.asarray(
                'factor_pixel_uses_not_independent_observations_or_accuracy'),
            'localization_acceptance': np.asarray('unvalidated_candidate'),
        }

    def report(self):
        if self.manifest is None:
            return None
        return {**self.provenance, 'factor_groups': {name: {
            **row, 'mean_multiplier': row['sum_multiplier']/max(1, row['observations'])}
            for name, row in self.groups.items()}}
