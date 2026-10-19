"""Append explicit real native-image ice curves to the preserved full source.

New source pixels are separate from old row inventories and geometry priors.
Held curves are reviewed, never silently fitted. World curves use the current
named marking family and the same installation camera, without full-scene
reconstruction for every small derivative column.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from complete_ice_template_candidate.complete_ice_numeric_geometry import _boundary_hit
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _native_frame_reference(row):
    """Normalize explicit provenance aliases on the derived row only."""
    paths = [row[k] for k in ('source_raw_frame', 'source_path') if k in row]
    hashes = [row[k] for k in ('source_file_sha256', 'source_frame_sha256') if k in row]
    if not paths or not hashes:
        raise ValueError('Native source frame path and SHA256 required')
    resolved = [Path(p).resolve() for p in paths]
    if any(p != resolved[0] for p in resolved[1:]) or any(h != hashes[0] for h in hashes[1:]):
        raise ValueError('Native source frame provenance aliases disagree')
    row['source_raw_frame'] = str(resolved[0])
    row['source_file_sha256'] = hashes[0]
    return resolved[0], hashes[0]


def _axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c, s, 0.]), np.array([-s, c, 0.])


class NativeIceCurveFactor:
    def __init__(self, state, packet_path, parameter_count):
        self.state = state
        self.parameter_count = int(parameter_count)
        self.packet_path = Path(packet_path).resolve()
        packet = json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema') != 'rinkspace.additional-native-ice-curves.v1'
                or packet.get('source_video_sha256') != SOURCE_SHA
                or packet.get('time_base') != '1/15360'):
            raise ValueError('Explicit same-video native-image ice-curve packet required')
        self.geometry = state.geometry.current
        self.unit = float(self.geometry.scene_unit_m)
        self.world_indices = WORLD_INDICES.copy()
        self.rows, self.held = [], []
        ids = set()
        for record in packet['rows']:
            row = deepcopy(record)
            if row['id'] in ids:
                raise ValueError('Additional observation IDs must be unique')
            ids.add(row['id'])
            native = int(row['pts'])
            if native not in state.original.aliases:
                raise ValueError('Source PTS has no authenticated exposure alias')
            actual = int(state.original.aliases[native])
            if 'actual_pts' in row and int(row['actual_pts']) != actual:
                raise ValueError('Declared additional observation exposure disagrees with authenticated alias')
            if actual not in state.bypts or not 229114112 <= native < 231705600:
                raise ValueError('Additional observation lies outside the authorized programme')
            frame_path, frame_sha = _native_frame_reference(row)
            if _sha(frame_path) != frame_sha:
                raise ValueError('Native source frame hash changed')
            uv = np.asarray(row['uv'], float)
            sigma = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(uv),)).copy()
            if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 2
                    or not np.isfinite(uv).all() or not np.isfinite(sigma).all()
                    or np.any(sigma <= 0) or np.any(uv < 0)
                    or np.any(uv >= np.array([1920., 1080.]))):
                raise ValueError('Finite original 1920x1080 source UV and positive sigma required')
            row.update(actual_pts=actual, uv=uv, sigma=sigma)
            self._validate_curve(row['curve'])
            if row['split'] == 'fit':
                self.rows.append(row)
            elif row['split'] == 'holdout':
                self.held.append(row)
            else:
                raise ValueError('Explicit new fit/holdout admission required')
        if not self.rows:
            raise ValueError('No real additional fit observations')
        # A dense line/ring trace is one correlated group. Raw per-point sigma
        # is preserved, while group objective normalization is explicit.
        self.scales = np.concatenate([np.repeat(r['sigma'], 2) * np.sqrt(len(r['uv']))
                                      for r in self.rows])
        self.quadratic = np.ones(len(self.scales), bool)
        self.row_count = len(self.scales)
        self.blocks = {r['actual_pts']: state.bypts[r['actual_pts']]
                       for r in self.rows + self.held}
        camera_columns = []
        self.camera_steps = {}
        for actual, block in self.blocks.items():
            slots = np.r_[block.xyz_slots, block.optics_chart.indices,
                          block.optics_chart.mount_indices,
                          block.start + 4 * block.bundle.index[actual] + np.arange(4)].astype(int)
            camera_columns.extend(slots)
            for col in slots:
                self.camera_steps[int(col)] = 1e-6
            for col in block.optics_chart.indices:
                self.camera_steps[int(col)] = 1e-4
            view = block.start + 4 * block.bundle.index[actual]
            for col in range(view, view + 3):
                self.camera_steps[col] = 1e-7
        self.camera_columns = np.unique(camera_columns)
        if np.intersect1d(self.camera_columns, self.world_indices).size:
            raise ValueError('Camera and named marking columns overlap')
        self._cache = OrderedDict()
        self.contract = dict(
            schema='rinkspace.additional-real-native-ice-curve-factor.v1',
            packet_path=str(self.packet_path), packet_sha256=_sha(self.packet_path),
            source_video_sha256=SOURCE_SHA, time_base='1/15360',
            new_fit_observation_ids=[r['id'] for r in self.rows],
            held_observation_ids=[r['id'] for r in self.held],
            new_scalar_rows=self.row_count, parameter_count=self.parameter_count,
            original_source_rows_UV_sigma_roles_profiles_and_clock_unchanged=True,
            new_target_objective_explicit=True,
            observation_operator='native full-Brown vector distance to current world curve',
            trace_objective='per correlated curve mean squared normalized native-pixel distance',
            raw_sigma_px_preserved=True,
            sigma_is_declared_working_weight_not_new_measured_uncertainty=True,
            no_plan_pixels_or_old_WORLD_as_image_observations=True,
            carrier_support_is_not_observed_paint_endpoint=True,
            held_rows_used_in_fit=False, accuracy_accepted=False)

    @staticmethod
    def _validate_curve(curve):
        kind = curve.get('kind')
        if kind in ('curling_centreline', 'house_ring'):
            if curve.get('row') not in (1, 2, 3, 4):
                raise ValueError('Explicit shared curling row required')
            if kind == 'house_ring' and (curve.get('side') not in ('F', 'R')
                    or curve.get('ring') not in ('outer', 'middle', 'inner', 'center')):
                raise ValueError('Explicit actual house and ring identity required')
        elif kind == 'hockey_circle':
            if curve.get('feature_id') not in ('C0', 'A1', 'A2', 'A3', 'A4'):
                raise ValueError('Explicit shared hockey-circle identity required')
        else:
            raise ValueError('Unsupported native ice-curve identity')

    def _world(self, q, curve):
        p = np.asarray(q, float)[self.world_indices]
        if curve['kind'] in ('curling_centreline', 'house_ring'):
            axis, across = _axes(p[14])
            center = np.r_[p[12:14], 0.] + across * ((curve['row'] - 2.5) * p[16])
            if curve['kind'] == 'curling_centreline':
                # The current ice-footprint intersections bound numerical
                # carrier coverage. They are not claimed paint endpoints,
                # sheet-length constraints or regulation hack/back dimensions.
                course, cq, _ = self.geometry.print_course_binding(p)
                bottom = course.sample_perimeter(cq, with_jacobian=False)[0]
                first = _boundary_hit(center / self.unit, -axis, np.asarray(bottom))
                last = _boundary_hit(center / self.unit, axis, np.asarray(bottom))
                return np.linspace(first, last, 1025)
            center += axis * p[15] / 2. * (-1. if curve['side'] == 'F' else 1.)
            ratio = {'outer': 1., 'middle': p[18], 'inner': p[19], 'center': p[20]}[curve['ring']]
            radius = p[17] * ratio
        else:
            axis, across = _axes(p[2])
            center = np.r_[p[:2], 0.]
            identity = curve['feature_id']
            if identity != 'C0':
                course, cq, _ = self.geometry.print_course_binding(p)
                bottom = course.sample_perimeter(cq, with_jacobian=False)[0]
                side = 'right' if identity in ('A1', 'A2') else 'left'
                inward = -axis if side == 'right' else axis
                wall = _boundary_hit(center / self.unit, -inward, np.asarray(bottom)) * self.unit
                goal = wall + inward * p[8 if side == 'right' else 7]
                sign = -1. if identity in ('A1', 'A4') else 1.
                center = goal + inward * p[4] + across * sign * p[5]
            radius = p[3]
        angle = np.linspace(0., 2. * np.pi, 361)
        return (center + radius * np.c_[np.cos(angle), np.sin(angle), np.zeros(len(angle))]) / self.unit

    def _cameras(self, q):
        core = np.asarray(q, float)[:CORE_COUNT]
        return {actual: block.bundle.camera(block.local(core), actual)
                for actual, block in self.blocks.items()}

    def _residual(self, q, rows, *, cameras=None):
        if cameras is None:
            cameras = self._cameras(q)
        values = []
        for row in rows:
            raw = curve_vector_error_full_brown(cameras[row['actual_pts']],
                self._world(q, row['curve']), row['uv'], with_jacobian=False)
            # The original routine returns residual directly for value-only.
            if isinstance(raw, tuple):
                raw = raw[0]
            values.append(np.asarray(raw).reshape(-1) /
                (np.repeat(row['sigma'], 2) * np.sqrt(len(row['uv']))))
        residual = np.concatenate(values) if values else np.empty(0)
        if not np.isfinite(residual).all():
            raise ValueError('Additional source leaves the valid optical domain')
        return residual

    def block(self, q, jacobian=True, *, original_start=0):
        q = np.asarray(q, float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Explicit full current physical state required')
        active_world = np.unique(np.concatenate([
            self.world_indices[[12, 13, 14, 15, 16, 17, 18, 19, 20, 44, 45, 46, 47, 48, 49]]
                if r['curve']['kind'] != 'hockey_circle'
                else self.world_indices[[0, 1, 2, 3, 4, 5, 7, 8, 44, 45, 46, 47, 48, 49]]
            for r in self.rows]))
        fit_cameras = np.unique(np.concatenate([
            np.r_[self.blocks[r['actual_pts']].xyz_slots,
                  self.blocks[r['actual_pts']].optics_chart.indices,
                  self.blocks[r['actual_pts']].optics_chart.mount_indices,
                  self.blocks[r['actual_pts']].start +
                    4 * self.blocks[r['actual_pts']].bundle.index[r['actual_pts']] + np.arange(4)]
            for r in self.rows])).astype(int)
        active = np.unique(np.r_[active_world, fit_cameras])
        key = q[active].tobytes()
        cached = self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):
            residual, matrix = cached
        else:
            cameras = self._cameras(q)
            residual = self._residual(q, self.rows, cameras=cameras)
            matrix = None
            if jacobian:
                compact = np.zeros((len(residual), len(active)))
                try:
                    for index, col in enumerate(active):
                        step = self.camera_steps.get(int(col), 1e-6)
                        plus, minus = q.copy(), q.copy()
                        plus[col] += step
                        minus[col] -= step
                        old_cameras = cameras if col in active_world else None
                        compact[:, index] = (self._residual(plus, self.rows, cameras=old_cameras)
                            - self._residual(minus, self.rows, cameras=old_cameras)) / (2. * step)
                finally:
                    self._cameras(q)
                rr, cc = np.nonzero(compact)
                matrix = csr_matrix((compact[rr, cc], (rr, active[cc])),
                    shape=(len(residual), self.parameter_count))
            self._cache[key] = (residual.copy(), matrix)
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        ledger = dict(family='additional_real_native_ice_curves',
            original_start=int(original_start), original_stop=int(original_start) + self.row_count,
            packet_sha256=self.contract['packet_sha256'],
            native_pts=[r['pts'] for r in self.rows],
            actual_pts=[r['actual_pts'] for r in self.rows],
            feature_ids=[r['feature_id'] for r in self.rows],
            fit_observation_ids=[r['id'] for r in self.rows],
            held_observation_ids=[r['id'] for r in self.held],
            plan_pixels_used=False, paint_endpoints_inferred=False)
        return SourceRowBlock(residual, matrix, self.quadratic, self.scales, ledger)

    def review(self, q):
        cameras = self._cameras(q)
        result = []
        for row in self.rows + self.held:
            normalized = self._residual(q, [row], cameras=cameras)
            error = normalized.reshape(-1, 2) * row['sigma'][:, None] * np.sqrt(len(row['uv']))
            distance = np.linalg.norm(error, axis=1)
            result.append(dict(id=row['id'], pts=row['pts'], actual_pts=row['actual_pts'],
                split=row['split'], feature_id=row['feature_id'], source_uv_count=len(distance),
                mean_native_px=float(distance.mean()), p90_native_px=float(np.percentile(distance, 90)),
                max_native_px=float(distance.max()), accuracy_accepted=False))
        return dict(contract=self.contract, rows=result, accuracy_accepted=False)


class SourceWithAdditionalNativeIceCurves:
    def __init__(self, base, factor):
        self.base, self.native_ice_factor = base, factor
        self.count, self.updater = int(base.count), base.updater
        self.quadratic = np.r_[base.quadratic, factor.quadratic]
        self.scales = np.r_[base.scales, factor.scales]
        self.contract = dict(schema='rinkspace.complete-source-plus-real-native-ice-curves.v1',
            original_source_contract=deepcopy(base.contract),
            additional_real_ice_curve_contract=deepcopy(factor.contract),
            original_source_rows=len(base.scales), additional_scalar_rows=factor.row_count,
            total_rows=len(self.scales), original_blocks_and_profiles_preserved=True,
            accuracy_accepted=False)

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def guard(self, q):
        self.base.guard(q)
        self.native_ice_factor.block(q, False)

    def blocks(self, q, jacobian=True):
        yield from self.base.blocks(q, jacobian)
        yield self.native_ice_factor.block(q, jacobian, original_start=len(self.base.scales))


def wrap_additional_native_ice_curves(base, state, packet_path):
    return SourceWithAdditionalNativeIceCurves(base,
        NativeIceCurveFactor(state, packet_path, base.count))
