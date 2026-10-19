"""Source-PTS camera tracking within reviewed shots using GTSAM fixed-lag iSAM2.

Accepted model-bound anchors establish absolute rink alignment. Fixed-mount
ray factors transfer static image evidence without assigning a false ice depth.
Optional surface factors require a verified ice/board identity. No motion
factor crosses a source cut; local flow quality is not absolute fit acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .runtime.bootstrap import add_geometry_dependencies
except ImportError:
    from runtime.bootstrap import add_geometry_dependencies

add_geometry_dependencies()

import cv2
import numpy as np

try:
    from .backend import IncrementalCameraGraph
    from .projection import (from_state, project_central_valid, project_fast,
                             require_central_projection, to_state)
    from .reference_board_pair import BoardIntersectionError
    from .robust_observations import pointwise_huber
except ImportError:
    from backend import IncrementalCameraGraph
    from projection import (from_state, project_central_valid, project_fast,
                            require_central_projection, to_state)
    from reference_board_pair import BoardIntersectionError
    from robust_observations import pointwise_huber


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
ACCEPTED_ANCHOR_STATUSES = frozenset({"accepted", "camera_anchor", "camera_accepted", "validated"})
BETWEEN_SIGMA = np.array([0.00025] * 3 + [0.016] * 3 + [0.025, 0.025, 0.006, 0.006, 0.005])
ANCHOR_SIGMA = np.array([0.006] * 3 + [0.04] * 3 + [0.12, 0.12, 0.10, 0.10, 0.06])
CENTER_SIGMA_SCENE = 0.5
FOCAL_REFERENCE_MAX_GAP_PTS = 15360  # Do not interpolate through a >1 s keyframe gap.
FOCAL_INITIAL_MAX_DEVIATION = 0.18
FOCAL_MOTION_SIGMA_LOG = 0.008


def _camera_vector(vector: object) -> np.ndarray:
    camera = np.asarray(vector, np.float64).reshape(-1)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("A camera anchor must provide 25 finite values")
    if np.any(np.abs(camera[23:25]) > 1e-12):
        raise ValueError("Brown sensor tilt must stay fixed to zero")
    return camera.copy()


def fast_world_rays(camera: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Invert fixed Brown14 optics with OpenCV, preserving image skew."""
    c = _camera_vector(camera)
    xy = np.ascontiguousarray(np.asarray(pixels, np.float64).reshape(-1, 2))
    if len(xy) == 0:
        return np.empty((0, 3), np.float64)
    fx, fy = np.exp(c[6:8])
    unskewed = xy.copy()
    unskewed[:, 0] -= c[10] * (xy[:, 1] - c[9]) / fy
    if np.max(np.abs(c[11:23])) <= 1e-14:
        undistorted = np.column_stack(((unskewed[:, 0] - c[8]) / fx,
                                       (unskewed[:, 1] - c[9]) / fy))
    else:
        k = np.array([[fx, 0., c[8]], [0., fy, c[9]], [0., 0., 1.]])
        undistorted = cv2.undistortPoints(
            unskewed.reshape(-1, 1, 2), k, np.ascontiguousarray(c[11:25])
        ).reshape(-1, 2)
    rays_camera = np.column_stack((undistorted, np.ones(len(xy))))
    rotation = cv2.Rodrigues(np.ascontiguousarray(c[3:6]))[0]
    return np.ascontiguousarray(rays_camera @ rotation)


@dataclass(frozen=True)
class SurfacePlanes:
    polygon_xy: np.ndarray
    segment_starts: np.ndarray
    segment_edges: np.ndarray
    board_top_z: float

    @classmethod
    def from_scene(cls, scene: dict) -> "SurfacePlanes":
        border = np.asarray(scene["ice_boundary"]["points_xyz"], np.float64)[:, :2]
        top = np.asarray(scene["board_surface"]["top_edge_xyz"], np.float64)
        return cls(
            np.ascontiguousarray(border.astype(np.float32)),
            border,
            np.roll(border, -1, axis=0) - border,
            float(top[:, 2].max()),
        )


@dataclass(frozen=True)
class SourceEdges:
    """Measured original-frame board edges, never projected pseudo-observations."""

    pts: np.ndarray
    offsets: np.ndarray
    observed_uv: np.ndarray
    edge_id: np.ndarray
    source_sha256: str
    model_sha256: str

    @classmethod
    def load(cls, path: Path, expected_pts: np.ndarray, *,
             source_sha256: str, model_sha256: str) -> "SourceEdges":
        with np.load(path, allow_pickle=False) as data:
            required = {"pts", "offsets", "observed_uv", "edge_id", "edge_names",
                        "source_sha256", "model_sha256"}
            if not required.issubset(data.files):
                raise ValueError(f"Incomplete measured source-edge cache: {path}")
            pts = np.asarray(data["pts"], np.int64).copy()
            offsets = np.asarray(data["offsets"], np.int64).copy()
            observed_uv = np.asarray(data["observed_uv"], np.float64).copy()
            edge_id = np.asarray(data["edge_id"], np.int8).copy()
            names = tuple(map(str, data["edge_names"]))
            source = str(data["source_sha256"])
            model = str(data["model_sha256"])
        if not np.array_equal(pts, expected_pts):
            raise ValueError(f"Source edges must cover every exact shot PTS: {path}")
        if source != source_sha256 or model != model_sha256:
            raise ValueError(f"Measured edges belong to a different source or rink model: {path}")
        if names != ("board_top", "ice_board_contact"):
            raise ValueError(f"Unexpected physical edge identities in {path}: {names}")
        if (offsets.shape != (len(pts) + 1,) or offsets[0] != 0
                or offsets[-1] != len(observed_uv) or np.any(np.diff(offsets) < 0)
                or observed_uv.shape != (len(edge_id), 2)
                or not np.isfinite(observed_uv).all()
                or np.any((edge_id < 0) | (edge_id > 1))):
            raise ValueError(f"Malformed measured source-edge offsets or pixels: {path}")
        return cls(pts, offsets, observed_uv, edge_id, source, model)

    def frame(self, index: int, edge_id: int) -> np.ndarray:
        start, end = map(int, self.offsets[index:index + 2])
        return self.observed_uv[start:end][self.edge_id[start:end] == edge_id]


def _closed_edge_ring(world: np.ndarray) -> np.ndarray:
    """The board perimeter in scene.json is a loop with an implicit last edge."""
    world = np.asarray(world, np.float64)
    if np.allclose(world[0], world[-1], rtol=0., atol=1e-9):
        return world
    return np.vstack((world, world[:1]))


def _edge_segment_candidates(camera: np.ndarray, world: np.ndarray,
                             observed: np.ndarray) -> np.ndarray:
    """Restrict curve search to projected segments near actual source pixels.

    This trims world geometry, never image observations or high-residual points.
    A generous pixel margin accommodates the imperfect starting camera.
    """
    projected, depth = project_fast(camera, world)
    start, finish = projected[:-1], projected[1:]
    low = observed.min(axis=0) - 350.
    high = observed.max(axis=0) + 350.
    crossing = (np.minimum(start, finish) <= high).all(axis=1) & (
        np.maximum(start, finish) >= low).all(axis=1)
    valid = np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
    valid &= (depth[:-1] > 0) & (depth[1:] > 0)
    selected = np.flatnonzero(crossing & valid)

    def central_segments(indices):
        if not len(indices):
            return indices
        # Endpoint and midpoint samples reject ghost chords; this finite
        # check does not certify every point on a continuous world segment.
        samples = np.concatenate((world[indices], world[indices + 1],
                                  .5 * (world[indices] + world[indices + 1])))
        central = project_central_valid(camera, samples)[-1].reshape(3, len(indices))
        return indices[np.all(central, axis=0)]

    trimmed = central_segments(selected)
    if len(trimmed):
        return trimmed
    # The pixel margin is only a speed hint.  On a distant trial camera,
    # valid world segments may all lie outside it; search the whole ring
    # before declaring an optical-domain failure.
    return central_segments(np.flatnonzero(valid))


try:
    from .acceleration import nearest_segments
except ImportError:
    from acceleration import nearest_segments


def _edge_curve_error(camera: np.ndarray, world: np.ndarray, observed: np.ndarray,
                      segment_index: np.ndarray, with_jacobian: bool = False,
                      *, finite_vector: bool = False):
    """Signed source-pixel to projected-curve distance with analytic camera Jacobian.

    Same contour residual and Jacobian as ice_refine._curve_error, using only
    the potentially visible physical segments for fixed-lag throughput.
    """
    if not len(segment_index):
        raise BoardIntersectionError('source_edge_no_central_segment', np.arange(len(observed)))
    point_index = np.unique(np.r_[segment_index, segment_index + 1])
    first_index = np.searchsorted(point_index, segment_index)
    second_index = np.searchsorted(point_index, segment_index + 1)
    if with_jacobian:
        projected, depth, raw_j, endpoint_central = project_central_valid(
            camera, world[point_index], with_jacobian=True)
        jac = raw_j.reshape(len(point_index), 2, 11)
    else:
        projected, depth, endpoint_central = project_central_valid(camera, world[point_index])
    start = projected[first_index]
    finish = projected[second_index]
    d = finish - start
    length2 = np.sum(d*d, axis=1)
    midpoint_central = project_central_valid(
        camera, .5 * (world[segment_index] + world[segment_index + 1]))[-1]
    valid = (endpoint_central[first_index] & endpoint_central[second_index]
             & midpoint_central
             & np.isfinite(start).all(axis=1) & np.isfinite(finish).all(axis=1)
             & (depth[first_index] > 0) & (depth[second_index] > 0)
             & (length2 > 1e-4))
    if not valid.any():
        raise BoardIntersectionError('source_edge_no_central_segment', np.arange(len(observed)))
    index, _, selected_t, nearest = nearest_segments(
        observed, start, d, length2, valid, compiled=False)
    a = start[index]
    direction = d[index]
    if finite_vector:
        residual = (nearest - observed).ravel()
        if not with_jacobian:
            return residual
        j0 = jac[first_index[index]]
        jd = jac[second_index[index]] - j0
        offset = observed-a
        length2_selected = length2[index]
        raw_t = np.sum(offset*direction, axis=1)/length2_selected
        numerator_j = (-np.einsum('nij,ni->nj', j0, direction)
                       + np.einsum('ni,nij->nj', offset, jd))
        length2_j = 2.*np.einsum('ni,nij->nj', direction, jd)
        t_j = (numerator_j-raw_t[:, None]*length2_j)/length2_selected[:, None]
        t_j[(raw_t <= 0.) | (raw_t >= 1.)] = 0.
        derivative = (j0+selected_t[:, None, None]*jd
                      + direction[:, :, None]*t_j[:, None, :])
        return residual, derivative.reshape(-1, 11)
    norm = np.sqrt(length2[index])
    u = a-observed
    cross = direction[:, 0]*u[:, 1] - direction[:, 1]*u[:, 0]
    residual = cross/norm
    if not with_jacobian:
        return residual
    j0 = jac[first_index[index]]
    jd = jac[second_index[index]] - j0
    dc = (jd[:, 0]*u[:, 1, None] - jd[:, 1]*u[:, 0, None]
          + direction[:, 0, None]*j0[:, 1] - direction[:, 1, None]*j0[:, 0])
    dn = np.sum(direction[:, :, None]*jd, axis=1)/norm[:, None]
    derivative = dc/norm[:, None] - cross[:, None]*dn/norm[:, None]**2
    return residual, derivative


def _add_source_edge_factors(graph: IncrementalCameraGraph, stamp: int,
                             index: int, template: np.ndarray,
                             initial_camera: np.ndarray, edges: SourceEdges,
                             world: dict[str, np.ndarray], *,
                             fixed_mount: bool, finite_vector: bool = False,
                             dynamic_candidates: bool = False,
                             source_quality=None) -> None:
    for edge_id, (name, sigma) in enumerate((("board_top", 2.),
                                             ("ice_board_contact", 3.))):
        observed = edges.frame(index, edge_id)
        if not len(observed):
            continue
        xyz = world[name]
        search_camera = initial_camera.copy()
        if fixed_mount:
            search_camera[:3] = template[:3]
        segments = _edge_segment_candidates(search_camera, xyz, observed)
        cache: dict[str, np.ndarray] = {}

        def evaluate_raw(q, with_jacobian=False, xyz=xyz, observed=observed,
                         segments=segments, cache=cache):
            state = np.asarray(q[0], np.float64)
            same = "state" in cache and np.array_equal(cache["state"], state)
            if not same:
                cache.clear()
                cache["state"] = state.copy()
            camera = from_state(state, template)
            if fixed_mount:
                camera[:3] = template[:3]
            if dynamic_candidates and not same:
                # Initializer-based trimming can become invalid after a warm
                # calibration update. Never turn that artificial trim into a
                # physical contour endpoint at a different camera state.
                cache['segments'] = _edge_segment_candidates(camera, xyz, observed)
            current_segments = cache.get('segments', segments)
            if with_jacobian and "jacobian" not in cache:
                cache["residual"], cache["jacobian"] = _edge_curve_error(
                    camera, xyz, observed, current_segments, True, finite_vector=finite_vector)
                if fixed_mount:
                    cache["jacobian"][:, :3] = 0.
            elif "residual" not in cache:
                cache["residual"] = _edge_curve_error(
                    camera, xyz, observed, current_segments, finite_vector=finite_vector)
            return cache["residual"], cache.get("jacobian")

        def residual(q, evaluate_raw=evaluate_raw):
            return evaluate_raw(q)[0]

        def jacobians(q, evaluate_raw=evaluate_raw):
            return [evaluate_raw(q, True)[1]]

        if source_quality is not None:
            residual, jacobians = source_quality.wrap(residual, jacobians,
                source_quality.scales(stamp, observed, 'native_edge:'+name),
                group_size=2 if finite_vector else 1)
        robust_residual, robust_jacobians = pointwise_huber(
            residual, jacobians, group_size=2 if finite_vector else 1, delta=2.*sigma)
        graph.add_observation([stamp], robust_residual,
                              residual_dim=(2 if finite_vector else 1)*len(observed),
                              sigma=sigma, jac_fn=robust_jacobians)


def _first_hit_planes(
    planes: SurfacePlanes, camera: np.ndarray, pixels: np.ndarray, hints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Freeze an observed ice/board identity and board segment, if valid."""
    rays = fast_world_rays(camera, pixels)
    centre = camera[:3]
    count = len(rays)
    point = np.zeros((count, 3), np.float64)
    normal = np.zeros((count, 3), np.float64)
    valid = np.zeros(count, bool)
    if count == 0:
        return point, normal, valid
    with np.errstate(invalid="ignore", divide="ignore"):
        ground_distance = np.where(rays[:, 2] < -1e-9, -centre[2] / rays[:, 2], np.inf)
    ground_hit = centre + ground_distance[:, None] * rays
    ground_good = np.isfinite(ground_distance) & (ground_distance > 0)
    for index in np.flatnonzero(ground_good & (hints == 1)):
        if cv2.pointPolygonTest(planes.polygon_xy, tuple(ground_hit[index, :2]), False) < 0:
            ground_good[index] = False

    edge = planes.segment_edges
    start = planes.segment_starts
    relative = start[None, :, :] - centre[None, None, :2]
    rx, ry = rays[:, 0, None], rays[:, 1, None]
    denominator = rx * edge[None, :, 1] - ry * edge[None, :, 0]
    numer = relative[:, :, 0] * edge[None, :, 1] - relative[:, :, 1] * edge[None, :, 0]
    along_numer = relative[:, :, 0] * ry - relative[:, :, 1] * rx
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = numer / denominator
        along = along_numer / denominator
    z = centre[2] + distance * rays[:, 2, None]
    edge_len = np.linalg.norm(edge, axis=1)
    wall_ok = (
        (np.abs(denominator) > 1e-10)
        & (edge_len[None, :] > 1e-5)
        & (distance > 0)
        & (along > 0.01)
        & (along < 0.99)
        & (z >= 0)
        & (z <= planes.board_top_z)
    )
    wall_distance = np.where(wall_ok, distance, np.inf)
    nearest_edge = np.argmin(wall_distance, axis=1)
    nearest_wall = wall_distance[np.arange(count), nearest_edge]
    ice = (hints == 1) & ground_good & (ground_distance < nearest_wall)
    board = (hints == 2) & np.isfinite(nearest_wall) & (nearest_wall < ground_distance)
    valid[ice | board] = True
    normal[ice, 2] = 1.
    if np.any(board):
        selected = nearest_edge[board]
        point[board, :2] = start[selected]
        normal[board, :2] = np.column_stack((edge[selected, 1], -edge[selected, 0]))
        normal[board] /= np.linalg.norm(normal[board], axis=1)[:, None]
    return point, normal, valid


def _distributed_indices(xy: np.ndarray, score: np.ndarray, limit: int) -> np.ndarray:
    if not len(xy):
        return np.empty(0, np.int32)
    ordered = np.argsort(score, kind="stable")
    col = np.clip((xy[:, 0] * 6 / 1920).astype(int), 0, 5)
    row = np.clip((xy[:, 1] * 4 / 1080).astype(int), 0, 3)
    counts = np.zeros((4, 6), np.int16)
    selected = []
    for index in ordered:
        r, c = row[index], col[index]
        if counts[r, c] >= max(2, int(np.ceil(limit / 24))):
            continue
        counts[r, c] += 1
        selected.append(int(index))
        if len(selected) == limit:
            break
    if len(selected) < limit:
        chosen = set(selected)
        selected.extend(int(index) for index in ordered if index not in chosen)
        selected = selected[:limit]
    return np.asarray(selected, np.int32)


@dataclass
class PairConstraint:
    xy0: np.ndarray
    xy1: np.ndarray
    plane_point: np.ndarray
    plane_normal: np.ndarray
    surface: np.ndarray
    camera_template: np.ndarray
    fixed_mount_rays: bool = False

    def _world_points(self, source_state: np.ndarray) -> np.ndarray:
        camera = from_state(source_state, self.camera_template)
        try:
            rays = fast_world_rays(camera, self.xy0)
        except cv2.error as error:
            raise BoardIntersectionError('pair_source_inverse_failed',
                                         np.arange(len(self.xy0))) from error
        if not np.isfinite(rays).all():
            raise BoardIntersectionError('pair_source_inverse_nonfinite',
                                         np.flatnonzero(~np.isfinite(rays).all(axis=1)))
        ray_points = np.ascontiguousarray(camera[:3] + 100. * rays)
        back_uv, _ = require_central_projection(
            camera, ray_points, stage='source_inverse', family='surface_pair',
            source_keys=None)
        inverse_error = np.linalg.norm(back_uv - self.xy0, axis=1)
        if np.any(~np.isfinite(inverse_error) | (inverse_error > .005)):
            raise BoardIntersectionError('pair_source_inverse_not_closed',
                                         np.flatnonzero(~np.isfinite(inverse_error)
                                                       | (inverse_error > .005)))
        if self.fixed_mount_rays:
            # A fixed optical centre transfers static viewing directions
            # independently of depth. No board/stand pixel is called ice.
            return np.ascontiguousarray(self.camera_template[:3]+rays*100.)
        numerator = np.sum((self.plane_point - camera[:3]) * self.plane_normal, axis=1)
        denominator = np.sum(rays * self.plane_normal, axis=1)
        safe_denominator = np.where(
            np.abs(denominator) >= 1e-9,
            denominator,
            np.where(denominator < 0, -1e-9, 1e-9),
        )
        distance = np.clip(numerator / safe_denominator, 0.001, 1e5)
        return np.ascontiguousarray(camera[:3] + distance[:, None] * rays)

    def projected(self, states: list[np.ndarray], jacobian: bool = False):
        xyz = self._world_points(states[0])
        target = from_state(states[1], self.camera_template)
        if self.fixed_mount_rays:
            target[:3] = self.camera_template[:3]
        if jacobian:
            uv, depth, derivative = require_central_projection(
                target, xyz, stage='target_projection', family='surface_pair',
                source_keys=None, with_jacobian=True)
            if self.fixed_mount_rays:
                derivative[:, :3] = 0.
            return uv, depth, derivative
        return require_central_projection(
            target, xyz, stage='target_projection', family='surface_pair',
            source_keys=None)

    def residual(self, states: list[np.ndarray]) -> np.ndarray:
        uv = self.projected(states)[0]
        return np.nan_to_num((uv - self.xy1).ravel(), nan=1000., posinf=1000., neginf=-1000.)

    def jacobians(self, states: list[np.ndarray]) -> list[np.ndarray]:
        source, target = states
        inverse_distortion = None
        analytic_rays = self.fixed_mount_rays
        if analytic_rays and np.max(np.abs(self.camera_template[11:25])) > 1e-12:
            camera = from_state(source, self.camera_template)
            fx, fy = np.exp(camera[6:8])
            qdy = (self.xy0[:, 1]-camera[9])/fy
            qdx = (self.xy0[:, 0]-camera[8]-camera[10]*qdy)/fx
            distorted = np.ascontiguousarray(np.c_[qdx, qdy])
            unit = cv2.undistortPoints(distorted.reshape(-1, 1, 2),
                                       np.eye(3), camera[11:25]).reshape(-1, 2)
            xyz_unit = np.ascontiguousarray(np.c_[unit, np.ones(len(unit))])
            roundtrip, distortion_jac = cv2.projectPoints(
                xyz_unit, np.zeros(3), np.zeros(3), np.eye(3), camera[11:25])
            # Implicit inverse derivatives are valid when OpenCV's inverse
            # has converged. Retain the numerical fallback for exotic lenses.
            analytic_rays = np.max(np.abs(roundtrip.reshape(-1,2)-distorted)) < 1e-8
            if analytic_rays:
                try:
                    inverse_distortion = np.linalg.inv(distortion_jac[:, 3:5].reshape(-1, 2, 2))
                except np.linalg.LinAlgError:
                    analytic_rays = False
        if analytic_rays:
            camera = from_state(source, self.camera_template)
            target_camera = from_state(target, self.camera_template)
            target_camera[:3] = self.camera_template[:3]
            xyz = self._world_points(source)
            derivative_target = project_fast(target_camera, xyz, with_jacobian=True)[2]
            # A world displacement is the negative of a camera-centre
            # displacement; state centre coordinates are scaled by 100.
            image_world = -derivative_target[:, :3].reshape(-1, 2, 3) / 100.
            fx, fy = np.exp(camera[6:8])
            qy = (self.xy0[:, 1]-camera[9])/fy
            qx = (self.xy0[:, 0]-camera[8]-camera[10]*qy)/fx
            rays = (np.column_stack((qx, qy, np.ones(len(qx))))
                    if inverse_distortion is None else xyz_unit)
            rotation, rotation_jac = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))
            world_state = np.zeros((len(qx), 3, 11))
            for component in range(3):
                world_state[:, :, 3+component] = 100.*(rays @ rotation_jac[component].reshape(3, 3))
            ray_state = np.zeros((len(qx), 3, 5))
            ray_state[:, 0, 0] = -qx
            ray_state[:, 0, 1] = camera[10]*qy/fx
            ray_state[:, 1, 1] = -qy
            ray_state[:, 0, 2] = -1000./fx
            ray_state[:, 0, 3] = 1000.*camera[10]/(fx*fy)
            ray_state[:, 1, 3] = -1000./fy
            ray_state[:, 0, 4] = -1000.*qy/fx
            if inverse_distortion is not None:
                ray_state[:, :2] = np.einsum('nij,njk->nik', inverse_distortion,
                                              ray_state[:, :2])
            world_state[:, :, 6:] = 100.*np.einsum('nij,ik->nkj', ray_state, rotation)
            derivative_source = np.einsum('nij,njk->nik', image_world, world_state).reshape(-1, 11)
            derivative_target[:, :3] = 0.
            return [derivative_source, derivative_target]
        derivative_source = np.empty((2 * len(self.xy0), 11), np.float64)
        for component in range(11):
            if self.fixed_mount_rays and component < 3:
                derivative_source[:, component] = 0.
                continue
            step = 1e-5 * max(1., abs(float(source[component])))
            high, low = source.copy(), source.copy()
            high[component] += step
            low[component] -= step
            vp = self.projected([high, target])[0].ravel()
            vm = self.projected([low, target])[0].ravel()
            derivative_source[:, component] = (vp - vm) / (2 * step)
        _, _, derivative_target = self.projected(states, jacobian=True)
        return [np.nan_to_num(derivative_source), np.nan_to_num(derivative_target)]

    def error(self, states: list[np.ndarray]) -> np.ndarray:
        return np.linalg.norm(self.residual(states).reshape(-1, 2), axis=1)


def prepare_pair(
    camera: np.ndarray,
    xy0: np.ndarray,
    xy1: np.ndarray,
    inlier: np.ndarray,
    hint: np.ndarray,
    fb_error: np.ndarray,
    planes: SurfacePlanes,
    *,
    max_matches: int = 120,
) -> PairConstraint | None:
    """Select static correspondences with verified current surface intersection."""
    camera = _camera_vector(camera)
    xy0 = np.asarray(xy0, np.float64).reshape(-1, 2)
    xy1 = np.asarray(xy1, np.float64).reshape(-1, 2)
    hint = np.asarray(hint, np.uint8).reshape(-1)
    inlier = np.asarray(inlier, bool).reshape(-1)
    fb_error = np.asarray(fb_error, np.float64).reshape(-1)
    if not all(len(item) == len(xy0) for item in (xy1, hint, inlier, fb_error)):
        raise ValueError("Inconsistent optical-flow pair arrays")
    good = inlier & np.isin(hint, [1, 2]) & (fb_error < 2.5)
    good &= np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1)
    if np.count_nonzero(good) < 8:
        return None
    candidates = np.flatnonzero(good)
    selected = _distributed_indices(xy0[candidates], fb_error[candidates], 240)
    candidates = candidates[selected]
    point, normal, geometric_valid = _first_hit_planes(
        planes, camera, xy0[candidates], hint[candidates]
    )
    if np.count_nonzero(geometric_valid) < 8:
        return None
    candidates = candidates[geometric_valid]
    point, normal = point[geometric_valid], normal[geometric_valid]
    selected = _distributed_indices(xy0[candidates], fb_error[candidates], max_matches)
    return PairConstraint(
        xy0[candidates][selected].copy(), xy1[candidates][selected].copy(),
        point[selected].copy(), normal[selected].copy(), hint[candidates][selected].copy(),
        camera.copy(),
    )


def prepare_ray_pair(camera, xy0, xy1, inlier, hint, fb_error, *, max_matches=120):
    """Use masked static correspondences under the fixed-mount ray model."""
    good = np.asarray(inlier, bool) & (np.asarray(fb_error) < 2.5)
    good &= np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1)
    indices = np.flatnonzero(good)
    if len(indices) < 8:
        return None
    chosen = _distributed_indices(xy0[indices], fb_error[indices], max_matches)
    indices = indices[chosen]
    count = len(indices)
    return PairConstraint(np.asarray(xy0[indices], float), np.asarray(xy1[indices], float),
                          np.zeros((count, 3)), np.zeros((count, 3)),
                          np.full(count, 3, np.uint8), camera.copy(), True)


def prepare_surface_ray_pair(camera0, camera1, xy0, xy1, inlier, fb_error,
                             planes: SurfacePlanes, *, max_matches=120):
    """Keep only source-pixel links visible on the same physical rink surface.

    Surface classification uses two current camera guesses. The fitted factor
    still uses fixed-centre source-to-target rays and never treats a tentative
    plane intersection as a measured world landmark.
    """
    xy0, xy1 = np.asarray(xy0, float), np.asarray(xy1, float)
    good = (np.asarray(inlier, bool) & (np.asarray(fb_error, float) < 2.5)
            & np.isfinite(xy0).all(axis=1) & np.isfinite(xy1).all(axis=1))
    candidates = np.flatnonzero(good)
    if len(candidates) < 8:
        return None
    selected = _distributed_indices(xy0[candidates], np.asarray(fb_error)[candidates], 240)
    candidates = candidates[selected]
    classified = []
    for camera, pixels in ((camera0, xy0[candidates]), (camera1, xy1[candidates])):
        count = len(candidates)
        point = np.zeros((count, 3))
        normal = np.zeros((count, 3))
        identity = np.zeros(count, np.uint8)
        for label in (1, 2):
            found_point, found_normal, valid = _first_hit_planes(
                planes, camera, pixels, np.full(count, label, np.uint8)
            )
            point[valid], normal[valid], identity[valid] = (
                found_point[valid], found_normal[valid], label
            )
        classified.append((point, normal, identity))
    p0, n0, s0 = classified[0]
    p1, n1, s1 = classified[1]
    same = (s0 > 0) & (s0 == s1)
    same &= (np.sum(n0*n1, axis=1) > .998)
    same &= (np.abs(np.sum((p0-p1)*n0, axis=1)) < .15)
    candidates = candidates[same]
    if len(candidates) < 8:
        return None
    selected = _distributed_indices(xy0[candidates], np.asarray(fb_error)[candidates],
                                    max_matches)
    candidates = candidates[selected]
    count = len(candidates)
    return PairConstraint(xy0[candidates].copy(), xy1[candidates].copy(),
                          np.zeros((count, 3)), np.zeros((count, 3)),
                          s0[same][selected].copy(), camera0.copy(), True)


@dataclass(frozen=True)
class CameraAnchor:
    pts: int
    camera_id: str
    camera: np.ndarray
    xyz: np.ndarray
    uv: np.ndarray
    surface: np.ndarray
    inliers: int
    ice_inliers: int
    kind: str = "model_anchor"


def load_anchors(
    cameras_json: Path,
    *,
    accepted_statuses: Iterable[str] = ACCEPTED_ANCHOR_STATUSES,
    model_sha256: str | None = None,
) -> dict[int, CameraAnchor]:
    manifest = json.loads(cameras_json.read_text(encoding="utf-8"))
    if model_sha256 is not None and manifest.get("model_sha256") != model_sha256:
        raise ValueError("Accepted camera anchors lack the exact released scene SHA256")
    statuses = set(accepted_statuses)
    anchors: dict[int, CameraAnchor] = {}
    for item in manifest["frames"]:
        if item["status"] not in statuses:
            continue
        pts = int(item["pts"])
        if pts in anchors:
            raise ValueError(f"Duplicate accepted camera anchor at native PTS {pts}")
        with np.load(cameras_json.parent / f"{pts}.npz", allow_pickle=False) as data:
            if ('xyz_model_sha256' in data.files and
                    str(data['xyz_model_sha256']) != manifest['model_sha256']):
                raise ValueError('Updated camera has legacy base-scene XYZ; use its camera state '
                                 'as an initializer, not as recalibrated model/pixel observations')
            camera = _camera_vector(data["camera"])
            xyz = np.asarray(data["xyz"], np.float64).reshape(-1, 3).copy()
            uv = np.asarray(data["uv"], np.float64).reshape(-1, 2).copy()
            surface = np.asarray(data["surface"]).astype(str).copy()
        if (len(xyz) != len(uv) or len(xyz) != len(surface) or len(xyz) < 6
                or not np.isfinite(xyz).all() or not np.isfinite(uv).all()):
            raise ValueError(f"Accepted anchor {pts} lacks six finite model/pixel correspondences")
        if "camera_vector25" in item and not np.allclose(camera, _camera_vector(item["camera_vector25"]),
                                                          rtol=0, atol=1e-10):
            raise ValueError(f"Anchor {pts} JSON camera and numeric asset disagree")
        anchors[pts] = CameraAnchor(
            pts, str(item["camera_id"]), camera, xyz, uv, surface,
            int(item.get("inliers", len(xyz))), int(item.get("ice_inliers", 0)),
        )
    return anchors


def _absolute_factor(graph: IncrementalCameraGraph, anchor: CameraAnchor, template: np.ndarray,
                     *, fixed_mount: bool = False,
                     temporal_prior_sigmas=(1.5, .0005, 1., 1., 1.)) -> None:
    pts = anchor.pts
    camera = anchor.camera.copy()
    if fixed_mount:
        camera[:3] = template[:3]
    state = to_state(camera)
    # A visually established anchor must retain its ice alignment. Atlas
    # texture has inherited calibration uncertainty and cannot overwhelm it.
    sigma = ANCHOR_SIGMA.copy()
    if anchor.kind == "temporal_keyframe":
        rotation_px, focal_log, cx_px, cy_px, skew_px = temporal_prior_sigmas
        sigma[3:6] = rotation_px / max(np.exp(anchor.camera[6:8]))
        sigma[6:8] = focal_log
        sigma[8:11] = np.asarray([cx_px, cy_px, skew_px]) / 1000.
    else:
        sigma[3:6] = .8 / max(np.exp(anchor.camera[6:8]))
        sigma[6:8] = .0005
        sigma[8:11] = .001
    if fixed_mount:
        sigma[:3] = 1e-8
    graph.add_prior(pts, state, sigma)

    initial_error = np.linalg.norm(project_fast(camera, anchor.xyz)[0]-anchor.uv, axis=1)
    good = np.isfinite(initial_error) & (initial_error <= 8.)
    if np.count_nonzero(good) < 6:
        return
    xyz, measured_uv = anchor.xyz[good], anchor.uv[good]

    def constrained_camera(state):
        camera = from_state(state, template)
        if fixed_mount:
            camera[:3] = template[:3]
        return camera

    def residual(states: list[np.ndarray]) -> np.ndarray:
        uv = project_fast(constrained_camera(states[0]), xyz)[0]
        return (uv - measured_uv).ravel()

    def jacobians(states: list[np.ndarray]) -> list[np.ndarray]:
        derivative = project_fast(constrained_camera(states[0]), xyz, with_jacobian=True)[2]
        if fixed_mount:
            derivative[:, :3] = 0.
        return [derivative]

    graph.add_observation(
        [pts], residual, residual_dim=2 * len(xyz), sigma=8.0,
        jac_fn=jacobians, huber_k=3.0,
    )


def _fixed_mount_factor(graph: IncrementalCameraGraph, pts: int,
                        centre_scene: np.ndarray, sigma_scene: float) -> None:
    """Keep pan/tilt/zoom free while bounding the camera's installed centre."""
    if not np.isfinite(sigma_scene) or sigma_scene <= 0:
        raise ValueError("Centre prior sigma must be finite and positive")
    centre_state = np.asarray(centre_scene, np.float64) / 100.
    jacobian = np.zeros((3, 11), np.float64)
    jacobian[:, :3] = np.eye(3)

    def residual(states: list[np.ndarray]) -> np.ndarray:
        return states[0][:3] - centre_state

    def jacobians(_states: list[np.ndarray]) -> list[np.ndarray]:
        return [jacobian]

    graph.add_observation([pts], residual, residual_dim=3, sigma=sigma_scene / 100.,
                          jac_fn=jacobians)


def _calibration_references(anchors: dict[int, CameraAnchor]):
    """Use solved temporal calibration when propagating between keyframes.

    A lone original anchor must not override the locally solved lens/aspect.
    Joint fitting without temporal keyframes retains its original references.
    """
    if any(a.kind == 'temporal_keyframe' for a in anchors.values()):
        return sorted(anchors.items())
    originals = sorted((t, a) for t, a in anchors.items() if a.kind == 'model_anchor')
    return originals or sorted(anchors.items())


def _lens_reference(anchors: dict[int, CameraAnchor], stamp: int) -> np.ndarray:
    """Interpolate solved calibration, not accumulating principal-point drift."""
    originals = _calibration_references(anchors)
    stamps = np.asarray([t for t, _ in originals])
    values = np.asarray([a.camera[8:11] for _, a in originals])
    return np.asarray([np.interp(stamp, stamps, values[:, j]) for j in range(3)])


def _lens_factor(graph: IncrementalCameraGraph, pts: int, reference_px: np.ndarray,
                 sigma_px=(8., 8., 20.)) -> None:
    """Soft lens calibration prior; a prior is not a new image observation."""
    reference = np.asarray(reference_px, float)/1000.
    derivative = np.zeros((3, 11))
    derivative[:, 8:11] = np.eye(3)
    def residual(states):
        return states[0][8:11]-reference
    def jacobian(_states):
        return [derivative]
    graph.add_observation([pts], residual, residual_dim=3,
                          sigma=np.asarray(sigma_px)/1000., jac_fn=jacobian)


def _aspect_factor(graph: IncrementalCameraGraph, pts: int,
                   anchors: dict[int, CameraAnchor], sigma_log_ratio=.02) -> None:
    """Zoom changes scale; it cannot freely stretch one image axis by 60%."""
    originals = _calibration_references(anchors)
    reference = float(np.interp(pts, [t for t, _ in originals],
                                [a.camera[6]-a.camera[7] for _, a in originals]))
    derivative = np.zeros((1, 11))
    derivative[0, 6:8] = (1., -1.)
    def residual(states):
        return np.asarray([states[0][6]-states[0][7]-reference])
    def jacobian(_states):
        return [derivative]
    graph.add_observation([pts], residual, residual_dim=1, sigma=sigma_log_ratio,
                          jac_fn=jacobian)


def _focal_reference(anchors: dict[int, CameraAnchor], stamp: int) -> np.ndarray | None:
    """Return a nearby *solved keyframe* focal estimate, never a pixel label.

    A lone model anchor or a long unobserved interval cannot establish the
    intervening zoom.  Normal native frames are bracketed by measured source
    keyframes about half a second apart; interpolation is only a soft prior.
    """
    stamps = np.asarray(sorted(anchors), np.int64)
    if len(stamps) < 2:
        return None
    right = int(np.searchsorted(stamps, stamp))
    if right < len(stamps) and stamps[right] == stamp:
        return anchors[int(stamp)].camera[6:8].copy()
    if right == 0 or right == len(stamps):
        return None
    previous, following = int(stamps[right - 1]), int(stamps[right])
    if following - previous > FOCAL_REFERENCE_MAX_GAP_PTS:
        return None
    fraction = (stamp - previous) / (following - previous)
    return ((1. - fraction) * anchors[previous].camera[6:8]
            + fraction * anchors[following].camera[6:8])


def _state_initial_reference(anchors: dict[int, CameraAnchor], stamp: int) -> np.ndarray | None:
    """Interpolate solved keyframes only to initialize a nearby native frame.

    This never adds a measurement; original source pixels and camera factors
    still determine the optimized state. Adjacent source keyframes here are
    close enough that their Rodrigues-vector branches are continuous.
    """
    stamps = np.asarray(sorted(anchors), np.int64)
    right = int(np.searchsorted(stamps, stamp))
    if right == 0 or right == len(stamps):
        return None
    previous, following = int(stamps[right - 1]), int(stamps[right])
    if following - previous > FOCAL_REFERENCE_MAX_GAP_PTS:
        return None
    fraction = (stamp - previous) / (following - previous)
    before = to_state(anchors[previous].camera)
    after = to_state(anchors[following].camera)
    if np.linalg.norm(after[3:6] - before[3:6]) > np.pi:
        return None
    return (1. - fraction) * before[3:11] + fraction * after[3:11]


def _focal_factor(graph: IncrementalCameraGraph, pts: int,
                  reference_logf: np.ndarray, sigma_logf: float) -> None:
    """Prevent a single bad local solve from choosing an alternate zoom branch."""
    if not np.isfinite(sigma_logf) or sigma_logf <= 0:
        raise ValueError("Focal prior sigma must be finite and positive")
    target = np.asarray(reference_logf, np.float64).reshape(2)
    derivative = np.zeros((2, 11), np.float64)
    derivative[:, 6:8] = np.eye(2)

    def residual(states: list[np.ndarray]) -> np.ndarray:
        return states[0][6:8] - target

    def jacobians(_states: list[np.ndarray]) -> list[np.ndarray]:
        return [derivative]

    graph.add_observation([pts], residual, residual_dim=2, sigma=sigma_logf,
                          jac_fn=jacobians)


def _frame_quality(pair: PairConstraint | None, states: list[np.ndarray]) -> tuple[str, tuple[int, int, int, float, float, float]]:
    if pair is None:
        return "predicted", (0, 0, 0, np.nan, np.nan, 0.)
    error = pair.error(states)
    count = len(error)
    if not count or not np.isfinite(error).all():
        return "predicted", (count, int(np.sum(pair.surface == 1)),
                             int(np.sum(pair.surface == 2)), np.nan, np.nan, 0.)
    hull = cv2.convexHull(pair.xy0.astype(np.float32)) if count >= 3 else None
    coverage = float(cv2.contourArea(hull) / (1920 * 1080)) if hull is not None else 0.
    median, p95 = float(np.median(error)), float(np.percentile(error, 95))
    observed = count >= 24 and coverage >= 0.006 and median <= 3.5 and p95 <= 12.
    return ("observed" if observed else "predicted",
            (count, int(np.sum(pair.surface == 1)), int(np.sum(pair.surface == 2)),
             median, p95, coverage))


def _pair_data(motion: dict[str, np.ndarray], index: int) -> tuple[np.ndarray, ...]:
    start, end = int(motion["offsets"][index]), int(motion["offsets"][index + 1])
    return tuple(motion[name][start:end] for name in
                 ("xy0", "xy1", "homography_inlier", "region_hint", "fb_error_px"))


def _motion_initial(camera: np.ndarray, pair: PairConstraint | None, direction: int) -> np.ndarray:
    """Initialize pinhole pan/tilt/zoom from measured static image homography.

    RQ decomposition only supplies the next iSAM2 linearization point. The
    graph still fits the original pixel pairs and absolute model anchors.
    Fixed nonzero Brown optics need nonlinear inversion and use the previous
    state instead of silently interpreting distorted pixels as pinhole rays.
    """
    if pair is None or not pair.fixed_mount_rays or np.max(np.abs(camera[11:25])) > 1e-12:
        return to_state(camera)
    source, target = (pair.xy0, pair.xy1) if direction == 1 else (pair.xy1, pair.xy0)
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
    if homography is None or inliers is None or int(inliers.sum()) < 8:
        return to_state(camera)
    fx, fy = np.exp(camera[6:8])
    intrinsic = np.array([[fx, camera[10], camera[8]], [0., fy, camera[9]], [0., 0., 1.]])
    rotation = cv2.Rodrigues(camera[3:6])[0]
    matrix = homography @ intrinsic @ rotation
    if np.linalg.det(matrix) < 0:
        matrix = -matrix
    _, new_k, new_r, *_ = cv2.RQDecomp3x3(np.ascontiguousarray(matrix))
    signs = np.diag(np.where(np.diag(new_k) < 0, -1., 1.))
    new_k, new_r = new_k @ signs, signs @ new_r
    if np.linalg.det(new_r) < 0 or abs(new_k[2, 2]) < 1e-10:
        return to_state(camera)
    new_k /= new_k[2, 2]
    if (not np.isfinite(new_k).all() or new_k[0, 0] <= 0 or new_k[1, 1] <= 0
            or np.max(np.abs(np.log([new_k[0, 0]/fx, new_k[1, 1]/fy]))) > .15):
        return to_state(camera)
    result = camera.copy()
    rvec = cv2.Rodrigues(new_r)[0].ravel()
    angle = np.linalg.norm(rvec)
    if angle > 1e-10:
        equivalent = rvec * (1 - 2*np.pi/angle)
        if np.linalg.norm(equivalent-camera[3:6]) < np.linalg.norm(rvec-camera[3:6]):
            rvec = equivalent
    result[3:6] = rvec
    result[6:8] = np.log([new_k[0, 0], new_k[1, 1]])
    result[8:11] = [new_k[0, 2], new_k[1, 2], new_k[0, 1]]
    return to_state(result)


METRIC_DTYPE = np.dtype([
    ("pair_matches", "i4"), ("ice_matches", "i4"), ("board_matches", "i4"),
    ("pair_median_px", "f4"), ("pair_p95_px", "f4"), ("pair_hull_fraction", "f4"),
    ("anchor_median_px", "f4"), ("anchor_p95_px", "f4"),
    ("anchor_inliers", "i4"), ("anchor_ice_inliers", "i4"),
])


def track_shot(
    shot: dict,
    anchors: dict[int, CameraAnchor],
    planes: SurfacePlanes,
    *,
    window_size: int = 120,
    max_matches: int = 120,
    model_sha256: str | None = None,
    center_sigma_scene: float = CENTER_SIGMA_SCENE,
    motion_model: str = "fixed_mount_rays",
    lens_sigma_px=(8., 8., 20.),
    focal_sigma_log: float = 0.06,
    aspect_sigma_log: float = 0.02,
    pair_sigma_px: float = 2.0,
    solver_refinement_updates: int = 1,
    temporal_prior_sigmas=(1.5, .0005, 1., 1., 1.),
    solver_mode: str = "gauss_newton",
    source_edges: Path | None = None,
    edge_world: dict[str, np.ndarray] | None = None,
    source_sha256: str | None = None,
    update_stride: int = 1,
    native_landmarks=None,
    graph_scope: str = "fixed_lag",
    update_strategy: str = "incremental",
    deferred_refinement_updates: int = 30,
    focal_prior_mode: str = "per_frame",
) -> dict:
    """Track one shot from an accepted anchor toward both source-time ends."""
    with np.load(shot["motion_npz"], allow_pickle=False) as data:
        motion = {name: data[name].copy() for name in data.files}
    pairs = np.asarray(motion["pair_pts"], np.int64)
    if len(pairs) != int(shot["pair_count"]) or len(pairs) == 0:
        raise ValueError(f"Shot {shot['shot_index']} has no complete pair index")
    if not np.all(pairs[1:, 0] == pairs[:-1, 1]) or np.any(pairs[:, 1] - pairs[:, 0] != 256):
        raise ValueError(f"Shot {shot['shot_index']} pair chain is not contiguous native video")
    pts = np.r_[pairs[0, 0], pairs[:, 1]].astype(np.int64)
    if (pts[0] != shot["start_pts"] or pts[-1] != shot["end_pts"] - 256
            or len(pts) != (shot["end_pts"] - shot["start_pts"]) // 256):
        raise ValueError("Motion pair chain crosses or misses the reviewed shot boundary")
    edges = None
    if source_edges is not None:
        if edge_world is None or source_sha256 is None or model_sha256 is None:
            raise ValueError("Source-edge factors require released physical curves and source/model hashes")
        edge_world = {name: _closed_edge_ring(edge_world[name])
                      for name in ("board_top", "ice_board_contact")}
        for name in ("board_top", "ice_board_contact"):
            curve = edge_world[name]
            if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2 or not np.isfinite(curve).all():
                raise ValueError(f"Invalid released physical curve: {name}")
        edges = SourceEdges.load(Path(source_edges), pts,
                                 source_sha256=source_sha256, model_sha256=model_sha256)
    shot_pts = set(pts.tolist())
    own = {int(t): a for t, a in anchors.items()
           if t in shot_pts and a.camera_id == shot["camera_id"]}
    if own:
        lens = next(iter(own.values())).camera[11:25]
        for anchor in own.values():
            if np.max(np.abs(anchor.camera[11:25] - lens)) > 1e-10:
                raise ValueError(
                    f"Shot {shot['shot_index']} anchors have incompatible fixed Brown optics; "
                    "the 11-state tracker cannot interpolate different lens distortion"
                )
    if not np.isfinite(focal_sigma_log) or focal_sigma_log <= 0:
        raise ValueError("focal_sigma_log must be finite and positive")
    if focal_prior_mode not in ("per_frame", "none"):
        raise ValueError("focal_prior_mode must be per_frame or none")
    if not np.isfinite(aspect_sigma_log) or aspect_sigma_log <= 0:
        raise ValueError("aspect_sigma_log must be finite and positive")
    if not np.isfinite(pair_sigma_px) or pair_sigma_px <= 0:
        raise ValueError("pair_sigma_px must be finite and positive")
    prior_sigmas = np.asarray(temporal_prior_sigmas, np.float64).reshape(-1)
    if prior_sigmas.shape != (5,) or not np.isfinite(prior_sigmas).all() or np.any(prior_sigmas <= 0):
        raise ValueError("temporal_prior_sigmas need five finite positive sigmas")
    if not 0 <= solver_refinement_updates <= 50:
        raise ValueError("solver_refinement_updates must be between 0 and 50")
    if not 1 <= update_stride <= window_size:
        raise ValueError("update_stride must be between 1 and the fixed-lag window_size")
    if graph_scope not in ("fixed_lag", "whole_shot"):
        raise ValueError("graph_scope must be fixed_lag or whole_shot")
    if update_strategy not in ("incremental", "deferred_whole"):
        raise ValueError("update_strategy must be incremental or deferred_whole")
    if not 0 <= deferred_refinement_updates <= 100:
        raise ValueError("deferred_refinement_updates must be between 0 and 100")
    if motion_model not in ("fixed_mount_rays", "fixed_mount_surface_rays", "surface_planes"):
        raise ValueError("Unknown camera motion model")
    fixed_mount = motion_model != "surface_planes"
    camera = np.full((len(pts), 25), np.nan)
    status = np.full(len(pts), "unavailable", dtype="<U16")
    metrics = np.zeros(len(pts), dtype=METRIC_DTYPE)
    for field in ("pair_median_px", "pair_p95_px", "anchor_median_px", "anchor_p95_px"):
        metrics[field] = np.nan
    identity = {int(stamp): index for index, stamp in enumerate(pts)}
    if not own:
        return {"pts": pts, "camera25": camera, "status": status, "metrics": metrics,
                "shot_index": int(shot["shot_index"]), "camera_id": shot["camera_id"],
                "anchor_pts": np.empty(0, np.int64), "model_sha256": np.asarray(model_sha256 or ""),
                "center_deviation_scene": np.full(len(pts), np.nan, np.float32)}
    middle = float(pts[len(pts) // 2])
    first = max(own.values(), key=lambda a: (a.ice_inliers > 0, a.inliers, -abs(a.pts - middle)))
    seed_index = int(np.searchsorted(pts, first.pts))
    if pts[seed_index] != first.pts:
        raise ValueError("Anchor PTS is not a source frame in its shot")
    status[seed_index] = "keyframe" if first.kind == "temporal_keyframe" else "anchor"
    constraints: list[PairConstraint | None] = [None] * len(pairs)
    branch_seed_camera25 = np.full((2, 25), np.nan, np.float64)
    effective_window_size = (max(window_size, len(pts) + 1)
                             if graph_scope == "whole_shot" else window_size)
    whole_shot_graph = effective_window_size >= len(pts)
    if update_strategy == "deferred_whole" and not whole_shot_graph:
        raise ValueError("deferred_whole requires a whole-shot iSAM2 graph")
    graph = None
    template = first.camera.copy()

    for branch_index, direction in enumerate((1, -1)):
        if graph is None or not whole_shot_graph:
            graph = IncrementalCameraGraph(11, window_size=effective_window_size,
                                           relinearize_threshold=0.00005,
                                           refinement_updates=solver_refinement_updates,
                                           optimization=solver_mode)
            _absolute_factor(graph, first, template, fixed_mount=fixed_mount,
                             temporal_prior_sigmas=prior_sigmas)
            _lens_factor(graph, first.pts, _lens_reference(own, first.pts), lens_sigma_px)
            _aspect_factor(graph, first.pts, own, aspect_sigma_log)
            if edges is not None:
                _add_source_edge_factors(graph, first.pts, seed_index, template,
                                         first.camera, edges, edge_world,
                                         fixed_mount=fixed_mount)
            if native_landmarks is not None:
                native_landmarks.add_factors(graph, first.pts, template,
                                             fixed_mount=fixed_mount)
            active = graph.update({first.pts: to_state(first.camera)})
        else:
            # The full shot fits inside one iSAM2 graph. Keep the solved seed
            # live while attaching the reverse-time half and jointly refine it.
            active = graph.active_estimates()
        camera[seed_index] = _camera_vector(from_state(active[first.pts], template))
        current_index = seed_index
        pending_states: dict[int, np.ndarray] = {}
        while 0 <= current_index + direction < len(pts):
            next_index = current_index + direction
            existing_pts = int(pts[current_index])
            next_pts = int(pts[next_index])
            pair_index = current_index if direction == 1 else next_index
            previous_pts, following_pts = map(int, pairs[pair_index])
            if {previous_pts, following_pts} != {existing_pts, next_pts}:
                raise ValueError("Frame and optical-flow pair PTS disagree")
            estimate = (pending_states[existing_pts] if existing_pts in pending_states
                        else graph.estimate(existing_pts))
            initial_camera = from_state(estimate, template)
            xy0, xy1, inlier, hint, fb_error = _pair_data(motion, pair_index)
            if motion_model == "fixed_mount_surface_rays":
                state_reference = _state_initial_reference(own, next_pts)
                tentative = from_state(estimate, template)
                if state_reference is not None:
                    tentative_state = to_state(tentative)
                    tentative_state[3:11] = state_reference
                    tentative = from_state(tentative_state, template)
                if direction == 1:
                    camera0, camera1 = initial_camera, tentative
                else:
                    camera0, camera1 = tentative, initial_camera
                pair = prepare_surface_ray_pair(camera0, camera1, xy0, xy1, inlier,
                                                fb_error, planes, max_matches=max_matches)
            elif motion_model == "fixed_mount_rays":
                pair = prepare_ray_pair(initial_camera, xy0, xy1, inlier, hint, fb_error,
                                        max_matches=max_matches)
            else:
                pair = prepare_pair(initial_camera, xy0, xy1, inlier, hint, fb_error,
                                    planes, max_matches=max_matches)
            constraints[pair_index] = pair
            prior_delta = np.zeros(11)
            between_sigma = BETWEEN_SIGMA.copy()
            previous_focal = _focal_reference(own, previous_pts)
            following_focal = _focal_reference(own, following_pts)
            if previous_focal is not None and following_focal is not None:
                # This is a local camera-motion prior from two independently
                # solved source keyframes. It constrains the *per-frame* zoom
                # rate, so a bad incremental solve cannot jump focal branches
                # in a single native frame while legitimate gradual zoom can
                # accumulate over the entire keyframe interval.
                prior_delta[6:8] = following_focal - previous_focal
                between_sigma[6:8] = FOCAL_MOTION_SIGMA_LOG
            graph.add_between(previous_pts, following_pts, prior_delta, between_sigma)
            _fixed_mount_factor(graph, next_pts, first.camera[:3], center_sigma_scene)
            _lens_factor(graph, next_pts, _lens_reference(own, next_pts), lens_sigma_px)
            _aspect_factor(graph, next_pts, own, aspect_sigma_log)
            focal_reference = _focal_reference(own, next_pts)
            if (focal_prior_mode == "per_frame" and focal_reference is not None
                    and next_pts not in own):
                _focal_factor(graph, next_pts, focal_reference, focal_sigma_log)
            if pair is not None:
                pair_residual, pair_jacobians = pointwise_huber(
                    pair.residual, pair.jacobians, group_size=2,
                    delta=3.0 * pair_sigma_px,
                )
                graph.add_observation(
                    [previous_pts, following_pts], pair_residual,
                    residual_dim=2 * len(pair.xy0), sigma=pair_sigma_px,
                    jac_fn=pair_jacobians,
                )
            other_anchor = own.get(next_pts)
            if other_anchor is not None:
                _absolute_factor(graph, other_anchor, template, fixed_mount=fixed_mount,
                                 temporal_prior_sigmas=prior_sigmas)
                initial = to_state(other_anchor.camera)
                if fixed_mount:
                    initial[:3] = template[:3]/100.
            else:
                state_reference = _state_initial_reference(own, next_pts)
                if state_reference is not None:
                    initial = to_state(initial_camera)
                    initial[3:11] = state_reference
                else:
                    initial = _motion_initial(initial_camera, pair, direction)
                if (focal_reference is not None and
                        np.max(np.abs(initial[6:8] - focal_reference)) >
                        FOCAL_INITIAL_MAX_DEVIATION):
                    # This only selects the optimizer's starting point. The
                    # source-pixel factors still determine the fitted state.
                    initial[6:8] = focal_reference
            if edges is not None:
                _add_source_edge_factors(graph, next_pts, next_index, template,
                                         from_state(initial, template), edges, edge_world,
                                         fixed_mount=fixed_mount)
            if native_landmarks is not None:
                native_landmarks.add_factors(graph, next_pts, template,
                                             fixed_mount=fixed_mount)
            pending_states[next_pts] = initial
            # Keep every native state and real image factor in the same fixed-lag
            # graph; batch only the expensive nonlinear update. A solved
            # keyframe and the final frame always trigger an immediate solve.
            at_branch_end = not 0 <= next_index + direction < len(pts)
            should_update = (at_branch_end if update_strategy == "deferred_whole" else
                             len(pending_states) >= update_stride or
                             other_anchor is not None or at_branch_end)
            if should_update:
                if update_strategy == "deferred_whole":
                    graph.refinement_updates = deferred_refinement_updates
                active = graph.update(pending_states)
                # The smoother retires old states after the batch solve. Their
                # just-optimized pretrim estimates must replace the older
                # exported camera rows as well as the still-active states.
                for stamp, vector in {**active, **graph.last_retired_estimates}.items():
                    camera[identity[stamp]] = _camera_vector(from_state(vector, template))
                pending_states = {}
            if other_anchor is not None:
                status[next_index] = "keyframe" if other_anchor.kind == "temporal_keyframe" else "anchor"
            current_index = next_index
        # The two native-time directions are optimized in separate graphs.
        # Retain both fitted seed estimates so a branch seam is observable.
        branch_seed_camera25[branch_index] = camera[seed_index]
    # Score the camera vectors actually exported, after all fixed-lag updates.
    # `observed` refers to local image motion, not validated global ice fit.
    for index, stamp in enumerate(pts):
        anchor = own.get(int(stamp))
        if anchor is not None:
            metrics[index]["anchor_inliers"] = anchor.inliers
            metrics[index]["anchor_ice_inliers"] = anchor.ice_inliers
            if not len(anchor.xyz):
                continue
            projected = project_fast(camera[index], anchor.xyz)[0]
            error = np.linalg.norm(projected - anchor.uv, axis=1)
            if np.isfinite(error).all():
                metrics[index]["anchor_median_px"] = np.median(error)
                metrics[index]["anchor_p95_px"] = np.percentile(error, 95)
            else:
                status[index] = "unavailable"
            continue
        pair_index = index - 1 if index > seed_index else index
        pair = constraints[pair_index]
        states = [to_state(camera[pair_index]), to_state(camera[pair_index + 1])]
        status[index], row = _frame_quality(pair, states)
        for field, value in zip(METRIC_DTYPE.names[:6], row):
            metrics[index][field] = value
    return {"pts": pts, "camera25": camera, "status": status, "metrics": metrics,
            "shot_index": int(shot["shot_index"]), "camera_id": shot["camera_id"],
            "anchor_pts": np.asarray(sorted(t for t, a in own.items() if a.kind == "model_anchor"), np.int64),
            "temporal_keyframe_pts": np.asarray(sorted(t for t, a in own.items() if a.kind == "temporal_keyframe"), np.int64),
            "motion_model": np.asarray(motion_model),
            "aspect_sigma_log": np.asarray(aspect_sigma_log),
            "pair_sigma_px": np.asarray(pair_sigma_px),
            "solver_refinement_updates": np.asarray(solver_refinement_updates),
            "temporal_prior_sigmas": prior_sigmas.copy(),
            "branch_seed_camera25": branch_seed_camera25,
            "graph_topology": np.asarray(
                "whole_shot_bidirectional_isam2" if whole_shot_graph
                else "two_fixed_lag_directional_isam2"),
            "graph_scope": np.asarray(graph_scope),
            "effective_window_size": np.asarray(effective_window_size),
            "update_strategy": np.asarray(update_strategy),
            "deferred_refinement_updates": np.asarray(deferred_refinement_updates),
            "focal_prior_mode": np.asarray(focal_prior_mode),
            "native_landmarks_metadata_json": np.asarray(
                json.dumps(native_landmarks.metadata(), sort_keys=True)
                if native_landmarks is not None else ""),
            "update_stride": np.asarray(update_stride),
            "source_edge_fixed_center": np.asarray(edges is not None and fixed_mount),
            "model_sha256": np.asarray(model_sha256 or ""),
            "source_edge_counts": (np.column_stack([
                np.asarray([len(edges.frame(i, edge_id)) for i in range(len(pts))], np.int16)
                for edge_id in (0, 1)]) if edges is not None else np.zeros((len(pts), 2), np.int16)),
            "center_deviation_scene": np.linalg.norm(camera[:, :3] - first.camera[:3], axis=1).astype(np.float32)}


def track_program(
    *,
    motion_index: Path = DEFAULT_ROOT / "three_frame_refit/program_tracking/motion_index.json",
    keyframes_json: Path = DEFAULT_TEMP / "keyframes/cameras.json",
    scene_path: Path = DEFAULT_ROOT / "three_frame_refit/deliverables/scene.json",
    release_path: Path = DEFAULT_ROOT / "three_frame_refit/deliverables/release.json",
    output_dir: Path = DEFAULT_TEMP / "tracked",
    accepted_statuses: Iterable[str] = ACCEPTED_ANCHOR_STATUSES,
    window_size: int = 120,
    max_matches: int = 120,
    center_sigma_scene: float = CENTER_SIGMA_SCENE,
    shot_indices: Iterable[int] | None = None,
    motion_model: str = "fixed_mount_rays",
    candidate_review: bool = False,
    temporal_keyframes: Path | None = None,
    lens_sigma_px=(8., 8., 20.),
    focal_sigma_log: float = 0.06,
    aspect_sigma_log: float = 0.02,
    pair_sigma_px: float = 2.0,
    solver_refinement_updates: int = 1,
    temporal_prior_sigmas=(1.5, .0005, 1., 1., 1.),
    solver_mode: str = "gauss_newton",
    source_edges: Path | None = None,
    native_landmarks_config: Path | None = None,
    native_landmarks_joint_dir: Path | None = None,
    update_stride: int = 1,
    graph_scope: str = "fixed_lag",
    update_strategy: str = "incremental",
    deferred_refinement_updates: int = 30,
    focal_prior_mode: str = "per_frame",
) -> list[dict]:
    manifest = json.loads(motion_index.read_text(encoding="utf-8"))
    if manifest["coordinate_system"] != "native_source_pixels_1920x1080":
        raise ValueError("Source motion is not in native 1920x1080 coordinates")
    scene_bytes = scene_path.read_bytes()
    scene_sha256 = hashlib.sha256(scene_bytes).hexdigest()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release["version"] != "0.25" or scene_sha256 != release["model_sha256"]:
        raise ValueError("Tracking scene is not the released v0.25 rink model")
    scene = json.loads(scene_bytes)
    planes = SurfacePlanes.from_scene(scene)
    edge_world = None
    if source_edges is not None:
        edge_world = {
            "board_top": np.asarray(scene["board_surface"]["top_edge_xyz"], np.float64),
            "ice_board_contact": np.asarray(scene["ice_boundary"]["points_xyz"], np.float64),
        }
    anchor_sha256 = hashlib.sha256(keyframes_json.read_bytes()).hexdigest()
    if candidate_review:
        accepted_statuses = set(accepted_statuses) | {"camera_candidate", "joint_camera_candidate_unvalidated"}
    anchors = load_anchors(keyframes_json, accepted_statuses=accepted_statuses,
                           model_sha256=scene_sha256)
    for anchor in anchors.values():
        matching = [shot for shot in manifest["shots"]
                    if shot["start_pts"] <= anchor.pts < shot["end_pts"]]
        if len(matching) != 1 or matching[0]["camera_id"] != anchor.camera_id:
            raise ValueError(f"Accepted anchor {anchor.pts} is outside its reviewed camera shot")
    source_shots = json.loads(Path(manifest["source_shots"]).read_text(encoding="utf-8"))
    if (native_landmarks_config is None) != (native_landmarks_joint_dir is None):
        raise ValueError("Native landmarks require both joint configuration and solved directory")
    if native_landmarks_config is not None and temporal_keyframes is None:
        raise ValueError("Native landmarks require the matching solved temporal keyframes")
    if len(source_shots["shots"]) != len(manifest["shots"]):
        raise ValueError("Motion index and reviewed cut count differ")
    for index, (reviewed, tracked_shot) in enumerate(zip(source_shots["shots"], manifest["shots"])):
        if any(reviewed[key] != tracked_shot[key] for key in ("start_pts", "end_pts", "camera_id")):
            raise ValueError(f"Motion index differs from reviewed source shot {index}")
    if output_dir.resolve().is_relative_to(DEFAULT_ROOT.resolve()):
        raise ValueError("Large tracked-frame cache must remain outside the OneDrive project")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    selected_shots = None if shot_indices is None else set(shot_indices)
    if selected_shots is not None and not selected_shots.issubset(set(range(len(manifest["shots"])))):
        raise ValueError("Requested shot index is not in the reviewed source")
    for shot in manifest["shots"]:
        if selected_shots is not None and shot["shot_index"] not in selected_shots:
            continue
        native_landmarks = None
        if native_landmarks_config is not None:
            try:
                from .source_landmarks import NativeLandmarks
            except ImportError:
                from source_landmarks import NativeLandmarks
            native_landmarks = NativeLandmarks.load(
                native_landmarks_config, native_landmarks_joint_dir, scene,
                int(shot["shot_index"]),
                source_sha256=source_shots["source_sha256_from_timeline"],
                model_sha256=scene_sha256,
            )
        shot_anchors = dict(anchors)
        if temporal_keyframes is not None:
            path = temporal_keyframes/f'shot{shot["shot_index"]:02d}_keyframes.npz'
            with np.load(path) as data:
                if str(data['model_sha256']) != scene_sha256:
                    raise ValueError('Temporal keyframes belong to a different rink model')
                if str(data['source_sha256']) != source_shots['source_sha256_from_timeline']:
                    raise ValueError('Temporal keyframes belong to a different video')
                for stamp, camera in zip(data['pts'], data['camera25']):
                    stamp = int(stamp)
                    if not shot['start_pts'] <= stamp < shot['end_pts']:
                        raise ValueError('Temporal keyframe crosses a source cut')
                    if stamp not in shot_anchors:
                        shot_anchors[stamp] = CameraAnchor(
                            stamp, shot['camera_id'], _camera_vector(camera),
                            np.empty((0, 3)), np.empty((0, 2)), np.empty(0, '<U16'),
                            0, 0, 'temporal_keyframe')
        result = track_shot(
            shot, shot_anchors, planes, window_size=window_size, max_matches=max_matches,
            model_sha256=scene_sha256, center_sigma_scene=center_sigma_scene,
            motion_model=motion_model,
            lens_sigma_px=lens_sigma_px,
            focal_sigma_log=focal_sigma_log,
            aspect_sigma_log=aspect_sigma_log,
            pair_sigma_px=pair_sigma_px,
            solver_refinement_updates=solver_refinement_updates,
            temporal_prior_sigmas=temporal_prior_sigmas,
            solver_mode=solver_mode,
            source_edges=(source_edges / f"shot{shot['shot_index']:02d}.npz"
                          if source_edges is not None else None),
            edge_world=edge_world,
            source_sha256=source_shots["source_sha256_from_timeline"],
            update_stride=update_stride,
            native_landmarks=native_landmarks,
            graph_scope=graph_scope,
            update_strategy=update_strategy,
            deferred_refinement_updates=deferred_refinement_updates,
            focal_prior_mode=focal_prior_mode,
        )
        result["source_sha256"] = np.asarray(source_shots["source_sha256_from_timeline"])
        edge_file = (source_edges / f"shot{shot['shot_index']:02d}.npz"
                     if source_edges is not None else None)
        result["source_edges_npz_sha256"] = np.asarray(
            hashlib.sha256(edge_file.read_bytes()).hexdigest() if edge_file is not None else "")
        result["source_edge_validation_scope"] = np.asarray(
            "pose_guided_measured_pixels_fit_check_not_holdout" if edge_file is not None else "")
        result["anchor_manifest_sha256"] = np.asarray(anchor_sha256)
        result["center_sigma_scene"] = np.asarray(center_sigma_scene)
        result["validation_scope"] = np.asarray(
            "unverified_candidate_review_only" if candidate_review else "anchored_tracking_needs_full_video_review")
        path = output_dir / f"shot{shot['shot_index']:02d}.npz"
        writing = path.with_name(path.stem + ".writing.npz")
        np.savez_compressed(writing, **result)
        writing.replace(path)
        counts = {label: int(np.count_nonzero(result["status"] == label))
                  for label in ("anchor", "keyframe", "observed", "predicted", "unavailable")}
        summary.append({"shot_index": shot["shot_index"], "camera_id": shot["camera_id"],
                        "path": str(path), "frames": len(result["pts"]),
                        "validation_scope": str(result["validation_scope"]),
                        "model_sha256": scene_sha256, "anchor_manifest_sha256": anchor_sha256,
                        "center_max_scene": float(np.nanmax(result["center_deviation_scene"]))
                        if np.isfinite(result["center_deviation_scene"]).any() else None,
                        **counts})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-index", type=Path,
                        default=DEFAULT_ROOT / "three_frame_refit/program_tracking/motion_index.json")
    parser.add_argument("--keyframes", type=Path, default=DEFAULT_TEMP / "keyframes/cameras.json")
    parser.add_argument("--scene", type=Path, default=DEFAULT_ROOT / "three_frame_refit/deliverables/scene.json")
    parser.add_argument("--release", type=Path, default=DEFAULT_ROOT / "three_frame_refit/deliverables/release.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEMP / "tracked")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--max-matches", type=int, default=120)
    parser.add_argument("--candidate-review", action="store_true",
                        help="Propagate explicitly unvalidated candidates for external-cache visual review")
    parser.add_argument("--temporal-keyframes", type=Path,
                        help="Directory of model/source-bound keyframe graphs with long image links")
    parser.add_argument("--source-edges", type=Path,
                        help="Directory of exact-PTS measured original-frame board edge NPZ files")
    parser.add_argument("--native-landmarks-config", type=Path,
                        help="Joint config whose original ice/ad source pixels will be reused")
    parser.add_argument("--native-landmarks-joint-dir", type=Path,
                        help="Matching solved joint directory with frozen ad geometry")
    parser.add_argument("--update-stride", type=int, default=1,
                        help="Insert this many native states and all observations per iSAM2 update")
    parser.add_argument("--graph-scope", choices=("fixed_lag", "whole_shot"),
                        default="fixed_lag",
                        help="whole_shot keeps every reviewed shot frame in one iSAM2 graph")
    parser.add_argument("--update-strategy", choices=("incremental", "deferred_whole"),
                        default="incremental",
                        help="deferred_whole inserts a complete half-shot before each iSAM2 solve")
    parser.add_argument("--deferred-refinement-updates", type=int, default=30,
                        help="Dogleg refinements after each deferred half-shot insertion")
    parser.add_argument("--lens-sigma-px", type=float, nargs=3, default=(8., 8., 20.),
                        help="Soft cx, cy and skew prior sigma around source calibration anchors")
    parser.add_argument("--focal-sigma-log", type=float, default=0.06,
                        help="Soft log-focal prior sigma between nearby solved keyframes")
    parser.add_argument("--focal-prior-mode", choices=("per_frame", "none"),
                        default="per_frame",
                        help="Whether to repeat the solved-keyframe focal guide on each native frame")
    parser.add_argument("--aspect-sigma-log", type=float, default=0.02,
                        help="Soft log(fx/fy) prior sigma around solved keyframe calibration")
    parser.add_argument("--pair-sigma-px", type=float, default=2.0,
                        help="Per-original-match optical-flow sigma in source pixels")
    parser.add_argument("--solver-refinement-updates", type=int, default=1,
                        help="Additional iSAM2 nonlinear updates per inserted native batch")
    parser.add_argument("--temporal-prior-sigmas", type=float, nargs=5,
                        default=(1.5, .0005, 1., 1., 1.),
                        metavar=("ROT_PX", "LOG_F", "CX_PX", "CY_PX", "SKEW_PX"),
                        help="Source-keyframe camera prior uncertainty; actual source landmarks remain factors")
    parser.add_argument("--solver-mode", choices=("gauss_newton", "dogleg"),
                        default="gauss_newton", help="iSAM2 nonlinear update method")
    parser.add_argument("--motion-model", choices=("fixed_mount_rays", "fixed_mount_surface_rays", "surface_planes"),
                        default="fixed_mount_rays")
    parser.add_argument("--center-sigma-scene", type=float, default=CENTER_SIGMA_SCENE,
                        help="Per-frame fixed-mount centre prior sigma in conditional scene units")
    parser.add_argument("--shot-index", type=int, action="append",
                        help="Track only these reviewed shot indices; may be repeated")
    args = parser.parse_args()
    for row in track_program(motion_index=args.motion_index, keyframes_json=args.keyframes,
                             scene_path=args.scene, release_path=args.release, output_dir=args.output_dir,
                             window_size=args.window_size, max_matches=args.max_matches,
                             center_sigma_scene=args.center_sigma_scene,
                             motion_model=args.motion_model,
                             candidate_review=args.candidate_review,
                             temporal_keyframes=args.temporal_keyframes,
                             lens_sigma_px=args.lens_sigma_px,
                             focal_sigma_log=args.focal_sigma_log,
                             aspect_sigma_log=args.aspect_sigma_log,
                             pair_sigma_px=args.pair_sigma_px,
                             solver_refinement_updates=args.solver_refinement_updates,
                             temporal_prior_sigmas=args.temporal_prior_sigmas,
                             solver_mode=args.solver_mode,
                             source_edges=args.source_edges,
                             native_landmarks_config=args.native_landmarks_config,
                             native_landmarks_joint_dir=args.native_landmarks_joint_dir,
                             update_stride=args.update_stride,
                             graph_scope=args.graph_scope,
                             update_strategy=args.update_strategy,
                             deferred_refinement_updates=args.deferred_refinement_updates,
                             focal_prior_mode=args.focal_prior_mode,
                             shot_indices=args.shot_index):
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
