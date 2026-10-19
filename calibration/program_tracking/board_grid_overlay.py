"""Finite board lattice from the same current scene and camera.

No solver, source observation, camera, scene coordinate or qualification is
changed.  Inject with ``grid.SceneRenderer = make_board_grid_renderer(
grid.SceneRenderer, scene_unit_m=body.metres_per_scene_unit)``.

When no explicit scale is supplied, use the scene's single closest60x30
conversion. Geometry and camera coordinates remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys

import cv2
import numpy as np
from rinkspace.metric_scale import boundary_span, estimate_uniform_scale


def _array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class BoardGridGeometry:
    """Fixed world paths with spacing from the same uniform metre conversion."""

    def __init__(self, scene, *, scene_unit_m=None, station_spacing_m=1.0):
        scale_policy = 'explicit_uniform_scene_to_metre_scale'
        if scene_unit_m is None:
            scene_unit_m = estimate_uniform_scale(boundary_span(scene))
            scale_policy = 'uniform_closest60x30_after_geometry'
        if not np.isfinite(scene_unit_m) or scene_unit_m <= 0:
            raise ValueError('Explicit positive source scene_unit_m is required')
        if not np.isfinite(station_spacing_m) or station_spacing_m <= 0:
            raise ValueError('Board station spacing must be positive')
        original_bottom = np.asarray(scene['ice_boundary']['points_xyz'], np.float64)
        original_top = np.asarray(scene['board_surface']['top_edge_xyz'], np.float64)
        if (original_bottom.ndim != 2 or original_bottom.shape[1:] != (3,)
                or original_top.shape != original_bottom.shape
                or len(original_bottom) < 3
                or not np.isfinite(original_bottom).all()
                or not np.isfinite(original_top).all()
                or not np.array_equal(original_bottom[:, :2], original_top[:, :2])):
            raise ValueError('Actual finite top/contact must pair exactly in count and XY')
        bottom, top = original_bottom.copy(), original_top.copy()
        if np.array_equal(bottom[0], bottom[-1]):
            if not np.array_equal(top[0], top[-1]):
                raise ValueError('Top/contact closure differs')
            bottom, top = bottom[:-1], top[:-1]
        elif np.array_equal(top[0], top[-1]):
            raise ValueError('Top/contact closure differs')
        if np.any(top[:, 2] <= bottom[:, 2]):
            raise ValueError('Every actual board top must be above its contact')
        self.bottom, self.top = bottom, top
        self.edge = np.roll(bottom[:, :2], -1, axis=0) - bottom[:, :2]
        lengths = np.linalg.norm(self.edge, axis=1)
        if np.any(lengths <= 0):
            raise ValueError('Actual finite board contains a degenerate facet')
        cumulative = np.r_[0., np.cumsum(lengths)]
        station_m = np.arange(0., cumulative[-1] * scene_unit_m, station_spacing_m)
        station = station_m / scene_unit_m
        facet = np.searchsorted(cumulative, station, side='right') - 1
        facet = np.minimum(facet, len(bottom)-1)
        t = (station - cumulative[facet]) / lengths[facet]
        b = bottom[facet] + t[:, None] * (np.roll(bottom, -1, axis=0)[facet]-bottom[facet])
        z = top[facet] + t[:, None] * (np.roll(top, -1, axis=0)[facet]-top[facet])
        # Original endpoint rows are preserved literally at the seam.
        b[0], z[0] = bottom[0], top[0]
        self.station_bottom, self.station_top = b, z
        self.station_m, self.station_facet, self.station_fraction = station_m, facet, t
        self.paths = []
        for fraction in (0., .25, .5, .75, 1.):
            ring = bottom if fraction == 0 else top if fraction == 1 else bottom + fraction*(top-bottom)
            self.paths.append(('board_horizontal', np.vstack((ring, ring[0]))))
        for first, last in zip(b, z):
            self.paths.append(('board_vertical', np.stack((first, last))))
        self.first = np.vstack([p[:-1] for _, p in self.paths])
        self.second = np.vstack([p[1:] for _, p in self.paths])
        self.kind = np.concatenate([np.full(len(p)-1, 0 if k == 'board_horizontal' else 1, np.int8)
                                    for k, p in self.paths])
        self.report = dict(
            schema='rinkspace.same-frozen-world-board-grid-overlay.v1',
            source_fields=['ice_boundary.points_xyz', 'board_surface.top_edge_xyz'],
            source_field_sha256={'ice_boundary.points_xyz': _array_sha(original_bottom),
                                 'board_surface.top_edge_xyz': _array_sha(original_top)},
            source_field_canonical_json_sha256=hashlib.sha256(json.dumps(
                {'ice_boundary.points_xyz': scene['ice_boundary']['points_xyz'],
                 'board_surface.top_edge_xyz': scene['board_surface']['top_edge_xyz']},
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            actual_finite_facets=len(bottom), original_world_coordinates_unchanged=True,
            top_contact_same_count_and_XY_exact=True, closed_finite_course=True,
            horizontal_height_fractions=[0., .25, .5, .75, 1.], horizontal_paths=5,
            vertical_paths=len(b), world_path_count=len(self.paths),
            original_horizontal_segments=5*len(bottom), original_vertical_segments=len(b),
            scene_unit_m=float(scene_unit_m), station_spacing_m=float(station_spacing_m),
            scaled_perimeter_m=float(cumulative[-1]*scene_unit_m),
            metre_scale_is_uniform_conversion_not_new_measurement=True,
            scale_policy=scale_policy,
            station_rule='arclength interpolation only within each actual finite original facet',
            confidence_or_fit_residual_gating=False, source_UV_sigma_roles_changed=False,
            camera_or_solver_changed=False, no_accuracy_claim=True)
        for a in (self.bottom, self.top, self.first, self.second, self.kind,
                  self.station_bottom, self.station_top, self.station_m,
                  self.station_facet, self.station_fraction):
            a.flags.writeable = False

    def unoccluded(self, camera, points):
        """Actual finite vertical wall intersections before the drawn point.

        This is geometric self-occlusion, independent of fit errors. Camera and
        points remain in source scene units; both sides of the wall are drawn.
        """
        points = np.asarray(points, float)
        C = np.asarray(camera[:3], float)
        # For an inside camera and the convex released course, no earlier wall
        # intersection exists; pointPolygonTest avoids the dense ray test.
        inside = cv2.pointPolygonTest(self.bottom[:, :2].astype(np.float32),
                                     (float(C[0]), float(C[1])), False)
        if inside >= 0:
            return np.ones(len(points), bool)
        result = np.ones(len(points), bool)
        e = self.edge
        w = self.bottom[:, :2]-C[:2]
        cross_w_e = w[:, 0]*e[:, 1]-w[:, 1]*e[:, 0]
        db = np.roll(self.bottom[:, 2], -1)-self.bottom[:, 2]
        dt = np.roll(self.top[:, 2], -1)-self.top[:, 2]
        for start in range(0, len(points), 256):
            p = points[start:start+256]
            d = p[:, :2]-C[:2]
            den = d[:, 0, None]*e[None, :, 1]-d[:, 1, None]*e[None, :, 0]
            nonparallel = np.abs(den) > 1e-14
            ray_t = np.divide(cross_w_e[None, :], den,
                              out=np.zeros_like(den), where=nonparallel)
            edge_t = np.divide(w[None, :, 0]*d[:, 1, None]-w[None, :, 1]*d[:, 0, None],
                               den, out=np.zeros_like(den), where=nonparallel)
            crossing = nonparallel & (ray_t > 1e-9) & (ray_t < 1.-1e-8) & (edge_t >= 0.) & (edge_t <= 1.)
            height = C[2]+ray_t*(p[:, 2, None]-C[2])
            lo = self.bottom[None, :, 2]+edge_t*db[None, :]
            hi = self.top[None, :, 2]+edge_t*dt[None, :]
            result[start:start+len(p)] = ~np.any(crossing & (height >= lo) & (height <= hi), axis=1)
        return result


def make_board_grid_renderer(base_renderer, *, scene_unit_m=None,
                             station_spacing_m=1.0, color=(255, 185, 70),
                             tolerance_px=.25, max_refinement_depth=10,
                             wall_self_occlusion=True):
    """Return SceneRenderer-compatible subclass with its existing byte-key LRU.

    The original ice/ads draw runs first. Board lines are painted afterwards;
    original person/graphics occlusion is then restored. The original wall mask
    and all original model lines/advertisement IDs retain their original draw.
    """
    module = sys.modules[base_renderer.__module__]
    if tolerance_px <= 0 or max_refinement_depth < 1:
        raise ValueError('Positive refinement controls required')

    class BoardGridSceneRenderer(base_renderer):
        def __init__(self, scene, *args, **kwargs):
            super().__init__(scene, *args, **kwargs)
            self.board_grid = BoardGridGeometry(scene, scene_unit_m=scene_unit_m,
                                                station_spacing_m=station_spacing_m)
            self.model_paths = tuple(self.model_paths) + tuple(self.board_grid.paths)
            self.path_counts = dict(self.path_counts,
                                   board_horizontal=5,
                                   board_vertical=len(self.board_grid.station_bottom))

        def _project_board_grid(self, camera, projection_context=None):
            a, b, kind = self.board_grid.first, self.board_grid.second, self.board_grid.kind
            commands = [[], []]
            unresolved = rejected = occluded = 0
            width, height = module.FRAME_SIZE
            fractions = np.array([0., .25, .5, .75, 1.])
            for level in range(max_refinement_depth+1):
                if not len(a):
                    break
                world = a[:, None, :]+fractions[None, :, None]*(b-a)[:, None, :]
                project=module.safe_project if projection_context is None else projection_context.project
                uv, depth, valid = project(camera, world.reshape(-1, 3))
                uv, depth, valid = uv.reshape(-1, 5, 2), depth.reshape(-1, 5), valid.reshape(-1, 5)
                valid &= np.isfinite(uv).all(axis=2) & (depth > .05)
                active = np.any(valid & (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width)
                                & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height), axis=1)
                # Retain a long valid chord that crosses the sensor even if
                # the five samples happen to fall outside the sensor.
                _, _, chord = module.clip_segments(uv[:, 0], uv[:, -1], width, height)
                active |= chord & valid[:, 0] & valid[:, -1]
                line = uv[:, -1]-uv[:, 0]
                length2 = np.sum(line*line, axis=1)
                t = np.clip(np.sum((uv[:, 1:4]-uv[:, :1])*line[:, None, :], axis=2)
                            /np.maximum(length2[:, None], 1e-30), 0., 1.)
                error = np.max(np.linalg.norm(uv[:, 1:4]-(uv[:, :1]+t[:, :, None]*line[:, None, :]), axis=2), axis=1)
                refine = active & (~valid.all(axis=1) | (error > tolerance_px))
                accept = active & valid.all(axis=1) & ~refine
                if level == max_refinement_depth:
                    unresolved += int(refine.sum())
                    # Never bridge an optical-domain transition. A valid
                    # unresolved curved leaf keeps all quarter samples.
                    accept |= refine & valid.all(axis=1)
                    refine[:] = False
                rejected += int((~active).sum())
                selected = np.flatnonzero(accept)
                if len(selected):
                    if wall_self_occlusion and projection_context is not None:
                        points=world[selected,2];row_uv,_,_=project(camera,points)
                        rows=np.clip(row_uv[:,1],0.,projection_context.context.image_height-1.)
                        row_cameras=projection_context.context.camera25_at_rows(rows,geometry_navigation=True)
                        visible=np.array([self.board_grid.unoccluded(c,p[None,:])[0] for c,p in zip(row_cameras,points,strict=True)])
                    else:
                        visible = (self.board_grid.unoccluded(camera, world[selected, 2])
                                   if wall_self_occlusion else np.ones(len(selected), bool))
                    occluded += int((~visible).sum())
                    selected = selected[visible]
                    for code in (0, 1):
                        ids = selected[kind[selected] == code]
                        if len(ids):
                            first = uv[ids, :-1].reshape(-1, 2)
                            last = uv[ids, 1:].reshape(-1, 2)
                            first, last, _ = module.clip_segments(first, last, width, height)
                            if len(first):
                                commands[code].append(np.rint(np.stack((first, last), axis=1)*256).astype(np.int32))
                selected = np.flatnonzero(refine)
                middle = .5*(a[selected]+b[selected])
                a, b, kind = (np.concatenate((a[selected], middle)),
                              np.concatenate((middle, b[selected])), np.tile(kind[selected], 2))
            paths = [np.ascontiguousarray(np.concatenate(parts), np.int32)
                     if parts else np.empty((0, 2, 2), np.int32) for parts in commands]
            for p in paths:
                p.flags.writeable = False
            return paths, dict(horizontal_drawn_segments=len(paths[0]), vertical_drawn_segments=len(paths[1]),
                               unresolved_sampled_curvature_leaves=unresolved,
                               rejected_behind_or_outside_leaves=rejected,
                               geometrically_self_occluded_leaves=occluded,
                               sampled_Brown_chord_tolerance_px=tolerance_px,
                               wall_self_occlusion=wall_self_occlusion)

        def _project_geometry(self, camera):
            geometry = super()._project_geometry(camera)
            paths, metrics = self._project_board_grid(camera)
            geometry['board_grid_paths'] = paths
            geometry['metrics']['board_grid'] = dict(self.board_grid.report, **metrics)
            geometry['array_bytes'] += sum(p.nbytes for p in paths)
            return geometry

        def _project_geometry_rows(self,camera,projection_context):
            geometry=super()._project_geometry_rows(camera,projection_context)
            paths,metrics=self._project_board_grid(camera,projection_context)
            geometry['board_grid_paths']=paths
            geometry['metrics']['board_grid']=dict(self.board_grid.report,**metrics)
            geometry['array_bytes']+=sum(p.nbytes for p in paths)
            return geometry

        def draw(self, frame, camera, person_mask=None, title=None, *, projection_context=None):
            if projection_context is None:
                image, metrics = super().draw(frame, camera, person_mask=person_mask, title=None)
            else:
                image, metrics = super().draw(frame,camera,person_mask=person_mask,title=None,projection_context=projection_context)
            geometry = self._geometry(camera,projection_context)
            for p in geometry['board_grid_paths']:
                if len(p):
                    cv2.polylines(image, p, False, color, 1, cv2.LINE_AA, shift=8)
            # Keep advertisement frames/IDs readable after the finer lattice.
            self._paint_advertisements(image, geometry['advertisements'])
            if person_mask is not None:
                cv2.copyTo(frame, person_mask.astype(np.uint8), image)
            for left, top, right, bottom in self.graphics_regions:
                image[top:bottom, left:right] = frame[top:bottom, left:right]
            if title:
                cv2.rectangle(image, (0, 0), (min(module.FRAME_SIZE[0], 1540), 58), (19, 29, 34), -1)
                cv2.putText(image, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, .83, (255, 255, 255), 2, cv2.LINE_AA)
            return image, metrics

    BoardGridSceneRenderer.__name__ = 'BoardGridSceneRenderer'
    return BoardGridSceneRenderer


def install_board_grid_renderer(grid_module, *, scene_unit_m, **kwargs):
    """Inject into a private rendering module; return original renderer class."""
    old = grid_module.SceneRenderer
    grid_module.SceneRenderer = make_board_grid_renderer(old, scene_unit_m=scene_unit_m, **kwargs)
    return old
