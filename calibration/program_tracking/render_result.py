"""Render a hash-matched v0.25 scene and grid on original programme frames.

Keyframe QA accepts unvalidated camera candidates and labels them as such.
Final video requires a complete native-PTS camera stream with accepted states.
Encoded intermediates must be placed outside OneDrive. Candidate videos remain
visibly unverified and may be saved in this task's result directory.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import av
import cv2
import numpy as np
from rinkspace.storage import AssetStore

try:
    from .projection import ground_xy, safe_project, scene_paths
    from .adaptive_board_curve import refine_board_paths, refine_vertical_edges
except ImportError:
    from projection import ground_xy, safe_project, scene_paths
    from adaptive_board_curve import refine_board_paths, refine_vertical_edges


ROOT = Path(__file__).resolve().parents[2]
STORAGE = AssetStore(ROOT)
DEFAULT_SCENE = ROOT / "calibration/deliverables/scene.json"
DEFAULT_RELEASE = ROOT / "calibration/deliverables/release.json"
DEFAULT_SHOTS = Path(__file__).with_name("source_shots.json")
DEFAULT_PEOPLE = Path("D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json")
DEFAULT_MASKS = Path("D:/CodexData/RinkSpace_program_tracking_tmp/people/masks")
DEFAULT_TRACKED = Path("D:/CodexData/RinkSpace_program_tracking_tmp/tracked")
DEFAULT_RENDER_TEMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp/render")
PROGRAMME_RESULT = STORAGE.output("programmes/okada_mei_sp/results")
DEFAULT_RESULT = PROGRAMME_RESULT / "okada_sp_grid_v025.mp4"
DEFAULT_REVIEW = DEFAULT_RENDER_TEMP / "okada_sp_candidate_review.mp4"
FRAME_SIZE = (1920, 1080)
# Optical flow's `observed` does not establish correct rink localization.
ACCEPTED_STATES = frozenset({"accepted", "validated", "visually_accepted"})
MASK_STATES = frozenset({"measured_mask", "no_large_person_detection"})
REVIEW_STATES = ACCEPTED_STATES | frozenset({"anchor", "keyframe", "observed", "predicted", "unavailable", "camera_candidate", "temporal_candidate"})


def read_scene(scene_file: Path, release_file: Path) -> tuple[dict, dict]:
    scene_file, release_file = map(STORAGE.resolve, (scene_file, release_file))
    data = scene_file.read_bytes()
    release = json.loads(release_file.read_text(encoding="utf-8"))
    digest = hashlib.sha256(data).hexdigest()
    if digest != release["model_sha256"] or release["version"] != "0.25":
        raise ValueError("Rink model differs from the released v0.25 scene")
    return json.loads(data), release


def camera_vector(record: dict) -> np.ndarray:
    value = record.get("camera_vector25", record.get("camera25", record.get("camera")))
    camera = np.asarray(value, dtype=np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError(f"Invalid 25-value camera at native PTS {record.get('pts')}")
    return camera


def clip_segments(first: np.ndarray, second: np.ndarray, width: int, height: int):
    """Clip finite forward-facing projected segments before integer conversion.

    A zoomed physical straight edge can cross the entire image even when
    both sampled vertices lie far outside it. Its length is not invalidity.
    """
    delta = second-first
    low = np.zeros(len(first))
    high = np.ones(len(first))
    good = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    for axis, limit in ((0, width-1), (1, height-1)):
        moving = np.abs(delta[:, axis]) > 1e-12
        good &= moving | ((first[:, axis] >= 0) & (first[:, axis] <= limit))
        a = np.divide(-first[:, axis], delta[:, axis],
                      out=np.full(len(first), -np.inf), where=moving)
        b = np.divide(limit-first[:, axis], delta[:, axis],
                      out=np.full(len(first), np.inf), where=moving)
        low = np.maximum(low, np.minimum(a, b))
        high = np.minimum(high, np.maximum(a, b))
    good &= high >= low
    return first[good]+low[good, None]*delta[good], first[good]+high[good, None]*delta[good], good


def clip_camera_segments_near_plane(first, second, near=.05):
    """Keep the exact forward part of each finite physical 3D segment."""
    first=np.asarray(first,float).copy();second=np.asarray(second,float).copy()
    good=np.isfinite(first).all(axis=1)&np.isfinite(second).all(axis=1)
    good&=(first[:,2]>=near)|(second[:,2]>=near)
    delta=second-first
    change_first=good&(first[:,2]<near)
    change_second=good&(second[:,2]<near)
    t=np.zeros(len(first));cross=change_first|change_second
    t[cross]=(near-first[cross,2])/delta[cross,2]
    crossing=first+t[:,None]*delta
    first[change_first]=crossing[change_first];second[change_second]=crossing[change_second]
    return first,second,good


def clip_polygon_halfspace(polygon,axis,limit,greater=True):
    """Clip a finite convex physical facet without inventing an extension."""
    polygon=np.asarray(polygon,float)
    if not len(polygon) or not np.isfinite(polygon).all():return np.empty((0,polygon.shape[-1]))
    output=[]
    for first,second in zip(polygon,np.roll(polygon,-1,axis=0)):
        a=(first[axis]>=limit) if greater else (first[axis]<=limit)
        b=(second[axis]>=limit) if greater else (second[axis]<=limit)
        if a:output.append(first)
        if a!=b:
            t=(limit-first[axis])/(second[axis]-first[axis]);output.append(first+t*(second-first))
    return np.asarray(output,float).reshape(-1,polygon.shape[-1])


def pinhole_camera_pixels(camera,local):
    normalized=local[:,:2]/local[:,2:]
    return np.column_stack((np.exp(camera[6])*normalized[:,0]+camera[10]*normalized[:,1]+camera[8],
                            np.exp(camera[7])*normalized[:,1]+camera[9]))


def clipped_pinhole_facet(camera,local,width,height):
    polygon=clip_polygon_halfspace(local,2,.05)
    if len(polygon)<3:return np.empty((0,2))
    polygon=pinhole_camera_pixels(camera,polygon)
    for axis,limit,greater in((0,0,True),(0,width-1,False),(1,0,True),(1,height-1,False)):
        polygon=clip_polygon_halfspace(polygon,axis,limit,greater)
        if len(polygon)<3:return np.empty((0,2))
    return polygon


def advertisement_boundary_paths(scene):
    """Use current solved corners on the same finite board as the ad factors.

    Historical candidate exports updated corners but left the old fitted-curve
    sidecar untouched. Rebuild only the existing sides; preserve every physical
    board vertex between the endpoints instead of drawing a corner-cutting chord.
    This changes rendering only, never the scene, source pixels or solver state.
    """
    try:
        from .ad_landmarks import AdGeometry
    except ImportError:
        from ad_landmarks import AdGeometry
    paths = []
    print_descriptor = scene.get('source_conditioned_board_prints') or {}
    explicit_print_ids = {path['number'] for path in source_conditioned_board_print_paths(scene)}
    retained_ids = print_descriptor.get('retain_outer_canvas_border_advertisement_ids')
    if retained_ids is None:
        retained_ids = explicit_print_ids if print_descriptor.get('retain_outer_canvas_border') else ()
    explicit_print_ids -= {int(number) for number in retained_ids}
    for ad in scene.get('advertisements', []):
        if int(ad['user_number']) in explicit_print_ids:
            # These real printed contours have one source/export producer;
            # a generic rectangle is neither their support nor a missing edge.
            continue
        geometry = AdGeometry(scene, int(ad['user_number']))
        state = geometry.initial
        corners = {role: geometry.corner(state, role) for role in geometry.roles}
        curves = {}
        for side in ('left', 'right'):
            if side+'_bottom' in corners and side+'_top' in corners:
                curves[side+'_vertical_xyz'] = np.linspace(corners[side+'_bottom'], corners[side+'_top'], 17)
        for level in ('top', 'bottom'):
            if 'left_'+level not in corners or 'right_'+level not in corners:
                continue
            start = geometry._base_s['left']
            end = start+geometry._width_sign*geometry._nominal_width
            low, high = sorted((start, end))
            stations = [start, end]
            for turn in range(int(np.floor(low/geometry._perimeter))-1,
                              int(np.ceil(high/geometry._perimeter))+1):
                stations.extend(float(s) for s in geometry._cum+turn*geometry._perimeter if low < s < high)
            stations = sorted(set(stations), reverse=end < start)
            samples = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(abs(b-a)/.25))+1))[:-1]
                                      for a, b in zip(stations[:-1], stations[1:])] + [np.array([end])])
            xy = np.array([geometry._point_at_arc(s) for s in samples])
            curves[level+'_edge_xyz'] = np.column_stack((xy, np.full(len(xy), corners['left_'+level][2])))
        for role, xyz in curves.items():
            paths.append(dict(number=int(ad['user_number']), role=role,
                              extent=ad.get('extent_scope', 'unspecified'), xyz=xyz))
    return paths


def shared_board_pattern_paths(scene):
    """Actual solved board-print world support, shared by camera and plan views.

    Circle polylines are engineering samples of the solver's shared stroke.
    Shield material points stay independent markers; no unseen outline or
    connecting triangle is created. Source pixels never enter this builder.
    """
    descriptor = scene.get('shared_board_patterns')
    if descriptor is None:
        return []
    if not isinstance(descriptor, dict) or descriptor.get('schema') != 'rinkspace.shared-native-board-pattern-candidate.v1':
        raise ValueError('Explicit shared board-pattern world descriptor required')
    circles = descriptor.get('circle_outer_stroke_paths_xyz_scene_units', {})
    shield = descriptor.get('shield_fixed_material_points_xyz_scene_units', {})
    if not isinstance(circles, dict) or not isinstance(shield, dict):
        raise ValueError('Shared pattern world paths must be explicitly user-numbered')
    paths = []
    for number, value in circles.items():
        number = int(number)
        xyz = np.asarray(value, np.float64)
        if (number not in (13, 26) or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 17
                or not np.isfinite(xyz).all() or not np.allclose(xyz[0], xyz[-1], rtol=0., atol=1e-8)):
            raise ValueError('Ad13/26 require the actual finite closed world stroke')
        paths.append(dict(number=number, role='circle_outer_stroke_xyz',
                          extent='printed_circle_outer_stroke', xyz=xyz.copy(),
                          primitive='polyline', pattern_kind='circle_outer_stroke'))
    roles = ('upper_left_printed_outline_corner', 'upper_right_printed_outline_corner',
             'bottom_printed_outline_tip')
    for number, value in shield.items():
        number = int(number)
        if number not in (12, 25) or not isinstance(value, dict) or set(value) != set(roles):
            raise ValueError('Ad12/25 require exactly their three confirmed material roles')
        xyz = np.asarray([value[role] for role in roles], np.float64)
        if xyz.shape != (3, 3) or not np.isfinite(xyz).all():
            raise ValueError('Three finite shared world material points required')
        paths.append(dict(number=number, role='fixed_material_points_xyz',
                          extent='three_confirmed_printed_material_points', xyz=xyz.copy(),
                          primitive='points', pattern_kind='fixed_material_points',
                          physical_point_roles=roles))
    by_number = {p['number']: p for p in paths}
    placement_binding = descriptor.get('placement_binding')
    if placement_binding is None:
        # Historical descriptors encode exact half-turn placement. This checks
        # their own stored geometry, not a user rule about print positions.
        for low, high in ((12, 25), (13, 26)):
            if low in by_number and high in by_number:
                a, b = by_number[low]['xyz'], by_number[high]['xyz']
                if a.shape != b.shape or not np.allclose(a*np.array([-1., -1., 1.]), b, rtol=0., atol=1e-8):
                    raise ValueError('Historical pattern descriptor has inconsistent half-turn geometry')
    elif placement_binding == 'same_template_independent_station_z.v1':
        offsets = descriptor.get('instance_station_z_offsets_m', {})
        material = descriptor.get('instance_material_coordinates_m', {})
        if not isinstance(offsets, dict) or set(offsets) != {'25', '26'} or not isinstance(material, dict):
            raise ValueError('Explicit independent pattern placement materials required')
        for high in ('25', '26'):
            offset = np.asarray(offsets[high], np.float64)
            if offset.shape != (2,) or not np.isfinite(offset).all():
                raise ValueError('Finite intrinsic station/Z placement offset required')
        a = np.asarray(material.get('shield12'), np.float64)
        b = np.asarray(material.get('shield25'), np.float64)
        if (a.shape != (3, 2) or b.shape != a.shape or not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or not np.allclose(b-a, offsets['25'], rtol=0., atol=1e-10)):
            raise ValueError('Ad12/25 must share their intrinsic print template at independent locations')
        a = np.asarray(material.get('circle13'), np.float64)
        b = np.asarray(material.get('circle26'), np.float64)
        if (a.shape != (3,) or b.shape != a.shape or not np.isfinite(a).all()
                or not np.isfinite(b).all() or a[2] <= 0. or a[2] != b[2]
                or not np.allclose(b[:2]-a[:2], offsets['26'], rtol=0., atol=1e-10)):
            raise ValueError('Ad13/26 must share their radius at independent locations')
    else:
        raise ValueError('Unknown board-pattern placement binding')
    return paths


def additional_board_circle_paths(scene):
    """Read actual solved Ad11/24 strokes without importing solver runtime.

    The descriptor supplies the complete closed XYZ curves used by fitting.
    These print strokes add neither an unseen advertisement rectangle nor a
    synthetic circular approximation made from its centre/radius metadata.
    """
    descriptor = scene.get('additional_board_circle_patterns')
    if descriptor is None:
        return []
    if (not isinstance(descriptor, dict) or descriptor.get('schema') !=
            'rinkspace.additional-independent-native-board-circles.v1'):
        raise ValueError('Actual independent board-circle scene descriptor required')
    circles = descriptor.get('circles')
    if not isinstance(circles, dict):
        raise ValueError('Actual board-circle curves must be explicitly numbered')
    paths = []
    for number, item in circles.items():
        n = int(number)
        xyz = np.asarray(item['circle_outer_stroke_xyz_scene_units'], np.float64)
        if (n not in (11, 24) or xyz.ndim != 2 or xyz.shape[1:] != (3,)
                or len(xyz) < 17 or not np.isfinite(xyz).all()
                or not np.array_equal(xyz[0], xyz[-1])):
            raise ValueError('Actual closed finite Ad11/24 print world required')
        paths.append(dict(number=n, role='circle_outer_stroke_xyz',
                          extent='printed_circle_outer_stroke', xyz=xyz.copy(),
                          primitive='polyline',
                          pattern_kind=('shared_radius_circle_outer_stroke'
                                        if descriptor.get('radius_shared')
                                        else 'independent_circle_outer_stroke')))
    return paths


def source_conditioned_board_print_paths(scene):
    """Exact solved free outlines for both camera and plan review views."""
    descriptor=scene.get('source_conditioned_board_prints')
    if descriptor is None:return []
    if descriptor.get('schema')!='rinkspace.current-source-conditioned-board-print-contours.v1':
        raise ValueError('Typed current source-conditioned print scene required')
    paths=[]
    for item in descriptor['patterns']:
        for contour in item['external_contours']:
            xyz=np.asarray(contour['xyz_scene_units'],np.float64)
            if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
                raise ValueError('Finite actual current print contour required')
            paths.append(dict(number=int(item['advertisement_id']),role=contour['role'],
                extent='source_conditioned_external_printed_component',xyz=xyz.copy(),
                primitive='polyline',pattern_kind='source_conditioned_free_print_contour'))
    return paths


def advertisement_review_paths(scene):
    """One actual ad-border and print inventory for both review views."""
    return (advertisement_boundary_paths(scene) + shared_board_pattern_paths(scene)
            + additional_board_circle_paths(scene)+source_conditioned_board_print_paths(scene))


# Legacy private-run installation can reuse this producer without appending
# the same solved print curves a second time to either view.
advertisement_review_paths._additional_native_board_circles = True


def fixed_scene_paths(scene, *, draw_grid=True):
    """One complete model inventory for both camera and plan review views.

    It is independent of image detections, frame confidence and fit quality.
    These paths are sampled from the supplied solved scene, without replacing
    missing observations or changing its geometry.
    """
    # A Lane polygon also retains proxy vertices for unresolved endpoints.
    # Its explicit open-edge inventory defines physical paint; closing it
    # would draw an unmeasured fourth side. Keep the parameter polygon in the
    # scene and build only its known painted sides for both review views.
    nonpolygon_scene = dict(scene, ice_features=[f for f in scene["ice_features"]
                                                if f["kind"] != "polygon"])
    paths = scene_paths(nonpolygon_scene, grid=False)
    if draw_grid:
        edges={}
        for cell in scene['ice_grid']['cells']:
            polygon=np.asarray(cell['polygon_xyz'],float)
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                if np.linalg.norm(a-b)<=1e-10:continue
                # Keep every finite modeled edge, including tiny boundary
                # cells. A world-length display cutoff could change the
                # inventory when the parent revises the rink width.
                key=tuple(sorted((tuple(np.round(a,9)),tuple(np.round(b,9)))))
                edges[key]=('grid',np.linspace(a,b,9))
        paths.extend(edges.values())
    for feature in scene["ice_features"]:
        if feature["kind"] == "polygon":
            polygon = np.asarray(feature["polygon_xyz"], float)
            omitted = set(feature.get("open_edges", []))
            if not omitted:
                paths.append(("marking", np.vstack((polygon, polygon[:1]))))
                continue
            for i, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
                if i not in omitted:
                    paths.append(("marking", np.linspace(a, b, 32)))
            continue
        if feature["kind"] != "circle":
            continue
        for ring in feature.get("rings", []):
            if ring["role"] == "outer":
                continue
            angles = np.linspace(0, 2*np.pi, 181)
            center = np.asarray(ring["center_xyz"], float)
            xyz = center + np.column_stack((np.cos(angles), np.sin(angles),
                                             np.zeros(len(angles)))) * float(ring["radius"])
            paths.append(("marking", xyz))
    return tuple((kind, np.asarray(xyz, float).copy()) for kind, xyz in paths)


class SceneRenderer:
    COLORS = {
        "grid": (85, 220, 65),       # BGR green
        "marking": (220, 55, 205),   # magenta
        "contact": (25, 150, 255),   # orange
        "board_top": (40, 235, 250), # yellow
    }

    def __init__(self, scene: dict, draw_grid: bool = True,
                 graphics_regions: tuple[tuple[int, int, int, int], ...] = (),
                 draw_advertisements: bool = True,
                 geometry_cache_size: int = 8):
        self.graphics_regions = graphics_regions
        if not isinstance(geometry_cache_size, int) or geometry_cache_size < 0:
            raise ValueError("Geometry cache size must be a nonnegative integer")
        # Each renderer/worker owns its bounded LRU. No native pixels, person
        # masks, titles or confidence enter this cache. Zero disables it for
        # exact replay/profile checks without a separate rendering algorithm.
        self.geometry_cache_size = geometry_cache_size
        self._geometry_cache = OrderedDict()
        self._geometry_cache_hits = self._geometry_cache_misses = 0
        self.advertisement_paths = []
        border = np.asarray(scene['ice_boundary']['points_xyz'], float)[:, :2]
        if len(border) > 1 and np.array_equal(border[0], border[-1]):
            border = border[:-1]
        edges = np.roll(border, -1, axis=0)-border
        edge_length2 = np.sum(edges*edges, axis=1)
        inward = np.column_stack((-edges[:, 1], edges[:, 0]))
        inward *= np.where(np.sum(inward*(border.mean(axis=0)-border), axis=1) >= 0, 1., -1.)[:, None]
        for path in advertisement_review_paths(scene) if draw_advertisements else []:
            xyz = path['xyz']
            delta = xyz[:, None, :2]-border[None, :, :]
            along = np.clip(np.sum(delta*edges, axis=2)/np.maximum(edge_length2, 1e-20), 0., 1.)
            nearest = border[None, :, :]+along[:, :, None]*edges
            facet = np.argmin(np.sum((xyz[:, None, :2]-nearest)**2, axis=2), axis=1)
            self.advertisement_paths.append(dict(path, inward=inward[facet].copy()))
        self.advertisement_offsets = np.r_[0, np.cumsum([len(p['xyz']) for p in self.advertisement_paths])]
        self.advertisement_xyz = (np.vstack([p['xyz'] for p in self.advertisement_paths])
                                  if self.advertisement_paths else np.empty((0, 3)))
        # This same fixed inventory also drives the right-hand plan map.
        self.model_paths = fixed_scene_paths(scene, draw_grid=draw_grid)
        paths = self.model_paths
        self.kinds = [kind for kind, _ in paths]
        self.path_counts = {kind: self.kinds.count(kind) for kind in self.COLORS}
        self.offsets = np.r_[0, np.cumsum([len(xyz) for _, xyz in paths])]
        self.xyz = np.concatenate([xyz for _, xyz in paths])
        self.segment_kind = np.full(len(self.xyz)-1, -1, np.int8)
        self.kind_codes = {kind: index for index, kind in enumerate(self.COLORS)}
        for kind, start, end in zip(self.kinds, self.offsets[:-1], self.offsets[1:]):
            self.segment_kind[start:end-1] = self.kind_codes[kind]
        if self.kinds[:2] != ["contact", "board_top"] or self.offsets[1] != self.offsets[2] - self.offsets[1]:
            raise ValueError("The released ice boundary and board top do not pair point-for-point")
        # The released vertices are the geometry; Brown refinement only adds
        # rendering samples on these same world segments for each camera.
        self.board_contact_xyz = self.xyz[:self.offsets[1]-1].copy()
        self.board_top_xyz = self.xyz[self.offsets[1]:self.offsets[2]-1].copy()

    def _advertisement_geometry(self, camera, projection_context=None):
        """Project unchanged finite ad borders and labels, independent of frame."""
        commands, labels, markers = [], [], []
        result = dict(mode='current_solved_corners_on_finite_board', modeled_ids=sorted({p['number'] for p in self.advertisement_paths}),
                      projected_ids=[], segments=0, extents={}, markers=0,
                      shared_pattern_modeled_ids=sorted({p['number'] for p in self.advertisement_paths if 'pattern_kind' in p}),
                      shared_pattern_projected_ids=[], shield_points_connected=False,
                      source_confidence_or_fit_residual_gating=False)
        if not len(self.advertisement_xyz):
            return dict(paths=commands, labels=labels, markers=np.empty((0, 2), np.int32), metrics=result)
        project = safe_project if projection_context is None else projection_context.project
        uv, depth, valid = project(camera, self.advertisement_xyz)
        valid &= np.isfinite(uv).all(axis=1) & (depth > .05)
        pinhole=projection_context is None and np.max(np.abs(camera[11:25]))<=1e-14
        if pinhole:
            rotation=cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
            local=(self.advertisement_xyz-camera[:3])@rotation.T
        label_points = {}
        known_extents = {}
        for path in self.advertisement_paths:
            known_extents.setdefault(str(path['number']), path['extent'])
        for p, a, b in zip(self.advertisement_paths, self.advertisement_offsets[:-1], self.advertisement_offsets[1:]):
            # Advertisements are printed on the rink-facing side of the wall.
            # Do not draw their reverse side through the near boards.
            centres = camera[:3] if projection_context is None else projection_context.centres(camera,p['xyz'])
            front = np.sum((centres[...,:2]-p['xyz'][:, :2])*p['inward'], axis=1) > 0
            good = valid[a:b] & front
            if p.get('primitive') == 'points':
                visible = good & (uv[a:b, 0] >= 0) & (uv[a:b, 0] < FRAME_SIZE[0]) & (uv[a:b, 1] >= 0) & (uv[a:b, 1] < FRAME_SIZE[1])
                points = uv[a:b][visible]
                if len(points):
                    markers.extend(np.rint(points).astype(np.int32))
                    result['markers'] += len(points)
                    result['extents'][str(p['number'])] = known_extents[str(p['number'])]
                    label_points.setdefault(p['number'], []).append(points)
                continue
            if projection_context is not None:
                forward=front[:-1]&front[1:]
                lines,_=projection_context.segments(camera,p['xyz'][:-1][forward],p['xyz'][1:][forward])
                first,second=lines[:,0],lines[:,1];segments=np.ones(len(lines),bool)
            elif pinhole:
                first,second,forward=clip_camera_segments_near_plane(local[a:b-1],local[a+1:b])
                segments=forward&front[:-1]&front[1:]
                first=pinhole_camera_pixels(camera,first[segments]);second=pinhole_camera_pixels(camera,second[segments])
            else:
                segments = good[:-1] & good[1:]
                first,second=uv[a:b-1][segments],uv[a+1:b][segments]
            if not segments.any():
                continue
            begin, end, _ = clip_segments(first,second,*FRAME_SIZE)
            if not len(begin):
                continue
            paths = np.ascontiguousarray(np.rint(np.stack((begin, end), axis=1)*256), np.int32)
            commands.append(paths)
            result['segments'] += len(begin)
            result['extents'][str(p['number'])] = known_extents[str(p['number'])]
            label_points.setdefault(p['number'], []).extend((begin, end))
        for number, pieces in label_points.items():
            points = np.vstack(pieces)
            anchor = points[np.argmin(points[:, 1])]
            extent = result['extents'][str(number)]
            label = f'ad{number}' + (' inner' if extent == 'internal_graphic_only' else ' edge' if extent == 'observed_edge_only' else '')
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
            x = int(np.clip(anchor[0]+5, 2, FRAME_SIZE[0]-size[0]-4))
            y = int(np.clip(anchor[1]+19, size[1]+2, FRAME_SIZE[1]-4))
            labels.append((label, (x, y)))
        result['projected_ids'] = sorted(label_points)
        result['shared_pattern_projected_ids'] = sorted(set(label_points)&set(result['shared_pattern_modeled_ids']))
        return dict(paths=commands, labels=labels, markers=np.asarray(markers, np.int32).reshape(-1, 2), metrics=result)

    @staticmethod
    def _paint_advertisements(image, geometry):
        color = (255, 215, 35)
        for paths in geometry['paths']:
            cv2.polylines(image, paths, False, color, 2, cv2.LINE_AA, shift=8)
        for marker in geometry.get('markers', ()):
            cv2.drawMarker(image, tuple(map(int, marker)), color, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
        for label, anchor in geometry['labels']:
            cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .55, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)

    def _draw_advertisements(self, image, camera):
        geometry = self._advertisement_geometry(camera)
        self._paint_advertisements(image, geometry)
        return geometry['metrics']

    def _project_geometry(self, camera):
        adaptive = None
        count = int(self.offsets[1])
        segment_kind = self.segment_kind
        if np.max(np.abs(camera[11:25])) <= 1e-14:
            # Zero Brown coefficients give the ordinary central pinhole
            # map everywhere forward. Avoid computing nine radial Jacobian
            # samples for every grid vertex in this common camera family.
            rotation = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
            local = (self.xyz-camera[:3]) @ rotation.T
            depth = local[:, 2]
            valid = np.isfinite(local).all(axis=1) & (depth > .05)
            xy = np.full((len(local), 2), np.nan)
            xy[valid] = local[valid, :2]/depth[valid, None]
            uv = np.column_stack((np.exp(camera[6])*xy[:, 0]+camera[10]*xy[:, 1]+camera[8],
                                  np.exp(camera[7])*xy[:, 1]+camera[9]))
        else:
            adaptive = refine_board_paths(
                camera, self.board_top_xyz, self.board_contact_xyz,
                tolerance_px=.125, image_size_wh=FRAME_SIZE)
            count = len(adaptive.top_xyz)
            # The paired refined rings drive both the drawn contours and the
            # wall mask. All other scene paths retain their released vertices.
            tail_uv, tail_depth, tail_valid = safe_project(camera, self.xyz[self.offsets[2]:])
            uv = np.concatenate((adaptive.contact_uv, adaptive.top_uv, tail_uv))
            valid = np.r_[adaptive.contact_valid, adaptive.top_valid, tail_valid]
            depth = np.r_[np.where(adaptive.contact_valid, 1., 0.),
                          np.where(adaptive.top_valid, 1., 0.), tail_depth]
            segment_kind = np.full(len(uv)-1, -1, np.int8)
            segment_kind[:count-1] = self.kind_codes["contact"]
            segment_kind[count:2*count-1] = self.kind_codes["board_top"]
            segment_kind[2*count:] = self.segment_kind[self.offsets[2]:]
        valid = valid & np.isfinite(uv).all(axis=1) & (depth > 0.05)
        grid_paths = {}
        segments = {key: 0 for key in self.COLORS}
        width, height = FRAME_SIZE
        # Submit independent clipped two-point paths in four batches.
        # Positive-depth edges remain visible even at extreme zoom.
        first, second = uv[:-1], uv[1:]
        segment_good = valid[:-1] & valid[1:] & (segment_kind >= 0)
        near_clipped_segments=0
        if adaptive is None:
            # Endpoint validity alone deletes a visible Lane/centreline when
            # its other endpoint lies behind the camera. Clip in 3D before
            # perspective division, then let the ordinary viewport clip retain
            # every visible part. Sampling density is not the coverage rule.
            ca,cb,forward=clip_camera_segments_near_plane(local[:-1],local[1:])
            near_clipped_segments=int(((segment_kind>=0)&forward&~segment_good).sum())
            def image_points(xyz):
                normalized=xyz[:,:2]/np.maximum(xyz[:,2:],.05)
                return np.column_stack((np.exp(camera[6])*normalized[:,0]+camera[10]*normalized[:,1]+camera[8],
                    np.exp(camera[7])*normalized[:,1]+camera[9]))
            first,second=image_points(ca),image_points(cb)
            segment_good=forward&(segment_kind>=0)
        segment_good &= (np.minimum(first[:, 0], second[:, 0]) < width)
        segment_good &= (np.maximum(first[:, 0], second[:, 0]) >= 0)
        segment_good &= (np.minimum(first[:, 1], second[:, 1]) < height)
        segment_good &= (np.maximum(first[:, 1], second[:, 1]) >= 0)
        for kind, code in self.kind_codes.items():
            selected = segment_good & (segment_kind == code)
            if not selected.any():
                continue
            begin, end, clipped = clip_segments(first[selected], second[selected], width, height)
            if not len(begin):
                continue
            # Quantize only the raster command, at 1/256 native pixel.
            paths = np.ascontiguousarray(np.rint(np.stack((begin, end), axis=1)*256), np.int32)
            grid_paths[kind] = paths
            segments[kind] = int(clipped.sum())
        # The solid board wall is in front of any ice/grid ray that appears in
        # its image footprint.  Restore original board pixels, then show the
        # actual board top.  This also handles near-side boards as the pan moves.
        bottom = uv[:count]
        top = uv[count:2 * count]
        board_good = valid[:count] & valid[count:2 * count]
        board_mask = np.zeros((height, width), np.uint8)
        vertical_edges = (refine_vertical_edges(camera, adaptive, image_size_wh=FRAME_SIZE)
                          if adaptive is not None else None)
        skipped_wall_facets = 0
        near_clipped_wall_facets=0
        skipped_reasons = dict(behind=0, sampled_offscreen_guard=0, unknown_crossing=0)
        if adaptive is not None:
            rotation = cv2.Rodrigues(np.ascontiguousarray(camera[3:6]))[0]
            wall_depth = np.column_stack(((adaptive.top_xyz-camera[:3]) @ rotation[2],
                                          (adaptive.contact_xyz-camera[:3]) @ rotation[2]))
        for index in range(count):
            nxt = (index + 1) % count
            if adaptive is None:
                facet=local[[index,nxt,count+nxt,count+index]]
                near_clipped_wall_facets+=int(np.any(facet[:,2]<.05)&np.any(facet[:,2]>=.05))
                polygon=clipped_pinhole_facet(camera,facet,width,height)
                if len(polygon)>=3:cv2.fillConvexPoly(board_mask,np.rint(polygon).astype(np.int32),1)
                continue
            if not (board_good[index] and board_good[nxt]):
                if adaptive is not None:
                    skipped_wall_facets += 1
                    reason = ('behind' if np.all(wall_depth[[index, nxt]] <= 0)
                              else 'unknown_crossing')
                    skipped_reasons[reason] += 1
                continue
            if vertical_edges is None:
                polygon = np.asarray([bottom[index], bottom[nxt], top[nxt], top[index]])
            else:
                side0, side1 = vertical_edges[index], vertical_edges[nxt]
                if not (side0.valid.all() and side1.valid.all()):
                    skipped_wall_facets += 1
                    skipped_reasons['unknown_crossing'] += 1
                    continue
                # top_i -> top_next -> curved side_next -> bottom_i ->
                # curved side_i in reverse. Adjacent facets reuse the same
                # vertical side vertices, with no crack between polygons.
                polygon = np.concatenate((top[index:index+1], side1.uv,
                                          bottom[index:index+1],
                                          side0.uv[-2:0:-1]))
            if not np.isfinite(polygon).all() or np.max(np.abs(polygon)) > 100000:
                if adaptive is not None:
                    skipped_wall_facets += 1
                    finite = np.isfinite(polygon).all()
                    outside = finite and (np.max(polygon[:, 0]) < 0
                        or np.min(polygon[:, 0]) >= width or np.max(polygon[:, 1]) < 0
                        or np.min(polygon[:, 1]) >= height)
                    skipped_reasons['sampled_offscreen_guard' if outside
                                    else 'unknown_crossing'] += 1
                continue
            integer_polygon = np.rint(polygon).astype(np.int32)
            if adaptive is not None:
                cv2.fillPoly(board_mask, [integer_polygon], 1)
            else:
                cv2.fillConvexPoly(board_mask, integer_polygon, 1)
        if adaptive is None:
            top_good=segment_good&(segment_kind==self.kind_codes['board_top'])
            begin,end,_=clip_segments(first[top_good],second[top_good],width,height)
        else:
            top_good = board_good[:-1] & board_good[1:]
            begin, end, _ = clip_segments(top[:-1][top_good], top[1:][top_good], width, height)
        top_paths = (np.ascontiguousarray(np.rint(np.stack((begin, end), axis=1)*256), np.int32)
                     if len(begin) else np.empty((0, 2, 2), np.int32))
        advertisements = self._advertisement_geometry(camera)
        advertisement_metrics = advertisements['metrics']
        forward_segment = (forward if adaptive is None else valid[:-1]&valid[1:]) & (segment_kind >= 0)
        metrics = {"drawn_segments": segments, "valid_scene_points": int(valid.sum()),
                   "advertisements": advertisement_metrics,
                   "fixed_model_path_counts": self.path_counts,
                   "visibility_policy": "all_modeled_paths_every_frame; geometric_projection_and_real_occlusion_only",
                   "confidence_gating": False,
                   "near_plane_clipped_segments": near_clipped_segments,
                   "near_plane_clipped_wall_facets":near_clipped_wall_facets,
                   "geometric_rejection_counts": {
                       "invalid_or_behind_segment": int(((segment_kind >= 0) & ~forward_segment).sum()),
                       "outside_image_segment": int((forward_segment & ~segment_good).sum())},
                   "occlusion": {"projected_board_pixels": int(np.count_nonzero(board_mask)),
                       "source_foreground_pixels": 0,
                       "broadcast_graphics_regions": len(self.graphics_regions)}}
        if adaptive is not None:
            metrics["board_refinement"] = {
                "inserted_points": int(len(adaptive.top_xyz)-len(self.board_top_xyz)-1),
                "unresolved_leaves": int(sum(row["unresolved_leaf_count"]
                                             for row in adaptive.segment_report)),
                "unresolved_vertical_leaves": int(sum(edge.unresolved_leaves
                                                       for edge in vertical_edges[:-1])),
                "skipped_wall_facets": skipped_wall_facets,
                "skipped_wall_facet_reasons": skipped_reasons,
            }
        arrays = [board_mask, top_paths, *grid_paths.values(), *advertisements['paths'], advertisements['markers']]
        for array in arrays:
            array.flags.writeable = False
        return dict(grid_paths=grid_paths, board_mask=board_mask, top_paths=top_paths,
                    advertisements=advertisements, metrics=metrics,
                    array_bytes=sum(array.nbytes for array in arrays))

    def _project_geometry_rows(self, camera, projection_context):
        """Same complete world inventory on the implicit native row camera."""
        grid_paths={};segments={key:0 for key in self.COLORS};near=unresolved=0
        for kind,world in self.model_paths:
            if kind not in self.kind_codes:continue
            lines,detail=projection_context.segments(camera,world[:-1],world[1:])
            near+=detail['near_plane_retained_leaves'];unresolved+=detail['unresolved_leaves']
            if not len(lines):continue
            first,last,clipped=clip_segments(lines[:,0],lines[:,1],*FRAME_SIZE)
            if not len(first):continue
            commands=np.ascontiguousarray(np.rint(np.stack((first,last),axis=1)*256),np.int32)
            grid_paths.setdefault(kind,[]).append(commands);segments[kind]+=int(clipped.sum())
        grid_paths={kind:np.concatenate(parts) for kind,parts in grid_paths.items()}
        board_mask=np.zeros((FRAME_SIZE[1],FRAME_SIZE[0]),np.uint8);near_facets=0;wall_unresolved=wall_subdivided=0
        from .row_scene_projection import clip_viewport_polygon
        for i in range(len(self.board_contact_xyz)):
            j=(i+1)%len(self.board_contact_xyz)
            world=np.array([self.board_contact_xyz[i],self.board_contact_xyz[j],self.board_top_xyz[j],self.board_top_xyz[i]])
            _,depth,_=projection_context.project(camera,world)
            near_facets+=int(np.any(depth<.05)&np.any(depth>=.05))
            polygons,detail=projection_context.wall_polygons(camera,world)
            wall_unresolved+=detail['unresolved_leaves'];wall_subdivided+=int(detail['subdivided'])
            for polygon in polygons:
                # Viewport clip before conversion, including fallback triangles.
                if len(polygon)>=3 and np.isfinite(polygon).all():
                    polygon=clip_viewport_polygon(polygon,FRAME_SIZE)
                    if len(polygon)>=3:cv2.fillPoly(board_mask,[np.rint(polygon).astype(np.int32)],1)
        top_paths=grid_paths.get('board_top',np.empty((0,2,2),np.int32))
        advertisements=self._advertisement_geometry(camera,projection_context)
        metrics=dict(drawn_segments=segments,valid_scene_points=int(projection_context.project(camera,self.xyz)[2].sum()),
            advertisements=advertisements['metrics'],fixed_model_path_counts=self.path_counts,
            visibility_policy='all_modeled_paths_every_frame; geometric_projection_and_real_occlusion_only',confidence_gating=False,
            near_plane_clipped_segments=near,near_plane_clipped_wall_facets=near_facets,
            geometric_rejection_counts=dict(unresolved_row_curve_leaves=unresolved,unresolved_row_wall_leaves=wall_unresolved,subdivided_row_wall_facets=wall_subdivided),
            row_projection=projection_context.provenance(),occlusion=dict(projected_board_pixels=int(np.count_nonzero(board_mask)),source_foreground_pixels=0,broadcast_graphics_regions=len(self.graphics_regions)))
        arrays=[board_mask,top_paths,*grid_paths.values(),*advertisements['paths'],advertisements['markers']]
        for array in arrays:array.flags.writeable=False
        return dict(grid_paths=grid_paths,board_mask=board_mask,top_paths=top_paths,advertisements=advertisements,metrics=metrics,array_bytes=sum(a.nbytes for a in arrays))

    def _geometry(self, camera, projection_context=None):
        camera = np.ascontiguousarray(camera)
        if camera.shape != (25,) or not np.isfinite(camera).all():
            raise ValueError("Renderer requires a finite 25-value camera")
        # No rounding, exposure/fps inference, approximate equality or camera
        # model reduction. Aliases hit only when all physical values match.
        key = (camera.dtype.str, camera.tobytes())
        if projection_context is not None:key=(*key,projection_context.cache_key)
        if self.geometry_cache_size and key in self._geometry_cache:
            self._geometry_cache_hits += 1
            self._geometry_cache.move_to_end(key)
            return self._geometry_cache[key]
        self._geometry_cache_misses += 1
        geometry = (self._project_geometry(camera) if projection_context is None
                    else self._project_geometry_rows(camera,projection_context))
        if self.geometry_cache_size:
            self._geometry_cache[key] = geometry
            while len(self._geometry_cache) > self.geometry_cache_size:
                self._geometry_cache.popitem(last=False)
        return geometry

    def geometry_cache_info(self):
        return dict(hits=self._geometry_cache_hits, misses=self._geometry_cache_misses,
                    entries=len(self._geometry_cache), maximum_entries=self.geometry_cache_size,
                    array_bytes=sum(g['array_bytes'] for g in self._geometry_cache.values()))

    def draw(self, frame: np.ndarray, camera: np.ndarray, person_mask: np.ndarray | None = None,
             title: str | None = None, *, projection_context=None) -> tuple[np.ndarray, dict]:
        if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            raise ValueError("Renderer requires original 1920x1080 frames")
        geometry = self._geometry(camera,projection_context)
        layer = frame.copy()
        for kind, paths in geometry['grid_paths'].items():
            cv2.polylines(layer, paths, False, self.COLORS[kind],
                          1 if kind == 'grid' else 2, lineType=cv2.LINE_AA, shift=8)
        image = cv2.addWeighted(layer, 0.78, frame, 0.22, 0)
        cv2.copyTo(frame, geometry['board_mask'], image)
        if len(geometry['top_paths']):
            cv2.polylines(image, geometry['top_paths'], False, self.COLORS["board_top"], 2, cv2.LINE_AA, shift=8)
        self._paint_advertisements(image, geometry['advertisements'])
        if person_mask is not None:
            if person_mask.shape != frame.shape[:2]:
                raise ValueError("Person mask size differs from native video")
            cv2.copyTo(frame, person_mask.astype(np.uint8), image)
        for left, top, right, bottom in self.graphics_regions:
            image[top:bottom, left:right] = frame[top:bottom, left:right]
        if title:
            cv2.rectangle(image, (0, 0), (min(FRAME_SIZE[0], 1540), 58), (19, 29, 34), -1)
            cv2.putText(image, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.83,
                        (255, 255, 255), 2, cv2.LINE_AA)
        # Metrics/occlusion describe this native frame. Callers cannot mutate
        # a cached list/dict and affect the next alias presentation.
        metrics = deepcopy(geometry['metrics'])
        metrics['occlusion']['source_foreground_pixels'] = (0 if person_mask is None
            else int(np.count_nonzero(person_mask)))
        metrics['occlusion']['broadcast_graphics_regions'] = len(self.graphics_regions)
        return image, metrics


class PersonMasks:
    def __init__(self, mask_dir: Path, people_path: Path):
        mask_dir, people_path = map(STORAGE.resolve, (mask_dir, people_path))
        self.mask_dir = mask_dir
        people = json.loads(people_path.read_text(encoding="utf-8"))
        self.people_manifest = people
        self.boxes = {int(item["pts"]): item["boxes"] for item in people["frames"]}
        self.lookup = {}
        self.active_path = None
        self.active = None
        for path in sorted(mask_dir.glob("*.npz")):
            try:
                with np.load(path) as data:
                    pts = data["pts"]
                    if tuple(data["shape"].tolist()) != (540, 960):
                        raise ValueError(f"Wrong SAM2 mask dimensions in {path}")
                    for index, (stamp, status) in enumerate(zip(pts, data["status"])):
                        stamp = int(stamp)
                        if stamp in self.lookup:
                            raise ValueError(f"Duplicate SAM2 mask PTS {stamp}")
                        self.lookup[stamp] = (path, index, str(status))
            except ValueError:
                raise
            except Exception:
                # A mask chunk may still be in the middle of being written.
                continue

    def validate_coverage(self, expected_pts: np.ndarray) -> dict[str, int]:
        expected = set(map(int, expected_pts))
        if set(self.boxes) != expected or len(self.people_manifest["frames"]) != len(expected):
            raise ValueError("Person detections do not cover the exact native source frames")
        missing = expected - self.lookup.keys()
        unexpected = self.lookup.keys() - expected
        invalid = {pts for pts in expected & self.lookup.keys()
                   if self.lookup[pts][2] not in MASK_STATES}
        if missing or unexpected or invalid:
            raise ValueError(
                f"SAM2 mask coverage invalid: missing={len(missing)}, "
                f"unexpected={len(unexpected)}, invalid_status={len(invalid)}; "
                f"examples={sorted(missing | unexpected | invalid)[:5]}"
            )
        return {state: sum(item[2] == state for item in self.lookup.values())
                for state in sorted(MASK_STATES)}

    def get(self, pts: int, require_sam2: bool = False) -> tuple[np.ndarray, str]:
        item = self.lookup.get(pts)
        if item is not None:
            path, index, status = item
            if require_sam2 and status not in MASK_STATES:
                raise ValueError(f"Invalid SAM2 mask status at native PTS {pts}: {status}")
            if self.active_path != path:
                # NPZ members are decompressed on every indexing operation.
                # Keep one packed chunk in RAM when reading consecutive frames.
                with np.load(path) as chunk:
                    self.active = {name: chunk[name] for name in ('pts', 'masks')}
                self.active_path = path
            data = self.active
            if index < len(data["pts"]) and int(data["pts"][index]) == pts:
                packed = data["masks"][index]
                mask = np.unpackbits(packed, axis=1)[:, :960].astype(np.uint8)
                mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
                mask = cv2.resize(mask, FRAME_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)
                return mask, status
        if require_sam2:
            raise ValueError(f"No measured SAM2 person mask at native PTS {pts}")
        mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), np.uint8)
        for box in self.boxes.get(pts, []):
            x1, y1, x2, y2 = [int(value * 2) for value in box[:4]]
            cv2.rectangle(mask, (max(0, x1 - 28), max(0, y1 - 28)),
                          (min(FRAME_SIZE[0] - 1, x2 + 28), min(FRAME_SIZE[1] - 1, y2 + 28)), 1, -1)
        return mask.astype(bool), "box_fallback"

    def static_scene_exclusion(self, pts: int) -> tuple[np.ndarray, str]:
        """Union the measured segmentation and every detected person box.

        A valid SAM2 chunk does not establish that small spectators or staff
        were segmented. Source measurements must also retain all detector
        exclusions. This method never changes the historical mask files.
        """
        mask, status = self.get(pts, require_sam2=True)
        mask = np.asarray(mask, bool).copy()
        if pts not in self.boxes:
            raise ValueError(f"Missing original person detections at PTS {pts}")
        for box in self.boxes[pts]:
            xy = np.asarray(box[:4], float)
            if xy.shape != (4,) or not np.isfinite(xy).all():
                raise ValueError(f"Invalid original person box at PTS {pts}")
            # Same native-pixel box support as the existing box fallback.
            x1, y1, x2, y2 = np.rint(xy * 2).astype(int)
            left, top = max(0, x1-28), max(0, y1-28)
            right, bottom = min(FRAME_SIZE[0], x2+29), min(FRAME_SIZE[1], y2+29)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = True
        return mask, status + "+all_detected_person_boxes"


def decode_selected(source: Path, pts_values: list[int]):
    with av.open(str(STORAGE.source(source))) as container:
        stream = container.streams.video[0]
        for target in sorted(set(pts_values)):
            container.seek(max(0, target - 15360), stream=stream, backward=True)
            frame = next((frame for frame in container.decode(stream) if frame.pts is not None and frame.pts >= target), None)
            if frame is None or frame.pts != target:
                raise ValueError(f"Could not decode original native PTS {target}")
            yield target, frame.to_ndarray(format="bgr24")


def select_qa_frames(shots: list[dict], candidates: list[dict]) -> list[dict]:
    selected = []
    for index, shot in enumerate(shots):
        within = [record for record in candidates if record.get("status") == "camera_candidate"
                  and shot["start_pts"] <= record["pts"] < shot["end_pts"]]
        if not within:
            selected.append({"shot_index": index, "labels": [], "camera_id": shot["camera_id"]})
            continue
        ranked = {
            "first": min(within, key=lambda record: record["pts"]),
            "best": max(within, key=lambda record: (record.get("ice_inliers", 0), record.get("inliers", 0))),
            "last": max(within, key=lambda record: record["pts"]),
        }
        selected.append({"shot_index": index, "camera_id": shot["camera_id"],
                         "labels": [{"role": label, **record} for label, record in ranked.items()]})
    return selected


def keyframe_qa(
    camera_file: Path, shot_file: Path, scene_file: Path, release_file: Path,
    people_file: Path, mask_dir: Path, output_dir: Path,
) -> dict:
    camera_file, shot_file, scene_file, release_file, people_file, mask_dir, output_dir = map(
        STORAGE.resolve, (camera_file, shot_file, scene_file, release_file, people_file, mask_dir, output_dir))
    if STORAGE.configured and not output_dir.is_relative_to(STORAGE.data):
        raise ValueError("Review products must use the registered data root")
    scene, release = read_scene(scene_file, release_file)
    shots_data = json.loads(shot_file.read_text(encoding="utf-8"))
    candidates = json.loads(camera_file.read_text(encoding="utf-8"))
    picks = select_qa_frames(shots_data["shots"], candidates["frames"])
    images = {pts: image for pts, image in decode_selected(
        Path(shots_data["source_path"]), [item["pts"] for shot in picks for item in shot["labels"]])}
    renderer = SceneRenderer(scene)
    people = PersonMasks(mask_dir, people_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews = []
    for shot in picks:
        rows = []
        records = []
        for item in shot["labels"]:
            pts = int(item["pts"])
            original = images[pts]
            mask, mask_status = people.get(pts)
            title = (f"CANDIDATE  shot {shot['shot_index']:02d} {shot['camera_id']}  "
                     f"{pts/15360:.3f}s  {item['role']}  inliers {item.get('inliers',0)} "
                     f"ice {item.get('ice_inliers',0)}")
            overlay, counts = renderer.draw(original, camera_vector(item), mask, title)
            left = cv2.resize(original, (960, 540), interpolation=cv2.INTER_AREA)
            right = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
            rows.append(cv2.hconcat((left, right)))
            records.append({"role": item["role"], "pts": pts, "inliers": item.get("inliers"),
                            "ice_inliers": item.get("ice_inliers"), "mask": mask_status,
                            **counts})
        if rows:
            path = output_dir / f"shot_{shot['shot_index']:02d}_candidate_review.jpg"
            cv2.imwrite(str(path), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 91])
            reviews.append({"shot_index": shot["shot_index"], "camera_id": shot["camera_id"],
                            "image": str(path), "samples": records})
        else:
            reviews.append({"shot_index": shot["shot_index"], "camera_id": shot["camera_id"],
                            "image": None, "samples": [], "status": "no_camera_candidate"})
    result = {"schema": "rinkspace.program.render-qa.v1", "candidate_file": str(camera_file),
              "candidate_status": candidates.get("status"), "scene_sha256": release["model_sha256"],
              "source_sha256": shots_data["source_sha256_from_timeline"], "shots": reviews,
              "acceptance": "unvalidated_visual_diagnostic_only"}
    (output_dir / "qa_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def pixel_to_ice(pts: int, pixels: np.ndarray, camera_record: dict, scene: dict,
                 *, allow_candidate: bool = False) -> dict:
    """Map native pixels to model ice coordinates with explicit uncertainty."""
    if int(camera_record["pts"]) != pts:
        raise ValueError("Camera PTS does not match queried source PTS")
    status = str(camera_record["status"])
    if status not in ACCEPTED_STATES and not (allow_candidate and status in {"camera_candidate", "temporal_candidate"}):
        raise ValueError(f"Frame {pts} has no accepted ice localization: {status}")
    xy, valid = ground_xy(camera_vector(camera_record), np.asarray(pixels, float).reshape(-1, 2), scene)
    return {"pts": pts, "scene_xy": xy.tolist(), "valid": valid.tolist(),
            "coordinate_units": "conditional_scene_units_not_verified_meters", "pose_status": status}


def expected_source_pts(shots_data: dict) -> np.ndarray:
    if shots_data["source_resolution"] != list(FRAME_SIZE) or shots_data["source_rate"] != "60/1":
        raise ValueError("Source metadata differs from the original native 1920x1080/60p video")
    if shots_data["source_time_base"] != "1/15360":
        raise ValueError("Unexpected source time base")
    start, stop = map(int, shots_data["window_pts"])
    pts = np.arange(start, stop, 256, dtype=np.int64)
    if len(pts) != int(shots_data["native_frame_count"]):
        raise ValueError("Programme PTS count differs from reviewed source")
    cursor = start
    for index, shot in enumerate(shots_data["shots"]):
        if int(shot["start_pts"]) != cursor or int(shot["end_pts"]) <= cursor:
            raise ValueError(f"Reviewed cut chain is broken at shot {index}")
        if (int(shot["end_pts"]) - cursor) // 256 != int(shot["source_native_frame_count"]):
            raise ValueError(f"Shot {index} native frame count differs from its PTS span")
        cursor = int(shot["end_pts"])
    if cursor != stop:
        raise ValueError("Reviewed shots do not reach the original end PTS")
    return pts


def _np_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Tracked {name} must be scalar")
    return str(array.item())


POSE_QUALITY_FIELDS = ('image_quality_status', 'image_quality_sigma_multiplier',
    'local_quality_max_sigma_multiplier', 'direct_source_factor_pixel_count',
    'motion_source_factor_pixel_count', 'local_quality_soft_pixel_count',
    'local_quality_unknown_pixel_count', 'pose_evidence',
    'ad_source_component_unreadable_component_count')
POSE_PROVENANCE_SCALARS = ('joint_diagnostics_sha256',
                           'ad_source_component_gate_manifest_sha256',
                           'localization_acceptance')


def _read_pose_file(path: Path, expected_hash: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "model_sha256" not in data or _np_scalar(data["model_sha256"], "model_sha256") != expected_hash:
            raise ValueError(f"Tracked poses do not identify the released scene hash: {path}")
        needed = {"pts", "camera25", "status", "metrics"}
        if not needed.issubset(data.files):
            raise ValueError(f"Tracked poses lack {sorted(needed - set(data.files))}: {path}")
        result = {name: data[name].copy() for name in needed}
        for name in ("shot_index", "camera_id", "anchor_pts", "source_sha256",
                     *POSE_QUALITY_FIELDS, *POSE_PROVENANCE_SCALARS):
            if name in data:
                result[name] = data[name].copy()
    return result


def load_tracked_poses(path: Path, shots_data: dict, model_sha256: str,
                       *, candidate_review: bool = False) -> dict:
    """Require one model-bound, accepted pose for every original native frame."""
    expected = expected_source_pts(shots_data)
    shots = shots_data["shots"]
    if path.is_dir():
        parts = []
        for index, shot in enumerate(shots):
            file = path / f"shot{index:02d}.npz"
            if not file.is_file():
                raise ValueError(f"Missing tracked shot {index}: {file}")
            data = _read_pose_file(file, model_sha256)
            if ("source_sha256" not in data or
                    _np_scalar(data["source_sha256"], "source_sha256") != shots_data["source_sha256_from_timeline"]):
                raise ValueError(f"Tracked shot {index} belongs to another source video")
            shot_pts = np.arange(shot["start_pts"], shot["end_pts"], 256, dtype=np.int64)
            if not np.array_equal(data["pts"], shot_pts):
                raise ValueError(f"Tracked shot {index} PTS differ from reviewed hard cuts")
            if ("shot_index" not in data or int(data["shot_index"]) != index
                    or "camera_id" not in data or _np_scalar(data["camera_id"], "camera_id") != shot["camera_id"]):
                raise ValueError(f"Tracked shot {index} camera identity differs from reviewed source")
            parts.append(data)
        result = {name: np.concatenate([part[name] for part in parts])
                  for name in ("pts", "camera25", "status", "metrics")}
        for name in POSE_QUALITY_FIELDS:
            present = [name in part for part in parts]
            if any(present) and not all(present):
                raise ValueError(f'Partial source-quality coverage for {name}')
            if all(present):
                result[name] = np.concatenate([part[name] for part in parts])
    else:
        result = _read_pose_file(path, model_sha256)
        if ("source_sha256" not in result or
                _np_scalar(result["source_sha256"], "source_sha256") != shots_data["source_sha256_from_timeline"]):
            raise ValueError("Tracked programme belongs to another source video")
        if "shot_index" not in result or "camera_id" not in result:
            raise ValueError("Single-file tracking needs per-frame shot_index and camera_id")
        indices = np.asarray(result["shot_index"])
        identities = np.asarray(result["camera_id"]).astype(str)
        if indices.shape != expected.shape or identities.shape != expected.shape:
            raise ValueError("Single-file shot and camera identities must cover each frame")
        for index, shot in enumerate(shots):
            select = (expected >= shot["start_pts"]) & (expected < shot["end_pts"])
            if not np.all(indices[select] == index) or not np.all(identities[select] == shot["camera_id"]):
                raise ValueError(f"Tracked camera identity differs from reviewed shot {index}")
    if not np.array_equal(result["pts"], expected):
        raise ValueError("Tracking omitted, repeated, reordered, or invented original source PTS")
    cameras = np.asarray(result["camera25"], dtype=np.float64)
    statuses = np.asarray(result["status"]).astype(str)
    metrics = np.asarray(result["metrics"])
    if cameras.shape != (len(expected), 25):
        raise ValueError("Every native frame needs one 25-value camera record")
    finite = np.isfinite(cameras).all(axis=1)
    if np.any(finite & (cameras[:, 2] <= 0)) or np.any(finite & (np.abs(cameras[:, 23:25]).max(axis=1) > 1e-9)):
        raise ValueError("Camera centres or frozen Brown sensor tilt violate the rink model")
    if statuses.shape != expected.shape or metrics.shape[0] != len(expected):
        raise ValueError("Tracking states or metrics do not cover every native frame")
    allowed = REVIEW_STATES if candidate_review else ACCEPTED_STATES
    bad = np.flatnonzero(~np.isin(statuses, list(allowed)))
    if len(bad):
        sample = [(int(expected[i]), statuses[i]) for i in bad[:8]]
        raise ValueError(f"{len(bad)} native frames have disallowed localization state; examples={sample}")
    bad_camera = np.flatnonzero(~finite & (statuses != "unavailable"))
    if len(bad_camera) or (not candidate_review and not finite.all()):
        raise ValueError(f"{len(bad_camera)} renderable frames lack a finite camera")
    quality = {name: result[name] for name in POSE_QUALITY_FIELDS if name in result}
    if any(np.asarray(values).shape != expected.shape for values in quality.values()):
        raise ValueError('Source-quality fields must cover the same exact native PTS')
    gate_count = quality.get('ad_source_component_unreadable_component_count')
    gate_hash = result.get('ad_source_component_gate_manifest_sha256')
    if (gate_count is None) != (gate_hash is None):
        raise ValueError('Ad source component gate count and manifest hash must occur together')
    if gate_count is not None:
        if (not candidate_review or 'joint_diagnostics_sha256' not in result
                or 'localization_acceptance' not in result
                or _np_scalar(result['localization_acceptance'], 'localization_acceptance')
                    != 'unvalidated_candidate'
                or not np.issubdtype(np.asarray(gate_count).dtype, np.integer)
                or np.any(gate_count < 0)):
            raise ValueError('Ad source component gate metadata require an unvalidated candidate')
        for name in ('joint_diagnostics_sha256', 'ad_source_component_gate_manifest_sha256'):
            value = _np_scalar(result[name], name)
            if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
                raise ValueError(f'Invalid {name} in ad component gate candidate')
    provenance = {name: _np_scalar(result[name], name) for name in POSE_PROVENANCE_SCALARS
                  if name in result}
    return {"pts": expected, "camera25": cameras, "status": statuses, "metrics": metrics,
            **quality, **provenance,
            "status_counts": {state: int(np.count_nonzero(statuses == state))
                              for state in sorted(allowed) if np.any(statuses == state)}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg_executable() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _available_encoder(ffmpeg: str) -> tuple[list[str], str]:
    """Probe actual one-frame NVENC execution, then use existing libx264."""
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0"]
    trial = subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=black:s=320x180:r=60", "-frames:v", "1", "-an", *nvenc,
        "-f", "null", "-"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    if trial.returncode == 0:
        return nvenc, "h264_nvenc"
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"], "libx264"


def _verify_encoded_video(path: Path, expected_count: int) -> dict:
    count = 0
    with av.open(str(path)) as video:
        if len(video.streams.video) != 1:
            raise ValueError("Encoded result does not have exactly one video stream")
        stream = video.streams.video[0]
        if (stream.width, stream.height) != FRAME_SIZE or float(stream.average_rate) != 60.:
            raise ValueError("Encoded result changed source dimensions or cadence")
        for _frame in video.decode(stream):
            count += 1
        audio_count = len(video.streams.audio)
    if count != expected_count:
        raise ValueError(f"Encoded result has {count} frames; expected {expected_count}")
    return {"decoded_frames": count, "video_size_wh": list(FRAME_SIZE),
            "video_rate": "60/1", "audio_stream_count": audio_count}


def _encode_raw_video(
    source: Path, expected_pts: np.ndarray, cameras: np.ndarray, scene: dict,
    statuses: np.ndarray, masks: PersonMasks, temporary_video: Path, ffmpeg: str,
    encoder_args: list[str], candidate_review: bool,
    cadence_context: dict | None = None,
    foreground_masks=None,
) -> tuple[int, float, object | None, dict]:
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "bgr24", "-video_size", "1920x1080", "-framerate", "60",
        "-i", "pipe:0", "-an", *encoder_args, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary_video),
    ]
    # These two native-source rectangles were checked in this broadcast.
    # They are screen graphics, not physical ice or board observations.
    renderer = SceneRenderer(scene, graphics_regions=((28, 24, 554, 253),
                                                        (1690, 944, 1865, 1000)))
    cadence_observer = None
    if cadence_context is not None:
        try:
            from .source_cadence_qa import SourceCadenceQA
        except ImportError:
            from source_cadence_qa import SourceCadenceQA
        grid_vertices = np.concatenate([
            renderer.xyz[start:end]
            for kind, start, end in zip(renderer.kinds, renderer.offsets[:-1], renderer.offsets[1:])
            if kind == "grid"
        ])
        cadence_observer = SourceCadenceQA(grid_vertices, cadence_context["shots"],
            source=str(source.resolve()), source_sha256=cadence_context["source_sha256"],
            model_sha256=cadence_context["model_sha256"])
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    count = 0
    render_diagnostics = dict(brown_frames=0, unresolved_frames=0,
        maximum_unresolved_leaves=0, maximum_skipped_wall_facets=0,
        source_foreground_frames=0, source_foreground_pixels=0,
        unresolved_source_pts=[],
        skipped_wall_facet_reason_totals=dict(behind=0, sampled_offscreen_guard=0,
                                             unknown_crossing=0),
        unknown_wall_mapping_frames=0,
        skipped_facets_interpretation='Includes offscreen and non-forward facets; not an accuracy failure count')
    try:
        with av.open(str(source)) as container:
            if len(container.streams.video) != 1:
                raise ValueError("Source must have exactly one original video stream")
            stream = container.streams.video[0]
            if (stream.width, stream.height) != FRAME_SIZE or str(stream.time_base) != "1/15360":
                raise ValueError("Original source dimensions or time base changed")
            container.seek(max(0, int(expected_pts[0]) - 15360), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < expected_pts[0]:
                    continue
                if frame.pts > expected_pts[-1]:
                    break
                if count >= len(expected_pts) or int(frame.pts) != int(expected_pts[count]):
                    raise ValueError(f"Original video native PTS gap at output frame {count}: {frame.pts}")
                original = frame.to_ndarray(format="bgr24")
                if cadence_observer is not None:
                    cadence_observer.observe(int(frame.pts), original, cameras[count])
                person, mask_state = masks.get(int(frame.pts), require_sam2=True)
                if mask_state not in MASK_STATES:
                    raise ValueError(f"Person mask is not measured at native PTS {frame.pts}")
                if foreground_masks is not None:
                    foreground = foreground_masks.get(int(frame.pts))
                    if foreground is not None:
                        if foreground.shape != person.shape or foreground.dtype != np.bool_:
                            raise ValueError('Reviewed source foreground mask has invalid shape/type')
                        person = person | foreground
                        render_diagnostics['source_foreground_frames'] += 1
                        render_diagnostics['source_foreground_pixels'] += int(foreground.sum())
                state = str(statuses[count])
                if state == "unavailable":
                    picture = original.copy()
                    cv2.rectangle(picture, (0, 0), (1540, 66), (0, 0, 170), -1)
                    cv2.putText(picture, f"UNAVAILABLE  source PTS {int(frame.pts)}", (14, 43),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    title = (f"UNVERIFIED CAMERA  {state}  source PTS {int(frame.pts)}"
                             if candidate_review else None)
                    picture, _metrics = renderer.draw(original, cameras[count], person, title)
                    refinement = _metrics.get('board_refinement')
                    if refinement is not None:
                        render_diagnostics['brown_frames'] += 1
                        unresolved = (refinement['unresolved_leaves']
                                      + refinement['unresolved_vertical_leaves'])
                        render_diagnostics['maximum_unresolved_leaves'] = max(
                            render_diagnostics['maximum_unresolved_leaves'], unresolved)
                        render_diagnostics['maximum_skipped_wall_facets'] = max(
                            render_diagnostics['maximum_skipped_wall_facets'],
                            refinement['skipped_wall_facets'])
                        reasons = refinement['skipped_wall_facet_reasons']
                        for reason, value in reasons.items():
                            render_diagnostics['skipped_wall_facet_reason_totals'][reason] += value
                        render_diagnostics['unknown_wall_mapping_frames'] += int(
                            reasons['unknown_crossing'] > 0)
                        if unresolved:
                            render_diagnostics['unresolved_frames'] += 1
                            render_diagnostics['unresolved_source_pts'].append(int(frame.pts))
                assert process.stdin is not None
                process.stdin.write(picture.tobytes())
                count += 1
                if count % 600 == 0:
                    print(json.dumps({"rendered_frames": count, "source_pts": int(frame.pts)}), flush=True)
        if count != len(expected_pts):
            raise ValueError(f"Original source ended at {count} frames; expected {len(expected_pts)}")
        assert process.stdin is not None
        process.stdin.close()
        result = process.wait()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if result != 0:
            raise RuntimeError(f"ffmpeg video encoding failed ({result}): {error[-2000:]}")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary_video.is_file():
            temporary_video.unlink()
        raise
    return count, time.perf_counter() - started, cadence_observer, render_diagnostics


def _mux_existing_audio(
    ffmpeg: str, video: Path, audio_source: Path, start_s: float, duration_s: float, output: Path,
) -> None:
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-ss", f"{start_s:.9f}", "-t", f"{duration_s:.9f}", "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "192k", "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Could not mux existing source audio: {result.stderr.decode(errors='replace')[-2000:]}")


def render_program_video(
    *, tracked: Path, source_shots: Path = DEFAULT_SHOTS, scene_file: Path = DEFAULT_SCENE,
    release_file: Path = DEFAULT_RELEASE, people_file: Path = DEFAULT_PEOPLE,
    mask_dir: Path = DEFAULT_MASKS, temporary_dir: Path = DEFAULT_RENDER_TEMP,
    output: Path = DEFAULT_RESULT, audio_source: Path | None = None, overwrite: bool = False,
    candidate_review: bool = False,
    cadence_qa: Path | None = None,
    foreground_review: Path | None = None,
) -> dict:
    """Export only a complete, verified source-PTS programme localization."""
    tracked, source_shots, scene_file, release_file, people_file, mask_dir, temporary_dir, output = map(
        STORAGE.resolve, (tracked, source_shots, scene_file, release_file, people_file, mask_dir, temporary_dir, output))
    if audio_source is not None:
        audio_source = STORAGE.source(audio_source)
    scene, release = read_scene(scene_file, release_file)
    shots_data = json.loads(source_shots.read_text(encoding="utf-8"))
    expected = expected_source_pts(shots_data)
    source = STORAGE.source(shots_data["source_path"])
    if source != STORAGE.source(json.loads(people_file.read_text(encoding="utf-8"))["source"]):
        raise ValueError("Person detections were computed from a different original video")
    if _sha256_file(source) != shots_data["source_sha256_from_timeline"]:
        raise ValueError("Original video changed since camera-cut review")
    poses = load_tracked_poses(tracked, shots_data, release["model_sha256"],
                               candidate_review=candidate_review)
    masks = PersonMasks(mask_dir, people_file)
    mask_counts = masks.validate_coverage(expected)
    foreground_masks = None
    if foreground_review is not None:
        try:
            from .reviewed_foreground_render_mask import ReviewedForegroundMasks
        except ImportError:
            from reviewed_foreground_render_mask import ReviewedForegroundMasks
        foreground_masks = ReviewedForegroundMasks(
            foreground_review, shots_data['source_sha256_from_timeline'], expected)
    mask_manifest = json.loads((mask_dir.parent / "masks.json").read_text(encoding="utf-8"))
    if (STORAGE.source(mask_manifest["source"]) != source
            or int(mask_manifest["frame_count"]) != len(expected)
            or mask_manifest.get("device") != "cuda"):
        raise ValueError("SAM2 mask run does not match the original source or CUDA execution")
    with av.open(str(source)) as source_container:
        selected_audio = audio_source or (source if source_container.streams.audio else None)
    if selected_audio is not None:
        with av.open(str(selected_audio)) as audio_container:
            if not audio_container.streams.audio:
                raise ValueError(f"Selected source has no audio stream: {selected_audio}")
    result_root = PROGRAMME_RESULT.resolve()
    if candidate_review:
        candidate_path = output.resolve()
        if not (candidate_path.is_relative_to(DEFAULT_RENDER_TEMP.parent.resolve())
                or (candidate_path.is_relative_to(PROGRAMME_RESULT.resolve())
                    and "candidate" in output.stem)):
            raise ValueError("Candidate review video must use the external cache or a candidate-named programme result")
    elif not output.resolve().is_relative_to(result_root):
        raise ValueError("Final result must use the registered programme results folder")
    if temporary_dir.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Large temporary video must remain outside the OneDrive project")
    manifest_path = output.with_suffix(".json")
    if (output.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Existing video or manifest needs explicit --overwrite: {output}, {manifest_path}")
    if cadence_qa is not None:
        cadence_qa = Path(cadence_qa)
        if cadence_qa.resolve() in (output.resolve(), manifest_path.resolve()):
            raise ValueError("Cadence QA cannot replace the video or its manifest")
        if cadence_qa.exists() and not overwrite:
            raise FileExistsError(f"Existing cadence QA needs explicit --overwrite: {cadence_qa}")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_executable()
    encoder_args, encoder_name = _available_encoder(ffmpeg)
    temp_video = temporary_dir / f"program_grid_{os.getpid()}.partial.mp4"
    temp_muxed = temporary_dir / f"program_grid_{os.getpid()}.muxed.partial.mp4"
    if temp_video.exists() or temp_muxed.exists():
        raise FileExistsError("A render intermediate with this process id already exists")
    try:
        rendered, elapsed, cadence_observer, render_diagnostics = _encode_raw_video(
            source, expected, poses["camera25"], scene, poses["status"], masks,
            temp_video, ffmpeg, encoder_args, candidate_review,
            cadence_context=({"shots": shots_data["shots"],
                              "source_sha256": shots_data["source_sha256_from_timeline"],
                              "model_sha256": release["model_sha256"]}
                             if cadence_qa is not None else None),
            foreground_masks=foreground_masks,
        )
        target = temp_video
        if selected_audio is not None:
            start_s, stop_s = [value / 15360 for value in shots_data["window_pts"]]
            _mux_existing_audio(ffmpeg, temp_video, selected_audio, start_s, stop_s - start_s, temp_muxed)
            target = temp_muxed
        verification = _verify_encoded_video(target, len(expected))
        if (selected_audio is None and verification["audio_stream_count"] != 0) or (
            selected_audio is not None and verification["audio_stream_count"] < 1
        ):
            raise ValueError("Encoded audio streams disagree with original source")
        cadence_artifact = (cadence_observer.save(cadence_qa, expected_pair_count=len(expected)-1,
                                                  overwrite=overwrite)
                            if cadence_observer is not None else None)
        os.replace(target, output)
        manifest = {
            "schema": "rinkspace.program-render-result.v1",
            "video": str(output.resolve()), "source": str(source.resolve()),
            "source_sha256": shots_data["source_sha256_from_timeline"],
            "scene_sha256": release["model_sha256"], "scene_version": release["version"],
            "source_pts_start_inclusive": int(expected[0]),
            "source_pts_end_exclusive": int(shots_data["window_pts"][1]),
            "source_pts_step": 256, "source_time_base": "1/15360",
            "frame_count": rendered, "cut_count": len(shots_data["shots"]) - 1,
            "pose_status_counts": poses["status_counts"], "person_mask_status_counts": mask_counts,
            "encoder": encoder_name, "audio_source": str(selected_audio) if selected_audio else None,
            "render_seconds": round(elapsed, 3), **verification,
            "render_diagnostics": render_diagnostics,
            "renderer_sha256": _sha256_file(Path(__file__)),
            "validation_scope": ("unverified_candidate_review_only" if candidate_review else
                                 "complete_native_frame_export;visual_ice_fit_review_separate"),
        }
        if cadence_artifact is not None:
            manifest["source_cadence_qa"] = cadence_artifact
        if foreground_masks is not None:
            manifest['reviewed_source_foreground'] = foreground_masks.metadata()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        # Only these two paths, created for this invocation in the designated
        # external cache, may be removed.  Never touch historical project data.
        for intermediate in (temp_video, temp_muxed):
            if intermediate.is_file():
                intermediate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qa-candidates", type=Path)
    mode.add_argument("--render-video", action="store_true")
    mode.add_argument("--candidate-review-video", action="store_true")
    parser.add_argument("--qa-dir", type=Path, default=Path("D:/CodexData/RinkSpace_program_tracking_tmp/render_qa"))
    parser.add_argument("--tracked", type=Path, default=DEFAULT_TRACKED)
    parser.add_argument("--render-temp", type=Path, default=DEFAULT_RENDER_TEMP)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio-source", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-shots", type=Path, default=DEFAULT_SHOTS)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--foreground-review', type=Path,
                        help='Frozen original-pixel foreground review; union with person masks')
    parser.add_argument("--cadence-qa", type=Path,
                        help="Write all source-image cadence pairs and camera grid shifts to this QA JSON")
    args = parser.parse_args()
    if args.qa_candidates is not None:
        if args.foreground_review is not None:
            parser.error('--foreground-review applies to video rendering, not --qa-candidates')
        if args.cadence_qa is not None:
            parser.error("--cadence-qa applies to video rendering, not --qa-candidates")
        result = keyframe_qa(args.qa_candidates, args.source_shots, args.scene, args.release,
                             args.people, args.masks, args.qa_dir)
        for item in result["shots"]:
            print(item["shot_index"], item["image"], len(item["samples"]), flush=True)
    else:
        output = args.output or (DEFAULT_REVIEW if args.candidate_review_video else DEFAULT_RESULT)
        result = render_program_video(
            tracked=args.tracked, source_shots=args.source_shots, scene_file=args.scene,
            release_file=args.release, people_file=args.people, mask_dir=args.masks,
            temporary_dir=args.render_temp, output=output,
            audio_source=args.audio_source, overwrite=args.overwrite,
            candidate_review=args.candidate_review_video,
            cadence_qa=args.cadence_qa,
            foreground_review=args.foreground_review,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
