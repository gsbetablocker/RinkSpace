"""Sequential source-image advertisement boundary measurement.

Named source annotations seed identity search. SIFT and adjacent-frame LK only
move search windows; emitted UV is independently reread in the current RGB.
No scene geometry, fitted camera, projection, or optimizer enters this reader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibration.program_tracking.source_board_measurements import (
    measure_board_source_frame, board_bracket_candidates)


SCHEMA = 'rinkspace.sequential-source-board-boundaries.v1'
SOURCE_SAMPLING_OPERATOR = 'native_BGR_bilinear_coordinates_round_nearest_even_1_over_32'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _identity_reference_allows(reference, number, shot_id, authorities):
    for authority in authorities:
        if (str(Path(authority['path']).resolve()) != str(Path(reference['path']).resolve())
                or authority['sha256'] != reference['sha256']):
            continue
        if ('approved_advertisement_numbers' in authority
                and int(number) not in authority['approved_advertisement_numbers']):
            continue
        if 'approved_shot_ids' in authority and shot_id not in authority['approved_shot_ids']:
            continue
        return True
    return False


def _root_sift(values):
    values = np.asarray(values, np.float32)
    return np.sqrt(values / np.maximum(values.sum(1, keepdims=True), 1e-12))


def _sample(image, xy):
    xy = np.asarray(xy, np.float32)
    # OpenCV 4's INTER_LINEAR quantizes coordinates to a 1/32-pixel table.
    # OpenCV 5 float remap no longer does. Make the source readout operator
    # explicit so changing the installed version cannot alter a material edge.
    xy = np.rint(xy*32.)/32.
    return cv2.remap(image.astype(np.float32, copy=False), xy[..., 0], xy[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _source_orientation_preserved(matrix, points):
    """Reject an image reflection of a printed instance as that same board."""
    centre = np.asarray(points, np.float32).mean(0)
    probe = np.asarray([centre, centre+[3., 0.], centre+[0., 3.]], np.float32)
    mapped = cv2.perspectiveTransform(probe[None], matrix)[0]
    first, second = mapped[1]-mapped[0], mapped[2]-mapped[0]
    signed_area = first[0]*second[1]-first[1]*second[0]
    return bool(np.isfinite(mapped).all() and signed_area > 1e-3)


def _source_visible_arc(image, source_uv):
    """Navigate a named outer arc through an actual RGB contour, never chords."""
    source_uv = np.asarray(source_uv, np.float64)
    closed = len(source_uv) > 3 and np.linalg.norm(source_uv[0]-source_uv[-1]) < 1.
    if closed:
        source_uv = source_uv[:-1]
    lo = np.maximum(0, np.floor(source_uv.min(0)-80)).astype(int)
    hi = np.minimum([1920, 1080], np.ceil(source_uv.max(0)+81)).astype(int)
    gray = cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 3., 9.)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        values = contour[:, 0].astype(np.float64)+lo
        if len(values) < 20:
            continue
        distance = np.linalg.norm(source_uv[:, None]-values[None], axis=2)
        positions = distance.argmin(1)
        errors = distance[np.arange(len(source_uv)), positions]
        if errors.max() > 5.:
            continue
        for direction in (1, -1):
            relative = ((positions-positions[0])*direction) % len(values)
            if (np.diff(relative) <= 0).any():
                continue
            span = len(values) if closed else int(relative[-1])+1
            indices = (positions[0]+direction*np.arange(span)) % len(values)
            path = values[indices]
            length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            supplied_span = np.linalg.norm(np.diff(source_uv, axis=0), axis=1).sum()
            if closed:
                supplied_span += np.linalg.norm(source_uv[-1]-source_uv[0])
            if length > 2.5*supplied_span:
                continue
            candidates.append((float(errors.sum()+.0001*length), path, errors))
    if not candidates:
        return None
    candidates.sort(key=lambda value: value[0])
    _, path, errors = candidates[0]
    arc_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    queries = np.r_[np.arange(0., arc_length[-1], 4.), arc_length[-1]]
    navigation = np.c_[np.interp(queries, arc_length, path[:, 0]),
                       np.interp(queries, arc_length, path[:, 1])]
    return navigation.astype(np.float32), dict(
        method='current_original_source_Canny_contour_through_named_outer_arc_anchors',
        source_anchor_to_contour_distance_px=errors.tolist(),
        source_RGB_contour_chain_length_px=float(arc_length[-1]),
        interpolated_carrier_is_navigation_only=True)


def conservative_ice_exclusion(exclusion, bands, *, boundary_margin=9):
    """Mask above actual yellow support; missing columns stay fully unknown.

    Interpolation only rasterizes a mask between adjacent actual measurements.
    It is never returned or counted as a measured UV.
    """
    output = np.ones(np.asarray(exclusion).shape, bool)
    for band in bands:
        lower = np.asarray(band['yellow_lower_uv'], np.float64)
        for a, b in zip(lower[:-1], lower[1:]):
            if b[0]-a[0] > 29. or b[0] <= a[0]:
                continue
            xs = np.arange(int(np.ceil(a[0])), int(np.floor(b[0]))+1)
            ys = np.interp(xs, [a[0], b[0]], [a[1], b[1]])
            for x, y in zip(xs, ys):
                output[min(1080, int(np.ceil(y))+boundary_margin):, x] = False
    return output | np.asarray(exclusion, bool)


def source_edge_anchors(packet_paths, *, source_sha256, shot_manifest=None,
                        confirmed_identity_references=()):
    """Load named FIT external edges and their existing native RGB caches.

    Partial edges are sufficient. No rectangle is completed from geometry, and
    no HELD annotation becomes a training anchor. Unknown/internal roles skip.
    """
    shots = []
    if shot_manifest is not None:
        content = json.loads(Path(shot_manifest).read_text(encoding='utf8'))
        shots = content['shots']
        if content['source_sha256_from_timeline'] != source_sha256:
            raise ValueError('Anchor source-shot identity differs')
    anchors = []
    confirmed = {(str(Path(r['path']).resolve()), r['sha256']) for r in confirmed_identity_references}
    for packet_path in packet_paths:
        packet_path = Path(packet_path).resolve()
        packet = json.loads(packet_path.read_text(encoding='utf-8-sig'))
        if packet.get('source_video_sha256', packet.get('source_sha256')) != source_sha256:
            raise ValueError('Advertisement source packet belongs to another video')
        reference = dict(path=str(packet_path), sha256=_sha(packet_path))
        for index, row in enumerate(packet.get('rows', [])):
            curve = row.get('curve', {})
            kinds = ('advertisement_boundary', 'board_circle', 'shared_board_pattern_circle_curve')
            if (row.get('split') != 'fit' or curve.get('kind') not in kinds
                    or curve.get('extent_scope', 'external_border') != 'external_border'
                    or str(row.get('geometry_role', '')).startswith('retained_internal')):
                continue
            number = curve.get('advertisement_number', curve.get('advertisement_id'))
            edge = curve.get('edge', 'outermost_visible_circular_print_stroke')
            if number is None or (curve['kind'] == 'advertisement_boundary'
                    and not any(k in edge for k in ('left', 'right', 'top', 'bottom'))):
                continue
            path = row.get('source_raw_frame', row.get('source_path'))
            if path is not None:
                try:
                    if not Path(path).is_file():
                        path = None
                except OSError:
                    # Defer inaccessible historical caches to this exact
                    # exposure of the caller's verified original stream.
                    path = None
            expected = row.get('source_file_sha256', row.get('source_frame_sha256'))
            if path is not None and expected is not None and _sha(path) != expected:
                raise ValueError('Existing source anchor RGB changed')
            uv = np.asarray(row.get('uv'), np.float64)
            if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 3
                    or not np.isfinite(uv).all() or np.any(uv < 0)
                    or np.any(uv >= [1920, 1080])):
                continue
            # Some reviewed arcs store [side, upper, lower] in the numeric
            # array. Their preserved visible-path order, not array order,
            # identifies adjacent source contour samples.
            paths = row.get('source_visible_paths', row.get('source_integration_paths'))
            if paths is not None and len(paths) == 1:
                order = np.asarray(paths[0], int)
                if len(order) < 3 or (order < 0).any() or (order >= len(uv)).any():
                    raise ValueError('Preserved visible source boundary path is invalid')
                uv = uv[order]
                supplied_normal = row.get('normal')
                if supplied_normal is not None and np.asarray(supplied_normal).shape == np.asarray(row['uv']).shape:
                    supplied_normal = np.asarray(supplied_normal)[order]
            else:
                supplied_normal = row.get('normal')
            native = int(row['pts'])
            actual = int(row.get('actual_pts', native))
            shot = [s['id'] for s in shots
                    if int(s['start_pts']) <= native < int(s['end_pts'])]
            anchors.append(dict(anchor_id=f'{packet_path.stem}:{index}',
                physical_material_id=f'Ad{int(number)}.{edge}',
                advertisement_number=int(number), edge=edge, uv=uv,
                native_pts=native, actual_pts=actual,
                shot_id=shot[0] if len(shot) == 1 else None,
                source_raw_frame=str(Path(path).resolve()) if path is not None else None,
                source_image_sha256=(expected or _sha(path)) if path is not None else None,
                source_annotation_reference=reference, source_row_index=index,
                source_annotation_id=row.get('id'),
                curve=dict(curve), source_normal=supplied_normal,
                source_identity_confirmed=_identity_reference_allows(reference, number,
                    shot[0] if len(shot) == 1 else None, confirmed_identity_references),
                source_identity_reference_requires_current_shot=True,
                identity_status='named_source_annotation_requires_current_instance_confirmation'))
            if paths is not None and len(paths) > 1:
                base = anchors.pop()
                for path_number, path_order in enumerate(paths):
                    order = np.asarray(path_order, int)
                    if len(order) < 3:
                        continue
                    if (order < 0).any() or (order >= len(uv)).any():
                        raise ValueError('Visible source path index is outside its preserved UV')
                    path_normal = (np.asarray(supplied_normal)[order]
                        if supplied_normal is not None and np.asarray(supplied_normal).shape == uv.shape else None)
                    anchors.append(dict(base, uv=uv[order], source_normal=path_normal,
                        anchor_id=base['anchor_id']+f':visible_path{path_number}',
                        tracking_id=base['physical_material_id']+f':visible_path{path_number}'))
    return anchors


def source_user_stroke_seeds(constraint_path, *, source_sha256):
    """Use explicit human external-border identities as source search seeds."""
    path = Path(constraint_path).resolve()
    content = json.loads(path.read_text(encoding='utf8'))
    if content['source_video_sha256'] != source_sha256:
        raise ValueError('Registered user physical identities belong to another video')
    reference = dict(path=str(path), sha256=_sha(path))
    seeds = []
    entity_source_guides = {}
    for stroke in content['user_marked_strokes']:
        curve = stroke['physical_curve']
        if (curve.get('kind') != 'advertisement_boundary'
                or curve.get('extent_scope') != 'external_border'):
            continue
        trace_ref = stroke['trace_reference']
        if _sha(trace_ref['path']) != trace_ref['sha256']:
            raise ValueError('Registered original human source trace changed')
        trace = json.loads(Path(trace_ref['path']).read_text(encoding='utf8'))
        if trace.get('model_projection_used') is not False or trace['source_pts'] != content['source_pts']:
            raise ValueError('Human trace must be registered directly to the original source')
        uv = np.asarray(trace['user_stroke_raw_native_px'], np.float32)
        # The drawing names a physical border. Its fitted source line is only
        # navigation; every admitted pixel must be independently reread in RGB.
        vx, vy, cx, cy = cv2.fitLine(uv, cv2.DIST_L2, 0., .01, .01).reshape(-1)
        direction, centre = np.asarray([vx, vy]), np.asarray([cx, cy])
        coordinate = (uv-centre)@direction
        a, b = coordinate[0], coordinate[-1]
        locations = np.linspace(a, b, max(3, int(abs(b-a)//4)+1))
        carrier = centre+locations[:, None]*direction
        number, edge = int(curve['advertisement_number']), curve['edge']
        entity_source_guides.setdefault(number, []).append(uv)
        seeds.append(dict(anchor_id=stroke['id'], physical_material_id=f'Ad{number}.{edge}',
            advertisement_number=number, edge=edge, uv=carrier,
            native_pts=int(trace['source_pts']), actual_pts=int(trace['source_pts']),
            curve=curve, source_annotation_reference=reference,
            source_trace_reference=trace_ref, source_identity_confirmed=True,
            source_raw_frame_reference=dict(path=trace['source_raw_frame'], sha256=trace['source_file_sha256']),
            human_physical_identity_anchor=True,
            search_origin='direct_user_registered_source_stroke',
            raw_user_drawn_stroke_is_identity_navigation_only=True,
            model_projection_or_fit_residual_used_for_identity=False))
    for seed in seeds:
        points = np.concatenate(entity_source_guides[seed['advertisement_number']])
        lo, hi = points.min(0), points.max(0)
        seed['source_entity_context_uv'] = np.asarray([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]], np.float32)
        seed['source_entity_context_origin'] = 'combined_registered_user_source_strokes_navigation_only'
    return seeds


class BoardProgrammeSourceReader:
    """One ordered original source stream; no seeks or frame-cache creation."""
    def __init__(self, anchors, *, source_sha256, time_base='1/15360',
                 actual_exposure_pts=None, native_to_actual=None, corner_annotation_paths=(),
                 confirmed_identity_references=(), source_annotation_seeds=(), device='cuda'):
        self.source_sha256, self.time_base = source_sha256, time_base
        self.anchors = list(anchors)
        self.templates = []
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.014)
        cv2.setNumThreads(2)
        self.device = device
        self.torch = None
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Requested GPU descriptor matching is unavailable')
            self.torch = torch
        elif device != 'cpu':
            raise ValueError('Explicit cuda/cpu matching device required')
        self.actual_successor = {}
        if actual_exposure_pts is not None:
            actual = np.asarray(actual_exposure_pts, np.int64)
            if not np.array_equal(actual, np.unique(actual)):
                raise ValueError('Exact sorted verified exposure inventory required')
            self.actual_successor = dict(zip(map(int, actual[:-1]), map(int, actual[1:])))
        self.native_to_actual = dict(native_to_actual or {})
        self.corner_seeds = {}
        self.confirmed_identity_references = {(str(Path(r['path']).resolve()), r['sha256'])
                                              for r in confirmed_identity_references}
        self.identity_authorities = list(confirmed_identity_references)
        groups = {}
        for anchor in self.anchors:
            if anchor['source_raw_frame'] is None:
                self.corner_seeds.setdefault(anchor['actual_pts'], []).append(anchor)
                continue
            groups.setdefault(anchor['source_raw_frame'], []).append(anchor)
        for path, group in groups.items():
            image = cv2.imread(path)
            if image is None or image.shape != (1080, 1920, 3):
                raise ValueError('Native source anchor RGB required')
            source_BGR_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            for anchor in group:
                anchor['source_BGR_sha256'] = source_BGR_sha256
            bounds = [(np.maximum(0, np.floor(np.asarray(a['uv']).min(0)-80)).astype(int),
                       np.minimum([1920, 1080], np.ceil(np.asarray(a['uv']).max(0)+81)).astype(int))
                      for a in group]
            lo = np.min([b[0] for b in bounds], axis=0)
            hi = np.max([b[1] for b in bounds], axis=0)
            mask = np.zeros((hi[1]-lo[1], hi[0]-lo[0]), np.uint8)
            for first, second in bounds:
                mask[first[1]-lo[1]:second[1]-lo[1], first[0]-lo[0]:second[0]-lo[0]] = 255
            gray = cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2GRAY)
            keys, desc = self.sift.detectAndCompute(gray, mask)
            if desc is None:
                continue
            coordinates = np.asarray([k.pt for k in keys])+lo
            for anchor, (first, second) in zip(group, bounds):
                take = np.flatnonzero((coordinates>=first).all(1)&(coordinates<second).all(1))
                if len(take) >= 8:
                    self._install_template(anchor, image, np.asarray(anchor['uv'], np.float32),
                        coordinates[take], desc[take])
        self.previous_gray = None
        self.previous_pts = None
        self.previous_shot = None
        self.tracks = {}
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.discovery_indices = {}
        self.source_seed_descriptor_cache = {}
        for path in corner_annotation_paths:
            self._load_corner_annotations(path)
        source_annotation_seeds = [dict(seed) for seed in source_annotation_seeds]
        entity_groups = {}
        for seed in source_annotation_seeds:
            if seed.get('source_identity_confirmed') is True:
                key = (int(seed['advertisement_number']), int(seed['native_pts']), int(seed['actual_pts']))
                entity_groups.setdefault(key, []).append(np.asarray(seed['uv'], np.float32))
        for seed in source_annotation_seeds:
            key = (int(seed['advertisement_number']), int(seed['native_pts']), int(seed['actual_pts']))
            if seed.get('source_entity_context_uv') is None and key in entity_groups:
                entity = np.concatenate(entity_groups[key])
                seed['source_entity_context_uv'] = np.asarray([entity.min(0), entity.max(0)], np.float32)
                seed['source_entity_context_origin'] = 'same_named_original_source_annotation_queries_navigation_only'
            self.add_source_seed(seed)

    @classmethod
    def from_source_packets(cls, packet_paths, *, source_sha256,
                            time_base='1/15360', shot_manifest=None,
                            confirmed_identity_references=(), actual_exposure_pts=None,
                            native_to_actual=None, corner_annotation_paths=(),
                            source_annotation_seeds=(), device='cuda'):
        return cls(source_edge_anchors(packet_paths, source_sha256=source_sha256,
                   shot_manifest=shot_manifest, confirmed_identity_references=confirmed_identity_references),
                   source_sha256=source_sha256, time_base=time_base,
                   actual_exposure_pts=actual_exposure_pts, native_to_actual=native_to_actual,
                   corner_annotation_paths=corner_annotation_paths,
                   confirmed_identity_references=confirmed_identity_references,
                   source_annotation_seeds=source_annotation_seeds, device=device)

    def add_source_seed(self, seed):
        """Register an existing named source polyline at its exact stream frame.

        This does not create a template from a projected/candidate curve. The
        caller supplies preserved physical annotation identity and authority;
        all pixels will be reread when that original exposure reaches read().
        """
        seed = dict(seed)
        required = ('anchor_id', 'physical_material_id', 'advertisement_number',
                    'edge', 'uv', 'actual_pts', 'native_pts', 'curve',
                    'source_annotation_reference')
        if any(seed.get(k) is None for k in required):
            raise ValueError('Named original source seed identity is incomplete')
        if seed.get('model_projection_or_fit_residual_used_for_identity') is not False:
            raise ValueError('Source-only annotation origin must be explicit')
        uv = np.asarray(seed['uv'], np.float32)
        if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 3
                or not np.isfinite(uv).all() or (uv < 0).any()
                or (uv >= [1920, 1080]).any()):
            raise ValueError('Existing native boundary polyline with at least three actual UV required')
        seed['uv'] = uv
        ref = seed['source_annotation_reference']
        if not ref.get('path') or len(ref.get('sha256', '')) != 64:
            raise ValueError('Preserved source annotation authority required')
        seed['source_identity_confirmed'] = seed.get('source_identity_confirmed') is True
        if seed.get('load_preserved_source_template') is True:
            reference = seed.get('source_raw_frame_reference')
            if not reference or not seed.get('shot_id'):
                raise ValueError('Preloaded source identity needs its exact raw frame and original shot')
            path = str(Path(reference['path']).resolve())
            if path not in self.source_seed_descriptor_cache:
                if _sha(path) != reference['sha256']:
                    raise ValueError('Preserved native identity source RGB changed')
                image = cv2.imread(path)
                if image is None or image.shape != (1080, 1920, 3):
                    raise ValueError('Preserved identity source must be native BGR')
                keys, descriptors = self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
                coordinates = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)
                self.source_seed_descriptor_cache[path] = (reference['sha256'], image, coordinates, descriptors)
            digest, image, coordinates, descriptors = self.source_seed_descriptor_cache[path]
            if digest != reference['sha256']:
                raise ValueError('Conflicting source RGB reference')
            seed['source_BGR_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
            seed['source_identity_reference_requires_current_shot'] = True
            lo, hi = uv.min(0)-80., uv.max(0)+81.
            take = np.flatnonzero((coordinates >= lo).all(1)&(coordinates < hi).all(1))
            if descriptors is None or len(take) < 8:
                if seed.get('human_physical_identity_anchor') is not True:
                    return
                points, values = np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
            else:
                points, values = coordinates[take], descriptors[take]
            self._install_template(seed, image, uv, points, values)
            self.discovery_indices.clear()
            return
        self.corner_seeds.setdefault(int(seed['actual_pts']), []).append(seed)

    def _install_template(self, anchor, image, uv, keys, desc):
        circular = anchor['curve'].get('kind') in ('board_circle', 'shared_board_pattern_circle_curve')
        anchor = dict(anchor)
        if circular:
            arc = _source_visible_arc(image, uv)
            anchor['source_outer_contour_confirmed'] = arc is not None
            anchor['sparse_original_outer_anchor_queries_only'] = arc is None
            if arc is not None:
                uv, anchor['source_actual_contour_evidence'] = arc
        supplied = anchor.get('source_normal')
        if supplied is not None and np.asarray(supplied).shape == uv.shape:
            normal = np.asarray(supplied, np.float64)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        else:
            tangent = np.gradient(uv.astype(np.float64), axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        # Densify only the navigation polyline. Each new search location must
        # still produce an independently visible current-source transition.
        if (not anchor.get('sparse_original_outer_anchor_queries_only')
                and anchor.get('model_target', {}).get('kind') != 'current_scene_board_print_contour'):
            navigation, navigation_normals = [], []
            for k, (a, b) in enumerate(zip(uv[:-1], uv[1:])):
                count = max(1, int(np.ceil(np.linalg.norm(b-a)/4.)))
                fractions = np.arange(count)/count
                navigation.extend(a[None]*(1-fractions[:, None])+b[None]*fractions[:, None])
                navigation_normals.extend(normal[k][None]*(1-fractions[:, None])
                                          + normal[k+1][None]*fractions[:, None])
            navigation.append(uv[-1]); navigation_normals.append(normal[-1])
            uv = np.asarray(navigation, np.float32)
            normal = np.asarray(navigation_normals, np.float64)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        if anchor.get('search_origin') == 'direct_user_registered_source_stroke':
            offsets = np.arange(-12., 12.001, .5)
            xy = uv[:, None]+offsets[None, :, None]*normal[:, None]
            values = _sample(image, xy)
            deltas = values[:, 4:]-values[:, :-4]
            magnitude = np.linalg.norm(deltas, axis=2)
            peak = []
            for profile in magnitude:
                local = np.flatnonzero((profile[1:-1] >= profile[:-2])
                    & (profile[1:-1] >= profile[2:]) & (profile[1:-1] >= 12.))+1
                # A brighter rail can sit next to an explicitly named printed
                # panel. Follow the nearest source interface to the registered
                # human guide; the globally strongest edge may be another item.
                peak.append(int(min(local, key=lambda k: (abs(offsets[k+2]), -profile[k])))
                            if len(local) else int(np.argmax(profile)))
            peak = np.asarray(peak, int)
            # Only improve the search location from this real source image;
            # read() still independently measures the actual boundary UV.
            shift = offsets[peak+2]
            uv = (uv+shift[:, None]*normal).astype(np.float32)
        vertical = 'vertical' in anchor.get('edge', '')
        source_side_distance = 1.5 if circular else 6. if vertical else 3.
        sides = _sample(image, uv[:, None]+normal[:, None]*np.array([-source_side_distance, source_side_distance])[None, :, None])
        contrast = sides[:, 1]-sides[:, 0]
        magnitude = np.linalg.norm(contrast, axis=1)
        contrast /= np.maximum(magnitude[:, None], 1e-9)
        coordinates = (np.asarray(keys, np.float32) if isinstance(keys, np.ndarray)
                       else np.asarray([k.pt for k in keys], np.float32))
        descriptors = _root_sift(desc)
        device_desc = self.torch.as_tensor(descriptors, device='cuda') if self.torch is not None else None
        self.templates.append(dict(anchor=anchor, xy=coordinates,
            desc=descriptors, device_desc=device_desc, uv=uv, normal=normal,
            contrast=contrast, source_contrast=magnitude))

    def _load_corner_annotations(self, path):
        path = Path(path).resolve()
        packet = json.loads(path.read_text(encoding='utf8'))
        if packet['source_sha256'] != self.source_sha256:
            raise ValueError('Old source corner annotations belong to another source')
        ref = dict(path=str(path), sha256=_sha(path))
        groups = {}
        for row in packet.get('observations', []):
            if row.get('status') != 'measured' or row.get('role') not in ('left_top', 'left_bottom', 'right_top', 'right_bottom'):
                continue
            native = int(row['pts'])
            actual = int(self.native_to_actual.get(native, native))
            groups.setdefault((actual, int(row['user_number'])), {})[row['role']] = np.asarray(row['uv'], float)
        for (pts, number), corners in groups.items():
            for edge, roles in dict(left_vertical=('left_top', 'left_bottom'),
                                   right_vertical=('right_top', 'right_bottom'),
                                   top_edge=('left_top', 'right_top'),
                                   bottom_edge=('left_bottom', 'right_bottom')).items():
                if not all(r in corners for r in roles):
                    continue
                a, b = (corners[r] for r in roles)
                if np.linalg.norm(b-a) < 12.:
                    continue
                # Source corners define only the initial search polyline. Every
                # output sample is reread from its current actual visible edge.
                uv = np.linspace(a, b, max(3, int(np.linalg.norm(b-a)//14)+1)).astype(np.float32)
                self.corner_seeds.setdefault(pts, []).append(dict(
                    anchor_id=f'{path.stem}:{pts}:Ad{number}:{edge}',
                    physical_material_id=f'Ad{number}.{edge}', advertisement_number=number,
                    edge=edge, uv=uv, native_pts=pts, actual_pts=pts,
                    source_annotation_reference=ref, source_row_index=None,
                    source_identity_confirmed=False,
                    source_identity_reference_requires_current_shot=True,
                    curve=dict(kind='advertisement_boundary', advertisement_number=number,
                               edge=edge, extent_scope='external_border')))

    def _current_corner_templates(self, image, gray, mask, binding):
        pts = int(binding['actual_pts'])
        seeds = self.corner_seeds.pop(pts, [])
        native = int(binding.get('native_pts', binding.get('source_pts')))
        # An exposure alias has different decoded RGB. It cannot silently
        # turn a neighbouring native annotation into an exact current anchor.
        seeds = [seed for seed in seeds if int(seed['native_pts']) == native]
        cached_keys, cached_desc, cached_coordinates = [], None, np.empty((0, 2), np.float32)
        if seeds:
            bounds = [(np.maximum(0, np.floor(np.asarray(s['uv']).min(0)-80)).astype(int),
                       np.minimum([1920, 1080], np.ceil(np.asarray(s['uv']).max(0)+81)).astype(int))
                      for s in seeds]
            common_lo = np.min([b[0] for b in bounds], axis=0)
            common_hi = np.max([b[1] for b in bounds], axis=0)
            region = np.zeros((common_hi[1]-common_lo[1], common_hi[0]-common_lo[0]), np.uint8)
            for lo, hi in bounds:
                region[lo[1]-common_lo[1]:hi[1]-common_lo[1], lo[0]-common_lo[0]:hi[0]-common_lo[0]] = 255
            region[mask[common_lo[1]:common_hi[1], common_lo[0]:common_hi[0]]] = 0
            cached_keys, cached_desc = self.sift.detectAndCompute(
                gray[common_lo[1]:common_hi[1], common_lo[0]:common_hi[0]], region)
            cached_coordinates = np.asarray([k.pt for k in cached_keys], np.float32).reshape(-1, 2)+common_lo
        for seed in seeds:
            if seed.get('source_identity_reference_requires_current_shot'):
                seed['source_identity_confirmed'] = _identity_reference_allows(
                    seed['source_annotation_reference'], seed['advertisement_number'],
                    binding.get('shot_id'), self.identity_authorities)
            uv = seed['uv']; lo = np.maximum(0, np.floor(uv.min(0)-80)).astype(int)
            hi = np.minimum([1920, 1080], np.ceil(uv.max(0)+81)).astype(int)
            take = np.flatnonzero((cached_coordinates >= lo).all(1)&(cached_coordinates < hi).all(1))
            coordinates = cached_coordinates[take]
            desc = cached_desc[take] if cached_desc is not None else None
            if desc is None or len(take) < 8:
                if seed.get('human_physical_identity_anchor') is not True:
                    continue
                coordinates, desc = np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
            seed.update(shot_id=binding.get('shot_id'), decoded_BGR_sha256=binding.get('decoded_native_BGR_sha256'))
            self._install_template(seed, image, uv, coordinates, desc)
        if seeds:
            self.discovery_indices.clear()

    def _template_matches(self, gray, exclusion, bands, shot_id, *, tracked_names=()):
        # Feature discovery is source-bounded. Green bands do not approve UV.
        tracked_names = set(tracked_names)
        if not any(t['anchor']['shot_id'] in (None, shot_id)
                   and t['anchor'].get('tracking_id', t['anchor']['physical_material_id']) not in tracked_names
                   for t in self.templates):
            return {}
        mask = np.zeros((1080, 1920), np.uint8)
        for band in bands:
            upper = np.asarray(band['blue_upper_uv'])
            lower = np.asarray(band['yellow_lower_uv'])
            polygon = np.r_[upper, lower[::-1]].astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)
        mask[exclusion] = 0
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return {}
        lo, hi = np.array([xs.min(), ys.min()]), np.array([xs.max()+1, ys.max()+1])
        keys, descriptors = self.sift.detectAndCompute(gray[lo[1]:hi[1], lo[0]:hi[0]],
            mask[lo[1]:hi[1], lo[0]:hi[0]])
        if descriptors is None or len(keys) < 8:
            return {}
        xy = np.asarray([k.pt for k in keys], np.float32)+lo
        descriptors = _root_sift(descriptors)
        # One approximate source-descriptor index proposes relevant templates.
        # Exact template ratio/RANSAC checks below still establish each search.
        if shot_id not in self.discovery_indices:
            indices = [i for i, t in enumerate(self.templates)
                       if t['anchor']['shot_id'] in (None, shot_id)]
            owners = np.concatenate([np.full(len(self.templates[i]['desc']), i, np.int32)
                                     for i in indices]) if indices else np.empty(0, np.int32)
            bank = np.concatenate([self.templates[i]['desc'] for i in indices]) if indices else None
            matcher = None
            if bank is not None and len(bank) >= 2:
                if self.torch is not None:
                    matcher = self.torch.as_tensor(bank, device='cuda')
                else:
                    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
                    matcher.add([bank]); matcher.train()
            self.discovery_indices[shot_id] = matcher, owners
        matcher, owners = self.discovery_indices[shot_id]
        if matcher is None:
            return {}
        target_device = None
        instance_distances = {}
        owner_numbers = np.asarray([self.templates[int(i)]['anchor']['advertisement_number']
                                    for i in owners], np.int32)
        if self.torch is not None:
            target_device = self.torch.as_tensor(descriptors, device='cuda')
            with self.torch.inference_mode():
                selected_parts = []
                by_instance = {int(n): [] for n in np.unique(owner_numbers)}
                columns = {n: self.torch.as_tensor(np.flatnonzero(owner_numbers == n), device='cuda')
                           for n in by_instance}
                for start in range(0, len(descriptors), 128):
                    distance = self.torch.cdist(target_device[start:start+128], matcher)
                    values, ids = distance.min(1)
                    selected_parts.append(ids[values<.55].cpu().numpy())
                    for number, cols in columns.items():
                        by_instance[number].append(distance[:, cols].min(1).values.cpu().numpy())
                    del distance
                selected = np.concatenate(selected_parts)
                counts = np.bincount(owners[selected], minlength=len(self.templates))
                instance_distances = {n: np.concatenate(v) for n, v in by_instance.items()}
        else:
            proposed = matcher.match(descriptors)
            counts = np.bincount([int(owners[m.trainIdx]) for m in proposed if m.distance < .55],
                                 minlength=len(self.templates))
        matches = {}
        # The same print has several source views/edge anchors. A target feature
        # may choose any of their nearly equal descriptors; pool only discovery
        # votes by named instance, then verify each exact source template.
        instance_votes = {}
        for i, count in enumerate(counts):
            number = self.templates[i]['anchor']['advertisement_number']
            instance_votes[number] = instance_votes.get(number, 0)+int(count)
        selected_templates = [i for i, t in enumerate(self.templates)
                              if instance_votes.get(t['anchor']['advertisement_number'], 0) >= 8
                              and t['anchor'].get('tracking_id', t['anchor']['physical_material_id']) not in tracked_names]
        for template_index in selected_templates:
            template = self.templates[int(template_index)]
            anchor = template['anchor']
            if anchor['shot_id'] is not None and anchor['shot_id'] != shot_id:
                continue
            if self.torch is not None:
                with self.torch.inference_mode():
                    values, ids = self.torch.cdist(template['device_desc'], target_device).topk(2, largest=False)
                    valid = values[:, 0] < .76*values[:, 1]
                    source_ids = self.torch.nonzero(valid).reshape(-1).cpu().numpy()
                    target_ids = ids[valid, 0].cpu().numpy()
                good = [cv2.DMatch(int(a), int(b), 0.) for a, b in zip(source_ids, target_ids)]
            else:
                pairs = self.bf.knnMatch(template['desc'], descriptors, k=2)
                good = [m for m, n in pairs if m.distance < .76*n.distance]
            if len(good) < 8:
                continue
            a = template['xy'][[m.queryIdx for m in good]]
            b = xy[[m.trainIdx for m in good]]
            matrix, inside = cv2.findHomography(a, b, cv2.RANSAC, 2.5,
                                               maxIters=1000, confidence=.995)
            if matrix is None or inside is None or int(inside.sum()) < 8:
                continue
            inside = inside.reshape(-1).astype(bool)
            if inside.mean() < .55:
                continue
            if not _source_orientation_preserved(matrix, a[inside]):
                continue
            support = b[inside]
            span = np.ptp(support, axis=0)
            if span.min() < 8. or span.max() < 40.:
                continue
            number = int(anchor['advertisement_number'])
            targets = np.asarray([good[k].trainIdx for k in np.flatnonzero(inside)], int)
            others = [values[targets] for n, values in instance_distances.items() if n != number]
            unique = np.zeros(len(targets), bool)
            if others and number in instance_distances:
                unique = instance_distances[number][targets] < .76*np.min(others, axis=0)
            unique_support = support[unique]
            unique_context = (len(unique_support) >= 8
                and np.ptp(unique_support, axis=0).min() >= 8.
                and np.ptp(unique_support, axis=0).max() >= 40.)
            mapped = cv2.perspectiveTransform(template['uv'][None], matrix)[0]
            normal_end = cv2.perspectiveTransform((template['uv']+template['normal']*5)[None].astype(np.float32), matrix)[0]
            normals = normal_end-mapped
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
            name = anchor.get('tracking_id', anchor['physical_material_id'])
            result = dict(template=template, search_uv=mapped, normals=normals,
                support_xy=b[inside], identity_matches=int(inside.sum()),
                matched_target_indices=set(int(good[k].trainIdx) for k in np.flatnonzero(inside)),
                identity_context_span_native_px=span.tolist(),
                current_instance_identity_confirmed=bool(unique_context),
                named_instance_discriminating_context_features=int(unique.sum()),
                compared_source_named_instances=sorted(instance_distances),
                identity_method='same_shot_source_annotation_SIFT', continuity=False)
            if anchor.get('source_entity_context_uv') is not None:
                result['source_entity_context_uv'] = cv2.perspectiveTransform(
                    np.asarray(anchor['source_entity_context_uv'], np.float32)[None], matrix)[0]
            if name not in matches or result['identity_matches'] > matches[name]['identity_matches']:
                matches[name] = result
        # Repeated graphics must not name two physical advertisements at once.
        conflicts = set()
        named = list(matches)
        for i, name in enumerate(named):
            first = matches[name]
            for other in named[i+1:]:
                second = matches[other]
                if first['template']['anchor']['advertisement_number'] == second['template']['anchor']['advertisement_number']:
                    continue
                a, b = first['matched_target_indices'], second['matched_target_indices']
                if len(a & b) >= .5*min(len(a), len(b)):
                    conflicts.update((name, other))
        for name in conflicts:
            matches.pop(name)
        return matches

    def _direct_current_anchors(self, binding, exclusion):
        """Use exact current source annotations only when their RGB bytes match."""
        current = int(binding['actual_pts'])
        digest = binding.get('decoded_native_BGR_sha256', binding.get('decoded_BGR_sha256'))
        matches = {}
        for template in self.templates:
            anchor = template['anchor']
            if anchor['actual_pts'] != current:
                continue
            expected = anchor.get('source_BGR_sha256', anchor.get('decoded_BGR_sha256'))
            if expected != digest:
                continue
            points = template['xy']; pixels = np.rint(points).astype(int)
            good = (pixels>=0).all(1)&(pixels<[1920, 1080]).all(1)
            bounded = np.clip(pixels, [0, 0], [1919, 1079])
            good &= ~exclusion[bounded[:, 1], bounded[:, 0]]
            if good.sum() < 8 and anchor.get('human_physical_identity_anchor') is not True:
                continue
            name = anchor.get('tracking_id', anchor['physical_material_id'])
            matches[name] = dict(template=template, search_uv=template['uv'].copy(),
                normals=template['normal'].copy(), support_xy=points[good],
                identity_matches=int(good.sum()), continuity=False,
                current_instance_identity_confirmed=True,
                identity_method='exact_same_exposure_source_annotation_and_BGR_bytes')
            if anchor.get('source_entity_context_uv') is not None:
                matches[name]['source_entity_context_uv'] = np.asarray(anchor['source_entity_context_uv'], np.float32).copy()
        return matches

    def _track(self, gray, exclusion):
        updated = {}
        actual_navigation = {}
        params = dict(winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, .01))
        for name, track in self.tracks.items():
            a = track['support_xy'].astype(np.float32)
            if len(a) < 8:
                continue
            b, forward, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, a[:, None], None, **params)
            if b is None:
                continue
            back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, b, None, **params)
            if back is None:
                continue
            b = b[:, 0]; back = back[:, 0]
            good = (forward[:, 0]>0)&(backward[:, 0]>0)&(np.linalg.norm(back-a, axis=1)<1.)
            pixel = np.clip(np.rint(b).astype(int), [0, 0], [1919, 1079])
            good &= (b>0).all(1)&(b<[1920, 1080]).all(1)&~exclusion[pixel[:, 1], pixel[:, 0]]
            if good.sum() < 8:
                continue
            matrix, inside = cv2.findHomography(a[good], b[good], cv2.RANSAC, 2., maxIters=1000)
            if matrix is None or inside is None or inside.sum() < 8:
                continue
            if not _source_orientation_preserved(matrix, a[good][inside.reshape(-1)>0]):
                continue
            search = cv2.perspectiveTransform(track['search_uv'][None].astype(np.float32), matrix)[0]
            end = cv2.perspectiveTransform((track['search_uv']+track['normals']*5)[None].astype(np.float32), matrix)[0]
            normal = end-search
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            updated[name] = dict(track, search_uv=search, normals=normal,
                support_xy=b[good][inside.reshape(-1)>0], continuity=True,
                identity_method='same_shot_actual_adjacent_RGB_LK_and_current_edge_reread')
            if track.get('source_entity_context_uv') is not None:
                updated[name]['source_entity_context_uv'] = cv2.perspectiveTransform(
                    np.asarray(track['source_entity_context_uv'], np.float32)[None], matrix)[0]
            actual_navigation[name] = matrix
        for name, track in self.tracks.items():
            if name in updated or not track.get('previous_frame_RGB_measured'):
                continue
            anchor = track['template']['anchor']
            chain = anchor.get('same_board_chain_id')
            if not chain or anchor.get('human_physical_identity_anchor') is not True:
                continue
            neighbours = [other for other in updated
                if self.tracks[other].get('previous_frame_RGB_measured')
                and updated[other]['template']['anchor'].get('same_board_chain_id') == chain]
            if not neighbours:
                continue
            neighbour = max(neighbours, key=lambda other: len(updated[other]['support_xy']))
            matrix = actual_navigation[neighbour]
            search = cv2.perspectiveTransform(track['search_uv'][None].astype(np.float32), matrix)[0]
            end = cv2.perspectiveTransform((track['search_uv']+track['normals']*5)[None].astype(np.float32), matrix)[0]
            normal = end-search
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            updated[name] = dict(track, search_uv=search, normals=normal,
                support_xy=updated[neighbour]['support_xy'].copy(), continuity=True,
                identity_method='continuous_source_boundary_linked_to_confirmed_adjoining_board_print',
                confirmed_source_board_chain=chain,
                actual_source_navigation_neighbour=updated[neighbour]['template']['anchor']['physical_material_id'])
        return updated

    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        template, observed = track['template'], []
        printed = template['anchor'].get('model_target', {}).get('kind') == 'current_scene_board_print_contour'
        vertical = 'vertical' in template['anchor'].get('edge', '')
        window = 6. if printed else 24. if vertical else 10.
        offsets = np.arange(-window, window+.001, .5)
        guesses = np.asarray(track['search_uv'])
        normals = np.asarray(track['normals'])
        tangents = np.c_[-normals[:, 1], normals[:, 0]]
        sample_xy = (guesses[:, None, None]+offsets[None, :, None, None]*normals[:, None, None]
                     + np.array([-2., 0., 2.])[None, None, :, None]*tangents[:, None, None])
        # One remap for all samples of this visible edge. This bounded readout
        # avoids a separate OpenCV invocation at each four-pixel source sample.
        sampled = _sample(image, sample_xy.reshape(-1, 3, 2)).reshape(len(guesses), len(offsets), 3, 3)
        source_values = np.einsum('npc,nc->np', np.median(sampled, axis=2), template['contrast'])
        circular = template['anchor']['curve'].get('kind') in ('board_circle', 'shared_board_pattern_circle_curve')
        independently_known_board = bool(known_source_board_material
            and known_source_board_material.get('physical_material_id') == template['anchor']['physical_material_id']
            and known_source_board_material.get('source_identity_confirmed') is True
            and known_source_board_material.get('actual_original_source_printed_board_membership_confirmed') is True
            and known_source_board_material.get('model_projection_or_fit_residual_used_for_identity') is False)
        for i, (guess, normal) in enumerate(zip(track['search_uv'], track['normals'])):
            if template['source_contrast'][i] < (12. if circular or vertical or printed else 24.):
                continue
            tangent = np.array([-normal[1], normal[0]])
            xy = sample_xy[i]
            pix = np.rint(xy).astype(int)
            if ((pix<0).any() or (pix>=[1920, 1080]).any()
                    or exclusion[pix[..., 1], pix[..., 0]].any()):
                continue
            values = source_values[i]
            delta = values[4:]-values[:-4]
            k = int(np.argmax(delta)); strength = float(delta[k])
            if strength < (8. if circular or vertical or printed else 24.) or k == 0 or k == len(delta)-1:
                continue
            # Multiple strong separated transitions cannot identify an exterior edge.
            competing = np.flatnonzero(delta>.8*strength)
            if len(competing) and np.ptp(competing)*.5 > 4.:
                continue
            # Read the visible material transition itself, rather than emitting
            # the transported search coordinate or a strongest-edge integer.
            if circular or printed:
                centre = k+2
                before = values[max(0, centre-8):centre]
                after = values[centre+4:min(len(values), centre+10)]
                if not len(before) or not len(after):
                    continue
                low, high = float(before.min()), float(np.median(after))
                crossing_lo, crossing_hi = max(0, centre-8), min(len(values)-1, centre+8)
            else:
                low, high = float(np.median(values[:5])), float(np.median(values[-5:]))
                crossing_lo, crossing_hi = 0, len(values)-1
            if high-low < (12. if circular or printed else 24.):
                continue
            half = .5*(low+high)
            crossings = np.flatnonzero((values[:-1] <= half)&(values[1:] > half))
            crossings = crossings[(crossings >= crossing_lo)&(crossings < crossing_hi)]
            if len(crossings) != 1:
                continue
            crossing = int(crossings[0])
            offset = offsets[crossing]+.5*(half-values[crossing])/(values[crossing+1]-values[crossing])
            if abs(offset-offsets[k+2]) > 3.:
                continue
            if (circular or printed) and abs(offset) > 3.:
                continue
            uv = guess+offset*normal
            if (board_bracket_candidates(uv[None], bands, exclusion)[0] < 0
                    and not independently_known_board):
                continue
            transition = []
            for fraction in (.25, .75):
                level = low+fraction*(high-low)
                locations = np.flatnonzero((values[:-1] <= level)&(values[1:] > level))
                locations = locations[(locations >= crossing_lo)&(locations < crossing_hi)]
                if len(locations) != 1:
                    break
                j = int(locations[0])
                transition.append(float(offsets[j]+.5*(level-values[j])/(values[j+1]-values[j])))
            if len(transition) != 2 or not transition[0] <= offset <= transition[1]:
                continue
            observed.append((i, uv, normal, strength,
                             np.asarray(transition)-offset))
        return observed

    def read(self, image, mask, binding, *, precomputed_measurement=None, person_components=None):
        source = binding.get('source_sha256', binding.get('video_sha256'))
        if source != self.source_sha256 or binding['time_base'] != self.time_base:
            raise ValueError('Sequential board reader source identity differs')
        pts = int(binding.get('source_pts', binding.get('native_pts')))
        shot = binding.get('shot_id')
        if self.previous_pts is not None and pts <= self.previous_pts:
            raise ValueError('Board reader requires strictly ordered native PTS')
        if precomputed_measurement is None:
            board = measure_board_source_frame(image, mask, binding)
            bands = board['boundary_candidates']
            ice_bands = board.get('ice_scope_boundary_candidates', bands)
        elif isinstance(precomputed_measurement, dict):
            bands = precomputed_measurement['boundary_candidates']
            ice_bands = precomputed_measurement.get('ice_scope_boundary_candidates', bands)
        else:
            bands = precomputed_measurement
            ice_bands = bands
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        actual = int(binding['actual_pts'])
        if self.actual_successor:
            clock_adjacent = self.actual_successor.get(getattr(self, 'previous_actual_pts', None)) == actual
        else:
            # A caller that has not supplied the frozen clock gets candidates,
            # never a verified temporal admission from an assumed source rate.
            clock_adjacent = False
        consecutive = (self.previous_gray is not None and self.previous_shot == shot
                       and clock_adjacent)
        tracking_mask = mask
        local_clearance = None
        if person_components is not None:
            from .vision_board_local_clearance import KnownBoardMaterialBoxClearance
            if not hasattr(self, 'box_clearance'):
                self.box_clearance = KnownBoardMaterialBoxClearance(self.actual_successor)
            local_clearance = self.box_clearance
            local_clearance.begin(image, mask, binding, person_components)
            # This mask navigates source-confirmed entities only. Admitted UV
            # remains subject to the original union or the two-image local
            # boundary proof below; no global exclusion is changed.
            tracking_mask = local_clearance.current['hard']
        self._current_corner_templates(image, gray, tracking_mask, binding)
        tracks = self._track(gray, tracking_mask) if consecutive else {}
        # All frames can discover newly visible source instances; no temporal skip.
        measured_tracks = [name for name, track in tracks.items()
                           if track.get('previous_frame_RGB_measured')]
        discovered = self._template_matches(gray, mask, bands, shot, tracked_names=measured_tracks)
        discovered.update(self._direct_current_anchors(binding, tracking_mask))
        for name, found in discovered.items():
            if (found['identity_method'] == 'exact_same_exposure_source_annotation_and_BGR_bytes'
                    and found['template']['anchor'].get('human_physical_identity_anchor') is True):
                tracks[name] = found
                continue
            if name not in tracks:
                tracks[name] = found
            else:
                # Different source anchors can sample different along-edge points.
                # Compare the observed edge's normal position, not point array sizes.
                edge = np.asarray(found['search_uv'], np.float64)
                centre = edge.mean(0)
                _, _, axes = np.linalg.svd(edge-centre, full_matrices=False)
                mismatch = np.median(np.abs((tracks[name]['search_uv']-centre)@axes[-1]))
                if mismatch <= 8.:
                    if not tracks[name].get('previous_frame_RGB_measured'):
                        tracks[name] = found
                    continue
                # A disagreeing match is an identity jump, never a replacement UV.
                tracks.pop(name)
        rows = []
        profile_image = image.astype(np.float32)
        for name, track in tracks.items():
            anchor = track['template']['anchor']
            physical_name = anchor['physical_material_id']
            current_identity_authorized = anchor.get('source_identity_confirmed') is True
            if anchor.get('source_identity_reference_requires_current_shot'):
                current_identity_authorized &= _identity_reference_allows(
                    anchor['source_annotation_reference'], anchor['advertisement_number'],
                    shot, self.identity_authorities)
            board_material = None
            if current_identity_authorized and track.get('current_instance_identity_confirmed') is True:
                board_material = dict(kind='known_original_source_printed_board_material',
                    physical_material_id=physical_name, source_identity_confirmed=True,
                    source_identity_authority=anchor['source_annotation_reference'],
                    source_anchor_id=anchor['anchor_id'], current_actual_pts=actual,
                    current_native_pts=pts, decoded_native_BGR_sha256=binding.get('decoded_native_BGR_sha256', binding.get('decoded_BGR_sha256')),
                    source_entity_identity_method=track['identity_method'],
                    actual_original_source_printed_board_membership_confirmed=True,
                    source_physical_surface='actual_confirmed_advertisement_on_rink_board_face',
                    unknown_rail_column_does_not_invent_rail_UV=True,
                    model_projection_or_fit_residual_used_for_identity=False)
            points = self._read_edge(profile_image, mask, bands, track,
                known_source_board_material=board_material)
            local_proof = {}
            if local_clearance is not None:
                extra, proof = local_clearance.additional_points(self, track,
                    physical_identity_confirmed=(current_identity_authorized
                        and track.get('current_instance_identity_confirmed') is True),
                    known_source_board_material=board_material)
                existing = {int(p[0]) for p in points}
                added = [p for p in extra if int(p[0]) not in existing]
                points = sorted([*points, *added], key=lambda p: p[0])
                local_proof = {int(p[0]): proof[int(p[0])] for p in added}
            admitted = (bool(points) and anchor.get('source_identity_confirmed') is True
                        and current_identity_authorized
                        and track.get('current_instance_identity_confirmed') is True)
            method = ('same_shot_actual_RGB_temporal_identity' if track['continuity']
                      else 'confirmed_original_source_identity_anchor')
            temporal = dict(kind=method, physical_material_id=physical_name, shot_id=shot,
                current_actual_pts=actual, same_physical_entity_confirmed=admitted,
                current_frame_RGB_measured=bool(points),
                model_projection_or_fit_residual_used_for_identity=False)
            if track['continuity']:
                temporal.update(previous_shot_id=self.previous_shot,
                    previous_actual_pts=self.previous_actual_pts,
                    previous_decoded_BGR_sha256=self.previous_BGR_sha256,
                    previous_frame_RGB_measured=bool(track.get('previous_frame_RGB_measured') or local_proof),
                    source_RGB_tracking_confirmed=True,
                    current_feature_semantics_remeasured=bool(points))
                admitted &= temporal['previous_frame_RGB_measured']
                temporal['same_physical_entity_confirmed'] = admitted
            else:
                temporal.update(source_identity_authority=anchor['source_annotation_reference'],
                    source_identity_confirmed=current_identity_authorized,
                    unique_current_source_context_verified=track.get('current_instance_identity_confirmed') is True,
                    named_instance_discriminating_context_features=track.get('named_instance_discriminating_context_features'),
                    compared_source_named_instances=track.get('compared_source_named_instances'))
            row = dict(physical_material_id=physical_name, advertisement_number=anchor['advertisement_number'],
                edge=anchor['edge'], curve=anchor['curve'], shot_id=shot,
                actual_pts=actual, native_pts=pts,
                uv=np.asarray([p[1] for p in points], np.float64).reshape(-1, 2),
                normal=np.asarray([p[2] for p in points], np.float64).reshape(-1, 2),
                anchor_point_indices=np.asarray([p[0] for p in points], np.int32),
                source_annotation_reference=anchor['source_annotation_reference'],
                source_entity_context_uv=(track.get('source_entity_context_uv')
                    if track.get('source_entity_context_uv') is not None else np.asarray([
                        track['search_uv'].min(0), track['search_uv'].max(0)], np.float64)),
                source_entity_context_is_navigation_only=True,
                source_anchor_id=anchor['anchor_id'], identity_method=track['identity_method'],
                source_actual_contour_evidence=anchor.get('source_actual_contour_evidence'),
                sparse_original_outer_anchor_queries_only=anchor.get('sparse_original_outer_anchor_queries_only', False),
                source_temporal_continuity=bool(track['continuity']), identity_support=track['identity_matches'],
                actual_source_RGB_measured=bool(points),
                temporal_evidence=temporal,
                model_projection_or_fit_residual_used_for_identity=False,
                source_identity_authority=anchor['source_annotation_reference'],
                source_identity_confirmed=current_identity_authorized,
                source_authority_current_shot_confirmed=current_identity_authorized,
                source_readout_uncertainty=dict(
                    kind='actual_source_readout_blur_and_quantization',
                    actual_source_RGB_measured=bool(points),
                    paint_width_used_as_center_tolerance=False,
                    measurement_method='visible_outer_transition_q25_q50_q75_source_chroma_projection',
                    quantization_step_px=1.,
                    source_native_pixel_quantization_step_px=1.,
                    source_interpolation_grid_px=1./32., profile_sampling_step_px=.5,
                    source_sampling_operator=SOURCE_SAMPLING_OPERATOR,
                    actual_readout_blur_interval_px=np.asarray([p[4] for p in points], np.float64).reshape(-1, 2),
                    normal_interval_px=np.asarray([p[4] for p in points], np.float64).reshape(-1, 2)),
                normal_measurement_origin='source_anchor_transport_navigation_only',
                source_neighbour_adjacency_confirmed=False,
                admission_status=('admitted_source_advertisement_temporal_identity' if admitted
                                  else 'pending_source_identity_or_current_visible_pixel'),
                status=('measured_visible_source_boundary' if admitted else
                        'measured_visible_boundary_requires_current_instance_and_pixel_review' if points
                        else 'unknown_no_unoccluded_unique_visible_edge'),
                static_material_confirmed=admitted, pixel_location_confirmed=admitted, allowed_for_fit=admitted)
            if local_proof:
                row.update(candidate_local_source_clearance=local_proof,
                    historical_person_union_modified=False,
                    unknown_material_cleared=False)
            if board_material is not None and points:
                direct_scope = board_bracket_candidates(row['uv'], bands, mask) >= 0
                if not direct_scope.all():
                    row.update(source_board_material_membership_evidence=board_material,
                        current_unknown_rail_column_anchor_point_indices=row['anchor_point_indices'][~direct_scope])
            if anchor.get('model_target', {}).get('kind') == 'current_scene_board_print_contour':
                row.update(model_target=dict(anchor['model_target']),
                    observation_semantics='board_print_feature',
                    source_component_correspondence=anchor.get('source_component_correspondence'),
                    source_outer_contour_navigation_is_not_current_measured_UV=True)
            # Missing source transitions split the contour. Normals can become
            # source truth only from adjacent actually measured current UV,
            # never from the anchor homography or an occluded span.
            indices = row['anchor_point_indices']
            distances = np.linalg.norm(np.diff(row['uv'], axis=0), axis=1)
            cuts = np.flatnonzero((np.diff(indices) != 1) | (distances > 6.)) + 1
            for segment_number, take in enumerate(np.split(np.arange(len(points)), cuts)):
                segment = dict(row, source_boundary_segment_index=segment_number,
                    uv=row['uv'][take], normal=row['normal'][take],
                    anchor_point_indices=indices[take],
                    source_readout_uncertainty=dict(row['source_readout_uncertainty']))
                if local_proof:
                    segment['candidate_local_source_clearance'] = {
                        int(index): local_proof[int(index)] for index in indices[take]
                        if int(index) in local_proof}
                if row.get('current_unknown_rail_column_anchor_point_indices') is not None:
                    segment['current_unknown_rail_column_anchor_point_indices'] = np.intersect1d(
                        indices[take], row['current_unknown_rail_column_anchor_point_indices'])
                blur = row['source_readout_uncertainty']['actual_readout_blur_interval_px'][take]
                if len(take) >= 2:
                    tangent = np.gradient(segment['uv'], axis=0)
                    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
                    current_normal = np.c_[-tangent[:, 1], tangent[:, 0]]
                    orientation = np.sum(current_normal*segment['normal'], axis=1)
                    current_normal[orientation < 0] *= -1
                    projection = np.sum(current_normal*segment['normal'], axis=1)
                    segment.update(normal=current_normal,
                        normal_measurement_origin='same_current_source_visible_contour_adjacent_measured_UV',
                        normal_from_actual_source_neighbour_pixels=True,
                        source_neighbour_adjacency_confirmed=True)
                    track['normals'][indices[take]] = current_normal
                    blur = blur*projection[:, None]
                segment['source_readout_uncertainty'].update(
                    actual_readout_blur_interval_px=blur, normal_interval_px=blur)
                rows.append(segment)
            for point in points:
                track['search_uv'][point[0]] = point[1]
            track['previous_frame_RGB_measured'] = admitted
        self.previous_gray, self.previous_pts, self.previous_shot = gray, pts, shot
        self.previous_actual_pts = actual
        self.previous_BGR_sha256 = binding.get('decoded_native_BGR_sha256', binding.get('decoded_BGR_sha256'))
        self.tracks = tracks
        if local_clearance is not None:
            local_clearance.finish()
        return dict(schema=SCHEMA, source=binding, boundary_candidates=bands, advertisement_rows=rows,
            source_exposure_processed=True, unknown_never_admitted=True,
            source_image_navigation_not_fitting_UV=True,
            conservative_ice_exclusion_mask=conservative_ice_exclusion(mask, ice_bands))
