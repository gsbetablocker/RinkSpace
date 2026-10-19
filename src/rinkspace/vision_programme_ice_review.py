"""Every-exposure native RGB measurements for the complete modeled ice inventory.

World paths locate broad searches only. None of their projected coordinates are
output observations. Each source candidate retains both measured paint sides or
the declared material interface. Candidates need physical source-image review
before admission; absence, occlusion, competing paint and unresolved roles remain
explicit. The caller owns decoding, exclusion masks, storage and visual review.
"""
from __future__ import annotations

from collections import defaultdict
import cv2
import numpy as np

from .vision_source_shapes import (_MaterialProfileFrame,
    read_source_chromatic_stroke_centers, read_source_dark_stroke_profiles,
    read_connected_source_material_profiles)


STATUS = {"no_source_query_in_frame": 0, "actual_RGB_candidate_pending_review": 1,
          "source_material_missing": 2, "competing_or_unresolved_material": 3,
          "source_exclusion_intersects_candidate": 4, "physical_role_unresolved": 5}
CANDIDATE_FLAGS = {"source_material_unresolved_or_truncated": 1,
                   "source_exclusion_intersects_candidate": 2,
                   "multiple_competing_source_candidates": 4,
                   "physical_identity_surface_and_reflection_review_pending": 8}


def _batched_actual_profiles(frame, query_uv, normals, intervals, signals):
    """Read complete native source profiles once, with every query retained.

    This is the same signed-interface q50 operator used by the public material
    reader. Grouping only accelerates RGB sampling; native offsets and measured
    transitions remain separate from search navigation and paint footprint.
    """
    count = len(query_uv)
    size = int(np.ceil(np.max(intervals[:, 1]-intervals[:, 0])))+1 if count else 0
    relative = np.arange(size, dtype=float)
    offsets = intervals[:, :1]+relative[None]
    valid_offset = offsets <= intervals[:, 1:]+.25
    tangent = np.c_[-normals[:, 1], normals[:, 0]]
    profiles = np.empty((count, size), np.float64)
    rgb = frame.rgb.astype(np.float32)
    luminance = frame.luminance.astype(np.float32)
    fields = {"dark": luminance, "red": rgb[..., 0]-rgb[..., 1],
              "blue": rgb[..., 2]-rgb[..., 0]}
    # OpenCV remap destination dimensions must stay below SHRT_MAX. Every
    # query is streamed through a chunk; this is not a search/sample cap.
    for signal in np.unique(signals):
        chosen = np.flatnonzero(signals == signal)
        for start in range(0, len(chosen), 4096):
            ids = chosen[start:start+4096]
            samples = (query_uv[ids, None, None]+normals[ids, None, None]*offsets[ids, None, :, None]
                +tangent[ids, None, None]*np.arange(-2, 3)[None, :, None, None])
            shape = samples.shape
            xy = samples.reshape(-1, size, 2).astype(np.float32)
            # OpenCV 5 changed float-map interpolation. Keep the established
            # native 1/32 source sampling lattice in either installed version.
            xy = np.rint(xy*32.)/32.
            native = cv2.remap(fields[str(signal)], xy[..., 0], xy[..., 1], cv2.INTER_LINEAR)
            profiles[ids] = np.median(native.reshape(shape[:3]), axis=1)
    # Tail extension is for smoothing only. It is never a material support or
    # output point; full native endpoint intervals remain in the result.
    for i in range(count):
        end = int(valid_offset[i].sum())
        if end < size:profiles[i, end:] = profiles[i, end-1]
    smoothed = cv2.GaussianBlur(profiles.astype(np.float64), (5, 1), 1., borderType=cv2.BORDER_REPLICATE)
    return offsets, profiles, smoothed, valid_offset


def _actual_profile_candidates(point, normal, offsets, profile, smooth, *, operator, signal, transition,
                               with_quantiles=False):
    """Complete observed bands/interfaces, retaining all competing candidates."""
    if len(offsets) < 20:
        return []
    second = np.diff(smooth, n=2)
    noise = float(1.4826*np.median(np.abs(second-np.median(second))))
    gradient = np.diff(smooth)
    floor = max(1., 3*noise)
    candidates = []
    def quantile_interface(a, b, sign):
        mass = np.maximum(sign*gradient[a:b], 0.)
        amount = float(mass.sum())
        if amount <= floor:
            return None
        q = np.interp([.25, .5, .75], np.r_[0., np.cumsum(mass)]/amount, offsets[a:b+1])
        return point[None]+normal[None]*q[:, None]
    if operator == "material_interface":
        sign = 1 if transition == "dark_to_light" else -1
        mass = sign*gradient
        active = mass > max(.25, noise)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])+1
        merged = []
        for a, b in zip(starts, ends):
            if merged and a-merged[-1][1] < 5:merged[-1][1] = int(b)
            else:merged.append([int(a), int(b)])
        for a, b in merged:
            lo, hi = max(0, a-4), min(len(offsets)-1, b+4)
            quantiles = quantile_interface(lo, hi, sign)
            if quantiles is None:continue
            uv = quantiles[1]
            cropped = lo == 0 or hi == len(offsets)-1
            reversal = bool((sign*gradient[a:b] < -floor).any())
            item = (uv, np.array([uv, uv]), cropped or reversal)
            candidates.append((*item, np.array([quantiles, quantiles])) if with_quantiles else item)
        return candidates
    flank = min(10, max(3, len(offsets)//10))
    background = np.linspace(float(np.median(profile[:flank])), float(np.median(profile[-flank:])), len(profile))
    material = np.maximum(background-smooth if signal == "dark" else smooth-background, 0.)
    material_floor = max(2., 3*noise, .1*float(material.max()))
    active = material > material_floor
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    components = []
    for a, b in zip(starts, ends):
        if b-a < 3:continue
        if components and a-components[-1][1]-1 <= 3:components[-1][1] = int(b)
        else:components.append([int(a), int(b)])
    sign = -1 if signal == "dark" else 1
    for a, b in components:
        lo, hi = max(0, a-10), min(len(offsets)-1, b+10)
        mass = material[lo:hi+1]
        total = float(mass.sum())
        if total <= 0:continue
        # Intensity mass only divides two searches; it is never the output
        # center. The center is the midpoint of two actual signed interfaces.
        split_offset = float(np.interp(.5, (np.cumsum(mass)-mass*.5)/total, offsets[lo:hi+1]))
        split = int(np.clip(np.searchsorted(offsets, split_offset), lo+1, hi-1))
        entering = quantile_interface(lo, split, sign)
        leaving = quantile_interface(split, hi, -sign)
        if entering is None or leaving is None:
            continue
        center = .5*(entering[1]+leaving[1])
        cropped = lo == 0 or hi == len(offsets)-1
        item = (center, np.asarray([entering[1], leaving[1]]), cropped)
        candidates.append((*item, np.asarray([entering, leaving])) if with_quantiles else item)
    return candidates


def ice_curve_inventory(scene, role_contract=None):
    """Expand the supplied complete scene, including all house material layers."""
    roles = {r["curve_id"]: r for r in (role_contract or {}).get("curve_records", [])}
    paths = []
    def add(feature, suffix, xyz, operator, signal="dark", transition=None):
        key = feature["id"] + ":" + suffix
        role = roles.get(key, {})
        if role.get("semantic_definition", "").startswith(("unresolved", "physical_paint_raster_role_unresolved")):
            operator = "unresolved"
        if role.get('confirmed_measurement_operator') in ('paint_center', 'paired_material_boundary_midline'):
            operator = 'paint_center'
        elif role.get('confirmed_measurement_operator') in ('material_interface', 'signed_material_transition'):
            operator = 'material_interface'
        if role.get('confirmed_material_signal') in ('dark', 'red', 'blue'):
            signal = role['confirmed_material_signal']
        if role.get('confirmed_material_transition') in ('light_to_dark', 'dark_to_light'):
            transition = role['confirmed_material_transition']
        paths.append(dict(curve_id=key, feature_id=feature["id"], world_curve_role=suffix,
            xyz=np.asarray(xyz, np.float64), operator=operator, material_signal=signal,
            material_transition=transition, source_role_authority=role.get("semantic_definition"),
            source_role_evidence=role.get('source_role_evidence'),
            physical_identity_confirmed_by_navigation=False))
    for feature in scene["ice_features"]:
        kind = feature["kind"]
        if kind == "circle":
            rings = feature.get("rings") or [dict(role="outer", center_xyz=feature["center_xyz"], radius=feature["radius"])]
            angles = np.linspace(0., 2*np.pi, 257)
            directions = np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))]
            for ring in rings:
                suffix = ring["role"]
                xyz = np.asarray(ring["center_xyz"])+float(ring["radius"])*directions
                if "rings" in feature:
                    transition = "dark_to_light" if suffix in ("outer", "inner") else "light_to_dark"
                    add(feature, suffix, xyz, "material_interface", transition=transition)
                    paths[-1]["center_xyz"] = np.asarray(ring["center_xyz"], float)
                else:
                    add(feature, suffix, xyz, "paint_center", "red")
        elif kind == "polyline":
            for index, points in enumerate(feature["segments_xyz"]):
                xyz = np.concatenate([np.linspace(a, b, 129)[:-1] for a, b in zip(points[:-1], points[1:])] + [np.asarray(points[-1:])])
                name = feature["id"]
                signal = "blue" if name in ("B1", "B2") else ("red" if name == "C" or name.startswith(("goal_line", "goal_trapezoid")) else "dark")
                add(feature, f"segment{index}", xyz, "paint_center", signal)
        elif kind == "polygon" and feature["id"].startswith("lane"):
            for index, points in enumerate(feature.get("paint_segments_xyz", [])):
                add(feature, f"edge{index}", np.linspace(points[0], points[-1], 257), "paint_center")
        elif kind == "polygon":
            polygon = np.asarray(feature["polygon_xyz"], float)
            xyz = np.vstack([np.linspace(a, b, 65)[:-1] for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])
            add(feature, "boundary", xyz, "unresolved")
        else:
            raise ValueError("Unhandled modeled ice feature: " + str(feature["id"]))
    keys = [p["curve_id"] for p in paths]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate modeled ice paths")
    return paths


def _source_queries(path, projected, depth, native_hw, spacing, radius):
    """Sample visible path search stations without converting samples to UV facts."""
    h, w = native_hw
    uv = np.asarray(projected, float)
    good = np.isfinite(uv).all(1) & np.isfinite(depth) & (depth > .05)
    output, normals = [], []
    travelled = spacing
    for index in range(len(uv)-1):
        if not good[index:index+2].all():
            travelled = spacing
            continue
        a, b = uv[index:index+2]
        vector = b-a
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        # Extremely off-screen segments need no huge intermediate arange.
        lo, hi = 0., 1.
        for axis, limit in ((0, w-1), (1, h-1)):
            if abs(vector[axis]) < 1e-12:
                if not 0 <= a[axis] <= limit:
                    hi = -1.
                    break
            else:
                ends = sorted((-a[axis]/vector[axis], (limit-a[axis])/vector[axis]))
                lo, hi = max(lo, ends[0]), min(hi, ends[1])
        if hi < lo:
            continue
        start = a+lo*vector
        stop = a+hi*vector
        visible_length = float(np.linalg.norm(stop-start))
        unit = vector/length
        normal = np.array([-unit[1], unit[0]])
        distances = np.arange(0., visible_length+1e-6, spacing)
        # Include the real visible segment endpoint as a search location,
        # while every emitted candidate still comes from RGB material reads.
        if not len(distances) or visible_length-distances[-1] > spacing*.5:
            distances = np.r_[distances, visible_length]
        for distance in distances:
            point = start+distance*unit
            if output and np.linalg.norm(point-output[-1]) < spacing*.5:
                continue
            if path.get("_projected_center") is not None and (point-path["_projected_center"]) @ normal < 0:
                normal_use = -normal
            else:
                normal_use = normal
            output.append(point)
            normals.append(normal_use)
    points = np.asarray(output, float).reshape(-1, 2)
    normals = np.asarray(normals, float).reshape(-1, 2)
    intervals = np.tile([-radius, radius], (len(points), 1)).astype(float)
    # Fit the broad search into actually present native pixels with a halo.
    for axis, limit in ((0, w-1), (1, h-1)):
        movement = np.abs(normals[:, axis]) > 1e-12
        first = np.divide(4-points[:, axis], normals[:, axis], out=np.full(len(points), -np.inf), where=movement)
        second = np.divide(limit-4-points[:, axis], normals[:, axis], out=np.full(len(points), np.inf), where=movement)
        intervals[:, 0] = np.maximum(intervals[:, 0], np.minimum(first, second))
        intervals[:, 1] = np.minimum(intervals[:, 1], np.maximum(first, second))
    enough = intervals[:, 1]-intervals[:, 0] >= 20.
    return points[enough], normals[enough], intervals[enough]


def _candidate_context_excluded(mask, candidate, normal):
    center = np.asarray(candidate, float)
    if center.shape != (2,) or not np.isfinite(center).all():
        return True
    tangent = np.array([-normal[1], normal[0]])
    # Actual candidate context, not a projected paint footprint.
    context = center + np.arange(-8, 9)[:, None]*normal
    context = (context[None]+np.arange(-3, 4)[:, None, None]*tangent).reshape(-1, 2)
    ix = np.rint(context).astype(int)
    valid = (ix >= 0).all(1) & (ix < [mask.shape[1], mask.shape[0]]).all(1)
    return not valid.all() or bool(mask[ix[valid, 1], ix[valid, 0]].any())


class IceProgrammeSourceReader:
    """Single-stream complete ice measurement; no decode, solve, PNG or network."""
    def __init__(self, scene, *, source_sha256, time_base, role_contract=None,
                 sample_spacing_px=24., search_radius_px=128., use_batched_profiles=True,
                 profile_backend='cpu'):
        self.paths = ice_curve_inventory(scene, role_contract)
        self.source_sha256, self.time_base = source_sha256, time_base
        self.spacing, self.radius = float(sample_spacing_px), float(search_radius_px)
        self.use_batched_profiles = bool(use_batched_profiles)
        if profile_backend not in ('cpu', 'cuda'):
            raise ValueError('Source profile backend must be cpu or cuda')
        self.profile_backend = profile_backend
        if self.spacing <= 0 or self.radius < 10:
            raise ValueError("Positive native sampling and complete material context required")
        self._feedback_base_paths = [dict(p) for p in self.paths]
        self.inventory = [{k: v for k, v in p.items() if k not in ("xyz", "center_xyz")}
                          for p in self.paths]

    def read(self, rgb, pts, camera25, exclusion_mask, *, shot_id):
        from calibration.program_tracking.projection import project_fast
        binding = dict(source_sha256=self.source_sha256, source_pts=pts,
                       time_base=self.time_base, native_hw=list(rgb.shape[:2]), shot_id=shot_id)
        from .feedback_runtime import SourceFeedbackRead
        feedback = SourceFeedbackRead(binding,
            entity_ids=[p["feature_id"] for p in self._feedback_base_paths] + [p["curve_id"] for p in self._feedback_base_paths],
            tags=["ice_paint", "measurement_operator"])
        self.paths = feedback.apply_inventory(self._feedback_base_paths)
        self.inventory = [{k: v for k, v in p.items() if k not in ("xyz", "center_xyz")} for p in self.paths]
        frame = _MaterialProfileFrame(rgb, binding)
        rgb, binding = frame.rgb, frame.binding
        exclusion = np.asarray(exclusion_mask, bool)
        if exclusion.shape != rgb.shape[:2]:
            raise ValueError("Same-exposure native exclusion mask required")
        query_uv, query_normal, query_interval, query_curve = [], [], [], []
        curve_status = np.zeros(len(self.paths), np.uint8)
        batches = defaultdict(list)
        for ci, path in enumerate(self.paths):
            projected, depth = project_fast(camera25, path["xyz"])
            search_path = dict(path)
            if "center_xyz" in path:
                search_path["_projected_center"] = project_fast(camera25, path["center_xyz"][None])[0][0]
            points, normal, intervals = _source_queries(search_path, projected, depth, rgb.shape[:2], self.spacing, self.radius)
            if not len(points):
                continue
            first = len(query_uv)
            query_uv.extend(points); query_normal.extend(normal); query_interval.extend(intervals)
            query_curve.extend([ci]*len(points))
            if path["operator"] == "unresolved":
                curve_status[ci] = STATUS["physical_role_unresolved"]
                # Unknown semantics never suppress source-pixel inspection.
                # Read the complete band hypothesis; role is still withheld.
                key = ("paint_center", path["material_signal"], None)
            else:
                key = (path["operator"], path["material_signal"], path["material_transition"])
            batches[key].extend(range(first, first+len(points)))
        query_uv = np.asarray(query_uv, float).reshape(-1, 2)
        query_normal = np.asarray(query_normal, float).reshape(-1, 2)
        query_interval = np.asarray(query_interval, float).reshape(-1, 2)
        query_curve = np.asarray(query_curve, np.int16)
        query_status = np.full(len(query_uv), STATUS["physical_role_unresolved"], np.uint8)
        candidate_uv, candidate_sides, candidate_query, candidate_flags = [], [], [], []
        candidate_quantiles = []
        fast_profiles = None
        if self.use_batched_profiles and len(query_uv):
            signals = np.array([self.paths[ci]["material_signal"] for ci in query_curve])
            if self.profile_backend == 'cuda':
                from .vision_source_cuda_profiles import batched_actual_profiles_cuda
                fast_profiles = batched_actual_profiles_cuda(frame, query_uv, query_normal, query_interval, signals)
            else:
                fast_profiles = _batched_actual_profiles(frame, query_uv, query_normal, query_interval, signals)
        for (operator, signal, transition), indices in batches.items():
            if fast_profiles is not None:
                offsets, profiles, smoothed, valid_offsets = fast_profiles
                for qi in indices:
                    valid = valid_offsets[qi]
                    values = _actual_profile_candidates(query_uv[qi], query_normal[qi], offsets[qi, valid],
                        profiles[qi, valid], smoothed[qi, valid], operator=operator, signal=signal,
                        transition=transition, with_quantiles=True)
                    clean_count = 0
                    for uv, sides, source_bad, quantiles in values:
                        excluded = any(_candidate_context_excluded(exclusion, p, query_normal[qi]) for p in [uv, *sides])
                        bits = int(source_bad) | (int(excluded) << 1) | (int(len(values) > 1) << 2) | 8
                        candidate_uv.append(uv); candidate_sides.append(sides)
                        candidate_quantiles.append(quantiles)
                        candidate_query.append(qi); candidate_flags.append(bits)
                        clean_count += not source_bad and not excluded
                    query_status[qi] = (STATUS["source_material_missing"] if not values else
                        STATUS["actual_RGB_candidate_pending_review"] if len(values) == clean_count == 1 else
                        STATUS["competing_or_unresolved_material"])
                    ci = query_curve[qi]
                    if self.paths[ci]['operator'] != 'unresolved':
                        curve_status[ci] = max(curve_status[ci], query_status[qi])
                continue
            common = dict(binding=binding, search_points_uv=query_uv[indices], normals=query_normal[indices],
                search_intervals_px=query_interval[indices], keep_profiles=False, _validated_frame=frame)
            if operator == "material_interface":
                readings = read_connected_source_material_profiles(rgb, transition=transition, **common)["readings"]
            elif signal in ("red", "blue"):
                readings = read_source_chromatic_stroke_centers(rgb,
                    material_signal="red_minus_green" if signal == "red" else "blue_minus_red", **common)["readings"]
            else:
                readings = read_source_dark_stroke_profiles(rgb, **common)["readings"]
            for qi, reading in zip(indices, readings):
                if operator == "material_interface":
                    candidates = reading["all_actual_source_material_candidates"]
                else:
                    key = "all_actual_source_chromatic_stroke_candidates" if signal in ("red", "blue") else "all_actual_source_dark_stroke_candidates"
                    candidates = reading[key]
                local_count = 0
                clean_count = 0
                for candidate in candidates:
                    if operator == "material_interface":
                        uv = np.asarray(candidate["actual_source_transition_candidate_uv"], float)
                        sides = np.array([uv, uv])
                    elif signal in ("red", "blue"):
                        uv = np.asarray(candidate["actual_source_center_candidate_uv"], float)
                        sides = np.asarray([x["actual_source_uv"] for x in candidate["actual_RGB_band_interfaces"]], float)
                    else:
                        uv = np.asarray(candidate["actual_source_dark_stroke_center_candidate_uv"], float)
                        sides = np.asarray([candidate["actual_RGB_entering_interface"]["actual_source_transition_candidate_uv"],
                                            candidate["actual_RGB_leaving_interface"]["actual_source_transition_candidate_uv"]], float)
                    if uv.shape != (2,) or sides.shape != (2, 2) or not np.isfinite(np.r_[uv, sides.ravel()]).all():
                        continue
                    source_bad = any(candidate["source_flags"].values())
                    excluded = any(_candidate_context_excluded(exclusion, p, query_normal[qi]) for p in [uv, *sides])
                    bits = int(source_bad) | (int(excluded) << 1) | (int(len(candidates) > 1) << 2)
                    # Physical identity and reflection/material review are
                    # required even when both actual interfaces read cleanly.
                    bits |= 8
                    candidate_uv.append(uv); candidate_sides.append(sides)
                    # Public fallback reader does not expose its full native
                    # gradient quantiles. Missing support stays explicit.
                    candidate_quantiles.append(np.full((2, 3, 2), np.nan))
                    candidate_query.append(qi); candidate_flags.append(bits)
                    local_count += 1
                    clean_count += not source_bad and not excluded
                query_status[qi] = (STATUS["source_material_missing"] if local_count == 0 else
                    STATUS["actual_RGB_candidate_pending_review"] if local_count == clean_count == 1 else
                    STATUS["competing_or_unresolved_material"])
                ci = query_curve[qi]
                curve_status[ci] = max(curve_status[ci], query_status[qi])
        result = dict(source_binding=binding, decoded_native_RGB_sha256=frame.digest,
            curve_status=curve_status, query_curve=query_curve, query_uv=query_uv.astype(np.float32),
            query_normal=query_normal.astype(np.float32), query_interval=query_interval.astype(np.float32),
            query_status=query_status, candidate_uv=np.asarray(candidate_uv, np.float32).reshape(-1, 2),
            candidate_sides=np.asarray(candidate_sides, np.float32).reshape(-1, 2, 2),
            candidate_side_quantiles_uv=np.asarray(candidate_quantiles, np.float64).reshape(-1, 2, 3, 2),
            source_readout_quantization_step_px=1., source_interpolation_grid_px=1./32.,
            candidate_query=np.asarray(candidate_query, np.int32), candidate_flags=np.asarray(candidate_flags, np.uint8),
            fitting_admissible=False, all_modeled_ice_paths_processed=True,
            source_RGB_measured=True, trained_network_loaded=False,
            source_measurement_method=("batched_native_RGB_material_candidates" if self.use_batched_profiles
                                       else "existing_public_source_shape_profiles"),
            model_surface_membership_verified=False,
            visual_review_complete=False, projected_UV_used_as_measurements=False)

        result['resolved_measurement_roles'] = self.inventory
        result['consumed_case_sha256'] = sorted({ref['content_sha256'] for ref in feedback.used.values()})
        result['consumed_fact_sha256'] = sorted({ref['sha256'] for match in getattr(feedback, 'applicable', []) if match['case_id'] in feedback.used for ref in match['case'].get('evidence_refs', [])})
        result['source_feedback'] = feedback.finish(rgb, dict(
            query_count=len(query_uv), candidate_count=len(candidate_uv),
            candidates_remain_pending_physical_review=True,
            operators={p['curve_id']:p['operator'] for p in self.paths}),
            tool='IceProgrammeSourceReader.read')
        return result
