"""Source-bound regression diagnostics against the existing accepted S01 outlines.

This is an error detector, not another reader or an approval mechanism.  It reads
the *actual_source_uv* and the saved query connectivity, never coherent/fitted
coordinates.  An accepted packet is deliberately a partial reference: ads and
unannotated parts of curves are unscored, not false positives.  The tolerances
below are engineering settings, not a statistical confidence guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "rinkspace.source-quality.v1"


@dataclass(frozen=True)
class QualityProtocol:
    localization_tolerance_px: float = 3.0
    coverage_tolerance_px: float = 5.0
    minimum_support_fraction: float = 0.8
    sample_spacing_px: float = 4.0
    endpoint_margin_px: float = 3.0

    def __post_init__(self):
        for name in ("localization_tolerance_px", "coverage_tolerance_px",
                     "sample_spacing_px", "endpoint_margin_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite pixel distance")
        if not 0 < self.minimum_support_fraction <= 1:
            raise ValueError("minimum_support_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class EvaluationUsage:
    """Record the run's dependencies; absent evidence is not clean provenance."""

    training_exposure: str = "unknown"
    tuning_exposure: str = "unknown"
    human_mask_used: bool | None = None
    same_exposure_reference_used: bool | None = None
    human_seed_pts: tuple[int, ...] = ()
    data_dependencies_audited: bool = False

    def __post_init__(self):
        for name in ("training_exposure", "tuning_exposure"):
            if getattr(self, name) not in {"unknown", "included", "excluded"}:
                raise ValueError(f"{name} must be unknown, included or excluded")
        for name in ("human_mask_used", "same_exposure_reference_used"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true, false or null")
        if any(type(pts) is not int or pts < 0 for pts in self.human_seed_pts):
            raise ValueError("human_seed_pts must contain native integer PTS")


@dataclass(frozen=True)
class ReferenceSegment:
    feature_id: str
    target_id: str
    segment_index: int
    layer: str
    measurement: str
    uv: tuple[tuple[float, float], ...]
    closed: bool
    geometry: dict[str, Any]


@dataclass(frozen=True)
class ReferenceFrame:
    source_sha256: str
    native_pts: int
    time_base: str
    shot_id: str
    native_hw: tuple[int, int]
    decoded_rgb_sha256: str | None
    packet_sha256: str
    binding_sha256: str
    acceptance_sha256: str
    segments: tuple[ReferenceSegment, ...]
    reviewed_layers: tuple[str, ...]


def _load_file(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def _points(value, name):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite native UV pairs")
    return array


def _layer(identity):
    if identity.startswith("ice."):
        return "ice"
    if identity.startswith(("board_top.", "board_contact.")):
        return "board_boundary"
    return None


def load_accepted_reference(packet_path, binding_path, acceptance_path, *,
                            reviewed_layers=("ice", "board_boundary"),
                            shot_id="s01"):
    """Verify exact receipt/file/source bindings, retaining only reviewed scope.

    Stored Windows paths are descriptions; the supplied files and their byte
    hashes determine identity.  This works in the offline audit directory too.
    No loose `user_accepted` boolean can replace the signed-off file receipt.
    """
    if not set(reviewed_layers) <= {"ice", "board_boundary"}:
        raise ValueError("This acceptance scope covers ice and board boundaries only")
    packet, packet_sha = _load_file(packet_path)
    binding, binding_sha = _load_file(binding_path)
    acceptance, acceptance_sha = _load_file(acceptance_path)
    if acceptance.get("accepted") is not True:
        raise ValueError("Reference acceptance is not affirmative")
    source = packet["source"]
    pts = source["native_pts"]
    if type(pts) is not int or pts < 0 or source.get("actual_pts", pts) != pts:
        raise ValueError("Reference is not a native source exposure")
    entries = [row for row in acceptance.get("frames", []) if row.get("native_pts") == pts]
    if len(entries) != 1:
        raise ValueError("Reference needs one exact native-PTS acceptance receipt")
    receipt = entries[0]
    if receipt.get("packet_sha256") != packet_sha or receipt.get("binding_sha256") != binding_sha:
        raise ValueError("Accepted packet or binding SHA does not match supplied bytes")
    if binding.get("source_packet", {}).get("sha256") != packet_sha:
        raise ValueError("Binding does not refer to this accepted source packet")
    clock = Fraction(source["time_base"])
    if (clock <= 0 or binding.get("source_pts") != pts
            or Fraction(binding["source_time_base"]) != clock
            or packet.get("source_pts", pts) != pts):
        raise ValueError("Packet and binding source clocks/exposures differ")
    digest = source["source_video_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("Reference lacks an actual source video SHA256")
    hw = tuple(source["native_hw"])
    if len(hw) != 2 or any(type(n) is not int or n <= 0 for n in hw):
        raise ValueError("Invalid native image dimensions")
    mapped = {}
    for observation in binding.get("observations", []):
        key = (observation["source_feature_id"], observation["source_segment_index"])
        mapped.setdefault(key, set()).add(observation["target_id"])
    segments = []
    for feature in packet.get("features", []):
        identity = feature["id"]
        layer = _layer(identity)
        if layer not in reviewed_layers:
            continue
        if feature.get("projected_uv_used_as_observations") is True:
            raise ValueError(f"{identity} stores projection-derived reference coordinates")
        for index, segment in enumerate(feature.get("segments", [])):
            uv = _points(segment.get("uv", []), f"{identity}/{index}")
            if not len(uv):
                continue
            targets = mapped.get((identity, index), set())
            # One source segment may bind several model primitives.  Never pick
            # a target by nearest position or discard that identity ambiguity.
            target = feature.get("target_id", identity)
            if not targets or target not in targets:
                raise ValueError(f"Accepted segment {identity}/{index} has no matching physical binding")
            measurement = "centreline" if layer == "board_boundary" or feature.get("measurement_operator") == "paint_center" else "contour"
            segments.append(ReferenceSegment(identity, target, index, layer, measurement,
                tuple(map(tuple, uv.tolist())), bool(segment.get("closed", False)),
                dict(feature.get("geometric_interpretation") or {})))
    return ReferenceFrame(digest.lower(), pts, str(clock), shot_id, hw,
        source.get("decoded_RGB_sha256"), packet_sha, binding_sha, acceptance_sha,
        tuple(segments), tuple(reviewed_layers))


def _curve(segment, spacing):
    """Interpolate within each saved segment, never across its gaps.

    Accepted circle packets save ellipse vertices.  Their saved analytic ellipse
    may interpolate *between those vertices* only when it reproduces them.  This
    avoids scoring sparse chords as the reference outline.  No unseen arc or
    fitted reader coordinate is created.
    """
    uv = np.asarray(segment.uv, dtype=float)
    geometry = segment.geometry
    if len(uv) > 1 and geometry.get("kind") == "projected_physical_circle_ellipse":
        center = np.asarray(geometry.get("center_uv", []), dtype=float)
        axes = np.asarray(geometry.get("axes_px", []), dtype=float) / 2
        if center.shape == (2,) and axes.shape == (2,) and np.isfinite(axes).all() and np.all(axes > 0):
            angle = np.deg2rad(geometry.get("angle_degrees", 0))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            local = ((uv - center) @ rotation) / axes
            theta = np.unwrap(np.arctan2(local[:, 1], local[:, 0]))
            recovered = np.stack((np.cos(theta), np.sin(theta)), axis=1) * axes @ rotation.T + center
            if np.max(np.linalg.norm(recovered - uv, axis=1)) <= 0.25:
                vertices = []
                for first, last in zip(theta[:-1], theta[1:]):
                    count = max(2, int(np.ceil(abs(last-first) * max(axes) / spacing)) + 1)
                    t = np.linspace(first, last, count)
                    vertices.extend((np.stack((np.cos(t), np.sin(t)), axis=1) * axes @ rotation.T + center)[:-1])
                uv = np.vstack([vertices, recovered[-1]])
    if segment.closed and len(uv) > 1 and np.linalg.norm(uv[-1] - uv[0]) > 1e-9:
        uv = np.vstack([uv, uv[0]])
    return uv


def _project(points, curve, endpoint_margin):
    """Distance and arclength, plus whether a partial label covers the query."""
    points = _points(points, "source readouts")
    if len(curve) == 1:
        distance = np.linalg.norm(points - curve[0], axis=1)
        return distance, np.zeros(len(points)), distance <= endpoint_margin
    starts, directions = curve[:-1], np.diff(curve, axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    keep = lengths > 1e-9
    if not np.any(keep):
        return _project(points, curve[:1], endpoint_margin)
    starts, directions, lengths = starts[keep], directions[keep], lengths[keep]
    raw = np.sum((points[:, None, :] - starts) * directions, axis=2) / lengths**2
    clipped = np.clip(raw, 0, 1)
    distance = np.linalg.norm(points[:, None, :] - starts - clipped[:, :, None] * directions, axis=2)
    nearest = np.argmin(distance, axis=1)
    row = np.arange(len(points))
    local = clipped[row, nearest]
    arc = np.r_[0, np.cumsum(lengths)][nearest] + local * lengths[nearest]
    best = distance[row, nearest]
    # Labels stop at the endpoints.  Do not call an extension into an unmarked
    # region wrong simply because the nearest marked point is far away.
    eligible = ((raw[row, nearest] >= 0) & (raw[row, nearest] <= 1)) | (best <= endpoint_margin)
    return best, arc, eligible


def _sample_curve(curve, spacing):
    if len(curve) == 1:
        return curve.copy(), np.zeros(1)
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(lengths)]
    if distance[-1] < 1e-9:
        return curve[:1].copy(), np.zeros(1)
    stations = np.linspace(0, distance[-1], max(2, int(np.ceil(distance[-1] / spacing)) + 1))
    return np.stack([np.interp(stations, distance, curve[:, i]) for i in range(2)], axis=1), stations


def _project_reference(points, segment, curve, endpoint_margin):
    distance, station, eligible = _project(points, curve, endpoint_margin)
    geometry = segment.geometry
    # A partial lower circle arc is not a label for an unmarked upper arc.
    # Euclidean nearest alone would incorrectly call the latter a 20 px error.
    if (len(segment.uv) > 1 and not segment.closed
            and geometry.get("kind") == "projected_physical_circle_ellipse"):
        center = np.asarray(geometry.get("center_uv", []), dtype=float)
        axes = np.asarray(geometry.get("axes_px", []), dtype=float) / 2
        if center.shape == (2,) and axes.shape == (2,) and np.all(axes > 0):
            angle = np.deg2rad(geometry.get("angle_degrees", 0))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            local = (np.asarray(segment.uv) - center) @ rotation / axes
            theta = np.unwrap(np.arctan2(local[:, 1], local[:, 0]))
            lo, hi = float(np.min(theta)), float(np.max(theta))
            candidate = (points - center) @ rotation / axes
            candidate_angle = np.arctan2(candidate[:, 1], candidate[:, 0])
            middle = (lo + hi) / 2
            candidate_angle = (candidate_angle - middle + np.pi) % (2 * np.pi) - np.pi + middle
            in_arc = (candidate_angle >= lo) & (candidate_angle <= hi)
            eligible &= in_arc | (distance <= endpoint_margin)
    return distance, station, eligible


def _read_runs(row):
    """Validate saved measured-run connectivity against the actual readout list."""
    uv = _points(row.get("actual_source_uv", []), "actual_source_uv")
    indices = row.get("source_indices")
    if indices is None:
        # Older point-only outputs can be localized, but are not invented lines.
        return uv, list(range(len(uv))), [[i] for i in range(len(uv))]
    indices = list(indices)
    if (len(indices) != len(uv) or len(set(indices)) != len(indices)
            or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 0 for i in indices)):
        raise ValueError("actual_source_uv lost its unique native query indices")
    index_to_position = {int(index): position for position, index in enumerate(indices)}
    segments = row.get("segment_ids")
    if segments is not None and (len(segments) != len(uv) or any(
            not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in segments)):
        raise ValueError("Actual source segment IDs lost their measured query correspondence")
    runs, used = [], set()
    for run in row.get("measured_runs", []):
        run_uv = _points(run.get("actual_source_uv", []), "measured_runs actual_source_uv")
        run_indices = list(run.get("source_indices", []))
        if len(run_uv) != len(run_indices):
            raise ValueError("Measured run lost its actual source indices")
        positions = []
        for index, point in zip(run_indices, run_uv):
            if index not in index_to_position or index in used:
                raise ValueError("Measured run invents or duplicates an actual query")
            position = index_to_position[index]
            if not np.allclose(point, uv[position], atol=1e-6, rtol=0):
                raise ValueError("Measured run substitutes fitted UV for actual source UV")
            if segments is not None and run.get("segment_id") != segments[position]:
                raise ValueError("Measured run contradicts the saved source segment IDs")
            if positions and index != indices[positions[-1]] + 1:
                runs.append(positions)
                positions = []
            positions.append(position)
            used.add(index)
        if positions:
            runs.append(positions)
    # Missing connectivity is a point, never an implicit bridge over occlusion.
    runs.extend([[position] for index, position in index_to_position.items() if index not in used])
    return uv, list(map(int, indices)), runs


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "median_px": None, "p95_px": None, "max_px": None}
    return {"count": len(values), "median_px": float(np.median(values)),
            "p95_px": float(np.percentile(values, 95)), "max_px": float(np.max(values))}


def _independence(reference, record, usage):
    reasons = []
    if usage.human_mask_used is True:
        reasons.append("human_reference_mask_used")
    if usage.same_exposure_reference_used is True or record.get("accepted_reference_used_for_guides") is True:
        reasons.append("same_exposure_reference_used")
    if usage.human_seed_pts:
        reasons.append("same_exposure_human_seed" if reference.native_pts in usage.human_seed_pts else "human_seed_in_temporal_chain")
    for row in (record.get("guided") or {}).get("readings", []):
        authority = row.get("identity_authority") or {}
        if authority.get("human_authority") is True:
            digest = (authority.get("packet") or {}).get("sha256")
            reasons.append("same_exposure_human_seed" if digest == reference.packet_sha256 else "human_seed_in_temporal_chain")
    for name in ("training_exposure", "tuning_exposure"):
        if getattr(usage, name) == "included":
            reasons.append(name + "_included")
    unknown = [name for name in ("training_exposure", "tuning_exposure") if getattr(usage, name) == "unknown"]
    unknown += [name for name in ("human_mask_used", "same_exposure_reference_used") if getattr(usage, name) is None]
    if usage.data_dependencies_audited is not True:
        unknown.append("data_dependencies_not_audited")
    return {"independent": not reasons and not unknown,
            "known_dependencies": sorted(set(reasons)), "unknown_dependencies": unknown,
            "usage": asdict(usage)}


def compare_source_frame(reference, reader_record, *, protocol=None, usage=None):
    """Return source-bound failures and machine repair actions without mutation.

    Forward errors measure actual pixels against the reviewed outline.  Reverse
    errors retain distance to *measured runs* and a separate covered-support
    fraction.  Run coverage follows only consecutive actual query indices within
    the same reviewed segment.  It does not assert every intervening pixel was
    independently observed.  No score writes `human_confirmed` or fit eligibility.
    """
    protocol = protocol or QualityProtocol()
    usage = usage or EvaluationUsage()
    binding = reader_record["source_binding"]
    if (binding.get("source_sha256", "").lower() != reference.source_sha256
            or binding.get("source_pts") != reference.native_pts
            or Fraction(binding.get("time_base", "0")) != Fraction(reference.time_base)
            or binding.get("shot_id") != reference.shot_id
            or tuple(binding.get("native_hw", [])) != reference.native_hw):
        raise ValueError("Reader and reference are different source/shot/native exposures")
    if (reference.decoded_rgb_sha256 and binding.get("decoded_native_rgb_sha256")
            and binding["decoded_native_rgb_sha256"] != reference.decoded_rgb_sha256):
        raise ValueError("Reader and reference decoded different RGB pixels at this PTS")
    independence = _independence(reference, reader_record, usage)
    report = {"schema": SCHEMA, "source_sha256": reference.source_sha256,
        "shot_id": reference.shot_id, "native_pts": reference.native_pts,
        "time_base": reference.time_base,
        "evaluation_role": "independent" if independence["independent"] else "regression",
        "reference": {"packet_sha256": reference.packet_sha256,
                      "binding_sha256": reference.binding_sha256,
                      "acceptance_sha256": reference.acceptance_sha256,
                      "reviewed_layers": list(reference.reviewed_layers),
                      "basis": "accepted_saved_outlines_only; not exhaustive pixel ground truth"},
        "protocol": asdict(protocol), "independence": independence,
        "failures": [], "repair_actions": [], "features": [], "unscored": [],
        "human_state_modified": False, "accuracy_accepted": False,
        "cross_shot_accuracy_claim": False}

    def fail(code, feature_id, details, action):
        report["failures"].append({"code": code, "feature_id": feature_id,
                                   "source_pts": reference.native_pts, "details": details})
        report["repair_actions"].append({"action": action, "shot_id": reference.shot_id,
            "first_pts": reference.native_pts, "last_pts": reference.native_pts,
            "feature_id": feature_id, "reason_code": code, "details": details})

    if independence["known_dependencies"]:
        fail("leakage", None, {"dependencies": independence["known_dependencies"],
            "meaning": "Keep and use these regression data; they are not independent accuracy evidence",
            "preserve_regression_data": True, "independent_accuracy_usable": False,
            "required_change": "Correct evaluation claims/setup; do not delete existing masked or seeded results"}, "repair_evaluation_setup")
    features, aliases = {}, {}
    for segment in reference.segments:
        features.setdefault(segment.feature_id, []).append(segment)
        for alias in (segment.feature_id, segment.target_id):
            aliases.setdefault(alias, set()).add(segment.feature_id)
    # The current continuous board reader can name the whole physical boundary.
    # These are explicit ontology aliases, not an image-nearest identity guess.
    for alias, prefix in (("board/continuous/top", "board_top."),
                          ("board/continuous/bottom", "board_contact.")):
        aliases[alias] = {identity for identity in features if identity.startswith(prefix)}
    readings = (reader_record.get("guided") or {}).get("readings", [])
    if any(reader_record.get(key) is True for key in
           ("hint_pixels_used_as_output", "observation_coordinates_from_projection", "projected_uv_used_as_observations")):
        fail("invalid_source_evidence", None, {"meaning": "Reader declares projected/hint pixels as measurements"}, "resolve_source_support")
        readings = []
    grouped = {identity: [] for identity in features}
    for row_index, row in enumerate(readings):
        uv, indices, runs = _read_runs(row)
        identity = row.get("physical_id")
        native_board_label = {"board.top": "board/continuous/top",
                              "board.bottom": "board/continuous/bottom"}.get(identity)
        if (native_board_label and row.get("native_path_id") == native_board_label
                and (row.get("native_geometry") or {}).get("path_id") == native_board_label):
            # The native label and its path name denote the same continuous
            # physical boundary. Preserve the source record; resolve only this
            # explicit, already bound alias for comparison with partial labels.
            report.setdefault("identity_alias_resolutions", []).append(dict(
                reading_index=row_index, source_physical_id=identity,
                native_path_id=native_board_label, source_UV_modified=False))
            identity = native_board_label
        # expected_id and guide coordinates are navigation hints, not observed ID.
        matched = aliases.get(identity, set()) if identity else set()
        if not matched:
            reason = "outside_reviewed_scope" if row.get("kind") == "ad_boundary" else "identity_not_in_partial_reference"
            report["unscored"].append({"reading_index": row_index, "physical_id": identity,
                                       "actual_point_count": len(uv), "reason": reason})
            if not identity and len(uv) and row.get("kind") != "ad_boundary":
                fail("unresolved_identity", None, {"reading_index": row_index,
                    "expected_id_is_only_a_hint": row.get("expected_id"), "actual_point_count": len(uv)}, "reassociate_entity")
            continue
        claimed = row.get("source_feature_id")
        if claimed in aliases and not (aliases[claimed] & matched):
            fail("wrong_identity", identity, {"reading_index": row_index,
                "physical_id": identity, "recorded_source_feature_id": claimed,
                "basis": "conflicting explicit physical/source identities; not nearest distance"}, "reassociate_entity")
            continue
        expected_kinds = {"board_top" if f.startswith("board_top.") else "ice_contact" if f.startswith("board_contact.") else "ice_paint" for f in matched}
        if row.get("kind") and row["kind"] not in expected_kinds:
            fail("wrong_identity", identity, {"reading_index": row_index,
                "kind": row["kind"], "expected_kinds": sorted(expected_kinds),
                "basis": "physical boundary semantics conflict"}, "reassociate_entity")
            continue
        for feature_id in matched:
            expected_measurements = {segment.measurement for segment in features[feature_id]}
            if row.get("native_measurement") and row["native_measurement"] not in expected_measurements:
                fail("measurement_semantics", feature_id, {"reading_index": row_index,
                    "actual": row["native_measurement"], "expected": sorted(expected_measurements)}, "reassociate_entity")
                continue
            grouped[feature_id].append((row_index, uv, indices, runs, matched))

    for feature_id, segments in features.items():
        curves = [_curve(segment, protocol.sample_spacing_px) for segment in segments]
        samples = [_sample_curve(curve, protocol.sample_spacing_px) for curve in curves]
        supported = [np.zeros(len(uv), dtype=bool) for uv, _ in samples]
        forward, reverse_runs, measured_count, unscored_count = [], [], 0, 0
        for row_index, uv, indices, runs, matched in grouped[feature_id]:
            if not len(uv):
                continue
            projections = [_project_reference(uv, segment, curve, protocol.endpoint_margin_px)
                           for segment, curve in zip(segments, curves)]
            distances = np.stack([p[0] for p in projections], axis=1)
            eligibility = np.stack([p[2] for p in projections], axis=1)
            nearest = np.argmin(np.where(eligibility, distances, np.inf), axis=1)
            best = distances[np.arange(len(uv)), nearest]
            eligible = np.array([projections[c][2][i] for i, c in enumerate(nearest)])
            # A whole-boundary identity can legitimately cover several named
            # reference segments. Assign its scoring support within that already
            # known entity; do not score a top-right sample against the far top
            # boundary too. This does not infer or rewrite physical identity.
            for other_id in matched - {feature_id}:
                for other in features[other_id]:
                    other_distance, _, other_eligible = _project_reference(
                        uv, other, _curve(other, protocol.sample_spacing_px), protocol.endpoint_margin_px)
                    eligible &= ~(other_eligible & (other_distance + 1e-6 < best))
            measured_count += len(uv)
            unscored_count += int(np.sum(~eligible))
            forward.extend(best[eligible].tolist())
            bad = eligible & (best > protocol.localization_tolerance_px)
            if np.any(bad):
                fail("localization", feature_id, {"reading_index": row_index,
                    "query_indices": [indices[i] for i in np.flatnonzero(bad)],
                    "actual_source_uv": uv[bad].tolist(), "distance_px": best[bad].tolist(),
                    "tolerance_px": protocol.localization_tolerance_px,
                    "identity_not_reassigned_by_distance": True}, "reread_material")
            for run in runs:
                reverse_runs.append(uv[run])
                for position in run:
                    segment = nearest[position]
                    distance, station, _ = projections[segment]
                    if eligible[position] and distance[position] <= protocol.coverage_tolerance_px:
                        supported[segment] |= np.abs(samples[segment][1] - station[position]) <= protocol.coverage_tolerance_px
                for first, last in zip(run[:-1], run[1:]):
                    segment = nearest[first]
                    if (segment != nearest[last] or not eligible[first] or not eligible[last]
                            or max(best[first], best[last]) > protocol.coverage_tolerance_px):
                        continue
                    stations = projections[segment][1]
                    lo, hi = sorted((stations[first], stations[last]))
                    supported[segment] |= ((samples[segment][1] >= lo) & (samples[segment][1] <= hi))
        reverse = []
        segment_results = []
        for segment, (sample_uv, _), mask in zip(segments, samples, supported):
            if reverse_runs:
                directed = np.min(np.stack([_project(sample_uv, run, protocol.endpoint_margin_px)[0]
                                            for run in reverse_runs], axis=1), axis=1)
                reverse.extend(directed.tolist())
            fraction = float(np.mean(mask))
            segment_results.append({"segment_index": segment.segment_index,
                "support_fraction": fraction, "sample_count": len(mask),
                "missing_sample_count": int(np.sum(~mask))})
        sample_count = sum(len(mask) for mask in supported)
        support_fraction = sum(int(np.sum(mask)) for mask in supported) / sample_count
        if support_fraction < protocol.minimum_support_fraction or any(
                item["support_fraction"] < protocol.minimum_support_fraction for item in segment_results):
            fail("missing_support", feature_id, {"support_fraction": support_fraction,
                "required_fraction": protocol.minimum_support_fraction,
                "segments": segment_results, "actual_point_count": measured_count,
                "scope": "only accepted visible segments; unmarked regions are unscored"},
                "discover_material" if measured_count == 0 else "reread_material")
        report["features"].append({"feature_id": feature_id, "actual_point_count": measured_count,
            "prediction_to_reviewed_outline": _statistics(forward),
            "reviewed_outline_to_measured_runs": _statistics(reverse),
            "support_fraction": support_fraction, "segments": segment_results,
            "points_outside_reviewed_support": unscored_count})
    if not features:
        report["evaluation_role"] = "unscored"
    report["summary"] = {"reviewed_feature_count": len(features),
        "actual_reading_count": len(readings), "failure_count": len(report["failures"]),
        "unscored_reading_count": len(report["unscored"])}
    return report
