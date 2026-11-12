"""Follow an already identified lower steel edge in native source images."""

from collections import defaultdict

import cv2
import numpy as np

from inputs import PROGRAMME, SIZE, direct_rows, masked_distance, read
from scripts.track_programme_skates import advance, resample


CUTTING_EDGE = {"runner_lower_edge"}
OTHER_BOUNDARIES = {"outsole", "front_mount", "rear_mount", "runner_upper_edge"}
SOURCE_REVIEW = PROGRAMME / "skate_cutting_edge_review_1112.json"
LEGACY_PARTS = {"boot_sole": "outsole", "front_stanchion": "front_mount", "rear_stanchion": "rear_mount",
                "runner_upper_edge": "runner_upper_edge"}


def lower_edge_seeds(inputs):
    """Use the audited native cuts and preserve the historical ledger verbatim."""
    seeds = {}
    for row in inputs.document.get("full_source_skate_structures", {}).get("rows", []):
        i = inputs.index.get(row["pts"])
        if (i is None or inputs.withheld[i] or row.get("structure_feature") not in CUTTING_EDGE
                or row["role"] != "fit" or row["foot"] not in (0, 1)
                or row.get("optimization_enabled") is False or row.get("source_status") != "observed"
                or row.get("bounded_ambiguous_boundary")):
            continue
        seeds[i, row["foot"]] = np.asarray(row["polyline_uv"], dtype=np.float32)
    for key, rows in inputs.fit.items():
        direct = direct_rows(rows, CUTTING_EDGE)
        if direct:
            seeds[key] = np.asarray(max(direct, key=lambda r: len(r["polyline_uv"]))["polyline_uv"], dtype=np.float32)
    audit = read(SOURCE_REVIEW)
    if (audit["schema"] != "rinkspace.skate-cutting-edge-source-review.v1"
            or audit["source_video_sha256"] != inputs.media_sha or tuple(audit["time_base"]) != tuple(inputs.tb)
            or tuple(audit["image_size"]) != SIZE or audit["role"] != "fit" or audit["feature"] not in CUTTING_EDGE):
        raise ValueError("Cutting-edge review does not match the native source")
    selected = {}
    for pts, f in audit["accepted_existing"]:
        key = inputs.index[pts], f
        selected[key] = seeds[key]
    for row in audit["revised_pixel_traces"]:
        key = inputs.index[row["pts"]], row["foot"]
        uv = np.asarray(row["polyline_uv"], dtype=np.float32)
        if (key not in seeds or key in selected or uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2
                or not np.isfinite(uv).all() or (uv < 0).any() or (uv >= SIZE).any()):
            raise ValueError("Invalid revised source rail")
        selected[key] = uv
    return selected


def reference_parts(inputs, seeds):
    """Track the same shoe's named upper web, mounts and sole as context."""
    references = defaultdict(dict)
    for row in inputs.document.get("full_source_skate_structures", {}).get("rows", []):
        key = (inputs.index.get(row["pts"]), row["foot"])
        feature = LEGACY_PARTS.get(row["structure_feature"])
        if (key in seeds and feature and row["role"] == "fit" and row.get("optimization_enabled") is not False
                and row.get("source_status") == "observed" and not row.get("bounded_ambiguous_boundary")):
            references[key][feature] = row["polyline_uv"]
    for key in seeds:
        for row in direct_rows(inputs.fit.get(key, []), OTHER_BOUNDARIES):
            references[key][row["semantic"]] = row["polyline_uv"]
    return references


def track_step(frames, bounds, previous, current, parts):
    # Crop intersection keeps one original pixel coordinate system for the
    # two-way motion calculation, even though the source window is moving.
    lo = np.maximum(bounds[previous, :2], bounds[current, :2])
    hi = np.minimum(bounds[previous, 2:], bounds[current, 2:])
    if (hi - lo).min() < 32 or ((parts[0]["uv"] <= lo + 2) | (parts[0]["uv"] >= hi - 2)).any():
        return []
    images = []
    for k in (previous, current):
        x, y = lo - bounds[k, :2]
        r, b = hi - bounds[k, :2]
        images.append(frames[k][y:b, x:r])
    inside = [dict(part, uv=part["uv"] - lo) for part in parts
              if ((part["uv"] > lo + 2) & (part["uv"] < hi - 2)).all()]
    return [dict(part, uv=part["uv"] + lo) for part in advance(inside, *images)]


def attached_to_same_boot(poly, other, old, new, flow, negatives):
    """Reject transfer to the boot shell, another foot, or a named non-cutting edge."""
    if not np.isfinite(poly).all():
        return False
    scale = np.linalg.norm(np.ptp(poly, axis=0))
    gap = -np.array([cv2.pointPolygonTest(poly, tuple(point), True) for point in new])
    if np.median(gap) < max(1., .022 * scale) or np.median(gap) > max(10., .33 * scale):
        return False
    if np.isfinite(other).all() and np.mean([cv2.pointPolygonTest(other, tuple(point), False) >= 0 for point in new]) > .25:
        return False
    if any(np.mean(masked_distance(new, line) < 2.2) > .36 for line in negatives):
        return False
    if np.isfinite(flow).all():
        residual = masked_distance(new, old + flow)
        if np.median(residual) > max(4., .085 * scale):
            return False
    return True


def trace_lower_edges(inputs, start, packet, frames, seeds, references):
    """Advance exact cutting-edge seeds and keep only compatible optical traces."""
    bounds = packet["chip_bounds"]
    boot = packet["boot"]
    candidates = defaultdict(list)
    for (i, foot), edge in seeds.items():
        seed = i - start
        if not 0 <= seed < len(frames):
            continue
        if np.linalg.norm(np.diff(edge, axis=0), axis=-1).sum() < 8.:
            continue
        uv = resample(edge).astype(np.float32)
        initial = [dict(uv=uv, seed_points=len(uv), feature="runner_lower_edge")]
        for feature, source in references.get((i, foot), {}).items():
            curve = resample(np.asarray(source, dtype=np.float32)).astype(np.float32)
            if (np.linalg.norm(np.diff(curve, axis=0), axis=-1).sum() >= 8.
                    and np.mean(masked_distance(uv, curve) < 2.2) <= .25):
                initial.append(dict(uv=curve, seed_points=len(curve), feature=feature))
        for direction in (-1, 1):
            parts = initial
            previous = seed
            confidence = 1.
            # Same short-horizon source flow as the programme evidence path.
            # A loss of material or visibility ends the trace immediately.
            for offset in range(1, 22):
                current = seed + direction * offset
                # Held-out frames may be read by optical inference. Their
                # labels and protected anchors never enter the fit ledger.
                if not 0 <= current < len(frames):
                    break
                if not np.isfinite(boot[current, foot]).all():
                    break
                old = parts[0]["uv"]
                parts = track_step(frames, bounds, previous, current, parts)
                if not parts or parts[0]["feature"] != "runner_lower_edge":
                    break
                part = parts[0]
                delta = (packet["boot_flow_from_previous"][current, foot] if direction == 1
                         else -packet["boot_flow_from_previous"][previous, foot])
                other = boot[current, 1-foot]
                forbidden = [r["polyline_uv"] for r in inputs.fit.get((start+current, foot), [])
                             if r["semantic"] in OTHER_BOUNDARIES]
                forbidden.extend(other_part["uv"] for other_part in parts[1:])
                if not attached_to_same_boot(boot[current, foot], other, old, part["uv"], delta, forbidden):
                    break
                confidence = min(confidence, part["confidence"])
                if offset > 12 and confidence < .80:
                    break
                candidates[current, foot].append((confidence - .012 * offset, i, part["uv"]))
                previous = current
    output = {}
    for key, choices in candidates.items():
        choices.sort(key=lambda row: row[0], reverse=True)
        confidence, anchor, curve = choices[0]
        if any(max(np.median(masked_distance(curve, test)), np.median(masked_distance(test, curve))) > 4.
               for _, _, test in choices[1:] if len(test) > 3):
            continue
        output[key] = (confidence, anchor, curve)
    return output
