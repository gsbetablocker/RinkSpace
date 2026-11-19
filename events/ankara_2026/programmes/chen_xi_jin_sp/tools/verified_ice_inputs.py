"""Replace earlier ice line rows with original-width measurements and retain history."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import sys

import cv2
import numpy as np

from common import HEIGHT, JOINT, OUTPUT, frozen_scene, observation_packet, save, scope, source_pts, write
from ice_evidence import CONTOUR, ICE_EVIDENCE_FIELDS, MIDPOINT, assert_ice_evidence, ice_roles
from measure import DT, FIRST, load_camera, motion_clock
from precision_observations import exclude
from stability_observations import ROW_FIELDS, read_arrays


def corroborate_widths(rows):
    """Compare nearby raw widths before admitting a complete paired stroke."""
    get = np.flatnonzero(rows["eligible"])
    if not len(get):
        return 0
    ordered = get[np.lexsort((rows["station"][get], rows["path"][get], rows["pts"][get]))]
    separate = ((rows["pts"][ordered][1:] != rows["pts"][ordered][:-1]) |
                (rows["path"][ordered][1:] != rows["path"][ordered][:-1]))
    width = np.linalg.norm(rows["source_bracket"][:, 1] - rows["source_bracket"][:, 0], axis=1)
    rejected = np.zeros(len(width), bool)
    for run in np.split(ordered, np.flatnonzero(separate) + 1):
        if len(run) < 4:
            rejected[run] = True
            continue
        uv = rows["uv"][run]
        distance = np.linalg.norm(uv[:, None] - uv[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        nearby = np.argsort(distance, axis=1)[:, :6]
        supported_width = np.median(width[run[nearby]], axis=1)
        ratio = width[run] / supported_width
        lo, hi = (.73, 1.39) if rows["family"][run[0]] in ("B1", "B2", "C") else (.64, 1.54)
        rejected[run[(ratio < lo) | (ratio > hi)]] = True
    exclude(rows, rejected, "paired_stroke_width_conflicts_with_neighbouring_visible_flanks")
    return int(rejected.sum())


def corroborate_lanes(rows, labels):
    """A weak local intensity pair does not identify a lane painted on ice."""
    lane = rows["eligible"] & np.char.startswith(rows["family"], "lane")
    get = np.flatnonzero(lane)
    if not len(get):
        return 0
    ordered = get[np.lexsort((rows["station"][get], rows["path"][get], rows["pts"][get]))]
    separate = ((rows["pts"][ordered][1:] != rows["pts"][ordered][:-1]) |
                (rows["path"][ordered][1:] != rows["path"][ordered][:-1]) |
                (np.linalg.norm(np.diff(rows["uv"][ordered], axis=0), axis=1) > 48.))
    supported = np.zeros(len(lane), bool)
    width = np.linalg.norm(rows["source_bracket"][:, 1] - rows["source_bracket"][:, 0], axis=1)
    strength = rows["source_side_contrast"].min(axis=1)
    for run in np.split(ordered, np.flatnonzero(separate) + 1):
        short = labels[rows["path"][run[0]]].endswith(":F_transverse")
        if len(run) < (4 if short else 6) or np.median(strength[run]) < 6.6:
            continue
        station = rows["station"][run]
        design = np.c_[np.ones(len(run)), station - np.mean(station)]
        width_fit = np.exp(design @ np.linalg.lstsq(design, np.log(width[run]), rcond=None)[0])
        good = (width[run] >= .61 * width_fit) & (width[run] <= 1.6 * width_fit)
        if np.count_nonzero(good) >= (4 if short else 6):
            supported[run[good]] = True
    rejected = lane & ~supported
    exclude(rows, rejected, "source_does_not_show_continuous_two_flank_lane_segment")
    return int(rejected.sum())


def signed_bend(points):
    a, b, c = [np.median(x, axis=0) for x in np.array_split(points, 3)]
    chord = c - a
    span = np.linalg.norm(chord)
    if span <= 0.:
        return 0., 0.
    side = (chord[0] * (b[1] - a[1]) - chord[1] * (b[0] - a[0])) / span
    return float(side), float(span)


def match_ice_identity(rows, old, shot, native, labelled, raw_outlines, seed):
    """Use guide geometry only to detect overlap or distinguish source curvature."""
    is_lane = np.char.startswith(rows["family"], "lane")
    is_arc = np.isin(rows["path"], [k for k in raw_outlines if labelled[k].endswith(":centreline")])
    to_check = rows["eligible"] & (is_lane | is_arc)
    if not to_check.any():
        return 0, 0
    pts, controls, rig = load_camera(shot, seed)
    times = motion_clock(shot, seed, pts)
    names = list(raw_outlines)
    limits = np.r_[0, np.cumsum([len(raw_outlines[k][0]) for k in names])]
    xyz = np.vstack([raw_outlines[k][0] for k in names])
    ids = np.flatnonzero(to_check)
    chronological = ids[np.argsort(rows["idx"][ids], kind="stable")]
    overlap = np.zeros(len(to_check), bool)
    ambiguous_arc = np.zeros(len(to_check), bool)
    observed_ring = old["eligible"] & np.char.startswith(old["material"], "paired_annulus_")
    ring_sources = defaultdict(list)
    for j in np.flatnonzero(observed_ring):
        ring_sources[(old["idx"][j], old["path"][j])].append(old["uv"][j])
    for current in np.split(chronological, np.flatnonzero(np.diff(rows["idx"][chronological])) + 1):
        if not len(current):
            continue
        index = rows["idx"][current[0]]
        projection = native.core.project_motion(rig, controls, FIRST, DT, HEIGHT, times[index], xyz)
        uv = projection["uv"]
        uv[projection["status"] != 0] = np.nan
        by_path = {k: uv[lo:hi] for k, lo, hi in zip(names, limits[:-1], limits[1:], strict=True)}
        source = rows["uv"][current]
        for path, polygon in by_path.items():
            if not labelled[path].endswith(":outer"):
                continue
            finite = polygon[np.isfinite(polygon).all(1)]
            if len(finite) < 18:
                continue
            ring = cv2.convexHull(finite.astype(np.float32))
            raw = ring_sources.get((index, path), ())
            offset = [abs(cv2.pointPolygonTest(ring, tuple(map(float, p)), True)) for p in raw]
            padding = 13. + (min(28., float(np.median(offset))) if offset else 0.)
            lo = finite.min(0) - padding
            hi = finite.max(0) + padding
            near = current[((source >= lo) & (source <= hi)).all(1)]
            for j in near:
                if cv2.pointPolygonTest(ring, tuple(map(float, rows["uv"][j])), True) >= -padding:
                    overlap[j] = True
        examine = current[is_arc[current] & ~overlap[current]]
        for path in np.unique(rows["path"][examine]):
            q = examine[rows["path"][examine] == path]
            q = q[np.argsort(rows["station"][q], kind="stable")]
            far_apart = np.linalg.norm(np.diff(rows["uv"][q], axis=0), axis=1) > 50.
            for pieces in np.split(q, np.flatnonzero(far_apart) + 1):
                if len(pieces) < 7:
                    ambiguous_arc[pieces] = True
                    continue
                station = rows["station"][pieces]
                track = by_path[path]
                guide = np.column_stack([np.interp(station, raw_outlines[path][1], track[:, j]) for j in range(2)])
                measured, span = signed_bend(rows["uv"][pieces])
                expected, _ = signed_bend(guide)
                curvature = (np.isfinite(expected) and span >= 105. and abs(measured) >= 2.0
                             and measured * expected > 0. and .32 * abs(expected) <= abs(measured) <= 3.0 * abs(expected))
                if not curvature:
                    ambiguous_arc[pieces] = True
    exclude(rows, overlap, "lane_or_arc_read_lies_on_target_ring_material")
    exclude(rows, ambiguous_arc, "original_red_marks_do_not_resolve_identity_of_this_curve")
    return int(overlap.sum()), int(ambiguous_arc.sum())


def join_shot(record, shot, native_labels, roles, previous, readings, out, native, outlines, guide):
    sid = shot["id"]
    original = observation_packet(previous, sid)
    source = observation_packet(readings, sid)
    old = read_arrays(original)
    clocks = source_pts(shot)
    np.testing.assert_array_equal(clocks[old["idx"]], old["pts"])
    for field, expected in (("source_sha256", record["source"]["sha256"]),
                           ("geometry_sha256", record["geometry_binding"]["value_sha256"])):
        if old[field].item() != expected:
            raise ValueError("Mismatched historical observation packet " + sid)

    midlines = np.isin(old["path"], [p for p, r in roles.items() if r == MIDPOINT])
    old_ice_contours = np.isin(old["path"], [p for p, r in roles.items() if r == CONTOUR])
    labels = {k: native_labels[k] for k in np.unique(old["path"][old_ice_contours])}
    mapped = np.array([labels.get(k, "") for k in old["path"]])
    qualified_contour = ((old["material"] == "paired_annulus_outer") & np.char.endswith(mapped, ":outer")
                         | (old["material"] == "paired_annulus_middle") & np.char.endswith(mapped, ":middle"))
    previous_active = midlines & old["eligible"] & (old["fold"] == 0)
    # Retain measured arrays and exact historical roles, including disabled rows.
    exclude(old, midlines, "replaced_by_original_visible_two_flank_evidence")
    exclude(old, old_ice_contours & ~qualified_contour, "undeclared_source_ring_boundary")
    count = len(old["pts"])
    old["source_bracket"] = np.full((count, 2, 2), np.nan)
    old["source_side_contrast"] = np.full((count, 2), np.nan)
    old["ice_measurement_role"] = np.where(qualified_contour, CONTOUR, "")
    data = [old]
    fresh_count = 0
    unclear_lanes = 0
    width_conflicts = 0
    ring_conflicts = arc_identity = 0
    by_family = {}
    if source.is_file():
        fresh = read_arrays(source)
        for field in ("source_sha256", "geometry_sha256"):
            np.testing.assert_array_equal(fresh[field], old[field])
        np.testing.assert_array_equal(clocks[fresh["idx"]], fresh["pts"])
        width_conflicts = corroborate_widths(fresh)
        ring_conflicts, arc_identity = match_ice_identity(fresh, old, shot, native, native_labels, outlines, guide)
        unclear_lanes = corroborate_lanes(fresh, native_labels)
        assert_ice_evidence(fresh, fresh["eligible"], roles)
        training = fresh["eligible"] & (fresh["fold"] == 0)
        fresh_count = int(training.sum())
        by_family = dict(Counter(fresh["family"][training]))
        data.append(fresh)

    frame_hashes = {}
    for item in data:
        for p, digest in zip(item["source_frame_pts"], item["source_frame_BGR_sha256"], strict=True):
            if frame_hashes.setdefault(int(p), digest) != digest:
                raise ValueError(f"Decoded source mismatch in {sid} at {p}")
    packet = {k: np.concatenate([item[k] for item in data]) for k in (*ROW_FIELDS, *ICE_EVIDENCE_FIELDS)}
    assert_ice_evidence(packet, packet["eligible"], roles)
    sort = np.argsort(packet["pts"], kind="stable")
    frame_pts = np.array(sorted(frame_hashes), dtype=np.int64)
    save(out / f"{sid}.npz", **{k: v[sort] for k, v in packet.items()},
         source_frame_pts=frame_pts,
         source_frame_BGR_sha256=np.array([frame_hashes[int(v)] for v in frame_pts]),
         source_sha256=old["source_sha256"], geometry_sha256=old["geometry_sha256"])
    summary = dict(shot=sid, prior_active_midpoints_removed=int(previous_active.sum()),
                   new_active_width_midpoints=fresh_count, new_width_midpoints_by_family=by_family,
                   ambiguous_new_lane_rows=unclear_lanes,
                   excluded_width_conflicts=width_conflicts,
                   excluded_target_ring_overlap=ring_conflicts, excluded_unresolved_arc=arc_identity,
                   prior_midpoint_rows_preserved=int(midlines.sum()),
                   material_contour_rows_retained=int((old_ice_contours & old["eligible"]).sum()),
                   previously_read=str(original), replacement=str(source) if source.is_file() else None)
    print(sid, summary["prior_active_midpoints_removed"], "former points", fresh_count, "two-sided points",
          summary["material_contour_rows_retained"], "material contour points", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--readings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guide", required=True, help="Camera used only to delimit overlapping ice structures")
    args = parser.parse_args()
    record = scope()
    native = frozen_scene(record, verify=True)
    roles = ice_roles(native)
    labels = {k: v["label"] for k, v in native.paths.items()}
    sys.path.insert(0, str(JOINT))
    from joint_inputs import native_paths

    selected = [key for key, role in roles.items()
                if ((role == CONTOUR and labels[key].endswith(":outer"))
                    or (role == MIDPOINT and labels[key].endswith(":centreline")))]
    outlines = native_paths(native, selected, count=143)
    previous, readings, out = (OUTPUT / name for name in (args.previous, args.readings, args.out))
    out.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(join_shot, record, shot, labels, roles, previous, readings, out,
                            native, outlines, OUTPUT / args.guide)
                for shot in record["shots"]]
        summary = [f.result() for f in jobs]
    write(out / "inputs.json", dict(previous=str(previous), original_width_readings=str(readings),
                                   geometry_sha256=record["geometry_binding"]["value_sha256"], shots=summary))


if __name__ == "__main__":
    main()
