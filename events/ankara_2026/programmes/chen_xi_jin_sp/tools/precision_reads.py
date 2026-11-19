"""Recover visible material edges in the reviewed Chen source intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256

import cv2
import numpy as np

from common import HEIGHT, JOINT, OUTPUT, TB, WIDTH, frozen_scene, save, scope, source_frames, visibility_packet, write
from measure import DT, FIRST, load_camera, localize, motion_clock
from physical_reads import boundary_candidates, coherent, samples, source_profiles, stripe_candidates
from review import restore_source_graphics


# Guide centres are readings of the 1080p50 source, in raw image coordinates.
RED_SECONDS = (107.78, 107.88, 108.08, 108.22, 108.26, 108.44)
RED_X_AT_800 = (1177., 1295., 1528., 1726., 1784., 2045.)
BLUE = {
    "s04": ("d78_material_165/path0", (39.16, 39.50), (26., 1090.), (20., 44.)),
    "s07": ("d78_material_143/path0", (61.30, 61.72), (50., 1050.), (32., 26.)),
    "s12": ("d78_material_142/path0", (107.94, 108.42), (75., 1125.), (23., 16.)),
}
PANELS = {
    "s16": ("Ad4", (166.52, 167.08), ("top_edge", "bottom_edge", "left_vertical")),
    "s20": ("Ad2", (197.14, 198.12), ("bottom_edge", "right_vertical")),
}
RINGS = {
    "s06": ("Ad24", (55.12, 55.78)),
    "s14": ("Ad11", (144.54, 145.24)),
}


def stripe(raw, mask, points, axis, tone, width):
    radius = 148
    normal = np.repeat(np.eye(2)[axis][None, :], len(points), axis=0)
    tangent = normal[:, ::-1].copy()
    value, hidden, _ = source_profiles(raw, mask, points, normal, tangent, radius, tone)
    options = [stripe_candidates(v, h, radius, expected_min=max(5, int(.58 * w)),
                    max_width=int(1.48 * w + 7), min_contrast=3.2)
               for v, h, w in zip(value, hidden, width, strict=True)]
    accepted = coherent(options, points, stripe=True)
    if not accepted:
        return None
    idx = np.array([i for i, _ in accepted])
    data = np.array([row for _, row in accepted])
    uv = points[idx] + data[:, 0, None] * normal[idx]
    edges = points[idx, None, :] + data[:, 2:4, None] * normal[idx, None, :]
    return uv, normal[idx], data[:, 4], edges


def red_centre(raw, mask, seconds):
    y = np.arange(441., 1047., 29.)
    x = np.interp(seconds, RED_SECONDS, RED_X_AT_800) - .43 * (y - 800.)
    return stripe(raw, mask, np.c_[x, y], 0, "red", .158 * (y - 440.) + 88.)


def blue_centre(raw, mask, uv, station, config):
    _, _, limits, widths = config
    q = samples(uv, station, step=29.)
    if q is None:
        return None
    coords = q[0]
    use = (coords[:, 0] >= limits[0]) & (coords[:, 0] <= limits[1])
    coords = coords[use]
    if len(coords) < 4:
        return None
    # The rough width only selects two supported raw flanks. It is not a
    # substitute for either flank or a fitted pixel position.
    width = np.interp(coords[:, 0], limits, widths)
    return stripe(raw, mask, coords, 1, "blue", width)


def ring_edge(raw, mask, uv, station):
    radius = 54
    sampled = samples(uv, station, step=18.)
    if sampled is None:
        return None
    points, normal, tangent, _ = sampled
    outside = points - np.mean(uv[np.isfinite(uv).all(1)], axis=0)
    normal *= np.where(np.sum(outside * normal, axis=1) <= 0., 1., -1.)[:, None]
    profiles, hidden, _ = source_profiles(raw, mask, points, normal, tangent, radius, "dark")
    options = [stripe_candidates(v, h, radius, expected_min=2, max_width=12, min_contrast=6.)
               for v, h in zip(profiles, hidden, strict=True)]
    # Use the first flank into the black perimeter. The source ring is dark
    # ink on light board and the model path identifies its outside edge.
    options = [[(left, score, left, right, contrast) for _, score, left, right, contrast in a]
               for a in options]
    accepted = coherent(options, points, stripe=True)
    if not accepted:
        return None
    i = np.array([j for j, _ in accepted])
    a = np.array([v for _, v in accepted])
    edge = points[i, None, :] + a[:, 2:4, None] * normal[i, None, :]
    return edge[:, 0], normal[i], a[:, 4], edge


def panel_edge(raw, mask, uv, station, role):
    axis = int(role.endswith("_edge"))
    radius = 91
    direction = 1 if role in ("top_edge", "left_vertical") else -1
    projected = samples(uv, station, axis=axis, step=25.)
    if projected is None:
        return None
    coords, normal, tangent, st = projected
    inside = (st > .10) & (st < .90)
    coords, normal, tangent = coords[inside], normal[inside], tangent[inside]
    if len(coords) < 4:
        return None
    profiles, hidden, colours = source_profiles(raw, mask, coords, normal, tangent, radius, "purple")
    accepted = coherent([boundary_candidates(v, h, c, radius, sign=direction, tone="purple")
             for v, h, c in zip(profiles, hidden, colours, strict=True)], coords, stripe=False)
    if not accepted:
        return None
    i = np.array([i for i, _ in accepted])
    data = np.array([row for _, row in accepted])
    points = coords[i] + data[:, 0, None] * normal[i]
    return points, normal[i], data[:, 4], np.full((len(i), 2, 2), np.nan)


def process(record, native, shot, guide, folder, family, native_paths):
    sid = shot["id"]
    pts, controls, rig = load_camera(shot, guide)
    clock = motion_clock(shot, guide, pts)
    ticks_per_second = TB.denominator
    descriptors = {}
    if sid == "s12":
        path = next(k for k in native.paths if family(k, native) == "C")
        descriptors[path] = (RED_SECONDS[0], RED_SECONDS[-1], red_centre,
                             "source_reviewed_red_two_flanks")
    if sid in BLUE:
        config = BLUE[sid]
        path = config[0]
        if path not in native.paths:
            raise ValueError("Reviewed ice path missing from the frozen scene")
        descriptors[path] = (*config[1], None, "source_reviewed_blue_two_flanks")
    if sid in RINGS:
        ad, seconds = RINGS[sid]
        path = f"print_instance/{ad}::Ad11.outer_ring.outer"
        descriptors[path] = (*seconds, ring_edge, "source_reviewed_dark_ring_outer_flank")
    if sid in PANELS:
        ad, seconds, edges = PANELS[sid]
        for role in edges:
            path = next(k for k in native.paths if k.startswith(f"print_instance/{ad}::")
                        and k.split("::", 1)[1].split("/")[0].endswith("." + role))
            descriptors[path] = (*seconds, role, "source_reviewed_purple_boundary")
    if not descriptors:
        return
    frame_rows = defaultdict(list)
    for k, (lo, hi, _, _) in descriptors.items():
        for p in pts[(pts >= round(lo * ticks_per_second)) & (pts <= round(hi * ticks_per_second))]:
            frame_rows[int(p)].append(k)
    samples_world = native_paths(native, descriptors, count=301)
    masks = visibility_packet(OUTPUT / "visibility_final", sid)
    with np.load(masks, allow_pickle=False) as m:
        np.testing.assert_array_equal(m["pts"], pts)
        assert m["source_sha256"].item() == record["source"]["sha256"]
        packed, valid = m["packed"], m["valid"]
    fields = defaultdict(list)
    hashes = {}
    for p, frame in source_frames(record["source"]["registered_path"], sorted(frame_rows)):
        idx = int(np.searchsorted(pts, p))
        if not valid[idx]:
            continue
        raw = frame.to_ndarray(format="bgr24")
        hashes[p] = sha256(raw.tobytes()).hexdigest()
        matte = np.unpackbits(packed[idx], axis=1, count=WIDTH)
        matte = restore_source_graphics(raw, p, cv2.dilate(matte, np.ones((15, 15), np.uint8)))
        for k in frame_rows[p]:
            _, _, reader, label = descriptors[k]
            world, station = samples_world[k]
            q = native.core.project_motion(rig, controls, FIRST, DT, HEIGHT, clock[idx], world)
            q["uv"][q["status"] != 0] = np.nan
            if reader is red_centre:
                result = red_centre(raw, matte, float(p * TB))
            elif reader is None:
                result = blue_centre(raw, matte, q["uv"], station, BLUE[sid])
            elif reader is ring_edge:
                result = ring_edge(raw, matte, q["uv"], station)
            else:
                result = panel_edge(raw, matte, q["uv"], station, reader)
            if result is None:
                continue
            uv, normal, contrast, edges = result
            on, error = localize(q["uv"], station, uv)
            keep = np.isfinite(on) & np.isfinite(error)
            count = int(keep.sum())
            if count < 4:
                continue
            row = dict(pts=np.repeat(p, count), idx=np.repeat(idx, count), path=np.repeat(k, count),
                       family=np.repeat(family(k, native), count), material=np.repeat(label, count),
                       uv=uv[keep], station=on[keep], normal=normal[keep], contrast=contrast[keep],
                       guide_error=error[keep], source_bracket=edges[keep])
            for field, values in row.items():
                fields[field].append(values)
    a = {k: np.concatenate(v) for k, v in fields.items()}
    if not a:
        write(folder / f"{sid}.json", dict(rows=0, reason="paired_flanks_not_resolved"))
        return
    has_neighbour = np.zeros(len(a["pts"]), bool)
    for path in np.unique(a["path"]):
        match = a["path"] == path
        frames = np.unique(a["idx"][match])
        gap = np.diff(frames)
        repeated = frames[np.r_[False, gap <= 2] | np.r_[gap <= 2, False]]
        has_neighbour |= match & np.isin(a["idx"], repeated)
    a.update(eligible=has_neighbour, eligible_before_quality_review=np.ones(len(a["pts"]), bool),
             excluded_reason=np.where(has_neighbour, "", "isolated_material_profile"),
             fold=np.zeros(len(a["pts"]), np.int8),
             source_sha256=record["source"]["sha256"],
             geometry_sha256=record["geometry_binding"]["value_sha256"],
             source_frame_pts=np.array(list(hashes), np.int64),
             source_frame_BGR_sha256=np.array(list(hashes.values())))
    save(folder / f"{sid}.npz", **a)
    receipt = {str(k): dict(rows=int(np.sum(m)), frames=len(np.unique(a["pts"][m])),
                           max_guide_distance=float(np.max(a["guide_error"][m])))
               for k in np.unique(a["path"][has_neighbour])
               if (m := has_neighbour & (a["path"] == k)).any()}
    write(folder / f"{sid}.json", receipt)
    print(sid, receipt, flush=True)


def main():
    import sys
    sys.path.insert(0, str(JOINT))
    from joint_inputs import family, native_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    folder = OUTPUT / args.out
    folder.mkdir()
    record = scope()
    native = frozen_scene(record, verify=True)
    cv2.setNumThreads(2)
    for shot in record["shots"]:
        process(record, native, shot, OUTPUT / args.guide, folder, family, native_paths)


if __name__ == "__main__":
    main()
