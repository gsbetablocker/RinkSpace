"""Read the original contours of recognizable print during fast pans and zooms."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from common import (HEIGHT, OUTPUT, WIDTH, frozen_scene, observation_packet, read, save,
                    scope, sha, source_frames, source_pts, visibility_packet, write)
from measure import DT, FIRST, fold_map, load_camera, localize, motion_clock
from review import restore_source_graphics
from stability_tracks import ji

WINDOW = np.arange(-22., 23.)
SHORT_SIDE = {"left_vertical", "right_vertical"}
PANEL_SIDE = SHORT_SIDE | {"top_edge", "bottom_edge"}
TONE_LEVEL = dict(red=7., blue=8., shield_blue=7., brown=9., panel=15.,
                  app_blue=14., purple_panel=13., dark=8., ring=6., white=10., green=7.)


def raw_edges(P, signal, mask, q, normal, shift, mode):
    axis = np.c_[-normal[:, 1], normal[:, 0]]
    centre = q + shift
    xy = (centre[:, None, None, :] + normal[:, None, None, :] * WINDOW[None, None, :, None]
          + axis[:, None, None, :] * np.array([-2.5, 0., 2.5])[None, :, None, None])
    intensity = gaussian_filter1d(np.median(P.sample(signal, xy), axis=1), .94, axis=1)
    valid = P.valid(mask, xy).all(axis=1)
    level = TONE_LEVEL[mode]
    hits = []
    for j, profile in enumerate(intensity):
        g = -np.gradient(profile)
        peaks = find_peaks(g, height=level * .10, prominence=level * .055)[0]
        widths = peak_widths(g, peaks, rel_height=.5)[0]
        scores = []
        for k, width in zip(peaks, widths, strict=True):
            # Horizontal smear can broaden the normal of a vertical edge.
            # Reject merged or poorly localized transitions.
            if abs(WINDOW[k]) > 7 or not 1.30 <= width <= 6.5 + 7.5 * abs(normal[j, 0]):
                continue
            flank = max(3, int(np.ceil(.71 * width)))
            before = slice(k - flank - 4, k - flank)
            after = slice(k + flank + 1, k + flank + 5)
            if not valid[j, before.start:after.stop].all():
                continue
            a, b = float(np.median(profile[before])), float(np.median(profile[after]))
            if a - b < level or (mode == "ring" and b > -130.):
                continue
            if min(a - profile[k], profile[k] - b) < .18 * (a - b):
                continue
            curvature = g[k - 1] - 2. * g[k] + g[k + 1]
            sub = .5 * np.clip((g[k - 1] - g[k + 1]) / min(-1.e-7, curvature), -1., 1.)
            distance = float(WINDOW[k] + sub)
            scores.append((g[k] / level - .038 * abs(distance), distance, a - b, width))
        if not scores:
            continue
        scores.sort(reverse=True)
        _, offset, contrast, width = scores[0]
        if len(scores) > 1 and abs(offset - scores[1][1]) > 4. and scores[1][0] > .80 * scores[0][0]:
            continue
        hits.append((j, centre[j] + offset * normal[j], contrast, width))
    return hits


def coarse(P, queries, signals, masks, offsets):
    """Score only the signed transitions of the candidate physical instance."""
    rank = np.zeros(len(offsets))
    support = np.zeros(len(offsets))
    horizontal = np.zeros(len(offsets))
    for mode, (q, normal) in queries.items():
        xy = q[None, :, :] + offsets[:, None, :]
        distance = 3.6 + 4.5 * abs(normal[:, 0])
        a = xy - normal[None, :, :] * distance[None, :, None]
        b = xy + normal[None, :, :] * distance[None, :, None]
        good = P.valid(masks, a) & P.valid(masks, b)
        inside, outside = P.sample(signals[mode], a), P.sample(signals[mode], b)
        if mode == "ring":
            good &= outside < -130.
        level = TONE_LEVEL[mode]
        strength = np.where(good, np.clip((inside - outside) / level, -.40, 3.0), 0.)
        rank += np.sum(strength, axis=1)
        support += np.sum(good, axis=1)
        horizontal += np.sum((strength >= .72) & (abs(normal[None, :, 0]) >= .57), axis=1)
    result = rank / np.maximum(support, 1.) ** .75
    result -= .0015 * np.linalg.norm(offsets, axis=1)
    result[(support < 8) | (horizontal < 3)] = -np.inf
    return result


class PrintReader:
    def __init__(self, record, panels=None):
        from current_sources import scan
        from read_current_material import print_configs

        self.native = frozen_scene(record, verify=True)
        self.P = scan.print_read
        self.B = scan.board_read
        self.config = {k: v for k, v in print_configs(self.native, tuple(range(1, 27))).items()
                       if "qr_" not in k.lower() and v[1] != "badge" and v[-1] != "badge"}
        for k in self.native.paths:
            m = self.P.ID.match(k)
            if not m:
                continue
            ad = int(m[1])
            if k.endswith(".shield_print_component001"):
                self.config[k] = (ad, "shield_blue", 1., None, "shield_outline")
            if ad in (13, 26) and "ministry_native" in k:
                self.config[k] = (ad, "red", 1., None, "ministry_letters")
        if panels is not None:
            self.config = {
                k: (ad, "app_blue" if ad in (1, 14) else "purple_panel" if ad in (2, 3, 15, 16) else mode,
                    sign, side, category)
                for k, (ad, mode, sign, side, category) in self.config.items()
                if ad in panels and side in PANEL_SIDE
            }
        self.panel_only = panels is not None
        self.paths = tuple(self.config)
        self.samples = ji.native_paths(self.native, self.paths, count=111)
        stops = np.r_[0, np.cumsum([len(self.samples[k][0]) for k in self.paths])]
        self.slices = dict(zip(self.paths, [slice(a, b) for a, b in zip(stops[:-1], stops[1:], strict=True)], strict=True))
        self.world = np.vstack([self.samples[k][0] for k in self.paths])
        self.board = self.native.scene["board"]
        u = np.linspace(0., self.native.core.board_perimeter(self.board), 701)
        self.course = np.asarray(self.native.core.board_points(self.board, u, np.zeros(len(u))))
        d = np.gradient(self.course, axis=0)
        self.inward = np.c_[d[:, 1], -d[:, 0]]
        self.inward *= np.where(np.sum(-self.course[:, :2] * self.inward, axis=1) < 0., -1., 1.)[:, None]
        from scipy.spatial import cKDTree
        self.station = cKDTree(self.course[:, :2]).query(self.world[:, :2])[1]

    def search(self, uv, status, centre):
        visible = (status == 0) & np.isfinite(uv).all(1)
        facing = np.sum((self.course[:, :2] - centre[:2]) * self.inward, axis=1) < 0.
        visible &= facing[self.station]
        groups = defaultdict(dict)
        board_sides = defaultdict(list)
        for k, (ad, _, _, side, _) in self.config.items():
            sl = self.slices[k]
            if side in PANEL_SIDE and np.count_nonzero(visible[sl]) >= 4:
                board_sides[ad].append(np.mean(uv[sl][visible[sl]], axis=0))
        inside = {ad: np.mean(q, axis=0) for ad, q in board_sides.items() if len(q) > 1}
        for k, (ad, mode, sign, side, category) in self.config.items():
            p, ok = uv[self.slices[k]], visible[self.slices[k]]
            potentially_visible = (ok & (p[:, 0] > -95) & (p[:, 0] < WIDTH + 95) &
                                   (p[:, 1] > 20) & (p[:, 1] < HEIGHT - 18))
            if potentially_visible.sum() < 4 or ok.sum() < .83 * len(p):
                continue
            v = np.where(ok, 0, 1)
            if side:
                if side not in (PANEL_SIDE if self.panel_only else SHORT_SIDE) or ad not in inside:
                    continue
                q, normal = self.B.open_queries(p, v, side, centre=inside[ad], spacing=12.)
            else:
                q, normal = self.P.projected_queries(p, v, 9., sign)
            if len(q) < 3:
                continue
            if len(q) > 18:
                select = np.rint(np.linspace(0, len(q) - 1, 18)).astype(int)
                q, normal = q[select], normal[select]
            groups[ad][k] = (mode, q, normal, category)
        return groups

    def read(self, raw, mask, projected, centre):
        P = self.P
        uv, status = projected["uv"], projected["status"]
        groups = self.search(uv, status, centre)
        if not groups:
            return
        signals = P.signals(np.ascontiguousarray(raw[:, :, ::-1]))
        b, g, r = np.moveaxis(raw.astype(np.float32), 2, 0)
        signals["panel"] = signals["dark"]
        signals["green"] = 1.04 * g - .48 * r - .56 * b - .10 * signals["white"]
        signals["gray_40"] = signals["dark"]
        signals["shield_blue"] = (b - .55 * g - .45 * r) + .26 * signals["dark"]
        signals["app_blue"] = .80 * (b - r) + .34 * signals["dark"]
        signals["purple_panel"] = .65 * b + .50 * r - 1.15 * g + .28 * signals["dark"]
        for ad, paths in groups.items():
            by_tone = defaultdict(list)
            for mode, q, n, category in paths.values():
                if mode not in signals:
                    continue
                by_tone["dark" if mode == "gray_40" else mode].append(np.c_[q, n])
            supported = {mode: np.vstack(q) for mode, q in by_tone.items()}
            queries = {mode: (x[:, :2], x[:, 2:]) for mode, x in supported.items()}
            if sum(len(q) for q, _ in queries.values()) < 8:
                continue
            translations = P.offsets(0., 0., 70, 25, 5.)
            score = coarse(P, queries, signals, mask, translations)
            m = np.argmax(score)
            if score[m] < 2.10:
                continue
            rival = np.linalg.norm(translations - translations[m], axis=1) > 22.
            if np.max(score[rival], initial=-np.inf) > .92 * score[m]:
                continue
            small = P.offsets(*translations[m], 5, 5, 1.)
            value = coarse(P, queries, signals, mask, small)
            best = np.argmax(value)
            if value[best] < 2.15:
                continue
            shift = small[best]
            found = []
            for k, (mode, q, normal, category) in paths.items():
                mode = "dark" if mode == "gray_40" else mode
                if mode not in signals:
                    continue
                edges = raw_edges(P, signals[mode], mask, q, normal, shift, mode)
                if len(edges) < 3:
                    continue
                j = np.asarray([item[0] for item in edges])
                points = np.asarray([item[1] for item in edges])
                contrast = np.asarray([item[2] for item in edges])
                width = np.asarray([item[3] for item in edges])
                poly = uv[self.slices[k]]
                station, distance = localize(poly + shift, self.samples[k][1], points)
                use = np.isfinite(station) & (distance < 8.1)
                if np.count_nonzero(use) < 3:
                    continue
                found.append((k, points[use], station[use], normal[j][use], contrast[use], width[use], q[j][use],
                              category))
            if not found:
                continue
            normals = np.vstack([item[3] for item in found])
            positions = np.vstack([item[1] for item in found])
            x_leverage = abs(normals[:, 0]) >= .58
            independent_x = (np.ptp(positions[x_leverage, 1 if self.panel_only else 0])
                             if np.count_nonzero(x_leverage) >= 4 else 0.)
            if (len(normals) < 7 or independent_x < 14. or
                    np.linalg.svd(normals, compute_uv=False)[-1] < .72):
                continue
            has_anchor = any(item[-1] in {"panel_edge", "signed_logo", "shield_outline"} for item in found)
            if not has_anchor and len(found) < 4:
                continue
            for path, points, station, normal, contrast, width, guide, category in found:
                yield path, f"Ad{ad}", points, station, normal, contrast, width, guide, value[best]


def tasks_from_coverage(shot, previous, coverage, windows, only_windows):
    pts = source_pts(shot)
    want = np.zeros(len(pts), bool)
    for lo, hi in windows:
        want |= (pts >= lo) & (pts <= hi)
    if only_windows:
        return np.flatnonzero(want)
    # Re-read contiguous material gaps including their entrance and exit.
    for start, stop, _ in coverage["no_ads_ice_spans"]:
        a, b = np.searchsorted(pts, np.rint(np.asarray([start, stop]) * 12800).astype(np.int64))
        want[max(0, a-3):b+4] = True
    for start, stop, _ in coverage["fast_intervals"]:
        a, b = np.searchsorted(pts, np.rint(np.asarray([start, stop]) * 12800).astype(np.int64))
        want[max(0, a-2):b+3] = True
    with np.load(observation_packet(previous, shot["id"])) as f:
        printed = f["eligible"] & np.char.startswith(f["family"], "Ad")
        count = np.bincount(f["idx"][printed], minlength=len(pts))
    return np.flatnonzero(want | (count < 4))



def process(reader, record, shot, coverage, windows, previous, guide, visibility, output, only_windows=False):
    sid = shot["id"]
    pts, controls, rig = load_camera(shot, guide)
    times = motion_clock(shot, guide, pts)
    centre = reader.native.core.evaluate_motion_camera(rig, controls, FIRST, DT, times)["centre"]
    folds = fold_map(shot, len(pts))
    indices = tasks_from_coverage(shot, previous, coverage, windows, reader.panel_only or only_windows)
    print(sid, len(indices), "original frames for new print profiles", flush=True)
    hashes = {}
    entries = defaultdict(list)
    with np.load(visibility_packet(visibility, sid)) as f:
        np.testing.assert_array_equal(f["pts"], pts)
        assert f["source_sha256"].item() == record["source"]["sha256"]
        packed, valid = f["packed"], f["valid"]
    if indices.size:
        for i, (p, frame) in zip(indices, source_frames(record["source"]["registered_path"], pts[indices]), strict=True):
            if not valid[i]:
                continue
            raw = frame.to_ndarray(format="bgr24")
            person = np.unpackbits(packed[i], axis=1, count=WIDTH)
            mask = restore_source_graphics(raw, p, cv2.dilate(person, np.ones((17, 17), np.uint8))).astype(bool)
            pr = reader.native.core.project_motion(rig, controls, FIRST, DT, HEIGHT, times[i], reader.world)
            for path, family, xy, station, normal, contrast, width, hint, score in reader.read(raw, mask, pr, centre[i]):
                n = len(station)
                row = dict(path=np.repeat(path, n), family=np.repeat(family, n),
                    pts=np.repeat(p, n), idx=np.repeat(i, n), fold=np.repeat(folds[i], n),
                    uv=xy, station=station, normal=normal, contrast=contrast,
                    guide_error=np.linalg.norm(xy - hint, axis=1),
                    material=np.repeat("source_board_print_contour", n),
                    source_transition_fwhm_px=width, source_registration_score=np.repeat(score, n))
                for key, data in row.items():
                    entries[key].append(data)
                if p not in hashes:
                    hashes[p] = hashlib.sha256(raw.tobytes()).hexdigest()
    additions = {k: np.concatenate(v) for k, v in entries.items()}
    receipt = dict(shot=sid, target_exposures=len(indices), material_rows=0, exposures_with_new_print=0)
    if additions:
        # Require the same physical print to be seen repeatedly in the local source interval.
        frame_index = additions["idx"]
        families = additions["family"]
        supported = np.zeros(len(frame_index), bool)
        for family in np.unique(families):
            own = families == family
            seen = np.unique(frame_index[own])
            near = np.diff(seen) <= 3
            good = seen[np.r_[False, near] | np.r_[near, False]]
            supported |= own & np.isin(frame_index, good)
        additions["eligible"] = supported
        additions["eligible_before_quality_review"] = np.ones(len(supported), bool)
        additions["excluded_reason"] = np.where(supported, "", "isolated_raw_print_read")
        additions["source_frame_pts"] = np.array(list(hashes), np.int64)
        additions["source_frame_BGR_sha256"] = np.array(list(hashes.values()))
        save(output / f"{sid}.npz", **additions, source_sha256=record["source"]["sha256"],
             geometry_sha256=record["geometry_binding"]["value_sha256"])
        receipt.update(material_rows=int(np.sum(supported)),
                       exposures_with_new_print=len(np.unique(frame_index[supported])),
                       groups=dict(Counter(families[supported])))
    write(output / f"{sid}.json", receipt)
    print(receipt, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guide", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--visibility", required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--windows", type=Path)
    p.add_argument("--only-windows",action="store_true",help="Use the explicit native source intervals")
    p.add_argument("--panel-ids", type=int, nargs="+", help="Read the four material borders in explicit windows")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shots", nargs="+")
    a = p.parse_args()
    cv2.setNumThreads(1)
    record = scope()
    if sha(record["source"]["registered_path"]) != record["source"]["sha256"]:
        raise ValueError("The original file has changed")
    ji.enable_historical_source_readers()
    if (a.panel_ids or a.only_windows) and not a.windows:
        p.error("Explicit source windows are required")
    reader = PrintReader(record, a.panel_ids)
    coverage = {r["shot"]: r for r in read(a.coverage)}
    windows = read(a.windows) if a.windows else {}
    output = OUTPUT / a.out
    output.mkdir(exist_ok=True)
    shots = [s for s in record["shots"] if a.shots is None or s["id"] in a.shots]
    shots.sort(key=lambda s: s["id"] not in {"s02", "s04", "s14", "s18", "s20"})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(process, reader, record, shot, coverage[shot["id"]], windows.get(shot["id"], []),
                            OUTPUT/a.previous, OUTPUT/a.guide, OUTPUT/a.visibility, output, a.only_windows)
                for shot in shots if not (output / f'{shot["id"]}.json').is_file()]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
