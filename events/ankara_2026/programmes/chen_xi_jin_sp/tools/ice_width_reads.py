"""Read both outside material transitions of a visible original ice stroke."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import os
import re
import sys
from time import monotonic, sleep

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import cv2
import numpy as np
from scipy.signal import find_peaks

from common import HEIGHT, JOINT, OUTPUT, WIDTH, read, save, scope, source_frames, visibility_packet, write
from ice_evidence import MIDPOINT, assert_ice_evidence, ice_roles
from ice_paint_profiles import read_paint_profiles
from measure import BOTTOM, DT, FIRST, TOP, fold_map, load_camera, motion_clock
from physical_reads import PhysicalReader, coherent, samples, source_profiles, subpixel
from review import restore_source_graphics

KNOWN_LANE_STROKE = re.compile(r"ice\.lane[1-4]:(?:edge0|edge2|F_transverse)$")

def tone_of(label):
    name = label.removeprefix("ice.").split(":", 1)[0]
    if name in {"B1", "B2"}:
        return "blue"
    if name in {"C", "C0", "A1", "A2", "A3", "A4"} or name.startswith(("goal_line_", "goal_trapezoid_")):
        return "red"
    if name.startswith(("lane", "house_connection_")):
        return "dark"
    return None


def paired_flanks(signal, hidden, radius, tone, permitted_width, *, minimum_contrast=None,
                  minimum_inside_chroma=3.5):
    """Check polarity at both sides and compare the interior to both ice flanks."""
    gradient = np.gradient(signal)
    peaks = find_peaks(gradient, height=.30, distance=3)[0]
    troughs = find_peaks(-gradient, height=.30, distance=3)[0]
    pixels = np.arange(len(signal))
    minimum, maximum = permitted_width
    options = []
    min_contrast = minimum_contrast if minimum_contrast is not None else (4.15 if tone == "dark" else 2.75)
    for left in peaks:
        for right in troughs[(troughs - left >= minimum) & (troughs - left <= maximum)]:
            width = right - left
            margin = min(19, max(5, int(width * .21)))
            centre = .5 * (left + right) - radius
            if abs(centre) > (68. if tone != "dark" else 30.) or left < margin or right + margin >= len(signal):
                continue
            if hidden[left - margin:right + margin + 1].any():
                continue
            baseline_left = float(np.median(signal[left - margin:left - 2]))
            baseline_right = float(np.median(signal[right + 3:right + margin + 1]))
            inset = max(1, int(width * .19))
            centre_samples = pixels[left + inset:right - inset + 1]
            interior = signal[centre_samples]
            strengths = float(np.quantile(interior, .31)) - np.array([baseline_left, baseline_right])
            if (strengths < min_contrast).any():
                continue
            if tone != "dark" and float(np.median(interior)) < minimum_inside_chroma:
                continue
            # Reject a transition deep inside the same painted band.
            outward = [
                (signal[left - margin:left - 2] - baseline_right),
                (signal[right + 3:right + margin + 1] - baseline_left),
            ]
            similar = min(strengths)
            if any(np.count_nonzero(x > max(2.7, .79 * similar)) >= int(np.ceil(.8 * len(x))) for x in outward):
                continue
            if min(gradient[left], -gradient[right]) < .22 * max(gradient[left], -gradient[right]):
                continue
            a = left + subpixel(gradient, left) - radius
            b = right + subpixel(-gradient, right) - radius
            score = similar * np.sqrt(min(gradient[left], -gradient[right])) * np.sqrt(min(width, 80.))
            score /= 1. + abs(centre) / 57.
            options.append(((a + b) * .5, score, a, b, similar, strengths))
    # A same-tone internal ridge cannot compete with its better-supported enclosing material band.
    options = [v for v in options if not any(w[2] < v[2] - 2. and w[3] > v[3] + 2.
               and w[4] >= .69 * v[4] for w in options)]
    return sorted(options, key=lambda q: -q[1])[:3]


def restrict_competing_reads(groups):
    """Remove an ice patch claimed by two different nearby registered strokes."""
    if len(groups) < 2:
        return
    from scipy.spatial import cKDTree

    uv = np.vstack([row["uv"] for row in groups])
    edges = np.vstack([row["source_bracket"] for row in groups])
    group_id = np.repeat(np.arange(len(groups)), [len(row["uv"]) for row in groups])
    radius = 12.
    suspect = np.zeros(len(uv), bool)
    delta = edges[:, 1] - edges[:, 0]
    widths = np.linalg.norm(delta, axis=1)
    normal = delta / widths[:, None]
    neighbors = cKDTree(uv).query_pairs(radius, output_type="ndarray")
    if len(neighbors):
        left, right = neighbors.T
        similar = (group_id[left] != group_id[right]) & (abs((normal[left] * normal[right]).sum(1)) > .87)
        similar &= (abs(((uv[left] - uv[right]) * normal[left]).sum(1))
                    < 2.5 + .15 * np.minimum(widths[left], widths[right]))
        similar &= (abs(widths[left] - widths[right]) < np.maximum(3., .20 * np.minimum(widths[left], widths[right])))
        suspect[neighbors[similar].ravel()] = True
    for i, row in enumerate(groups):
        row["ambiguous_identity"] = suspect[group_id == i]


class WidthReader:
    def __init__(self, record):
        sys.path.insert(0, str(JOINT))
        from joint_inputs import family, native_paths

        self.rails = PhysicalReader(record)
        self.native = self.rails.native
        self.core = self.native.core
        self.roles = ice_roles(self.native)
        self.paths = {}
        for key, role in self.roles.items():
            label = self.native.paths[key]["label"]
            tone = tone_of(label)
            if role == MIDPOINT and tone is not None:
                self.paths[key] = (label, tone, family(key, self.native))
        world = native_paths(self.native, [*self.paths, TOP, BOTTOM], count=241)
        self.station = {k: v[1] for k, v in world.items()}
        offsets = np.r_[0, np.cumsum([len(v[0]) for v in world.values()])]
        self.slices = {k: slice(int(a), int(b)) for k, a, b in zip(world, offsets[:-1], offsets[1:], strict=True)}
        self.world = np.vstack([v[0] for v in world.values()])

    def frame(self, raw, mask, rig, controls, when, centre, families=None):
        projected = self.core.project_motion(rig, controls, FIRST, DT, HEIGHT, when, self.world)
        uv = projected["uv"]
        uv[projected["status"] != 0] = np.nan
        visible = np.sum((self.rails.course - centre[:2]) * self.rails.inward, axis=1) <= 0.
        foot = uv[self.slices[BOTTOM]].copy()
        # Same native continuous perimeter sampled by the source rail reader.
        board_station = self.station[BOTTOM]
        known_station = self.rails.data[BOTTOM][1]
        side = np.interp(board_station, known_station, visible.astype(float)) >= .5
        foot[~side] = np.nan
        baseline = self.rails.board_foot(foot)
        floor = mask | ((np.arange(HEIGHT)[:, None] < baseline[None, :] + 13.) & (baseline[None, :] > .1))
        floor = floor.astype(np.uint8, copy=False)
        groups = []
        for path, (label, tone, family) in self.paths.items():
            if families is not None and family not in families:
                continue
            broad = family in ("B1", "B2", "C")
            curved = self.native.paths[path]["kind"] != 0
            radius = 164 if broad else 69 if tone != "dark" else 46
            step = 28. if broad else 21. if curved else 25.
            guide = samples(uv[self.slices[path]], self.station[path], step=step)
            if guide is None:
                continue
            points, normal, tangent, station = guide
            x, y = np.rint(points).astype(int).T
            visible_ice = (y > baseline[x] + 9.) & (floor[y, x] == 0)
            points, normal, tangent, station = (z[visible_ice] for z in (points, normal, tangent, station))
            if len(points) < 4:
                continue
            if tone == "blue":
                guess = self.rails.stripe_footprint(path, station, normal, rig, controls, when)
                reads, _ = read_paint_profiles(raw, floor, points, normal,
                                               np.clip(.87 * guess, 4., 190.), [tone] * len(points), new=True)
                options = []
                for value, footprint in zip(reads, guess, strict=True):
                    if value is None or footprint <= 0.:
                        options.append([])
                        continue
                    score, low, high, contrast, _, _ = value
                    options.append([(.5 * (low + high), score, low, high, min(contrast), contrast)])
            else:
                values, hidden, _ = source_profiles(raw, floor, points, normal, tangent, radius, tone)
                if broad:
                    guess = self.rails.stripe_footprint(path, station, normal, rig, controls, when)
                    bounds = np.column_stack((np.clip(.52 * guess, 3., 156.), np.clip(2.07 * guess + 4., 8., 250.)))
                else:
                    bounds = np.tile([3., 72. if tone != "dark" else 24.], (len(station), 1))
                options = [paired_flanks(p, m, radius, tone, lim)
                           for p, m, lim in zip(values, hidden, bounds, strict=True)]
            # A circle centreline may see different normal offsets on separate arcs.
            # Measure each arc from its own visible material samples.
            segments = np.array_split(np.arange(len(station)), int(np.ceil(len(station) / 15.))) if curved else (np.arange(len(station)),)
            selected = []
            for section in segments:
                selection = coherent([options[i] for i in section], points[section], stripe=True)
                selected.extend((int(section[i]), q) for i, q in selection)
            if not selected:
                continue
            index = np.array([i for i, _ in selected])
            q = [v for _, v in selected]
            left_right = np.array([v[2:4] for v in q])
            edges = points[index, None, :] + left_right[:, :, None] * normal[index, None, :]
            measured = edges.mean(axis=1)
            amount = len(measured)
            groups.append(dict(path=np.repeat(path, amount), family=np.repeat(family, amount),
                station=station[index], uv=measured, normal=normal[index],
                source_bracket=edges, source_side_contrast=np.array([v[5] for v in q]),
                contrast=np.array([v[4] for v in q]),
                guide_error=np.linalg.norm(points[index] - measured, axis=1),
                material=np.repeat("source_verified_width_centre", amount),
                ice_measurement_role=np.repeat(MIDPOINT, amount),
                ambiguous_identity=np.zeros(amount, bool),
                identified_stroke=np.repeat(tone != "dark" or bool(KNOWN_LANE_STROKE.fullmatch(label)), amount)))
        restrict_competing_reads(groups)
        return groups


def init(record):
    global RECORD, READER
    cv2.setNumThreads(1)
    RECORD = record
    READER = WidthReader(record)


def collect(job):
    shot, guide, destination, stride, windows, families, visibility = job
    start = monotonic()
    sid = shot["id"]
    pts, controls, rig = load_camera(shot, guide)
    times = motion_clock(shot, guide, pts)
    samples_camera = READER.core.evaluate_motion_camera(rig, controls, FIRST, DT, times)
    if np.any(samples_camera["status"]):
        raise ValueError("Undefined guide camera for " + sid)
    roles = fold_map(shot, len(pts))
    use = np.ones(len(pts), bool)
    if windows is not None:
        use.fill(False)
        for lo, hi in windows.get(sid, []):
            use |= (pts >= lo) & (pts <= hi)
    selected = np.flatnonzero(use)[::stride]
    if not len(selected):
        return
    with np.load(visibility_packet(visibility, sid), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["pts"], pts)
        assert z["source_sha256"].item() == RECORD["source"]["sha256"]
        packed, valid = z["packed"], z["valid"]
    values = defaultdict(list)
    digests = {}
    for p, f in source_frames(RECORD["source"]["registered_path"], pts[selected]):
        i = int(np.searchsorted(pts, p))
        if not valid[i]:
            continue
        raw = f.to_ndarray(format="bgr24")
        mask = cv2.dilate(np.unpackbits(packed[i], axis=1, count=WIDTH), np.ones((17, 17), np.uint8))
        mask = restore_source_graphics(raw, p, mask)
        digests[p] = sha256(raw.tobytes()).hexdigest()
        for group in READER.frame(raw, mask, rig, controls, times[i], samples_camera["centre"][i], families):
            n = len(group["uv"])
            group.update(pts=np.full(n, p, np.int64), idx=np.full(n, i, np.int32),
                         fold=np.full(n, roles[i], np.int8))
            for key, data in group.items():
                values[key].append(data)
        if len(digests) % 150 == 0:
            print(sid, len(digests), "original frames", flush=True)
    receipt = dict(shot=sid, original_frames=len(digests), source_width_midpoints=0,
                   by_path={}, seconds=monotonic() - start)
    if values:
        arrays = {k: np.concatenate(v) for k, v in values.items()}
        repeated = np.zeros(len(arrays["pts"]), bool)
        for path in np.unique(arrays["path"]):
            which = arrays["path"] == path
            visible = np.unique(arrays["idx"][which & ~arrays["ambiguous_identity"]])
            delta = np.diff(visible)
            if len(visible):
                near = visible[np.r_[False, delta <= 2 * stride] | np.r_[delta <= 2 * stride, False]]
                repeated |= which & np.isin(arrays["idx"], near)
        ambiguous = arrays.pop("ambiguous_identity")
        typed = arrays.pop("identified_stroke")
        if sid == "s20":
            # Source review has no confirmed ice-line identity in this close-up.
            typed.fill(False)
        arrays.update(eligible=repeated & ~ambiguous & typed,
                      eligible_before_quality_review=np.ones(len(repeated), bool),
                      excluded_reason=np.where(~typed, "source_stroke_identity_unresolved",
                          np.where(ambiguous, "two_ice_path_identities_claim_same_material",
                              np.where(repeated, "", "material_pair_missing_in_nearby_exposure"))),
                      source_sha256=RECORD["source"]["sha256"],
                      geometry_sha256=RECORD["geometry_binding"]["value_sha256"],
                      source_frame_pts=np.array(list(digests), dtype=np.int64),
                      source_frame_BGR_sha256=np.array(list(digests.values())))
        assert_ice_evidence(arrays, arrays["eligible"], READER.roles)
        save(destination / f"{sid}.npz", **arrays)
        receipt.update(source_width_midpoints=int(arrays["eligible"].sum()),
                       by_path=dict(Counter(arrays["path"][arrays["eligible"]])))
    write(destination / f"{sid}.json", receipt)
    print(sid, receipt["source_width_midpoints"], "two-sided material midpoints", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--windows")
    parser.add_argument("--families",help="Optional JSON shot-to-family selection for specific source re-reads")
    parser.add_argument("--visibility",default="visibility_final")
    parser.add_argument("--watch-visibility", action="store_true")
    parser.add_argument("--shots", nargs="+")
    args = parser.parse_args()
    record = scope()
    out = OUTPUT / args.out
    out.mkdir(exist_ok=True)
    windows = read(args.windows) if args.windows else None
    families = read(args.families) if args.families else {}
    work = [s for s in record["shots"] if (args.shots is None or s["id"] in args.shots)
            and (windows is None or s["id"] in windows) and not (out / f"{s['id']}.json").is_file()]
    visibility = OUTPUT / args.visibility
    jobs = [(s, OUTPUT / args.guide, out, args.stride, windows, families.get(s["id"]), visibility)
            for s in work]
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(record,)) as pool:
        if not args.watch_visibility:
            list(pool.map(collect, jobs))
        else:
            active = []
            while jobs or active:
                ready = [job for job in jobs if visibility_packet(visibility, job[0]["id"]).is_file()]
                for job in ready:
                    active.append(pool.submit(collect, job))
                    jobs.remove(job)
                done = [future for future in active if future.done()]
                for future in done:
                    future.result()
                    active.remove(future)
                if jobs or active:
                    sleep(.4)


if __name__ == "__main__":
    main()
