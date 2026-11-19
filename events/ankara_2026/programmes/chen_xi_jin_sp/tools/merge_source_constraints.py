"""Bind additional current-source print and motion measurements for joint fitting."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from common import OUTPUT, frozen_scene, observation_packet, save, scope, source_pts, write
from fit import measurement_rows
from ice_evidence import ICE_EVIDENCE_FIELDS, ice_roles
from precision_observations import exclude
from stability_audit import load_tracks
from stability_observations import ROW_FIELDS, read_arrays

PRINT_FIELDS = ("source_transition_fwhm_px", "source_registration_score")
TRACK_FIELDS = ("index", "lag", "start", "fold", "world", "uv", "left_uv", "prev2_uv",
                "anchor_uv", "track_id", "family", "quality", "role")


def available_print(rows):
    return rows["eligible"] & np.char.startswith(rows["family"], "Ad")


def thin(rows, indexes, count):
    """Take spatially distributed signed contours instead of dense repeats."""
    if len(indexes) <= count:
        return indexes
    xy = rows["uv"][indexes]
    normal = rows["normal"][indexes]
    confidence = np.minimum(35., rows["contrast"][indexes]) - rows["source_transition_fwhm_px"][indexes]
    # Give the dimension unsupported by the nearly horizontal board rails
    # the first selection and then keep points spread over the actual print.
    seed = np.argmax(confidence + 12. * abs(normal[:, 0]))
    take = [int(seed)]
    distance = np.full(len(xy), np.inf)
    for _ in range(1, count):
        distance = np.minimum(distance, np.linalg.norm((xy - xy[take[-1]]) * [1., .72], axis=1))
        rank = distance * (.75 + .25 * abs(normal[:, 0])) + .18 * confidence
        rank[take] = -np.inf
        take.append(int(np.argmax(rank)))
    return indexes[take]


def qualified_new(older, new, *, per_path=False):
    """Use current print only when it supplies distinct measurable locations."""
    grouping = "path" if per_path else "family"
    old = available_print(older)
    groups = {}
    for subset in iter_groups(older, old, grouping):
        key = int(older["idx"][subset[0]]), str(older[grouping][subset[0]])
        groups[key] = subset
    collected = np.zeros(len(new["idx"]), bool)
    for subset in iter_groups(new, new["eligible"], grouping):
        i = subset[0]
        key = int(new["idx"][i]), str(new[grouping][i])
        existing = groups.get(key)
        if existing is not None:
            # Independently measured points on other shapes add position
            # evidence. Preserve any existing confirmed read for this path.
            subset = subset[~np.isin(new["path"][subset], older["path"][existing])]
            if len(subset):
                d = new["uv"][subset, None, :] - older["uv"][existing][None, :, :]
                subset = subset[np.min(np.sum(d*d, axis=2), axis=1) > 8.**2]
        if len(subset):
            chosen = thin(new, subset, 14 if existing is None else 8)
            collected[chosen] = True
    return collected


def iter_groups(rows, selected, grouping="family"):
    take = np.flatnonzero(selected)
    if not len(take):
        return
    take = take[np.lexsort((rows[grouping][take], rows["idx"][take]))]
    between = ((np.diff(rows["idx"][take]) != 0) |
               (rows[grouping][take[1:]] != rows[grouping][take[:-1]]))
    for group in np.split(take, np.flatnonzero(between) + 1):
        yield group


def observation_join(shot, older, fresh, output, *, per_path=False):
    old = read_arrays(older)
    new = read_arrays(fresh)
    hashes = {}
    for packet in (old, new):
        for key in ("source_sha256", "geometry_sha256"):
            np.testing.assert_array_equal(packet[key], old[key])
        np.testing.assert_array_equal(packet["pts"], source_pts(shot)[packet["idx"]])
        for p, digest in zip(packet["source_frame_pts"], packet["source_frame_BGR_sha256"], strict=True):
            if hashes.setdefault(int(p), digest) != digest:
                raise ValueError(f"Source decode differs at PTS {p}")
    if not np.all(np.char.startswith(new["family"], "Ad")):
        raise ValueError("Only named print contours here")
    previous = available_print(old)
    newly_available = new["eligible"].copy()
    admit = qualified_new(old, new, per_path=per_path)
    exclude(new, ~admit, "qualified_same_exposure_print_already_present_or_spatially_sampled")
    absent = ~np.isin(new["idx"][admit], old["idx"][previous])
    for field, shape, dtype in (
            ("source_bracket", (2,2), float), ("source_side_contrast", (2,), float),
            ("ice_measurement_role", (), "<U1"),
            ("source_transition_fwhm_px", (), float), ("source_registration_score", (), float)):
        for packet in (old, new):
            if field not in packet:
                packet[field] = np.full((len(packet["pts"]), *shape), "" if dtype == "<U1" else np.nan,
                                        dtype=dtype)
    rows = {key: np.concatenate((old[key], new[key])) for key in (*ROW_FIELDS, *ICE_EVIDENCE_FIELDS, *PRINT_FIELDS)}
    ordered = np.argsort(rows["pts"], kind="stable")
    pts = np.array(sorted(hashes), np.int64)
    save(output, **{k: row[ordered] for k, row in rows.items()}, source_frame_pts=pts,
         source_frame_BGR_sha256=np.array([hashes[p] for p in pts]),
         source_sha256=old["source_sha256"], geometry_sha256=old["geometry_sha256"])
    return dict(qualified_source_rows=int(newly_available.sum()), distinct_active_rows=int(admit.sum()),
                new_training_rows=int(np.sum(admit & (new["fold"] == 0))),
                all_new_frames=int(np.unique(new["idx"][admit]).size),
                frames_previously_without_print=int(np.unique(new["idx"][admit][absent]).size))


class BoardArtwork:
    def __init__(self, native):
        board = native.scene["board"]
        s = np.linspace(0., native.core.board_perimeter(board), 3201)
        self.station = s
        self.foot = cKDTree(np.asarray(native.core.board_points(board, s, np.zeros(len(s))))[:, :2])
        self.reference_only = []
        for ad in (1, 14):
            prefix = f"print_instance/Ad{ad}::"
            # The app board has QR cells and phone interfaces. Motion on
            # those details has no role in the fitted scene. The separately
            # read named panel edges and glyphs supply absolute evidence.
            uv = np.vstack([v["vertices"] for k, v in native.paths.items()
                            if k.startswith(prefix) and k.endswith(("left_vertical", "right_vertical"))])
            self.reference_only.append((uv[:, 0].min() - .002, uv[:, 0].max() + .002))

    def track_gate(self, data):
        board = data["family"] == 0
        s = self.station[self.foot.query(data["world"][board, :2])[1]]
        reject = np.zeros(len(s), bool)
        for a, b in self.reference_only:
            reject |= (s >= a) & (s <= b)
        keep = np.ones(len(board), bool)
        keep[board] = ~reject
        return keep


def temporal_join(record, shot, older, fresh, output, artwork, reviewed_track_exclusions=()):
    old = load_tracks(older, record, shot)
    new = load_tracks(fresh, record, shot)
    for data in (old, new):
        if np.min(data["index"]-data["lag"], initial=0) < 0:
            raise ValueError("A track crosses a hard cut")
    if np.intersect1d(old["track_id"], new["track_id"]).size:
        raise ValueError("Track IDs collide in " + shot["id"])
    old_valid, new_valid = artwork.track_gate(old), artwork.track_gate(new)
    source_detail_exclusions = [int((~old_valid).sum()), int((~new_valid).sum())]
    athlete_exclusions = []
    for data, keep in ((old, old_valid), (new, new_valid)):
        clear = ice_clear_of_athlete(data, shot)
        for lo, hi, rectangle in reviewed_track_exclusions:
            is_ice = data["family"] == 1
            for index, uv in ((data["index"], data["uv"]), (data["index"] - data["lag"], data["left_uv"]),
                              (data["start"], data["anchor_uv"]), (data["index"] - 2, data["prev2_uv"])):
                pts = data["pts"][np.maximum(0, index)]
                px = (uv >= rectangle[:2]) & (uv <= rectangle[2:])
                clear &= ~(is_ice & (pts >= lo) & (pts <= hi) & px.all(1))
        athlete_exclusions.append(int(np.sum(keep & ~clear)))
        keep &= data["usable"] & clear
    rows = {k: np.concatenate((old[k][old_valid], new[k][new_valid])) for k in TRACK_FIELDS}
    independent = np.r_[np.zeros(int(old_valid.sum()), bool), np.ones(int(new_valid.sum()), bool)]
    centres = distinct_track_centres(rows)
    rows = {k: v[centres] for k, v in rows.items()}
    save(output, **rows, pts=new["pts"], source_sha256=new["source_sha256"],
         geometry_sha256=new["geometry_sha256"], graphics_mask_per_exposure=True,
         independent_patch=independent[centres])
    info = dict(reference_only_detail_rows_excluded=source_detail_exclusions,
                athlete_rows_excluded=athlete_exclusions, repeated_source_patch_centres_excluded=int((~centres).sum()))
    for fam, label in ((0, "board"), (1, "ice")):
        for lag in (1, 8):
            def coverage(data):
                selected = (data["family"] == fam) & (data["lag"] == lag) & (data["role"] == 0)
                count = np.bincount(data["index"][selected], minlength=len(data["pts"]))
                return int(np.count_nonzero(count >= 2))
            info[f"{label}_{lag}_frames"] = [coverage(old), coverage(rows | {"pts": new["pts"]})]
    return info


def ice_clear_of_athlete(data, shot):
    """Reject patch centres on the skater in all exposures of a motion row."""
    with np.load(OUTPUT/"foreground"/f'{shot["id"]}.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(data["pts"], f["pts"])
        np.testing.assert_array_equal(data["source_sha256"], f["source_sha256"])
        boxes, offsets = f["boxes"], f["offsets"]
    passed = np.ones(len(data["index"]), bool)
    ice = np.flatnonzero(data["family"] == 1)
    for t, uv in ((data["index"], data["uv"]),
                  (data["index"] - data["lag"], data["left_uv"]),
                  (data["start"], data["anchor_uv"]),
                  (data["index"] - 2, data["prev2_uv"])):
        subset = ice[np.isfinite(uv[ice]).all(1) & (t[ice] >= 0)]
        order = subset[np.argsort(t[subset], kind="stable")]
        split = np.flatnonzero(np.diff(t[order])) + 1
        for own in np.split(order, split):
            if not len(own):
                continue
            j = t[own[0]]
            b = boxes[offsets[j]:offsets[j + 1], :4]
            margin = np.asarray([-34., -25., 34., 59.])
            v = b + margin
            close = ((uv[own, None, :] >= v[None, :, :2]) &
                     (uv[own, None, :] <= v[None, :, 2:])).all(2)
            passed[own] &= ~close.any(1)
    return passed


def distinct_track_centres(data):
    """Repeated seeds on the same original patch count once per motion pair."""
    index, lag, family = (data[k] for k in ("index", "lag", "family"))
    order = np.lexsort((data["quality"], -data["role"].astype(int), lag, family, index))
    cell = np.column_stack((index[order], family[order], lag[order]))
    borders = np.flatnonzero(np.any(np.diff(cell, axis=0), axis=1)) + 1
    kept = np.zeros(len(index), bool)
    for q in np.split(order, borders):
        if not len(q):
            continue
        p = data["uv"][q]
        min_spacing = 13. if family[q[0]] == 1 else 9.
        invalid = np.zeros(len(q), bool)
        for i, j in enumerate(q):
            if invalid[i]:
                continue
            kept[j] = True
            invalid[i+1:] |= np.sum((p[i+1:] - p[i])**2, axis=1) < min_spacing**2
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", required=True)
    p.add_argument("--print", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--track-out", required=True)
    args = p.parse_args()
    record = scope()
    native = frozen_scene(record, verify=True)
    roles = ice_roles(native)
    artwork = BoardArtwork(native)
    dst, temporal = OUTPUT/args.out, OUTPUT/args.track_out
    dst.mkdir()
    temporal.mkdir()
    raw_print, patch = OUTPUT/args.print, OUTPUT/args.patch

    def job(shot):
        sid = shot["id"]
        if not (raw_print/f"{sid}.json").is_file() or not (patch/f"{sid}.json").is_file():
            raise ValueError("Finish original source measurement for " + sid)
        original = observation_packet(OUTPUT/args.observations, sid)
        added = raw_print/f"{sid}.npz"
        pdf = {}
        if added.is_file():
            pdf = observation_join(shot, original, added, dst/f"{sid}.npz")
        report = temporal_join(record, shot, observation_packet(OUTPUT/args.tracks, sid),
                               observation_packet(patch, sid), temporal/f"{sid}.npz", artwork)
        print(sid, pdf, report, flush=True)
        return sid, original, pdf, report

    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(job, record["shots"]))
    write(dst/"shot_packets.json", {sid: str(dst/f"{sid}.npz" if (dst/f"{sid}.npz").is_file() else original)
                                    for sid, original, _, _ in result})
    for shot in record["shots"]:
        measurement_rows(dst, shot, record, roles)
    write(dst/"inputs.json", dict(qualified_print=str(raw_print), observations=args.observations,
                                 shots={sid: pdf for sid, _, pdf, _ in result}))
    write(temporal/"inputs.json", dict(independent_source_patches=str(patch), previous_tracks=args.tracks,
                                      shots={sid: details for sid, _, _, details in result}))


if __name__ == "__main__":
    main()
