"""Combine signed readings with earlier per-exposure observations."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import OUTPUT, STEP, observation_packet, read, save, scope, write
from physical_merge import qualified_profiles
from stability_observations import ROW_FIELDS, groups, read_arrays


def exclude(rows, mask, reason):
    rows["eligible_before_quality_review"] |= rows["eligible"]
    rejected = mask & rows["eligible"]
    rows["eligible"][rejected] = False
    rows["excluded_reason"] = np.where(rejected, reason, rows["excluded_reason"])


def replace_group(older, newer):
    current = groups(newer)
    overlap = np.fromiter(((p, k) in current for p, k in zip(older["pts"], older["path"], strict=True)),
                          bool, count=len(older["pts"]))
    exclude(older, overlap, "same_material_exposure_reread_from_original")


def merge(shot, sources, output, windows):
    sid = shot["id"]
    paths = [observation_packet(p, sid) for p in sources]
    old, named, *extra = [read_arrays(p) if p.is_file() else None for p in paths]
    if old is None or named is None:
        raise ValueError("Missing full-shot observation input " + sid)
    physical, reviewed = extra
    protected = old["eligible"] & (np.char.startswith(old["material"], "source_")
                   | np.char.startswith(old["material"], "direct_")
                   | (old["material"] == "close_ad2_glyph_exact"))
    bound = set(zip(old["pts"][protected], old["path"][protected], strict=True))
    duplicate = np.fromiter(((p, k) in bound for p, k in zip(named["pts"], named["path"], strict=True)),
                            bool, count=len(named["pts"]))
    # A guide can distinguish repeated print only within a narrow search; the
    # manually reviewed signed profiles below have their own pixel evidence.
    exclude(named, duplicate | (named["guide_error"] > 17.)
            | ((sid == "s20") & ~np.char.startswith(named["family"], "Ad")),
            "existing_signed_material_or_ambiguous_named_read")
    if physical is not None:
        physical["eligible"], physical["excluded_reason"] = qualified_profiles(physical)
        reread = np.zeros(len(physical["pts"]), bool)
        for lo, hi in windows:
            use = (physical["pts"] >= lo) & (physical["pts"] <= hi)
            reread |= use
        exclude(physical, ~reread | ~np.isin(physical["family"], ("C", "B1", "B2")),
                "outside_selected_ice_stroke_rereads")
        physical["fold"][reread] = 0
        replace_group(old, physical)
        replace_group(named, physical)
    replace_group(old, named)
    if reviewed is not None:
        for item in (old, named, physical):
            if item is not None:
                replace_group(item, reviewed)
    # Keep all original measurements and their dispositions, including nearby
    # former holdouts, which can no longer serve as independent examinations.
    for lo, hi in windows:
        reused = (old["pts"] >= lo - 5 * STEP) & (old["pts"] <= hi + 5 * STEP)
        old["fold"][reused & (old["fold"] > 0)] = 3
    data = [z for z in (old, named, physical, reviewed) if z is not None]
    hashes = {}
    for z in data:
        for k in ("source_sha256", "geometry_sha256"):
            np.testing.assert_array_equal(z[k], old[k])
        for p, digest in zip(z["source_frame_pts"], z["source_frame_BGR_sha256"], strict=True):
            if hashes.setdefault(p, digest) != digest:
                raise ValueError("Different original frame decode at source PTS " + str(p))
    rows = {k: np.concatenate([z[k] for z in data]) for k in ROW_FIELDS}
    order = np.argsort(rows["pts"], kind="stable")
    hashes = dict(sorted(hashes.items()))
    save(output / f"{sid}.npz", **{k: v[order] for k, v in rows.items()},
         source_sha256=old["source_sha256"], geometry_sha256=old["geometry_sha256"],
         source_frame_pts=np.array(list(hashes), np.int64),
         source_frame_BGR_sha256=np.array(list(hashes.values())))
    training = rows["eligible"] & (rows["fold"] == 0)
    receipt = dict(shot=sid, training=int(training.sum()), frames=len(np.unique(rows["pts"][training])),
                   added_named=int(np.sum(named["eligible"] & (named["fold"] == 0))),
                   added_physical=int(np.sum(physical["eligible"] & (physical["fold"] == 0)))
                       if physical is not None else 0,
                   added_reviewed=int(np.sum(reviewed["eligible"] & (reviewed["fold"] == 0)))
                       if reviewed is not None else 0,
                   inputs=[str(p) for p in paths if p.is_file()])
    if reviewed is not None:
        receipt["reviewed_source_groups"] = dict(Counter(reviewed["family"][reviewed["eligible"]]))
    print(sid, {k: v for k, v in receipt.items() if k != "inputs"}, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--named", required=True)
    parser.add_argument("--physical", required=True)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    periods = read(args.windows)
    output = OUTPUT / args.out
    output.mkdir()
    sources = [OUTPUT / name for name in (args.baseline, args.named, args.physical, args.reviewed)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        work = [pool.submit(merge, shot, sources, output, periods.get(shot["id"], []))
                for shot in scope()["shots"]]
        results = [future.result() for future in work]
    write(output / "inputs.json", dict(reused_exposures=periods, shots=results))


if __name__ == "__main__":
    main()
