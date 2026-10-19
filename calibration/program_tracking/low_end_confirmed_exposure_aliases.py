"""Source-only alias map for confirmed repeated low-end programme exposures.

This module does not build, change, or solve a camera graph. It keeps every
native source PTS and only supplies a canonical frame key for groups already
confirmed by the frozen full60-to-official50 pixel mapping. A later graph
wrapper must preserve every original factor, UV, prior target, noise model,
and frame output, while deduplicating aliased keys within each factor and
summing their Jacobian blocks. Repeated image evidence stays correlated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
V13 = TMP / "joint_programme_atlas_provenance_full_v13"
CADENCE = TMP / "source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment"
SOURCE_SHOTS = ROOT / "calibration/program_tracking/source_shots.json"
QUALIFIED = TMP / "low_end_board_edges_qualified_only_v1"
OUTPUT = TMP / "low_end_confirmed_exposure_alias_preflight_v1"
SHOTS = (3, 5, 8)
SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
MODEL_SHA256 = "c7c59e2efb9782e6a3a45cf21ae7ad39958ad4ccdacb50437d4ad87056cd107a"
SOURCE_SHOTS_SHA256 = "feb7f42539bab4a7def84e74eaaf40dd0808710e3c0c0c101e64c381ddc63197"
GROUPS_SHA256 = "5d43945541c5d6621827a49f9c23f025f0a90941bdc6480b059decab64eb74f2"
MAPPING_SHA256 = "a9be37308b9a39fb55519a0be69c741c53a8216a81a60a2066fff25bdcf54dcc"
QUALIFIED_PHASE_SHA256 = "046a9f09d9e8c682063f18b80fa1a613114faa1d1393d51feb24529d44bf0714"
QUALIFIED_CAMERAS_SHA256 = "73d5010132552444a4086152af2ee43e1439b0d15f17d0737cc2044a1711ff01"
QUALIFIED_STATES_SHA256 = "41e356a453670e1f5b1e53884864d6ebc36cf4d1829fcce3bdef47dafeda07c8"


SUPPORT_FILES = {
    (3, "board_top"): TMP / "shot03_board_top_support_v1/board_top_support.npz",
    (3, "contact"): TMP / "shot03_contact_support_v2/contact_support.npz",
    (5, "board_top"): TMP / "low_end_source_edge_support_05_08_adaptive_v2/full/shot05_board_top_support.npz",
    (5, "contact"): TMP / "low_end_source_edge_support_05_08_adaptive_v2/full/shot05_contact_support.npz",
    (8, "board_top"): TMP / "low_end_source_edge_support_05_08_adaptive_v2/full/shot08_board_top_support.npz",
    (8, "contact"): TMP / "low_end_source_edge_support_05_08_adaptive_v2/full/shot08_contact_support.npz",
}
SUPPORT_SHA256 = {
    (3, "board_top"): "2f805f049c886feea66b139304f339665aeba885b9e961676d854d9c2d2d84c4",
    (3, "contact"): "f13106520945833664ac37eae926bf82293a1fa3379b4f759734c4c00f2833f5",
    (5, "board_top"): "270ecf8d5db57f3aafbb23ce95ae8837d156d5f62cf0a2cb7a3de68d528af191",
    (5, "contact"): "1a2a44ca67c82747cb2c4ceb95df0cebcb231fa921f615cfb05b5eed86245c42",
    (8, "board_top"): "a4ef29f314319ecaa93005be20c4d62ac9d83ca9dbe8c11b50d3669a906cd039",
    (8, "contact"): "8928cb2850c0b8cb4c617167a491b3e774f3d7162d9c753789811e6c2319f535",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alias_jacobian_blocks(keys: tuple[int, ...], blocks: list[np.ndarray],
                          aliases: dict[int, int]) -> tuple[tuple[int, ...], list[np.ndarray]]:
    """Preserve one factor's raw Jacobian after repeated keys share a state.

    The factor residual must separately expand each original key position to
    its aliased state. In particular, (a,b,mount) with a,b in one exposure
    becomes (representative,mount), with J_representative = J_a + J_b.
    """
    require(len(keys) == len(blocks), "Original factor Jacobian omitted a key block")
    target = tuple(aliases.get(int(key), int(key)) for key in keys)
    unique = tuple(dict.fromkeys(target))
    merged = {key:np.zeros_like(np.asarray(blocks[0],np.float64)) for key in unique}
    for key, block in zip(target, blocks):
        matrix = np.asarray(block,np.float64)
        require(matrix.shape == merged[key].shape and np.isfinite(matrix).all(),
                "Original factor Jacobian block changed shape or became nonfinite")
        merged[key] += matrix
    return unique, [merged[key] for key in unique]


def load_confirmed_aliases() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return exact original PTS, representative PTS, seed PTS, and audit."""
    paths = dict(source_shots=SOURCE_SHOTS,
        groups=CADENCE/"exposure_groups.json",
        mapping=CADENCE/"native60_to_official50.jsonl",
        qualified_phase=QUALIFIED/"phase_00.json",
        qualified_cameras=QUALIFIED/"fixed/shot_cameras.npz",
        qualified_states=QUALIFIED/"fixed/states.npz")
    expected = dict(source_shots=SOURCE_SHOTS_SHA256,groups=GROUPS_SHA256,
        mapping=MAPPING_SHA256,qualified_phase=QUALIFIED_PHASE_SHA256,
        qualified_cameras=QUALIFIED_CAMERAS_SHA256,
        qualified_states=QUALIFIED_STATES_SHA256)
    require(all(digest(paths[name]) == sha for name,sha in expected.items()),
            "Frozen source exposure or qualified camera evidence changed")
    source = json.loads(SOURCE_SHOTS.read_text(encoding="utf8"))
    packet = json.loads(paths["groups"].read_text(encoding="utf8"))
    phase = json.loads(paths["qualified_phase"].read_text(encoding="utf8"))
    require(source["source_sha256_from_timeline"] == SOURCE_SHA256
            and source["native_frame_count"] == 10140
            and len(source["shots"]) == 10 and len(source["native_cuts"]) == 9
            and packet["schema"] == "rinkspace.official50-confirmed-exposure-groups.v1"
            and packet["source_binding"]["full60"]["sha256"] == SOURCE_SHA256
            and packet["validation"]["confirmed_pair_count"] == 1620
            and packet["validation"]["all_native_pts_exactly_once"] is True
            and phase["schema"] == "rinkspace.low-end-board-edge-qualified-only-ablation.v1"
            and phase["source_sha256"] == SOURCE_SHA256
            and phase["scene_sha256"] == MODEL_SHA256
            and phase["frames"] == 2219 and phase["states"] == 2221,
            "Frozen source/scene/shot/qualified phase identity changed")
    with np.load(source["native_packet_pts_cache"],allow_pickle=False) as raw:
        all_source_pts = np.asarray(raw["pts"],np.int64)
    require(all_source_pts.shape == (10140,) and np.all(np.diff(all_source_pts)>0),
            "Full original native source PTS cache changed")
    with np.load(paths["qualified_cameras"],allow_pickle=False) as packet_cameras:
        pts = np.asarray(packet_cameras["pts"],np.int64)
        cameras = np.asarray(packet_cameras["camera25"],np.float64)
        shot_index = np.asarray(packet_cameras["shot_index"],np.int16)
        require(str(packet_cameras["source_sha256"]) == SOURCE_SHA256
                and str(packet_cameras["model_sha256"]) == MODEL_SHA256,
                "Completed qualified camera packet source or scene changed")
    require(pts.shape == (2219,) and cameras.shape == (2219,25)
            and np.isfinite(cameras).all() and np.isin(shot_index,SHOTS).all()
            and np.all(np.diff(pts)>0) and np.isin(pts,all_source_pts).all(),
            "Qualified camera stream PTS/Brown source changed")
    with np.load(paths["qualified_states"],allow_pickle=False) as states:
        require(states["keys"].shape == (2221,)
                and states["states"].shape == (2221,11)
                and np.isfinite(states["states"]).all(),
                "Completed qualified frame states changed")
    by_pts = {int(stamp):i for i,stamp in enumerate(pts)}
    original_keyframes = {}
    v13_hashes = {}
    for shot in SHOTS:
        path = V13/f"shot{shot:02d}.npz"
        v13_hashes[str(shot)] = digest(path)
        with np.load(path,allow_pickle=False) as original:
            original_pts = np.asarray(original["pts"],np.int64)
            require(np.array_equal(original_pts,pts[shot_index==shot])
                    and str(original["source_sha256"]) == SOURCE_SHA256
                    and str(original["model_sha256"]) == MODEL_SHA256,
                    f"Shot{shot:02d} original source PTS/scene changed")
            original_keyframes[shot] = set(map(int,original["original_keyframe_pts"]))
    mapping_rows = {}
    with paths["mapping"].open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = int(row["full_pts"])
            if stamp in by_pts:
                require(stamp not in mapping_rows, "Duplicate official mapping source PTS")
                mapping_rows[stamp] = row
    require(set(mapping_rows) == set(map(int,pts)),
            "Official mapping omitted a low-end source PTS")
    aliases = {int(stamp):int(stamp) for stamp in pts}
    seeds = aliases.copy()
    groups = []
    covered = set()
    for group in packet["groups"]:
        stamps = list(map(int,group["full_pts"]))
        if int(group["shot_index"]) not in SHOTS or len(stamps)==1:
            continue
        shot = int(group["shot_index"])
        require(len(stamps) in (2,3) and stamps == sorted(stamps)
                and group["criterion"] == "confirmed_same_official_exposure_pair_chain"
                and group["confirmed_official_pts"] is not None
                and len(group["confirmed_pair_evidence"]) == len(stamps)-1
                and all(p in by_pts and int(shot_index[by_pts[p]]) == shot
                        and p not in covered for p in stamps),
                "Low-end repeated exposure crosses a shot, reuses PTS, or is unconfirmed")
        official = int(group["confirmed_official_pts"])
        for p in stamps:
            row = mapping_rows[p]
            require(row["match_status"] == "unique_pixel_match"
                    and int(row["accepted_official_pts"]) == official,
                    f"Source PTS {p} is not a unique official pixel match")
        for evidence,(a,b) in zip(group["confirmed_pair_evidence"],zip(stamps[:-1],stamps[1:])):
            require(int(evidence["previous_full_pts"]) == a
                    and int(evidence["full_pts"]) == b and b-a == 256
                    and int(evidence["official_pts"]) == official
                    and evidence["criterion"] ==
                        "accepted_official_pts_equal_and_adjacent_within_shot_and_full_pair_roi_gray_mae_le_0.45"
                    and float(evidence["full_pair_roi_gray_mae"]) <= .45
                    and mapping_rows[b]["previous_pair_status"] ==
                        "confirmed_same_official_exposure",
                    "Official same-exposure pair failed the frozen pixel/adjacency rule")
        brown = cameras[by_pts[stamps[0]],11:25]
        require(all(np.array_equal(cameras[by_pts[p],11:25],brown) for p in stamps),
                "Same exposure has different Brown14 camera templates")
        representative = stamps[0]
        keyframe_seeds = [p for p in stamps if p in original_keyframes[shot]]
        seed = keyframe_seeds[0] if keyframe_seeds else representative
        for p in stamps:
            aliases[p] = representative
            seeds[p] = seed
        covered.update(stamps)
        groups.append(dict(group_id=group["group_id"],shot_index=shot,
            source_pts=stamps,official_pts=official,
            canonical_pts=representative,seed_source_pts=seed,
            keyframe_pts=keyframe_seeds,
            pair_roi_gray_mae=[float(e["full_pair_roi_gray_mae"])
                               for e in group["confirmed_pair_evidence"]]))
    canonical = np.asarray([aliases[int(p)] for p in pts],np.int64)
    seed_pts = np.asarray([seeds[int(p)] for p in pts],np.int64)
    require(len(groups)==366 and sum(len(x["source_pts"])-1 for x in groups)==368
            and len(np.unique(canonical))==1851
            and all(aliases[int(p)] in by_pts and seeds[int(p)] in by_pts for p in pts),
            "Low-end confirmed aliases unexpectedly change frame variable count")
    supports = {}
    support_hashes = {}
    for key,path in SUPPORT_FILES.items():
        shot,kind = key
        require(digest(path) == SUPPORT_SHA256[key],
                f"Shot{shot:02d} {kind} frozen source packet SHA changed")
        with np.load(path,allow_pickle=False) as support:
            sp = np.asarray(support["pts"],np.int64)
            offsets = np.asarray(support["offsets"],np.int64)
            eligible = np.asarray(support["eligible"],bool)
        require(np.array_equal(sp,pts[shot_index==shot])
                and offsets.shape == (len(sp)+1,)
                and offsets[0]==0 and offsets[-1]==len(eligible),
                f"Shot{shot:02d} {kind} reviewed source packet changed")
        supports[key] = (sp,offsets,eligible)
        support_hashes[f"shot{shot}_{kind}"] = digest(path)
    edge_stats = {}
    for shot in SHOTS:
        own = [g for g in groups if g["shot_index"]==shot]
        for kind in ("board_top","contact"):
            sp,offsets,eligible = supports[shot,kind]
            count = Counter()
            for group in own:
                for position,p in enumerate(group["source_pts"]):
                    i = int(np.searchsorted(sp,p))
                    lo,hi = map(int,offsets[i:i+2])
                    count["frames"] += 1
                    count["factors"] += int(hi>lo)
                    count["original_uv_rows"] += hi-lo
                    count["qualified_rows"] += int(eligible[lo:hi].sum())
                    if position:
                        count["later_same_exposure_factors"] += int(hi>lo)
                        count["later_same_exposure_qualified_rows"] += int(eligible[lo:hi].sum())
            edge_stats[f"shot{shot}_{kind}"] = dict(count)
    by_shot = {}
    for shot in SHOTS:
        own = [g for g in groups if g["shot_index"]==shot]
        keyframes = original_keyframes[shot]
        by_shot[str(shot)] = dict(frames=int((shot_index==shot).sum()),
            repeated_groups=len(own),group_size_counts=dict(Counter(len(g["source_pts"]) for g in own)),
            confirmed_links=sum(len(g["source_pts"])-1 for g in own),
            frame_variables_after_alias=int((shot_index==shot).sum())-
                sum(len(g["source_pts"])-1 for g in own),
            repeated_group_original_keyframe_priors=sum(p in keyframes for g in own
                for p in g["source_pts"]),
            nonrepresentative_keyframe_priors=sum(p in keyframes for g in own
                for p in g["source_pts"][1:]),
            repeated_dummy_center_priors=sum(len(g["source_pts"]) for g in own),
            repeated_frame_lens_ties=sum(len(g["source_pts"]) for g in own))
    # This tiny algebra check is independent of source pixels and factor values.
    ja,jb,jm = (np.full((2,11),value) for value in (2.,3.,5.))
    keys,blocks = alias_jacobian_blocks((1,2,99),[ja,jb,jm],{1:1,2:1})
    require(keys==(1,99) and np.array_equal(blocks[0],ja+jb)
            and np.array_equal(blocks[1],jm),
            "Duplicate-key Jacobian accumulation failed")
    audit = dict(schema="rinkspace.low-end-confirmed-exposure-alias-preflight.v1",
        source_sha256=SOURCE_SHA256,model_sha256=MODEL_SHA256,
        scope="shot03/05/08 qualified-only candidate; source-only confirmed exposure aliases",
        input_sha256={**expected,"v13_shots":v13_hashes,
                      "source_support_npz":support_hashes,"script":digest(Path(__file__))},
        source_policy="Only frozen same-shot groups with both unique official-pixel matches, equal accepted official PTS, adjacent exact full60 source PTS, and source ROI gray MAE <=0.45. No modulo-six rule.",
        original_source_pts=2219,original_camera_states=2221,
        aliased_frame_variables=1851,aliased_total_states_including_mount_lens=1853,
        repeated_groups=len(groups),confirmed_same_exposure_native_pairs=368,
        by_shot=by_shot,edge_stats=edge_stats,groups=groups,
        alias_semantics=dict(representative="first exact native source PTS of confirmed group",
            optional_seed_hint="first original keyframe within group if present, otherwise representative; a later wrapper may instead use the representative original state",
            output="expand canonical solved 11D frame to every original source PTS; retain each per-frame Brown14 template",
            preserved="all original UV, per-frame factors/priors/targets, quality, Huber, native temporal pairs, source PTS and cuts",
            duplicate_keys="Within each factor use unique mapped keys; expand original key positions before raw callback and sum Jacobian blocks for aliases, including Ja+Jb of same-exposure native pairs.",
            factor_count="Keep every original 11199 factor, including same-exposure native pairs and repeated per-frame edge/prior/lens factors; do not drop or prelabel them constant.",
            risk="Repeated source pixels and priors become multiple factors on one state; no exposure covariance is modeled, so evidence remains correlated and apparent precision may be overstated."),
        no_graph_built=True,no_solver_started=True,no_uv_or_prior_changed=True,
        final_holdout_read=False,not_accuracy_acceptance=True)
    return pts,canonical,seed_pts,audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(TMP.resolve()),
            "Alias preflight output must stay under programme TMP")
    require(not args.output.exists(), "Alias preflight output exists; do not overwrite")
    pts,canonical,seed,audit=load_confirmed_aliases()
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"source_pts_aliases.npz",
        pts=pts,canonical_pts=canonical,seed_source_pts=seed,
        source_sha256=SOURCE_SHA256,model_sha256=MODEL_SHA256)
    audit["aliases_npz"] = dict(path=str((args.output/"source_pts_aliases.npz").resolve()),
                                sha256=digest(args.output/"source_pts_aliases.npz"))
    path=args.output/"phase_00.json"
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(dict(path=str(path),sha256=digest(path),
        alias_packet_sha256=audit["aliases_npz"]["sha256"],
        groups=len(audit["groups"]),frame_variables=1851,
        no_solver_started=True)),flush=True)


if __name__=="__main__":
    main()
