#!/usr/bin/env python3
"""Construct the exact pinned ExMR BodyNet+HandMR / FootMR-off ablation.

This is a module ablation, not per-frame model selection or rotation averaging.
The fixed ExMR implementation changes ankles in FootMR and elbows/wrists/fingers
in HandMR. HandMR's input shoulder/elbow/wrist FK has no ankle ancestors. Thus
the saved final hand output is unchanged if FootMR is skipped. No inference,
SMPL-X assets, CUDA, interpolation, or optimization is needed here.

Only validated raw camera-stage archives from the same call may be combined.
The resulting archive is a *prior*, not a validated trajectory. The downstream
RinkSpace importer must propagate these separate hand channels.

Adapted from exmr/build_exmr_hand_only.py in the supplied
RinkSpace_3d6f51f_Audit_Integration package. Its pinned stage-validation and
module-ablation construction are retained; this entry resolves registered Data
assets through AssetStore and refuses output-file replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.storage import AssetStore

EXMR_COMMIT = "eac72e47339b404c0443f5fb323412e2ce432410"
REQUIRED_KEYS = {
    "pts", "source_pts", "R_w2c", "K_rect", "body_pose", "betas",
    "global_orient", "left_hand_pose", "right_hand_pose", "transl",
}
HAND_KEYS = ("left_hand_pose", "right_hand_pose")
ANKLES = slice(18, 24)  # body_pose axes for native SMPL-X joints 7, 8
ELBOWS_WRISTS = slice(51, 63)  # native joints 18, 19, 20, 21


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identical(a: np.ndarray, b: np.ndarray, label: str) -> None:
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
            "Stage mismatch: " + label)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_protocol(record: dict) -> None:
    require(record.get("exmr_commit") == EXMR_COMMIT, "Unreviewed ExMR revision")
    for flag in ("world_branch_executed", "postprocess", "snap_to_ground",
                 "smooth_incam", "smooth_global", "orientation_normalization"):
        require(record.get(flag) is False, "Camera-stage protocol required: " + flag)
    require(record.get("one_continuous_model_call_per_input") is True,
            "Expected one continuous inference call per input")
    require(record.get("source_time_base") == [1, 12800], "Unexpected source time base")


def construct(pre: dict, post: dict, final: dict) -> tuple[dict, dict]:
    """Return copied candidate arrays and exact structural-check summary."""
    require(set(pre) == set(post) == set(final), "Stage archive fields differ")
    require(REQUIRED_KEYS <= set(pre), "Missing required raw-camera-stage field")
    for name in pre:
        require(pre[name].shape == post[name].shape == final[name].shape and
                pre[name].dtype == post[name].dtype == final[name].dtype,
                "Stage shape/dtype mismatch: " + name)
    n = len(pre["pts"])
    shapes = {
        "pts": (n,), "source_pts": (n,), "R_w2c": (n, 3, 3),
        "K_rect": (n, 3, 3), "body_pose": (n, 63), "betas": (n, 10),
        "global_orient": (n, 3), "left_hand_pose": (n, 45),
        "right_hand_pose": (n, 45), "transl": (n, 3),
    }
    require(n > 1, "A continuous sequence is required")
    for label, stage in (("prefoot", pre), ("postfoot", post), ("final", final)):
        for name, shape in shapes.items():
            value = stage[name]
            require(value.shape == shape, f"{label}/{name}: wrong shape")
            require(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(),
                    f"{label}/{name}: finite numeric values required")
        require(np.issubdtype(stage["pts"].dtype, np.integer), "PTS must be integers")
        identical(stage["pts"], stage["source_pts"], label + " PTS")
        require(np.all(np.diff(stage["pts"]) > 0), "PTS must be strictly increasing")
    # Anything except the specifically documented pose fields must be identical.
    for name in sorted(set(pre) - {"body_pose", *HAND_KEYS}):
        identical(pre[name], post[name], "prefoot/postfoot " + name)
        identical(post[name], final[name], "postfoot/final " + name)
    # Raw stage isolation is tested against the arrays, not trusted from a log.
    no_ankles = np.ones(63, dtype=bool)
    no_ankles[ANKLES] = False
    identical(pre["body_pose"][:, no_ankles], post["body_pose"][:, no_ankles],
              "FootMR changed a joint outside native 7,8")
    for name in HAND_KEYS:
        identical(pre[name], post[name], "FootMR changed " + name)
    no_arms = np.ones(63, dtype=bool)
    no_arms[ELBOWS_WRISTS] = False
    identical(post["body_pose"][:, no_arms], final["body_pose"][:, no_arms],
              "HandMR changed a body joint outside native 18,19,20,21")
    candidate = {key: value.copy() for key, value in pre.items()}
    candidate["body_pose"][:, ELBOWS_WRISTS] = final["body_pose"][:, ELBOWS_WRISTS]
    for name in HAND_KEYS:
        candidate[name] = final[name].copy()
    # Protect root/camera/shape/other joints and retain the hand parameters exactly.
    identical(candidate["body_pose"][:, ANKLES], pre["body_pose"][:, ANKLES],
              "Candidate ankles must equal BodyNet")
    identical(candidate["body_pose"][:, no_ankles], final["body_pose"][:, no_ankles],
              "Candidate non-ankle body poses must equal final")
    for name in HAND_KEYS:
        identical(candidate[name], final[name], "Candidate must retain " + name)
    changed = np.any(pre["body_pose"].reshape(n, 21, 3) !=
                     candidate["body_pose"].reshape(n, 21, 3), axis=(0, 2))
    return candidate, {
        "frames": n,
        "source_pts": [int(pre["pts"][0]), int(pre["pts"][-1])],
        "source_time_base": [1, 12800],
        "candidate": "ExMR BodyNet + HandMR; FootMR disabled",
        "changed_body_joint_ids_relative_to_prefoot": (np.flatnonzero(changed) + 1).tolist(),
        "unchanged": ["root translation", "root rotation", "shape", "cameras", "PTS",
                      "native body joints 1..17 including both ankles"],
        "hand_parameters": "left/right 45D flat-hand-mean axis-angle copied exactly from final",
        "not_per_frame_selection": True,
        "not_a_validated_fit": True,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prefoot", "postfoot", "final", "inference-json", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    store = AssetStore(args.project, data_root=args.data_root)
    require(store.configured, "A registered Data root is required")
    # Relative CLI assets use the registered Data root, absolute registered
    # project/Data assets retain their identity. No machine paths are guessed.
    for name in ("prefoot", "postfoot", "final", "inference_json"):
        setattr(args, name, store.resolve(getattr(args, name), base=store.data, must_exist=True))
    args.output = guard_project_write(
        store.resolve(args.output, base=store.data), project_root=store.project)
    require(args.output.is_relative_to(store.data), "Output must stay in registered Data")
    report_path = args.output.with_suffix(".provenance.json")
    for path in (args.output, report_path):
        require(not path.exists(), "Refusing to overwrite: " + str(path))
    record = json.loads(args.inference_json.read_text(encoding="utf-8"))
    validate_protocol(record)
    sources = {name: getattr(args, name) for name in ("prefoot", "postfoot", "final")}
    arrays, report = construct(*(load(sources[key]) for key in sources))
    require(report["frames"] == record.get("frames"), "Inference/archive frame count mismatch")
    require(report["source_pts"] == record.get("source_pts"), "Inference/archive interval mismatch")
    report.update({
        "exmr_commit": EXMR_COMMIT,
        "architecture_source": "https://github.com/twehrbein/ExMR/blob/" + EXMR_COMMIT + "/exmr/model/exmr.py",
        "inputs": {name: {"name": path.name, "sha256": digest(path)} for name, path in sources.items()},
        "inference_record_sha256": digest(args.inference_json),
        "scope": "Construct prior only; rebuild geometry and re-evaluate before use in fitting/player",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures no source/archive is overwritten, even by a race.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["output_sha256"] = digest(args.output)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "report": str(report_path),
                      "frames": report["frames"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
