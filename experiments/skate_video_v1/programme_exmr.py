"""Run the pinned ExMR input and inference stages on all programme shots."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .inputs import PROGRAMME, PROJECT, ROOT, atomic_json, atomic_npz, read
from calibration.program_tracking.body_tracking import exmr_prior
from calibration.program_tracking.body_tracking.action_clip_input import camera25_from_snapshot
from calibration.program_tracking.body_tracking.camera_bridge import native_to_rectified, rectified_intrinsics
from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_contract import required_clock


def prepare(output, recovery):
    record = read(recovery)
    scope = read(PROGRAMME / "rebuild_scope.json")
    pts, shots, clock = required_clock(scope)
    packet = exmr_prior.load(Path(record["prepared"]) / "input.npz")
    np.testing.assert_array_equal(packet["pts"], pts)
    np.testing.assert_array_equal(packet["shot_ids"], shots)
    source = Path(record["source"]["path"])
    if exmr_prior.digest(source) != record["source"]["sha256"]:
        raise ValueError("Recovered source binding changed")
    camera = ProgrammeCamera(PROJECT, packet, shots, record_path=read(source)["programme_camera_candidate"]["record"])
    if camera.info["state_sha256"] != record["camera_sha256"]:
        raise ValueError("Programme source camera changed")
    rows = np.arange(len(pts))
    cuts = np.r_[0, np.flatnonzero(shots[1:] != shots[:-1]) + 1, len(pts)]
    selected = np.concatenate([rows[a:b:2] for a, b in zip(cuts[:-1], cuts[1:])])
    boxes = packet["source_bbox"].astype(float)
    good = np.isfinite(boxes).all(axis=1) & (np.diff(boxes[:, [0, 2]], axis=1)[:, 0] > 0)
    good &= boxes[:, 3] > boxes[:, 1]
    for a, b in zip(cuts[:-1], cuts[1:]):
        group = rows[a:b]
        valid = group[good[group]]
        if len(valid) < 2:
            raise ValueError(f"No source person ROI for {shots[a]}")
        missing = group[~good[group]]
        for component in range(4):
            boxes[missing, component] = np.interp(missing, valid, boxes[valid, component])
    camera25 = camera25_from_snapshot(camera.midpoint)[selected]
    intrinsics = np.stack([rectified_intrinsics(row) for row in camera25])
    coco = np.zeros((len(selected), 17, 3), np.float32)
    for j, i in enumerate(selected):
        valid = packet["source_qualified"][i] & np.isfinite(packet["source_uv"][i]).all(axis=1)
        xy, ok = native_to_rectified(packet["source_uv"][i, valid], camera25[j], K_rect=intrinsics[j])
        ids = np.flatnonzero(valid)[ok]
        coco[j, ids, :2] = xy[ok]
        coco[j, ids, 2] = 1
    output.mkdir(parents=True, exist_ok=True)
    bridge = output / "programme_camera_input.npz"
    if not bridge.exists():
        atomic_npz(bridge, source_pts=pts[selected], shot_ids=shots[selected], camera25=camera25,
                   K_rect=intrinsics, R_w2c=camera.midpoint["rotation"][selected], rect_coco17=coco,
                   roi_native_xyxy=boxes[selected], bbox_inferred=~good[selected],
                   source_mapping_missing=~good[selected], source_time_base=np.asarray(clock),
                   source_video_sha256=np.array(scope["source"]["sha256"]),
                   camera_sha256=np.array(record["camera_sha256"]))
    else:
        cached = exmr_prior.load(bridge)
        np.testing.assert_array_equal(cached["source_pts"], pts[selected])
        np.testing.assert_array_equal(cached["camera25"], camera25)
        np.testing.assert_array_equal(cached["roi_native_xyxy"], boxes[selected])
        np.testing.assert_array_equal(cached["source_video_sha256"], scope["source"]["sha256"])
    contract = output / "programme_source.json"
    atomic_json(contract, dict(source=dict(path=scope["source"]["registered_path"], sha256=scope["source"]["sha256"]),
                               source_pts=[int(pts[0]), int(pts[-1])], source_exposures=len(pts),
                               camera_sha256=record["camera_sha256"], time_base=list(map(int, clock)),
                               inferred_roi_pts=pts[selected][~good[selected]].tolist(),
                               source_network_fps=25, output_fitting_fps=50))
    exmr_prior.prepare(bridge, contract, output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "keypoints", "sam", "hands", "infer"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.recovery is None:
            parser.error("prepare requires --recovery")
        prepare(args.output, args.recovery)
    elif args.stage == "sam":
        exmr_prior.sam_features(ROOT, args.output, save_native=False)
    elif args.stage == "infer":
        exmr_prior.infer(ROOT, args.output, include_shared=False)
    else:
        {"keypoints": exmr_prior.keypoints, "hands": exmr_prior.hand_features}[args.stage](ROOT, args.output)


if __name__ == "__main__":
    main()
