"""Verify the saved boot attachment and the programme's native 3D player."""
from argparse import ArgumentParser
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .inputs import PROGRAMME, PROJECT, ROOT, atomic_json, digest, emit, read
from calibration.program_tracking.body_tracking.programme_contract import required_clock


def check_recovery(binding, archive, receipt):
    record = read(binding)
    original = read(PROGRAMME / "programme_skate_chain_v1.json")
    if (record["schema"] != "rinkspace.skate-video-recovery-binding.v1"
            or record["candidate_id"] != original["candidate_id"]
            or record["camera_sha256"] != original["camera_sha256"]
            or record["parameter"]["sha256"] != original["parameter"]["sha256"]):
        raise ValueError("Recovery binding does not identify the registered candidate")
    for name in ("parameter", "source", "result", "body_observations"):
        if digest(record[name]["path"]) != record[name]["sha256"]:
            raise ValueError(f"Recovered {name} changed")
    member = "01_attached/whole_programme.npz"
    with ZipFile(archive) as package:
        index = json.loads(package.read("MANIFEST.json"))["members"]
        row, = (row for row in index if row["member"] == member)
        raw = package.read(member)
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Archived whole-programme arrays do not match the manifest")
    pts, shots, time_base = required_clock(read(PROGRAMME / "rebuild_scope.json"))
    fields = ("P_world_mid", "G_world_mid", "Q_world_mid", "com_world_mid",
              "skate_R_world_mid", "skate_origin_world_mid", "skate_reference_world_mid")
    with (np.load(BytesIO(raw), allow_pickle=False) as frozen,
          np.load(record["result"]["path"], allow_pickle=False) as recovered,
          np.load(record["body_observations"]["path"], allow_pickle=False) as observations):
        np.testing.assert_array_equal(frozen["parameter_sha256"], record["parameter"]["sha256"])
        np.testing.assert_array_equal(frozen["camera_sha256"], record["camera_sha256"])
        for packet in (frozen, recovered):
            np.testing.assert_array_equal(packet["source_pts"], pts)
            np.testing.assert_array_equal(packet["time_base"], time_base)
        np.testing.assert_array_equal(frozen["shot_ids"], shots)
        np.testing.assert_array_equal(recovered["source_shot_ids"], shots)
        np.testing.assert_array_equal(observations["pts"], pts)
        np.testing.assert_array_equal(recovered["source_body_model_joint_ids"], observations["model_joint_ids"])
        for name in fields:
            np.testing.assert_array_equal(recovered[name], frozen[name], err_msg=name)
        admitted = int(observations["use"].sum())
    report = dict(passed=True, source_exposures=len(pts), source_pts=[int(pts[0]), int(pts[-1])],
                  time_base=[int(value) for value in time_base], exact_admitted_body_rows=admitted,
                  exact_archive_fields=fields, archived_member_sha256=row["sha256"],
                  source_camera_sha256=record["camera_sha256"],
                  original_objective_resume_verified=False)
    atomic_json(receipt, report)
    emit("recovery_verified", **report)


def check_socket_clock(visual, checkpoint, receipt):
    from .body_skate_fit import Problem, quantiles
    from .deliver_body_skate_fit import saved_ref
    from calibration.program_tracking.body_tracking.skate_physics import fk_state

    problem = Problem(visual, receipt.parent)
    base = saved_ref(problem, checkpoint, apply_socket=False)
    final = saved_ref(problem, checkpoint)
    if not hasattr(final, "socket_shift"):
        raise ValueError("This checkpoint has no calibrated socket")
    max_rotation = max_origin = 0.
    correction = []
    for clock in np.array_split(problem.times, 66):
        query = clock.ravel()
        zero = np.zeros((len(query), 69))
        a, b = (fk_state(ref, ref.at(query), zero) for ref in (base, final))
        max_rotation = max(max_rotation, float(np.abs(a["R"] - b["R"]).max()))
        max_origin = max(max_origin, float(np.abs(a["O"] - b["O"]).max()) * final.binding.ratio)
        delta = (b["native_world"] - a["native_world"]).reshape(len(clock), 5, 22, 3)[:, 2]
        correction.append(delta)
    if max_rotation >= 1.e-8 or max_origin >= 1.e-8:
        raise ValueError("Calibrated attachment moved the fitted rigid skate")
    motion = np.concatenate(correction) * final.binding.ratio * 1000
    adjacent = problem.i.shots[:-1] == problem.i.shots[1:]
    change = np.linalg.norm(np.diff(motion, axis=0), axis=-1)
    rows = {}
    for key, joints in (("root", [0]), ("knees", [4, 5]), ("ankles", [7, 8])):
        source = np.flatnonzero(adjacent)
        j = source[np.argmax(change[source][:, joints].max(axis=1))]
        rows[key] = dict(corrective_frame_to_frame_mm=quantiles(change[source][:, joints]),
                         max_mm=float(change[j, joints].max()), max_source_seconds=float(problem.i.times[j + 1]))
    report = dict(source_exposures=len(problem.i.pts), motion_nodes=int(problem.times.size),
                  rigid_skate_orientation_max_abs_difference=max_rotation,
                  rigid_skate_origin_max_difference_m=max_origin, motion_correction=rows)
    atomic_json(receipt, report)
    emit("native_attachment_verified", **report)


def check_viewer(config, receipt):
    from rinkspace.viewer3d.app import ResultsPlayer
    from rinkspace.viewer3d.data import ResultCatalog

    catalog = ResultCatalog(PROJECT, ROOT, config)
    player = ResultsPlayer(catalog, restore=False, size=(1330, 750))
    path = catalog.result_path(catalog.variant(catalog.config["default_variant"]))
    with np.load(path, allow_pickle=False) as state:
        pts = state["source_pts"]
        com = state["com_world_mid"]
        ice = state["com_projection_world_mid"]
    scale = player.sequence.meta["uniform_unit_reexpression_ratio"]
    sample = [0, catalog.frame_for_seconds(89.32), catalog.frame_for_seconds(186.84), len(pts) - 1]

    class Verify:
        def __init__(self):
            self.index = 0
            self.seen = []
            self.passed = False

        def tick(self, p, when):
            try:
                if self.index == 0:
                    assert p.trajectory_reference == "com" and p.follow
                    assert all(p.visible[key] for key in ("com", "trajectory", "com_projection", "surface", "shoes"))
                    assert not p.visible["pelvis"] and not p.visible["pelvis_trajectory"]
                    np.testing.assert_array_equal(pts, catalog.pts)
                    np.testing.assert_allclose(p.tracking_points, com * scale, rtol=0, atol=1.e-10)
                    np.testing.assert_allclose(p.layers["com_projection"][0].lines.reshape(-1, 3),
                                               ice * scale, rtol=0, atol=2.e-5)
                    assert p.sequence.vertices.shape == (len(pts), 10475, 3) and len(p.layers["shoes"]) >= 2
                if self.index < len(sample):
                    j = sample[self.index]
                    p.seek(j)
                    assert p.frame == j
                    self.seen.append(int(pts[j]))
                    self.index += 1
                    return
                old, new = catalog.variants[1]["id"], catalog.config["default_variant"]
                p.load_variant(old)
                assert p.variant["id"] == old
                p.load_variant(new)
                assert p.variant["id"] == new
                p.seek(0)
                p.set_view("side")
                report = dict(passed=True, exposures=len(pts), source_pts=self.seen,
                              variant=new, gl=p.ctx.info.get("GL_RENDERER"), model_vertices=p.sequence.vertices.shape[1],
                              trajectory_reference="saved model COM", projection="world Z=0",
                              previous_variant_switch=True)
                atomic_json(receipt, report)
                emit("native_viewer_verified", **report)
                self.passed = True
                p.close()
            except Exception:
                p.close()
                raise

    player.verification = Verify()
    player.run()
    if not player.verification.passed:
        raise RuntimeError("OpenGL player did not finish the programme check")


def main():
    parser = ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    clock = modes.add_parser("clock")
    clock.add_argument("visual", type=Path)
    clock.add_argument("checkpoint", type=Path)
    clock.add_argument("receipt", type=Path)
    viewer = modes.add_parser("viewer")
    viewer.add_argument("config", type=Path)
    viewer.add_argument("receipt", type=Path)
    recovery = modes.add_parser("recovery")
    recovery.add_argument("binding", type=Path)
    recovery.add_argument("archive", type=Path)
    recovery.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if not args.receipt.resolve().is_relative_to(ROOT.resolve()) or args.receipt.exists():
        raise ValueError("Choose a new receipt in Data")
    if args.mode == "clock":
        check_socket_clock(args.visual, args.checkpoint, args.receipt)
    elif args.mode == "viewer":
        check_viewer(args.config, args.receipt)
    else:
        check_recovery(args.binding, args.archive, args.receipt)


if __name__ == "__main__":
    main()
