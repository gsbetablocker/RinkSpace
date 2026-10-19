"""Compare actual Ad7 projection errors with independently read blur supports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from calibration.solver_runtime import install
install()
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from calibration.program_tracking.render_result import SceneRenderer
from z190_source_camera_diagnostic import absolute_value_jac15, source_errors
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-identity", type=Path, required=True)
    parser.add_argument("--board-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-prefix", default="actual_tail_", help="Review another qualified native advertisement source through the same operator")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads(args.state_identity.read_text(encoding="utf8"))
    scene_path = args.state_identity.parent / "scene_UNACCEPTED.json"
    if digest(scene_path) != identity["scene_sha256"] or digest(identity["camera_path"]) != identity["camera_sha256"]:
        raise ValueError("The exact immutable scene and physical camera are required")
    renderer = SceneRenderer(json.loads(scene_path.read_text(encoding="utf8")))
    camera = np.load(identity["camera_path"], allow_pickle=False)
    packet = json.loads(args.board_source.read_text(encoding="utf8"))
    edge_names = dict(left_vertical="left_vertical_xyz", right_vertical="right_vertical_xyz",
                      top_edge="top_edge_xyz", bottom_edge="bottom_edge_xyz")
    metrics = []
    for row in packet["rows"]:
        if not row["id"].startswith(args.row_prefix):
            continue
        world = next(path["xyz"] for path in renderer.advertisement_paths
                     if path["number"] == row["curve"]["advertisement_number"]
                     and path["role"] == edge_names[row["curve"]["edge"]])
        derived = dict(row, world=world, sigma=np.asarray(row["sigma_px"]),
                       uv=np.asarray(row["uv"]), normal=np.asarray(row["normal"]),
                       normal_interval=np.asarray(row["normal_interval_px"]))
        cam = camera["camera25"][np.flatnonzero(camera["pts"] == row.get("actual_pts", row["pts"]))[0]]
        errors = source_errors(cam, [derived])[0]
        residual = absolute_value_jac15(cam, [derived], False)
        metrics.append(dict(**errors, weighted_residual=residual.tolist(),
                            weighted_cost=float(residual @ residual) / 2.,
                            optical_supports_px=row["normal_interval_px"]))
    report = dict(schema="rinkspace.native-board-source-centre-and-optical-support-comparison.v1",row_prefix=args.row_prefix,
        q_sha256=identity["q_sha256"], state_identity=str(args.state_identity),
        actual_camera_path=identity["camera_path"], scene_path=str(scene_path),
        scene_sha256=identity["scene_sha256"], camera_sha256=identity["camera_sha256"],
        board_source=str(args.board_source), board_source_sha256=digest(args.board_source),
        rows=metrics, source_centre_error_not_optical_support_error=True,
        within_blur_support_does_not_mean_visually_aligned=True,
        camera_geometry_or_pixels_not_changed=True, whole_video_accuracy_accepted=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(output=str(args.output), rows=len(metrics), q=identity["q_sha256"])))


if __name__ == "__main__":
    main()
