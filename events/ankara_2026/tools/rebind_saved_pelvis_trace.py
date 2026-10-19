"""Prepare the saved pelvis trace for an authenticated replacement grid camera.

Reuse the existing chart reexpression; preserve original source pixels and
conditional heights. This does not fit a body or claim a new world trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import numpy as np
from calibration.program_tracking.body_tracking.saved_root_trace_camera_rebind import rebind_saved_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-binding", type=Path, required=True,
                        help="Original full video's FROZEN_INPUTS_UNACCEPTED.json")
    parser.add_argument("--grid-state-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-source", type=Path)
    args = parser.parse_args()
    data_root = Path("D:/CodexData/RinkSpace_Data").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(data_root):
        raise ValueError("Saved trace arrays must stay in the event Data directory")
    report_path = output.with_suffix(".binding.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Preserve every previous trace reexpression")
    original = json.loads(args.original_binding.read_text(encoding="utf-8-sig"))
    identity = json.loads(args.grid_state_identity.read_text(encoding="utf-8-sig"))
    started = time.perf_counter()
    arrays, report = rebind_saved_trace(
        original["body"]["path"], original["camera"]["path"], identity["camera_path"],
        expected_trace_sha256=original["body"]["sha256"],
        expected_camera_sha256=identity["camera_sha256"],
        camera_q_sha256=identity["q_sha256"],
        grid_state_identity_path=args.grid_state_identity,
        scale_source=args.scale_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report.update(output=str(output), preparation_wall_seconds=time.perf_counter()-started,
                  output_bytes=output.stat().st_size)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "output", "preparation_wall_seconds", "output_bytes",
        "actual_exposures", "native_presentations",
        "preserved_original_model_P_pixel_max_error_native_px")}), flush=True)


if __name__ == "__main__":
    main()
