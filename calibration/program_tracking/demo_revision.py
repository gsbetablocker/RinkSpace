"""Small presentation-only corrections requested after the full demo review.

The native source, body estimates, camera model and support-blade spin records
remain unchanged. The selected jump route is a straight interpolated display
projection; it is not an observation of ground contact or a recovered 3D flight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.demo_source import check_output_path

DATA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp")
BASELINE = DATA / "results/demo_full_20261016/contact_packet.npz"
SOURCE = DATA / "results/demo_last_shot_20261016/source_packet.npz"
OUTPUT = DATA / "results/demo_full_20261016/v2"
INTRO_FRAMES = 410
FIRST_PTS = 256256
PTS_STEP = 256
TIME_BASE = 12800


def _load(file):
    with np.load(file, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _sha(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bitexact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def build_packet(output=OUTPUT, baseline=BASELINE, source=SOURCE,
                 start_frame=1812, end_frame=1848):
    """Build an independent full-range display packet, using 1-based demo frames."""
    output = Path(output)
    packet_file = check_output_path(output / "trajectory_packet.npz")
    receipt_file = check_output_path(output / "trajectory_packet.json")
    baseline, source = Path(baseline), Path(source)
    contact, native = _load(baseline), _load(source)
    pts = contact["pts"]
    n = len(pts)
    if not np.array_equal(pts, native["pts"]):
        raise ValueError("Contact and official native source clocks differ")
    if not np.array_equal(pts, FIRST_PTS + np.arange(n) * PTS_STEP):
        raise ValueError("Expected continuous official native50 PTS")
    lo, hi = int(start_frame) - 1 - INTRO_FRAMES, int(end_frame) - 1 - INTRO_FRAMES
    if not 0 <= lo < hi < n:
        raise ValueError("The selected 1-based demo frames must be inside the source")
    if np.any(contact["spin_id"][lo:hi + 1]):
        raise ValueError("A jump revision must not overwrite the reviewed spin records")
    if native["shot_index"][lo] != native["shot_index"][hi]:
        raise ValueError("The jump segment cannot cross a broadcast cut")
    old_xy = contact["xy_m"].copy()
    xy = old_xy.copy()
    weight = (pts[lo:hi + 1] - pts[lo]) / float(pts[hi] - pts[lo])
    line = old_xy[lo] + weight[:, None] * (old_xy[hi] - old_xy[lo])
    # Keep both anchors bit exact, including any existing model provenance.
    xy[lo + 1:hi] = line[1:-1]
    revised = {key: value.copy() for key, value in contact.items()}
    revised["xy_m"] = xy
    revised["presentation_revision_baseline_xy_m"] = old_xy
    selected = np.zeros(n, bool)
    selected[lo:hi + 1] = True
    changed = np.zeros(n, bool)
    changed[lo + 1:hi] = np.any(xy[lo + 1:hi] != old_xy[lo + 1:hi], axis=1)
    revised["presentation_linear_segment"] = selected
    revised["presentation_linearized_xy"] = changed
    # These labels are confined to this independently reviewed jump. Frames
    # 1817/1847 are motion-blurred boundaries, not asserted contact observations.
    airborne = np.zeros(n, bool)
    uncertain = np.zeros(n, bool)
    demo_frames = np.arange(n) + INTRO_FRAMES + 1
    airborne[(demo_frames >= 1818) & (demo_frames <= 1846)] = True
    uncertain[np.isin(demo_frames, (1817, 1847))] = True
    revised["presentation_airborne_observed"] = airborne
    revised["presentation_airborne_boundary_uncertain"] = uncertain
    revised["presentation_revision_method"] = selected.astype(np.uint8)
    revised["presentation_revision_baseline_path"] = np.asarray(str(baseline))
    revised["presentation_revision_baseline_sha256"] = np.asarray(_sha(baseline))
    unchanged_fields = [key for key in contact if key != "xy_m"]
    if not all(_bitexact(contact[key], revised[key]) for key in unchanged_fields):
        raise AssertionError("An existing contact/source record was changed")
    if not _bitexact(xy[~selected], old_xy[~selected]):
        raise AssertionError("The revision escaped the selected display window")
    spin = contact["spin_id"] != 0
    if not _bitexact(xy[spin], old_xy[spin]):
        raise AssertionError("A reviewed spin route was changed")
    if not np.isfinite(xy).all():
        raise AssertionError("The revised route must remain finite")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(packet_file, **revised)
    saved = _load(packet_file)
    if not all(_bitexact(revised[key], saved[key]) for key in revised):
        raise AssertionError("Saved display packet failed byte-exact reload")
    distance = float(np.linalg.norm(xy[hi] - xy[lo]))
    duration = float((pts[hi] - pts[lo]) / TIME_BASE)
    max_deviation = float(np.max(np.linalg.norm(old_xy[lo:hi + 1] - line, axis=1)))
    receipt = {
        "schema": "rinkspace.presentation-jump-linear-revision.v1",
        "packet": {"path": str(packet_file), "sha256": _sha(packet_file)},
        "baseline": {"path": str(baseline), "sha256": _sha(baseline)},
        "source_packet": {"path": str(source), "sha256": _sha(source)},
        "request": {
            "demo_frame_numbering": "1-based, including the 410-frame opening",
            "approximate_start_frame": 1812, "approximate_landing_frame": 1846,
            "method": "Straight endpoint-to-endpoint XY interpolation at native PTS"
        },
        "window": {
            "demo_start_frame": int(start_frame), "demo_end_frame": int(end_frame),
            "first_pts": int(pts[lo]), "last_pts_inclusive": int(pts[hi]),
            "source_start_seconds": float(pts[lo] / TIME_BASE),
            "source_end_seconds": float(pts[hi] / TIME_BASE),
            "inclusive_frames": hi - lo + 1, "changed_frames": int(changed.sum()),
            "start_xy": old_xy[lo].tolist(), "end_xy": old_xy[hi].tolist(),
            "duration_seconds": duration,
            "straight_distance_conditional_m": distance,
            "linear_speed_conditional_mps": distance / duration,
            "old_max_displacement_from_new_line_conditional_m": max_deviation,
            "anchor_selection": (
                "User's takeoff-side frame retained; first clearly visible landing "
                "at demo frame1848/source48.76s, two frames after approximate1846"
            ),
        },
        "flight_provenance": {
            "observed_airborne_demo_frames_inclusive": [1818, 1846],
            "motion_blurred_boundary_demo_frames": [1817, 1847],
            "first_clear_landing_demo_frame": 1848,
            "ground_contact_invented": False,
            "revised_xy_is": "Presentation-only ice-plane route, not blade contact",
        },
        "invariants": {
            "all_existing_fields_except_xy_m_bitexact": True,
            "xy_outside_selected_window_bitexact": True,
            "anchors_bitexact": True,
            "all_three_reviewed_spin_routes_and_records_bitexact": True,
            "native_pts_and_source_mapping_bitexact": True,
            "source_body_camera_and_model_unchanged": True,
            "saved_packet_reload_bitexact": True,
        },
        "existing_fields_preserved": unchanged_fields,
        "new_fields": [key for key in revised if key not in contact],
        "metric_scale_status": "Inherited conditional model scale; not surveyed metres",
        "evidence": [
            str(output / "jump_endpoint_review.jpg"),
            str(output / "jump_grounded_edges.jpg"),
        ],
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
    }
    receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--start-frame", type=int, default=1812)
    parser.add_argument("--end-frame", type=int, default=1848)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.output, args.baseline, args.source,
                                  args.start_frame, args.end_frame), ensure_ascii=False))


if __name__ == "__main__":
    main()
