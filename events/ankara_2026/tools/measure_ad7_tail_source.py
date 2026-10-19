"""Read independently inspected Ad7 tail borders with the public RGB helper.

The source windows below come from actual native images. Existing Ad7 held
frames 230507008/230507264 remain outside this new calibration packet.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

BASE = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/cut6_clear_source_constraints")
SHA = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return value


def main():
    output = BASE / "ad7_tail_source_20261007"
    target = output / "Ad7_230506752_native_tail_source_packet_v2.json"
    if target.exists():
        raise FileExistsError(target)
    prior = json.loads((BASE / "late_230501120_qualified_Ad7_external_source_packet.json").read_text(encoding="utf8"))
    fit, controls, rejected, single_point_diagnostics = [], [], [], []
    for pts, left in ((230506752, 1490), (230508032, 1820)):
        raw = output / f"raw_native_{pts}.png"
        rgb = cv2.cvtColor(cv2.imread(str(raw)), cv2.COLOR_BGR2RGB)
        binding = dict(source_sha256=SHA, source_pts=pts, time_base="1/15360",
                       shot_id="223403520:R0402_039", native_hw=[1080, 1920],
                       image_domain="native_brown_pixels")
        xs = [1600, 1700] if pts == 230506752 else [1850, 1880]
        cases = (
            ("left_vertical", [[left, y] for y in (360, 430, 510)], [1, 0], [-70, 55], "light_to_dark"),
            ("top_edge", [[x, 315] for x in xs], [0, 1], [-20, 25], "light_to_dark"),
            ("bottom_edge", [[x, 552] for x in xs], [0, 1], [-25, 25], "dark_to_light"),
        )
        for edge, points, normal, interval, transition in cases:
            reading = read_source_material_transition_profiles(
                rgb, binding=binding, search_points_uv=points,
                normals=[normal] * len(points), search_intervals_px=[interval] * len(points),
                transition=transition, keep_profiles=True)
            accepted = [row for row in reading["readings"]
                        if row["disposition"] == "measured_material_transition"]
            rejected.extend(dict(pts=pts, edge=edge, **serial(row)) for row in reading["readings"]
                            if row["disposition"] != "measured_material_transition")
            if not accepted:
                continue
            if pts == 230508032 and edge == "bottom_edge":
                single_point_diagnostics.extend(dict(pts=pts, edge=edge, **serial(q)) for q in accepted)
                continue  # Four HELD controls: two left + two top; no extra target.
            authority = dict(prior["rows"][0]["physical_identity_authority"])
            authority["current_source_context"] = "Same continuous secondary shot and ISU blue print; actual native frames inspected before source windows were declared"
            row = dict(id=f"actual_tail_{pts}_Ad7_{edge}_source",
                pts=pts, actual_pts=pts,
                curve=dict(kind="advertisement_boundary", extent_scope="external_border",
                           advertisement_number=7, edge=edge),
                geometry_role="printed_colour_block_external_border",
                uv=[q["actual_source_uv"].tolist() for q in accepted],
                sigma_px=[max(4., float(np.ptp(q["observed_normal_interval_px"])) / 2.) for q in accepted],
                sigma_semantics="Existing 4px board working floor or half measured optical support width; empirical blur is not precise boundary accuracy",
                normal=[q["source_search_normal"].tolist() for q in accepted],
                normal_interval_px=[q["observed_normal_interval_px"].tolist() for q in accepted],
                physical_identity_authority=authority,
                source_RGB_actual_measurement=True,
                model_projection_used_to_generate_or_select_uv=False,
                source_raw_frame=str(raw), source_file_sha256=digest(raw),
                source_native_RGB_sha256=reading["RGB_sha256"],
                measured_RGB_sources=serial(accepted))
            (fit if pts == 230506752 else controls).append(row)
    packet = dict(schema="rinkspace.source-qualified-native-Ad7-tail-anchor.v1",
        source_video_sha256=SHA, time_base="1/15360", rows=fit, evaluation_rows=controls,
        original_230507008_230507264_HELD_retained_and_not_reclassified=True,
        shared_print_geometry_or_camera_not_used_for_UV=True,
        source_windows_manual_original_RGB_inspection=True,
        source_clock="Each new PTS is its own representative in existing authenticated original alias clock",
        public_source_reader=dict(path="src/rinkspace/vision_source_shapes.py", sha256=digest(ROOT / "src/rinkspace/vision_source_shapes.py")),
        rejected_source_queries=rejected, accuracy_accepted=False)
    packet["unused_bottom_source_diagnostics"] = single_point_diagnostics
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(dict(path=str(target), fit_points=sum(len(r["uv"]) for r in fit),
                         held_points=sum(len(r["uv"]) for r in controls),
                         rejected_queries=len(rejected), bytes=target.stat().st_size)))


if __name__ == "__main__":
    main()
