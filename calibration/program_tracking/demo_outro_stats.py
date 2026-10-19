"""Traceable statistics for the standalone V3 demo ending.

These are presentation-model estimates, not independently surveyed metrics.
Source native PTS remain the sole clock. Existing models are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.program_tracking.demo_layers import IceLayers

PROGRAMME_START_SOURCE_SECONDS = 22.12
PROGRAMME_END_SOURCE_SECONDS = 188.12
PROGRAMME_SECONDS = 166.0
TIME_BASE_DENOMINATOR = 12800
# Visual action intervals, separate from support-point replacement windows.
# The flying entry belongs to the first spin; the second spin continues beyond
# its contact replacement window, until its visible step-out at about 166.5 s.
SPIN_SOURCE_INTERVALS = (
    (81.40, 92.80, "旋转 1"),
    (156.40, 166.60, "旋转 2"),
    (171.90, 185.00, "旋转 3"),
)
COVERAGE_THRESHOLD = .20
COVERAGE_SIGMA_M = 1.5
COVERAGE_ROUTE_RADIUS_M = 1.0
DATA = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp")
OUT = DATA / "results/demo_outro_stats_20261016"


def programme_indices(sample):
    first = int(round(PROGRAMME_START_SOURCE_SECONDS*TIME_BASE_DENOMINATOR))
    last = int(round(PROGRAMME_END_SOURCE_SECONDS*TIME_BASE_DENOMINATOR))
    indices = np.flatnonzero((sample.pts >= first) & (sample.pts <= last))
    expected = np.arange(first, last+256, 256, dtype=np.int64)
    if not np.array_equal(sample.pts[indices], expected):
        raise ValueError("Statistics require every official native-50 PTS in the timer window")
    return indices


def summary_layers(sample):
    """The same final style, restricted to the reported 166-second window."""
    indices = programme_indices(sample)
    return IceLayers(sample.scene, float(sample.packet["unit"]), sample.xy[indices],
                     sample.times[indices], sigma_m=COVERAGE_SIGMA_M,
                     coverage_radius_m=COVERAGE_ROUTE_RADIUS_M,
                     heat_palette="champagne_gold", particle_style="particles",
                     line_width_m=.09, fade_seconds=2.5, old_alpha=.22,
                     heat_update_anchor=0)


def coverage_field(layers):
    """Numeric pre-colour field, exactly matching IceLayers' existing algorithm."""
    occupancy = np.zeros(layers._shape, np.uint8)
    layers._stamp_prefix(occupancy, 0, len(layers.times_s)-1,
                         layers._coverage_thickness, 255)
    sigma = layers.sigma_m*layers._pixels_per_metre
    field = cv2.GaussianBlur((occupancy > 0).astype(np.float32), (0, 0),
                             sigmaX=sigma, sigmaY=sigma,
                             borderType=cv2.BORDER_CONSTANT)
    return np.clip(field, 0., 1.)


def coverage_percentage(field, layers, threshold=COVERAGE_THRESHOLD):
    # Fractional AA boundary weights avoid counting background or whole edge pixels.
    weights = layers._clip.astype(np.float64)
    return float(100*np.sum(weights*(field >= threshold))/np.sum(weights))


def _single_pass_half_width(threshold):
    # Continuous isolated infinitely long 2-metre-wide strip, Gaussian sigma1.5.
    def response(distance):
        z = math.sqrt(2)*COVERAGE_SIGMA_M
        return .5*(math.erf((COVERAGE_ROUTE_RADIUS_M-distance)/z)
                   - math.erf((-COVERAGE_ROUTE_RADIUS_M-distance)/z))
    lo, hi = 0., 10.
    for _ in range(60):
        mid = (lo+hi)/2
        if response(mid) >= threshold:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2, response(0.)


def build_stats(sample=None):
    if sample is None:
        from calibration.program_tracking.demo_v3_sample import V3Sample
        sample = V3Sample()
    indices = programme_indices(sample)
    source_times = sample.times[indices]
    times = source_times-PROGRAMME_START_SOURCE_SECONDS
    speed = np.asarray(sample.speed[indices], dtype=float)
    if not np.isfinite(speed).all():
        raise ValueError("A missing speed may not be silently converted to zero")
    excluded = np.zeros(len(times), bool)
    excluded_segments = np.zeros(len(times)-1, bool)
    mid_source = (source_times[:-1]+source_times[1:])/2
    intervals = []
    for start, end, label in SPIN_SOURCE_INTERVALS:
        excluded |= (source_times >= start) & (source_times < end)
        excluded_segments |= (mid_source >= start) & (mid_source < end)
        intervals.append(dict(start_seconds=start-PROGRAMME_START_SOURCE_SECONDS,
                              end_seconds=end-PROGRAMME_START_SOURCE_SECONDS,
                              source_start_seconds=start, source_end_seconds=end,
                              label=label, boundary_uncertainty_seconds=.10,
                              basis="official source visual spin entry/step-out review"))
    dt = np.diff(times)
    segment_distance = (speed[:-1]+speed[1:])*.5/3.6*dt
    active = ~excluded_segments
    active_seconds = float(np.sum(dt[active]))
    distance = float(np.sum(segment_distance[active]))
    average = distance/active_seconds*3.6
    peak_index = int(np.argmax(np.where(~excluded, speed, -np.inf)))
    layers = summary_layers(sample)
    field = coverage_field(layers)
    coverage = coverage_percentage(field, layers)
    bounds = layers.bounds_xy
    pixel_area_m2 = ((bounds[2]-bounds[0])*layers.unit/(layers.texture_wh[0]-1)
                    *(bounds[3]-bounds[1])*layers.unit/(layers.texture_wh[1]-1))
    rink_area_m2 = float(np.sum(layers._clip.astype(np.float64))*pixel_area_m2)
    half_width, centre_value = _single_pass_half_width(COVERAGE_THRESHOLD)
    full_field = coverage_field(sample.layers)
    full_coverage = coverage_percentage(full_field, sample.layers)
    record = {
        "version": "v3_outro_stats_v1",
        "athlete": {"name_zh": "冈田芽衣", "name_latin": "Mei OKADA",
                    "nation_code": "JPN", "nation_zh": "日本"},
        "event": {"display_name": "2026 ISU 青年大奖赛安卡拉站",
                  "official_name": "ISU Figure Skating Junior Grand Prix Ankara 2026",
                  "discipline": "女子短节目",
                  "official_url": "https://www.isu-skating.com/en/figure-skating/events/eventdetail/isu-jgp-ankara-2026/?tab=overview_16055"},
        "programme_seconds": PROGRAMME_SECONDS,
        "programme_time_display": "2:46",
        "active_seconds": active_seconds,
        "excluded_seconds": PROGRAMME_SECONDS-active_seconds,
        "active_distance_m": distance,
        "average_speed_kmh": average,
        "max_speed_kmh": float(speed[peak_index]),
        "max_speed_programme_seconds": float(times[peak_index]),
        "coverage_percent": coverage,
        "coverage": {
            "sigma_m": COVERAGE_SIGMA_M, "threshold": COVERAGE_THRESHOLD,
            "route_radius_m": COVERAGE_ROUTE_RADIUS_M,
            "definition": "union of 1m-radius route neighbourhood; Gaussian sigma1.5m; fraction of ice with numeric field >=0.20",
            "threshold_basis": "fixed visual-demo contour; isolated straight pass has centre response about0.495 and contour about2.17m from route centre; not an international standard",
            "isolated_pass_half_width_m": half_width,
            "isolated_pass_centre_value": centre_value,
            "ice_area_m2": rink_area_m2,
            "covered_area_m2": coverage/100*rink_area_m2,
            "threshold_sensitivity_percent": {str(t): coverage_percentage(field,layers,t) for t in (.10,.15,.20,.25,.30,.40)},
            "full_source_coverage_percent": full_coverage,
            "full_source_minus_timer_percentage_points": full_coverage-coverage,
            "includes_spin_positions": True,
            "dwell_time_weighted": False,
            "texture_wh": list(layers.texture_wh),
            "boundary_rule": "fractional actual model ice-boundary mask, outside excluded",
        },
        "exclusion_intervals": intervals,
        "speed_curve": {"programme_time_seconds": times.tolist(), "speed_kmh": speed.tolist(),
                        "excluded_mask": excluded.tolist()},
        "clock": {
            "source_path": str(sample.packet["source_path"]),
            "native_fps": 50, "time_base": "1/12800",
            "first_pts": int(sample.pts[indices[0]]), "last_pts": int(sample.pts[indices[-1]]),
            "source_start_seconds": float(source_times[0]),
            "source_end_seconds": float(source_times[-1]),
            "duration_integration": "8300 native source intervals, endpoints included; exactly166s, no retiming",
            "timer_evidence": [
                "0:00 remains visible at source23.08; first0:01 at23.10",
                "2:45 remains visible at188.10; first2:46 at188.12",
                "166s window ends on first2:46 frame; start22.12 inferred by subtracting166s; initial0:01 phase differs by one native frame(20ms)",
                "0:00 graphic animates in around22.40; its first visible frame is not assumed to be programme zero",
            ],
            "start_is_inferred": True, "timer_phase_uncertainty_seconds": .02,
        },
        "method": {
            "distance": "trapezoid integral of V3 speed curve over active source intervals; spin distance and time both removed",
            "average": "active_distance_m / active_seconds *3.6",
            "speed_curve": "V3 existing speedometer: XY Gaussian sigma0.16s, native-clock gradient norm, speed Gaussian sigma0.26s",
            "peak": "maximum of same smoothed speed curve outside grey exclusion intervals",
            "stationary_dance_intervals": [],
            "stationary_dance_policy": "no additional confirmed stationary-dance interval; no automatic low-speed exclusion",
            "metres_per_scene_unit": float(sample.packet["unit"]),
            "scale_status": "inherited conditional model scale; not independently surveyed",
            "all_source_distance_m": float(np.sum((sample.speed[:-1]+sample.speed[1:])*.5/3.6*np.diff(sample.times))),
            "timer_window_all_distance_m": float(np.sum(segment_distance)),
            "uses_final_v3_display_trajectory": True,
            "trajectory_disclaimer": "includes previously documented inferred positions, spin contact proxies and local display corrections; no new model solve",
            "timer_rows_inferred_in_source_packet": int(np.sum(sample.packet["inferred"][indices])),
        },
        "footer": "开发中版本 · 所示数据为模型估算，仅供参考",
        "display_rounding": {"distance_m_decimals": 0,"speed_kmh_decimals": 1,"coverage_percent_decimals": 0},
    }
    if abs(record["active_seconds"]+record["excluded_seconds"]-166.)>1e-9:
        raise AssertionError("Active and excluded time do not sum to programme duration")
    return record


def load_stats(path=OUT/"stats.json"):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save_stats(record, output_dir=OUT, sample=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    (output_dir/"stats.json").write_text(text,encoding="utf8")
    curve = record["speed_curve"]
    np.savez_compressed(output_dir/"timeline.npz",
                        programme_time_seconds=curve["programme_time_seconds"],
                        speed_kmh=curve["speed_kmh"],
                        excluded_mask=curve["excluded_mask"])
    if sample is not None:
        layers=summary_layers(sample)
        field=coverage_field(layers)
        np.savez_compressed(output_dir/"coverage_field.npz",field=field,
                            ice_boundary_mask=layers._clip,
                            bounds_xy=layers.bounds_xy,unit=layers.unit,
                            threshold=COVERAGE_THRESHOLD,sigma_m=COVERAGE_SIGMA_M)
        layers.texture_prepared(len(layers.times_s)-1,summary=True)
        if not np.array_equal(field, np.clip(cv2.GaussianBlur(
                (layers._coverage>0).astype(np.float32),(0,0),
                sigmaX=layers.sigma_m*layers._pixels_per_metre,
                sigmaY=layers.sigma_m*layers._pixels_per_metre,
                borderType=cv2.BORDER_CONSTANT),0.,1.)):
            raise AssertionError("Reported numeric coverage differs from renderer heat field")
    return output_dir/"stats.json"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    from calibration.program_tracking.demo_v3_sample import V3Sample
    sample=V3Sample()
    record=build_stats(sample)
    path=save_stats(record,args.output_dir,sample)
    compact={k:v for k,v in record.items() if k not in ("speed_curve",)}
    print(json.dumps({"path":str(path),"stats":compact},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
