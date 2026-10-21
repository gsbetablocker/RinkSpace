"""Read competing native-pixel band boundaries without a camera model.

This is a small inspectable reference candidate generator, not a replacement
source-qualification system. It consumes a source-bound uint8 RGB image, an
optional exact exclusion mask and a navigation ROI. A confirmed material policy
can select the blue/white/yellow/ice ordering; it does not confer physical
instance identity, subpixel truth, covariance, fit admission, or generalization.

The audit examples are already-development Ankara images. Output positions are
optical half-height proposals with 10--90% transition intervals, not precise
geometric contact truths. No historical UV or model projection is an input to
the measurement function. Unknown and competing candidates remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ProfileConfig:
    horizontal_halfwidth_px: int = 2
    colour_floor_8bit: float = 4.0
    colour_noise_multiple: float = 3.0
    neutral_chroma_max_8bit: float = 24.0
    minimum_luminance_8bit: float = 64.0
    support_shoulder_px: int = 8
    min_band_samples: int = 2
    version: str = "native_material_band_candidates_reference_v1"


ORDERED_POLICY = "blue_cap_white_body_yellow_strip_ice_lower_interface"


def _runs(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _noise(signal):
    second = np.diff(signal, n=2)
    return float(1.4826 * np.median(np.abs(second - np.median(second)))) if len(second) else 0.0


def _transition(signal, a, b, polarity, base, level):
    """Return observed ordered intensity crossings; preserve broad intervals."""
    if b <= a or level <= 0:
        return None
    values = (signal[a:b + 1] - base) / level
    if polarity < 0:
        values = 1.0 - values
    target = []
    for quantile in (0.1, 0.5, 0.9):
        crossings = np.flatnonzero((values[:-1] <= quantile) & (values[1:] >= quantile))
        if len(crossings) != 1:
            return None
        i = int(crossings[0])
        den = float(values[i + 1] - values[i])
        target.append(a + i + (quantile - float(values[i])) / den)
    return target


def _bands(signal, floor, config):
    rows = []
    for a, b in _runs(signal > floor):
        if b - a < config.min_band_samples:
            continue
        if a == 0 or b == len(signal):
            rows.append(dict(status="truncated_material_band", sample_range=[int(a), int(b)]))
            continue
        lo = max(0, a - config.support_shoulder_px)
        hi = min(len(signal) - 1, b + config.support_shoulder_px)
        peak = a + int(np.argmax(signal[a:b]))
        before = float(np.median(signal[lo:a]))
        after = float(np.median(signal[b:hi + 1]))
        material = float(signal[peak])
        entry = _transition(signal, lo, peak, 1, before, material - before)
        exit = _transition(signal, peak, hi, -1, after, material - after)
        if entry is None or exit is None:
            rows.append(dict(status="competing_or_incomplete_optical_transition", sample_range=[int(a), int(b)]))
            continue
        rows.append(dict(status="paired_current_colour_transitions", sample_range=[int(a), int(b)],
            entry_q10_q50_q90=entry, exit_q10_q50_q90=exit,
            optical_band_width_px=float(exit[1] - entry[1]),
            peak_colour_8bit=material, before_colour_8bit=before, after_colour_8bit=after))
    return rows


def read_candidates(rgb, *, source, columns, vertical_roi, policy=None, mask=None, config=ProfileConfig()):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("A complete native uint8 RGB frame is required")
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if source.get("decoded_native_RGB_sha256") != digest:
        raise ValueError("Source binding must name the exact decoded RGB bytes")
    if not isinstance(source.get("native_pts"), int) or not source.get("time_base") or not source.get("video_sha256"):
        raise ValueError("Original integer PTS, time base and video SHA are required")
    if policy not in (None, ORDERED_POLICY):
        raise ValueError("Unknown material policy")
    height, width = image.shape[:2]
    y0, y1 = map(int, vertical_roi)
    if not 0 <= y0 < y1 <= height:
        raise ValueError("Navigation ROI is outside the native image")
    if mask is not None and (np.asarray(mask).shape != (height, width) or np.asarray(mask).dtype != np.bool_):
        raise ValueError("Optional exclusions must be a same-frame native boolean mask")
    output = []
    for column in columns:
        x = int(column)
        half = config.horizontal_halfwidth_px
        if x - half < 0 or x + half >= width:
            output.append(dict(column=x, status="profile_outside_native_image", candidates=[]))
            continue
        profile = np.median(image[y0:y1, x - half:x + half + 1].astype(float), axis=1)
        yellow = np.minimum(profile[:, 0], profile[:, 1]) - profile[:, 2]
        blue = profile[:, 2] - np.maximum(profile[:, 0], profile[:, 1])
        yellow_floor = max(config.colour_floor_8bit, config.colour_noise_multiple * _noise(yellow))
        blue_floor = max(config.colour_floor_8bit, config.colour_noise_multiple * _noise(blue))
        warm = _bands(yellow, yellow_floor, config)
        caps = _bands(blue, blue_floor, config)
        accepted = []
        for index, band in enumerate(warm):
            candidate = dict(band, candidate_id=f"column{x}:yellow{index}",
                physical_instance_status="unknown", geometric_position_precision="unverified")
            if band["status"] == "paired_current_colour_transitions":
                candidate.update(upper_optical_uv=[float(x), y0 + band["entry_q10_q50_q90"][1]],
                    lower_optical_uv=[float(x), y0 + band["exit_q10_q50_q90"][1]],
                    lower_optical_interval_uv=[[float(x), y0 + q] for q in (band["exit_q10_q50_q90"][0], band["exit_q10_q50_q90"][2])])
            if policy is None:
                candidate["selection_status"] = "material_role_unknown_without_policy"
                accepted.append(candidate)
                continue
            if band["status"] != "paired_current_colour_transitions":
                candidate["selection_status"] = "unknown_optical_support"
                accepted.append(candidate)
                continue
            start, end = band["sample_range"]
            cap_options = []
            for cap in caps:
                if cap["status"] != "paired_current_colour_transitions":
                    continue
                cap_end = cap["sample_range"][1]
                if cap_end + 2 >= start:
                    continue
                cap_options.append(cap)
            reasons = []
            # Require a directly observed preceding cap; do not manufacture one
            # from a model or treat a bright wall as a blue material.
            if len(cap_options) != 1:
                reasons.append("missing_or_competing_blue_cap")
            shoulder = min(config.support_shoulder_px, start, len(profile) - end)
            if shoulder < 2:
                reasons.append("missing_adjacent_current_material")
            else:
                above = profile[start - shoulder:start]
                below = profile[end:end + shoulder]
                for name, values in (("board_body", above), ("iceward_region", below)):
                    if float(np.median(values.mean(axis=1))) < config.minimum_luminance_8bit:
                        reasons.append(f"dark_or_unresolved_{name}")
                    if float(np.median(np.ptp(values, axis=1))) > config.neutral_chroma_max_8bit:
                        reasons.append(f"competing_coloured_{name}")
            if mask is None:
                reasons.append("visibility_mask_not_supplied")
            elif np.asarray(mask)[max(y0, y0 + start - shoulder):min(y1, y0 + end + shoulder), x - half:x + half + 1].any():
                reasons.append("candidate_support_intersects_exclusion")
            candidate["policy_reasons"] = reasons
            candidate["selection_status"] = "ordered_material_candidate" if not reasons else "unknown_or_rejected_by_current_material_evidence"
            accepted.append(candidate)
        candidates = [r for r in accepted if r.get("selection_status") == "ordered_material_candidate"]
        output.append(dict(column=x, status=("unique_current_optical_candidate" if len(candidates) == 1 else "multiple_current_optical_candidates" if candidates else "unknown"),
            selected_candidate_id=candidates[0]["candidate_id"] if len(candidates) == 1 else None,
            observed_yellow_threshold_8bit=yellow_floor, observed_blue_threshold_8bit=blue_floor,
            blue_bands=caps, candidates=accepted,
            profile_native_y=list(range(y0, y1)), profile_RGB=profile.tolist(),
            yellow_signal=yellow.tolist(), blue_signal=blue.tolist()))
    return dict(schema=config.version, source=dict(source), native_hw=[height, width],
        navigation_roi_y=[y0, y1], policy=policy, config=asdict(config), columns=output,
        original_source_pixels_modified=False, old_observation_modified=False,
        camera_or_model_used=False, fitting_admissible=False,
        precision_estimate_type="optical_half_height_and_q10_q90_interval_only",
        physical_identity_must_be_verified=True,
        limitations=["Example implementation is tuned only by the described signal model, not validated across events.",
                     "Colour casts, blur, reflections, nonvertical boundaries and competing cap/strip instances can leave unknown.",
                     "The supplied ROI and exclusion mask require their own source/semantic provenance.",
                     "An optical transition midpoint is not proof of an exact geometric contact pixel."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--mask", help="Same-source grayscale exclusion PNG; nonzero means excluded")
    parser.add_argument("--columns", nargs="+", type=int, required=True)
    parser.add_argument("--roi-y", nargs=2, type=int, required=True)
    parser.add_argument("--policy", choices=[ORDERED_POLICY])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = None if args.mask is None else np.asarray(Image.open(args.mask).convert("L")) > 0
    result = read_candidates(image, source=json.loads(Path(args.binding).read_text()), columns=args.columns,
        vertical_roi=args.roi_y, policy=args.policy, mask=mask)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Do not overwrite earlier candidate evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
