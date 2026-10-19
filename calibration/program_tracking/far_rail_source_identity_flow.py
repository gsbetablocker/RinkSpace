"""Conservative source-image continuity between *reviewed UV subsets*.

Optical flow assigns a provisional identity to another frame's already
measured candidate UVs. It never creates UV observations, promotes an entire
connected component, or chooses the physically correct seed. Seed authority
must come from a separate human/source review manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from far_rail_source_components import source_exclusion


@dataclass(frozen=True)
class MatchPolicy:
    max_source_gap_px: float = 56.0
    max_target_gap_px: float = 56.0
    max_adjacent_vertical_jump_px: float = 8.0
    max_fb_error_px: float = 1.5
    max_normal_error_px: float = 3.5
    endpoint_allowance_px: float = 14.0
    min_flow_count: int = 4
    min_match_count: int = 4
    min_match_fraction: float = 0.55
    min_winner_margin_count: int = 2


@dataclass(frozen=True)
class PointwisePolicy:
    """Conservative gates for sparse, explicitly reviewed original UV seeds."""
    max_target_gap_px: float = 56.0
    max_adjacent_vertical_jump_px: float = 8.0
    max_fb_error_px: float = 1.5
    max_normal_error_px: float = 3.5
    max_x_snap_px: float = 16.0
    min_competing_normal_margin_px: float = 1.5


def split_measured_uv(uv: np.ndarray, gap_px: float = 56.0,
                      y_jump_px: float = 8.0) -> list[np.ndarray]:
    """Return local indices for smooth, x-contiguous measured source points."""
    xy = np.asarray(uv, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError("Candidate observed_uv must be finite Nx2")
    if len(xy) == 0:
        return []
    order = np.argsort(xy[:, 0], kind="stable")
    delta = np.diff(xy[order], axis=0)
    cuts = np.flatnonzero((delta[:, 0] > gap_px)
                             | (np.abs(delta[:, 1]) > y_jump_px)) + 1
    return [segment.astype(np.int32) for segment in np.split(order, cuts) if len(segment)]


def _mask_at(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    rounded = np.rint(xy).astype(np.int32)
    inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < mask.shape[1])
              & (rounded[:, 1] >= 0) & (rounded[:, 1] < mask.shape[0]))
    keep = np.zeros(len(xy), bool)
    keep[inside] = ~mask[rounded[inside, 1], rounded[inside, 0]]
    return keep


def _candidate_segments(candidates: list[dict], gap_px: float,
                        y_jump_px: float) -> list[dict]:
    segments = []
    for rank, candidate in enumerate(candidates):
        uv = np.asarray(candidate["observed_uv"], np.float64)
        for number, indices in enumerate(split_measured_uv(uv, gap_px, y_jump_px)):
            if len(indices) < 3:
                continue
            segments.append(dict(rank=rank, segment=number, uv_indices=indices,
                                 uv=uv[indices],
                                 component_id=int(candidate["component_id"])))
    return segments


def expand_between_reviewed_seeds(candidates: list[dict],
                                  point_indices: list[tuple[int, int]],
                                  *, gap_px: float = 56.0,
                                  y_jump_px: float = 8.0,
                                  max_transition_width_px: float = 24.0) -> list[dict]:
    """Assign only measured UV *between* ≥2 anchors on one thin source strip.

    Three reviewed 20/50/80%-position UVs may bound a longer continuous
    colour strip. The intervening observed UV are from the frozen detector,
    with its blue colour and thin-column tests. We additionally require no
    x-gap, y-jump, or anomalously wide transition inside that bounded span.
    One anchor alone remains one point. Nothing outside seed bounds inherits
    identity, even when it shares the same connected-component label.
    """
    if len(set(point_indices)) != len(point_indices):
        raise ValueError("Duplicate reviewed source UV point")
    by_rank: dict[int, set[int]] = {}
    for rank, index in point_indices:
        if rank < 0 or rank >= len(candidates):
            raise IndexError("Reviewed candidate rank is absent")
        uv = np.asarray(candidates[rank]["observed_uv"])
        if index < 0 or index >= len(uv):
            raise IndexError("Reviewed UV index is absent")
        by_rank.setdefault(int(rank), set()).add(int(index))
    expanded = []
    for rank, reviewed in sorted(by_rank.items()):
        candidate = candidates[rank]
        uv = np.asarray(candidate["observed_uv"], np.float64)
        intervals = np.asarray(candidate["vertical_transition_interval_px"], np.float64)
        peaks = np.asarray(candidate["color_peak_score"], np.float64)
        if (intervals.shape != uv.shape or peaks.shape != (len(uv),)):
            raise ValueError("Original candidate UV/colour interval shapes disagree")
        widths = intervals[:, 1] - intervals[:, 0]
        for segment_number, whole in enumerate(split_measured_uv(uv, gap_px, y_jump_px)):
            # A colour/blur break splits an otherwise geometrically smooth
            # component; it cannot be filled merely because two anchors lie
            # on opposite sides of the break.
            good = (np.isfinite(widths[whole]) & (widths[whole] > 0)
                    & (widths[whole] <= max_transition_width_px)
                    & np.isfinite(peaks[whole]) & (peaks[whole] >= 14))
            cuts = np.flatnonzero(~good)
            starts = np.r_[0, cuts + 1]
            ends = np.r_[cuts, len(whole)]
            for subnumber, (start, end) in enumerate(zip(starts, ends)):
                part = whole[start:end]
                if not len(part):
                    continue
                seeds = [int(i) for i in part if int(i) in reviewed]
                if not seeds:
                    continue
                positions = [int(np.flatnonzero(part == i)[0]) for i in seeds]
                selected = part[min(positions):max(positions)+1]
                expanded.append(dict(candidate_rank=rank,
                                     component_id=int(candidate["component_id"]),
                                     segment=segment_number, subsegment=subnumber,
                                     reviewed_seed_indices=seeds,
                                     observed_uv_indices=selected.astype(int).tolist(),
                                     identity_evidence=("reviewed_source_points_plus_"
                                                        "continuous_thin_blue_strip_between"
                                                        if len(seeds) >= 2 else
                                                        "single_reviewed_source_point"),
                                     colour_peak_min=float(np.min(peaks[selected])),
                                     transition_width_max_px=float(np.max(widths[selected])),
                                     x_span_px=float(np.ptp(uv[selected, 0]))))
    return expanded


def _normal_distances(predicted: np.ndarray, segment_uv: np.ndarray,
                      endpoint_allowance: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = segment_uv[:, 0], segment_uv[:, 1]
    in_range = ((predicted[:, 0] >= x[0] - endpoint_allowance)
                & (predicted[:, 0] <= x[-1] + endpoint_allowance))
    y_nearest = np.interp(np.clip(predicted[:, 0], x[0], x[-1]), x, y)
    nearest_segment = np.clip(np.searchsorted(x, predicted[:, 0], side="right") - 1,
                              0, len(x) - 2)
    dx = x[nearest_segment + 1] - x[nearest_segment]
    dy = y[nearest_segment + 1] - y[nearest_segment]
    slope = np.divide(dy, dx, out=np.zeros_like(dy), where=np.abs(dx) > 1e-8)
    normal_distance = np.abs(predicted[:, 1] - y_nearest) / np.sqrt(1 + slope*slope)
    return normal_distance, in_range


def match_adjacent(source_image: np.ndarray, target_image: np.ndarray,
                   source_candidates: list[dict], target_candidates: list[dict],
                   source_candidate_rank: int, source_uv_indices: np.ndarray,
                   *, source_person_mask: np.ndarray | None = None,
                   target_person_mask: np.ndarray | None = None,
                   policy: MatchPolicy = MatchPolicy()) -> dict:
    """Match one reviewed UV fragment to next-frame *raw measured* UV indices.

    The caller must verify that images are adjacent native PTS in the same
    shot. This function deliberately does not accept a camera or scene.
    """
    if (source_image.shape != target_image.shape or source_image.ndim != 3
            or source_image.shape[2] != 3 or source_image.dtype != np.uint8
            or target_image.dtype != np.uint8):
        raise ValueError("Expected two same-size BGR uint8 source frames")
    if not (0 <= source_candidate_rank < len(source_candidates)):
        raise IndexError("Source candidate rank is not present in raw frame")
    candidate = source_candidates[source_candidate_rank]
    source_uv = np.asarray(candidate["observed_uv"], np.float64)
    indices = np.asarray(source_uv_indices, np.int32)
    if (indices.ndim != 1 or len(indices) != len(np.unique(indices))
            or np.any(indices < 0) or np.any(indices >= len(source_uv))):
        raise ValueError("Source UV indices must be unique existing measured indices")
    answer = dict(status="unknown", reason=None, source_candidate_rank=source_candidate_rank,
                  source_component_id=int(candidate["component_id"]),
                  source_uv_indices=indices.tolist(), target_candidate_rank=None,
                  target_component_id=None, target_segment=None,
                  target_uv_indices=[], flow_count=0, fb_valid_count=0,
                  matched_track_count=0, second_best_track_count=0,
                  median_fb_error_px=None, median_normal_error_px=None,
                  p95_normal_error_px=None)
    if len(indices) < policy.min_flow_count:
        answer["reason"] = "too_few_reviewed_source_uv"
        return answer
    selected = source_uv[indices]
    if len(split_measured_uv(selected, policy.max_source_gap_px,
                             policy.max_adjacent_vertical_jump_px)) != 1:
        answer["reason"] = "reviewed_source_uv_crosses_large_gap"
        return answer
    source_exclude = source_exclusion(source_image, source_person_mask)
    target_exclude = source_exclusion(target_image, target_person_mask)
    selected = selected[_mask_at(source_exclude, selected)]
    if len(selected) < policy.min_flow_count:
        answer["reason"] = "source_occluded_or_excluded"
        return answer
    gray0 = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    p0 = selected.astype(np.float32).reshape(-1, 1, 2)
    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .01))
    p1, status01, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    if p1 is None or status01 is None:
        answer["reason"] = "forward_lk_failed"
        return answer
    p0_again, status10, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    if p0_again is None or status10 is None:
        answer["reason"] = "reverse_lk_failed"
        return answer
    predicted = p1.reshape(-1, 2).astype(np.float64)
    fb = np.linalg.norm(p0_again.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    valid = (status01.ravel().astype(bool) & status10.ravel().astype(bool)
             & np.isfinite(predicted).all(axis=1) & np.isfinite(fb)
             & (fb <= policy.max_fb_error_px) & _mask_at(target_exclude, predicted))
    answer["flow_count"] = len(selected)
    answer["fb_valid_count"] = int(np.count_nonzero(valid))
    if np.count_nonzero(valid) < policy.min_flow_count:
        answer["reason"] = "insufficient_forward_backward_flow"
        return answer
    predicted, fb = predicted[valid], fb[valid]
    answer["median_fb_error_px"] = float(np.median(fb))
    options = []
    for segment in _candidate_segments(target_candidates, policy.max_target_gap_px,
                                       policy.max_adjacent_vertical_jump_px):
        distances, inside = _normal_distances(predicted, segment["uv"],
                                               policy.endpoint_allowance_px)
        matched = inside & (distances <= policy.max_normal_error_px)
        options.append(dict(rank=segment["rank"], segment=segment["segment"],
                            uv_indices=segment["uv_indices"],
                            component_id=segment["component_id"],
                            uv=segment["uv"], matched=matched,
                            distances=distances, count=int(np.count_nonzero(matched))))
    if not options:
        answer["reason"] = "no_target_measured_segment"
        return answer
    options.sort(key=lambda row: (-row["count"],
                                  float(np.median(row["distances"][row["matched"]]))
                                  if row["count"] else np.inf))
    best = options[0]
    second = options[1]["count"] if len(options) > 1 else 0
    answer["matched_track_count"] = best["count"]
    answer["second_best_track_count"] = second
    if (best["count"] < policy.min_match_count
            or best["count"] / len(predicted) < policy.min_match_fraction):
        answer["reason"] = "insufficient_target_normal_support"
        return answer
    if second and best["count"] - second < policy.min_winner_margin_count:
        answer["reason"] = "competing_blue_segments"
        return answer
    matched_prediction = predicted[best["matched"]]
    xlo = np.min(matched_prediction[:, 0]) - policy.endpoint_allowance_px
    xhi = np.max(matched_prediction[:, 0]) + policy.endpoint_allowance_px
    actual = best["uv"]
    target_indices = best["uv_indices"][(actual[:, 0] >= xlo)
                                          & (actual[:, 0] <= xhi)]
    if len(target_indices) < 2:
        answer["reason"] = "too_few_actual_target_uv"
        return answer
    residuals = best["distances"][best["matched"]]
    answer.update(status="candidate_link", reason="measured_uv_normal_match",
                  target_candidate_rank=best["rank"],
                  target_component_id=best["component_id"],
                  target_segment=best["segment"],
                  target_uv_indices=target_indices.astype(int).tolist(),
                  median_normal_error_px=float(np.median(residuals)),
                  p95_normal_error_px=float(np.percentile(residuals, 95)))
    return answer


def match_bidirectional(image0: np.ndarray, image1: np.ndarray,
                        candidates0: list[dict], candidates1: list[dict],
                        rank0: int, uv_indices0: np.ndarray,
                        *, mask0: np.ndarray | None = None,
                        mask1: np.ndarray | None = None,
                        policy: MatchPolicy = MatchPolicy()) -> dict:
    """Require the measured target subset to link back to the original rank."""
    forward = match_adjacent(image0, image1, candidates0, candidates1, rank0,
                             uv_indices0, source_person_mask=mask0,
                             target_person_mask=mask1, policy=policy)
    result = dict(forward=forward, reverse=None, status="unknown", reason=None)
    if forward["status"] != "candidate_link":
        result["reason"] = f"forward:{forward['reason']}"
        return result
    reverse = match_adjacent(image1, image0, candidates1, candidates0,
                             forward["target_candidate_rank"],
                             np.asarray(forward["target_uv_indices"], np.int32),
                             source_person_mask=mask1, target_person_mask=mask0,
                             policy=policy)
    result["reverse"] = reverse
    original = set(map(int, uv_indices0))
    returned = set(reverse["target_uv_indices"])
    overlap = len(original & returned)
    result["roundtrip_source_uv_overlap"] = overlap
    if (reverse["status"] != "candidate_link"
            or reverse["target_candidate_rank"] != rank0
            or overlap < max(2, int(np.ceil(.4 * len(original))))):
        result["reason"] = "adjacent_reverse_inconsistent"
        return result
    result["status"] = "candidate_link_bidirectional"
    result["reason"] = "measured_uv_both_directions"
    return result


def match_sparse_reviewed_points(source_pts: int, target_pts: int,
                                 source_image: np.ndarray, target_image: np.ndarray,
                                 source_candidates: list[dict],
                                 target_candidates: list[dict],
                                 source_seed_indices: list[tuple[int, int]],
                                 *, source_person_mask: np.ndarray | None = None,
                                 target_person_mask: np.ndarray | None = None,
                                 policy: PointwisePolicy = PointwisePolicy()) -> dict:
    """LK+FB match sparse reviewed points to other *measured* candidate UV.

    `source_seed_indices` holds (candidate_rank, observed_uv_index) pairs, not
    whole candidate IDs. Every returned target index names an actual detector
    UV. A mixed connected component never conveys identity to its unlisted UV.
    This runs one native 256-PTS step; it does not infer through missing frames.
    """
    if abs(int(target_pts) - int(source_pts)) != 256:
        raise ValueError("Source identity may move only to adjacent native frames")
    if (source_image.shape != target_image.shape
            or source_image.shape != (1080, 1920, 3)
            or source_image.dtype != np.uint8 or target_image.dtype != np.uint8):
        raise ValueError("Expected two exact native BGR uint8 frames")
    if len(set(source_seed_indices)) != len(source_seed_indices):
        raise ValueError("Duplicate pointwise source seed")
    point_uv = []
    for rank, index in source_seed_indices:
        if rank < 0 or rank >= len(source_candidates):
            raise IndexError("Seed candidate rank not in source frame")
        raw = np.asarray(source_candidates[rank]["observed_uv"], np.float64)
        if index < 0 or index >= len(raw):
            raise IndexError("Seed UV index not in source candidate")
        point_uv.append(raw[index])
    rows = [dict(source_candidate_rank=int(rank), source_uv_index=int(index),
                 source_uv=point_uv[number].tolist(), target_candidate_rank=None,
                 target_uv_index=None, target_uv=None, status="unknown", reason=None,
                 flow_predicted_uv=None, fb_error_px=None, normal_error_px=None,
                 x_snap_error_px=None, competitor_margin_px=None)
            for number, (rank, index) in enumerate(source_seed_indices)]
    if not rows:
        return dict(source_pts=int(source_pts), target_pts=int(target_pts),
                    status="no_reviewed_seeds", rows=[])
    source_exclude = source_exclusion(source_image, source_person_mask)
    target_exclude = source_exclusion(target_image, target_person_mask)
    available = _mask_at(source_exclude, np.asarray(point_uv, np.float64))
    gray0 = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    p0 = np.asarray(point_uv, np.float32).reshape(-1, 1, 2)
    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .01))
    p1, forward_ok, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    if p1 is None or forward_ok is None:
        for row in rows:
            row["reason"] = "forward_lk_failed"
        return dict(source_pts=int(source_pts), target_pts=int(target_pts),
                    status="unknown", rows=rows)
    p0_again, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    if p0_again is None or reverse_ok is None:
        for row in rows:
            row["reason"] = "reverse_lk_failed"
        return dict(source_pts=int(source_pts), target_pts=int(target_pts),
                    status="unknown", rows=rows)
    predicted = p1.reshape(-1, 2).astype(np.float64)
    fb = np.linalg.norm(p0_again.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    valid = (available & forward_ok.ravel().astype(bool)
             & reverse_ok.ravel().astype(bool)
             & np.isfinite(predicted).all(axis=1) & np.isfinite(fb)
             & (fb <= policy.max_fb_error_px)
             & _mask_at(target_exclude, predicted))
    segments = _candidate_segments(target_candidates, policy.max_target_gap_px,
                                   policy.max_adjacent_vertical_jump_px)
    for number, row in enumerate(rows):
        if not available[number]:
            row["reason"] = "reviewed_source_pixel_occluded_or_excluded"
            continue
        if not valid[number]:
            row["reason"] = "forward_backward_or_target_mask_failed"
            continue
        row["flow_predicted_uv"] = predicted[number].tolist()
        row["fb_error_px"] = float(fb[number])
        options = []
        for segment in segments:
            distance, inside = _normal_distances(predicted[number:number+1],
                                                  segment["uv"], policy.max_x_snap_px)
            if not inside[0] or distance[0] > policy.max_normal_error_px:
                continue
            nearest = int(np.argmin(np.abs(segment["uv"][:, 0] - predicted[number, 0])))
            x_error = float(abs(segment["uv"][nearest, 0] - predicted[number, 0]))
            if x_error > policy.max_x_snap_px:
                continue
            original_index = int(segment["uv_indices"][nearest])
            if not _mask_at(target_exclude,
                            segment["uv"][nearest:nearest+1])[0]:
                continue
            options.append(dict(rank=int(segment["rank"]),
                                index=original_index,
                                uv=segment["uv"][nearest].tolist(),
                                normal_error=float(distance[0]), x_error=x_error))
        options.sort(key=lambda entry: (entry["normal_error"], entry["x_error"]))
        if not options:
            row["reason"] = "no_actual_target_uv_on_flow_normal"
            continue
        best = options[0]
        margin = (options[1]["normal_error"] - best["normal_error"]
                  if len(options) > 1 else np.inf)
        row["competitor_margin_px"] = (float(margin) if np.isfinite(margin) else None)
        if margin < policy.min_competing_normal_margin_px:
            row["reason"] = "competing_measured_blue_segments"
            continue
        row.update(target_candidate_rank=best["rank"],
                   target_uv_index=best["index"], target_uv=best["uv"],
                   normal_error_px=best["normal_error"],
                   x_snap_error_px=best["x_error"], status="candidate_link",
                   reason="lk_fb_to_actual_source_uv")
    destinations = {}
    for number, row in enumerate(rows):
        if row["status"] == "candidate_link":
            destinations.setdefault((row["target_candidate_rank"],
                                     row["target_uv_index"]), []).append(number)
    for numbers in destinations.values():
        if len(numbers) > 1:
            for number in numbers:
                rows[number].update(status="unknown", reason="many_to_one_source_match",
                                    target_candidate_rank=None, target_uv_index=None,
                                    target_uv=None)
    return dict(source_pts=int(source_pts), target_pts=int(target_pts),
                status="pointwise_candidate_links_only", rows=rows)


def match_sparse_bidirectional(pts0: int, pts1: int, image0: np.ndarray,
                               image1: np.ndarray, candidates0: list[dict],
                               candidates1: list[dict],
                               seed_indices0: list[tuple[int, int]],
                               *, mask0: np.ndarray | None = None,
                               mask1: np.ndarray | None = None,
                               policy: PointwisePolicy = PointwisePolicy()) -> dict:
    """Keep a point only if independently reread source UV maps back exactly."""
    forward = match_sparse_reviewed_points(
        pts0, pts1, image0, image1, candidates0, candidates1, seed_indices0,
        source_person_mask=mask0, target_person_mask=mask1, policy=policy)
    target_pairs = [(int(row["target_candidate_rank"]), int(row["target_uv_index"]))
                    for row in forward["rows"] if row["status"] == "candidate_link"]
    reverse = match_sparse_reviewed_points(
        pts1, pts0, image1, image0, candidates1, candidates0, target_pairs,
        source_person_mask=mask1, target_person_mask=mask0, policy=policy)
    inverse = {(row["source_candidate_rank"], row["source_uv_index"]): row
               for row in reverse["rows"]}
    accepted = []
    for row in forward["rows"]:
        if row["status"] != "candidate_link":
            continue
        target_key = (row["target_candidate_rank"], row["target_uv_index"])
        back = inverse.get(target_key)
        if (back is None or back["status"] != "candidate_link"
                or (back["target_candidate_rank"], back["target_uv_index"])
                != (row["source_candidate_rank"], row["source_uv_index"])):
            row.update(status="unknown", reason="reverse_actual_uv_disagrees",
                       target_candidate_rank=None, target_uv_index=None,
                       target_uv=None)
        else:
            row.update(status="bidirectional_source_match",
                       reason="actual_uv_matches_both_directions")
            accepted.append(row)
    return dict(source_pts=int(pts0), target_pts=int(pts1),
                status="source_pointwise_review_seed_propagation",
                accepted_count=len(accepted), rows=forward["rows"],
                reverse_rows=reverse["rows"])
