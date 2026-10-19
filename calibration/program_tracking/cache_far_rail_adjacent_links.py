"""Native adjacent-frame, source-only links between measured blue-strip UVs.

This creates no physical identity and never treats optical-flow locations as
observations. Every accepted target is an index into the frozen source
candidate cache, after forward/backward LK and measured-line competition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import av
import numpy as np

from far_rail_source_identity_flow import match_sparse_bidirectional
from render_result import DEFAULT_MASKS, DEFAULT_PEOPLE, PersonMasks


CACHE_DIR = Path("D:/CodexData/RinkSpace_program_tracking_tmp/far_rail_source_components_v1")
OUTPUT_DIR = Path("D:/CodexData/RinkSpace_program_tracking_tmp/far_rail_source_flow_v1")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CandidateCache:
    """Readonly offsets into all real source UVs; local component IDs can move."""

    def __init__(self, directory: Path = CACHE_DIR):
        manifest_path = directory / "far_rail_all_main_source_candidates.json"
        self.manifest_sha256 = _sha(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        packet = directory / "far_rail_all_main_source_candidates.npz"
        self.packet_sha256 = _sha(packet)
        if self.packet_sha256 != self.manifest["packet_sha256"]:
            raise ValueError("Source candidate packet hash differs from manifest")
        with np.load(packet, allow_pickle=False) as archive:
            self.arrays = {key: archive[key].copy() for key in archive.files}
        a = self.arrays
        self.frame_pts = a["frame_pts"]
        self.frame_shot = a["frame_shot_index"]
        self.frame_offset = a["frame_candidate_offsets"]
        self.uv_offset = a["candidate_uv_offsets"]
        self.pts_to_index = {int(pts): i for i, pts in enumerate(self.frame_pts)}
        if (len(self.pts_to_index) != len(self.frame_pts)
                or len(self.frame_offset) != len(self.frame_pts) + 1
                or len(self.uv_offset) != len(a["component_id"]) + 1
                or int(self.uv_offset[-1]) != len(a["observed_uv"])):
            raise ValueError("Frozen source candidate index is inconsistent")

    def candidates(self, index: int) -> list[dict]:
        a = self.arrays
        result = []
        for absolute in range(int(self.frame_offset[index]), int(self.frame_offset[index + 1])):
            lo, hi = map(int, self.uv_offset[absolute:absolute + 2])
            result.append(dict(component_id=int(a["component_id"][absolute]),
                               component_rank=int(a["component_rank"][absolute]),
                               component_bbox_xywh=a["component_bbox_xywh"][absolute].tolist(),
                               observed_uv=a["observed_uv"][lo:hi],
                               vertical_transition_interval_px=(
                                   a["vertical_transition_interval_px"][lo:hi]),
                               color_peak_score=a["color_peak_score"][lo:hi]))
        if any(i != row["component_rank"] for i, row in enumerate(result)):
            raise ValueError("Candidate rank changed inside frozen frame")
        return result

    def flat_uv_index(self, frame_index: int, rank: int, local_index: int) -> int:
        component = int(self.frame_offset[frame_index]) + int(rank)
        if component < self.frame_offset[frame_index] or component >= self.frame_offset[frame_index+1]:
            raise IndexError("Candidate rank outside frozen frame")
        flat = int(self.uv_offset[component]) + int(local_index)
        if flat < self.uv_offset[component] or flat >= self.uv_offset[component+1]:
            raise IndexError("UV local index outside frozen candidate")
        return flat


def build_adjacent_links(output_dir: Path = OUTPUT_DIR, *, cache_dir: Path = CACHE_DIR,
                         masks_dir: Path = DEFAULT_MASKS,
                         people_path: Path = DEFAULT_PEOPLE,
                         progress_interval: int = 500) -> dict:
    cache = CandidateCache(cache_dir)
    manifest = cache.manifest
    source = Path(manifest["source_video"])
    if _sha(source) != manifest["source_video_sha256"]:
        raise ValueError("Native source video changed since candidate extraction")
    if _sha(people_path) != manifest["person_manifest_sha256"]:
        raise ValueError("Person-detection manifest changed")
    chunks = {entry["name"]: entry["sha256"] for entry in manifest["sam2_chunks"]}
    if {path.name for path in masks_dir.glob("*.npz")} != set(chunks):
        raise ValueError("SAM2 mask chunk inventory changed")
    if any(_sha(masks_dir / name) != digest for name, digest in chunks.items()):
        raise ValueError("SAM2 mask chunk changed")
    people = PersonMasks(masks_dir, people_path)
    total_uv = len(cache.arrays["observed_uv"])
    forward_target = np.full(total_uv, -1, np.int64)
    forward_fb = np.full(total_uv, np.nan, np.float32)
    forward_normal = np.full(total_uv, np.nan, np.float32)
    forward_x_snap = np.full(total_uv, np.nan, np.float32)
    reason_counts = Counter()
    per_shot = defaultdict(Counter)
    seen = set()
    previous = None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if (stream.time_base.numerator, stream.time_base.denominator) != (1, 15360):
            raise ValueError("Native video time base changed")
        container.seek(max(0, int(cache.frame_pts[0]) - 15360),
                       stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < cache.frame_pts[0]:
                continue
            if frame.pts > cache.frame_pts[-1]:
                break
            pts = int(frame.pts)
            index = cache.pts_to_index.get(pts)
            if index is None:
                continue
            if pts in seen:
                raise ValueError(f"Duplicate original native frame {pts}")
            seen.add(pts)
            image = frame.to_ndarray(format="bgr24")
            if (image.shape != (1080, 1920, 3)
                    or hashlib.sha256(image.tobytes()).hexdigest()
                    != cache.arrays["frame_source_pixel_sha256"][index]):
                raise ValueError(f"Original pixel hash changed at PTS {pts}")
            mask, mask_status = people.get(pts, require_sam2=True)
            if mask_status != cache.arrays["frame_sam2_status"][index]:
                raise ValueError(f"SAM2 status changed at PTS {pts}")
            candidates = cache.candidates(index)
            if previous is not None:
                prior_index, prior_pts, prior_image, prior_mask, prior_candidates = previous
                if (cache.frame_shot[prior_index] == cache.frame_shot[index]
                        and pts - prior_pts == 256):
                    seeds = [(rank, local)
                             for rank, candidate in enumerate(prior_candidates)
                             for local in range(len(candidate["observed_uv"]))]
                    matched = match_sparse_bidirectional(
                        prior_pts, pts, prior_image, image,
                        prior_candidates, candidates, seeds,
                        mask0=prior_mask, mask1=mask)
                    shot_counts = per_shot[int(cache.frame_shot[index])]
                    shot_counts["native_pairs"] += 1
                    for row in matched["rows"]:
                        src = cache.flat_uv_index(prior_index,
                                                  row["source_candidate_rank"],
                                                  row["source_uv_index"])
                        reason_counts[row["reason"]] += 1
                        if row["status"] == "bidirectional_source_match":
                            dst = cache.flat_uv_index(index,
                                                      row["target_candidate_rank"],
                                                      row["target_uv_index"])
                            if forward_target[src] >= 0:
                                raise AssertionError("A source UV received two target links")
                            forward_target[src] = dst
                            forward_fb[src] = row["fb_error_px"]
                            forward_normal[src] = row["normal_error_px"]
                            forward_x_snap[src] = row["x_snap_error_px"]
                            shot_counts["linked_uv"] += 1
                        else:
                            shot_counts["unknown_uv"] += 1
            previous = (index, pts, image, mask, candidates)
            if progress_interval and len(seen) % progress_interval == 0:
                print(json.dumps({"decoded_frames": len(seen), "expected": len(cache.frame_pts),
                                  "linked_uv": int(np.count_nonzero(forward_target >= 0))}),
                      flush=True)
    if seen != set(map(int, cache.frame_pts)):
        missing = sorted(set(map(int, cache.frame_pts)) - seen)
        raise ValueError(f"Native adjacent decode missed {len(missing)} frames; {missing[:5]}")
    linked = np.flatnonzero(forward_target >= 0)
    targets = forward_target[linked]
    if len(np.unique(targets)) != len(targets):
        raise AssertionError("Local links are not one-to-one")
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = output_dir / "far_rail_adjacent_source_links.npz"
    pending = output_dir / "far_rail_adjacent_source_links.writing.npz"
    np.savez_compressed(pending,
                        forward_target_uv_flat_index=forward_target,
                        forward_fb_error_px=forward_fb,
                        forward_normal_error_px=forward_normal,
                        forward_x_snap_error_px=forward_x_snap,
                        candidate_packet_sha256=np.asarray(cache.packet_sha256),
                        source_video_sha256=np.asarray(manifest["source_video_sha256"]),
                        semantics=np.asarray("unassigned_bidirectional_adjacent_actual_uv_links"))
    pending.replace(packet)
    report = dict(schema="rinkspace.far_rail_adjacent_source_links.v1",
                  semantics="source-only bidirectional LK+FB actual-UV correspondence; no physical identity",
                  packet=str(packet), packet_sha256=_sha(packet),
                  candidate_packet_sha256=cache.packet_sha256,
                  candidate_manifest_sha256=cache.manifest_sha256,
                  source_video_sha256=manifest["source_video_sha256"],
                  sam2_chunk_collection_sha256=manifest["sam2_chunk_collection_sha256"],
                  flow_module=str(Path(__file__).with_name("far_rail_source_identity_flow.py")),
                  flow_module_sha256=_sha(Path(__file__).with_name("far_rail_source_identity_flow.py")),
                  counts=dict(frames=len(cache.frame_pts), total_raw_candidate_uv=total_uv,
                              adjacent_matched_uv=len(linked),
                              by_shot={str(k): dict(v) for k, v in sorted(per_shot.items())},
                              outcome_reasons=dict(reason_counts)),
                  unknown_semantics="No link is unknown; never backfill by projected or flow UV")
    (output_dir / "far_rail_adjacent_source_links.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--progress-interval", type=int, default=500)
    args = parser.parse_args()
    result = build_adjacent_links(args.output_dir, progress_interval=args.progress_interval)
    print(json.dumps({"packet": result["packet"], "packet_sha256": result["packet_sha256"],
                      "counts": result["counts"]}), flush=True)
