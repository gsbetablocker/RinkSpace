"""GPU training and source-pixel review for the shared-rink visual module.

Source observations, drawing support, propagated estimates and camera priors
retain different roles. In particular, neither a fitted curve nor a smooth
trajectory is used as a pixel label. All transient files use the Data root.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(path):
    config = read_json(path)
    data = Path(config["data_directory"]).resolve()
    if data.is_relative_to(ROOT) or "OneDrive" in data.parts:
        raise ValueError("Training intermediates must be outside OneDrive")
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(data / "runtime_cache"))
    os.environ.setdefault("TORCH_HOME", str(data / "runtime_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return config, data


@lru_cache(maxsize=24)
def _image(path, frame_key=None):
    if frame_key is not None:
        with np.load(path, allow_pickle=False) as source:
            image = source[frame_key].copy()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("A source bundle member must contain native RGB uint8 pixels")
        return image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def load_source_clock_registry(registry=None):
    """Load only explicitly supplied event clock facts; no sample is a default."""
    if registry is None:
        return dict(sources=[], unregistered_source_policy="native_frames_without_alias_inference")
    if isinstance(registry, (str, Path)):
        path = Path(registry)
        path = path if path.is_absolute() else ROOT / path
        packet = read_json(path)
        packet = {**packet, "_registry_path": str(path.resolve()), "_registry_sha256": sha256(path)}
    elif isinstance(registry, dict):
        packet = dict(registry)
    else:
        raise ValueError("Source clock registry must be an explicit path or mapping")
    if not isinstance(packet.get("sources"), list):
        raise ValueError("Source clock registry requires per-source records")
    return packet


def confirmed_exposure_groups(source_sha256, time_base, *, registry=None):
    """Select proven exposure aliases by source and clock before opening files.

    Unregistered sources keep native frame identities. Equal pixels or nominal
    rates cannot establish exposure aliases. An explicitly configured proof
    fails on broken files or bindings, including before sequence-cache reuse.
    """
    clock = str(Fraction(time_base))
    if Fraction(clock) <= 0:
        raise ValueError("Source clock must be positive")
    registry = load_source_clock_registry(registry)
    records = [r for r in registry["sources"] if r["source_sha256"] == source_sha256]
    if len(records) > 1:
        raise ValueError("Source clock registry contains ambiguous source records")
    metadata = dict(source_sha256=source_sha256, time_base=clock, proof_sha256=None,
                    status="native_frames_without_verified_exposure_aliases",
                    proof_path=None, confirmed_repeat_pairs=0,
                    registry_path=registry.get("_registry_path"),
                    registry_sha256=registry.get("_registry_sha256"))
    if not records:
        return {}, metadata
    record = records[0]
    if str(Fraction(record["time_base"])) != clock:
        raise ValueError("Source clock differs from its registered original video")
    spec = record.get("exposure_proof")
    if spec is None:
        return {}, metadata
    path = Path(spec["path"])
    path = path if path.is_absolute() else ROOT / path
    expected = spec["sha256"]
    if sha256(path) != expected:
        raise ValueError("Configured original-pixel exposure proof changed")
    proof = read_json(path)
    if proof.get("schema") != spec.get("schema") or proof.get("schema") != "rinkspace.official50-confirmed-exposure-groups.v1":
        raise ValueError("Unsupported or mismatched configured exposure proof schema")
    native = proof["source_binding"]["full60"]
    paired = proof["source_binding"]["official50"]
    if native["sha256"] != source_sha256 or str(Fraction(native["native_pts_time_base"])) != clock:
        raise ValueError("Exposure proof belongs to another original source or clock")
    paired_spec = spec.get("paired_source_binding")
    if paired_spec and (paired_spec["source_sha256"] != paired["sha256"] or
                        Fraction(paired_spec["time_base"]) != Fraction(paired["native_pts_time_base"])):
        raise ValueError("Exposure proof paired-source binding differs")
    validation = proof["validation"]
    if not validation["all_groups_single_shot"] or not validation["all_nine_cut_pairs_different_groups"]:
        raise ValueError("Exposure proof lacks its verified source-cut identity")
    mapping_spec = spec.get("mapping_evidence")
    if mapping_spec:
        mapping_path = Path(mapping_spec["path"])
        mapping_path = mapping_path if mapping_path.is_absolute() else ROOT / mapping_path
        if (sha256(mapping_path) != mapping_spec["sha256"] or
                mapping_spec["sha256"] not in proof["input_sha256"].values()):
            raise ValueError("Configured original exposure mapping evidence changed")
    mapping = {}
    for group in proof["groups"]:
        pts = list(map(int, group["full_pts"]))
        if not pts or pts != sorted(set(pts)):
            raise ValueError("Exposure group native source PTS are empty, duplicated or unordered")
        official = group["confirmed_official_pts"]
        if len(pts) > 1 and (official is None or len(group["confirmed_pair_evidence"]) != len(pts) - 1):
            raise ValueError("Repeated exposure lacks original pair evidence")
        for i, stamp in enumerate(pts):
            if stamp in mapping:
                raise ValueError("Native source PTS belongs to multiple exposure groups")
            accepted = group["frame_accepted_official_pts"][i]
            mapping[stamp] = dict(canonical=pts[0], shot=group["shot_index"],
                                  official_pts=official if official is not None else accepted,
                                  official_time_base=str(Fraction(paired["native_pts_time_base"])),
                                  status=group["frame_match_status"][i])
    coverage = spec.get("coverage")
    if coverage and (len(mapping) != coverage["native_frame_count"] or
                     min(mapping) != coverage["start_pts"] or max(mapping) != coverage["last_native_pts"]):
        raise ValueError("Configured exposure proof coverage differs from original evidence")
    metadata.update(proof_path=str(path.resolve()), proof_sha256=expected,
                    status="registered_original_pixel_exposure_proof_for_covered_PTS_only",
                    confirmed_repeat_pairs=validation["confirmed_pair_count"], coverage=coverage)
    return mapping, metadata


def frame_neighbours(video_path, sample, radius_seconds=.12, *, exposure_registry=None):
    """Decode neighbouring real PTS, never manufacture an FPS-derived key.

    Only explicitly proven exposure repeats share a neighbour group. Pixel
    equality is diagnostic and preserves every unproven native frame identity.
    """
    import av
    target = int(sample["source_pts"])
    clock = Fraction(sample["time_base"])
    scope = sample.get("source_shot_scope")
    if scope is not None:
        if (scope["source_sha256"] != sample["source_hash"] or
                Fraction(scope["time_base"]) != clock or
                not scope["start_pts"] <= target < scope["end_pts_exclusive"]):
            raise ValueError("Source neighbours need the label's exact native shot and clock")
    frames = []
    exposures, exposure_proof = confirmed_exposure_groups(sample["source_hash"], clock,
                                                          registry=exposure_registry)
    target_group = exposures.get(target)
    decoded_target = None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.time_base != clock:
            raise ValueError("Video and label time bases differ")
        margin = int(radius_seconds / float(clock))
        container.seek(max(0, target - margin), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = int(frame.pts)
            if scope is not None and pts < scope["start_pts"]:
                continue
            if scope is not None and pts >= scope["end_pts_exclusive"]:
                break
            if pts < target - margin:
                continue
            if pts > target + margin:
                break
            group = exposures.get(pts)
            if target_group and (not group or group["shot"] != target_group["shot"]):
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if pts == target:
                decoded_target = rgb
            same_confirmed = bool(frames and group and exposures.get(frames[-1][0]) and
                                  group["canonical"] == exposures[frames[-1][0]]["canonical"])
            if same_confirmed:
                frames[-1][2].append(pts)
            else:
                frames.append([pts, rgb, [pts]])
    original = _image(sample["frame_path"], sample.get("frame_key"))
    if not frames:
        raise ValueError(f"No source video neighbours for {target}")
    centres = [i for i, (_, _, native_pts) in enumerate(frames) if target in native_pts]
    if len(centres) != 1:
        raise ValueError("Annotated source PTS is absent from decoded video")
    centre = centres[0]
    if decoded_target is None:
        raise ValueError("Exact annotated source frame was not decoded")
    error = float(np.abs(decoded_target.astype(float) - original).mean())
    if error > 1:
        raise ValueError(f"Stored annotation frame differs from source video: {error}")
    previous = frames[max(0, centre - 1)]
    following = frames[min(len(frames) - 1, centre + 1)]
    # PNG remains the annotation source. Neighbour images are actual video.
    return np.stack([previous[1], original, following[1]]), {
        "native_pts": [previous[0], target, following[0]],
        "time_base": str(clock), "central_video_png_mean_rgb_error": error,
        "display_repeat_groups": [f[2] for f in frames],
        "equal_RGB_neighbour_pairs_are_exposure_proof": False,
        "canonical_exposure_pts": [exposures.get(p, {}).get("canonical", p)
                                   for p in [previous[0], target, following[0]]],
        "official_exposure_records": [exposures.get(p)
                                      for p in [previous[0], target, following[0]]],
        "official_exposure_proof": exposure_proof,
        "official_exposure_identity": "existing pixel-match evidence only; other frames unknown",
        "source_pixels_modified": False,
        "native_source_shot_scope": scope,
    }


def source_previous_observation(row):
    """Distinguish real previous evidence from a padded copy of the centre.

    Pixel equality supplies no exposure identity. Distinct native frames may
    alias only through the explicitly verified source-clock proof already
    checked before cache loading. Missing history does not advance memory.
    """
    native = row.get("native_pts")
    if (not isinstance(native, (list, tuple)) or len(native) != 3 or
            any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in native)
            or native[0] > native[1]):
        raise ValueError("Sequence memory needs three source-native PTS with previous at or before centre")
    if native[0] == native[1]:
        return dict(available=False, reason="no_previous_source_observation_context_padding")
    canonical = row.get("canonical_exposure_pts", native)
    if (not isinstance(canonical, (list, tuple)) or len(canonical) != 3 or
            any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in canonical)):
        raise ValueError("Sequence canonical exposures require their native source identity")
    if canonical[0] == canonical[1]:
        records = row.get("official_exposure_records")
        proof = row.get("official_exposure_proof", {})
        if (not proof.get("proof_sha256") or not row.get("source_video_sha256") or
                proof.get("source_sha256") != row.get("source_video_sha256") or
                Fraction(row.get("time_base", "0")) <= 0 or
                Fraction(proof.get("time_base", "0")) != Fraction(row.get("time_base", "0")) or
                not isinstance(records, (list, tuple)) or len(records) != 3 or
                not all(isinstance(r, dict) and r.get("canonical") == canonical[0] for r in records[:2]) or
                records[0].get("shot") != records[1].get("shot")):
            raise ValueError("Distinct native frames sharing an exposure need their explicit source-clock proof")
        return dict(available=False, reason="same_confirmed_source_exposure_no_distinct_previous")
    return dict(available=True, reason="distinct_previous_source_observation")


def prepare_sequences(config, data, manifest):
    """Reuse a single compressed bundle instead of exporting frame folders."""
    from .vision_supervision import validate_sequence_shot_scope
    source_hashes = {}
    for sample in manifest["samples"]:
        source_video = str(Path(sample.get("source_video_path", config["source_video"])).resolve())
        if source_video not in source_hashes:
            source_hashes[source_video] = sha256(source_video)
        if sample.get("source_hash") != source_hashes[source_video]:
            raise ValueError("Training labels and the corresponding original video have different source identities")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    clock_registry = load_source_clock_registry(config.get("source_clock_registry"))
    source_clocks = {(sample["source_hash"], str(Fraction(sample["time_base"])))
                     for sample in manifest["samples"]}
    clock_evidence = [confirmed_exposure_groups(digest, clock, registry=clock_registry)[1]
                      for digest, clock in sorted(source_clocks)]
    clock_digest = hashlib.sha256(json.dumps(clock_evidence, sort_keys=True).encode()).hexdigest()
    clock_revision = "explicit_source_clock_proofs_native_frames_and_exact_shot_scope_v3"
    metadata_path = data / "sequence_sources.json"
    bundle_path = data / "training_sequences.npz"
    if metadata_path.exists() and bundle_path.exists():
        old = read_json(metadata_path)
        if (old.get("manifest_sha256") == manifest_hash and
                old.get("source_videos_sha256") == source_hashes and old.get("clock_revision") == clock_revision
                and old.get("clock_evidence_sha256") == clock_digest):
            if sha256(bundle_path) == old.get("bundle_sha256"):
                for sample, row in zip(manifest["samples"], old["rows"]):
                    validate_sequence_shot_scope(sample, row)
                with np.load(bundle_path, allow_pickle=False) as stored:
                    return [stored[f"sample_{i}"].copy()
                            for i in range(len(manifest["samples"]))], old
    sequences, rows = [], []
    last_progress = time.monotonic()
    for index, sample in enumerate(manifest["samples"]):
        sequence, row = frame_neighbours(sample.get("source_video_path", config["source_video"]), sample,
                                         exposure_registry=clock_registry)
        row["source_video_sha256"] = sample["source_hash"]
        row["source_video_path"] = sample.get("source_video_path", config["source_video"])
        validate_sequence_shot_scope(sample, row)
        sequences.append(sequence)
        rows.append(row)
        if time.monotonic() - last_progress >= 15:
            print(json.dumps(dict(stage="prepare_actual_source_sequences", completed=index+1,
                                  total=len(manifest["samples"]), source_pts=sample["source_pts"])), flush=True)
            last_progress = time.monotonic()
    np.savez_compressed(bundle_path,
                        **{f"sample_{i}": image for i, image in enumerate(sequences)})
    metadata = dict(schema="rinkspace.vision-source-sequences.v1",
                    manifest_sha256=manifest_hash, source_videos_sha256=source_hashes,
                    source_video_sha256=source_hashes[str(Path(config["source_video"]).resolve())],
                    bundle_sha256=sha256(bundle_path), rows=rows,
                    clock_revision=clock_revision,
                    clock_evidence_sha256=clock_digest, source_clock_evidence=clock_evidence,
                    future_frame_used=True, realtime_causal_claim=False)
    write_json(metadata_path, metadata)
    return sequences, metadata


def validate_training_sources(manifest):
    """Check evidence bindings before supervision; integrity is not accuracy.

    User-confirmed identities and positions directly supervise the network.
    Source review has separate authority; FIT membership and camera projections
    never supply image labels.
    """
    rows, source_file_hashes = [], {}
    accepted_states = {"user_confirmed_visual_supervision", "user_trace_weak",
                       "source_reviewed_interval", "source_reviewed_weak_stroke"}
    for sample in manifest["samples"]:
        source_path = str(Path(sample["frame_path"]).resolve())
        if source_path not in source_file_hashes:
            source_file_hashes[source_path] = sha256(source_path)
        digest = source_file_hashes[source_path]
        if digest != sample.get("frame_sha256"):
            raise ValueError(f"Original supervision image hash changed: {sample['source_pts']}")
        if sample.get("pixel_ground_truth"):
            raise ValueError("Source accuracy declarations require their own explicit pixel evidence")
        image = _image(sample["frame_path"],sample.get("frame_key"))
        if sample.get("decoded_native_RGB_sha256") and hashlib.sha256(image.tobytes()).hexdigest() != sample["decoded_native_RGB_sha256"]:
            raise ValueError("The original native RGB source bundle member changed")
        if list(image.shape[1::-1]) != sample["frame_size_wh"]:
            raise ValueError("Source label canvas differs from original native pixels")
        for annotation in sample["annotations"]:
            if annotation.get("state") in ("ignored", "occluded", "unreviewed"):
                continue
            if annotation.get("state") not in accepted_states or not annotation.get("authority"):
                raise ValueError("A training annotation lacks explicit source review authority")
            if annotation.get("pixel_ground_truth") and annotation.get("state") != "user_confirmed_visual_supervision":
                raise ValueError("Assistant source intervals do not establish exact pixel truth")
            authority = annotation["authority"]
            if isinstance(authority, dict) and authority.get("path") and authority.get("sha256"):
                if sha256(authority["path"]) != authority["sha256"]:
                    raise ValueError("The original user drawing changed after digitization")
            if annotation.get("original_quality", {}).get("model_projection_used_to_generate_or_select_uv"):
                raise ValueError("Model-projected points cannot become image labels")
        for region in sample.get("regions", []):
            if region.get("state") not in (None, "source_reviewed_region") or not region.get("authority"):
                raise ValueError("Training regions require actual source review authority")
            if region.get("pixel_ground_truth") or region.get("boundary_pixel_GT"):
                raise ValueError("Region semantics do not establish exact boundary pixel truth")
            if region.get("rgb_mask") and region.get("mask_origin") != "reviewed_original_RGB_colour_in_named_source_region":
                raise ValueError("Pixel colour masks must come from an explicitly reviewed source region")
        rows.append(dict(source_pts=int(sample["source_pts"]), time_base=sample["time_base"],
                         source_sha256=sample["source_hash"],
                         frame_sha256=digest, original_canvas_wh=sample["frame_size_wh"],
                         semantic_group_count=len(sample["annotations"]),
                         reviewed_region_count=len(sample.get("regions", [])),
                         pixel_accuracy_confirmed=False))
    return rows


def _annotations_in_crop(sample, origin, size):
    from .vision_supervision import active_source_supervision_mask
    result = []
    height, width = (size, size) if np.isscalar(size) else size
    offset = np.asarray(origin, float)
    for annotation in sample["annotations"]:
        if annotation.get("state") in ("ignored", "occluded", "unreviewed"):
            continue
        uv = np.asarray(annotation["uv"], float)
        if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 1:
            continue
        uv = uv - offset
        keep = ((uv[:, 0] >= 0) & (uv[:, 0] < width) &
                (uv[:, 1] >= 0) & (uv[:, 1] < height))
        active = np.asarray(active_source_supervision_mask(annotation), dtype=bool)
        keep &= active
        if annotation.get("source_visible_mask") is not None:
            # A human can correctly indicate a boundary behind a person or a
            # broadcast graphic. Preserve every authoritative position in the
            # annotation, but supervise visible source pixels only. This mask
            # records source visibility, never a network confidence decision.
            visible = np.asarray(annotation["source_visible_mask"])
            if visible.dtype != np.bool_ or visible.shape != (len(uv),):
                raise ValueError("Source visibility must be one explicit boolean per original UV")
            keep &= visible
        if not keep.any():
            continue
        if "normal" in annotation and annotation["normal"] is not None:
            normal = np.asarray(annotation["normal"], float)
        elif len(uv) == 1:
            if annotation.get("state") != "user_confirmed_visual_supervision":
                raise ValueError("A reviewed single-point interval needs its measured source normal")
            # Direct human UV classification needs no invented tangent. This
            # zero normal is not used by the direct confirmed-position loss.
            normal = np.zeros_like(uv)
        else:
            tangent = np.gradient(uv, axis=0)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        if normal.shape != uv.shape:
            raise ValueError("Source annotation normals must match its native points")
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        interval = np.asarray(annotation.get("interval_px", [-3, 3]), float)
        if interval.ndim == 0:
            interval = np.tile([-float(interval), float(interval)], (len(uv), 1))
        elif interval.shape == (2,):
            interval = np.tile(interval, (len(uv), 1))
        if interval.shape != (len(uv), 2):
            raise ValueError("Annotation support interval has incompatible shape")
        indices = np.flatnonzero(keep)
        if len(indices) > 96 and annotation.get("state") != "user_confirmed_visual_supervision":
            indices = indices[np.linspace(0, len(indices) - 1, 96).astype(int)]
        cropped = {**annotation, "uv": uv[indices], "normal": normal[indices],
                   "interval_px": interval[indices], "source_point_indices": indices}
        if annotation.get("source_integration_weights") is not None:
            cropped["source_integration_weights"] = np.asarray(annotation["source_integration_weights"])[indices]
        if annotation.get("source_visible_mask") is not None:
            cropped["source_visible_mask"] = visible[indices]
        if annotation.get("source_active_supervision_mask") is not None:
            cropped["source_active_supervision_mask"] = active[indices].tolist()
        result.append(cropped)
    return result


def interval_supervision(logits, annotations, classes):
    """Direct supervision at user-confirmed UVs; source-review interval loss.

    Each connected drawing is one correlated supervision group. Unlabelled
    regions do not become background and fitted geometry supplies no target.
    """
    import torch
    from torch.nn import functional as F
    from .vision_supervision import source_weighted_mean, reduce_source_group_losses
    c, h, w = logits.shape[1:]
    offsets = torch.arange(-24, 25, device=logits.device, dtype=torch.float32)
    losses, diagnostics = [], []
    for annotation in annotations:
        if annotation["kind"] not in classes:
            continue
        category = classes.index(annotation["kind"])
        uv = torch.as_tensor(annotation["uv"], device=logits.device, dtype=torch.float32)
        if annotation.get("state") == "user_confirmed_visual_supervision":
            # Nearby confirmed strokes must not compete in a normal-window
            # softmax: both sides of a circle can be correct within 24 pixels.
            grid = uv * uv.new_tensor([2 / (w - 1), 2 / (h - 1)]) - 1
            scores = F.grid_sample(logits.float(), grid[None, :, None], align_corners=True)[0, :, :, 0].T
            target = torch.full((len(uv),), category, device=logits.device, dtype=torch.long)
            loss, mass = source_weighted_mean(F.cross_entropy(scores, target, reduction="none"), annotation)
            losses.append((annotation, loss, mass))
            diagnostics.append(dict(group_id=annotation["group_id"], kind=annotation["kind"],
                                    sample_count=len(uv), supervision="direct_user_confirmed_native_UV",
                                    model_confidence_filter_used=False, competing_normal_window_used=False))
            continue
        normal = torch.as_tensor(annotation["normal"], device=logits.device, dtype=torch.float32)
        interval = torch.as_tensor(annotation["interval_px"], device=logits.device, dtype=torch.float32)
        pixels = uv[:, None] + normal[:, None] * offsets[None, :, None]
        inside = ((pixels[..., 0] >= 0) & (pixels[..., 0] < w) &
                  (pixels[..., 1] >= 0) & (pixels[..., 1] < h))
        grid = pixels * pixels.new_tensor([2 / (w - 1), 2 / (h - 1)]) - 1
        scores = F.grid_sample(logits.float(), grid[None], align_corners=True)[0]
        profile = scores[category].masked_fill(~inside, -1e4)  # [N, 49]
        support = ((offsets[None] >= interval[:, :1]) &
                   (offsets[None] <= interval[:, 1:]) & inside)
        valid = support.any(dim=1)
        if not valid.any():
            continue
        normal_log_probability = F.log_softmax(profile, dim=-1)
        log_mass = torch.logsumexp(normal_log_probability.masked_fill(~support, -1e4), dim=-1)
        localization, mass = source_weighted_mean(-log_mass[valid], annotation, valid)
        # A coarse locus is a bag containing a possible boundary location.
        # Its uncertainty width is not a dense positive paint/rail mask.
        # The maximizing pixel is a latent prediction, never a source label.
        class_log_probability = F.log_softmax(scores, dim=0)[category]
        semantic = -class_log_probability.masked_fill(~support, -1e4).max(dim=1).values
        semantic_mean, _ = source_weighted_mean(semantic[valid], annotation, valid)
        loss = localization + .5 * semantic_mean
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite source supervision")
        losses.append((annotation, loss, mass))
        selected = offsets[profile.argmax(dim=-1)]
        outside = torch.maximum(interval[:, 0] - selected, selected - interval[:, 1]).clamp_min(0)
        diagnostics.append(dict(group_id=annotation["group_id"], kind=annotation["kind"],
                                sample_count=int(valid.sum()),
                                mean_outside_drawing_support_px=float(outside[valid].mean().detach()),
                                native_pixel_accuracy_proven=False))
    if not losses:
        return logits.sum() * 0, []
    return reduce_source_group_losses(losses, logits.sum()*0), diagnostics


def primitive_supervision(logits, annotations):
    """Learn line/circle identity from source annotations, never guessed masks."""
    import torch
    from torch.nn import functional as F
    from .vision_supervision import source_weighted_mean, reduce_source_group_losses
    primitives = ("line", "circle", "other")
    h, w = logits.shape[-2:]
    losses = []
    for annotation in annotations:
        # Legacy unconfirmed source supports do not locate an image primitive.
        # User-confirmed annotations always train their stated primitive.
        if annotation.get("state") in ("user_trace_weak", "source_reviewed_weak_stroke"):
            continue
        kind = annotation.get("primitive")
        if kind not in primitives:
            continue
        uv = torch.as_tensor(annotation["uv"], device=logits.device, dtype=torch.float32)
        grid = uv * uv.new_tensor([2 / (w - 1), 2 / (h - 1)]) - 1
        sampled = F.grid_sample(logits.float(), grid[None, :, None], align_corners=True)[0, :, :, 0].T
        target = torch.full((len(uv),), primitives.index(kind), device=logits.device, dtype=torch.long)
        loss, mass = source_weighted_mean(F.cross_entropy(sampled, target, reduction="none"), annotation)
        losses.append((annotation, loss, mass))
    return reduce_source_group_losses(losses, logits.sum()*0)


def source_background_mask(sample, origin, size, device):
    """Only the clearly non-rink area above manually traced board tops.

    No model prediction supplies this mask. Unknown columns and a broad band
    near the rough drawing remain ignored. Ice and board advertising interiors
    are not automatically labelled background.
    """
    import torch
    height, width = (size, size) if np.isscalar(size) else size
    mask = np.zeros((height, width), bool)
    if sample.get("background_above_board_top") is False:
        return torch.from_numpy(mask).to(device)[None, None]
    for annotation in sample["annotations"]:
        if annotation["kind"] != "board_top":
            continue
        for point in annotation["uv"]:
            x = int(round(point[0] - origin[0]))
            high = int(point[1] - origin[1] - 80)
            if 0 <= x < width and high > 0:
                mask[:min(height, high), x] = True
    return torch.from_numpy(mask).to(device)[None, None]


def source_reviewed_region_masks(sample, origin, size, device):
    """Explicit original-image review regions, kept apart from inferred curves."""
    import torch
    height, width = (size, size) if np.isscalar(size) else size
    for region in sample.get("regions", []):
        x0, y0, x1, y1 = region["rectangle_xyxy"]
        x0, x1 = max(0, int(x0 - origin[0])), min(width, int(x1 - origin[0]))
        y0, y1 = max(0, int(y0 - origin[1])), min(height, int(y1 - origin[1]))
        mask = torch.zeros((1, 1, height, width), device=device, dtype=torch.bool)
        if x1 > x0 and y1 > y0:
            mask[:, :, y0:y1, x0:x1] = True
        rule = region.get("rgb_mask")
        if rule and bool(mask.any()):
            image = _image(sample["frame_path"],sample.get("frame_key"))
            ox, oy = map(int, origin)
            native = torch.from_numpy(image[oy:oy+height, ox:ox+width].copy()).to(device).float()
            r, g, b = native.unbind(-1)
            if rule["signal"] == "blue_excess":
                colour = (b-r >= rule["minimum_blue_red_255"]) & (b-g >= rule["minimum_blue_green_255"])
            elif rule["signal"] == "red_excess":
                colour = (r-b >= rule["minimum_red_blue_255"]) & (r-g >= rule["minimum_red_green_255"])
            else:
                raise ValueError("Only reviewed original RGB red/blue paint rules are supported")
            mask &= colour[None, None]
        yield region, mask


def semantic_exclusion_loss(logits, mask, excluded_kinds, classes, annotations=()):
    """Learn only a source-reviewed role's absence, leaving other roles open.

    An upper-board region can be definitely outside the rink contact boundary
    without every pixel being background. This loss does not assign a sibling
    semantic class or a primitive. Confirmed human positions retain priority.
    """
    import torch
    names = tuple(classes)
    excluded = tuple(excluded_kinds)
    if (not excluded or len(set(excluded)) != len(excluded) or
            any(kind not in names for kind in excluded) or len(excluded) >= len(names)):
        raise ValueError("Semantic exclusions require distinct known roles and at least one allowed role")
    if (logits.ndim != 4 or logits.shape[1] != len(names) or
            mask.shape != (logits.shape[0], 1, *logits.shape[-2:]) or
            mask.dtype != torch.bool or mask.device != logits.device):
        raise ValueError("Semantic exclusion masks must share the actual logits' source pixels")
    h, w = logits.shape[-2:]
    for annotation in annotations:
        if (annotation.get("state") != "user_confirmed_visual_supervision" or
                annotation.get("kind") not in excluded):
            continue
        uv = torch.as_tensor(annotation["uv"], device=logits.device, dtype=torch.float32)
        low = uv.floor().long()
        fraction = uv-low
        # Direct confirmed-position CE samples logits bilinearly. Protect all
        # source pixels with nonzero weight, including subpixel drawing points.
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            x, y = low[:, 0]+dx, low[:, 1]+dy
            weight = (fraction[:, 0] if dx else 1-fraction[:, 0]) * (fraction[:, 1] if dy else 1-fraction[:, 1])
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (weight > 0)
            if bool(inside.any()) and bool(mask[:, 0, y[inside], x[inside]].any()):
                raise ValueError("A reviewed semantic exclusion conflicts with confirmed human source positions")
    values = logits.float().permute(0, 2, 3, 1)[mask[:, 0]]
    stats = dict(supervised_pixels=len(values), excluded_kinds=list(excluded),
                 positive_semantic_class_assigned=False, primitive_class_assigned=False,
                 human_positions_filtered_by_model_confidence=False)
    if not len(values):
        return logits.sum()*0, stats
    allowed = [index for index, name in enumerate(names) if name not in excluded]
    loss = (torch.logsumexp(values, dim=1)-torch.logsumexp(values[:, allowed], dim=1)).mean()
    return loss, stats


def _crop(sequence, sample, size, rng, fixed=False):
    image_h, image_w = sequence.shape[1:3]
    eligible = [a for a in sample["annotations"] if len(a.get("uv", [])) > 1]
    regions = sample.get("regions", [])
    if regions and not fixed and rng.random() < .5:
        box = rng.choice(regions)["rectangle_xyxy"]
        point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    else:
        annotation = eligible[0] if fixed else rng.choice(eligible)
        uv = np.asarray(annotation["uv"], float)
        point = uv[len(uv) // 2] if fixed else uv[rng.randrange(len(uv))]
    x = int(np.clip(point[0] - size // 2, 0, image_w - size))
    y = int(np.clip(point[1] - size // 2, 0, image_h - size))
    return sequence[:, y:y + size, x:x + size].copy(), _annotations_in_crop(sample, (x, y), size), (x, y)


def _tensor(sequence, device):
    import torch
    return torch.from_numpy(sequence).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)[None] / 255


def _atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class RinkVisionModule:
    """Source-bound GPU inference with aligned memory and optional real cameras.

    Returned class maps are predictions; consumers retain their source identity,
    semantic uncertainty and training scope when proposing solver observations.
    Every class remains available. This interface does not hide failed regions.
    """
    def __init__(self, model_path, flow_weights, *, device="cuda", exposure_registry=None):
        import torch
        from .vision_network import RinkVisionNet
        from .vision_geometry import GPUFlow
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("Rink vision inference uses the configured GPU")
        self.device = torch.device(device)
        self.model_path = Path(model_path).resolve()
        model_bytes = self.model_path.read_bytes()
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        payload = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
        del model_bytes
        self.network = RinkVisionNet(**payload["model_config"]).to(self.device).eval()
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.flow = GPUFlow(flow_weights, device=device, num_updates=8)
        self.classes = tuple(payload["classes"])
        self.training_scope = payload["training_report"]
        self.previous_image = self.hidden = self.previous_binding = self.previous_result = None
        self.current_image = self.current_binding = self.exposure_input_hidden = self.last_motion = None
        self.current_prediction = None
        self.exposure_registry = load_source_clock_registry(exposure_registry)
        self.source_exposure_cache = {}

    def model_usage(self):
        """Publish selected weights and their retained source-regression status."""
        scope = self.training_scope
        return dict(path=str(self.model_path), sha256=self.model_sha256,
            training_step=scope.get('latest_step'), training_status=scope.get('status'),
            source_retention_failed=scope.get('source_retention_guard', {}).get('failed'),
            accuracy_accepted=scope.get('accuracy_accepted') is True,
            inference_selection_is_formal_model_promotion=False)

    def reset(self):
        self.previous_image = self.hidden = self.previous_binding = self.previous_result = None
        self.current_image = self.current_binding = self.exposure_input_hidden = self.last_motion = None
        self.current_prediction = None

    def _camera_prior(self, camera25, world_curves, source_binding, camera_source_binding,
                      native_hw, model_revision):
        import torch
        from torch.nn import functional as F
        from .vision_geometry import CameraShapeAdapter
        h, w = native_hw
        prior = torch.zeros((1, 1, len(self.classes), h, w), device=self.device)
        metadata = []
        shapes = []
        adapter = CameraShapeAdapter()
        for curve in world_curves:
            if curve["kind"] not in self.classes:
                raise ValueError("World curve has no typed landmark class")
            projected = adapter.project(
                curve["world_xyz"], camera25, geometry_id=curve["id"],
                model_revision=model_revision, source_binding=source_binding,
                camera_source_binding=camera_source_binding, native_hw=native_hw,
                segment_ids=curve.get("segment_ids"))
            pixels = torch.as_tensor(projected.pixels, device=self.device, dtype=torch.float32)
            normals = torch.full_like(pixels, float("nan"))
            segment_ids = torch.full((len(pixels),), -1, device=self.device, dtype=torch.int64)
            point_batches = []
            for segment_index, segment in enumerate(projected.segments):
                if len(segment) < 2:
                    continue
                points = pixels[list(segment)]
                tangent = torch.empty_like(points)
                tangent[0], tangent[-1] = points[1] - points[0], points[-1] - points[-2]
                if len(points) > 2:
                    tangent[1:-1] = (points[2:] - points[:-2]) * .5
                normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
                normal /= torch.linalg.vector_norm(normal, dim=1).clamp_min(1e-8)[:, None]
                normals[list(segment)] = normal
                segment_ids[list(segment)] = segment_index
                start, direction = points[:-1], points[1:] - points[:-1]
                safe = torch.where(direction.abs() > 1e-8, direction, torch.ones_like(direction))
                limits = start.new_tensor([w - 1, h - 1])
                t0, t1 = -start / safe, (limits - start) / safe
                parallel = direction.abs() <= 1e-8
                legal_parallel = (~parallel | ((start >= 0) & (start <= limits))).all(dim=1)
                # A parallel in-image axis imposes no interval constraint.
                amin = torch.where(parallel, start.new_full(start.shape, -float("inf")), torch.minimum(t0, t1))
                amax = torch.where(parallel, start.new_full(start.shape, float("inf")), torch.maximum(t0, t1))
                lo, hi = amin.amax(dim=1).clamp_min(0), amax.amin(dim=1).clamp_max(1)
                legal = legal_parallel & (hi >= lo)
                if not bool(legal.any()):
                    continue
                start = start[legal] + direction[legal] * lo[legal, None]
                direction = direction[legal] * (hi[legal] - lo[legal])[:, None]
                count = torch.linalg.vector_norm(direction, dim=1).ceil().long().clamp_min(1)
                steps = torch.arange(int(count.max()) + 1, device=self.device)
                t = steps[None].float() / count[:, None]
                points = start[:, None] + direction[:, None] * t[:, :, None]
                point_batches.append(points[(steps[None] <= count[:, None])])
            if point_batches:
                xy = torch.cat(point_batches).round().long()
                inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                xy = xy[inside]
                prior[0, 0, self.classes.index(curve["kind"]), xy[:, 1], xy[:, 0]] = 1
            metadata.append(projected.metadata)
            shapes.append(dict(id=curve["id"], kind=curve["kind"],
                               primitive=curve.get("primitive", "unresolved"),
                               number=curve.get("number"),
                               extent=curve.get("extent"),
                               world_xyz=torch.as_tensor(curve["world_xyz"], device=self.device,
                                                         dtype=torch.float64),
                               segments=projected.segments,
                               pixels=pixels, normals=normals, segment_ids=segment_ids,
                               valid=torch.as_tensor(projected.valid, device=self.device),
                               inside_native=torch.as_tensor(projected.inside_native, device=self.device),
                               search_radius_px=float(curve.get("search_radius_px", 32)),
                               source_binding=projected.metadata["source_binding"],
                               metadata=projected.metadata))
        prior = F.max_pool2d(prior[0], 7, stride=1, padding=3)[None]
        return prior, metadata, shapes

    def measure_shapes(self, rgb_tensor, result, shapes):
        """Batch native profiles by class, retaining each physical path's identity."""
        import torch
        from .vision_geometry import measure_boundary_profiles
        grouped = {}
        for shape in shapes:
            if shape["primitive"] == "points":
                continue  # Independent material points have no invented edge/normal.
            grouped.setdefault((shape["kind"], shape["search_radius_px"]), []).append(shape)
        for (kind, radius), members in grouped.items():
            counts = [len(s["pixels"]) for s in members]
            ids, segment_base = [], 0
            for shape in members:
                original = shape["segment_ids"]
                ids.append(torch.where(original >= 0, original + segment_base, original))
                segment_base += max(1, int(original.max()) + 1)
            profiles = measure_boundary_profiles(
                rgb_tensor, result["class_probabilities"],
                torch.cat([s["pixels"] for s in members]),
                torch.cat([s["normals"] for s in members]),
                class_index=self.classes.index(kind), search_radius=radius,
                segment_ids=torch.cat(ids), rgb_source_binding=result["source_binding"],
                probability_source_binding=result["prediction_source_binding"],
                projection_source_binding=result["source_binding"],
                shadow_probabilities=result["class_probabilities"][:, self.classes.index("shadow_or_reflection"):
                                                                   self.classes.index("shadow_or_reflection")+1],
                shadow_source_binding=result["prediction_source_binding"])
            total, cursor = sum(counts), 0

            def sliced(value, start, end, key=None):
                if isinstance(value, dict):
                    return {k: sliced(v, start, end, k) for k, v in value.items()}
                if (isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == total
                        and key not in ("offsets_native_px", "rgb_gradient_offsets_native_px")):
                    return value[start:end]
                return value

            for shape, count in zip(members, counts):
                shape["source_profiles"] = sliced(profiles, cursor, cursor + count)
                cursor += count
        return shapes

    def predict(self, rgb, source_binding, *, camera25=None, world_curves=None,
                camera_source_binding=None, model_revision=None, reset=False):
        import torch
        from torch.nn import functional as F
        from .vision_geometry import _source_binding, warp
        binding = _source_binding(source_binding)
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must be unmodified decoded RGB uint8 pixels")
        h, w = image.shape[:2]
        if "native_hw" in binding and binding["native_hw"] != [h, w]:
            raise ValueError("Source image resolution differs from its binding")
        binding["native_hw"] = [h, w]
        clock_key = (binding["source_sha256"], str(Fraction(binding["time_base"])))
        if clock_key not in self.source_exposure_cache:
            self.source_exposure_cache[clock_key] = confirmed_exposure_groups(
                *clock_key, registry=self.exposure_registry)
        exposure_map, exposure_proof = self.source_exposure_cache[clock_key]
        exposure = exposure_map.get(binding["source_pts"])
        if exposure:
            if ("canonical_exposure_pts" in binding and
                    binding["canonical_exposure_pts"] != exposure["canonical"]):
                raise ValueError("Caller exposure identity conflicts with the original pixel proof")
            binding.update(canonical_exposure_pts=exposure["canonical"],
                           official_exposure_pts=exposure["official_pts"],
                           official_exposure_time_base=exposure["official_time_base"],
                           exposure_proof_sha256=exposure_proof["proof_sha256"])
        elif binding.get("canonical_exposure_pts") not in (None, binding["source_pts"]):
            raise ValueError("An unverified repeated-exposure alias cannot update memory")
        if reset:
            self.reset()
        old = self.previous_binding
        if old and (any(old.get(k) != binding.get(k) for k in
                        ("source_sha256", "time_base", "shot_id", "installation_id", "native_hw")) or
                    int(binding["source_pts"]) < int(old["source_pts"])):
            self.reset()
            old = None
        tensor = torch.from_numpy(image.copy()).to(self.device, dtype=torch.float32).permute(2, 0, 1)[None] / 255
        if (self.current_binding and binding["source_pts"] == self.current_binding["source_pts"]
                and not torch.equal(tensor, self.current_image)):
            raise ValueError("Different RGB pixels were supplied for the same displayed source frame identity")
        self.current_image, self.current_binding, self.last_motion = tensor, binding, None
        same_stamp = bool(old and binding["source_pts"] == old["source_pts"])
        repeated = bool(old and exposure and
                        binding["canonical_exposure_pts"] == old.get("canonical_exposure_pts"))
        if same_stamp and not torch.equal(tensor, self.previous_image):
            raise ValueError("Different RGB pixels were supplied for the same original frame identity")
        # Decode/read each displayed frame's actual RGB. A verified repeated
        # exposure reuses the recurrent input, without advancing physical memory.
        hidden = self.exposure_input_hidden if same_stamp or repeated else self.hidden
        motion = None
        prior = None
        prior_metadata = []
        shapes = []
        with torch.inference_mode():
            if hidden is not None and not (same_stamp or repeated):
                pair = self.flow.compute(self.previous_image, tensor, analysis_hw=(384, 640),
                                         source0=old, source1=binding)
                mh, mw = hidden.shape[-2:]
                flow = F.interpolate(pair.backward, size=(mh, mw), mode="bilinear", align_corners=False)
                flow *= flow.new_tensor([mw / w, mh / h])[None, :, None, None]
                hidden, inside = warp(hidden, flow)
                valid = F.interpolate(pair.valid_backward.float(), size=(mh, mw), mode="nearest")
                hidden = hidden * valid.to(hidden.dtype) * inside.to(hidden.dtype)
                motion = pair.metadata
                self.last_motion = pair
            if world_curves is not None:
                if camera25 is None or camera_source_binding is None or not model_revision:
                    raise ValueError("World shapes require the actual matching camera and model revision")
                prior, prior_metadata, shapes = self._camera_prior(
                    camera25, world_curves, binding, camera_source_binding, (h, w), model_revision)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.network(tensor[:, None], hidden=hidden, geometry_prior=prior)
            result = dict(class_probabilities=output["logits"][:, 0].float().softmax(dim=1),
                          class_log_probabilities=output["logits"][:, 0].float().log_softmax(dim=1),
                          primitive_probabilities=output["primitive_logits"][:, 0].float().softmax(dim=1),
                          source_binding=binding, prediction_source_binding=binding,
                          model_sha256=self.model_sha256,
                          classes=self.classes, native_hw=[h, w], motion=motion,
                          camera_shape_priors=prior_metadata,
                          projected_shapes=shapes,
                          probabilities_conditioned_on_geometry=prior is not None,
                          class_log_probabilities_source="raw_network_logits_float32_log_softmax",
                          log_probability_profile_policy="interpolate_each_class_then_compute_class_vs_rest_log_odds",
                          geometry_prior_is_pixel_truth=False,
                          accuracy_accepted=False,
                          memory_update=("reused_confirmed_exposure" if repeated else
                                         "reused_same_source_frame" if same_stamp else
                                         "image_aligned" if old else "shot_reset"),
                          new_observation=not (same_stamp or repeated))
            result["source_clock_evidence"] = exposure_proof
            if shapes:
                self.measure_shapes(tensor, result, shapes)
            if not (same_stamp or repeated):
                self.exposure_input_hidden = hidden
                self.hidden = output["hidden"].detach()
                self.previous_image = tensor
                self.previous_binding = binding
                self.previous_result = result
        self.current_prediction = result
        return result


def _source_json(value):
    """Serialize native measurements; unavailable numbers remain explicitly null."""
    import torch
    if isinstance(value, torch.Tensor):
        return _source_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _source_json(value.tolist())
    if isinstance(value, dict):
        return {key: _source_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_source_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def confirmed_source_guides(sample, *, role_definitions=None, return_packet=False):
    """Use the shared source-role adapter without deriving material from shape.

    Human coordinates remain intact. A circle can be a printed rim, a filled
    disc boundary, or a painted axis; unresolved material roles stay explicit.
    This adapter is a source search input and creates no pixel measurements.
    """
    from .vision_geometry import source_annotations_to_guides
    binding = dict(source_sha256=sample["source_hash"], source_pts=int(sample["source_pts"]),
                   time_base=sample["time_base"],
                   shot_id=sample.get("shot_id") or f"single_source_frame_{int(sample['source_pts'])}")
    if sample.get("frame_size_wh"):
        binding["native_hw"] = list(reversed(sample["frame_size_wh"]))
    annotations = [a for a in sample["annotations"]
                   if a.get("state") in ("user_confirmed_visual_supervision", "user_trace_weak")
                   or a.get("manual_annotation_ground_truth") is True
                   or (isinstance(a.get("authority"), dict)
                       and a["authority"].get("human_authority") is True)]
    packet = source_annotations_to_guides(
        annotations, source_binding=binding, annotation_source_binding=binding,
        role_definitions=role_definitions, default_search_radius_px=24.)
    for guide in packet["guides"]:
        authority = guide.get("identity_authority")
        alias = authority.get("user_text_alias") if isinstance(authority, dict) else None
        guide.update(expected_id=guide["id"],
                     display_label=alias or guide["annotation_feature_id"])
    return packet if return_packet else packet["guides"]


def source_ignored_region_box(region, native_hw):
    """Normalize stored native rectangles and historical full-height x ranges."""
    h, w = native_hw
    if "rectangle_xyxy" in region:
        box = region["rectangle_xyxy"]
    elif "x_range_source_px" in region:
        left, right = region["x_range_source_px"]
        box = [left, 0, right, h]
    else:
        raise ValueError("Source exclusions require a native rectangle or x range")
    values = np.asarray(box, float)
    if values.shape != (4,) or not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Source exclusions require a finite ordered native rectangle")
    x0, y0 = np.floor(values[:2]).astype(int)
    x1, y1 = np.ceil(values[2:]).astype(int)
    return tuple(int(value) for value in (max(0, x0), max(0, y0), min(w, x1), min(h, y1)))


def source_guide_visibility_override_crop(guide, binding, native_hw):
    """A later same-frame RGB review can retire legacy ignores for this guide.

    This cannot authorize a pixel measurement or override another feature's
    exclusions. It expires on the next source frame, including temporal hints.
    """
    from .vision_geometry import source_visibility_matches_frame
    evidence = guide.get("source_visibility_evidence", {})
    if (guide.get("origin") != "user_confirmed_visual_supervision"
            or evidence.get("original_legacy_sample_ignored_x_ranges_are_historical_and_do_not_veto_this_new_source_review") is not True
            or evidence.get("model_confidence_used") is not False):
        return None
    if not source_visibility_matches_frame(binding, evidence):
        return None
    visible = np.asarray(guide.get("source_visible_mask", []))
    points = np.asarray(guide["points_uv"], float)
    if visible.dtype != np.bool_ or visible.shape != (len(points),) or not visible.all():
        return None
    box = source_ignored_region_box({"rectangle_xyxy": evidence["source_crop_native_xyxy"]}, native_hw)
    x0, y0, x1, y1 = box
    if not len(points) or not ((points[:, 0] >= x0-.5) & (points[:, 0] < x1-.5)
                              & (points[:, 1] >= y0-.5) & (points[:, 1] < y1-.5)).all():
        raise ValueError("Reviewed source visibility crop must contain its own native guide")
    return box


class SourceRinkReader:
    """Read actual source pixels. A search hint never becomes the measurement."""
    def __init__(self, config_path, *, model_path=None, original_fit_request_path=None, native_geometry=None,
                 source_reidentification=None):
        self.config, self.data = configure(config_path)
        self.vision = RinkVisionModule(Path(model_path) if model_path else ROOT / self.config["final_model"],
                                       ROOT / self.config["flow_weights"], device=self.config["device"],
                                       exposure_registry=self.config.get("source_clock_registry"))
        if native_geometry is None and self.config.get("native_geometry"):
            from .vision_native import NativeSourceGeometry
            spec = self.config["native_geometry"]
            native_geometry = NativeSourceGeometry(ROOT / spec["scene"], ROOT / spec["library"])
        self.native_geometry = native_geometry
        self.material_reader = None
        if native_geometry is not None:
            from .vision_material_reader import CurrentNativeMaterialReader
            self.material_reader = CurrentNativeMaterialReader(native_geometry, device=self.vision.device)
        self.board_material_authority = None
        if self.config.get("board_foot_material_contract"):
            from .vision_source_topology import board_foot_material_authority
            self.board_material_authority = board_foot_material_authority(ROOT / self.config["board_foot_material_contract"])
        self.reidentifier = None
        self.reidentification_active = False
        source_reidentification = source_reidentification or self.config.get('source_reidentification')
        if source_reidentification:
            from .vision_reidentify import AcceptedSourceReidentifier
            self.reidentifier = AcceptedSourceReidentifier(source_reidentification, self.native_geometry, device=self.vision.device)
        self.instance_provider = None
        self.active_guides = []
        self.active_global_exclusion = None
        self.active_binding = None
        self.original_fit_reader = None
        if original_fit_request_path is not None:
            from .vision_source_fit import OriginalFITSourceReader
            self.original_fit_reader = OriginalFITSourceReader(original_fit_request_path)

    def read(self, rgb, source_binding, *, guides_2d=None, ignored_regions=(), reset=False,
             source_region_constraints=None, precomputed_prediction=None,
             extract_source_components=True, current_source_masks_by_kind=None):
        import torch
        from torch.nn import functional as F
        from .vision_geometry import extract_source_primitives, guided_source_reading, coherent_named_source_reading
        from .vision_geometry import warp, _same_frame
        from .vision_feedback import begin_read, check_guide_operators, finish_read
        feedback = begin_read(source_binding, guides_2d if guides_2d is not None else self.active_guides)
        if precomputed_prediction is None:
            prediction = self.vision.predict(rgb, source_binding, reset=reset)
        else:
            if precomputed_prediction is not self.vision.current_prediction:
                raise ValueError("Reader requires the current prediction from its own vision module")
            prediction = precomputed_prediction
            expected = {**source_binding, "native_hw": list(rgb.shape[:2])}
            _same_frame(expected, prediction["source_binding"])
            if reset and prediction["memory_update"] != "shot_reset":
                raise ValueError("An explicit cut requires a freshly reset prediction")
            native = torch.from_numpy(np.asarray(rgb).copy()).to(self.vision.device).permute(2, 0, 1)[None].float()/255
            if not torch.equal(native, self.vision.current_image):
                raise ValueError("Precomputed prediction belongs to different original RGB")
        binding = prediction["source_binding"]
        with torch.inference_mode():
            h, w = rgb.shape[:2]
            if prediction["memory_update"] == "shot_reset":
                self.active_guides, self.active_binding, self.active_global_exclusion = [], None, None
                self.active_front_face = None
            explicit_guides = guides_2d is not None
            if explicit_guides:
                guides_2d = [{**g} for g in guides_2d]
            else:
                guides_2d = [{**g} for g in self.active_guides]
                pair = self.vision.last_motion
                for guide in guides_2d:
                    points = torch.as_tensor(guide["points_uv"], device=self.vision.device, dtype=torch.float32)
                    if pair is not None:
                        grid = points * points.new_tensor([2/(w-1),2/(h-1)])-1
                        movement = F.grid_sample(pair.forward, grid[None,:,None], align_corners=True)[0,:,:,0].T
                        flow_valid = F.grid_sample(pair.valid_forward.float(), grid[None,:,None],
                                                   mode="nearest", align_corners=True)[0,0,:,0] > .5
                        if guide.get("normals") is not None:
                            normal = torch.as_tensor(guide["normals"], device=self.vision.device, dtype=torch.float32)
                            tangent = torch.stack((normal[:, 1], -normal[:, 0]), 1)
                            tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-8)
                            tip_grid = (points+tangent) * points.new_tensor([2/(w-1),2/(h-1)])-1
                            tip_move = F.grid_sample(pair.forward, tip_grid[None,:,None], align_corners=True)[0,:,:,0].T
                            moved_tangent = tangent + tip_move - movement
                            guide["normals"] = torch.stack((-moved_tangent[:, 1], moved_tangent[:, 0]), 1)
                        points = points + movement
                        guide["temporal_flow_supported"] = flow_valid
                        if guide.get("source_exclusion_mask") is not None:
                            mask, inside = warp(guide["source_exclusion_mask"].float(), pair.backward)
                            guide["source_exclusion_mask"] = (mask > .5) & inside
                    if guide.get("rgb_transition_polarity") and guide.get("identity_authority"):
                        if not guide.get("transition_authority"):
                            guide["transition_authority"] = guide["identity_authority"]
                    guide.update(points_uv=points, source_binding=binding,
                                 origin="temporal_source_search_hint",
                                 identity_confirmed=bool(guide.get("identity_authority")),
                                 hint_previous_source_binding=self.active_binding,
                                 hint_is_source_measurement=False)
                    if guide.get("source_exclusion_mask") is not None:
                        guide["source_exclusion_binding"] = binding
            reidentification = None
            material_matches = None
            instances = None
            people, graphics = [], dict(candidates=[])
            if getattr(self, 'reidentifier', None) is not None:
                if prediction['memory_update'] == 'shot_reset':
                    self.reidentification_active = not bool(guides_2d)
                exclusion, people, graphics = self.reidentifier.exclusions(rgb, binding)
                current_source_masks_by_kind = dict(current_source_masks_by_kind or {})
                for kind in ('board_top','ice_contact','ad_boundary','ice_paint'):
                    supplied = current_source_masks_by_kind.get(kind)
                    fresh = torch.as_tensor(exclusion, device=self.vision.device)
                    current_source_masks_by_kind[kind] = fresh if supplied is None else fresh | torch.as_tensor(supplied, device=self.vision.device)
                if self.reidentification_active:
                    recovered, reidentification = self.reidentifier.search(rgb, binding,
                        reset=prediction['memory_update']=='shot_reset', person_boxes=people,
                        prediction=prediction, classes=self.vision.classes, exclusion=exclusion, flow_engine=getattr(self.vision,'flow',None))
                    recovered_ids = {g['id'] for g in recovered}
                    guides_2d = [g for g in guides_2d if g['id'] not in recovered_ids] + recovered
                    material_matches=reidentification
                    reidentification['current_occlusion'] = dict(person_boxes=people,
                        graphic_boxes=[g['box_xyxy'] for g in graphics.get('candidates',[])],
                        excluded_pixels=int(exclusion.sum()), person_model_sha256=self.reidentifier.person.weights_sha256)
                elif prediction['memory_update']=='shot_reset' and getattr(self,'config',{}).get('source_instances'):
                    material_matches=self.reidentifier.matcher.read(rgb,binding,person_boxes=people,
                        overlay_boxes=graphics.get('candidates',[]),
                        source_identity_anchors=self.reidentifier.accepted_ad_anchors(guides_2d,binding))
                    native_ads,native_identities=self.reidentifier.advertisement_guides(rgb,binding,material_matches,native_only=True)
                    if native_ads:
                        preserved=[]
                        for guide in guides_2d:
                            definition=self.native_geometry.lookup(guide['id'])
                            physical=(definition or {}).get('instance_id','').rsplit('/',1)[-1]
                            if physical in native_identities and guide.get('kind')=='ad_boundary':
                                preserved.append(dict(id=guide['id'],points_uv=guide['points_uv'],
                                    identity_authority=guide.get('identity_authority'),source_binding=guide.get('source_binding')))
                        replaced_paths={g['id'] for g in native_ads}
                        replaced={row['id'] for row in preserved if self.native_geometry.lookup(row['id'])['id'] in replaced_paths}
                        guides_2d=[g for g in guides_2d if g['id'] not in replaced]+native_ads
                        material_matches['preserved_accepted_source_navigation']=preserved
                        material_matches['current_native_registered_correspondence_used']=True
            if getattr(self, 'config', {}).get('source_instances'):
                from .vision_instances import Sam2SourceInstances
                from .vision_review import source_region_constraints as instance_constraints
                if self.instance_provider is None:
                    self.instance_provider = Sam2SourceInstances(ROOT / self.config['backbone'],
                        device=self.vision.device, reread_rgb=False)
                instances = self.instance_provider.read_auto(rgb, binding,
                    class_probabilities=prediction['class_probabilities'],classes=self.vision.classes,
                    probabilities_conditioned_on_geometry=False,person_boxes=people,
                    overlay_boxes=graphics.get('candidates',[]),reset=prediction['memory_update']=='shot_reset')
                if material_matches and material_matches.get('prompts'):
                    from .vision_review import retain_same_source_refinement_evidence
                    from .vision_instances import source_correspondence_identity_verified
                    # One authenticated native physical instance per SAM object;
                    # dozens of reference views cannot create duplicate objects.
                    chosen={}
                    for prompt in material_matches['prompts']:
                        physical=prompt.get('physical_id')
                        from .vision_instances import HUMAN_IDENTITIES
                        authority=prompt.get('identity_authority',{}) or {}
                        human=authority.get('kind') in HUMAN_IDENTITIES and authority.get('reference')
                        if physical and (human or source_correspondence_identity_verified(authority,binding,physical)):
                            if physical not in chosen or len(prompt.get('actual_pattern_keypoints_uv',[]))>len(chosen[physical].get('actual_pattern_keypoints_uv',[])):
                                chosen[physical]=dict(prompt,object_id=2000+int(physical.removeprefix('Ad')))
                    if chosen:
                        before=self.instance_provider.frame_index
                        instances=retain_same_source_refinement_evidence(instances,
                            self.instance_provider.read(rgb,binding,prompts=list(chosen.values()),reset=False))
                        if self.instance_provider.frame_index!=before:raise ValueError('Same-RGB print refinement advanced exposure')
                    instances['current_native_ad_reference_search']=dict(
                        verified_physical_ids=sorted(chosen),reference_matching_is_boundary_measurement=False)
                from .vision_review import current_source_pixel_exclusion
                pixel_exclusion,pixel_evidence=current_source_pixel_exclusion(
                    instances,binding,native_hw=rgb.shape[:2],device=self.vision.device)
                instances['current_source_exclusion_mask']=pixel_exclusion
                instances['current_source_pixel_exclusion_evidence']=pixel_evidence
                instances['unreviewed_occluder_predictions_used_as_RGB_veto']=bool(pixel_evidence['regions'])
                current_source_masks_by_kind=dict(current_source_masks_by_kind or {})
                for kind in ('board_top','ice_contact','ad_boundary','ice_paint'):
                    prior=current_source_masks_by_kind.get(kind)
                    current_source_masks_by_kind[kind]=pixel_exclusion if prior is None else pixel_exclusion|torch.as_tensor(prior,device=self.vision.device)
                constraints = instance_constraints(instances,native_hw=rgb.shape[:2],device=self.vision.device)
                if source_region_constraints is None:
                    source_region_constraints = constraints
                else:
                    old_masks=dict(source_region_constraints.get('masks',{}))
                    for kind, mask in constraints.get('masks',{}).items():
                        old_masks[kind]=old_masks.get(kind,torch.ones_like(mask))&mask
                    source_region_constraints=dict(source_region_constraints,masks=old_masks)
            # Re-read the top-deck/ice interface on this exact RGB before any ad read.
            from .vision_near_board import read_near_board
            near_board = read_near_board(rgb, binding, guides=guides_2d,
                gray_history=getattr(self, "config", {}).get("near_board_source_evidence", {}).get("gray_history"))
            near_exclusion = torch.as_tensor(near_board["ad_exclusion_mask"], device=self.vision.device)[None, None]
            if instances is not None:
                from .vision_review import reread_source_advertisements
                mask=np.asarray(near_board['ad_exclusion_mask'],bool).copy()
                mask|=instances['current_source_exclusion_mask'].detach().cpu().numpy()
                for kind in ('ad_boundary',):
                    current=(current_source_masks_by_kind or {}).get(kind)
                    if current is not None:mask|=torch.as_tensor(current).reshape(h,w).cpu().numpy()
                instances['current_source_exclusion_mask']=mask
                instances=reread_source_advertisements(rgb,instances,self.instance_provider,near_board=near_board)
                instances['named_output_requires_current_native_retained_path']=True
            if bool(near_exclusion.any()):
                source_region_constraints = dict(source_region_constraints or {}, source_binding=binding)
                regions = dict(source_region_constraints.get("masks", {}))
                regions["ad_boundary"] = regions.get("ad_boundary", torch.ones_like(near_exclusion)) & ~near_exclusion
                source_region_constraints["masks"] = regions
                source_region_constraints["metadata"] = dict(source_region_constraints.get("metadata", {}),
                    actual_top_face_exterior_excluded=True, model_board_thickness=0)
            if near_board.get("qualified_ice_visible_mask") is not None:
                source_region_constraints = dict(source_region_constraints or {}, source_binding=binding)
                regions = dict(source_region_constraints.get("masks", {}))
                ice_visible = torch.as_tensor(near_board["qualified_ice_visible_mask"],device=self.vision.device)[None,None]
                regions["ice_paint"] = regions.get("ice_paint",torch.ones_like(ice_visible)) & ice_visible
                source_region_constraints["masks"] = regions
            geometry_report = None
            if getattr(self, "native_geometry", None) is not None:
                configured_event = getattr(self, "config", {}).get("native_geometry", {}).get("event_id")
                if feedback.event_id and configured_event and feedback.event_id != configured_event:
                    raise ValueError("Source event differs from the configured current native geometry")
                guides_2d, geometry_report = self.native_geometry.bind(guides_2d)
            operator_conflicts = check_guide_operators(feedback, guides_2d)
            global_mask = torch.zeros((1, 1, h, w), device=self.vision.device, dtype=torch.bool)
            if not explicit_guides and self.active_global_exclusion is not None:
                pair = self.vision.last_motion
                if pair is None:
                    global_mask = self.active_global_exclusion
                else:
                    mask, inside = warp(self.active_global_exclusion.float(), pair.backward)
                    global_mask = (mask > .5) & inside
            feature_masks = {}
            for region in ignored_regions:
                x0, y0, x1, y1 = source_ignored_region_box(region, (h, w))
                ids = region.get("applies_only_to_feature_ids")
                if ids:
                    for name in ids:
                        mask = feature_masks.setdefault(name, torch.zeros_like(global_mask))
                        mask[:, :, y0:y1, x0:x1] = True
                else:
                    global_mask[:, :, y0:y1, x0:x1] = True
            for guide in guides_2d:
                current_mask = (current_source_masks_by_kind or {}).get(guide.get("kind"))
                if current_mask is not None:
                    current_mask = torch.as_tensor(current_mask, device=self.vision.device)
                    if current_mask.shape == (h, w):
                        current_mask = current_mask[None, None]
                    if current_mask.shape != (1, 1, h, w) or current_mask.dtype != torch.bool:
                        raise ValueError("Current source masks must be native boolean masks")
                    if guide.get("exclusion_origin") == "current_source_mask":
                        guide["source_exclusion_mask"] = current_mask
                    elif guide.get("source_exclusion_mask") is not None:
                        guide["source_exclusion_mask"] = current_mask | guide["source_exclusion_mask"]
                    else:
                        guide["source_exclusion_mask"] = current_mask
                    guide.update(source_exclusion_binding=binding, exclusion_origin="current_source_mask")
                # Preserve legacy exclusions for every other feature. The new
                # explicitly reviewed source crop applies only to its guide.
                mask = global_mask.clone()
                if guide.get("source_exclusion_mask") is not None:
                    _same_frame(binding, guide["source_exclusion_binding"])
                    mask |= guide["source_exclusion_mask"]
                if guide["id"] in feature_masks:
                    mask |= feature_masks[guide["id"]]
                crop = source_guide_visibility_override_crop(guide, binding, (h, w))
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    mask[:, :, y0:y1, x0:x1] = False
                if guide.get("kind") in ("ad_boundary", "ice_paint"):
                    mask |= near_exclusion
                if guide.get("kind") == "ice_paint" and near_board.get("qualified_ice_visible_mask") is not None:
                    mask |= ~torch.as_tensor(near_board["qualified_ice_visible_mask"],device=self.vision.device)[None,None]
                guide.update(source_exclusion_mask=mask, source_exclusion_binding=binding)
            guided = (guided_source_reading(
                self.vision.current_image, prediction["class_probabilities"], guides_2d,
                classes=self.vision.classes, rgb_source_binding=binding,
                probability_source_binding=prediction["prediction_source_binding"],
                guide_source_binding=binding, probabilities_conditioned_on_geometry=False,
                source_exclusion_mask=None, exclusion_source_binding=binding)
                if guides_2d else dict(readings=[], metadata={}))
            if guides_2d:
                guided = coherent_named_source_reading(guided)
            board_material = None
            if getattr(self, "board_material_authority", None) and guides_2d:
                from .vision_board_reader import refine_board_readings
                board_material = refine_board_readings(rgb, binding, guides_2d, guided,
                    authority=self.board_material_authority, feedback=feedback, prediction=prediction,
                    classes=self.vision.classes, device=self.vision.device, near_board=near_board)
            if getattr(self, "board_material_authority", None):
                from .vision_board_reader import (board_face_admission,filter_ad_readings,
                    dense_front_face_admission,continue_front_face_admission)
                admitted_face, face_receipt = board_face_admission(guides_2d,guided,(h,w),device=self.vision.device)
                board_exclusion = global_mask[0,0].cpu().numpy().copy()
                for kind in ('board_top','ice_contact','ad_boundary'):
                    current = (current_source_masks_by_kind or {}).get(kind)
                    if current is not None:
                        board_exclusion |= torch.as_tensor(current).reshape(h,w).cpu().numpy()
                board_exclusion |= near_exclusion[0,0].cpu().numpy()
                dense_face, dense_receipt, dense_packet = dense_front_face_admission(
                    rgb, binding, guides_2d, guided, board_exclusion, device=self.vision.device)
                admitted_face |= dense_face
                face_receipt['dense_current_RGB'] = dense_receipt
                near_board['current_source_board_measurement'] = dense_packet
                if near_board.get("qualified_vertical_face_mask") is not None:
                    admitted_face = torch.as_tensor(near_board["qualified_vertical_face_mask"],device=self.vision.device)[None,None]
                    face_receipt.update(current_RGB_material_face_reference=near_board["source_reference"],
                        source_qualification="existing_exact_native_generation")
                face_exclusion=torch.as_tensor(board_exclusion,device=self.vision.device)[None,None]
                if any(g.get('source_board_view')=='front_face' for g in guides_2d):
                    admitted_face,continued_face=continue_front_face_admission(admitted_face,
                        getattr(self,'active_front_face',None),binding,self.active_binding,
                        self.vision.last_motion,face_exclusion)
                    face_receipt['continuous_verified_front_face']=continued_face
                admitted_face &= ~face_exclusion
                if prediction['new_observation']:self.active_front_face=admitted_face.detach().clone()
                near_board["vertical_face_admission_mask"] = admitted_face[0,0].cpu().numpy()
                filter_ad_readings(guides_2d,guided,admitted_face)
                source_region_constraints = dict(source_region_constraints or {},source_binding=binding)
                masks = dict(source_region_constraints.get("masks", {}))
                masks["ad_boundary"] = masks.get("ad_boundary",torch.ones_like(admitted_face)) & admitted_face
                source_region_constraints["masks"] = masks
                source_region_constraints["metadata"] = dict(source_region_constraints.get("metadata",{}),
                    vertical_board_face_admission=face_receipt)
                if board_material is not None:board_material["advertisement_face_admission"] = face_receipt
            material_result = None
            if getattr(self, 'material_reader', None) is not None:
                material_result = self.material_reader.refine_ice(rgb, binding, guides_2d, guided,
                    near_board=near_board, global_exclusion=global_mask,
                    source_instances=instances, person_boxes=people,
                    reset=prediction['memory_update']=='shot_reset')
                material_result['advertisements'] = self.material_reader.refine_print(
                    rgb, binding, guides_2d, guided, near_board=near_board)
            if extract_source_components:
                source = extract_source_primitives(
                    self.vision.current_image, prediction["class_probabilities"], classes=self.vision.classes,
                    rgb_source_binding=binding, probability_source_binding=prediction["prediction_source_binding"],
                    probabilities_conditioned_on_geometry=False,
                    source_region_constraints=source_region_constraints)
                source["full_frame_component_extraction_performed"] = True
            else:
                # Intermediate application frames still reread the current
                # RGB along transported guides below. Absence of a global
                # search must not be reported as a failed detection.
                source = dict(source_binding=binding, native_hw=[h, w], components=[],
                    full_frame_component_extraction_performed=False,
                    component_search_status="not_performed_on_intermediate_tracking_frame",
                    source_region_rejection_masks={}, probabilities_conditioned_on_geometry=False,
                    fitted_or_flow_pixels_used_as_measurements=False)
            if material_result is not None:
                material_result['unassigned_material_search'] = self.material_reader.read_unassigned_material(
                    rgb, binding, prediction, source, flow_pair=self.vision.last_motion,
                    reset=prediction['memory_update']=='shot_reset',guides=guides_2d)
                material_result['output_scope'] = self.native_geometry.check_output(guides_2d, guided)
            for reading, guide in zip(guided["readings"], guides_2d):
                reading["display_label"] = guide.get("display_label", reading.get("physical_id"))
                reading["original_source_group_id"] = guide.get("source_group_id")
                reading["original_dependence_groups_per_point"] = guide.get("original_dependence_groups_per_point")
                reading["source_group_id"] = (str(binding["source_sha256"])+":"+str(binding["time_base"])+":"+
                    str(binding["source_pts"])+":"+str(guide["id"]))
                reading["temporal_dependence_group"] = (str(binding["source_sha256"])+":"+
                    str(binding.get("shot_id"))+":"+str(guide["id"]))
                reading["temporal_hint_previous_source_binding"] = guide.get("hint_previous_source_binding")
                reading["temporal_flow_supported"] = guide.get("temporal_flow_supported")
            if prediction["new_observation"]:
                self.active_guides = []
                for guide, reading in zip(guides_2d, guided["readings"]):
                    points = torch.as_tensor(guide["points_uv"], device=self.vision.device,
                                             dtype=torch.float32).clone()
                    indices = reading.get("coherent_source_indices", reading["source_indices"])
                    actual = reading.get("coherent_source_uv", reading["actual_source_uv"])
                    points[indices] = actual.to(dtype=points.dtype)
                    self.active_guides.append({**guide, "points_uv": points,
                                               "origin": "temporal_source_search_hint",
                                               "source_binding": binding,
                                               "identity_confirmed": bool(reading.get("identity_authority"))})
                self.active_binding, self.active_global_exclusion = binding, global_mask
        fit_revision = None
        if getattr(self, 'original_fit_reader', None) is not None:
            fit_revision = self.original_fit_reader.read(rgb, binding)
        feedback.check_top_face_exclusion(near_board)
        source_feedback = finish_read(feedback, rgb, guided, operator_conflicts=operator_conflicts,
                                      geometry=geometry_report)
        return dict(source_binding=binding, model_sha256=prediction["model_sha256"],
                    source_feedback=source_feedback, native_geometry=geometry_report, board_material=board_material,
                    native_material=material_result, source_instances=instances,
                    near_board=near_board, source_reidentification=reidentification,
                    source_print_reference_search=material_matches,
                    model_usage=self.vision.model_usage(),
                    original_FIT_source_revision=fit_revision,
                    memory_update=prediction["memory_update"], source=source, guided=guided,
                    source_clock_evidence=prediction["source_clock_evidence"],
                    new_observation=prediction["new_observation"],
                    source_motion_metadata=prediction["motion"],
                    hint_pixels_used_as_output=False, world_or_camera_optimized=False,
                    grid_shown=False, accuracy_accepted=False)

    @staticmethod
    def draw(rgb, result, advertisements=()):
        """Show measured source support only, retaining gaps and occlusions."""
        import cv2
        image = rgb.copy()
        colors = dict(ice_contact=(255,125,30), board_top=(255,235,40),
                      ice_paint=(220,55,205), ad_boundary=(35,215,255))
        # Preserve all source output; only named/guided paint goes in this
        # review layer, so unrelated screen/text edges are not shown as ice.
        for kind in (() if result.get("native_geometry") else ("ice_contact", "board_top")):
            for component in result["source"]["components"]:
                if component["typed_class_candidate"] != kind:
                    continue
                xy = component["source_support_uv"].long().cpu().numpy()
                image[xy[:, 1], xy[:, 0]] = colors[kind]
        labeled = set()
        for reading in result["guided"]["readings"]:
            points = reading.get("coherent_source_uv", reading["actual_source_uv"]).detach().cpu().numpy()
            if not len(points):
                continue
            runs = reading.get("coherent_measured_runs", reading["measured_runs"])
            for run in runs:
                uv = run["actual_source_uv"].detach().cpu().numpy()
                if len(uv) > 1:
                    cv2.polylines(image, [np.rint(uv*256).astype(np.int32)], False,
                                  colors[reading["kind"]], 2, cv2.LINE_AA, shift=8)
                elif len(uv):
                    cv2.circle(image, tuple(np.rint(uv[0]).astype(int)), 2, colors[reading["kind"]], -1)
            label = reading.get("display_label") or reading.get("physical_id") or reading.get("expected_id") or "paint"
            if label in labeled:
                continue
            anchor = points[len(points)//2]
            xy = (int(np.clip(anchor[0]+4,0,rgb.shape[1]-160)),
                  int(np.clip(anchor[1]-6,18,rgb.shape[0]-4)))
            cv2.putText(image, str(label), xy, cv2.FONT_HERSHEY_SIMPLEX, .55, (15,15,15), 3, cv2.LINE_AA)
            cv2.putText(image, str(label), xy, cv2.FONT_HERSHEY_SIMPLEX, .55, colors[reading["kind"]], 1, cv2.LINE_AA)
            labeled.add(label)
        for advertisement in (() if result.get("native_geometry") else advertisements):
            for contour in advertisement.get("source_contours", []):
                uv = np.asarray(contour["uv"], float)
                if len(uv) > 1:
                    cv2.polylines(image, [np.rint(uv*256).astype(np.int32)],
                                  bool(contour.get("closed", False)), colors["ad_boundary"], 2,
                                  cv2.LINE_AA, shift=8)
            points = [np.asarray(c["uv"], float) for c in advertisement.get("source_contours", [])
                      if len(c["uv"])]
            if points:
                uv = np.concatenate(points)
                xy = (int(np.clip(np.median(uv[:,0]), 4, rgb.shape[1]-80)),
                      int(np.clip(uv[:,1].min()+18, 18, rgb.shape[0]-4)))
                label = f"Ad{advertisement['number']}"
                cv2.putText(image,label,xy,cv2.FONT_HERSHEY_SIMPLEX,.55,(15,15,15),3,cv2.LINE_AA)
                cv2.putText(image,label,xy,cv2.FONT_HERSHEY_SIMPLEX,.55,colors["ad_boundary"],1,cv2.LINE_AA)
        return image


def read_frame(config_path, source_pts, guide_path=None, *, model_path=None, original_fit_request_path=None):
    """Exact native frame reading with optional source-bound search hints."""
    import av
    reader = SourceRinkReader(config_path, model_path=model_path, original_fit_request_path=original_fit_request_path)
    config, data = reader.config, reader.data
    digest = sha256(config["source_video"])
    guides, sample, source_guide_packet = [], None, None
    if guide_path:
        packet = read_json(guide_path)
        sample = packet["sample_confirmed"]
        if sample["source_pts"] != source_pts or sample["source_hash"] != digest:
            raise ValueError("Visual hints belong to a different original source frame")
        role_path = data / "paint_role_contract_current.json"
        roles = ({row["curve_id"]: row for row in read_json(role_path)["curve_records"]}
                 if role_path.exists() else {})
        source_guide_packet = confirmed_source_guides(sample, role_definitions=roles, return_packet=True)
        guides = source_guide_packet["guides"]
    with av.open(config["source_video"]) as container:
        stream = container.streams.video[0]
        container.seek(source_pts, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts == source_pts:
                rgb = frame.to_ndarray(format="rgb24")
                clock = str(stream.time_base)
                break
            if frame.pts is not None and frame.pts > source_pts:
                raise ValueError("Exact requested native PTS was not decoded")
        else:
            raise ValueError("Requested native source frame is absent")
    if sample and not np.array_equal(rgb, _image(sample["frame_path"])):
        raise ValueError("User-confirmed annotations differ from the original video RGB")
    binding = dict(source_sha256=digest, source_pts=source_pts, time_base=clock,
                   shot_id=f"single_source_frame_{source_pts}", native_hw=list(rgb.shape[:2]))
    result = reader.read(rgb, binding, guides_2d=guides,
                         ignored_regions=sample.get("ignored_regions", ()) if sample else (), reset=True)
    fit_revision = result.get('original_FIT_source_revision')
    if fit_revision is not None:
        fit_revision['model_usage'] = result['model_usage']
        write_json(data / f"original_FIT_revision_pts{source_pts}.json", _source_json(fit_revision))
    ads = []
    ad_path = data / "source_ads_complete_frame.json"
    if ad_path.exists():
        from .vision_data import iter_complete_source_ad_boundaries
        by_number = {}
        for curve in iter_complete_source_ad_boundaries(ad_path):
            source = curve["source_binding"]
            if (source["source_pts"] != source_pts or source["source_sha256"] != digest or
                    str(Fraction(source["time_base"])) != clock):
                continue
            entry = by_number.setdefault(curve["advertisement_number"],
                dict(number=curve["advertisement_number"], identity_authority="human_confirmed_event_catalog",
                     source_binding=source, source_contours=[]))
            entry["source_contours"].append(dict(uv=curve["uv"], closed=curve["closed"],
                                                boundary_role=curve["boundary_role"]))
        ads = list(by_number.values())
    image = reader.draw(rgb, result, ads)
    output = data / "module_frame.png"
    Image.fromarray(image).save(output)
    packet = dict(schema="rinkspace.native-source-reading.v1", source_binding=result["source_binding"],
                  model_sha256=result["model_sha256"], grid_shown=False,
                  model_usage=result['model_usage'],
                  source_clock_evidence=result["source_clock_evidence"],
                  world_or_camera_optimized=False, source_pixels_modified_for_reading=False,
                  hint_pixels_used_as_measurements=False,
                  source_annotation_adapter_metadata=(source_guide_packet["metadata"]
                                                      if source_guide_packet else None),
                  unresolved_source_annotation_roles=(source_guide_packet["unresolved"]
                                                      if source_guide_packet else []),
                  guided_readings=[{k:v for k,v in r.items() if k != "guide"}
                                   for r in result["guided"]["readings"]],
                  advertisement_source_readings=ads,
                  original_FIT_source_revision=fit_revision,
                  output_image=str(output), accuracy_accepted=False)
    write_json(data / "module_frame.json", _source_json(packet))
    print(json.dumps(dict(output_image=str(output), source_pts=source_pts,
                         read_source_points=sum(len(r.get("coherent_source_uv",r["actual_source_uv"]))
                                                for r in result["guided"]["readings"]),
                         grid_shown=False, world_or_camera_optimized=False)), flush=True)
    return packet


def review_shot(config_path, video, start_pts, end_pts, time_base, shot_id,
                expected_sha256, *, reset_pts=(), model_path=None, original_fit_request_path=None):
    """Review every decoded original frame in a declared half-open shot.

    Only source-bound numerical predictions are saved to the fixed Data files
    shot_review.jsonl and shot_review.json. Declared reset PTS are explicit cut
    instructions, not automatic guesses from motion. No camera mapping, pixel
    labels, images, prediction arrays or videos are generated.
    """
    import av
    import heapq
    import torch

    if (isinstance(start_pts, bool) or isinstance(end_pts, bool)
            or not isinstance(start_pts, int) or not isinstance(end_pts, int)
            or start_pts < 0 or end_pts <= start_pts or not shot_id):
        raise ValueError("Review requires an integer nonempty [start_pts,end_pts) and shot_id")
    clock = Fraction(str(time_base))
    digest = str(expected_sha256).lower()
    if clock <= 0 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Review requires a positive source time base and expected SHA256")
    cuts = set(reset_pts)
    if any(isinstance(p, bool) or not isinstance(p, int) or not start_pts <= p < end_pts for p in cuts):
        raise ValueError("Declared reset PTS must be original frame stamps inside this interval")
    video = Path(video).resolve()
    actual_digest = sha256(video)
    if actual_digest != digest:
        raise ValueError("Review video SHA256 differs from the expected original source")
    config, data = configure(config_path)
    model_path = Path(model_path) if model_path is not None else ROOT / config["final_model"]
    module = RinkVisionModule(model_path, ROOT / config["flow_weights"], device=config["device"],
                              exposure_registry=config.get("source_clock_registry"))
    fit_reader = None
    if original_fit_request_path is not None:
        from .vision_source_fit import OriginalFITSourceReader
        fit_reader = OriginalFITSourceReader(original_fit_request_path)
    module.reset()
    torch.cuda.reset_peak_memory_stats(module.device)
    started = time.monotonic()
    rows_path, summary_path = data / "shot_review.jsonl", data / "shot_review.json"
    summary = dict(
        schema="rinkspace.vision-continuous-shot-review.v1", status="running",
        model_usage=module.model_usage(),
        original_FIT_request_reference=fit_reader.request_reference if fit_reader is not None else None,
        source_video=str(video), source_sha256=actual_digest, source_sha256_verified=True,
        requested_start_pts=start_pts, requested_end_pts_exclusive=end_pts,
        time_base=str(clock), shot_id=str(shot_id),
        physical_camera_mapping="unknown_not_used", camera_prior_used=False,
        declared_reset_pts=sorted(cuts), reset_policy="shot_start_and_explicit_declared_cuts_only",
        model_path=str(model_path), model_sha256=module.model_sha256,
        training_step=module.training_scope.get("latest_step"),
        gpu=torch.cuda.get_device_name(module.device), torch_version=str(torch.__version__),
        all_decoded_inrange_frames_processed=True, sampling="none",
        frame_count=0, first_processed_pts=None, last_processed_pts=None,
        first_excluded_end_pts=None, source_pts_strictly_increasing=True,
        missing_pts_frames=0, source_pts_integrity_accepted=False,
        accuracy_accepted=False, independent_pixel_ground_truth_available=False,
        interpretation="Confidence and temporal smoothness are predictions, not pixel accuracy",
        prediction_change_domain="unaligned native pixels; includes camera motion, not tracked-boundary drift",
        rows_path=str(rows_path), classes=list(module.classes),
    )
    write_json(summary_path, summary)
    gap_counts, memory_counts = Counter(), Counter()
    candidates = {name: [] for name in ("rgb_change", "prediction_change", "uncertainty")}
    repeat_candidates = []
    observed_cuts = set()
    class_sums = {name: dict(pixel_fraction=0., mean_probability=0.) for name in module.classes}
    previous_rgb = previous_probability = previous_labels = previous_pts = None
    last_progress = started

    def probability_stats(probability, names):
        if not bool(torch.isfinite(probability).all()):
            raise FloatingPointError("Nonfinite probabilities during source shot review")
        labels = probability.argmax(dim=1)
        confidence = probability.amax(dim=1)
        counts = torch.bincount(labels.flatten(), minlength=len(names))
        confidence_sums = torch.zeros(len(names), device=probability.device).scatter_add_(
            0, labels.flatten(), confidence.flatten())
        means = probability.mean(dim=(0, 2, 3))
        count_values, mean_values, confidence_values = (
            counts.cpu().tolist(), means.cpu().tolist(), confidence_sums.cpu().tolist())
        stats = {
            name: dict(pixel_fraction=count_values[i] / labels.numel(),
                       mean_probability=mean_values[i],
                       argmax_confidence_mean=(confidence_values[i] / count_values[i]
                                               if count_values[i] else None))
            for i, name in enumerate(names)
        }
        entropy = float(-(probability * probability.clamp_min(1e-8).log()).sum(dim=1).mean())
        return stats, labels, entropy, float(confidence.mean())

    def candidate(kind, score, pts):
        if score is None:
            return
        heap = candidates[kind]
        heapq.heappush(heap, (float(score), int(pts)))
        if len(heap) > 8:
            heapq.heappop(heap)

    def finalize(status):
        count = summary["frame_count"]
        summary.update(
            status=status, elapsed_seconds=time.monotonic() - started,
            pts_gap_counts={str(k): v for k, v in sorted(gap_counts.items())},
            memory_update_counts=dict(memory_counts), observed_reset_pts=sorted(observed_cuts),
            unobserved_declared_reset_pts=sorted(cuts - observed_cuts),
            exact_rgb_repeat_candidate_pts=repeat_candidates,
            exact_rgb_repeat_is_official_exposure_proof=False,
            candidate_source_pts={
                kind: [dict(source_pts=p, score=s) for s, p in sorted(heap, reverse=True)]
                for kind, heap in candidates.items()
            },
            frame_mean_class_statistics={
                name: {key: value / count if count else None for key, value in values.items()}
                for name, values in class_sums.items()
            },
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(module.device),
        )
        write_json(summary_path, summary)

    try:
        with av.open(str(video)) as container, rows_path.open("w", encoding="utf-8") as output:
            stream = container.streams.video[0]
            if stream.time_base != clock:
                raise ValueError("Original video stream time base differs from requested review")
            summary["stream_time_base"] = str(stream.time_base)
            summary["stream_average_rate"] = str(stream.average_rate) if stream.average_rate else None
            nominal_step = Fraction(1, 1) / stream.average_rate / clock if stream.average_rate else None
            integer_step = int(nominal_step) if nominal_step and nominal_step.denominator == 1 else None
            summary["nominal_pts_step"] = integer_step
            summary["nominal_frame_count"] = (
                (end_pts - start_pts) // integer_step
                if integer_step and (end_pts - start_pts) % integer_step == 0 else None)
            container.seek(start_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    summary["missing_pts_frames"] += 1
                    raise ValueError("Decoded frame lacks an original source PTS")
                pts = int(frame.pts)
                if pts < start_pts:
                    continue
                if pts >= end_pts:
                    summary["first_excluded_end_pts"] = pts
                    break
                if frame.time_base != clock:
                    raise ValueError("Decoded original frame changed its source time base")
                if previous_pts is not None:
                    gap = pts - previous_pts
                    gap_counts[gap] += 1
                    if gap <= 0:
                        summary["source_pts_strictly_increasing"] = False
                rgb = frame.to_ndarray(format="rgb24")
                explicit_reset = summary["frame_count"] == 0 or pts in cuts
                if pts in cuts:
                    observed_cuts.add(pts)
                binding = dict(source_sha256=actual_digest, source_pts=pts,
                               time_base=str(clock), shot_id=str(shot_id),
                               native_hw=list(rgb.shape[:2]))
                result = module.predict(rgb, binding, reset=explicit_reset)
                probability = result["class_probabilities"]
                stats, labels, entropy, confidence = probability_stats(probability, module.classes)
                primitive_stats, _, _, _ = probability_stats(
                    result["primitive_probabilities"], ("line", "circle", "other"))
                comparison = (not explicit_reset and previous_rgb is not None
                              and rgb.shape == previous_rgb.shape)
                rgb_change = float(np.abs(rgb.astype(np.int16) - previous_rgb.astype(np.int16)).mean()) if comparison else None
                label_change = float((labels != previous_labels).float().mean()) if comparison else None
                probability_change = float((probability - previous_probability).abs().sum(dim=1).mean()) if comparison else None
                exact_repeat = bool(np.array_equal(rgb, previous_rgb)) if comparison else False
                if exact_repeat and len(repeat_candidates) < 8:
                    repeat_candidates.append(pts)
                hidden = module.hidden
                if hidden is None or not bool(torch.isfinite(hidden).all()):
                    raise FloatingPointError("Missing or nonfinite within-shot memory")
                row = dict(
                    source_pts=pts, time_base=str(clock), source_time_seconds=float(pts * clock),
                    source_binding=result["source_binding"],
                    prediction_source_binding=result["prediction_source_binding"],
                    native_hw=list(rgb.shape[:2]), class_statistics=stats,
                    primitive_statistics=primitive_stats, confidence_mean=confidence,
                    entropy_mean_nats=entropy, rgb_mean_absolute_change_0_255=rgb_change,
                    argmax_changed_pixel_fraction=label_change,
                    probability_l1_mean_change=probability_change,
                    exact_rgb_repeat_candidate=exact_repeat,
                    temporal_comparison="same_declared_shot" if comparison else "reset_or_first_frame",
                    memory_update=result["memory_update"], new_observation=result["new_observation"],
                    explicit_reset=explicit_reset, hidden_shape=list(hidden.shape), hidden_finite=True,
                    hidden_rms=float(hidden.float().square().mean().sqrt()), motion=result["motion"],
                    accuracy_accepted=False, independent_pixel_ground_truth_available=False,
                )
                if fit_reader is not None:
                    row['original_FIT_source_revision'] = _source_json(fit_reader.read(rgb, result['source_binding']))
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                summary["frame_count"] += 1
                if summary["first_processed_pts"] is None:
                    summary["first_processed_pts"] = pts
                summary["last_processed_pts"] = pts
                memory_counts[result["memory_update"]] += 1
                for name, values in stats.items():
                    for key in class_sums[name]:
                        class_sums[name][key] += values[key]
                candidate("rgb_change", rgb_change, pts)
                candidate("prediction_change", label_change, pts)
                candidate("uncertainty", entropy, pts)
                previous_rgb, previous_probability, previous_labels, previous_pts = rgb, probability, labels, pts
                now = time.monotonic()
                if now - last_progress >= 15:
                    output.flush()
                    finalize("running")
                    print(json.dumps(dict(command="review-shot", frames=summary["frame_count"],
                                          last_pts=pts, elapsed_seconds=now - started)), flush=True)
                    last_progress = now
        nominal_count = summary["nominal_frame_count"]
        summary["frame_count_matches_nominal_rate"] = (
            summary["frame_count"] == nominal_count if nominal_count is not None else None)
        summary["source_pts_integrity_accepted"] = bool(
            summary["frame_count"] and summary["first_processed_pts"] == start_pts
            and summary["first_excluded_end_pts"] == end_pts
            and summary["source_pts_strictly_increasing"] and not summary["missing_pts_frames"]
            and not (cuts - observed_cuts)
            and (integer_step is None or all(gap == integer_step for gap in gap_counts))
            and summary["frame_count_matches_nominal_rate"] is not False)
        finalize("decoded_range_complete" if summary["source_pts_integrity_accepted"] else "source_pts_review_required")
    except Exception as error:
        summary.update(error_type=type(error).__name__, error=str(error),
                       all_decoded_inrange_frames_processed=False, source_pts_integrity_accepted=False)
        finalize("failed")
        raise
    print(json.dumps({key: summary[key] for key in
                      ("status", "frame_count", "first_processed_pts", "last_processed_pts",
                       "source_pts_integrity_accepted", "model_sha256", "accuracy_accepted")}), flush=True)
    return summary


def load_training_manifest(config, data):
    """Assemble the exact source labels shared by audits and GPU training."""
    from .vision_supervision import (source_updates_by_key, source_sample_key,
        merge_user_boundary_samples, merge_source_reviewed_regions,
        attach_native_shot_scopes, apply_supervision_roles)
    manifest = read_json(data / "manifest.json")
    supplementary_path = data / "supplementary_samples.json"
    if supplementary_path.exists():
        additions = read_json(supplementary_path)
        updates = source_updates_by_key(manifest["samples"], additions.get("sample_updates", []))
        merged = []
        for sample in manifest["samples"]:
            update = updates.get(source_sample_key(sample))
            if update:
                if sha256(sample["frame_path"]) != update["frame_sha256"]:
                    raise ValueError("Reviewed sample additions do not match the original frame")
                annotations = [{**row} for row in sample["annotations"]]
                for review in update.get("local_support_reviews", []):
                    row = next(a for a in annotations if a["group_id"] == review["group_id"])
                    point_index = int(review["source_point_index"])
                    uv = np.asarray(row["uv"], float)
                    if not np.allclose(uv[point_index], review["original_uv"], rtol=0, atol=1e-6):
                        raise ValueError("Local source support review references a different original point")
                    if row.get("state") == "user_confirmed_visual_supervision":
                        row["local_source_support_review"] = review
                        continue  # Assistant intervals cannot replace confirmed human positions.
                    interval = np.asarray(row["interval_px"], float).copy()
                    if interval.shape == (2,):
                        interval = np.tile(interval, (len(uv), 1))
                    if not np.allclose(interval[point_index], review["original_interval_px"], rtol=0, atol=1e-6):
                        raise ValueError("Local source support interval changed before review")
                    row["original_interval_px_before_source_review"] = row["interval_px"]
                    interval[point_index] = review["training_interval_union_px"]
                    row["interval_px"] = interval.tolist()
                    row["local_source_support_review"] = review
                sample = {**sample, "annotations": annotations,
                          "regions": sample.get("regions", []) + update.get("regions", [])}
            merged.append(sample)
        manifest = {**manifest, "samples": merged + additions["samples"],
                    "supplementary_source_sha256": sha256(supplementary_path)}
    temporal_path = data / "temporal_supervision_samples.json"
    if temporal_path.exists():
        temporal = read_json(temporal_path)
        existing_frames = {source_sample_key(s) for s in manifest["samples"]}
        extra = [s for s in temporal.get("samples", [])
                 if source_sample_key(s) not in existing_frames]
        manifest = {**manifest, "samples": manifest["samples"] + extra,
                    "temporal_supervision_sha256": sha256(temporal_path)}
    if config.get("user_boundary_supervision"):
        user_boundary_path = Path(config["user_boundary_supervision"])
        user_boundary_path = user_boundary_path if user_boundary_path.is_absolute() else ROOT / user_boundary_path
        manifest = merge_user_boundary_samples(manifest, read_json(user_boundary_path))
        manifest["user_boundary_supervision_source"] = dict(
            path=str(user_boundary_path.resolve()), sha256=sha256(user_boundary_path),
            authority="explicit_source_bound_direct_user_revision",
            original_source_positions_preserved=True,
            source_visibility_separate_from_human_geometry=True)
    for reference in config.get("additional_user_boundary_supervision", []):
        path = Path(reference["path"])
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != reference["sha256"]:
            raise ValueError("Additional direct user supervision changed after registration")
        manifest = merge_user_boundary_samples(manifest, read_json(path))
        manifest.setdefault("additional_user_boundary_supervision_sources", []).append(
            dict(path=str(path.resolve()), sha256=reference["sha256"]))
    if config.get("source_material_region_supervision"):
        material_path = Path(config["source_material_region_supervision"])
        material_path = material_path if material_path.is_absolute() else ROOT / material_path
        manifest = merge_source_reviewed_regions(manifest, read_json(material_path))
        manifest["source_material_region_supervision_source"] = dict(
            path=str(material_path.resolve()), sha256=sha256(material_path),
            authority="separate_review_of_actual_native_RGB_material_interiors",
            precise_boundary_ground_truth_created=False)
    if config.get("source_semantic_rejections"):
        rejection_path = Path(config["source_semantic_rejections"])
        rejection_path = rejection_path if rejection_path.is_absolute() else ROOT / rejection_path
        manifest = merge_source_reviewed_regions(manifest, read_json(rejection_path))
        manifest["source_semantic_rejections_source"] = dict(
            path=str(rejection_path.resolve()), sha256=sha256(rejection_path),
            authority="actual_source_RGB_role_absence_review",
            other_positive_class_or_primitive_assigned=False,
            precise_boundary_ground_truth_created=False)
    index_path = data / "supervision_index.json"
    if not index_path.exists():
        raise ValueError("Prepare the source supervision index and whole-shot split before training")
    shot_plan = read_json(index_path)["shot_plan"]
    if config.get("shot_plan"):
        reference = config["shot_plan"]
        path = Path(reference["path"])
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != reference["sha256"]:
            raise ValueError("Explicit training shot plan changed")
        shot_plan = read_json(path)
    manifest = attach_native_shot_scopes(apply_supervision_roles(manifest), shot_plan)
    allowed = config.get("training_annotation_authorities")
    if allowed is not None:
        from .vision_supervision import select_training_authorities
        manifest = select_training_authorities(manifest, allowed)
    if config.get("source_support_integration"):
        from .vision_supervision import attach_source_support_integration
        manifest = attach_source_support_integration(manifest)
    return manifest, shot_plan


def train(config_path, steps=None, resume=False, candidate=False, resume_checkpoint=None,
          retention_guard_every=None,update_scope=None,learning_rate=None,position_rank_weight=None,
          retention_reference=None):
    if read_json(config_path).get('strict_source_stream_training'):
        from .strict_source_supervision import train_strict_source_stream
        return train_strict_source_stream(config_path,steps=steps,resume=resume,candidate=candidate,
            resume_checkpoint=resume_checkpoint,retention_guard_every=retention_guard_every,
            update_scope=update_scope,learning_rate=learning_rate,position_rank_weight=position_rank_weight,
            retention_reference=retention_reference)
    import torch
    from torch.nn import functional as F
    from .vision_network import RinkVisionNet, masked_region_loss
    from .vision_geometry import GPUFlow, warp
    from .vision_supervision import (apply_supervision_roles, split_indices, shot_sampling_groups,
                                     attach_native_shot_scopes, source_sample_key, source_updates_by_key,
                                     merge_user_boundary_samples, merge_source_reviewed_regions,
                                     training_supervision_consumption_record,
                                     confirmed_position_interval_localization_loss,
                                     confirmed_source_point_coverage,confirmed_source_crop_target)
    from .vision_retention import source_retention_snapshot, source_retention_regressions

    config, data = configure(config_path)
    localization_weight=float(config.get("human_position_localization_weight",1.))
    run_data = data
    if config.get("training_run_directory"):
        if not candidate:
            raise ValueError("A separate training run directory stages candidates only")
        run_data = Path(config["training_run_directory"]).resolve()
        if not run_data.is_relative_to(data):
            raise ValueError("Candidate training artifacts must stay within the configured Data directory")
        run_data.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(localization_weight) or localization_weight<0:
        raise ValueError("Human position localization weight must be finite and nonnegative")
    guard_every=config.get("source_retention_guard_every",20) if retention_guard_every is None else retention_guard_every
    if isinstance(guard_every,bool) or not isinstance(guard_every,int) or guard_every<1:
        raise ValueError("Source retention guard cadence must be a positive optimizer-step interval")
    guard_action = config.get("source_retention_guard_action", "stop")
    if guard_action not in ("stop", "record_candidate") or guard_action == "record_candidate" and not candidate:
        raise ValueError("Recording intermediate regressions without stopping is allowed for staged candidates only")
    rank_weight=float(config.get("human_position_rank_weight",0.) if position_rank_weight is None else position_rank_weight)
    primitive_weight=float(config.get("primitive_supervision_weight",.2))
    if not np.isfinite(primitive_weight) or primitive_weight<0:
        raise ValueError("Primitive supervision weight must be finite and nonnegative")
    if not np.isfinite(rank_weight) or rank_weight<0:
        raise ValueError("Confirmed-position rank weight must be finite and nonnegative")
    if learning_rate is not None and (isinstance(learning_rate,bool) or not np.isfinite(learning_rate) or learning_rate<=0):
        raise ValueError("An explicit learning rate must be finite and positive")
    if resume_checkpoint is not None and not resume:
        raise ValueError("An explicit resume checkpoint requires resume=True")
    if not torch.cuda.is_available():
        raise RuntimeError("Configured GPU is required for this training run")
    device = torch.device(config["device"])
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    manifest, shot_plan = load_training_manifest(config, data)
    supervision_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    input_reviews = validate_training_sources(manifest)
    sequences, sequence_metadata = prepare_sequences(config, run_data, manifest)
    previous_roles = [source_previous_observation(row) for row in sequence_metadata["rows"]]
    samples = manifest["samples"]
    if len(samples) < 2:
        raise ValueError("Need separately sourced annotation frames")
    indices = split_indices(samples, shot_plan)
    if indices["pending_source_timeline"]:
        raise ValueError("Each training source needs its own verified complete-shot classification")
    training, held = indices["train"], indices["development"]
    if config.get("training_annotation_authorities") is not None:
        training = [i for i in training if manifest["samples"][i].get("regions") or any(
            a.get("vision_supervision", {}).get("semantic_supervision_enabled")
            for a in manifest["samples"][i]["annotations"])]
    if not training:
        raise ValueError("Whole-shot split has no training sources")
    sampling_groups = shot_sampling_groups(samples, shot_plan, training)
    held_pts = {int(samples[i]["source_pts"]) for i in held}
    checkpoint_path = run_data / ("checkpoint_candidate.pt" if candidate else "checkpoint.pt")
    report_path = run_data / ("training_report_candidate.json" if candidate else "training_report.json")
    resume_path = (Path(resume_checkpoint) if resume_checkpoint is not None else
                   checkpoint_path if checkpoint_path.exists() else data / "checkpoint.pt")
    if not resume_path.is_absolute():
        resume_path=ROOT/resume_path
    if resume and not resume_path.is_file():
        raise FileNotFoundError("Requested resume checkpoint is missing; never restart the network silently")
    resume_checkpoint_sha256 = sha256(resume_path) if resume and resume_path.exists() else None
    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("supervision_sha256") != supervision_digest:
            prior_manifest = saved.get("source_manifest")
            if prior_manifest is None:
                prior_payload = torch.load(ROOT / config["final_model"], map_location="cpu", weights_only=True)
                if int(prior_payload["training_report"]["latest_step"]) != int(saved["step"]):
                    raise ValueError("Checkpoint and prior source manifest have different training revisions")
                prior_manifest = prior_payload["source_manifest"]
            current_samples = {source_sample_key(s):s for s in manifest["samples"]}
            if len(current_samples) != len(manifest["samples"]):
                raise ValueError("Resume source manifest contains duplicate native source identities")
            for original in prior_manifest["samples"]:
                updated = current_samples.get(source_sample_key(original))
                if updated is None or any(original[k] != updated[k] for k in
                                          ("frame_sha256","source_hash","frame_size_wh")) or (
                        updated is not None and Fraction(original["time_base"]) != Fraction(updated["time_base"])):
                    raise ValueError("Resume requalification changed an original source image")
                current_groups = {a["group_id"]:a for a in updated["annotations"]}
                for original_group in original["annotations"]:
                    updated_group = current_groups.get(original_group["group_id"])
                    if (updated_group is None or original_group["kind"] != updated_group["kind"]
                            or original_group.get("primitive") != updated_group.get("primitive")
                            or not np.array_equal(original_group["uv"],updated_group["uv"])):
                        raise ValueError("Resume requalification changed previously recorded source coordinates")
            saved["source_requalification"] = dict(previous_manifest_sha256=saved["manifest_sha256"],
                                                   current_manifest_sha256=sha256(data / "manifest.json"),
                                                   current_supervision_sha256=supervision_digest,
                                                   original_images_and_UV_preserved=True,
                                                   source_review_validated=True)
            del prior_manifest
        network = RinkVisionNet(**saved["model_config"]).to(device)
        network.load_state_dict(saved["state_dict"], strict=True)
        start_step = int(saved["step"])
    else:
        saved = None
        network = RinkVisionNet(num_classes=len(config["classes"]),
                                backbone_path=config["backbone"], freeze_backbone=True).to(device)
        start_step = 0
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],
                                  lr=config["learning_rate"], weight_decay=1e-4)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        if saved.get("rng_state") is not None:
            rng.setstate(saved["rng_state"])
        if saved.get("torch_rng_state") is not None:
            torch.set_rng_state(saved["torch_rng_state"])
        if saved.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
    actual_scope=update_scope or (saved or {}).get("training_update_scope",config.get("training_update_scope","all_trainable"))
    if actual_scope not in ("all_trainable","semantic_head"):
        raise ValueError("Training update scope must be all_trainable or semantic_head")
    if actual_scope=="semantic_head":
        # Preserve the complete optimizer/moments layout. Frozen parameters
        # have grad=None and AdamW does not update them; no primitive/shared
        # path is duplicated, reset or changed by the positional experiment.
        for name,parameter in network.named_parameters():
            parameter.requires_grad_(name.startswith("semantic_head."))
    if learning_rate is not None:
        for group in optimizer.param_groups:group["lr"]=float(learning_rate)
    initial_frozen_parameter_sha={name:hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,parameter in network.named_parameters() if not parameter.requires_grad}
    crop_coverage={};coverage_revision_resets=[]
    balanced_sampling = bool(config.get("balanced_source_cycles"))
    sampling_state = (saved or {}).get("balanced_source_sampling_state", {}) if balanced_sampling else {}
    for sample in samples:
        key=json.dumps(list(source_sample_key(sample)),separators=(",",":"))
        current=confirmed_source_point_coverage(sample,[])
        prior=(saved or {}).get("confirmed_source_crop_coverage",{}).get(key)
        if prior:
            if prior.get("source_key")!=current["source_key"]:
                raise ValueError("Saved human crop coverage belongs to another original source")
            for identity in current["groups"]:
                if prior.get("original_annotation_sha256",{}).get(identity)==current["original_annotation_sha256"][identity]:
                    counts=np.asarray(prior.get("groups",{}).get(identity))
                    if counts.shape!=(len(current["groups"][identity]),) or counts.dtype.kind not in "iu" or (counts<0).any():
                        raise ValueError("Saved actual human crop coverage has changed source indices")
                    current["groups"][identity]=counts.tolist()
                else:coverage_revision_resets.append(dict(source_key=current["source_key"],group_id=identity))
        crop_coverage[key]=current
    flow_reader = GPUFlow(ROOT / config["flow_weights"], device=str(device), num_updates=8)
    # Flow is measured once on full source frames. Crops retain native units.
    needed_indices = sorted(set(training + held))
    flow_cache = [None] * len(sequences)
    for completed, index in enumerate(needed_indices, 1):
        sequence = sequences[index]
        tensor = _tensor(sequence, device)[0]
        pair = flow_reader.compute(tensor[0:1], tensor[1:2], analysis_hw=(384, 640))
        # Preserve native flow; keep only the two fields used by training in
        # host memory. Growing the source inventory must not grow GPU state.
        flow_cache[index] = (pair.backward.cpu(), pair.valid_backward.cpu())
        del pair, tensor
        if completed % 10 == 0 or completed == len(needed_indices):
            print(json.dumps(dict(stage="cache_actual_source_flow", completed=completed, total=len(needed_indices))), flush=True)
    # Keep the pretrained encoder's view identical to full-frame inference.
    # A native crop receives an ROI of this full-frame FPN; magnifying that
    # crop into a fresh 512-square SAM image changes landmark/context scale.
    context_cache = [None] * len(sequences)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for completed, index in enumerate(needed_indices, 1):
            sequence = sequences[index]
            context_cache[index] = [level.cpu() for level in network.encode_context(_tensor(sequence, device))]
            if completed % 10 == 0 or completed == len(needed_indices):
                print(json.dumps(dict(stage="cache_full_source_context", completed=completed, total=len(needed_indices))), flush=True)
    class_coverage = Counter(a["kind"] for i in training for a in samples[i]["annotations"])
    class_coverage.update(region["kind"] for i in training for region in samples[i].get("regions", [])
                          if region.get("kind"))
    class_coverage["background"] += sum(any(a["kind"] == "board_top" for a in samples[i]["annotations"]) for i in training)
    report = dict(schema="rinkspace.rink-vision-training.v1", status="running",
                  stage=config["training_stage"], accuracy_accepted=False,
                  pixel_ground_truth=False, classes=config["classes"],
                  supervised_class_groups=dict(class_coverage),
                  reject_only_region_count=sum("excluded_kinds" in region for i in training
                                               for region in samples[i].get("regions", [])),
                  excluded_semantic_class_groups=dict(Counter(kind for i in training
                      for region in samples[i].get("regions", [])
                      for kind in region.get("excluded_kinds", []))),
                  source_semantic_rejections_source=manifest.get("source_semantic_rejections_source"),
                  unsupervised_classes=[c for c in config["classes"] if c not in class_coverage],
                  manifest_sha256=sha256(data / "manifest.json"),
                  supervision_sha256=supervision_digest,
                  source_video_sha256=sequence_metadata["source_videos_sha256"][str(Path(config["source_video"]).resolve())],
                  source_videos_sha256=sequence_metadata["source_videos_sha256"],
                  verified_input_sources=input_reviews,
                  source_rgb_cache_sha256=sequence_metadata["bundle_sha256"],
                  source_clock_evidence=sequence_metadata["source_clock_evidence"],
                  source_clock_evidence_sha256=sequence_metadata["clock_evidence_sha256"],
                  gpu=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
                  training_source_pts=[samples[i]["source_pts"] for i in training],
                  training_source_bindings=[dict(source_sha256=samples[i]["source_hash"],
                                                time_base=str(Fraction(samples[i]["time_base"])),
                                                source_pts=int(samples[i]["source_pts"])) for i in training],
                  sampling_policy=("balanced_cycles_uniform_shot_then_pool_then_frame" if balanced_sampling else
                                   "uniform_whole_shot_then_available_human_or_source_RGB_pool_then_frame"),
                  shot_sampling_groups=sampling_groups,
                  held_source_pts=sorted(held_pts),
                  held_source_bindings=[dict(source_sha256=samples[i]["source_hash"],
                                            time_base=str(Fraction(samples[i]["time_base"])),
                                            source_pts=int(samples[i]["source_pts"])) for i in held],
                  whole_shot_split=shot_plan,
                  independent_source_samples_used_for_training=False,
                  training_authority_selection=manifest.get("training_authority_selection"),
                  candidate_only=bool(candidate),
                  source_retention_guard_action=guard_action,
                  source_support_integration_enabled=bool(config.get("source_support_integration")),
                  primitive_supervision_weight=primitive_weight,
                  held_label_authority="original_user_confirmed_annotations",
                  full_shot_accuracy_validated=False, camera_prior_used_as_target=False,
                  future_frame_used=True, realtime_causal_claim=False,
                  development_frame_used_to_diagnose_errors=True,
                  held_is_final_independent_validation=False,
                  encoder_context="full_source_frame_FPN_native_pixel_ROI",
                  source_cache_storage="CPU_native_flow_and_frozen_FPN_selected_sample_to_GPU",
                  previous_memory_policy="only_distinct_actual_source_observation; no_memory_advance_for_padded_or_proven_same_exposure",
                  missing_previous_source_observation_count=sum(not role["available"] for role in previous_roles),
                  other_interval_semantic_loss="existential_max_in_supported_bag_not_dense_positive_band",
                  confirmed_user_semantic_loss="direct_native_UV_class_CE_no_competing_window",
                  confirmed_user_position_localization="original_normal_interval_spatial_log_odds_mass_with_all_active_same_kind_manual_support",
                  confirmed_user_position_localization_weight=localization_weight,
                  confirmed_user_position_rank_weight=rank_weight,
                  training_update_scope=actual_scope,
                  optimizer_parameter_list_and_moments_preserved=bool(saved),
                  actual_optimizer_learning_rates=[group["lr"] for group in optimizer.param_groups],
                  updated_parameter_names=[name for name,parameter in network.named_parameters() if parameter.requires_grad],
                  human_crop_sampling="least_actual_consumed_original_point_bins_within_the_selected_source; reviewed regions retain separate crop sampling",
                  human_crop_coverage_history_backfilled=False,
                  human_crop_coverage_annotation_revision_resets=coverage_revision_resets,
                  confirmed_user_primitive_supervision=True,
                  confirmed_user_positions_filtered_by_model_confidence=False,
                  full_frame_every=config.get("full_frame_every", 4),
                  history=[], resumed_from_step=start_step)
    write_json(report_path, report)
    started = time.monotonic()

    def evaluate():
        network.eval()
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in held:
                crop, origin = sequences[i], (0, 0)
                annotations = _annotations_in_crop(samples[i], origin, crop.shape[1:3])
                first = 0 if previous_roles[i]["available"] else 1
                output = network(_tensor(crop[first:], device),
                                 context_features=[level[:, first:].to(device) for level in context_cache[i]])
                loss, diagnostics = interval_supervision(output["logits"][:, 1-first], annotations, config["classes"])
                localization, localization_diagnostics = confirmed_position_interval_localization_loss(
                    output["logits"][:, 1-first],annotations,config["classes"],
                    original_sample=samples[i],crop_origin_xy=origin,rank_weight=rank_weight)
                rows.append(dict(source_pts=samples[i]["source_pts"], crop_xy=list(origin),
                                 source_sha256=samples[i]["source_hash"], time_base=samples[i]["time_base"],
                                 evaluation_domain="full_native_source_frame",
                                 native_hw=list(crop.shape[1:3]),
                                 source_supervision_loss=float(loss), groups=diagnostics,
                                 confirmed_position_localization_loss=float(localization),
                                 position_localization_groups=localization_diagnostics))
        network.train()
        return rows

    report["held_before_training"] = evaluate()
    guard_indices=[i for i in training if samples[i].get("regions") or any(
        a.get("state")=="user_confirmed_visual_supervision" for a in samples[i]["annotations"])]
    if not guard_indices:
        raise ValueError("Training needs existing confirmed source positions or reviewed regions for retention checks")

    def native_retention_snapshots():
        """Causal full native observations; cached encoder features supply no pixel targets."""
        network.eval();snapshots=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for i in guard_indices:
                images=_tensor(sequences[i][:2],device)
                hidden=None
                if previous_roles[i]["available"]:
                    observed=network(images[:,:1],context_features=[level[:,:1].to(device) for level in context_cache[i]])
                    sh,sw=observed["hidden"].shape[-2:]
                    native_backward,native_valid=flow_cache[i]
                    flow=F.interpolate(native_backward.to(device),size=(sh,sw),mode="bilinear",align_corners=False)
                    flow=flow*flow.new_tensor([sw/images.shape[-1],sh/images.shape[-2]])[None,:,None,None]
                    valid=F.interpolate(native_valid.to(device).float(),size=(sh,sw),mode="nearest")
                    hidden,inside=warp(observed["hidden"],flow)
                    hidden=hidden*inside.to(hidden.dtype)*valid.to(hidden.dtype)
                actual=network(images[:,1:2],hidden=hidden,
                    context_features=[level[:,1:2].to(device) for level in context_cache[i]])
                h,w=images.shape[-2:];sample=samples[i]
                annotations=_annotations_in_crop(sample,(0,0),(h,w))
                binding=dict(source_sha256=sample["source_hash"],time_base=sample["time_base"],
                    source_pts=int(sample["source_pts"]),shot_id=sample["source_shot_scope"]["shot_id"],
                    native_hw=[h,w],image_domain="native_brown_pixels")
                regions=[({**region,"region_id":region.get("region_id") or f"source_reviewed_region_{index}"},mask)
                    for index,(region,mask) in enumerate(source_reviewed_region_masks(sample,(0,0),(h,w),device))]
                snapshots.append(source_retention_snapshot(actual["logits"][:,0].float().softmax(1),
                    actual["primitive_logits"][:,0].float().softmax(1),source_binding=binding,
                    annotations=annotations,classes=config["classes"],reviewed_regions=regions))
                del actual,images,hidden
        network.train()
        return snapshots

    baseline_retention=native_retention_snapshots()
    reference_record=None;reference_start_comparison=None
    if retention_reference is None and saved and saved.get("source_retention_guard_reference"):
        reference_record=saved["source_retention_guard_reference"]
        retention_reference=Path(reference_record["path"])
    if retention_reference is not None:
        retention_reference=Path(retention_reference)
        if reference_record and sha256(retention_reference)!=reference_record["sha256"]:
            raise ValueError("Saved source retention reference has changed")
        retained=read_json(retention_reference)
        if retained.get("schema")!="rinkspace.training-native-source-retention-guard.v1" or retained["checked_sample_indices"]!=guard_indices:
            raise ValueError("Existing source retention reference has different original-source coverage")
        reference_start_comparison=[source_retention_regressions(current,baseline)
            for current,baseline in zip(baseline_retention,retained["baseline"])]
        if len(baseline_retention)!=len(retained["baseline"]):
            raise ValueError("Source retention reference lost an original observation")
        baseline_retention=retained["baseline"]
        reference_record=dict(path=str(retention_reference),sha256=sha256(retention_reference),
            baseline_step=retained["baseline_step"])
    guard_path=run_data/f"source_retention_guard_{'candidate' if candidate else 'formal'}_{start_step+1}_{time.time_ns()}.json"
    guard_record=dict(schema="rinkspace.training-native-source-retention-guard.v1",
        comparison_basis="actual native current frame with the distinct actual previous observation only when available",
        future_frame_supplied=False,encoder_context_is_not_pixel_truth=True,
        checked_sample_indices=guard_indices,baseline_step=start_step,baseline=baseline_retention,checks=[],
        failed=False,guard_every_successful_optimizer_steps=guard_every,
        reference_record=reference_record,initial_checkpoint_vs_reference=reference_start_comparison,
        unknown_pixels_assigned_targets=False,source_boundary_accuracy_accepted=False)
    if reference_record is None:
        immutable_reference_path=guard_path.with_name(guard_path.stem+"_reference.json")
        write_json(immutable_reference_path,dict(schema=guard_record["schema"],checked_sample_indices=guard_indices,
            baseline_step=start_step,baseline=baseline_retention,reference_is_immutable=True,
            source_boundary_accuracy_accepted=False))
        reference_record=dict(path=str(immutable_reference_path),sha256=sha256(immutable_reference_path),baseline_step=start_step)
        guard_record["reference_record"]=reference_record
    write_json(guard_path,guard_record)
    report["source_retention_guard"]=dict(path=str(guard_path),cadence=guard_every,
        reference_source_count=len(guard_indices),failed=False,
        independent_or_physical_pixel_accuracy_check=False)
    if reference_record:guard_record["baseline_step"]=reference_record["baseline_step"]
    if guard_action == "stop" and reference_start_comparison and any(not row["retained_without_new_classification_errors"] for row in reference_start_comparison):
        guard_record["failed"]=True;write_json(guard_path,guard_record)
        report.update(status="existing_checkpoint_failed_original_source_retention_reference",latest_step=start_step)
        report["source_retention_guard"].update(failed=True,sha256=sha256(guard_path))
        write_json(report_path,report)
        print(json.dumps(dict(status=report["status"],latest_step=start_step,optimizer_steps=0)),flush=True)
        return report
    write_json(guard_path,guard_record)
    total_steps = int(steps if steps is not None else config["training_steps"])
    consumption_path = run_data / f"training_consumption_{'candidate' if candidate else 'formal'}_{start_step+1}_{start_step+total_steps}_{time.time_ns()}.jsonl"
    consumption_header = dict(schema="rinkspace.optimizer-source-consumption-ledger.v1", record_type="header",
        source_supervision_sha256=supervision_digest, source_RGB_cache_sha256=sequence_metadata["bundle_sha256"],
        config_sha256=sha256(config_path), source_videos_sha256=sequence_metadata["source_videos_sha256"],
        start_step=start_step, requested_optimizer_steps=total_steps,
        resume_checkpoint_path=str(resume_path) if resume_checkpoint_sha256 else None,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        start_rng_state_sha256=hashlib.sha256(json.dumps(rng.getstate()).encode()).hexdigest(),
        source_code_sha256={name:sha256(ROOT/"src/rinkspace"/name)
                            for name in ("vision.py","vision_network.py","vision_supervision.py",
                                         "vision_retention.py","source_sampling.py")},
        human_position_localization_weight=localization_weight,
        human_position_rank_weight=rank_weight,training_update_scope=actual_scope,
        actual_learning_rates=[group["lr"] for group in optimizer.param_groups],
        human_crop_sampling=report["human_crop_sampling"],human_crop_coverage_history_backfilled=False,
        human_position_localization_basis=report["confirmed_user_position_localization"],
        written_only_after_successful_optimizer_step=True, past_training_records_backfilled=False)
    with consumption_path.open("x",encoding="utf-8",newline="\n") as ledger:
        ledger.write(json.dumps(consumption_header,sort_keys=True,allow_nan=False)+"\n")
    consumption_steps = 0
    report["optimizer_source_consumption"] = dict(path=str(consumption_path), successful_steps_recorded=0,
        original_source_positions_verified=True, historical_training_backfilled=False)
    for step in range(start_step, start_step + total_steps):
        if balanced_sampling:
            from .vision_supervision import balanced_source_selection
            sample_group,i=balanced_source_selection(sampling_groups,sampling_state,rng)
        else:
            sample_group = rng.choice(sampling_groups)
            i = rng.choice(rng.choice(sample_group["sampling_pools"]))
        full_frame = (step + 1) % int(config.get("full_frame_every", 4)) == 0
        crop_target=None
        coverage_key=json.dumps(list(source_sample_key(samples[i])),separators=(",",":"))
        if full_frame:
            crop, (x, y) = sequences[i], (0, 0)
            annotations = _annotations_in_crop(samples[i], (x, y), crop.shape[1:3])
        else:
            regions=samples[i].get("regions",[])
            if regions and rng.random()<.5:
                box=rng.choice(regions)["rectangle_xyxy"]
                size=config["crop_size"];h,w=sequences[i].shape[1:3]
                x=int(np.clip((box[0]+box[2])/2-size//2,0,w-size))
                y=int(np.clip((box[1]+box[3])/2-size//2,0,h-size))
                crop=sequences[i][:,y:y+size,x:x+size].copy()
                annotations=_annotations_in_crop(samples[i],(x,y),size)
            else:
                crop_target=confirmed_source_crop_target(samples[i],crop_coverage[coverage_key],rng,crop_size=config["crop_size"])
                if crop_target is None:
                    crop,annotations,(x,y)=_crop(sequences[i],samples[i],config["crop_size"],rng)
                else:
                    x,y=crop_target["crop_origin_native_xy"];size=config["crop_size"]
                    crop=sequences[i][:,y:y+size,x:x+size].copy()
                    annotations=_annotations_in_crop(samples[i],(x,y),size)
        crop_h, crop_w = crop.shape[1:3]
        if not annotations and not samples[i].get("regions"):
            crop, (x, y), full_frame = sequences[i], (0, 0), True
            crop_h, crop_w = crop.shape[1:3]
            annotations = _annotations_in_crop(samples[i], (x, y), (crop_h, crop_w))
            if not annotations:
                raise ValueError("Selected training source has no usable original-source supervision")
        images = _tensor(crop, device)
        background_mask = source_background_mask(samples[i], (x, y), (crop_h, crop_w), device)
        backward_cpu, valid_cpu = flow_cache[i]
        backward = backward_cpu[:, :, y:y + crop_h, x:x + crop_w].to(device)
        valid = valid_cpu[:, :, y:y + crop_h, x:x + crop_w].to(device)
        optimizer.zero_grad(set_to_none=True)
        # The previous full-frame forward is no_grad. Reusing its autocast
        # weight cache can silently detach the current trainable head.
        with torch.autocast("cuda", dtype=torch.bfloat16,cache_enabled=False):
            context_args = dict(context_crop=(x, y, crop_w, crop_h),
                                context_native_hw=sequences[i].shape[1:3])
            previous = None
            aligned_hidden = None
            if previous_roles[i]["available"]:
                if full_frame:
                    with torch.no_grad():
                        previous = network(images[:, :1],
                                           context_features=[level[:, :1].to(device) for level in context_cache[i]],
                                           **context_args)
                else:
                    previous = network(images[:, :1],
                                       context_features=[level[:, :1].to(device) for level in context_cache[i]],
                                       **context_args)
                hidden = previous["hidden"]
                sh, sw = hidden.shape[-2:]
                hidden_flow = F.interpolate(backward, size=(sh, sw), mode="bilinear", align_corners=False)
                hidden_flow = hidden_flow * hidden_flow.new_tensor([sw / crop_w, sh / crop_h])[None, :, None, None]
                aligned_hidden, hidden_inside = warp(hidden, hidden_flow)
                hidden_valid = F.interpolate(valid.float(), size=(sh, sw), mode="nearest")
                aligned_hidden = aligned_hidden * hidden_valid * hidden_inside
            current = network(images[:, 1:2], hidden=aligned_hidden,
                              context_features=[level[:, 1:2].to(device) for level in context_cache[i]],
                              **context_args)
            logits = current["logits"][:, 0]
            source_loss, diagnostics = interval_supervision(logits, annotations, config["classes"])
            localization_loss, localization_diagnostics = confirmed_position_interval_localization_loss(
                logits,annotations,config["classes"],original_sample=samples[i],crop_origin_xy=(x,y),rank_weight=rank_weight)
            primitive_loss = primitive_supervision(current["primitive_logits"][:, 0], annotations)
            background_loss, background_counts = masked_region_loss(
                logits, current["primitive_logits"][:, 0], background_mask,
                semantic_class=0, primitive_class=2)
            region_losses, region_consumption_counts = [], []
            for region_index, (region, region_mask) in enumerate(source_reviewed_region_masks(
                    samples[i], (x, y), (crop_h, crop_w), device)):
                if region.get("excluded_kinds"):
                    if region.get("kind") is not None:
                        raise ValueError("A source semantic exclusion cannot silently assign another positive class")
                    region_loss, counts = semantic_exclusion_loss(
                        logits, region_mask, region["excluded_kinds"], config["classes"], annotations)
                else:
                    region_loss, counts = masked_region_loss(
                        logits, current["primitive_logits"][:, 0], region_mask,
                        semantic_class=config["classes"].index(region["kind"]),
                        primitive_class=("line", "circle", "other").index(region.get("primitive", "other")))
                region_consumption_counts.append(dict(region_index=region_index,region_id=region.get("region_id"),
                    supervised_pixels=counts["supervised_pixels"],protected_human_pixels=0,
                    human_protection_policy=("fail_on_human_overlap" if region.get("excluded_kinds")
                                             else "no_human_mask_subtraction")))
                if counts["supervised_pixels"]:
                    region_losses.append(region_loss)
            reviewed_region_loss = torch.stack(region_losses).mean() if region_losses else logits.sum() * 0
            temporal_loss = logits.sum() * 0
            if previous is not None:
                previous_probability = previous["logits"][:, 0].float().softmax(dim=1)
                aligned_probability, inside = warp(previous_probability, backward)
                divergence = F.kl_div(logits.float().log_softmax(dim=1), aligned_probability.detach(), reduction="none").sum(dim=1, keepdim=True)
                mask = valid.float() * inside.float()
                temporal_loss = (divergence * mask).sum() / mask.sum().clamp_min(1)
            loss = source_loss + localization_weight*localization_loss + .15 * temporal_loss + primitive_weight * primitive_loss + background_loss + reviewed_region_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        if not loss.requires_grad:
            raise RuntimeError("Current source training loss has no trainable parameter path")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite training gradient")
        optimizer.step()
        consumption = training_supervision_consumption_record(
            step=step+1,sample_index=i,sample=samples[i],selected_annotations=annotations,
            source_loss_diagnostics=diagnostics,classes=config["classes"],shot_key=sample_group["shot_key"],
            crop_origin_xy=(x,y),crop_hw=(crop_h,crop_w),full_frame=full_frame,
            previous_source_observation=previous_roles[i],optimizer_step_succeeded=True,
            reviewed_region_counts=region_consumption_counts,background_counts=background_counts,
            localization_diagnostics=localization_diagnostics)
        consumption['actual_loss_values']=dict(source_classification=float(source_loss.detach()),
            confirmed_position_localization=float(localization_loss.detach()),
            confirmed_position_localization_weight=localization_weight,total=float(loss.detach()))
        consumption['confirmed_human_crop_target']=crop_target
        increment=confirmed_source_point_coverage(samples[i],[consumption])
        for identity,values in increment["groups"].items():
            crop_coverage[coverage_key]["groups"][identity]=(
                np.asarray(crop_coverage[coverage_key]["groups"][identity],np.int64)+np.asarray(values,np.int64)).tolist()
        with consumption_path.open("a",encoding="utf-8",newline="\n") as ledger:
            ledger.write(json.dumps(consumption,sort_keys=True,allow_nan=False)+"\n")
        consumption_steps += 1
        report["optimizer_source_consumption"]["successful_steps_recorded"] = consumption_steps
        guard_failed=False
        if consumption_steps%guard_every==0 or consumption_steps==total_steps:
            current_retention=native_retention_snapshots()
            comparisons=[source_retention_regressions(current,baseline)
                         for current,baseline in zip(current_retention,baseline_retention)]
            guard_failed=any(not row["retained_without_new_classification_errors"] for row in comparisons)
            guard_record["checks"].append(dict(step=step+1,comparisons=comparisons,snapshots=current_retention))
            guard_record["failed"]=guard_failed
            write_json(guard_path,guard_record)
            report["source_retention_guard"].update(failed=guard_failed,last_checked_step=step+1,
                new_error_source_count=sum(not row["retained_without_new_classification_errors"] for row in comparisons))
            print(json.dumps(dict(stage="actual_native_source_retention_guard",step=step+1,
                failed=guard_failed,new_error_source_count=report["source_retention_guard"]["new_error_source_count"])),flush=True)
        if (step + 1) % 20 == 0 or step == start_step or guard_failed or consumption_steps==total_steps:
            record = dict(step=step + 1, source_pts=samples[i]["source_pts"],
                          source_hash=samples[i]["source_hash"], shot_key=sample_group["shot_key"],
                          native_training_hw=[crop_h, crop_w], full_frame=full_frame,
                          loss=float(loss.detach()), source_loss=float(source_loss.detach()),
                          confirmed_position_localization_loss=float(localization_loss.detach()),
                          confirmed_position_localization_weight=localization_weight,
                          temporal_loss=float(temporal_loss.detach()), gradient_norm=float(norm),
                          primitive_loss=float(primitive_loss.detach()),
                          background_loss=float(background_loss.detach()),
                          background_pixels=background_counts["supervised_pixels"],
                          reviewed_region_loss=float(reviewed_region_loss.detach()),
                          previous_source_observation_available=previous_roles[i]["available"],
                          previous_memory_role=previous_roles[i]["reason"],
                          seconds=time.monotonic() - started)
            report["history"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            checkpoint = dict(model_config=network.model_config, state_dict=network.state_dict(),
                              optimizer=optimizer.state_dict(), step=step + 1,
                              manifest_sha256=report["manifest_sha256"], classes=config["classes"],
                              supervision_sha256=supervision_digest,
                              rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state(),
                              cuda_rng_states=torch.cuda.get_rng_state_all(),
                              source_manifest=manifest,
                              training_update_scope=actual_scope,
                              confirmed_source_crop_coverage=crop_coverage,
                              balanced_source_sampling_state=sampling_state,
                              source_retention_guard_reference=reference_record,
                              accuracy_accepted=False,
                              supervision="direct_user_confirmed_UV_and_original_source_review")
            _atomic_checkpoint(checkpoint_path, checkpoint)
            report["latest_step"] = step + 1
            write_json(report_path, report)
        if guard_failed and guard_action == "stop":
            break
    report["held_after_training"] = evaluate()
    report["optimizer_source_consumption"].update(
        sha256=sha256(consumption_path), requested_steps_recorded=(consumption_steps==total_steps))
    report["source_retention_guard"].update(sha256=sha256(guard_path))
    final_frozen_parameter_sha={name:hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,parameter in network.named_parameters() if not parameter.requires_grad}
    if final_frozen_parameter_sha!=initial_frozen_parameter_sha:
        raise RuntimeError("Frozen primitive/shared model parameters changed during the scoped positional experiment")
    report["frozen_parameter_sha256"]=initial_frozen_parameter_sha
    report["all_frozen_parameters_preserved_byte_exact"]=True
    report.update(status=(("stopped_after_actual_confirmed_source_regression" if guard_action == "stop" else "candidate_trained_with_recorded_source_regressions") if report["source_retention_guard"]["failed"]
                          else "source_visual_supervision_trained_full_accuracy_pending"),
                  elapsed_seconds=time.monotonic() - started,
                  latest_step=start_step + consumption_steps,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
    final_path = run_data / "rink_vision_candidate.pt" if candidate else ROOT / config["final_model"]
    _atomic_checkpoint(final_path, dict(model_config=network.model_config,
                                       state_dict=network.state_dict(), classes=config["classes"],
                                       training_report=report, source_manifest=manifest,
                                       accuracy_accepted=False))
    report.update(final_model=str(final_path), final_model_sha256=sha256(final_path),
                  final_model_bytes=final_path.stat().st_size)
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("status", "latest_step", "final_model", "final_model_bytes")}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="RinkSpace GPU visual module")
    parser.add_argument("command", choices=("prepare", "train", "review-shot", "read-frame"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rink_vision.json")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--pts", type=int)
    parser.add_argument("--guides", type=Path)
    parser.add_argument("--model", type=Path, help="explicit inference weights; selecting a training candidate does not promote it")
    parser.add_argument("--original-fit-request", type=Path, help="authenticated original FIT locators and physical measurement definitions to reread")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint",type=Path,help="explicit frozen checkpoint with complete optimizer and RNG state")
    parser.add_argument("--retention-guard-every",type=int,help="native confirmed-source regression check cadence in successful optimizer steps")
    parser.add_argument("--update-scope",choices=("all_trainable","semantic_head"),help="explicit minimal parameter-update scope, with full optimizer state retained")
    parser.add_argument("--learning-rate",type=float,help="explicit learning-rate override, preserving optimizer moments")
    parser.add_argument("--position-rank-weight",type=float,help="auxiliary confirmed position peak-ranking weight; original CE and mass retained")
    parser.add_argument("--retention-reference",type=Path,help="preserve an existing actual native retention baseline across a resumed experiment")
    parser.add_argument("--candidate", action="store_true", help="Stage parameters and checkpoint in Data for real-image comparison")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-pts", type=int)
    parser.add_argument("--end-pts", type=int)
    parser.add_argument("--time-base")
    parser.add_argument("--shot-id")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--reset-pts", type=int, action="append", default=[])
    args = parser.parse_args()
    config, data = configure(args.config)
    if args.command == "prepare":
        from .vision_data import prepare_dataset
        print(prepare_dataset(ROOT, data), flush=True)
    elif args.command == "read-frame":
        if args.pts is None:
            parser.error("read-frame requires --pts in the original video time base")
        read_frame(args.config, args.pts, args.guides, model_path=args.model,
                   original_fit_request_path=args.original_fit_request)
    elif args.command == "review-shot":
        required = ("video", "start_pts", "end_pts", "time_base", "shot_id", "expected_sha256")
        missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
        if missing:
            parser.error("review-shot requires " + ", ".join(missing))
        review_shot(args.config, args.video, args.start_pts, args.end_pts, args.time_base,
                    args.shot_id, args.expected_sha256, reset_pts=args.reset_pts, model_path=args.model,
                    original_fit_request_path=args.original_fit_request)
    else:
        train(args.config, steps=args.steps, resume=args.resume, candidate=args.candidate,
              resume_checkpoint=args.resume_checkpoint,retention_guard_every=args.retention_guard_every,
              update_scope=args.update_scope,learning_rate=args.learning_rate,position_rank_weight=args.position_rank_weight,
              retention_reference=args.retention_reference)


if __name__ == "__main__":
    main()
