"""Restore saved source uncertainties after sparse edge expansion.

The accepted UV and opaque identities remain immutable. ``space_boundary_samples``
keeps pre-expansion vertex scales in endpoint_evidence arrays and stores only new
interior scales in point_bindings. Their original ordered complement is the exact
index map; it is not interpolation or a default uncertainty. A fresh current RGB
profile is also recorded without using its midpoint to move the accepted UV.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .single_frame_source_review import transition

SCHEMA = "rinkspace.accepted-source-scale-recovery.v1"
PROFILE_FEATURES = {"board_contact": "yellow", "Ad2.top_edge": "magenta",
                    "Ad2.bottom_edge": "magenta"}


def reference(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest)


def read_reference(ref):
    if reference(ref["path"]) != ref:
        raise ValueError("Immutable source reference changed: " + ref["path"])
    return json.loads(Path(ref["path"]).read_text(encoding="utf8"))


def point_id(feature_id, segment_index, segment, vertex_index):
    ids = segment.get("point_ids")
    if ids is not None:
        if len(ids) != len(segment["uv"]):
            raise ValueError("Opaque point IDs no longer index the accepted UV")
        return ids[vertex_index]
    return f"{feature_id}/segment/{segment_index}/vertex/{vertex_index}"


def recover_endpoint_scale(segment, vertex_index):
    """Recover an original vertex's exact compact-array index, sigma and IQR."""
    n = len(segment["uv"])
    bindings = segment.get("point_bindings", [])
    sigmas = np.asarray(segment.get("endpoint_evidence_localization_sigma_px", []), float)
    widths = np.asarray(segment.get("endpoint_evidence_transition_iqr_px", []), float)
    if len(bindings) != n or sigmas.ndim != 1 or widths.shape != sigmas.shape:
        raise ValueError("Exact expanded bindings and original endpoint evidence required")
    original = [i for i, binding in enumerate(bindings)
                if "localization_sigma_px" not in binding]
    if len(original) != len(sigmas) or vertex_index not in original:
        raise ValueError("Endpoint evidence cannot be mapped uniquely to original vertices")
    if not (np.isfinite(sigmas).all() and np.isfinite(widths).all()
            and (sigmas > 0).all() and (widths > 0).all()):
        raise ValueError("Original source scales must be finite and positive")
    # compact_run rounds widths and sigmas independently to 0.001 px.
    if not np.allclose(sigmas, widths / 1.349, rtol=0, atol=0.001):
        raise ValueError("Original sigma no longer agrees with source transition IQR")
    index = original.index(vertex_index)
    return dict(original_endpoint_evidence_index=index,
                ordered_original_endpoint_vertex_indices=original,
                sigma_px=float(sigmas[index]), source_transition_iqr_px=float(widths[index]),
                scale_field=f"endpoint_evidence_localization_sigma_px[{index}]",
                width_field=f"endpoint_evidence_transition_iqr_px[{index}]")


def native_profile_evidence(raw, uv, feature_id, mask=None):
    """Read the existing vertical chroma half-transition at unchanged UV."""
    signal = PROFILE_FEATURES.get(feature_id)
    if signal is None:
        raise ValueError("No source-authorized material signal for this feature")
    uv = np.asarray(uv, float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("Finite native source UV required")
    x, y = np.rint(uv).astype(int)
    radius = 28  # Same actual native profile radius as the opening source reader.
    ys = np.arange(y-radius, y+radius+1)
    xs = np.arange(max(0, x-2), min(raw.shape[1], x+3))
    if ys[0] < 0 or ys[-1] >= raw.shape[0] or not len(xs):
        raise ValueError("Complete native profile support unavailable")
    colors = raw[np.ix_(ys, xs)].astype(float)
    b, g, r = colors[:, :, 0], colors[:, :, 1], colors[:, :, 2]
    scalar = (r+g)/2-b if signal == "yellow" else (r+b)/2-g
    profile = np.median(scalar, axis=1)
    result = transition(profile, radius)
    if result is None:
        raise ValueError("Current RGB material quartiles are unresolved")
    quantiles, amplitude = result
    q = ys[0] + np.asarray(quantiles)
    width = float(q[2]-q[0])
    if width <= 0 or not np.isfinite(width):
        raise ValueError("Positive measured material transition width required")
    lo, hi = int(np.floor(q[0])), int(np.ceil(q[2]))
    return dict(method="existing_native_colour_half_transition",
        signal=signal, profile_normal_xy=[0., 1.], normal_radius_px=radius,
        tangent_native_columns=xs.tolist(), normal_native_rows=ys.tolist(),
        source_support_xyxy_inclusive=[int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])],
        median_native_chroma_profile=profile.tolist(),
        quantile_fractions=[.25, .5, .75], material_quantile_native_y=q.tolist(),
        material_quantile_uv=[[float(x), float(v)] for v in q],
        measured_transition_iqr_px=width, measured_sigma_px=width/1.349,
        native_side_contrast=amplitude,
        diagnostic_midpoint_offset_from_accepted_uv_px=float(q[1]-uv[1]),
        accepted_UV_modified=False, current_profile_does_not_reestimate_accepted_source_position=True,
        profile_person_mask_any=(bool(mask[np.ix_(ys, xs)].any()) if mask is not None else None),
        transition_iqr_person_mask_any=(bool(mask[lo:hi+1, xs[0]:xs[-1]+1].any())
                                       if mask is not None else None))


def _manual_before(packet, frame, acceptance):
    manual = packet.get("manual_editor")
    if not manual:
        return None
    tx_id = manual["revision_id"]
    candidates = [r for r in acceptance.get("manual_edits", {}).get("transactions", [])
                  if r["id"] == tx_id]
    if len(candidates) != 1:
        raise ValueError("Accepted manual edit transaction reference required")
    tx_ref = {k: candidates[0][k] for k in ("path", "sha256")}
    tx = read_reference(tx_ref)
    if (tx["status"] != "committed" or tx["index"] != frame["index"]
            or tx["source_pts"] != frame["native_pts"] or tx["packet_after"] != frame["packet"]):
        raise ValueError("Manual transaction no longer binds the accepted packet")
    folder = Path(tx_ref["path"]).parent
    binary = folder / tx["files"][0]["before_file"]
    if reference(binary)["sha256"] != tx["action"]["base_revision"]:
        raise ValueError("Manual before-packet revision differs")
    before_path = folder / "before_packet.json"
    before = json.loads(before_path.read_text(encoding="utf8"))
    if before != json.loads(binary.read_text(encoding="utf8")):
        raise ValueError("Manual before-packet snapshot differs from its original bytes")
    return before, dict(transaction=tx_ref, before_packet=reference(before_path),
                        original_before_bytes=reference(binary))


def _verify_manual_point(before, source_feature_id, opaque_id, uv, recovered):
    found = []
    for feature in before["features"]:
        if feature["id"] != source_feature_id:
            continue
        for si, segment in enumerate(feature["segments"]):
            for vi, point in enumerate(segment["uv"]):
                if point_id(feature["id"], si, segment, vi) == opaque_id:
                    found.append((segment, vi, point))
    if len(found) != 1 or found[0][2] != uv:
        raise ValueError("Manual source identity/UV was not preserved")
    prior = recover_endpoint_scale(found[0][0], found[0][1])
    for key in ("sigma_px", "source_transition_iqr_px", "original_endpoint_evidence_index"):
        if prior[key] != recovered[key]:
            raise ValueError("Manual editing changed the source uncertainty address")


def recover_dataset(selection_path, output_path):
    selection_ref = reference(selection_path)
    selection = read_reference(selection_ref)
    records, failures, methods, seen = [], [], Counter(), set()
    requested_roles = Counter()
    for frame in selection["frames"]:
        if not any(o["selected_vertex_indices"].get("missing_source_scale")
                   or o["source_feature_id"] == "Ad2.visible_left_outline"
                   for o in frame["observations"]):
            continue
        packet, binding = read_reference(frame["packet"]), read_reference(frame["binding"])
        acceptance = read_reference(frame["user_acceptance"])
        accepted = next(r for r in acceptance["frames"] if r["index"] == frame["index"])
        if (acceptance.get("accepted") is not True
                or accepted["packet_sha256"] != frame["packet"]["sha256"]
                or accepted["binding_sha256"] != frame["binding"]["sha256"]
                or binding["source_packet"] != frame["packet"]):
            raise ValueError("Explicit user acceptance does not authenticate this source packet")
        native = packet["source"]
        if native["time_base"] != "1/12800" or native["native_pts"] != native["actual_pts"]:
            raise ValueError("Official native source clock required")
        raw = cv2.imread(native["image_path"])
        if raw is None or hashlib.sha256(raw.tobytes()).hexdigest() != native["decoded_BGR_sha256"]:
            raise ValueError("Current native RGB identity differs")
        mask_ref = dict(path=native["mask_path"], sha256=native["mask_sha256"])
        if reference(mask_ref["path"]) != mask_ref:
            raise ValueError("Native person mask bytes changed")
        mask = cv2.imread(mask_ref["path"], cv2.IMREAD_GRAYSCALE) != 0
        features = {f["id"]: f for f in packet["features"]}
        manual = _manual_before(packet, frame, acceptance)
        for observation in frame["observations"]:
            fid, si = observation["source_feature_id"], observation["source_segment_index"]
            segment = features[fid]["segments"][si]
            candidates = {vi: "missing_source_scale" for vi in
                          observation["selected_vertex_indices"].get("missing_source_scale", [])}
            if fid == "Ad2.visible_left_outline":
                for vi in observation["selected_vertex_indices"].get("shared_line_evidence", []):
                    pb = segment["point_bindings"][vi]
                    if (pb.get("kind") == "edge_sample"
                            and pb.get("physical_target") in ("Ad2.top_edge", "Ad2.bottom_edge")
                            and "localization_sigma_px" not in pb):
                        candidates[vi] = "shared_native_endpoint_without_scale"
            for vi, role in candidates.items():
                requested_roles[role] += 1
                pid = point_id(fid, si, segment, vi)
                key = (native["native_pts"], pid)
                if key in seen:
                    raise ValueError("Duplicate accepted source point identity")
                seen.add(key)
                address = dict(frame=frame["index"], native_pts=native["native_pts"],
                    source_feature_id=fid, source_segment_index=si, source_vertex_index=vi,
                    source_point_id=pid, uv=segment["uv"][vi], target_id=observation["target_id"],
                    original_selection_role=role,
                    source_packet=frame["packet"], binding=frame["binding"],
                    user_acceptance=frame["user_acceptance"],
                    source_video_sha256=native["source_video_sha256"], time_base=native["time_base"],
                    source_image=reference(native["image_path"]),
                    decoded_BGR_sha256=native["decoded_BGR_sha256"])
                try:
                    profile_feature = observation["target_id"] if fid == "Ad2.visible_left_outline" else fid
                    profile = native_profile_evidence(raw, address["uv"], profile_feature, mask)
                    if role == "missing_source_scale":
                        recovered = recover_endpoint_scale(segment, vi)
                        if manual:
                            _verify_manual_point(manual[0], fid, pid, address["uv"], recovered)
                        recovery_kind = "original_compact_endpoint_source_sigma_index_restored"
                    else:
                        recovered = dict(sigma_px=profile["measured_sigma_px"],
                            source_transition_iqr_px=profile["measured_transition_iqr_px"],
                            scale_field="native_RGB_profile.measured_sigma_px",
                            width_field="native_RGB_profile.measured_transition_iqr_px")
                        recovery_kind = "fresh_accepted_native_endpoint_material_profile"
                    record = dict(address, **recovered,
                        recovery_kind=recovery_kind,
                        source_estimator="native_material_transition_IQR_divided_by_1.349",
                        native_RGB_profile=profile,
                        accepted_source_position_and_original_sigma_reestimated=False,
                        source_uncertainty_newly_measured=(role == "shared_native_endpoint_without_scale"),
                        source_uncertainty_is_not_geometric_fit_error=True,
                        manual_edit_source_address_verified=(manual[1] if manual else None))
                    records.append(record)
                    methods[record["recovery_kind"]] += 1
                except ValueError as error:
                    failures.append(dict(address, reason=str(error), default_sigma_filled=False))
    if requested_roles["missing_source_scale"] != selection["counts"]["missing_source_scale"]:
        raise ValueError("Entire declared missing-scale inventory must be accounted for")
    if len(records)+len(failures) != sum(requested_roles.values()):
        raise ValueError("Every requested source endpoint needs an explicit result")
    result = dict(schema=SCHEMA, selection=selection_ref, recovered_points=len(records),
        failed_points=len(failures), methods=dict(methods), records=records, failures=failures,
        requested_roles=dict(requested_roles),
        source_coordinates_modified=False, original_packets_modified=False,
        default_sigma_filled=False, new_source_pixels_added=False,
        source_only_RGB_profile_measurement=True, solver_or_camera_projection_used=False,
        optimization_performed=False, accuracy_accepted=False,
        implementation=reference(__file__),
        source_estimator_implementation=reference(Path(__file__).with_name("single_frame_source_review.py")))
    output_path = Path(output_path)
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != data:
        raise ValueError("Existing frozen recovery must be preserved")
    if not output_path.exists():
        output_path.write_bytes(data)
    return reference(output_path), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref, result = recover_dataset(args.selection, args.output)
    print(json.dumps(dict(reference=ref, recovered_points=result["recovered_points"],
                          failed_points=result["failed_points"])))


if __name__ == "__main__":
    main()
