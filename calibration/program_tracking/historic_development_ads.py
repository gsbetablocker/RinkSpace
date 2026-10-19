"""Previously inspected, real source-UV advertisement corners for development fits.

These three whole historical files are training inputs, never fresh validation.
The 24-point 2026-09-27 source-only set is an exposure-level exclusion gate;
its pixel values and scores are deliberately never accessed here.  No model
prediction/error stored beside a source observation is used for selection.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from continuous_brown_adapter import adapt_factor
from projection import from_state, require_central_projection
from reference_board_pair import BoardIntersectionError


HERE = Path(__file__).resolve().parent
TMP = Path("D:/CodexData/RinkSpace_program_tracking_tmp")
BASE_SCENE_SHA256 = "08b4e0f68340a279354f735e5b501a0a0a8e254f451caef0f431e3d29328215b"
SOURCE_SHA256 = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
SOURCE_FILES = (
    ("old_frozen_development", TMP / "native_ad_holdout_v8/measurements/corners.json",
     "d7608cd7a10e09872953fb9cb2fc278eba1b6372c14aa9b613c00ee0213575b1", 102),
    ("shot06_frozen_development", TMP / "shot06_ad_source/v2_native_holdout_230477056/frozen_observations.json",
     "aebafd4c2ffff0c5c1ddbbc03d50caa6252c0c7a46477a345e58def2550c6c52", 6),
    ("shot09_old_development", TMP / "native_ad_holdout_v9/shots06_09.json",
     "f999cf8cba896815f5dbfcb5f4df7df64aaa033599b54d3f84c9eb2e5f31db96", 55),
)
GENERATOR_FILES = {
    HERE / "ad_corner_holdout.py": "1c40ef374bcd251054a2251eac0498a79a55b1624f9da384c9e03526da504a99",
    HERE / "ad_corners.py": "7f6bacee351510702e4e8cc5dc61cc8a66b9db3bbfb3a3b4733f80511e7c56cf",
}
VALIDATION_GATE = HERE / "validation_source_only_20260927.json"
VALIDATION_GATE_SHA256 = "da797d253e2be8d2c2f088468727195a5105e86f938e0510d3db36acdf54d1dd"
SENSOR_SIZE = np.array([1920., 1080.])
ROLES = {"left_bottom", "left_top", "right_bottom", "right_top"}


def _read_pinned(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Pinned source or generator changed: {path}: {actual}")
    return json.loads(payload)


def _uv(value, source: str) -> tuple[float, float]:
    uv = np.asarray(value, np.float64)
    if (uv.shape != (2,) or not np.isfinite(uv).all() or
            np.any(uv < 0.) or np.any(uv >= SENSOR_SIZE)):
        raise ValueError(f"Invalid original 1920x1080 source UV: {source}")
    return float(uv[0]), float(uv[1])


def _canonical(pts: int, number: int, role: str,
               aliases: dict[int, int]) -> tuple[int, int, str]:
    return (int(aliases[pts]), int(number), str(role))


def _validation_exposures(aliases: dict[int, int], source_sha256: str) -> set[int]:
    # The gate is parsed for PTS/exposure/identity ONLY.  Its observations'
    # pixel values, errors, predictions and scores are not referenced.
    gate = _read_pinned(VALIDATION_GATE, VALIDATION_GATE_SHA256)
    if gate["source_sha256_from_reviewed_timeline"] != source_sha256:
        raise ValueError("24-point validation gate belongs to another source")
    if len(gate["frames"]) != 10 or sum(len(frame["observations"])
                                       for frame in gate["frames"]) != 24:
        raise ValueError("24-point validation identity manifest changed")
    excluded = set()
    for frame in gate["frames"]:
        pts = int(frame["pts"])
        if pts in aliases:
            excluded.add(int(aliases[pts]))
    return excluded


def _fitted_keys(data: dict, aliases: dict[int, int]) -> tuple[set[tuple], dict]:
    keys = set()
    report = {}
    observed_inputs = data["diag"]["observation_input_sha256"]
    source_hash = data["captured"]["source_hash"]
    for name in data["config"]["ad_corner_observations"]:
        path = Path(name).resolve()
        expected = observed_inputs.get(str(path))
        if expected is None:
            raise ValueError(f"Fitted advertisement evidence not in v13 provenance: {path}")
        manifest = _read_pinned(path, expected)
        if manifest["source_sha256"] != source_hash:
            raise ValueError(f"Fitted advertisement evidence has wrong source: {path}")
        selected = 0
        for raw in manifest["observations"]:
            if raw.get("status") != "measured":
                continue
            pts = int(raw["pts"])
            if pts in aliases:
                keys.add(_canonical(pts, raw["user_number"], raw["role"], aliases))
                selected += 1
        report[str(path)] = {"sha256": expected, "selected_measured_rows": selected}
    return keys, report


def prepare_development_ads(data: dict, aliases: dict[int, int]) -> tuple[list[dict], dict]:
    """Return complete-file historical source corners selected by this camera group.

    Eligibility, ID and uncertainty are fixed by pinned source/generator files,
    not by the current candidate's residual.  An alias exposure/ID/role that
    was already fitted can add no second weight.  The whole 24-point validation
    exposure is excluded, regardless of ad or corner role.
    """
    aliases = {int(p): int(rep) for p, rep in aliases.items()}
    if set(aliases) != set(data["pts"]) or any(rep not in aliases or aliases[rep] != rep
                                              for rep in aliases.values()):
        raise ValueError("Exposure aliases must cover exactly the selected source PTS")
    if data["captured"]["source_hash"] != SOURCE_SHA256 or data["diag"]["base_model_sha256"] != BASE_SCENE_SHA256:
        raise ValueError("Development advertisement evidence belongs to another source/scene")
    if data["scene"]["programme_calibration_candidate"]["base_model_sha256"] != BASE_SCENE_SHA256:
        raise ValueError("Current candidate scene has a different base model")
    for path, digest in GENERATOR_FILES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source-corner generator policy changed: {path}")
    ads = {int(ad["user_number"]): ad for ad in data["scene"]["advertisements"]}
    fitted, fitted_report = _fitted_keys(data, aliases)
    validation = _validation_exposures(aliases, SOURCE_SHA256)
    seen = set(fitted)
    selected_rows = []
    files = {}
    counts = Counter()
    for category, path, digest, expected_count in SOURCE_FILES:
        source = _read_pinned(path, digest)
        raw_rows = source["observations"]
        if (source["source_sha256"] != SOURCE_SHA256 or
                source["model_sha256"] != BASE_SCENE_SHA256 or
                len(raw_rows) != expected_count):
            raise ValueError(f"Historical development source/model/scope changed: {path}")
        if category == "shot09_old_development" and not source["not_used_in_joint_corner_fit"]:
            raise ValueError("Shot09 historical observations have a changed fit role")
        file_count = Counter()
        for raw in raw_rows:
            pts, number, role = int(raw["pts"]), int(raw["user_number"]), str(raw["role"])
            if number == 4:
                # #4's short board side has no supported rectangular absolute corners.
                file_count["excluded_ad4_geometry"] += 1
                continue
            if number not in {1, 2, 3, 5, 6, 7} or role not in ROLES:
                raise ValueError(f"Unaudited development advertisement identity: {category}, {pts}")
            wanted_status = ("holdout_actual_source_pixel_not_joint_fit"
                             if category == "shot09_old_development" else "measured")
            if raw.get("status") != wanted_status:
                raise ValueError(f"Historical development measurement status changed: {path}")
            if category != "shot06_frozen_development" and raw.get("mask_status") != "measured_mask":
                raise ValueError(f"Historical measurement mask status changed: {path}")
            if number == 5 and raw.get("measurement_component") != "meiwa_color_bar_envelope":
                raise ValueError("Ad #5 source component differs from internal graphic")
            if number != 5 and raw.get("measurement_component") == "meiwa_color_bar_envelope":
                raise ValueError("Internal graphic was assigned to another advertisement")
            # Never read predicted_uv, error_px, error_xy_px, modelXYZ, or any
            # projection-derived field from a historical measurement packet.
            observed = _uv(raw["observed_uv"] if category == "shot09_old_development"
                           else raw["uv"], f"{category}, {pts}, {number}, {role}")
            if category == "shot09_old_development":
                if "sigma_px" in raw:
                    raise ValueError("Shot09 now stores sigma: re-audit its policy")
                source_sigma = 4. if number == 5 else 3.
                sigma_provenance = "reconstructed_from_generator_policy"
            else:
                source_sigma = float(raw["sigma_px"])
                sigma_provenance = "stored_source_measurement_policy"
            if not np.isfinite(source_sigma) or source_sigma <= 0:
                raise ValueError("Invalid historical source sigma")
            file_count["audited_source_rows"] += 1
            if pts not in aliases:
                file_count["outside_selected_camera_group"] += 1
                continue
            key = _canonical(pts, number, role, aliases)
            if key[0] in validation:
                file_count["validation_exposure_excluded"] += 1
                continue
            if key in seen:
                file_count["already_fitted_or_alias_duplicate"] += 1
                continue
            ad = ads.get(number)
            if ad is None or role not in ad["corners_xyz_scene_units"]:
                raise ValueError(f"Current v13 candidate scene lacks ad {number} {role}")
            expected_scope = "internal_graphic_only" if number == 5 else "external_border"
            if ad.get("extent_scope") != expected_scope:
                raise ValueError(f"Advertisement {number} has a changed physical extent")
            xyz = np.asarray(ad["corners_xyz_scene_units"][role], np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError("Candidate scene advertisement corner is invalid")
            seen.add(key)
            selected_rows.append(dict(
                pts=pts, representative_pts=key[0], user_number=number, role=role,
                observed_uv=observed, scene_xyz=tuple(map(float, xyz)),
                source_sigma_px=source_sigma, sigma_provenance=sigma_provenance,
                source_category=category, source_path=str(path.resolve()),
                source_sha256=digest, measurement_origin=raw["measurement_origin"],
                measurement_component=raw.get("measurement_component"),
                identity_evidence=raw.get("identity_evidence")))
            file_count["attached_candidates"] += 1
        files[category] = dict(path=str(path.resolve()), sha256=digest,
                               source_rows=len(raw_rows), **dict(file_count))
        counts.update(file_count)
    selected_rows.sort(key=lambda r: (r["pts"], r["user_number"], r["role"]))
    metadata = dict(
        role="historical_development_training; previously_scored_not_new_validation",
        source_sha256=SOURCE_SHA256,
        base_scene_sha256=BASE_SCENE_SHA256,
        candidate_scene_sha256=data["diag"]["model_sha256"],
        selected_camera_id=data["camera_id"],
        files=files, fitted_inputs=fitted_report,
        fitted_unique_exposure_identity_count=len(fitted),
        excluded_validation_manifest=dict(path=str(VALIDATION_GATE.resolve()),
            sha256=VALIDATION_GATE_SHA256, frames=10, observations=24,
            selected_exposure_count=len(validation),
            policy="whole_confirmed_exposure; pixels/scores never accessed"),
        missing_source_fields={"shot09_old_development": ["sigma_px"]},
        reconstructed_sigma_policy=dict(
            label="reconstructed_from_generator_policy",
            sigma_px_by_ad={"2": 3., "3": 3., "5": 4., "6": 3., "7": 3.},
            original_packet_sigma_stored=False,
            meaning="generator default uncertainty, not measured covariance",
            generator_sha256={str(p.resolve()): h for p, h in GENERATOR_FILES.items()}),
        counts=dict(counts), prepared_rows=len(selected_rows),
        ad4_policy="exclude_all_absolute_rectangular_corners; do not invent rectangle",
        residual_selection_used=False, historic_final_ads_used=False)
    return selected_rows, metadata


def attach_development_ad_factors(graph, data: dict, model, initial: dict,
                                  rows: list[dict]) -> dict:
    """Attach observed-minus-predicted source-UV factors through physical15.

    The exact same 3px Huber threshold and source-quality sigma multiplier as
    existing direct ad corners are used.  Core11 projection derivatives chain
    through the rig model; the Brown adapter adds independent Brown4 columns.
    """
    active = set(map(int, initial))
    quality = data["captured"]["source_quality"]
    effective_sigmas, multipliers = [], []
    counts = Counter()
    attached_identities = set()
    for row in rows:
        pts = int(row["pts"])
        identity = (int(row["representative_pts"]), int(row["user_number"]),
                    str(row["role"]))
        if identity in attached_identities:
            raise ValueError(f"Development ad source identity repeated: {identity}")
        attached_identities.add(identity)
        if pts not in model.frames or pts not in data["pts"]:
            raise ValueError(f"Development ad source PTS absent from selected model: {pts}")
        if int(model.frames[pts].frame_key) != int(row["representative_pts"]):
            raise ValueError("Development ad exposure alias differs from model")
        uv = np.asarray(row["observed_uv"], np.float64)
        xyz = np.asarray(row["scene_xyz"], np.float64).reshape(1, 3)
        template = model.frames[pts].template25
        source_sigma = float(row["source_sigma_px"])
        multiplier = float(quality.scales(pts, [uv], "ad_corner")[0])
        sigma = source_sigma * multiplier
        if not np.isfinite(sigma) or sigma <= 0 or multiplier < 1.:
            raise ValueError("Invalid development ad quality/sigma")

        def residual(states, world=xyz, observed=uv, camera_template=template,
                     stamp=pts):
            projected, _ = require_central_projection(
                from_state(states[0], camera_template), world,
                stage="residual", family="historical_development_ad_corner",
                source_keys=(stamp,))
            return projected[0] - observed

        def jacobian(states, world=xyz, camera_template=template, stamp=pts):
            camera = from_state(states[0], camera_template)
            _, _, derivative = require_central_projection(
                camera, world, with_jacobian=True, stage="jacobian",
                family="historical_development_ad_corner", source_keys=(stamp,))
            return [derivative]

        adapted = adapt_factor((pts,), residual, jacobian,
                               dict(residual_dim=2, sigma=sigma, huber_k=3.),
                               active, data["states"], model)
        if adapted is None:
            raise ValueError("Development ad lost all active camera dependencies")
        keys, value, derivative, options = adapted
        def checked_residual(q, *, fn=value, stamp=pts):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "residual"
                exc.factor_family = "historical_development_ad_corner"
                exc.source_keys = (stamp,)
                raise

        def checked_derivative(q, *, fn=derivative, stamp=pts):
            try:
                return fn(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage = "jacobian"
                exc.factor_family = "historical_development_ad_corner"
                exc.source_keys = (stamp,)
                raise

        graph.add_observation(keys, checked_residual,
                              jac_fn=checked_derivative, **options)
        effective_sigmas.append(sigma)
        multipliers.append(multiplier)
        counts[row["source_category"]] += 1
    return dict(attached_factors=len(rows), by_source=dict(counts),
                quality_kind="ad_corner", huber_k=3.,
                sigma_rule="stored_or_generator_source_sigma_px * existing_source_quality_multiplier",
                effective_sigma_px_min=min(effective_sigmas) if effective_sigmas else None,
                effective_sigma_px_max=max(effective_sigmas) if effective_sigmas else None,
                source_quality_multiplier_max=max(multipliers) if multipliers else None,
                scene_geometry="fixed_current_v13_candidate_corners_scene_units",
                no_source_uv_changed=True)
