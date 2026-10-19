"""Resolve the current shared 3D scene without changing frozen revisions.

This selector is for the scene3d calibration model. The older rink_model.json
delivery selector used by video localization is a separate artifact format.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .advertisement_topology import validate_user_confirmed_advertisements
from .scene_camera import Scene
from ..storage import resolve_asset


ACTIVE_CONFIG = Path("configs/rink_model/active_shared_scene3d.json")
ANKARA_CONFIRMED_ADVERTISEMENTS = Path(
    "configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json"
)


def _project_file(project: Path, relative_name: str, label: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute():
        raise ValueError(f"{label} path must be project-relative")
    return resolve_asset(project, relative, must_exist=True)


def _checked_file(project: Path, relative_name: str, expected_sha: str, label: str) -> Path:
    path = _project_file(project, relative_name, label)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _validate_refit_release(project: Path, config: dict, data: dict) -> None:
    """Validate the separate release record without changing the fitted scene bytes."""
    release_path = _checked_file(
        project, config["release_path"], config["release_sha256"], "active scene release"
    )
    release = json.loads(release_path.read_text(encoding="utf-8-sig"))
    if release.get("schema_version") != "rinkspace.shared-scene-release.v1":
        raise ValueError("invalid shared scene release schema")
    for field in ("revision", "model_id", "model_path", "model_sha256", "model_schema_version"):
        if release.get(field) != config.get(field):
            raise ValueError(f"active scene release {field} mismatch")
    if (release["model_schema_version"] != "rinkspace.three_frame_refit.scene.v1"
            or data.get("schema_version") != release["model_schema_version"]
            or data.get("model_id") != release["model_id"]):
        raise ValueError("active shared scene identity mismatch")
    if (release.get("event_model_id") != "ankara_2026_shared"
            or config.get("event_model_id") != release["event_model_id"]):
        raise ValueError("active scene release event identity mismatch")
    if (config.get("user_confirmed_advertisement_contract_sha256")
            != release.get("semantic_contract_sha256")):
        raise ValueError("active scene release advertisement contract hash mismatch")
    contract_path = _checked_file(
        project, release["semantic_contract_path"],
        release["semantic_contract_sha256"], "user-confirmed advertisement contract"
    )
    if contract_path != _project_file(
        project, str(ANKARA_CONFIRMED_ADVERTISEMENTS), "advertisement contract"
    ):
        raise ValueError("active scene release references a different advertisement contract")
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    if (contract.get("schema_version") != "rinkspace.user-confirmed-advertisement-topology.v1"
            or len(contract.get("positions", [])) != 26):
        raise ValueError("user-confirmed advertisement contract is incomplete")
    # The refit estimates only observed advertisements. Its ten candidate boundaries
    # are not a replacement for the separate 26-instance identity contract.
    numbers = [row.get("user_number") for row in data.get("advertisements", [])]
    if len(numbers) != len(set(numbers)) or any(n not in range(1, 27) for n in numbers):
        raise ValueError("refit advertisement numbers are invalid")
    _checked_file(project, release["fit_path"], release["fit_sha256"], "refit fit")
    _checked_file(
        project, release["observations_path"], release["observations_sha256"],
        "refit observations"
    )
    source = data.get("source", {})
    if (Path(source.get("selected_fit_path", "")) != Path(release["fit_path"])
            or source.get("selected_fit_sha256") != release["fit_sha256"]
            or Path(source.get("observation_file", "")) != Path(release["observations_path"])
            or source.get("observation_sha256") != release["observations_sha256"]
            or source.get("selected_fit_complete_flag") is not True):
        raise ValueError("active scene release does not match fitted source evidence")


def active_scene_path(root: str | Path | None = None) -> Path:
    """Return the selected scene after verifying its revision and file hash."""
    project = (Path(root) if root is not None else Path(__file__).resolve().parents[3]).resolve()
    config = json.loads((project / ACTIVE_CONFIG).read_text(encoding="utf-8"))
    if config.get("schema_version") != "rinkspace.active-shared-scene3d.v1":
        raise ValueError("invalid active shared scene selector schema")
    path = _checked_file(
        project, config["model_path"], config["model_sha256"], "active shared scene"
    )
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if "release_path" in config:
        _validate_refit_release(project, config, data)
        return path
    if (data.get("schema_version") != "rinkspace.scene3d.v1"
            or data.get("revision") != config["revision"]
            or data.get("model_id") != config["model_id"]):
        raise ValueError("active shared scene identity mismatch")
    if config["model_id"] == "ankara_2026_shared":
        contract_path = _project_file(project, str(ANKARA_CONFIRMED_ADVERTISEMENTS), "advertisement contract")
        contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if config.get("user_confirmed_advertisement_contract_sha256") != contract_sha:
            raise ValueError("active shared scene advertisement contract hash mismatch")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        validate_user_confirmed_advertisements(data, contract, contract_sha)
        snapshot_relative = Path(data["advertisement_topology"]["source_pptx_snapshot"])
        if snapshot_relative.is_absolute():
            raise ValueError("user-confirmed review snapshot path must be project-relative")
        snapshot = _project_file(project, str(snapshot_relative), "user-confirmed review snapshot")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != contract["source_pptx_sha256"]:
            raise ValueError("user-confirmed review snapshot SHA-256 mismatch")
        observation_relative = Path(data["source"]["high_view_ad_corner_observations"])
        if observation_relative.is_absolute():
            raise ValueError("high-view advertisement observation path must be project-relative")
        observation_path = _project_file(project, str(observation_relative), "high-view advertisement observation")
        observation_sha = hashlib.sha256(observation_path.read_bytes()).hexdigest()
        if (config.get("high_view_ad_corner_observations_sha256") != observation_sha
                or data["source"].get("high_view_ad_corner_observations_sha256") != observation_sha):
            raise ValueError("active scene lost high-view advertisement observations")
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        if (observation.get("schema_version") != "rinkspace.high-view-advert-corners.v1"
                or observation.get("source_frame", {}).get("source_pts") != 56218880
                or [row.get("user_number") for row in observation.get("advertisements", [])] != list(range(1, 8))
                or sum(len(row.get("corner_observations", [])) for row in observation["advertisements"]) != 26):
            raise ValueError("high-view advertisement observation record is incomplete")
        for evidence in (observation["user_annotation"], observation["source_frame"]):
            evidence_relative = Path(evidence["snapshot_path"] if "snapshot_path" in evidence else evidence["image_path"])
            if evidence_relative.is_absolute():
                raise ValueError("high-view advertisement image path must be project-relative")
            evidence_path = _project_file(project, str(evidence_relative), "high-view advertisement image")
            expected_sha = evidence["sha256"] if "snapshot_path" in evidence else evidence["image_sha256"]
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != expected_sha:
                raise ValueError("high-view advertisement evidence image SHA-256 mismatch")
        by_number = {ad["user_number"]: ad for ad in data["advertisements"]}
        for row in observation["advertisements"]:
            evidence = by_number[row["user_number"]].get("high_view_corner_evidence", {})
            if (evidence.get("record_sha256") != observation_sha
                    or evidence.get("source_pts") != 56218880
                    or evidence.get("point_letters") != [p["letter"] for p in row["corner_observations"]]):
                raise ValueError("active scene dropped a user-marked advertisement corner")
    return path


def load_active_scene(root: str | Path | None = None) -> Scene:
    """Load the selected scene3d model, including unresolved lane metadata."""
    project = (Path(root) if root is not None else Path(__file__).resolve().parents[3]).resolve()
    scene = Scene(active_scene_path(project))
    # The refit keeps its frozen fit-linked scene immutable; its human version
    # belongs to the separately hashed release manifest and selector.
    config = json.loads((project / ACTIVE_CONFIG).read_text(encoding="utf-8"))
    scene.revision = config["revision"]
    return scene
