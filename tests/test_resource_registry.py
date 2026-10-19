"""Relocation and source-identity checks without solver, native or media imports."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from rinkspace.resource_registry import ResourceRegistry


@pytest.fixture
def registry(tmp_path):
    project, data = tmp_path / "relocated-project", tmp_path / "relocated-data"
    (project / "configs").mkdir(parents=True)
    data.mkdir()
    storage = {"data_root": "D:/old/Data", "legacy_root": "D:/old/RinkSpace",
               "project_paths": {"calibration/inputs": "events/example/models/inputs"},
               "legacy_paths": {"outputs": "events/example/history"}}
    (project / "configs/storage.json").write_text(json.dumps(storage), encoding="utf-8")
    asset = project / "events/example/observations/source.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'{"original_reference":"D:/old/Data/unchanged.npz"}\n')
    manifest = {"assets": [{"source_path": "D:/old/Data/source.json",
                            "project_path": "events/example/observations/source.json",
                            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}]}
    (project / "events/example/asset_preservation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ResourceRegistry.from_project(project, data)


def test_preserved_alias_precedes_data_prefix_and_preserves_bytes(registry):
    resolved = registry.resolve("d:\\OLD\\data\\SOURCE.JSON", must_exist=True, verify=True)
    assert resolved.root_kind == "project"
    assert resolved.alias_registered
    assert b"D:/old/Data/unchanged.npz" in resolved.path.read_bytes()
    assert resolved.path == registry.project / "events/example/observations/source.json"
    assert registry.resolve(resolved.template_reference(), verify=True).expected_sha256 == resolved.expected_sha256
    assert registry.resolve(resolved.path, verify=True).expected_sha256 == resolved.expected_sha256


def test_relocated_data_and_specific_legacy_roots(registry):
    assert registry.resolve("D:/old/Data/checkpoints/seed.npz").path == registry.data / "checkpoints/seed.npz"
    assert registry.resolve("D:/old/RinkSpace/outputs/x.json").path == registry.data / "events/example/history/x.json"
    assert registry.resolve("D:/old/RinkSpace/calibration/inputs/a.json").path == registry.project / "events/example/models/inputs/a.json"
    assert registry.resolve("@project/events/example/observations/source.json").path.is_file()
    assert registry.resolve("relative/file.json").path == registry.project / "relative/file.json"


@pytest.mark.parametrize("reference", ["@project/../private", "@data/../../private",
                                       "D:/old/Data/../private", "D:/unregistered/private",
                                       "@project/C:/private", "@unknown/private", "../private", "bad\x00path"])
def test_unregistered_roots_and_traversal_are_rejected(registry, reference):
    with pytest.raises(ValueError):
        registry.resolve(reference)


def test_identity_check_rejects_modified_copy(registry):
    resolved = registry.resolve("D:/old/Data/source.json")
    resolved.path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        registry.resolve("D:/old/Data/source.json", verify=True)


def test_relative_reference_is_independent_of_process_working_directory(registry, monkeypatch):
    monkeypatch.chdir(registry.data)
    assert registry.resolve("relative/file.json").path == registry.project / "relative/file.json"


def test_missing_and_unhashed_resource_are_distinct(registry):
    with pytest.raises(FileNotFoundError):
        registry.resolve("@data/missing.npz", must_exist=True)
    with pytest.raises(ValueError, match="No expected SHA"):
        registry.resolve("@data/missing.npz", verify=True)


def test_conflicting_alias_and_overlapping_roots_are_rejected(registry):
    with pytest.raises(ValueError, match="Conflicting"):
        registry.add_alias("D:/old/Data/source.json", root_kind="data", relative_path="other.json")
    with pytest.raises(ValueError, match="separate"):
        ResourceRegistry(registry.project, registry.project / "data")


def test_template_and_materialization_use_relocated_roots(registry):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_portable_profile.py"))
    profile = registry.project / "configs/demo.json"
    profile.write_text(json.dumps({"schema": "rinkspace.reusable-solver-inputs.v1", "driver": "named66",
                                  "resource_root": "D:/old/Data", "source_options": {
                                      "--source": "D:/old/Data/source.json", "--workers": "6", "--flag": True}}), encoding="utf-8")
    before = profile.read_bytes()
    prepared = tool["template"](registry, profile)
    assert profile.read_bytes() == before
    assert prepared["source_options"]["--source"] == "@project/events/example/observations/source.json"
    assert prepared["source_options"]["--workers"] == "6"
    assert prepared["source_options"]["--flag"] is True
    report = tool["materialize"](registry, prepared)
    assert report["schema"] == "rinkspace.resolved-input-preparation.v1"
    assert report["runtime_activated"] is False
    assert report["missing_direct_resources"] == []
    assert report["candidate_profile"]["resource_root"] == str(registry.data)
    assert report["candidate_profile"]["source_options"]["--source"].startswith(str(registry.project))
    assert report["verified_registered_asset_fields"] == ["--source"]


def test_unresolved_profile_cannot_materialize(registry):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_portable_profile.py"))
    profile = registry.project / "configs/demo.json"
    profile.write_text(json.dumps({"schema": "rinkspace.reusable-solver-inputs.v1", "driver": "named66",
                                  "resource_root": "D:/old/Data", "source_options": {"--source": "C:/outside/a.json"}}), encoding="utf-8")
    prepared = tool["template"](registry, profile)
    assert len(prepared["unresolved_direct_references"]) == 1
    with pytest.raises(ValueError, match="unresolved direct references"):
        tool["materialize"](registry, prepared)


def test_scope_and_continuation_metadata_survive_preparation(registry):
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_portable_profile.py"))
    policy = {"allowed_static_surfaces": ["ice", "board"], "unknown_or_unreviewed_default": "excluded"}
    profile = registry.project / "configs/strict.json"
    document = {"schema": "rinkspace.reusable-solver-inputs.v1", "driver": "named66", "event_id": "example",
                "resource_root": "D:/old/Data", "source_options": {"--source": "D:/old/Data/source.json"},
                "strict_model_scope_policy": policy, "continuation": {"fresh_J_required": True}, "accuracy_accepted": False}
    profile.write_text(json.dumps(document), encoding="utf-8")
    prepared = tool["template"](registry, profile, verify_registered_assets=False)
    report = tool["materialize"](registry, prepared, verify_registered_assets=False)
    candidate = report["candidate_profile"]
    assert candidate["strict_model_scope_policy"] == policy
    assert candidate["continuation"] == document["continuation"]
    assert candidate["accuracy_accepted"] is False
    assert report["source_eligibility_certified_by_preparation"] is False
    assert report["verified_registered_asset_fields"] == []
    assert report["registered_asset_fields_not_read"] == ["--source"]
    assert prepared["registered_asset_verification_performed"] is False


def test_prefix_conflict_is_rejected_and_identical_registration_is_idempotent(registry):
    count = len(registry.prefixes)
    registry.add_prefix("d:/OLD/data", root_kind="data")
    assert len(registry.prefixes) == count
    with pytest.raises(ValueError, match="Conflicting resource prefix"):
        registry.add_prefix("D:/old/Data", root_kind="project", relative_path="other")
    assert registry.resolve("D:/old/Data/new.npz").root_kind == "data"


def test_later_hash_registration_promotes_all_aliases(registry):
    registry.add_alias("C:/source/a.json", root_kind="project", relative_path="new.json")
    expected = "a" * 64
    registry.add_alias("C:/source/b.json", root_kind="project", relative_path="new.json", expected_sha256=expected)
    assert registry.resolve("C:/source/a.json").expected_sha256 == expected
    assert registry.resolve("@project/new.json").expected_sha256 == expected
    with pytest.raises(ValueError, match="identity"):
        registry.add_alias("C:/source/c.json", root_kind="project", relative_path="new.json", expected_sha256="b" * 64)
    assert registry.resolve("C:/source/a.json").expected_sha256 == expected


def test_exact_project_relocation_keeps_original_and_portable_hashes(registry):
    old_relative = "events/example/observations/source.json"
    new_relative = "events/example/observations/s1.json"
    original = registry.project / old_relative
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    target = registry.project / new_relative
    original.rename(target)
    (registry.project / "configs/path_aliases.json").write_text(json.dumps({
        "schema": "rinkspace.project-path-aliases.v1", "aliases": [{
            "old_path": old_relative, "new_path": new_relative, "sha256": expected}]}), encoding="utf-8")
    # The preservation manifest intentionally still contains the old target path.
    relocated = ResourceRegistry.from_project(registry.project, registry.data)
    references = (old_relative, "@project/" + old_relative, registry.project / old_relative,
                  "D:/old/RinkSpace/" + old_relative, "D:/old/Data/source.json")
    for reference in references:
        resource = relocated.resolve(reference, must_exist=True, verify=True)
        assert resource.path == target
        assert resource.expected_sha256 == expected
        assert resource.alias_registered
    assert relocated.resolve("@project/" + new_relative, verify=True).path == target
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_portable_profile.py"))
    prepared = {"schema": "rinkspace.portable-input-template.v1", "source_profile": {}, "driver": "named66",
                "resource_root": "@data/", "source_options": {"--source": "@project/" + old_relative},
                "unresolved_direct_references": [], "resources": [{"field": "--source",
                    "reference": "@project/" + old_relative, "expected_sha256": expected}]}
    report = tool["materialize"](relocated, prepared)
    assert report["candidate_profile"]["source_options"]["--source"] == str(target)
    assert report["verified_registered_asset_fields"] == ["--source"]
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        relocated.resolve(old_relative, verify=True)
    with pytest.raises(ValueError, match="identity differs"):
        tool["materialize"](relocated, prepared)


def test_relocation_hash_conflict_with_preservation_manifest_is_rejected(registry):
    (registry.project / "configs/path_aliases.json").write_text(json.dumps({
        "schema": "rinkspace.project-path-aliases.v1", "aliases": [{
            "old_path": "events/example/observations/source.json", "new_path": "events/example/observations/s1.json",
            "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        ResourceRegistry.from_project(registry.project, registry.data)


def write_privacy_derivative(registry, relative, original, public):
    document = {"schema": "rinkspace.privacy-derivatives.v1", "files": [{
        "path": relative, "original_sha256": original, "public_sha256": public}]}
    (registry.project / "configs/privacy_derivatives.json").write_text(json.dumps(document), encoding="utf-8")
    return document


def test_public_derivative_uses_current_identity_without_changing_history(registry):
    relative = "events/example/observations/source.json"
    asset = registry.project / relative
    manifest = registry.project / "events/example/asset_preservation_manifest.json"
    historical_manifest = manifest.read_bytes()
    original_bytes = asset.read_bytes()
    original = hashlib.sha256(original_bytes).hexdigest()
    public_bytes = b'{"public_reference":"local-source"}\n'
    current = hashlib.sha256(public_bytes).hexdigest()
    asset.write_bytes(public_bytes)
    write_privacy_derivative(registry, relative, original, current)
    public = ResourceRegistry.from_project(registry.project, registry.data)
    for reference in ("D:/old/Data/source.json", "@project/" + relative, relative, asset):
        resource = public.resolve(reference, must_exist=True, verify=True)
        assert resource.path == asset
        assert resource.expected_sha256 == current
    assert manifest.read_bytes() == historical_manifest
    public.add_alias("C:/explicit/public.json", root_kind="project", relative_path=relative, expected_sha256=current)
    with pytest.raises(ValueError, match="identity"):
        public.add_alias("C:/wrong.json", root_kind="project", relative_path=relative, expected_sha256="0" * 64)
    # A retained Data original keeps its original identity, independent of the project path.
    data_original = registry.data / "original.json"
    data_original.write_bytes(original_bytes)
    public.add_alias("C:/private/original.json", root_kind="data", relative_path="original.json", expected_sha256=original)
    assert public.resolve("C:/private/original.json", verify=True).expected_sha256 == original
    # Registration is not a whitelist: the original bytes also fail the public identity check.
    asset.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="identity differs"):
        public.resolve(relative, verify=True)


def test_public_derivative_after_project_path_relocation(registry):
    old_relative = "events/example/observations/source.json"
    new_relative = "events/example/observations/s1.json"
    old = registry.project / old_relative
    original = hashlib.sha256(old.read_bytes()).hexdigest()
    target = registry.project / new_relative
    old.rename(target)
    target.write_bytes(b'{"public":true}\n')
    current = hashlib.sha256(target.read_bytes()).hexdigest()
    (registry.project / "configs/path_aliases.json").write_text(json.dumps({
        "schema": "rinkspace.project-path-aliases.v1", "aliases": [{
            "old_path": old_relative, "new_path": new_relative, "sha256": original}]}), encoding="utf-8")
    write_privacy_derivative(registry, new_relative, original, current)
    public = ResourceRegistry.from_project(registry.project, registry.data)
    for reference in (old_relative, "@project/" + old_relative, "@project/" + new_relative,
                      "D:/old/RinkSpace/" + old_relative, "D:/old/Data/source.json"):
        resource = public.resolve(reference, verify=True)
        assert resource.path == target
        assert resource.expected_sha256 == current
    public.add_alias("C:/new-public.json", root_kind="project", relative_path=old_relative, expected_sha256=current)
    assert public.resolve("C:/new-public.json", verify=True).path == target


def test_public_derivative_wrong_original_does_not_override_preservation(registry):
    relative = "events/example/observations/source.json"
    write_privacy_derivative(registry, relative, "0" * 64, "1" * 64)
    with pytest.raises(ValueError, match="derivative identity"):
        ResourceRegistry.from_project(registry.project, registry.data)


@pytest.mark.parametrize("bad_path", ["../private.json", "@project/../../private", "D:/private.json", "."])
def test_public_derivative_paths_are_bounded(registry, bad_path):
    write_privacy_derivative(registry, bad_path, "0" * 64, "1" * 64)
    with pytest.raises(ValueError):
        ResourceRegistry.from_project(registry.project, registry.data)


def test_public_derivative_configuration_rejects_duplicate_and_invalid_identity(registry):
    relative = "events/example/observations/source.json"
    original = hashlib.sha256((registry.project / relative).read_bytes()).hexdigest()
    document = write_privacy_derivative(registry, relative, original, "1" * 64)
    document["files"].append(document["files"][0].copy())
    path = registry.project / "configs/privacy_derivatives.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ResourceRegistry.from_project(registry.project, registry.data)
    document["files"] = [dict(document["files"][0], public_sha256="invalid")]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        ResourceRegistry.from_project(registry.project, registry.data)
