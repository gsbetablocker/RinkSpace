"""Prepare opt-in resource templates; never invoke a solver or rewrite its inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PureWindowsPath
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rinkspace.resource_registry import ResourceRegistry, sha256
from rinkspace.path_policy import guard_project_write


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def is_absolute_reference(value):
    return isinstance(value, str) and (Path(value).is_absolute() or bool(PureWindowsPath(value).drive))


def template(registry: ResourceRegistry, profile: Path, *, verify_registered_assets: bool = True):
    profile = profile.resolve()
    if not profile.is_relative_to(registry.project):
        raise ValueError("Source profile must be inside the selected checkout")
    original_sha = sha256(profile)
    document = json_read(profile)
    if document.get("schema") != "rinkspace.reusable-solver-inputs.v1":
        raise ValueError("Unsupported source profile; this tool does not infer another input contract")
    declared = copy.deepcopy(document["source_options"])
    resources, unresolved = [], []
    fields = {"resource_root": document["resource_root"], **declared}
    for field, value in fields.items():
        if not is_absolute_reference(value):
            continue
        try:
            resource = registry.resolve(value)
        except ValueError as error:
            unresolved.append({"field": field, "original_reference": value, "reason": str(error)})
            continue
        item = {"field": field, "original_reference": value,
                "reference": resource.template_reference(), "root_kind": resource.root_kind,
                "relative_path": resource.relative_path, "expected_sha256": resource.expected_sha256,
                "alias_registered": resource.alias_registered, "exists_at_capture": resource.path.exists()}
        item["identity_verification"] = ("verified" if verify_registered_assets else "not_read_in_metadata_mode") if resource.expected_sha256 else "no_registry_hash"
        if resource.expected_sha256 is not None and verify_registered_assets:
            registry.resolve(value, must_exist=True, verify=True)
        resources.append(item)
        if field != "resource_root":
            declared[field] = resource.template_reference()
    if sha256(profile) != original_sha:
        raise ValueError("Source profile changed during preparation; no template was written")
    root_resource = next((row for row in resources if row["field"] == "resource_root"), None)
    return {
        "schema": "rinkspace.portable-input-template.v1",
        "source_profile": {"project_path": profile.relative_to(registry.project).as_posix(), "sha256": original_sha},
        "driver": document["driver"],
        "profile_metadata": {key: copy.deepcopy(value) for key, value in document.items()
                             if key not in {"schema", "driver", "resource_root", "source_options"}},
        "resource_root": root_resource["reference"] if root_resource else document["resource_root"],
        "source_options": declared,
        "resources": resources, "unresolved_direct_references": unresolved,
        "runtime_activated": False,
        "portable_complete_run_verified": False,
        "source_eligibility_certified_by_preparation": False,
        "registered_asset_verification_performed": verify_registered_assets,
        "scope": "Only explicit profile options and resource_root. Immutable source JSON contents, pickle identities and native contracts are unchanged.",
    }


def materialize(registry: ResourceRegistry, prepared: dict, *, verify_registered_assets: bool = True):
    if prepared.get("schema") != "rinkspace.portable-input-template.v1":
        raise ValueError("Input is not a portable preparation template")
    if prepared["unresolved_direct_references"]:
        raise ValueError("Template has unresolved direct references")
    candidate = copy.deepcopy(prepared.get("profile_metadata", {}))
    candidate.update({"schema": "rinkspace.reusable-solver-inputs.v1", "driver": prepared["driver"],
                      "resource_root": str(registry.resolve(prepared["resource_root"]).path),
                      "source_options": copy.deepcopy(prepared["source_options"])})
    unresolved, verified, not_read = [], [], []
    for item in prepared["resources"]:
        resource = registry.resolve(item["reference"])
        if not resource.path.exists():
            unresolved.append({"field": item["field"], "path": str(resource.path)})
        if item["expected_sha256"] is not None and verify_registered_assets:
            if sha256(resource.path) != item["expected_sha256"]:
                raise ValueError(f"Prepared input identity differs: {item['field']}")
            verified.append(item["field"])
        elif item["expected_sha256"] is not None:
            not_read.append(item["field"])
        if item["field"] != "resource_root":
            candidate["source_options"][item["field"]] = str(resource.path)
    return {
        "schema": "rinkspace.resolved-input-preparation.v1",
        "source_profile": prepared["source_profile"],
        "project_root": str(registry.project), "data_root": str(registry.data),
        "candidate_profile": candidate,
        "verified_registered_asset_fields": verified,
        "registered_asset_fields_not_read": not_read,
        "profile_metadata_preserved": "profile_metadata" in prepared,
        "profile_metadata_paths_remapped": False,
        "missing_direct_resources": unresolved,
        "runtime_activated": False, "portable_complete_run_verified": False,
        "source_eligibility_certified_by_preparation": False,
        "remaining_validation": ["Nested source and profile-metadata references must use the same opt-in registry in a reviewed adapter.",
                                 "Preserved scope policy is not evidence that every source point satisfies current surface rules.",
                                 "Existing import/pickle producer identities and source hashes must be retained.",
                                 "Native build and complete-state continuation require independent validation."],
    }


def write_new(path: Path, content: dict, *, project_root: Path = PROJECT):
    path = guard_project_write(path, project_root=project_root)
    rendered = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            print(f"Already prepared: {path}")
            return
        raise FileExistsError(f"Refusing to replace an existing preparation file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(f"Prepared: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("template", "materialize"))
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--template-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="Read the selected profile and resource metadata; skip hashing registered asset contents.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = (args.data_root or Path(json_read(root / "configs/storage.json")["data_root"])).resolve()
    registry = ResourceRegistry.from_project(root, data)
    output = (args.output if args.output.is_absolute() else (root if args.mode == "template" else data) / args.output).resolve()
    if args.mode == "template":
        if not args.profile or not output.is_relative_to(root / "configs/portable"):
            parser.error("Template requires --profile and a new output under configs/portable")
        selected = args.profile if args.profile.is_absolute() else root / args.profile
        document = template(registry, selected, verify_registered_assets=not args.metadata_only)
    else:
        if not args.template_file or not output.is_relative_to(data / "runtime_tools/repository_validation/portable_profiles"):
            parser.error("Materialize requires --template-file and a new output under Data/runtime_tools/repository_validation/portable_profiles")
        selected = args.template_file if args.template_file.is_absolute() else root / args.template_file
        document = materialize(registry, json_read(selected), verify_registered_assets=not args.metadata_only)
    write_new(output, document, project_root=root)


if __name__ == "__main__":
    main()
