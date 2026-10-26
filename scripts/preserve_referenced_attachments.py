"""Preserve only registered clipboard annotation originals, without changing refs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rinkspace.storage import configured_data_root
from rinkspace.resource_registry import ResourceRegistry, reference_key, sha256


def state(path: Path):
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino, "device": value.st_dev}


def main():
    data = configured_data_root(PROJECT)
    registry = ResourceRegistry.from_project(PROJECT, data)
    manifest_path = PROJECT / "events/ankara_2026/asset_preservation_manifest.json"
    output = PROJECT / "events/ankara_2026/external_resource_preservation.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        for row in existing["assets"]:
            target = (data / row["data_relative_path"]).resolve()
            if not target.is_relative_to(data) or sha256(target) != row["sha256"]:
                raise ValueError("Existing registered attachment identity differs")
        print(f"Already preserved and verified: {len(existing['assets'])} attachments")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    dependencies = manifest["external_dependencies"]
    selected = [row for row in dependencies
                if row["category"] == "media_or_frame_evidence_retained_external"
                and "/temp/" in row["source_path"].replace("\\", "/").casefold()
                and Path(row["source_path"]).name.startswith("codex-clipboard-")
                and Path(row["source_path"]).suffix.lower() == ".png"
                and any("attachment" in ref for ref in row["referenced_by"])]
    candidates, skipped_candidates = [], []
    for row in dependencies:
        original = row["source_path"].split("#", 1)[0]
        if not reference_key(original).startswith(reference_key(data).rstrip("/") + "/"):
            continue
        try:
            path = Path(original).resolve()
            if path.is_relative_to(data) and path.suffix.lower() == ".png" and path.is_file():
                candidates.append(path)
        except OSError as error:
            skipped_candidates.append({"path": original, "error": str(error), "existence": "unknown"})
    records = []
    for row in selected:
        original = Path(row["source_path"])
        before = state(original)
        payload = original.read_bytes()
        expected = hashlib.sha256(payload).hexdigest()
        if state(original) != before:
            raise ValueError(f"Attachment changed during read: {original.name}")
        identical = None
        for candidate in candidates:
            try:
                if candidate.stat().st_size == len(payload) and sha256(candidate) == expected:
                    identical = candidate
                    break
            except OSError as error:
                skipped_candidates.append({"path": str(candidate), "error": str(error), "existence": "unknown"})
        if identical is None:
            target = data / "events/ankara_2026/evidence/preserved_user_attachments" / (expected + ".png")
            target = target.resolve()
            if not target.is_relative_to(data):
                raise ValueError("Attachment target escapes Data")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError("Existing content-addressed target has different bytes")
                action = "reused_existing_preserved_data_asset"
            else:
                with target.open("xb") as stream:
                    stream.write(payload)
                action = "copied_to_data_byte_identical"
        else:
            target, action = identical, "reused_existing_registered_data_asset"
        if sha256(target) != expected or state(original) != before or sha256(original) != expected:
            raise ValueError("Attachment source/copy stability or identity check failed")
        registry.add_alias(str(original), root_kind="data", relative_path=target.relative_to(data).as_posix(), expected_sha256=expected)
        records.append({"source_path": str(original), "data_relative_path": target.relative_to(data).as_posix(),
                        "sha256": expected, "bytes": len(payload), "action": action,
                        "role": "registered_original_user_annotation_attachment",
                        "referenced_by": row["referenced_by"], "source_stat_at_capture": before,
                        "source_read_and_copy_stable": True, "source_and_target_sha256_match": True,
                        "active_source_references_rewritten": False})
    report = {"schema": "rinkspace.external-resource-preservation.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_manifest": manifest_path.relative_to(PROJECT).as_posix(), "source_manifest_sha256": sha256(manifest_path),
              "event_id": "ankara_2026", "assets": records, "unreadable_candidates_not_claimed_missing": skipped_candidates,
              "policy": "Only attachment originals already registered by source facts; media bytes remain Data. Existing identical Data copies are reused. No decoding, image generation, move or deletion.",
              "runtime_activated": False, "original_references_changed": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"registered_attachments": len(records), "new_copies": sum(row['action'] == 'copied_to_data_byte_identical' for row in records),
                      "bytes_registered": sum(row['bytes'] for row in records), "manifest": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
