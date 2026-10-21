"""Focused publication-boundary and redaction tests; no repository mutations."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_github_readiness.py"
spec = importlib.util.spec_from_file_location("check_github_readiness", SCRIPT)
readiness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = readiness
spec.loader.exec_module(readiness)


def test_github_ordinary_boundaries_are_strictly_above():
    mib = readiness.inventory.MIB
    assert readiness.size_class(50 * mib) == "within_github_ordinary_file_warning"
    assert readiness.size_class(50 * mib + 1) == "above_github_ordinary_file_warning"
    assert readiness.size_class(100 * mib) == "above_github_ordinary_file_warning"
    assert readiness.size_class(100 * mib + 1) == "above_github_ordinary_file_limit"


def test_lfs_attribute_does_not_convert_raw_index_or_committed_blob():
    assert readiness.inspect_lfs_blob(101 * readiness.inventory.MIB, None, {"filter": "lfs"}) == "raw_blob_despite_lfs_attribute"
    pointer = {"payload_sha256": "a" * 64, "payload_bytes": 101 * readiness.inventory.MIB}
    assert readiness.inspect_lfs_blob(133, pointer, {"filter": "lfs"}) == "lfs_pointer"
    assert readiness.inspect_lfs_blob(133, pointer, {}) == "lfs_pointer_without_effective_lfs_filter"


def test_remote_userinfo_queries_and_fragments_never_reach_report():
    remote = readiness.safe_remote("https://username:private_value@github.com/owner/project.git?access_token=hidden#hidden_fragment")
    serialized = json.dumps(remote)
    assert remote["host"] == "github.com"
    assert remote["repository_path"] == "/owner/project.git"
    for forbidden in ("username", "private_value", "access_token", "hidden_fragment"):
        assert forbidden not in serialized
    assert readiness.safe_remote("git@github.com:owner/project.git")["host"] == "github.com"
    assert readiness.safe_remote("D:/private/repo.git") == {"kind": "local_or_unrecognized", "endpoint": "redacted"}
    assert readiness.safe_remote("ssh://user:pw@invalid:broken/repo")["host"] == "invalid"


def test_tracked_cache_and_local_config_require_review_but_arrays_and_images_do_not():
    assert "tracked_runtime_or_cache" in readiness.tracked_review_reasons("pkg/__pycache__/cached.pyc")
    assert "tracked_compiled_artifact" in readiness.tracked_review_reasons("pkg/__pycache__/cached.pyc")
    assert readiness.tracked_review_reasons("configs/local.json")
    assert readiness.tracked_review_reasons(".env")
    assert not readiness.tracked_review_reasons(".env.example")
    assert not readiness.tracked_review_reasons("events/example/scene.npz")
    assert not readiness.tracked_review_reasons("events/example/source_evidence.png")


def test_credential_scanner_returns_locations_without_values():
    credential = "ghp_" + "aB3c" * 10
    found, _ = readiness.inventory.scan_text(("credential = '" + credential + "'").encode(), "sample.txt", "index")
    assert found == [{"path": "sample.txt", "line": 1, "category": "github_token_format", "layer": "index"}]
    assert credential not in json.dumps(found)


def test_public_guides_cannot_require_local_panels_or_missing_demo(tmp_path):
    names = set(readiness.PUBLIC_GUIDES + readiness.QUICKSTART_FILES)
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public entry", encoding="utf-8")
    assert readiness.public_entry_findings(tmp_path, names) == []
    (tmp_path / "START_HERE.html").write_text("local task state", encoding="utf-8")
    (tmp_path / "README.md").write_text("[results](START_HERE.html)", encoding="utf-8")
    assert {"path": "README.md", "reason": "public_guide_links_to_local_only_entry"} in readiness.public_entry_findings(tmp_path, names)
    (tmp_path / "src/rinkspace/demo.py").unlink()
    assert {"path": "src/rinkspace/demo.py", "reason": "missing_public_entry_or_quickstart_file"} in readiness.public_entry_findings(tmp_path, names)
    assert readiness.tracked_review_reasons("events/example/programmes/test/source_datasets/batch/review.html")
    assert not readiness.tracked_review_reasons("src/rinkspace/static/frame_point_editor.html")
