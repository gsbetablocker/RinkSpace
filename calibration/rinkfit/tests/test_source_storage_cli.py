"""Public commit entry: evidence pins, actual records, immutable publication."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts/source_storage.py"
spec = importlib.util.spec_from_file_location("source_storage_cli_tested", CLI)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
Store = entry.SourceRecordStore


def dump(path, rows):
    with gzip.open(path, "wt", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SourceStorageCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("RINKSPACE_STORAGE_TEST_TMP"))
        self.root = Path(self.temp.name)
        self.db = self.root / "source.sqlite"
        self.rows = [{"selected_row_id": f"old/s05/{pts}/{i}", "source_point_id": f"raw/s05/{pts}/{i}",
                      "shot_id": "s05", "native_pts": pts, "time_base": "1/12800", "uv": [20.0 + i, 40.0],
                      "display_class": "ice", "independent_fit_weight_issued": False,
                      "physical_world_correspondence_issued": False}
                     for pts in (256, 512) for i in range(3)]
        self.frames = [{"shot_id": "s05", "native_pts": pts, "decoded_native_BGR_sha256": "c" * 64,
                        "points": [{k: r[k] for k in ("selected_row_id", "uv", "display_class")} for r in self.rows if r["native_pts"] == pts]}
                       for pts in (256, 512)]
        self.rowpath = self.root / "selected_rows.jsonl.gz"
        self.framepath = self.root / "selected_frames.jsonl.gz"
        dump(self.rowpath, self.rows)
        dump(self.framepath, self.frames)
        with Store(self.db) as store:
            ri = store.import_jsonl_stream(self.rowpath)
            fi = store.import_selected_frames(self.framepath, self.rowpath)
        self.evidence = self.root / "source_review.json"
        self.evidence.write_text('{"qualification":"explicit test fixture; no pixels certified"}', encoding="utf8")
        self.request = {"schema": entry.REQUEST_SCHEMA, "db": str(self.db),
                        "base_rows": {"stream": str(self.rowpath), "stream_semantic_sha256": ri["semantic_sha256"]},
                        "base_frames": {"stream": str(self.framepath), "stream_semantic_sha256": fi["semantic_sha256"]},
                        "new_rows_stream": "direct/candidate/rows", "new_frames_stream": "direct/candidate/frames",
                        "changes": [{"shot_id": "s05", "native_pts": 256, "remove": [self.rows[1]["selected_row_id"]],
                                     "update": {self.rows[0]["selected_row_id"]: {"display_class": "ice_inferred"}}}],
                        "source_evidence": [{"file": str(self.evidence), "sha256": hashlib.sha256(self.evidence.read_bytes()).hexdigest(), "role": "test fixture"}]}
        self.reqpath = self.root / "request.json"
        self.out = self.root / "candidate.json"
        self.write_request()

    def write_request(self):
        self.reqpath.write_text(json.dumps(self.request), encoding="utf8")

    def tearDown(self):
        self.temp.cleanup()

    def test_public_cli_commits_direct_delta_and_verifies_after_container_append(self):
        run = subprocess.run([sys.executable, "-X", "utf8", str(CLI), "commit", "--request", str(self.reqpath), "--out", str(self.out)], capture_output=True, text=True, encoding="utf8")
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["point_rows"], 5)
        manifest = json.loads(self.out.read_text(encoding="utf8"))
        self.assertEqual(manifest["quality_status"], "candidate_source_only")
        self.assertTrue(manifest["store_content_addressed"])
        self.assertEqual(manifest["qualification_delta"]["stored_preview_header_overrides"], 0)
        self.assertEqual(manifest["qualification_delta"]["actual_transition"]["removed_selected_rows"], 1)
        self.assertEqual(manifest["qualification_delta"]["actual_transition"]["changed_retained_qualification_rows"], 1)
        self.assertFalse(manifest["full_selected_rows_file_written"])
        self.assertFalse(manifest["full_selected_frames_file_written"])
        self.assertNotIn("sha256", manifest["selected_frames"])
        self.assertEqual(len(list(self.root.glob("*.jsonl.gz"))), 2)
        with Store(self.db) as store:
            self.assertEqual(store.stream_info(self.rowpath)["record_count"], 6)
            store.fork_rows_stream("another/later/stream", str(self.rowpath), {("s05", 512): {"remove": [self.rows[-1]["selected_row_id"]]}})
        # Old immutable manifest remains readable when the SQLite container appends.
        inspected = entry.inspect_manifest(self.out, verify=True)
        self.assertTrue(inspected["record_content_reverified"])
        self.assertEqual(inspected["point_rows"], 5)
        self.assertLess(len(json.dumps(inspected)), 4000)

    def test_bad_evidence_or_geometry_request_does_not_publish_or_mutate(self):
        self.request["source_evidence"][0]["sha256"] = "0" * 64
        self.write_request()
        with self.assertRaises(ValueError):
            entry.commit(self.reqpath, self.out)
        self.assertFalse(self.out.exists())
        with Store(self.db, readonly=True) as store:
            self.assertEqual(store.stats()["streams"], 2)
        self.request["source_evidence"][0]["sha256"] = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        self.request["changes"][0]["update"] = {self.rows[0]["selected_row_id"]: {"uv": [999, 99]}}
        self.write_request()
        with self.assertRaises(ValueError):
            entry.commit(self.reqpath, self.out)
        with Store(self.db, readonly=True) as store:
            self.assertEqual(store.stats()["streams"], 2)

    def test_actual_content_is_checked_even_when_descriptor_sha_is_unchanged(self):
        with Store(self.db) as store:
            sid = store.stream_info(self.rowpath)["id"]
            row = store.db.execute("SELECT blob_id FROM frames WHERE stream_id=? AND pts=256", (sid,)).fetchone()
            payload = store._get(row[0])
            payload["qualification"]["common"]["display_class"] = "board_top"
            blob = store._put(payload)
            store.db.execute("UPDATE frames SET blob_id=? WHERE stream_id=? AND pts=256", (blob, sid))
            store.db.commit()
        # Metadata still matches request; recovered records intentionally do not.
        with self.assertRaises(ValueError):
            entry.commit(self.reqpath, self.out)
        self.assertFalse(self.out.exists())
        with Store(self.db, readonly=True) as store:
            self.assertEqual(store.stats()["streams"], 2)

    def test_existing_manifest_is_never_overwritten(self):
        self.out.write_text("existing immutable receipt", encoding="utf8")
        with self.assertRaises(FileExistsError):
            entry.commit(self.reqpath, self.out)
        self.assertEqual(self.out.read_text(encoding="utf8"), "existing immutable receipt")
        with Store(self.db, readonly=True) as store:
            self.assertEqual(store.stats()["streams"], 2)

    def test_projection_failure_keeps_unpublished_recoverable_stream(self):
        with patch.object(Store, "project_selected_frames", side_effect=ValueError("deliberate projection failure")):
            with self.assertRaises(entry.CommitFailure) as error:
                entry.commit(self.reqpath, self.out)
        self.assertEqual(error.exception.sealed_streams, [self.request["new_rows_stream"]])
        self.assertFalse(self.out.exists())
        with Store(self.db, readonly=True) as store:
            self.assertEqual(store.stream_info(self.request["new_rows_stream"])["record_count"], 5)
            self.assertEqual(store.stream_info(self.rowpath)["record_count"], 6)


if __name__ == "__main__":
    unittest.main()
