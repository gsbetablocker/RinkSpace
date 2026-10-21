"""Lossless storage and qualification-delta invariants, without video decoding."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "src/rinkfit_joint/source_record_store.py"
spec = importlib.util.spec_from_file_location("source_record_store_tested", MODULE)
store_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store_module)
Store = store_module.SourceRecordStore
Error = store_module.SourceStoreError
Unknown = store_module.UnknownSchemaError


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def point(pts, i, shot="s05"):
    return {"selected_row_id": f"old/{shot}/{pts}/{i}", "shot_id": shot,
            "native_pts": pts, "time_base": "1/12800", "source_point_id": f"source/{shot}/{pts}/{i}",
            "uv": [i, 0.0 if i else -0.0], "display_class": "ice", "feature_id": "ice.C",
            "qualification": {"accepted": True, "reason": "Source proof " + "一致的原片说明" * 40,
                              "proof": {"file": "D:/dataset/same/path/" + "a" * 90, "sha256": "b" * 64}},
            "typed_values": [True, 1, 1.0, None, ["@s", 17], ["@l", ["@s", 4]]],
            "independent_fit_weight_issued": False}


def frames(rows):
    result = []
    for r in rows:
        key = r["shot_id"], r["native_pts"]
        if not result or (result[-1]["shot_id"], result[-1]["native_pts"]) != key:
            result.append({"shot_id": key[0], "native_pts": key[1], "decoded_native_BGR_sha256": str(key[1]) * 8, "points": []})
        result[-1]["points"].append({k: r[k] for k in ("selected_row_id", "uv", "display_class")})
    return result


class SourceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dbpath = self.root / "records.sqlite"
        self.store = Store(self.dbpath)
        self.rows = [point(p, i) for p in (256, 512) for i in range(4)]
        self.path = self.root / "selected_rows.jsonl.gz"
        write_rows(self.path, self.rows)
        self.info = self.store.import_jsonl_stream(self.path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def assert_json(self, a, b):
        self.assertEqual(store_module._json(a), store_module._json(b))

    def test_roundtrip_dictionary_types_order_readonly_and_immutability(self):
        self.assert_json(list(self.store.iterate_records(self.path)), self.rows)
        self.assertEqual(self.info["record_count"], 8)
        self.assertEqual(self.info["frame_count"], 2)
        self.assertEqual(self.info["original_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(self.store.stats()["source_points"], 8)
        self.assertGreater(self.store.stats()["atoms"], 0)
        one = list(self.store.iterate_records(self.path, frame=("s05", 256)))
        one[0]["uv"][0] = 999
        self.assertEqual(list(self.store.iterate_records(self.path, frame=256))[0]["uv"][0], 0)
        with self.assertRaises(Error):
            self.store.import_jsonl_stream(self.path)
        with self.assertRaises(KeyError):
            list(self.store.iterate_records(self.path, frame=999))
        with Store(self.dbpath, readonly=True) as reader:
            self.assert_json(list(reader.iterate_records(self.path)), self.rows)
            with self.assertRaises(Error):
                reader.fork_rows_stream("new", self.path, {})

    def test_import_delta_membership_qualifications_and_single_source_catalog(self):
        revised = json.loads(json.dumps(self.rows))
        revised.pop(2)
        revised[0]["qualification"]["accepted"] = False
        revised[0]["new_null"] = None
        revised = [revised[1], revised[0], *revised[2:]]
        second = self.root / "v2.jsonl.gz"
        write_rows(second, revised)
        info = self.store.import_jsonl_stream(second, parent_stream=self.path, kind="selected_rows")
        self.assertEqual(info["stored_frame_overrides"], 1)
        self.assertEqual(self.store.stats()["source_points"], 8)
        self.assert_json(list(self.store.iterate_records(second)), revised)
        self.assert_json(list(self.store.iterate_records(self.path)), self.rows)

    def test_direct_qualification_fork_and_projection_without_staged_full_tables(self):
        oldframes = self.root / "selected_frames.jsonl.gz"
        write_rows(oldframes, frames(self.rows))
        self.store.import_selected_frames(oldframes, self.path)
        removed = self.rows[1]["selected_row_id"]
        target = "direct/new_rows"
        info = self.store.fork_rows_stream(target, self.path, {("s05", 256): {"remove": [removed], "update": {self.rows[0]["selected_row_id"]: {"display_class": "ice_corrected"}}}})
        expected = json.loads(json.dumps(self.rows))
        expected.pop(1)
        expected[0]["display_class"] = "ice_corrected"
        self.assert_json(list(self.store.iterate_records(target)), expected)
        self.assertIsNone(info["original_sha256"])
        projected = self.store.project_selected_frames("direct/new_frames", target, oldframes)
        self.assertIsNone(projected["original_sha256"])
        self.assertEqual(projected["stored_frame_overrides"], 0)
        self.assert_json(list(self.store.iterate_records("direct/new_frames")), frames(expected))
        self.assertEqual(self.store.stats()["source_points"], 8)

    def test_fail_closed_rollback_source_uv_unknown_schema_and_invalid_forks(self):
        before = self.store.stats()
        bad = json.loads(json.dumps(self.rows))
        bad[0]["uv"][0] = 2.5
        p = self.root / "bad.jsonl.gz"
        write_rows(p, bad)
        with self.assertRaises(Error):
            self.store.import_jsonl_stream(p, parent_stream=self.path, kind="selected_rows")
        self.assertEqual(self.store.stats()["source_points"], before["source_points"])
        self.assertEqual(self.store.stats()["streams"], before["streams"])
        with self.assertRaises(KeyError):
            self.store.stream_info(p)
        for patch in ({"uv": [1, 2]}, {"display_class": None}):
            with self.assertRaises(Error):
                self.store.fork_rows_stream("badfork", self.path, {("s05", 256): {"update": {self.rows[0]["selected_row_id"]: patch}}})
        with self.assertRaises(Error):
            self.store.fork_rows_stream("no_exposure", self.path, {("s05", 256): {"remove": [r["selected_row_id"] for r in self.rows[:4]]}})
        unknown = self.root / "unknown.jsonl.gz"
        bad[0]["schema"] = "untrusted.new.schema"
        write_rows(unknown, bad)
        with self.assertRaises(Unknown):
            self.store.import_jsonl_stream(unknown, kind="selected_rows")
        mixed = self.root / "mixed.jsonl.gz"
        bad = json.loads(json.dumps(self.rows))
        bad[0]["source_row_id"] = "another"
        write_rows(mixed, bad)
        with self.assertRaises(Unknown):
            self.store.import_jsonl_stream(mixed, kind="selected_rows")

    def test_projection_rejects_wrong_uv_and_unknown_point_fields(self):
        p = self.root / "wrong_frames.jsonl.gz"
        values = frames(self.rows)
        values[0]["points"][0]["uv"] = [999, 0]
        write_rows(p, values)
        with self.assertRaises(Error):
            self.store.import_selected_frames(p, self.path)
        self.assertEqual(self.store.stats()["streams"], 1)
        values[0]["points"][0]["extra"] = 1
        write_rows(p, values)
        with self.assertRaises(Unknown):
            self.store.import_selected_frames(p, self.path)

    def test_concat_and_native_frame_metadata_factor_roundtrip(self):
        rows6 = [point(p, i, "s06") for p in (768, 1024) for i in range(2)]
        p = self.root / "six.jsonl.gz"
        write_rows(p, rows6)
        self.store.import_jsonl_stream(p, kind="selected_rows")
        info = self.store.register_concat("both", [self.path, p])
        self.assertEqual(info["stored_frame_overrides"], 0)
        self.assert_json(list(self.store.iterate_records("both")), self.rows + rows6)
        native = []
        for pts in (1280, 1536):
            source_rows = []
            for i in range(5):
                r = point(pts, i)
                r["source_row_id"] = r.pop("source_point_id")
                r.pop("selected_row_id")
                r["source_video_sha256"] = "e" * 64
                source_rows.append(r)
            frame = {"schema": "rinkfit.dense_native_board_source_frame.v1", "shot_id": "s05", "native_pts": pts,
                     "rows": source_rows, "source_video_sha256": "e" * 64, "source_original_UV_modified": False}
            f = self.root / (str(pts) + ".json")
            f.write_text(json.dumps(frame, ensure_ascii=False), encoding="utf8")
            native.append((f, frame))
        self.store.import_frame_jsons("native", [p for p, r in native])
        self.assert_json(list(self.store.iterate_records("native")), [r for p, r in native])
        self.assertEqual(len(self.store.native_file_refs("native")), 2)

    def test_unknown_stored_kind_rejected(self):
        self.store.db.execute("UPDATE streams SET kind='unexpected_schema' WHERE id=?", (self.info["id"],))
        self.store.db.commit()
        with self.assertRaises(Unknown):
            list(self.store.iterate_records(self.path))


if __name__ == "__main__":
    unittest.main()
