"""One-time state import must preserve exact scalar-array identity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rinkspace.compat.d78_state import import_d78_state


class D78LocalStateIdentity(unittest.TestCase):
    def make_input(self, root, raw, *, name="station.f64", declared_sha=None, relocated=False):
        def write(path, value):
            path.write_text(json.dumps(value), encoding="utf-8")
        model = root / "model.json"
        write(model, {"blocks": [], "camera": {}})
        scope = root / "scope.json"
        write(scope, dict(required_shot_ids=["s01"],
            source=dict(id="synthetic", width=1920, height=1080),
            source_clock=dict(time_base_num=1, time_base_den=12800),
            shots=[dict(id="s01", rig_id="main", processing=dict(first_pts=100, stop_pts_exclusive=200))]))
        local = root / name
        local.write_bytes(raw)
        expected = declared_sha if declared_sha is not None else hashlib.sha256(raw).hexdigest()
        original_path = root / "absent-original" / name if relocated else local
        complete = root / "complete.json"
        write(complete, {"row_state_files": {name: {"path": str(original_path), "sha256": expected}}})
        asset_map = None
        if relocated:
            asset_map = root / "asset_map.json"
            write(asset_map, {"states": [{"model": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
                "assets": {name: {"file": str(local), "sha256": expected}}}]})
        return dict(model_path=model, out_dir=root / "out", scope_path=scope,
                    library_path="not-used-by-identity-test", complete_state_path=complete,
                    asset_map_path=asset_map, allow_unverified_replay=True)

    def import_state(self, arguments):
        # Optical and scene math have their own real-native tests. Isolate this
        # regression to byte-preserving migration; no source image is claimed.
        with patch("rinkspace.compat.d78_state.NativeCore"), \
             patch("rinkspace.compat.d78_state.decode_camera", return_value=({}, np.zeros((4, 3)))), \
             patch("rinkspace.compat.d78_state.decode_entities", return_value={"revision": "r0"}):
            return import_d78_state(**arguments)

    def test_existing_original_path_with_wrong_hash_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arguments = self.make_input(root, np.array([.125], dtype="<f8").tobytes(), declared_sha="0" * 64)
            with self.assertRaisesRegex(ValueError, "differs from its declared identity"):
                self.import_state(arguments)
            self.assertFalse((root / "out").exists())

    def test_original_and_relocated_bytes_are_preserved(self):
        for relocated in (False, True):
            with self.subTest(relocated=relocated), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                values = np.array([.125, .75], dtype="<f8")
                arguments = self.make_input(root, values.tobytes(), relocated=relocated)
                result = self.import_state(arguments)
                np.testing.assert_array_equal(np.load(result.parent / "local_values.npy"), values)
                state = json.loads(result.read_text())
                self.assertTrue(state["origin"]["complete_local_state_imported"])
                layout = json.loads((result.parent / "local_layout.json").read_text())
                self.assertEqual(layout["station.f64"]["source_ref"]["sha256"],
                                 hashlib.sha256(values.tobytes()).hexdigest())

    def test_relocated_wrong_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arguments = self.make_input(root, np.array([.125], dtype="<f8").tobytes(),
                                        declared_sha="0" * 64, relocated=True)
            with self.assertRaisesRegex(ValueError, "differs from its declared identity"):
                self.import_state(arguments)
            self.assertFalse((root / "out").exists())

    def test_matching_hash_does_not_authorize_partial_scalar(self):
        for name, dtype in (("station.f64", "<f8"), ("path.u32", "<u4")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = np.array([1], dtype=dtype).tobytes() + b"x"
                arguments = self.make_input(root, raw, name=name)
                with self.assertRaisesRegex(ValueError, "byte count is not aligned"):
                    self.import_state(arguments)
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
