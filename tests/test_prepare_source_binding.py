"""Native source/shot binding checks, using synthetic data rather than footage."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rinkspace.session.dataset import DatasetError, load_array, read_json, write_array, write_json
from rinkspace.session.prepare import prepare


class PrepareSourceBinding(unittest.TestCase):
    def make_input(self, root, rows, *, shot_source="official"):
        root.mkdir()
        pts_ref = write_array(root, "source_pts", [50, 100, 150, 200, 250], "<i8")
        sources = [dict(id=sid, sha256=sha * 64, width=1920, height=1080,
                        time_base_num=1, time_base_den=12800, pts_ref=pts_ref)
                   for sid, sha in (("official", "a"), ("other", "b"))]
        shots = []
        for sid, first in (("s01", 100), ("s02", 200)):
            processing = dict(first_pts=first, last_pts=first + 50,
                              stop_pts_exclusive=first + 100, native_frame_count=2)
            shots.append(dict(id=sid, source_id=shot_source if sid == "s01" else "official",
                              processing=processing, camera_domain=dict(
                                  status="defined", first_pts=first, stop_pts_exclusive=first + 100)))
        scope = dict(required_shot_ids=["s01", "s02"], shots=shots)
        n = len(rows)
        arrays = dict(
            uv=np.tile([10., 20.], (n, 1)).astype("<f8"),
            pts=np.array([r.get("pts", 100) for r in rows], dtype="<i8"),
            source_index=np.array([r.get("source_index", 0) for r in rows], dtype="<i4"),
            shot_index=np.array([r.get("shot_index", 0) for r in rows], dtype="<i4"),
            feature_index=np.zeros(n, dtype="<i4"), kind=np.ones(n, dtype="u1"),
            active=np.array([r.get("active", 1) for r in rows], dtype="u1"),
            split=np.zeros(n, dtype="u1"), noise_group_index=np.full(n, -1, dtype="<i4"),
            covariance=np.tile([1., 0., 1.], (n, 1)).astype("<f8"),
            support=np.tile([0., 1.], (n, 1)).astype("<f8"),
            local_index=np.arange(n, dtype="<i8"))
        write_json(root / "obs.json", dict(arrays={k: write_array(root, k, v) for k, v in arrays.items()}))
        write_json(root / "sources.json", dict(sources=sources))
        write_json(root / "shots.json", dict(shots=shots))
        write_json(root / "features.json", dict(features=[dict(
            id="ice.C", label="ice.C", kind="finite_curve", measurement="centreline",
            identity="confirmed", surface="ice")]))
        write_json(root / "groups.json", dict(groups=[],
            members_ref=write_array(root, "members", [], "<i8"),
            weights_ref=write_array(root, "weights", [], "<f8")))
        write_json(root / "scene.json", dict(paths=[]))
        write_json(root / "rigs.json", dict(rigs=[]))
        write_json(root / "provenance.json", {})
        write_json(root / "scope.json", scope)
        write_json(root / "input.json", dict(schema="rinkspace.input.v1", dataset_id="synthetic-binding-test",
            revision="r0", scene_ref="scene.json", rigs_ref="rigs.json", shots_ref="shots.json",
            sources_ref="sources.json", observations_ref="obs.json", features_ref="features.json",
            noise_groups_ref="groups.json", provenance_ref="provenance.json"))
        return arrays

    def test_legal_first_and_last_exposures_preserve_source_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.make_input(root / "input", [dict(pts=100), dict(pts=150)])
            receipt = prepare(root / "input/input.json", root / "input/scope.json", "s01", root / "prepared")
            manifest = read_json(receipt.manifest_path)
            for key, value in expected.items():
                np.testing.assert_array_equal(load_array(root / "prepared", manifest["arrays"][key]), value)
            self.assertEqual(manifest["coverage"][0]["frames"], 2)

    def test_different_source_with_same_native_pts_is_rejected(self):
        for active in (0, 1):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_input(root / "input", [dict(source_index=1, active=active)])
                with self.assertRaisesRegex(DatasetError, "source differs from its shot") as raised:
                    prepare(root / "input/input.json", root / "input/scope.json", "s01", root / "prepared")
                self.assertEqual(raised.exception.context["shot_id"], "s01")
                self.assertEqual(raised.exception.context["source_row"], 0)
                self.assertFalse((root / "prepared").exists())

    def test_valid_source_pts_outside_shot_are_rejected(self):
        for active in (0, 1):
            for pts in (50, 200):
                with self.subTest(active=active, pts=pts), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.make_input(root / "input", [dict(pts=pts, active=active)])
                    with self.assertRaisesRegex(DatasetError, "outside its shot processing domain") as raised:
                        prepare(root / "input/input.json", root / "input/scope.json", "s01", root / "prepared")
                    self.assertEqual(raised.exception.context["pts"], pts)
                    self.assertFalse((root / "prepared").exists())

    def test_unselected_shot_bindings_do_not_block_selected_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_input(root / "input", [dict(pts=100), dict(shot_index=1, source_index=1, pts=50)])
            receipt = prepare(root / "input/input.json", root / "input/scope.json", "s01", root / "prepared")
            manifest = read_json(receipt.manifest_path)
            self.assertEqual(manifest["observations"], 1)
            self.assertEqual(manifest["selected_shot_ids"], ["s01"])

    def test_unknown_selected_shot_source_reports_contract_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_input(root / "input", [dict()], shot_source="absent-source")
            with self.assertRaisesRegex(DatasetError, "references an absent source"):
                prepare(root / "input/input.json", root / "input/scope.json", "s01", root / "prepared")
            self.assertFalse((root / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
