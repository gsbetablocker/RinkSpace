"""CPU-only contract tests for explicit camera ablations, not fit validation."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


# Load this exact delivered module in an overlay or a normal repository checkout.
# No installed package, native DLL, GPU, or source/state file is required.
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "camera_experiment_under_test", ROOT / "src/rinkspace/domain/camera_experiment.py")
ce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ce)
EXAMPLES = ROOT / "events/ankara_2026/programmes/okada_mei_sp/camera_experiments"


def rig(size=43):
    value = np.linspace(0.01, 0.43, size)
    value[:9] = np.eye(3).ravel()
    value[9:15] = [0, 0, 1, 1, 0, 0]
    value[24:29] = [1000., 3., 317.25, 197.75, 0.13]
    value[37] = 0.0317
    return value


def profile():
    return {
        "schema": ce.SCHEMA, "id": "test-explicit",
        "basis": {"kind": "engineering_assumption", "not_sony_measurement": True,
                  "description": "Synthetic ablation only; all fixed values are assumptions."},
        "active_camera_names": ["readout", "pivot_z", "cx"],
        "fixed_rig_values": {"eta": 0., "pupil2": 0.},
        "scene_mode": "frozen",
    }


class CameraExperimentTests(unittest.TestCase):
    def test_copy_no_mutation_and_native_scalar_mapping(self):
        seed, config = rig(), profile()
        config["fixed_rig_values"]["wide_p2"] = -0.007
        original, original_config = seed.copy(), copy.deepcopy(config)
        result, indices, metadata = ce.apply_camera_experiment(seed, config)
        expected = original.copy()
        expected[23], expected[28], expected[42] = 0., 0., -0.007
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(seed, original)
        self.assertEqual(config, original_config)
        self.assertEqual(indices, (8, 15, 26))
        self.assertEqual(metadata["active_camera_names"], ["pivot_z", "cx", "readout"])
        self.assertFalse(np.shares_memory(seed, result))
        result[0] = -2
        metadata["profile"]["basis"]["description"] = "changed result only"
        self.assertEqual(config, original_config)
        np.testing.assert_array_equal(seed, original)
        json.dumps(metadata, allow_nan=False)

    def test_conflicts_unknowns_and_undeclared_assumptions_fail(self):
        changes = [
            {"active_camera_names": ["cx", "cx"]},
            {"active_camera_names": ["zoom"]},
            {"fixed_rig_values": {"cx": 960.}},
            {"fixed_rig_values": {"R0_x": 0.}},
            {"fixed_rig_values": {"pan": [0, 0, 1]}},
            {"fixed_rig_values": {"fref": 1000.}},
            {"fixed_rig_values": {"wide_boundary": 6.}},
            {"fixed_rig_values": {"pupil1": True}},
            {"scene_mode": "infer"},
            {"basis": {"kind": "sony_measurement", "not_sony_measurement": False,
                       "description": "This interface does not accept that claim."}},
            {"default_image_size": [1920, 1080]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ce.apply_camera_experiment(rig(), {**profile(), **change})
        missing = profile()
        del missing["scene_mode"]
        with self.assertRaises(ValueError):
            ce.apply_camera_experiment(rig(), missing)

    def test_json_duplicate_keys_nonfinite_and_numeric_overflow_fail(self):
        text = json.dumps(profile())
        bad_json = [
            text.replace('"eta": 0.0', '"eta": 0.0, "eta": 1.0'),
            text.replace('"eta": 0.0', '"eta": NaN'),
            text.replace('"eta": 0.0', '"eta": Infinity'),
            text.replace('"eta": 0.0', '"eta": 1e400'),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            for invalid in bad_json:
                path.write_text(invalid, encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    ce.load_camera_experiment(path, rig())
        for value in (float("nan"), float("inf"), -float("inf")):
            config = profile()
            config["fixed_rig_values"]["eta"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                ce.apply_camera_experiment(rig(), config)

    def test_storage_and_chart_compatibility_are_not_interchanged(self):
        for storage, chart in ((38, 27), (43, 31)):
            with self.subTest(storage=storage):
                result, _, metadata = ce.apply_camera_experiment(rig(storage), profile())
                self.assertEqual(result.shape, (storage,))
                self.assertEqual(metadata["camera_chart_size"], chart)
        for change in ({"active_camera_names": ["wide_k1"]},
                       {"fixed_rig_values": {"wide_p2": 0.}}):
            with self.assertRaises(ValueError):
                ce.apply_camera_experiment(rig(38), {**profile(), **change})
        for seed in (np.zeros(31), np.zeros((1, 43)), np.full(43, np.nan)):
            with self.assertRaises(ValueError):
                ce.apply_camera_experiment(seed, profile())

    def test_compact_columns_match_full_selection_without_null_camera_columns(self):
        seed = rig()
        chosen = (6, 15, 27, 30)
        scales = np.linspace(.001, .031, 31)
        rd, columns = ce.build_camera_directions(
            seed, chosen, scales, motion_columns=2, scene_columns=3)
        self.assertEqual(rd.shape, (31, 9))  # 2 motion + 4 camera + 3 scene
        np.testing.assert_array_equal(columns, [2, 3, 4, 5])
        self.assertEqual(np.linalg.matrix_rank(rd[:, columns]), len(chosen))
        self.assertEqual(np.count_nonzero(rd), len(chosen))
        old = np.zeros((31, 2 + 31 + 3))
        old[:, 2:33] = np.diag(scales)
        keep = ce.retained_column_indices(
            seed, chosen, motion_columns=2, scene_columns=3)
        np.testing.assert_array_equal(keep, [0, 1, 8, 17, 29, 32, 33, 34, 35])
        np.testing.assert_array_equal(old[:, keep], rd)

    def test_empty_camera_selection_and_invalid_directions(self):
        rd, columns = ce.build_camera_directions(
            rig(38), (), np.ones(27), motion_columns=2, scene_columns=0)
        self.assertEqual(rd.shape, (27, 2))
        self.assertEqual(columns.size, 0)
        cases = [((0, 0), np.ones(27), 1), ((27,), np.ones(27), 1),
                 ((True,), np.ones(27), 1), ((1,), np.ones(38), 1),
                 ((1,), np.zeros(27), 1), ((1,), np.full(27, np.nan), 1),
                 ((1,), np.ones(27), -1), ((), np.ones(27), 0)]
        for indices, scales, motion in cases:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                ce.build_camera_directions(rig(38), indices, scales,
                                           motion_columns=motion, scene_columns=0)

    def test_examples_preserve_frame_and_mechanical_initial_values(self):
        seed = rig()
        results = {}
        for path in sorted(EXAMPLES.glob("*.json")):
            result, indices, metadata = ce.load_camera_experiment(path, seed)
            results[path.stem] = result
            self.assertEqual(metadata["scene_mode"], "frozen")
            self.assertFalse(metadata["physical_camera_validated"])
            self.assertTrue(set(range(6, 15)).issubset(indices))
            np.testing.assert_array_equal(result[15:26], seed[15:26])
        self.assertEqual(len(results), 4)
        current = results["square_current_pp_k1_zoom"]
        free = results["square_free_pp_k1_zoom"]
        zero = results["square_free_pp_k1_zoom_zero_readout"]
        center = results["square_raster_center"]
        # No automatic raster centre, readout-from-fps, or invented physical height.
        np.testing.assert_array_equal(current[26:28], [317.25, 197.75])
        np.testing.assert_array_equal(current, free)
        self.assertEqual(free[37], .0317)
        different = np.flatnonzero(zero != free)
        np.testing.assert_array_equal(different, [37])
        self.assertEqual(zero[37], 0.)
        np.testing.assert_array_equal(center[26:28], [959.5, 539.5])


class JointExperimentProtectionTests(unittest.TestCase):
    def test_joint_baseline_directory_and_cross_profile_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'model.json';config.write_text(json.dumps(profile()),encoding='utf-8')
            (root/'activity.json').write_text(json.dumps({'phase':'joint'}),encoding='utf-8')
            with self.assertRaises(ValueError):ce.guard_experiment_run_directory(root,config)
            _,_,metadata=ce.apply_camera_experiment(rig(),profile())
            (root/'activity.json').write_text(json.dumps({'camera_experiment':metadata}),encoding='utf-8')
            ce.guard_experiment_run_directory(root,config)
            other=profile();other['id']='different';config.write_text(json.dumps(other),encoding='utf-8')
            with self.assertRaises(ValueError):ce.guard_experiment_run_directory(root,config)

    def test_tagged_snapshot_cannot_silently_reopen_another_model(self):
        _,_,metadata=ce.apply_camera_experiment(rig(),profile())
        self.assertFalse(ce.validate_experiment_resume(None,metadata))
        self.assertFalse(ce.validate_experiment_resume(metadata['profile_sha256'],metadata))
        with self.assertRaises(ValueError):ce.validate_experiment_resume('saved',None)
        with self.assertRaises(ValueError):ce.validate_experiment_resume('different',metadata)
        self.assertTrue(ce.validate_experiment_resume('different',metadata,allow_change=True))


if __name__ == "__main__":
    unittest.main()
