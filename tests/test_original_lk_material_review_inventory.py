"""Original measured motion and unknown material must not be conflated."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "events/ankara_2026/tools"))
from audit_original_low_motion_board_overlap import (
    ORIGINAL_LK_MODE, original_candidate_records,
    original_lk_review_inventory_reasons,
)
from audit_original_board_material_regions import observed_column_region, endpoint_region_status


class OriginalLKMaterialReviewInventoryTest(unittest.TestCase):
    def test_full_native_search_finds_visible_blue_cap_above_old_y255_floor(self):
        rgb = np.full((1080, 1920, 3), 240, np.uint8)
        rgb[232:239] = [255, 0, 0]
        rgb[395:404] = [0, 230, 230]
        original = rgb.copy()
        old = observed_column_region(rgb, [])
        current = observed_column_region(rgb, [], source_y_search_range=[0, 1080])
        uv = np.array([[1400., 290.], [1400., 231.], [1400., 240.]])
        old_reasons = endpoint_region_status(uv, old, True)
        current_reasons = endpoint_region_status(uv, current, True)
        self.assertEqual(int(old_reasons[0]), 129)
        self.assertEqual(int(current_reasons[0]), 0)
        # Background and the mixed cap border do not become vertical board.
        self.assertEqual(int(current_reasons[1]) & 128, 128)
        self.assertEqual(int(current_reasons[2]) & 128, 128)
        self.assertTrue(np.array_equal(rgb, original))

    def test_colour_region_does_not_fill_missing_columns_or_missing_people_review(self):
        rgb = np.full((1080, 1920, 3), 240, np.uint8)
        rgb[232:239] = [255, 0, 0]
        rgb[395:404] = [0, 230, 230]
        rgb[232:239, 1400] = 240
        bounds = observed_column_region(rgb, [], source_y_search_range=[0, 1080])
        self.assertNotEqual(int(endpoint_region_status([[1400., 290.]], bounds, True)[0]), 0)
        self.assertEqual(int(endpoint_region_status([[1500., 290.]], bounds, False)[0]), 256)

    def test_colour_region_respects_people_and_UI_even_inside_two_colour_strips(self):
        rgb = np.full((1080, 1920, 3), 240, np.uint8)
        rgb[232:239] = [255, 0, 0]
        rgb[395:404] = [0, 230, 230]
        bounds = observed_column_region(rgb, [[640, 130, 720, 190]], source_y_search_range=[0, 1080])
        reason = endpoint_region_status([[1400., 290.], [100., 240.]], bounds, True)
        self.assertEqual(int(reason[0]) & 64, 64)
        self.assertEqual(int(reason[1]) & 64, 64)

    def test_existing_measurement_does_not_require_a_second_corner_or_colour_cap(self):
        source = np.array([[400., 380.], [700., 390.]], np.float32)
        target = source + np.array([4., 0.], np.float32)
        original_source, original_target = source.copy(), target.copy()
        status = original_lk_review_inventory_reasons(source, target,
            source_people_clear=np.ones(2, bool), target_people_clear=np.ones(2, bool),
            source_UI_clear=np.ones(2, bool), target_UI_clear=np.ones(2, bool))
        self.assertTrue(np.array_equal(status, np.zeros((2, 2), np.uint16)))
        self.assertTrue(np.array_equal(source, original_source))
        self.assertTrue(np.array_equal(target, original_target))
        # There is intentionally no independent-detector or blue-strip input.
        # A zero review-navigation reason is NOT a material qualification.

    def test_people_UI_and_invalid_pixels_remain_explicit_unknown(self):
        source = np.array([[30., 40.], [600., 500.], [-1., 550.]])
        target = np.array([[34., 40.], [605., 500.], [3., np.nan]])
        status = original_lk_review_inventory_reasons(source, target,
            source_people_clear=np.array([True, False, True]),
            target_people_clear=np.array([True, False, True]),
            source_UI_clear=np.array([False, True, True]),
            target_UI_clear=np.array([False, True, True]))
        self.assertTrue(np.array_equal(status, [[1024, 1024], [64, 64], [512, 512]]))

    def test_unaligned_or_non_boolean_navigation_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Boolean"):
            original_lk_review_inventory_reasons(np.ones((1, 2)), np.ones((1, 2)),
                source_people_clear=[1], target_people_clear=[True],
                source_UI_clear=[True], target_UI_clear=[True])

    def test_compact_review_keeps_old_navigation_failures_without_material_admission(self):
        arrays = dict(current_original_prefix_bundle_index=np.array([269954, 269964]),
            source_uv=np.array([[123., 456.], [124., 457.]], np.float32),
            target_uv=np.array([[127., 456.], [128., 457.]], np.float32),
            source_sigma_px=np.full((2, 2), 2., np.float64),
            source_navigation_reason_bits=np.zeros((2, 2), np.uint16),
            existing_source_region_reason_bits=np.array([[129, 129], [131, 131]], np.uint16))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inventory.npz"
            np.savez_compressed(path, **arrays)
            packet = dict(candidate_mode=ORIGINAL_LK_MODE, shot_index=5, records=[],
                summary=dict(source_candidate_count=2), candidate_numeric_archive=dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                candidate_numeric_archive_keys=list(arrays))
            rows = list(original_candidate_records(packet))
            self.assertEqual(rows[0]["existing_source_region_reason_bits"], [129, 129])
            self.assertEqual(rows[1]["existing_source_region_reason_bits"], [131, 131])
            self.assertTrue(np.array_equal([r["source_uv"] for r in rows], arrays["source_uv"]))
            self.assertTrue(np.array_equal([r["target_uv"] for r in rows], arrays["target_uv"]))
            self.assertTrue(np.array_equal([r["source_sigma_px"] for r in rows], arrays["source_sigma_px"]))
            self.assertTrue(all(not r["both_end_material_reviewed"] for r in rows))
            self.assertTrue(all(not r["automatically_material_qualified"] for r in rows))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                list(original_candidate_records(packet))


if __name__ == "__main__":
    unittest.main()
