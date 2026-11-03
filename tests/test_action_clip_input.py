import unittest
import numpy as np
from calibration.program_tracking.body_tracking.action_clip_input import (
    camera25_from_snapshot, clip_pts, map_observations, roi_proposals)


class ActionClipInputTests(unittest.TestCase):
    def test_official_slots_retain_native_tail_and_constant_model_spacing(self):
        native, rows = clip_pts((44.98, 52.))
        self.assertEqual((len(native), len(rows)), (352, 176))
        self.assertEqual(int(native[-1]), 665600)
        np.testing.assert_array_equal(np.diff(native[rows]), 512)
        self.assertEqual(int(native[rows[-1]]), 665344)
        with self.assertRaises(ValueError):
            clip_pts((44.981, 52.))

    def test_roi_proposal_never_fills_measured_pixels_or_bbox(self):
        native = dict(pts=np.array([0, 256, 512, 768, 1024]),
                      bbox=np.array([[0, 0, 10, 20], [1, 1, 11, 21],
                                     [np.nan] * 4, [3, 3, 13, 23], [4, 4, 14, 24]]),
                      source_mapping_missing=np.array([False, False, True, False, False]))
        original = native['bbox'].copy()
        with self.assertRaisesRegex(ValueError, '512'):
            roi_proposals(native, np.array([0, 2, 4]))
        proposed = roi_proposals(native, np.array([0, 2, 4]), missing_policy='roi_proposal')
        np.testing.assert_array_equal(native['bbox'], original)
        np.testing.assert_array_equal(proposed['roi_source_official_pts'][1], [256, 768])
        np.testing.assert_allclose(proposed['roi_native_xyxy'][1], [2, 2, 12, 22])
        self.assertTrue(proposed['bbox_inferred'][1])
        self.assertEqual(proposed['roi_interpolation_alpha'][1], .5)
        with self.assertRaisesRegex(ValueError, 'both sides'):
            roi_proposals(dict(pts=np.array([0, 256]), bbox=np.array([[np.nan]*4, [1, 1, 2, 2]]),
                               source_mapping_missing=np.array([True, False])),
                          np.array([0]), missing_policy='roi_proposal')

    def test_mapping_keeps_missing_unknown_and_prefers_real_valid_row(self):
        uv = np.full((2, 17, 2), 100., np.float32)
        uv[1, :, 0] = 200.
        observations = dict(pts=np.array([100, 200]), shot_index=np.array([2, 2]),
                            coco17=np.concatenate((uv, np.ones((2, 17, 1))), axis=2),
                            native_coco17_xy=uv, keypoint_scores=np.ones((2, 17)),
                            bbox_native_xyxy=np.array([[50, 50, 150, 300], [150, 50, 250, 300]]),
                            valid=np.array([False, True]))
        ledger = [dict(full_pts=old, accepted_official_pts=512, shot_index=2,
                       best_roi_gray_mae=mae, match_status='unique_pixel_match')
                  for old, mae in [(100, .1), (200, 1.)]]
        mapped = map_observations(observations, ledger, np.array([512, 768]), 's02')
        np.testing.assert_array_equal(mapped['body_uv'][0], uv[1])
        self.assertTrue(mapped['body_use'][0].all())
        self.assertTrue(np.isnan(mapped['body_uv'][1]).all())
        self.assertFalse(mapped['body_use'][1].any())
        self.assertEqual(mapped['source_legacy_pts'][1], -1)
        self.assertTrue(mapped['source_mapping_missing'][1])

    def test_camera25_layout_preserves_effective_native_brown_coefficients(self):
        import cv2
        from calibration.program_tracking.body_tracking.camera_bridge import project_world
        rotation = cv2.Rodrigues(np.array([.1, -.2, .3]))[0]
        pose = dict(status=np.array([0]), centre=np.array([[1., 2., 3.]]),
                    rotation=rotation[None], fx=np.array([1200.]), fy=np.array([1100.]),
                    cx=np.array([960.]), cy=np.array([540.]), k1=np.array([.02]),
                    k2=np.array([-.003]), p1=np.array([.001]), p2=np.array([-.002]))
        camera = camera25_from_snapshot(pose)[0]
        local = np.array([[.2, -.1, 1.], [-.1, .3, 2.]])
        world = local @ rotation + pose['centre'][0]
        actual, valid = project_world(world, camera)
        intrinsic = np.array([[1200., 0., 960.], [0., 1100., 540.], [0., 0., 1.]])
        expected = cv2.projectPoints(local, np.zeros(3), np.zeros(3), intrinsic,
                                     np.array([.02, -.003, .001, -.002, 0.]))[0][:, 0]
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(camera[15:], 0.)


if __name__ == '__main__':
    unittest.main()
