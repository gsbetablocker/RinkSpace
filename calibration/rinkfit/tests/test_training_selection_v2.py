"""Source-domain failures and per-exposure budget preservation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from rinkfit_joint import training_source_selection_v2 as selection


class TrainingSelectionTests(unittest.TestCase):
    def test_roles_are_original_shot_and_native_exposure_authority(self):
        roles = {'100': {'shot_id': 's01', 'role': 'development_candidate'},
                 '200': {'shot_id': 's01', 'role': 'validation_guard'}}
        actual = selection.exposure_permission(np.array([0, 0]), np.array([100, 200]), ['s01'], roles)
        np.testing.assert_array_equal(actual, [True, False])
        for shots, pts in ((['s02'], [100]), (['s01'], [300])):
            with self.assertRaisesRegex(ValueError, 'Unknown or cross-shot'):
                selection.exposure_permission(np.array([0]), np.array(pts), shots, roles)

    def test_current_verdict_cannot_relabel_changed_pixels_or_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / 'packet.json'
            packet.write_text('{}', 'utf8')
            arrays = dict(uv=np.array([[3., 4.]]), native_pts=np.array([100]), shot_index=np.array([0]),
                          material_index=np.array([2]), path=np.array([0]))
            class Reader:
                def __init__(self):
                    self.path = packet
                    self.packet = {'row_count': 1}
                def load(self, key): return arrays[key]
            row = dict(raw_row=0, source_row_id='original', uv=[3., 4.], native_pts=100,
                       shot_index=0, material_index=2, path=0, uses_fitted_model_projection=False,
                       decision='rejected')
            verdict = dict(schema='rinkfit.source_identity_verdicts.v1',
                           source_packet_sha256=selection.file_sha256(packet), rows=[row])
            with patch.object(selection, '_source_identities', return_value=({0: 'original'}, None)):
                self.assertEqual(selection.source_verdict_rows(Reader(), verdict), [row])
                original_path = row.pop('path')
                with self.assertRaisesRegex(ValueError, 'Incomplete source identity verdict tuple'):
                    selection.source_verdict_rows(Reader(), verdict)
                row['path'] = original_path
                row['uv'] = [8., 9.]
                with self.assertRaisesRegex(ValueError, 'Source verdict changed uv'):
                    selection.source_verdict_rows(Reader(), verdict)
                row['uv'] = [3., 4.]
                verdict['source_packet_sha256'] = 'old packet'
                with self.assertRaisesRegex(ValueError, 'another source packet'):
                    selection.source_verdict_rows(Reader(), verdict)

    def test_removed_exposure_does_not_inflate_surviving_exposure_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / 'packet.json'; packet.write_text('{}', 'utf8')
            proof = root / 'proof.json'; proof.write_text('{}', 'utf8')
            rules = root / 'canonical.json'
            rules.write_text(json.dumps(dict(groups=[], source_review=selection.reference(proof))), 'utf8')
            arrays = dict(shot_index=np.array([0, 0, 0]), native_pts=np.array([100, 100, 200]),
                          material_index=np.array([0, 0, 0]), kind=np.array([0, 0, 0]))
            class Reader:
                path = packet
                def load(self, key): return arrays[key]
            model = dict(materials=[dict(label='ice.C')])
            config = dict(engineering_support_budget_v2=dict(source_packet_sha256=selection.file_sha256(packet),
                          scheme='equal_native_exposure', calibrated_covariance=False, total_budget=10.,
                          canonical_material_entities=selection.reference(rules)))
            policy, weights, motion, report = selection.recalculate_budget(
                Reader(), model, config, np.array([0]), np.array([True, True, True]), {}, {})
            self.assertEqual(report['original_per_exposure_budget'], 5.)
            self.assertEqual(report['new_per_exposure_budget'], 5.)
            self.assertEqual(policy['total_budget'], 5.)
            np.testing.assert_array_equal(weights, [5.])
            self.assertEqual(motion.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
