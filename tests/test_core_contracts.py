"""Admission and immutable-source regressions for the new domain boundary.

These small synthetic records test software semantics, not real observations
or visual accuracy. No historical runtime and no GPU dependency are loaded.
"""
from dataclasses import replace
import unittest

from rinkspace.domain.contracts import ContractError, Feature, Observation, Source, validate_dataset


class CoreContracts(unittest.TestCase):
    def setUp(self):
        self.source = Source("source", "a"*64, 1920, 1080, 1, 12800, frozenset({256256, 256512}))
        self.feature = Feature("ice.C", "finite_curve", "ice", "confirmed", "centreline", has_station_parameter=True)
        self.row = Observation("old-17", "source", "a"*64, 256256, (800.0, 600.0), "ice.C", "centreline",
                               "legacy_import", "measured", "human_confirmed", "old-source-row-17", (1.0, 0.1, 2.0))

    def validate(self, row):
        return validate_dataset({"source": self.source}, {"ice.C": self.feature}, (row,))

    def test_confirmed_legacy_and_verified_automatic_remain_usable(self):
        self.assertEqual(self.validate(self.row)["active"], 1)
        machine = replace(self.row, observation_id="auto-18", origin="automatic", quality="machine_verified")
        self.assertEqual(self.validate(machine)["active"], 1)

    def test_source_identity_and_native_pts_cannot_change(self):
        for row in (replace(self.row, pts=256257), replace(self.row, source_sha256="b"*64)):
            with self.assertRaises(ContractError):
                self.validate(row)

    def test_math_predictions_and_rejected_rows_cannot_enter_fitting(self):
        for row in (replace(self.row, source_evidence="mathematical_only"),
                    replace(self.row, origin="synthetic"), replace(self.row, quality="rejected")):
            with self.assertRaises(ContractError):
                self.validate(row)

    def test_entity_or_measurement_cannot_be_relabelled_silently(self):
        with self.assertRaises(ContractError):
            self.validate(replace(self.row, measurement="outer_boundary"))
        with self.assertRaises(ContractError):
            self.validate(replace(self.row, feature_id="some_other_line"))

    def test_mutable_coordinates_are_rejected(self):
        with self.assertRaises(ContractError):
            replace(self.row, uv=[800.0, 600.0])
        with self.assertRaises(ContractError):
            replace(self.source, time_base_den=12800.0)

    def test_fixed_junction_cannot_gain_a_sliding_station(self):
        with self.assertRaises(ContractError):
            Feature("junction", "fixed_junction", "ice", "confirmed", "centreline", ("a", "b"), "left", True)


if __name__ == "__main__":
    unittest.main()
