"""Finite ice objectives exercised with the actual contact kernel, without assets."""
from dataclasses import FrozenInstanceError, fields, replace
import unittest

import numpy as np

from calibration.program_tracking.body_tracking.ice_soft import IceSoftPolicy, ice_soft_rows
from calibration.program_tracking.body_tracking.skate_physics import ALLOWANCE, constraints


# Deliberately explicit toy choices for regression arithmetic, not fit defaults.
POLICY = IceSoftPolicy(penetration_deadzone_m=.004, penetration_scale_m=.03, penetration_weight=.5,
                       support_hover_deadzone_m=.006, support_scale_m=.04, support_weight=.75)


class PointBinding:
    ratio = 1.

    def packet(self):
        return {'template': {'contact_arcs': [], 'contact_point_groups': [
            {'id': 'runner', 'points': np.zeros((1, 3)), 'semantic': 'runner', 'support_eligible': True}],
            'support_feature_ids': {'glide': ['runner'], 'pick': ['runner']}}}


def contact(heights, times=(.02,), evidence=()):
    times = np.asarray(times)
    R = np.tile(np.eye(3), (len(times), 2, 1, 1))
    O = np.zeros((len(times), 2, 3))
    O[:, :, 2] = heights
    return constraints(PointBinding(), {'R': R, 'O': O}, times, 0., evidence)


def evidence(**extra):
    return dict(id='reviewed_support', pts=256, feet=[0], mode='glide', **extra)


def support_positions(result):
    return np.flatnonzero(np.array(result['diagnostics']['kinds']) == 'ice_support_hover')


class IceSoftTests(unittest.TestCase):
    def test_deadzones_have_zero_residual_and_zero_derivative_including_hinge(self):
        packet = contact([-.005, .006], evidence=[dict(evidence(), feet=[1])])
        result = ice_soft_rows(packet, POLICY)
        np.testing.assert_array_equal(result['residual'], 0.)
        np.testing.assert_array_equal(result['derivative_per_margin_m'], 0.)
        np.testing.assert_array_equal(result['indices'], [0, 1, 3])
        np.testing.assert_allclose(result['diagnostics']['raw_height_m'], [-.005, .006, .006])
        np.testing.assert_allclose(result['diagnostics']['margin_m'][:2], [-.003, .008])
        self.assertEqual(result['diagnostics']['skipped_rows'][0]['reason'], 'support_lower_covered_by_nonpenetration')
        at_penetration_hinge = ice_soft_rows(contact([-ALLOWANCE['runner']-POLICY.penetration_deadzone_m]*2), POLICY)
        np.testing.assert_allclose(at_penetration_hinge['residual'], 0., atol=1e-16)
        np.testing.assert_array_equal(at_penetration_hinge['derivative_per_margin_m'], 0.)

    def test_small_penetration_has_finite_light_pressure_and_deeper_cost_grows(self):
        costs = []
        for depth in (0., .001, .01, .1):
            height = -ALLOWANCE['runner']-POLICY.penetration_deadzone_m-depth
            result = ice_soft_rows(contact([height, height]), POLICY)
            expected = np.sqrt(.02*POLICY.penetration_weight)*depth/POLICY.penetration_scale_m
            np.testing.assert_allclose(result['residual'], [expected, expected], atol=1e-16)
            costs.append(result['parts']['ice_penetration'])
            self.assertEqual(result['parts']['ice_support_hover'], 0.)
        self.assertTrue(np.isfinite(costs).all())
        self.assertTrue(np.all(np.diff(costs) > 0))
        self.assertLess(costs[1], costs[3]/1000)
        np.testing.assert_allclose(costs[2]/costs[1], 100.)

    def test_support_uses_raw_height_with_its_own_deadzone_and_never_lower_band(self):
        costs = []
        for height in (.006, .007, .02, .2):
            result = ice_soft_rows(contact([height, .05], evidence=[evidence()]), POLICY)
            chosen = support_positions(result)
            self.assertEqual(len(chosen), 1)
            expected = np.sqrt(.02*POLICY.support_weight)*max(height-.006, 0)/.04
            np.testing.assert_allclose(result['residual'][chosen], [expected], atol=1e-16)
            self.assertEqual(result['parts']['ice_penetration'], 0.)
            costs.append(result['parts']['ice_support_hover'])
        self.assertTrue(np.all(np.diff(costs) > 0))

    def test_derivatives_are_per_margin_with_correct_penetration_and_hover_signs(self):
        step = 1e-7
        pen = ice_soft_rows(contact([-.04, -.04]), POLICY)
        pen_plus = ice_soft_rows(contact([-.04+step, -.04+step]), POLICY)
        pen_minus = ice_soft_rows(contact([-.04-step, -.04-step]), POLICY)
        numeric = (pen_plus['residual']-pen_minus['residual'])/(2*step)
        self.assertTrue(np.all(pen['derivative_per_margin_m'] < 0))
        np.testing.assert_allclose(numeric, pen['derivative_per_margin_m'], rtol=1e-9)
        hover = ice_soft_rows(contact([.05, .05], evidence=[evidence()]), POLICY)
        hover_plus = ice_soft_rows(contact([.05+step, .05], evidence=[evidence()]), POLICY)
        hover_minus = ice_soft_rows(contact([.05-step, .05], evidence=[evidence()]), POLICY)
        chosen = support_positions(hover)
        numeric = (hover_plus['residual'][chosen]-hover_minus['residual'][chosen])/(2*step)
        self.assertTrue(np.all(hover['derivative_per_margin_m'][chosen] < 0))
        # support upper margin = bound - h, hence dm/dh=-1.
        np.testing.assert_allclose(numeric, -hover['derivative_per_margin_m'][chosen], rtol=1e-9)

    def test_source_measure_and_support_derivative_do_not_depend_on_collision_grid(self):
        outputs = []
        for times in ([0., .02, .04], np.arange(17)*.0025):
            packet = contact([.05, .06], times=times,
                             evidence=[evidence(source_support_weight_s=.04, confidence=.01)])
            result = ice_soft_rows(packet, POLICY)
            chosen = support_positions(result)
            np.testing.assert_allclose(packet['weights'][result['indices'][chosen]], [.04])
            outputs.append([result['residual'][chosen], result['derivative_per_margin_m'][chosen]])
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-14)

    def test_constant_penetration_integral_remains_grid_invariant(self):
        costs = [ice_soft_rows(contact([-.04, -.04], times=times), POLICY)['parts']['ice_penetration']
                 for times in ([0., .02, .04], np.arange(17)*.0025)]
        np.testing.assert_allclose(costs[0], costs[1], rtol=0, atol=1e-14)

    def test_same_pts_groups_both_remain_and_unknown_airborne_create_no_support(self):
        rows = [evidence(), dict(evidence(), id='other_foot', feet=[1])]
        result = ice_soft_rows(contact([.05, .06], evidence=rows), POLICY)
        self.assertEqual(len(support_positions(result)), 2)
        for mode in ('unknown', 'airborne'):
            result = ice_soft_rows(contact([.05, .06], evidence=[dict(evidence(), mode=mode)]), POLICY)
            self.assertEqual(len(support_positions(result)), 0)
            self.assertEqual(len(result['residual']), 2)

    def test_policy_has_no_defaults_is_frozen_and_rejects_nonfinite_or_invalid_values(self):
        with self.assertRaises(TypeError):
            IceSoftPolicy()
        with self.assertRaises(FrozenInstanceError):
            POLICY.penetration_weight = 2.
        for field in fields(POLICY):
            invalid = [np.nan, np.inf, -np.inf, -1., True, [1.]]
            if not field.name.endswith('deadzone_m'):
                invalid.append(0.)
            for value in invalid:
                with self.subTest(field=field.name, value=value):
                    with self.assertRaises(ValueError):
                        replace(POLICY, **{field.name: value})
        replace(POLICY, penetration_deadzone_m=0., support_hover_deadzone_m=0.)

    def test_empty_rows_and_non_ice_rows_add_no_gate_or_multiplier_state(self):
        empty = dict(margin_m=np.empty(0), weights=np.empty(0), values=np.empty(0), labels=[])
        result = ice_soft_rows(empty, POLICY)
        self.assertEqual(result['residual'].shape, (0,))
        self.assertEqual(result['indices'].dtype.kind, 'i')
        self.assertEqual(result['derivative_per_margin_m'].shape, (0,))
        self.assertEqual(set(result), {'residual', 'indices', 'derivative_per_margin_m', 'parts', 'diagnostics'})
        packet = dict(margin_m=np.array([-3.]), weights=np.array([1.]), values=np.array([-3.]),
                      labels=[('neutral', 'left_foot_cavity')])
        result = ice_soft_rows(packet, POLICY)
        self.assertEqual(len(result['residual']), 0)
        self.assertEqual(result['diagnostics']['skipped_rows'][0]['reason'], 'not_an_ice_row')

    def test_input_vectors_are_checked_and_missing_raw_witness_is_explicit(self):
        packet = dict(margin_m=np.array([-.02]), weights=np.array([.02]), values=np.array([-.02]),
                      labels=[(0, 'left_nonpenetration')])
        result = ice_soft_rows(packet, POLICY)
        self.assertTrue(np.isnan(result['diagnostics']['raw_height_m'][0]))
        self.assertEqual(result['diagnostics']['raw_height_reasons'], ['raw_witness_metadata_unavailable'])
        for key, value in (('weights', [0.]), ('weights', [np.inf]), ('values', [np.nan]),
                           ('margin_m', [[-.02]]), ('values', []), ('labels', [])):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    ice_soft_rows(dict(packet, **{key: value}), POLICY)


if __name__ == '__main__':
    unittest.main()