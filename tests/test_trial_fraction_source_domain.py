"""Independent trial-fraction support domains preserve the original sources."""
from copy import deepcopy

import numpy as np
import pytest

from calibration.solver_runtime.trial_fraction_source_domain import attach_trial_fraction_domain


@pytest.fixture
def original_domain():
    previous = np.array([0., 2., 1e-8, 4., 0., 3.])
    owners = np.array([-1, 0, 1, 1, 2, 3, 4, -1, 5, 5])
    original = np.array([2., .7, .3, .9, -.4, 1.3, -2.1, .6, 4.2, -3.7])
    coefficient = np.array([0., 0., 2., 2., 0., 4., 0., 0., 3., 3.])
    rho = .6
    targets = previous.copy()
    targets[previous > 1e-8] *= rho
    responses = {
        "native": dict(row_start=1, row_stop=5, dynamic_group_limits=targets[:3].copy(),
                       dynamic_row_group_index=np.array([0, 1, 1, 2]),
                       nominal_carrier=np.array([1., 2.])),
        "wide": dict(row_start=5, row_stop=7, dynamic_group_limits=targets[3:5].copy(),
                     dynamic_row_group_index=np.array([0, 1]), nominal_carrier=np.array([3.])),
        "confirmed": dict(row_start=7, row_stop=10, dynamic_group_limits=targets[5:].copy(),
                          dynamic_row_group_index=np.array([-1, 0, 0]), nominal_carrier=np.array([4.])),
    }
    domain = dict(matrix=np.arange(20., dtype=float).reshape(10, 2),
                  lower=original-rho*coefficient, curvature_responses=responses,
                  report={"original_physical_guard_retained": True})
    options = dict(original_constraint_lower=original, row_group_index=owners,
                   previous_group_violations=previous, common_fraction=rho)
    return domain, options, coefficient


def assert_tree_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    else:
        assert actual == expected


def test_full_fraction_keeps_original_lower_targets_and_rows_bitexact(original_domain):
    domain, options, _ = original_domain
    attached = attach_trial_fraction_domain(domain, **options)
    full = attached["at_trial_fraction"](1.)
    assert full["matrix"] is domain["matrix"]
    assert full["lower"].tobytes() == domain["lower"].tobytes()
    for family, original in domain["curvature_responses"].items():
        restored = full["curvature_responses"][family]
        assert restored["dynamic_group_limits"].tobytes() == original["dynamic_group_limits"].tobytes()
        np.testing.assert_array_equal(restored["dynamic_row_group_index"], original["dynamic_row_group_index"])
        np.testing.assert_array_equal(restored["nominal_carrier"], original["nominal_carrier"])
    assert "at_trial_fraction" not in domain


@pytest.mark.parametrize("alpha", [.5, .125, 1./256.])
def test_short_fraction_uses_same_parent_group_interpolation(original_domain, alpha):
    domain, options, coefficient = original_domain
    trial = attach_trial_fraction_domain(domain, **options)["at_trial_fraction"](alpha)
    fraction = 1.-alpha*(1.-options["common_fraction"])
    np.testing.assert_array_equal(trial["lower"], options["original_constraint_lower"]-fraction*coefficient)
    previous = options["previous_group_violations"]
    targets = previous.copy()
    targets[previous > 1e-8] *= fraction
    actual = np.concatenate([response["dynamic_group_limits"]
                             for response in trial["curvature_responses"].values()])
    np.testing.assert_array_equal(actual, targets)
    unchanged = coefficient == 0.
    # Physical/unowned faces and zero (including exactly 1e-8) groups do not expand.
    assert trial["lower"][unchanged].tobytes() == domain["lower"][unchanged].tobytes()
    assert_tree_equal(trial["matrix"], domain["matrix"])


def test_calls_do_not_mutate_inputs_and_do_not_inherit_full_SOC_cuts(original_domain):
    domain, options, coefficient = original_domain
    saved_domain, saved_options = deepcopy(domain), deepcopy(options)
    attached = attach_trial_fraction_domain(domain, **options)
    first = attached["at_trial_fraction"](.5)
    first["lower"] += 100.
    first["curvature_responses"]["native"]["dynamic_group_limits"] += 100.
    second = attached["at_trial_fraction"](.5)
    fraction = 1.-.5*(1.-options["common_fraction"])
    np.testing.assert_array_equal(second["lower"], options["original_constraint_lower"]-fraction*coefficient)
    assert_tree_equal(domain, saved_domain)
    assert_tree_equal(options, saved_options)
    # Simulate the full-step SOC accumulator mutating its retained numerical lower.
    attached["lower"] += 1000.
    third = attached["at_trial_fraction"](.5)
    full = attached["at_trial_fraction"](1.)
    np.testing.assert_array_equal(third["lower"], saved_options["original_constraint_lower"]-fraction*coefficient)
    assert full["lower"].tobytes() == saved_domain["lower"].tobytes()
    # The callback also freezes the original parent values rather than caller aliases.
    options["original_constraint_lower"] += 300.
    options["previous_group_violations"] += 300.
    options["row_group_index"][:] = -1
    np.testing.assert_array_equal(attached["at_trial_fraction"](.5)["lower"], third["lower"])


@pytest.mark.parametrize("alpha", [0., -.1, 1.01, np.nan, np.inf])
def test_invalid_trial_fraction_is_rejected(original_domain, alpha):
    domain, options, _ = original_domain
    with pytest.raises(ValueError):
        attach_trial_fraction_domain(domain, **options)["at_trial_fraction"](alpha)


@pytest.mark.parametrize("bad", [-2, 6, .5])
def test_invalid_global_group_ownership_is_rejected(original_domain, bad):
    domain, options, _ = original_domain
    owners = options["row_group_index"].astype(float)
    owners[0] = bad
    with pytest.raises(ValueError):
        attach_trial_fraction_domain(domain, **dict(options, row_group_index=owners))


def test_mismatched_original_full_contraction_is_rejected(original_domain):
    domain, options, _ = original_domain
    domain["lower"][2] += .01
    with pytest.raises(ValueError, match="Full-step"):
        attach_trial_fraction_domain(domain, **options)


@pytest.mark.parametrize("bad", [-2, 3, .5])
def test_invalid_family_local_group_ownership_is_rejected(original_domain, bad):
    domain, options, _ = original_domain
    local = domain["curvature_responses"]["native"]["dynamic_row_group_index"].astype(float)
    local[0] = bad
    domain["curvature_responses"]["native"]["dynamic_row_group_index"] = local
    with pytest.raises(ValueError):
        attach_trial_fraction_domain(domain, **options)


def test_family_global_row_mapping_must_refer_to_same_original_groups(original_domain):
    domain, options, _ = original_domain
    domain["curvature_responses"]["native"]["dynamic_row_group_index"][0] = 1
    with pytest.raises(ValueError, match="row ownership"):
        attach_trial_fraction_domain(domain, **options)


def test_family_order_or_missing_parent_groups_is_rejected(original_domain):
    domain, options, _ = original_domain
    domain["curvature_responses"] = {key: domain["curvature_responses"][key]
                                      for key in ("wide", "native", "confirmed")}
    with pytest.raises(ValueError, match="group target order"):
        attach_trial_fraction_domain(domain, **options)


def test_independent_trial_callbacks_cannot_recursively_reuse_SOC_domains(original_domain):
    domain, options, _ = original_domain
    attached = attach_trial_fraction_domain(domain, **options)
    assert "at_trial_fraction" not in attached["at_trial_fraction"](.5)
