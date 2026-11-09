"""Guard each confirmed flight on its actual weighted COM and momentum rows."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.programme_flight_guard import ProgrammeFlightGuard
from calibration.program_tracking.body_tracking.weighted_blocks import block, full_columns, pack_weighted


EPISODES = ("entry_flight_core", "jump_3f", "jump_3t", "jump_2a", "jump_3lz")
COM = "flight_integrated_com"
MOMENTUM = "flight_specific_momentum"
FLIGHT_NAMES = tuple(f"episode/{episode}/{family}" for episode in EPISODES for family in (COM, MOMENTUM))
PREFIX = "__flight__/"


@pytest.fixture
def weighted_flights():
    names = (*FLIGHT_NAMES, "body_pixels", "episode/old/flight_specific_angular_impulse")
    entries = []
    weights = {}
    for i, name in enumerate(names):
        j = csr_matrix(
            ([1., -.5, .3, 2.], ([0, 0, 1, 2], [0, i + 2, 23, 32])),
            shape=(3, 33),
        )
        entries.append(full_columns(block(name, [3. + i, -4., 0.], j)))
        weights[name] = [4., 1., .25]
    _, _, packed = pack_weighted(entries, weights, jacobian=True)
    return {"weighted_blocks": packed}


def replaced(value, name, **updates):
    return {"weighted_blocks": [replace(b, **updates) if b.name == name else b
                                for b in value["weighted_blocks"]]}


def test_caps_domains_and_dependencies_use_complete_actual_weighted_rows(weighted_flights):
    guard = ProgrammeFlightGuard(weighted_flights)
    assert set(guard.caps) == set(FLIGHT_NAMES)
    assert guard.caps[FLIGHT_NAMES[0]] == pytest.approx(np.sqrt(6.**2 + 4.**2))
    assert guard.dependency(PREFIX + FLIGHT_NAMES[0]) is None

    domains = {d["source_id"]: d for d in guard.domains(weighted_flights)}
    assert set(domains) == {PREFIX + name for name in FLIGHT_NAMES}
    direction = np.zeros(50)
    direction[[17, 21, 40, 49]] = [.4, -.6, 1.1, .7]
    for term in weighted_flights["weighted_blocks"][:len(FLIGHT_NAMES)]:
        key = PREFIX + term.name
        domain = domains[key]
        assert domain["offset"].shape == (1, 3)
        np.testing.assert_array_equal(domain["offset"][0], term.residual)
        assert not np.shares_memory(domain["offset"], term.residual)
        np.testing.assert_array_equal(domain["radius"], [np.linalg.norm(term.residual)])
        assert domain["labels"] == [key] and domain["model_foot"] is None
        matrix = domain["jacobian"]
        dependency = guard.dependency(key)
        assert matrix.format == "csr" and matrix.shape == term.jacobian.shape == (3, 50)
        assert matrix[:, :17].nnz == 0 and matrix[2, 49] != 0.
        for field in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(matrix, field), getattr(term.jacobian, field))
            np.testing.assert_array_equal(getattr(dependency["jacobian"], field), getattr(matrix, field))
        np.testing.assert_array_equal(matrix @ direction, term.jacobian @ direction)
        assert dependency["model_foot"] is None
    assert guard.dependency(PREFIX + "unrelated") is None


@pytest.mark.parametrize("worsened", FLIGHT_NAMES)
def test_each_episode_family_can_reject_a_large_aggregate_improvement(weighted_flights, worsened):
    guard = ProgrammeFlightGuard(weighted_flights)
    base = weighted_flights["weighted_blocks"]
    trial = [replace(b, residual=b.residual * (1.1 if b.name == worsened else .01)) for b in base]
    assert sum(np.linalg.norm(b.residual)**2 for b in trial) < sum(np.linalg.norm(b.residual)**2 for b in base)

    checks = guard.checks({"weighted_blocks": list(reversed(trial))})

    assert {key for key, check in checks.items() if check["rejected"]} == {PREFIX + worsened}
    violation = checks[PREFIX + worsened]
    assert violation["before"] == guard.caps[worsened]
    assert violation["after"] == pytest.approx(1.1 * guard.caps[worsened])
    assert violation["unit"] == "weighted flight residual norm"


def test_persisted_caps_never_accumulate_evaluation_allowance(weighted_flights):
    original = ProgrammeFlightGuard(weighted_flights)
    name = FLIGHT_NAMES[0]
    block_zero = weighted_flights["weighted_blocks"][0]
    cap = original.caps[name]
    allowance = original.tolerance(cap)
    barely_above = replaced(weighted_flights, name, residual=block_zero.residual * (1. + .75 * allowance / cap))

    inherited = original
    for _ in range(4):
        assert not inherited.checks(barely_above)[PREFIX + name]["rejected"]
        inherited = ProgrammeFlightGuard(barely_above, previous=json.loads(json.dumps(inherited.record())))
        assert inherited.caps[name] == cap
    too_high = replaced(weighted_flights, name, residual=block_zero.residual * (1. + 1.25 * allowance / cap))
    assert inherited.checks(too_high)[PREFIX + name]["rejected"]
    with pytest.raises(ValueError, match="preceding flight guard"):
        ProgrammeFlightGuard(too_high, previous=inherited.record())

    better = replaced(weighted_flights, name, residual=.8 * block_zero.residual)
    improved = ProgrammeFlightGuard(better, previous=inherited.record())
    assert improved.caps[name] == pytest.approx(.8 * cap)
    assert improved.checks(weighted_flights)[PREFIX + name]["rejected"]


def test_missing_family_and_changed_flight_population_are_rejected(weighted_flights):
    guard = ProgrammeFlightGuard(weighted_flights)
    name = FLIGHT_NAMES[0]
    terms = weighted_flights["weighted_blocks"]
    missing = {"weighted_blocks": [b for b in terms if b.name != name]}
    with pytest.raises(ValueError, match="both COM and specific momentum"):
        ProgrammeFlightGuard(missing)
    with pytest.raises(ValueError, match="both COM and specific momentum"):
        guard.checks(missing)
    fewer_rows = replaced(weighted_flights, name, residual=terms[0].residual[:-1])
    changed_episode = {"weighted_blocks": [
        replace(b, name=b.name.replace("episode/entry_flight_core/", "episode/other_flight/")) for b in terms
    ]}
    for changed in (fewer_rows, changed_episode):
        with pytest.raises(ValueError, match="population changed"):
            guard.checks(changed)
        with pytest.raises(ValueError, match="population changed"):
            guard.domains(changed)
        with pytest.raises(ValueError, match="population changed"):
            ProgrammeFlightGuard(changed, previous=guard.record())


def test_nonfinite_residual_and_inherited_cap_are_rejected(weighted_flights):
    guard = ProgrammeFlightGuard(weighted_flights)
    name = FLIGHT_NAMES[0]
    nonfinite = replaced(weighted_flights, name, residual=np.array([1., 0., np.nan]))
    with pytest.raises(ValueError, match="Invalid flight residual"):
        ProgrammeFlightGuard(nonfinite)
    with pytest.raises(ValueError, match="Invalid flight residual"):
        guard.checks(nonfinite)
    previous = json.loads(json.dumps(guard.record()))
    previous["caps"][name] = np.inf
    with pytest.raises(ValueError, match="Invalid previous flight norm cap"):
        ProgrammeFlightGuard(weighted_flights, previous=previous)


def test_linear_domain_requires_actual_jacobian_with_the_original_rows(weighted_flights):
    guard = ProgrammeFlightGuard(weighted_flights)
    term = weighted_flights["weighted_blocks"][0]
    for missing in (None, term.jacobian[:-1]):
        with pytest.raises(ValueError, match="Actual flight Jacobian"):
            guard.domains(replaced(weighted_flights, term.name, jacobian=missing))


def test_linear_domain_rejects_nonfinite_jacobian_values(weighted_flights):
    guard = ProgrammeFlightGuard(weighted_flights)
    term = weighted_flights['weighted_blocks'][0]
    broken = term.jacobian.copy()
    broken.data[0] = np.inf
    with pytest.raises(ValueError, match='Nonfinite flight Jacobian'):
        guard.domains(replaced(weighted_flights, term.name, jacobian=broken))

def test_same_flight_definition_recovers_persisted_caps(weighted_flights):
    identity = "same mass proxy, source clock and flight operators"
    original = ProgrammeFlightGuard(weighted_flights, definition_identity=identity)
    previous = json.loads(json.dumps(original.record()))
    assert previous["definition_identity"] == identity

    recovered = ProgrammeFlightGuard(
        weighted_flights, previous=previous, definition_identity=identity,
    )

    assert recovered.rows == original.rows
    assert recovered.caps == original.caps
    assert recovered.record()["definition_identity"] == identity


@pytest.mark.parametrize("changed_identity", ["same-length rows, different mass proxy", None])
def test_same_named_and_sized_rows_cannot_recover_a_different_definition(weighted_flights, changed_identity):
    original = ProgrammeFlightGuard(weighted_flights, definition_identity="original flight definition")
    previous = json.loads(json.dumps(original.record()))

    with pytest.raises(ValueError, match="(?i)definition|identity"):
        ProgrammeFlightGuard(weighted_flights, previous=previous, definition_identity=changed_identity)
