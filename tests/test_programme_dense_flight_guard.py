"""Dense gravity caps use their own episodes, identity and sparse domains."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking.programme_flight_guard import (
    ProgrammeDenseFlightGuard,
    ProgrammeFlightGuard,
)
from calibration.program_tracking.body_tracking.weighted_blocks import block


NAMES = tuple(f"episode/{name}/flight_dense_com_acceleration" for name in ("entry", "jump"))
PREFIX = "__flight_dense__/"
IDENTITY = "same mass proxy, source PTS and dense stencils"


@pytest.fixture
def combined():
    dense = [
        block(NAMES[0], [3., 4., 0.], csr_matrix(([2., -.3, .7], ([0, 1, 1], [17, 36, 74])), shape=(3, 80))),
        block(NAMES[1], [2., 0., 2.], csr_matrix(([.4, .8, 1.], ([0, 2, 2], [29, 45, 77])), shape=(3, 80))),
    ]
    native = [
        block(f"episode/{name}/{family}", [2., 2., 0.])
        for name in ("entry", "jump") for family in ("flight_integrated_com", "flight_specific_momentum")
    ]
    return {"weighted_blocks": [*dense, *native, block("body_pixels", [100., 100.])]}


def rescale(value, scales):
    return {"weighted_blocks": [
        replace(term, residual=term.residual * scales.get(term.name, 1.))
        for term in value["weighted_blocks"]
    ]}


@pytest.mark.parametrize("worsened", NAMES)
def test_one_dense_flight_rejects_despite_improved_native_and_other_dense_norms(combined, worsened):
    old = ProgrammeFlightGuard(combined)
    dense = ProgrammeDenseFlightGuard(combined, definition_identity=IDENTITY)
    trial = rescale(combined, {term.name: 1.15 if term.name == worsened else .1
                              for term in combined["weighted_blocks"]})
    assert all(not row["rejected"] for row in old.checks(trial).values())
    checks = dense.checks(trial)
    assert set(checks) == {PREFIX + name for name in NAMES}
    assert {key for key, row in checks.items() if row["rejected"]} == {PREFIX + worsened}


def test_dense_domains_and_dependencies_keep_actual_sparse_global_columns(combined):
    guard = ProgrammeDenseFlightGuard(combined, definition_identity=IDENTITY)
    assert guard.dependency(PREFIX + NAMES[0]) is None
    domains = {row["source_id"]: row for row in guard.domains(combined)}
    assert set(domains) == {PREFIX + name for name in NAMES}
    for term in combined["weighted_blocks"][:2]:
        key = PREFIX + term.name
        row = domains[key]
        jacobian = guard.dependency(key)["jacobian"]
        assert jacobian.format == "csr" and jacobian.shape == (3, 80)
        assert jacobian[:, :17].nnz == 0
        assert (row["jacobian"] - term.jacobian).nnz == 0
        assert (jacobian - term.jacobian).nnz == 0
        np.testing.assert_array_equal(row["offset"], term.residual[None, :])
        np.testing.assert_array_equal(row["radius"], [np.linalg.norm(term.residual)])
    assert guard.dependency("__flight__/episode/entry/flight_integrated_com") is None


def test_dense_resume_requires_matching_definition_and_population_without_cap_growth(combined):
    initial = ProgrammeDenseFlightGuard(combined, definition_identity=IDENTITY)
    saved = json.loads(json.dumps(initial.record()))
    assert saved["schema"] == "rinkspace.dense-flight-nondegradation.v1"
    assert saved["definition_identity"] == IDENTITY
    with pytest.raises(ValueError, match="definition identity changed"):
        ProgrammeDenseFlightGuard(combined, previous=saved, definition_identity="same rows, different dense clock")
    previous_native = ProgrammeFlightGuard(combined, definition_identity=IDENTITY).record()
    with pytest.raises(ValueError, match="population changed"):
        ProgrammeDenseFlightGuard(combined, previous=previous_native, definition_identity=IDENTITY)
    missing = {"weighted_blocks": [term for term in combined["weighted_blocks"] if term.name != NAMES[0]]}
    with pytest.raises(ValueError, match="population changed"):
        ProgrammeDenseFlightGuard(missing, previous=saved, definition_identity=IDENTITY)

    near = rescale(combined, {NAMES[0]: 1. + .6e-7, NAMES[1]: .8})
    inherited = ProgrammeDenseFlightGuard(near, previous=saved, definition_identity=IDENTITY)
    assert inherited.caps[NAMES[0]] == initial.caps[NAMES[0]]
    assert inherited.caps[NAMES[1]] == pytest.approx(.8 * initial.caps[NAMES[1]])
    above = rescale(combined, {NAMES[0]: 1. + 1.2e-7})
    with pytest.raises(ValueError, match="preceding flight guard"):
        ProgrammeDenseFlightGuard(above, previous=inherited.record(), definition_identity=IDENTITY)
