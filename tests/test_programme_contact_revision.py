"""Preserve the full temporal contact association across an input revision."""
import json

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.programme_continue import inherited_contact_modes
from calibration.program_tracking.body_tracking.sequence_dynamics import MODE_NAMES


FIELDS = (
    "sequence_contact_modes",
    "sequence_contact_status",
    "sequence_contact_mode_names",
    "sequence_contact_association_json",
)


@pytest.fixture
def contact_packet():
    pts = np.array([283136, 283392, 284160], dtype=np.int64)
    record = {
        "method": "duration-expanded Viterbi",
        "mode_names": MODE_NAMES.tolist(),
        "source_episode_ids": ["support", "flight"],
        "inference_is_not_new_source_observation": True,
    }
    parent = {
        FIELDS[0]: np.array([[0, 6], [3, 5], [5, 5]], dtype=np.int16),
        FIELDS[1]: np.array([["inferred", "unknown"], ["inferred", "inferred"], ["inferred", "inferred"]]),
        FIELDS[2]: MODE_NAMES.copy(),
        FIELDS[3]: np.array(json.dumps(record, indent=2)),
        "action_objective_source_pts": pts.copy(),
        "x": np.array([4., -3.]),
    }
    return parent, pts


@pytest.mark.parametrize("legacy", [False, True], ids=["with-source-pts", "legacy-no-source-pts"])
def test_contact_payload_is_preserved_as_independent_arrays(contact_packet, legacy):
    parent, pts = contact_packet
    if legacy:
        del parent["action_objective_source_pts"]
    old = {key: value.copy() for key, value in parent.items()}

    child = inherited_contact_modes(parent, pts)

    assert set(child) == set(FIELDS)
    for key in FIELDS:
        np.testing.assert_array_equal(child[key], old[key])
        assert not np.shares_memory(child[key], parent[key])
    child[FIELDS[0]][0, 0] = 4
    child[FIELDS[1]][0, 0] = "unknown"
    child[FIELDS[2]][0] = "heel"
    child[FIELDS[3]][()] = "{}"
    for key, value in old.items():
        np.testing.assert_array_equal(parent[key], value)


@pytest.mark.parametrize("missing", FIELDS)
def test_missing_contact_payload_field_is_rejected(contact_packet, missing):
    parent, pts = contact_packet
    del parent[missing]

    with pytest.raises(ValueError, match=missing):
        inherited_contact_modes(parent, pts)


@pytest.mark.parametrize("field", FIELDS[2:], ids=["mode-list", "association-record"])
def test_wrong_contact_mode_vocabulary_is_rejected(contact_packet, field):
    parent, pts = contact_packet
    reordered = MODE_NAMES[::-1]
    if field == FIELDS[2]:
        parent[field] = reordered.copy()
    else:
        record = json.loads(str(parent[field]))
        record["mode_names"] = reordered.tolist()
        parent[field] = np.array(json.dumps(record))

    with pytest.raises(ValueError, match="vocabulary"):
        inherited_contact_modes(parent, pts)


@pytest.mark.parametrize("field", FIELDS[:2], ids=["modes", "status"])
def test_contact_payload_must_cover_every_exposure(contact_packet, field):
    parent, pts = contact_packet
    parent[field] = parent[field][:-1]

    with pytest.raises(ValueError, match="layout"):
        inherited_contact_modes(parent, pts)


def test_out_of_vocabulary_contact_index_is_rejected(contact_packet):
    parent, pts = contact_packet
    parent[FIELDS[0]][1, 0] = len(MODE_NAMES)

    with pytest.raises(ValueError, match="layout"):
        inherited_contact_modes(parent, pts)


def test_equal_length_contact_payload_at_different_native_pts_is_rejected(contact_packet):
    parent, pts = contact_packet
    shifted = pts.copy()
    shifted[1] += 1

    with pytest.raises((AssertionError, ValueError), match="native source PTS"):
        inherited_contact_modes(parent, shifted)
