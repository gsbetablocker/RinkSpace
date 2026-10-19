"""Source-scale identity mapping and actual native chroma evidence."""
from copy import deepcopy

import numpy as np
import pytest

from calibration.program_tracking.recover_accepted_source_scales import (
    recover_endpoint_scale, native_profile_evidence, _verify_manual_point)


def expanded_segment():
    return dict(uv=[[40., 100.], [60., 100.], [80., 100.], [100., 100.]],
        point_ids=["opaque endpoint A", "opaque inserted B", "opaque endpoint C", "opaque endpoint D"],
        point_bindings=[{}, {"localization_sigma_px": 7.}, {}, {}],
        endpoint_evidence_transition_iqr_px=[2.698, 5.396, 8.094],
        endpoint_evidence_localization_sigma_px=[2., 4., 6.])


def test_expanded_endpoint_sigma_uses_original_complement_not_current_index():
    segment = expanded_segment()
    assert recover_endpoint_scale(segment, 2)["sigma_px"] == 4.
    assert recover_endpoint_scale(segment, 3)["sigma_px"] == 6.
    assert recover_endpoint_scale(segment, 2)["original_endpoint_evidence_index"] == 1
    with pytest.raises(ValueError, match="mapped uniquely"):
        recover_endpoint_scale(segment, 1)


@pytest.mark.parametrize("change", ["count", "invalid", "width"])
def test_ambiguous_or_changed_original_scale_does_not_get_a_default(change):
    segment = expanded_segment()
    if change == "count": segment["endpoint_evidence_localization_sigma_px"].pop()
    elif change == "invalid": segment["endpoint_evidence_localization_sigma_px"][1] = 0.
    else: segment["endpoint_evidence_transition_iqr_px"][1] = 30.
    with pytest.raises(ValueError): recover_endpoint_scale(segment, 2)


def test_manual_opaque_identity_restores_same_source_scale_after_segment_split():
    segment = expanded_segment()
    before = dict(features=[dict(id="board_contact", segments=[segment])])
    result = recover_endpoint_scale(segment, 2)
    _verify_manual_point(before, "board_contact", "opaque endpoint C", [80., 100.], result)
    with pytest.raises(ValueError, match="identity/UV"):
        _verify_manual_point(before, "board_contact", "opaque endpoint C", [81., 100.], result)


@pytest.mark.parametrize("signal", ["board_contact", "Ad2.top_edge"])
def test_actual_native_chroma_profile_measures_sigma_without_moving_uv(signal):
    y = np.arange(180)
    fraction = .5*(1+np.tanh((y-90)/4))
    raw = np.zeros((180, 80, 3), np.uint8)
    raw[:, :, 0] = 20
    raw[:, :, 2] = np.rint(20+160*fraction)[:, None]
    raw[:, :, 1] = raw[:, :, 2] if signal == "board_contact" else 20
    uv = [40.25, 91.5]
    result = native_profile_evidence(raw, uv, signal)
    assert result["measured_sigma_px"] > 0
    assert result["material_quantile_native_y"][0] < result["material_quantile_native_y"][2]
    assert result["accepted_UV_modified"] is False
    assert uv == [40.25, 91.5]


def test_no_material_transition_or_truncated_support_cannot_fabricate_sigma():
    raw = np.zeros((180, 80, 3), np.uint8)
    with pytest.raises(ValueError, match="unresolved"):
        native_profile_evidence(raw, [40., 90.], "board_contact")
    with pytest.raises(ValueError, match="support unavailable"):
        native_profile_evidence(raw, [40., 5.], "board_contact")
