"""State/coordinate contracts for bounded fallback; synthetic stubs, not accuracy."""
from __future__ import annotations

import numpy as np
import pytest

from rinkspace import model_localization as module
from rinkspace.geometry import project


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.setattr(module, 'load_model', lambda path: {'model_id': 'fixture', 'revision': 'v0'})
    loc = module.ModelLocalizer('disposable-fixture', remove_anchors=True,
                                refinement_motion_model='similarity', reobserve_every=5)
    image = np.full((540, 960, 3), 120, np.uint8)
    domain = np.full((540, 960), 255, np.uint8)
    monkeypatch.setattr(module, 'ice_mask', lambda image: {'ice': domain, 'usable': domain})
    monkeypatch.setattr(module, 'row_foreground_exclusion', lambda image, ice: {
        'usable': domain, 'foreground': np.zeros_like(domain), 'method': 'synthetic_stub', 'config': {}})
    monkeypatch.setattr(module, '_features', lambda image: ([], None))
    monkeypatch.setattr(module, '_proposal', lambda *args: None)

    class NoLocalGeometry:
        def __init__(self):
            self.breaks = []

        def push(self, image, sample, masks, force_break=False):
            self.breaks.append(force_break)
            return {'status': 'unavailable', 'H_reference_from_current': None,
                    'graph_component_id': 'local_after_lk_loss'}

    loc.local = NoLocalGeometry()
    support = [[80, 240], [430, 240], [430, 470], [80, 470]]
    h = np.array([[1.2, .08, 15.], [0, 1.1, 21.], [0, 0, 1.]])
    loc.last_global = {'H_model_from_sample': h.copy(), 'anchor_id': 'actual_model_anchor',
        'physical_ids': ['R1'], 'support_model': project(h, support).tolist(),
        'camera_group_id': 'camera_candidate', 'error_summary': {'p90_px': 1.},
        'visible_curve_support': [{'physical_id': 'R1', 'curve_samples_in_support': 30}],
        'is_reference_self_frame': False, 'anchor_source_pts': 30000, 'anchor_source_time_base': '1/1000'}
    loc.last_anchor_time = 10.
    loc.last_component = 'local_before_lk_loss'
    loc.bridge = None
    loc.last_render = {'gray': np.full((540, 960), 100, np.uint8),
        'H_model_from_sample': h.copy(), 'source_time_s': 10.,
        'support_mask': module.polygon_mask(domain.shape, support)}
    delta = np.array([[1., 0, 2.], [0, 1., -1.], [0, 0, 1.]])
    calls = []

    def fit(reference, current, reference_mask, current_mask, initial, *, config):
        calls.append({'reference': reference, 'reference_mask': reference_mask,
                      'current_mask': current_mask, 'initial': np.asarray(initial).copy()})
        return {'status': 'candidate', 'reason': 'synthetic_pair_candidate',
                'H_current_from_reference': delta.tolist(),
                'support_current_polygon': support, 'final_curve_residual': {'p90_px': 1.}}

    monkeypatch.setattr(module, 'refine_model_projection', fit)

    def sample(time, index=1):
        return {'sample_index': index, 'source_time_s': time, 'source_sha256': 'b' * 64,
                'pts': round(time * 1000), 'time_base': '1/1000', 'width': 1920, 'height': 1080,
                'source_frame_index': index, 'shot_id': 'same_shot'}

    return loc, image, sample, h, delta, calls, support


def test_fallback_composes_in_the_correct_direction_and_remains_prediction(fallback):
    loc, image, sample, h, delta, calls, _ = fallback
    anchor = loc.last_global
    row = loc.push(image, sample(10.04))
    assert row['status'] == 'predicted'
    assert row['reason'] == 'bounded_masked_ice_photometric_propagation_from_model_anchor'
    np.testing.assert_allclose(row['H_model_from_image'], h @ np.linalg.inv(delta) @ np.diag([.5, .5, 1]))
    assert loc.last_anchor_time == 10.
    assert loc.last_global is anchor
    assert loc.bridge is None
    assert loc.last_render['source_time_s'] == 10.04
    assert row['reobserved_physical_ids'] == []
    assert not row['independent_accuracy_validated']
    assert row['propagated_from']['anchor_source_time_s'] == 10.
    assert row['anchor_source_pts'] == 30000
    assert row['bounded_photometric_propagation']['role'] == 'local_propagation_only_not_a_new_model_observation'
    assert len(calls) == 1


def test_many_successful_pairs_cannot_refresh_or_extend_the_anchor_expiry(fallback):
    loc, image, sample, _, _, calls, _ = fallback
    for index in range(1, 16):
        row = loc.push(image, sample(10. + .04 * index, index))
        assert row['status'] == 'predicted'
        assert loc.last_anchor_time == 10.
        assert row['propagated_from']['anchor_source_time_s'] == 10.
    assert len(calls) == 15
    row = loc.push(image, sample(10.64, 16))
    assert row['status'] == 'unavailable' and row['H_model_from_image'] is None
    assert len(calls) == 15
    assert loc.last_render['source_time_s'] == pytest.approx(10.6)
    assert loc.last_anchor_time == 10.


@pytest.mark.parametrize('time', [10.081, 10., 9.96])
def test_pair_gap_repeated_pts_and_backwards_time_do_not_use_the_fallback(fallback, time):
    loc, image, sample, _, _, calls, _ = fallback
    row = loc.push(image, sample(time))
    assert row['status'] == 'unavailable' and row['H_model_from_image'] is None
    assert calls == []


def test_a_cut_clears_previous_render_even_when_the_pair_would_otherwise_fit(fallback):
    loc, image, sample, _, _, calls, _ = fallback
    row = loc.push(image, sample(10.04), force_break=True)
    assert row['status'] == 'unavailable' and row['H_model_from_image'] is None
    assert calls == []
    assert loc.last_render is None and loc.last_global is None
    assert loc.last_anchor_time is None and loc.bridge is None
    assert loc.local.breaks == [True]


def test_last_render_without_a_model_anchor_cannot_bootstrap_global_geometry(fallback):
    loc, image, sample, _, _, calls, _ = fallback
    loc.last_global = None
    row = loc.push(image, sample(10.04))
    assert row['status'] == 'unavailable' and row['H_model_from_image'] is None
    assert calls == []


def test_a_refused_pair_cannot_return_an_old_map_or_advance_last_render(fallback, monkeypatch):
    loc, image, sample, _, _, _, _ = fallback
    monkeypatch.setattr(module, 'refine_model_projection', lambda *args, **kwargs: {
        'status': 'unavailable', 'reason': 'curve_support_missing', 'H_current_from_reference': None})
    row = loc.push(image, sample(10.04))
    assert row['status'] == 'unavailable' and row['H_model_from_image'] is None
    assert row['bounded_photometric_propagation']['status'] == 'unavailable'
    assert loc.last_anchor_time == 10. and loc.last_render['source_time_s'] == 10.


@pytest.mark.parametrize('fixed_fit_success', [False, True])
def test_prediction_can_seed_a_fixed_template_but_only_its_fresh_fit_is_a_new_anchor(
        fallback, monkeypatch, fixed_fit_success):
    loc, image, sample, h, delta, _, support = fallback
    fixed_gray = np.full((540, 960), 80, np.uint8)
    fixed_h = np.array([[2., 0, 100.], [0, 2., 50.], [0, 0, 1.]])
    loc.refs = [{'view': {'id': 'actual_model_anchor', 'H_model_from_sample': fixed_h.tolist(),
        'physical_ids': ['R1'], 'observations': [{'physical_id': 'R1', 'ellipse': [140, 160, 40, 15, 0]}],
        'source_sha256': 'a' * 64, 'pts': 30000, 'time_base': '1/1000'},
        'gray': fixed_gray, 'mask': np.full((540, 960), 255, np.uint8), 'key': [], 'desc': None}]
    calls = []

    def fit(reference, current, reference_mask, current_mask, initial, *, config):
        is_fixed = reference is fixed_gray
        calls.append((is_fixed, np.asarray(initial)))
        if is_fixed and not fixed_fit_success:
            return {'status': 'unavailable', 'reason': 'fixed_template_curve_mismatch'}
        return {'status': 'candidate', 'reason': 'synthetic_fit',
            'H_current_from_reference': np.asarray(initial).tolist() if is_fixed else delta.tolist(),
            'support_current_polygon': support, 'final_curve_residual': {'p90_px': 1.}}

    monkeypatch.setattr(module, 'refine_model_projection', fit)
    row = loc.push(image, sample(10.04, index=5))
    assert calls[0][0] is False and calls[1][0] is True
    expected_prediction = h @ np.linalg.inv(delta)
    np.testing.assert_allclose(calls[1][1], np.linalg.inv(expected_prediction) @ fixed_h)
    assert row['reobservation_attempts'][0]['proposal_role'] == 'bounded_ice_photometric_motion_to_fixed_model_reobservation'
    if fixed_fit_success:
        assert row['status'] == 'candidate'
        assert loc.last_anchor_time == 10.04
        assert row['propagated_from'] is None
        assert row['reobserved_physical_ids'] == ['R1']
    else:
        assert row['status'] == 'predicted'
        assert loc.last_anchor_time == 10.
        assert row['reobserved_physical_ids'] == []
    np.testing.assert_allclose(row['H_model_from_image'], expected_prediction @ np.diag([.5, .5, 1]))
