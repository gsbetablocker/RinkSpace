"""Source-qualified gray identity re-entry without stale coordinates as LK support.

This successor consumes the ordinary model-context reader exactly once. Only a
new, current, two-image RGB seed can retire a stale navigation entry for the
same named gray material. Every ordinary selector pixel and identity gate then
runs unchanged. It creates neither a global alias nor a measured endpoint.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np

from .vision_ice_temporal_identity import TemporalIceIdentitySelector
from .vision_ice_single_gray_support_identity import (
    ConfirmedSingleGraySupportModelContextIdentity, F3,
)

F2 = 'house_connection_F2_R2:segment0'


class FreshConfirmedSingleGraySupportModelContextIdentity(ConfirmedSingleGraySupportModelContextIdentity):
    def consume(self, *args, **kwargs):
        # A no-candidate exposure must not display the previous frame's support
        # certificate. This changes recording only, never numeric acceptance.
        self.latest_single_gray_support_certificate = None
        return super().consume(*args, **kwargs)


class _CurrentContextOnce:
    def __init__(self, context, result, measurement):
        self.context = context
        self.result = result
        self.measurement = measurement
        self.called = False

    def consume(self, rgb, measurement, exclusion_mask, *, source_ice_scope_mask):
        if self.called or measurement is not self.measurement:
            raise ValueError('Current source model context must be consumed once')
        self.called = True
        return self.result

    def __getattr__(self, name):
        return getattr(self.context, name)


class CurrentSourceGrayReentrySelector(TemporalIceIdentitySelector):
    def _exact_current_gray_seed(self, seed, measurement, previous_context_binding):
        name = seed.get('curve_id')
        if name not in (F2, F3) or name not in self.inventory:
            return False
        binding = measurement['source_binding']
        pts = int(binding['source_pts'])
        evidence = seed.get('source_identity_navigation_evidence', {})
        if (seed.get('actual_pts') != pts or seed.get('source_sha256') != self.source_sha256
                or str(Fraction(seed['time_base'])) != self.time_base
                or seed.get('decoded_native_RGB_sha256') != measurement['decoded_native_RGB_sha256']
                or seed.get('decoded_native_BGR_sha256') != binding['decoded_native_BGR_sha256']
                or seed.get('shot_id') != binding['shot_id']
                or seed.get('source_material_class') != 'static_painted_ice'
                or seed.get('surface_and_physical_identity_confirmed') is not True
                or not seed.get('authority') or previous_context_binding is None):
            return False
        previous = evidence.get('previous_source_binding', {})
        if (previous != previous_context_binding or previous.get('shot_id') != binding['shot_id']
                or int(previous.get('actual_pts', pts)) >= pts
                or previous.get('source_sha256') != self.source_sha256
                or previous.get('time_base') != self.time_base):
            return False
        for field in ('actual_material_correspondence_confirmed','actual_field_order_confirmed',
                      'physical_model_scope_confirmed','same_shot_source_continuity_confirmed',
                      'source_identity_confirmed','current_feature_semantics_remeasured',
                      'competing_actual_material_interpretations_rejected'):
            if evidence.get(field) is not True:
                return False
        if evidence.get('actual_material_local_LK', {}).get('source_RGB_tracking_confirmed') is not True:
            return False
        current = evidence.get('source_binding', {})
        if (current.get('actual_pts') != pts or current.get('native_pts') != binding.get('native_pts', pts)
                or current.get('decoded_native_BGR_sha256') != binding['decoded_native_BGR_sha256']
                or current.get('shot_id') != binding['shot_id']):
            return False
        ids = np.asarray(evidence.get('current_actual_candidate_indices', ()), np.int64)
        if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
            return False
        queries = np.asarray(measurement['candidate_query'])
        if np.any(ids < 0) or np.any(ids >= len(queries)):
            return False
        ci = self.names.index(name)
        if (np.any(np.asarray(measurement['query_curve'])[queries[ids]] != ci)
                or np.any(np.asarray(measurement['candidate_flags'])[ids] & 3)
                or not np.array_equal(queries[ids], evidence.get('current_actual_query_indices'))):
            return False
        quantiles = np.asarray(measurement.get('candidate_side_quantiles_uv', ()), float)
        if quantiles.shape != (len(queries), 2, 3, 2) or not np.isfinite(quantiles[ids]).all():
            return False
        sides = quantiles[ids, :, 1]
        centers = sides.mean(1)
        if (not np.array_equal(sides, evidence.get('current_actual_two_sides_uv'))
                or not np.array_equal(centers, seed.get('navigation_uv'))
                or hashlib.sha256(np.ascontiguousarray(centers).tobytes()).hexdigest()
                != evidence.get('current_actual_measured_UV_sha256')):
            return False
        return True

    def consume(self, rgb, measurement, exclusion_mask, *, seeds=(), source_ice_scope_mask=None):
        self.latest_current_source_context_reentry = []
        context = self.model_context_identity
        if context is None or source_ice_scope_mask is None:
            return super().consume(rgb, measurement, exclusion_mask, seeds=seeds,
                                   source_ice_scope_mask=source_ice_scope_mask)
        prior = None if context.previous_binding is None else dict(context.previous_binding)
        current = context.consume(rgb, measurement, exclusion_mask,
                                  source_ice_scope_mask=source_ice_scope_mask)
        # The previous gray image is the immediately preceding exposure. A
        # track measured in an older image cannot be LK input in that image.
        if (self.previous_binding is not None
                and self.previous_binding['shot_id'] == measurement['source_binding']['shot_id']):
            previous_pts = int(self.previous_binding['source_pts'])
            for seed in current['seeds']:
                name = seed['curve_id']
                track = self.tracks.get(name)
                if (track is None or int(track['actual_pts']) == previous_pts
                        or not self._exact_current_gray_seed(seed, measurement, prior)):
                    continue
                self.latest_current_source_context_reentry.append(dict(
                    curve_id=name, stale_actual_pts=int(track['actual_pts']),
                    previous_gray_actual_pts=previous_pts,
                    current_actual_pts=int(measurement['source_binding']['source_pts']),
                    current_actual_candidate_indices=seed['source_identity_navigation_evidence']['current_actual_candidate_indices'],
                    current_actual_query_indices=seed['source_identity_navigation_evidence']['current_actual_query_indices'],
                    own_two_image_RGB_seed_qualified=True, stale_pixels_used_as_LK_support=False,
                    source_precision_scope_or_competition_gate_relaxed=False))
                del self.tracks[name]
        proxy = _CurrentContextOnce(context, current, measurement)
        self.model_context_identity = proxy
        try:
            result = super().consume(rgb, measurement, exclusion_mask, seeds=seeds,
                                     source_ice_scope_mask=source_ice_scope_mask)
            if not proxy.called:
                raise ValueError('Ordinary temporal selector did not consume its current context')
            return result
        finally:
            self.model_context_identity = context
