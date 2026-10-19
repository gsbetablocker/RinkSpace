"""Reuse native RGB review with exact original search-query precision.

An appended source guide promotes an array to float64. Restore the original
query prefix before that append rather than promoting rounded cache values.
Candidate pixels, material sides, masks, and source clocks remain unchanged.
"""
from __future__ import annotations

from .native_alias_source_review import review_native_alias as _review_native_alias
from .programme_source_measurement_cache import restore_exact_navigation_queries


class ExactNativeQueryReader:
    def __init__(self, reader):
        self.reader = reader

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def read(self, rgb, pts, camera25, exclusion_mask, *, shot_id=None):
        measurement = self.reader.read(rgb, pts, camera25, exclusion_mask,
                                       shot_id=shot_id)
        restore_exact_navigation_queries(measurement, self.reader, camera25)
        return measurement


def review_native_alias(*, navigation_reader, **kwargs):
    return _review_native_alias(
        navigation_reader=ExactNativeQueryReader(navigation_reader), **kwargs)
