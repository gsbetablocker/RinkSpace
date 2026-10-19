"""Reuse exact original projections in one native review.

Support and raw-pixel review operators run in their original order. Only the
pure arrays for exactly repeated inputs are reused. Source qualification,
optical support application and the final review reports still run normally.
"""
from __future__ import annotations

from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np


_SCOPE = ContextVar('rinkspace_native_review_projection', default=None)
_RECEIPT = None


def _copy_result(value):
    return tuple(part.copy() for part in value)


def _projection_wrapper(original, module, *, central):
    @functools.wraps(original)
    def evaluate(camera, xyz, *args, **kwargs):
        scope = _SCOPE.get()
        if (len(args) > (0 if central else 1)
                or (args and 'with_jacobian' in kwargs)
                or set(kwargs) - {'with_jacobian', '_brown4_output'}):
            return original(camera, xyz, *args, **kwargs)
        jacobian = kwargs.get('with_jacobian', args[0] if args else False)
        if scope is None or jacobian or kwargs.get('_brown4_output', False):
            return original(camera, xyz, *args, **kwargs)
        try:
            c = np.asarray(camera, dtype=float)
            world = np.ascontiguousarray(np.asarray(xyz, dtype=float).reshape(-1, 3))
            if c.shape != (25,) or not np.isfinite(c).all() or not np.isfinite(world).all():
                return original(camera, xyz, *args, **kwargs)
            key = (original, c.tobytes(), world.shape, world.tobytes(),
                   module._central_projection_backend,
                   os.environ.get('RINKSPACE_PROJECTION_BACKEND'),
                   os.environ.get('RINKSPACE_MOTION_BACKEND'),
                   os.environ.get('RINKSPACE_PROJECTION_VALUE_BACKEND'))
        except (TypeError, ValueError):
            return original(camera, xyz, *args, **kwargs)
        saved = scope['values'].get(key)
        if saved is not None:
            scope['central_hits' if central else 'fast_hits'] += 1
            return _copy_result(saved)
        result = original(camera, xyz, *args, **kwargs)
        if (type(result) is tuple and len(result) == (3 if central else 2)
                and all(isinstance(part, np.ndarray) and not part.dtype.hasobject for part in result)):
            scope['values'][key] = _copy_result(result)
        return result
    return evaluate


def install_review_projection_reuse_runtime(*, report_directory=None):
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    import projection
    import ice_refine
    import z190_source_camera_diagnostic as diagnostic
    from structured_programme_source_candidate import native_ice_curve_source
    from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
    NativeIceCurveFactor = native_ice_curve_source.NativeIceCurveFactor
    original_reviews = {NativeBoardEdgeFactor: NativeBoardEdgeFactor.review,
                        NativeIceCurveFactor: NativeIceCurveFactor.review}
    originals = {name: getattr(projection, name) for name in ('project_fast', 'project_central_valid')}
    aliases = []
    # These are the original projection consumers reached by support/raw
    # board review. Replace only aliases that still name the original method.
    for name, original in originals.items():
        wrapped = _projection_wrapper(original, projection, central=name == 'project_central_valid')
        for module in (projection, ice_refine, diagnostic):
            if getattr(module, name, None) is original:
                setattr(module, name, wrapped)
                aliases.append(module.__name__ + '.' + name)

    def wrap_review(original_review, factor_name):
        @functools.wraps(original_review)
        def review(owner, q):
            scope = dict(values={}, central_hits=0, fast_hits=0, curve_hits=0)
            token = _SCOPE.set(scope)
            try:
                return original_review(owner, q)
            finally:
                _SCOPE.reset(token)
                if report_directory is not None:
                    event = dict(factor=factor_name, original_projection_entries=len(scope['values']),
                                 repeated_central_projection_calls=scope['central_hits'],
                                 repeated_fast_projection_calls=scope['fast_hits'],
                                 repeated_curve_queries=scope['curve_hits'])
                    with (Path(report_directory) / 'native_board_review_projection_reuse.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(event) + '\n')
                scope['values'].clear()
        return review
    for factor, original_review in original_reviews.items():
        factor.review = wrap_review(original_review, factor.__name__)
    path = Path(__file__).resolve()
    _RECEIPT = dict(schema='rinkspace.exact-native-review-projection-reuse.v1',
        helper_path=str(path), helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_review_sha256={factor.__name__:hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                               for factor, function in original_reviews.items()},
        original_projection_sha256={name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                                    for name, function in originals.items()},
        exact_original_aliases=aliases,
        scope='One original native board/ice review call; exact camera/world/source-query bytes',
        original_support_raw_pixel_qualification_and_HELD_review_retained=True,
        Jacobian_calls_unchanged=True, returned_arrays_independently_owned=True,
        failures_not_cached=True, registered_source_files_modified=False, accuracy_accepted=False)
    if report_directory is not None:
        destination = Path(report_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'review_projection_reuse_receipt.json').write_text(
            json.dumps(_RECEIPT, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return _RECEIPT
