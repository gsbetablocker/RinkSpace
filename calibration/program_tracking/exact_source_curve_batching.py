"""Read-only lossless reuse of one original finite curve across source rows.

Every UV and original sigma is kept. FIT/HELD, native clocks, physical target,
operator and model state identities remain separate. This is not a solver
adapter and does not qualify source observations or install new weights.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import numpy as np


_CONTEXT_FIELDS = ('source_pack', 'source_model_identity', 'actual_pts', 'native_pts',
    'decoded_BGR_sha256', 'scope', 'physical_material_id', 'model_target',
    'observation_semantics', 'split')
_FINITE_FIELDS = frozenset(('world', 'uv', 'sigma', 'point_correspondence',
    'curve_observation_kind', 'observation_semantics'))


def prepare_exact_source_curve_batches(entries):
    """Group only the explicitly ordinary finite nearest-curve operator.

    Typed extents, branches, point correspondences and optical support keep
    singleton original calls. Full original entries and source locators remain
    referenced, including each independent visible source path.
    """
    groups = OrderedDict()
    scalar_cursor = 0
    entries = tuple(entries)
    for index, entry in enumerate(entries):
        context, row = entry['source_context'], entry['evaluation_row']
        if set(context) != set(_CONTEXT_FIELDS) or context['split'] not in ('fit', 'held'):
            raise ValueError('Exact model target, source clocks and explicit FIT/HELD context required')
        uv, sigma, world = (np.asarray(row[k]) for k in ('uv', 'sigma', 'world'))
        if (uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv)
                or sigma.shape != (len(uv),) or not np.isfinite(uv).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0)
                or world.ndim != 2 or world.shape[1:] != (3,)
                or len(world) < 2 or not np.isfinite(world).all()
                or row.get('observation_semantics') != context['observation_semantics']):
            raise ValueError('Finite original source UV/sigma and exact finite primitive required')
        ordinary = (set(row).issubset(_FINITE_FIELDS)
            and row.get('point_correspondence') is False
            and row.get('curve_observation_kind')=='measured_transition_location')
        # Digest is only a grouping index. Literal primitive bytes and all
        # context/attributes are checked before sharing a projection.
        world_bytes = np.ascontiguousarray(world).tobytes()
        operator = {k:row[k] for k in row if k not in ('world', 'uv', 'sigma')}
        identity = json.dumps(dict(context=context, operator=operator), sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        key = (identity, uv.dtype.str, sigma.dtype.str, world.dtype.str, world.shape,
            hashlib.sha256(world_bytes).digest()) if ordinary else ('singleton', index)
        if key not in groups:
            groups[key] = dict(indices=[], original_scalar_ranges=[], original_point_counts=[],
                first_row=row, world_bytes=world_bytes, actual_pts=context['actual_pts'],
                ordinary_finite_curve_reuse=ordinary)
        group = groups[key]
        if group['world_bytes'] != world_bytes:
            raise ValueError('Same digest identifies different finite primitive bytes')
        group['indices'].append(index)
        group['original_scalar_ranges'].append((scalar_cursor, scalar_cursor+2*len(uv)))
        group['original_point_counts'].append(len(uv))
        scalar_cursor += 2*len(uv)
    result = []
    for group in groups.values():
        indices = group['indices']
        if len(indices)==1:
            combined = group['first_row']
        else:
            combined = dict(group['first_row'],
                uv=np.concatenate([entries[i]['evaluation_row']['uv'] for i in indices]),
                sigma=np.concatenate([entries[i]['evaluation_row']['sigma'] for i in indices]))
        result.append(dict(original_entry_indices=indices,
            original_scalar_ranges=group['original_scalar_ranges'],
            original_point_counts=group['original_point_counts'],
            source_locators=[entries[i]['source_locator'] for i in indices],
            independent_source_visible_paths=[entries[i]['source_visible_paths'] for i in indices],
            actual_pts=group['actual_pts'], evaluation_row=combined,
            ordinary_finite_curve_reuse=group['ordinary_finite_curve_reuse']))
    return dict(original_entries=entries, groups=result, original_scalar_rows=scalar_cursor,
        original_points=scalar_cursor//2, source_points_removed=0,
        source_rows_before=len(entries), source_operator_calls_after=len(result),
        original_source_pack_or_loss_changed=False, solver_adapter_installed=False)


def evaluate_exact_source_curve_batches(prepared, cameras, evaluator, *, jacobian=True):
    """Evaluate grouped primitives and restore original scalar row order."""
    residual = np.empty(prepared['original_scalar_rows'], float)
    jac = None
    for group in prepared['groups']:
        output = evaluator(cameras[group['actual_pts']], [group['evaluation_row']], jacobian)
        if jacobian:
            value, block = output
            if jac is None:
                jac = np.empty((len(residual), block.shape[1]), block.dtype)
        else:
            value = output
        cursor = 0
        for lo, hi in group['original_scalar_ranges']:
            stop = cursor+hi-lo
            residual[lo:hi] = value[cursor:stop]
            if jacobian:
                jac[lo:hi] = block[cursor:stop]
            cursor = stop
        if cursor != len(value):
            raise ValueError('Grouped finite source operator changed scalar point count')
    if jacobian:
        return residual, jac
    return residual
