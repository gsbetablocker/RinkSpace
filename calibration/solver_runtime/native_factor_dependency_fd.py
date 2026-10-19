"""Original native-source FD with declared per-exposure camera dependencies.

The complete nominal source is evaluated before this helper. Material columns
retain every row; only camera coordinates outside an original typed camera
chart have identically zero omitted rows. Original residual operators, physical
checks, FD widths, inward stencils and source row order remain authoritative.
"""
from __future__ import annotations
import time
import numpy as np


def native_factor_dependency_jacobian(factor, q, rows, active, material_columns,
                                      base, evaluate, height_columns):
    from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
    q = np.asarray(q, float)
    active = np.asarray(active, int)
    base = np.asarray(base, float)
    sizes = np.asarray([2 * len(row['uv']) for row in rows], int)
    offsets = np.r_[0, np.cumsum(sizes)]
    if base.shape != (int(offsets[-1]),):
        raise ValueError('Original complete native scalar row order required')
    dependencies = []
    for row in rows:
        pts = int(row['actual_pts'])
        block = factor.blocks[pts]
        dependencies.append(np.unique(np.r_[material_columns, block.xyz_slots,
            block.optics_chart.indices, block.optics_chart.mount_indices,
            block.start + 4 * block.bundle.index[pts] + np.arange(4)]).astype(int))
    compact = np.zeros((len(base), len(active)), float)
    contexts = []
    started = time.perf_counter()
    for ordinal, column in enumerate(active):
        groups = np.asarray([i for i, columns in enumerate(dependencies)
                             if int(column) in columns], int)
        if not len(groups):
            # Existing factors include cameras of HELD-only exposures. Their
            # columns stay present and have exactly zero FIT derivatives.
            contexts.append(dict(column=int(column), source_groups=0, evaluated_scalar_rows=0))
            continue
        indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in groups])
        h = factor.camera_steps.get(int(column), 1e-6)
        compact[indices, ordinal] = positive_pivot_coordinate_derivative(
            lambda value: evaluate(value, groups), q, int(column), h,
            height_columns, base=base[indices])
        contexts.append(dict(column=int(column), source_groups=int(len(groups)),
                             evaluated_scalar_rows=int(len(indices))))
    factor._source_dependency_fd_report = dict(
        schema='rinkspace.original-native-factor-camera-dependency-FD.v1',
        complete_scalar_rows=len(base), original_active_columns=len(active),
        original_sample_scalar_rows=len(base) * len(active),
        actual_sample_scalar_rows=sum(x['evaluated_scalar_rows'] for x in contexts),
        material_columns=np.asarray(material_columns, int).tolist(),
        contexts=contexts, seconds=time.perf_counter() - started,
        original_source_pixels_support_sigma_roles_FD_and_full_state_checks_preserved=True,
        accuracy_accepted=False)
    return compact
