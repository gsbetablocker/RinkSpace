"""Exact-stencil native human constraint FD with declared camera dependencies.

Every world column remains active for every group. Only camera columns whose
declared exposure/installation chart excludes the group are skipped. Each
actual sample still enters the original complete-q and physical-domain checks.
"""
from __future__ import annotations

import time
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix


def group_dependencies(truth, records):
    dependencies = []
    for item in records:
        actual = int(item['row']['actual_pts'])
        block = truth.blocks[actual]
        columns = np.r_[truth.world_indices, block.xyz_slots,
                        block.optics_chart.indices, block.optics_chart.mount_indices,
                        block.start + 4 * block.bundle.index[actual] + np.arange(4)].astype(int)
        dependencies.append(np.unique(columns))
    return dependencies


def dependent_carrier_jacobian(truth, q, *, full_stroke=False):
    from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
    from structured_programme_source_candidate.human_stroke_truth import finite_stroke_queries
    from structured_programme_source_candidate.native_camera_local import native_camera_local
    records = truth.full_stroke_records if full_stroke else truth.records
    if records is None:
        raise ValueError('Original complete stroke records required')
    dependencies = group_dependencies(truth, records)
    sizes = np.asarray([2 * len(row['uv']) for row in records], int)
    offsets = np.r_[0, np.cumsum(sizes)]
    # Original full evaluation authenticates every group before any omission.
    nominal_queries=truth.queries(q,full_stroke=full_stroke)
    base=np.concatenate([np.c_[row['query']['signed_normal_px'],row['query']['coverage_tangent_px']].ravel()
                         for row in nominal_queries])
    frozen_segments={id(record):result['query'].get('nominal_finite_segments')
                     for record,result in zip(records,nominal_queries)}
    frozen_categories={id(record):result['query'].get('nominal_finite_segment_categories')
                       for record,result in zip(records,nominal_queries)}
    frozen_faces={id(record):result['query'].get('nominal_source_support_faces')
                  for record,result in zip(records,nominal_queries)}
    frozen_stations={id(record):result['query'].get('nominal_source_station_branches')
                     for record,result in zip(records,nominal_queries)}
    compact = np.zeros((len(base), len(truth.active)), float)
    contexts = []
    query_caches = {id(record): OrderedDict() for record in records}
    source_keys = {id(record): tuple((np.asarray(record[name]).shape,
        np.asarray(record[name]).dtype.str, np.asarray(record[name]).tobytes())
        for name in ('uv', 'normals', 'bounds', 'tangent_bounds') if name in record)
        for record in records}
    base_keys = {}
    world_cache = {}
    import pickle
    world_curve_inputs = {id(record):pickle.dumps(record['row']['curve'],protocol=5)for record in records}
    memo_counts = dict(hits=0, misses=0)

    def memoized_queries(value, selected_records):
        # Literal original complete-state and world-domain checks. The finite
        # query is pure and its complete numeric inputs authenticate each hit.
        value = np.asarray(value, float)
        if value.shape != (truth.parameter_count,):
            raise ValueError('Complete shared source q required for human truth')
        if not np.isfinite(value).all() or np.any(value[truth.height_columns] <= 0.):
            raise ValueError('Human truth retains the existing strictly positive physical pivots')
        truth.state.geometry.validate_parameters(value[truth.world_indices])
        core = value[:truth.state.structured_programme_coordinate_chart.core_count]
        cameras = {p: b.bundle.camera(native_camera_local(b, core), p) for p, b in truth.blocks.items()}
        result = []
        for record in selected_records:
            camera = cameras[record['row']['actual_pts']]
            # Complete q and physical world admission ran above. A curve
            # producer has no camera inputs; reuse only exact world bytes and
            # its literal unchanged curve within this one FD construction.
            world_key=(id(record['factor']),world_curve_inputs[id(record)],value[truth.world_indices].tobytes())
            if world_key not in world_cache:
                world_cache[world_key]=record['factor']._world(value, record['row']['curve'])
            world = world_cache[world_key]
            key = (source_keys[id(record)], record['row']['curve']['kind'],
                   camera.shape, camera.dtype.str, camera.tobytes(),
                   world.shape, world.dtype.str, world.tobytes())
            cache = query_caches[id(record)]
            if key in cache:
                query = cache[key]
                cache.move_to_end(key)
                memo_counts['hits'] += 1
            else:
                query = finite_stroke_queries(camera, world, record['uv'], record['normals'], record['bounds'],
                    tangent_bounds=record.get('tangent_bounds'),
                    frozen_segments=frozen_segments[id(record)],
                    frozen_segment_categories=frozen_categories[id(record)],
                    frozen_support_faces=frozen_faces[id(record)],frozen_station_branches=frozen_stations[id(record)],
                    association='nearest_finite_segment' if record['row']['curve']['kind'] in
                    ('board_top', 'ice_board_contact') else 'normal_scanline')
                cache[key] = query
                if np.array_equal(value, q):
                    base_keys[id(record)] = key
                while len(cache) > 3:
                    discard = next(k for k in cache if k != base_keys.get(id(record)))
                    del cache[discard]
                memo_counts['misses'] += 1
            result.append(dict(id=record['row']['id'], pts=record['row']['pts'],
                               actual_pts=record['row']['actual_pts'], query=query))
        return result

    begin = time.perf_counter()
    # Keep exact base camera/material queries through all FD perturbations.
    memoized_queries(q, records)
    try:
        for ordinal, column in enumerate(truth.active):
            groups = [i for i, columns in enumerate(dependencies) if int(column) in columns]
            if not groups:
                raise ValueError('Declared original active column has no group dependency')
            selected_records = [records[i] for i in groups]
            row_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in groups])

            def carriers(value):
                rows = memoized_queries(value, selected_records)
                return np.concatenate([np.c_[r['query']['signed_normal_px'],
                                            r['query']['coverage_tangent_px']].ravel() for r in rows])

            tick = time.perf_counter()
            compact[row_indices, ordinal] = positive_pivot_coordinate_derivative(
                carriers, q, int(column), truth.steps.get(int(column), 1e-6),
                truth.height_columns, base=base[row_indices])
            contexts.append(dict(column=int(column), groups=groups,
                                 evaluated_scalar_rows=len(row_indices), seconds=time.perf_counter() - tick))
    finally:
        # Same final nominal camera/material restoration as the original J.
        truth.queries(q)
    rows, columns = np.nonzero(compact)
    truth._dependency_fd_report = dict(
        schema='rinkspace.declared-camera-dependency-human-FD.v1',
        complete_scalar_rows=len(base), original_active_columns=len(truth.active),
        original_full_sample_scalar_rows=len(base) * len(truth.active),
        actual_sample_scalar_rows=sum(x['evaluated_scalar_rows'] for x in contexts),
        full_world_dependencies_preserved=truth.world_indices.tolist(),
        source_pixels_bounds_rows_columns_and_stencils_changed=False,
        camera_dependencies_from_original_typed_chart=True,
        complete_sample_domain_checks_retained=True, contexts=contexts,
        nominal_finite_segment_and_endpoint_categories_replayed_for_local_derivative_only=True,
        exact_finite_query_input_byte_memoization=memo_counts,
        jacobian_seconds=time.perf_counter() - begin)
    return csr_matrix((compact[rows, columns], (rows, truth.active[columns])),
                      shape=(len(base), truth.parameter_count))

