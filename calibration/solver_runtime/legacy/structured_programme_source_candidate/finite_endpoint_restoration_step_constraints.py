"""Finite endpoint inward restoration branches for the QP, diagnostic draft.

No query, physical curve, source support, normal face, or nonlinear guard is
changed. For an authenticated open physical endpoint currently missing a
scanline, the inward coverage branch saturates at zero on entering the finite
primitive. When zero is in the original tangent support, its opposite linear
face is inactive on this branch. The complete original nonlinear query remains
the authority for accepting every trial.
"""
import numpy as np

def authenticated_open_curve(curve,world):
    """Use producer semantics, never a floating-point circle seam comparison."""
    kind=curve.get('kind')
    if kind=='advertisement_boundary':
        declared=curve.get('edge')in('left_vertical','right_vertical','top_edge','bottom_edge')
    else:declared=kind in('current_ice_segment','curling_centreline')
    world=np.asarray(world)
    return bool(declared and world.ndim==2 and world.shape[1:]==(3,)and len(world)>=2
        and np.any(world[-1]!=world[0]))

def endpoint_inward_restoration_constraints(truth, q, selected, *, full_stroke=True, group_limits=None, return_step_carrier_response=False):
    selected = np.asarray(selected, int)
    if not np.isin(truth.active, selected).all():
        raise ValueError('Human camera/world columns must be in explicit selected Schur chart')
    records = truth.full_stroke_records if full_stroke else truth.records
    if records is None:
        raise ValueError('All complete human strokes required')
    rows = truth.queries(q, full_stroke=full_stroke)
    carrier = np.concatenate([np.c_[r['query']['signed_normal_px'], r['query']['coverage_tangent_px']].ravel() for r in rows])
    J = truth.jacobian(q, carriers=True, full_stroke=full_stroke)[:, selected].toarray()
    bounds = np.vstack([r['bounds'] for r in records])
    tangent_bounds = np.vstack([r.get('tangent_bounds', np.c_[-np.max(abs(r['bounds']), axis=1), np.max(abs(r['bounds']), axis=1)]) for r in records])
    lower = np.c_[bounds[:, 0], tangent_bounds[:, 0]].ravel()
    upper = np.c_[bounds[:, 1], tangent_bounds[:, 1]].ravel()
    if group_limits is not None:
        limits = np.asarray(group_limits, float)
        if limits.shape != (len(records),) or np.any(limits < 0.0) or (not np.isfinite(limits).all()):
            raise ValueError('Existing nonlinear native group guard limits required')
        expansion = np.repeat(limits, [2 * len(row['uv']) for row in records])
        lower = lower - expansion
        upper = upper + expansion
    keep_lower = np.ones(len(carrier), bool)
    keep_upper = keep_lower.copy()
    inventory = []
    offset = 0
    for group_index, (record, result) in enumerate(zip(records, rows)):
        query = result['query']
        world = record['factor']._world(q, record['row']['curve'])
        source_support = record.get('tangent_bounds', np.c_[-np.max(abs(record['bounds']), axis=1), np.max(abs(record['bounds']), axis=1)])
        if (query.get('finite_source_association')!='original_support_rectangle_Linf'
                and authenticated_open_curve(record['row']['curve'], world)):
            segment = np.asarray(query['nearest_finite_segment_index'])
            station = np.asarray(query['nearest_finite_segment_station'])
            endpoint = (segment == 0) & (station == 0.0) | (segment == len(world) - 2) & (station == 1.0)
            eligible = ~np.asarray(query['finite_scanline_intersection'], bool) & endpoint & (source_support[:, 0] <= 0.0) & (source_support[:, 1] >= 0.0)
            for vertex in np.flatnonzero(eligible):
                ordinal = offset + 2 * int(vertex) + 1
                c = float(carrier[ordinal])
                if c < 0.0:
                    keep_upper[ordinal] = False
                    side = 'upper'
                    model = 'minimum(linear_carrier,0)'
                elif c > 0.0:
                    keep_lower[ordinal] = False
                    side = 'lower'
                    model = 'maximum(linear_carrier,0)'
                else:
                    continue
                inventory.append(dict(group=group_index, id=result['id'], actual_pts=result['actual_pts'], vertex=int(vertex), curve_kind=record['row']['curve']['kind'], original_carrier_row=ordinal, current_coverage_px=c, original_tangent_support=source_support[vertex].tolist(), segment=int(segment[vertex]), station=float(station[vertex]), inactive_opposite_step_face=side, inward_branch_carrier=model))
        offset += 2 * len(record['uv'])
    matrix = np.vstack((J, -J))
    target = np.r_[lower - carrier, carrier - upper]
    nonzero = np.any(matrix != 0.0, axis=1)
    if np.any(target[~nonzero] > 0.0):
        raise ValueError('Human support face is locally infeasible in declared chart')
    retained = nonzero & np.r_[keep_lower, keep_upper]
    result = dict(matrix=matrix[retained], lower=target[retained], report=dict(truth.contract, hard_halfspaces_in_native_source_pixels=True, finite_curve_nonlinear_guard_required=True, complete_guard_vertices_in_hard_halfspaces=full_stroke, extra_source_FIT_weights=0, endpoint_inward_restoration=dict(source_vertices_normal_faces_and_supports_unchanged=True, source_query_curve_and_nonlinear_guard_unchanged=True, original_nonzero_step_faces=int(nonzero.sum()), inward_branch_step_faces=int(retained.sum()), inactive_opposite_faces=int(np.sum(nonzero & ~retained)), branches=inventory)))
    if return_step_carrier_response:
        from .source_step_curvature_tightening import carrier_step_response
        result['step_carrier_response'] = carrier_step_response(carrier, retained)
    return result
