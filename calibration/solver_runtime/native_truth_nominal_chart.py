"""Replay the native human J's nominal query chart without changing guards."""
from __future__ import annotations
import numpy as np

_FIELDS=('nominal_finite_segments','nominal_finite_segment_categories',
         'nominal_source_support_faces','nominal_source_station_branches')

def capture_native_nominal_query_chart(truth,q):
    rows=truth.queries(q,full_stroke=True)
    records=truth.full_stroke_records
    if records is None or len(rows)!=len(records):raise ValueError('Complete original native truth groups required')
    contexts=[]
    for record,row in zip(records,rows):
        query=row['query']
        contexts.append(dict(record=record,**{name:None if query.get(name)is None else np.asarray(query[name]).copy()for name in _FIELDS}))
    return dict(contexts=contexts,nominal_carrier=np.concatenate([
        np.c_[r['query']['signed_normal_px'],r['query']['coverage_tangent_px']].ravel()for r in rows]),
        contract=dict(native_J_nominal_segment_category_face_station_formula=True,
            no_fixed_observed_world_correspondence_added=True,nonlinear_guard_uses_dynamic_query=True))

def native_nominal_query_carriers(truth,q,chart):
    from structured_programme_source_candidate.human_stroke_truth import finite_stroke_queries
    from structured_programme_source_candidate.native_camera_local import native_camera_local
    q=np.asarray(q,float)
    if(q.shape!=(truth.parameter_count,)or not np.isfinite(q).all()
        or np.any(q[truth.height_columns]<=0.)):raise ValueError('Original complete native truth physical chart required')
    truth.state.geometry.validate_parameters(q[truth.world_indices])
    core=q[:truth.state.structured_programme_coordinate_chart.core_count]
    cameras={p:b.bundle.camera(native_camera_local(b,core),p)for p,b in truth.blocks.items()}
    contexts=chart['contexts'];records=truth.full_stroke_records
    if len(contexts)!=len(records)or any(c['record']is not r for c,r in zip(contexts,records)):
        raise ValueError('Original nominal native source group identity changed')
    values=[]
    for context in contexts:
        record=context['record'];curve=record['row']['curve'];world=record['factor']._world(q,curve)
        query=finite_stroke_queries(cameras[int(record['row']['actual_pts'])],world,
            record['uv'],record['normals'],record['bounds'],tangent_bounds=record.get('tangent_bounds'),
            association='nearest_finite_segment'if curve['kind']in('board_top','ice_board_contact')else'normal_scanline',
            frozen_segments=context['nominal_finite_segments'],
            frozen_segment_categories=context['nominal_finite_segment_categories'],
            frozen_support_faces=context['nominal_source_support_faces'],
            frozen_station_branches=context['nominal_source_station_branches'])
        values.append(np.c_[query['signed_normal_px'],query['coverage_tangent_px']].ravel())
    return np.concatenate(values)
