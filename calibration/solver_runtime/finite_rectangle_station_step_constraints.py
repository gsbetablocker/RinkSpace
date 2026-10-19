"""Finite-station local SQP faces on the unchanged full human rectangles.

Only current producer-certified boundary-clipped straight primitives and
whole vertical print borders enter this route. It changes a local model, never source pixels, support
arrays, the finite nonlinear query, an objective row or a physical parameter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .finite_rectangle_station_faces import finite_rectangle_station_halfspaces


def _current_boundary_clipped_primitive(record, q):
    curve=record['row']['curve'];factor=record['factor']
    if curve.get('kind')!='current_ice_segment':return None
    world=np.asarray(factor._world(q,curve),float)
    view=getattr(factor,'_ice_material_view',None)
    if view is None:return None
    p=factor.geometry.effective_world(np.asarray(q,float)[factor.world_indices])
    feature=view.features(p).get(curve['feature_id'],{})
    if (feature.get('kind')!='polyline'or feature.get('surface_id')!='ice_surface'
        or feature.get('structured_material_relation')!='rinkspace.direct-coupled-physical-hockey-curling-template-latent48.v1'
        or feature.get('boundary_rule')!='Tilted physical line extended and clipped to the fitted superellipse ice boundary'):
        return None
    segment=np.asarray(feature.get('segments_xyz'),float)
    if segment.shape!=(1,2,3)or curve.get('segment_index')!=0:return None
    if not np.array_equal(world,np.linspace(segment[0,0],segment[0,1],len(world))):
        raise ValueError('Finite station model must use the exact current registered whole primitive')
    from structured_programme_source_candidate.native_ice_curve_source import _boundary_hit
    course,cq,_=factor.geometry.print_course_binding(p)
    polygon=np.asarray(course.sample_perimeter(cq,with_jacobian=False)[0])
    direction=world[-1,:2]-world[0,:2];length=np.linalg.norm(direction)
    if not np.isfinite(world).all()or not np.isfinite(length)or length<=0.:
        raise ValueError('Current finite physical line must be nondegenerate')
    axis=direction/length;center=.5*(world[0,:2]+world[-1,:2])
    ends=np.stack((_boundary_hit(np.r_[center,0.],-np.r_[axis,0.],polygon),
                   _boundary_hit(np.r_[center,0.],np.r_[axis,0.],polygon)))
    # This is a coordinate arithmetic certificate, not a source pixel band.
    allowance=64*np.finfo(float).eps*max(1.,float(np.abs(world[[0,-1]]).max()),
        float(np.abs(polygon).max()))
    if ends.shape!=(2,3)or np.max(np.abs(ends-world[[0,-1]]))>allowance:
        raise ValueError('Registered finite primitive endpoints differ from the current ice boundary')
    return world


def _current_vertical_print_primitive(record,q):
    """Certify an existing whole physical vertical border, not a sample cell.

    The paired material producer fixes developed course station at constant
    u and uses zmin + v*height. Only its actual external left/right border is
    admitted; horizontal print edges and inferred ink contours keep their
    original operator. Arithmetic checks do not expand any pixel support.
    """
    curve=record['row']['curve'];factor=record['factor'];geometry=getattr(factor,'geometry',None)
    edge=curve.get('edge')
    if (curve.get('kind')!='advertisement_boundary' or edge not in ('left_vertical','right_vertical')
        or curve.get('extent_scope')!='external_border' or geometry is None
        or getattr(geometry,'paired_material_protocol_schema',None)!='rinkspace.same-size-paired-original-materials-independent-station-zmin.v1'):
        return None
    number=curve['advertisement_number'];key=edge+'_xyz'
    u,v=geometry._materials[number,key];u=np.asarray(u,float);v=np.asarray(v,float)
    if (u.ndim!=1 or v.shape!=u.shape or len(u)<2 or not np.isfinite(u).all()
        or not np.isfinite(v).all() or not np.all(u==u[0])):
        raise ValueError('Whole vertical print border requires one exact developed station')
    allowance=64*np.finfo(float).eps*max(1.,float(np.abs(v).max()),abs(float(u[0])))
    expected_u=0. if edge=='left_vertical'else 1.
    if (abs(float(u[0])-expected_u)>allowance or abs(float(v[0]))>allowance
        or abs(float(v[-1])-1.)>allowance or np.max(abs(v-np.linspace(0.,1.,len(v))))>allowance):
        raise ValueError('Vertical print samples must cover the original full lower-to-upper material border')
    world=np.asarray(factor._world(q,curve),float)
    if world.shape!=(len(u),3)or not np.isfinite(world).all():
        raise ValueError('Exact original complete vertical print primitive required')
    arithmetic=64*np.finfo(float).eps*max(1.,float(abs(world).max()))
    if (np.max(abs(world[:,:2]-world[0,:2]))>arithmetic
        or np.max(abs(world-np.linspace(world[0],world[-1],len(world))))>arithmetic
        or world[-1,2]<=world[0,2]):
        raise ValueError('Current whole print border must remain a nondegenerate vertical physical line')
    return world


def actual_station_model_carriers(truth,q,response,*,original_carriers=None):
    """Evaluate retained guard carriers and the same fixed-foot local chart.

    Local FM face values are not the trial's dynamic guard violation. The
    guard may use a different finite witness after a source branch changes.
    """
    base=(truth._carriers(q,full_stroke=True)if original_carriers is None else np.asarray(original_carriers,float))
    expected=response.get('original_native_carrier_count')
    if (base.ndim!=1 or not np.isfinite(base).all()or(expected is not None and len(base)!=expected)):
        raise ValueError('Complete finite original native carrier chart required')
    contexts=response.get('finite_station_contexts')
    if not contexts:return base
    from .finite_rectangle_projection_fd import projected_fixed_foot_carriers
    xy=projected_fixed_foot_carriers(truth,q,contexts)
    extra=[]
    for context,point in zip(contexts,xy):
        record=context['record'];index=context['source_vertex']
        normal=record['normals'][index];tangent=np.array([-normal[1],normal[0]])
        extra.extend((point-record['uv'][index])@np.stack((normal,tangent)).T)
    return np.r_[base,np.asarray(extra).ravel()]


def nominal_station_contexts(truth,q,*,full_queries=None):
    """Value-only original witness inventory; no truth or source J is built."""
    records=truth.full_stroke_records
    if records is None:raise ValueError('Original complete stroke records required')
    queries=truth.queries(q,full_stroke=True)if full_queries is None else full_queries
    if len(queries)!=len(records):raise ValueError('Complete original full query inventory required')
    contexts=[];offsets=np.r_[0,np.cumsum([2*len(r['uv'])for r in records])]
    for group,(record,result)in enumerate(zip(records,queries)):
        query=result['query']
        if query.get('finite_source_association')!='original_support_rectangle_Linf':continue
        vertical=_current_vertical_print_primitive(record,q)
        boundary=None if vertical is not None else _current_boundary_clipped_primitive(record,q)
        if vertical is None and boundary is None:continue
        world=vertical if vertical is not None else boundary
        segments=np.asarray(query['nearest_finite_segment_index'],int)
        stations=np.asarray(query['nearest_finite_segment_station'],float)
        categories=np.asarray(query['nominal_finite_segment_categories'],int)
        for vertex in range(len(record['uv'])):
            category=int(categories[vertex]);segment=int(segments[vertex])
            side=('lower'if segment==0 and category==1 else
                'upper'if segment==len(world)-2 and category==2 else None)
            original_station=float(stations[vertex])
            # The original selected-face/category semantics may classify an
            # endpoint one arithmetic ulp before t reaches literal 0 or 1.
            # Its local clamped foot is the true endpoint, while the original
            # query value remains unchanged and independently checked below.
            station=0. if side=='lower'else 1. if side=='upper'else original_station
            contexts.append(dict(record=record,group=group,source_vertex=int(vertex),
                segment=segment,station=station,original_nominal_segment_station=original_station,
                nominal_source_segment_category=category,nominal_physical_endpoint_side=side,
                whole_vertical_print=vertical is not None,whole_boundary_clipped_straight=boundary is not None,
                original_carrier_offset=int(offsets[group]+2*vertex)))
    return contexts


def _verify_nominal_face(face_value,actual_vertex_violation,*,context,provenance,operands):
    """Arithmetic consistency, never an added source-pixel tolerance."""
    scale=max(1.,float(np.sum(abs(np.asarray(operands,float)))))
    arithmetic=64.*np.finfo(float).eps*scale
    if face_value>actual_vertex_violation+arithmetic:
        record=context['record'];row=record['row']
        raise ValueError('Nominal finite-station face excludes its original support witness: '
            +str(dict(group=context['group'],id=row.get('id'),actual_pts=row['actual_pts'],
                source_vertex=context['source_vertex'],face=provenance,
                nominal_face_value=float(face_value),original_actual_vertex_violation=float(actual_vertex_violation),
                arithmetic_roundoff_bound=arithmetic,pixel_support_or_zero_tolerance_added=False)))
    return arithmetic


def finite_rectangle_station_constraints(truth,q,selected,declared_fixed=(),*,
        full_stroke=True,group_limits=None,endpoint_inward_restoration=False,
        return_step_carrier_response=False):
    """Use one station chart on each certified complete physical line."""
    from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart
    from .finite_rectangle_projection_fd import projected_station_contexts
    if not full_stroke:raise ValueError('Finite station route requires the original full strokes')
    selected=np.asarray(selected,int)
    base=constraints_on_declared_free_chart(truth,q,selected,declared_fixed,
        full_stroke=True,group_limits=group_limits,
        endpoint_inward_restoration=endpoint_inward_restoration,return_step_carrier_response=True)
    records=truth.full_stroke_records;queries=truth.queries(q,full_stroke=True)
    offsets=np.r_[0,np.cumsum([2*len(r['uv'])for r in records])]
    contexts=nominal_station_contexts(truth,q,full_queries=queries)
    if not contexts:return base
    limits=np.zeros(len(records))if group_limits is None else np.asarray(group_limits,float)
    if limits.shape!=(len(records),)or not np.isfinite(limits).all()or np.any(limits<0.):
        raise ValueError('Exact original nonnegative native group limits required')
    projections=projected_station_contexts(truth,q,contexts,selected)
    if isinstance(projections,list):
        projections={key:np.stack([item[key]for item in projections])for key in
            ('projected_foot_xy','fixed_foot_jacobian','station_screen_tangent','nominal_physical_station',
                'physical_endpoints_xy','endpoint_J')}
        projections['endpoint_jacobian']=projections.pop('endpoint_J')
    cone=np.tile([-np.inf,np.inf],(len(contexts),1))
    for point,context in enumerate(contexts):
        # A global world-line certificate does not certify an affine Brown
        # projection. Use only its local station tangent. Internal sampling
        # endpoints/knots never constrain this computational increment.
        side=context['nominal_physical_endpoint_side']
        if side=='lower':cone[point,0]=0.
        elif side=='upper':cone[point,1]=0.
    normal=np.stack([c['record']['normals'][c['source_vertex']]for c in contexts])
    uv=np.stack([c['record']['uv'][c['source_vertex']]for c in contexts])
    nb=np.stack([c['record']['bounds'][c['source_vertex']]for c in contexts])
    tb=np.stack([c['record']['tangent_bounds'][c['source_vertex']]for c in contexts])
    local_limits=limits[[c['group']for c in contexts]]
    model=finite_rectangle_station_halfspaces(projections['projected_foot_xy'],
        projections['fixed_foot_jacobian'],projections['station_screen_tangent'],
        projections['nominal_physical_station'],np.tile([0.,1.],(len(contexts),1)),
        uv,normal,nb,tb,local_limits,station_tangent_cone=cone)
    response=base['step_carrier_response'];carrier=np.asarray(response['nominal_carrier'],float)
    mask=np.asarray(response['retained_face_mask'],bool);count=len(carrier)
    original_columns=np.r_[np.arange(count),np.arange(count)][mask]
    original_signs=np.r_[np.ones(count),-np.ones(count)][mask]
    original_group=np.repeat(np.arange(len(records)),np.diff(offsets))
    owned=original_group[original_columns]
    substituted=np.array([c['original_carrier_offset']+axis for c in contexts for axis in (0,1)])
    kept=~np.isin(original_columns,substituted)
    matrices=[base['matrix'][kept]];targets=[base['lower'][kept]]
    groups=[owned[kept]];expansions=[limits[owned[kept]]]
    pr=list(np.arange(np.count_nonzero(kept)));pc=list(original_columns[kept]);pv=list(original_signs[kept])
    lower_rows=[];local_matrices=[];local_groups=[];local_provenance=[]
    nominal=actual_station_model_carriers(truth,q,dict(finite_station_contexts=contexts))
    row_cursor=len(pr)
    for source_face,provenance in enumerate(model['face_provenance']):
        point=provenance['source_point_index'];context=contexts[point]
        parents=provenance['original_source_faces'];kind=provenance['kind']
        if kind=='nonnegative_epigraph':continue # literal 0 <= nonnegative existing limit
        columns=[];coefficients=[];support=0.
        weights=provenance.get('source_face_weights',[1.])
        matrix=model['matrix'][source_face].copy()
        for weight,parent in zip(weights,parents):
            axis_index=0 if parent['axis']=='normal'else 1
            sign=-1. if parent['side']=='lower'else 1.
            bounds=nb if axis_index==0 else tb
            support+=weight*sign*bounds[point,0 if sign<0 else 1]
            column=count+2*point+axis_index
            columns.append(column);coefficients.append(weight*sign)
        lower=float(np.dot(coefficients,nominal[columns])-support-local_limits[point])
        original_query=queries[context['group']]['query']
        original_violation=float(np.max(abs(np.asarray(original_query['residual']).reshape(-1,2)[context['source_vertex']])))
        arithmetic=_verify_nominal_face(lower+local_limits[point],original_violation,
            context=context,provenance=provenance,
            operands=np.r_[np.asarray(coefficients)*nominal[columns],support,local_limits[point],
                context['record']['uv'][context['source_vertex']],projections['projected_foot_xy'][point]])
        model_value=float(provenance['local_epigraph_face_at_nominal'])
        if abs(lower+local_limits[point]-model_value)>arithmetic:
            raise ValueError('Finite-station nominal value and J/SOC chart differ: '+str(provenance))
        if not np.any(matrix):
            if lower>0.:raise ValueError('An exact constant original finite rectangle face is infeasible')
            continue
        local_matrices.append(matrix);lower_rows.append(lower);local_groups.append(context['group'])
        # A is the negative derivative of the upper epigraph face.
        pr.extend([row_cursor]*len(columns));pc.extend(columns);pv.extend([-x for x in coefficients]);row_cursor+=1
        local_provenance.append(dict(provenance,group=context['group'],source_vertex=context['source_vertex'],
            original_query_vertex_violation_px=original_violation,
            nominal_arithmetic_consistency_bound=arithmetic))
    if local_matrices:
        matrices.append(np.asarray(local_matrices));targets.append(np.asarray(lower_rows))
        groups.append(np.asarray(local_groups,int));expansions.append(limits[local_groups])
    matrix=np.vstack(matrices);lower=np.concatenate(targets)
    projection=csr_matrix((pv,(pr,pc)),shape=(len(lower),len(nominal)))
    report=dict(base['report'],finite_rectangle_station_local_model=dict(model['contract'],
        contexts=[dict(group=c['group'],source_vertex=c['source_vertex'],segment=c['segment'],station=c['station'],
            original_nominal_segment_station=c['original_nominal_segment_station'],
            nominal_source_segment_category=c['nominal_source_segment_category'],
            nominal_physical_endpoint_side=c['nominal_physical_endpoint_side'],
            whole_vertical_print=c.get('whole_vertical_print',False),
            whole_boundary_clipped_straight=c.get('whole_boundary_clipped_straight',False))for c in contexts],
        original_local_carrier_faces_replaced=int(np.count_nonzero(~kept)),
        eliminated_finite_faces_added=len(local_matrices),face_provenance=local_provenance,
        nominal_original_query_support_consistency_checked_for_every_local_face=True,
        nominal_consistency_uses_only_float_arithmetic_bound_not_pixel_tolerance=True,
        local_carrier_values_J_and_curvature_share_fixed_projected_foot=True,
        distant_physical_cap_projection_not_substituted_into_local_station_chart=True,
        primitive_dispatch='registered-current-boundary-clipped-straight-or-complete-vertical-print-border',
        whole_physical_primitives_use_all_original_vertices_and_local_station_tangent_cones=True,
        physical_endpoint_one_sided_station_release_only_when_nominal_is_true_global_endpoint=True,
        FM_local_face_values_are_not_dynamic_nonlinear_guard_violations=True,
        no_computational_station_or_source_FIT_parameter_added=True))
    result=dict(matrix=matrix,lower=lower,report=report)
    if return_step_carrier_response:
        result['step_carrier_response']=dict(nominal_carrier=nominal,
            row_carrier_projection=projection,row_group_index=np.concatenate(groups),
            row_limit_expansion=np.concatenate(expansions),finite_station_contexts=contexts,
            original_native_carrier_count=count,
            finite_station_carrier_contract=dict(basis='fixed_nominal_projected_segment_foot',
                local_model_only=True,original_dynamic_guard_is_independent=True,
                original_base_count=count,added_fixed_foot_axis_carriers=2*len(contexts)))
    return result


def station_model_implementation_sources():
    files=[Path(__file__),Path(__file__).with_name('finite_rectangle_station_faces.py'),
        Path(__file__).with_name('finite_rectangle_projection_fd.py')]
    return [dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in files]
