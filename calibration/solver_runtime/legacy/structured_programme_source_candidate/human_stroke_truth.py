"""Direct human stroke envelopes on the existing finite physical producers.

This factor is a feasibility interface, not an additional rink or solver.
Raster footprints and an explicitly recorded engineering location envelope
protect the user-drawn position. Source pixels query the actual finite projected
curve with its measured normal; they are not exact
two-coordinate point correspondences. The original source packets stay intact.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from rinkspace.vision_camera_initialization import unit_source_normals
from .native_camera_local import native_camera_local
from .fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative,positive_pivot_height_columns


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _visual_location_envelope(path,family,authority,ids):
    """A fixed engineering drift limit; it is not drawing precision."""
    path=Path(path).resolve();meta=json.loads(path.read_text(encoding='utf8'))
    if (meta.get('schema')!='rinkspace.fixed-human-visual-initializer-location-envelope.v1'
        or meta.get('user_rejected_subpixel_zero_band_refinement')is not True
        or meta.get('location_envelope_is_measured_statistical_precision')is not False
        or meta.get('fixed_for_subsequent_full_source_trials')is not True):
        raise ValueError('Explicit practical manual-location policy required')
    ref=meta['authority_references'][family]
    if Path(ref['path']).resolve()!=Path(authority['path']).resolve()or ref['sha256']!=authority['sha256']or _sha(ref['path'])!=ref['sha256']:
        raise ValueError('Manual location policy differs from actual human authority')
    anchor=meta['complete_initializer_reference']
    if _sha(anchor['path'])!=anchor['sha256']:raise ValueError('Visual initializer bytes changed')
    with np.load(anchor['path'],allow_pickle=False)as arrays:
        if hashlib.sha256(np.asarray(arrays['q'],dtype=np.float64).tobytes()).hexdigest()!=meta['canonical_q_sha256']:
            raise ValueError('Visual initializer canonical q identity changed')
    rows=[r for r in meta['records']if r['family']==family]
    if [r['id']for r in rows]!=list(ids):raise ValueError('Every actual manual stroke needs its own fixed location envelope')
    for row in rows:
        values=np.array([row['normal_expansion_px'],row['tangent_expansion_px']],float)
        if not np.isfinite(values).all()or np.any(values<0.):raise ValueError('Finite nonnegative manual location drift limits required')
    return rows,dict(path=str(path),sha256=_sha(path),canonical_visual_initializer=meta['canonical_q_sha256'],
        human_drawings_not_pixel_exact_physical_measurements=True,fixed_drift_envelope_not_source_sigma=True,
        old_soft_cost_cannot_exceed_this_location_envelope=True)


def _apply_location_envelope(item,policy):
    bounds=item['bounds'];radius=np.max(abs(bounds),axis=1)
    tangent=item.get('tangent_bounds',np.c_[-radius,radius])
    item['original_raster_normal_bounds']=bounds.copy();item['original_source_tangent_bounds']=tangent.copy()
    item['bounds']=bounds+np.array([-policy['normal_expansion_px'],policy['normal_expansion_px']])
    item['tangent_bounds']=tangent+np.array([-policy['tangent_expansion_px'],policy['tangent_expansion_px']])


def direct_manual_centre_factors(state,truth,board_packet,ice_packet,parameter_count,*,actual_pts=None):
    """Existing finite source operators target sparse hand-drawing centres.

    The separate HumanStrokeTruth guard retains the drawing/registration/hand
    uncertainty. Its support boundary is not the target of this secondary
    objective. New raw-sigma correlated traces use a mean, while historical
    traces retain the normalization already supplied by their producer.
    """
    from .native_board_edge_source import NativeBoardEdgeFactor
    from .native_ice_curve_source import NativeIceCurveFactor
    ids={item['row']['id']for item in truth.records}
    result=[]
    for factory,packet in ((NativeBoardEdgeFactor,board_packet),(NativeIceCurveFactor,ice_packet)):
        factor=factory(state,packet,parameter_count)
        rows=[row for row in factor.rows if row['id']in ids and(actual_pts is None or row['actual_pts']==actual_pts)]
        if not rows:continue
        for row in rows:
            row.pop('normal_interval',None)
            row['_source_optical_support']=None
        factor.rows=rows;factor.held=[]
        factor.blocks={row['actual_pts']:state.bypts[row['actual_pts']]for row in rows}
        factor.scales=np.concatenate([np.repeat(row['sigma'],2)for row in rows])
        factor.quadratic=np.ones(len(factor.scales),bool);factor.row_count=len(factor.scales);factor._cache.clear()
        factor._manual_centre_mean_scale=np.concatenate([np.full(2*len(row['uv']),
            np.sqrt(len(row['uv']))if row.get('practical_manual_location_uncertainty_native_px',0.)>0. else 1.)for row in rows])
        if isinstance(factor,NativeBoardEdgeFactor):
            factor.active=truth.active.copy();factor.active_world=truth.world_indices.copy()
            factor.contract['fit_ids']=[row['id']for row in rows];factor.contract['held_ids']=[]
        factor.contract.update(direct_manual_centre_secondary_objective=True,
            practical_support_retained_as_separate_guard=True,dense_guard_vertices_not_extra_FIT_rows=True,
            original_UV_and_raw_packet_sigma_not_modified=True,normal_optical_dead_zone_not_used_as_centre_target=True)
        result.append(factor)
    return result


def direct_manual_centre_block(factor,value,jacobian=True,*,original_start=0):
    block=factor.block(value,jacobian,original_start=original_start)
    scale=factor._manual_centre_mean_scale
    return SourceRowBlock(block.r/scale,None if block.J is None else block.J.multiply((1./scale)[:,None]).tocsr(),
        block.quadratic,block.scales*scale,dict(block.row_ledger,
            family='direct_manual_sparse_correlated_centre_secondary',retained_practical_hard_guard=True,
            correlated_trace_mean_normalized=True,unmodified_raw_source_sigma_and_UV=True))


class SourceWithDirectManualCentres:
    """Append the same centre operators to the owned canonical full source."""
    def __init__(self,parent,state,truth,board_packet,ice_packet,layout):
        self.parent=parent;self.layout=layout
        self.factors=direct_manual_centre_factors(state,truth,board_packet,ice_packet,parent.count)
        self.quadratic=np.r_[parent.quadratic,*[factor.quadratic for factor in self.factors]]
        self.scales=np.r_[parent.scales,*[factor.scales*factor._manual_centre_mean_scale for factor in self.factors]]
        self.contract=dict(parent.contract,direct_manual_centre_secondary=dict(
            authority=truth.contract['authority'],added_scalar_rows=sum(factor.row_count for factor in self.factors),
            source_UV_and_raw_sigma_unchanged=True,correlated_trace_mean=True,
            direct_manual_supports_remain_hard_guards=True,geometry_or_camera_parameters_added=0))
    def __getattr__(self,name):return getattr(self.parent,name)
    def blocks(self,value,jacobian=True):
        yield from self.parent.blocks(value,jacobian)
        evaluation=self.parent.evaluation_q(value);offset=len(self.parent.quadratic)
        for factor in self.factors:
            block=direct_manual_centre_block(factor,evaluation,jacobian,original_start=offset)
            if block.J is not None:
                block=SourceRowBlock(block.r,block.J[:,self.layout.canonical_to_evaluation].tocsr(),
                    block.quadratic,block.scales,block.row_ledger)
            yield block;offset+=len(block.r)


def human_support_operator_contract():
    from .normal_scanline_human_query_acceleration import RECTANGLE_OPERATOR
    files=[Path(__file__),Path(__file__).with_name('human_constraint_dependency_fd.py'),
        Path(__file__).with_name('normal_scanline_human_query_acceleration.py'),
        Path(__file__).with_name('native')/'normal_scanline_human_query.c',
        Path(__file__).with_name('finite_endpoint_restoration_step_constraints.py')]
    return dict(schema=RECTANGLE_OPERATOR,normal_and_tangent_support_arrays_unchanged=True,
        all_original_positive_central_finite_segments_retained=True,
        board_nearest_finite_query_unchanged=True,nonlinear_guard_uses_dynamic_minimum=True,
        derivative_replays_only_nominal_original_finite_segment=True,
        derivative_replays_nominal_finite_endpoint_category=True,
        finite_endpoint_categories_from_exact_original_active_support_faces=True,
        derivative_replays_nominal_rectangle_support_and_station_active_faces=True,
        old_query_caps_require_explicit_same_q_semantic_migration=True,
        implementation_sources=[dict(path=str(p.resolve()),sha256=_sha(p))for p in files])


def finite_stroke_queries(camera,world,uv,normals,bounds,*,association='normal_scanline',tangent_bounds=None,frozen_segments=None,frozen_segment_categories=None,frozen_support_faces=None,frozen_station_branches=None):
    """Declared finite curve against the original 2D source support rectangle.

    Positive central endpoints and midpoints are required for each segment,
    exactly as in the retained finite-curve observation operator. Disconnected
    optical components are never joined. The nonboard route minimizes the
    original maximum outside-normal/tangent support over all finite segments.
    A remote normal-ray tangency cannot override a nearer source support point.
    """
    world=np.asarray(world,float);uv=np.asarray(uv,float);normals=np.asarray(normals,float)
    projected,depth,central=project_central_valid_full_brown(camera,world)
    midpoint=project_central_valid_full_brown(camera,.5*(world[:-1]+world[1:]))[-1]
    if association=='nearest_finite_segment':
        from .finite_human_query_acceleration import nearest_finite_human_projected_queries
        return nearest_finite_human_projected_queries(projected,depth,central,midpoint,uv,normals,bounds,
            tangent_bounds=tangent_bounds,world_closed=np.array_equal(world[0],world[-1]))
    if association=='normal_scanline':
        from .normal_scanline_human_query_acceleration import finite_source_rectangle_projected_queries
        return finite_source_rectangle_projected_queries(projected,depth,central,midpoint,
            uv,normals,bounds,tangent_bounds=tangent_bounds,frozen_segments=frozen_segments,
            frozen_segment_categories=frozen_segment_categories,frozen_support_faces=frozen_support_faces,
            frozen_station_branches=frozen_station_branches)
    start,delta=projected[:-1],np.diff(projected,axis=0)
    length2=np.einsum('ni,ni->n',delta,delta)
    valid=(central[:-1]&central[1:]&midpoint&np.isfinite(start).all(axis=1)&
           np.isfinite(delta).all(axis=1)&(depth[:-1]>0)&(depth[1:]>0)&(length2>1e-4))
    if not np.any(valid):raise ValueError('Human finite physical primitive has no positive central segment')
    ids=np.flatnonzero(valid);a=start[ids];d=delta[ids];l2=length2[ids]
    difference=uv[:,None,:]-a[None,:,:]
    determinant=_cross(d[None,:,:],normals[:,None,:])
    nonparallel=np.abs(determinant)>32*np.finfo(float).eps*np.sqrt(l2)[None,:]
    station=np.divide(_cross(difference,normals[:,None,:]),determinant,
                      out=np.full(determinant.shape,np.nan),where=nonparallel)
    signed=np.divide(_cross(difference,d[None,:,:]),determinant,
                     out=np.full(determinant.shape,np.nan),where=nonparallel)
    intersections=nonparallel&(station>=0.)&(station<=1.)
    chosen=np.argmin(np.where(intersections,np.abs(signed),np.inf),axis=1)
    coverage=intersections.any(axis=1);index=np.arange(len(uv))
    selected_signed=signed[index,chosen]
    selected_station=(ids[chosen]+station[index,chosen])/(len(world)-1)
    tangent=np.c_[-normals[:,1],normals[:,0]]
    # This is the same nearest finite projected segment, with clamped ends,
    # used by the source operator. It drives a missing finite coverage query
    # toward its actual primitive instead of an infinite supporting line.
    nearest_station=np.clip(np.einsum('nmi,mi->nm',difference,d)/l2[None,:],0.,1.)
    nearest=a[None,:,:]+nearest_station[:,:,None]*d[None,:,:]
    vectors=nearest-uv[:,None,:]
    closest=np.argmin(np.einsum('nmi,nmi->nm',vectors,vectors),axis=1)
    vector=vectors[index,closest]
    fallback_signed=np.einsum('ni,ni->n',vector,normals)
    fallback_tangent=np.einsum('ni,ni->n',vector,tangent)
    if association=='nearest_finite_segment':
        # Original wide sources use nearest finite projected curve feet.
        # A normal ray can hit the opposite side of a closed course while
        # the actual nearest foot is still on the visible local material.
        # Internal knots are not observed physical endpoints. Only an end
        # of an actual valid optical component may need cap coverage.
        segment=ids[closest];s=nearest_station[index,closest]
        previous=np.zeros(len(valid),bool);following=previous.copy()
        previous[1:]=valid[:-1];following[:-1]=valid[1:]
        if np.array_equal(world[0],world[-1]):previous[0]=valid[-1];following[-1]=valid[0]
        coverage=((s>0.)&(s<1.))|((s==0.)&previous[segment])|((s==1.)&following[segment])
        selected_signed=fallback_signed
        selected_station=(segment+s)/(len(world)-1)
    elif association!='normal_scanline':raise ValueError('Declared existing finite source association required')
    selected_signed=np.where(coverage,selected_signed,fallback_signed)
    selected_station=np.where(coverage,selected_station,(ids[closest]+nearest_station[index,closest])/(len(world)-1))
    radius=np.max(np.abs(bounds),axis=1)
    tangent_bounds=np.c_[-radius,radius]if tangent_bounds is None else np.asarray(tangent_bounds,float)
    if tangent_bounds.shape!=bounds.shape or not np.isfinite(tangent_bounds).all()or np.any(tangent_bounds[:,0]>tangent_bounds[:,1]):
        raise ValueError('Finite authenticated drawing/context tangent envelope required')
    coverage_raw=np.where(coverage,0.,fallback_tangent)
    normal_violation=selected_signed-np.clip(selected_signed,bounds[:,0],bounds[:,1])
    coverage_violation=coverage_raw-np.clip(coverage_raw,tangent_bounds[:,0],tangent_bounds[:,1])
    violation=np.c_[normal_violation,coverage_violation].ravel()
    if not np.isfinite(violation).all():raise ValueError('Nonfinite direct human stroke query')
    return dict(residual=violation,signed_normal_px=selected_signed,coverage_tangent_px=coverage_raw,
                finite_scanline_intersection=coverage,finite_primitive_station=selected_station,
                normal_violation_px=normal_violation,coverage_violation_px=coverage_violation,
                nearest_finite_vector_px=vector,nearest_finite_segment_index=ids[closest],
                nearest_finite_segment_station=nearest_station[index,closest],
                nearest_finite_signed_normal_px=fallback_signed,nearest_finite_tangent_px=fallback_tangent,
                finite_source_association=association,
                valid_projected_segment_count=len(ids),endpoint_coverage_allowance_px=radius)


def _truth_record_input_bytes(value):
    if isinstance(value,np.ndarray):return ('array',value.shape,value.dtype.str,value.tobytes())
    if isinstance(value,np.generic):return _truth_record_input_bytes(np.asarray(value))
    if isinstance(value,dict):return ('dict',tuple((k,_truth_record_input_bytes(v))for k,v in sorted(value.items())))
    if isinstance(value,(list,tuple)):return ('sequence',tuple(_truth_record_input_bytes(v)for v in value))
    if value is None or isinstance(value,(str,int,float,bool)):return (type(value).__name__,value)
    if hasattr(value,'normals')and hasattr(value,'bounds'):
        return (type(value).__module__,type(value).__name__,_truth_record_input_bytes(value.normals),_truth_record_input_bytes(value.bounds))
    raise TypeError('Unrecognized source input cannot enter exact truth memo: '+type(value).__name__)

def _truth_record_memo_nominal(owner,q):
    owner._truth_record_memo_nominal_q=np.asarray(q,float).copy()
    owner._truth_record_memo_base_keys={}

def _truth_record_memo(owner,item,key,q,evaluate):
    from collections import OrderedDict
    memo=getattr(owner,'_truth_record_memo_cache',None)
    if memo is None:owner._truth_record_memo_cache=memo={}
    cache=memo.setdefault(id(item),OrderedDict())
    counts=getattr(owner,'_truth_record_memo_counts',None)
    if counts is None:owner._truth_record_memo_counts=counts=dict(hits=0,misses=0)
    basekeys=getattr(owner,'_truth_record_memo_base_keys',{})
    nominal=getattr(owner,'_truth_record_memo_nominal_q',None)
    if nominal is not None and np.array_equal(q,nominal):basekeys[id(item)]=key
    if key in cache:
        result=cache[key];cache.move_to_end(key);counts['hits']+=1
    else:
        result=evaluate();cache[key]=result;counts['misses']+=1
        while len(cache)>3:
            discard=next(k for k in cache if k!=basekeys.get(id(item)))
            del cache[discard]
    # The public producer never exposes mutable memo arrays/details.
    def clone(v):
        if isinstance(v,np.ndarray):return v.copy()
        if isinstance(v,dict):return {k:clone(x)for k,x in v.items()}
        if isinstance(v,tuple):return tuple(clone(x)for x in v)
        if isinstance(v,list):return [clone(x)for x in v]
        return v
    return clone(result)

def _confirmed_record_memo_key(owner,q,item):
    # Preserve ALL world/material columns, even when one particular row does
    # not use them. Only unrelated exposure camera columns can be omitted.
    inventory=(owner.active.tobytes(),owner.world_indices.tobytes(),tuple(sorted(owner.blocks)))
    if getattr(owner,'_truth_record_memo_dependency_inventory',None)!=inventory:
        cameras={actual:np.unique(np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,
            block.start+4*block.bundle.index[actual]+np.arange(4)]).astype(int)for actual,block in owner.blocks.items()}
        material=np.setdiff1d(owner.active,np.unique(np.concatenate(list(cameras.values()))))
        owner._truth_record_memo_dependency_columns={actual:np.unique(np.r_[owner.world_indices,material,columns]).astype(int)
            for actual,columns in cameras.items()}
        owner._truth_record_memo_dependency_inventory=inventory
    columns=owner._truth_record_memo_dependency_columns[item['row']['actual_pts']]
    source=dict(row=item['row'],radius=item['radius'],normals=item['normals'],bounds=item['bounds'],
        original_support_operator=item.get('original_support_operator'))
    # Source rows retain large historical trace metadata. Serialize their
    # complete in-memory inputs in C rather than recursively hashing many
    # thousands of Python tuple nodes on every FD call. This is a literal
    # byte key only; nothing is deserialized or persisted as a pickle.
    import pickle
    return ('confirmed-carrier',columns.tobytes(),q[columns].tobytes(),pickle.dumps(source,protocol=5))


def _confirmed_pure_query_memo(owner,item,q,camera,world_or_derived,evaluate):
    """Memoize only an actual projection/query input, never a world-domain check."""
    import pickle
    source=dict(row=item['row'],radius=item['radius'],normals=item['normals'],bounds=item['bounds'],
        original_support_operator=item.get('original_support_operator'))
    actual_world_input=(_truth_record_input_bytes(world_or_derived)if isinstance(world_or_derived,np.ndarray)
        else ('literal-derived-rows',pickle.dumps(world_or_derived,protocol=5)))
    key=('confirmed-pure-query',_truth_record_input_bytes(np.asarray(camera)),
        actual_world_input,pickle.dumps(source,protocol=5))
    return _truth_record_memo(owner,item,key,q,evaluate)


class UserConfirmedSourceTruth:
    """Protect explicitly confirmed native measurements on their original operator.

    RGB readings are not registered drawings. This separate authority binds the
    unchanged original row and finite curve, and adds no independent FIT weight.
    """
    def __init__(self,state,authority_path,board_factor,ice_factor,*,parameter_count,
                 circle_factor=None,material_factor=None,placement_source=None):
        self.state=state;self.path=Path(authority_path).resolve()
        authority=json.loads(self.path.read_text(encoding='utf8'))
        if authority.get('schema')!='rinkspace.user-confirmed-native-source-priority-authorities.v1':
            raise ValueError('Explicit user-confirmed native source authority required')
        self.parameter_count=int(parameter_count)
        self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        self.records=[];self.blocks={};self.steps={};active=list(self.world_indices);ids=set()
        self.placement_source=placement_source
        factors={'native_board_edge':board_factor,'native_ice_curve':ice_factor,
                 'native_board_circle':circle_factor,'native_board_material_point':material_factor}
        packets={}
        for record in authority['records']:
            if (record.get('user_confirmed_physical_identity')is not True
                or record.get('model_projection_used_to_generate_or_select_uv')is not False
                or record.get('source_sigma_or_UV_changed')is not False):
                raise ValueError('Confirmed physical identity and unchanged native measurements required')
            factor=factors[record['family']]
            if factor is None:raise ValueError('Confirmed original source family is not present in the complete model')
            ref=record['packet_reference'];path=Path(ref['path']).resolve()
            if path!=factor.packet_path or _sha(path)!=ref['sha256']:
                raise ValueError('Confirmed source packet identity differs')
            if path not in packets:packets[path]=json.loads(path.read_text(encoding='utf8'))
            original=packets[path]['rows'][int(record['original_row_index'])]
            raw=json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
            source_id=record.get('source_row_id',record['id'])
            companion=source_id!=record['id']
            if companion and (record.get('raw_support_companion_of')!=source_id
                or record['id']!=source_id+'__original_optical_support_companion'
                or source_id not in ids
                or record.get('carrier_binding')!='unchanged_original_source_support_v1'
                or record.get('source_support_unchanged')is not True):
                raise ValueError('Original FIT support companion must preserve its existing parent guard and identity')
            if original['id']!=source_id or record['id']in ids or hashlib.sha256(raw).hexdigest()!=record['source_row_canonical_json_sha256']:
                raise ValueError('Confirmed original row metadata/identity differs')
            ids.add(record['id'])
            row=next((r for r in factor.rows if r['id']==source_id),None)
            if row is None or row['curve']!=record['curve']or row['pts']!=record['pts']or row['actual_pts']!=record['actual_pts']:
                raise ValueError('Only unchanged actual FIT source rows may be confirmed')
            for field in ('uv','sigma_px'):
                if not np.array_equal(np.asarray(record[field],float),np.asarray(original[field],float)):
                    raise ValueError('Confirmed UV/raw sigma differs from immutable source')
            frame=Path(record['source_raw_frame']).resolve()
            if frame!=Path(original['source_raw_frame']).resolve()or record['source_file_sha256']!=original['source_file_sha256']or _sha(frame)!=record['source_file_sha256']:
                raise ValueError('Confirmed raw frame identity differs')
            for key in ('user_confirmation_reference','source_measurement_reference'):
                ref=record[key]
                if _sha(ref['path'])!=ref['sha256']:raise ValueError('Confirmed human/RGB measurement evidence changed')
            radius=np.broadcast_to(np.asarray(record['practical_location_uncertainty_px'],float),(len(row['uv']),)).copy()
            if not np.isfinite(radius).all()or np.any(radius<=0.):
                raise ValueError('Finite positive engineering measurement uncertainty required')
            if record.get('practical_uncertainty_is_user_pixel_precision')is not False:
                raise ValueError('Engineering uncertainty must not be claimed as user pixel precision')
            normals=bounds=None
            if 'normal' in original:
                normals,_=unit_source_normals(np.asarray(original['normal'],float),len(row['uv']))
                bounds=np.asarray(original['normal_interval_px'],float)+np.c_[-radius,radius]
            block=factor.blocks[row['actual_pts']];self.blocks[row['actual_pts']]=block
            cols=np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,
                       block.start+4*block.bundle.index[row['actual_pts']]+np.arange(4)].astype(int)
            active.extend(cols)
            for col in cols:self.steps[int(col)]=factor.camera_steps.get(int(col),1e-6)
            # Alias only this private guard record; never mutate original FIT rows.
            if companion:row=dict(row,id=record['id'],original_source_row_id=source_id)
            item=dict(record=record,factor=factor,row=row,radius=radius,normals=normals,bounds=bounds)
            if record.get('carrier_binding')=='unchanged_original_source_support_v1':
                self._bind_original_support(item,original,active)
            self.records.append(item)
        if not self.records:raise ValueError('At least one explicitly confirmed source required')
        self.active=np.unique(active).astype(int)
        self.height_columns=positive_pivot_height_columns(self.blocks.values())
        self.contract=dict(schema='rinkspace.user-confirmed-existing-native-source-guard.v1',
            authority=dict(path=str(self.path),sha256=_sha(self.path)),ids=[r['row']['id']for r in self.records],
            source_UV_sigma_roles_unchanged=True,registered_drawing_provenance_not_claimed=True,
            finite_source_operator_unchanged=True,extra_independent_FIT_weight_rows=0,
            engineering_uncertainty_not_user_pixel_precision=True,accuracy_accepted=False,
            human_truth_support_operator=human_support_operator_contract())


    def _bind_original_support(self,item,original,active):
        record,row,factor=item['record'],item['row'],item['factor']
        if record.get('source_support_unchanged') is not True:
            raise ValueError('The original actual source support must remain unchanged')
        support={key:original[key]for key in ('normal','normal_interval_px','point_support_bounds_xy_px')if key in original}
        if record.get('original_source_support')!=support:
            raise ValueError('Confirmed support differs from the immutable native source')
        operator=record['original_source_operator'];family=record['family'];kind=row['curve']['kind']
        if operator in ('printed_component_horizontal_extent','original_complete_curve_horizontal_extent'):
            admitted=((operator=='printed_component_horizontal_extent'and family=='native_board_edge'and kind=='printed_component_contour')
                or(operator=='original_complete_curve_horizontal_extent'and family=='native_ice_curve'and kind=='house_ring'))
            if (not admitted or row.get('curve_observation_kind')!='horizontal_extent'
                or not np.array_equal(np.asarray(original['normal'],float),np.array([[1.,0.]]))
                or len(row['uv'])!=1):
                raise ValueError('Existing source-qualified complete-curve horizontal extent required')
            if operator=='original_complete_curve_horizontal_extent'and(row.get('_source_optical_support')is None
                or original.get('point_correspondence')is not False or original.get('endpoint_observation')is not False):
                raise ValueError('Original house-ring optical support and no point/endpoint correspondence required')
            size=len(row['uv'])
        elif operator=='original_board_nearest_finite_normal_interval':
            if (family!='native_board_edge' or kind!='advertisement_boundary'
                or row.get('curve_observation_kind')!='normal_interval'
                or 'normal_interval_px'not in original):
                raise ValueError('Existing source-observed finite advertisement boundary required')
            size=len(row['uv'])
        elif operator=='original_nearest_finite_normal_interval':
            if ((family,kind)not in (('native_board_circle','board_circle'),
                ('native_board_material_point','shared_board_pattern_circle_curve'))
                or row.get('_source_optical_support') is None):
                raise ValueError('Existing admitted circle nearest-finite normal support required')
            size=len(row['uv'])
        elif operator=='original_named_point_XY_box':
            if (family!='native_board_material_point' or kind!='shared_board_pattern_material_points'
                or row.get('source_point_bounds_authorized')is not True):
                raise ValueError('Existing named material point source XY support required')
            size=2*len(row['uv'])
        else:raise ValueError('Unknown original source support carrier')
        item.update(original_support_operator=operator,carrier_size=size,expand_lower_faces=True)
        active.extend(np.asarray(factor.world_indices if family=='native_ice_curve'else factor.active_world,int).tolist())
        if family=='native_board_circle':active.extend(np.asarray(factor.active_circle_indices,int).tolist())
        if family=='native_board_material_point':active.extend(np.asarray(factor.pattern_columns,int).tolist())
        if family=='native_board_material_point':
            if self.placement_source is None:
                raise ValueError('Original independent material placement context is required')
            active.extend(np.asarray(self.placement_source.placement_indices,int).tolist())

    def _original_support_carrier(self,q,item,*,factor_cameras=None):
        from contextlib import nullcontext
        from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
        row,factor=item['row'],item['factor'];operator=item['original_support_operator']
        prefix=np.asarray(q,float)[:factor.parameter_count]
        context=(self.placement_source.pattern_placement_context(q)
                 if item['record']['family']=='native_board_material_point'else nullcontext())
        with context:
            if factor_cameras is None:
                all_cameras=factor._cameras(prefix)
            else:
                # Each unchanged-q factor still validates/builds ALL its cameras.
                # Other records may reuse that exact result within this one sample.
                key=id(factor)
                if key not in factor_cameras:factor_cameras[key]=factor._cameras(prefix)
                all_cameras=factor_cameras[key]
            camera=all_cameras[row['actual_pts']]
            if operator in ('printed_component_horizontal_extent','original_complete_curve_horizontal_extent'):
                from z190_source_camera_diagnostic import horizontal_extent_value_jac15
                world=factor._world(prefix,row['curve'])
                value=_confirmed_pure_query_memo(self,item,q,camera,world,
                    lambda:horizontal_extent_value_jac15(camera,dict(row,world=world),False)[::2])
                bounds=(row['_source_optical_support'].bounds if operator=='original_complete_curve_horizontal_extent'
                        else np.asarray(row['normal_interval'],float))
                return value,bounds[:,0],bounds[:,1]
            if operator=='original_board_nearest_finite_normal_interval':
                normals,_=unit_source_normals(np.asarray(row['normal'],float),len(row['uv']))
                bounds=np.asarray(row['normal_interval'],float)
                # Reuse the exact original finite-foot source producer. Its
                # tangent/endpoint errors stay in the unchanged soft FIT;
                # there is no independently measured tangent/cap hard band.
                from z190_source_camera_diagnostic import absolute_value_jac15
                # Use the public producer's exact derived rows and original raw
                # support-off operation. Material/domain construction always runs.
                derived=factor._source_rows(prefix,row,camera)
                derived=[{k:v for k,v in part.items()if k!='normal_interval'}for part in derived]
                def original_raw():
                    residual=absolute_value_jac15(camera,derived,False)
                    if not np.isfinite(residual).all():raise ValueError('Additional board source has invalid projection')
                    return residual*np.repeat(row['sigma'],2)
                raw=_confirmed_pure_query_memo(self,item,q,camera,derived,original_raw)
                value=np.einsum('ni,ni->n',raw.reshape(-1,2),normals)
                return value,bounds[:,0],bounds[:,1]
            world=(factor.world(prefix,row['advertisement_number'])
                   if item['record']['family']=='native_board_circle'else factor._world(prefix,row))
            if operator=='original_named_point_XY_box':
                from z190_shared_board_pattern_source import _project
                value=_confirmed_pure_query_memo(self,item,q,camera,world,
                    lambda:(_project(camera,world,False)[0]-row['uv']).ravel())
                bounds=row['_source_point_bounds']-row['uv'][:,:,None]
                return value,bounds[:,:,0].ravel(),bounds[:,:,1].ravel()
            raw=_confirmed_pure_query_memo(self,item,q,camera,world,
                lambda:curve_vector_error_full_brown(camera,world,row['uv'],with_jacobian=False))
            if isinstance(raw,tuple):raw=raw[0]
            support=row['_source_optical_support']
            value=np.einsum('ni,ni->n',np.asarray(raw,float).reshape(-1,2),support.normals)
            return value,support.bounds[:,0],support.bounds[:,1]

    def _one_record_carrier(self,q,item,cameras,*,factor_cameras=None):
        from .native_ice_curve_source import curve_vector_error_full_brown
        values=[];lower=[];upper=[];details=[]
        row=item['row']
        if 'original_support_operator'in item:
            value,lo,hi=self._original_support_carrier(q,item,factor_cameras=factor_cameras)
            values.append(value);lower.append(lo);upper.append(hi)
            violation=value-np.clip(value,lo,hi)
            details.append(dict(id=row['id'],pts=row['pts'],actual_pts=row['actual_pts'],
                maximum_violation_px=float(np.max(abs(violation))),finite_source_query=item['original_support_operator'],
                maximum_carrier_magnitude_px=float(np.max(abs(value))),original_actual_source_support_unchanged=True))
            return values[0],lower[0],upper[0],details[0]
        camera=cameras[row['actual_pts']];world=item['factor']._world(q,row['curve'])
        horizontal_extent=row.get('curve_observation_kind')=='horizontal_extent'and row['curve'].get('kind')=='house_ring'
        if horizontal_extent:
            from z190_source_camera_diagnostic import horizontal_extent_value_jac15
            value=_confirmed_pure_query_memo(self,item,q,camera,world,
                lambda:horizontal_extent_value_jac15(camera,dict(row,world=world),False))
            if item['normals']is None or not np.array_equal(item['normals'],np.array([[1.,0.]])):
                raise ValueError('Confirmed whole-ring extent needs its actual horizontal source support')
            lo=np.c_[item['bounds'][:,0],np.zeros(len(row['uv']))].ravel()
            hi=np.c_[item['bounds'][:,1],np.zeros(len(row['uv']))].ravel()
            query=dict(complete_curve_horizontal_extent_offset_xy_px=value)
        elif item['normals']is not None:
            query=_confirmed_pure_query_memo(self,item,q,camera,world,
                lambda:finite_stroke_queries(camera,world,row['uv'],item['normals'],item['bounds'],
                    tangent_bounds=np.c_[-item['radius'],item['radius']],association='nearest_finite_segment'))
            value=np.c_[query['signed_normal_px'],query['coverage_tangent_px']].ravel()
            lo=np.c_[item['bounds'][:,0],-item['radius']].ravel()
            hi=np.c_[item['bounds'][:,1],item['radius']].ravel()
        else:
            # An original ring reading has no measured source normal.
            # Keep its full-Brown nearest finite curve distance, rather
            # than inventing a tangent or a physical endpoint observation.
            vector=_confirmed_pure_query_memo(self,item,q,camera,world,
                lambda:curve_vector_error_full_brown(camera,world,row['uv']))
            if isinstance(vector,tuple):vector=vector[0]
            value=np.linalg.norm(np.asarray(vector,float).reshape(-1,2),axis=1)
            lo=np.zeros(len(value));hi=item['radius'];query=dict(nearest_finite_distance_px=value)
        values.append(value);lower.append(lo);upper.append(hi)
        violation=value-np.clip(value,lo,hi)
        details.append(dict(id=row['id'],pts=row['pts'],actual_pts=row['actual_pts'],
            maximum_violation_px=float(np.max(abs(violation))),
            finite_source_query='complete_curve_horizontal_extent'if horizontal_extent else'measured_normal_and_finite_coverage'if item['normals']is not None else'original_nearest_finite_curve_distance',
            maximum_carrier_magnitude_px=float(np.max(abs(value)))))
        return values[0],lower[0],upper[0],details[0]

    def _carriers(self,q,*,record_memo=None):
        from .native_ice_curve_source import curve_vector_error_full_brown
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all()or np.any(q[self.height_columns]<=0.):
            raise ValueError('Existing complete physical q required for confirmed source guard')
        self.state.geometry.validate_parameters(q[self.world_indices])
        core=q[:self.state.structured_programme_coordinate_chart.core_count]
        cameras={p:b.bundle.camera(native_camera_local(b,core),p)for p,b in self.blocks.items()}
        values=[];lower=[];upper=[];details=[];factor_cameras={}
        if record_memo is not None:
            # Retain every original factor camera admission on each complete
            # FD sample, even when an unrelated exposure admits a pure hit.
            for item in self.records:
                if 'original_support_operator' in item:
                    factor=item['factor'];key=id(factor)
                    if key not in factor_cameras:
                        factor_cameras[key]=factor._cameras(q[:factor.parameter_count])
        for item in self.records:
            key=None if record_memo is None else _confirmed_record_memo_key(self,q,item)
            cached=None if key is None else record_memo.get((id(item),key))
            if cached is None:
                cached=self._one_record_carrier(q,item,cameras,factor_cameras=factor_cameras)
                if key is not None:record_memo[(id(item),key)]=cached
            value,lo,hi,detail=cached
            values.append(value);lower.append(lo);upper.append(hi);details.append(dict(detail))
        return np.concatenate(values),np.concatenate(lower),np.concatenate(upper),details

    def guard(self,q):
        value,lo,hi,_=self._carriers(q);violation=value-np.clip(value,lo,hi)
        if np.any(abs(violation)>1e-8):
            raise ValueError(f'User-confirmed native source location guard violated by {np.max(abs(violation)):.12g} px')
        return True

    def constraints(self, q, selected, *, group_limits=None, return_step_carrier_response=False):
        q = np.asarray(q, float)
        _truth_record_memo_nominal(self,q)
        selected = np.asarray(selected, int)
        # This memo exists for one exact FD construction only. Its keys retain
        # every world/material column and the actual declared camera chart.
        record_memo={}
        value, lo, hi, details = self._carriers(q,record_memo=record_memo)
        J = np.zeros((len(value), len(selected)))
        active = set(self.active.tolist())
        for index, column in enumerate(selected):
            if int(column) in active:
                J[:, index] = positive_pivot_coordinate_derivative(lambda p: self._carriers(p,record_memo=record_memo)[0], q, int(column), self.steps.get(int(column), 1e-06), self.height_columns, base=value)
        if group_limits is not None:
            limits = np.asarray(group_limits, float)
            if limits.shape != (len(self.records),) or np.any(limits < 0.0) or (not np.isfinite(limits).all()):
                raise ValueError('Existing nonlinear confirmed group guard limits required')
            sizes = [item.get('carrier_size',len(item['row']['uv']) * (2 if item['normals'] is not None else 1)) for item in self.records]
            original_lower = lo.copy()
            expansion = np.repeat(limits, sizes)
            lo = lo - expansion
            hi = hi + expansion
            cursor = 0
            for item, size in zip(self.records, sizes):
                if not item.get('expand_lower_faces',item['normals'] is not None):
                    lo[cursor:cursor + size] = original_lower[cursor:cursor + size]
                cursor += size
        matrix = np.vstack((J, -J))
        target = np.r_[lo - value, value - hi]
        nonzero = np.any(matrix != 0.0, axis=1)
        if np.any(~nonzero & (target > 1e-08)):
            raise ValueError('Confirmed source violation has no active shared coordinate direction')
        result = dict(matrix=matrix[nonzero], lower=target[nonzero], report=dict(schema='rinkspace.user-confirmed-existing-native-source-step-guard.v1', authority=self.contract['authority'], groups=details, extra_independent_FIT_weight_rows=0))
        if return_step_carrier_response:
            from .source_step_curvature_tightening import carrier_step_response
            result['step_carrier_response'] = carrier_step_response(value, nonzero)
        return result


class HumanStrokeTruth:
    """Bind an authenticated authority index to already-admitted source rows."""
    def __init__(self,state,authority_path,board_factor,ice_factor,*,parameter_count=None):
        self.state=state;self.path=Path(authority_path).resolve()
        authority=json.loads(self.path.read_text(encoding='utf8'))
        if authority.get('schema')!='rinkspace.registered-direct-human-hard-truth-authorities.v1':
            raise ValueError('Direct registered human hard-truth authority required')
        self.parameter_count=board_factor.parameter_count if parameter_count is None else int(parameter_count)
        if (self.parameter_count<board_factor.parameter_count or self.parameter_count<ice_factor.parameter_count
            or (parameter_count is None and ice_factor.parameter_count!=self.parameter_count)):
            raise ValueError('Human facts must share one complete q and retain every existing material suffix')
        self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        self.records=[];self.blocks={};self.steps={};active=list(self.world_indices)
        factors={'native_board_edge':board_factor,'native_ice_curve':ice_factor}
        packets={}
        for record in authority['records']:
            if (record.get('direct_human_GT')is not True or record.get('model_or_RGB_transport_used_to_generate_UV')is not False
                or record.get('historical_sigma_used_as_hard_tolerance')is not False):
                raise ValueError('Only directly drawn and registered human truth may enter this priority interface')
            factor=factors[record['family']];ref=record['packet_reference'];path=Path(ref['path']).resolve()
            if _sha(path)!=ref['sha256']:raise ValueError('Human source packet identity differs')
            if path not in packets:packets[path]=json.loads(path.read_text(encoding='utf8'))
            original=packets[path]['rows'][int(record['original_row_index'])]
            if path!=factor.packet_path:
                if factor.packet_path not in packets:
                    packets[factor.packet_path]=json.loads(factor.packet_path.read_text(encoding='utf8'))
                current=packets[factor.packet_path]['rows'][int(record['original_row_index'])]
                if current!=original:
                    raise ValueError('Appended target changed the exact authenticated original human source row')
            row_id=record.get('id',record.get('source_row_id',record.get('original_row_id')))
            if row_id is None:raise ValueError('Explicit original human source row ID required')
            if original['id']!=row_id:raise ValueError('Human source row position/identity differs')
            raw=json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
            if hashlib.sha256(raw).hexdigest()!=record['source_row_canonical_json_sha256']:
                raise ValueError('Original human source row metadata differs')
            row=next((r for r in factor.rows+factor.held if r['id']==row_id),None)
            if row is None or row['curve']!=record['curve'] or row['pts']!=record['pts'] or row['actual_pts']!=record['actual_pts']:
                raise ValueError('Human source curve/actual exposure differs from producer')
            uv=np.asarray(record['uv'],float);original_normals=np.asarray(record['normal'],float)
            if not np.array_equal(uv,np.asarray(original['uv'],float))or not np.array_equal(original_normals,np.asarray(original['normal'],float)):
                raise ValueError('Direct human UV/normal arrays differ from immutable original source')
            normals,roundoff=unit_source_normals(original_normals,len(uv))
            bounds=np.asarray(record['normal_interval_px'],float)
            if bounds.shape!=uv.shape or not np.isfinite(bounds).all()or np.any(bounds[:,0]>bounds[:,1]):
                raise ValueError('One finite actual human stroke envelope per query required')
            if not np.array_equal(bounds[:,0],np.asarray(record['lower_bound_px'],float))or not np.array_equal(bounds[:,1],np.asarray(record['upper_bound_px'],float)):
                raise ValueError('Human envelope bound aliases differ')
            if _sha(record['source_raw_frame'])!=record['source_file_sha256']:
                raise ValueError('Original direct human native frame bytes changed')
            for refkey in ('source_annotation_registration','source_annotation_attachment'):
                ref=record[refkey]
                if _sha(ref['path'])!=ref['sha256']:raise ValueError('Human annotation/registration identity changed')
            block=factor.blocks[row['actual_pts']];self.blocks[row['actual_pts']]=block
            cols=np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,
                       block.start+4*block.bundle.index[row['actual_pts']]+np.arange(4)].astype(int)
            active.extend(cols)
            for col in cols:self.steps[int(col)]=factor.camera_steps.get(int(col),1e-6)
            self.records.append(dict(record=record,factor=factor,row=row,uv=uv,normals=normals,bounds=bounds,
                                     normal_unit_roundoff=roundoff))
        self.active=np.unique(active).astype(int);self.row_count=2*sum(len(r['uv'])for r in self.records)
        self.quadratic=np.ones(self.row_count,bool);self.scales=np.ones(self.row_count)
        self.height_columns=positive_pivot_height_columns(self.blocks.values())
        self.contract=dict(schema='rinkspace.direct-human-stroke-priority-factor.v1',authority=dict(path=str(self.path),sha256=_sha(self.path)),
            source_groups=len(self.records),source_UV_queries=self.row_count//2,complete_q_count=self.parameter_count,
            observed_UV_not_point_correspondences=True,working_source_sigma_used_as_hard_tolerance=False,
            same_physical_material_and_complete_camera_producers=True,all_queries_required=True,
            physical_course_scope_authority=board_factor.physical_course_scope_authority,
            no_new_geometry_or_camera_DOF=True,accuracy_accepted=False,
            human_truth_support_operator=human_support_operator_contract())
        self.full_stroke_records=None
        self.phase_I_records=None
        self.phase_I_guard_vertex_indices=None

    def bind_full_stroke_guard(self,path):
        path=Path(path).resolve();meta=json.loads(path.read_text(encoding='utf8'))
        if (meta.get('schema')!='rinkspace.registered-direct-human-full-stroke-hard-guard.v1'
            or meta.get('source_only_reading')is not True or meta.get('model_projection_or_RGB_transport_used')is not False
            or meta['primary_authority_reference']!=self.contract['authority']):
            # Paths can use slash aliases while referring to the same immutable
            # source; compare the reference identity explicitly below instead.
            ref=meta.get('primary_authority_reference',{})
            if (meta.get('schema')!='rinkspace.registered-direct-human-full-stroke-hard-guard.v1'
                or meta.get('source_only_reading')is not True or meta.get('model_projection_or_RGB_transport_used')is not False
                or Path(ref.get('path','')).resolve()!=self.path or ref.get('sha256')!=self.contract['authority']['sha256']):
                raise ValueError('Full human stroke guard authority differs')
        if len(meta['records'])!=len(self.records):raise ValueError('Every direct human primary group needs its full guard')
        ref=meta['guard_arrays_reference']
        if _sha(ref['path'])!=ref['sha256']:raise ValueError('Complete human stroke arrays changed')
        full=[]
        with np.load(ref['path'],allow_pickle=False)as arrays:
            for i,(record,primary)in enumerate(zip(meta['records'],self.records)):
                if (record['id']!=primary['row']['id']or record['authority_record_index']!=i
                    or record['curve']!=primary['row']['curve']or record['actual_pts']!=primary['row']['actual_pts']
                    or record['authority_source_row_canonical_json_sha256']!=primary['record']['source_row_canonical_json_sha256']
                    or record.get('full_trace_is_guard_only')is not True):
                    raise ValueError('Full stroke guard physical/source row identity differs')
                prefix=record['array_prefix'];uv=arrays[prefix+'_uv'].copy();normal=arrays[prefix+'_normal'].copy()
                bounds=arrays[prefix+'_normal_interval_px'].copy();source_normals=normal.copy();normal,error=unit_source_normals(normal,len(uv))
                if (uv.shape!=(record['vertex_count'],2)or bounds.shape!=uv.shape or not np.isfinite(uv).all()
                    or not np.isfinite(bounds).all()or np.any(bounds[:,0]>bounds[:,1])):
                    raise ValueError('Complete finite source-only human stroke arrays required')
                full.append(dict(primary,uv=uv,normals=normal,source_normals=source_normals,bounds=bounds,normal_unit_roundoff=error))
        self.full_stroke_records=full
        self.contract['full_stroke_guard']=dict(path=str(path),sha256=_sha(path),arrays_reference=ref,
            vertices=sum(len(r['uv'])for r in full),dense_vertices_added_as_independent_FIT_rows=False)

    def bind_stroke_cap_support(self,path):
        if self.full_stroke_records is None:raise ValueError('Complete human stroke must be bound before real cap support')
        path=Path(path).resolve();meta=json.loads(path.read_text(encoding='utf8'))
        if (meta.get('schema')!='rinkspace.direct-human-stroke-end-source-2D-raster-footprint-support.v2'
            or meta.get('endpoint_record_count')!=len(meta.get('records',[]))
            or meta.get('endpoint_record_count')!=2*len(self.full_stroke_records)
            or meta.get('source_only_reading')is not True or meta.get('model_projection_or_current_residual_used')is not False
            or meta.get('physical_corner_or_object_endpoint_observation_added')is not False
            or meta.get('guard_or_parent_UV_normals_normal_bands_modified')is not False):
            raise ValueError('Actual unchanged human pixel footprint and registration cap support required')
        for field,expected in (('primary_authority_reference',self.contract['authority']),
            ('full_trace_reference',self.contract['full_stroke_guard']),('full_trace_arrays_reference',self.contract['full_stroke_guard']['arrays_reference'])):
            ref=meta[field]
            if Path(ref['path']).resolve()!=Path(expected['path']).resolve()or ref['sha256']!=expected['sha256']or _sha(ref['path'])!=ref['sha256']:
                raise ValueError('Source cap support differs from frozen human authority/complete stroke')
        for item in self.records+self.full_stroke_records:
            radius=np.max(abs(item['bounds']),axis=1);item['tangent_bounds']=np.c_[-radius,radius]
        seen=set()
        def projected_tangent_bounds(record,normal):
            tangent=np.array([-normal[1],normal[0]])
            # The frozen source basis can differ from unit length by one ULP.
            # Convert its actual coordinate intervals through the inverse,
            # rather than relaxing equality or treating it as an exact rotation.
            basis=np.array([record['normal'],record['tangent']],float)
            coefficients=tangent@np.linalg.inv(basis)
            source=np.array([record['normal_cell_support_px'],record['tangent_cell_support_px']],float)
            products=source*coefficients[:,None]
            return np.array([np.min(products,axis=1).sum(),np.max(products,axis=1).sum()])
        for record in meta['records']:
            group=int(record['source_full_guard_record_index']);index=int(record['source_vertex_index'])
            full=self.full_stroke_records[group]
            if (record['id']!=full['row']['id']or index not in (0,len(full['uv'])-1)or(group,index)in seen
                or not np.array_equal(record['uv'],full['uv'][index])or not np.array_equal(record['normal'],full['source_normals'][index])
                or record['physical_endpoint_or_corner_observation']is not False or record['normal_band_from_frozen_guard_not_replaced']is not True):
                raise ValueError('Exact source-only first/last drawing vertex required; no physical endpoint observation')
            seen.add((group,index))
            for field in ('source_annotation_attachment','source_annotation_registration'):
                ref=record[field]
                if _sha(ref['path'])!=ref['sha256']:raise ValueError('Actual source cap annotation/registration changed')
            if _sha(record['source_raw_frame'])!=record['source_file_sha256']:raise ValueError('Actual source cap native RGB changed')
            full['tangent_bounds'][index]=projected_tangent_bounds(record,full['normals'][index])
            primary=self.records[group]
            matches=np.flatnonzero(np.all(abs(primary['uv']-full['uv'][index])<=1e-10,axis=1))
            for j in matches:primary['tangent_bounds'][j]=projected_tangent_bounds(record,primary['normals'][j])
        self.contract['source_2D_stroke_cap_support']=dict(path=str(path),sha256=_sha(path),drawing_caps=len(meta['records']),
            physical_corner_or_endpoint_observations_added=False,original_normal_hard_bands_unchanged=True,
            support_projected_in_actual_query_tangent_basis=True)

    def bind_practical_location_envelope(self,path):
        if self.full_stroke_records is None or 'source_2D_stroke_cap_support'not in self.contract:
            raise ValueError('Actual complete strokes and drawing caps must precede practical location policy')
        policies,ref=_visual_location_envelope(path,'native',self.contract['authority'],[r['row']['id']for r in self.records])
        for primary,full,policy in zip(self.records,self.full_stroke_records,policies):
            _apply_location_envelope(primary,policy);_apply_location_envelope(full,policy)
        self.contract['practical_manual_location_envelope']=ref
        self.contract['raster_support_not_claimed_as_physical_ground_truth_precision']=True

    def _queries_for_records(self,q,records):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,):raise ValueError('Complete shared source q required for human truth')
        if not np.isfinite(q).all()or np.any(q[self.height_columns]<=0.):
            raise ValueError('Human truth retains the existing strictly positive physical pivots')
        self.state.geometry.validate_parameters(q[self.world_indices])
        core=q[:self.state.structured_programme_coordinate_chart.core_count]
        cameras={p:b.bundle.camera(native_camera_local(b,core),p)for p,b in self.blocks.items()}
        rows=[]
        if records is None:raise ValueError('Complete direct human stroke guard has not been bound')
        for r in records:
            query=finite_stroke_queries(cameras[r['row']['actual_pts']],r['factor']._world(q,r['row']['curve']),
                                       r['uv'],r['normals'],r['bounds'],tangent_bounds=r.get('tangent_bounds'),
                                       association='nearest_finite_segment'if r['row']['curve']['kind']in('board_top','ice_board_contact')else'normal_scanline')
            rows.append(dict(id=r['row']['id'],pts=r['row']['pts'],actual_pts=r['row']['actual_pts'],query=query))
        return rows

    def queries(self,q,*,full_stroke=False):
        return self._queries_for_records(q,self.full_stroke_records if full_stroke else self.records)

    def set_phase_I_guard_vertices(self,indices_by_group,*,interior_target_px=.01):
        """Select authenticated violated guard vertices for feasibility only.

        These cuts are never extra source FIT weights. The small interior
        target prevents the solver's native-step numerical stop at an outer
        band face; the actual hard guard always uses the unchanged authority
        bounds. Failure to reach this stricter initializer target is not a
        certificate that the original human band is infeasible.
        """
        if self.full_stroke_records is None:raise ValueError('Full human guard required before selecting cuts')
        if len(indices_by_group)!=len(self.records):raise ValueError('One explicit guard-index list per human group required')
        if not np.isfinite(interior_target_px)or interior_target_px<0.:raise ValueError('Nonnegative initializer interior target required')
        records=[];inventory=[]
        for primary,full,indices in zip(self.records,self.full_stroke_records,indices_by_group):
            indices=np.asarray(indices,int)
            if indices.ndim!=1 or len(np.unique(indices))!=len(indices)or np.any(indices<0)or np.any(indices>=len(full['uv'])):
                raise ValueError('Actual full-stroke vertex identities required')
            bounds=np.vstack((primary['bounds'],full['bounds'][indices]))
            margin=np.minimum(interior_target_px,.01*(bounds[:,1]-bounds[:,0]))
            target=bounds+np.c_[margin,-margin]
            primary_radius=np.max(abs(primary['bounds']),axis=1);full_radius=np.max(abs(full['bounds']),axis=1)
            tangent_bounds=np.vstack((primary.get('tangent_bounds',np.c_[-primary_radius,primary_radius]),
                full.get('tangent_bounds',np.c_[-full_radius,full_radius])[indices]))
            tangent_margin=np.minimum(interior_target_px,.01*(tangent_bounds[:,1]-tangent_bounds[:,0]))
            records.append(dict(primary,uv=np.vstack((primary['uv'],full['uv'][indices])),
                normals=np.vstack((primary['normals'],full['normals'][indices])),bounds=target,
                tangent_bounds=tangent_bounds+np.c_[tangent_margin,-tangent_margin]))
            inventory.append(dict(id=primary['row']['id'],guard_vertex_indices=indices.tolist(),
                primary_query_count=len(primary['uv']),initializer_interior_target_px=float(interior_target_px)))
        self.phase_I_records=records;self.phase_I_guard_vertex_indices=inventory
        self.row_count=2*sum(len(r['uv'])for r in records)
        self.quadratic=np.ones(self.row_count,bool);self.scales=np.ones(self.row_count)
        return inventory

    def residual(self,q):
        records=self.records if self.phase_I_records is None else self.phase_I_records
        return np.concatenate([r['query']['residual']for r in self._queries_for_records(q,records)])

    def _carriers(self,q,*,full_stroke=False):
        rows=self.queries(q,full_stroke=full_stroke)
        return np.concatenate([np.c_[r['query']['signed_normal_px'],r['query']['coverage_tangent_px']].ravel()for r in rows])

    def jacobian(self,q,*,carriers=False,full_stroke=False):
        if carriers:
            from .human_constraint_dependency_fd import dependent_carrier_jacobian
            return dependent_carrier_jacobian(self,q,full_stroke=full_stroke)
        return self._reference_jacobian(q,carriers=carriers,full_stroke=full_stroke)

    def _reference_jacobian(self,q,*,carriers=False,full_stroke=False):
        evaluate=(lambda value:self._carriers(value,full_stroke=full_stroke))if carriers else self.residual
        base=evaluate(q)
        compact=np.empty((len(base),len(self.active)))
        try:
            for i,col in enumerate(self.active):
                compact[:,i]=positive_pivot_coordinate_derivative(evaluate,q,int(col),self.steps.get(int(col),1e-6),
                                                                  self.height_columns,base=base)
        finally:self.queries(q)
        rows,cols=np.nonzero(compact)
        return csr_matrix((compact[rows,cols],(rows,self.active[cols])),shape=(len(base),self.parameter_count))

    def block(self,q,jacobian,original_start=0):
        r=self.residual(q);J=self.jacobian(q)if jacobian else None
        ledger=dict(family='direct_human_stroke_phase_I',original_start=original_start,original_stop=original_start+len(r),
                    authority_sha256=self.contract['authority']['sha256'],query_ids=[r['row']['id']for r in self.records],
                    engineering_sigma_not_used_to_loosen_truth=True,phase_I_is_not_old_source_fit=True,
                    guard_cuts_are_feasibility_not_independent_FIT_weights=True,
                    phase_I_guard_vertex_indices=self.phase_I_guard_vertex_indices)
        return SourceRowBlock(r,J,self.quadratic,self.scales,ledger)

    def constraints(self, q, selected, *, full_stroke=True, group_limits=None, endpoint_inward_restoration=False, return_step_carrier_response=False):
        """Original-unit hard support half-spaces for the selected Schur chart."""
        if endpoint_inward_restoration:
            from .finite_endpoint_restoration_step_constraints import endpoint_inward_restoration_constraints
            return endpoint_inward_restoration_constraints(self, q, selected, full_stroke=full_stroke, group_limits=group_limits, return_step_carrier_response=return_step_carrier_response)
        selected = np.asarray(selected, int)
        if not np.isin(self.active, selected).all():
            raise ValueError('Human camera/world columns must be in explicit selected Schur chart')
        records = self.full_stroke_records if full_stroke else self.records
        if records is None:
            raise ValueError('All complete human strokes must be bound for the hard step')
        carrier = self._carriers(q, full_stroke=full_stroke)
        J = self.jacobian(q, carriers=True, full_stroke=full_stroke)[:, selected].toarray()
        bounds = np.vstack([np.c_[r['bounds'][:, 0], r['bounds'][:, 1]] for r in records])
        tangents = np.vstack([r.get('tangent_bounds', np.c_[-np.max(abs(r['bounds']), axis=1), np.max(abs(r['bounds']), axis=1)]) for r in records])
        lower = np.c_[bounds[:, 0], tangents[:, 0]].ravel()
        upper = np.c_[bounds[:, 1], tangents[:, 1]].ravel()
        if group_limits is not None:
            limits = np.asarray(group_limits, float)
            if limits.shape != (len(records),) or np.any(limits < 0.0) or (not np.isfinite(limits).all()):
                raise ValueError('Existing nonlinear native group guard limits required')
            expansion = np.repeat(limits, [2 * len(row['uv']) for row in records])
            lower = lower - expansion
            upper = upper + expansion
        matrix = np.vstack((J, -J))
        target = np.r_[lower - carrier, carrier - upper]
        nonzero = np.any(matrix != 0.0, axis=1)
        if np.any(target[~nonzero] > 0.0):
            raise ValueError('Human support face is locally infeasible in the declared physical camera/world chart')
        result = dict(matrix=matrix[nonzero], lower=target[nonzero], report=dict(self.contract, hard_halfspaces_in_native_source_pixels=True, finite_curve_nonlinear_guard_required=True, complete_guard_vertices_in_hard_halfspaces=full_stroke, extra_source_FIT_weights=0))
        if return_step_carrier_response:
            from .source_step_curvature_tightening import carrier_step_response
            result['step_carrier_response'] = carrier_step_response(carrier, nonzero)
        return result

    def report(self,q):
        queries=self.queries(q);rows=[];all_residual=[]
        for r in queries:
            query=r['query'];values=np.asarray(query['residual']).reshape(-1,2);norm=np.linalg.norm(values,axis=1)
            rows.append(dict(id=r['id'],pts=r['pts'],actual_pts=r['actual_pts'],source_queries=len(norm),
                maximum_outside_human_support_px=float(norm.max()),mean_outside_human_support_px=float(norm.mean()),
                complete_finite_scanline_intersections=int(query['finite_scanline_intersection'].sum()),
                finite_endpoint_coverage_queries=int((~query['finite_scanline_intersection']).sum()),
                signed_normal_px=query['signed_normal_px'].tolist(),finite_primitive_station=query['finite_primitive_station'].tolist()))
            all_residual.append(query['residual'])
        values=np.concatenate(all_residual)
        result=dict(contract=self.contract,rows=rows,feasible=bool(np.all(values==0.)),
            maximum_violation_px=float(abs(values).max()),primary_squared_violation=float(values@values/2),
            native_query_count=sum(len(r['uv'])for r in self.records),old_source_accuracy_accepted=False)
        if self.full_stroke_records is not None:
            full=self.queries(q,full_stroke=True);summaries=[];maximum=0.
            for row in full:
                query=row['query'];vectors=query['residual'].reshape(-1,2);distance=np.linalg.norm(vectors,axis=1)
                worst=int(np.argmax(distance));value=float(distance[worst]);maximum=max(maximum,value)
                summaries.append(dict(id=row['id'],vertices=len(distance),maximum_violation_px=value,
                    worst_full_stroke_vertex=worst,finite_scanline_intersections=int(query['finite_scanline_intersection'].sum()),
                    endpoint_coverage_queries=int((~query['finite_scanline_intersection']).sum()),
                    full_guard_passed=bool(np.all(vectors==0.))))
            result['primary_queries_feasible']=result['feasible']
            result['full_stroke_guard']=dict(rows=summaries,vertices=sum(r['vertices']for r in summaries),
                maximum_violation_px=maximum,feasible=all(r['full_guard_passed']for r in summaries),
                independent_FIT_weights_added=False)
            result['feasible']=result['feasible']and result['full_stroke_guard']['feasible']
        return result

    def guard(self,q):
        report=self.report(q)
        maximum=max(report['maximum_violation_px'],report.get('full_stroke_guard',{}).get('maximum_violation_px',0.))
        if not report['feasible']:raise ValueError('Direct human stroke hard support violated: '+str(maximum)+' native px')
        return True


class WideStrokeTruth:
    """Original direct human TOP contours on the existing private wide operator."""
    def __init__(self,factor,authority_path,parameter_count,camera_indices):
        self.factor=factor;self.path=Path(authority_path).resolve();self.parameter_count=int(parameter_count)
        self.camera_indices=np.asarray(camera_indices,int);self.world_indices=np.asarray(factor.geometry_parameter_indices,int)
        if self.camera_indices.shape!=(25,)or len(np.unique(np.r_[self.world_indices,self.camera_indices]))!=len(self.world_indices)+25:
            raise ValueError('Existing full world and independent private wide25 chart required')
        authority=json.loads(self.path.read_text(encoding='utf8'))
        if authority.get('schema')!='rinkspace.direct-human-original-wide-top-contour-hard-authorities.v1':
            raise ValueError('Original direct wide human TOP authority required')
        trace_ref=authority['trace_packet_reference'];arrays_ref=authority['array_reference']
        if _sha(trace_ref['path'])!=trace_ref['sha256']or _sha(arrays_ref['path'])!=arrays_ref['sha256']:
            raise ValueError('Original wide source/support bytes changed')
        trace=json.loads(Path(trace_ref['path']).read_text(encoding='utf8'))
        if trace!=factor.trace_packet:raise ValueError('Wide authority differs from actual CurrentWideSource trace packet')
        self.records=[]
        with np.load(arrays_ref['path'],allow_pickle=False)as arrays:
            for i,record in enumerate(authority['records']):
                original=trace['contours'][record['original_contour_index']]
                raw=json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
                if (record.get('direct_human_GT')is not True or record['id']!=original['id']
                    or hashlib.sha256(raw).hexdigest()!=record['source_row_canonical_json_sha256']
                    or record['native_pts']!=56218880 or record['actual_pts']is not None):
                    raise ValueError('Original independent wide source identity differs; programme alias cannot be assumed')
                for ref in (record['source_annotation_attachment'],record['human_semantic_authority']):
                    if _sha(ref['path'])!=ref['sha256']:raise ValueError('Original wide manual authority changed')
                if _sha(record['source_raw_frame'])!=record['source_file_sha256']:raise ValueError('Original wide raw image changed')
                prefix=record['array_prefix'];uv=arrays[prefix+'_uv'].copy();normal=arrays[prefix+'_normal'].copy()
                bounds=arrays[prefix+'_normal_interval_px'].copy();normal,error=unit_source_normals(normal,len(uv))
                if (uv.shape!=(record['vertex_count'],2)or bounds.shape!=uv.shape
                    or not np.array_equal(uv,np.asarray(original['observed_centres_source_px'],float))
                    or not np.array_equal(uv,factor.trace_uv[i])or not np.isfinite(bounds).all()or np.any(bounds[:,0]>bounds[:,1])):
                    raise ValueError('All original wide human UV and finite raster supports required')
                self.records.append(dict(record=record,uv=uv,normals=normal,bounds=bounds))
        self.active=np.unique(np.r_[self.world_indices,self.camera_indices]);self.height_columns=np.empty(0,int)
        self.phase_I_records=None;self.phase_I_guard_vertex_indices=None
        self.row_count=2*sum(len(r['uv'])for r in self.records);self.quadratic=np.ones(self.row_count,bool);self.scales=np.ones(self.row_count)
        self.contract=dict(schema='rinkspace.existing-wide-direct-human-TOP-hard-guard.v1',authority=dict(path=str(self.path),sha256=_sha(self.path)),
            source_pts=56218880,programme_actual_alias_not_assumed=True,guard_vertices=self.row_count//2,
            existing_private_wide25_and_current_TOP_course=True,working_sigma_not_hard_tolerance=True,
            extra_FIT_weight_rows=0,no_new_geometry_or_camera_DOF=True)
        self.contract['source_association']='Original CurrentWideSource nearest finite projected curve; measured normal raster support, internal knots not physical endpoints'

    def bind_practical_location_envelope(self,path):
        policies,ref=_visual_location_envelope(path,'wide',self.contract['authority'],[r['record']['id']for r in self.records])
        for item,policy in zip(self.records,policies):_apply_location_envelope(item,policy)
        self.contract['practical_manual_location_envelope']=ref
        self.contract['raster_support_not_claimed_as_physical_ground_truth_precision']=True

    def _queries_for_records(self,q,records):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,):raise ValueError('Existing complete q required for original wide human guard')
        world=q[self.world_indices];camera=self.factor.camera(q[self.camera_indices])
        cache=getattr(self,'_constraint_exact_world_cache',None)
        if cache is None:top=self.factor._curve_worlds(world)[1]
        else:
            # The original world validator remains authoritative per sample.
            self.factor.geometry.validate_parameters(world)
            key=world.tobytes()
            if key not in cache:cache[key]=self.factor._curve_worlds(world)[1]
            top=cache[key]
        rows=[]
        for r in records:
            # Literal complete-world/current-camera calculation above still
            # runs on every FD sample; exact finite-query inputs certify hits.
            key=('wide-finite-query',_truth_record_input_bytes((camera,top,r['uv'],r['normals'],r['bounds'],r.get('tangent_bounds'))))
            query=_truth_record_memo(self,r,key,q,lambda r=r:finite_stroke_queries(camera,top,r['uv'],r['normals'],r['bounds'],
                association='nearest_finite_segment',tangent_bounds=r.get('tangent_bounds')))
            rows.append(dict(id=r['record']['id'],query=query))
        return rows

    def queries(self,q):return self._queries_for_records(q,self.records)

    def set_phase_I_guard_vertices(self,indices_by_group,*,interior_target_px=.01):
        if len(indices_by_group)!=len(self.records):raise ValueError('Every original wide human group requires explicit cuts')
        records=[];inventory=[]
        for full,indices in zip(self.records,indices_by_group):
            indices=np.asarray(indices,int)
            if indices.ndim!=1 or not len(indices)or len(np.unique(indices))!=len(indices)or np.any(indices<0)or np.any(indices>=len(full['uv'])):
                raise ValueError('Explicit existing wide human guard vertices required')
            bounds=full['bounds'][indices];margin=np.minimum(interior_target_px,.01*(bounds[:,1]-bounds[:,0]))
            records.append(dict(full,uv=full['uv'][indices],normals=full['normals'][indices],bounds=bounds+np.c_[margin,-margin]))
            inventory.append(dict(id=full['record']['id'],guard_vertex_indices=indices.tolist(),initializer_interior_target_px=interior_target_px))
        self.phase_I_records=records;self.phase_I_guard_vertex_indices=inventory
        self.row_count=2*sum(len(r['uv'])for r in records);self.quadratic=np.ones(self.row_count,bool);self.scales=np.ones(self.row_count)
        return inventory

    def residual(self,q):return np.concatenate([r['query']['residual']for r in self._queries_for_records(q,self.phase_I_records or self.records)])

    def _carriers(self,q):
        return np.concatenate([np.c_[r['query']['signed_normal_px'],r['query']['coverage_tangent_px']].ravel()for r in self.queries(q)])

    def jacobian(self,q,*,carriers=False):
        _truth_record_memo_nominal(self,q)
        self._constraint_exact_world_cache={}
        evaluate=self._carriers if carriers else self.residual;base=evaluate(q);compact=np.empty((len(base),len(self.active)))
        try:
            for i,col in enumerate(self.active):
                h=1e-7 if int(col)in set(self.camera_indices[11:].tolist())else 1e-6
                compact[:,i]=positive_pivot_coordinate_derivative(evaluate,q,int(col),h,self.height_columns,base=base)
        finally:
            self._constraint_exact_world_cache=None
            self.queries(q)
        rows,cols=np.nonzero(compact)
        return csr_matrix((compact[rows,cols],(rows,self.active[cols])),shape=(len(base),self.parameter_count))

    def block(self,q,jacobian,original_start=0):
        r=self.residual(q)
        return SourceRowBlock(r,self.jacobian(q)if jacobian else None,self.quadratic,self.scales,
            dict(family='existing_wide_direct_human_phase_I',original_start=original_start,original_stop=original_start+len(r),
                authority_sha256=self.contract['authority']['sha256'],phase_I_is_not_old_source_fit=True,
                guard_cuts_are_feasibility_not_independent_FIT_weights=True,phase_I_guard_vertex_indices=self.phase_I_guard_vertex_indices))

    def constraints(self, q, selected, *, group_limits=None, return_step_carrier_response=False):
        selected = np.asarray(selected, int)
        if not np.isin(self.active, selected).all():
            raise ValueError('Wide private camera and shared world must be selected Schur columns')
        carrier = self._carriers(q)
        J = self.jacobian(q, carriers=True)[:, selected].toarray()
        bounds = np.vstack([r['bounds'] for r in self.records])
        tangents = np.vstack([r.get('tangent_bounds', np.c_[-np.max(abs(r['bounds']), axis=1), np.max(abs(r['bounds']), axis=1)]) for r in self.records])
        lower = np.c_[bounds[:, 0], tangents[:, 0]].ravel()
        upper = np.c_[bounds[:, 1], tangents[:, 1]].ravel()
        if group_limits is not None:
            limits = np.asarray(group_limits, float)
            if limits.shape != (len(self.records),) or np.any(limits < 0.0) or (not np.isfinite(limits).all()):
                raise ValueError('Existing nonlinear wide group guard limits required')
            expansion = np.repeat(limits, [2 * len(row['uv']) for row in self.records])
            lower = lower - expansion
            upper = upper + expansion
        matrix = np.vstack((J, -J))
        target = np.r_[lower - carrier, carrier - upper]
        nonzero = np.any(matrix != 0.0, axis=1)
        if np.any(target[~nonzero] > 0.0):
            raise ValueError('Wide human support face is locally unobservable in existing camera/world chart')
        result = dict(matrix=matrix[nonzero], lower=target[nonzero], report=dict(self.contract, hard_halfspaces_in_native_source_pixels=True))
        if return_step_carrier_response:
            from .source_step_curvature_tightening import carrier_step_response
            result['step_carrier_response'] = carrier_step_response(carrier, nonzero)
        return result

    def report(self,q):
        rows=[]
        for row in self.queries(q):
            query=row['query'];values=query['residual'].reshape(-1,2);distance=np.linalg.norm(values,axis=1);worst=int(np.argmax(distance))
            rows.append(dict(id=row['id'],vertices=len(values),maximum_violation_px=float(distance[worst]),worst_vertex=worst,
                finite_scanline_intersections=int(query['finite_scanline_intersection'].sum()),guard_passed=bool(np.all(values==0.))))
        return dict(contract=self.contract,rows=rows,feasible=all(r['guard_passed']for r in rows),
                    maximum_violation_px=max(r['maximum_violation_px']for r in rows),original_source_sigma_and_weights_unchanged=True)

    def guard(self,q):
        report=self.report(q)
        if not report['feasible']:raise ValueError('Original direct wide human TOP hard band violated: '+str(report['maximum_violation_px']))
        return True
