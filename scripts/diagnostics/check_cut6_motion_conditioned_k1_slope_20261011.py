"""Current cut6 source tangent-space test, stdout only, no fitted state.

One prospective physical coefficient k1(logf) is compared with the existing
source chart. Original motion depth dispatch, profiles, scalar losses and
source budgets are retained. The linear projection is not a feasible fit or
a whole-programme identifiability claim.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import os
import json
import time
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='4'

import check_secondary_split_zoom_k1_chain_20261011 as split
import joint_absolute_source_context_20261011 as ctx
import joint_absolute_source_derivatives_20261011 as deriv
import joint_absolute_global_tail_context_20261011 as tail
import low_shot_camera_dependent_tail_values_20261011 as lowtail
import replay_cut6_fixed_finite_RF4_mode_20261011 as replay
from verify_original_measured_locations_actual_20261011 import install_on_source_context,current_revised
from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
from calibration.solver_runtime.source_loss import SourceLossWeights,source_irls_weights,weighted_source_cost,loss_weight_arrays
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_coordinate_derivative
from shared_board_course_candidate.finite_plane_motion_columns import prepare_finite_plane_motion_cache,finite_plane_motion_columns
from z190_fixed_target_cross_view_factor import fixed_target_source_factor
from z190_motion_native import native_motion
from scipy.sparse import csr_matrix,vstack,hstack,diags
from scipy.linalg import eigh
np,a=ctx.np,ctx.a


def emit(stage,**items):
    print(json.dumps(dict(stage=stage,**items),allow_nan=False,default=a.plain),flush=True)


@contextmanager
def slope_cameras(c,amount):
    """Same public camera25, new k1 only; no per-view camera coordinate."""
    b=c.secondary;old=b.bundle.camera;cal=b.bundle.calibration
    def camera(value,pts):
        result=old(value,pts).copy()
        result[11]+=amount*(result[6]-np.log(1920.)-cal.q_reference)
        return result
    b.bundle.camera=camera
    try:yield
    finally:b.bundle.camera=old


def sparse_columns(columns,J,n):
    J=np.asarray(J,float);rows,cols=np.nonzero(J)
    return csr_matrix((J[rows,cols],(rows,np.asarray(columns,int)[cols])),shape=(len(J),n))


def groups(c,points,revised):
    output=[]
    for family,factor in [('additional_real_native_ice_curves',c.ice),('additional_real_native_board_edges',c.board),
                         ('additional_real_native_board_material_points',c.material),('additional_real_native_board_circles',c.circle)]:
        view=factor._qualified_active_fit_view()if hasattr(factor,'_qualified_active_fit_view')else None
        view=factor if view is None else view
        w,norm=loss_weight_arrays(c.integration._payloads.get(family),view.row_count);cursor=0
        for row in view.rows:
            size=2*len(row['uv']);end=cursor+size
            if int(row['actual_pts'])in points:
                r=ctx.native_value(c,factor,c.q,[row]);raw=ctx.native_value(c,factor,c.q,[row],True)
                loss=None if w is None else SourceLossWeights(w[cursor:end],None if norm is None else norm[cursor:end])
                output.append(dict(family=family,actual_pts=int(row['actual_pts']),id=row['id'],row=row,factor=factor,
                    physical_group=a.selected_curve(row),source_kind='native',scalar_rows=size,scales=np.repeat(row['sigma'],2),
                    raw=raw,actual=r,quadratic=view.quadratic[cursor:end].copy(),loss=loss))
            cursor=end
        assert cursor==view.row_count
    for ordinal,factor in enumerate(c.centres):
        cursor=0
        for row in factor.rows:
            size=2*len(row['uv']);end=cursor+size
            if int(row['actual_pts'])in points:
                mean=factor._manual_centre_mean_scale[cursor:end]
                r=factor._residual(c.q,[row],cameras={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])})/mean
                output.append(dict(family='direct_manual_sparse_correlated_centre_secondary',ordinal=ordinal,
                    actual_pts=int(row['actual_pts']),id=row['id'],row=row,factor=factor,physical_group=a.selected_curve(row),
                    source_kind='manual_centre',scalar_rows=size,mean_scale=mean,scales=np.repeat(row['sigma'],2)*mean,
                    raw=r.copy(),actual=r.copy(),quadratic=np.ones(size,bool),loss=None))
            cursor=end
    for index,item in enumerate(c.originals):
        if item['collection_role']!='FIT'or int(item['group_actual_pts'])not in points:continue
        r,row,keep=replay.original_value(c,index,revised,c.q)
        if not len(keep):continue
        raw=replay.original_value(c,index,revised,c.q,True)[0]
        retired=(item['block']=='secondary'and int(item['group_actual_pts'])==230445824 and item['group_order']==131 and item['row_order']==2)
        if retired:r=np.zeros_like(r)
        output.append(dict(family='original_absolute:'+item['block'],actual_pts=int(item['group_actual_pts']),id=row['id'],
            row=row,physical_group=str(row.get('target',row.get('curve',row['id']))),packet_row_index=index,source_kind='original',
            scalar_rows=len(r),scales=np.repeat(row['sigma'],2),raw=raw,actual=r,
            quadratic=ctx.original_quadratic(c,index),loss=ctx.original_loss(c,index),retired=retired))
    assert sum(g['source_kind']=='original'for g in output)==183
    return output


def absolute_matrices(c,multiplier):
    # This public derivative helper evaluates the current operators in batches
    # by family/exposure, including the original connected-cap STRICT chart.
    all_groups=c.groups;native_indices=[i for i,g in enumerate(all_groups)if g['source_kind']!='original']
    c.groups=[all_groups[i]for i in native_indices]
    try:native_answers,report=deriv.programme_derivatives(c,multiplier)
    finally:c.groups=all_groups
    answers={i:native_answers[k]for k,i in enumerate(native_indices)}
    original=defaultdict(list)
    for i,g in enumerate(c.groups):
        if g['source_kind']=='original':original[(g['family'],g['actual_pts'])].append((i,g))
    certificates=set()
    for (family,at),records in original.items():
        cam=ctx.camera(c,c.q,at);rows=[g['row']for i,g in records];call=c.calls[family.split(':')[1]]
        bound=[c.provider.rebind(row,c.world)for row in rows]
        r,camJ=call(cam,bound,True);cols,cmap=deriv.camera_map(c,at)
        G=evaluate_geometry_group(c.provider,c.world,cam,rows,call,1e-6*multiplier,certificates=certificates)
        J=np.column_stack((G,np.asarray(camJ)@cmap));cursor=0
        for i,g in records:
            stop=cursor+g['scalar_rows']
            if not g.get('retired'):assert np.max(abs(r[cursor:stop]-g['actual']))<1e-10
            answers[i]=(np.r_[c.world_indices,cols],J[cursor:stop],None);cursor=stop
    matrices=[];slope=[];cursor=0
    for i,g in enumerate(c.groups):
        columns,J,_=answers[i]
        if g.get('retired'):J=np.zeros_like(J)
        matrices.append(sparse_columns(columns,J,len(c.q)))
        h=1e-6*multiplier
        if g.get('retired'):s=np.zeros(g['scalar_rows'])
        elif g['source_kind']=='original':
            row=c.provider.rebind(g['row'],c.world);cam=ctx.camera(c,c.q,g['actual_pts'])
            _,camJ=c.calls[g['family'].split(':')[1]](cam,[row],True)
            s=np.asarray(camJ)[:,11]*(cam[6]-np.log(1920.)-c.secondary.bundle.calibration.q_reference)
        else:
            f=g['factor'];row=g['row']
            def value(amount):
                with slope_cameras(c,amount):
                    if g['source_kind']=='manual_centre':
                        return f._residual(c.q,[row],cameras={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])})/g['mean_scale']
                    if hasattr(f,'_derivative_residual'):
                        cams={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])}
                        view=copy(f);view.rows=[row];view._cache=OrderedDict();view.scales=np.repeat(row['sigma'],2)
                        view.source_point_qualification=None;view._active_fit_view_cache=None
                        nominal={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])}
                        # Associations are frozen at the unperturbed source state.
                        with slope_cameras(c,-amount):
                            nominal={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])}
                        assoc=view._nominal_associations(c.q,nominal)if hasattr(view,'_nominal_associations')else None
                        if assoc is None:
                            from structured_programme_source_candidate.native_ice_curve_source import _axis_scanline,_scanline_value,_curve_error_on_association
                            assoc=[_scanline_value(nominal[row['actual_pts']],view._world(c.q,row['curve']),row,with_association=True)[1]if _axis_scanline(row)else
                                   None if row.get('curve_observation_kind')=='horizontal_extent'else
                                   _curve_error_on_association(nominal[row['actual_pts']],view._world(c.q,row['curve']),row['uv'])[1]]
                        return view._derivative_residual(c.q,cameras=cams,associations=assoc,apply_optical_support=True)
                    return ctx.native_value(c,f,c.q,[row])
            s=(value(h)-value(-h))/(2*h)
        slope.append(s);cursor+=len(s)
    return vstack(matrices,format='csr'),np.concatenate(slope),report


def motion_matrix(c,data,border):
    k,wa,wb=split.original_inputs(c.state,c.q,data,border)
    floor=data['floor'];result,audit,workspace=split.native_motion_with_optics(**k,minimum_source_depth_scene_units=floor)
    raw,Ja,Jb,Js,Jopt=result;raw=np.asarray(raw).reshape(-1,2);trace=workspace['native_trace'];sigma=k['sigma']
    slope,dispatch=split.profiled_split_column(k,trace,wa,wb,floor,workspace['profile_nuisance_tangent_native'])
    n=len(raw);b=c.secondary;ia,ib=data['ia'],data['ib']
    columns=np.c_[np.broadcast_to(b.xyz_slots,(n,3)),b.start+4*ia[:,None]+np.arange(4),
        b.start+4*ib[:,None]+np.arange(4),np.broadcast_to(b.optics_chart.indices,(n,3)),np.broadcast_to(b.optics_chart.mount_indices,(n,3))]
    mount=dict(k);dsa=np.zeros((n,3,4));dsb=np.zeros((n,3,4))
    dsa[:,:,:3]=np.transpose(k['ra'],(0,2,1))/c.provider.scene_unit_m
    dsb[:,:,:3]=np.transpose(k['rb'],(0,2,1))/c.provider.scene_unit_m
    mount.update(da=np.zeros((3,3,3)),db=np.zeros((3,3,3)),dca_shared=dsa,dcb_shared=dsb,
                 dca_view=np.zeros((3,4)),dcb_view=np.zeros((3,4)))
    mounted,_,mt=native_motion(**mount,minimum_source_depth_scene_units=floor,schur=True,return_trace=True)
    np.testing.assert_array_equal(np.asarray(mounted[0]).reshape(n,2),raw);np.testing.assert_array_equal(mt[:,:13],trace[:,:13])
    local=np.concatenate((Js.reshape(n,2,4)[:,:,:3]/(100*c.provider.scene_unit_m),Ja.reshape(n,2,4),
        Jb.reshape(n,2,4),Jopt.reshape(n,2,3),mounted[3].reshape(n,2,4)[:,:,:3]),axis=2)/sigma[:,:,None]
    rows=np.broadcast_to(np.arange(2*n).reshape(n,2,1),local.shape).ravel()
    cols=np.broadcast_to(columns[:,None,:],local.shape).ravel()
    J=csr_matrix((local.ravel(),(rows,cols)),shape=(2*n,len(c.q)));J.sum_duplicates();J.eliminate_zeros()
    course,cq,chain=c.provider.print_course_binding(c.world)
    sampled=course.sample_perimeter(cq,with_jacobian=True);edge=np.asarray(sampled[0]);jed=np.asarray(sampled[2])@chain
    if np.array_equal(edge[0],edge[-1]):edge=edge[:-1];jed=jed[:-1]
    planesJ=np.zeros((int(data['board'].sum()),4,len(c.world_indices)));plane_values=np.zeros((len(planesJ),4))
    board_ids=np.flatnonzero(data['board']);positions={int(i):j for j,i in enumerate(board_ids)}
    cameras=np.asarray([ctx.camera(c,c.q,int(p))for p in data['stamps']])
    for view in np.unique(ia[data['board']]):
        ids=np.flatnonzero(data['board']&(ia==view));owner=SimpleNamespace(_facet_course_cache=None,records=[None]*len(ids),
            source_row_groups={0:np.arange(len(ids))},source_cameras={0:cameras[view]},source_uv=data['a'][ids],
            exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
        proxy=SimpleNamespace(sample_perimeter=lambda unused,with_jacobian=False:(border,))
        facets=split.s.OriginalMotionFarBoardRoleBinding._current_source_facets(owner,proxy,None,('current',split.s.token(border)))
        for point,face in zip(ids,facets):
            plane_values[positions[int(point)]],planesJ[positions[int(point)]]=split.s.vertical_plane_value_jacobian(
                edge[face],edge[(face+1)%len(edge)],jed[face],jed[(face+1)%len(edge)])
    np.testing.assert_allclose(plane_values,k['planes'][data['board']],rtol=0,atol=0)
    geom=workspace['geometry'];direction=np.einsum('ni,nij->nj',np.c_[geom['na'],np.ones(n)],k['ra'])
    cache=prepare_finite_plane_motion_cache(k,data['board'],source_direction_selected=direction[data['board']])
    wr,wJ,world_report=finite_plane_motion_columns(cache,plane_values,planesJ,sigma_normalized=True)
    rr,cc=np.nonzero(wJ);J+=csr_matrix((wJ[rr,cc],(wr[rr],c.world_indices[cc])),shape=J.shape)
    active=np.flatnonzero(data['incident']);selected=(2*active[:,None]+np.arange(2)).ravel()
    emit('motion_original_J_ready',complete_batch=n,selected_points=len(active),selected_scalars=len(selected),
         original_profiles_and_roles=True,trace=dispatch,world_columns=world_report,sigma_divisions=1)
    return (raw/sigma).ravel()[selected],J[selected],(slope/sigma).ravel()[selected],trace,data['incident']


def extra_matrices(c,points,multiplier,auth,track,pf):
    groups=[];matrices=[];slope=[]
    # Pattern operator retains all original source material and shared Q9;
    # the other external frames stay in the original full ledger and are
    # selected only after the full original evaluate/J.
    with tail._public_derivative_widths(c,multiplier):pr,pJ=tail._evaluate_pattern(c,pf,c.q,True)
    saved=tail._saved_pattern_values(c,pf);assert np.max(abs(pr-saved))<1e-12
    saved_records=a.read(c.directory/'source_review_UNACCEPTED.json')['independent_board_pattern_source']['fit']
    sizes=[sum(1 if 'native_sigma_scaled_excess'in item or 'distance_native_px'in item else len(item['outside_support_native_px'])for item in record['source_reports'])for record in saved_records]
    assert sum(sizes)==len(pr)
    cursor=0;selected=[]
    for f,size in zip(pf.fit,sizes):
        if f['observer_state_owner']=='programme'and int(f['source_actual_pts'])in points:selected.extend(range(cursor,cursor+size))
        cursor+=size
    selected=np.asarray(selected,int);P=pJ[selected]
    # Existing independent material placement suffix participates through its
    # public context, and is not silently omitted from the native pattern J.
    additions=[]
    for col in c.placement.placement_indices:
        h=1e-6*multiplier;v=c.q.copy();v[col]+=h;w=c.q.copy();w[col]-=h
        additions.append((tail._evaluate_pattern(c,pf,v)-tail._evaluate_pattern(c,pf,w))/(2*h))
    if additions:P+=sparse_columns(c.placement.placement_indices,np.column_stack(additions)[selected],len(c.q))
    h=1e-6*multiplier
    with slope_cameras(c,h):sp=tail._evaluate_pattern(c,pf,c.q)
    with slope_cameras(c,-h):sm=tail._evaluate_pattern(c,pf,c.q)
    groups.append(dict(family='shared_board_patterns:cut6',id='actual_pattern_source',actual=pr[selected],quadratic=np.ones(len(selected),bool),loss=None,scalar_rows=len(selected)))
    matrices.append(P);slope.append(((sp-sm)/(2*h))[selected])
    # Source-row order is the actual factor fit order. Includes the qualified
    # initialized LOW target association, while its source RF4 is not invented.
    for row in track.fit_evaluation_rows:
        pts={int(row[k])for k in ('actual_pts','source_actual_pts','target_actual_pts')if k in row}
        if not (pts&points or np.intersect1d(row['_columns'],c.shot_columns).size):continue
        r=track._evaluate(c.q,row)[0];cols=row['_columns'];J=np.empty((len(r),len(cols)))
        for j,col in enumerate(cols):
            J[:,j]=positive_pivot_coordinate_derivative(lambda value:track._evaluate(value,row)[0],c.q,int(col),row['_steps'][int(col)]*multiplier,track.height_columns,base=r)
        h=1e-6*multiplier
        with slope_cameras(c,h):sp=track._evaluate(c.q,row)[0]
        with slope_cameras(c,-h):sm=track._evaluate(c.q,row)[0]
        groups.append(dict(family='additional_native_static_board_tracks:cut6',id=row['id'],actual=r,
            quadratic=np.ones(len(r),bool),loss=None,scalar_rows=len(r),
            operator_basis='frozen_initialized_source_conditioned_material'if row.get('_anchor')or row.get('initialization')else'current_ray_same_board_texture'))
        matrices.append(sparse_columns(cols,J,len(c.q)));slope.append((sp-sm)/(2*h))
    roles={'long_side_main':0,'long_side_secondary':1,'low_end':2}
    for path,expected in lowtail.CROSS_REFS:
        assert auth(path)==expected
        for row in a.read(path)['rows']:
            if row['split']!='fit':continue
            pts=int(c.state.original.aliases[int(row['source_native_pts'])])
            if pts not in points:continue
            assert pts==int(row['source_actual_pts']);ref=row['source_native_reference'];assert auth(ref['raw_png'])==ref['raw_png_sha256']
            bounds=[[float('-inf')if l is None else l,float('inf')if u is None else u]for l,u in zip(row['physical_axis_broad_support_xy_min'],row['physical_axis_broad_support_xy_max'])]
            fs=dict(source_visible=True,source_support_manually_qualified=True,model_projection_used_to_generate_or_select_uv=False,
                physical_target_kind='fixed_pan_head_pivot',fixed_pivot_source_qualified=True,physical_target_authority=dict(whole_head_mechanical_origin_support=True),
                native_support_bounds_xy_px=bounds,sigma_native_px=np.broadcast_to(row['source_sigma_engineering_native_px'],2).tolist(),sigma_semantics=row['sigma_semantics'])
            observer=c.state.ids[roles[row['observer_installation']]];target=c.state.ids[roles[row['target_installation']]];b=c.state.bypts[pts]
            start=b.start+4*b.bundle.index[pts]
            cols=np.unique(np.r_[deriv.camera_columns(c,pts),c.state.slots[target]])
            def evaluate(value,amount=0.):
                cal=b.bundle.current_calibration(value);view=value[start:start+4]
                k1=cal.k1[0]+amount*(view[3]-np.log(1920.)-cal.q_reference)
                return fixed_target_source_factor(fs,value[c.state.slots[observer]],value[c.state.slots[target]],view[:3],view[3],
                    np.asarray(cal.lever_xyz)*c.provider.scene_unit_m,np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]]),k1).residual
            r=evaluate(c.q);J=[]
            for col in cols:
                h=(1e-4 if col in b.optics_chart.indices else 1e-7 if start<=col<start+3 else 1e-6)*multiplier
                J.append(positive_pivot_coordinate_derivative(evaluate,c.q,int(col),h,track.height_columns,base=r))
            h=1e-6*multiplier;s=(evaluate(c.q,h)-evaluate(c.q,-h))/(2*h)
            groups.append(dict(family='physical_target:cut6',id=row['id'],actual=r,quadratic=np.ones(len(r),bool),loss=None,scalar_rows=len(r)))
            matrices.append(sparse_columns(cols,np.column_stack(J),len(c.q)));slope.append(s)
    emit('associated_tail_J_ready',step_multiplier=multiplier,groups=len(groups),scalar_rows=sum(len(g['actual'])for g in groups),
         families={f:sum(g['scalar_rows']for g in groups if g['family']==f)for f in sorted({g['family']for g in groups})})
    return groups,vstack(matrices,format='csr'),np.concatenate(slope)


def copy_pattern(owner,frames):
    result=copy(owner);result.fit=frames;return result


def summarize_solution(c,columns,direction,groups,offsets,response,remaining):
    output=[]
    for family in sorted({g['family']for g in groups}):
        ids=np.concatenate([np.arange(offsets[i],offsets[i+1])for i,g in enumerate(groups)if g['family']==family])
        r=np.concatenate([g['actual']for g in groups if g['family']==family]);metric=np.concatenate([source_irls_weights(g['actual'],g['quadratic'],g['loss'])for g in groups if g['family']==family])
        output.append(dict(family=family,scalar_rows=len(ids),weighted_norm2_before=float(np.sum(metric*r*r)),
            residual_projection_energy=float(np.sum(remaining[ids]**2)),
            response_dot_residual=float(np.sum(response[ids]*np.sqrt(metric)*r))))
    lookup={int(col):float(d)for col,d in zip(columns,direction)};canonical=c.state.canonical_parameter_layout.evaluation_to_canonical
    largest=sorted(zip(columns.tolist(),direction.tolist()),key=lambda x:abs(x[1]),reverse=True)[:12]
    active_mount=[(34199,'lower'),(34200,'upper'),(34196,'lower'),(34197,'upper'),(34201,'lower'),(34202,'lower'),(34203,'upper')]
    mount=[]
    for cc,side in active_mount:
        col=int(cc);d=lookup.get(col,0.)
        mount.append(dict(canonical_index=int(canonical[col]),evaluation_index=col,parent_bound=side,delta=d,outward=d<0 if side=='lower'else d>0))
    rf=[lookup.get(int(col),0.)for col in c.shot_columns];rf=np.asarray(rf).reshape(-1,4)
    return dict(family_direction=output,largest_parameter_directions=[dict(evaluation_index=i,canonical_index=int(canonical[i]),delta=d)for i,d in largest],
        RF4_direction_max_abs=np.max(abs(rf),axis=0).tolist(),RF4_direction_p95_abs=np.quantile(abs(rf),.95,axis=0).tolist(),
        active_mount_direction=mount,any_active_mount_outward=any(r['outward']for r in mount))


def projection(c,groups,J1,J2,s1,s2):
    r=np.concatenate([g['actual']for g in groups]);w=np.concatenate([source_irls_weights(g['actual'],g['quadratic'],g['loss'])for g in groups]);rw=np.sqrt(w)*r
    columns=np.union1d(J1.indices,J2.indices);columns=np.setdiff1d(columns,c.fixed_evaluation)
    A1=diags(np.sqrt(w))@J1[:,columns];A2=diags(np.sqrt(w))@J2[:,columns];S1=np.sqrt(w)*s1;S2=np.sqrt(w)*s2
    norms=np.sqrt(np.asarray(((A1.multiply(A1)+A2.multiply(A2))*.5).sum(axis=0)).ravel());used=norms>0
    columns=columns[used];norms=norms[used];A1=A1[:,used]@diags(1/norms);A2=A2[:,used]@diags(1/norms)
    offsets=np.r_[0,np.cumsum([g['scalar_rows']for g in groups])]
    emit('local_matrix_ready',rows=len(r),existing_free_columns=len(columns),nonzeros=A1.nnz,original_source_cost=sum(weighted_source_cost(g['actual'],g['quadratic'],g['loss'])for g in groups),
         weighted_IRLS_residual_energy=float(rw@rw),two_stencil_J_relative_Frobenius=float(np.sqrt(((A1-A2).multiply(A1-A2)).sum())/np.sqrt((A1.multiply(A1)).sum())),
         kappa_stencil_difference_L2=float(np.linalg.norm(S1-S2)),candidate_nominal_k1=c.secondary.bundle.calibration.k1,
         local_unconstrained_projection_not_feasible_fit=True,other_shots_global_wide_external_roll_and_full86_constraints_not_in_projection=True)
    for scale,A,S in [('h',A1,S1),('h_over_2',A2,S2)]:
        H=(A.T@A).toarray();evals,V=eigh(H,check_finite=False,driver='evd');evals=np.maximum(evals,0);rhs=np.asarray(A.T@rw).ravel();srhs=np.asarray(A.T@S).ravel()
        emit('local_column_spectrum',stencil=scale,normalized_column_gram_eigenvalue_quantiles=np.quantile(evals,[0,.01,.1,.5,.9,.99,1]),
             raw_common_column_norm_quantiles=np.quantile(norms,[0,.01,.1,.5,.9,.99,1]),
             spectrum_is_Gram_not_high_precision_small_singular_value_certificate=True)
        for relative in (1e-12,1e-10,1e-8,1e-6):
            keep=evals>relative*evals[-1];VK=V[:,keep];lam=evals[keep]
            solve=lambda b:VK@((VK.T@b)/lam)
            coef=-solve(rhs);base_response=np.asarray(A@coef).ravel();remaining=rw+base_response
            scoef=solve(srhs);orth=S-np.asarray(A@scoef).ravel();information=float(orth@orth);score=float(orth@remaining)
            kap=-score/information if information>0 else 0.;jointcoef=coef-kap*scoef
            jointresponse=np.asarray(A@jointcoef).ravel()+kap*S;jointremaining=rw+jointresponse
            emit('existing_DOF_and_incremental_k1_slope',stencil=scale,Gram_relative_cutoff=relative,retained_rank=int(keep.sum()),
                existing_remaining_energy=float(remaining@remaining),candidate_raw_column_energy=float(S@S),candidate_after_existing_information=information,
                kappa_local_linear_coefficient=kap,unit_metric_formal_standard_error=1/np.sqrt(information)if information>0 else None,
                incremental_projection_gain=float(remaining@remaining-jointremaining@jointremaining),joint_remaining_energy=float(jointremaining@jointremaining),
                candidate_existing_nuisance_coefficient_norm=float(np.linalg.norm(kap*scoef/norms)),
                existing=summarize_solution(c,columns,coef/norms,groups,offsets,base_response,remaining),
                joint=summarize_solution(c,columns,jointcoef/norms,groups,offsets,jointresponse,jointremaining),
                formal_unit_metric_error_not_measurement_covariance_or_profile_likelihood=True,
                rank_cutoffs_fixed_before_residual_no_true_rank_or_finite_feasible_attainment_claim=True)


def main():
    started=time.perf_counter();a.RUN=split.DIRECTORY.parent.parent;a.CHILD=split.Q;a.CONFIG=split.s.CONFIG
    c=ctx.load_context();c.secondary=c.state.bypts[split.s.PTS]
    c._global_tail_roll=tail._install_camera_hooks(c);external,ef=tail._external_owner(c);pf=tail._install_pattern_owner(c,external)
    shot=a.read(a.ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10][5]
    points={p for p in c.state.bypts if int(shot['start_pts'])<=p<int(shot['end_pts'])}
    c.shot_columns=np.concatenate([c.state.bypts[p].start+4*c.state.bypts[p].bundle.index[p]+np.arange(4)for p in sorted(points)])
    assert len(points)==492 and not np.intersect1d(c.shot_columns,c.fixed_evaluation).size
    with np.load(c.path,allow_pickle=False)as z:
        ordinal=z['aux__block_names'].tolist().index('secondary');keys=['aux__block_names','aux__block_lengths','aux__canonical_pts','aux__current_depth_and_material_domain_contract_json','profile__weights_'+str(ordinal)]
        saved={key:z[key].copy()for key in keys}
    data=split.s.secondary_inputs(c.manifest,saved);data['incident']=np.isin(data['pa'],list(points))|np.isin(data['pb'],list(points))
    course,cq,_=c.provider.print_course_binding(c.world);border=np.asarray(course.sample_perimeter(cq,with_jacobian=False)[0])
    mr,mJ,ms,trace,selected=motion_matrix(c,data,border)
    motion_group=dict(family='programme:secondary:cut6_motion',id='all_original_directed_motion',actual=mr,quadratic=np.zeros(len(mr),bool),loss=None,scalar_rows=len(mr))
    packet,revised=install_on_source_context(c,c.options['--original-measured-transition-location-source']);c.groups=groups(c,points,revised)
    emit('absolute_scope_ready',state_sha256=split.Q,original_FIT_rows=sum(g['source_kind']=='original'for g in c.groups),
         current_native_and_centres_rows=sum(g['source_kind']!='original'for g in c.groups),groups=len(c.groups),
         per_family_original_cost={f:sum(weighted_source_cost(g['actual'],g['quadratic'],g['loss'])for g in c.groups if g['family']==f)for f in sorted({g['family']for g in c.groups})})
    lowtail.AUTH=split.s.R/'okada_review86_q276_vertical_print_station_one_cycle_20261011.bootstrap/startup_source_auth_execution/actual_startup_source_auth_reuse.json'
    auth=lowtail.SourceBytes()
    with lowtail.authenticated_track_constructor(auth):track=lowtail.track_module.NativeBoardTrackFactor(c.state,c.options['--additional-native-board-track-source'],len(c.q))
    arrays=[];extra_reference=None
    for multiplier in (1.,.5):
        AJ,AS,_=absolute_matrices(c,multiplier);eg,EJ,ES=extra_matrices(c,points,multiplier,auth,track,pf)
        if extra_reference is None:extra_reference=eg
        else:
            assert [g['id']for g in eg]==[g['id']for g in extra_reference]
            for x,y in zip(eg,extra_reference):np.testing.assert_array_equal(x['actual'],y['actual'])
        arrays.append((vstack((mJ,AJ,EJ),format='csr'),np.r_[ms,AS,ES]))
    auth.finish();all_groups=[motion_group]+c.groups+extra_reference
    projection(c,all_groups,arrays[0][0],arrays[1][0],arrays[0][1],arrays[1][1])
    emit('completed',seconds=time.perf_counter()-started,source_images_decoded=0,source_images_written=0,
         saved_J_H_dq_or_new_state=False,complete_programme_optimization=False)


if __name__=='__main__':main()
