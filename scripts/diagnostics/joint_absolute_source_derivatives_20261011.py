"""Local original absolute-source derivatives, no source graph or model fit."""
from copy import copy
from collections import OrderedDict,defaultdict
import time
import json
import joint_absolute_source_context_20261011 as ctx
a=ctx.a
np=ctx.np
from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import positive_pivot_height_columns
from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group


def camera_columns(c,at):
    block=c.state.bypts[int(at)]
    return np.unique(np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,
        block.start+4*block.bundle.index[int(at)]+np.arange(4)]).astype(int)


def camera_map(c,at,*,value=None):
    """Map the actual RF4/install chart to the original conditioned J15.

    Original projection.to_state owns center/100 and principal-point/1000
    columns. Brown4 columns remain unconditioned. A camera25 Jacobian cannot
    be multiplied by J15 until this conversion has been applied exactly once.
    """
    from calibration.program_tracking.z190_physical_camera import ViewState
    from dataclasses import replace
    value=c.q if value is None else np.asarray(value,float)
    block=c.state.bypts[int(at)];cal=block.bundle.calibration
    reference=c.state.evaluation_q(c.arrays['q'])
    dp=(value[block.xyz_slots]-reference[block.xyz_slots])/c.provider.scene_unit_m
    dm=(value[block.optics_chart.mount_indices]-reference[block.optics_chart.mount_indices])/c.provider.scene_unit_m
    do=value[block.optics_chart.indices]-reference[block.optics_chart.indices]
    cal=replace(cal,pivot_xyz=tuple(np.asarray(cal.pivot_xyz)+dp),
        lever_xyz=tuple(np.asarray(cal.lever_xyz)+dm),
        principal_x_fraction=(cal.principal_x_fraction[0]+do[0]/1920.,),
        principal_y_fraction=(cal.principal_y_fraction[0]+do[1]/1080.,),
        k1=(cal.k1[0]+.001*do[2],))
    begin=block.start+4*block.bundle.index[int(at)];view=value[begin:begin+4]
    cam,shared,local=cal.camera25_and_jacobians(ViewState(tuple(view[:3]),float(view[3]-np.log(1920.))))
    assert np.max(abs(cam-ctx.camera(c,value,at)))<1e-11
    columns=camera_columns(c,at);matrix=np.zeros((15,len(columns)))
    index={int(col):i for i,col in enumerate(columns)};names=cal.shared_state_names()
    for j,col in enumerate(block.xyz_slots):matrix[:,index[int(col)]]=shared[:15,j]/(100*c.provider.scene_unit_m)
    for j,col in enumerate(block.optics_chart.mount_indices):matrix[:,index[int(col)]]=shared[:15,j+3]/(100*c.provider.scene_unit_m)
    for col,name,scale in zip(block.optics_chart.indices,
        ('principal_x_fraction_degree_0','principal_y_fraction_degree_0','k1_degree_0'),(1/1920.,1/1080.,.001)):
        matrix[:,index[int(col)]]=shared[:15,names.index(name)]*scale
    for j in range(4):matrix[:,index[begin+j]]=local[:15,j]
    matrix*=np.r_[np.full(3,.01),np.ones(5),np.full(3,.001),np.ones(4)][:,None]
    return columns,matrix


def native_derivative(c,groups,multiplier):
    """Original dependency FD and nominal support, preserving both views."""
    source=groups[0]['factor'];view=copy(source);view.rows=[group['row']for group in groups]
    view.scales=np.concatenate([np.repeat(row['sigma'],2)for row in view.rows]);view._cache=OrderedDict()
    view.source_point_qualification=None;view._active_fit_view_cache=None
    view.camera_steps={int(col):float(step)*multiplier for col,step in source.camera_steps.items()}
    material=np.asarray(c.world_indices,int)
    if source is c.material:material=np.r_[material,source.active_material,c.placement.placement_indices]
    elif source is c.circle:material=np.r_[material,source.active_circle_indices]
    cameras=np.unique(np.concatenate([camera_columns(c,row['actual_pts'])for row in view.rows]))
    active=np.unique(np.r_[material,cameras]).astype(int)
    cams={row['actual_pts']:ctx.camera(c,c.q,row['actual_pts'])for row in view.rows}
    raw=np.concatenate([group['raw']for group in groups])
    manual=groups[0]['source_kind']=='manual_centre'
    mean=np.concatenate([group['mean_scale']for group in groups])if manual else np.ones(len(raw))
    # Native ice / board production explicitly freezes nearest-segment/clamp.
    if hasattr(view,'_derivative_residual'):
        if hasattr(view,'_nominal_associations'):associations=view._nominal_associations(c.q,cams)
        else:
            from structured_programme_source_candidate.native_ice_curve_source import _curve_error_on_association,_axis_scanline,_scanline_value
            associations=[]
            for row in view.rows:
                associations.append(_scanline_value(cams[row['actual_pts']],view._world(c.q,row['curve']),row,with_association=True)[1]if _axis_scanline(row)else
                    None if row.get('curve_observation_kind')=='horizontal_extent'else
                    _curve_error_on_association(cams[row['actual_pts']],view._world(c.q,row['curve']),row['uv'])[1])
        def evaluate(value,indices):
            cameras_now=cams if np.array_equal(value[cameras],c.q[cameras])else{
                view.rows[i]['actual_pts']:ctx.camera(c,value,view.rows[i]['actual_pts'])for i in indices}
            got=view._derivative_residual(value,cameras=cameras_now,associations=associations,
                apply_optical_support=False,row_indices=indices)
            sizes=np.r_[0,np.cumsum([2*len(row['uv'])for row in view.rows])]
            slots=np.concatenate([np.arange(sizes[i],sizes[i+1])for i in indices])
            return got/mean[slots]
        nominal=evaluate(c.q,np.arange(len(view.rows)))
        assert np.max(abs(nominal-raw))<1e-9
        # Factor helper obtains its default material h through camera_steps.
        for column in material:view.camera_steps.setdefault(int(column),1e-6*multiplier)
        rawJ=native_factor_dependency_jacobian(view,c.q,view.rows,active,material,raw,evaluate,
            positive_pivot_height_columns(view.blocks.values()))
        actualJ=rawJ.copy();offset=0
        from source_edge_intervals import SourceNormalIntervals
        for row in view.rows:
            stop=offset+2*len(row['uv']);support=row.get('_source_optical_support')
            if hasattr(view,'_uses_optical_support')and view._uses_optical_support(row):
                support=SourceNormalIntervals(row['normal'],row['normal_interval'],np.ones(len(row['uv']),bool))
            if support is not None:
                physical=raw[offset:stop]*view.scales[offset:stop]*mean[offset:stop]
                actualJ[offset:stop]=support.apply(physical,[actualJ[offset:stop]])[1][0]
            offset=stop
    else:
        # Material and circle factors retain their own actual FD operator.
        # Raw support removal is a separate diagnostic, not a new source J.
        for column in material:view.camera_steps.setdefault(int(column),1e-6*multiplier)
        def evaluate(value,indices,raw_view=False):
            rows=[view.rows[i]for i in indices]
            return ctx.native_value(c,source,value,rows,raw=raw_view)
        actual=np.concatenate([group['actual']for group in groups])
        actualJ=native_factor_dependency_jacobian(view,c.q,view.rows,active,material,actual,
            lambda value,indices:evaluate(value,indices),positive_pivot_height_columns(view.blocks.values()))
        if np.array_equal(raw,actual):rawJ=actualJ.copy()
        else:rawJ=native_factor_dependency_jacobian(view,c.q,view.rows,active,material,raw,
            lambda value,indices:evaluate(value,indices,True),positive_pivot_height_columns(view.blocks.values()))
    declared=getattr(source,'active_world',None)
    if declared is None:
        declared=np.unique(np.concatenate([c.world_indices[np.r_[np.arange(28),np.arange(44,50)]]
            if row['curve']['kind']=='current_ice_segment'else
            c.world_indices[np.r_[[0,1,2,7,8,9,44,45,46,47,48,49],list(getattr(view.geometry,'goal_crease_transverse_indices',{}).values())].astype(int)]
            if row['curve']['kind']=='current_goal_crease'else
            c.world_indices[[12,13,14,15,16,17,18,19,20,44,45,46,47,48,49]]
            if row['curve']['kind']!='hockey_circle'else
            c.world_indices[[0,1,2,3,4,5,7,8,44,45,46,47,48,49]]for row in view.rows]))
        if getattr(view.geometry,'shared_hockey_curling_axis',False):
            declared=np.unique(np.where(declared==c.world_indices[2],c.world_indices[14],declared))
    extra=np.setdiff1d(c.world_indices,declared);positions=[int(np.flatnonzero(active==col)[0])for col in extra]
    extension=dict(original_declared_world_columns=np.asarray(declared,int).tolist(),diagnostic_added_world_columns=extra.tolist(),
        added_column_raw_l2_norm=np.linalg.norm(rawJ[:,positions],axis=0).tolist(),
        added_column_actual_l2_norm=np.linalg.norm(actualJ[:,positions],axis=0).tolist(),
        added_nonzero_raw_world_columns=extra[np.any(rawJ[:,positions]!=0.,axis=0)].tolist())
    return active,actualJ,rawJ,dict(view._source_dependency_fd_report,world_dependency_extension=extension,
        nominal_branch=('original_frozen_nearest_segment_clamp_and_nominal_support'if hasattr(view,'_derivative_residual')else
                        'original_public_material_or_circle_FD_operator'),step_multiplier=multiplier)


def programme_derivatives(c,multiplier):
    started=time.perf_counter();answers={};reports=[]
    grouped=defaultdict(list)
    for i,group in enumerate(c.groups):
        if group['source_kind']in ('native','manual_centre'):
            grouped[(id(group['factor']),group['source_kind'],group.get('ordinal'))].append((i,group))
    for records in grouped.values():
        indices,groups=zip(*records);columns,J,rawJ,report=native_derivative(c,list(groups),multiplier)
        offset=0
        for i,group in records:
            stop=offset+group['scalar_rows'];answers[i]=(columns,J[offset:stop],rawJ[offset:stop]);offset=stop
        reports.append(dict(family=groups[0]['family'],ordinal=groups[0].get('ordinal'),**report))
        print(json.dumps(dict(stage='programme_native_local_J',family=groups[0]['family'],ordinal=groups[0].get('ordinal'),
            step_multiplier=multiplier,scalar_rows=sum(g['scalar_rows']for g in groups),columns=len(columns),seconds=report['seconds'])),flush=True)
    original=defaultdict(list)
    for i,group in enumerate(c.groups):
        if group['source_kind']=='original':original[(group['family'],group['actual_pts'])].append((i,group))
    certificates=set()
    for (family,at),records in original.items():
        group_started=time.perf_counter()
        block=family.split(':')[1];call=c.calls[block];rows=[group['row']for i,group in records]
        cam=ctx.camera(c,c.q,at);columns,cmap=camera_map(c,at)
        rebound=[c.provider.rebind(row,c.world)for row in rows]
        actual,camJ=call(cam,rebound,True);raw,rawCamJ=call(cam,[a.raw_row(row)for row in rebound],True)
        actualG=evaluate_geometry_group(c.provider,c.world,cam,rows,call,1e-6*multiplier,certificates=certificates)
        rawG=evaluate_geometry_group(c.provider,c.world,cam,[a.raw_row(row)for row in rows],call,1e-6*multiplier,certificates=certificates)
        J=np.column_stack((actualG,np.asarray(camJ)@cmap));rawJ=np.column_stack((rawG,np.asarray(rawCamJ)@cmap))
        all_columns=np.r_[c.world_indices,columns];offset=0
        for i,group in records:
            stop=offset+group['scalar_rows']
            assert np.max(abs(actual[offset:stop]-group['actual']))<1e-10
            assert np.max(abs(raw[offset:stop]-group['raw']))<1e-10
            answers[i]=(all_columns,J[offset:stop],rawJ[offset:stop]);offset=stop
        reports.append(dict(family=family,actual_pts=at,scalar_rows=len(actual),world_columns=len(c.world_indices),
            camera_columns=len(columns),geometry_step=1e-6*multiplier,camera_method='original_nominal_J15_times_exact_public_current_RF4_installation_mapping',
            physical_domain_stencils=list(getattr(c.provider,'_last_absolute_geometry_fd_domain_checks',[]))))
        print(json.dumps(dict(stage='programme_original_local_J',family=family,actual_pts=at,
            step_multiplier=multiplier,scalar_rows=len(actual),seconds=time.perf_counter()-group_started)),flush=True)
    assert len(answers)==len(c.groups)
    return answers,dict(step_multiplier=multiplier,seconds=time.perf_counter()-started,factors=reports)


def wide_derivatives(c,multiplier):
    from structured_programme_source_candidate import current_wide_source as module
    original_stencil=module._feasible_world_stencil
    core=c.state.structured_programme_coordinate_chart.core_count
    private=np.asarray(c.state.structured_programme_coordinate_chart.high_indices,int)
    c.wide._cache.clear()
    module._feasible_world_stencil=lambda geometry,world,column,h:original_stencil(geometry,world,column,h*multiplier)
    started=time.perf_counter()
    try:value,coreJ,privateJ=c.wide.evaluate(c.q[:core],c.q[private],True)
    finally:module._feasible_world_stencil=original_stencil;c.wide._cache.clear()
    columns=np.unique(np.r_[coreJ.indices,private]).astype(int)
    J=np.zeros((len(value),len(columns)));index={int(col):i for i,col in enumerate(columns)}
    for col in np.unique(coreJ.indices):J[:,index[int(col)]]=coreJ[:,int(col)].toarray().ravel()
    J[:,[index[int(col)]for col in private]]=privateJ
    return columns,J,dict(step_multiplier=multiplier,seconds=time.perf_counter()-started,unique_rows=len(value),
        core_columns=np.unique(coreJ.indices).tolist(),private_camera25_columns=private.tolist(),
        cache_cleared_between_scales=True,method='original_prepared_nominal_nearest_foot_with_original_world_domain_stencil_and_analytic_private25')
