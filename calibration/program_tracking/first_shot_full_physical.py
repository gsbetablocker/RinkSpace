"""Explicit complete physical direction adapter for a newly constructed owner.

All source mathematics and the solver's original trial/acceptance/control loop
remain unchanged. No fixed cache matrix is used: each direction obtains its
physical domain at the actual current q and pulls back the complete chart.
"""
from copy import deepcopy
from types import FunctionType, MethodType
import hashlib
import numpy as np
from scipy.sparse import csr_matrix, vstack

from .first_shot_source_transition import reference


def _existing_quadratic_range(model):
    """Use the very polynomial-range producer used by the original guard."""
    callback=model.existing_quadratic.total_mount_envelope.__globals__.get('quadratic_range')
    if not callable(callback):raise ValueError('Original mount guard polynomial-range callback unavailable')
    return callback


def complete_mount_domain(owner, q):
    """Replace the old mount prefix by the complete current canonical tangent."""
    from . import first_shot_fixed_head_optics as model
    from .first_shot_derived_inactive import PhysicalCourseChart, DerivedInactiveChart
    from .first_shot_mount_guard_tangent import guard_mount_tangent
    state=owner.programme.state
    q=np.asarray(q);physical,chain=owner.effective_q_and_chain(q);chain=csr_matrix(chain)
    if chain.shape!=(len(physical),len(q)):
        raise ValueError('Complete actual physical-to-canonical chain required')
    physical=np.asarray(physical)[:state.count];physical_chain=chain[:state.count]
    _,scale,ratio,support,value,box_lower,box_upper=model._mount_context(state,physical)
    _,columns,_,M,absolute,_=model._physical_halfspaces(state,physical)
    shared=np.asarray(owner.shared,dtype=int)
    inherited=owner.linear_constraints(q,shared.copy());old=csr_matrix(inherited['matrix'])
    margins=np.asarray(inherited['lower']);mount_rows=len(absolute)
    current_margin=absolute-M@value
    prefix_error=abs(margins[:mount_rows]-current_margin)
    prefix_roundoff=64*np.finfo(float).eps*np.maximum(1.,abs(absolute)+abs(M@value))
    if old.shape!=(len(margins),len(shared)) or np.any(prefix_error>prefix_roundoff):
        raise ValueError('Complete unchanged current mount prefix and inherited domain required')
    values,partial=PhysicalCourseChart(state,physical)(physical)
    analytic_scale,ds=DerivedInactiveChart._scale(*values[3:])
    if not np.isclose(analytic_scale,scale,rtol=32*np.finfo(float).eps,atol=0.):
        raise ValueError('Actual physical-course and mount scales differ')
    log_scale=(csr_matrix((ds[0]*partial[3]+ds[1]*partial[4])[None,:])@physical_chain)/scale
    views=np.asarray([state.view_columns(i)for i in range(1,state.frame_count+1)])
    if views.shape!=(state.frame_count,4):
        raise ValueError('Existing fixed-head native view chart required')
    tilt=physical[views[:,1]];driver=physical[views[:,3]]-state.calibration(physical).q_reference
    install=state.fixed_head.installation(physical[state.head_axes_indices])
    u=install['R0']@install['n']
    du=np.column_stack([install['dR0'][:,:,k]@install['n']+install['R0']@install['dn'][:,k]
        for k in range(len(state.head_axes_indices))])
    a=np.cross(u,install['b'])
    da=np.column_stack([np.cross(du[:,k],install['b'])+np.cross(u,install['db'][:,k])
        for k in range(len(state.head_axes_indices))])
    limits=np.tile([0.,1.],(state.frame_count-1,1));limits[0,0]=-.5;limits[-1,1]=1.5
    mount=guard_mount_tangent(value=value,value_jacobian=physical_chain[columns],
        lower=box_lower/ratio,upper=box_upper/ratio,
        log_scale_gradient=log_scale.toarray().ravel(),support=support,
        tilt=tilt,tilt_jacobian=physical_chain[views[:,1]],driver=driver,
        driver_jacobian=physical_chain[views[:,3]],interval_limits=limits,
        vector_a=a,vector_b=u,vector_a_jacobian=csr_matrix(da)@physical_chain[state.head_axes_indices],
        vector_b_jacobian=csr_matrix(du)@physical_chain[state.head_axes_indices],
        quadratic_interval_range=_existing_quadratic_range(model))
    tail=old[mount_rows:].tocoo()
    other=csr_matrix((tail.data,(tail.row,shared[tail.col])),shape=(old.shape[0]-mount_rows,len(q)))
    return dict(matrix=vstack((mount['matrix'],other),format='csr'),
        lower=np.r_[mount['lower'],margins[mount_rows:]],
        q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        report=dict(producer=reference(__file__),mount=mount['report'],
            inherited_mount_prefix_rows=mount_rows,inherited_other_rows=other.shape[0],
            mount_prefix_replay_roundoff_max=float(np.max(prefix_error,initial=0.)),
            complete_actual_physical_chart_pulled_back=True,
            conservative_corner_prefix_replaced_by_existing_actual_guard=True,
            current_uniform_m_per_scene_unit=float(scale),
            original_chart_unit_m_per_scene_unit=float(state.scene_unit_working_m),
            current_metric_to_old_chart_ratio=float(ratio),
            working_box_lower_camera_m=np.asarray(box_lower).tolist(),
            working_box_upper_camera_m=np.asarray(box_upper).tolist(),
            working_box_is_user_search_assumption_not_measured_hardware=True,
            independent_metric_scale_verified=False,
            source_model_and_existing_working_bounds_unchanged=True,
            full_nonlinear_original_guard_and_source_cost_required=True))


def recovery_step_receipt(parent, proposed, recovered, normal, gradient):
    """Describe the actual recovered step without extending the QP KKT claim."""
    parent=np.asarray(parent);proposed=np.asarray(proposed);recovered=np.asarray(recovered)
    before=proposed-parent;after=recovered-parent;changed=np.flatnonzero(proposed!=recovered)
    prediction=lambda step:float(-np.asarray(gradient)@step-.5*step@(normal@step))
    return dict(schema='rinkspace.actual-physical-recovery-step-comparison.v1',
        parent_q_sha256=hashlib.sha256(parent.tobytes()).hexdigest(),
        proposed_q_sha256=hashlib.sha256(proposed.tobytes()).hexdigest(),
        recovered_q_sha256=hashlib.sha256(recovered.tobytes()).hexdigest(),
        changed_columns=changed.tolist(),parent_values=parent[changed].tolist(),
        proposed_values=proposed[changed].tolist(),recovered_values=recovered[changed].tolist(),
        proposed_step_norm=float(np.linalg.norm(before)),recovered_step_norm=float(np.linalg.norm(after)),
        recovery_displacement_norm=float(np.linalg.norm(after-before)),
        proposed_source_prediction=prediction(before),recovered_source_prediction=prediction(after),
        original_QP_KKT_does_not_certify_a_changed_recovered_step=True,
        actual_nonlinear_source_cost_and_original_guard_remain_authoritative=True)


def install(owner, *, driver, entry_reference, domain_builder=None):
    """Install explicitly after prediction-refresh, before any source solve.

    The compiled public solver already dispatches its trial helper through its
    private globals. Cloning those globals changes only that numerical dispatch;
    the complete original solver statements and signature stay byte unchanged.
    """
    from calibration.solver_runtime import full_free_physical
    if hasattr(owner,'complete_physical_direction_receipt'):
        raise ValueError('Complete physical direction policy already installed')
    old_kwargs=owner.solver_kwargs;old_identity=owner.checkpoint_identity
    selected=np.asarray(owner.shared,dtype=int);fixed=np.asarray(owner.fixed,dtype=int)
    builder=(lambda q:complete_mount_domain(owner,q))if domain_builder is None else domain_builder
    if not callable(builder):raise ValueError('An actual-current-q complete domain builder is required')
    context={};original=driver.solve_joint_source
    if 'schur_trial_linear_step' not in original.__globals__:
        raise ValueError('Existing public source solver trial dispatch required')
    check_physical=original.__globals__.get('_check_physical')
    if not callable(check_physical):raise ValueError('Original complete physical guard checker required')
    refs=[reference(__file__),reference(full_free_physical.__file__),deepcopy(entry_reference)]
    if domain_builder is None:
        from . import first_shot_mount_guard_tangent
        from . import first_shot_fixed_head_optics
        refs.append(reference(first_shot_mount_guard_tangent.__file__))
        refs.append(reference(_existing_quadratic_range(first_shot_fixed_head_optics).__code__.co_filename))
    else:
        import inspect
        producer=inspect.getsourcefile(domain_builder)
        if producer is None:raise ValueError('Custom current-domain callback requires an identifiable source producer')
        domain_ref=reference(producer)
        if domain_ref not in refs:refs.append(domain_ref)
    receipt=dict(schema='rinkspace.complete-physical-source-direction.v1',producers=refs,
        complete_q_H_g_diagonal_and_original_Schur_LM_used=True,
        physical_domain_recomputed_at_each_actual_q=True,GN_reference_path_unchanged=True,
        source_cost_profiles_acceptance_bounds_and_checkpoints_unchanged=True,
        original_guard_admissible_trials_preserved_byte_exact_before_recovery=True,
        development_mode_equalities_added=False,enabled_before_solve_only=True)

    def trial(normal,gradient,diagonal,**options):
        if not context:raise ValueError('Current physical domain was not built for this direction')
        actual=(np.arange(options['global_count'])if options.get('global_parameter_indices')is None
            else np.asarray(options['global_parameter_indices']))
        if not np.array_equal(actual,selected) or normal.shape!=(context['count'],context['count']):
            raise ValueError('Current complete source/shared chart changed')
        supplied=csr_matrix(options.get('global_step_constraint_matrix'))
        if (supplied.shape!=context['shared_matrix'].shape or (supplied-context['shared_matrix']).nnz
                or not np.array_equal(options.get('global_step_constraint_lower'),context['shared_lower'])):
            raise ValueError('A changed shared SOC domain needs its own complete-column adapter')
        domain=context['full'];matrix=csr_matrix(domain['matrix']);lower=np.asarray(domain['lower'])
        extra=[];targets=[]
        lo=np.asarray(options['global_step_lower_bounds']);hi=np.asarray(options['global_step_upper_bounds'])
        for index,column in enumerate(selected):
            if np.isfinite(lo[index]):
                extra.append(csr_matrix(([1.],([0],[column])),shape=(1,normal.shape[0])));targets.append(lo[index])
            if np.isfinite(hi[index]):
                extra.append(csr_matrix(([-1.],([0],[column])),shape=(1,normal.shape[0])));targets.append(-hi[index])
        for column in options.get('nonnegative_global_step_indices',()):
            extra.append(csr_matrix(([1.],([0],[column])),shape=(1,normal.shape[0])));targets.append(0.)
        if extra:matrix=vstack((matrix,*extra),format='csr');lower=np.r_[lower,targets]
        result=full_free_physical.physical_trial_step(normal,gradient,diagonal,shared_indices=selected,
            fixed_indices=fixed,damping=options['damping'],schur_view_lm_ratio=options.get('schur_view_lm_ratio',0.),
            matrix=matrix,lower=lower)
        context['source_normal']=normal;context['source_gradient']=np.asarray(gradient)
        result.report['complete_current_physical_domain']=deepcopy(domain.get('report',{}))
        return result

    namespace=dict(original.__globals__);namespace['schur_trial_linear_step']=trial
    wrapped=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    wrapped.__kwdefaults__=deepcopy(original.__kwdefaults__);wrapped.__annotations__=dict(original.__annotations__)
    wrapped.__module__=original.__module__;wrapped.__qualname__=original.__qualname__
    for name in ('source_producers','recovery_references','_references'):
        if name=='_references' and not hasattr(owner,name):continue
        refs_list=getattr(owner,name)
        for ref in refs:
            if ref not in refs_list:refs_list.append(deepcopy(ref))

    def identity(this):
        result=old_identity();result['complete_physical_direction']=deepcopy(receipt);return result

    def kwargs(this):
        for ref in refs:
            if reference(ref['path'])!=ref:raise ValueError('Complete physical direction producer changed')
        result=old_kwargs()
        for name in ('shared_step_curvature_tightening','source_physical_endpoint_event','bounded_motion_model_factory'):
            if result.get(name)is not None:raise ValueError('Complete physical adapter needs an explicit full-column adapter for '+name)
        if result.get('conditional_view_backtracking'):
            raise ValueError('Conditional view directions need complete physical re-solving')
        shared_callback=result.get('shared_linear_constraints')
        if not callable(shared_callback):raise ValueError('Original current shared-domain callback required')
        if (not np.array_equal(result.get('global_schur_indices'),selected)
                or not np.array_equal(result.get('fixed_parameter_indices'),fixed)):
            raise ValueError('Existing complete fixed/shared selection must be retained')
        def constraints(q,columns):
            record=shared_callback(q,columns);full=builder(np.asarray(q).copy())
            if (full.get('q_sha256')!=hashlib.sha256(np.asarray(q).tobytes()).hexdigest()
                    or csr_matrix(full['matrix']).shape[1]!=len(q)):
                raise ValueError('Complete physical domain must bind this exact current canonical q')
            context.clear();context.update(count=len(q),full=full,
                shared_matrix=csr_matrix(record['matrix']),shared_lower=np.asarray(record['lower']).copy())
            return record
        result['shared_linear_constraints']=constraints
        original_recovery=result.get('trial_feasibility_recovery')
        if original_recovery is not None:
            def recovery(parent,proposed):
                try:
                    check_physical(owner.guard,np.asarray(proposed).copy())
                    valid=True
                except (ValueError,RuntimeError):
                    valid=False
                if valid:
                    result=dict(q=np.asarray(proposed).copy(),report=dict(
                        kind='original_complete_guard_admissible_trial_preserved',
                        original_full_guard_passed_before_optional_numerical_recovery=True,
                        modified_parameter_indices=[],conservative_corner_projection_skipped=True,
                        actual_source_cost_and_final_original_guard_still_required=True))
                else:
                    result=original_recovery(parent,proposed)
                if 'source_normal' not in context:raise ValueError('Recovery trace has no original source linearization')
                result=dict(result);result['report']=dict(result['report'],actual_step_comparison=
                    recovery_step_receipt(parent,proposed,result['q'],context['source_normal'],context['source_gradient']))
                return result
            result['trial_feasibility_recovery']=recovery
        result['checkpoint_identity']['complete_physical_direction']=deepcopy(receipt)
        return result
    owner.checkpoint_identity=MethodType(identity,owner);owner.solver_kwargs=MethodType(kwargs,owner)
    driver.solve_joint_source=wrapped;owner.complete_physical_direction_receipt=deepcopy(receipt)
    return deepcopy(receipt)
