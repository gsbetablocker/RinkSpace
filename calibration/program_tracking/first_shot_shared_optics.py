"""Candidate: one shared installation and quadratic zoom optics.

Retain every original 7674 slot; require all original native PP slots zero.
Reuse public SharedCalibration, native RowCamera and existing head transform.
No owner, optimizer, source identity changes, measurements or bounds are made.
"""
from dataclasses import replace
import sys,numpy as np
from native_transverse_bundle import reference,load_model
BASE_REFERENCE=dict(path=r'D:\CodexData\RinkSpace_Data\events\ankara_2026\programmes\okada_mei_sp\results\shot1_physical_fit_20261014\head_axis_audit\transverse_native_pp.py',sha256='07f69df94e02248ff9b481c179c1163b181f20600190bbd8f57f4e76d03a6f98')
base=load_model(BASE_REFERENCE)
import quadratic_axial_native_pp as existing_quadratic
from calibration.program_tracking.first_shot_row_camera import FrameRowCamera
from calibration.program_tracking.z190_physical_camera import ViewState
head=base.head
BASE_COUNT=7674
PARAMETERS=('axial_quadratic_working_length','principal_x_quadratic_px',
    'principal_y_quadratic_px','k1_quadratic_milli','k2_constant_milli',
    'k2_linear_milli','k2_quadratic_milli','p1_milli','p2_milli')
COUNT=BASE_COUNT+len(PARAMETERS)
CENTER_BASIS=base.CENTER_BASIS;WIDTH_BASIS=base.WIDTH_BASIS
world_and_jacobian=base.world_and_jacobian;describe=base.describe

class TransverseNativePrincipalState:
    """Append nine shared optics slots; native PP is inactive and fixed zero."""
    def __init__(self,original):
        self.head_state=(original if isinstance(original,base.TransverseNativePrincipalState)
            else base.TransverseNativePrincipalState(original))
        if self.head_state.count!=BASE_COUNT:raise ValueError('Exact current Transverse7674 required')
        self.original=self.head_state.original;self.count=COUNT
        self.quadratic_index=BASE_COUNT;self.optics_indices=np.arange(BASE_COUNT,COUNT)
        self.reduced_camera_fixed_columns=np.arange(self.head_state.old_count,self.head_state.pp_count)
        if len(self.reduced_camera_fixed_columns)!=2*self.frame_count:raise ValueError('Exact native PP coordinate inventory required')
        self.initial_q=np.r_[self.head_state.initial_q,np.zeros(len(PARAMETERS))]
        self.initial_q[self.reduced_camera_fixed_columns]=0.
        self.global_shared_indices=np.r_[self.head_state.global_shared_indices,self.optics_indices].astype(int)
        if np.intersect1d(self.global_shared_indices,self.reduced_camera_fixed_columns).size:
            raise ValueError('Inactive native PP must not be shared')
    def __getattr__(self,name):
        old=self.__dict__.get('head_state')
        if old is None:raise AttributeError(name)
        return getattr(old,name)
    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(COUNT,)or not np.isfinite(q).all():raise ValueError('Complete finite reduced optics state required')
        if np.any(q[self.reduced_camera_fixed_columns]!=0.):raise ValueError('Native PP coordinates must be explicitly fixed to zero')
        return q
    def head_prefix(self,q):return self._q(q)[:BASE_COUNT]
    def quadratic(self,q):return float(self._q(q)[self.quadratic_index])
    def _activate(self,q):
        q=self._q(q);self.head_state._activate(q[:BASE_COUNT]);return q
    def split(self,q):return self.head_state.split(self.head_prefix(q))
    def axis(self,q):return self.head_state.axis(self.head_prefix(q))
    def prefix(self,q):return self.head_state.prefix(self.head_prefix(q))
    def deviations(self,q):return self.head_state.deviations(self.head_prefix(q))
    def calibration(self,q):
        q=self._q(q);cal=self.head_state.calibration(q[:BASE_COUNT]);v=q[self.optics_indices]
        if any(len(getattr(cal,k))!=2 for k in ('principal_x_fraction','principal_y_fraction','k1','axial')):
            raise ValueError('Authenticated inherited linear zoom chart required')
        if np.any(np.asarray(cal.k2)!=0.):raise ValueError('Inherited k2 must be explicit constant zero')
        return replace(cal,allow_effective_pp_offset=False,
            principal_x_fraction=(*cal.principal_x_fraction,v[1]/1920.),
            principal_y_fraction=(*cal.principal_y_fraction,v[2]/1080.),
            k1=(*cal.k1,.001*v[3]),k2=tuple(.001*v[4:7]),
            axial=(*cal.axial,v[0]/self.scene_unit_working_m),
            tangential_p1_p2=tuple(np.asarray(cal.tangential_p1_p2)+.001*v[7:9]))
    def _world_input(self,q):
        q=np.asarray(q,float);return q[:BASE_COUNT]if q.shape==(COUNT,)else q
    def scene(self,q):return self.head_state.scene(self._world_input(q))
    def review_scene(self,q):return self.head_state.review_scene(self._world_input(q))
    def world_paths(self,q,descriptors):return self.head_state.world_paths(self._world_input(q),descriptors)
    def templates_at(self,q):return self.head_state.templates_at(self._world_input(q))
    def guard_geometry(self,q):return self.head_state.guard_geometry(self._world_input(q))
    def camera25(self,q,index):
        q=self._q(q);camera=self.calibration(q).camera25(ViewState.from_state4(q[self.view_columns(index)]))
        return camera if self.axis(q)==0. else head.camera25_and_conditioned_chain(camera,self.axis(q),self.scene_unit_working_m)[0]
    def camera15_mapping(self,q,index):return prepare(self,q,index).camera15_mapping(539.5,jacobian=True)
    def guard(self,q):
        q=self._activate(q);result=self.head_state.guard(q[:BASE_COUNT]);self.calibration(q)
        report=mount_envelope(self,q)
        if not report['valid']:raise ValueError('Shared physical mount leaves existing user working internal/total boxes')
        result=dict(result);result['physical_mount_search_domain']=report
        result['reduced_shared_optics']=dict(native_PP_fixed_zero=len(self.reduced_camera_fixed_columns),
            shared_optics_parameters=len(PARAMETERS),no_independent_per_frame_C=True,
            physical_installation_metrology_not_claimed=True)
        return result

class SharedOpticsRowCamera(FrameRowCamera):
    """Public curve value/RS driver, plus explicit shared curve-column mapping."""
    def __init__(self,state,q,index,*,readout_index=None):
        super().__init__(state,q,index,readout_index=readout_index)
        self.state=state;self.extra_columns=state.optics_indices.copy()
        self.columns=np.r_[self.columns[:-1],self.extra_columns,self.columns[-1]].astype(int)
        if len(np.unique(self.columns))!=len(self.columns):raise ValueError('Shared nonlinear optics columns overlap')
        self.columns.flags.writeable=False
        self._column_locations={int(c):i for i,c in enumerate(self.columns)}
        self._extra_shared_names=('axial_degree_2_scene_div100','principal_x_fraction_degree_2',
            'principal_y_fraction_degree_2','k1_degree_2','k2_degree_0','k2_degree_1','k2_degree_2','p1','p2')
        names=self.calibration.shared_state_names()
        self._extra_selected=[names.index(n) for n in self._extra_shared_names]
        self._extra_scales=np.r_[1./(100.*self.scene_unit_working_m),1./1920.,1./1080.,np.full(6,.001)]
    def _sample(self,row,geometry_navigation=False,jacobian=False):
        offsets=self._row_offsets(np.asarray([row],float),geometry_navigation)
        left,alpha=self._brackets(offsets,geometry_navigation=geometry_navigation)
        if jacobian and self.readout_milliseconds==0. and row<539.5 and self.index>1:
            left[0]=self.index-2;alpha[0]=1.
        pair=self._pair(left[0]);a=float(alpha[0]);view,*_=self._interpolate(pair,a,False)
        return self.calibration.camera25(view),pair,a
    def camera15_mapping(self,row,*,jacobian=True,geometry_navigation=False):
        camera,cols,chain=super().camera15_mapping(row,jacobian=jacobian,geometry_navigation=geometry_navigation)
        if not jacobian:return camera,cols,None
        _,pair,a=self._sample(row,geometry_navigation,True)
        view,*_=self._interpolate(pair,a,False)
        _,shared,_=self.calibration.camera25_and_jacobians(view)
        conditioning=np.r_[np.full(3,.01),np.ones(5),np.full(3,.001),np.ones(4)]
        locations=[self._column_locations[int(c)]for c in self.extra_columns]
        chain[:,locations]=shared[:15,self._extra_selected]*conditioning[:,None]*self._extra_scales
        return camera,cols,chain
    def provenance(self):return dict(super().provenance(),
        schema='rinkspace.shared-quadratic-optics-native-row-camera.v1',native_PP_added=False,
        shared_parameter_names=list(PARAMETERS),shared_parameter_columns=self.extra_columns.tolist(),
        calibration_value_and_logf_and_row_velocity='public SharedCalibration/FrameRowCamera unchanged')

def prepare(state,q,index,*,readout_index=None):
    if isinstance(state,TransverseNativePrincipalState):
        context=SharedOpticsRowCamera(state,q,index,readout_index=readout_index)
        return head.TwoAxisTrajectory(state,q,index,readout_index=readout_index,base_context=context)
    return base.prepare(state,q,index,readout_index=readout_index)

def source_chunk(task):
    from calibration.program_tracking import first_shot_fit as fit
    import os
    q,world,D,frames,expected=task;state,_,_=fit._source_worker;q=state._q(q)
    if head.sha(q)!=expected:raise ValueError('Complete reduced worker state identity differs')
    r,J,_=fit._source_chunk((q,world,D,frames))
    if D is not None and J.shape!=(len(r),state.count):raise ValueError('Shared optics source Jacobian lost columns')
    old,pp=state.split(q)
    return r,J,dict(pid=os.getpid(),augmented_q_sha256=expected,old_q_sha256=head.sha(old),PP_sha256=head.sha(pp),
        axis_a_working_m=float(state.axis(q)),frames_0based=frames,world_derivatives_received=D is not None,
        backend=state.worker_backend,full_parameter_count=state.count,native_PP_fixed_zero=True,
        source_sampler='original first_shot_fit._source_chunk')
def install_dispatch():
    head.row_api.prepare_frame_row_camera=prepare
    module=sys.modules.get('calibration.program_tracking.row_scene_projection')
    if module is not None:module.prepare_frame_row_camera=prepare
    runtime=sys.modules.get('native_transverse_bundle')
    if runtime is not None and runtime.source_chunk.__module__=='native_transverse_bundle':runtime.source_chunk=source_chunk

def make_unshifted_world_state(state):return base.make_unshifted_world_state(state.head_state)
def init_geometry_state(binding,delta):return base.init_geometry_state(binding,delta)
MOUNT_REFERENCE=dict(path=r'D:\OneDrive\codex_projs\RinkSpace\events\ankara_2026\model_constraints\camera_hardware\user_camera_mount_and_height_20261006_v4.json')
def _mount_context(state,q):
    from rinkspace.metric_scale import boundary_span,estimate_uniform_scale
    import json
    q=state._q(q);scene=state.scene(q);scale=estimate_uniform_scale(boundary_span(scene))
    facts=json.loads(open(MOUNT_REFERENCE['path'],encoding='utf8').read())
    box=facts['v4_amendment']['working_mount_range']
    lower=np.asarray(box['lower_camera_xyz_m'],float);upper=np.asarray(box['upper_camera_xyz_m'],float)
    ratio=scale/state.scene_unit_working_m
    support=head.envelope_support(state.head_state,state.head_prefix(q))
    values=np.r_[head._mount_values(state.head_state,state.head_prefix(q)),state.quadratic(q)]
    return q,scale,ratio,support,values,lower,upper

def mount_envelope(state,q):
    q,scale,ratio,support,value,lo,hi=_mount_context(state,q)
    out=existing_quadratic.total_mount_envelope(support,value,lo/ratio,hi/ratio)
    internal=value[:3]*ratio
    total_lo=np.asarray(out['total_offset_min_working_m'])*ratio
    total_hi=np.asarray(out['total_offset_max_working_m'])*ratio
    internal_valid=bool(np.all(internal>=lo-1e-12)and np.all(internal<=hi+1e-12))
    return dict(valid=bool(out['valid']and internal_valid),total_offset_valid=bool(out['valid']),
        internal_lever_valid=internal_valid,internal_lever_camera_m=internal.tolist(),
        total_camera_local_offset_min_m=total_lo.tolist(),total_camera_local_offset_max_m=total_hi.tolist(),
        lower_m=lo.tolist(),upper_m=hi.tolist(),current_uniform_metres_per_scene_unit=float(scale),
        original_chart_working_m_per_scene_unit=float(state.scene_unit_working_m),
        current_metric_to_old_working_ratio=float(ratio),source=reference(MOUNT_REFERENCE['path']),
        user_working_search_ranges_not_measured_dimensions=True,
        all_native_SO3_arcs_and_terminal_half_intervals_certified=True,
        quadratic_axial_interior_extrema_checked=out['interior_quadratic_extrema_checked'],
        zero_axial_mechanical_reference_included=True,internal_d_and_total_both_constrained=True,
        independent_axis_a_or_axial_coefficient_box_added=False,
        axis_and_internal_lever_large_cancellation_not_allowed=True,
        verified_metre_scale=False,uniform_closest60x30_conversion=True)

def _physical_halfspaces(state,q):
    q,scale,ratio,support,value,lo,hi=_mount_context(state,q)
    total=existing_quadratic.extended_mount_halfspaces(support,lo/ratio,hi/ratio)
    internal=np.c_[np.eye(3),np.zeros((3,3))]
    M=np.r_[total['matrix'],internal,-internal]
    lower=np.r_[total['lower'],lo/ratio,-hi/ratio]
    columns=np.r_[np.arange(121,124),int(state.original.zoom_parameter_indices[3]),state.axis_index,state.quadratic_index].astype(int)
    return q,columns,value,M,lower,scale

def mount_linear_constraints(state,q,selected):
    q,columns,value,M,lower,scale=_physical_halfspaces(state,q);selected=np.asarray(selected,int)
    matrix=np.zeros((len(lower),len(selected)))
    for j,c in enumerate(columns):
        pos=np.flatnonzero(selected==c)
        if len(pos)>1:raise ValueError('Duplicate selected shared physical column')
        if len(pos):matrix[:,pos[0]]=M[:,j]
    return dict(matrix=matrix,lower=lower-M@value,
        report=dict(kind='existing_user_internal_and_continuous_total_mount_boxes',
            parameter_indices=columns.tolist(),current_uniform_metres_per_scene_unit=scale,
            all_fixed_coordinate_contributions_retained_in_current_margin=True,
            QP_pose_and_scale_are_current_state_only=True,
            proposed_trial_recovery_recomputes_full_RF4_and_geometry_metric_scale=True,
            original_working_search_range_source=reference(MOUNT_REFERENCE['path']),new_numeric_bounds_invented=False))

def recover_mount_trial(state,parent,proposed):
    from scipy.optimize import minimize
    q,cols,value,M,lower,scale=_physical_halfspaces(state,proposed)
    if np.min(M@value-lower)>=-1e-12:
        if not mount_envelope(state,q)['valid']:raise ValueError('Conservative halfspaces did not certify physical mount')
        return dict(q=q.copy(),report=dict(kind='shared_physical_mount_already_feasible',modified_parameter_indices=[]))
    fun=lambda x:.5*float((x-value)@(x-value));jac=lambda x:x-value
    result=minimize(fun,value,jac=jac,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda x:M@x-lower,jac=lambda x:M)],
        options=dict(ftol=1e-14,maxiter=300,disp=False))
    if not result.success or np.min(M@result.x-lower)<-1e-10:
        raise ValueError('Shared physical internal/total mount projection failed: '+str(result.message))
    out=q.copy();x=result.x;out[121:124]+=x[:3]-value[:3]
    out[int(state.original.zoom_parameter_indices[3])]=x[3];out[state.axis_index]=x[4];out[state.quadratic_index]=x[5]
    report=mount_envelope(state,out)
    if not report['valid']:raise ValueError('Projected reduced camera lacks continuous physical mount certificate')
    changed=np.flatnonzero(out!=q)
    if not np.isin(changed,cols).all():raise ValueError('Mount recovery changed an unrelated state coordinate')
    return dict(q=out,report=dict(kind='six_shared_coordinate_convex_internal_and_total_mount_projection',
        modified_parameter_indices=changed.tolist(),before=value.tolist(),after=x.tolist(),
        all_proposed_RF4_world_print_PP_Brown_and_wide_camera_values_preserved=True,
        trial_geometry_and_pose_and_metric_scale_recomputed=True,physical_mount_domain=report,
        projection_success=True,iterations=int(result.nit),actual_source_cost_and_full_guard_still_required=True))
def provenance():return dict(schema='rinkspace.reduced-shared-optics-candidate.v1',
    parameter_count=COUNT,inherited_slot_count=BASE_COUNT,shared_optics_parameters=list(PARAMETERS),
    added_shared_columns=list(range(BASE_COUNT,COUNT)),native_PP_value_policy='all fixed0; not used in projection',
    independent_per_frame_camera_parameters='rotation3 + log_focal only',
    source_changed=False,geometry5_slots_changed=False,new_bounds_or_temporal_priors=False,
    public_camera_producer=reference(sys.modules[FrameRowCamera.__module__].__file__),
    existing_head_producer=reference(head.__file__),inherited_model=BASE_REFERENCE)