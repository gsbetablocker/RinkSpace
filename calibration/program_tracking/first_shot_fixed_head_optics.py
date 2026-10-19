"""Candidate fixed shared head with pan/tilt-angle native RS and shared optics.

Independent successor producer; never installs into a running owner. The four
shared axis coordinates are installation unknowns. The standard unit axes are
only chart origins, not manufacturer or visual metrology.
"""
from collections import OrderedDict
from types import SimpleNamespace
import sys,numpy as np
from calibration.program_tracking import first_shot_shared_optics as optics
from calibration.program_tracking.fixed_pan_tilt import FixedPanTilt
from calibration.program_tracking.z190_physical_camera import ViewState
from native_transverse_bundle import reference
base=optics.base;head=optics.head;existing_quadratic=optics.existing_quadratic
MOUNT_REFERENCE=optics.MOUNT_REFERENCE
BASE_COUNT=7683;COUNT=7687
CENTER_BASIS=base.CENTER_BASIS;WIDTH_BASIS=base.WIDTH_BASIS
world_and_jacobian=base.world_and_jacobian;describe=base.describe


def _batch_rotations(angles,installation):
    n,b,R0=(installation[k]for k in('n','b','R0'))
    def exp_axis(axis,a):
        c=np.cos(a)[:,None,None];s=np.sin(a)[:,None,None]
        return c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return exp_axis(b,angles[:,1])@R0@exp_axis(n,angles[:,0])


class TransverseNativePrincipalState:
    def __init__(self,original):
        self.optics_state=(original if getattr(original,'count',None)==BASE_COUNT and hasattr(original,'optics_indices')
            else optics.TransverseNativePrincipalState(original))
        self.original=self.optics_state.original;self.count=COUNT
        self.fixed_head=FixedPanTilt((0.,0.,1.),(1.,0.,0.))
        self.head_axes_indices=np.arange(BASE_COUNT,COUNT)
        self.third_pose_slot_indices=np.asarray([self.optics_state.view_columns(i)[2]for i in range(1,self.frame_count+1)])
        self.reduced_camera_fixed_columns=self.optics_state.reduced_camera_fixed_columns.copy()
        self.fixed_camera_parameter_indices=np.r_[self.reduced_camera_fixed_columns,self.third_pose_slot_indices]
        self.initial_q=np.r_[self.optics_state.initial_q,np.zeros(4)]
        self.initial_q[self.third_pose_slot_indices]=0.
        self.global_shared_indices=np.r_[self.optics_state.global_shared_indices,self.head_axes_indices].astype(int)
        self._fixed_head_support_cache=OrderedDict()
    def __getattr__(self,name):
        old=self.__dict__.get('optics_state')
        if old is None:raise AttributeError(name)
        return getattr(old,name)
    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(COUNT,)or not np.isfinite(q).all():raise ValueError('Complete fixed-head/optics state required')
        if np.any(q[self.fixed_camera_parameter_indices]!=0.):raise ValueError('Inactive native PP and third pose slots must be fixed zero')
        return q
    def optics_prefix(self,q):return self._q(q)[:BASE_COUNT]
    def _activate(self,q):
        q=self._q(q);self.optics_state._activate(q[:BASE_COUNT]);return q
    def split(self,q):return self.optics_state.split(self.optics_prefix(q))
    def axis(self,q):return self.optics_state.axis(self.optics_prefix(q))
    def quadratic(self,q):return self.optics_state.quadratic(self.optics_prefix(q))
    def calibration(self,q):return self.optics_state.calibration(self.optics_prefix(q))
    def deviations(self,q):return self.optics_state.deviations(self.optics_prefix(q))
    def _world_input(self,q):
        q=np.asarray(q,float);return q[:BASE_COUNT]if q.shape==(COUNT,)else q
    def scene(self,q):return self.optics_state.scene(self._world_input(q))
    def review_scene(self,q):return self.optics_state.review_scene(self._world_input(q))
    def templates_at(self,q):return self.optics_state.templates_at(self._world_input(q))
    def world_paths(self,q,descriptors):return self.optics_state.world_paths(self._world_input(q),descriptors)
    def guard_geometry(self,q):return self.optics_state.guard_geometry(self._world_input(q))
    def camera25(self,q,index):return prepare(self,q,index).camera25_at_row(539.5)
    def camera15_mapping(self,q,index):return prepare(self,q,index).camera15_mapping(539.5)
    def guard(self,q):
        q=self._activate(q);old,_=self.split(q)
        # Do not feed angle slots into old R3 head/envelope code. The original
        # base guard uses geometry, print, outside pivot, readout and focal.
        result=self.original.guard(old,check_mount_bounds=False)
        self.calibration(q);self.fixed_head.installation(q[self.head_axes_indices])
        report=mount_envelope(self,q)
        if not report['valid']:raise ValueError('Fixed-head shared optics leaves user internal/continuous total working domain')
        return dict(result,physical_mount_search_domain=report,physical_pan_tilt_angles_interpolated=True,
            per_frame_only_R3_and_zoom1=False,per_frame_only_pan_tilt_and_zoom=True,
            native_PP_fixed_zero_count=len(self.reduced_camera_fixed_columns),third_pose_slots_fixed_zero_count=len(self.third_pose_slot_indices),
            old_R3_guard_or_envelope_not_called=True)


class FixedHeadRowCamera(optics.SharedOpticsRowCamera):
    def __init__(self,state,q,index,*,readout_index=None):
        self.axis_shared=state._q(q)[state.head_axes_indices].copy();self.fixed_head=state.fixed_head
        super().__init__(state,q,index,readout_index=readout_index)
        self.columns=np.r_[self.columns[:-1],state.head_axes_indices,self.columns[-1]].astype(int)
        if len(np.unique(self.columns))!=len(self.columns):raise ValueError('Fixed-head axis columns overlap')
        self._column_locations={int(c):i for i,c in enumerate(self.columns)}
        # Core construction interprets raw3 only for this cached mid camera;
        # replace it immediately with the actual two-angle physical rotation.
        mid=self._views[self.index-1];r=self.fixed_head.rotvec(mid[:2],self.axis_shared)
        self._mid_camera=self.calibration.camera25(ViewState(tuple(r),float(mid[3])))
        self.cache_key=(*self.cache_key,'shared-fixed-head-angle-clock',tuple(self.axis_shared))
    def _pair(self,left):
        left=int(left)
        if left not in self._pairs:self._pairs[left]=SimpleNamespace(left=left,right=left+1,a=self._views[left],b=self._views[left+1],
            seconds=float((self.pts[left+1]-self.pts[left])*self.time_base))
        return self._pairs[left]
    def _interpolate(self,pair,alpha,jacobian):
        if not jacobian:
            r=self.fixed_head.interpolate(pair.a[:2],pair.b[:2],alpha,self.axis_shared)
            return ViewState(tuple(r),float((1-alpha)*pair.a[3]+alpha*pair.b[3])),None,None,None
        r,J=self.fixed_head.interpolate(pair.a[:2],pair.b[:2],alpha,self.axis_shared,jacobian=True)
        left=np.zeros((4,4));right=np.zeros((4,4));da=np.zeros(4)
        left[:3,:2]=J[:,4:6];right[:3,:2]=J[:,6:8];left[3,3]=1-alpha;right[3,3]=alpha
        da[:3]=J[:,-1];da[3]=pair.b[3]-pair.a[3]
        return ViewState(tuple(r),float((1-alpha)*pair.a[3]+alpha*pair.b[3])),left,right,da
    def camera15_mapping(self,row,*,jacobian=True,geometry_navigation=False):
        camera,cols,chain=super().camera15_mapping(row,jacobian=jacobian,geometry_navigation=geometry_navigation)
        if not jacobian:return camera,cols,None
        _,pair,alpha=self._sample(row,geometry_navigation,True);view,*_=self._interpolate(pair,alpha,False)
        _,J=self.fixed_head.interpolate(pair.a[:2],pair.b[:2],alpha,self.axis_shared,jacobian=True)
        _,_,per_view=self.calibration.camera25_and_jacobians(view)
        cond=np.r_[np.full(3,.01),np.ones(5),np.full(3,.001),np.ones(4)]
        locations=[self._column_locations[int(c)]for c in self.state.head_axes_indices]
        chain[:,locations]=(per_view[:15,:3]@J[:,:4])*cond[:,None]
        return camera,cols,chain
    def camera25_at_rows(self,rows,*,geometry_navigation=False):
        rows=np.asarray(rows,float);offsets=self._row_offsets(rows,geometry_navigation)
        left,alpha=self._brackets(offsets,geometry_navigation=geometry_navigation)
        a=self._views[left];b=self._views[left+1]
        angles=(1-alpha[:,None])*a[:,:2]+alpha[:,None]*b[:,:2]
        rotations=_batch_rotations(angles,self.fixed_head.installation(self.axis_shared))
        from calibration.program_tracking.first_shot_row_camera import _rotation_logs
        rvec=_rotation_logs(rotations);logf=(1-alpha)*a[:,3]+alpha*b[:,3]
        cal=self.calibration;mode=cal.output_mode;z=logf-cal.q_reference
        poly=lambda v:np.polynomial.polynomial.polyval(z,v)
        lever=np.tile(cal.lever_xyz,(len(rows),1));lever[:,2]+=poly(cal.axial)
        cameras=np.zeros((len(rows),25));cameras[:,:3]=np.asarray(cal.pivot_xyz)+np.einsum('mji,mj->mi',rotations,lever)
        cameras[:,3:6]=rvec;f=mode.domain_wh[0]*np.exp(logf);root=np.sqrt(cal.aspect_fx_over_fy)
        K=np.zeros((len(rows),3,3));K[:,0,0]=f*root;K[:,1,1]=f/root;K[:,2,2]=1.
        K[:,0,2]=mode.domain_wh[0]*poly(cal.principal_x_fraction);K[:,1,2]=mode.domain_wh[1]*poly(cal.principal_y_fraction)
        K=mode.A[None]@K;cameras[:,6]=np.log(K[:,0,0]);cameras[:,7]=np.log(K[:,1,1]);cameras[:,8:10]=K[:,:2,2];cameras[:,10]=K[:,0,1]
        cameras[:,11]=poly(cal.k1);cameras[:,12]=poly(cal.k2);cameras[:,13:15]=cal.tangential_p1_p2
        if not np.isfinite(cameras).all():raise ValueError('Nonfinite fixed-head row camera')
        return cameras


class FixedHeadCentreTrajectory:
    def __init__(self,state,q,index,*,readout_index=None):
        self.state=state;self.q=state._q(q).copy();self.base=FixedHeadRowCamera(state,q,index,readout_index=readout_index)
        self.a=state.axis(q);self.unit=state.scene_unit_working_m
        self.columns=np.r_[self.base.columns,state.axis_index].astype(int)
        if len(np.unique(self.columns))!=len(self.columns):raise ValueError('Head separation duplicated')
        self._locations={int(c):i for i,c in enumerate(self.columns)}
        self.cache_key=(*self.base.cache_key,'same-n-b-common-perpendicular',self.a)
    def __getattr__(self,name):return getattr(self.base,name)
    def _kinematics(self,row,nav=False,jacobian=False):
        _,pair,alpha=self.base._sample(row,nav,jacobian)
        angles=(1-alpha)*pair.a[:2]+alpha*pair.b[:2]
        result=self.state.fixed_head.common_perpendicular(angles,self.base.axis_shared,jacobian=jacobian)
        return result,pair,alpha
    def camera25_at_row(self,row,*,geometry_navigation=False):return self.camera15_mapping(row,jacobian=False,geometry_navigation=geometry_navigation)[0]
    def camera25_at_rows(self,rows,*,geometry_navigation=False):
        camera=self.base.camera25_at_rows(rows,geometry_navigation=geometry_navigation).copy()
        offsets=self.base._row_offsets(np.asarray(rows,float),geometry_navigation);left,alpha=self.base._brackets(offsets,geometry_navigation=geometry_navigation)
        a=self.base._views[left];b=self.base._views[left+1];angles=(1-alpha[:,None])*a[:,:2]+alpha[:,None]*b[:,:2]
        install=self.state.fixed_head.installation(self.base.axis_shared);R=_batch_rotations(angles,install)
        h=np.cross(install['n'],np.einsum('mji,j->mi',R,install['b']))
        camera[:,:3]+=(self.a/self.unit)*h;return camera
    def camera15_mapping(self,row,*,jacobian=True,geometry_navigation=False):
        camera,cols,chain=self.base.camera15_mapping(row,jacobian=jacobian,geometry_navigation=geometry_navigation)
        result,pair,alpha=self._kinematics(row,geometry_navigation,jacobian)
        h,H=result if jacobian else(result,None)
        camera=camera.copy();camera[:3]+=(self.a/self.unit)*h
        if not jacobian:return camera,self.columns,None
        full=np.c_[chain,np.r_[h/(100*self.unit),np.zeros(12)]]
        factor=self.a/(100*self.unit)
        for j,col in enumerate(self.state.head_axes_indices):full[:3,self._locations[int(col)]]+=factor*H[:,j]
        for knot,w in((pair.left,1-alpha),(pair.right,alpha)):
            for j,col in enumerate(self.state.view_columns(knot+1)[:2]):full[:3,self._locations[int(col)]]+=factor*w*H[:,4+j]
        d_alpha_ms=.001*(float(row)/1079.-.5)/pair.seconds
        full[:3,self._locations[int(self.readout_index)]]+=factor*(H[:,4:]@(pair.b[:2]-pair.a[:2]))*d_alpha_ms
        return camera,self.columns,full
    def camera15_row_derivative(self,row,*,geometry_navigation=False):
        value=self.base.camera15_row_derivative(row,geometry_navigation=geometry_navigation).copy()
        (h,H),pair,alpha=self._kinematics(row,geometry_navigation,True)
        speed=self.base.readout_milliseconds*.001/(1079.*pair.seconds)
        value[:3]+=(self.a/(100*self.unit))*(H[:,4:]@(pair.b[:2]-pair.a[:2]))*speed
        return value
    def provenance(self):return dict(schema='rinkspace.fixed-pan-tilt-shared-optics-native-RS.v1',
        camera_centre='P + R.T(d+axial(z)*ez) + a*(n cross R.T*b)',
        rotation='Exp(beta*b) R0 Exp(alpha*n)',native_and_RS_pose_are_same_fixed_head_surface=True,
        ordinary_observed_rows_and_native_implicit_query_semantics_unchanged=True,per_frame_PP_or_C_added=False)


def prepare(state,q,index,*,readout_index=None):
    if isinstance(state,TransverseNativePrincipalState):return FixedHeadCentreTrajectory(state,q,index,readout_index=readout_index)
    return optics.prepare(state,q,index,readout_index=readout_index)
def source_chunk(task):return optics.source_chunk(task)
def install_dispatch():
    head.row_api.prepare_frame_row_camera=prepare
    module=sys.modules.get('calibration.program_tracking.row_scene_projection')
    if module is not None:module.prepare_frame_row_camera=prepare
    runtime=sys.modules.get('native_transverse_bundle')
    if runtime is not None and runtime.source_chunk.__module__=='native_transverse_bundle':runtime.source_chunk=source_chunk

def make_unshifted_world_state(state):return optics.make_unshifted_world_state(state.optics_state)
def init_geometry_state(binding,delta):return optics.init_geometry_state(binding,delta)

def _mount_context(state,q):
    from rinkspace.metric_scale import boundary_span,estimate_uniform_scale
    import json
    q=state._q(q);scene=state.scene(q);scale=estimate_uniform_scale(boundary_span(scene));ratio=scale/state.scene_unit_working_m
    facts=json.loads(open(MOUNT_REFERENCE['path'],encoding='utf8').read());box=facts['v4_amendment']['working_mount_range']
    lo=np.asarray(box['lower_camera_xyz_m']);hi=np.asarray(box['upper_camera_xyz_m'])
    cal=state.calibration(q);views=np.asarray([q[state.view_columns(i)]for i in range(1,state.frame_count+1)])
    token=(views.tobytes(),q[state.head_axes_indices].tobytes(),float(cal.q_reference))
    support=state._fixed_head_support_cache.get(token)
    if support is None:
        install=state.fixed_head.installation(q[state.head_axes_indices]);u=install['R0']@install['n'];A=np.cross(u,install['b']);B=u
        vmin=np.full(3,np.inf);vmax=np.full(3,-np.inf);intervals=[]
        for i in range(state.frame_count-1):
            left=-.5 if i==0 else 0.;right=1.5 if i==state.frame_count-2 else 1.
            beta0=views[i,1];delta=views[i+1,1]-beta0
            C=A*np.cos(beta0)+B*np.sin(beta0);D=-A*np.sin(beta0)+B*np.cos(beta0)
            signed_D=D if delta>=0. else-D
            bounds=np.asarray([head._trig_component_bounds(0.,C[j],signed_D[j],abs(delta),left,right)for j in range(3)])
            driver=np.asarray([(1-t)*views[i,3]+t*views[i+1,3]-cal.q_reference for t in(left,right)])
            vmin=np.minimum(vmin,bounds[:,0]);vmax=np.maximum(vmax,bounds[:,1])
            intervals.append(dict(left=i,alpha=[left,right],driver=[float(driver.min()),float(driver.max())],v_bounds=bounds.tolist()))
        support=dict(v_min=vmin,v_max=vmax,intervals=intervals,interval_count=len(intervals),
            actual_fixed_head_angle_intervals=True,all_native_and_terminal_rows_covered=True)
        state._fixed_head_support_cache[token]=support
        while len(state._fixed_head_support_cache)>2:state._fixed_head_support_cache.popitem(last=False)
    value=np.r_[np.asarray(cal.lever_xyz)*state.scene_unit_working_m,q[int(state.original.zoom_parameter_indices[3])],state.axis(q),state.quadratic(q)]
    return q,scale,ratio,support,value,lo,hi
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
        all_native_pan_tilt_angle_intervals_and_terminal_half_intervals_certified=True,
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
def provenance():return dict(schema='rinkspace.fixed-head-angle-RS-shared-optics-candidate.v1',parameter_count=COUNT,
    old_optics_count=BASE_COUNT,optics9_columns=list(range(7674,7683)),shared_axis4_columns=list(range(7683,7687)),
    pose_slots='native RF4 slots now pan,tilt,fixed_zero,logf',native_PP_count=2496,third_pose_fixed_count=1248,
    fixed_head_math=reference(sys.modules[FixedPanTilt.__module__].__file__),shared_optics_producer=reference(optics.__file__),
    unknown_shared_axes_not_fixed_by_standard_chart_origins=True,old_geometry171_unchanged=True,
    original_source_projection_and_whitening_semantics_retained=True)
