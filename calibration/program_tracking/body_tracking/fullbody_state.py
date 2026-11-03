"""One continuous body state, with frozen J1/B5 corrections folded exactly once.

Computations retain the input's declared length unit (60 historical metres per
scene unit). Export/display multiply *all* length arrays by the camera binding's
uniform ratio. This conversion is not a fit or an accuracy improvement.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation, Slerp
from .rebuild_prepare import load, digest
from .rebuild_body_geometry import CompactBody, POSE_DOFS
from .footwear_geometry import LEG_DOFS, extra_rotations, equipment_world


def matrices(v):
    v=np.asarray(v,float)
    return Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(*v.shape[:-1],3,3)


class FrozenReference:
    def __init__(self,root):
        self.root=Path(root);self.p=load(self.root/'input.npz')
        self.base_path=self.root/'next_round_20261029/B5/result.npz'
        self.joint_path=self.root/'next_round_20261029/J1_s2/result.npz'
        self.base=load(self.base_path);self.joint=load(self.joint_path)
        for z in (self.base,self.joint):np.testing.assert_array_equal(z['pts'],self.p['pts'])
        self.body=CompactBody(self.root/'compact_smplx.npz',self.p['shared_betas'],float(self.p['body_scale']))
        self.times=self.p['fit_time_seconds'];self.geometry=self.joint['shared_geometry'].copy()
        self.pose_slerps=[Slerp(self.times,Rotation.from_rotvec(self.p['pose_axis_angle'][:,j])) for j in range(21)]
        self.qslerp=Slerp(self.times,Rotation.from_matrix(self.p['raw_root_prior_Q_world']))
        self.m=len(self.base['position_knots'])-4
        controls=self.base['x'];self.position=BSpline(self.base['position_knots'],controls[:self.m*3].reshape(self.m,3),3)
        self.orientation=BSpline(self.base['position_knots'],controls[self.m*3:self.m*6].reshape(self.m,3),3)
        self.b9=BSpline(self.base['position_knots'],controls[self.m*6:].reshape(self.m,9),3)
        self.legs=BSpline(self.joint['correction_knots'],self.joint['leg_controls'],3)
        self.root_delta=BSpline(self.joint['correction_knots'],self.joint['root_delta_controls'],3)
        self.hashes={str(f):digest(f) for f in (self.root/'input.npz',self.root/'compact_smplx.npz',self.base_path,self.joint_path)}

    def at(self,t):
        t=np.atleast_1d(t).astype(float);clipped=np.clip(t,self.times[0],self.times[-1])
        local=np.stack([s(clipped).as_matrix() for s in self.pose_slerps],axis=1)
        b9=self.b9(t)
        for d,(j,axis) in enumerate(POSE_DOFS):
            a=np.zeros((len(t),3));a[:,axis]=b9[:,d]
            local[:,j-1]=local[:,j-1]@matrices(a)
        legs=extra_rotations(self.legs(t))
        for j,angle in legs.items():local[:,j-1]=local[:,j-1]@matrices(angle)
        return dict(P=self.position(t)+self.root_delta(t),
                    Q=matrices(self.orientation(t))@self.qslerp(clipped).as_matrix(),local_R=local,times=t)

    def world(self,reference,delta=None):
        n=len(reference['P']);delta=np.zeros((n,69)) if delta is None else np.asarray(delta)
        if delta.shape!=(n,69):raise ValueError('State increment must be P(3), Q(3), 21 local rotvecs(63)')
        P=reference['P']+delta[:,:3]
        Q=reference['Q']@matrices(delta[:,3:6])
        local=reference['local_R']@matrices(delta[:,6:].reshape(n,21,3))
        coco,native,_,frames=self.body.forward_rotations(local,with_frames=True)
        gear,_,boot_R,boot_O=equipment_world(P,Q,native,frames,self.geometry)
        coco_world=P[:,None]+np.einsum('nij,nkj->nki',Q,coco)
        native_world=P[:,None]+np.einsum('nij,nkj->nki',Q,native)
        return dict(P=P,Q=Q,local_R=local,native=native_world,coco=coco_world,
                    gear=gear,boot_R=boot_R,boot_O=boot_O,frames=frames,
                    world=np.concatenate((gear.reshape(n,96,3),coco_world),axis=1))


class WindowState:
    """C2 attachment to the unchanged reference, local cubic spline support."""
    def __init__(self,reference,seconds,mode='full',spacing=.4,taper=.2):
        self.reference=reference;self.seconds=tuple(map(float,seconds));self.mode=mode
        self.bounds_time=np.asarray(seconds)-reference.p['pts'][0]/12800.
        a,b=self.bounds_time;self.taper=float(taper)
        inner=np.arange(a+spacing,b-1e-9,spacing)
        self.knots=np.r_[[a]*4,inner,[b]*4];self.m=len(self.knots)-4
        active=[0,1,2]+[6+(j-1)*3+axis for j,axis in LEG_DOFS]
        if mode=='full':
            active += [3,4,5]+[6+(j-1)*3+axis for j in (3,6,9,16,17,18,19) for axis in range(3)]
        elif mode!='legacy':raise ValueError(mode)
        self.active=np.asarray(active,int);self.d=len(active);self.size=self.m*self.d
        self._basis_spline=BSpline(self.knots,np.eye(self.m),3,extrapolate=False)

    def basis(self,t,derivative=0):
        t=np.atleast_1d(t).astype(float);a,b=self.bounds_time
        inside=(t>=a)&(t<=b);tc=np.clip(t,a,b)
        u=np.clip(np.minimum(tc-a,b-tc)/self.taper,0,1)
        envelope=6*u**5-15*u**4+10*u**3
        sign=np.where(tc<(a+b)/2,1.,-1.)
        first=(30*u**4-60*u**3+30*u**2)*sign/self.taper
        second=(120*u**3-180*u**2+60*u)/self.taper**2
        if derivative==0:value=self._basis_spline(tc)*envelope[:,None]
        elif derivative==1:value=self._basis_spline(tc,1)*envelope[:,None]+self._basis_spline(tc)*first[:,None]
        elif derivative==2:value=self._basis_spline(tc,2)*envelope[:,None]+2*self._basis_spline(tc,1)*first[:,None]+self._basis_spline(tc)*second[:,None]
        else:raise ValueError(derivative)
        value[~inside]=0
        return value

    def increments(self,x,t):
        out=np.zeros((len(np.atleast_1d(t)),69))
        out[:,self.active]=self.basis(t)@np.asarray(x).reshape(self.m,self.d)
        return out

    def state_at(self,x,t,reference=None):
        reference=self.reference.at(t) if reference is None else reference
        return self.reference.world(reference,self.increments(x,t))


def camera_binding(project,reference,record_path=None):
    """Use the selected frozen native camera, without re-running preparation."""
    from rinkspace.core.native import NativeCore
    from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
    from rinkspace.metric_scale import estimate_uniform_scale
    from .rebuild_body_fit import FixedNativeProjector
    project=Path(project)
    record_path=Path(record_path or project/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.json')
    record=json.loads(record_path.read_text(encoding='utf8'))
    state=load(record['state']['compact_path']);dep=json.loads(str(state['joint_numeric_dependencies_json']))
    shot=next(s for s in dep['shots'] if s['id']=='s01')
    timeline=ShotMotionTimeline(dep['shots'],dep['control_dt'],dep['first_control_time'])
    p=dict(reference.p)
    # Keep the complete packed control array and its explicit time translation.
    p.update(rig=state['rig'][shot['camera_index']],controls=state['controls'],
        first_control_time=np.array(timeline.first),control_dt=np.array(timeline.dt),
        motion_times=timeline.exposure_time(shot,p['pts']))
    library=record['native_library']['path'];projector=FixedNativeProjector(p,library)
    pose=projector.core.evaluate_motion_camera(p['rig'],p['controls'],timeline.first,timeline.dt,p['motion_times'])
    if pose['status'].any():raise ValueError('Current camera failed at a source midpoint')
    scale=estimate_uniform_scale(2*state['board'][:2])
    info=dict(record=str(record_path),state_path=record['state']['compact_path'],
        state_sha256=digest(record['state']['compact_path']),native_library=library,native_library_sha256=digest(library),
        shot=shot,metres_per_scene_unit=scale,solver_metres_per_scene_unit=float(p['metres_per_scene_unit']),
        length_export_ratio=scale/float(p['metres_per_scene_unit']),camera_changed_during_fit=False,
        source_time_base=[1,12800],old_reference_reprojected_on_current_camera=True,
        prepared=record['prepared'])
    return projector,info,state,pose
