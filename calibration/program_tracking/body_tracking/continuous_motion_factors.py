"""Finite priors on actual absolute motion, including every terminal orientation."""
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from .absolute_motion_curve import _log
from .absolute_motion_state import inverse_right_jacobian
from .fullbody_fit import right_jacobian
from .skate_contact import trapezoid_time_weights

def robust_vector(value,delta=3.):
    s=np.sum(value*value,axis=-1);u=np.sqrt(1+s/delta**2);k=np.sqrt(2/(u+1));dk=-k/(4*delta**2*u*(u+1))
    return value*k[...,None],k[...,None,None]*np.eye(3)+2*dk[...,None,None]*value[...,None]*value[...,None,:]

def track_local(blocks):
    n,j=blocks.shape[:2];out=np.zeros((n,j,3,69))
    for k in range(j):out[:,k,:,3+k*3:6+k*3]=blocks[:,k]
    return out.reshape(n,j*3,69)

class ActualMotionFactors:
    def __init__(self,problem,parent):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference;self.times=self.ref.times
        self.source=self.ref.p['pts']/12800.;self.base=self.ref.at(self.times);self.parent=parent.at(self.times)
        self.prior_R=np.concatenate((self.parent['Q'][:,None],self.parent['local_R']),axis=1)
        self.weights=trapezoid_time_weights(self.times);self.last=None
        # These finite engineering scales are not acceptance bands or maximum speeds.
        self.config=dict(pose_prior_weight=.2,pose_prior_sigma_rad=.6,root_prior_sigma_solver=.75,
            actual_rotation_acceleration_scale_rad_s2=80.,root_acceleration_scale_solver_s2=12.,
            transition_weight=.16,stable_weight=1.,right_wrist_outlier_prior_weight=.035,
            horizontal_entry_curvature='finite existing-scale prior; not a COM or contact measurement',
            vertical_entry_prior_relaxation_seconds=[81.30,81.62])
    def evaluate(self,x,jacobian=False):
        d=self.body.state.increments(x,self.times);s=self.ref.world(self.base,d)
        R=np.concatenate((s['Q'][:,None],s['local_R']),axis=1);n=len(R);rows=[];mats=[];parts={}
        def add(name,r,J):
            r=np.asarray(r).ravel();rows.append(r);parts[name]=float(r@r/2)
            if jacobian:mats.append(J)
        weight=np.ones((n,22))*np.sqrt(self.weights[:,None]*self.config['pose_prior_weight'])/.6
        weight[((self.source>=89.24)&(self.source<=89.44)),21]*=np.sqrt(.035)
        error=_log(self.prior_R.swapaxes(-1,-2)@R);z=error*weight[...,None];r,D=robust_vector(z,delta=2.)
        J=None
        if jacobian:
            blocks=D@((weight[...,None,None]*inverse_right_jacobian(error))@right_jacobian(np.asarray(d)[:,3:].reshape(n,22,3)))
            J=d.rotation_chain(blocks)
        add('finite_network_pose_prior',r,J)
        w=np.sqrt(self.weights*.2)/.75;rp=(s['P']-self.parent['P'])*w[:,None]
        if jacobian:
            local=np.zeros((n,3,69));local[:,:,:3]=w[:,None,None]*np.eye(3);J=d.chain(local)
        add('finite_parent_root_prior',rp,J)
        # SO(3) interval velocities are expressed in the same world/parent axes.
        h=np.diff(self.times);rel=R[1:]@R[:-1].swapaxes(-1,-2);phi=_log(rel);omega=phi/h[:,None,None]
        a=2/(h[:-1]+h[1:]);alpha=a[:,None,None]*np.diff(omega,axis=0)
        t=self.source[1:-1];phase=np.where((t>=86.8)&(t<=92.46),1.,.16)
        # Allow genuine changes at takeoff, landing, grasp/release rather than forcing uniform spin.
        phase[((t>=80.8)&(t<=82.4))|((t>=92.4)&(t<=93.2))]=.10
        scale=np.full(22,80.);scale[0]=60.;w=np.sqrt((h[:-1]+h[1:])[:,None]/2*phase[:,None])/scale[None]
        z=alpha*w[...,None];r,D=robust_vector(z,delta=3.)
        if jacobian:
            K=inverse_right_jacobian(phi)@R[:-1]
            T=right_jacobian(np.asarray(d)[:,3:].reshape(n,22,3))
            coeff=[a[:,None,None,None]*K[:-1]@T[:-2]/h[:-1,None,None,None],
                   -a[:,None,None,None]*(K[:-1]/h[:-1,None,None,None]+K[1:]/h[1:,None,None,None])@T[1:-1],
                   a[:,None,None,None]*K[1:]@T[2:]/h[1:,None,None,None]]
            J=csr_matrix((r.size,self.body.size))
            for offset,C in enumerate(coeff):
                C=D@(w[...,None,None]*C)
                J=J+d.rotation_chain(C,sample_indices=np.arange(n-2)+offset)
        add('actual_so3_angular_acceleration',r,J)
        # Actual translation, not derivatives of a correction. No flight-root zero-acceleration target.
        Pacc=self.body.state.basis(self.times,2)@self.body.state.current_curve(x).position_controls
        # The old all-axis hole allowed large camera-ray translation spikes
        # around the entry. Keep a finite horizontal curvature prior there.
        # Vertical relaxation is retained; the confirmed-flight COM factor,
        # rather than a pelvis zero-acceleration target, supplies gravity.
        w=np.repeat((np.sqrt(self.weights)/12.)[:,None],3,axis=1)
        vertical_relax=(self.source>=81.30)&(self.source<=81.62)
        w[vertical_relax,2]=0.;r=Pacc*w
        if jacobian:
            from scipy.sparse import kron,diags
            A=csr_matrix(self.body.state.basis(self.times,2))
            K=kron(A,csr_matrix(np.c_[np.eye(3),np.zeros((3,66))]),format='csr')
            J=diags(w.ravel())@K
        add('actual_root_translation_acceleration',r,J)
        residual=np.concatenate(rows);J=vstack(mats,format='csr') if jacobian else None
        self.last=dict(residual=residual,jac=hstack((csr_matrix((len(residual),17)),J),format='csr') if jacobian else None,parts=parts,
            metrics=dict(actual_root_omega_world_rad_s=omega[:,0].tolist(),actual_right_wrist_interval_degrees=np.rad2deg(np.linalg.norm(phi[:,21],axis=1)).tolist(),
                all_terminal_joint_orientations_regularized=True,finite_prior_config=self.config,no_zero_angular_velocity_target=True))
        return self.last
