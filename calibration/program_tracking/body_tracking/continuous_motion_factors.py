"""Finite priors on actual absolute motion, including every terminal orientation."""
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from .absolute_motion_curve import _log
from .absolute_motion_state import inverse_right_jacobian
from .fullbody_fit import right_jacobian
from .skate_contact import trapezoid_time_weights

def reviewed_cut_weight(reference, times):
    """Increase acceleration weight only around declared live camera edits."""
    repair = getattr(reference,'source_document',{}).get('scoped_candidate_import',{})
    policy = repair.get('camera_cut_continuity')
    weight = np.ones(len(times))
    if policy is None:
        return weight
    pts = reference.p['pts']
    cuts = np.asarray(policy['cut_pts'])
    indices = np.searchsorted(pts, cuts)
    if (cuts.ndim != 1 or cuts.dtype.kind not in 'iu' or not len(cuts)
            or np.any(np.diff(cuts) <= 0) or np.any(indices == 0) or np.any(indices >= len(pts))):
        raise ValueError('Expected ordered camera cut PTS')
    np.testing.assert_array_equal(pts[indices], cuts)
    shots = reference.p['shot_ids']
    if np.any(shots[indices] == shots[indices-1]):
        raise ValueError('Camera cut weighting requires an actual change of shot')
    half = float(policy['half_width_s'])
    multiplier = float(policy['acceleration_weight_multiplier'])
    if not np.isfinite([half,multiplier]).all() or half <= 0 or multiplier < 1:
        raise ValueError('Invalid camera cut acceleration weighting')
    mid = (reference.times[indices-1]+reference.times[indices])*.5
    q = np.clip(np.abs(np.asarray(times)[:,None]-mid)/half,0.,1.)
    return weight + (multiplier-1)*np.max((1.-q*q)**2,axis=1)


def robust_vector(value,delta=3.):
    s=np.sum(value*value,axis=-1);u=np.sqrt(1+s/delta**2);k=np.sqrt(2/(u+1));dk=-k/(4*delta**2*u*(u+1))
    return value*k[...,None],k[...,None,None]*np.eye(3)+2*dk[...,None,None]*value[...,None]*value[...,None,:]

def track_local(blocks):
    n,j=blocks.shape[:2];out=np.zeros((n,j,3,69))
    for k in range(j):out[:,k,:,3+k*3:6+k*3]=blocks[:,k]
    return out.reshape(n,j*3,69)


class ActualSkateAngularAcceleration:
    """Finite acceleration prior on the composed world skate rotations.

    Smooth local hip/knee/ankle tracks can still compose into a sharp skate
    turn. This term evaluates the actual FK product, without a zero-speed
    target, image observations, or inferred contact labels.
    """
    name='actual_skate_world_angular_acceleration'

    def __init__(self,problem,times,scale_rad_s2=80.,feet=(0,1)):
        self.problem=problem;self.times=np.asarray(times,float)
        self.feet=np.asarray(feet,int)
        if not len(self.feet) or np.any((self.feet<0)|(self.feet>1)) or len(np.unique(self.feet))!=len(self.feet):
            raise ValueError('Expected distinct skate indices')
        if len(self.times)<3 or np.any(np.diff(self.times)<=0):
            raise ValueError('Skate acceleration needs increasing physical times')
        self.scale=float(scale_rad_s2)
        if not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('Positive finite skate acceleration scale required')
        self.base=problem.reference.at(self.times)

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import block_diag,diags
        from .skate_physics import fk_state
        from .weighted_blocks import block,full_columns
        p=self.problem;state=p.body.state;delta=state.increments(x,self.times)
        pose=fk_state(p.reference,self.base,delta);R=pose['R'];n=len(R)
        h=np.diff(self.times);relative=R[1:]@R[:-1].swapaxes(-1,-2)
        phi=_log(relative);omega=phi/h[:,None,None]
        a=2/(h[:-1]+h[1:]);alpha=a[:,None,None]*np.diff(omega,axis=0)
        measure=np.sqrt((h[:-1]+h[1:])/2)/self.scale
        take=(np.arange(n-2)[:,None,None]*6+self.feet[None,:,None]*3+np.arange(3)).ravel()
        residual=(alpha*measure[:,None,None]).ravel()[take];J=None
        if jacobian:
            # Spatial tangent of the complete Q * hip * knee * ankle * mount
            # product. Each leg joint affects only its own skate.
            local=np.zeros((n,2,3,69));d=np.asarray(delta)
            root=pose['Q']@right_jacobian(d[:,3:6])
            local[:,:,:,3:6]=root[:,None]
            for foot,joints in enumerate(((1,4,7),(2,5,8))):
                for joint in joints:
                    cols=6+3*(joint-1)+np.arange(3)
                    axes=pose['Q']@pose['frames'][:,joint]@right_jacobian(d[:,cols])
                    local[:,foot,:,cols[0]:cols[-1]+1]=axes
            tangent=delta.chain(local.reshape(n,6,69))
            if hasattr(state,'shoe_chain'):
                shoe=np.zeros((n,6,6))
                shoe[:,:3,:3]=R[:,0];shoe[:,3:,3:]=R[:,1]
                tangent=tangent+state.shoe_chain(shoe,x,self.times)
            inverse=inverse_right_jacobian(phi)
            previous=block_diag(inverse.reshape(-1,3,3),format='csr')
            following=block_diag((inverse@relative.swapaxes(-1,-2)).reshape(-1,3,3),format='csr')
            velocity=diags(np.repeat(1/h,6))@(following@tangent[6:]-previous@tangent[:-6])
            J=(diags(np.repeat(a*measure,6))@(velocity[6:]-velocity[:-6]))[take]
        result=full_columns(block(self.name,residual,J,dict(
            quadrature_times=self.times,quadrature_is_source_observation=False,
            scale_rad_s2=self.scale,zero_angular_velocity_target=False,
            complete_articulated_world_rotation=True,feet=self.feet)))
        return dict(residual=residual,jac=result.jacobian,
            parts={self.name:float(residual@residual/2)},weighted_blocks=[result],
            metrics={self.name:dict(maximum_angular_speed_rad_s=float(np.linalg.norm(omega[:,self.feet],axis=-1).max()),
                maximum_angular_acceleration_rad_s2=float(np.linalg.norm(alpha[:,self.feet],axis=-1).max()))})

class ActualSkateInclinationRate:
    """Finite continuity prior on sole inclination, invariant to world yaw.

    The world-up direction expressed in skate coordinates is R[2, :]. Its
    change detects heel/toe or edge rocking while allowing arbitrary yaw.
    This is an explicitly selected motion prior, not a measured ice contact
    or a flat-foot target. The mean inclination remains image determined.
    """
    name='actual_skate_inclination_rate'

    def __init__(self,problem,times,feet=(0,1),scale_rad_s=1.,fade_seconds=.18):
        self.problem=problem;self.times=np.asarray(times,float)
        self.feet=np.asarray(feet,int);self.scale=float(scale_rad_s)
        if len(self.times)<2 or np.any(np.diff(self.times)<=0):
            raise ValueError('Inclination continuity needs increasing physical times')
        if (not len(self.feet) or np.any((self.feet<0)|(self.feet>1))
                or len(np.unique(self.feet))!=len(self.feet)):
            raise ValueError('Expected distinct skate indices')
        if not np.isfinite([self.scale,fade_seconds]).all() or self.scale<=0 or fade_seconds<0:
            raise ValueError('Invalid inclination prior scale or fade')
        h=np.diff(self.times);mid=(self.times[1:]+self.times[:-1])/2
        taper=np.ones(len(h)) if fade_seconds==0 else np.minimum(
            np.clip((mid-self.times[0])/fade_seconds,0.,1.),
            np.clip((self.times[-1]-mid)/fade_seconds,0.,1.))
        self.measure=np.sqrt(h*taper)/self.scale/h
        self.base=problem.reference.at(self.times)

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import diags
        from .skate_physics import fk_state
        from .weighted_blocks import block,full_columns
        p=self.problem;state=p.body.state;delta=state.increments(x,self.times)
        pose=fk_state(p.reference,self.base,delta);R=pose['R'];n=len(R)
        vertical=R[:,:,2,:]
        take=(np.arange(n-1)[:,None,None]*6+self.feet[None,:,None]*3+np.arange(3)).ravel()
        residual=(np.diff(vertical,axis=0)*self.measure[:,None,None]).ravel()[take]
        J=None
        if jacobian:
            d=np.asarray(delta);axes=np.zeros((n,2,3,69))
            axes[:,:,:,3:6]=(pose['Q']@right_jacobian(d[:,3:6]))[:,None]
            for foot,joints in enumerate(((1,4,7),(2,5,8))):
                for joint in joints:
                    first=6+3*(joint-1);cols=slice(first,first+3)
                    axes[:,foot,:,cols]=pose['Q']@pose['frames'][:,joint]@right_jacobian(d[:,cols])
            # d (ez . R ej) / d spatial rotation = (R ej) cross ez.
            cross=np.stack((R[:,:,1,:],-R[:,:,0,:],np.zeros((n,2,3))),axis=-1)
            local=np.einsum('nfki,nfid->nfkd',cross,axes).reshape(n,6,69)
            tangent=delta.chain(local)
            if hasattr(state,'shoe_chain'):
                shoe=np.zeros((n,6,6))
                for foot in (0,1):
                    shoe[:,foot*3:foot*3+3,foot*3:foot*3+3]=cross[:,foot]@R[:,foot]
                tangent+=state.shoe_chain(shoe,x,self.times)
            J=(diags(np.repeat(self.measure,6))@(tangent[6:]-tangent[:-6]))[take]
        result=full_columns(block(self.name,residual,J,dict(
            quadrature_times=self.times,quadrature_is_source_observation=False,
            scale_rad_s=self.scale,feet=self.feet,world_yaw_unconstrained=True,
            absolute_inclination_target=None,contact_measurement=False)))
        rate=np.diff(vertical[:,self.feet],axis=0)/np.diff(self.times)[:,None,None]
        pitch=np.rad2deg(np.arcsin(np.clip(vertical[:,self.feet,0],-1,1)))
        return dict(residual=residual,jac=result.jacobian,
            parts={self.name:float(residual@residual/2)},weighted_blocks=[result],
            metrics={self.name:dict(maximum_inclination_rate_rad_s=float(np.linalg.norm(rate,axis=-1).max()),
                forward_axis_elevation_degrees_range=[float(pitch.min()),float(pitch.max())])})


class ActualSkateAnkleAcceleration:
    """Finite acceleration prior on each actual world ankle pivot after FK.

    Coupled hip/knee changes can translate a skate despite a stable sole
    orientation. This term constrains that trajectory without setting a
    fixed position, velocity, ice height, or contact point.
    """
    name='actual_skate_ankle_acceleration'

    def __init__(self,problem,times,feet=(0,1),scale_m_s2=8.):
        self.problem=problem;self.times=np.asarray(times,float)
        self.feet=np.asarray(feet,int);self.scale=float(scale_m_s2)
        if len(self.times)<3 or np.any(np.diff(self.times)<=0):
            raise ValueError('Ankle acceleration needs increasing physical times')
        if (not len(self.feet) or np.any((self.feet<0)|(self.feet>1))
                or len(np.unique(self.feet))!=len(self.feet)):
            raise ValueError('Expected distinct skate indices')
        if not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('Positive finite ankle acceleration scale required')
        self.base=problem.reference.at(self.times)

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import diags
        from .skate_physics import fk_state
        from .weighted_blocks import block,full_columns
        p=self.problem;state=p.body.state;delta=state.increments(x,self.times)
        pose=fk_state(p.reference,self.base,delta);n=len(self.times)
        pivot=pose['native_world'][:,[7,8]];ratio=p.binding.ratio
        h=np.diff(self.times);a=2/(h[:-1]+h[1:])
        velocity=np.diff(pivot,axis=0)/h[:,None,None]
        acceleration=a[:,None,None]*np.diff(velocity,axis=0)*ratio
        measure=np.sqrt((h[:-1]+h[1:])/2)/self.scale
        take=(np.arange(n-2)[:,None,None]*6+self.feet[None,:,None]*3+np.arange(3)).ravel()
        residual=(acceleration*measure[:,None,None]).ravel()[take];J=None
        if jacobian:
            d=np.asarray(delta);local=np.zeros((n,2,3,69))
            local[:,:,:,:3]=np.eye(3)
            root=pose['Q']@right_jacobian(d[:,3:6])
            arm=pivot-pose['P'][:,None]
            local[:,:,:,3:6]=np.cross(root[:,None].swapaxes(-1,-2),arm[:,:,None,:]).swapaxes(-1,-2)
            for foot,joints in enumerate(((1,4),(2,5))):
                for joint in joints:
                    first=6+3*(joint-1);cols=slice(first,first+3)
                    axes=pose['Q']@pose['frames'][:,joint]@right_jacobian(d[:,cols])
                    arm=pivot[:,foot]-pose['native_world'][:,joint]
                    local[:,foot,:,cols]=np.cross(axes.swapaxes(-1,-2),arm[:,None,:]).swapaxes(-1,-2)
            position=delta.chain(local.reshape(n,6,69))*ratio
            speed=diags(np.repeat(1/h,6))@(position[6:]-position[:-6])
            J=(diags(np.repeat(a*measure,6))@(speed[6:]-speed[:-6]))[take]
        result=full_columns(block(self.name,residual,J,dict(
            quadrature_times=self.times,quadrature_is_source_observation=False,
            scale_m_s2=self.scale,feet=self.feet,position_or_velocity_target=None,
            contact_measurement=False,physical_scale_is_conditional=True)))
        return dict(residual=residual,jac=result.jacobian,
            parts={self.name:float(residual@residual/2)},weighted_blocks=[result],
            metrics={self.name:dict(maximum_ankle_acceleration_conditional_m_s2=
                float(np.linalg.norm(acceleration[:,self.feet],axis=-1).max()))})


class ActualMotionFactors:
    def __init__(self,problem,parent,*,substeps=1,rotation_acceleration_loss='pseudo_huber'):
        self.problem=problem;self.body=problem.body;self.ref=problem.reference
        if rotation_acceleration_loss not in ('pseudo_huber','quadratic'):
            raise ValueError('Unknown rotation acceleration loss')
        if not isinstance(substeps,(int,np.integer)) or substeps<1:
            raise ValueError('Motion quadrature requires a positive integer subdivision')
        native_times=np.asarray(self.ref.times)
        fractions=np.arange(substeps,dtype=float)/substeps
        self.times=np.r_[(native_times[:-1,None]+np.diff(native_times)[:,None]*fractions).ravel(),native_times[-1]]
        from .body_clock import source_seconds
        self.source=np.interp(self.times,native_times,source_seconds(self.ref.p))
        self.sample_pts=np.interp(self.times,native_times,self.ref.p['pts'])
        self.base=self.ref.at(self.times);self.parent=parent.at(self.times)
        self.prior_R=np.concatenate((self.parent['Q'][:,None],self.parent['local_R']),axis=1)
        self.weights=trapezoid_time_weights(self.times);self.last=None
        # These finite engineering scales are not acceptance bands or maximum speeds.
        self.config=dict(pose_prior_weight=.2,pose_prior_sigma_rad=.6,root_prior_sigma_solver=.75,
            actual_rotation_acceleration_scale_rad_s2=80.,root_acceleration_scale_solver_s2=12.,
            transition_weight=.16,stable_weight=1.,right_wrist_outlier_prior_weight=.035,
            horizontal_entry_curvature='finite existing-scale prior; not a COM or contact measurement',
            vertical_entry_prior_relaxation_seconds=[81.30,81.62],
            quadrature_substeps=int(substeps),quadrature_samples=len(self.times),
            quadrature_is_source_observation=False,
            rotation_acceleration_loss=rotation_acceleration_loss)
        policy=getattr(self.ref,'source_document',{}).get('root_prior_policy',{})
        confidence=float(policy.get('camera_ray_depth_confidence',1.))
        if not np.isfinite(confidence)or not 0<confidence<=1:raise ValueError('Finite camera-ray depth confidence required')
        self.root_transform=np.broadcast_to(np.eye(3),(len(self.times),3,3)).copy()
        if confidence!=1.:
            centres=np.asarray(self.body.camera_centres)
            if substeps!=1:
                centres=np.column_stack([np.interp(self.times,native_times,centres[:,k]) for k in range(3)])
            ray=self.parent['P']-centres;ray/=np.maximum(np.linalg.norm(ray,axis=1),1e-12)[:,None]
            self.root_transform+=(np.sqrt(confidence)-1)*ray[:,:,None]*ray[:,None,:]
        self.config['camera_ray_depth_confidence']=confidence
    def evaluate(self,x,jacobian=False):
        from .fullbody_state import matrices
        d=self.body.state.increments(x,self.times)
        pose=np.asarray(d)
        # These priors consume the pose before FK and selected skinning.
        s=dict(P=self.base['P']+pose[:,:3],
               Q=self.base['Q']@matrices(pose[:,3:6]),
               local_R=self.base['local_R']@matrices(pose[:,6:].reshape(-1,21,3)))
        R=np.concatenate((s['Q'][:,None],s['local_R']),axis=1);n=len(R);rows=[];mats=[];parts={};named=[]
        def add(name,r,J):
            r=np.asarray(r).ravel();rows.append(r);parts[name]=float(r@r/2)
            if jacobian and not getattr(self.problem,'bounded_memory',False):mats.append(J)
            from .weighted_blocks import block,full_columns
            named.append(full_columns(block(name,r,J,dict(source_pts=self.ref.p['pts'],
                quadrature_source_seconds=self.source,quadrature_is_source_observation=False))))
        weight=np.ones((n,22))*np.sqrt(self.weights[:,None]*self.config['pose_prior_weight'])/.6
        generic=getattr(self.ref,'source_document',{}).get('generic_context',False)
        if not generic:weight[((self.source>=89.24)&(self.source<=89.44)),21]*=np.sqrt(.035)
        error=_log(self.prior_R.swapaxes(-1,-2)@R);z=error*weight[...,None];r,D=robust_vector(z,delta=2.)
        J=None
        if jacobian:
            pose_blocks=D@((weight[...,None,None]*inverse_right_jacobian(error))@right_jacobian(np.asarray(d)[:,3:].reshape(n,22,3)))
        add('finite_network_pose_prior',r,J)
        w=np.sqrt(self.weights*.2)/.75;rp=np.einsum('nij,nj->ni',self.root_transform,s['P']-self.parent['P'])*w[:,None]
        if jacobian:
            root_local=np.zeros((n,3,69));root_local[:,:,:3]=w[:,None,None]*self.root_transform
        add('finite_parent_root_prior',rp,J)
        # SO(3) interval velocities are expressed in the same world/parent axes.
        h=np.diff(self.times);rel=R[1:]@R[:-1].swapaxes(-1,-2);phi=_log(rel);omega=phi/h[:,None,None]
        a=2/(h[:-1]+h[1:]);alpha=a[:,None,None]*np.diff(omega,axis=0)
        t=self.source[1:-1];phase=np.where((t>=86.8)&(t<=92.46),1.,.16)
        # Allow genuine changes at takeoff, landing, grasp/release rather than forcing uniform spin.
        phase[((t>=80.8)&(t<=82.4))|((t>=92.4)&(t<=93.2))]=.10
        if generic:
            from .phase_ledger import phase_weights
            phase=phase_weights(self.ref,self.sample_pts[1:-1],default=.16)
        phase*=reviewed_cut_weight(self.ref,self.times[1:-1])
        scale=np.full(22,80.);scale[0]=60.;w=np.sqrt((h[:-1]+h[1:])[:,None]/2*phase[:,None])/scale[None]
        z=alpha*w[...,None]
        if self.config['rotation_acceleration_loss']=='quadratic':
            r=z;D=np.broadcast_to(np.eye(3),(*z.shape[:-1],3,3))
        else:
            r,D=robust_vector(z,delta=3.)
        if jacobian:
            K=inverse_right_jacobian(phi)@R[:-1]
            T=right_jacobian(np.asarray(d)[:,3:].reshape(n,22,3))
            coeff=[a[:,None,None,None]*K[:-1]@T[:-2]/h[:-1,None,None,None],
                   -a[:,None,None,None]*(K[:-1]/h[:-1,None,None,None]+K[1:]/h[1:,None,None,None])@T[1:-1],
                   a[:,None,None,None]*K[1:]@T[2:]/h[1:,None,None,None]]
            pose_tiles=[];root_tiles=[];acceleration_tiles=[]
            state=self.body.state
            # One tile plus its two-sample acceleration halo owns the mapping.
            # Nothing is cached across x, clocks, seed/layout or binding changes.
            for first in range(0,n,512):
                last=min(first+512,n);stop=min(last+2,n)
                _,sample,mapping=state.linearization(x,self.times[first:stop])
                pick=np.arange(last-first)
                pose_tiles.append(state._rotation_chain_with_mapping(pose_blocks[first:last],sample,mapping,pick))
                root_tiles.append(state._chain_with_mapping(root_local[first:last],sample,mapping,pick))
                end=min(last,n-2)
                if end>first:
                    piece=csr_matrix(((end-first)*22*3,self.body.size))
                    for offset,C in enumerate(coeff):
                        blocks=D[first:end]@(w[first:end,...,None,None]*C[first:end])
                        piece=piece+state._rotation_chain_with_mapping(blocks,sample,mapping,np.arange(end-first)+offset)
                    acceleration_tiles.append(piece)
            from .weighted_blocks import block,full_columns
            for i,tiles in enumerate((pose_tiles,root_tiles)):
                matrix=vstack(tiles,format='csr');tiles.clear()
                b=named[i];b.jacobian=full_columns(block(b.name,b.residual,matrix,b.provenance,b.association)).jacobian
                if mats:mats[i]=matrix
            J=vstack(acceleration_tiles,format='csr') if acceleration_tiles else csr_matrix((0,self.body.size))
            acceleration_tiles.clear()
        add('actual_so3_angular_acceleration',r,J)
        if getattr(self.ref,'source_document',{}).get('trajectory_reference')!='model_com':
            # Actual translation, not derivatives of a correction. No flight-root zero-acceleration target.
            Pacc=self.body.state.basis(self.times,2)@self.body.state.current_curve(x).position_controls
            # The old all-axis hole allowed large camera-ray translation spikes
            # around the entry. Keep a finite horizontal curvature prior there.
            # Vertical relaxation is retained; the confirmed-flight COM factor,
            # rather than a pelvis zero-acceleration target, supplies gravity.
            w=np.repeat((np.sqrt(self.weights)/12.)[:,None],3,axis=1)
            vertical_relax=(self.source>=81.30)&(self.source<=81.62)
            if generic:
                from .phase_ledger import flight_mask
                vertical_relax=flight_mask(self.ref,self.sample_pts)
            w[vertical_relax,2]=0.;r=Pacc*w
            if jacobian:
                from scipy.sparse import kron,diags
                A=csr_matrix(self.body.state.basis(self.times,2))
                K=kron(A,csr_matrix(np.c_[np.eye(3),np.zeros((3,66))]),format='csr')
                J=diags(w.ravel())@K
                if J.shape[1]<self.body.size:J=hstack((J,csr_matrix((J.shape[0],self.body.size-J.shape[1]))),format='csr')
            add('actual_root_translation_acceleration',r,J)
        residual=np.concatenate(rows);J=vstack(mats,format='csr') if mats else None
        self.last=dict(residual=residual,jac=hstack((csr_matrix((len(residual),17)),J),format='csr') if J is not None else None,parts=parts,weighted_blocks=named,
            metrics=dict(actual_root_omega_world_rad_s=omega[:,0].tolist(),actual_right_wrist_interval_degrees=np.rad2deg(np.linalg.norm(phi[:,21],axis=1)).tolist(),
                all_terminal_joint_orientations_regularized=True,finite_prior_config=self.config,no_zero_angular_velocity_target=True))
        return self.last
