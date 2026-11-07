"""Model-calibrated anatomy on the shared FK and attached-skate state."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.transform import Rotation

from .fullbody_state import matrices
from .memory_bounded import slice_delta,slice_reference
from .sequence_skate_state import perturb_relative
from .skate_contact import trapezoid_time_weights
from .skate_physics import fk_state
from .weighted_blocks import block, full_columns, signature


@dataclass(frozen=True)
class Capsule:
    joint: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    skin_group: tuple


@dataclass(frozen=True)
class SkateBox:
    foot: int
    centre: np.ndarray
    half: np.ndarray
    triangles: np.ndarray


def skin_assignment(surface):
    weights = surface.weights[:, :22].copy()
    weights[:, 15] += surface.weights[:, 22:25].sum(axis=1)
    weights[:, 20] += surface.weights[:, 25:40].sum(axis=1)
    weights[:, 21] += surface.weights[:, 40:55].sum(axis=1)
    return weights.argmax(axis=1)


def make_capsule(surface, labels, joint, endpoint, groups, ratio):
    scale = surface.body.scale*ratio
    origin = surface.body.J[joint]
    offsets = (surface.shaped[np.isin(labels,groups)]-origin)*scale
    direction = np.asarray(endpoint)*scale
    length = np.linalg.norm(direction)
    axis = direction/length
    longitudinal = offsets@axis
    radial = np.linalg.norm(offsets-longitudinal[:,None]*axis,axis=-1)
    # Profile the central section of this specific shaped mesh. Endpoints
    # stay within the part, away from overlapping adjacent joint caps.
    lo, hi = np.quantile(longitudinal, [.23,.78])
    lo, hi = np.clip([lo,hi],0,length)
    in_part = (longitudinal>=lo)&(longitudinal<=hi)
    core = radial[in_part] if np.any(in_part) else radial
    radius = float(np.quantile(core,.72))
    return Capsule(joint,lo*axis,hi*axis,radius,tuple(groups))


def anatomy_geometry(reference):
    surface=reference.binding.surface
    labels=skin_assignment(surface)
    joints=surface.body.J
    ratio=reference.binding.ratio
    capsules={}
    for side, ids in [('left',(1,4,7,16,18,20)),('right',(2,5,8,17,19,21))]:
        hip,knee,ankle,shoulder,elbow,wrist=ids
        for name,start,end in [('thigh',hip,knee),('shank',knee,ankle),
                               ('upper_arm',shoulder,elbow),('forearm',elbow,wrist)]:
            capsules[side+'_'+name]=make_capsule(surface,labels,start,joints[end]-joints[start],[start],ratio)
        palm=(surface.shaped[labels==wrist].mean(axis=0)-joints[wrist])*2
        capsules[side+'_palm']=make_capsule(surface,labels,wrist,palm,[wrist],ratio)
    capsules['torso']=make_capsule(surface,labels,0,joints[12]-joints[0],[0,3,6,9,12],ratio)
    packet=reference.binding.packet()
    template=packet['template']
    tri=template['vertices'][template['faces']]*ratio
    parts=np.asarray(template['face_parts'])
    shell=np.array(['outsole' in part or 'heel_block' in part for part in parts])
    upper=packet['upper'][reference.binding.faces]*ratio
    boot=np.concatenate((tri[shell],upper))
    blade=tri[~shell]
    split=np.mean([blade[...,0].min(),blade[...,0].max()])
    rear=blade[...,0].mean(axis=1)<split
    boxes={}
    for foot,side in enumerate(('left','right')):
        for name,triangles in [('boot',boot),('blade_rear',blade[rear]),('blade_front',blade[~rear])]:
            low=triangles.min(axis=(0,1))
            high=triangles.max(axis=(0,1))
            boxes[side+'_'+name]=SkateBox(foot,.5*(low+high),.5*(high-low),triangles)
    return capsules,boxes,labels


def closest_segments_batch(a,b,c,d):
    """Closest points, including endpoint and parallel branches."""
    u=b-a
    v=d-c
    w=a-c
    uu=np.sum(u*u,axis=-1)
    vv=np.sum(v*v,axis=-1)
    uv=np.sum(u*v,axis=-1)
    uw=np.sum(u*w,axis=-1)
    vw=np.sum(v*w,axis=-1)
    denom=uu*vv-uv*uv
    safe=np.maximum(denom,1e-30)
    s=(uv*vw-vv*uw)/safe
    t=(uu*vw-uv*uw)/safe
    zero=np.zeros_like(s)
    one=np.ones_like(s)
    S=np.stack((s,zero,one,np.clip(-uw/np.maximum(uu,1e-30),0,1),
                np.clip((uv-uw)/np.maximum(uu,1e-30),0,1)),axis=1)
    T=np.stack((t,np.clip(vw/np.maximum(vv,1e-30),0,1),
                np.clip((vw+uv)/np.maximum(vv,1e-30),0,1),zero,one),axis=1)
    delta=w[:,None]+S[...,None]*u[:,None]-T[...,None]*v[:,None]
    sq=np.sum(delta*delta,axis=-1)
    valid=(denom>1e-20)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    sq[:,0]=np.where(valid,sq[:,0],np.inf)
    choice=np.argmin(sq,axis=1)
    i=np.arange(len(a))
    return delta[i,choice],S[i,choice],T[i,choice]


def capsule_box_separation(a,b,centre,R,half):
    """Sample the line at box-face crossings, interval minima and endpoints."""
    endpoints=np.einsum('nki,nij->nkj',np.stack((a-centre,b-centre),axis=1),R)
    lo=endpoints[:,0]
    v=endpoints[:,1]-lo
    denom=np.where(np.abs(v)>1e-14,v,np.inf)
    cuts=np.concatenate((np.zeros((len(a),1)),np.ones((len(a),1)),
                         np.clip((-half-lo)/denom,0,1),np.clip((half-lo)/denom,0,1)),axis=1)
    cuts.sort(axis=1)
    middle=.5*(cuts[:,:-1]+cuts[:,1:])
    q=lo[:,None]+middle[...,None]*v[:,None]
    outside=np.abs(q)>half
    edge=np.sign(q)*half
    quadratic=np.sum((v[:,None]*outside)**2,axis=-1)
    stationary=np.sum((edge-lo[:,None])*v[:,None]*outside,axis=-1)/np.maximum(quadratic,1e-30)
    stationary=np.clip(stationary,cuts[:,:-1],cuts[:,1:])
    times=np.concatenate((cuts,middle,stationary),axis=1)
    q=np.abs(lo[:,None]+times[...,None]*v[:,None])-half
    sdf=np.linalg.norm(np.maximum(q,0.),axis=-1)+np.minimum(np.max(q,axis=-1),0.)
    return np.min(sdf,axis=1)


def box_separation(a,A,ha,b,B,hb):
    """Signed separating-axis margin of the two oriented boxes."""
    axesA=A.swapaxes(-1,-2)
    axesB=B.swapaxes(-1,-2)
    cross=np.cross(axesA[:,:,None,:],axesB[:,None,:,:]).reshape(-1,9,3)
    axes=np.concatenate((axesA,axesB,cross),axis=1)
    norm=np.linalg.norm(axes,axis=-1)
    axes=axes/np.maximum(norm[...,None],1e-14)
    margin=np.abs(np.einsum('ni,nki->nk',b-a,axes))-np.abs(axes@A)@ha-np.abs(axes@B)@hb
    return np.max(np.where(norm>1e-10,margin,-np.inf),axis=1)


def hinge_angles(matrices, axis):
    q=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat().reshape(*matrices.shape[:-2],4)
    axial=np.sum(q[...,:3]*axis,axis=-1)
    signed=2*np.arctan2(axial,q[...,3])
    signed=(signed+np.pi)%(2*np.pi)-np.pi
    swing=2*np.arccos(np.clip(np.hypot(axial,q[...,3]),0,1))
    return signed,swing


class ProgrammeAnatomy:
    """Finite envelope and signed local-joint terms, evaluated on one curve."""
    def __init__(self,problem,document):
        self.problem=problem
        self.ref=problem.reference
        self.state=problem.body.state
        self.ratio=self.ref.binding.ratio
        self.native_times=self.ref.times
        self.times=np.sort(np.r_[self.native_times,.5*(self.native_times[1:]+self.native_times[:-1])])
        self.base=self.ref.at(self.times)
        self.weights=np.sqrt(trapezoid_time_weights(self.times))
        self.capsules,self.boxes,self.labels=anatomy_geometry(self.ref)
        self.pairs=[('left_shank','right_shank'),('left_thigh','right_thigh'),
                    ('left_thigh','right_shank'),('right_thigh','left_shank'),
                    ('left_boot','right_boot')]
        for side,other in [('left','right'),('right','left')]:
            self.pairs.extend((side+'_'+part,'torso') for part in ('forearm','palm'))
            self.pairs.extend((side+'_'+part,other+'_'+leg)
                for part in ('forearm','palm') for leg in ('thigh','shank'))
            self.pairs.extend([(side+'_boot',other+'_shank'),(side+'_boot',other+'_thigh')])
            self.pairs.extend((side+'_'+part,leg)
                for part in ('blade_front','blade_rear')
                for leg in (side+'_shank',other+'_shank',other+'_thigh',other+'_boot'))
            self.pairs.extend((side+'_palm',shoe)
                for shoe in (side+'_boot',other+'_boot',side+'_blade_front',side+'_blade_rear',
                             other+'_blade_front',other+'_blade_rear'))
        self.pairs.extend(('left_'+a,'right_'+b)
                          for a in ('blade_rear','blade_front') for b in ('blade_rear','blade_front'))
        parents=self.ref.body.parents
        ancestry={}
        for name,part in {**self.capsules,**self.boxes}.items():
            joint=part.joint if isinstance(part,Capsule) else (7,8)[part.foot]
            chain=set()
            while joint>0:
                chain.add(joint)
                joint=parents[joint]
            ancestry[name]=chain
        self.pair_joint_dependencies=np.asarray([
            [joint in (ancestry[a]|ancestry[b]) for joint in range(1,22)]
            for a,b in self.pairs])
        self.pair_shoe_dependencies=np.asarray([
            [any(name in self.boxes and self.boxes[name].foot==foot for name in (a,b))
             for foot in (0,1)] for a,b in self.pairs])
        grasp=getattr(self.ref,'grasp_relations',None) or {}
        grip=grasp.get('grip_interval',{})
        self.left_rear_grip_seconds=None
        if grip.get('hand')=='left' and grip.get('foot')==0:
            tick=np.asarray(grasp.get('time_base',self.ref.p['time_base']),dtype=float)
            self.left_rear_grip_seconds=np.asarray([grip['start_pts'],grip['end_pts']])*tick[0]/tick[1]
        self.hinge_joints=np.array([4,5,18,19])
        J=self.ref.body.J
        axes=[]
        for knee,ankle,toe in ((4,7,10),(5,8,11)):
            axes.append(np.cross(J[ankle]-J[knee],-(J[toe]-J[ankle])))
        for elbow,wrist in ((18,20),(19,21)):
            axes.append(np.cross(J[wrist]-J[elbow],[0.,0.,1.]))
        self.axes=np.asarray(axes)
        self.axes/=np.linalg.norm(self.axes,axis=1,keepdims=True)
        self.hinge_low=np.deg2rad([-18.,-18.,-24.,-24.])
        self.hinge_high=np.deg2rad([170.,170.,170.,170.])
        self.swing=np.deg2rad([42.,42.,65.,65.])
        self.hinge_scale=np.deg2rad([24.,24.,37.,37.])
        self.nearest_source=self.ref.p['pts'][np.minimum(np.arange(len(self.times))//2,len(self.native_times)-1)]
        self.origin_seconds=float(self.ref.p['pts'][0]*self.ref.p['time_base'][0]/self.ref.p['time_base'][1]-self.native_times[0])

    def world_geometry(self,state,names):
        parts={}
        bounds={}
        for name,c in self.capsules.items():
            if name not in names:continue
            m=state['Q']@state['frames'][:,c.joint]
            origin=state['native_world'][:,c.joint]*self.ratio
            a=origin+m@c.start
            b=origin+m@c.end
            parts[name]=(a,b,c.radius)
            bounds[name]=(np.minimum(a,b)-c.radius,np.maximum(a,b)+c.radius)
        for name,b in self.boxes.items():
            if name not in names:continue
            axes=state['R'][:,b.foot]
            centre=state['O'][:,b.foot]*self.ratio+axes@b.centre
            extents=np.abs(axes)@b.half
            parts[name]=(centre,axes,b.half)
            bounds[name]=(centre-extents,centre+extents)
        return parts,bounds

    def collision_values(self,state,samples,requested=None):
        use=np.arange(len(self.pairs)) if requested is None else np.flatnonzero(requested.any(axis=0))
        parts,bounds=self.world_geometry(state,{name for p in use for name in self.pairs[p]})
        values=np.zeros((len(samples),len(self.pairs)))
        singular=0
        broad_count=0
        for p in use:
            a,b=self.pairs[p]
            if requested is None:
                l1,u1=bounds[a]
                l2,u2=bounds[b]
                mask=np.all((l1<=u2)&(l2<=u1),axis=1)
            else:
                mask=requested[:,p]
            take=np.flatnonzero(mask)
            if not len(take):
                continue
            broad_count+=len(take)
            ac,ar,ah=parts[a]
            bc,br,bh=parts[b]
            ca=a in self.capsules
            cb=b in self.capsules
            if ca and cb:
                delta,_,_=closest_segments_batch(ac[take],ar[take],bc[take],br[take])
                distance=np.linalg.norm(delta,axis=-1)
                # Branch only at numerical centreline coincidence. The local
                # side axis supplies a nonzero, repeatable separating tangent.
                coincident=distance<1e-6
                if np.any(coincident):
                    singular+=int(np.count_nonzero(coincident))
                    direction=state['Q'][take[coincident],:,0]
                    distance[coincident]=np.sum(delta[coincident]*direction,axis=-1)
                raw=ah+bh-distance
                slack=.12*min(ah,bh)
            elif ca or cb:
                if not ca:
                    ac,ar,ah,bc,br,bh=bc,br,bh,ac,ar,ah
                raw=ah-capsule_box_separation(ac[take],ar[take],bc[take],br[take],bh)
                slack=.22*min(ah,float(np.min(bh)))
            else:
                raw=-box_separation(ac[take],ar[take],ah,bc[take],br[take],bh)
                slack=.16*min(float(np.min(ah)),float(np.min(bh)))
            depth=np.maximum(raw-slack,0.)
            # Only the observed left palm/rear-web pairing during the S05
            # grip is allowed. The rest of the hand, blade and boot stay active.
            if self.left_rear_grip_seconds is not None and {a,b}=={'left_palm','left_blade_rear'}:
                t=self.times[samples[take]]+self.origin_seconds
                start,end=self.left_rear_grip_seconds
                allowed=(t>=start)&(t<=end)
                depth[allowed]=0.
            values[take,p]=depth
        return values,broad_count,singular

    def collision(self,x,jacobian):
        d=self.state.increments(x,self.times)
        state=fk_state(self.ref,self.base,d)
        ids=np.arange(len(self.times))
        depth,broad,singular=self.collision_values(state,ids)
        weight=self.weights/.055
        residual=(depth*weight[:,None]).ravel()
        J=None
        if jacobian:
            selected,pairs=np.nonzero(depth>0.)
            active,lookup=np.unique(selected,return_inverse=True)
            n=len(selected)
            if n:
                times=self.times[active]
                d=self.state.increments(x,times)
                base=self.ref.at(times)
                step=1e-6
                local=np.zeros((n,1,69))
                shoe=np.zeros((n,1,6))
                for target,dependency,offset in ((local,self.pair_joint_dependencies,6),
                                                  (shoe,self.pair_shoe_dependencies,0)):
                    for group in range(dependency.shape[1]):
                        rows=np.flatnonzero(dependency[pairs,group])
                        if not len(rows):continue
                        samples,reindex=np.unique(lookup[rows],return_inverse=True)
                        mask=np.zeros((len(samples),len(self.pairs)),bool)
                        mask[reindex,pairs[rows]]=True
                        sub=slice_delta(d,samples)
                        subbase=slice_reference(base,samples,len(active))
                        for axis in range(3):
                            column=offset+3*group+axis
                            if offset:
                                plus,minus=sub.copy(),sub.copy()
                                plus[:,column]+=step
                                minus[:,column]-=step
                            else:
                                plus=perturb_relative(sub,group,axis,step)
                                minus=perturb_relative(sub,group,axis,-step)
                            dp=self.collision_values(fk_state(self.ref,subbase,plus),active[samples],mask)[0]
                            dm=self.collision_values(fk_state(self.ref,subbase,minus),active[samples],mask)[0]
                            target[rows,0,column]=weight[selected[rows]]*(dp-dm)[reindex,pairs[rows]]/(2*step)
                body=d.chain(local,sample_indices=lookup)
                body+=d.owner.shoe_chain(shoe,d.parameters,d.sample_times,sample_indices=lookup)
                rows=np.ravel_multi_index((selected,pairs),depth.shape)
                scatter=csr_matrix((np.ones(n),(rows,np.arange(n))),shape=(len(residual),n))
                J=scatter@body
            else:
                J=csr_matrix((len(residual),self.state.size))
        diagnostics=[]
        for i,(a,b) in enumerate(self.pairs):
            v=depth[:,i]
            worst=int(np.argmax(v))
            hot=np.flatnonzero(v>0.)
            groups=np.split(hot,1+np.flatnonzero(np.diff(hot)>1))
            intervals=[dict(start_seconds=float(self.times[group[0]]+self.origin_seconds),
                            end_seconds=float(self.times[group[-1]]+self.origin_seconds),
                            peak_m=float(v[group].max()))
                       for group in groups if len(group)]
            diagnostics.append(dict(pair=[a,b],positive_samples=len(hot),peak_model_proxy_m=float(v[worst]),
                worst_sample=int(worst),nearest_source_pts=int(self.nearest_source[worst]),
                sample_is_native=bool(worst%2==0),sample_seconds=float(self.times[worst]+self.origin_seconds),
                intervals=intervals))
        b=full_columns(block('self_collision',residual,J,
            dict(source_pts=self.nearest_source,scale=.055,unit='m'),signature(depth>0)))
        return b,dict(pairs=diagnostics,native_exposures=len(self.native_times),
            sample_count=len(self.times),includes_between_exposure_midpoints=True,
            broadphase_candidate_pair_samples=broad,singular_separating_branch_samples=singular,
            geometry='shaped SMPL-X central capsules and neutral attached skate OBBs',
            capsule_radii_m={name:part.radius for name,part in self.capsules.items()},
            one_phase_local_grip_exception='left palm, left rear blade web; inherited S05 grip interval')

    def joint_rows(self,R):
        twist,swing=hinge_angles(R,self.axes)
        r=np.stack((np.maximum(self.hinge_low-twist,0.),np.maximum(twist-self.hinge_high,0.),
            np.maximum(swing-self.swing-.28*np.abs(twist),0.)),axis=-1)
        return r/self.hinge_scale[:,None],twist,swing

    def hinges(self,x,jacobian):
        d=self.state.increments(x,self.native_times)
        joints=self.hinge_joints-1
        values=np.asarray(d[:,6:]).reshape(-1,21,3)[:,joints]
        bases=self.base['local_R'][::2,joints]
        residual,twist,swing=self.joint_rows(bases@matrices(values))
        weight=np.sqrt(trapezoid_time_weights(self.native_times))
        residual*=weight[:,None,None]
        active,joint=np.nonzero(np.any(residual>0,axis=-1))
        J=None
        if jacobian:
            n=len(active)
            local=np.zeros((n,3,69))
            if n:
                centre=values[active].copy()
                base=bases[active]
                i=np.arange(n)
                for axis in range(3):
                    plus,minus=centre.copy(),centre.copy()
                    plus[i,joint,axis]+=1e-6
                    minus[i,joint,axis]-=1e-6
                    change=(self.joint_rows(base@matrices(plus))[0]-
                            self.joint_rows(base@matrices(minus))[0])/(2e-6)
                    columns=6+3*joints[joint]+axis
                    local[i,:,columns]=weight[active,None]*change[i,joint]
                rows=3*(4*active+joint)
                rowids=(rows[:,None]+np.arange(3)).ravel()
                scatter=csr_matrix((np.ones(3*n),(rowids,np.arange(3*n))),
                    shape=(residual.size,3*n))
                J=scatter@d.chain(local,sample_indices=active)
            else:
                J=csr_matrix((residual.size,self.state.size))
        pts=self.ref.p['pts']
        b=full_columns(block('signed_joint_relation',residual.ravel(),J,
            dict(source_pts=pts,unit='rad'),signature(residual>0)))
        peak=np.argmax(np.linalg.norm(residual,axis=-1),axis=0)
        return b,dict(native_joint_ids=self.hinge_joints.tolist(),
            neutral='shaped native rest; local identity; knee flex toward posterior, elbow toward anterior',
            positive_flexion_axes_parent_rest=self.axes.tolist(),
            finite_engineering_ranges_degrees=np.degrees(np.stack((self.hinge_low,self.hinge_high),axis=1)).tolist(),
            positive_samples=np.count_nonzero(np.any(residual>0,axis=-1),axis=0).tolist(),
            source_pts=pts.tolist(),signed_flexion_deg=np.degrees(twist).tolist(),
            off_axis_swing_deg=np.degrees(swing).tolist(),worst_source_pts=pts[peak].tolist(),
            ankles_and_hips_use_existing_articulation_and_footwear=True,no_world_blade_bank_as_ankle_twist=True)

    def evaluate(self,x,jacobian=False):
        collision,collision_metrics=self.collision(x,jacobian)
        hinges,hinge_metrics=self.hinges(x,jacobian)
        blocks=[collision,hinges]
        return dict(residual=np.concatenate([b.residual for b in blocks]),
            jac=vstack([b.jacobian for b in blocks],format='csr') if jacobian and not getattr(self.problem,'bounded_memory',False) else None,
            parts={b.name:float(b.residual@b.residual/2) for b in blocks},weighted_blocks=blocks,
            metrics=dict(self_collision=collision_metrics,signed_joint_relation=hinge_metrics))
