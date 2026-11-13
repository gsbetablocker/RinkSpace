"""Optimize a local window of the saved body and rigid skate curve."""
from argparse import ArgumentParser
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times, _log, _exp
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from ..inputs import atomic_npz, digest, read
from ..programme_motion_followup import bind
from .source_audit import static_assembly


DEVICE=torch.device('cuda')
DTYPE=torch.float32
def tn(x):
    return torch.as_tensor(np.ascontiguousarray(x),dtype=DTYPE,device=DEVICE)
def ti(x):
    return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.long,device=DEVICE)
def cpu(x):
    return x.detach().cpu().numpy().astype(float)
def qprod(a,b):
    v,w=a[...,:3],a[...,3:]
    u,z=b[...,:3],b[...,3:]
    return torch.cat((w*u+z*v+torch.linalg.cross(v,u),w*z-(v*u).sum(dim=-1,keepdim=True)),dim=-1)
def conj(q):
    return torch.cat((-q[...,:3],q[...,3:]),dim=-1)
def exp(v):
    half=torch.linalg.vector_norm(v,dim=-1,keepdim=True)*.5
    return torch.cat((v*.5*torch.sinc(half/torch.pi),torch.cos(half)),dim=-1)
def log(q):
    v,w=q[...,:3],q[...,3:]
    n=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return v*(2*torch.atan2(n,w)/n.clamp_min(1.e-7)).where(n>1.e-7,2.)
def quat(r):
    return Rotation.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(*r.shape[:-2],4)
def mat(q):
    # W positive is not needed, since both lifts give identical matrices.
    x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
      2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
      2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(*q.shape[:-1],3,3)

def geometry(curve,p,r):
    rr=r[:-1].swapaxes(-1,-2)@r[1:]
    q=quat(rr);old=curve.edge_reference_quaternion
    q=np.where((q*old).sum(-1,keepdims=True)>=0.,q,-q)
    return AbsoluteMotionCurve(curve.knots,p,r,q)

def save_candidate(z,curve,path,*,parent):
    dest=dict(z)
    dest.update(absolute_position_controls=curve.position_controls,
        absolute_rotation_controls=curve.rotation_controls,
        absolute_rotation_edge_reference_quaternion=curve.edge_reference_quaternion)
    x=z['x'].copy()
    position=curve.position_controls-z['absolute_seed_position_controls']
    q=_log(z['absolute_seed_rotation_controls'].swapaxes(-1,-2)@curve.rotation_controls)
    updated=np.c_[position,q.reshape(len(position),66)]
    old=z['absolute_position_controls']
    r0=z['absolute_rotation_controls']
    changed=np.any(curve.position_controls!=old,axis=1)|np.any(curve.rotation_controls!=r0,axis=(1,2,3))
    xx=x[17:17+updated.size].reshape(updated.shape)
    xx[changed]=updated[changed]
    np.testing.assert_allclose(z['absolute_seed_position_controls']+xx[:,:3],curve.position_controls,atol=1e-9,rtol=0)
    np.testing.assert_allclose(z['absolute_seed_rotation_controls']@_exp(xx[:,3:].reshape(-1,22,3)),curve.rotation_controls,atol=1e-8,rtol=0)
    dest['x']=x
    dest['scoped_repair_parent_sha256']=np.array(digest(parent))
    dest['scoped_repair_report']=np.array(str(path.parent/'assessment.json'))
    dest['historical_solver_diagnostics_apply_to_parent']=np.array(True)
    atomic_npz(path,**dest)
    s=curve_from_packet(load(path))
    np.testing.assert_array_equal(s.position_controls,curve.position_controls)
    np.testing.assert_array_equal(s.rotation_controls,curve.rotation_controls)

class Fit:
    def __init__(self,r,c,curve,frames,active,ctrl,fit_position=True):
        self.ref=r;self.camera=c;self.parent=curve
        self.frames=np.asarray(frames);self.time=r.times[frames];self.pts=r.p['pts'][frames]
        self.active=np.array(active,int);self.ctrl=np.array(ctrl,int);self.fit_position=fit_position
        self.pc=(curve.position_controls.copy());self.rc=curve.rotation_controls.copy()
        tau=c.readout_times(frames)
        self.clock=(self.time[:,None]+tau[:,None]*np.linspace(-.5,.5,5)).ravel()
        self.support=np.clip(np.searchsorted(curve.knots,self.clock,side='right')-4,0,len(self.pc)-4)[:,None]+np.arange(4)
        b=curve.basis_sparse(self.clock)[np.arange(len(self.clock))[:,None],self.support].toarray()
        self.weights=tn(b)
        self.beta=tn(np.cumsum(b[:,::-1],axis=1)[:,::-1][:,1:])
        self.indices=ti(self.support)
        self.t0=tn(self.pc);self.r0=tn(quat(self.rc));self.chart=tn(curve.edge_reference_quaternion[self.support[:,:-1]])
        self.ctl=ti(self.ctrl);self.ri=ti(self.active)
        self.hand=r.at(self.clock).get('hand_local_R')
        self.rel=r.at(self.clock).get('skate_relative_R')
        self.relR,self.relT=map(tn,r.binding.relative_parameters(self.rel))
        self.eye=torch.eye(3,device=DEVICE,dtype=DTYPE)
        self.body=r.body
        self.J=tn(self.body.J);self.offset=tn(self.body._joint_offsets)
        self.default=tn(self.body.default)
        self.posedirs=tn(self.body.posedirs);self.shaped=tn(self.body.shaped)
        self.skin=tn(self.body.W);self.reg=tn(self.body.reg.T)
        _,self.material=static_assembly(r)
        boot=r.binding.packet()
        shell=boot['upper'][:len(boot['xs'])*6]
        steel=boot['template']['vertices']
        # Dense arc samples for a smooth envelope during the trial. Validate
        # the accepted result with the existing exact continuous arcs.
        sampled=[]
        for a in boot['template']['contact_arcs']:
            if a['semantic'] not in ('runner','toe_pick'):continue
            angle=np.linspace(a['theta_start_rad'],a['theta_end_rad'],101)
            sampled.extend(np.asarray(a['center'])+a['radius_m']*(np.cos(angle)[:,None]*a['axis_cos']+
                            np.sin(angle)[:,None]*a['axis_sin']))
        self.steel=tn(np.asarray(sampled))
        self.shell=tn(np.r_[shell,steel[::5]])
        self.mats=tn(self.material)
        self.hand_tensor=None if self.hand is None else tn(self.hand)
        self.parameter=torch.nn.Parameter(torch.zeros((len(ctrl),len(active)+int(fit_position),3),device=DEVICE,dtype=DTYPE))
        self.record={}
        self.n=len(frames)
    def set_control_goal(self,new_p,new_r):
        p=new_p[self.ctrl]-self.pc[self.ctrl]
        rr=_log(self.rc[self.ctrl][:,self.active].swapaxes(-1,-2)@new_r[self.ctrl][:,self.active])
        d=np.concatenate([p[:,None],rr],axis=1) if self.fit_position else rr
        self.parameter.data.copy_(tn(d))
    def geometry(self):
        d=cpu(self.parameter)
        p=self.pc.copy();r=self.rc.copy()
        if self.fit_position:p[self.ctrl]+=d[:,0];d=d[:,1:]
        r[self.ctrl[:,None],self.active]=r[self.ctrl[:,None],self.active]@_exp(d)
        return geometry(self.parent,p,r)
    def state(self):
        d=self.parameter
        if self.fit_position:
            pos=self.t0.index_add(0,self.ctl,d[:,0]);d=d[:,1:]
        else:
            pos=self.t0
        # Only four control entries per shutter sample affect the curve.
        updates=qprod(self.r0[self.ctl[:,None],self.ri],exp(d))
        qc=self.r0.clone();qc[self.ctl[:,None],self.ri]=updates
        q=qc[self.indices]
        edge=qprod(conj(q[:,:-1]),q[:,1:])
        edge=edge*torch.where((edge*self.chart).sum(-1,keepdim=True)>=0.,1.,-1.)
        e=exp(self.beta[:,:,None,None]*log(edge))
        rot=mat(qprod(qprod(qprod(q[:,0],e[:,0]),e[:,1]),e[:,2]))
        P=(self.weights[:,:,None]*pos[self.indices]).sum(dim=1)
        local=rot[:,1:]
        n=len(local)
        # Native 55-joint SMPL-X with the same frozen hand curve and regressor.
        tail=self.default[None].expand(n,-1,-1,-1)
        if self.hand_tensor is not None:
            tail=torch.cat((tail[:,:3],self.hand_tensor.reshape(n,30,3,3)),dim=1)
        locals_=torch.cat((local,tail),dim=1)
        R=[self.eye.expand(n,3,3)]; J=[self.J[0].expand(n,3)]
        for j in range(1,55):
            a=self.body.parents[j]
            R.append(R[a]@locals_[:,j-1])
            J.append(J[a]+(R[a]@self.offset[j,:,None]).squeeze(-1))
        allR=torch.stack(R,dim=1);allJ=torch.stack(J,dim=1)
        root=rot[:,0];native=(allJ[:,:22]-self.J[0])*self.body.scale
        world=P[:,None]+(root[:,None]@native[...,None]).squeeze(-1)
        U=root[:,None]@allR[:,[7,8]]
        bootR=U@self.relR
        bootP=world[:,[7,8]]+(U@self.relT[...,None]).squeeze(-1)
        return P,root,local,allR,allJ,world,bootR,bootP,locals_
    def project(self,w,inds=None):
        n=self.n if inds is None else len(inds)
        w=w.reshape(n,5,-1,3);arr=cpu(w)
        ids=self.frames if inds is None else self.frames[inds]
        pr=moving_projection(self.camera,arr,ids,jacobian=True)
        return tn(pr['uv'])+torch.einsum('nkulc,nlkc->nku',tn(pr['pixel_jac_nodes']),w-w.detach())
    def coco(self,state):
        P,Q,_,R,J,_,_,_,local=state;n=len(P)
        translation=J-(R@self.J[None,:,:,None]).squeeze(-1)
        posed=self.shaped+((local-self.eye).reshape(n,486)@self.posedirs).reshape(n,-1,3)
        blend_R=(self.skin@R.reshape(n,55,9)).reshape(n,-1,3,3)
        blend_t=self.skin@translation
        v=(blend_R@posed[...,None]).squeeze(-1)+blend_t
        coco=(self.reg@v-self.J[0])*self.body.scale
        return P[:,None]+(Q[:,None]@coco[...,None]).squeeze(-1)
    def contact_heights(self,state):
        _,_,_,_,_,_,R,O,_=state
        hi=torch.einsum('nfj,mj->nfm',R[2::5,:,2,:],self.steel)+O[2::5,:,2,None]
        return hi.min(dim=-1).values*self.ref.binding.ratio
    def material_world(self,state):
        R,O=state[6:8]
        return O[:,:,None]+torch.einsum('nfij,mj->nfmi',R,self.mats)
    def world_native(self,state):
        return state[5]
    def prior_rows(self,state,base_poses):
        P,Q,R=state[:3]
        # Smooth actual changes across the curved segment. Measured endpoint
        # continuity belongs in separate signed terms.
        dp=P[2::5]-base_poses['P']
        rr=R[2::5]
        local_delta=(base_poses['local_R'].transpose(-1,-2)@rr-self.eye)
        active=self.active[self.active>0]-1
        dev=local_delta[:,active]
        dq=(base_poses['Q'].transpose(-1,-2)@Q[2::5]-self.eye)
        return (self.parameter.square().mean()*.4 + dp.square().mean()*1.1 +
                dev.square().mean()*.9 + dq.square().mean()*.9)
    def run(self,fn,steps,name):
        start=time.monotonic()
        state=dict(min=float('inf'),x=None)
        def evaluate(back=True):
            z=self.state();parts=fn(z)
            total=sum(parts.values())
            if not torch.isfinite(total):raise ValueError('nonfinite source-window objective')
            if back:total.backward()
            return total,parts
        def logit(phase,step,total,parts,retained):
            print(json.dumps(dict(window=name,phase=phase,step=step,elapsed_s=round(time.monotonic()-start,3),
                total=float(total.detach()),components={k:float(v.detach()) for k,v in parts.items()},
                active_controls=self.ctrl.tolist(),active_tracks=self.active.tolist(),source_exposures=self.n,
                device=str(DEVICE),retained=retained)),flush=True)
        with torch.no_grad():
            initial,terms=evaluate(False);logit('initial',0,initial,terms,True)
            state.update(min=float(initial),x=self.parameter.detach().clone())
        opt=torch.optim.LBFGS([self.parameter],lr=1.,max_iter=4,max_eval=9,history_size=25,
                              tolerance_grad=4.e-5,tolerance_change=5.e-7,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(set_to_none=True)
            return evaluate(True)[0]
        stall=0
        for i in range(steps):
            opt.step(closure)
            with torch.no_grad():
                total,parts=evaluate(False);val=float(total)
                accepted=val<state['min']-1.e-6
                if accepted:
                    stall=0 if state['min']-val>2.e-5 else stall+1
                    state.update(min=val,x=self.parameter.detach().clone())
                else:
                    stall+=1
                logit('line_search',i+1,total,parts,accepted)
                if stall>=3:break
        with torch.no_grad():
            self.parameter.copy_(state['x'])
        return self.geometry()

def smooth(a,b,t):
    u=np.clip((t-a)/(b-a),0,1)
    return u*u*(3-2*u)

def opening(r,c,z,out,*,prepared,donor_path,source_path,parent,start=None):
    curve=r.absolute_curve
    g=greville_times(curve.knots)
    ctrl=np.flatnonzero(g<2.90)
    frames=np.arange(np.searchsorted(r.times,3.28,side='right'))
    active=[0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21]
    fit=Fit(r,c,curve,frames,active,ctrl)
    t=fit.pts/12800
    donor=load(donor_path)
    L=np.array([1,2,4,5,7,8,10,11])
    strength=(1-smooth(.69,1.37,g))
    p=curve.position_controls.copy();rr=curve.rotation_controls.copy()
    up=rr[:,0,:,1];target=np.tile([0.,0.,1.],(len(up),1))
    cross=np.cross(up,target);n=np.linalg.norm(cross,axis=-1)
    tilt=cross/np.maximum(1.e-9,n[:,None])*np.arctan2(n,up[:,2])[:,None]
    rr[:,0]=_exp(tilt*(strength*.82)[:,None])@rr[:,0]
    blend=_log(rr[:,L].swapaxes(-1,-2)@donor['local_R'][L-1])
    rr[:,L]=rr[:,L]@_exp(strength[:,None,None]*blend*.86)
    # Preserve actual camera-visible upper body at initialization: the lower
    # torso solve can independently compensate the smaller pelvis-roll prior.
    p[:,:2]=p[:,:2]+strength[:,None]*.25*(donor['P'][:2]-p[:,:2])
    if start is None:
        fit.set_control_goal(p,rr)
    else:
        warm=curve_from_packet(load(start))
        fit.set_control_goal(warm.position_controls,warm.rotation_controls)
    obs=load(prepared/'observations.npz')
    held=(np.arange(len(t))%8==4)
    uf,oj=np.nonzero(obs['use'][frames]&(~held)[:,None])
    uu=obs['uv'][uf,oj];jj=obs['model_joint_ids'][uf,oj]
    # Face eyes/ears in profile and occluded arms retain a robust finite weight.
    sigma=np.maximum(5.,obs['sigma_px'][uf,oj]*.78)
    sigma=np.where(jj>=11,sigma*.95,sigma)
    uf_t=ti(uf);jj_t=ti(jj);ut=tn(uu);sig=tn(sigma)
    base=r.at(fit.time)
    poses={k:tn(base[k]) for k in ('P','Q','local_R')}
    upright=tn(1-smooth(22.79,23.08,t));crossed=tn(1-smooth(22.82,23.31,t))
    expected_no_lift=tn(1-smooth(24.65,24.88,t))
    crop=tn(smooth(23.14,23.35,t)*(1-smooth(23.94,24.01,t)))
    quiet=tn((t<=22.86).astype(float))
    support_l=tn(smooth(24.08,24.22,t)*(1-smooth(24.74,24.88,t)))
    donorR=tn(donor['local_R'][L-1]);anchor=tn(donor['P'][:2])
    donor_state=fk_state(r,dict(P=donor['P'][None],Q=donor['Q'][None],local_R=donor['local_R'][None],skate_relative_R=r.at([0.])['skate_relative_R']),np.zeros((1,69)))
    donor_ankles=tn(donor_state['native_world'][0,[7,8],:2])
    donor_delta=donor_ankles[0]-donor_ankles[1]
    donor_dir=tn(donor_state['R'][0,:,:,0])
    origin=tn(donor_state['P'][0,:2])
    source=read(source_path)
    rows=[x for x in source['rows'] if x['role']=='fit']
    def row_values(name):
        q=[v for v in rows if name in v]
        k=np.array([int(round((v['seconds']-22.12)*50)) for v in q])
        pts=np.array([v[name] for v in q],float)
        return ti(k),tn(pts)
    leg_idx,leg_target=row_values('legs')
    sole_idx,sole_target=row_values('sole')
    steel_idx,steel_target=row_values('steel')
    # Each leg record is ordered by anatomical side once the original shows
    # the turn. Earlier silhouette cross-sections are read without side.
    thigh_idx,thigh_target=row_values('thigh')
    hip_idx,hip_target=row_values('hip')
    edge_control=tn(smooth(24.74,25.1,g[ctrl]+22.12))
    def objective(s):
        proj=fit.project(fit.coco(s))
        err=(proj[uf_t,jj_t]-ut)/sig[:,None]
        pixels=(torch.sqrt(1.+err.square())-1.).mean()*8.
        native=fit.world_native(s)
        legs=native[:,[1,2,4,5,7,8]]
        all_leg_uv=fit.project(legs)
        ankle_uv=all_leg_uv[:,2:]
        hip_uv=all_leg_uv[:,:2].mean(1)
        materials=fit.material_world(s)
        bootuv=fit.project(materials.flatten(1,2)).reshape(len(t),2,-1,2)
        p,q,loc=s[0][2::5],s[1][2::5],s[2][2::5]
        up=q[:,:,1]
        spine=native[2::5,12]-native[2::5,0]
        spine=spine/torch.linalg.vector_norm(spine,dim=-1,keepdim=True)
        # Earlier observed pose is a finite cross-shot prior in the quiet
        # phase. Preserve the turn and release that the closer shot shows.
        upright_loss=((up[:,:2].square().sum(-1)*1.3+spine[:,:2].square().sum(-1))*
                       upright).mean()*33.
        donor_loss=((loc[:,L-1]-donorR).square().mean(dim=(-1,-2,-3))*crossed).mean()*12.
        ground=fit.contact_heights(s)
        ankles=native[2::5][:,[7,8],:2]
        stance_diff=(((ankles[:,0]-ankles[:,1]-donor_delta)/.11).square().sum(-1)*upright).mean()*2.8
        stance_place=(((ankles-donor_ankles)/.20).square().mean(dim=(1,2))*upright).mean()*1.15
        stay=((torch.diff(ankles,dim=0)/.009).square().mean(dim=(1,2))*upright[1:]).mean()*.35
        foot_dirs=s[6][2::5,:,:,0]
        stance_dirs=((foot_dirs-donor_dir).square().mean(dim=(1,2))*upright).mean()*2.7
        rear_clear=(torch.relu(.018-ground[:,0])/.013).square()*upright
        front_stay=((ground[:,1]/.012).square()*upright).mean()*2.4+rear_clear.mean()*.2
        no_lift=((ground.min(dim=1).values/.015).square()*expected_no_lift).mean()*3.8
        prevent=torch.relu(-ground-.005).square().mean()/(.013**2)*3.5
        stand=((ground[:,0]/.013).square()*support_l).mean()*2.8
        # Crop boundary support only when the original contains neither
        # visible boot. This rules out old shoes drawn on bare ice.
        RR,OO=s[6:8]
        shell_world=OO[:,:,None]+torch.einsum('nfij,mj->nfmi',RR,fit.shell[::10])
        silhouette=fit.project(shell_world.flatten(1,2))
        outside=(torch.relu(1081.-silhouette[:,:,1].min(dim=1).values)*crop/38.).square().mean()*1.7
        position=(((p[:,:2]-anchor).square().sum(-1)*quiet).mean()/.20**2+
                   (torch.diff(p,dim=0).square().sum(-1)*quiet[1:]).mean()/.008**2)*1.2
        lk=ankle_uv[leg_idx]
        leg_mask=leg_target[:,:,0]>0
        e=(lk[leg_mask]-leg_target[leg_mask])/18.
        source_legs=(torch.sqrt(1.+e.square())-1.).mean()*5.5
        # Left/right order is not needed for a silhouette cross-section.
        uh=all_leg_uv[thigh_idx,:2];uk=all_leg_uv[thigh_idx,2:4]
        f=(thigh_target[:,0,None]-uh[:,:,1])/(uk[:,:,1]-uh[:,:,1]).clamp_min(35.)
        tv=torch.sort(uh[:,:,0]+f.clamp(.05,1.4)*(uk[:,:,0]-uh[:,:,0]),dim=1).values
        tshift=((tv-thigh_target[:,1:])/27.).square().mean()*2.
        hips=((hip_uv[hip_idx]-hip_target)/35.).square().mean()*1.7
        # Independent heel and toe on visible shoe portions.
        eu=(bootuv[sole_idx,sole_target[:,0].long(),sole_target[:,1].long()]-sole_target[:,2:])/12.
        sole_loss=(torch.sqrt(1.+eu.square())-1.).mean()*3.5
        zz=bootuv[steel_idx,steel_target[:,0].long(),2:4]-steel_target[:,1:].reshape(-1,2,2)
        steel_loss=(torch.sqrt(1.+(zz/12.).square())-1.).mean()*4.5
        knees=loc[:,[3,4]]
        safe=((knees[:,:,0,1].square()+knees[:,:,0,2].square()).mean()*3.5+
              torch.relu(knees[:,:,2,1]-.18).square().mean()*9.)
        continuity=((torch.diff(s[7][2::5],dim=0,n=2)/.018).square().mean()+
             ((s[6][7::5]-s[6][2:-5:5])/.22).square().mean())*.40
        close=(fit.parameter.square()*edge_control[:,None,None]).mean()*17.
        return dict(source_body=pixels,previous_upright=upright_loss,previous_crossed=donor_loss,
                    source_legs=source_legs,visible_thighs=tshift,visible_hips=hips,source_heel_toe=sole_loss,first_visible_steel=steel_loss,
                    stance_ankles=stance_diff+stance_place,stance_shoe_heading=stance_dirs,stance_front=front_stay,stance_drift=stay,
                    continuous_support=no_lift,ice_clearance=prevent,first_visible_left_support=stand,
                    cropped_shoes=outside,quiet_root=position,knee_anatomy=safe,material_continuity=continuity,
                    finite_parent=fit.prior_rows(s,poses)+close)
    opt=fit.run(objective,40,'opening')
    out.mkdir(exist_ok=True)
    save_candidate(z,opt,out/'candidate.npz',parent=parent)
    print(json.dumps(dict(status='wrote',parameter=str(out/'candidate.npz'))),flush=True)
    return opt

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("prepared", "evidence", "parameter", "donor", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ref, camera, _, _, controls = bind(args.prepared, args.evidence, args.parameter)
    opening(ref, camera, controls, args.output, prepared=args.prepared, donor_path=args.donor,
            source_path=args.source, parent=args.parameter, start=args.warm_start)
