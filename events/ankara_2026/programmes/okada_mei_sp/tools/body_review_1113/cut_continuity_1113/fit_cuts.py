"""Fit both sides of each real edit on the same programme curve."""
import json
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from scipy.optimize import minimize
import torch

from calibration.program_tracking.body_tracking.absolute_motion_curve import _log
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.programme_camera import ProgrammeCamera
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, NAMES, angles_joint
from calibration.program_tracking.body_tracking.anatomical_chart import JointChart, JointEnvelope
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from experiments.skate_video_v1.body_skate_fit import Assembly
from experiments.skate_video_v1.inputs import PROJECT, atomic_json, atomic_npz, read
from experiments.skate_video_v1.motion_review.scoped_fit import Fit, cpu, geometry, save_candidate, ti, tn
from experiments.skate_video_v1.programme_motion_followup import bind

RUN=Path(os.environ['RINKSPACE_AUDIT_RUN'])
JOB=Path(os.environ.get('RINKSPACE_OKADA_BODY_REVIEW', RUN.parent/'audit_af48_1113')).resolve()/'cut_continuity_1113'
PREP=RUN/'prepared_refined'
OLD=JOB.parent/'opening_lunge'
PARENT=OLD/'delivered/solve/candidate.npz'
r, old_camera, _, old_doc, packet=bind(PREP,OLD/'opening_source.json',PARENT)
doc=read(JOB/'cut_source.json')
camera=ProgrammeCamera(PROJECT,r.p,r.p['shot_ids'],record_path=doc['programme_camera_candidate']['record'])
camera.core.motion_projection_threads=3
r.source_document=doc
curve=curve_from_packet(packet)
clock=r.times
pts=r.p['pts']
ratio=r.binding.ratio
sec=pts/12800.
obs=load(PREP/'observations.npz')
source=load(Path(doc['whole_skate_source']['path']))
mass=ContinuousMassGeometry(r)
assembly=Assembly(r.binding)
config=doc['scoped_candidate_import']
anchor=load(Path(doc['programme_structure']['chart_anchor']))
chart=JointChart(anchor['rest_R'],anchor['anatomical_basis'],anchor['reference_twist'])
policy=read(doc['programme_structure']['policy'])
envelopes=[policy['envelopes'][name.split('_')[-1]] for name in NAMES]
exact_envelopes=[JointEnvelope(*np.deg2rad(e['flex_degrees']),np.deg2rad(e['flex_grid_degrees']),
                  np.deg2rad(e['swing_axes_degrees']),*np.deg2rad(e['scale_degrees']),policy['provenance'])
                 for e in envelopes]
bodies=tn(anchor['anatomical_basis'])
rest=tn(anchor['rest_R'])
joint_ids=JOINTS-1
bounds_data=[]
for e in envelopes:
    bounds_data.append(tuple(tn(np.deg2rad(e[k])) for k in
        ('flex_degrees','flex_grid_degrees','swing_axes_degrees','scale_degrees')))
base_world_rotation=None

def sq(x):
    return x.square().sum(dim=-1)

def robust(x,delta=2.):
    return 2.*delta*delta*(torch.sqrt(1.+x.square()/delta**2)-1.)

def quat_axis(rotation):
    # Away from the pi branch the original anatomical charts use this
    # representation. Safe sign at pi is supplied by the frozen twist lift.
    w=(1.+rotation.diagonal(dim1=-1,dim2=-2).sum(-1)).clamp_min(1.e-7).sqrt()/2.
    v=torch.stack((rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],
                   rotation[...,1,0]-rotation[...,0,1]),-1)/(4.*w[...,None])
    return v,w

def angular_steps(R):
    relative=R[1:]@R[:-1].transpose(-1,-2)
    q,w=quat_axis(relative)
    n=torch.linalg.vector_norm(q,dim=-1)
    return q*((2.*torch.atan2(n,w)/n.clamp_min(1.e-7)).where(n>1.e-7,2.))[...,None]

def angular_accel(R,dt):
    return torch.diff(angular_steps(R),dim=0)/dt**2

def refit_state(fit_curve,t):
    ref=absolute_reference(r,fit_curve)
    ref.relative_skate_curve=r.relative_skate_curve
    return fk_state(ref,ref.at(t),np.zeros((len(t),69)))

def anatomy_values(pose,frames):
    data=chart.evaluate(pose['local_R'][:,joint_ids],sample_indices=frames)
    return np.stack([env.residual(angles_joint(data,k)) for k,env in enumerate(exact_envelopes)],1)

def anatomical(local,branch):
    R=bodies.transpose(-1,-2)@rest.transpose(-1,-2)@local[:,joint_ids]@bodies
    q,w=quat_axis(R)
    x,y,z=q.unbind(-1)
    twist=2.*torch.atan2(x,w)
    twist=branch+torch.remainder(twist-branch+torch.pi,2.*torch.pi)-torch.pi
    n=torch.linalg.vector_norm(q[...,1:],dim=-1)
    p=torch.sqrt(w.square()+x.square())
    swing=2.*torch.atan2(n,p)
    u=torch.stack((w*y-x*z,w*z+x*y),-1)/(p*n.clamp_min(1.e-7))[...,None]
    out=[]
    for j,(ends,grid,radii,scales) in enumerate(bounds_data):
        a=twist[:,j]
        k=torch.searchsorted(grid,a.contiguous(),right=True).clamp(1,len(grid)-1)
        f=((a-grid[k-1])/(grid[k]-grid[k-1])).clamp(0.,1.)
        radius=radii[k-1]+f[:,None]*(radii[k]-radii[k-1])
        allowed=torch.linalg.vector_norm(u[:,j]/radius,dim=-1).clamp_min(1.e-6).reciprocal()
        out.append(torch.relu(torch.stack(((ends[0]-a)/scales[0],(a-ends[1])/scales[0],
                                          (swing[:,j]-allowed)/scales[1]),-1)))
    return torch.stack(out,dim=1)

def ground_samples():
    pack=r.binding.packet()
    tr=pack['template']
    samples=[]
    for name in ('runner','toe_pick'):
        vertices=[]
        for arc in tr['contact_arcs']:
            if arc['semantic']!=name:continue
            t=np.linspace(arc['theta_start_rad'],arc['theta_end_rad'],73)
            vertices.extend(np.asarray(arc['center'])+arc['radius_m']*(
                np.cos(t[:,None])*arc['axis_cos']+np.sin(t[:,None])*arc['axis_sin']))
        for group in tr['contact_point_groups']:
            if group['semantic']==name:vertices.extend(group['points'])
        samples.append(tn(vertices))
    return (*samples,tn(tr['vertices']),tn(pack['upper']))
edge=ground_samples()

class CutFit:
    def __init__(self, cut, window, strength=1.):
        self.cut=int(cut)
        self.source_time=float(sec[self.cut])
        self.active=np.arange(22)
        self.ctrl=np.flatnonzero((curve.knots[:-4] < clock[window[1]]) & (curve.knots[4:] > clock[window[0]]))
        support=curve.knots[np.r_[self.ctrl[0],self.ctrl[-1]+4]]
        self.frames=np.flatnonzero((clock>=support[0]-.045)&(clock<=support[1]+.045))
        self.fit=Fit(r,camera,curve,self.frames,self.active,self.ctrl)
        self.param=torch.nn.Parameter(torch.zeros_like(self.fit.parameter))
        self.scale=tn([.018]+[.12]*22)[None,:,None]
        self.strength=strength
        self.q=refit_state(curve,clock[self.frames])
        self.base_ice=component_ice(r.binding,self.q)[1]
        d=clock[self.frames]-(clock[cut-1]+clock[cut])/2.
        # Let the velocities approach both flanks of the cut. Do not tie
        # either the skate or the pelvis to a stationary world point.
        self.time_weight=tn(.17+.83*np.exp(-.5*(d[1:-1]/.115)**2))
        self.halo=tn(np.clip((abs(d)-.19)/.34,0.,1.)**2)
        self.near=tn(np.exp(-.5*(d/.12)**2))
        self.P0=tn(self.q['P'])
        self.Q0=tn(self.q['Q'])
        self.L0=tn(self.q['local_R'])
        self.S0=tn(self.q['O'])
        self.SR0=tn(self.q['R'])
        self.dense=np.unique(np.r_[np.arange(support[0],support[1],.008),
                                   curve.knots[self.ctrl[0]:self.ctrl[-1]+5]])
        self.D2=tn(curve.basis_sparse(self.dense,2)[:,self.ctrl].toarray()*ratio)
        self.D20=tn(curve.position_derivative(self.dense,2)*ratio)
        d_dense=self.dense-(clock[cut-1]+clock[cut])/2.
        self.dense_w=tn(.12+.88*np.exp(-.5*(d_dense/.115)**2))
        self.cap=tn(anatomy_values(self.q,self.frames).max(0))
        self.angle_anchor=tn(anchor['reference_twist'][self.frames])
        self.ice_floor=tn(np.minimum(self.base_ice.min(0)[:,[0,1]],[-.006,-.010])+.0003)
        self.part_floor=tn(np.minimum(self.base_ice[:,:,2:].min((0,2)),-.006))
        self.source_binding()
        self.cavity_binding()
        self.refine_guard = None
        self.free = None

    def source_binding(self):
        n=len(self.frames)
        f,j=np.nonzero(obs['use'][self.frames])
        q=obs['model_joint_ids'][self.frames[f],j]
        self.fj,self.model_id=ti(f),ti(q)
        self.uv=tn(obs['uv'][self.frames[f],j])
        self.sigma=tn(np.maximum(obs['sigma_px'][self.frames[f],j,None],3.))
        c=np.where(j<5,.4,np.where(j<13,1.,.85))
        self.body_weights=tn(c)
        base=self.fit.state()
        self.old_body_uv=self.fit.project(self.fit.coco(base)).detach()[self.fj,self.model_id]
        self.role=source['role'][self.frames]
        self.whole=[]
        for foot in (0,1):
            take=np.flatnonzero(self.role[:,foot]==1)
            if len(take)==0:continue
            nodes=refit_state(curve,self.fit.clock.reshape(n,5)[take].ravel())
            w=assembly.points(nodes,foot,transforms=(nodes['R'],nodes['O']))
            poly=moving_projection(camera,w.reshape(len(take),5,-1,3),self.frames[take])['uv']
            rows=[assembly.associations(z,source['region'][self.frames[k],foot]) for z,k in zip(poly,take,strict=True)]
            ab,frac,xy,normal=(np.stack(y) for y in zip(*rows,strict=True))
            x=assembly.vertices[ab[:, :, 0]]
            local=x+frac[...,None]*(assembly.vertices[ab[:,:,1]]-x)
            base_pose=refit_state(curve,self.fit.clock.reshape(n,5)[take].ravel())
            matched=assembly.points(base_pose,foot,(ab,frac),(base_pose['R'],base_pose['O']))
            m_uv=moving_projection(camera,matched.reshape(len(take),5,-1,3),self.frames[take])['uv']
            distance=np.median(np.linalg.norm(m_uv-xy,axis=-1),axis=1)
            radii=source['sigma_px'][self.frames[take],foot]
            self.whole.append((foot,take,ab,frac,tn(local),tn(xy),tn(normal),
                tn(np.maximum(radii[:,None],4.)),tn(distance+2.6*radii)))
        self.material=tn(assembly.vertices[::2])
        self.marks=[]
        path=RUN/'opening_direction_1113/cut_position_source.json'
        if path.is_file() and self.source_time==101.66:
            rows=[q for q in read(path)['rows'] if q['role']=='fit']
            ff=np.searchsorted(pts[self.frames],[q['pts'] for q in rows])
            assert np.all(pts[self.frames[ff]]==[q['pts'] for q in rows])
            self.marks.append((ff,tn([q['steel_centre_uv'] for q in rows]),tn([q['sigma_px'] for q in rows])))

    def cavity_binding(self):
        from cavity_torch import Cavity
        self.cavity=Cavity(r,self.fit,self.q)

    def model_com(self,st):
        P,Q,_,Rs,_,world,R,O,_=st
        M=tn(mass.mass)
        A=Q[2::5,None]@Rs[2::5][:,mass.ids]
        centres=world[2::5][:,mass.ids]*ratio+torch.einsum('nsij,sj->nsi',A,tn(mass.offsets[:mass.count]))
        shoes=O[2::5]*ratio+torch.einsum('nfij,fj->nfi',R[2::5],tn(mass.offsets[mass.count:]))
        return torch.einsum('s,nsc->nc',M,torch.cat((centres,shoes),dim=1))

    def loss(self,st,softness=1.):
        P,Q,L,_,_,world,R,O,_=st
        G=self.model_com(st)
        Qmid,Lmid,Rmid,Omid=Q[2::5],L[2::5],R[2::5],O[2::5]
        body=self.fit.project(self.fit.coco(st))[self.fj,self.model_id]
        image=(robust((body-self.uv)/self.sigma).sum(-1)*self.body_weights).mean()*8.
        # Weak constraint on model displacement where the source is a worse
        # discriminator of rapid edge/foreshortening changes.
        image_hold=((body-self.old_body_uv).square().sum(-1)*self.halo[self.fj]).mean()/14.**2*2.
        silhouette=body.sum()*0.
        for foot,indices,ab,frac,local,uv,normal,sigma,limit in self.whole:
            rotation=R.reshape(-1,5,2,3,3)[indices,:,foot]
            origin=O.reshape(-1,5,2,3)[indices,:,foot]
            if self.refine_guard is not None:
                paired, blend = self.refine_guard.contour_rows[foot]
                alpha = self.refine_guard.upper.alpha(self.fit,st[3],st[4]).reshape(-1,5,2,3)[indices,:,foot]
                u,v = [self.refine_guard.upper.apply(a,alpha) for a in paired]
                local=u+blend[:,None,:,None]*(v-u)
                XYZ=origin[:,:,None]+torch.einsum('nfij,nfkj->nfki',rotation,local)
            else:
                XYZ=origin[:,:,None]+torch.einsum('nfij,nkj->nfki',rotation,local)
            xy=self.fit.project(XYZ.flatten(0,1),indices)
            err=xy-uv
            silhouette+=robust((err*normal).sum(-1)/sigma).mean()*1.25
            silhouette+=robust(err/sigma[...,None]).mean()*.15
            if self.refine_guard is not None:
                extent=torch.linalg.vector_norm(err,dim=-1).median(dim=1).values
                silhouette+=14.*(torch.relu(extent-limit)/1.2).square().sum()
        for indices,uv,sigma in self.marks:
            mm=self.fit.material_world(st)[:,1,2:4].reshape(-1,5,2,3)[indices].flatten(0,1)
            err=self.fit.project(mm,indices).mean(1)-uv
            silhouette+=robust(err/sigma[:,None]).mean()*.60
        acc=self.D20+self.D2@self.fit.parameter[:,0]
        root=(sq(acc/23.)*self.dense_w).mean()*.54
        com=(sq(torch.diff(G,n=2,dim=0)/(.02**2*26.))*self.time_weight).mean()*.85
        local=torch.cat((Qmid[:,None],Lmid),dim=1)
        angular=(sq(angular_accel(local,.02)/115.)*self.time_weight[:,None]).mean()*.78
        shoe_turn=(sq(angular_accel(Rmid,.02)/100.)*self.time_weight[:,None]).mean()*.70
        joint_acc=torch.diff(world[2::5][:,[1,2,4,5,7,8,12,16,17,18,19,20,21]],n=2,dim=0)*ratio/.02**2
        articulation=(sq(joint_acc/75.)*self.time_weight[:,None]).mean()*.52
        shoe_move=(sq(torch.diff(Omid,n=2,dim=0)*ratio/(.02**2*52.))*self.time_weight[:,None]).mean()*.77
        # Ice is a one-sided finite surface clearance, and does not create a
        # contact from an occlusion. Preserve the original separation for a
        # clearly lifted foot while allowing shape of the transition to move.
        Z=Rmid[:,:,2,:]
        ground=[(torch.einsum('nfj,kj->nfk',Z,E)+Omid[:,:,2,None]).min(-1).values*ratio for E in edge]
        ice=torch.stack(ground[:2],-1)
        clearance=(torch.relu(self.ice_floor[None]-ice)/.009).square().mean()*18.
        clearance+=(torch.relu(self.part_floor[None]-torch.minimum(ground[2],ground[3]))/.012).square().mean()*12.
        ref_h=tn(self.base_ice[:,:,0])
        z=ground[0]-ref_h
        clearance+=(torch.relu(z.abs()-.037)/.075).square().mean()*.72
        lift_mask=ref_h>.16
        clearance+=(torch.relu(-z-.038).square()*lift_mask).mean()/.055**2*3.
        anatomy=anatomical(Lmid,self.angle_anchor)
        anatomical_loss=anatomy.square().mean()*26.+(
            torch.relu(anatomy-self.cap[None]-.013).square().amax(dim=0)).sum()*24.
        prior=(
            ((P[2::5]-self.P0).square().sum(-1)/.12**2)*(1.+4*self.halo)).mean()*.63
        prior+=((Lmid-self.L0).square().mean((1,2,3))*(1.+5*self.halo)).mean()*4.
        prior+=((Qmid-self.Q0).square().mean((1,2))*(1.+4*self.halo)).mean()*2.
        prior+=((Omid-self.S0).square().mean((1,2))*self.halo).mean()/.08**2*.48
        prior+=self.param.square().mean()*.014
        smooth=(root+com+angular+shoe_turn+articulation+shoe_move)*self.strength*softness
        parts=dict(body_pixels=image,whole_skate_and_visible_centre=silhouette,body_border=image_hold,
                   motion=smooth,ice=clearance,anatomy=anatomical_loss,pose=prior)
        parts.update(self.cavity.loss(st))
        if self.refine_guard is not None:
            parts.update(self.refine_guard.loss(st,local,anatomy))
        return parts

    def run(self,first_steps,second_steps,initial=None):
        print(json.dumps(dict(event='cut_fit_start',source_pts=int(pts[self.cut]),seconds=self.source_time,
            first_last_source_seconds=sec[self.frames[[0,-1]]].tolist(),
            control_indices=self.ctrl.tolist(),native_body_source_rows=len(self.fj),
            whole_skate_source_rows=int(np.sum(self.role==1)))),flush=True)
        if initial is not None:
            self.fit.set_control_goal(initial.position_controls,initial.rotation_controls)
            self.param.data.copy_((self.fit.parameter / self.scale).detach())
        x=cpu(self.param).ravel()
        def val(x, strength, back=True):
            self.param.data.copy_(tn(x.reshape(self.param.shape)))
            self.param.grad=None
            self.fit.parameter=self.param*self.scale
            if self.free is not None:
                self.fit.parameter=self.fit.parameter*self.free
            parts=self.loss(self.fit.state(),strength)
            total=sum(parts.values())
            if not torch.isfinite(total):
                raise ValueError({key:float(v.detach()) for key,v in parts.items()})
            if back:
                total.backward()
                return float(total.detach()),cpu(self.param.grad).ravel()
            return float(total.detach()),{k:float(v.detach()) for k,v in parts.items()}
        for strength,steps in ((1.,first_steps),(.72,second_steps)):
            if steps <= 0:
                continue
            n=0
            def tick(xx):
                nonlocal n
                n+=1
                if n%25==0:
                    v,p=val(xx,strength,False)
                    print(json.dumps(dict(event='cut_fit_iteration',seconds=self.source_time,iteration=n,
                        strength=strength,total=v,components=p)),flush=True)
            v,p=val(x,strength,False)
            print(json.dumps(dict(event='cut_fit_initial',seconds=self.source_time,objective=v,components=p)),flush=True)
            limits=[(-3.5,3.5)]*len(x) if self.free is None else [
                (-3.5,3.5) if flag else (0.,0.) for flag in np.broadcast_to(cpu(self.free),self.param.shape).ravel()]
            res=minimize(val,x,args=(strength,),jac=True,method='L-BFGS-B',
                         bounds=limits,callback=tick,
                         options=dict(maxiter=steps,ftol=1.e-8,maxcor=18,gtol=4.e-5))
            print(json.dumps(dict(event='cut_fit_stage',seconds=self.source_time,iter=int(res.nit),
                      objective=float(res.fun),status=str(res.message))),flush=True)
            x=res.x
        val(x,.72,False)
        return self.fit.geometry()

    def report(self,new):
        frames=self.frames
        t=clock[frames]
        dense=np.sort(np.r_[t,(t[1:]+t[:-1])/2.])
        near=abs(sec[frames]-(self.source_time-.01))<=.16
        mid=near[1:-1]
        dense_near=abs(dense-(clock[self.cut-1]+clock[self.cut])/2.)<=.16
        cf=component_ice(r.binding,refit_state(new,dense))[1]
        q=refit_state(new,t)
        centre=mass.com(mass.from_pose(q)[0])
        body=refit_state(new,self.fit.clock)
        model=r.body.forward_rotations(body['local_R'],hand_local_R=body.get('hand_local_R'))[0]
        yy=body['P'][:,None]+(body['Q'][:,None]@model[...,None]).squeeze(-1)
        uv=moving_projection(camera,yy.reshape(len(frames),5,17,3),frames)['uv']
        f,j=np.nonzero(obs['use'][frames])
        e=np.linalg.norm(uv[f,obs['model_joint_ids'][frames[f],j]]-obs['uv'][frames[f],j],axis=-1)
        b=e[near[f]];torso=e[near[f]&np.isin(j,(5,6,11,12))]
        d=np.degrees(np.linalg.norm(_log(q['R'][1:]@q['R'][:-1].swapaxes(-1,-2)),axis=-1))
        local_delta=np.degrees(np.linalg.norm(_log(q['local_R'][1:]@q['local_R'][:-1].swapaxes(-1,-2)),axis=-1))
        near_edges=near[1:]|near[:-1]
        a=np.linalg.norm(np.diff(centre,n=2,axis=0)/.02**2,axis=-1)
        shoe=np.linalg.norm(np.diff(q['O'],n=2,axis=0)*ratio/.02**2,axis=-1)
        root=np.linalg.norm(new.position_derivative(dense,2)*ratio,axis=-1)
        cavity=self.cavity.exact(q)
        return dict(seconds=self.source_time,pts=int(pts[self.cut]),
                    root_a=float(root[dense_near].max()),com_a=float(a[mid].max()),
                    skate_a=shoe[mid].max(0).tolist(),skate_step_deg=d[near_edges].max(0).tolist(),
                    local_angle_peak_deg=float(local_delta[near_edges].max()),
                    pixel_p50_p90=np.percentile(b,[50,90]).tolist(),torso_p90=float(np.percentile(torso,90)),
                    full_support_pixels=np.percentile(e,[50,90]).tolist(),
                    exact_runner_min_mm=(cf[:,:,0].min(0)*1000).tolist(),
                    anatomy_max=anatomy_values(q,self.frames).max(0).tolist(),
                    cavity_peak_mm=(cavity.max(0)*1000).tolist(),cavity_norm=np.linalg.norm(cavity,axis=0).tolist())

if __name__=='__main__':
    ap=ArgumentParser()
    ap.add_argument('--steps',type=int,nargs=2,default=(130,65))
    ap.add_argument('--strength',type=float,default=1.)
    ap.add_argument('--which',type=int,nargs='*',default=[])
    ap.add_argument('--name',default='initial')
    ap.add_argument('--seed',default=None)
    ap.add_argument('--targeted',action='store_true')
    ap.add_argument('--root-only',action='store_true')
    ap.add_argument('--start',default=None)
    ap.add_argument('--late-shoes',action='store_true')
    ap.add_argument('--early-ankles',action='store_true')
    args=ap.parse_args()
    out=JOB/args.name
    out.mkdir(exist_ok=True)
    reports=[]
    for k,window in enumerate(config['focus_pts']):
        if args.which and k+1 not in args.which:continue
        index=np.searchsorted(pts,config['camera_cut_continuity']['cut_pts'][k])
        f=CutFit(index,np.searchsorted(pts,window),args.strength)
        previous=f.report(curve)
        start=None
        if args.seed is not None:
            sys.modules.setdefault('fit_cuts',sys.modules[__name__])
            from refine_cut_guard import CutTrialGuard
            start=curve_from_packet(load(JOB/args.seed/f'cut_{k+1}.npz'))
            if args.start is not None:
                start=curve_from_packet(load(JOB/args.start/f'cut_{k+1}.npz'))
            if args.targeted or args.root_only or args.late_shoes or args.early_ankles:
                free = np.ones((len(f.ctrl),23,1))
                if args.root_only:
                    free[:,1:] = 0.
                elif args.early_ankles:
                    # Preserve the viewed silhouette at both flanks while
                    # distributing the old toe/ankle turn over the cut.
                    inside=(curve.knots[f.ctrl]>=clock[index]-.08)&(curve.knots[f.ctrl+4]<=clock[index]+.14)
                    free[~inside,1:]=0.
                    free[:,np.array([0,3,6,9,12,13,14,15,16,17,18,19,20,21])+1]=0.
                elif args.late_shoes:
                    earliest = clock[index]-(.042 if k==0 else .085)
                    latest = clock[index]+(.19 if k==0 else .26)
                    inside=(curve.knots[f.ctrl]>=earliest)&(curve.knots[f.ctrl+4]<=latest)
                    free[~inside,1:]=0.
                    if k==4:
                        free[:,np.array([2,5,8,11])+1]=0.
                elif k in (5,7):
                    free[:,np.array([3,6,9,12,13,14,15,16,17,18,19,20,21])+1]=0.
                    if k == 5:
                        inside = (curve.knots[f.ctrl]>=clock[index]-.34)&(curve.knots[f.ctrl+4]<=clock[index]+.34)
                        free[~inside,2:]=0.
                elif k in (0,4):
                    earliest = clock[index]-(.10 if k==4 else .34)
                    latest = clock[index]+(.39 if k==4 else .245)
                    inside = (curve.knots[f.ctrl]>=earliest)&(curve.knots[f.ctrl+4]<=latest)
                    free[~inside,1:]=0.
                f.free=tn(free)
                f.fit.set_control_goal(start.position_controls,start.rotation_controls)
                f.fit.parameter.data.mul_(f.free)
                start=f.fit.geometry()
            f.refine_guard=CutTrialGuard(f,start)
            if args.seed != 'refined' and not (args.targeted or args.root_only or args.late_shoes or args.early_ankles):
                surface_cache=f.refine_guard.surface_cache
                if not surface_cache.is_file():
                    surface_cache.parent.mkdir(exist_ok=True)
                    atomic_npz(surface_cache,**f.refine_guard.surface.current_scan)
        candidate=f.run(*args.steps,initial=start)
        after=f.report(candidate)
        # One continuous curve. The full-stage loader applies a declared
        # camera transition to the original parameter file, and verifies this
        # proposal against the new camera identity.
        ident=dict(packet,camera_binding_json=np.array(json.dumps(camera.info,sort_keys=True)))
        save_candidate(ident,candidate,out/f'cut_{k+1}.npz',parent=PARENT)
        report=dict(before=previous,after=after,changed_controls=f.ctrl.tolist())
        reports.append(report)
        atomic_json(out/f'cut_{k+1}.json',report)
        print(json.dumps(dict(event='cut_fit_result',**report),ensure_ascii=False),flush=True)
    atomic_json(out/'results.json',reports)
