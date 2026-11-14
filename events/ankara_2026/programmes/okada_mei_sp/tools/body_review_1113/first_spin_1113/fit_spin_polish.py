from spin_model import (
    JOB, r, camera, doc, curve, clock, pts, seconds, ratio, obs, roles, mass, assembly, config, anchor,
    state, Fit, Cavity, DenseIce, DeformedUpper, SurfaceBounds, cpu, ti, tn, load, read,
    ground_samples, anatomy_values, anatomical, angular_steps, angular_accel, sq, robust, log_event,
    save_candidate, PARENT, atomic_json, moving_projection, curve_from_packet,
)

import argparse
from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from experiments.skate_video_v1.inputs import PROJECT
from spin_source_rows import ProtectedSkateRows
import json
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import torch


class SpinFit:
    def __init__(self, start=None, *, replay=None):
        self.active=np.arange(22)
        start_time,end_time=clock[np.searchsorted(pts,config['focus_pts'][0])]
        self.ctrl=np.flatnonzero((curve.knots[:-4]<end_time)&(curve.knots[4:]>start_time))
        self.support=curve.knots[[self.ctrl[0],self.ctrl[-1]+4]]
        self.frames=np.flatnonzero((clock>self.support[0]-.06)&(clock<self.support[1]+.06))
        f=self.frames
        self.fit=Fit(r,camera,curve,f,self.active,self.ctrl)
        self.param=torch.nn.Parameter(torch.zeros_like(self.fit.parameter))
        self.scale=tn([.012]+[.10]*22)[None,:,None]
        self.origin=curve if start is None else start
        self.base=state(curve,clock[f])
        self.s=seconds[f]
        t=self.s
        # Eased emphasis through takeoff, landing and release. The rotation
        # path retains distinct rates for the camel and catch-foot positions.
        stable=(t>=81.62)&(t<=92.72)
        edge=np.minimum(np.clip((t-81.13)/.32,0.,1.),np.clip((93.18-t)/.4,0.,1.))
        self.hold=tn((1-edge)**2)
        self.stable=ti(np.flatnonzero(stable))
        self.flight=ti(np.flatnonzero((t[1:-1]>=81.36)&(t[1:-1]<=81.46)))
        self.entry=ti(np.flatnonzero((t[1:-1]>=81.16)&(t[1:-1]<=81.68)))
        self.entry_steps=ti(np.flatnonzero((t[1:]>=81.20)&(t[1:]<=81.68)))
        self.mid_steps=ti(np.flatnonzero((t[1:]>=81.78)&(t[1:]<=92.50)))
        self.outer=tn((1-np.minimum(np.clip((t-81.10)/.17,0.,1.),
                                     np.clip((93.24-t)/.17,0.,1.)))**2)
        self.w=tn(.18+.82*edge[1:-1])
        self.turn_w=tn((.12+.88*stable[1:-1])*edge[1:-1]+.20*np.exp(-((t[1:-1]-81.45)/.12)**2))
        self.co_w=tn((.35+.65*stable[1:-1])*edge[1:-1])
        self.angle_branch=tn(anchor['reference_twist'][f])
        self.anatomy0=tn(anatomy_values(self.base,f))
        guard=load(JOB/'parent_replay/scoped_guard_baseline.npz')
        self.anatomy_cap=tn(guard['anatomy'].max(axis=(0,2)))
        checked=load(JOB/'parent_replay/scoped_guard_baseline.npz')['angle']
        self.max_step=tn(np.deg2rad(checked[f[:-1]].max(0)+config['guard']['world_angular_step_budget_deg']-.45))
        self.P0=tn(self.base['P'])
        self.L0=tn(self.base['local_R'])
        self.Q0=tn(self.base['Q'])
        self.root_range_t=np.unique(np.r_[np.arange(self.support[0],self.support[1],.012), curve.knots[self.ctrl]])
        self.root_basis=tn(curve.basis_sparse(self.root_range_t,2)[:,self.ctrl].toarray()*ratio)
        self.root_accel=tn(curve.position_derivative(self.root_range_t,2)*ratio)
        root_s=self.root_range_t+seconds[0]
        self.root_w=tn(.3+.7*np.minimum(np.clip((root_s-81.1)/.35,0,1),np.clip((93.26-root_s)/.30,0,1)))
        self.edge=ground_samples()
        # Preserve the low-frequency direction of travel of the observed
        # support as it enters and leaves the spin.
        R0=self.base['R']
        support_pivot=(self.base['O'][:,1] + R0[:,1]@self.material_centre())*ratio
        self.support_line=tn(gaussian_filter1d(support_pivot,7.,axis=0,mode='nearest'))
        source_heading=np.unwrap(np.arctan2(self.base['Q'][:,1,0],self.base['Q'][:,0,0]))
        phase_step=np.diff(source_heading)
        self.guide_step=tn(gaussian_filter1d(phase_step,5.5,mode='nearest'))
        self.zlift=tn(component_ice(r.binding,self.base)[1][:,:,0])
        self.setup_image()
        self.protected=ProtectedSkateRows(self,r,doc,JOB/'parent_replay')
        baseline=self.fit.state()
        self.protected.replay(self.fit,baseline)
        self.setup_grasp()
        self.cavity=Cavity(r,self.fit,self.base)
        self.ice=None
        if replay is not None:
            alltimes=load(replay/'solve/candidate.npz')['physics_times']
            dense=alltimes[(alltimes>self.support[0])&(alltimes<self.support[1])]
            self.ice=DenseIce(self,dense,state(curve,dense),self.edge,config['ice']['tolerance_m'])
        self.surface=None

    def material_centre(self):
        geometry=r.binding.packet()
        # Mid-rocker under the foot, used as an actual material point for
        # tracking. Z contact is computed from the full runner separately.
        p=geometry['parameters']
        return np.array([.43*p.sole_length_m,0.,np.min(geometry['template']['vertices'][:,2])])

    def setup_image(self):
        f=self.frames
        u,k=np.nonzero(obs['use'][f])
        mapped=obs['model_joint_ids'][f[u],k]
        self.obs_frame,self.obs_label=ti(u),ti(mapped)
        self.uv=tn(obs['uv'][f[u],k])
        self.sigma=tn(np.maximum(obs['sigma_px'][f[u],k,None],2.5))
        self.image_gain=tn(np.where(k<5,.48,np.where(k<13,1.,1.05)))
        baseline=self.fit.state()
        self.body_uv0=self.fit.project(self.fit.coco(baseline)).detach()[self.obs_frame,self.obs_label]
        self.upper=DeformedUpper(self)
        self.whole=[]
        self.dirs=[]
        self.source_shoe_count=0
        for foot in (0,1):
            selected=np.flatnonzero(roles['role'][f,foot]==1)
            if not len(selected):
                continue
            self.source_shoe_count+=len(selected)
            nodes=state(curve,self.fit.clock.reshape(len(f),5)[selected].ravel())
            shape=assembly.points(nodes,foot,transforms=(nodes['R'],nodes['O']))
            uv=moving_projection(camera,shape.reshape(len(selected),5,-1,3),f[selected])['uv']
            match=[assembly.associations(a,roles['region'][f[i],foot]) for a,i in zip(uv,selected,strict=True)]
            idx,blend,target,normals=map(np.stack,zip(*match,strict=True))
            old=assembly.points(nodes,foot,(idx,blend),(nodes['R'],nodes['O']))
            old_uv=moving_projection(camera,old.reshape(len(selected),5,-1,3),f[selected])['uv']
            old_distance=np.median(np.linalg.norm(old_uv-target,axis=-1),axis=1)
            radius=roles['sigma_px'][f[selected],foot]
            endpoints=[self.upper.weights(assembly.vertices[idx[:,:,j]][:,None],
                         (idx[:,:,j]<assembly.upper_count)[:,None]) for j in (0,1)]
            self.whole.append((foot,selected,endpoints,tn(blend),tn(target),tn(normals),tn(radius),
                               tn(old_distance+2.6*radius)))
        for foot in (0,1):
            use=[row for row in doc['whole_skate_source']['directed_boot_rows'] if row['foot']==foot
                 and pts[f[0]]<=row['pts']<=pts[f[-1]]]
            if not use:continue
            ff=np.searchsorted(pts[f],[row['pts'] for row in use])
            foot_model=r.binding.packet()
            local=np.array([[0.,0.,foot_model['floor'][0]],[foot_model['xs'][-1],0.,foot_model['floor'][-1]]])
            target=np.stack([np.subtract(row['toe_uv'],row['heel_uv']) for row in use])
            sigma=np.array([row['sigma_px'] for row in use])
            self.dirs.append((foot,ff,tn(local),tn(target),tn(sigma)))
        self.facing=[]
        led=read(PROJECT/
                 'events/ankara_2026/programmes/okada_mei_sp/programme_action_ledger_1107.json')
        episode=next(row for row in led['episodes'] if row['episode_id']=='flying_camel_path')
        for e in episode['facing_observations']:
            i=int(np.searchsorted(pts[f],e['pts']))
            assert pts[f[i]]==e['pts']
            target=camera.camera_centres(f[i:i+1])[0,:2]
            self.facing.append((i,tn(target),1. if e['view']=='front' else -1.,np.cos(np.deg2rad(e['tolerance_deg']))))

    def setup_grasp(self):
        grasp=r.grasp_relations['grip_interval']
        held=(pts[self.frames]>=grasp['start_pts'])&(pts[self.frames]<=grasp['end_pts'])
        self.held=np.flatnonzero(held)[::2]
        self.hand=grasp['hand']
        self.foot=grasp['foot']
        spec=r.grasp_geometry['palm_patches'][self.hand]
        ids=spec['vertex_ids']
        surface=r.binding.surface
        self.palm_w=tn(spec['weights'])
        self.palm_skin=tn(surface.weights[ids])
        self.palm_pose=tn(surface.posedirs.reshape(486,-1,3)[:,ids].reshape(486,-1))
        self.palm_shape=tn(surface.shaped[ids])
        self.grip_segment=tn(r.grasp_geometry['segment_local_solver'])
        base=self.fit.state()
        self.grip_reference=self.grasp(base).detach()

    def grasp(self, st):
        P,Q,_,A,J,_,R,O,L=st
        keep=2+5*self.held
        A,J,L=A[keep],J[keep],L[keep]
        t=J-(A@self.fit.J[None,:,:,None]).squeeze(-1)
        local=self.palm_shape+((L-self.fit.eye).reshape(len(keep),486)@self.palm_pose).reshape(len(keep),-1,3)
        blend=(self.palm_skin@A.reshape(len(keep),55,9)).reshape(len(keep),-1,3,3)
        mesh=(blend@local[...,None]).squeeze(-1)+self.palm_skin@t
        palm=((mesh*self.palm_w[None,:,None]).sum(dim=1)-self.fit.J[0])*r.body.scale
        world=P[keep]+(Q[keep]@palm[...,None]).squeeze(-1)
        rel=((world-O[keep,self.foot])[...,None,:]@R[keep,self.foot]).squeeze(1)
        a,b=self.grip_segment
        step=b-a
        progress=(((rel-a)*step).sum(-1)/step.square().sum()).clamp(0,1)
        return torch.linalg.vector_norm(rel-a-progress[:,None]*step,dim=-1)*ratio

    def source_loss(self,st):
        P,Q,L,A,J,_,R,O,_=st
        uv=self.fit.project(self.fit.coco(st))[self.obs_frame,self.obs_label]
        image=12.*(robust((uv-self.uv)/self.sigma).sum(-1)*self.image_gain).mean()
        hold=3.*(sq((uv-self.body_uv0)/11.)*self.hold[self.obs_frame]).mean()
        alpha=self.upper.alpha(self.fit,A,J).reshape(-1,5,2,3)
        outline=uv.sum()*0.
        for foot,sel,endpoints,fraction,target,normal,radius,limit in self.whole:
            r1=R.reshape(-1,5,2,3,3)[sel,:,foot]
            p1=O.reshape(-1,5,2,3)[sel,:,foot]
            a,b=[self.upper.apply(v,alpha[sel,:,foot]) for v in endpoints]
            vertices=a+fraction[:,None,:,None]*(b-a)
            xyz=p1[:,:,None]+vertices@r1.transpose(-1,-2)
            uv1=self.fit.project(xyz.flatten(0,1),sel)
            error=uv1-target
            outline+=1.50*robust((error*normal).sum(-1)/radius[:,None]).mean()
            distance=torch.linalg.vector_norm(error,dim=-1).median(dim=1).values
            outline+=5.*(torch.relu(distance-limit)/2.).square().amax()
        direction=uv.sum()*0.
        for foot,sel,xy,target,sigma in self.dirs:
            r1=R.reshape(-1,5,2,3,3)[sel,:,foot]
            p1=O.reshape(-1,5,2,3)[sel,:,foot]
            world=p1[:,:,None]+xy@r1.transpose(-1,-2)
            img=self.fit.project(world.flatten(0,1),sel)
            chord=img[:,1]-img[:,0]
            direction+=1.65*robust((chord-target)/sigma[:,None]).sum(-1).mean()
        cone=uv.sum()*0.
        for i,centre,sign,cosine in self.facing:
            aim=(Q[2+5*i]@A[2+5*i,9])[:2,2]
            aim=aim/torch.linalg.vector_norm(aim).clamp_min(.08)
            view=centre-P[2+5*i,:2]
            view=view/torch.linalg.vector_norm(view).clamp_min(.08)
            cone+=3.*torch.relu(cosine-sign*(view*aim).sum()).square()
        return dict(body_pixels=image,source_flanks=hold,whole_skate=outline,directed_support=direction,
                    source_facing_cones=cone, source_shoe_rows=self.protected.loss(self.fit,st))

    def com(self,st):
        P,Q,L,A,J,world,R,O,_=st
        mid=slice(2,None,5)
        A_world=Q[mid,None]@A[mid][:,mass.ids]
        C=world[mid][:,mass.ids]*ratio+torch.einsum('nsij,sj->nsi',A_world,tn(mass.offsets[:mass.count]))
        shoes=O[mid]*ratio+torch.einsum('nfij,fj->nfi',R[mid],tn(mass.offsets[mass.count:]))
        return torch.einsum('s,nsc->nc',tn(mass.mass),torch.cat((C,shoes),dim=1))

    def motion_loss(self,st):
        P,Q,L,A,J,W,R,O,_=st
        mid=slice(2,None,5)
        p,q,l,w,boot,shoe=P[mid],Q[mid],L[mid],W[mid],R[mid],O[mid]
        G=self.com(st)
        local=torch.cat((q[:,None],l),dim=1)
        qx=q[:,:2,0]
        u=qx/torch.linalg.vector_norm(qx,dim=-1,keepdim=True).clamp_min(.06)
        x,y=u.unbind(-1)
        zeros,ones=torch.zeros_like(x),torch.ones_like(x)
        spin_axes=torch.stack((x,-y,zeros,y,x,zeros,zeros,zeros,ones),dim=-1).reshape(len(x),3,3)
        yaw=torch.atan2(x[:-1]*y[1:]-y[:-1]*x[1:],(u[:-1]*u[1:]).sum(-1))
        yaw_change=torch.diff(yaw)/.02**2
        # Same positive phase keeps accelerating or decelerating from the
        # observed cues. Smooth departures from the local rate, no target at 0.
        direction=torch.relu(.32*self.guide_step-yaw)[self.stable[:-1]]
        rate_loss=1.65*(robust(yaw_change/38.)*self.turn_w).mean()
        rate_loss+=.52*robust((yaw-self.guide_step)/.075).mean()
        rate_loss+=2.4*(direction/.07).square().mean()
        # Measured angular acceleration in the turning frame removes the
        # constant common body spin while exposing independent limb jitter.
        rotating_R=spin_axes[:,None].transpose(-1,-2)@(q[:,None]@A[mid][:,[0,9,12,15,4,5,16,17]])
        rotating_skate=spin_axes[:,None].transpose(-1,-2)@boot
        rotating_pose=torch.cat((rotating_R,rotating_skate),dim=1)
        rel_acc=angular_accel(rotating_pose,.02)
        axes=torch.cat((q[:,None]@A[mid][:,[9,15]],boot),dim=1)
        world_step=angular_steps(axes)
        world_norm=torch.linalg.vector_norm(world_step,dim=-1)
        body_angular=(robust(angular_accel(local,.02)/75.).sum(-1)*self.w[:,None]).mean()*.67
        true_shoe_angular=(robust(angular_accel(boot,.02)/100.).sum(-1)*self.w[:,None]).mean()*.50
        peak_step=(torch.relu(torch.linalg.vector_norm(angular_steps(local),dim=-1).amax(0)-self.max_step)/.019).square().sum()*24.
        phase_sync=(robust(rel_acc/68.).sum(-1)*self.co_w[:,None]).mean()*.88
        rs=angular_steps(boot)[:,1,2]
        phase_sync+=1.9*(torch.relu(-rs[self.stable[:-1]])/.07).square().mean()
        phase_sync+=.62*(robust((rs-yaw)/.14)[self.stable[:-1]]).mean()
        real_com=torch.diff(G,n=2,dim=0)/.02**2
        # The stable core has centripetal acceleration. Only its excursions
        # beyond this broad engineering scale are penalized in world space.
        vertical=robust(real_com[:,2]/9.)
        root_vertical=self.root_accel+self.root_basis@self.fit.parameter[:,0]
        root_loss=.20*(robust(root_vertical/38.).sum(-1)*self.root_w).mean()
        root_loss+=.32*(robust(root_vertical[:,2]/22.)*self.root_w).mean()
        ground=(shoe[:,1]+(boot[:,1]@tn(self.material_centre())[:,None]).squeeze(-1))*ratio
        rel=(w[:,[0,4,5,7,8,12,15,16,17,18,19,20,21]]*ratio-ground[:,None])@spin_axes
        com_rel=((G-ground)[:,None]@spin_axes).squeeze(1)
        body_pos=(robust(torch.diff(rel,n=2,dim=0)/(.02**2*18.)).sum(-1)*self.co_w[:,None]).mean()*1.0
        body_pos+=.85*(robust(torch.diff(com_rel,n=2,dim=0)/(.02**2*14.)).sum(-1)*self.co_w).mean()
        body_pos+=.70*(vertical*self.w).mean()
        # Same support over multiple rotations, with finite lateral drift.
        idx=self.stable
        drift=ground[idx,:2]-self.support_line[idx,:2]
        steady=robust(torch.diff(ground[idx,:2],n=2,dim=0)/(.02**2*18.)).sum(-1).mean()*.90
        steady+=.75*robust(drift/.065).sum(-1).mean()
        air=real_com[self.flight]-tn([0.,0.,-9.81])
        flight=.72*robust(air/13.).sum(-1).mean()
        # Retain onset and release posture while adjusting the full spin.
        prior=.6*(sq((p-self.P0)*ratio/.14)*(1.+6.*self.hold)).mean()
        prior+=3.*((l-self.L0).square().mean((1,2,3))*(1.+9.*self.hold)).mean()
        prior+=3.*((q-self.Q0).square().mean((1,2))*(1.+8.*self.hold)).mean()
        prior+=40.*(self.outer*(sq((p-self.P0)*ratio/.022)
                               +8.*(l-self.L0).square().mean((1,2,3))
                               +8.*(q-self.Q0).square().mean((1,2)))).mean()
        # The airborne entrance really turns. Penalize the isolated speed
        # excursions in all axes, and evaluate the entire transition.
        entry=torch.diff(world_step,dim=0)[self.entry,:2]
        entry_loss=.8*robust(entry/.12).sum(-1).mean()
        too_fast=world_norm[self.entry_steps,:2]-tn(np.deg2rad([30.,36.]))
        entry_loss+=4.*(torch.relu(too_fast)/.12).square().mean()
        overshoot=world_norm[self.mid_steps]-tn(np.deg2rad([19.,23.,25.,23.]))
        entry_loss+=3.*robust(torch.relu(overshoot)/.095).mean()
        entry_loss+=.90*(robust(angular_accel(rotating_skate,.02)/80.).sum(-1)*self.co_w[:,None]).mean()
        return dict(signed_spin=rate_loss,turning_frame=phase_sync,local_angular=body_angular, transient_rotation=entry_loss,
                    actual_shoe_angular=true_shoe_angular,relative_limb_and_com=body_pos,root_curvature=root_loss, joint_step_cap=peak_step,
                    support_drift=steady,visible_flight=flight,prior=prior)

    def structure_loss(self,st):
        _,_,L,A,J,_,R,O,_=st
        mid=slice(2,None,5)
        A,J,R,O=A[mid],J[mid],R[mid],O[mid]
        Z=R[...,2,:]
        height=torch.stack([(Z@sample.T+O[...,2,None]).amin(-1)*ratio for sample in self.edge[:2]],dim=-1)
        root_contact=height.amin(-1)
        support=root_contact[self.stable,1]
        contact=(torch.relu(support-.026)/.016).square().mean()*.66
        contact+=.85*(torch.relu(-support-.007)/.018).square().mean()
        contact+=.20*(torch.relu(self.zlift[:,0]-.10-root_contact[:,0])/.10).square().mean()
        anatomy=anatomical(L[mid],self.angle_branch)
        new_anatomy=(anatomy-self.anatomy0).clamp_min(0.)
        local=(anatomy.square().mean()*.9
               + 3.*(torch.relu(anatomy.amax((0,2))-self.anatomy_cap+1.e-5)/.010).square().sum()*30.
               + 5.*(torch.relu(new_anatomy.mean(0)-.018)/.025).square().sum())
        grip=self.grasp(st)
        hold=.52*robust(torch.relu(grip-.038)/.040).square().mean()
        hold+=2.8*robust(torch.relu(grip-self.grip_reference-.012)/.019).mean()
        if self.ice is not None:
            contact+=self.ice.loss(self.fit.parameter)
        else:
            cap=torch.minimum(self.zlift.amin(0),tn([-.009,-.009]))
            contact+=6.*(torch.relu(cap-height[:,:,0])/.011).square().mean()
        losses=dict(ice_contact=contact,anatomy=local,visible_grasp=hold)
        losses.update(self.cavity.loss(st))
        if self.surface is not None:
            losses['collision_witnesses']=self.surface.loss(self.fit.parameter)
        return losses

    def evaluate(self,v, strength,grad=True):
        self.param.data.copy_(tn(v.reshape(self.param.shape)))
        self.param.grad=None
        self.fit.parameter=self.param*self.scale
        st=self.fit.state()
        parts=self.source_loss(st)
        parts.update({k:(value if k in ('prior','visible_flight') else strength*value)
                      for k,value in self.motion_loss(st).items()})
        parts.update(self.structure_loss(st))
        loss=sum(parts.values())
        if not torch.isfinite(loss):
            raise ValueError({k:float(c.detach()) for k,c in parts.items()})
        if grad:
            loss.backward()
            return float(loss.detach()),cpu(self.param.grad).ravel()
        return float(loss.detach()),{k:float(c.detach()) for k,c in parts.items()}

    def solve(self,name,stage_steps,strengths):
        path=JOB/'proposals'
        path.mkdir(exist_ok=True)
        log_event(event='spin_fit_start',source_seconds=self.s[[0,-1]].tolist(),
                  active_controls=len(self.ctrl),body_rows=len(self.uv),source_whole_skate=self.source_shoe_count,
                  held_source_times=len(self.held))
        self.fit.set_control_goal(self.origin.position_controls,self.origin.rotation_controls)
        value=cpu(self.fit.parameter/self.scale).ravel()
        for stage,(strength,iterations) in enumerate(zip(strengths,stage_steps,strict=True)):
            before,parts=self.evaluate(value,strength,False)
            log_event(event='initial',stage=stage,strength=strength,total=before,components=parts)
            start=time.monotonic()
            i=0
            def progress(xx):
                nonlocal i
                i+=1
                if i%20==0:
                    loss,details=self.evaluate(xx,strength,False)
                    log_event(event='spin_iteration',stage=stage,iteration=i,total=loss,
                              elapsed_s=time.monotonic()-start,components=details)
                if i%60==0:
                    self.evaluate(xx,strength,False)
                    save_candidate(packet,self.fit.geometry(),path/f'{name}_stage_{stage}_{i}.npz',parent=PARENT)
            result=minimize(self.evaluate,value,args=(strength,),jac=True,method='L-BFGS-B',
                bounds=[(-4.5,4.5)]*len(value),callback=progress,
                options=dict(maxiter=iterations,ftol=2.e-8,maxcor=20,gtol=8.e-5))
            value=result.x
            after,parts=self.evaluate(value,strength,False)
            log_event(event='spin_stage',stage=stage,iteration=int(result.nit),total=after,
                elapsed_s=time.monotonic()-start,components=parts,status=str(result.message))
            save_candidate(packet,self.fit.geometry(),path/f'{name}.npz',parent=PARENT)
        return self.fit.geometry()


if __name__=='__main__':
    from spin_model import packet
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='spin_fit')
    p.add_argument('--strength',type=float,nargs='+',default=[1.,1.5])
    p.add_argument('--iterations',type=int,nargs='+',default=[120,100])
    p.add_argument('--seed')
    p.add_argument('--surface',action='store_true')
    p.add_argument('--full-ice',action='store_true')
    p.add_argument('--surface-target')
    args=p.parse_args()
    seed=None if args.seed is None else curve_from_packet(load(JOB/'proposals'/f'{args.seed}.npz'))
    fit=SpinFit(seed,replay=JOB/'parent_replay' if args.full_ice else None)
    if args.surface:
        scan=load(doc['programme_structure']['surface_witnesses'])
        target=fit.origin if args.surface_target is None else curve_from_packet(load(JOB/'proposals'/f'{args.surface_target}.npz'))
        fit.surface=SurfaceBounds(fit,target,r,scan,doc,fit.support,state)
    fit.solve(args.name,args.iterations,args.strength)
