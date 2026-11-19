"""World-space dynamics on the saved articulated spline, on the broadcast clock."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse as sp

from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, RotationChartDomainError, _exp, _log
from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve, curve_from_packet
from calibration.program_tracking.body_tracking.body_gpu_linear import CuDSS
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from compare import bind
from joint_source_objective import LANDMARK, material_points, source_shoe_rows
from opening_support import samples
from source_joint_refit import POLICY, log, save
from source_review import digest, whole_skate_observations


def block_matrix(blocks):
    n=len(blocks)
    mat=sp.bsr_matrix((blocks,np.arange(n),np.arange(n+1)),
                      shape=(n*blocks.shape[1],n*blocks.shape[2])).tocsr()
    mat.eliminate_zeros()
    return mat


def differences(n, order, dt):
    from math import comb
    coeff=[(-1)**(order-i)*comb(order,i)/dt**order for i in range(order+1)]
    return sp.diags(coeff,np.arange(order+1),shape=(n-order,n),format='csr')


def project(camera, world, scale):
    # Native projection finite differences at the middle of the exposure; the
    # native rolling-shutter residual at the parent is added separately.
    xyz=world/scale-camera[:,None,13:16]
    ray=np.einsum('nij,nkj->nki',camera[:,4:13].reshape(-1,3,3),xyz)
    x,y=(ray[:,:,:2]/ray[:,:,2:]).transpose(2,0,1)
    k1,k2,p1,p2=camera[:,20:].T[:,:,None];r=x*x+y*y
    rad=1+r*(k1+r*k2)
    u=camera[:,16,None]*(x*rad+2*p1*x*y+p2*(r+2*x*x))+camera[:,18,None]
    v=camera[:,17,None]*(y*rad+p1*(r+2*y*y)+2*p2*x*y)+camera[:,19,None]
    return np.stack((u,v),axis=-1)


class WorldMotion:
    def __init__(self, prepared, parent, ledger, output):
        self.parent,self.output=parent,output
        self.ref,self.cam,_,_,self.packet=bind(prepared,parent/'source.json',parent/'candidate.npz')
        self.seed=self.ref.absolute_curve
        self.ratio=self.ref.binding.ratio
        self.mass=ContinuousMassGeometry(self.ref)
        t=self.ref.times
        self.t=np.empty(len(t)*2-1)
        self.t[::2]=t
        self.t[1::2]=(t[:-1]+t[1:])*.5
        self.dt=float(np.median(np.diff(self.t)))
        np.testing.assert_allclose(np.diff(self.t),self.dt,atol=1e-10)
        self.B=self.seed.basis_sparse(self.t)
        self.D=[None]+[differences(len(self.t),i,self.dt) for i in (1,2,3)]
        self.restricted=np.unique(self.seed.basis_sparse(samples(self.ref,POLICY)).indices)
        base=self.ref.at(self.t)
        self.unchanged={k:base[k] for k in ('skate_relative_R','hand_local_R') if k in base}
        self.episodes=self.ref.source_document['source_episode_review']['episodes']
        self.stamps=self.ref.p['pts'][0]+self.t*12800.
        self.flight=np.zeros(len(self.t)-2,bool)
        self.support=np.zeros((len(self.t),2),bool)
        for e in self.episodes:
            on=(self.stamps>=e['start_pts']-.01)&(self.stamps<=e['end_pts_inclusive']+.01)
            if e['kind']=='flight_visible_candidate':
                self.flight |= on[:-2]&on[1:-1]&on[2:]
            if e['kind']=='support_candidate' and e.get('foot') in (0,1):
                self.support[on,e['foot']]=True
        self.support[self.stamps<=POLICY['pts'][1]+.01]=True
        self.mat=material_points(self.ref.binding)
        self.parents=self.ref.body.parents[:22]
        self.ancestor=np.zeros((22,22),bool)
        for i in range(22):
            k=i
            while k>=0:
                self.ancestor[i,k]=True
                if k==0:break
                k=int(self.parents[k])
        with np.load(parent/'result.npz') as result:
            rr=result['native_body_FK_world_R_mid'][:,LANDMARK]
            self.attach=np.einsum('ntji,ntj->nti',rr,result['COCO_world_mid']-result['native22_world_mid'][:,LANDMARK])
        with np.load(ledger) as z:
            np.testing.assert_array_equal(z['pts'],self.ref.p['pts'])
            self.use=np.zeros_like(z['use'])
            self.target=np.zeros_like(z['uv'],dtype=float)
            ii=np.arange(len(t))[:,None];ids=z['model_joint_ids']
            self.target[ii,ids]=np.nan_to_num(z['uv'])
            self.use[ii,ids]=z['use']
        with np.load(parent/'joint_source_measurements.npz') as z:
            if str(z['candidate'])!=digest(parent/'candidate.npz'):raise ValueError('Source measurement identity')
            self.parent_uv=np.concatenate((z['uv'],z['shoe'][:,:,:7].reshape(len(t),14,2)),axis=1)
        self.seed_state=self.state(self.seed)
        self.offset=self.parent_uv-project(self.cam.midpoint['values'],self.seed_state['photo'],self.cam.scale*self.ratio)
        self.protect=np.zeros((len(t),31))
        # The complete paired boot and blade photo estimate, including its
        # reviewed point/contour factors, travels with the preceding result.
        self.protect[:,17:]=1/6
        self.protect[:,17+6]*=2
        self.protect[:,17+7+6]*=2
        self.protect[abs(t+22.12-53.5)<=.08,23]=1.3
        self.protect[abs(t+22.12-24.32)<=.16,14]=.6
        self.protect[abs(t+22.12-63.50)<=.07,7]=.6
        self.camera_scale=self.cam.scale*self.ratio
        self.detailed=False
        self.edge_review=False
        self.root_trust=1.8

    def source_factors(self,work,shared=None):
        from types import SimpleNamespace
        from calibration.program_tracking.body_tracking.visible_skate_union import VisibleSkateUnion
        from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
        from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
        if shared is None:shared=self.parent.parent
        with np.load(shared/'parent_material_uv.npz') as z:self.surface=z['material'][7:]
        with np.load(self.parent/'joint_source_measurements.npz') as z:
            self.surface_uv=z['shoe'][:,:,7:]
        document=json.loads(whole_skate_observations().read_text(encoding='utf8'))
        pts=self.ref.p['pts']
        with np.load(shared/'before_source_measurements.npz') as z:
            self.named=tuple(map(np.asarray,zip(*source_shoe_rows(document,pts,z['shoe']))))
        with np.load(work/'input/shoe_occlusion_source.npz') as z:
            np.testing.assert_array_equal(pts,z['source_pts'])
            self.complete=(z['role']==1)&z['visible']&~z['partial_boundary']&(z['source_pair_overlap'][:,None]<.12)
            self.dirs=np.stack((np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)),axis=1)
            self.ext=np.max(np.nan_to_num(z['region'])@self.dirs.T,axis=2)
        for rec in document.get('source_outline_review_exclusions',[]):
            if 'pts' in rec and 'foot' in rec:self.complete[np.searchsorted(pts,rec['pts']),rec['foot']]=False
        s=self.seed_state;self.cameras=self.cam.midpoint['values']
        self.surface_offset=[]
        for ft in (0,1):
            xyz=self.surface_xyz(s,ft)
            self.surface_offset.append(self.surface_uv[:,ft]-project(self.cameras,xyz,self.camera_scale))
        state=SequenceState(self.ref,self.packet['seconds'])
        prob=SimpleNamespace(reference=self.ref,binding=self.ref.binding,camera=self.cam,body=SimpleNamespace(state=state))
        path=work/'source_review/current_visible_arcs.json'
        fac=VisibleSkateUnion(prob,dict(path=str(path),sha256=digest(path),exact_template=True,reviewed_open_arcs=True),np.arange(len(pts)))
        fac.associate(np.zeros(state.size));rows=[]
        for batch,match in zip(fac.batches,fac.matches):
            frames,times,base,target,radius,scale=batch;fk=fk_state(self.ref,base,np.zeros((len(times),69)))
            for foot in (0,1):
                points=fac.assembly.points(fk,foot,match[foot],(fk['R'],fk['O'])).reshape(len(frames),5,-1,3)
                uv=moving_projection(self.cam,points,frames)['uv'];mid=np.arange(len(frames))*5+2
                loc=np.einsum('nij,nkj->nki',fk['R'][mid,foot].swapaxes(-1,-2),points[:,2]-fk['O'][mid,foot,None])
                for k,j in zip(*np.where((match[2]==bool(foot))&(scale>0))):
                    rows.append((frames[k],foot,loc[k,j],target[k,j],uv[k,j],scale[k,j]/radius[k,j]))
        self.arcs=tuple(map(np.asarray,zip(*rows)))
        t,feet,local,target,uv,w=self.arcs
        xyz=s['boot_O'][2*t,feet]+(s['boot_R'][2*t,feet]@local[:,:,None])[:,:,0]*self.ratio
        self.arc_offset=uv-project(self.cameras[t],xyz[:,None],self.camera_scale)[:,0]
        self.detailed=True
        log(self.output,'measured_factors',source_body=int(self.use.sum()),named=len(self.named[0]),
            complete=int(self.complete.sum()),arcs=len(t),source_frames=len(pts))

    def surface_xyz(self,s,foot):
        return s['boot_O'][::2,foot,None]+np.einsum('nij,kj->nki',s['boot_R'][::2,foot],self.surface)*self.ratio

    def state(self,curve):
        v=evaluate_curve(curve,self.t)
        base={**self.unchanged,**v}
        s=fk_state(self.ref,base,np.zeros((len(self.t),69)))
        centres,_=self.mass.from_pose(s)
        pos=s['native_world']*self.ratio
        axes=s['Q'][:,None]@s['frames']
        shoes=(s['O'][:,:,None]+np.einsum('nfij,kj->nfki',s['R'],self.mat))*self.ratio
        coco=pos[::2,LANDMARK]+np.einsum('ntij,ntj->nti',axes[::2,LANDMARK],self.attach)
        photo=np.concatenate((coco,shoes[::2].reshape(len(coco),14,3)),axis=1)
        heights=[];low=[]
        for start in range(0,len(self.t),512):
            k=slice(start,start+512);features=batch_features(self.ref.binding,s['R'][k],s['O'][k])
            mask=np.isin(features['semantics'],['runner','toe_pick'])
            h=features['height'][:,:,mask];index=h.argmin(-1)
            local=np.take_along_axis(features['local_point'][:,:,mask],index[:,:,None,None],axis=2)[:,:,0]
            low.append((s['O'][k]+np.einsum('nfij,nfj->nfi',s['R'][k],local))*self.ratio)
            heights.append(h.min(-1))
        return dict(com=self.mass.com(centres),centres=centres,pos=pos,axes=axes,shoes=shoes,photo=photo,
                    low=np.concatenate(low),height=np.concatenate(heights),boot_R=s['R'],boot_O=s['O']*self.ratio)

    def jac(self,xyz,labels,group,st,rows,rigid_skate_points=0):
        n,v=xyz.shape[:2];jac=np.zeros((n,v,3,len(group)*3))
        for k,joint in enumerate(group):
            if joint==-2:
                # d/d log(body scale). Rigid skate geometry is translated
                # through its ankle; the hardware dimensions stay constant.
                lever=xyz-st['pos'][rows,0,None,:]
                if rigid_skate_points:
                    ankles=np.asarray(labels[-rigid_skate_points:])
                    lever[:,-rigid_skate_points:]=st['pos'][rows][:,ankles]-st['pos'][rows,0,None,:]
                jac[:,:,:,3*k]=lever
                continue
            if joint==-1:
                jac[:,:,:,3*k:3*k+3]=np.eye(3)*self.ratio
                continue
            mask=self.ancestor[np.asarray(labels),joint]
            if not mask.any():continue
            pivot=st['pos'][rows,joint];axes=st['axes'][rows,joint].swapaxes(-1,-2)
            arms=xyz[:,mask]-pivot[:,None]
            jac[:,mask,:,3*k:3*k+3]=np.cross(axes[:,None,:,:],arms[:,:,None,:]).swapaxes(-1,-2)
        return jac

    def system(self,curve,group,linear=True):
        st=self.state(curve);nd=len(group)*3;n=len(self.t);cost={}
        h=sp.csr_matrix((n*nd,n*nd));gradient=np.zeros(n*nd)
        def term(name,values,jac=None,op=None):
            nonlocal h,gradient
            r=np.asarray(values,dtype=float).ravel()
            cost[name]=cost.get(name,0.)+float(r@r)/len(self.ref.times)
            if linear:
                mat=block_matrix(jac)
                if op is not None:mat=op@mat
                h=h+mat.T@mat;gradient+=mat.T@r
        centres=st['centres'];com=st['com']
        if linear:
            mass_jac=self.jac(centres,list(self.mass.ids)+[7,8],group,st,slice(None),2)
            cj=np.einsum('k,nkij->nij',self.mass.mass,mass_jac)
            del mass_jac
        else:cj=None
        def motion(name,value,jac,order,scale,target=0.):
            derivative=self.D[order];w=np.broadcast_to(1./np.asarray(scale),(derivative.shape[0],))*.70710678
            r=(derivative@value-target)*w[:,None]
            op=sp.kron(sp.diags(w)@derivative,sp.eye(value.shape[1]),format='csr') if linear else None
            term(name,r,jac,op)
        grav=np.zeros((n-2,3));grav[self.flight,2]=-9.81
        com_weight=getattr(self,'com_weight',1.)
        motion('com_acc',com,cj,2,np.where(self.flight,.65,3.0)/com_weight,grav)
        motion('com_jerk',com,cj,3,75./com_weight)
        if self.detailed:
            for joint in (18,19,20,21):
                x=st['pos'][:,joint];wj=self.jac(x[:,None],[joint],group,st,slice(None))[:,0] if linear else None
                motion('arm_jerk',x,wj,3,540.)
        if hasattr(self,'left_knee_world_weight'):
            knee=st['pos'][:,4]
            weight=self.left_knee_world_weight
            kj=self.jac(knee[:,None],[4],group,st,slice(None))[:,0] if linear else None
            term('left_knee_displacement',(knee-self.seed_state['pos'][:,4])*weight[:,None],
                 kj*weight[:,None,None] if linear else None)
        for foot,joint in enumerate((7,8)):
            for k in (3,4):
                x=st['shoes'][:,foot,k];sj=self.jac(x[:,None],[joint],group,st,slice(None),1)[:,0] if linear else None
                if foot==0 and hasattr(self,'runner_height_cap'):
                    excess=np.maximum(0.,x[:,2]-self.runner_height_cap)
                    weight=self.runner_cap_weight*(excess>0)
                    term('reviewed_runner_height',excess*weight,sj[:,2:3]*weight[:,None,None] if linear else None)
                motion('runner_acc',x,sj,2,np.where(self.support[1:-1,foot],12.,32.)*np.sqrt(2.))
                motion('runner_jerk',x,sj,3,np.where(self.support[1:-2,foot],260.,580.)*np.sqrt(2.))
            x=st['low'][:,foot];z=x[:,2]
            # Source-reviewed glide intervals have finite support. Any airborne
            # blade retains its height, with one-sided physical nonpenetration.
            target=np.where(self.support[:,foot]|(z<0),0.,z)
            ice_scale,contact_scale=(1450.,420.) if self.edge_review else (600.,180.)
            if hasattr(self,'review_contact_weight'):
                contact_scale=self.review_contact_weight[:,foot]
                ice_scale=np.maximum(2400.,contact_scale)
            weight=np.where(z<0,ice_scale,np.where(self.support[:,foot],contact_scale,0.))*.70710678
            jac=self.jac(x[:,None],[joint],group,st,slice(None),1)[:,0,2:3] if linear else None
            term('ice',(z-target)*weight,jac*weight[:,None,None] if linear else None)
            # Lateral slip at the instantaneous material point nearest the
            # plane. Hold that material coordinate fixed during differentiation.
            R=st['boot_R'][:,foot];O=st['shoes'][:,foot,0]
            local=(R[1:-1].swapaxes(-1,-2)@(x[1:-1]-O[1:-1])[:,:,None])[:,:,0]
            tangent=R[1:-1,:2,0];side=np.column_stack((-tangent[:,1],tangent[:,0],np.zeros(n-2)))
            side/=np.maximum(np.linalg.norm(side,axis=1,keepdims=True),.1)
            slip_scale=.13 if self.edge_review else 1.1
            if hasattr(self,'support_slip_scale'):
                slip_scale=self.support_slip_scale[1:-1,foot]
            weight=(self.support[1:-1,foot]&self.support[:-2,foot]&self.support[2:,foot])*.70710678/slip_scale
            a=O[:-2]+(R[:-2]@local[:,:,None])[:,:,0];b=O[2:]+(R[2:]@local[:,:,None])[:,:,0]
            vel=np.sum((b-a)*side,axis=1)/(2*self.dt)
            cost['reviewed_lateral_slip']=cost.get('reviewed_lateral_slip',0.)+float((vel*weight)@(vel*weight))/len(self.ref.times)
            if linear:
                ja=self.jac(a[:,None],[joint],group,st,slice(None,-2),1)[:,0]
                jb=self.jac(b[:,None],[joint],group,st,slice(2,None),1)[:,0]
                A=block_matrix(np.einsum('ni,nij->nj',side,ja)[:,None,:])
                B=block_matrix(np.einsum('ni,nij->nj',side,jb)[:,None,:])
                zero=sp.csr_matrix((n-2,2*nd))
                J=sp.diags(weight/(2*self.dt))@(sp.hstack([zero,B])-sp.hstack([A,zero]))
                if self.edge_review:
                    # Signed side speed changes when an articulated ankle yaws
                    # the runner, including when its centre is stationary.
                    # Omit the derivative for a near-vertical runner with an
                    # unresolved horizontal tangent.
                    forward=R[1:-1,:,0]
                    horizontal=np.linalg.norm(tangent,axis=1)
                    relative_velocity=(b-a)/(2*self.dt)
                    ds=np.zeros((n-2,1,nd))
                    for col,j in enumerate(group):
                        if j<0 or not self.ancestor[joint,j]:continue
                        axes=st['axes'][1:-1,j].swapaxes(-1,-2)
                        df=np.cross(axes,forward[:,None,:])
                        draw=np.stack((-df[:,:,1],df[:,:,0],np.zeros_like(df[:,:,0])),axis=-1)
                        normal=side[:,None,:]
                        dn=(draw-normal*np.sum(draw*normal,axis=-1,keepdims=True))/np.maximum(horizontal[:,None,None],.1)
                        ds[:,0,3*col:3*col+3]=np.sum(dn*relative_velocity[:,None,:],axis=-1)*(horizontal>.1)[:,None]
                    if foot==0 and getattr(self,'differentiate_rocker',False):
                        # The material point touching the ice moves along the
                        # rocker as the boot tilts. Differentiate it at the
                        # middle instant, holding that point for the velocity.
                        rows=np.where(self.local_fit_rows[1:-1]&self.support[1:-1,0])[0]
                        mid=rows+1
                        scale=1.e-4
                        def witness(rot):
                            ft=batch_features(self.ref.binding,rot[:,None],np.zeros((len(rot),1,3)))
                            steel=np.isin(ft['semantics'],['runner','toe_pick'])
                            k=ft['height'][:,0,steel].argmin(1)
                            return ft['local_point'][:,0][:,steel][np.arange(len(rows)),k]*self.ratio
                        derivative=[]
                        for axis in np.eye(3)*scale:
                            plus=witness(R[mid]@_exp(axis))
                            minus=witness(R[mid]@_exp(-axis))
                            derivative.append((plus-minus)/(2*scale))
                        local_derivative=np.stack(derivative,axis=2)
                        dvel=(R[mid+1]-R[mid-1])@local_derivative/(2*self.dt)
                        for col,j in enumerate(group):
                            if j>=0 and self.ancestor[joint,j]:
                                axes=R[mid].swapaxes(1,2)@st['axes'][mid,j]
                                ds[rows,0,3*col:3*col+3]+=np.einsum('ni,nij->nj',side[rows],dvel@axes)
                    pad=sp.csr_matrix((n-2,nd))
                    J=J+sp.diags(weight)@sp.hstack([pad,block_matrix(ds),pad])
                h=h+J.T@J;gradient+=J.T@(vel*weight)
            if foot==0 and hasattr(self,'native_slip_scale'):
                mid=np.flatnonzero(self.local_fit_rows & self.support[:,0] & (np.arange(n)%2==0))
                mid=mid[(mid>=2)&(mid<n-2)]
                mid=mid[self.support[mid-2,0]&self.support[mid+2,0]]
                material=local[mid-1,:,None]
                lo=O[mid-2]+(R[mid-2]@material)[:,:,0]
                hi=O[mid+2]+(R[mid+2]@material)[:,:,0]
                native_vel=(hi-lo)/(4*self.dt)
                side50=side[mid-1]
                speed=(native_vel*side50).sum(1)/self.native_slip_scale
                cost['native_steel_slip']=float(speed@speed)/len(self.ref.times)
                if linear:
                    jlo=self.jac(lo[:,None],[joint],group,st,mid-2,1)[:,0]
                    jhi=self.jac(hi[:,None],[joint],group,st,mid+2,1)[:,0]
                    sidejac=np.zeros((len(mid),nd))
                    for col,j in enumerate(group):
                        if j<0 or not self.ancestor[joint,j]:continue
                        ax=st['axes'][mid,j].swapaxes(-1,-2)
                        direction=np.cross(ax,R[mid,None,:,0])
                        dsxy=np.stack([-direction[:,:,1],direction[:,:,0],np.zeros_like(direction[:,:,0])],axis=-1)
                        normal=side50[:,None]
                        length=horizontal[mid-1,None,None]
                        dsxy=(dsxy-normal*(dsxy*normal).sum(-1,keepdims=True))/np.maximum(length,.1)
                        dsxy*=length>.1
                        slot=slice(3*col,3*col+3)
                        sidejac[:,slot]=(dsxy*native_vel[:,None]).sum(-1)
                        if getattr(self,'differentiate_rocker',False):
                            ix=np.searchsorted(rows,mid-1)
                            dvel=(R[mid+2]-R[mid-2])@local_derivative[ix]/(4*self.dt)
                            sidejac[:,slot]+=np.einsum('ni,nij->nj',side50,dvel@(R[mid].swapaxes(1,2)@st['axes'][mid,j]))
                    lojac=-np.einsum('ni,nij->nj',side50,jlo)/(4*self.dt)
                    hijac=np.einsum('ni,nij->nj',side50,jhi)/(4*self.dt)
                    entries=np.stack((lojac,sidejac,hijac),axis=1)/self.native_slip_scale
                    cols=(mid[:,None]+np.array([-2,0,2]))[:,:,None]*nd+np.arange(nd)
                    jj=sp.coo_matrix((entries.ravel(),(np.repeat(np.arange(len(mid)),3*nd),cols.ravel())),
                                     shape=(len(mid),n*nd)).tocsr()
                    h=h+jj.T@jj;gradient+=jj.T@speed
        photo=st['photo'];cam=self.cam.midpoint['values'];uv=project(cam,photo,self.camera_scale)+self.offset
        # Source landmarks and preservation of separately reviewed rigid-boot
        # image features. Huber-like weighting reduces wrong-arm associations.
        if linear:
            optical=np.stack([(project(cam,photo+np.eye(3)[k]*1e-5,self.camera_scale)-uv+self.offset)/1e-5 for k in range(3)],axis=-1)
            xyzjac=self.jac(photo,LANDMARK+[7]*7+[8]*7,group,st,slice(None,None,2),14)
            uvjac=optical@xyzjac
        nx=len(uv)
        # Framewise sums avoid assembling a many-million-row pixel Jacobian.
        body_weight=self.use*(.33 if self.detailed else .125)
        if self.edge_review:
            body_weight[:,5:11]*=2.2
            body_weight[:,13:15]*=1.7
            body_weight[:,13:15]*=getattr(self,'knee_weight',1.)
        for name,residual,weight,start,end in [
                ('body_pixels',uv[:,:17]-self.target,body_weight,0,17),
                ('source_foot_and_review',uv-self.parent_uv,self.protect,0,31)]:
            length=np.linalg.norm(residual,axis=-1);delta=7.
            cost[name]=float(np.sum(weight**2*2*delta**2*(np.sqrt(1+(length/delta)**2)-1)))/nx
            if linear:
                wt=weight/(1+(length/delta)**2)**.25
                jj=(uvjac[:,start:end]*wt[:,:,None,None]).reshape(nx,-1,nd)
                rhs=(residual*wt[:,:,None]).reshape(nx,-1)
                blocks=np.zeros((n,nd,nd));blocks[::2]=jj.swapaxes(1,2)@jj
                h=h+block_matrix(blocks)
                gradient.reshape(n,nd)[::2]+=np.einsum('nki,nk->ni',jj,rhs)
        if self.detailed:
            def observations(name,indices,residual,weight,J):
                nonlocal h,gradient
                rho=np.sqrt(1+(residual/5.)**2)
                cost[name]=cost.get(name,0.)+float(np.sum(weight**2*50*(rho-1)))/nx
                if linear:
                    w=weight**2/rho
                    HH=np.zeros((n,nd,nd));gg=np.zeros((n,nd))
                    np.add.at(HH,2*indices,np.einsum('ni,nj,n->nij',J,J,w))
                    np.add.at(gg,2*indices,J*(w*residual)[:,None])
                    h=h+block_matrix(HH);gradient+=gg.ravel()
            if hasattr(self,'reviewed_image_regions'):
                fr,idx,lo,hi,weight=self.reviewed_image_regions
                residual=uv[fr,idx]-np.clip(uv[fr,idx],lo,hi)
                for axis in range(2):
                    observations('reviewed_image_region',fr,residual[:,axis],weight,
                                 uvjac[fr,idx,axis] if linear else None)
            fr,ft,idx,lo,hi,ww=self.named;ii=17+ft*7+idx
            delta=uv[fr,ii]-np.clip(uv[fr,ii],lo,hi)
            weights=np.clip(ww*4,.35,2.5)*(1.35 if self.edge_review else 1.)
            weights*=getattr(self,'shoe_source_weight',1.)
            for axis in range(2):
                observations('named_source',fr,delta[:,axis],weights,uvjac[fr,ii,axis] if linear else None)
            for foot,joint in enumerate((7,8)):
                fr=np.flatnonzero(self.complete[:,foot]);idx=np.arange(len(fr))
                xyz=self.surface_xyz(st,foot)[fr]
                mid_uv=project(cam[fr],xyz,self.camera_scale)
                projected=mid_uv+self.surface_offset[foot][fr]
                dots=projected@self.dirs.T;v=np.argmax(dots,axis=1)
                if linear:
                    optical=np.stack([(project(cam[fr],xyz+np.eye(3)[k]*1e-5,self.camera_scale)-mid_uv)/1e-5 for k in range(3)],axis=-1)
                    dj=optical@self.jac(xyz,[joint]*len(self.surface),group,st,2*fr,len(self.surface))
                for direction,unit in enumerate(self.dirs):
                    r=np.max(dots[:,:,direction],axis=1)-self.ext[fr,foot,direction]
                    J=np.einsum('i,nij->nj',unit,dj[idx,v[:,direction]]) if linear else None
                    weight=(.32 if self.edge_review else .22)*getattr(self,'shoe_source_weight',1.)
                    if foot==0:weight*=getattr(self,'left_outline_weight',1.)
                    observations('complete_outline',fr,r,weight,J)
                    if foot==0 and hasattr(self,'outline_reference_px'):
                        limits=self.outline_reference_px[fr,direction]
                        excess=np.sign(r)*np.maximum(np.abs(r)-limits,0.)
                        observations('outline_reference_guard',fr,excess,weight*1.8*(np.abs(r)>limits),J)
            fr,feet,local,target,direct,w=self.arcs
            for foot,joint in enumerate((7,8)):
                take=feet==foot;t=fr[take]
                xyz=st['boot_O'][2*t,foot]+(st['boot_R'][2*t,foot]@local[take,:,None])[:,:,0]*self.ratio
                uv0=project(cam[t],xyz[:,None],self.camera_scale)[:,0]
                err=uv0+self.arc_offset[take]-target[take]
                if linear:
                    opt=np.stack([(project(cam[t],xyz[:,None]+np.eye(3)[k]*1e-5,self.camera_scale)[:,0]-uv0)/1e-5 for k in range(3)],axis=-1)
                    jac=opt@self.jac(xyz[:,None],[joint],group,st,2*t,1)[:,0]
                for axis in range(2):
                    observations('observed_boot_blade_arc',t,err[:,axis],np.clip(17*w[take],.6,3.5),jac[:,axis] if linear else None)
        # A finite world trust anchors the monocular depth gauge. Translational
        # continuity across camera cuts is handled in the same world basis.
        x=st['pos'][:,0];jac=self.jac(x[:,None],[0],group,st,slice(None))[:,0] if linear else None
        term('world_position_trust',(x-self.seed_state['pos'][:,0])*self.root_trust,jac*self.root_trust if linear else None)
        rotations=[j for j in group if j>=0]
        # One boot orientation can be reproduced by many combinations of hip,
        # knee and ankle rotations. Regularize the actual accumulated change
        # from the source-reviewed articulated state, including its time rate.
        prior=getattr(self,'pose_reference',self.seed)
        pose_weight=getattr(self,'pose_trust_scale',20.)
        rate_weight=getattr(self,'pose_rate_trust_scale',1.25)
        for j in rotations:
            delta=_log(prior.rotation_controls[:,j].swapaxes(-1,-2)@curve.rotation_controls[:,j])
            v=self.B@delta
            cost['joint_pose_trust']=cost.get('joint_pose_trust',0.)+float(np.sum((v*pose_weight)**2)+np.sum((self.D[1]@v*rate_weight)**2))/len(self.ref.times)
            if linear:
                select=np.zeros((3,nd));i=3*group.index(j);select[:,i:i+3]=np.eye(3)
                jac=sp.kron(sp.eye(n),select,format='csr')
                A=sp.kron(self.D[1],select,format='csr')
                h=h+pose_weight**2*(jac.T@jac)+rate_weight**2*(A.T@A)
                gradient+=pose_weight**2*(jac.T@v.ravel())+rate_weight**2*(A.T@(self.D[1]@v).ravel())
        extra=getattr(self,'control_objective',None)
        if extra:
            extra_cost,extra_jac,extra_residual=extra(curve,group,linear)
            cost.update(extra_cost)
        if not linear:return cost
        if getattr(self,'exact_rotation_basis',False):
            rows=np.flatnonzero(self.local_fit_rows)
            geom=evaluate_curve(curve,self.t[rows],with_blocks=True)
            if min(group)<-1:raise ValueError('Unsupported variable in local curve basis')
            blocks=geom['rotation_right_blocks'][:,:,np.maximum(group,0),:,:].copy()
            if -1 in group:
                blocks[:,:,group.index(-1)]=geom['position_weights'][:,:,None,None]*np.eye(3)
            columns=geom['control_indices']
            nctrl=len(curve.position_controls)
            k=np.arange(len(group))*3
            rid=rows[:,None,None,None,None]*nd+k[None,None,:,None,None]+np.arange(3)[None,None,None,:,None]
            cid=columns[:,:,None,None,None]*nd+k[None,None,:,None,None]+np.arange(3)[None,None,None,None,:]
            M=sp.coo_matrix((blocks.ravel(),(np.broadcast_to(rid,blocks.shape).ravel(),
                np.broadcast_to(cid,blocks.shape).ravel())),shape=(n*nd,nctrl*nd)).tocsr()
        else:
            M=sp.kron(self.B,sp.eye(nd),format='csr')
        H=(M.T@h@M).tocsr();g=M.T@gradient
        if extra:
            H=H+extra_jac.T@extra_jac;g+=extra_jac.T@extra_residual
        return cost,H,g

    def optimize(self,curve,group):
        nd=len(group)*3;start=time.perf_counter()
        costs,H,g=self.system(curve,group)
        active=np.ones((len(curve.position_controls),nd),bool)
        if hasattr(self,'active_knots'):
            active[~self.active_knots]=False
        for key,j in enumerate(group):
            if j in getattr(self,'joint_knots',{}):
                active[~self.joint_knots[j],3*key:3*key+3]=False
            if j==-1:active[self.restricted,3*key+2]=False
            elif j in (0,1,2,4,5,7,8):active[self.restricted,3*key:3*key+3]=False
        indices=np.flatnonzero(active.ravel())
        # Damping of the physical angular/position increments includes the fine
        # control-grid null directions; no independent per-exposure shoe poses.
        damping=np.tile([2. if j==-1 else 14. for j in group],(len(active),1)).repeat(3,axis=1).ravel()**2
        A=(H+sp.diags(damping)).tocsr()[indices][:,indices].tocsr()
        A.sort_indices();A.eliminate_zeros()
        matrix=torch.sparse_csr_tensor(torch.as_tensor(A.indptr,device='cuda',dtype=torch.int32),
            torch.as_tensor(A.indices,device='cuda',dtype=torch.int32),torch.as_tensor(A.data,device='cuda'),size=A.shape)
        step,device=CuDSS().solve(matrix,torch.as_tensor(-g[indices],device='cuda'))
        dx=np.zeros_like(g);dx[indices]=step.cpu().numpy();dx=dx.reshape(len(active),len(group),3)
        if getattr(self,'preserve_step_direction',False):
            # Jointwise clipping can turn a coupled contact-preserving update
            # into an ascent direction. Scale the entire solved step together.
            dx*=min(1.,.32/max(np.linalg.norm(dx,axis=2).max(),1e-9))
        for k,j in enumerate(group):
            if j>=0:
                dx[:,k]*=np.minimum(1.,.32/np.maximum(1e-9,np.linalg.norm(dx[:,k],axis=1)))[:,None]
        alphas=(1.,.5,.25,.125,.0625,.03125)
        if getattr(self,'preserve_step_direction',False):alphas+=(.015625,.0078125,.00390625)
        for alpha in alphas:
            pos=curve.position_controls.copy();rot=curve.rotation_controls.copy()
            for k,j in enumerate(group):
                if j==-1:pos+=alpha*dx[:,k]
                else:rot[:,j]=rot[:,j]@_exp(alpha*dx[:,k])
            trial=AbsoluteMotionCurve(curve.knots,pos,rot,curve.edge_reference_quaternion)
            try:
                trial.check_continuous_edge_update(curve)
            except RotationChartDomainError as e:
                log(self.output,'world_trial_rejected',group=group,alpha=alpha,reason=str(e))
                continue
            loss=self.system(trial,group,False)
            accepted=sum(loss.values())<sum(costs.values())
            log(self.output,'world_motion_step',group=group,alpha=alpha,accepted=accepted,before=costs,after=loss,
                columns=len(indices),nnz=A.nnz,device=device['backend'],device_solve_seconds=device['solve_seconds'],
                elapsed_seconds=time.perf_counter()-start)
            if accepted:return trial,(sum(costs.values())-sum(loss.values()))/sum(costs.values())
        return curve,0.


def run(args):
    args.output.mkdir(exist_ok=True)
    shutil.copyfile(args.parent/'source.json',args.output/'source.json')
    stage=WorldMotion(args.prepared,args.parent,args.ledger,args.output)
    curve=stage.seed;cycle=0
    if args.work:
        stage.source_factors(args.work)
        stage.edge_review=args.edge_review
        with np.load(args.start) as z:curve=curve_from_packet(z)
    checkpoint=args.output/'candidate.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            receipt=json.loads(str(z['joint_source_fit_json']))
            if receipt['parent_sha256']!=digest(args.parent/'candidate.npz'):raise ValueError('Parent of resume checkpoint differs')
            cycle=receipt['cycle']+1;curve=curve_from_packet(z)
    while True:
        change=0.
        groups=[[-1]] if cycle==0 and not args.work else [[-1,1,4,7],[-1,2,5,8],[-1,0]]
        if stage.edge_review:groups.append([3,6,9,12])
        if args.work:groups.extend(([13,16,18,20],[14,17,19,21]))
        for group in groups:
            curve,gain=stage.optimize(curve,group);change+=gain
        report=dict(stage='world_dynamics_refit',cycle=cycle,improvement=change,
            parent=str(args.parent/'candidate.npz'),parent_sha256=digest(args.parent/'candidate.npz'),
            camera='s14_boards_1118',ice='lane4_side_1118',world_physics_rate_hz=1./stage.dt,
            blade_heading_jacobian=stage.edge_review,reviewed_glide_scale_mps=.13 if stage.edge_review else 1.1,
            refinement_start_sha256=digest(args.start) if args.start else None)
        save(args.output,'candidate.npz',stage.packet,curve,stage.ref,report)
        if cycle and change<.004:break
        cycle+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('prepared','parent','ledger','output'):
        parser.add_argument(key,type=lambda s:Path(s).resolve())
    parser.add_argument('--work',type=Path)
    parser.add_argument('--start',type=Path)
    parser.add_argument('--edge-review',action='store_true')
    run(parser.parse_args())
