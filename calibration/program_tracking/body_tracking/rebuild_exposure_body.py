"""Moving-body rolling shutter on the unchanged native camera core.

Three historical or five split articulated samples define exposure motion.
The five-node mode respects the source-pose velocity knot at frame center. Both camera and body use the same predicted row time.
The implicit body-row derivative is retained in every optimization factor.
"""
import time,json
import numpy as np
import cv2
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation,Slerp
from scipy.sparse import coo_matrix,diags
from .rebuild_body_fit import ArticulatedSplineProblem


def row_weights(row,node_count=3):
    row=np.asarray(row);inside=(row>=0)&(row<=1079)
    # Preserve the historical out-of-image endpoint convention. Such points
    # remain explicitly flagged; this is not physical exposure evidence.
    r=2*np.clip(row,0,1079)/1079-1
    if node_count==3:
        return np.stack((r*(r-1)/2,1-r*r,r*(r+1)/2),-1),np.stack((r-.5,-2*r,r+.5),-1)*(2/1079)*inside[...,None]
    if node_count!=5:raise ValueError('Exposure interpolation supports 3 or 5 nodes')
    # The source-pose Slerp has a velocity knot at the exposure midpoint.
    # Split there instead of smoothing both one-sided velocities together.
    left=r<0;q=np.where(left,2*r+1,2*r-1)
    w=np.stack((q*(q-1)/2,1-q*q,q*(q+1)/2),-1)
    dw=np.stack((q-.5,-2*q,q+.5),-1)*(4/1079)*inside[...,None]
    value=np.zeros((*row.shape,5));derivative=np.zeros_like(value)
    for j in range(3):
        value[...,j]+=w[...,j]*left;value[...,j+2]+=w[...,j]*(~left)
        derivative[...,j]+=dw[...,j]*left;derivative[...,j+2]+=dw[...,j]*(~left)
    return value,derivative


def moving_projection(projector,nodes,frames,*,jacobian=False):
    node_count=nodes.shape[1]
    if node_count not in (3,5):raise ValueError("Expected 3 or 5 exposure nodes")
    uv=projector.project(nodes[:,node_count//2],frames)
    for iteration in range(64):
        weights,_=row_weights(uv[:,:,1],node_count);world=np.einsum('nkl,nlkc->nkc',weights,nodes)
        new=projector.project(world,frames)
        error=float(np.max(np.abs(new-uv)));uv=new
        if error<1e-8:break
    else:raise ValueError('Moving-body row projection failed to converge')
    weights,dw=row_weights(uv[:,:,1],node_count);world=np.einsum('nkl,nlkc->nkc',weights,nodes)
    out=dict(uv=uv,world=world,row_weights=weights,row_iterations=iteration+1,inside_exposure_rows=(uv[:,:,1]>=0)&(uv[:,:,1]<=1079))
    if jacobian:
        _,jp=projector.with_jacobian(world,frames)
        wrow=np.einsum('nkl,nlkc->nkc',dw,nodes)
        a=np.einsum('nkpc,nkc->nkp',jp,wrow);denom=1-a[:,:,1]
        if np.any(np.abs(a[:,:,1])>=1):raise ValueError('Moving-body row map is not locally contractive')
        pixel=jp+a[:,:,:,None]*jp[:,:,1,None,:]/denom[:,:,None,None]
        metric=np.broadcast_to(np.eye(3),(*world.shape[:2],3,3))+wrow[:,:,:,None]*jp[:,:,1,None,:]/denom[:,:,None,None]
        out['pixel_jac_nodes']=pixel[:,:,:,None,:]*weights[:,:,None,:,None]
        out['metric_jac_nodes']=metric[:,:,:,None,:]*weights[:,:,None,:,None]
        out['body_row_contraction']=float(np.max(np.abs(a[:,:,1])))
    return out


class ExposureArticulatedSplineProblem(ArticulatedSplineProblem):
    body_motion_model='quadratic exposure motion, implicit native predicted row; camera and body share image row time; out-of-image body uses bounded endpoint for diagnostic drawing only'
    def __init__(self,*args,exposure_nodes=3,**kwargs):
        if exposure_nodes not in (3,5):raise ValueError('Expected 3 or 5 exposure nodes')
        self.exposure_nodes=int(exposure_nodes)
        super().__init__(*args,**kwargs);self.node_cache={}
        self.body_motion_model=('split quadratic, five exact articulated samples' if exposure_nodes==5 else 'historical quadratic, three exact articulated samples')+'; implicit native predicted row; camera and body share row time; outside-image body uses flagged endpoints'
        self.priors['body_motion']=self.body_motion_model
        self.priors['exposure_nodes']=self.exposure_nodes
    def node_inputs(self,frames):
        key=tuple(np.asarray(frames,int).tolist())
        if key in self.node_cache:return self.node_cache[key]
        t=self.p['fit_time_seconds'];half=float(self.p['rig'][37])/2
        node_count=getattr(self,'exposure_nodes',3)
        if node_count==5 and abs(half)>float(np.min(np.diff(t)))+1e-12:
            raise ValueError('Exposure spans additional source-pose knots; use exact or knot-adaptive exposure evaluation')
        times=(t[frames,None]+half*np.linspace(-1,1,node_count)[None]).ravel()
        B=BSpline.design_matrix(times,self.knots,3,extrapolate=True).tocsr()
        clipped=np.clip(times,t[0],t[-1]);pose=np.empty((len(times),21,3))
        for j in range(21):pose[:,j]=Slerp(t,Rotation.from_rotvec(self.p['pose_axis_angle'][:,j]))(clipped).as_rotvec()
        q=Slerp(t,Rotation.from_matrix(self.p['raw_root_prior_Q_world']))(clipped).as_matrix()
        out=dict(B=B,pose=pose,qprior=q,bc=B.indices.reshape(len(frames),node_count,4),bv=B.data.reshape(len(frames),node_count,4))
        self.node_cache[key]=out;return out
    def kinematics(self,x,frames,all_coco=False):
        frames=np.asarray(frames,int);nodes=self.node_inputs(frames);B=nodes['B'];n=len(frames);node_count=nodes['bc'].shape[1]
        P=B@x[:3*self.m].reshape(self.m,3);dr=B@x[3*self.m:6*self.m].reshape(self.m,3)
        Q=Rotation.from_rotvec(dr).as_matrix()@nodes['qprior']
        delta=B@x[6*self.m:].reshape(self.m,self.dof) if self.articulate else np.zeros((node_count*n,self.dof))
        coco,native,blade=self.body.forward(nodes['pose'],delta)
        indices=np.arange(17) if all_coco else self.indices
        local=np.concatenate((coco[:,indices],blade.reshape(node_count*n,10,3)),axis=1)
        world=(np.einsum('nij,nkj->nki',Q,local)+P[:,None]).reshape(n,node_count,-1,3)
        return dict(P=P,Q=Q,dr=dr,delta=delta,local=local,world_nodes=world,inputs=nodes,frames=frames)
    def output_projections(self,x,frames):
        s=self.kinematics(x,frames,True);out=moving_projection(self.projection,s['world_nodes'],frames)
        return out
    def output_root_projection(self,x,frames):
        """Dynamic pelvis image location; legacy P_uv remains a mid-world marker."""
        frames=np.asarray(frames,int);nodes=self.node_inputs(frames)
        P=(nodes['B']@x[:3*self.m].reshape(self.m,3)).reshape(len(frames),nodes['bc'].shape[1],1,3)
        return moving_projection(self.projection,P,frames)
    def evaluate(self,x,*,jacobian=False):
        if self.cache_x is not None and np.array_equal(x,self.cache_x) and (not jacobian or self.cache['jac'] is not None):return self.cache
        tick=time.perf_counter();s=self.kinematics(x,self.frames);n=self.n;node_count=s['world_nodes'].shape[1];k=s['world_nodes'].shape[2]
        projected=moving_projection(self.projection,s['world_nodes'],self.frames,jacobian=jacobian)
        uv=projected['uv'];world=projected['world'];nodes=s['inputs']
        residual=[];rows=[];cols=[];vals=[];cursor=0;parts={}
        if jacobian:
            jq=np.empty((n*node_count,k,3,3));ja=np.zeros((n*node_count,k,3,self.dof))
            for i in range(n*node_count):
                _,drv=cv2.Rodrigues(s['dr'][i]);dQ=np.einsum('rab,bc->rac',drv.reshape(3,3,3),nodes['qprior'][i])
                jq[i]=np.einsum('rab,kb->kar',dQ,s['local'][i])
            if self.articulate:
                eps=1e-5
                for d in range(self.dof):
                    a=s['delta'].copy();b=s['delta'].copy();a[:,d]+=eps;b[:,d]-=eps
                    ga=self.body.forward(nodes['pose'],a);gb=self.body.forward(nodes['pose'],b)
                    la=np.concatenate((ga[0][:,self.indices],ga[2].reshape(n*node_count,10,3)),1)
                    lb=np.concatenate((gb[0][:,self.indices],gb[2].reshape(n*node_count,10,3)),1)
                    ja[:,:,:,d]=np.einsum('nij,nkj->nki',s['Q'],(la-lb)/(2*eps))
            jq=jq.reshape(n,node_count,k,3,3).transpose(0,2,1,3,4)
            ja=ja.reshape(n,node_count,k,3,self.dof).transpose(0,2,1,3,4)
            jp=projected['pixel_jac_nodes'];jm=projected['metric_jac_nodes']
            jrot=np.einsum('nkplc,nklcd->nkpld',jp,jq);jpose=np.einsum('nkplc,nklcd->nkpld',jp,ja)
            hrot=np.einsum('nkplc,nklcd->nkpld',jm,jq);hpose=np.einsum('nkplc,nklcd->nkpld',jm,ja)
        def add(res,name,frame_ids=None,dp=None,dq=None,da=None):
            nonlocal cursor
            flat=np.asarray(res,float).ravel();residual.append(flat);parts[name]=float(flat@flat/2)
            if jacobian and frame_ids is not None:
                f=np.asarray(frame_ids,int);rr=np.repeat(np.arange(cursor,cursor+len(flat)),4*node_count)
                for derivative,offset,dim in [(dp,0,3),(dq,3*self.m,3),(da,6*self.m,self.dof)]:
                    if derivative is None:continue
                    for j in range(dim):
                        rows.append(rr);cols.append((nodes['bc'][f]*dim+j+offset).ravel())
                        vals.append((derivative[:,:,j,None]*nodes['bv'][f]).ravel())
            cursor+=len(flat)
        count=len(self.indices);w=self.weight;err=(uv[:,:count]-self.uv)*w[:,:,None]
        fids=np.repeat(np.arange(n),count*2)
        add(err,'source_pixels',fids,
            (jp[:,:count]*w[:,:,None,None,None]).reshape(-1,node_count,3) if jacobian else None,
            (jrot[:,:count]*w[:,:,None,None,None]).reshape(-1,node_count,3) if jacobian else None,
            (jpose[:,:count]*w[:,:,None,None,None]).reshape(-1,node_count,self.dof) if jacobian and self.articulate else None)
        blades=world[:,count:].reshape(n,2,5,3)
        if self.contact:
            fi,foot=np.indices((n,2));fi,foot=fi.ravel(),foot.ravel();best=np.argmin(blades[:,:,:,2],axis=2).ravel()
            point=count+foot*5+best;height=blades[fi,foot,best,2]
            g=((height<-.02)&(~self.p['lower_body_inferred'][self.frames[fi]])).astype(float)/.03
            add(np.minimum(height+.02,0)*g,'ice_nonpenetration',fi,
                jm[fi,point,2]*g[:,None,None] if jacobian else None,
                hrot[fi,point,2]*g[:,None,None] if jacobian else None,
                hpose[fi,point,2]*g[:,None,None] if jacobian and self.articulate else None)
            for ci,c in self.contacts:
                ft=c['foot'];kk=int(np.argmin(blades[ci,ft,:,2]));point=count+ft*5+kk;z=blades[ci,ft,kk,2]
                lo,hi=self.evidence['height_interval_m'];height=min(z-lo,0)+max(z-hi,0);g=float(z<lo or z>hi)/self.evidence['height_residual_scale_m']
                add([height/self.evidence['height_residual_scale_m']],f'contact_height_{c["pts"]}',[ci],
                    jm[ci,point,2][None]*g if jacobian else None,
                    hrot[ci,point,2][None]*g if jacobian else None,
                    hpose[ci,point,2][None]*g if jacobian and self.articulate else None)
                region=np.asarray(c['region']);value=uv[ci,point];d=np.minimum(value-region[:2],0)+np.maximum(value-region[2:],0)
                g=((value<region[:2])|(value>region[2:])).astype(float)/self.evidence['region_residual_scale_px']
                add(d/self.evidence['region_residual_scale_px'],f'contact_region_{c["pts"]}',[ci,ci],
                    jp[ci,point]*g[:,None,None] if jacobian else None,
                    jrot[ci,point]*g[:,None,None] if jacobian else None,
                    jpose[ci,point]*g[:,None,None] if jacobian and self.articulate else None)
        def prior(matrix,controls,scale,offset,name):
            nonlocal cursor
            scale=np.broadcast_to(np.asarray(scale,float),(controls.shape[1],));v=(matrix@controls/scale).ravel();residual.append(v);parts[name]=float(v@v/2)
            if jacobian:
                co=matrix.tocoo();dim=controls.shape[1]
                rows.append((co.row[:,None]*dim+np.arange(dim)+cursor).ravel());cols.append((co.col[:,None]*dim+np.arange(dim)+offset).ravel());vals.append((co.data[:,None]/scale).ravel())
            cursor+=len(v)
        prior(self.acc,x[:3*self.m].reshape(self.m,3),(15.,15.,25.),0,'root_acceleration')
        rc=x[3*self.m:6*self.m].reshape(self.m,3)
        prior(self.curvature,rc,np.deg2rad(15.),3*self.m,'root_rotation_curvature');prior(diags(np.ones(self.m)),rc,np.deg2rad(30.),3*self.m,'root_rotation_prior')
        if self.articulate:
            ac=x[6*self.m:].reshape(self.m,self.dof)
            prior(diags(np.ones(self.m)),ac,np.deg2rad(10.),6*self.m,'local_pose_prior');prior(self.acc,ac,np.deg2rad(400.),6*self.m,'local_pose_correction_acceleration')
        r=np.concatenate(residual);J=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(r),self.size)).tocsr() if jacobian else None
        self.cache_x=np.array(x,copy=True);self.cache=dict(residual=r,jac=J,state=s,uv=uv,parts=parts,seconds=time.perf_counter()-tick)
        if jacobian:self.jac_evaluations+=1
        else:
            self.evaluations+=1
            if self.log:self.log.write(json.dumps(dict(event='evaluation',mode=self.mode,evaluation=self.evaluations,objective=float(r@r/2),parts=parts,elapsed_seconds=time.perf_counter()-self.start,eval_seconds=self.cache['seconds'],domain_failures=0,row_iterations=projected['row_iterations']))+'\n')
        return self.cache
