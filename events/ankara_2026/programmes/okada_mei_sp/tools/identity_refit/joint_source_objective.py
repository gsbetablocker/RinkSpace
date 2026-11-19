"""Joint source-image and motion objective for the registered Okada calibration.

All variable positions are obtained from one continuous articulated spline.
The rehearsal projection uses native midpoint Brown optics with a per-landmark
rolling-shutter correction; delivery and final checks use the exact native camera.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import shutil

from compare import bind
from source_joint_refit import POLICY, log, save
from opening_support import samples
from source_camera_rebind import write
from source_review import digest, whole_skate_observations
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, _exp
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.skate_template import profile_height

LANDMARK = [15,15,15,15,15,16,17,18,19,20,21,1,2,4,5,7,8]
FEATURES = ['heel','toe','sole_middle','runner_rear','runner_front','heel_block_rear','cuff_center']


def tensor(a):
    return torch.tensor(np.ascontiguousarray(a), device='cuda', dtype=torch.float32)


def index(a):
    return torch.as_tensor(a, device='cuda', dtype=torch.long)


def qmul(a,b):
    x,w=a[...,:3],a[...,3:]
    y,v=b[...,:3],b[...,3:]
    return torch.cat((w*y+v*x+torch.linalg.cross(x,y),w*v-(x*y).sum(-1,keepdim=True)),dim=-1)


def conj(q):
    return torch.cat((-q[...,:3],q[...,3:]),-1)


def qexp(v):
    a=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return torch.cat((.5*torch.sinc(a/(2*np.pi))*v,torch.cos(a*.5)),-1)


def qlog(q):
    v=q[...,:3];a=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return v*(2*torch.atan2(a,q[...,3:])/a.clamp_min(1e-9)).where(a>1e-7,2*torch.ones_like(a))


def matrix(q):
    x,y,z,w=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),
        1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)


def robust(v,scale):
    return scale*(torch.sqrt(v.square()+scale*scale)-scale)


def material_points(binding):
    p=binding.packet();s=p['parameters']
    heel=[0.,0.,p['floor'][0]];toe=[p['xs'][-1],0.,p['floor'][-1]]
    u=np.array([-s.runner_rear_extension_m,s.sole_length_m-s.runner_front_inset_m])
    runner=np.column_stack((u,np.zeros(2),profile_height(u,s)))
    return np.vstack([heel,toe,np.mean([heel,toe],0),runner,[0.,0.,s.rear_mount_height_m],p['upper'][np.unique(p['collar'])].mean(0)])


def quat(r):
    return Rotation.from_matrix(r.reshape(-1,3,3).copy()).as_quat().reshape(*r.shape[:-2],4)


def source_shoe_rows(document,pts,prior_uv):
    rows=[r for r in document['samples'] if r['role']=='fit']
    # Apply the same physical-track grouping and exclusivity as NativeShoePoints.
    from calibration.program_tracking.body_tracking.source_motion_refit import NativeShoePoints
    factor=NativeShoePoints(None,dict(samples=rows))
    assoc={}; costs={}
    for name,ix in factor.groups.items():
        score=np.zeros(2)
        for i in ix:
            r=rows[i]; t=np.searchsorted(pts,r['pts']); k=FEATURES.index(r['feature'])
            v=prior_uv[t,:,k];lo,hi=r.get('pixel_box',[r['uv'],r['uv']])
            score+=((v-np.clip(v,lo,hi))**2).sum(1)*factor.weight[i]/factor.sigma[i]**2
        costs[name]=score;assoc[name]=int(score.argmin())
    graph={name:set() for name in assoc}
    for names in factor.exclusive.values():
        if len(names)==2:
            a,b=names;graph[a].add(b);graph[b].add(a)
    seen=set()
    for a,neighbors in graph.items():
        if a in seen or not neighbors:continue
        col={a:0};queue=[a]
        while queue:
            j=queue.pop()
            for k in graph[j]:
                if k not in col:col[k]=1-col[j];queue.append(k)
                elif col[k]==col[j]:raise ValueError('Conflicting shoe associations')
        flip=int(np.argmin([sum(costs[k][v^i] for k,v in col.items()) for i in range(2)]))
        for k,v in col.items():assoc[k]=v^flip
        seen.update(col)
    foot=np.array([next(iter(o)) for o in factor.proposal_indices])
    for group,ix in factor.groups.items():foot[ix]=assoc[group]
    return [(np.searchsorted(pts,r['pts']),int(f),FEATURES.index(r['feature']),
             *r.get('pixel_box',[r['uv'],r['uv']]),np.sqrt(factor.weight[i])/factor.sigma[i]) for i,(r,f) in enumerate(zip(rows,foot))]


def basis(curve,t):
    b=curve.basis_sparse(t);n=len(curve.position_controls)
    i=np.clip(np.searchsorted(curve.knots,t,side='right')-4,0,n-4)[:,None]+np.arange(4)
    selected=b[np.arange(len(t))[:,None],i]
    w=selected.toarray() if hasattr(selected,'toarray') else np.asarray(selected)
    if w.shape!=i.shape:w=w.reshape(i.shape)
    return index(i),tensor(w)


def native_measurements(ref,cam,materials):
    from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
    p=ref.binding.packet()
    coll=np.unique(p['collar'])
    bodies=[];shoes=[];local_body=[]
    for start in range(0,len(ref.times),48):
        ids=np.arange(start,min(len(ref.times),start+48))
        t=ref.times[ids,None]+cam.readout_times(ids)[:,None]*np.linspace(-.5,.5,5)
        base=ref.at(t.ravel());v=ref.world(base)
        f=v['footwear'];R=f['boot_R'];O=f['boot_O']
        xyz=O[:,:,None]+np.einsum('nfij,vj->nfvi',R,materials)
        cuff=f['upper_local'][:,:,coll].mean(2)
        xyz[:,:,6]=O+np.einsum('nfij,nfj->nfi',R,cuff)
        whole=np.concatenate([v['coco'],xyz.reshape(t.size,-1,3)],1)
        uv=moving_projection(cam,whole.reshape(len(ids),5,-1,3),ids)['uv']
        bodies.append(uv[:,:17]);shoes.append(uv[:,17:].reshape(len(ids),2,-1,2))
        mid=np.arange(len(ids))*5+2
        attitudes=v['Q'][mid,None]@v['frames'][mid][:,LANDMARK]
        local_body.append(np.einsum('nkji,nkj->nki',attitudes,v['coco'][mid]-v['native'][mid][:,LANDMARK]))
    return np.concatenate(bodies),np.concatenate(shoes),np.concatenate(local_body)


class Objective:
    def __init__(self,args,ref,cam,packet):
        self.args=args; self.ref=ref; self.cam=cam
        self.curve=curve_from_packet(packet)
        self.n=len(ref.times); self.pts=ref.p['pts']
        curve=self.curve
        self.idx,self.b=basis(curve,ref.times)
        self.beta=self.b.flip(-1).cumsum(-1).flip(-1)[:,1:]
        self.q=tensor(quat(curve.rotation_controls));self.p=tensor(curve.position_controls)
        self.chart=tensor(curve.edge_reference_quaternion)
        self.delta=torch.nn.Parameter(torch.zeros((*self.p.shape[:1],23,3),device='cuda'))
        mask=np.ones(self.delta.shape[:2]);restricted=np.unique(curve.basis_sparse(samples(ref,POLICY)).indices)
        mask[np.ix_(restricted,[0,1,2,3,5,6,8,9])]=0
        self.mask=tensor(mask[:,:,None]); self.parents=ref.body.parents[:22]
        self.bone=tensor((ref.body.J[:22]-ref.body.J[self.parents])*ref.body.scale)
        self.levels=[];depth=np.zeros(22,int)
        for j in range(1,22):depth[j]=depth[self.parents[j]]+1
        order=np.argsort(depth,kind='stable');self.reorder=index(np.argsort(order))
        for l in range(1,depth.max()+1):
            children=order[depth[order]==l];par=order[depth[order]==l-1]
            self.levels.append((index(children),index([np.where(par==self.parents[j])[0][0] for j in children])))
        self.mapping=index(LANDMARK)
        self.ratio=ref.binding.ratio
        with np.load(args.parent/'result.npz') as z:
            self.prior_joint=tensor(z['native22_world_mid']/self.ratio)
            prior_R=z['native_body_FK_world_R_mid']
            j=z['native22_world_mid'][:,LANDMARK];c=z['COCO_world_mid']
            self.body_attach=tensor(np.einsum('nkji,nkj->nki',prior_R[:,LANDMARK],(c-j)/self.ratio))
            self.correction=tensor(z['COCO_uv'])
            self.before_uv=z['COCO_uv']
            shoot=z['skate_R_world_mid'];orig=z['skate_origin_world_mid']/self.ratio
            attR=prior_R[:,[7,8]];ank=z['native22_world_mid'][:,[7,8]]/self.ratio
            self.shoeR=tensor(attR.swapaxes(-1,-2)@shoot)
            self.shoeT=tensor(np.einsum('nfji,nfj->nfi',attR,orig-ank))
            self.before_local=tensor(z['local_R_mid'])
            self.shots=z['source_shot_ids']
            lower=ref.binding.packet()['template']['vertices']
            upper=ref.binding.packet()['upper'];surface=np.vstack([lower,upper])
            rng=np.random.default_rng(1119);directions=rng.normal(size=(3,230))
            select=np.unique((surface@directions).argmax(0))
            self.material=tensor(np.vstack([material_points(ref.binding),surface[select]]))
            self.ice_old=tensor(z['skate_witness_world_mid'][:,:,2])
        with np.load(args.early/'fitted/result.npz') as z:
            self.early_R=tensor(z['local_R_mid']);self.early_pose=tensor((z['native22_world_mid']-z['P_world_mid'][:,None])/self.ratio)
        self.camera=tensor(cam.midpoint['values'])
        priorbody=tensor(c/self.ratio)
        sl=slice(None)
        self.correction-=self.proj(priorbody,sl)
        s0=tensor(orig)[:,:,None]+torch.einsum('nfij,vj->nfvi',tensor(shoot),self.material)
        self.shoe_corr=torch.zeros((*s0.shape[:-1],2),device='cuda')
        # Use actual native source rows at five exposure nodes for the parent
        # material targets, including the direction-dependent camera readout.
        from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
        from calibration.program_tracking.body_tracking.skate_physics import fk_state
        from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
        with np.load(args.parent/'candidate.npz') as old:
            baseline=absolute_reference(ref,curve_from_packet(old))
        surf=self.material.cpu().numpy()
        cache=args.output/'parent_material_uv.npz'
        if cache.exists():
            with np.load(cache) as data:
                if str(data['candidate'])!=digest(args.parent/'candidate.npz') or not np.array_equal(data['material'],surf):raise ValueError('Parent projection cache identity differs')
                actual=data['uv']
        else:
            blocks=[]
            for i in range(0,self.n,64):
                ix=np.arange(i,min(i+64,self.n));ts=ref.times[ix,None]+cam.readout_times(ix)[:,None]*np.linspace(-.5,.5,5)
                fk=fk_state(baseline,baseline.at(ts.ravel()),np.zeros((ts.size,69)))
                xyz=fk['O'][:,:,None]+np.einsum('nfij,vj->nfvi',fk['R'],surf)
                uv=moving_projection(cam,xyz.reshape(len(ix),5,-1,3),ix)['uv'].reshape(len(ix),2,len(surf),2)
                blocks.append(uv)
                if i%1024==0:log(args.output,'native_material_projection',frame=i)
            actual=np.concatenate(blocks);np.savez_compressed(cache,uv=actual,material=surf,candidate=digest(args.parent/'candidate.npz'))
        self.prior_shoe=tensor(actual)
        self.shoe_corr=self.prior_shoe-self.proj(s0.flatten(1,2),sl).reshape_as(self.prior_shoe)
        obs=np.load(args.work/'input/body_track_source.npz');ids=obs['model_joint_ids'];self.ids=index(ids)
        target=obs['uv'].copy();active=obs['use'].copy();weight=np.ones(active.shape)
        # Right knee was previously excluded along with cropped feet. Verify it
        # separately in the native source as it comes into frame during the turn.
        sec=self.pts/12800;chosen=(sec>=24.16)&(sec<=24.48)
        for i in np.flatnonzero(chosen):
            k=np.flatnonzero(ids[i]==14)[0];active[i,k]=True;weight[i,k]=5
        ledger={k:obs[k] for k in obs.files}
        ledger['use']=active
        ledger['joint_refit_admission_json']=np.array(json.dumps(dict(
            source_seconds=[24.16,24.48],joint=14,source='native raw video at original PTS',
            reason='Right knee is visible during this interval; raw pixel observations retained.')))
        np.savez_compressed(args.output/'body_observations.npz',**ledger)
        weight[(ids==7)&(abs(sec[:,None]-63.5)<=.18)]=5
        self.target=tensor(np.nan_to_num(target));self.weight=tensor(active*weight/np.where(obs['inside_tracked_shoe_proxy'],5.,1.))
        # Direct source reread: centres of left boot opening on native images.
        # Screened with the immediately preceding/following frames to keep foot identity.
        anchors=[(53.34,1302,826),(53.42,1298,832),(53.50,1280,831),(53.58,1220,796),(53.66,1102,771)]
        self.review=[(np.searchsorted(self.pts,round(t*12800)),x,y) for t,x,y in anchors]
        document=json.loads(whole_skate_observations(args.programme).read_text(encoding='utf8'))
        self.rows=source_shoe_rows(document,self.pts,actual)
        self.extra=[(np.searchsorted(self.pts,r['pts']),r['joint'],r['uv']) for r in document['body_samples'] if r['role']=='fit']
        sh=np.load(args.work/'input/shoe_occlusion_source.npz')
        # Intact isolated contours can constrain the silhouette on both sides.
        valid=(sh['role']==1)&sh['visible']&~sh['partial_boundary']&(sh['source_pair_overlap'][:,None]<.12)
        for row in document.get('source_outline_review_exclusions',[]):
            if 'pts' in row and 'foot' in row:valid[np.searchsorted(self.pts,row['pts']),row['foot']]=False
        self.shape_dirs=tensor(np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1))
        boundary=sh['region'];u=np.nan_to_num(boundary,nan=0.)@self.shape_dirs.cpu().numpy().T
        self.extent=tensor(u.max(-2));self.complete=tensor(valid)
        self.foot_visible=tensor(sh['visible'])
        wr=np.ones((self.n,2));wr[:,0]=1-.99*np.exp(-((sec-53.50)/.12)**4)
        wr[(sec>=24.12)&(sec<=24.56),1]=.25
        self.keep=tensor(wr)
        self.support_left=tensor((sec>23.96)&(sec<=24.18))
        log(args.output,'source_blocks',body=int(active.sum()),named_material=len(self.rows),complete_contours=int(valid.sum()),source_frames=self.n,controls=len(self.p),device=torch.cuda.get_device_name())

    def proj(self,xyz,sl):
        c=self.camera[sl]
        ray=torch.einsum('nij,nvj->nvi',c[:,4:13].reshape(-1,3,3),xyz/self.cam.scale-c[:,None,13:16])
        x,y=(ray[:,:,:2]/ray[:,:,2:]).unbind(-1);r=x*x+y*y
        k1,k2,p1,p2=c[:,20:,None].unbind(1);rad=1+r*(k1+r*k2)
        u=c[:,16,None]*(x*rad+2*p1*x*y+p2*(r+2*x*x))+c[:,18,None]
        v=c[:,17,None]*(y*rad+p1*(r+2*y*y)+2*p2*x*y)+c[:,19,None]
        return torch.stack([u,v],-1)

    def forward(self,sl):
        ix=self.idx[sl];v=self.delta[ix]*self.mask[ix]
        # Position parameter units are smaller than angular increments.
        P=((self.p[ix]+v[:,:,0]*.15)*self.b[sl,:,None]).sum(1)
        q=qmul(self.q[ix],qexp(v[:,:,1:]))
        dq=qmul(conj(q[:,:-1]),q[:,1:]);ch=self.chart[ix[:,:3]]
        dq=dq*torch.where((dq*ch).sum(-1,keepdim=True)>=0,1.,-1.)
        e=qexp(qlog(dq)*self.beta[sl,:,None,None])
        self.local_quat=qmul(qmul(qmul(q[:,0],e[:,0]),e[:,1]),e[:,2])
        local=matrix(self.local_quat)
        mats=[local[:,:1]];points=[P[:,None]]
        for child,par in self.levels:
            r=mats[-1][:,par];points.append(points[-1][:,par]+(r@self.bone[child,:,None]).squeeze(-1));mats.append(r@local[:,child])
        j=torch.cat(points,1)[:,self.reorder];r=torch.cat(mats,1)[:,self.reorder]
        body=j[:,self.mapping]+(r[:,self.mapping]@self.body_attach[sl,:,:,None]).squeeze(-1)
        uv=self.proj(body,sl)+self.correction[sl]
        rot=r[:,[7,8]];org=j[:,[7,8]]+(rot@self.shoeT[sl,...,None]).squeeze(-1);shoeR=rot@self.shoeR[sl]
        self.boot_origin,self.boot_rotation=org,shoeR
        s=org[:,:,None]+torch.einsum('nfij,vj->nfvi',shoeR,self.material)
        sh=self.proj(s.flatten(1,2),sl).reshape(*s.shape[:-1],2)+self.shoe_corr[sl]
        return uv,sh,s,j,local

    def loss(self,start,end):
        a=max(0,start-1);b=min(self.n,end+1);sl=slice(a,b);k=slice(start-a,end-a)
        uv,shoe,world,j,rot=self.forward(sl)
        x=uv[k];f=shoe[k];s=slice(start,end)
        ii=torch.arange(len(x),device='cuda')[:,None]
        error=x[ii,self.ids[s]]-self.target[s]
        parts=dict(body=(robust(error,9)*self.weight[s,:,None]).sum()/(17*2))
        rv=self.keep[s]
        parts['shoe_preserve']=(robust(f-self.prior_shoe[s],6).mean((-1,-2))*rv).sum()*.35
        outline=(f[:,:,7:]@self.shape_dirs.T).amax(-2)-self.extent[s]
        parts['shoe_outline']=(robust(outline,6).mean(-1)*self.complete[s]).sum()*.9
        ri=self.rows_by_block[start]
        if len(ri):
            t,ft,id,lo,hi,w=ri
            u=f[t,ft,id];dx=u-u.clamp(lo,hi)
            parts['shoe_material']=(robust(dx,8).sum(-1)*w).sum()*.40
        else:parts['shoe_material']=x.sum()*0
        for t,xx,yy in self.review:
            if start<=t<end:
                cuff_error=f[t-start,0,6]-tensor([xx,yy]);parts['review_shoe']=parts.get('review_shoe',0)+robust(cuff_error,12).sum()*3
        for t,ji,xy in self.extra:
            if start<=t<end:parts['body_review']=parts.get('body_review',0)+robust(x[t-start,ji]-tensor(xy),10).sum()*.8
        delta_xyz=(j[k]-j[k,:1])-self.early_pose[s]
        parts['earlier_joint_depth']=robust(delta_xyz*.99*100,.11*100).mean((-1,-2)).sum()*.11
        parts['earlier_orientation']=(1-(rot[k, :, :, :][:,1:]*self.early_R[s]).sum((-1,-2))/3).mean(-1).sum()*9
        clearance=world[k,:,:,2].amin(-1)*self.ratio
        # Sole geometry samples have a finite discretization; the final opening
        # rule always uses the exact continuous runner/toe-pick model.
        parts['left_release']=((clearance[:,0]*130.)**2*self.support_left[s]).sum()*.8
        if hasattr(self,'leg_reference'):
            kr=(self.ids[s]==13)|(self.ids[s]==14)
            parts['reviewed_knees']=(robust(error,6)*self.weight[s,:,None]*kr[:,:,None]).sum()*1.2
            kept=f-self.leg_shoe[s]
            parts['retain_boot_pixels']=robust(kept,3.).mean((-1,-2,-3)).sum()*.75
            parts['retain_leg_pose']=(1-(rot[k,1:]*self.leg_reference[s]).sum((-1,-2))/3).mean(-1).sum()*95
            height_diff=(clearance-self.leg_initial_height[s])*110
            parts['retain_ice_height']=robust(height_diff,3.).mean(-1).sum()*.75
            original_depth=self.leg_actual_height[s]
            parts['ice_nonpenetration']=((clearance-self.leg_initial_height[s]+original_depth).clamp_max(0)*160).square().sum()*.6
            if len(rot)>2:
                oldrot=self.leg_reference[sl]
                motion=rot[:-1,1:].transpose(-1,-2)@rot[1:,1:]
                oldmotion=oldrot[:-1].transpose(-1,-2)@oldrot[1:]
                parts['retain_leg_velocity']=robust((motion-oldmotion)*100,5).mean((-1,-2,-3)).sum()*2
            rows=self.arc_blocks.get(start)
            if rows:
                times,feet,local,target,offset,weight=rows
                base=self.boot_origin[k][times,feet]
                points=base+(self.boot_rotation[k][times,feet]@local[...,None]).squeeze(-1)
                uv_arc=self.proj(points[:,None],times+start)[:,0]+offset
                parts['source_material_arcs']=(robust(uv_arc-target,7).sum(-1)*weight).sum()*5
        if hasattr(self,'baseline_body'):
            parent=self.baseline_body[s][ii,self.ids[s]]-self.target[s]
            away=(error.norm(dim=-1)-parent.norm(dim=-1)-1.5).clamp_min(0)
            parts['retain_body']=((robust(away,6)*self.weight[s]).sum(-1)/17).sum()*.9
            # Keep the assessed visible material and complete-outline gains
            # while releasing unobservable anatomical rotations.
            ext=(self.baseline_shoe[s,:,7:]@self.shape_dirs.T).amax(-2)-self.extent[s]
            parts['retain_outline']=(robust((outline.abs()-ext.abs()-.7).clamp_min(0),5).mean(-1)*self.complete[s]).sum()*1.5
            if len(ri):
                t,ft,id,lo,hi,w=ri
                base=self.baseline_shoe[s][t,ft,id]
                parent=(base-base.clamp(lo,hi)).norm(dim=-1)
                away=(dx.norm(dim=-1)-parent-.75).clamp_min(0)
                parts['retain_material']=(robust(away,6)*w).sum()*1.4
            parts['anatomical_domain']=self.structure_loss(self.local_quat[k])*26
            parts['earlier_orientation']*=6
            parts['earlier_joint_depth']*=2
            parts['rotation_domain_prior']=((clearance+self.contact_bias[s]).clamp_max(0)*220).square().sum()
            parts['reviewed_glide']=(((clearance+self.contact_bias[s])*120).square()*self.glide[s]).sum()
            parts['preserve_skate_height']=robust((clearance+self.contact_bias[s]-self.baseline_h[s])*92.,6).mean(-1).sum()*.5
        if len(rot)>2:
            # body-relative and root rotation changes, retaining genuine steady turns.
            twist=rot[:-1].transpose(-1,-2)@rot[1:]
            second=twist[1:]-twist[:-1]
            parts['rotation_acceleration']=robust(second*100,12).mean((-1,-2,-3)).sum()*3
            rel=j-j[:,:1];acc=rel[2:]-2*rel[1:-1]+rel[:-2]
            parts['joint_acceleration']=robust(acc*200,8).mean((-1,-2)).sum()*2
        parts['update']=(self.delta[self.idx[s,1]].square().mean((-1,-2))).sum()*2
        return parts

    def prepare_blocks(self,block):
        self.rows_by_block={}
        for start in range(0,self.n,block):
            rows=[r for r in self.rows if start<=r[0]<start+block]
            if not rows:self.rows_by_block[start]=[];continue
            t,f,k,lo,hi,w=zip(*rows)
            self.rows_by_block[start]=(index(np.array(t)-start),index(f),index(k),tensor(lo),tensor(hi),tensor(np.clip(np.asarray(w)*5,.25,2.5)))

    def refine_source(self):
        """Relinearize source rows at the joint candidate; allow depth and twist
        to improve against anatomical and earlier-motion evidence."""
        from calibration.program_tracking.body_tracking.programme_joint_limits import authored_bases, JOINTS, NAMES
        self.structure=(index(JOINTS),tensor(quat(authored_bases(self.ref.body.J))))
        self.envelopes=json.loads(Path(self.ref.source_document['programme_structure']['policy']).read_text(encoding='utf8'))['envelopes']
        self.ranges=[]
        for i,n in enumerate(NAMES):
            cfg=self.envelopes[n.split('_')[-1]]
            self.ranges.append(tuple(tensor(np.deg2rad(cfg[key])) for key in ('flex_degrees','flex_grid_degrees','swing_axes_degrees','scale_degrees')))
        old=self.args.output/'body_stage1.npz';stamp=digest(old)
        cache=self.args.output/'stage1_projection.npz'
        if cache.is_file():
            with np.load(cache) as z:
                if str(z['sha'])!=stamp:raise ValueError('Refinement baseline was replaced')
                body,shoe,attachment=z['uv'],z['shoe'],z['attachments']
        else:
            body,shoe,attachment=native_measurements(self.ref,self.cam,self.material.cpu().numpy())
            np.savez_compressed(cache,sha=stamp,uv=body,shoe=shoe,attachments=attachment)
        self.body_attach=tensor(attachment)
        with torch.no_grad():
            for a in range(0,self.n,512):
                sl=slice(a,min(a+512,self.n))
                uv,sh,_,_,_=self.forward(sl)
                self.correction[sl]+=tensor(body[sl])-uv
                self.shoe_corr[sl]+=tensor(shoe[sl])-sh
        self.baseline_body=tensor(body);self.baseline_shoe=tensor(shoe)
        from calibration.program_tracking.body_tracking.skate_physics import fk_state,batch_features
        heights=[];offsets=[]
        for a in range(0,self.n,192):
            sl=slice(a,min(a+192,self.n)); t=self.ref.times[sl]
            state=fk_state(self.ref,self.ref.at(t),np.zeros((len(t),69)))
            feat=batch_features(self.ref.binding,state['R'],state['O'],state)
            minimum=feat['height'].min(-1)
            with torch.no_grad(): approx=self.forward(sl)[2][:,:,:,2].amin(-1).cpu().numpy()*self.ratio
            heights.append(minimum);offsets.append(minimum-approx)
        self.baseline_h=tensor(np.concatenate(heights));self.contact_bias=tensor(np.concatenate(offsets))
        # Prior projected boots outside the opening close-up have no visible source target.
        self.keep[self.pts<=round(24.18*12800)]=0
        self.glide=torch.zeros((self.n,2),device='cuda')
        for row in self.ref.source_document.get('source_support_evidence',[]):
            if row.get('optimization_enabled') and row.get('role')=='fit':
                self.glide[np.searchsorted(self.pts,row['pts']),row['foot']]=1

    def structure_loss(self,local_q):
        # Extract relative quaternion in the existing shaped anatomical basis.
        # Branch is the near-rest principal flexion for these finite soft domains.
        basis_q=self.structure[1]
        q=local_q[:,self.structure[0]]
        q=qmul(qmul(conj(basis_q).expand_as(q),q),basis_q.expand_as(q))
        x,y,z,w=q.unbind(-1)
        length=(y.square()+z.square()+1e-12).sqrt();projection=(w.square()+x.square()+1e-12).sqrt()
        angle=2*torch.atan2(length,projection)
        sw=torch.stack([w*y-x*z,w*z+x*y],-1)*(angle/(length*projection))[...,None]
        flex=torch.atan2(2*w*x,w.square()-x.square())
        vals=[]
        for i,(lims,grid,axes,scale) in enumerate(self.ranges):
            f=flex[:,i];j=torch.searchsorted(grid,f.contiguous()).clamp(1,len(grid)-1)
            mix=((f-grid[j-1])/(grid[j]-grid[j-1])).clamp(0,1)
            radii=axes[j-1]+mix[:,None]*(axes[j]-axes[j-1])
            radius=(sw[:,i]/radii).norm(dim=-1).clamp_min(1e-8)
            violation=sw[:,i]*(1-1/radius).clamp_min(0)[:,None]
            vals.append(((f-f.clamp(lims[0],lims[1]))/scale[0]).square()+(violation/scale[1]).square().sum(-1))
        return torch.stack(vals,-1).mean(-1).sum()

    def leg_sources(self,pkt,block):
        from types import SimpleNamespace
        from calibration.program_tracking.body_tracking.visible_skate_union import VisibleSkateUnion
        from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
        from calibration.program_tracking.body_tracking.skate_physics import fk_state
        from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
        self.mask[:,:1]=0
        fixed=[k for k in range(23) if k not in (2,3,5,6,8,9)]
        self.mask[:,fixed]=0
        with np.load(self.args.output/'joint_source_measurements.npz') as data:
            if str(data['candidate'])!=digest(self.args.output/'delivery_candidate.npz'):raise ValueError('Stale leg refinement targets')
            uv=data['uv'];shoe=data['shoe'];self.leg_actual_height=tensor(data['height'])
        with np.load(self.args.output/'result.npz') as data:
            mapping=LANDMARK
            rr=data['native_body_FK_world_R_mid'][:,mapping]
            offset=(data['COCO_world_mid']-data['native22_world_mid'][:,mapping])/self.ratio
            self.body_attach=tensor(np.einsum('nkji,nkj->nki',rr,offset))
            self.leg_reference=tensor(data['local_R_mid'])
        heights=[]
        with torch.no_grad():
            for a in range(0,self.n,block):
                sl=slice(a,min(a+block,self.n))
                v,sh,world,_,_=self.forward(sl)
                self.correction[sl]+=tensor(uv[sl])-v
                self.shoe_corr[sl]+=tensor(shoe[sl])-sh
                heights.append(world[...,2].amin(-1)*self.ratio)
        self.leg_shoe=tensor(shoe);self.leg_initial_height=torch.cat(heights)
        self.keep[self.pts<=round(24.18*12800)]=0
        state=SequenceState(self.ref,pkt['seconds'])
        problem=SimpleNamespace(reference=self.ref,camera=self.cam,binding=self.ref.binding,body=SimpleNamespace(state=state))
        path=self.args.work/'source_review/current_visible_arcs.json'
        spec=dict(path=str(path),sha256=digest(path),exact_template=True,reviewed_open_arcs=True)
        factor=VisibleSkateUnion(problem,spec,np.arange(self.n));factor.associate(np.zeros(state.size))
        row_list=[]
        for batch,match in zip(factor.batches,factor.matches):
            frames,times,base,target,radius,scale=batch;s=fk_state(self.ref,base,np.zeros((len(times),69)))
            for foot in (0,1):
                xyz=factor.assembly.points(s,foot,match[foot],(s['R'],s['O'])).reshape(len(frames),5,-1,3)
                exact=moving_projection(self.cam,xyz,frames)['uv']
                mid=np.arange(len(frames))*5+2;origin=s['O'][mid,foot]
                loc=np.einsum('nij,nkj->nki',s['R'][mid,foot].swapaxes(-1,-2),xyz[:,2]-origin[:,None])
                use=np.where((match[2]==bool(foot))&(scale>0))
                for k,j in zip(*use):
                    row_list.append((frames[k],foot,loc[k,j],target[k,j],exact[k,j],(scale[k,j]*6/radius[k,j])**2))
        self.arc_blocks={}
        with torch.no_grad():
            for a in range(0,self.n,block):
                rows=[r for r in row_list if a<=r[0]<a+block]
                if not rows:continue
                t,f,l,u,actual,w=zip(*rows);t=index(np.asarray(t)-a);f=index(f);l=tensor(l)
                self.forward(slice(a,min(a+block,self.n)))
                xyz=self.boot_origin[t,f]+(self.boot_rotation[t,f]@l[...,None]).squeeze(-1)
                raw=self.proj(xyz[:,None],t+a)[:,0]
                self.arc_blocks[a]=(t,f,l,tensor(u),tensor(actual)-raw,tensor(w))
        log(self.args.output,'leg_source_coverage',reviewed_material_arc_pixels=len(row_list),
            fixed_upper_body=True,fixed_body_root=True,preserve_confirmed_support=True)

    def curve_value(self):
        v=(self.delta*self.mask).detach().cpu().double().numpy()
        return AbsoluteMotionCurve(self.curve.knots,self.curve.position_controls+v[:,0]*.15,
            self.curve.rotation_controls@_exp(v[:,1:]),self.curve.edge_reference_quaternion)


def run(args):
    base=args.output/'supported_seed.npz'
    if args.step=='refine':
        base=args.output/'body_stage1.npz'
        if not base.exists():shutil.copyfile(args.output/'candidate.npz',base)
    if args.step=='legs':base=args.output/'delivery_candidate.npz'
    ref,cam,_,_,pkt=bind(args.prepared,args.output/'source.json',base)
    objective=Objective(args,ref,cam,pkt)
    if args.step=='refine':
        objective.refine_source()
    batch=400;objective.prepare_blocks(batch)
    if args.step=='legs':objective.leg_sources(pkt,batch)
    torch.set_num_threads(1)
    opt=torch.optim.LBFGS([objective.delta],lr=.85,max_iter=15,history_size=12,
        line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=2e-7)
    calls=0;last=None
    parent_hash=digest(base)
    def closure():
        nonlocal calls,last
        opt.zero_grad();totals={}
        for i in range(0,objective.n,batch):
            parts=objective.loss(i,min(i+batch,objective.n))
            value=sum(parts.values())/objective.n;value.backward()
            for name,v in parts.items():totals[name]=totals.get(name,0.)+v.detach()
        last={k:float(v/objective.n) for k,v in totals.items()};calls+=1
        if calls%8==1:log(args.output,'objective',evaluation=calls,parts=last,gpu_peak_bytes=torch.cuda.max_memory_allocated())
        return torch.tensor(sum(last.values()),device='cuda')
    previous=None;stage=0
    while True:
        stage+=1;opt.step(closure)
        curve=objective.curve_value();curve.check_continuous_edge_update(objective.curve)
        loss=sum(last.values())
        if previous is not None and loss>=previous:break
        destination={'refine':'refined_candidate.npz','fit':'candidate.npz','legs':'leg_candidate.npz'}[args.step]
        save(args.output,destination,pkt,curve,ref,dict(stage='full_programme_joint_'+args.step,round=stage,parent=str(base),parent_sha256=parent_hash,
            objective=last,evaluations=calls,calibration='s14_boards_1118',geometry='lane4_side_1118'))
        if previous is not None and previous-loss<max(.002,loss*.00035):break
        previous=loss
