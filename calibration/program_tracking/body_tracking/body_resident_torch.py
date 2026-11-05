"""Prepared FP64 native55/selected-skin foot block on one CUDA owner.

This is explicitly a hybrid backend: body image analytic derivatives, native
camera projection, CSR assembly and SciPy TRF remain CPU work. Every live
nonsmooth shoe/collar choice is evaluated on device for both perturbations.
Only a changed delta is uploaded, and residual plus local derivatives return
in one packed download per linearization. No per-column device readback.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from .torch_foot_compatibility import TorchFootCompatibility


class ResidentFootLinearization:
    def __init__(self,reference,ref,binding,*,device='cuda',direction_batch_size=3):
        from .compact_body_derivatives import resolve_reference
        from .selected_surface_linearization import supports_surface
        owner=resolve_reference(reference)
        if owner is None or not supports_surface(binding.surface):
            raise TypeError('Resident foot path requires audited original body, world and FullSurface')
        self.torch=torch
        if direction_batch_size not in (1,2,3):raise ValueError('Direction batch must fit one joint (1..3)')
        self.direction_batch_size=direction_batch_size
        self.device=torch.device(device)
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.dtype=torch.float64
        self.n=len(ref['P']);self.scale=float(owner.body.scale)
        self.fixed_x=np.array(binding.x,copy=True)
        self._prepare_bytes=0
        started=time.perf_counter()
        def upload(a,dtype=torch.float64):
            array=np.array(a,copy=True)
            result=torch.as_tensor(array,dtype=dtype,device=self.device)
            self._prepare_bytes+=result.numel()*result.element_size()
            return result
        body=owner.body;surface=binding.surface
        if surface.body.scale!=body.scale or not all(np.array_equal(getattr(surface.body,k),getattr(body,k)) for k in ('J','parents','default')):
            raise ValueError('Resident surface and FK model differ')
        self.parents=np.asarray(body.parents,dtype=int).copy()
        self.J=upload(body.J);self.eye=torch.eye(3,dtype=self.dtype,device=self.device)
        self.refP=upload(ref['P']);self.refQ=upload(ref['Q']);self.refLocal=upload(ref['local_R'])
        self.tail=upload(np.broadcast_to(body.default,(self.n,33,3,3)))
        if 'hand_local_R' in ref:self.tail[:,3:]=upload(ref['hand_local_R']).reshape(self.n,30,3,3)
        ids=np.asarray(binding.sample_ids,dtype=int)
        self.vertices=len(ids)
        self.shape=upload(np.asarray(surface.shaped)[ids])
        self.posedirs=upload(np.asarray(surface.posedirs).reshape(486,-1,3)[:,ids].reshape(486,-1))
        self.weights=upload(np.asarray(surface.weights)[ids].T)
        descendants=np.eye(55,dtype=bool)
        for j in range(1,55):descendants[:,j]|=descendants[:,self.parents[j]]
        self.groups={}
        for j in range(1,22):
            allids=np.flatnonzero(descendants[j]);skin=allids[np.any(np.asarray(surface.weights)[ids][:,allids]!=0,axis=0)]
            self.groups[j]=(upload(np.flatnonzero(descendants[j,:22]),torch.int64),
                upload(skin,torch.int64),upload(np.asarray(surface.weights)[ids][:,skin].T),
                upload(np.asarray(surface.weights,dtype=np.float64)[ids][:,skin].sum(axis=1)))
        self.levels=[]
        depth=np.zeros(55,dtype=int)
        for j in range(1,55):depth[j]=depth[self.parents[j]]+1
        for level in range(1,depth.max()+1):
            js=np.flatnonzero(depth==level);ps=self.parents[js]
            self.levels.append((upload(js,torch.int64),upload(ps,torch.int64),upload(np.asarray(body.J)[js]-np.asarray(body.J)[ps])))
        self.consumer=TorchFootCompatibility(binding,self.device)
        self._prepare_bytes+=self.consumer.prepare_uploaded_bytes
        properties=torch.cuda.get_device_properties(self.device)
        self.device_info=dict(name=properties.name,compute_capability=list(torch.cuda.get_device_capability(self.device)),total_memory=properties.total_memory,torch=torch.__version__,cuda=torch.version.cuda)
        torch.cuda.synchronize(self.device)
        self.prepare_seconds=time.perf_counter()-started
        self._last_delta=None;self._pose=None
        self.reset_measurement()

    def matches(self,binding):
        return np.array_equal(self.fixed_x,binding.x)

    def record(self):
        return dict(kind='cuda_resident_native55_selected_skin_foot',device=str(self.device),device_info=self.device_info,
            scope='native55 FK, full486 pose blend, selected union skin, exact +/- replay and live cavity/collar consumers; CPU analytic body images/native camera/CSR/TRF unchanged',
            selected_vertices=self.vertices,samples=self.n,prepare_seconds=self.prepare_seconds,
            direction_batch_size=self.direction_batch_size,
            prepare_uploaded_bytes=self._prepare_bytes,precision='float64',
            shared_parameters='fixed; changing shared geometry requires a new prepared owner',
            transfer_boundary='one changed N x 69 increment upload; one packed residual/local-Jacobian download per call; no per-column download')

    def reset_measurement(self):
        self._events=[]
        self._counts=dict(calls=0,linearize_calls=0,trial_calls=0,h2d_bytes=0,d2h_bytes=0,h2d_calls=0,d2h_calls=0,
            native55_fk_prepares=0,selected_surface_prepares=0,exact_fk_replay_pairs=0,selected_skin_replay_pairs=0,
            consumer_calls=0,h2d_host_seconds=0.,d2h_host_blocking_seconds=0.,compute_host_enqueue_seconds=0.)

    def clear_result_cache(self):
        self._last_delta=None;self._pose=None

    def measurement(self):
        # Caller synchronizes the owner before reading event intervals.
        result=dict(self._counts)
        for kind in ('h2d','compute','d2h'):
            result[kind+'_seconds_cuda_events']=sum(a.elapsed_time(b)*.001 for k,a,b in self._events if k==kind)
        result['timing_scope']='CUDA stream event intervals (include host submission gaps, not summed pure kernel time); complete wall separately includes host waits. Static preparation excluded.'
        return result

    def _event(self,kind):
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record();self._events.append((kind,first,last));return last

    def _rot(self,v):
        theta=torch.linalg.vector_norm(v,dim=-1)
        a=torch.sinc(theta/torch.pi)
        b=.5*torch.sinc(theta/(2*torch.pi)).square()
        z=torch.zeros_like(v[...,0]);x,y,w=v.unbind(-1)
        skew=torch.stack((z,-w,y,w,z,-x,-y,x,z),-1).reshape(*v.shape[:-1],3,3)
        return self.eye+a[...,None,None]*skew+b[...,None,None]*(skew@skew)

    def _local_roundtrip(self,native,P,Q):
        world=P[:,None]+torch.einsum('nij,nkj->nki',Q,native)
        return torch.einsum('nji,nkj->nki',Q,world-P[:,None])

    def _prepare_pose(self,delta):
        n=self.n
        local=self.refLocal@self._rot(delta[:,6:].reshape(n,21,3))
        all_local=torch.cat((self.eye.expand(n,1,3,3),local,self.tail),dim=1)
        rotation=all_local.clone();position=self.J.expand(n,55,3).clone()
        for js,ps,offset in self.levels:
            rotation[:,js]=rotation[:,ps]@all_local[:,js]
            position[:,js]=position[:,ps]+torch.einsum('nqab,qb->nqa',rotation[:,ps],offset)
        native=(position[:,:22]-self.J[0])*self.scale
        frames=rotation[:,:22].clone()
        P=self.refP+delta[:,:3];Q=self.refQ@self._rot(delta[:,3:6])
        # The surface reconstructs native22 local positions through world->local.
        base_native=self._local_roundtrip(native,P,Q)
        surface_position=position.clone();surface_rotation=rotation.clone()
        surface_position[:,:22]=base_native/self.scale+self.J[0]
        # Recompute the fixed tail from reconstructed native22, exactly as FullSurface.
        for j in range(22,55):
            parent=int(self.parents[j])
            surface_rotation[:,j]=surface_rotation[:,parent]@self.tail[:,j-22]
            surface_position[:,j]=surface_position[:,parent]+torch.einsum('nij,j->ni',surface_rotation[:,parent],self.J[j]-self.J[parent])
        recovered_local=torch.stack([frames[:,self.parents[j]].transpose(-1,-2)@frames[:,j] for j in range(1,22)],dim=1)
        feature=(torch.cat((recovered_local,self.tail),dim=1)-self.eye).reshape(n,486)
        posed=self.shape+(feature@self.posedirs).reshape(n,self.vertices,3)
        translation=surface_position-torch.einsum('njab,jb->nja',surface_rotation,self.J)
        blendR=(surface_rotation.reshape(n,55,9).transpose(1,2)@self.weights).transpose(1,2).reshape(n,self.vertices,3,3)
        blendT=(translation.transpose(1,2)@self.weights).transpose(1,2)
        samples=(torch.einsum('nvij,nvj->nvi',blendR,posed)+blendT-self.J[0])*self.scale
        return dict(local=local,frames=frames,position=position,native=native,P=P,Q=Q,
            recovered_local=recovered_local,surface_position=surface_position,rotation=surface_rotation,
            translation=translation,posed=posed,blendR=blendR,samples=samples,base_native=base_native,delta=delta)

    def _subtree(self,pose,joint):
        indices,skin,weight,total=self.groups[joint];n=self.n
        rotation=pose['rotation'][:,skin]
        subR=(rotation.reshape(n,len(skin),9).transpose(1,2)@weight).transpose(1,2).reshape(n,self.vertices,3,3)
        subT=(pose['translation'][:,skin].transpose(1,2)@weight).transpose(1,2)
        relative=(torch.einsum('nvij,nvj->nvi',subR,pose['posed'])+subT
            -total[None,:,None]*pose['surface_position'][:,joint,None])
        return indices,subR,relative

    def _pair(self,pose,column,step,subtree):
        joint=(column-6)//3+1;axis=(column-6)%3;parent=int(self.parents[joint])
        indices,subR,relative=subtree
        outputs=[];natives=[];frames_all=[]
        for sign in (1.,-1.):
            value=pose['delta'][:,6:].reshape(self.n,21,3)[:,joint-1].clone()
            value[:,axis]+=sign*step
            changed=self.refLocal[:,joint-1]@self._rot(value)
            current=pose['frames'][:,parent]@changed
            turn=current@pose['frames'][:,joint].transpose(-1,-2)
            pos=pose['position'][:,:22].clone();frames=pose['frames'].clone()
            pivot=pose['position'][:,joint,None]
            pos[:,indices]=pivot+torch.einsum('nij,nkj->nki',turn,pose['position'][:,indices]-pivot)
            frames[:,indices]=turn[:,None]@pose['frames'][:,indices]
            pos[:,joint]=pose['position'][:,joint];frames[:,joint]=current
            native=self._local_roundtrip((pos-self.J[0])*self.scale,pose['P'],pose['Q'])
            local=frames[:,parent].transpose(-1,-2)@frames[:,joint]
            dposed=((local-pose['recovered_local'][:,joint-1]).reshape(self.n,9)@self.posedirs[(joint-1)*9:joint*9]).reshape(self.n,self.vertices,3)
            pose_effect=torch.einsum('nvij,nvj->nvi',pose['blendR'],dposed)
            shifted=relative+torch.einsum('nvij,nvj->nvi',subR,dposed)
            surface_turn=frames[:,joint]@pose['frames'][:,joint].transpose(-1,-2)-self.eye
            pose_effect+=torch.einsum('nij,nvj->nvi',surface_turn,shifted)
            outputs.append(pose['samples']+pose_effect*self.scale)
            natives.append(native);frames_all.append(frames)
        both=self.consumer.evaluate(torch.cat(natives),torch.cat(frames_all),torch.cat(outputs))
        return (both[:self.n]-both[self.n:])/(2*step)

    def _pairs(self,pose,columns,step,subtree):
        """Replay all axes of one joint in one device consumer invocation.

        Direction/sign/frame axes stay explicit until the consumer boundary.
        Each +/- state retains its own live max, wall and collar choices.
        No perturbation, skin dependency or nonsmooth branch is discarded.
        """
        k=len(columns);n=self.n;joint=(columns[0]-6)//3+1
        if any((column-6)//3+1!=joint for column in columns):
            raise ValueError('A replay batch must share its joint subtree')
        parent=int(self.parents[joint]);indices,subR,relative=subtree
        axes=torch.as_tensor([(column-6)%3 for column in columns],device=self.device)
        directions=self.eye[axes]
        signs=torch.tensor([1.,-1.],dtype=self.dtype,device=self.device)
        value=(pose['delta'][:,6:].reshape(n,21,3)[:,joint-1][None,None]
            +step*signs[None,:,None,None]*directions[:,None,None])
        changed=self.refLocal[:,joint-1][None,None]@self._rot(value)
        current=pose['frames'][:,parent][None,None]@changed
        turn=current@pose['frames'][:,joint].transpose(-1,-2)[None,None]
        pos=pose['position'][:,:22][None,None].expand(k,2,n,22,3).clone()
        frames=pose['frames'][None,None].expand(k,2,n,22,3,3).clone()
        pivot=pose['position'][:,joint,None]
        pos[:,:,:,indices]=pivot[None,None]+torch.einsum('bsnij,nkj->bsnki',turn,pose['position'][:,indices]-pivot)
        frames[:,:,:,indices]=turn[:,:,:,None]@pose['frames'][:,indices][None,None]
        pos[:,:,:,joint]=pose['position'][:,joint][None,None]
        frames[:,:,:,joint]=current
        world=pose['P'][None,None,:,None]+torch.einsum('nij,bsnkj->bsnki',pose['Q'],(pos-self.J[0])*self.scale)
        native=torch.einsum('nji,bsnkj->bsnki',pose['Q'],world-pose['P'][None,None,:,None])
        local=frames[:,:,:,parent].transpose(-1,-2)@frames[:,:,:,joint]
        dposed=((local-pose['recovered_local'][:,joint-1][None,None]).reshape(k,2,n,9)
            @self.posedirs[(joint-1)*9:joint*9]).reshape(k,2,n,self.vertices,3)
        effect=torch.einsum('nvij,bsnvj->bsnvi',pose['blendR'],dposed)
        shifted=relative[None,None]+torch.einsum('nvij,bsnvj->bsnvi',subR,dposed)
        surface_turn=frames[:,:,:,joint]@pose['frames'][:,joint].transpose(-1,-2)[None,None]-self.eye
        effect+=torch.einsum('bsnij,bsnvj->bsnvi',surface_turn,shifted)
        samples=pose['samples'][None,None]+effect*self.scale
        residual=self.consumer.evaluate(native.reshape(k*2*n,22,3),frames.reshape(k*2*n,22,3,3),samples.reshape(k*2*n,self.vertices,3))
        residual=residual.reshape(k,2,n,self.consumer.compatibility_width)
        return (residual[:,0]-residual[:,1])/(2*step)

    @torch.no_grad()
    def evaluate(self,delta,columns=None,*,step=1e-5):
        values=np.asarray(delta,dtype=np.float64)
        if values.shape!=(self.n,69) or not np.isfinite(values).all():raise ValueError('Changed resident state layout or nonfinite delta')
        columns=None if columns is None else list(map(int,columns))
        if columns is not None and any(c<6 or c>=69 for c in columns):raise ValueError('Resident derivative column outside body21')
        if not np.isfinite(step) or step<=0:raise ValueError('Invalid difference step')
        self._counts['calls']+=1
        self._counts['trial_calls' if columns is None else 'linearize_calls']+=1
        changed=self._last_delta is None or not np.array_equal(values,self._last_delta)
        if changed:
            host=time.perf_counter()
            end=self._event('h2d')
            device_delta=torch.as_tensor(np.array(values,copy=True),dtype=self.dtype,device=self.device)
            end.record();self._counts['h2d_host_seconds']+=time.perf_counter()-host
            self._counts['h2d_calls']+=1;self._counts['h2d_bytes']+=device_delta.numel()*8
        host=time.perf_counter()
        compute_end=self._event('compute')
        if changed:
            self._pose=self._prepare_pose(device_delta);self._last_delta=values.copy()
            self._counts['native55_fk_prepares']+=1;self._counts['selected_surface_prepares']+=1
        pose=self._pose
        residual=self.consumer.evaluate(pose['base_native'],pose['frames'],pose['samples'])
        self._counts['consumer_calls']+=1
        derivatives=[];last_joint=None;subtree=None
        offset=0;columns=[] if columns is None else columns
        while offset<len(columns):
            column=columns[offset];joint=(column-6)//3+1
            if last_joint!=joint:subtree=self._subtree(pose,joint);last_joint=joint
            stop=offset+1
            while stop<len(columns) and stop-offset<self.direction_batch_size and (columns[stop]-6)//3+1==joint:stop+=1
            batch=columns[offset:stop]
            if len(batch)==1:derivatives.append(self._pair(pose,column,step,subtree))
            else:derivatives.extend(self._pairs(pose,batch,step,subtree).unbind(0))
            self._counts['exact_fk_replay_pairs']+=len(batch);self._counts['selected_skin_replay_pairs']+=len(batch);self._counts['consumer_calls']+=1
            offset=stop
        packed=torch.cat([residual.reshape(-1)]+[value.reshape(-1) for value in derivatives])
        compute_end.record()
        self._counts['compute_host_enqueue_seconds']+=time.perf_counter()-host
        host=time.perf_counter()
        end=self._event('d2h')
        array=packed.cpu().numpy();end.record()
        self._counts['d2h_host_blocking_seconds']+=time.perf_counter()-host
        self._counts['d2h_calls']+=1;self._counts['d2h_bytes']+=array.nbytes
        width=residual.numel()
        baseline=array[:width].reshape(self.n,self.consumer.compatibility_width)
        result=[array[(i+1)*width:(i+2)*width].reshape(baseline.shape) for i in range(len(derivatives))]
        return baseline,result
