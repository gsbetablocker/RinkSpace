"""Resident FP64 consumer of the existing fixed shared SkateBinding.

``evaluate(native, frames, samples)`` returns exactly the row layout of
``binding.compatibility_rows(...)/sqrt(2)``. Inputs use body-local solver units;
samples must follow ``binding.sample_ids`` including its collar union order.
FK, all pose blend shapes, and dynamic hand skinning belong to the caller.
Only construction reads NumPy or uploads constants. Evaluations never read a
device value on the host and retain the trial's live section topology choices.
The CUDA section is a nonsmooth finite-difference consumer, not an autograd op.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .body_foot_cuda_bridge import CollarKernel, REASONS


class TorchFootCompatibility:
    def __init__(self,binding,device='cuda',*,library=None):
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('TorchFootCompatibility requires CUDA')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.dtype=torch.float64
        self.prepare_uploaded_bytes=0
        def upload(a,dtype=torch.float64):
            value=torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=self.device)
            self.prepare_uploaded_bytes+=value.numel()*value.element_size()
            return value
        self.fixed_x=np.asarray(binding.x,dtype=np.float64).copy()
        self.sample_ids=np.asarray(binding.sample_ids,dtype=np.int64).copy()
        self.ratio=float(binding.ratio)
        self.compatibility_width=int(binding.compatibility_width)
        self.collar_enabled=bool(getattr(binding,'collar_constraints_enabled',False))
        self.R,self.t=map(upload,binding.parameters()[:2])
        p=binding.packet()
        self.length=float(p['parameters'].sole_length_m)
        self.xs,self.floor,self.targets,self.inner=map(upload,(p['xs'],p['floor'],p['targets'],p['inner']))
        self.section_count=len(p['xs'])
        station=int(np.argmin(abs(p['xs']/self.length-.28)))
        self.collar_height=float(p['floor'][station]+p['rise'][station])
        self.collar_width=float(p['width'][station])
        self.ankle=-(self.t[:,None,:]@self.R).squeeze(1)
        # Resident replay may flatten direction/sign/sample axes. Keep the
        # sample block intact and repeat it in that same order on device.
        self.skate_relative_R=None
        self.maximum=upload(np.array([.020,.015,.012])/self.ratio)
        self.foot_indices=[upload(ids,torch.int64) for ids in binding.foot_indices]
        self.patch_indices=[[upload(ids,torch.int64) for ids in patches] for patches in binding.patch_indices]
        # Deformation basis is fixed by the template, while alpha stays live.
        vertices=np.asarray(p['inner'])
        high=np.clip((vertices[:,2]-np.interp(vertices[:,0],p['xs'],p['floor']))/(.08/self.ratio),0,1)
        cuff=np.clip((.65*self.length-vertices[:,0])/(.20*self.length),0,1)*high
        weight=np.clip(vertices[:,0]/(.46*self.length),0,1)
        self.front,self.rear,self.side=map(upload,(weight*cuff,(1-weight)*cuff,cuff*np.sign(vertices[:,1])))
        self.last_collar_metrics=None
        self.reason_names=REASONS
        if self.collar_enabled:
            from .skate_collar_factors import prepare
            spec=prepare(binding)
            self.collar_positions=upload(binding.collar_sample_positions,torch.int64)
            self.topologies=[{name:upload(getattr(top,name),torch.int64) for name in
                ('edges','triangle_edges','edge_faces','edge_face_counts')} for top in spec.body]
            self.ends=upload(spec.inner_edge_vertices,torch.int64)
            self.fraction=upload(spec.inner_edge_fraction)
            self.section_height=float(spec.height_solver)
            self.tolerance=float(spec.tolerance_solver)
            self.domain=max(float(np.linalg.norm(np.ptp(spec.inner_vertices,axis=0))),np.finfo(float).eps)
            self.static_unknown=spec.inner_unknown_reason
            if self.static_unknown:self.reason_names=(*REASONS,self.static_unknown)
            self.kernel=CollarKernel(library) if not self.static_unknown else None

    def _interp(self,query,grid,values):
        # np.interp endpoint behavior, with a fixed ascending template grid.
        index=(torch.searchsorted(grid,query.contiguous(),right=True)-1).clamp(0,len(grid)-2)
        fraction=((query-grid[index])/(grid[index+1]-grid[index])).clamp(0,1)
        return values[index]*(1-fraction)+values[index+1]*fraction

    def _relative_parameters(self,relative_R,n):
        if relative_R is None:relative_R=getattr(self,'skate_relative_R',None)
        if relative_R is None:return self.R,self.t
        if (not isinstance(relative_R,torch.Tensor) or relative_R.device!=self.device or relative_R.dtype!=self.dtype
                or relative_R.ndim!=4 or tuple(relative_R.shape[1:])!=(2,3,3)):
            raise ValueError(f'skate_relative_R must be FP64 on {self.device} with shape (N,2,3,3)')
        count=len(relative_R)
        if count==0 or n%count:raise ValueError('Consumer batch must be an integer multiple of the shoe sample count')
        if count!=n:relative_R=relative_R.repeat(n//count,1,1,1)
        rotation=self.R[None]@relative_R
        origin=-torch.einsum('nfij,fj->nfi',rotation,self.ankle)
        return rotation,origin

    def _alpha(self,native,frames,skate_relative_R=None,*,parameters=None):
        R,_=self._relative_parameters(skate_relative_R,len(native)) if parameters is None else parameters
        values=[]
        for foot,(knee,ankle) in enumerate(((4,7),(5,8))):
            rotation=frames[:,ankle]@(R[foot] if R.ndim==3 else R[:,foot])
            direction=torch.einsum('nji,nj->ni',rotation,native[:,knee]-native[:,ankle])
            fraction=((self.collar_height-self.ankle[foot,2])/direction[:,2].clamp_min(1e-6)).clamp(0,1)
            centre=self.ankle[foot]+fraction[:,None]*direction
            radius=.027/self.ratio
            amount=torch.stack((centre[:,0]+radius-.46*self.length,-centre[:,0]+radius,
                centre[:,1].abs()+radius-self.collar_width),dim=1).clamp_min(0)
            values.append(torch.minimum(amount,self.maximum))
        return torch.stack(values,dim=1)

    def _deform(self,alpha):
        x=self.inner[None,None,:,0]+alpha[:,:,0,None]*self.front-alpha[:,:,1,None]*self.rear
        y=self.inner[None,None,:,1]+alpha[:,:,2,None]*self.side
        z=self.inner[None,None,:,2].expand_as(x)
        return torch.stack((x,y,z),dim=-1)

    def _collar(self,locals_,inner):
        n=len(inner)
        if self.static_unknown:
            reason=torch.full((n,2),len(REASONS),dtype=torch.int32,device=self.device)
            residual=torch.full((n,2),self.domain,dtype=self.dtype,device=self.device)
            self.last_collar_metrics=dict(residual_breach_solver=residual,breach_solver=torch.zeros_like(residual),
                minimum_sample_gap_solver=torch.full_like(residual,float('nan')),sample_count=torch.zeros((n,2),dtype=torch.int64,device=self.device),
                unknown_reason_code=reason,unknown=reason!=0)
            return residual
        results=[]
        stream=torch.cuda.current_stream(self.device)
        for foot in range(2):
            vertices=inner[:,foot]
            polygon=(vertices[:,self.ends[:,0],:2]*(1-self.fraction[None,:,None])+
                     vertices[:,self.ends[:,1],:2]*self.fraction[None,:,None]).contiguous()
            z_changed=torch.any((vertices[:,:,2]-self.inner[None,:,2]).abs()>self.tolerance,dim=1).contiguous()
            body=locals_[foot][:,self.collar_positions].contiguous()
            results.append(self.kernel(body,self.topologies[foot],polygon,z_changed,
                self.section_height,self.tolerance,self.domain,stream))
        self.last_collar_metrics={key:torch.stack([result[key] for result in results],dim=1) for key in results[0]}
        return self.last_collar_metrics['residual_breach_solver']

    @torch.no_grad()
    def evaluate(self,native,frames,samples,skate_relative_R=None):
        n=len(native)
        for name,tensor,shape in (('native',native,(n,22,3)),('frames',frames,(n,22,3,3)),
                                  ('samples',samples,(n,len(self.sample_ids),3))):
            if tensor.device!=self.device or tensor.dtype!=self.dtype or tuple(tensor.shape)!=shape:
                raise ValueError(f'{name} must be FP64 on {self.device} with shape {shape}')
        R,t=self._relative_parameters(skate_relative_R,n)
        alpha=self._alpha(native,frames,parameters=(R,t))
        deformed=self._deform(alpha)
        sections=deformed[:,:,:self.section_count*6].reshape(n,2,self.section_count,6,3)
        anchors=[];outside=[];support=[];locals_=[]
        for foot,ankle in enumerate((7,8)):
            if R.ndim==3:
                rotation=frames[:,ankle]@R[foot]
                origin=native[:,ankle]+torch.einsum('nij,j->ni',frames[:,ankle],t[foot])
            else:
                rotation=frames[:,ankle]@R[:,foot]
                origin=native[:,ankle]+torch.einsum('nij,nj->ni',frames[:,ankle],t[:,foot])
            local=torch.einsum('nji,nkj->nki',rotation,samples-origin[:,None])
            locals_.append(local)
            skin=local[:,self.foot_indices[foot]]
            anchor=torch.stack([local[:,ids].mean(1) for ids in self.patch_indices[foot]],dim=1)
            anchors.append(anchor)
            walls=[];query=skin[:,:,0]
            rows=torch.arange(n,device=self.device)[:,None]
            for side in range(6):
                grid=sections[:,foot,:,side,0]
                index=((query[:,:,None]>=grid[:,None,:]).sum(dim=-1)-1).clamp(0,self.section_count-2)
                left,right=grid[rows,index],grid[rows,index+1]
                fraction=((query-left)/(right-left)).clamp(0,1)
                yz=sections[:,foot,:,side,1:]
                walls.append(yz[rows,index]*(1-fraction[:,:,None])+yz[rows,index+1]*fraction[:,:,None])
            walls=torch.stack(walls,dim=2)
            edge=torch.roll(walls,-1,dims=2)-walls
            inward=torch.stack((-edge[:,:,:,1],edge[:,:,:,0]),dim=-1)/torch.linalg.vector_norm(edge,dim=-1)[:,:,:,None]
            margins=(inward*(skin[:,:,None,1:]-walls)).sum(dim=-1)
            lining=.003/self.ratio
            violations=torch.cat(((-margins).clamp_min(0),(lining-query).clamp_min(0)[:,:,None],
                (query-(self.length-lining)).clamp_min(0)[:,:,None]),dim=2)
            outside.append(violations.amax(dim=1))
            support.append(anchor[:,:,2]-self._interp(anchor[:,:,0],self.xs,self.floor+lining))
        rows=torch.cat(((torch.stack(anchors,dim=1)-self.targets[None,None]).reshape(n,12)/(.012/self.ratio),
            torch.stack(outside,dim=1).reshape(n,16)/(.003/self.ratio),
            torch.stack(support,dim=1).reshape(n,4)/(.005/self.ratio)),dim=1)
        if self.collar_enabled:rows=torch.cat((rows,self._collar(locals_,deformed)/(.003/self.ratio)),dim=1)
        return rows/math.sqrt(2.)
