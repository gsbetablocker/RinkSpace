import numpy as np
import torch

from experiments.skate_video_v1.motion_review.scoped_fit import ti, tn


class Cavity:
    def __init__(self, reference, fit, parent_world):
        self.reference = reference
        self.fit = fit
        self.binding = binding = reference.binding
        self.mid = slice(2, None, 5)
        surface = binding.surface
        ids = binding.sample_ids
        vshape = tn(surface.shaped[ids])
        W = tn(surface.weights[ids])
        posedirs = tn(surface.posedirs.reshape(486, -1, 3)[:, ids].reshape(486, -1))
        skin_ids=[ti(ix) for ix in binding.foot_indices]
        template=binding.packet()
        Rb,tb,_=binding.parameters()
        ankle=tn(-np.einsum('fi,fij->fj',tb,Rb))
        length=template['parameters'].sole_length_m
        xs=tn(template['xs'])
        floor=tn(template['floor'])
        sx=len(xs)
        inner=tn(template['inner'][:sx*6].reshape(sx,6,3))
        xp=inner[:,:,0]
        height=tn(template['floor']+template['rise'])
        station=int(np.argmin(abs(template['xs']/length-.28)))
        radius=.027/binding.ratio
        cuff=((.65*length-xp)/(.20*length)).clamp(0,1)*((inner[:,:,2]-floor[:,None])/(.08/binding.ratio)).clamp(0,1)
        front=(xp/(.46*length)).clamp(0,1)*cuff
        rear=(1-(xp/(.46*length)).clamp(0,1))*cuff
        collar_end=int(np.max(template['collar'])//6)
        ff=torch.arange(len(parent_world['frames']),device=xs.device)[:,None,None]
        wall_ids=torch.arange(6,device=xs.device)[None,None]
        self.vshape = vshape
        self.W = W
        self.posedirs = posedirs
        self.skin_ids = skin_ids
        self.ankle = ankle
        self.length = length
        self.xs = xs
        self.floor = floor
        self.sx = sx
        self.inner = inner
        self.xp = xp
        self.height = height
        self.station = station
        self.radius = radius
        self.cuff = cuff
        self.front = front
        self.rear = rear
        self.collar_end = collar_end
        self.ff = ff
        self.wall_ids = wall_ids
        self.width = tn(template['width'][station])
        baseline = self.exact(parent_world)
        self.norm = tn(np.linalg.norm(baseline, axis=0))
        self.cap = tn(np.maximum(baseline.max(0) - .0002, .0045))

    def exact(self, pose):
        data = self.binding.foot_metrics(
            pose['native'], pose['frames'], hand_local_R=pose.get('hand_local_R'),
            skate_relative_R=pose['skate_relative_R'])
        return np.stack([v.max(axis=(1, 2)) for v in data['violations']], axis=1) * self.binding.ratio

    def values(self, state):
        binding = self.binding
        vshape = self.vshape
        W = self.W
        posedirs = self.posedirs
        skin_ids = self.skin_ids
        ankle = self.ankle
        length = self.length
        sx = self.sx
        inner = self.inner
        xp = self.xp
        height = self.height
        station = self.station
        radius = self.radius
        cuff = self.cuff
        front = self.front
        rear = self.rear
        collar_end = self.collar_end
        ff = self.ff
        wall_ids = self.wall_ids
        _,_,_,A,J,_,_,_,L=state
        A,J,L=A[self.mid],J[self.mid],L[self.mid]
        t=J-(A@self.fit.J[None,:,:,None]).squeeze(-1)
        posed=vshape+((L-self.fit.eye).reshape(len(A),486)@posedirs).reshape(len(A),-1,3)
        blends=(W@A.reshape(len(A),55,9)).reshape(len(A),-1,3,3)
        vertices=(blends@posed[...,None]).squeeze(-1)+W@t
        vertices=(vertices-self.fit.J[0])*self.reference.body.scale
        j=(J-self.fit.J[0])*self.reference.body.scale
        result=[]
        for f,(k,a) in enumerate(((4,7),(5,8))):
            frame=A[:,a] @ self.fit.relR[self.mid,f]
            origin=j[:,a]+(A[:,a]@self.fit.relT[self.mid,f,:,None]).squeeze(-1)
            foot=(vertices[:,skin_ids[f]]-origin[:,None])@frame
            direction=((j[:,k]-j[:,a])[:,None]@frame).squeeze(1)
            lam=((height[station]-ankle[f,2])/direction[:,2].clamp_min(1e-6)).clamp(0,1)
            c=ankle[f]+lam[:,None]*direction
            alpha=torch.stack((c[:,0]+radius-.46*length,-c[:,0]+radius,
                                c[:,1].abs()+radius-self.width),dim=1)
            alpha=torch.minimum(alpha.clamp_min(0),tn([.020,.015,.012])/binding.ratio)
            grid=xp+alpha[:,0,None,None]*front-alpha[:,1,None,None]*rear
            iy=inner[:,:,1]+alpha[:,2,None,None]*cuff*torch.sign(inner[:,:,1])
            data=torch.stack((iy,inner[:,:,2].expand_as(iy)),dim=-1)
            query=foot[...,0]
            idx=((query[:,:,None,None]>=grid[:,None]).sum(2)-1).clamp(0,sx-2)
            aa,bb=grid[ff,idx,wall_ids],grid[ff,idx+1,wall_ids]
            s=((query[:,:,None]-aa)/(bb-aa)).clamp(0,1)
            low=data[ff,idx,wall_ids]
            walls=low+s[:,:,:,None]*(data[ff,idx+1,wall_ids]-low)
            edge=torch.roll(walls,shifts=-1,dims=2)-walls
            inward=torch.stack((-edge[...,1],edge[...,0]),dim=-1)
            inward=inward/torch.linalg.vector_norm(inward,dim=-1,keepdim=True)
            margin=(inward*(foot[:,:,None,1:]-walls)).sum(-1)
            if binding.respect_open_collar:
                open_roof=idx[:,:,3]<collar_end
                above=foot[:,:,2]>=torch.maximum(walls[:,:,3,1],walls[:,:,4,1])
                through=(foot[:,:,1]>=walls[:,:,4,0]) & (foot[:,:,1]<=walls[:,:,3,0])
                reject=(open_roof&above&through)[...,None] & (wall_ids!=0)
                reject |= open_roof[...,None] & (wall_ids==3)
                margin=torch.where(reject,torch.inf,margin)
            depths=torch.cat((torch.relu(-margin),torch.relu(.003/binding.ratio-query)[...,None],
                              torch.relu(query-(length-.003/binding.ratio))[...,None]),dim=2)
            result.append(depths.amax(dim=(1,2))*binding.ratio)
        return torch.stack(result,dim=1)

    def loss(self, state):
        breach = self.values(state)
        norm_rise = torch.linalg.vector_norm(breach, dim=0) - self.norm
        worst_rise = torch.relu(breach - self.cap).amax(dim=0)
        return {
            'cavity_total': 6.0 * (torch.relu(norm_rise + .00035) / .001).square().sum(),
            'cavity_peak': 5.0 * (worst_rise / .0008).square().sum(),
            'cavity_regularization': .30 * (breach / .006).square().mean(),
        }
