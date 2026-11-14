from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from calibration.program_tracking.body_tracking.programme_repair_factors import component_ice
from calibration.program_tracking.body_tracking.programme_surface import (
    CollisionSurface, MOVING_NORMAL_MODE, OrientedFaceWitness, witnesses_packet)
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from experiments.skate_video_v1.motion_review.scoped_fit import Fit, cpu, ti, tn


FIELDS = ('sample','vertex','face_vertices','bary','normal','initial_gap','face_a','face_b')


def at_centres(fit, times):
    r = fit.fit.ref
    ref = SimpleNamespace(times=times, p=r.p, at=r.at, binding=r.binding, body=r.body)
    probe = Fit(ref, SimpleNamespace(readout_times=lambda frames: np.zeros(len(frames))),
                fit.fit.parent, np.arange(len(times)), fit.active, fit.ctrl)
    for key in ('weights','beta','indices','chart','hand_tensor','relR','relT'):
        value = getattr(probe,key)
        if value is not None:
            setattr(probe,key,value[2::5])
    return probe


class DeformedUpper:
    def __init__(self, fit):
        binding = fit.fit.ref.binding
        pack = binding.packet()
        self.template = pack
        R, T, _ = binding.parameters()
        self.ankle = tn(-np.einsum('fi,fij->fj',T,R))
        self.length = pack['parameters'].sole_length_m
        i = int(np.argmin(abs(pack['xs']/self.length-.28)))
        self.height = tn(pack['floor'][i]+pack['rise'][i])
        self.width = tn(pack['width'][i])
        self.radius = .027/binding.ratio
        self.max_alpha = tn([.020,.015,.012])/binding.ratio
        self.ratio = binding.ratio

    def alpha(self, fit, A, J, *, nodes=slice(None)):
        a = np.array([7,8])
        K = J[:,[4,5]]
        ankle = J[:,a]
        axes = A[:,a] @ fit.relR[nodes]
        direction = ((K-ankle)*fit.body.scale)[...,None,:] @ axes
        direction = direction.squeeze(2)
        lam = ((self.height-self.ankle[:,2])/direction[:,:,2].clamp_min(1.e-6)).clamp(0,1)
        c = self.ankle + lam[...,None]*direction
        alpha = torch.stack((c[...,0]+self.radius-.46*self.length,-c[...,0]+self.radius,
                             c[...,1].abs()+self.radius-self.width),-1)
        return torch.minimum(alpha.clamp_min(0),self.max_alpha)

    def weights(self, vertices, mask=None):
        x, y, z = np.moveaxis(vertices,-1,0)
        floor = np.interp(x,self.template['xs'],self.template['floor'])
        weight = np.clip((.65*self.length-x)/(.20*self.length),0,1)*np.clip(
            (z-floor)/(.08/self.ratio),0,1)
        if mask is not None:
            weight *= mask
        front = np.clip(x/(.46*self.length),0,1)*weight
        return tuple(map(tn,(vertices,front,weight-front,weight*np.sign(y))))

    @staticmethod
    def apply(values, alpha):
        vertices, front, rear, side = values
        dx = alpha[...,0,None]*front-alpha[...,1,None]*rear
        dy = alpha[...,2,None]*side
        return vertices+torch.stack((dx,dy,torch.zeros_like(dx)),-1)


class DenseIce:
    def __init__(self, fit, times, reference_pose, edge, tolerance):
        self.fit = at_centres(fit,times)
        binding = fit.fit.ref.binding
        self.ratio = binding.ratio
        self.upper = DeformedUpper(fit)
        shape = binding.packet()
        self.upper_points = self.upper.weights(shape['upper'])
        tr = shape['template']
        self.parts = [*edge[:2]]
        for name, (first, last) in tr['part_ranges'].items():
            if name not in ('steel_runner','toe_pick_head'):
                self.parts.append(tn(tr['vertices'][first:last]))
        names, old = component_ice(binding,reference_pose)
        self.floor = tn(np.minimum(old.min(0),-np.array([tolerance.get(k,tolerance['solid']) for k in names]))+.00010)
        self.old = old

    def values(self, parameter):
        self.fit.parameter = parameter
        _,_,_,A,J,_,R,O,_ = self.fit.state()
        Z = R[...,2,:]
        height = [(Z@v.T+O[...,2,None]).amin(-1) for v in self.parts]
        alpha = self.upper.alpha(self.fit,A,J)
        upper = self.upper.apply(self.upper_points,alpha)
        height.append(((Z[...,None,:]*upper).sum(-1)+O[...,2,None]).amin(-1))
        return torch.stack(height,-1)*self.ratio

    def loss(self, parameter):
        breach = torch.relu(self.floor-self.values(parameter))/.0010
        return 12.*breach.square().amax(0).sum()+40.*breach.square().mean()


class SurfaceBounds:
    def __init__(self, fit, initial, reference, scan, document, support, fk, prior_scan=None):
        geom = CollisionSurface(reference,document)
        self.geom = geom
        first = np.searchsorted(scan['times'],support[0],side='right')
        last = np.searchsorted(scan['times'],support[1],side='left')
        times = scan['times'][first:last]
        self.times = times
        self.fk = fk
        self.moving = document['programme_structure']['surface_normal_mode']==MOVING_NORMAL_MODE
        old = geom.vertices(fk(fit.fit.parent,times))
        current = geom.vertices(fk(initial,times))
        saved = (scan['sample']>=first)&(scan['sample']<last)
        base = {k:scan[k][saved] for k in FIELDS}
        base['sample'] -= first
        current_scan = self.scan(current)
        self.current_scan = current_scan
        scans = [base,current_scan] if prior_scan is None else [base,current_scan,prior_scan]
        merged = {k:np.concatenate([v[k] for v in scans]) for k in FIELDS}
        ids = np.c_[merged['sample'],merged['vertex'],merged['face_a'],merged['face_b']]
        _, index = np.unique(ids,axis=0,return_index=True)
        merged = {k:v[index] for k,v in merged.items()}
        self.ready = len(index)>0
        if not self.ready:
            return
        pair = np.sort(geom.parts[np.c_[merged['face_a'],merged['face_b']]],axis=1)
        labels, groups = np.unique(pair,axis=0,return_inverse=True)
        h0,h1 = self.numpy_gaps(old,merged),self.numpy_gaps(current,merged)
        floor = np.zeros(len(labels))
        np.minimum.at(floor,groups,h0)
        # Keep a little slack inside the production per-pair tolerance.
        target = np.maximum(floor[groups]-.0003,-.100)
        near = np.minimum(h0,h1) < np.minimum(target+.025,-.001)
        self.ready = bool(near.any())
        initial_min = np.zeros(len(labels))
        np.minimum.at(initial_min,groups,h1)
        print('surface witnesses',len(index),'retained',int(near.sum()),'initial pressure',
              [(geom.part_names[a],geom.part_names[b],round(floor[k]*1000,1),round(initial_min[k]*1000,1))
               for k,(a,b) in enumerate(labels) if initial_min[k]<max(floor[k]-.002,-.1025)],flush=True)
        if not self.ready:
            return
        selection = {k:v[near] for k,v in merged.items()}
        self.bounds = tn(target[near])
        self.group = ti(groups[near])
        self.groups = len(labels)
        self.samples = np.unique(selection['sample'])
        self.frame = ti(np.searchsorted(self.samples,selection['sample']))
        self.ids = np.unique(np.r_[selection['vertex'],selection['face_vertices'].ravel()])
        self.va = ti(np.searchsorted(self.ids,selection['vertex']))
        self.vb = ti(np.searchsorted(self.ids,selection['face_vertices']))
        self.bary = tn(selection['bary'])
        self.normal = tn(selection['normal'])
        self.sign = tn(geom.response_signs(np.c_[selection['face_a'],selection['face_b']])[:,0])
        self.fit = at_centres(fit,times[self.samples])
        s = reference.binding.surface
        skin = self.ids[self.ids<geom.skin_count]
        self.vshape,self.W = tn(s.shaped[skin]),tn(s.weights[skin])
        self.posedirs = tn(s.posedirs.reshape(486,-1,3)[:,skin].reshape(486,-1))
        self.upper = DeformedUpper(fit)
        self.gear = []
        for foot in (0,1):
            first = geom.skin_count+foot*geom.gear_count
            vertices = self.ids[(self.ids>=first)&(self.ids<first+geom.gear_count)]-first
            xyz = np.r_[geom.template_vertices,geom.upper_vertices][vertices]
            self.gear.append(self.upper.weights(xyz,vertices>=len(geom.template_vertices)))

    def scan(self, vertices):
        geom = self.geom
        def one(item):
            i,v = item
            witness,_ = geom.scan_sample(v,self.times[i])
            keep = {}
            for w in witness:
                key = int(geom.parts[w[5]]),int(geom.parts[w[6]])
                unique = keep.setdefault(key,{})
                if w[0] not in unique or w[4]<unique[w[0]][4]:
                    unique[w[0]]=w
            return [(i,w) for g in keep.values() for w in sorted(g.values(),key=lambda row:row[4])[:2]]
        with ThreadPoolExecutor(max_workers=8) as workers:
            records = [v for group in workers.map(one,enumerate(vertices)) for v in group]
        return witnesses_packet(records)

    def numpy_gaps(self, vertices, pack):
        i = pack['sample']
        point = vertices[i,pack['vertex']]
        face = vertices[i[:,None],pack['face_vertices']]
        if self.moving:
            signs = self.geom.response_signs(np.c_[pack['face_a'],pack['face_b']])[:,0]
            return OrientedFaceWitness.from_geometry(point,face,pack['bary'],signs).gap
        return np.einsum('ni,ni->n',pack['normal'],point-np.einsum('ni,nij->nj',pack['bary'],face))

    def gaps(self, parameter):
        f = self.fit
        f.parameter = parameter
        _,_,_,A,J,_,_,_,L = f.state()
        translation = J-(A@f.J[None,:,:,None]).squeeze(-1)
        local = self.vshape+((L-f.eye).reshape(len(A),486)@self.posedirs).reshape(len(A),-1,3)
        blend = (self.W@A.reshape(len(A),55,9)).reshape(len(A),-1,3,3)
        skin = ((blend@local[...,None]).squeeze(-1)+self.W@translation-f.J[0])*f.body.scale
        alpha = self.upper.alpha(f,A,J)
        pieces = [skin]
        for foot, a in enumerate((7,8)):
            R = A[:,a]@f.relR[:,foot]
            O = (J[:,a]-f.J[0])*f.body.scale+(A[:,a]@f.relT[:,foot,:,None]).squeeze(-1)
            xyz = self.upper.apply(self.gear[foot],alpha[:,foot])
            pieces.append(O[:,None]+xyz@R.transpose(-1,-2))
        all_points = torch.cat(pieces,dim=1)*self.geom.ratio
        point = all_points[self.frame,self.va]
        face = all_points[self.frame[:,None],self.vb]
        normal = self.normal
        if self.moving:
            normal = torch.linalg.cross(face[:,1]-face[:,0],face[:,2]-face[:,0])
            normal = self.sign[:,None]*normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True).clamp_min(1.e-12)
        offset = point-(face*self.bary[...,None]).sum(1)
        return (offset*normal).sum(-1)

    def loss(self, parameter):
        if not self.ready:
            return parameter.sum()*0.
        breach = torch.relu(self.bounds-self.gaps(parameter))/.002
        peak = torch.zeros(self.groups,device=parameter.device).scatter_reduce(0,self.group,breach,reduce='amax')
        return 8.*peak.square().sum()+16.*breach.square().mean()
