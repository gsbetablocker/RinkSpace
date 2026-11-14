from pathlib import Path

import numpy as np
import torch

from calibration.program_tracking.body_tracking.source_visible_model import structure_geometry
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from experiments.skate_video_v1.inputs import read
from experiments.skate_video_v1.motion_review.scoped_fit import tn,ti,cpu


class ProtectedSkateRows:
    """Original rolling-shutter rows and their saved visible segment choices."""

    def __init__(self, owner, reference, document, stage):
        rows = [row for row in document['full_source_skate_structures']['rows']
                if row.get('evidence_status')=='observed' and row.get('role')=='fit'
                and row.get('optimization_enabled',True)]
        checkpoint=load(stage/'source_associations.npz')
        choice=checkpoint['source_association_contour_fixed_choice']
        limits=read(stage/'primary_quality_baseline.json')['source']
        models=structure_geometry(reference.binding)
        endpoints=models['points'][np.column_stack((models['starts'][choice],models['ends'][choice]))]
        cursor=0
        group={}
        caps=[]
        self.ids=[]
        self.old=[]
        self.radius=[]
        sizes=[]
        for row in rows:
            pixels=np.asarray(row['polyline_uv'])
            end=cursor+len(pixels)
            at=int(np.searchsorted(owner.fit.pts,row['pts']))
            if at<len(owner.fit.pts) and owner.fit.pts[at]==row['pts']:
                value=group.setdefault((at,row['foot']),[])
                old,band=limits[row['id']]
                assert band==row['pixel_uncertainty_radius_px']
                self.ids.append(row['id'])
                self.old.append(old)
                self.radius.append(band)
                caps.append(band+old+.25*band)
                sizes.append(len(pixels))
                value.append((endpoints[cursor:end],pixels,len(self.ids)-1))
            cursor=end
        assert cursor==len(choice)
        self.groups=list(group)
        self.frames=np.asarray([f for f,_ in self.groups],dtype=int)
        self.feet=ti([foot for _,foot in self.groups])
        n=max(sum(len(px) for _,px,_ in batch) for batch in group.values())
        q=np.zeros((len(group),n,2,3))
        uv=np.zeros((len(group),n,2))
        ii=[None]*len(caps)
        jj=[None]*len(caps)
        for i,batch in enumerate(group.values()):
            j=0
            for xyz,pixels,k in batch:
                end=j+len(xyz)
                q[i,j:end]=xyz
                uv[i,j:end]=pixels
                ii[k]=np.full(len(xyz),i,int)
                jj[k]=np.arange(j,end)
                j=end
        length=max(sizes)
        def pad(ix):return np.stack([np.pad(v,(0,length-len(v)),mode='edge') for v in ix])
        self.a,self.b=ti(pad(ii)),ti(pad(jj))
        # Sort padding behind the real samples, then use the same linear p95.
        self.keep=ti(np.arange(length)[None,:]<np.asarray(sizes)[:,None]).bool()
        fractional=np.asarray(sizes)*.95-.95
        self.lower,self.upper=map(ti,(np.floor(fractional),np.ceil(fractional)))
        self.mix=tn(fractional-np.floor(fractional))
        self.row=ti(np.arange(len(caps)))
        self.q=tn(q.reshape(len(group),2*n,3))
        self.uv=tn(uv)
        self.cap=tn(caps)
        self.band=tn(self.radius)
        self.old=tn(self.old)

    def distances(self, fit, state):
        R,O=state[6:8]
        R=R.reshape(-1,5,2,3,3)[self.frames,:,self.feet]
        O=O.reshape(-1,5,2,3)[self.frames,:,self.feet]
        xyz=O[:,:,None]+self.q[:,None]@R.transpose(-1,-2)
        uv=fit.project(xyz.flatten(0,1),self.frames).reshape(*self.uv.shape[:-1],2,2)
        start,end=uv.unbind(-2)
        v=end-start
        u=((self.uv-start)*v).sum(-1)/v.square().sum(-1).clamp_min(1.e-12)
        d=torch.linalg.vector_norm(start+u.clamp(0,1)[...,None]*v-self.uv,dim=-1)
        return d

    def band_p95(self, distance):
        ranked=torch.where(self.keep,(distance[self.a,self.b]-self.band[:,None]).clamp_min(0),1.e5)
        s=ranked.sort(dim=-1).values
        return s[self.row,self.lower]*(1-self.mix)+s[self.row,self.upper]*self.mix

    def loss(self, fit, state):
        outside=self.band_p95(self.distances(fit,state))
        cap=self.old+.25*self.band
        improve=torch.sqrt(1+(outside/self.band).square())-1
        excess=torch.relu(outside-cap+.08)
        # Prevent a single visible row from getting worse at the expense of
        # the other rows. The exact production guard still runs afterwards.
        return 6.*improve.mean()+20.*excess.square().mean()+10.*excess.square().amax()

    def replay(self,fit,state):
        p=cpu(self.band_p95(self.distances(fit,state)))
        e=p-cpu(self.old)
        print('source contour anchor',len(e),'peak delta',float(np.max(abs(e))),
              'positive cap excess',float(np.max(e-.25*cpu(self.band))),flush=True)
        if np.max(abs(e))>.05:
            ids=np.argsort(abs(e))[-8:]
            print([(self.ids[i],round(float(e[i]),4)) for i in ids],flush=True)
