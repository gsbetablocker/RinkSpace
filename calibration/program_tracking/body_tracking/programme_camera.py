"""Frozen native multi-shot dispatch and an explicit static crop camera.

Both expose the existing body projection interface; no camera is optimized.
"""
from types import SimpleNamespace
import numpy as np
from .fullbody_state import camera_binding
from .source_visible_model import camera_centres

class ProgrammeCamera:
    image_height=1080
    def __init__(self,project,packet,shot_ids):
        self.shot_ids=np.asarray(shot_ids);self.cameras={};self.first={};infos={};poses={}
        for shot in dict.fromkeys(self.shot_ids.tolist()):
            ix=np.flatnonzero(self.shot_ids==shot);part=dict(packet,pts=packet['pts'][ix])
            camera,info,state,pose=camera_binding(project,SimpleNamespace(p=part),shot_id=shot)
            self.cameras[shot]=camera;self.first[shot]=int(ix[0]);infos[shot]=info;poses[shot]=pose
        # Only sparse S05 source factors consume this historical scalar readout.
        # Full body, exporter and dispatcher use per-exposure rig/readout below.
        representative=self.cameras.get('s05',next(iter(self.cameras.values())))
        self.p=dict(packet,rig=representative.p['rig'],controls=representative.p['controls'],first_control_time=representative.p['first_control_time'],control_dt=representative.p['control_dt'])
        self.p['motion_times']=np.empty(len(self.shot_ids))
        for shot,camera in self.cameras.items():self.p['motion_times'][self.shot_ids==shot]=camera.p['motion_times']
        self.scale=representative.scale;self.core=representative.core
        self.info=dict(next(iter(infos.values())))
        self.info.update(shot='programme',shots=infos,no_per_shot_alignment=True,source_time_base=[1,12800])
        self.midpoint={key:np.empty((len(self.shot_ids),*np.asarray(next(iter(poses.values()))[key]).shape[1:]))for key in ('rotation','centre','values','status','fx','fy','cx','cy','k1','k2','p1','p2')}
        for shot,pose in poses.items():
            mask=self.shot_ids==shot
            for key in self.midpoint:self.midpoint[key][mask]=pose[key]
    def _groups(self,frames):
        frames=np.asarray(frames,dtype=int)
        for shot in dict.fromkeys(self.shot_ids[frames].tolist()):
            ix=np.flatnonzero(self.shot_ids[frames]==shot);yield ix,self.cameras[shot],frames[ix]-self.first[shot]
    def readout_times(self,frames):
        frames=np.asarray(frames);tau=np.empty(len(frames))
        for ix,camera,local in self._groups(frames):tau[ix]=camera.p['rig'][37]
        return tau
    def camera_centres(self,frames,rows=None):
        out=np.empty((len(frames),3))
        for ix,camera,local in self._groups(frames):out[ix]=camera_centres(camera,local,None if rows is None else np.asarray(rows)[ix])
        return out
    def project(self,points,frames):
        points=np.asarray(points);out=np.empty((*points.shape[:-1],2))
        for ix,camera,local in self._groups(frames):out[ix]=camera.project(points[ix],local)
        return out
    def with_jacobian(self,points,frames):
        points=np.asarray(points);uv=np.empty((*points.shape[:-1],2));J=np.empty((*points.shape[:-1],2,3))
        for ix,camera,local in self._groups(frames):uv[ix],J[ix]=camera.with_jacobian(points[ix],local)
        return uv,J

class StaticCropCamera:
    """Cam7 rectified crop pixels; world extrinsics are fixed official values."""
    def __init__(self,packet,K_input,R_w2c,C_world_m,*,image_height=1024):
        self.p=dict(packet,rig=np.zeros(43));self.K=np.asarray(K_input,float);self.R=np.asarray(R_w2c,float);self.C=np.asarray(C_world_m,float)
        if self.R.ndim==3:
            if not np.allclose(self.R,self.R[:1],rtol=0,atol=1e-12):raise ValueError('Static FS camera rotation changed')
            self.R=self.R[0]
        if self.C.shape!=(3,)or self.K.shape!=(len(packet['pts']),3,3):raise ValueError('Explicit crop camera arrays required')
        self.image_height=int(image_height);self.scale=1.
    def readout_times(self,frames):return np.zeros(len(frames))
    def camera_centres(self,frames,rows=None):return np.broadcast_to(self.C,(len(frames),3)).copy()
    def with_jacobian(self,points,frames):
        points=np.asarray(points);K=self.K[np.asarray(frames,int)];X=(points-self.C)@self.R.T;q=np.einsum('nij,nkj->nki',K,X)
        if np.any(q[...,2]<=1e-6):raise ValueError('FS point behind frozen camera')
        uv=q[...,:2]/q[...,2,None]
        D=(K[:,None,:2,:]-uv[...,None]*K[:,None,2:3,:])/q[...,2,None,None]
        return uv,D@self.R
    def project(self,points,frames):return self.with_jacobian(points,frames)[0]
