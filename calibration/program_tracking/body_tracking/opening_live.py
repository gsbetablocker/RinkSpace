"""Source-reviewed opening factors on the complete current body state.

Local per-time feature derivatives are scattered through their own spline
bases before the quiet-phase difference operator is applied.  The common
standing state remains free; hands/arms/head are absent from quiet rows.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix,csr_matrix
from .opening_motion import (START,HAND_RELEASE_START,HAND_RELEASE_END,
    hand_waist_skin_ids,hand_waist_vectors)
from .opening_timing import (QUIET_END,QUIET_BODY_JOINTS,
    QUIET_POSITION_SCALE,QUIET_ROTATION_SCALE_DEG)
from .opening_repair import smoothstep


class OpeningLiveFactors:
    def __init__(self,body):
        self.body=body;self.reference=body.reference
        seconds=self.reference.p['pts'][body.frames]/12800.
        self.selected=np.flatnonzero((seconds>=START-1e-9)&(seconds<=HAND_RELEASE_END+1e-9))
        self.times=body.mid_times[self.selected]
        self.width=6+3*len(QUIET_BODY_JOINTS)+9
        self.rows={};rr=[];cc=[];vv=[];row=0
        self.quiet=np.flatnonzero(seconds[self.selected]<=QUIET_END+1e-9)
        self.waist_ids=self.palm_ids=None
        if not len(self.selected):
            self.operator=csr_matrix((0,0));return
        self.ref=self.reference.at(self.times)
        self.waist_ids,self.palm_ids=hand_waist_skin_ids(self.reference.binding.surface)
        self.q_seed=self.ref['Q'][0].copy()
        self.local_seed=self.ref['local_R'][0,np.asarray(QUIET_BODY_JOINTS)-1].copy()
        scale=np.r_[np.full(3,QUIET_POSITION_SCALE),
            np.full(self.width-12,np.deg2rad(QUIET_ROTATION_SCALE_DEG))]
        first=row
        for frame in self.quiet[1:]:
            for j,sigma in enumerate(scale):
                rr.extend((row,row));cc.extend((frame*self.width+j,self.quiet[0]*self.width+j))
                vv.extend((1/sigma,-1/sigma));row+=1
        self.rows['source_confirmed_quiet_body']=slice(first,row)
        first=row
        weight=1-smoothstep((seconds[self.selected]-HAND_RELEASE_START)/(HAND_RELEASE_END-HAND_RELEASE_START))
        for frame,w in enumerate(weight):
            if w<=0:continue
            for j in range(9):
                rr.append(row);cc.append(frame*self.width+self.width-9+j)
                vv.append(w/(.02*np.sqrt(3)));row+=1
        self.rows['observed_right_hand_waist_relation']=slice(first,row)
        # Same discrete-source measure as the inherited opening image factors.
        self.operator=coo_matrix((np.asarray(vv)*np.sqrt(.02/2),(rr,cc)),
            shape=(row,len(self.selected)*self.width)).tocsr()

    def features(self,s):
        q=Rotation.from_matrix(self.q_seed.T@s['Q']).as_rotvec()
        local=s['local_R'][:,np.asarray(QUIET_BODY_JOINTS)-1]
        a=Rotation.from_matrix((self.local_seed.transpose(0,2,1)[None]@local).reshape(-1,3,3)).as_rotvec().reshape(len(q),-1)
        hand=hand_waist_vectors(self.reference,s,self.waist_ids,self.palm_ids)
        return np.c_[s['P'],q,a,hand]

    def evaluate(self,x,jacobian=False):
        if not len(self.selected):
            return dict(residual=np.empty(0),jac=csr_matrix((0,self.body.size)) if jacobian else None,
                parts={},metrics={},source_pts=[])
        state=self.body.state;d=state.increments(x,self.times)
        s=self.reference.world(self.ref,d);features=self.features(s)
        residual=np.asarray(self.operator@features.ravel()).ravel()
        J=None
        if jacobian:
            values=self.body.execution.state_differences(self.reference,self.ref,d,state.active,self.features,step=1e-5)
            local=np.stack(values,-1);B=state.basis(self.times);fr,kn=np.nonzero(B)
            v=local[fr]*B[fr,kn,None,None]
            rr=np.broadcast_to(fr[:,None,None]*self.width+np.arange(self.width)[None,:,None],v.shape)
            cc=np.broadcast_to(kn[:,None,None]*state.d+np.arange(state.d),v.shape)
            feature_J=coo_matrix((v.ravel(),(rr.ravel(),cc.ravel())),shape=(features.size,self.body.size)).tocsr()
            J=self.operator@feature_J
        parts={k:float(residual[v]@residual[v]/2)for k,v in self.rows.items()}
        metrics={}
        if len(self.quiet):
            qi=self.quiet;root=Rotation.from_matrix(s['Q'][qi[0]].T@s['Q'][qi]).magnitude()
            loc=s['local_R'][qi][:,np.asarray(QUIET_BODY_JOINTS)-1]
            angles=Rotation.from_matrix((loc[0].transpose(0,2,1)[None]@loc).reshape(-1,3,3)).magnitude()
            metrics=dict(quiet_position_range_m=float(np.linalg.norm(s['P'][qi]-s['P'][qi[0]],axis=1).max()),
                quiet_root_range_deg=float(np.rad2deg(root.max())),quiet_body_range_deg=float(np.rad2deg(angles.max())))
        metrics['hand_waist_max_distance_m']=float(np.linalg.norm(features[:,-9:].reshape(-1,3,3),axis=-1).max())
        return dict(residual=residual,jac=J,parts=parts,metrics=metrics,
            source_pts=self.reference.p['pts'][self.body.frames[self.selected]].tolist(),
            measure='fixed source observation measure 0.02s; not contact duration',
            prior_origin='source-reviewed quiet phase and sliding skin relation, evaluated on all saved layers plus current state')
