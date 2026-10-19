"""Tiny transport-only fixture; no programme observation or target is used."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix

def absolute_fixture(q):return q

class TinySourceBlock:
    def __init__(self,name,offset):
        self.name=name;self.offset=float(offset);self.absolute=absolute_fixture
        self.bundle=SimpleNamespace(motion_rows=4,profile_weights=None,robust_phase=False,abs_groups=())
        self.bundle.profile_motion_residual=self.profile_motion_residual
        self.actor_output_capacity=dict(rows=6,nnz=12,motion_rows=4)
    def local(self,q):return q
    def profile_motion_residual(self,q):
        q=np.asarray(q,float)
        return np.array([q[0]+q[3],2*q[1],3*q[2],q[0]-q[3]])+self.offset+np.arange(4)/8
    def evaluate(self,q,jacobian):
        q=np.asarray(q,float);J=np.array([[1.,0.,0.,1.],[0.,2.,0.,0.],[0.,0.,3.,0.],[1.,0.,0.,-1.],[0.,1.,1.,0.],[1.,0.,1.,0.]])
        residual=J@q+self.offset+np.arange(6)/8
        if self.bundle.robust_phase and self.bundle.profile_weights is not None:
            weights=np.r_[self.bundle.profile_weights.ravel(),np.ones(2)]
            residual=residual*weights;J=J*weights[:,None]
        return (residual,csr_matrix(J))if jacobian else residual
    def absolute_records(self,q,rows=None):
        roles=('FIT',)if rows is None else tuple(rows)
        for role in roles:yield dict(block=self.name,role=role,value=float(np.asarray(q).sum()+self.offset))
