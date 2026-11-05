"""Vectorized +/- geometry agrees with the retained scalar replay."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from calibration.program_tracking.body_tracking.body_resident_torch import ResidentFootLinearization

class Consumer:
    compatibility_width=5
    def evaluate(self,native,frames,samples):
        return torch.stack((native[:,7,0],native[:,8,1],frames[:,7,0,1],samples[:,:,2].amax(1),torch.clamp_min(samples[:,:,0],0).sum(1)),dim=1)

@pytest.mark.parametrize('columns',[[6,7,8],[8,6],[66,68]])
def test_same_joint_batch_preserves_each_direction_and_sign(columns):
    torch.manual_seed(19);n=3;v=5;r=object.__new__(ResidentFootLinearization);r.n=n;r.vertices=v;r.device=torch.device('cpu');r.dtype=torch.float64;r.eye=torch.eye(3,dtype=r.dtype);r.scale=1.3
    r.parents=np.r_[-1,np.zeros(54,dtype=int)];r.refLocal=r._rot(torch.randn(n,21,3,dtype=r.dtype)*.2);r.J=torch.randn(55,3,dtype=r.dtype);r.posedirs=torch.randn(486,v*3,dtype=r.dtype)*.01;r.consumer=Consumer()
    frames=r._rot(torch.randn(n,22,3,dtype=r.dtype)*.3)
    pose=dict(delta=torch.randn(n,69,dtype=r.dtype)*.1,frames=frames,position=torch.randn(n,55,3,dtype=r.dtype),P=torch.randn(n,3,dtype=r.dtype),Q=r._rot(torch.randn(n,3,dtype=r.dtype)*.2),recovered_local=frames[:,1:],blendR=torch.randn(n,v,3,3,dtype=r.dtype),samples=torch.randn(n,v,3,dtype=r.dtype))
    joint=(columns[0]-6)//3+1;subtree=(torch.arange(joint,min(22,joint+3)),torch.randn(n,v,3,3,dtype=r.dtype),torch.randn(n,v,3,dtype=r.dtype))
    scalar=torch.stack([r._pair(pose,c,1e-5,subtree) for c in columns]);batch=r._pairs(pose,columns,1e-5,subtree)
    torch.testing.assert_close(batch,scalar,rtol=1e-7,atol=2e-9)
