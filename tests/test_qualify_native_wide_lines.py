import numpy as np
from qualify_native_wide_lines import measured_piece_members


def test_remeasurement_keeps_opposite_arc_and_unseen_ends():
    uv=np.array([[0.,10.],[24.,10.],[48.,10.]])
    normal=np.array([0.,1.])
    sides=uv[:,None,:]+np.array([-2.,2.])[None,:,None]*normal
    quant=sides[:,:,None,:]+np.array([-.5,0.,.5])[None,None,:,None]*normal
    measured=dict(candidate_uv=uv,candidate_side_quantiles_uv=quant)
    parent=np.array([[10.,11.],[30.,6.],[10.,50.],[-2.,10.],[50.,10.]])
    # The nearby old paint-edge read is replaced together with its center;
    # the other side of a circle and unmeasured ends are retained.
    np.testing.assert_array_equal(measured_piece_members(parent,measured,[0,1,2]),[True,True,False,False,False])


def test_singleton_never_claims_an_unmeasured_interval():
    point=np.array([[7.,8.]])
    measured=dict(candidate_uv=point,candidate_side_quantiles_uv=np.zeros((1,2,3,2)))
    np.testing.assert_array_equal(measured_piece_members([[7.,8.],[7.,9.]],measured,[0]),[True,False])
