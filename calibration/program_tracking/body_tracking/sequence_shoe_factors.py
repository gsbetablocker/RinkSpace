"""Sparse right-shoe tangent chains, sharing native exposure and ankle pivots."""
import numpy as np
from .fullbody_fit import skew


def image_shoe_chain(helper,delta,state,projection,Jrotation,D=None):
    n=len(helper.rows);pick=np.arange(n*5);feet=np.repeat(helper.feet,5)
    R=state['R'][pick,feet]
    RFB,tFB,_=helper.binding.parameters();ankle=-np.einsum('fi,fij->fj',tFB,RFB)
    J=Jrotation+np.einsum('nlij,nljk->nlik',R.reshape(n,5,3,3),skew(ankle[feet]).reshape(n,5,3,3))
    if helper.planar_rows.any():J[helper.planar_rows,:,2,:]=0.
    pixel=np.einsum('nplc,nlca->npla',projection['pixel_jac_nodes'][:,0],J)
    local=np.zeros((n,2,5,6))
    for foot in (0,1):
        keep=helper.feet==foot;local[keep,:,:,foot*3:foot*3+3]=pixel[keep]
    if D is not None:local=np.einsum('nab,nbld->nald',np.asarray(D),local)
    return delta.owner.shoe_exposure_chain(local,delta.parameters,delta.sample_times)
