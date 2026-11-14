from argparse import ArgumentParser
import numpy as np

from spin_model import JOB,PARENT,packet,curve,seconds,load,curve_from_packet,save_candidate
from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp,_log
from experiments.skate_video_v1.motion_review.scoped_fit import geometry

p=ArgumentParser()
p.add_argument('name')
p.add_argument('target')
p.add_argument('--base',default='flanks_preserved')
p.add_argument('--keep',type=float,nargs='*',default=[])
p.add_argument('--taper',type=float,default=.10)
p.add_argument('--range',type=float,nargs=2)
p.add_argument('--fraction',type=float,default=1.)
a=p.parse_args()
read_curve=lambda name:curve_from_packet(load(JOB/'proposals'/f'{name}.npz'))
base=read_curve(a.base)
proposal=read_curve(a.name)
a0=curve.knots[:-4]+seconds[0]
a1=curve.knots[4:]+seconds[0]
w=np.full(len(a0),a.fraction)
if a.range:
    # Constrain the blend to this exact knot support, with a continuous
    # transition into the retained later spin.
    lo,hi=a.range
    z=np.clip((hi-a1)/a.taper,0,1)*np.clip((a0-lo)/a.taper,0,1)
    w*=z*z*(3-2*z)
for t in a.keep:
    distance=np.maximum(0.,np.maximum(a0-t,t-a1))
    z=np.clip(distance/a.taper,0,1)
    w=np.minimum(w,a.fraction*z**2*(3.-2.*z))
p=base.position_controls.copy()
R=base.rotation_controls.copy()
full=(w==1.)
between=(w>0)&~full
p[full]=proposal.position_controls[full]
R[full]=proposal.rotation_controls[full]
p[between]+=w[between,None]*(proposal.position_controls[between]-p[between])
j=np.argwhere(between[:,None] & np.any(proposal.rotation_controls!=R,axis=(-1,-2)))
i,k=j.T
R[i,k]=R[i,k]@_exp(w[i,None]*_log(R[i,k].swapaxes(-1,-2)@proposal.rotation_controls[i,k]))
save_candidate(packet,geometry(curve,p,R),JOB/'proposals'/f'{a.target}.npz',parent=PARENT)
print('combined',a.target,'protected',a.keep,'fully updated',int(full.sum()),flush=True)
