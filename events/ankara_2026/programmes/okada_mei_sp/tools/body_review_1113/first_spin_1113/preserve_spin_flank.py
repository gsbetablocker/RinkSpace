from argparse import ArgumentParser
import numpy as np

from spin_model import JOB,PARENT,packet,curve,clock,pts,seconds,load,curve_from_packet,save_candidate
from calibration.program_tracking.body_tracking.absolute_motion_curve import _exp,_log
from experiments.skate_video_v1.motion_review.scoped_fit import geometry

p=ArgumentParser()
p.add_argument('name')
p.add_argument('--target',default='flanks_preserved')
p.add_argument('--exact-before',type=float,default=81.09)
p.add_argument('--fraction',type=float,default=1.)
a=p.parse_args()
proposal=curve_from_packet(load(JOB/'proposals'/f'{a.name}.npz'))
keep=curve.knots[:-4]+seconds[0] >= a.exact_before
z=curve.position_controls.copy()
q=curve.rotation_controls.copy()
if a.fraction==1.:
    z[keep]=proposal.position_controls[keep]
    q[keep]=proposal.rotation_controls[keep]
else:
    z[keep]+=a.fraction*(proposal.position_controls[keep]-z[keep])
    q[keep]=q[keep]@_exp(a.fraction*_log(q[keep].swapaxes(-1,-2)@proposal.rotation_controls[keep]))
save_candidate(packet,geometry(curve,z,q),JOB/'proposals'/f'{a.target}.npz',parent=PARENT)
print('first changed knot',float(np.min(curve.knots[:-4][keep & np.any(q!=curve.rotation_controls,axis=(1,2,3))])+seconds[0]))
