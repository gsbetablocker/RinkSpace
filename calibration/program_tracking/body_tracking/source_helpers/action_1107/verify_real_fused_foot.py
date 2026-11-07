import json
from pathlib import Path
import sys
import numpy as np
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.body_resident_torch import ChunkedResidentFootLinearization
from calibration.program_tracking.body_tracking.body_clock import source_seconds

project,prepared,param,target=map(Path,sys.argv[1:])
ref,_,_,_,_=load_reference(project,prepared)
saved=load(param)
ref.absolute_curve=curve_from_packet(saved,seed=True)
ref.relative_skate_curve=shoe_curve_from_packet(saved,seed=True)
state=SequenceState(ref,source_seconds(ref.p)[[0,-1]])
shot=ref.p['shot_ids']
cuts=np.flatnonzero(shot[1:]!=shot[:-1])+1
indices=np.unique(np.r_[0,1,509:515,1021:1027,cuts-1,cuts,8325:8329])
sample=ref.times[indices]
d=state.increments(saved['x'][17:],sample)
owner=ChunkedResidentFootLinearization(ref,ref.at(sample),ref.binding,batch_size=7)
args=dict(skate_relative_R=d.skate_relative_R)
cols=np.arange(6,69)
separate,bodyA=owner.evaluate(d,cols,step=1e-8,**args)
maskA=owner.baseline_collar_unknown.copy()
_,shoeA=owner.shoe_derivatives(d,d.skate_relative_R,step=1e-8)
old=owner.measurement()
owner.clear_result_cache()
owner.reset_measurement()
fused,bodyB,shoeB=owner.linearize_body_and_shoes(d,cols,body_step=1e-8,shoe_step=1e-8,**args)
new=owner.measurement()
report=dict(exposures=indices.tolist(),pts=ref.p['pts'][indices].tolist(),mask_equal=bool(np.array_equal(maskA,owner.baseline_collar_unknown)),
    max_residual_error=float(np.max(abs(fused-separate))),max_body_local_error=float(np.max(abs(np.array(bodyB)-bodyA))),
    max_shoe_local_error=float(np.max(abs(shoeA-shoeB))),separate=old,fused=new)
assert report['mask_equal']
assert report['max_residual_error']<1e-13
assert report['max_body_local_error']<1e-7 and report['max_shoe_local_error']<1e-7
with target.open('x',encoding='utf8') as stream: json.dump(report,stream,indent=2)
print({k:v for k,v in report.items() if k.startswith('max') or k=='mask_equal'})
