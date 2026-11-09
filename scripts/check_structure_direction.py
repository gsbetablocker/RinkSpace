"""Real saved-state directional checks of new surface and anatomical blocks."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.programme_surface import ProgrammeSurfaceFactor
from calibration.program_tracking.body_tracking.programme_joint_limits import ProgrammeJointLimits

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','prepared','parameter','probe','policy','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        saved=load(a.parameter);root,camera,info,doc,_=load_reference(a.project,a.prepared)
        ref=absolute_reference(root,curve_from_packet(saved),root=a.prepared);ref.relative_skate_curve=shoe_curve_from_packet(saved)
        import copy
        ref.binding=copy.copy(root.binding);ref.binding.reference=ref;ref.binding.x=saved['x'][:17];ref.binding._packet=None;ref.binding._parameter_key=None
        state=SequenceState(ref,(ref.times[0],ref.times[-1]))
        p=SimpleNamespace(reference=ref,body=SimpleNamespace(state=state),size=state.size+17)
        packet=load(a.probe);packet['times']=ref.times
        factors=[ProgrammeSurfaceFactor(p,doc,packet),ProgrammeJointLimits(p,json.loads(a.policy.read_text('utf8')),np.zeros(state.size))]
        rng=np.random.default_rng(1108);x=np.zeros(state.size);results=[]
        for f in factors:
            tick=time.monotonic();value=f.evaluate(x,True)
            J=value['weighted_blocks'][0].jacobian[:,17:]
            direction=rng.normal(size=state.size);direction/=np.linalg.norm(direction)
            h=2e-6
            central=(f.evaluate(x+h*direction)['residual']-f.evaluate(x-h*direction)['residual'])/(2*h)
            exact=J@direction
            error=np.linalg.norm(exact-central)/max(np.linalg.norm(central),1e-12)
            row=dict(factor=type(f).__name__,seconds=time.monotonic()-tick,rows=len(exact),nnz=J.nnz,relative_direction_error=error,max_abs_error=float(np.max(abs(exact-central))),metrics=value['metrics'])
            if isinstance(f,ProgrammeSurfaceFactor):
                roots=(np.arange(state.body_size)%69)<6
                row['root_J_max']=float(np.max(abs(J[:,:state.body_size][:,roots].data),initial=0))
            else:f.export(x,a.output/'baseline_anatomical_frames.npz')
            print(json.dumps(row),flush=True);results.append(row)
        (a.output/'real_direction_check.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        assert all(r['relative_direction_error']<2e-4 for r in results)
if __name__=='__main__':main()
