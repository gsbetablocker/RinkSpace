"""Evaluate measured source contours with freshly associated model visibility."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.ordered_native_skate import OrderedNativeSkateFactors
from calibration.program_tracking.body_tracking.programme_skate_replay import write
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference


def main():
    p=ArgumentParser()
    for k in ('project','prepared','source','parent','candidate','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    ref,camera,_,doc,_=load_reference(a.project,a.prepared,a.source)
    parent=load(a.parent)
    new=load(a.candidate)
    ref.absolute_curve=curve_from_packet(parent,seed=True)
    ref.relative_skate_curve=shoe_curve_from_packet(parent,seed=True)
    state=SequenceState(ref,tuple(ref.times[[0,-1]]))
    problem=SimpleNamespace(body=SimpleNamespace(state=state),reference=ref,binding=ref.binding,camera=camera,size=len(parent['x']))
    scopes=['runner_lower_edge','boot_shell','outsole','front_mount','rear_mount']
    rows=[r for r in doc['programme_skate_evidence']['ordered_contours']
          if 76<=r['pts']/12800<=95 and r['semantic'] in scopes and r['role'] in ('fit','check')]
    f=OrderedNativeSkateFactors(problem,rows)
    obs={}
    for label,packet in [('old',parent),('candidate',new)]:
        x=packet['x'][17:]
        f.associate(x)
        value=f.evaluate(x,False)['metrics'][f.name]
        obs[label]=[dict(id=v['id'],pts=v['pts'],foot=v['foot'],semantic=v['semantic'],
            role=v['role'],radius_px=v['radius_px'],status=v['status'],
            visible_path_vertices=len(v.get('model_uv',[])),
            p95_px=float(np.percentile(v['distance_px'],95)) if 'distance_px'in v else None) for v in value]
    write(a.output,obs)


if __name__=='__main__':
    main()
