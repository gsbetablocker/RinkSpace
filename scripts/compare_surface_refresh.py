"""Compare both accepted-step endpoints on one freshly scanned witness set."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json
import numpy as np
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.programme_surface import ProgrammeSurfaceFactor
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('project','prepared','before','after','witnesses','stage','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-evidence',type=Path)
    a=p.parse_args();before=load(a.before);after=load(a.after);packet=load(a.witnesses)
    root,camera,info,doc,_=load_reference(a.project,a.prepared,a.source_evidence)
    ref=absolute_reference(root,curve_from_packet(before,seed=True),root=a.prepared)
    ref.relative_skate_curve=shoe_curve_from_packet(before,seed=True)
    ref.binding=copy.copy(root.binding);ref.binding.reference=ref;ref.binding.x=before['x'][:17].copy()
    ref.binding._packet=None;ref.binding._parameter_key=None
    state=SequenceState(ref,(ref.times[0],ref.times[-1]));problem=SimpleNamespace(reference=ref,body=SimpleNamespace(state=state))
    cfg=json.loads(Path(doc['programme_structure']['policy']).read_text(encoding='utf8'))
    factor=ProgrammeSurfaceFactor(problem,doc,packet,scale_m=cfg['geometry']['surface_scale_m'])
    with threadpool_limits(limits=1):
        first=factor.evaluate(before['x'][17:],False);last=factor.evaluate(after['x'][17:],False)
    original=json.loads((a.stage/'initial_metrics.json').read_text(encoding='utf8'))
    final=json.loads((a.stage/'final_metrics.json').read_text(encoding='utf8'))
    selection=json.loads((a.stage/'solve/candidate.json').read_text(encoding='utf8'))
    weight=16.
    oldcost=selection['run_policy']['baseline']['joint_cost'];newcost=selection['selection']['retained_metrics']['joint_cost']
    before_common=oldcost-original['weighted_parts']['surface_intersection']+weight*first['parts']['surface_intersection']
    after_common=newcost-final['weighted_parts']['surface_intersection']+weight*last['parts']['surface_intersection']
    report=dict(schema='rinkspace.surface-association-refresh.v1',before_parameter_sha256=digest(a.before),after_parameter_sha256=digest(a.after),
        witnesses_sha256=digest(a.witnesses),association=factor.association,same_refreshed_association=True,rows=len(packet['sample']),
        surface_before=first['metrics']['surface_intersection'],surface_after=last['metrics']['surface_intersection'],
        full_objective_before_on_new_association=before_common,full_objective_after_on_new_association=after_common,
        nonincrease_on_new_association=after_common<=before_common+1e-7,
        removed_witnesses_not_credited_as_cost_descent=True,accuracy_accepted=False)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
