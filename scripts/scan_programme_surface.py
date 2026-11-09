"""Scan actual saved surfaces on the existing complete native source clock."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState,shoe_curve_from_packet
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.programme_surface import scan_surface

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('project','prepared','parameter','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-evidence',type=Path)
    a=p.parse_args();packet=load(a.parameter)
    root,camera,info,doc,_=load_reference(a.project,a.prepared,a.source_evidence)
    reference=absolute_reference(root,curve_from_packet(packet),root=a.prepared)
    reference.relative_skate_curve=shoe_curve_from_packet(packet)
    reference.binding=copy.copy(root.binding);reference.binding.reference=reference
    reference.binding.x=packet['x'][:17].copy();reference.binding._packet=None;reference.binding._parameter_key=None
    state=SequenceState(reference,(reference.times[0],reference.times[-1]))
    started=time.monotonic()
    def emit(event,**values):print(json.dumps(dict(event=event,elapsed_s=time.monotonic()-started,**values)),flush=True)
    with threadpool_limits(limits=1):
        _,report=scan_surface(reference,state,np.zeros(state.size),doc,a.output,emit=emit)
    report.update(parameter=str(a.parameter),parameter_sha256=digest(a.parameter),camera_sha256=info.get('state_sha256'),
        input_sha256=digest(a.prepared/'input.npz'),source_document_sha256=digest(a.source_evidence if a.source_evidence is not None else a.prepared/'source_evidence.json'))
    (a.output/'surface_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    emit('complete',report=str(a.output/'surface_scan.json'))

if __name__=='__main__':main()
