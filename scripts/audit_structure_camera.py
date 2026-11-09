"""Inspect saved whole-programme structure without neural inference or realignment."""
from pathlib import Path
import argparse, json, time, sys
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference
from calibration.program_tracking.body_tracking.rebuild_prepare import load,digest
from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference,curve_from_packet
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.programme_anatomy import anatomy_geometry,hinge_angles

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('project','prepared','parameter','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scan',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def emit(event,**data):print(json.dumps(dict(event=event,elapsed_s=time.monotonic()-started,**data)),flush=True)
    with threadpool_limits(limits=1):
        packet=load(a.parameter);root,camera,info,doc,_=load_reference(a.project,a.prepared)
        ref=absolute_reference(root,curve_from_packet(packet),root=a.prepared)
        ref.relative_skate_curve=shoe_curve_from_packet(packet)
        import copy
        ref.binding=copy.copy(root.binding);ref.binding.reference=ref;ref.binding.x=packet['x'][:17].copy();ref.binding._packet=None;ref.binding._parameter_key=None
        s=fk_state(ref,ref.at(ref.times),np.zeros((len(ref.times),69)))
        np.savez_compressed(a.output/'baseline_fk.npz',source_pts=ref.p['pts'],source_seconds=ref.p['pts']/12800.,local_R=s['local_R'],Q=s['Q'],P=s['P'],frames=s['frames'],native_world=s['native_world'],skate_R=s['R'],skate_O=s['O'],rest_J=ref.body.J,parents=ref.body.parents,body_scale=ref.body.scale,length_ratio=ref.binding.ratio)
        J=ref.body.J
        axes=[]
        for knee,ankle,toe in ((4,7,10),(5,8,11)):axes.append(np.cross(J[ankle]-J[knee],-(J[toe]-J[ankle])))
        for elbow,wrist in ((18,20),(19,21)):axes.append(np.cross(J[wrist]-J[elbow],[0.,0.,1.]))
        axes=np.array(axes);axes/=np.linalg.norm(axes,axis=-1,keepdims=True)
        twist,swing=hinge_angles(s['local_R'][:,np.array([4,5,18,19])-1],axes)
        unwrapped=np.unwrap(twist,axis=0)
        frame=int(np.argmin(abs(ref.p['pts']/12800.-162.60)))
        emit('fk_exported',parameter_sha256=digest(a.parameter),camera_sha256=info['state_sha256'],exposures=len(ref.times),axes=axes.tolist(),rest_J=J[:22].tolist(),twist_min_deg=np.degrees(twist).min(0).tolist(),unwrapped_range_deg=np.degrees(np.stack([unwrapped.min(0),unwrapped.max(0)])).tolist(),knee_16260=dict(local_R=s['local_R'][frame,4].tolist(),principal_deg=float(np.degrees(twist[frame,1])),unwrapped_deg=float(np.degrees(unwrapped[frame,1])),off_axis_deg=float(np.degrees(swing[frame,1]))))
        for n in (1,32):
            indices=np.arange(frame,frame+n);state={k:v[indices] for k,v in s.items() if isinstance(v,np.ndarray) and len(v)==len(ref.times)}
            t=time.monotonic();vertices=ref.binding.surface.sample_vertices(state['native'],state['frames'],hand_local_R=state.get('hand_local_R'),double_precision=True)
            emit('posed_skin_cost',samples=n,vertices=vertices.shape,seconds=time.monotonic()-t)
        from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface,witnesses_packet
        geometry=CollisionSurface(ref,doc)
        emit('surface_layout',vertices=geometry.vertex_count,faces=len(geometry.faces),parts=geometry.part_names)
        selected=np.array([np.argmin(abs(ref.p['pts']/12800.-t)) for t in (49.96,97.82,104.56,162.60)])
        records=[]
        for i in selected:
            state={k:v[i:i+1] for k,v in s.items() if isinstance(v,np.ndarray) and len(v)==len(ref.times)}
            v=geometry.vertices(state)[0];tic=time.monotonic()
            witness,diagnostic=geometry.scan_sample(v,ref.times[i]);pairs=diagnostic.pop('pairs');grips=diagnostic.pop('grip_pairs')
            from collections import Counter
            counts=Counter(tuple(sorted((geometry.part_names[geometry.parts[a]],geometry.part_names[geometry.parts[b]]))) for a,b in pairs)
            diagnostic['pair_counts']=[dict(parts=k,count=v) for k,v in counts.most_common(12)]
            diagnostic['deepest_gap_m']=min([r[4] for r in witness],default=0.)
            emit('surface_probe',seconds=float(ref.p['pts'][i]/12800.),duration_s=time.monotonic()-tic,witnesses=len(witness),**diagnostic)
            records.extend((int(i),row) for row in witness)
        np.savez_compressed(a.output/'surface_probe.npz',**witnesses_packet(records))
        if a.scan:
            from calibration.program_tracking.body_tracking.programme_surface import scan_surface
            from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
            state=SequenceState(ref,(ref.times[0],ref.times[-1]))
            scan_surface(ref,state,np.zeros(state.size),doc,a.output,emit=emit)
        emit('complete')

if __name__=='__main__':main()
