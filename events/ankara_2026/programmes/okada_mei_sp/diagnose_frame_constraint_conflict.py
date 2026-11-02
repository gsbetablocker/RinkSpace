"""Local source-normal sensitivity of a frozen frame, not a production fit.

All original frame observations are retained. This distinguishes instantaneous
pan/tilt/zoom support from four existing lens directions without altering any
state, source UV, or deciding source validity by residual.
"""
from pathlib import Path
import argparse,json
import numpy as np
import prepare_native_multicamera as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import compile_geometry
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.domain.print_assets import evaluate_print_path


def summary(r):
    a=np.asarray(r);return dict(rows=len(a),rms=float(np.sqrt(np.mean(a*a))),median_abs=float(np.median(abs(a))),max_abs=float(np.max(abs(a))))


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    m=prepared.read(a.prepared/'joint_manifest.json');mapping=prepared.read(a.prepared/'mapping.json')
    state=prepared.npz(a.state);points=prepared.npz(a.prepared/'point_observations.npz')
    scene_path=Path(m['scene_path']);raw=prepared.read(scene_path);model=SceneParameterModel(raw,scene_path,optimize_scene=True)
    prepared.validate_snapshot(state,model.initial_state.layout,mapping,m,points)
    if prepared.digest(a.native_library)!=str(state['native_library_provenance_sha256']):raise ValueError('DLL differs')
    core=NativeCore(a.native_library);parameter=ParameterState(model.initial_state.layout,state['scene_values'])
    scene,paths=compile_geometry(raw,scene_path,core,parameter_state=parameter)
    print_assets={identity:asset for ref,asset in read_scene_assets(raw,scene_path) for identity in ref.get('instance_ids',[ref.get('instance_id')])}
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);reports=[]
    for pts in a.pts:
        shot=next(s for s in m['shots'] if s['first_pts']<=pts<=s['last_pts'])
        ix=np.flatnonzero(points['pts']==pts);names=np.asarray(mapping['path_ids'])[points['path_indices'][ix]]
        observed=points['uv'][ix];station=state['stations'][ix];groups={name:np.flatnonzero(names==name) for name in set(names)}
        def world(stations):
            xyz=np.empty((len(ix),3))
            for name,rows in groups.items():
                p=paths[name]
                if '::' in name:
                    instance,path=name.split('::',1)
                    xyz[rows]=evaluate_print_path(print_assets[instance],instance,path,scene['board'],core,stations[rows],parameter_state=parameter,jacobian=False)['xyz']
                else:xyz[rows]=core.sample_path(scene['board'],p['kind'],p['parameters'],p['vertices'],stations[rows])
            return xyz
        xyz=world(station);rig=state['rig'][shot['camera_index']];controls=state['controls']
        clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
        def project(coords,physical=rig,motion=controls):
            r=core.project_motion(physical,motion,timeline.first,timeline.dt,1080,clock,coords)
            if np.any(r['status']):raise ValueError('Native sensitivity probe invalid')
            return r['uv']
        prediction=project(xyz);residual=prediction-observed
        left=station.copy();right=station.copy();h=1e-6
        for name,rows in groups.items():
            support=paths[name].get('support',[0.,1.]);lo,hi=(-np.inf,np.inf) if support is None else support
            left[rows]=np.maximum(lo,station[rows]-h);right[rows]=np.minimum(hi,station[rows]+h)
        tangent=project(world(right))-project(world(left));length=np.linalg.norm(tangent,axis=1)
        normals=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(length[:,None],1e-30)
        fixed=(points['kind'][ix]==0)|(length<1e-10)
        if fixed.any():raise ValueError('This source-normal diagnostic requires ordinary finite curves')
        r=np.einsum('ni,ni->n',residual,normals);columns=[]
        for axis in range(3):
            high=controls.copy();low=controls.copy();high[:,axis]+=h;low[:,axis]-=h
            columns.append(np.einsum('ni,ni->n',(project(xyz,motion=high)-project(xyz,motion=low))/(2*h),normals))
        for index in (31,33,35,36,15,16,17,20,19):
            high=rig.copy();low=rig.copy();high[index]+=h;low[index]-=h
            columns.append(np.einsum('ni,ni->n',(project(xyz,physical=high)-project(xyz,physical=low))/(2*h),normals))
        J=np.stack(columns,axis=1);stages={};predictions={}
        for label,selected in [('instantaneous_motion',[0,1,2]),('instantaneous_motion_and_lens',list(range(7))),
                               ('motion_and_centre',[0,1,2,7,8,9]),('motion_lens_and_centre',list(range(10))),
                               ('motion_and_forward',[0,1,2,10]),('motion_lens_and_forward',list(range(7))+[10]),
                               ('motion_lens_forward_vertical',list(range(7))+[10,11])]:
            js=J[:,selected];scale=np.linalg.norm(js,axis=0)
            delta,_,rank,singular=np.linalg.lstsq(js/scale,-r,rcond=None);delta/=scale
            after=r+js@delta;predictions[label]=after
            stages[label]=dict(delta=delta.tolist(),rank=int(rank),before=summary(r),linear_after=summary(after))
        family=[]
        for name,rows in sorted(groups.items()):
            family.append(dict(path=name,before=summary(r[rows]),after_motion=summary(predictions['instantaneous_motion'][rows]),after_lens=summary(predictions['instantaneous_motion_and_lens'][rows]),after_centre=summary(predictions['motion_and_centre'][rows]),after_full=summary(predictions['motion_lens_and_centre'][rows]),after_forward=summary(predictions['motion_lens_and_forward'][rows])))
        focal=core.evaluate_motion_camera(rig,controls,timeline.first,timeline.dt,[clock])['fx'][0]
        reports.append(dict(shot=shot['id'],pts=pts,focal_pixels=float(focal),stages=stages,paths=family,
            rows=ix.tolist(),observed=observed.tolist(),predicted=prediction.tolist(),normals=normals.tolist(),normal_residual=r.tolist()))
    prepared.write(a.output,dict(schema='rinkspace.frame_constraint_conflict_linear_diagnostic.v1',state=dict(path=str(a.state),sha256=prepared.digest(a.state)),
        parameters=['pan','tilt','log_f','k1_at_frame','k2_at_frame','p1_at_frame','p2_at_frame','pivot_x','pivot_y','pivot_z','lever_z','lever_y'],frames=reports,
        production_fit_started=False,state_modified=False,source_UV_modified=False,
        limitations=['First-order sensitivity only; local lens values are not a cross-zoom or physical-camera model.',
                    'Curve tangent is eliminated locally, including endpoint samples; no constrained KKT claim.',
                    'A residual does not decide whether the source identity is correct.']))
    print(json.dumps([dict(pts=r['pts'],focal=r['focal_pixels'],stages=r['stages']) for r in reports]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','native-library','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pts',type=int,nargs='+',required=True)
    main(p.parse_args())
