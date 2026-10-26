"""Inspect actual source support and native state binding for the print audit.

Read-only diagnostic: no measurements, state updates, or optimizer steps.
"""
from pathlib import Path
import argparse, importlib.util, io, json
import numpy as np
import cv2

spec = importlib.util.spec_from_file_location('prep', Path(__file__).with_name('prepare_native_s01.py'))
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.domain.print_assets import sample_instance, evaluate_print_path
from rinkspace.review.render import source_frames
from rinkspace.session.gpu_geometry import NativeSceneGPU


def run(out=None, source_ads=None, extra_frames=()):
    out = out or prep.RUN/'print_shape_audit'
    report = prep.read(out/'report.json')
    with np.load(io.BytesIO((out/'audited_state.npz').read_bytes())) as f:
        saved = {k:f[k] for k in f.files}
    with np.load(prep.RUN/'point_observations.npz') as f:
        z = {k:f[k] for k in f.files}
    assert str(saved['point_input_sha256']) == prep.digest(prep.RUN/'point_observations.npz')
    mapping = prep.read(prep.RUN/'mapping.json')
    paths = np.asarray(mapping['path_ids'])[z['path_indices']]
    scene = prep.read(prep.SCENE)
    model = SceneParameterModel(scene, prep.SCENE, optimize_scene=True)
    state = ParameterState(model.initial_state.layout, saved['scene_values'])
    core = NativeCore(prep.LIBRARY)
    native = NativeSceneGPU(prep.SCENE, core, parameter_state=state)
    camera = {**prep.camera_for_snapshot(prep.read(prep.CAMERA), saved), 'rig':saved['rig'].tolist()}
    scope = prep.read(Path(__file__).with_name('rebuild_scope.json'))
    source = {**scope['source'], **scope['source_clock']}
    all_rows = []; all_world = []; groups = []; selections = []; drawings = {}
    assets = list(read_scene_assets(scene, prep.SCENE))
    for ref, asset in assets:
        n = ref['advertisement_number']
        template = next(t for t in asset['templates'] if t['id'] == ref['template_id'])
        identity = 'print_instance/Ad'+str(n)
        selected = np.flatnonzero(np.char.startswith(paths, identity+'::'))
        bindings = template['internal_geometry']['path_bindings']
        if len(selected):
            available, counts = np.unique(z['pts'][selected], return_counts=True)
            chosen = {int(available[np.argmax(counts)]), int(available[len(available)//2])}
            if n in (8, 10):chosen.add(int(available[np.argmin(abs(available-491776))]))
            chosen.update(int(pt) for ad,pt in extra_frames if ad==n)
            if source_ads is None or n in source_ads:
                for pt in sorted(chosen):
                    rows=selected[z['pts'][selected] == pt]
                    if not len(rows):raise ValueError('Requested source frame has no observations for selected ad')
                    selections.append((n,pt,rows))
            sampled = sample_instance(asset, identity, native.scene['board'], core,
                parameter_state=state,
                unobserved_surface_extensions=[p['id'] for p in template['paths']
                    if not np.any(paths == identity+'::'+p['id'])])
            drawings[n] = sampled['paths']
        for group in template['internal_geometry']['groups']:
            path_data = []
            for pid in group['path_ids']:
                rows = np.flatnonzero(paths == identity+'::'+pid)
                if len(rows):
                    pick = rows[np.unique(np.linspace(0,len(rows)-1,min(8,len(rows))).astype(int))]
                    evaluated = evaluate_print_path(asset, identity, pid, native.scene['board'], core,
                        saved['stations'][pick], parameter_state=state, jacobian=False)
                    assert not np.any(evaluated['status'])
                    all_rows.extend(pick.tolist());all_world.extend(evaluated['xyz'])
                path_data.append(dict(path=pid, observations=len(rows),
                    exposures=len(np.unique(z['pts'][rows])),
                    station_range=[float(saved['stations'][rows].min()),float(saved['stations'][rows].max())]
                        if len(rows) else None))
            value = state[group['parameter_block_id']]
            record = dict(ad=n,block=group['parameter_block_id'],paths=path_data,
                parameterization=group['parameterization'],values=value.tolist())
            if len(value)==10:
                record['axis_scale']=np.exp(value[2:4]).tolist()
                weights=[]
                for start in (4,7):
                    v=np.r_[value[start:start+3],0.];v=np.exp(v-v.max());weights.append(v/v.sum())
                record['axis_interval_fractions']=np.asarray(weights).tolist()
            groups.append(record)
    rows = np.asarray(all_rows,int);world=np.asarray(all_world)
    cpu = core.project_motion(saved['rig'],saved['controls'],camera['first_control_time'],
        camera['control_dt'],1080,(z['pts'][rows]-camera['reference_pts'])/12800,world)
    with native.resident(camera=camera,controls=saved['controls'],path_ids=paths[rows].tolist(),
            stations=saved['stations'][rows],pts=z['pts'][rows],observed=z['uv'][rows]) as resident:
        gpu=resident.evaluate('cuda')
    assert not np.any(cpu['status']) and not np.any(gpu['status'])
    binding_error=float(np.max(np.linalg.norm(cpu['uv']-gpu['values'][:,3:5],axis=1)))
    print(json.dumps(dict(native_CPU_GPU_checked_rows=len(rows),max_pixel_difference=binding_error)),flush=True)
    panels=[];projection_error=0.
    for n,pt,rows in selections:
        _,frame=next(source_frames(source['registered_path'],source,[pt]))
        active_paths=sorted(set(paths[rows]));palette={p:tuple(map(int,cv2.cvtColor(
            np.uint8([[[int(179*i/max(1,len(active_paths))),220,235]]]),cv2.COLOR_HSV2BGR)[0,0]))
            for i,p in enumerate(active_paths)}
        measured=frame.copy();fitted=frame.copy()
        for i in rows:
            xy=tuple(np.rint(z['uv'][i]).astype(int));cv2.circle(measured,xy,3,palette[paths[i]],1,cv2.LINE_AA)
        coords=[]
        for path in drawings[n]:
            xyz=np.asarray(path['xyz']);time=(pt-camera['reference_pts'])/12800
            prediction=core.project_motion(saved['rig'],saved['controls'],camera['first_control_time'],camera['control_dt'],1080,time,xyz)
            good=prediction['status']==0
            if not np.all(good):continue
            uv=prediction['uv'];coords.append(uv)
            full='print_instance/Ad'+str(n)+'::'+path['id'];color=palette.get(full,(60,255,70))
            cv2.polylines(fitted,[np.rint(np.clip(uv,-1e6,1e6)).astype(np.int32)],bool(path.get('closed')),color,1,cv2.LINE_AA)
        # Test the renderer's flattened XYZ/index path against direct native projection.
        xyz=np.vstack([np.asarray(p['xyz']) for p in drawings[n]])
        pick=np.unique(np.linspace(0,len(xyz)-1,min(129,len(xyz))).astype(int));xyz=xyz[pick]
        with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
                options=native.scene['shared_geometry_options'],rig=saved['rig'],controls=saved['controls'],
                first_control_time=camera['first_control_time'],control_dt=camera['control_dt'],image_height=1080,
                codes=np.zeros(len(xyz),np.int32),stations=np.linspace(0,1,len(xyz)),
                times=np.full(len(xyz),(pt-camera['reference_pts'])/12800),observed=np.zeros((len(xyz),2))) as resident:
            resident.configure_geometry([dict(kind=4,vertices=xyz,parameters=[])]);flattened=resident.evaluate('cuda')
        direct=core.project_motion(saved['rig'],saved['controls'],camera['first_control_time'],camera['control_dt'],
            1080,(pt-camera['reference_pts'])/12800,xyz)
        ok=(flattened['status']==0)&(direct['status']==0)
        if np.any(ok):projection_error=max(projection_error,float(np.max(np.linalg.norm(
            flattened['values'][ok,3:5]-direct['uv'][ok],axis=1))))
        q=z['uv'][rows];lo=np.maximum(np.floor(q.min(0)-[30,45]).astype(int),[0,0]);hi=np.minimum(np.ceil(q.max(0)+[30,45]).astype(int),[1920,1080])
        x0,y0=lo;x1,y1=hi
        crops=[v[y0:y1,x0:x1] for v in (frame,measured,fitted)]
        images=[]
        for label,crop in zip(('Original source','Original observed pixels; color = bound native path','Current native projection; same path colors'),crops):
            header=np.full((28,crop.shape[1],3),25,np.uint8)
            cv2.putText(header,f'Ad{n} | {pt/12800:.3f}s PTS {pt} | '+label,(6,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
            images.extend([header,crop])
        name=out/f'Ad{n:02d}_source_{pt}.png';cv2.imwrite(str(name),np.vstack(images))
        panels.append(dict(ad=n,pts=pt,path=str(name),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],
            observations=[dict(row=int(i),id=str(z['observation_ids'][i]),path=str(paths[i]),uv=z['uv'][i].tolist()) for i in rows]))
        print(json.dumps(dict(ad=n,pts=pt,source_points=len(rows),image=str(name))),flush=True)
    prep.write(out/'support.json',dict(snapshot_sha256=report['snapshot_sha256'],
        input_sha256=prep.digest(prep.RUN/'point_observations.npz'),groups=groups,source_panels=panels,
        native_material_vs_resident_check=dict(rows=len(all_rows),max_px=binding_error),
        renderer_flattened_vs_direct_max_px=projection_error,changed_parameters=False,changed_observations=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--scene-path',type=Path);parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--source-ads',type=int,nargs='+');parser.add_argument('--extra-frame',type=int,nargs=2,action='append',default=[])
    a=parser.parse_args()
    if a.scene_path is not None:prep.SCENE=a.scene_path.resolve(strict=True)
    if a.run_dir is not None:prep.RUN=a.run_dir.resolve(strict=True)
    run(a.audit_dir,a.source_ads,a.extra_frame)
