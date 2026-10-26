"""Initialize nuisance stations on named current native curves; never fit the camera."""
from pathlib import Path
import importlib.util,time,json,collections,argparse
import numpy as np
spec=importlib.util.spec_from_file_location("s01prep",Path(__file__).with_name("prepare_native_s01.py"));prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.domain.parameters import SceneParameterModel,ParameterState

def initialize(state_file=None):
    core=NativeCore(prep.LIBRARY);native=NativeSceneGPU(prep.SCENE,core);camera=prep.read(prep.CAMERA)
    controls=np.load(prep.CAMERA.parent/camera['controls_ref']);mapping=prep.read(prep.RUN/'mapping.json');z=dict(np.load(prep.RUN/'point_observations.npz'))
    snapshot=None
    if state_file:
        snapshot=dict(np.load(state_file));model=SceneParameterModel(prep.read(prep.SCENE),prep.SCENE,optimize_scene=True)
        if str(snapshot.get('point_input_sha256',''))!=prep.digest(prep.RUN/'point_observations.npz'):raise ValueError('Initial native state has different point input')
        native=NativeSceneGPU(prep.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,snapshot['scene_values']))
        camera={**camera,'rig':snapshot['rig'].tolist()};controls=snapshot['controls']
    station=z['stations'].copy() if snapshot is None else snapshot['stations'].copy();groups=collections.defaultdict(list)
    for row,(pts,path,kind) in enumerate(zip(z['pts'],z['path_indices'],z['kind'])):
        if kind==1:groups[(int(pts),int(path))].append(row)
    samples={}
    for _,pathid in groups:
        if pathid in samples:continue
        path=native.paths[mapping['path_ids'][pathid]]
        if path['kind']==7:s=np.linspace(0,core.board_perimeter(native.scene['board']),1025)
        elif path['kind']==8:s=np.unique(np.r_[path['knots'],np.linspace(0,1,129)])
        else:s=np.linspace(0,1,257 if path['kind'] in (1,6) else 129)
        samples[pathid]=s
    batch=[];batch_count=0;done=0;invalid_groups=[];distance=np.full(len(station),np.nan);start=time.perf_counter()
    def process(items):
        nonlocal done
        n=sum(len(samples[key[1]]) for key,rows in items);paths=[];ss=[];pts=[];ranges=[];offset=0
        for key,rows in items:
            grid=samples[key[1]];paths.extend([mapping['path_ids'][key[1]]]*len(grid));ss.extend(grid);pts.extend([key[0]]*len(grid));ranges.append((offset,offset+len(grid),rows));offset+=len(grid)
        with native.resident(camera=camera,controls=controls,path_ids=paths,stations=np.asarray(ss),pts=np.asarray(pts,np.int64),observed=np.zeros((n,2))) as resident:
            projected=resident.evaluate('cuda')
        for begin,end,rows in ranges:
            values=projected['values'][begin:end,3:5];status=projected['status'][begin:end];grid=np.asarray(ss[begin:end]);valid=(status[:-1]==0)&(status[1:]==0)
            if not np.any(valid):invalid_groups.append(dict(pts=int(z['pts'][rows[0]]),path=mapping['path_ids'][z['path_indices'][rows[0]]],rows=rows));continue
            left=values[:-1][valid];delta=np.diff(values,axis=0)[valid];length=np.sum(delta*delta,axis=1);lo=grid[:-1][valid];hi=grid[1:][valid]
            # Only station seeds change. Native path membership and raw UV/PTS do not.
            for k in range(0,len(rows),128):
                rr=np.asarray(rows[k:k+128],int);d=z['uv'][rr,None,:]-left[None,:,:]
                t=np.clip(np.einsum('ijk,jk->ij',d,delta)/np.maximum(length,1e-300),0,1)
                ds=np.sum((d-t[:,:,None]*delta)**2,axis=2);j=np.argmin(ds,axis=1);i=np.arange(len(rr))
                station[rr]=lo[j]+t[i,j]*(hi[j]-lo[j]);distance[rr]=np.sqrt(ds[i,j])
            done+=1
        print(json.dumps(dict(groups_done=done,groups_total=len(groups),seconds=time.perf_counter()-start,no_visible_seed_groups=len(invalid_groups))),flush=True)
    for key,rows in groups.items():
        if batch_count+len(samples[key[1]])>250000 and batch:process(batch);batch=[];batch_count=0
        batch.append((key,rows));batch_count+=len(samples[key[1]])
    if batch:process(batch)
    np.savez(prep.RUN/'initialized_points.npz',stations=station,seed_screen_distance=distance)
    if snapshot is not None:
        snapshot['stations']=station;np.savez(prep.RUN/'reseeded_pixels_state.npz',**snapshot)
    prep.write(prep.RUN/'station_initialization.json',dict(method='native_current_geometry_and_predicted_row_projection_named_curve_search',
        changed='nuisance_station_initial_values_only',camera_sha256=prep.digest(prep.CAMERA),scene_revision=native.scene_revision,
        dataset_native_parameters_overwritten=False,numeric_state_source=str(state_file) if state_file else None,unresolved_groups=invalid_groups,curve_rows=int(np.count_nonzero(z['kind']==1)),
        seed_error_quantiles_px=np.nanpercentile(distance,[0,50,90,95,99,100]).tolist()))
    with native.resident(camera=camera,controls=controls,path_ids=[mapping['path_ids'][i] for i in z['path_indices']],stations=station,pts=z['pts'],observed=z['uv']) as resident:
        result=resident.evaluate('cuda');np.savez(prep.RUN/'initialized_pixel_evaluation.npz',values=result['values'],status=result['status'])
        status=result['status'];err=np.linalg.norm(result['values'][:,9:11],axis=1)
        report=dict(invalid=int(np.count_nonzero(status)),status=dict(zip(*[x.tolist() for x in np.unique(status,return_counts=True)])),errors=np.nanpercentile(err,[0,50,90,95,99,100]).tolist(),paths=[])
        for pid,path in enumerate(mapping['path_ids']):
            rr=np.flatnonzero(z['path_indices']==pid)
            if len(rr):report['paths'].append(dict(path=path,rows=len(rr),invalid=int(np.count_nonzero(status[rr])),p50=float(np.nanmedian(err[rr])),p95=float(np.nanpercentile(err[rr],95))))
        prep.write(prep.RUN/'initialized_evaluation.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='paths'}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path);args=parser.parse_args();initialize(args.state)
