"""Exact-source native-state overlay; no camera smoothing or source interpolation."""
from pathlib import Path
from fractions import Fraction
import importlib.util,json,argparse,time,collections,io
import numpy as np
import cv2,av
spec=importlib.util.spec_from_file_location('s01prep',Path(__file__).with_name('prepare_native_s01.py'));prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU,complete_camera_rig
from rinkspace.review.scene import sample_review_scene
from rinkspace.review.render import source_frames,validate_video
from rinkspace.review.draw import text,COLORS


def render(snapshot_path,phase,label,stills=None,source_range_pts=None):
    started=time.monotonic();raw_scene=prep.read(prep.SCENE);model=SceneParameterModel(raw_scene,prep.SCENE,optimize_scene=True)
    snapshot_blob=snapshot_path.read_bytes();snap=dict(np.load(io.BytesIO(snapshot_blob),allow_pickle=False))
    z,mapping,_=prep.load_pixel_input(prep.RUN)
    camera=prep.validate_snapshot_identity(snap,model.initial_state.layout,prep.read(prep.CAMERA),mapping,points=z,
        point_input_sha256=prep.digest(prep.RUN/'point_observations.npz'))
    state=ParameterState(model.initial_state.layout,snap['scene_values']);core=NativeCore(prep.LIBRARY)
    native=NativeSceneGPU(prep.SCENE,core,parameter_state=state)
    camera={**camera,'rig':snap['rig'].tolist()};camera['rig']=complete_camera_rig(camera).tolist();controls=snap['controls']
    scope=prep.read(Path(__file__).with_name('rebuild_scope.json'));source={**scope['source'],**scope['source_clock']};tb=Fraction(1,12800);expected=prep.selected_s01_exposures()
    if not np.array_equal(np.unique(z['pts']),expected):raise ValueError('Incomplete S01 source exposure coverage')
    if prep.digest(source['registered_path'])!=source['sha256']:raise ValueError('Source video identity changed')
    ext=camera.get('inactive_extension_proofs',[])
    if ext and len(camera['rig'])==38 and np.min(controls[:,2])<ext[0]['activation_log_f_native']:raise ValueError('Cannot render outside verified native optics extension domain')
    observed_paths={mapping['path_ids'][i] for i in z['path_indices']}
    samples=sample_review_scene(raw_scene,core,prep.SCENE,parameter_state=state,observed_path_ids=observed_paths)
    # Increase representation density only; these are never fit observations.
    chunks=[];cuts=[];offset=0
    for cut in samples['cuts']:
        path=native.paths.get(cut['id']);points=samples['xyz'][cut['begin']:cut['end']]
        if path is not None and cut['layer']!='advertisement':
            if path['kind']==7:ss=np.linspace(0,core.board_perimeter(native.scene['board']),4097)
            else:ss=np.linspace(*(path.get('support') or [0.,1.]),1025 if path['kind'] in (1,6) else 129)
            points=np.asarray(core.sample_path(native.scene['board'],path['kind'],path['parameters'],path.get('vertices',[]),ss))
        chunks.append(points);cuts.append({**cut,'begin':offset,'end':offset+len(points)});offset+=len(points)
    world=np.vstack(chunks);out=prep.RUN/'review';out.mkdir(exist_ok=True)
    times=(expected-camera['reference_pts'])/12800;poses=core.evaluate_motion_camera(camera['rig'],controls,camera['first_control_time'],camera['control_dt'],times)
    if np.any(poses['status']):raise ValueError('Camera pose outside source time range')
    with native.resident(camera=camera,controls=controls,path_ids=[mapping['path_ids'][i] for i in z['path_indices']],stations=snap['stations'],pts=z['pts'],observed=z['uv']) as p:
        obs=p.evaluate('cuda')
    by_pts=collections.defaultdict(list)
    for i,pts in enumerate(z['pts']):by_pts[int(pts)].append(i)
    errors=np.linalg.norm(obs['values'][:,9:11],axis=1)
    side_width=640;side=np.full((1080,side_width,3),22,np.uint8);scale=540/(2*native.scene['board'][0]);center=np.array([320,475])
    def top(xyz):return np.rint(np.asarray(xyz)[...,:2]*[scale,-scale]+center).astype(np.int32)
    def color(cut):return (220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
    for cut in cuts:cv2.polylines(side,[top(world[cut['begin']:cut['end']])],bool(cut.get('closed')),color(cut),1,cv2.LINE_AA)
    text(side,'CURRENT NATIVE SCENE / TOP VIEW',(18,28),.57)
    text(side,f"{len(cuts)} paths | {len(native.unlocalized)} unlocalized ads",(18,57),.48)
    text(side,'Yellow ice | Blue boards | Green print',(18,85),.44)
    text(side,'Length unit L; source exposure midpoint pose',(18,109),.43)
    if samples['unobserved_print_surface_extensions']:
        text(side,'Magenta: unobserved print outside board; unfitted',(18,133),.40,(220,60,240))
    pivot=np.array(camera['rig'][15:18]);cv2.circle(side,tuple(top(pivot)),5,(100,210,255),2,cv2.LINE_AA)
    text(side,'TRIPOD',top(pivot)+[9,15],.4,(100,210,255))
    selection=expected if stills is None else np.asarray(stills,np.int64)
    if source_range_pts is not None:
        if stills is not None:raise ValueError('Select either stills or a continuous source interval')
        selection=selection[(selection>=source_range_pts[0])&(selection<=source_range_pts[1])]
        if not len(selection):raise ValueError('Empty continuous source interval')
    if not set(selection.tolist())<=set(expected.tolist()):raise ValueError('Requested review PTS outside full S01')
    stream=output=None;video=out/(label+'_s01.mp4');records=[]
    if stills is None:
        output=av.open(str(video),'w',options={'movflags':'+faststart','video_track_timescale':'12800'});stream=output.add_stream('libx264',rate=50);stream.width=2560;stream.height=1144;stream.pix_fmt='yuv420p';stream.time_base=tb;stream.codec_context.time_base=tb;stream.options={'crf':'17','preset':'fast'}
    frames=iter(source_frames(source['registered_path'],source,selection)) if stills is None else None
    with (out/(label+'_render.jsonl')).open('w',encoding='utf-8',buffering=1) as log:
        try:
            for index,pts in enumerate(selection):
                fi=int((pts-expected[0])//256);_,raw=next(source_frames(source['registered_path'],source,[int(pts)])) if stills is not None else next(frames)
                n=len(world)
                with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],options=native.scene['shared_geometry_options'],rig=camera['rig'],controls=controls,first_control_time=camera['first_control_time'],control_dt=camera['control_dt'],image_height=1080,codes=np.zeros(n,np.int32),stations=np.linspace(0,1,n),times=np.full(n,times[fi]),observed=np.zeros((n,2))) as p:
                    p.configure_geometry([dict(kind=4,vertices=world,parameters=[])]);prediction=p.evaluate('cuda')
                uv=prediction['values'][:,3:5];good=(prediction['status']==0)&np.isfinite(uv).all(1);pixels=np.rint(np.clip(np.nan_to_num(uv),-1e7,1e7)).astype(np.int32);picture=raw.copy()
                for cut in cuts:
                    begin,end=cut['begin'],cut['end'];transitions=np.flatnonzero(np.diff(np.r_[False,good[begin:end],False])).reshape(-1,2)
                    for lo,hi in transitions:
                        if hi-lo>1:cv2.polylines(picture,[pixels[begin+lo:begin+hi]],bool(cut.get('closed') and lo==0 and hi==end-begin),color(cut),1,cv2.LINE_AA)
                rows=np.asarray(by_pts[int(pts)],int)
                for row in rows:
                    measured=tuple(np.rint(z['uv'][row]).astype(int));cv2.circle(picture,measured,2,(0,0,255),-1,cv2.LINE_AA)
                    if obs['status'][row]==0:
                        fitted=tuple(np.rint(obs['values'][row,3:5]).astype(int));cv2.line(picture,measured,fitted,(30,130,240),1,cv2.LINE_AA);cv2.drawMarker(picture,fitted,(60,255,70),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
                panel=side.copy();C=poses['centre'][fi];direction=poses['rotation'][fi,2];at=top(C);look=direction[:2];look=look/max(np.linalg.norm(look),1e-12)
                cv2.arrowedLine(panel,tuple(at),tuple(np.rint(at+look*[95,-95]).astype(int)),(240,80,255),2,cv2.LINE_AA,tipLength=.12);cv2.circle(panel,tuple(at),4,(240,80,255),-1,cv2.LINE_AA)
                azimuth=np.degrees(np.arctan2(direction[1],direction[0]));elevation=np.degrees(np.arcsin(np.clip(direction[2],-1,1)))
                lines=[f'S01 main | frame {fi+1}/{len(expected)}',f'Source {pts/12800:.3f}s | PTS {pts}',f'Pan {np.degrees(poses["pan"][fi]):+.3f} deg',f'Tilt {np.degrees(poses["tilt"][fi]):+.3f} deg',f'View azimuth {azimuth:+.3f} deg',f'View elevation {elevation:+.3f} deg',f'Focal length {poses["fx"][fi]:.2f} px',f'Optical C: {C[0]:+.6f}, {C[1]:+.6f}, {C[2]:+.6f} L',f'Pivot: {pivot[0]:+.6f}, {pivot[1]:+.6f}, {pivot[2]:+.6f} L',f'Active source points {len(rows)} | invalid {np.count_nonzero(obs["status"][rows])}',f'Pixel error median {np.nanmedian(errors[rows]):.2f} | p95 {np.nanpercentile(errors[rows],95):.2f}']
                for j,line in enumerate(lines):text(panel,line,(16,760+24*j),.43)
                canvas=np.concatenate((picture,panel),axis=1);header=np.full((64,2560,3),15,np.uint8);text(header,f'S01 NATIVE GPU JOINT FIT | {phase.upper()} | {label}',(16,26),.6,(60,220,255));text(header,'All retained source UV: red dots | Native fitted locations: green crosses | Original source cadence: 50 fps',(16,52),.48);canvas=np.concatenate((header,canvas),axis=0)
                record=dict(pts=int(pts),frame=fi+1,source_rows=z['source_rows'][rows].tolist(),median_px=float(np.nanmedian(errors[rows])),p95_px=float(np.nanpercentile(errors[rows],95)),pose=poses['values'][fi].tolist());records.append(record)
                if stills is not None or fi in (0,312,624,936,1247):cv2.imwrite(str(out/f'{label}_{int(pts)}.png'),canvas)
                if output is not None:
                    frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=int(pts-selection[0]);frame.time_base=tb
                    for packet in stream.encode(frame):output.mux(packet)
                if index%50==0 or index==len(selection)-1:
                    progress=dict(frame=fi+1,total=len(expected),elapsed=time.monotonic()-started);log.write(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
            if output is not None:
                for packet in stream.encode():output.mux(packet)
        finally:
            if output is not None:output.close()
    verification=validate_video(video,selection,tb) if stills is None else None
    import hashlib
    prep.write(out/(label+'_review.json'),dict(snapshot=str(snapshot_path),snapshot_sha256=hashlib.sha256(snapshot_blob).hexdigest(),native_library=str(prep.LIBRARY),native_library_sha256=prep.digest(prep.LIBRARY),phase=phase,source_sha256=source['sha256'],source_first_seconds=float(selection[0]/12800),source_last_seconds=float(selection[-1]/12800),frames=len(selection),video=str(video) if stills is None else None,verification=verification,points_total=len(z['uv']),invalid_points=int(np.count_nonzero(obs['status'])),all_rows=records,native_parameters_postsmoothed=False,unlocalized_ads=native.unlocalized,unobserved_print_surface_extensions=samples['unobserved_print_surface_extensions'],complete_path_count=len(cuts),visual_acceptance='pending'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path);parser.add_argument('--phase',choices=['pixels'],default='pixels');parser.add_argument('--label',default='fit');parser.add_argument('--stills',type=int,nargs='+')
    parser.add_argument('--scene-path',type=Path);parser.add_argument('--run-dir',type=Path);parser.add_argument('--native-library',type=Path);parser.add_argument('--source-range-pts',type=int,nargs=2);args=parser.parse_args()
    if args.scene_path is not None:prep.SCENE=args.scene_path.resolve(strict=True)
    if args.run_dir is not None:prep.RUN=args.run_dir.resolve(strict=True)
    if args.native_library is not None:prep.LIBRARY=args.native_library.resolve(strict=True)
    render(args.snapshot or prep.RUN/'accepted_pixels_latest.npz',args.phase,args.label,args.stills,args.source_range_pts)
