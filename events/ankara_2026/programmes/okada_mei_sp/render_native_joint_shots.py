"""Complete source-PTS videos from one frozen joint state; no optimization."""
from pathlib import Path
from fractions import Fraction
import argparse, collections, json, time
import av
import cv2
import numpy as np
import prepare_native_joint_shots as p
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.scene import sample_review_scene
from rinkspace.review.render import source_frames, validate_video
from rinkspace.review.draw import COLORS, text


def main(args):
    started = time.monotonic()
    root = getattr(args,'run_root',None) or p.RUN; out = args.output; out.mkdir(parents=True, exist_ok=True)
    manifest = p.read(root/'joint_manifest.json'); mapping = p.read(root/'mapping.json')
    points = p.npz(root/'point_observations.npz'); state = p.npz(args.snapshot)
    for path, sha in [(root/'point_observations.npz', manifest['point_input_sha256']),
                      (Path(manifest['scene_path']), manifest['scene_file_sha256']),
                      (Path(manifest['native_library']), manifest['native_library_sha256'])]:
        if p.digest(path) != sha: raise ValueError('Current render dependency changed: '+str(path))
    core = p.NativeCore(manifest['native_library']); raw_scene = p.read(manifest['scene_path'])
    model = p.SceneParameterModel(raw_scene, Path(manifest['scene_path']), optimize_scene=True)
    multicamera=state['rig'].ndim==2
    if multicamera:
        import prepare_native_multicamera as multi
        multi.validate_snapshot(state,model.initial_state.layout,mapping,manifest,points)
    else:p.validate_joint_snapshot(state, model.initial_state.layout, mapping, manifest, points)
    parameter_state = p.ParameterState(model.initial_state.layout, state['scene_values'])
    native = p.NativeSceneGPU(manifest['scene_path'], core, parameter_state=parameter_state)
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    camera_definition=manifest['cameras'][0]['camera_definition'] if multicamera else manifest['shared_camera_definition']
    row_cameras=np.zeros(len(points['pts']),np.int32)
    if multicamera:
        for shot in manifest['shots']:row_cameras[shot['row_begin']:shot['row_stop']]=shot['camera_index']
    camera = {**p.read(camera_definition), 'rig':(state['rig'][0] if multicamera else state['rig']).tolist(),
              'first_control_time':timeline.first, 'control_dt':timeline.dt}
    controls = state['controls']; paths = np.asarray(mapping['path_ids'])[points['path_indices']]
    times = timeline.row_times(points['pts'])
    with native.resident(camera=camera, controls=controls, path_ids=paths.tolist(),
                         stations=state['stations'], pts=points['pts'], observed=points['uv'],
                         motion_times=times,rigs=state['rig'] if multicamera else None,camera_ids=row_cameras if multicamera else None) as resident:
        resident.configure_projection_time_support(half_seconds=manifest['max_readout']/2)
        observed = resident.evaluate('cuda')
    if observed['status'].any(): raise ValueError('Frozen state contains invalid observed projections')
    errors = np.linalg.norm(observed['values'][:,9:11], axis=1)
    samples = sample_review_scene(raw_scene, core, Path(manifest['scene_path']),
                                  parameter_state=parameter_state, observed_path_ids=set(paths))
    chunks = []; cuts = []; offset = 0
    for cut in samples['cuts']:
        path = native.paths.get(cut['id']); xyz = samples['xyz'][cut['begin']:cut['end']]
        if path is not None and cut['layer'] != 'advertisement':
            stations = (np.linspace(0, core.board_perimeter(native.scene['board']), 4097)
                        if path['kind'] == 7 else np.linspace(*(path.get('support') or [0.,1.]),
                                                             1025 if path['kind'] in (1,6) else 129))
            xyz = np.asarray(core.sample_path(native.scene['board'], path['kind'],
                                             path['parameters'], path.get('vertices',[]), stations))
        chunks.append(xyz); cuts.append({**cut,'begin':offset,'end':offset+len(xyz)}); offset += len(xyz)
    world = np.vstack(chunks); scope = p.read(p.HERE/'rebuild_scope.json')
    source = {**scope['source'], **scope['source_clock']}; tb = Fraction(1,12800)
    if p.digest(source['registered_path']) != source['sha256']: raise ValueError('Source video identity changed')
    by_pts = collections.defaultdict(list)
    for i, pts in enumerate(points['pts']): by_pts[int(pts)].append(i)
    side = np.full((1080,640,3),22,np.uint8); scale = 540/(2*native.scene['board'][0])
    def top(xyz): return np.rint(np.asarray(xyz)[...,:2]*[scale,-scale]+[320,475]).astype(np.int32)
    def color(cut): return (220,60,240) if cut.get('unobserved_surface_extension') else COLORS[cut['layer']]
    for cut in cuts:
        cv2.polylines(side,[top(world[cut['begin']:cut['end']])],bool(cut.get('closed')),color(cut),1,cv2.LINE_AA)
    text(side,'CURRENT NATIVE SCENE / TOP VIEW',(18,28),.57)
    text(side,f'{len(cuts)} paths | {len(native.unlocalized)} unlocalized ads',(18,57),.48)
    text(side,'Yellow ice | Blue boards | Green print',(18,85),.44)
    text(side,'Length unit L; source exposure midpoint pose',(18,109),.43)
    if samples['unobserved_print_surface_extensions']:
        text(side,'Magenta: unobserved print outside board; unfitted',(18,133),.40,(220,60,240))
    for i,physical_rig in enumerate(state['rig'] if multicamera else [state['rig']]):
        pivot=physical_rig[15:18];cv2.circle(side,tuple(top(pivot)),5,(100,210,255),2,cv2.LINE_AA)
        text(side,'TRIPOD '+str(i+1),top(pivot)+[9,15],.4,(100,210,255))
    reports = []
    joint_label = ' + '.join(shot['id'].upper() for shot in manifest['shots'])
    with (out/'render_progress.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for shot in manifest['shots']:
            rig=state['rig'][shot['camera_index']] if multicamera else state['rig'];pivot=rig[15:18]
            selection = np.arange(shot['first_pts'],shot['last_pts']+256,256,dtype=np.int64)
            shot_rows = np.arange(shot['row_begin'],shot['row_stop'])
            if not np.array_equal(np.unique(points['pts'][shot_rows]),selection):
                raise ValueError('Incomplete source exposure coverage: '+shot['id'])
            shot_times = (selection-shot['reference_pts'])/12800+timeline.offset(shot)
            poses = core.evaluate_motion_camera(rig,controls,timeline.first,timeline.dt,shot_times)
            if poses['status'].any(): raise ValueError('Invalid source camera pose')
            video = out/(args.label+'_'+shot['id']+'.mp4'); records = []
            key_frames = {0,len(selection)//2,len(selection)-1,
                          int((points['pts'][shot_rows[np.argmax(errors[shot_rows])]]-selection[0])//256)}
            with av.open(str(video),'w',options={'movflags':'+faststart','video_track_timescale':'12800'}) as output:
                stream = output.add_stream('libx264',rate=50); stream.width=2560; stream.height=1144
                stream.pix_fmt='yuv420p'; stream.time_base=tb; stream.codec_context.time_base=tb
                stream.options={'crf':'17','preset':'fast'}
                for fi,(pts,raw) in enumerate(source_frames(source['registered_path'],source,selection)):
                    n=len(world)
                    with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
                            options=native.scene['shared_geometry_options'],rig=rig,controls=controls,
                            first_control_time=timeline.first,control_dt=timeline.dt,image_height=1080,
                            codes=np.zeros(n,np.int32),stations=np.linspace(0,1,n),times=np.full(n,shot_times[fi]),
                            observed=np.zeros((n,2))) as resident:
                        resident.configure_geometry([dict(kind=4,vertices=world,parameters=[])])
                        resident.configure_projection_time_support(half_seconds=manifest['max_readout']/2)
                        projected=resident.evaluate('cuda')
                    uv=projected['values'][:,3:5]; good=(projected['status']==0)&np.isfinite(uv).all(1)
                    pixels=np.rint(np.clip(np.nan_to_num(uv),-1e7,1e7)).astype(np.int32); picture=raw.copy()
                    for cut in cuts:
                        a,b=cut['begin'],cut['end']
                        for lo,hi in np.flatnonzero(np.diff(np.r_[False,good[a:b],False])).reshape(-1,2):
                            if hi-lo>1: cv2.polylines(picture,[pixels[a+lo:a+hi]],bool(cut.get('closed') and lo==0 and hi==b-a),color(cut),1,cv2.LINE_AA)
                    rows=np.asarray(by_pts[int(pts)],int)
                    for row in rows:
                        measured=tuple(np.rint(points['uv'][row]).astype(int)); fitted=tuple(np.rint(observed['values'][row,3:5]).astype(int))
                        cv2.circle(picture,measured,2,(0,0,255),-1,cv2.LINE_AA)
                        cv2.line(picture,measured,fitted,(30,130,240),1,cv2.LINE_AA)
                        cv2.drawMarker(picture,fitted,(60,255,70),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
                    panel=side.copy(); centre=poses['centre'][fi]; direction=poses['rotation'][fi,2]
                    at=top(centre); look=direction[:2]/max(np.linalg.norm(direction[:2]),1e-12)
                    cv2.arrowedLine(panel,tuple(at),tuple(np.rint(at+look*[95,-95]).astype(int)),(240,80,255),2,cv2.LINE_AA,tipLength=.12)
                    cv2.circle(panel,tuple(at),4,(240,80,255),-1,cv2.LINE_AA)
                    azimuth=np.degrees(np.arctan2(direction[1],direction[0])); elevation=np.degrees(np.arcsin(np.clip(direction[2],-1,1)))
                    lines=[f'{shot["id"].upper()} | frame {fi+1}/{len(selection)}',f'Source {pts/12800:.3f}s | PTS {pts}',
                           f'Pan {np.degrees(poses["pan"][fi]):+.3f} deg',f'Tilt {np.degrees(poses["tilt"][fi]):+.3f} deg',
                           f'View azimuth {azimuth:+.3f} deg',f'View elevation {elevation:+.3f} deg',
                           f'Focal length {poses["fx"][fi]:.2f} px',
                           f'Optical C: {centre[0]:+.6f}, {centre[1]:+.6f}, {centre[2]:+.6f} L',
                           f'Pivot: {pivot[0]:+.6f}, {pivot[1]:+.6f}, {pivot[2]:+.6f} L',
                           f'Active source points {len(rows)} | invalid 0',
                           f'Max {errors[rows].max():.2f} px | >20px {np.count_nonzero(errors[rows]>20)}',
                           f'Median {np.median(errors[rows]):.2f} | p95 {np.percentile(errors[rows],95):.2f} px']
                    for j,line in enumerate(lines): text(panel,line,(16,752+24*j),.43)
                    header=np.full((64,2560,3),15,np.uint8)
                    text(header,f'{shot["id"].upper()} NATIVE GPU JOINT FIT | FROZEN CANDIDATE | {joint_label}',(16,26),.6,(60,220,255))
                    text(header,'Source UV: red dots | Native fitted: green crosses | Original 50 fps | No additional smoothing',(16,52),.48)
                    canvas=np.vstack((header,np.hstack((picture,panel))))
                    records.append(dict(pts=int(pts),frame=fi+1,source_rows=points['source_rows'][rows].tolist(),
                                        max_px=float(errors[rows].max()),above_20px=int(np.count_nonzero(errors[rows]>20)),
                                        pose=poses['values'][fi].tolist(),invalid_observations=0))
                    if fi in key_frames: cv2.imwrite(str(out/f'{args.label}_{shot["id"]}_{pts}.png'),canvas)
                    frame=av.VideoFrame.from_ndarray(canvas,format='bgr24'); frame.pts=int(pts-selection[0]); frame.time_base=tb
                    for packet in stream.encode(frame): output.mux(packet)
                    if fi%50==0 or fi==len(selection)-1:
                        progress=dict(shot=shot['id'],frame=fi+1,total=len(selection),elapsed_s=time.monotonic()-started)
                        log.write(json.dumps(progress)+'\n'); print(json.dumps(progress),flush=True)
                for packet in stream.encode(): output.mux(packet)
            verification=validate_video(video,selection,tb)
            report=dict(shot=shot['id'],video=str(video),snapshot=str(args.snapshot),snapshot_sha256=p.digest(args.snapshot),
                        input_sha256=manifest['point_input_sha256'],native_library_sha256=manifest['native_library_sha256'],
                        source_sha256=source['sha256'],source_first_seconds=float(selection[0]/12800),
                        source_last_seconds=float(selection[-1]/12800),frames=len(records),points_total=len(shot_rows),
                        invalid_points=0,verification=verification,all_rows=records,native_parameters_postsmoothed=False,
                        complete_path_count=len(cuts),unlocalized_ads=native.unlocalized,visual_acceptance='pending')
            p.write(out/(args.label+'_'+shot['id']+'_review.json'),report)
            reports.append({k:v for k,v in report.items() if k!='all_rows'})
    p.write(out/(args.label+'_videos.json'),reports)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--run-root',type=Path); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--label',default='paused')
    main(parser.parse_args())
