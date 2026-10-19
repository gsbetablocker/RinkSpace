"""Opt-in complete cut5/cut6 grid review on a frozen physical camera/scene.

Default: validate inputs and prepare the exact native/exposure clock only.
Small source checks are explicit. --video exports the selected COMPLETE cuts;
it never fits/rebinds a body, fills a camera, or substitutes a short subclip.
"""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av,cv2,numpy as np
from rinkspace.storage import AssetStore,resolve_asset
import render_programme_root_trace as movie
from render_result import SceneRenderer,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE,decode_selected,MASK_STATES
from z190_physical_camera import PhysicalCameraModel
from continuous_central_brown4_optin import enable_projection_backend
from three_frame_refit.program_tracking.body_tracking.pelvis_proxy_programme import PlanView,source_time

CUTS={5:(229945856,230369024),6:(230369024,230518016)}
GEOMETRY_KEY='shared_rink_numeric_geometry_lanes_ads_v1'
STORAGE=AssetStore(ROOT)
DEFAULT_SOURCE=STORAGE.source(ROOT/'data/input/oxXu1avhrvs.mp4')
SHOTS=Path(__file__).with_name('source_shots.json')

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def reference(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class FrozenProblemCutGridView:
    def __init__(self,camera_path,scene_path,source_path,masks_path,people_path,state_identity_path=None):
        enable_projection_backend();cv2.setNumThreads(1)
        camera_path,scene_path,masks_path,people_path=map(STORAGE.resolve,(camera_path,scene_path,masks_path,people_path))
        source_path=STORAGE.source(source_path)
        self.input_paths=list(map(Path,(camera_path,scene_path,source_path,masks_path,people_path)))
        self.frozen_paths=[Path(camera_path),Path(scene_path)]
        self.state_identity=None
        if state_identity_path is not None:
            state_identity_path=STORAGE.resolve(state_identity_path)
            self.input_paths.append(Path(state_identity_path))
            self.state_identity=json.loads(Path(state_identity_path).read_text(encoding='utf-8'))
            identity=self.state_identity
            complete=Path(state_identity_path).parent/'complete_state_UNACCEPTED.npz'
            if (Path(scene_path).resolve()!=Path(state_identity_path).parent/'scene_UNACCEPTED.json'
                    or Path(camera_path).resolve()!=Path(identity['camera_path']).resolve()
                    or sha(scene_path)!=identity['scene_sha256']
                    or sha(camera_path)!=identity['camera_sha256']
                    or sha(complete)!=identity['state_npz_sha256']):
                raise ValueError('Complete state, actual camera and scene identity differ')
            with np.load(complete,allow_pickle=False)as arrays:
                canonical=np.asarray(arrays['q'],np.float64)
                if (hashlib.sha256(canonical.tobytes()).hexdigest()!=identity['q_sha256']
                        or canonical.shape!=(int(identity['parameter_count']),)):
                    raise ValueError('Complete canonical q identity differs')
            self.frozen_paths.extend((Path(state_identity_path),complete))
        self.input_refs=[reference(path)for path in self.frozen_paths]
        self.model,self.pts,self.aliases,self.views=PhysicalCameraModel.load_npz(camera_path)
        with np.load(camera_path,allow_pickle=False)as z:
            self.camera25=z['camera25'];metadata=json.loads(str(z['metadata_json'].item()))
        shots=json.loads(SHOTS.read_text(encoding='utf-8-sig'));scope=shots['analysis_scope']
        self.cuts=(dict(CUTS)if self.state_identity is None else
            {i:(int(shot['start_pts']),int(shot['end_pts']))
             for i,shot in enumerate(shots['shots'][1:10],1)})
        expected=np.arange(scope['start_pts'],scope['end_pts_exclusive'],256,dtype=np.int64)
        if(not np.array_equal(self.pts,expected)or len(self.pts)!=10123 or len(self.views)!=8503 or
           len(self.model.calibrations)!=3 or metadata['source_time_base']!='1/15360' or
           metadata['source_sha256']!=shots['source_sha256_from_timeline'] or
           source_path!=STORAGE.source(shots['source_path'])):
            raise ValueError('Complete original 3-installation/8503-actual/10123-native camera required')
        if any(np.linalg.norm(c.lever_xyz)<=0 for c in self.model.calibrations.values()):
            raise ValueError('Keep all three original nonzero shared camera levers')
        self.reps=np.asarray([self.aliases[int(p)]for p in self.pts],np.int64)
        self.lookup={int(p):i for i,p in enumerate(self.pts)}
        self.scene=json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
        geometry=self.scene.get(GEOMETRY_KEY,{})
        if self.state_identity is None and geometry.get('parameter_count')!=11:
            raise ValueError('Supply the frozen actual q11 shared Lane/Ad1/14 scene, not its old base')
        if self.state_identity is None:
            self.renderer=SceneRenderer(self.scene,draw_grid=True,draw_advertisements=True)
            self.plan=PlanView(self.scene,0.)
        else:
            contract=self.scene.get('source_event_state_contract',{})
            if (contract.get('q_sha256')!=self.state_identity['q_sha256']
                    or contract.get('q_shape')!=[self.state_identity['parameter_count']]
                    or contract.get('frozen_profile_sha256')!=self.state_identity['frozen_profile_token']):
                raise ValueError('Scene is not the same complete q/profile as its immutable identity')
            from board_grid_overlay import make_board_grid_renderer
            from rinkspace.metric_scale import boundary_span,estimate_uniform_scale
            self.scene_unit_m=estimate_uniform_scale(boundary_span(self.scene))
            Renderer=make_board_grid_renderer(SceneRenderer,scene_unit_m=self.scene_unit_m)
            self.renderer=Renderer(self.scene,draw_grid=True,draw_advertisements=True,
                graphics_regions=((28,24,554,253),(1690,944,1865,1000)))
            self.plan=PlanView(self.scene,0.,board_paths=self.renderer.board_grid.paths)
        if self.renderer.path_counts!=self.plan.path_counts:
            raise ValueError('Source projection and plan must draw the same complete ice inventory')
        left_ids=sorted({int(row['number'])for row in self.renderer.advertisement_paths})
        if left_ids!=self.plan.advertisement_ids:
            raise ValueError('Source and plan must draw the same actual advertisements/patterns')
        if self.state_identity is not None and (left_ids!=list(range(1,27))
                or self.renderer.path_counts.get('marking')!=65):
            raise ValueError('All 65 current ice paths and Ad1-26 must remain enabled')
        self.base=self.plan.base.copy();self.base[510:]=248
        for i,label in enumerate(('全部网格、靶圈、Lane 与广告','同一冻结场景，线条持续全开',
                                  '三台固定安装、原非零镜头偏移','原片 60Hz 展示；沿用实际曝光别名',
                                  '当前相机候选，仍需完整镜头审阅','本诊断只显示冰场定位')):
            self.base=self.plan.text(self.base,(20,535+58*i),label,small=True)
        self.header=np.full((70,movie.SIZE[0],3),20,np.uint8)
        self.header=self.plan.text(self.header,(18,14),'完整镜头网格审阅（候选）',color=(245,245,245))
        self.masks=PersonMasks(Path(masks_path),Path(people_path))
        if STORAGE.source(self.masks.people_manifest['source'])!=source_path:
            raise ValueError('Existing source person masks belong to another video')
        self.source_metadata=dict(path=str(Path(source_path).resolve()),
            sha256_from_authoritative_source_timeline=metadata['source_sha256'],
            file_rehashed_in_this_renderer=False,time_base='1/15360',native_rate='60/1',
            official_actual_exposure_clock_reused_without_average_fill_or_SLERP=True)
        self.current_points=None
        if self.state_identity is not None and self.state_identity.get('current_source_points'):
            from current_source_points import CurrentSourcePoints
            binding=self.state_identity['current_source_points']
            self.current_points=CurrentSourcePoints(binding['path'],camera_path,
                q_sha256=self.state_identity['q_sha256'])
            manifest=self.current_points.metadata['input_manifest']
            if (self.current_points.reference['sha256']!=binding['sha256'] or
                    manifest['sha256']!=self.state_identity['input_manifest_sha256'] or
                    sha(manifest['path'])!=manifest['sha256']):
                raise ValueError('Current source pixels differ from the immutable fitting inputs')
            self.frozen_paths.extend((self.current_points.path,Path(manifest['path'])))
            self.input_refs=[reference(path)for path in self.frozen_paths]

    def cut_pts(self,index):
        start,end=self.cuts[index];selected=self.pts[(self.pts>=start)&(self.pts<end)]
        expected=np.arange(start,end,256,dtype=np.int64)
        if not np.array_equal(selected,expected):raise ValueError('Camera omits a native presentation inside the complete cut')
        if any(not start<=self.aliases[int(p)]<end for p in selected):
            raise ValueError('An official exposure alias crosses a camera cut')
        if len({self.views[self.aliases[int(p)]].installation_id for p in selected})!=1:
            raise ValueError('A complete camera cut changes installation identity')
        missing=[int(p)for p in selected if p not in self.masks.lookup or self.masks.lookup[int(p)][2]not in MASK_STATES]
        if missing:raise ValueError(f'Original measured person masks unavailable: {missing[:5]}')
        return selected

    def draw(self,native,pts):
        pts=int(pts);i=self.lookup[pts];actual=self.aliases[pts]
        if not any(start<=pts<end for start,end in self.cuts.values()):
            raise ValueError('Native PTS is outside the selected complete source cuts')
        mask,mask_status=self.masks.get(pts,require_sam2=True)
        left,metrics=self.renderer.draw(native,self.camera25[i],mask,title=None)
        if self.current_points is not None:self.current_points.draw(left,actual)
        canvas=np.full(movie.OUTPUT_SHAPE,20,np.uint8)
        canvas[:70]=self.header;canvas[70:1150,:1920]=left;canvas[70:1150,1920:]=self.base
        cut=next(k for k,(a,b)in self.cuts.items()if a<=pts<b)
        cv2.putText(canvas,f'cut{cut} | source {source_time(pts)} | native PTS {pts} | actual {actual}',
            (650,43),cv2.FONT_HERSHEY_SIMPLEX,.73,(245,245,245),2,cv2.LINE_AA)
        self.last_metrics=dict(pts=pts,actual_pts=actual,mask_status=mask_status,
            source_native_bgr_sha256=hashlib.sha256(native).hexdigest(),render_metrics=metrics)
        return canvas

    def assert_frozen_inputs(self):
        if self.input_refs!=[reference(path)for path in self.frozen_paths]:
            raise ValueError('Frozen camera/actual scene changed during review')

def _initialize_grid_worker(paths,expected_refs,native_name,output_name,slot_count):
    # Reuse the already-audited ordered pixel ring and chunk worker. Only its
    # view/compositor is replaced; no ProgrammeRootView/body is instantiated.
    movie._WORKER_SHM=(shared_memory.SharedMemory(name=native_name),shared_memory.SharedMemory(name=output_name))
    movie._WORKER_NATIVE=np.ndarray((slot_count,*movie.NATIVE_SHAPE),np.uint8,buffer=movie._WORKER_SHM[0].buf)
    movie._WORKER_OUTPUT=np.ndarray((slot_count,*movie.OUTPUT_SHAPE),np.uint8,buffer=movie._WORKER_SHM[1].buf)
    movie._WORKER_VIEW=FrozenProblemCutGridView(*paths)
    if movie._WORKER_VIEW.input_refs!=expected_refs:raise ValueError('Drawing worker loaded different frozen camera/scene')

class GridOnlyOrderedSharedDraw(movie.OrderedSharedDraw):
    def __init__(self,view,workers=4,slots=None,chunk_size=4,verify_hash=False):
        super().__init__(view.input_paths,workers,slots,chunk_size,verify_hash,
            representative_by_pts=view.aliases)
        self.expected_refs=view.input_refs

    def __enter__(self):
        self.native_memory=shared_memory.SharedMemory(create=True,size=self.slots*int(np.prod(movie.NATIVE_SHAPE)))
        try:self.output_memory=shared_memory.SharedMemory(create=True,size=self.slots*int(np.prod(movie.OUTPUT_SHAPE)))
        except BaseException:
            self.native_memory.close();self.native_memory.unlink();raise
        self.native=np.ndarray((self.slots,*movie.NATIVE_SHAPE),np.uint8,buffer=self.native_memory.buf)
        self.output=np.ndarray((self.slots,*movie.OUTPUT_SHAPE),np.uint8,buffer=self.output_memory.buf)
        self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),
            initializer=_initialize_grid_worker,initargs=(self.paths,self.expected_refs,
            self.native_memory.name,self.output_memory.name,self.slots))
        return self

def verify_video(path,selected):
    count=0;max_clock_error=0.;clock_digest=hashlib.sha256()
    with av.open(str(path))as video:
        if len(video.streams.video)!=1:raise ValueError('One grid diagnostic video stream required')
        stream=video.streams.video[0]
        if (stream.width,stream.height)!=movie.SIZE or Fraction(stream.average_rate)!=60:
            raise ValueError('Complete cut output size/native60 presentation cadence changed')
        for frame in video.decode(stream):
            if count>=len(selected)or frame.pts is None:raise ValueError('Encoded native presentation count/PTS changed')
            error=abs(float(Fraction(frame.pts)*frame.time_base-Fraction(count,60)))
            max_clock_error=max(max_clock_error,error);clock_digest.update(np.int64(selected[count]).tobytes());count+=1
    if count!=len(selected)or max_clock_error>1./15360:
        raise ValueError('Encoded complete cut lost a native frame or presentation time')
    return dict(decoded_native_presentations=count,encoded_relative_clock_maximum_error_seconds=max_clock_error,
        output_size_wh=list(movie.SIZE),native_rate='60/1',absolute_source_time_on_every_frame=True,
        verified_original_native_pts_sha256=clock_digest.hexdigest())

def render_video(args,view,cut,selected,*,filename=None):
    target=args.output/(filename or f'cut{cut:02d}_whole_grid_ONLY_UNACCEPTED.mp4')
    if target.exists():raise FileExistsError(target)
    ffmpeg=movie._ffmpeg_executable();encoder,encoder_name=movie._available_encoder(ffmpeg)
    command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{movie.SIZE[0]}x{movie.SIZE[1]}','-framerate','60','-i','pipe:0','-an',
        *encoder,'-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    count=0;started=time.perf_counter();digest=hashlib.sha256();ring_stats=None
    def consume(pts,canvas):
        nonlocal count
        if count>=len(selected)or pts!=int(selected[count]):raise ValueError('Encoder input changed complete-cut native order')
        process.stdin.write(memoryview(canvas).cast('B'));digest.update(np.int64(pts).tobytes());count+=1
        if count%600==0:print(json.dumps(dict(stage='whole_cut_grid_only',cut=cut,frames=count,total=len(selected))),flush=True)
    try:
        decoded=movie.decode_programme_frames(args.source,selected)
        if args.draw_workers>1:
            with GridOnlyOrderedSharedDraw(view,args.draw_workers,args.draw_slots,args.draw_chunk_size)as ring:
                for pts,canvas,_ in ring.frames(decoded):consume(pts,canvas)
                ring_stats=ring.stats.copy()
        else:
            for pts,native in decoded:consume(pts,view.draw(native,pts))
        process.stdin.close();code=process.wait(timeout=60)
        if code or count!=len(selected):raise RuntimeError(process.stderr.read().decode(errors='replace'))
    except BaseException:
        if process.poll()is None:process.kill();process.wait(timeout=30)
        raise
    view.assert_frozen_inputs()
    return dict(file=str(target),cut=cut,selected_encoder_name=encoder_name,selected_encoder_args=encoder,
        drawing_pipeline=ring_stats or dict(mode='serial',workers=1),
        encoded_native_count=count,encoded_input_original_pts_sha256=digest.hexdigest(),
        wall_seconds=time.perf_counter()-started,verification=verify_video(target,selected))

def run(args):
    args.source=STORAGE.source(args.source)
    args.output=STORAGE.resolve(args.output)
    if not args.output.resolve().is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('All diagnostic products must remain in D:/CodexData')
    args.output.mkdir(parents=True,exist_ok=True)
    view=FrozenProblemCutGridView(args.camera,args.scene,args.source,args.masks,args.people)
    scopes=[];native=[];indices=[]
    for cut in sorted(set(args.cut)):
        pts=view.cut_pts(cut);a,b=CUTS[cut];native.extend(pts);indices.extend([cut]*len(pts))
        scopes.append(dict(cut=cut,start_pts=a,end_pts_exclusive=b,source_start=source_time(a),
            source_end_exclusive=source_time(b),source_duration_seconds=(b-a)/15360,
            native_presentations=len(pts),actual_exposures=len({view.aliases[int(p)]for p in pts}),
            source_last_native_pts=int(pts[-1]),complete_cut=True))
    native=np.asarray(native,np.int64);actual=np.asarray([view.aliases[int(p)]for p in native],np.int64)
    clock=args.output/'complete_cut_native_actual_clock.npz'
    if clock.exists():raise FileExistsError(clock)
    np.savez_compressed(clock,native_pts=native,actual_pts=actual,cut_index=np.asarray(indices,np.int64),
        source_time_base=np.array('1/15360'),native_rate=np.array('60/1'))
    selected=sorted(set(args.check_pts));allowed=set(map(int,native))
    if not set(selected).issubset(allowed):raise ValueError('Small-check PTS must belong to the requested complete cuts')
    checks=[];serial_hashes=[];frames=[]
    for pts,source in decode_selected(args.source,selected):
        image=view.draw(source,pts);target=args.output/f'{pts}_grid_ONLY_check.png'
        if target.exists():raise FileExistsError(target)
        if not cv2.imwrite(str(target),image):raise RuntimeError('Cannot save small grid-only interface check')
        checks.append(dict(image=str(target),**view.last_metrics));serial_hashes.append(hashlib.sha256(image).hexdigest())
        if args.check_parallel:frames.append((pts,source))
    parallel=None
    if args.check_parallel:
        if len(frames)==0 or args.draw_workers<=1:raise ValueError('Parallel check needs explicit real source PTS and multiple drawing workers')
        results=[]
        with GridOnlyOrderedSharedDraw(view,args.draw_workers,args.draw_slots,args.draw_chunk_size,True)as ring:
            for pts,canvas,digest in ring.frames(frames):results.append((pts,digest))
            stats=ring.stats.copy()
        if results!=list(zip(selected,serial_hashes)):raise ValueError('Grid-only serial/process canvases or native order differ')
        parallel=dict(exact_serial_process_canvas_sha256=True,source_pts=selected,pipeline=stats)
    videos=[]
    if args.video:
        for cut in sorted(set(args.cut)):videos.append(render_video(args,view,cut,view.cut_pts(cut)))
    view.assert_frozen_inputs()
    record=dict(schema='rinkspace.complete-problem-cuts-grid-only-review.v1',scopes=scopes,
        camera_and_actual_scene=view.input_refs,source=view.source_metadata,clock=reference(clock),
        geometry_parameter_count=11,actual_geometry_parameters_m=view.scene[GEOMETRY_KEY]['parameters_m'],
        source_and_plan_all_geometry=view.renderer.path_counts,advertisement_ids=view.plan.advertisement_ids,
        every_frame_projects_the_same_complete_scene=True,geometry_hidden_by_alignment_or_confidence=False,
        shared_geometry_cache_reused=True,original_native_decode_reused=True,old_movie_ordered_pixel_ring_reused=True,
        source_person_masks_reused=True,body_input_loaded=False,body_P_G_or_trajectory_drawn=False,
        source_alias_states_replayed_by_PhysicalCameraModel=True,source_and_actual_exposure_clocks_kept_distinct=True,
        small_checks=checks,small_parallel_check=parallel,videos=videos,
        fullcut_video_export_executed=bool(videos),fullprogramme_video_generated=False,
        video_encoder_initialized_only_for_video_export=True,accuracy_accepted=False,
        small_checks_do_not_validate_complete_cuts=True,production_camera_scene_body_video_changed=False)
    write(args.output/'grid_only_review_entry.json',record)
    print(json.dumps(dict(output=str(args.output),scopes=scopes,small_checks=len(checks),
        parallel_pixel_exact=parallel is not None,fullcut_videos=len(videos))),flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('camera','scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--masks',type=Path,default=DEFAULT_MASKS);parser.add_argument('--people',type=Path,default=DEFAULT_PEOPLE)
    parser.add_argument('--cut',type=int,choices=(5,6),nargs='+',default=[5,6])
    parser.add_argument('--check-pts',type=int,nargs='*',default=[])
    parser.add_argument('--check-parallel',action='store_true')
    parser.add_argument('--draw-workers',type=int,default=min(4,os.cpu_count()or 1))
    parser.add_argument('--draw-chunk-size',type=int,default=4);parser.add_argument('--draw-slots',type=int)
    parser.add_argument('--video',action='store_true',help='Explicit export of each selected COMPLETE cut; otherwise prepare/small checks only')
    run(parser.parse_args())
