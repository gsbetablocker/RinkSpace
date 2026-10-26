"""Check and render the delivered S01 ad replacement without running a solver."""
from pathlib import Path
from fractions import Fraction
import sys, hashlib, json, collections, time, zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replace_package_s01 as package
import ad_replacement_s01 as b
import numpy as np
import cv2
import av

sys.path.insert(0,str(b.ROOT/'src'))
OUT=b.OUT
BUNDLE=package.BUNDLE
REVIEW=OUT/'review'
AUDIT=OUT/'handoff_audit'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inputs():
    receipt=b.read(BUNDLE/'point_input.json')
    if digest(BUNDLE/'point_observations.npz')!=receipt['sha256']:raise ValueError('Delivered input changed')
    z=dict(np.load(BUNDLE/'point_observations.npz',allow_pickle=False))
    mapping=b.read(BUNDLE/'mapping.json')
    paths=np.asarray(mapping['path_ids'])[z['path_indices']]
    ads=np.array([int(p.split('::')[0].split('/Ad')[1]) if p.startswith('print_instance/Ad') else 0 for p in paths])
    if np.count_nonzero(ads)!=receipt['new_ad_rows']:raise ValueError('Unexpected advertisement scope')
    return z,mapping,paths,ads


def check():
    package.validate()
    z,mapping,paths,ads=inputs()
    from rinkspace.domain.parameters import ParameterState
    core,model,native=package.current_native()
    camera=b.read(mapping['camera_path']);controls=np.load(Path(mapping['camera_path']).parent/camera['controls_ref'])
    stations=np.load(BUNDLE/'initialized_points.npz')['stations']
    statefile=BUNDLE/'rebound_numeric_state.npz'
    if statefile.exists():
        from rinkspace.session.gpu_geometry import NativeSceneGPU
        state=dict(np.load(statefile,allow_pickle=False))
        if str(state['point_input_sha256'])!=digest(BUNDLE/'point_observations.npz'):raise ValueError('State/input mismatch')
        native=NativeSceneGPU(b.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']))
        camera={**camera,'rig':state['rig'].tolist()};controls=state['controls'];stations=state['stations']
    with native.resident(camera=camera,controls=controls,path_ids=paths.tolist(),stations=stations,pts=z['pts'],observed=z['uv']) as resident:
        result=resident.evaluate('cuda')
    if np.any(result['status']) or not np.isfinite(result['values'][:,9:11]).all():
        raise ValueError('Delivered full input contains invalid native projections')
    vis={r['pts']:r for r in b.read(OUT/'visibility.json')['frames']}
    for pt in np.unique(z['pts'][ads>0]):
        rows=np.flatnonzero((z['pts']==pt)&(ads>0));uv=np.rint(z['uv'][rows]).astype(int)
        mask=cv2.imread(vis[int(pt)]['mask_path'],0)
        if np.any(uv<0) or np.any(uv[:,0]>=mask.shape[1]) or np.any(uv[:,1]>=mask.shape[0]):raise ValueError('Source pixels outside frame')
        if np.any(mask[uv[:,1],uv[:,0]]):raise ValueError('Delivered advertisement point in existing person exclusion')
    manifest=b.read(package.ADS/'manifest.json');reason=collections.Counter()
    for group in manifest['per_ad'].values():
        for key,value in group.items():
            if key.startswith('reason/'):reason[key[7:]]+=value
    bysource=np.load(package.ADS/'source_provenance.npz',allow_pickle=False)['original_dataset']
    err=np.linalg.norm(result['values'][:,9:11],axis=1)
    report=dict(schema='rinkspace.ad_handoff_consumer_check.v1',passed=True,
        input_sha256=digest(BUNDLE/'point_observations.npz'),full_consumer_rows=len(paths),ad_rows=int(np.count_nonzero(ads)),
        nonad_rows=int(np.count_nonzero(ads==0)),ad_source_frames=len(np.unique(z['pts'][ads>0])),
        complete_source_frames=1248,frames_without_ad_points=1248-len(np.unique(z['pts'][ads>0])),
        ad_counts={str(n):int(np.count_nonzero(ads==n)) for n in range(2,13)},
        accepted_source_dataset_counts=dict(collections.Counter(bysource.tolist())),
        source_candidates=manifest['source_rows'],excluded_candidates=manifest['excluded_rows'],exclusion_reasons=dict(reason),
        ad_fixed_material_corners=int(np.count_nonzero((ads>0)&(z['kind']==0))),
        ad_finite_curve_points=int(np.count_nonzero((ads>0)&(z['kind']==1))),
        all_accepted_ad_source_pixels_rechecked_against_SAM2=True,
        all_rows_evaluated_by_actual_native_GPU_consumer=True,native_invalid_rows=int(np.count_nonzero(result['status'])),
        numeric_state=str(statefile) if statefile.exists() else 'current model/camera initial values',
        diagnostic_residual_quantiles_px=np.percentile(err,[0,50,95,99,100]).tolist(),
        residuals_used_for_identity_or_exclusion=False,optimizer_started=False,
        model_accuracy_or_human_acceptance_claimed=False)
    b.write(AUDIT/'consumer_check.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


COLORS={2:(0,255,255),3:(120,255,40),4:(255,170,0),5:(40,255,160),6:(255,80,220),7:(70,220,255),8:(190,240,50),9:(220,170,255),10:(120,200,255),11:(255,255,120),12:(100,255,240)}


def render():
    from rinkspace.review.render import source_frames,validate_video
    z,mapping,paths,ads=inputs();REVIEW.mkdir(exist_ok=True);input_digest=digest(BUNDLE/'point_observations.npz')
    source=b.read(b.ROOT/'events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json')['primary_source']
    if digest(source['path'])!=source['sha256']:raise ValueError('Original video identity mismatch')
    scope=b.read(b.ROOT/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json')
    shot=next(s for s in scope['shots'] if s['id']=='s01')['processing']
    clock=scope['source_clock'];tb=Fraction(clock['time_base_num'],clock['time_base_den'])
    expected=np.arange(shot['first_pts'],shot['stop_pts_exclusive'],clock['native_pts_step'],dtype=np.int64)
    vis={r['pts']:r for r in b.read(OUT/'visibility.json')['frames'] if r['pts'] in set(map(int,expected))}
    if set(vis)!=set(map(int,expected)):raise ValueError('Incomplete native exposure index')
    if not set(map(int,z['pts'][ads>0]))<=set(vis):raise ValueError('Ad source frame outside video')
    selected={int(pt):np.flatnonzero((z['pts']==pt)&(ads>0)) for pt in expected}
    path=REVIEW/'S01_replaced_ad_points.mp4';temp=path.with_suffix('.partial.mp4')
    if path.exists():raise ValueError('Existing delivered review must be explicitly revised')
    count=0;source_hashes=hashlib.sha256();start=time.monotonic()
    with av.open(str(temp),'w',options={'movflags':'+faststart','video_track_timescale':str(tb.denominator)}) as output, (REVIEW/'frame_index.jsonl').open('w',encoding='utf-8') as ledger:
        stream=output.add_stream('libx264',rate=Fraction(clock['native_fps_num'],clock['native_fps_den']))
        stream.width=source['width'];stream.height=source['height']+96;stream.pix_fmt='yuv420p'
        stream.time_base=tb;stream.codec_context.time_base=tb;stream.codec_context.thread_count=4
        stream.options={'crf':'16','preset':'fast'}
        metadata=dict(source,time_base_num=tb.numerator,time_base_den=tb.denominator)
        for i,(pt,raw) in enumerate(source_frames(Path(source['path']),metadata,expected)):
            rawsha=hashlib.sha256(raw.tobytes()).hexdigest()
            if rawsha!=vis[pt]['source_BGR_sha256']:raise ValueError('Review image differs from the measured original exposure')
            source_hashes.update(bytes.fromhex(rawsha));rows=selected[pt]
            canvas=np.zeros((source['height']+96,source['width'],3),np.uint8);canvas[:source['height']]=raw
            for row in rows:
                x,y=np.rint(z['uv'][row]).astype(int);color=COLORS[int(ads[row])]
                if z['kind'][row]==0:
                    cv2.rectangle(canvas,(x-3,y-3),(x+3,y+3),(0,0,0),2,cv2.LINE_AA)
                    cv2.rectangle(canvas,(x-2,y-2),(x+2,y+2),color,1,cv2.LINE_AA)
                else:
                    cv2.circle(canvas,(x,y),2,(0,0,0),-1,cv2.LINE_AA)
                    cv2.circle(canvas,(x,y),1,color,-1,cv2.LINE_AA)
            base=source['height']
            cv2.putText(canvas,f'S01 | SOURCE {float(pt*tb):06.3f}s | PTS {pt} | FRAME {i+1}/{len(expected)} | ACTIVE AD POINTS {len(rows)}',(12,base+23),0,.58,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(canvas,'Original measured UV | dots: curve points | squares: fixed material corners | prepared fit input / no new fit',(12,base+45),0,.48,(210,210,210),1,cv2.LINE_AA)
            for n in range(2,13):
                label=f'Ad{n}: {int(np.count_nonzero(ads[rows]==n))}'
                cv2.putText(canvas,label,(12+(n-2)*170,base+76),0,.55,COLORS[n],1,cv2.LINE_AA)
            frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=pt-int(expected[0]);frame.time_base=tb
            for packet in stream.encode(frame):output.mux(packet)
            record=dict(video_frame_index=i,video_pts=frame.pts,source_pts=pt,source_seconds=float(pt*tb),source_BGR_sha256=rawsha,
                point_input_sha256=input_digest,bundle_row_indices=rows.tolist(),ad_point_count=len(rows))
            ledger.write(json.dumps(record)+'\n');count+=1
            if i%100==0 or i==len(expected)-1:print(json.dumps(dict(stage='render',frames=count,total=len(expected),seconds=round(time.monotonic()-start,1))),flush=True)
        for packet in stream.encode():output.mux(packet)
    verification=validate_video(temp,expected,tb);temp.replace(path)
    if sum(len(v) for v in selected.values())!=int(np.count_nonzero(ads)):raise ValueError('Not every delivered ad row displayed exactly once')
    report=dict(schema='rinkspace.ad_replacement_video.v1',path=str(path),sha256=digest(path),
        point_input=str(BUNDLE/'point_observations.npz'),point_input_sha256=digest(BUNDLE/'point_observations.npz'),
        source_video=source['path'],source_sha256=source['sha256'],source_time_base=str(tb),
        source_first_pts=int(expected[0]),source_last_pts=int(expected[-1]),source_first_seconds=float(expected[0]*tb),source_last_seconds=float(expected[-1]*tb),
        complete_native_frames=count,source_frame_BGR_hashes_verified=count,source_frame_hash_sequence_sha256=source_hashes.hexdigest(),
        displayed_ad_rows=int(np.count_nonzero(ads)),frames_without_ad_points=sum(not len(v) for v in selected.values()),
        all_source_pixels_at_original_resolution=True,original_image_size=[source['width'],source['height']],output_size=[source['width'],source['height']+96],
        current_exposure_only=True,projected_or_interpolated_points=0,withdrawn_points_shown=0,extra_point_ID_labels=False,
        verification=verification,audio_included=False,player_opened=False,human_visual_acceptance=False,
        frame_index=dict(path=str(REVIEW/'frame_index.jsonl'),sha256=digest(REVIEW/'frame_index.jsonl')))
    b.write(REVIEW/'video.json',report);print(json.dumps(report),flush=True)


def seal():
    required=[OUT/'HANDOFF.md',OUT/'validation.json',AUDIT/'consumer_check.json',AUDIT/'visual_review.json',REVIEW/'video.json',REVIEW/'frame_index.jsonl']
    required+=sorted(p for p in package.ADS.iterdir() if p.is_file())
    required+=sorted(p for p in BUNDLE.iterdir() if p.is_file())
    required+=[OUT/'boundary_identity_review.json',OUT/'source_lineage.json',OUT/'current_binding_color_audit.json',OUT/'current_explicit_conflicts.json',OUT/'direct_color_source_review.json']
    entries=[dict(path=str(p.relative_to(OUT)),sha256=digest(p),bytes=p.stat().st_size) for p in required]
    report=dict(schema='rinkspace.s01_ad_handoff.v1',status='ready_for_consumer_and_user_visual_review',files=entries,
        source_scope='S01 / 20.020-44.960 seconds / 1248 native exposures',
        model_and_camera_parameters_from_dataset=False,optimizer_started=False,human_visual_acceptance=False,
        video=b.read(REVIEW/'video.json'),history_preserved_at=str(OUT/'review_revisions'))
    b.write(OUT/'handoff_manifest.json',report)
    path=OUT/'ad_replacement_handoff_s01.zip';temporary=path.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in required+[OUT/'handoff_manifest.json']:archive.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('Handoff archive CRC failure')
        for entry in entries:
            if hashlib.sha256(archive.read(Path(entry['path']).as_posix())).hexdigest()!=entry['sha256']:raise ValueError('Handoff archive member mismatch')
    temporary.replace(path)
    b.write(OUT/'delivery.json',dict(package=str(path),package_sha256=digest(path),manifest_sha256=digest(OUT/'handoff_manifest.json'),video=str(REVIEW/'S01_replaced_ad_points.mp4'),archive_members_verified=len(entries)+1))
    print(json.dumps(b.read(OUT/'delivery.json')),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['check','render','seal']);args=parser.parse_args()
    {'check':check,'render':render,'seal':seal}[args.stage]()
