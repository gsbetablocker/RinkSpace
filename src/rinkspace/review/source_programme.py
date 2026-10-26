"""Full native-exposure source reading and separate frozen-model review videos."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
import hashlib,json,time
import av,cv2,numpy as np
from .render import source_frames,validate_video,project,overlay,top_down
from ..session.dataset import write_json,digest


def draw_points(raw,record):
    canvas=raw.copy();groups={}
    for c in record['candidates']:
        if c['localization']!='measured' or c['actual_uv'] is None or c['withdrawal']:continue
        status='identified' if c['identity']=='confirmed' else 'unresolved'
        key=(c.get('native_path_id') or c['source_feature_id'],status)
        groups.setdefault(key,[]).append(c['actual_uv'])
    for (name,status),points in groups.items():
        color=(80,245,80) if status=='identified' else (0,185,255)
        pts=np.unique(np.rint(points).astype(int),axis=0)
        for x,y in pts:
            if 0<=x<canvas.shape[1] and 0<=y<canvas.shape[0]:cv2.circle(canvas,(int(x),int(y)),2,color,-1,cv2.LINE_AA)
        if len(pts):
            x,y=np.median(pts,axis=0).astype(int);label=name.rsplit('/',1)[-1]
            cv2.putText(canvas,label,(max(4,min(x,canvas.shape[1]-180)),max(55,min(y-8,canvas.shape[0]-8))),
                cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    b=record['source_binding'];pts=b['source_pts']
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],42),(22,22,22),-1)
    cv2.putText(canvas,f"{b['shot_id'].upper()} | source {pts/12800:.3f}s | PTS {pts} | MACHINE - NOT REVIEWED",
        (12,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,'GREEN: source pixels with identity | AMBER: source pixels, identity pending | NO MODEL GRID',
        (12,36),cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1,cv2.LINE_AA)
    return canvas


def current_review_navigation(job,ledger,geometry,navigation_root,navigation_results=None):
    """Load the selected solve revision only after its source/native checks."""
    from ..session.source_programme import navigation_dependencies
    cameras={};references={}
    scope_path=Path(job['roots']['repo'])/job['scope_ref']['path']
    for shot in job['sequences']:
        sid=shot['shot_id']
        path=Path(navigation_results[sid]) if navigation_results else Path(navigation_root)/sid/'result.json'
        saved=json.loads(path.read_text(encoding='utf-8-sig'))
        dependencies=navigation_dependencies(ledger,sid,geometry,saved['camera_seed'],scope_path)
        if saved.get('source_dependencies')!=dependencies:
            raise ValueError('Review navigation does not belong to the current source/native revision: '+sid)
        outputs=saved.get('output_sha256',{})
        if set(outputs)!={'controls.npy','stations.npy','camera.json'} or any(
                not (path.parent/name).is_file() or digest(path.parent/name)!=sha for name,sha in outputs.items()):
            raise ValueError('Review navigation output changed: '+sid)
        camera=json.loads((path.parent/'camera.json').read_text(encoding='utf-8-sig'))
        camera['controls']=np.load(path.parent/'controls.npy',allow_pickle=False)
        cameras[sid]=camera;references[sid]=dict(path=str(path.resolve()),sha256=digest(path),dependencies=dependencies)
    return cameras,references


def render_programme(job,ledger,geometry,navigation_root,out,report,*,navigation_results=None):
    """Reuse existing native projection, overlay, decoder and video verification."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tb=Fraction(job['source_clock']['time_base_num'],job['source_clock']['time_base_den'])
    source=dict(job['source'],time_base_num=tb.numerator,time_base_den=tb.denominator)
    expected=np.array([p for s in job['sequences'] for p in range(s['first_pts'],s['stop_pts_exclusive'],256)],np.int64)
    actual=np.array([p for p, in ledger.db.execute('select pts from frames order by pts')],np.int64)
    if not np.array_equal(expected,actual):raise ValueError('Full source review cannot omit native exposures')
    cameras,references=current_review_navigation(job,ledger,geometry,navigation_root,navigation_results)
    receipt=out/'review.json'
    if receipt.is_file() and json.loads(receipt.read_text(encoding='utf-8')).get('navigation_references')!=references:
        token=hashlib.sha256(json.dumps(references,sort_keys=True).encode()).hexdigest()
        out=out/'revisions'/token;out.mkdir(parents=True,exist_ok=True);receipt=out/'review.json'
    if receipt.is_file():
        saved=json.loads(receipt.read_text(encoding='utf-8'))
        for v in saved['videos']:
            if digest(out/v['path'])!=v.get('sha256'):raise ValueError('Saved source review video changed')
            validate_video(out/v['path'],expected,tb)
        return saved
    paths=[out/'S02-S09_source_points.mp4',out/'S02-S09_native_GPU_navigation.mp4']
    temporaries=[p.with_suffix('.partial.mp4') for p in paths]
    outputs=[];streams=[];timings=dict(decode_seconds=0.,point_draw_seconds=0.,native_projection_seconds=0.,grid_draw_seconds=0.,encode_seconds=0.)
    start=time.perf_counter();samples=geometry.samples
    overview=top_down(samples['xyz'],samples['cuts'],height=source['height'],missing_count=40)
    for p,width in zip(temporaries,(1920,2560)):
        container=av.open(str(p),'w',options={'movflags':'+faststart','video_track_timescale':str(tb.denominator)})
        stream=container.add_stream('libx264',rate=Fraction(50));stream.width=width;stream.height=1080;stream.pix_fmt='yuv420p'
        stream.time_base=tb;stream.codec_context.time_base=tb;stream.options={'crf':'18','preset':'fast'}
        outputs.append(container);streams.append(stream)
    artifacts=[];count=0
    try:
        for shot in job['sequences']:
            sid=shot['shot_id'];camera=cameras[sid]
            geometry.assert_unchanged()
            pts_array=np.array([p for p in expected if shot['first_pts']<=p<shot['stop_pts_exclusive']],np.int64)
            decoder=iter(source_frames(Path(source['registered_path']),source,pts_array))
            for index in range(len(pts_array)):
                before=time.perf_counter();pts,raw=next(decoder);timings['decode_seconds']+=time.perf_counter()-before
                record=ledger.read(sid,int(pts))
                if hashlib.sha256(raw[:,:,::-1].copy().tobytes()).hexdigest()!=record['source_binding']['decoded_native_rgb_sha256']:
                    raise ValueError('Review RGB differs from measured native exposure')
                before=time.perf_counter();points=draw_points(raw,record);timings['point_draw_seconds']+=time.perf_counter()-before
                before=time.perf_counter();prediction=project(geometry.core,camera,samples['xyz'],int(pts));timings['native_projection_seconds']+=time.perf_counter()-before
                before=time.perf_counter();grid,_=overlay(raw,prediction,samples['cuts'],overview,int(pts),tb,np.empty((0,2)),sid,index,len(pts_array));timings['grid_draw_seconds']+=time.perf_counter()-before
                cv2.putText(grid,'GPU CAMERA NAVIGATION / MODEL PROJECTION / NOT SOURCE POINTS',(12,45),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                for container,stream,pixels in zip(outputs,streams,(points,grid)):
                    before=time.perf_counter();frame=av.VideoFrame.from_ndarray(pixels,format='bgr24');frame.pts=int(pts-expected[0]);frame.time_base=tb
                    for packet in stream.encode(frame):container.mux(packet)
                    timings['encode_seconds']+=time.perf_counter()-before
                if index in (0,len(pts_array)//2,len(pts_array)-1):
                    for label,pixels in [('source_points',points),('native_grid',grid)]:
                        name=f'{sid}_{pts}_{label}.jpg';cv2.imwrite(str(out/name),pixels,[cv2.IMWRITE_JPEG_QUALITY,95]);artifacts.append(dict(path=name,source_pts=int(pts),source_seconds=float(pts*tb)))
                count+=1
                if index%50==0 or index==len(pts_array)-1:report(dict(stage='review_video',shot=sid,pts=int(pts),encoded_frames=count,total=len(expected)))
        for container,stream in zip(outputs,streams):
            for packet in stream.encode():container.mux(packet)
    finally:
        for container in outputs:container.close()
    _,after=current_review_navigation(job,ledger,geometry,navigation_root,navigation_results)
    if after!=references:raise ValueError('Source or native navigation changed during review rendering')
    videos=[]
    for temporary,path in zip(temporaries,paths):
        check=validate_video(temporary,expected,tb);temporary.replace(path)
        videos.append(dict(path=path.name,sha256=digest(path),verification=check,source_first_seconds=float(expected[0]*tb),source_last_seconds=float(expected[-1]*tb)))
    result=dict(status='machine_result_pending_scoped_visual_review',native_exposures=count,shot_ids=job['selected_shot_ids'],
        videos=videos,stills=artifacts,timing=dict(timings,total_seconds=time.perf_counter()-start),human_approval=False,
        navigation_references=references,output_directory=str(out.resolve()))
    write_json(receipt,result);return result
