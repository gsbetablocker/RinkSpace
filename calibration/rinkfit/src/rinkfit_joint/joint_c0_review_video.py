"""Render the complete native S05 with only C0 paint midline and current C0 dots."""
from pathlib import Path
from fractions import Fraction
import argparse,gzip,hashlib,json,os,shutil,subprocess,time
import av,cv2,numpy as np
from .io_utils import file_sha256
from .joint_metrics import PacketArrays
from .joint_model import graph_from_joint,as_event_view
from .camera_graph import CameraGraph
from .joint_acceleration import install
from .joint_replay import fast_native_project
from rinkfit_start.native_scene import NativeScene

def render(*,run,packet,scope,output,compiled_dir):
    started=time.perf_counter();run=Path(run);packet=Path(packet);out=Path(output)
    if out.exists():raise FileExistsError(str(out))
    packet_sha=file_sha256(packet)
    while True:
        latest_bytes=(run/'recoverable_latest.json').read_bytes();latest=json.loads(latest_bytes)
        candidate=Path(latest['directory']);binding_bytes=(candidate/'observed_row_initialization_state_binding.json').read_bytes()
        binding=json.loads(binding_bytes);model_bytes=(candidate/'model.json').read_bytes()
        model_sha=hashlib.sha256(model_bytes).hexdigest()
        if (model_sha==latest['model_sha256'] and binding['state_token']==latest['state_token']
            and (run/'recoverable_latest.json').read_bytes()==latest_bytes):break
    if not latest['complete_checkpoint'] or binding['source_packet_sha256']!=packet_sha:raise ValueError('Incomplete or mismatched actual state')
    scope=json.loads(Path(scope).read_text('utf8'));source=scope['source'];shot=next(s for s in scope['shots'] if s['id']=='s05')
    first,last=shot['first_pts'],shot['last_pts'];expected=(last-first)//256+1
    if expected!=shot['native_frame_count'] or source['native_fps']!='50/1' or source['time_base']!='1/12800':raise ValueError('Source clock changed')
    if file_sha256(source['path'])!=source['sha256']:raise ValueError('Source video bytes changed')
    model=json.loads(model_bytes);mi=next(i for i,m in enumerate(model['materials']) if m['label']=='ice.C0:outer')
    material=model['materials'][mi]
    if len(material['paths'])!=1 or material['paths'][0]['radius_index']!=2:raise ValueError('C0 must be the paint midline')
    scene=NativeScene(as_event_view(model,'s05'));world=scene.world(mi,0,np.linspace(0,1,2049))
    graph=CameraGraph(graph_from_joint(model));reader=PacketArrays(packet)
    uv=reader.load('uv');pts=reader.load('native_pts');mat=reader.load('material_index');sid=reader.load('shot_index')
    admitted=(reader.load('admission')==1)&(reader.load('fit_role')==1)
    ids=np.flatnonzero(admitted&(mat==mi)&(sid==reader.packet['shot_ids'].index('s05')))
    by_pts={}
    for i in ids:
        stamp=int(pts[i])
        if not first<=stamp<=last or (stamp-first)%256:raise ValueError('C0 source clock mismatch')
        by_pts.setdefault(stamp,[]).append(uv[i].copy())
    out.mkdir(parents=True);snapshot=out/'model.json';snapshot.write_bytes(model_bytes)
    (out/'complete_state_receipt.json').write_bytes(latest_bytes)
    (out/'observed_row_initialization_state_binding.json').write_bytes(binding_bytes)
    install(graph,snapshot,out/'acceleration',world,Path(compiled_dir),shots=['s05'])
    camera=graph.native_camera('s05');encoded=out/'encoded.mp4';count=0;cv2.setNumThreads(1)
    with av.open(source['path'])as media,av.open(str(encoded),'w')as sink,gzip.open(out/'frames.jsonl.gz','wt',encoding='utf8')as ledger:
        stream=media.streams.video[0];stream.thread_count=2;media.seek(first,stream=stream,backward=True)
        target=sink.add_stream('libx264',rate=Fraction(50,1));target.width=1920;target.height=1080;target.pix_fmt='yuv420p'
        target.time_base=Fraction(1,12800);target.codec_context.thread_count=2;target.options={'crf':'18','preset':'fast'}
        for frame in media.decode(stream):
            stamp=int(frame.pts)
            if stamp<first:continue
            if stamp>last:break
            if stamp!=first+count*256:raise ValueError('Missing native source frame')
            image=frame.to_ndarray(format='bgr24');local=float(graph.local_seconds('s05',np.array([stamp],np.int64))[0])
            projected,valid,equation=fast_native_project(camera,world,local)
            valid&=np.isfinite(projected).all(axis=1)&(np.abs(projected)<1e8).all(axis=1)
            visible=np.flatnonzero(valid)
            for segment in np.split(visible,np.flatnonzero(np.diff(visible)>1)+1):
                if len(segment)>1:cv2.polylines(image,[np.rint(projected[segment]).astype(np.int32).reshape(-1,1,2)],False,(0,0,255),1,cv2.LINE_AA)
            points=by_pts.get(stamp,[])
            for xy in points:cv2.circle(image,tuple(np.rint(xy).astype(int)),2,(255,255,0),1,cv2.LINE_AA)
            result=av.VideoFrame.from_ndarray(image,format='bgr24');result.pts=stamp-first;result.time_base=Fraction(1,12800)
            for part in target.encode(result):sink.mux(part)
            ledger.write(json.dumps(dict(native_pts=stamp,current_C0_source_points=len(points),other_points=0,other_model_paths=0))+'\n')
            count+=1
            if count==1 or count%100==0 or count==expected:
                progress=dict(frames=count,total=expected,seconds=time.perf_counter()-started)
                (out/'progress.json').write_text(json.dumps(progress),'utf8');print(json.dumps(progress),flush=True)
        for part in target.encode(None):sink.mux(part)
    if count!=expected:raise ValueError('Incomplete S05')
    executable=shutil.which('ffmpeg')
    if executable is None:
        import imageio_ffmpeg
        executable=imageio_ffmpeg.get_ffmpeg_exe()
    video=out/'s05_C0_only.mp4'
    command=[str(executable),'-hide_banner','-nostdin','-y','-i',str(encoded),'-ss',f'{first/12800:.8f}','-i',source['path'],'-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-t',f'{expected/50:.8f}','-movflags','+faststart',str(video)]
    with (out/'audio_mux.log').open('w',encoding='utf8')as log:subprocess.run(command,check=True,stdout=log,stderr=log)
    decoded=0
    with av.open(str(video))as media:
        for frame in media.decode(video=0):
            if Fraction(frame.pts)*frame.time_base!=Fraction(decoded,50):raise ValueError('Output native time discontinuity')
            decoded+=1
    if decoded!=expected:raise ValueError('Output frame count mismatch')
    encoded.unlink()
    report=dict(schema='rinkfit.C0_midline_only_native_review.v1',video=str(video),frames=count,source_seconds_inclusive=[first/12800,last/12800],source_pts_inclusive=[first,last],native_fps='50/1',source_packet=dict(file=str(packet),sha256=packet_sha),model_sha256=model_sha,state_token=latest['state_token'],global_iteration=latest['global_iteration'],model_material_index=mi,historical_label=material['label'],physical_definition='paint_width_midline',radius_index=2,model_color='red',source_dot_color='cyan',current_source_points_total=len(ids),other_points=0,other_model_paths=0,top_view=False,added_information_overlay=False,quality_accepted=False,seconds=time.perf_counter()-started)
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','packet','scope','output','compiled-dir'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args());render(**{k.replace('-','_'):v for k,v in a.items()})
if __name__=='__main__':main()
