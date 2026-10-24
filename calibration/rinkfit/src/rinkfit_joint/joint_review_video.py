"""Render one complete native shot from an actual saved solver state."""
from pathlib import Path
from fractions import Fraction
import argparse, gzip, json, os, shutil, subprocess, time
import av, cv2, numpy as np
from .io_utils import file_sha256
from .joint_model import graph_from_joint, as_event_view
from .joint_replay import inventory, fast_native_project, render_visibility, draw_fast
from .joint_display import CurrentSourcePoints, SharedModelTopView
from .joint_review_camera import CurrentJointCameras
from .joint_acceleration import install
from rinkfit_start.native_scene import NativeScene


def _checked_current_ref(reference, base):
    path=Path(reference['file']);path=path if path.is_absolute() else Path(base)/path
    path=path.resolve()
    if file_sha256(path)!=reference['sha256']:raise ValueError('Current video input bytes changed: '+str(path))
    return path


def _current_review_inputs(candidate, packet, binding, model_sha, current_input=None):
    """Bind current bytes and their logical source separately, without history."""
    candidate=Path(candidate).resolve();packet=Path(packet).resolve()
    transport_file=candidate/'current_checkpoint.json';transport_bytes=transport_file.read_bytes()
    transport=json.loads(transport_bytes)
    if transport.get('schema')!='rinkfit.current_complete_primal_transport.v1':
        raise ValueError('Video needs a current complete checkpoint transport')
    model_file=_checked_current_ref(transport['model'],candidate)
    if model_file!=candidate/'model.json' or transport['model']['sha256']!=model_sha:
        raise ValueError('Current checkpoint and displayed model differ')
    if transport['from_state_token']!=binding['state_token']:
        raise ValueError('Current checkpoint and displayed complete state token differ')
    manifest_ref=transport['current_input_v1'];manifest_file=_checked_current_ref(manifest_ref,candidate)
    if current_input is not None:
        supplied=(Path(current_input).resolve() if isinstance(current_input,(str,Path))
                  else _checked_current_ref(current_input,Path.cwd()))
        if supplied!=manifest_file:raise ValueError('Video current input differs from complete checkpoint')
    current=json.loads(manifest_file.read_bytes())
    if current.get('schema')!='rinkfit.current_joint_input.v1':raise ValueError('Unsupported current video input')
    actual_packet=_checked_current_ref(current['packet'],manifest_file.parent)
    if actual_packet!=packet:raise ValueError('Displayed packet differs from current input actual bytes')
    logical=current['logical_source_id']
    if binding['source_packet_sha256']!=logical:
        raise ValueError('Displayed state and current logical source differ')
    if transport['raw_rows']!=current['raw_rows']:
        raise ValueError('Current checkpoint and source raw denominator differ')
    checkpoint=_checked_current_ref(transport['checkpoint'],candidate)
    from .joint_review_packet import CurrentReviewPacket
    manifest_ref=dict(file=str(manifest_file),sha256=transport['current_input_v1']['sha256'])
    reader=CurrentReviewPacket(packet,current_input=manifest_ref)
    if transport_file.read_bytes()!=transport_bytes:raise RuntimeError('Current checkpoint changed during review binding')
    return reader,dict(current_input_v1=manifest_ref,logical_source_id=logical,
        actual_source_packet=dict(file=str(packet),sha256=reader.packet_sha),
        complete_checkpoint=dict(file=str(transport_file),sha256=file_sha256(transport_file)),
        engineering_checkpoint=dict(file=str(checkpoint),sha256=transport['checkpoint']['sha256']),
        state_token=binding['state_token'],raw_rows=current['raw_rows'],historical_source_replay=False)


def render(*, candidate, packet, scope, shot, output, compiled_dir, gpu_dll=None, gpu_validation=None, ffmpeg=None, current_input=None):
    started=time.perf_counter();candidate=Path(candidate).resolve();packet=Path(packet).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Immutable review output already exists: '+str(output))
    scope=json.loads(Path(scope).read_text('utf8'));item=next(s for s in scope['shots'] if s['id']==shot)
    source=scope['source'];first,last=int(item['first_pts']),int(item['last_pts'])
    if source['native_fps']!='50/1' or source['time_base']!='1/12800' or (last-first)%256:raise ValueError('Native source clock changed')
    binding_file=candidate/'observed_row_initialization_state_binding.json';binding_bytes=binding_file.read_bytes();binding=json.loads(binding_bytes)
    model_file=candidate/'model.json';model_sha=file_sha256(model_file);model_bytes=model_file.read_bytes()
    if file_sha256(model_file)!=model_sha or binding_file.read_bytes()!=binding_bytes:raise RuntimeError('State changed during snapshot; retry latest complete state')
    reader,current_binding=_current_review_inputs(candidate,packet,binding,model_sha,current_input)
    if source['sha256']!=file_sha256(source['path']):raise ValueError('Official native source bytes changed')
    model=json.loads(model_bytes)
    clock=model['source_clock']
    if (clock['source_video_sha256']!=source['sha256'] or clock['time_base']!=[1,12800]
            or clock['frame_step_pts']!=256 or clock['image_size']!=[1920,1080]):
        raise ValueError('Displayed model and official native source clock differ')
    output.mkdir(parents=True);snapshot=output/'model.json';snapshot.write_bytes(model_bytes)
    (output/'observed_row_initialization_state_binding.json').write_bytes(binding_bytes)
    if file_sha256(snapshot)!=model_sha:raise ValueError('Incomplete model snapshot')
    graph=CurrentJointCameras(model);world,records=inventory(model)
    scene=NativeScene(as_event_view(model,shot))
    from .joint_review_packet import source_point_reader
    with source_point_reader(reader):points=CurrentSourcePoints(packet,model,scope,shot_ids=[shot])
    top=SharedModelTopView(world,records,scene)
    acceleration=install(graph,snapshot,output/'acceleration',world,Path(compiled_dir),shots=[shot],gpu_dll=Path(gpu_dll) if gpu_dll else None,gpu_validation=Path(gpu_validation) if gpu_validation else None)
    camera=graph.native_camera(shot);expected=(last-first)//256+1;count=0;encoded=output/'encoded.mp4'
    cv2.setNumThreads(1)
    with av.open(source['path']) as media,av.open(str(encoded),'w') as sink,gzip.open(output/'frames.jsonl.gz','wt',encoding='utf8') as ledger:
        stream=media.streams.video[0]
        if Fraction(stream.time_base)!=Fraction(1,12800):raise ValueError('Actual native PTS time base differs')
        stream.thread_count=2;media.seek(first,stream=stream,backward=True)
        target=sink.add_stream('libx264',rate=Fraction(50,1));target.width=1920+top.width;target.height=1080;target.pix_fmt='yuv420p';target.time_base=Fraction(1,12800);target.codec_context.thread_count=2;target.options={'crf':'18','preset':'fast'}
        for frame in media.decode(stream):
            stamp=int(frame.pts)
            if stamp<first:continue
            if stamp>last:break
            if stamp!=first+256*count:raise ValueError('Missing native source exposure')
            raw=frame.to_ndarray(format='bgr24');local=float(graph.local_seconds(shot,np.array([stamp],np.int64))[0])
            uv,valid,eq=fast_native_project(camera,world,local);valid_indices=np.flatnonzero(valid)
            if len(valid_indices):
                subset=[dict(r,begin=int(np.searchsorted(valid_indices,r['begin'])),end=int(np.searchsorted(valid_indices,r['end']))) for r in records]
                visible=render_visibility(camera,scene,world[valid_indices],subset,local,uv[valid_indices])
                valid[valid_indices]&=visible
            image,metrics=draw_fast(raw,uv,valid,records,'');metrics.update(points.draw(image,stamp));map_image=top.frame(camera,local,shot)
            cv2.putText(map_image,f'source {stamp/12800:.3f}s PTS {stamp}',(22,90),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,245,245),1,cv2.LINE_AA)
            cv2.putText(map_image,'model '+model_sha[:16],(22,112),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,245,245),1,cv2.LINE_AA)
            canvas=np.concatenate((image,map_image),axis=1);result=av.VideoFrame.from_ndarray(canvas,format='bgr24');result.pts=stamp-first;result.time_base=Fraction(1,12800)
            for part in target.encode(result):sink.mux(part)
            ledger.write(json.dumps(dict(native_pts=stamp,source_seconds=stamp/12800,model_sha256=model_sha,**metrics),separators=(',',':'))+'\n')
            count+=1
            if count==1 or count%10==0 or count==expected:
                progress=dict(shot=shot,frames=count,total=expected,source_pts=stamp,model_sha256=model_sha,state_token=binding['state_token'],seconds=time.perf_counter()-started)
                temporary=output/'progress.tmp';temporary.write_text(json.dumps(progress),'utf8');os.replace(temporary,output/'progress.json')
                print(json.dumps(progress),flush=True)
        for part in target.encode(None):sink.mux(part)
    if count!=expected:raise ValueError('Incomplete native shot')
    executable=ffmpeg or shutil.which('ffmpeg')
    if executable is None:
        import imageio_ffmpeg
        executable=imageio_ffmpeg.get_ffmpeg_exe()
    video=output/(shot+'.mp4');duration=Fraction(expected,50)
    command=[str(executable),'-hide_banner','-nostdin','-y','-i',str(encoded),'-ss',f'{first/12800:.8f}','-i',source['path'],'-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-t',f'{float(duration):.8f}','-movflags','+faststart',str(video)]
    with (output/'audio_mux.log').open('w',encoding='utf8') as log:subprocess.run(command,check=True,stdout=log,stderr=log)
    decoded=0
    with av.open(str(video)) as media:
        for frame in media.decode(video=0):
            if Fraction(frame.pts)*frame.time_base!=Fraction(decoded,50):raise ValueError('Review output source-clock discontinuity')
            decoded+=1
    if decoded!=expected:raise ValueError('Review output is incomplete')
    encoded.unlink()
    report=dict(schema='rinkfit.current_native_shot_review.v1',video=str(video),source=source,source_seconds_inclusive=[first/12800,last/12800],source_pts_inclusive=[first,last],frames=count,fps='50/1',model_sha256=model_sha,parent_model=str(model_file),complete_state_token=binding['state_token'],source_packet=current_binding['actual_source_packet'],current_source_binding=current_binding,all_model_paths_enabled=True,same_world_top_view=True,source_frame_pixels_not_covered_by_metadata=True,fit_quality_accepted=False,seconds=time.perf_counter()-started,acceleration=acceleration)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','packet','scope','shot','output','compiled-dir'):p.add_argument('--'+key,required=True)
    for key in ('gpu-dll','gpu-validation','ffmpeg','current-input'):p.add_argument('--'+key)
    args=vars(p.parse_args());render(**{k.replace('-','_'):v for k,v in args.items()})

if __name__=='__main__':main()
