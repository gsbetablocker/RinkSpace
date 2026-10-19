"""Orbit/overlay opening with lossless packet reuse of the completed main shot.

Only the new opening is encoded. Frozen first_shot_v2 H264 packet payloads are
copied unchanged after a matching-codec opening. Fresh source audio is delayed
using continuous1024-sample AAC framing. Model, body, and main effects unchanged.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import av
import cv2
import numpy as np
from calibration.program_tracking.render_demo_last_shot import DATA,BASE_OUT,WH,FPS,decode,digest,write_json
from calibration.program_tracking.demo_particle_pass import _encoder_quality
from calibration.program_tracking.render_result import _ffmpeg_executable,_available_encoder
from calibration.program_tracking.demo_source import check_output_path
FIRST,LAST=256256,575488
MAIN_FRAMES=(LAST-FIRST)//256+1
PARENT_OUT=DATA/'results/demo_intro_20261016'
OUT=PARENT_OUT/'v2'
MAIN_SOURCE=PARENT_OUT/'first_shot_v2.mp4'
INTRO_VIDEO=OUT/'intro_only.mp4'
JOINED_VIDEO=OUT/'joined_video.mp4'
FINAL_VIDEO=OUT/'RinkSpace_intro_orbit_overlay.mp4'
PROFILE='orbit_overlay'
TICK_STEP=256
TIME_BASE=Fraction(1,12800)


def configure(version='v2'):
    """Select an isolated opening profile/output; immutable main stays fixed."""
    global OUT,INTRO_VIDEO,JOINED_VIDEO,FINAL_VIDEO,PROFILE
    profiles={'v2':'orbit_overlay','v3':'continuous_orbit_overlay'}
    if version not in profiles:raise ValueError('Unknown opening output version')
    OUT=PARENT_OUT/version
    PROFILE=profiles[version]
    INTRO_VIDEO=check_output_path(OUT/'intro_only.mp4')
    JOINED_VIDEO=check_output_path(OUT/'joined_video.mp4')
    FINAL_VIDEO=check_output_path(OUT/'RinkSpace_intro_orbit_overlay.mp4')
    return dict(version=version,profile=PROFILE,output=str(OUT),main_source=str(MAIN_SOURCE))


def out(name):
    return check_output_path(OUT/name)


def reference(file):
    return dict(path=str(file),sha256=digest(file))


def intro_state():
    from calibration.program_tracking.demo_intro import IntroTransform
    with np.load(BASE_OUT/'source_packet.npz',allow_pickle=False) as z:
        indices=np.flatnonzero(z['pts']==FIRST)
        if len(indices)!=1:raise ValueError('Official first source PTS unavailable')
        camera=z['camera25'][int(indices[0])].copy()
        source=str(z['source_path'])
        scene_path=str(z['scene_path'])
    scene=json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
    native=next(decode(source,FIRST,FIRST))[1]
    mask_file=DATA/'results/frame_by_frame_review_20261013/opening_100_50fps/masks_current/000001.png'
    mask=cv2.imread(str(mask_file),cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape!=(WH[1],WH[0]):raise ValueError('Official source first person mask missing')
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(np.float32)/255.
    mask=cv2.GaussianBlur(mask,(0,0),.9)
    intro=IntroTransform(scene,camera,output_wh=WH,profile=PROFILE,fps=FPS)
    frames=int(intro.total_frames)
    if frames<1 or abs(frames/FPS-float(intro.total_seconds))>1e-8:
        raise ValueError('Opening duration must align to the native50 frame grid')
    return intro,native,mask,source,scene_path,mask_file


def _frame(intro,native,mask,elapsed):
    image=intro.frame(float(elapsed),native,first_foreground=mask)
    if image.shape!=(WH[1],WH[0],3) or image.dtype!=np.uint8:
        raise ValueError('Opening must return uint8 BGR1920x1080')
    return image


def preview():
    OUT.mkdir(parents=True,exist_ok=True)
    intro,native,mask,_,_,_=intro_state()
    duration=float(intro.total_seconds)
    base=[0.,.7,1.2,2.2,3.3,4.6,6.2,6.85,7.3,7.75,duration-.12,duration-.02]
    times=[float(np.clip(t,0.,duration-.02)) for t in base]
    tiles=[]
    for t in times:
        image=_frame(intro,native,mask,t)
        tile=cv2.resize(image,(480,270),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'intro +{t:.2f}s',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
        tiles.append(tile)
    if not cv2.imwrite(str(out('intro_preview.jpg')),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,12,4)])):
        raise IOError('Cannot save opening preview')
    overlay_time=min(7.3,duration-.02)
    if not cv2.imwrite(str(out('overlay_7p3.jpg')),_frame(intro,native,mask,overlay_time),[cv2.IMWRITE_JPEG_QUALITY,96]):
        raise IOError('Cannot save source/grid overlap preview')
    exact=bool(np.array_equal(_frame(intro,native,mask,duration),native))
    if not exact:raise ValueError('Intro endpoint must exactly equal original source first frame')
    report=dict(profile=PROFILE,output_version=OUT.name,preview=str(out('intro_preview.jpg')),overlay_preview=str(out('overlay_7p3.jpg')),
                elapsed_seconds=times,overlay_elapsed_seconds=overlay_time,
                intro_seconds=duration,intro_frames=intro.total_frames,
                endpoint_source_pixels_exact=exact,intro_selfcheck=intro.selfcheck(),
                parameters=intro.parameters(),no_video_generated=True,
                main_source_unmodified=str(MAIN_SOURCE))
    write_json(out('preview_report.json'),report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


# Remaining export/verification functions are defined below before CLI dispatch.


def progress(stage,frames,total,started):
    value=dict(status='running',stage=stage,frames=frames,total=total,
               elapsed_seconds=time.perf_counter()-started)
    write_json(out('render_status.json'),value)
    print(json.dumps(value),flush=True)


def _check_video(stream):
    if stream.average_rate!=FPS or (stream.width,stream.height)!=WH or stream.time_base!=TIME_BASE:
        raise ValueError('Copy segment must be1920x1080 at50fps with1/12800 time base')
    if stream.codec_context.name!='h264':raise ValueError('Copy segment must be H264')


def _contains_idr(payload,extradata):
    if len(extradata)<5 or extradata[0]!=1:raise ValueError('Expected avcC length-prefixed H264')
    length_bytes=(extradata[4]&3)+1
    index=0
    while index+length_bytes<=len(payload):
        length=int.from_bytes(payload[index:index+length_bytes],'big')
        index+=length_bytes
        if length<1 or index+length>len(payload):raise ValueError('Malformed length-prefixed H264 sample')
        if payload[index]&31==5:return True
        index+=length
    return False


def _video_packets(file):
    """Read packet identities without decoding or changing sample payloads."""
    identities=[]
    with av.open(str(file)) as container:
        stream=container.streams.video[0]
        _check_video(stream)
        extradata=bytes(stream.codec_context.extradata)
        previous=None
        for packet in container.demux(stream):
            if packet.size<=0:continue
            if packet.pts is None or packet.dts is None:raise ValueError('Video packet timestamp absent')
            if previous is not None and packet.dts-previous!=TICK_STEP:
                raise ValueError('Video DTS is not strictly continuous20ms')
            previous=int(packet.dts)
            if int(packet.duration)!=TICK_STEP:raise ValueError('Video packet duration is not20ms')
            if not identities and not _contains_idr(bytes(packet),extradata):
                raise ValueError('Independent copied video segment does not start with an actual H264 IDR slice')
            identities.append((int(packet.pts),int(packet.dts),int(packet.duration),
                               hashlib.sha256(bytes(packet)).hexdigest(),bool(packet.is_keyframe)))
    return extradata,identities


def _sequence_hash(identities):
    h=hashlib.sha256()
    for item in identities:h.update(bytes.fromhex(item[3]))
    return h.hexdigest()


def encode_intro(intro,native,mask):
    started=time.perf_counter()
    ffmpeg=_ffmpeg_executable()
    args,name=_available_encoder(ffmpeg)
    args=_encoder_quality(args,name)
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','bgr24','-video_size','1920x1080','-framerate','50','-i','pipe:0',
             *args,'-pix_fmt','yuv420p','-an','-frames:v',str(intro.total_frames),
             '-video_track_timescale','12800','-movflags','+faststart',str(out(INTRO_VIDEO.name))]
    write_json(out('intro_command.json'),command)
    count=0
    with out('intro_encode.stderr.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
        try:
            for index in range(int(intro.total_frames)):
                process.stdin.write(_frame(intro,native,mask,index/FPS).tobytes())
                count+=1
                if count%25==0:progress('intro_encode',count,intro.total_frames,started)
        finally:
            process.stdin.close()
            return_code=process.wait()
        if return_code:raise RuntimeError('Intro encoder failed: '+out('intro_encode.stderr.log').read_text(errors='replace')[-3000:])
    extra,packets=_video_packets(INTRO_VIDEO)
    if len(packets)!=int(intro.total_frames):raise ValueError('Intro encoded packet count differs')
    if sorted(p[0] for p in packets)!=list(range(0,count*TICK_STEP,TICK_STEP)):
        raise ValueError('Intro presentation PTS lost native50 grid')
    return dict(video=reference(INTRO_VIDEO),frames=count,seconds=count/FPS,
                encoder=name,encoder_args=args,extradata_sha256=hashlib.sha256(extra).hexdigest(),
                wall_seconds=time.perf_counter()-started)


def join_video_packets(intro_frames):
    """Losslessly join codec-identical independent IDR segments with exact ticks."""
    intro_extra,intro_packets=_video_packets(INTRO_VIDEO)
    main_extra,main_packets=_video_packets(MAIN_SOURCE)
    if intro_extra!=main_extra:
        raise ValueError('H264 SPS/PPS/avcC differs between intro and main. Stop and report; no fallback reencode authorized.')
    if len(intro_packets)!=int(intro_frames) or len(main_packets)!=MAIN_FRAMES:
        raise ValueError('Join source frame counts differ from requested complete segments')
    if not intro_packets[0][4] or not main_packets[0][4] or main_packets[0][0]!=0:
        raise ValueError('Both independent copy segments must start with IDR atPTS0')
    offset=int(intro_frames)*TICK_STEP
    count=0
    previous=None
    with av.open(str(INTRO_VIDEO)) as first, av.open(str(JOINED_VIDEO),'w',format='mp4',
            options={'movflags':'+faststart','video_track_timescale':'12800'}) as target:
        output_stream=target.add_stream_from_template(first.streams.video[0])
        output_stream.time_base=TIME_BASE
        for file,shift in ((INTRO_VIDEO,0),(MAIN_SOURCE,offset)):
            with av.open(str(file)) as source:
                stream=source.streams.video[0]
                _check_video(stream)
                for packet in source.demux(stream):
                    if packet.size<=0:continue
                    packet.pts=int(packet.pts)+shift
                    packet.dts=int(packet.dts)+shift
                    if previous is not None and packet.dts-previous!=TICK_STEP:
                        raise ValueError('B-frame/GOP boundary gives non-continuous20ms DTS; stop without timestamp repair')
                    previous=int(packet.dts)
                    packet.stream=output_stream
                    packet.time_base=TIME_BASE
                    target.mux(packet)
                    count+=1
    _,joined=_video_packets(JOINED_VIDEO)
    if len(joined)!=int(intro_frames)+MAIN_FRAMES or count!=len(joined):
        raise ValueError('Joined output packet count differs')
    copied=[(p[0]-offset,p[1]-offset,p[2],p[3],p[4]) for p in joined[int(intro_frames):]]
    if copied!=main_packets:
        raise ValueError('Main copied payload/PTS/DTS changed unexpectedly')
    if joined[:int(intro_frames)]!=intro_packets:
        raise ValueError('Intro copied payload/PTS/DTS changed unexpectedly')
    if sorted(p[0] for p in joined)!=list(range(0,len(joined)*TICK_STEP,TICK_STEP)):
        raise ValueError('Joined display PTS lost native50 grid')
    report=dict(video=reference(JOINED_VIDEO),frames=count,intro_frames=int(intro_frames),main_frames=MAIN_FRAMES,
                main_source=reference(MAIN_SOURCE),main_packet_payload_sequence_sha256=_sequence_hash(main_packets),
                copied_main_payload_sequence_sha256=_sequence_hash(copied),main_payload_and_shifted_pts_dts_exact=True,
                main_pts_dts_shift_ticks=offset,time_base='1/12800',dts_interval_ticks=TICK_STEP,
                extradata_identical=True,extradata_sha256=hashlib.sha256(main_extra).hexdigest())
    write_json(out('join_report.json'),report)
    return report


def mux_audio(intro_frames,source):
    duration=(int(intro_frames)+MAIN_FRAMES)/FPS
    delay_ms=round(int(intro_frames)*1000/FPS)
    ffmpeg=_ffmpeg_executable()
    audio_filter=(f'atrim=duration={MAIN_FRAMES/FPS},asetpts=PTS-STARTPTS,'
                  f'adelay={delay_ms}:all=1,asetnsamples=n=1024:p=1,asetpts=N/SR/TB')
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(JOINED_VIDEO),
             '-ss',str(FIRST/12800.),'-i',source,'-map','0:v:0','-map','1:a:0?',
             '-c:v','copy','-c:a','aac','-b:a','192k','-af',audio_filter,
             '-t',str(duration),'-video_track_timescale','12800','-movflags','+faststart',str(out(FINAL_VIDEO.name))]
    write_json(out('final_mux_command.json'),command)
    with out('final_mux.stderr.log').open('wb') as log:
        result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
    if result.returncode:raise RuntimeError('Audio mux failed: '+out('final_mux.stderr.log').read_text(errors='replace')[-3000:])
    _,joined=_video_packets(JOINED_VIDEO)
    _,final=_video_packets(FINAL_VIDEO)
    if final!=joined:raise ValueError('Final audio mux changed copied video payload or clock')
    return dict(source_audio_delay_milliseconds=delay_ms,audio_filter=audio_filter,
                video_copy_payload_pts_dts_exact=True,total_frames=len(final),duration_seconds=duration)


def render():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    intro,native,mask,source,scene_path,mask_file=intro_state()
    selfcheck=intro.selfcheck()
    if not np.array_equal(_frame(intro,native,mask,float(intro.total_seconds)),native):
        raise ValueError('Intro does not end in exact source first frame')
    progress('intro_prepare',0,intro.total_frames,started)
    encoded=encode_intro(intro,native,mask)
    progress('packet_join',intro.total_frames,intro.total_frames+MAIN_FRAMES,started)
    joined=join_video_packets(intro.total_frames)
    progress('audio_mux',intro.total_frames+MAIN_FRAMES,intro.total_frames+MAIN_FRAMES,started)
    audio=mux_audio(intro.total_frames,source)
    files=dict(source_packet=BASE_OUT/'source_packet.npz',scene=Path(scene_path),first_person_mask=mask_file,
               main_source=MAIN_SOURCE,render_code=Path(__file__),
               intro_code=Path(__file__).with_name('demo_intro.py'))
    report=dict(status='complete',video=reference(FINAL_VIDEO),profile=PROFILE,output_version=OUT.name,
                intro_frames=intro.total_frames,intro_seconds=intro.total_frames/FPS,
                main_frames=MAIN_FRAMES,total_frames=intro.total_frames+MAIN_FRAMES,
                duration_seconds=(intro.total_frames+MAIN_FRAMES)/FPS,fps=FPS,width=WH[0],height=WH[1],
                source=source,source_first_pts=FIRST,source_last_pts=LAST,source_time_base='1/12800',
                original_source_audio=True,main_rendered_again=False,main_payload_copied_exact=True,
                main_effects_modified=False,solver_modified=False,body_fit_run=False,
                new_music=False,new_text=False,new_ending=False,
                parameters=intro.parameters(),intro_selfcheck=selfcheck,
                parameter_sources={k:reference(v) for k,v in files.items()},
                intro_encode=encoded,join=joined,audio_mux=audio,
                wall_seconds=time.perf_counter()-started)
    write_json(out('final_report.json'),report)
    write_json(out('render_status.json'),report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def _verify_audio(file,intro_seconds,total_seconds):
    with av.open(str(file)) as container:
        if len(container.streams.audio)!=1:raise ValueError('One original programme audio stream required')
        stream=container.streams.audio[0]
        rate=int(stream.rate)
        if rate!=48000:raise ValueError('Original audio must remain48kHz')
        packets=[]
        for packet in container.demux(stream):
            if packet.size<=0:continue
            if packet.pts is None or packet.dts is None:raise ValueError('AAC packet timestamp missing')
            scaled=[Fraction(v)*stream.time_base*rate for v in (packet.pts,packet.dts,packet.duration)]
            if any(v.denominator!=1 for v in scaled):raise ValueError('AAC timestamps leave the sample grid')
            packets.append(tuple(int(v) for v in scaled))
    if not packets:raise ValueError('AAC stream empty')
    max_packet_gap=0
    max_packet_step=0
    for i,(pts,dts,duration) in enumerate(packets):
        if not 0<duration<=1024 or (i<len(packets)-1 and duration!=1024):
            raise ValueError(f'AAC packet{i} duration{duration} is not1024 samples')
        if i:
            steps=(pts-packets[i-1][0],dts-packets[i-1][1])
            max_packet_step=max(max_packet_step,*steps)
            max_packet_gap=max(max_packet_gap,*(abs(step-1024) for step in steps))
            if steps!=(1024,1024):raise ValueError(f'AAC packet clock gap{i}: {steps}')
    decoded_frames=0;decoded_samples=0;max_sample_gap=0
    previous_start=None;previous_samples=None
    silent_peak=0.;silent_squares=0.;silent_values=0;silent_samples=0
    main_peak=0.;main_squares=0.;main_values=0
    with av.open(str(file)) as container:
        for frame in container.decode(container.streams.audio[0]):
            if frame.pts is None:raise ValueError('Decoded audio frame timestamp missing')
            exact=Fraction(frame.pts)*frame.time_base*frame.sample_rate
            if exact.denominator!=1:raise ValueError('Decoded audio leaves integral sample grid')
            start=int(exact)
            expected=0 if previous_start is None else previous_start+previous_samples
            gap=start-expected
            max_sample_gap=max(max_sample_gap,abs(gap))
            if gap:raise ValueError(f'Decoded audio sample gap at frame{decoded_frames}: {gap}')
            previous_start,previous_samples=start,int(frame.samples)
            decoded_frames+=1;decoded_samples+=int(frame.samples)
            raw=frame.to_ndarray()
            values=raw.astype(np.float64)
            if np.issubdtype(raw.dtype,np.integer):values/=np.iinfo(raw.dtype).max
            if values.ndim==1:values=values[None,:]
            if values.shape[-1]!=frame.samples:values=values.reshape(-1,frame.samples)
            times=start/frame.sample_rate+np.arange(frame.samples)/frame.sample_rate
            silence=(times>=0.)&(times<intro_seconds-.02)
            if silence.any():
                value=values[:,silence]
                silent_peak=max(silent_peak,float(np.abs(value).max()))
                silent_squares+=float(np.sum(value*value));silent_values+=value.size
                silent_samples+=int(silence.sum())
            main=(times>=intro_seconds)&(times<total_seconds)
            if main.any():
                value=values[:,main]
                main_peak=max(main_peak,float(np.abs(value).max()))
                main_squares+=float(np.sum(value*value));main_values+=value.size
    required=math.floor((intro_seconds-.02)*rate)
    silent_rms=math.sqrt(silent_squares/max(silent_values,1))
    main_rms=math.sqrt(main_squares/max(main_values,1))
    if silent_samples<required or silent_peak>1e-3 or silent_rms>1e-5 or main_rms<1e-5:
        raise ValueError(f'Audio silence/coverage failed: samples={silent_samples}/{required},peak={silent_peak},rms={silent_rms},main={main_rms}')
    return dict(sample_rate=rate,packet_count=len(packets),packet_first_pts_samples=packets[0][0],
                packet_last_pts_samples=packets[-1][0],packet_max_step_samples=max_packet_step,
                packet_max_gap_samples=max_packet_gap,packet_max_gap_seconds=max_packet_gap/rate,
                decoded_frames=decoded_frames,decoded_samples_per_channel=decoded_samples,
                decoded_max_gap_samples=max_sample_gap,decoded_max_gap_seconds=max_sample_gap/rate,
                intro_silent_samples_per_channel=silent_samples,required_intro_silent_samples_per_channel=required,
                intro_silence_verified_to_seconds=intro_seconds-.02,
                intro_audio_peak=silent_peak,intro_audio_rms=silent_rms,
                programme_audio_peak=main_peak,programme_audio_rms=main_rms,
                silence_limits=dict(peak=1e-3,rms=1e-5),main_minimum_rms=1e-5)


def verify():
    report=json.loads(out('final_report.json').read_text(encoding='utf8'))
    intro_frames=int(report['intro_frames'])
    total=intro_frames+MAIN_FRAMES
    offset=intro_frames*TICK_STEP
    _,main_packets=_video_packets(MAIN_SOURCE)
    _,final_packets=_video_packets(FINAL_VIDEO)
    if len(main_packets)!=MAIN_FRAMES or len(final_packets)!=total:
        raise ValueError('Final/main video packet count differs')
    copied=[(v[0]-offset,v[1]-offset,v[2],v[3],v[4]) for v in final_packets[intro_frames:]]
    if copied!=main_packets:
        raise ValueError('Final main-segment payload or shifted packet timestamps differ from immutable first_shot_v2')
    if sorted(v[0] for v in final_packets)!=list(range(0,total*TICK_STEP,TICK_STEP)):
        raise ValueError('Video packet PTS are not the complete native50 presentation grid')
    indices=sorted(set([0,round(.7*FPS),round(1.2*FPS),round(2.2*FPS),round(3.3*FPS),
                        round(4.6*FPS),round(6.85*FPS),round(min(7.3,intro_frames/FPS-.02)*FPS),
                        intro_frames-1,intro_frames,intro_frames+500,total-1]))
    pictures={};count=0
    with av.open(str(FINAL_VIDEO)) as container:
        stream=container.streams.video[0]
        _check_video(stream)
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None or Fraction(frame.pts)*frame.time_base!=Fraction(index,FPS):
                raise ValueError(f'Final decoded video clock mismatch at{index}')
            if index in indices:
                tile=cv2.resize(frame.to_ndarray(format='bgr24'),(480,270),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f'{index/FPS:.2f}s frame{index}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,
                            (245,245,245),1,cv2.LINE_AA)
                pictures[index]=tile
            count+=1
    if count!=total:raise ValueError('Decoded final video does not contain complete intro and1248-frame main')
    audio=_verify_audio(FINAL_VIDEO,intro_frames/FPS,total/FPS)
    tiles=[pictures[index] for index in indices]
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    if not cv2.imwrite(str(out('encoded_review.jpg')),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])):
        raise IOError('Cannot save final encoded review')
    result=dict(status='verified',video=reference(FINAL_VIDEO),profile=PROFILE,output_version=OUT.name,
                fps=50,decoded_frames=count,
                video_packets=len(final_packets),video_interval_seconds=.02,
                intro_frames=intro_frames,intro_seconds=intro_frames/FPS,
                main_frames=MAIN_FRAMES,total_seconds=total/FPS,
                main_packet_payload_sequence_sha256=_sequence_hash(main_packets),
                final_main_payload_sequence_sha256=_sequence_hash(copied),
                all_main_packet_payloads_equal=True,all_main_pts_dts_equal_after_offset=True,
                main_source=reference(MAIN_SOURCE),source_first_pts=FIRST,source_last_pts=LAST,
                source_time_base='1/12800',source_native_pts_step=256,
                audio=audio,encoded_review=str(out('encoded_review.jpg')),
                solver_modified=False,main_rendered_again=False)
    write_json(out('verification.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['v2','v3'],default='v2',
                        help='v2=orbit then travel; v3=continuous simultaneous orbit/travel')
    selected=parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--preview',action='store_true')
    selected.add_argument('--render-only',action='store_true')
    selected.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    configure(args.version)
    cv2.setNumThreads(2)
    if args.preview:return preview()
    if args.render_only:return render()
    return verify()


if __name__=='__main__':
    main()
