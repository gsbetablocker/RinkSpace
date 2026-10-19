"""Native50 preview: 5.2-second3Dintro plus the complete opening source shot.

Uses frozen demo camera/body presentation state, original native50 person masks,
seeded particle billboards, and original programme audio. No solver/fitting.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import av
import cv2
import numpy as np
from calibration.program_tracking.render_demo_last_shot import Sample,decode,digest,write_json,DATA,BASE_OUT,WH,FPS
from calibration.program_tracking.demo_particle_pass import ParticleOverlay,_encoder_quality
from calibration.program_tracking.demo_source import check_output_path
from calibration.program_tracking.render_result import _ffmpeg_executable,_available_encoder
FIRST,LAST=256256,575488
MAIN_FRAMES=(LAST-FIRST)//256+1
INTRO_SECONDS=5.2
INTRO_FRAMES=260
OUT=DATA/'results/demo_intro_20261016'
MASK_ROOT=DATA/'results/frame_by_frame_review_20261013'
MAIN_VIDEO=OUT/'first_shot_v2.mp4'
FINAL_VIDEO=OUT/'RinkSpace_intro_first_shot_PotPlayer.mp4'


def _out(name):
    return check_output_path(OUT/name)


def _progress(stage,frames,total,started):
    value={'status':'running','stage':stage,'frames':frames,'total':total,
           'elapsed_seconds':time.perf_counter()-started}
    write_json(_out(stage+'_status.json'),value)
    print(json.dumps(value),flush=True)


class IntroShotSample(Sample):
    """Sample plus official source masks and clear particle billboards."""
    def __init__(self):
        super().__init__(version='v2')
        self.particles=ParticleOverlay(self,first_pts=FIRST,last_pts=LAST)

    def foreground(self,index):
        pts=int(self.packet['pts'][int(index)])
        number=(pts-FIRST)//256+1
        if not 1<=number<=MAIN_FRAMES or pts!=FIRST+(number-1)*256:
            raise ValueError('Official first-shot mask requested outside exact source range')
        folder=('opening_100_50fps' if number<=100 else
                'continuation_101_200_50fps' if number<=200 else 'first_shot_remaining_50fps')
        file=MASK_ROOT/folder/'masks_current'/f'{number:06d}.png'
        mask=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=(WH[1],WH[0]):
            raise ValueError('Missing original1080p official person mask: '+str(file))
        mask=cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(np.float32)/255.
        mask=cv2.GaussianBlur(mask,(0,0),.9)
        mask[0:255,0:555]=1.
        mask[934:1010,1670:1880]=1.
        self.last_mask_label='official_native_person_mask'
        self.mask_counts[self.last_mask_label]=self.mask_counts.get(self.last_mask_label,0)+1
        return mask

    def draw(self,native,pts):
        rendered,base,texture,foreground=super().draw(native,int(pts))
        frame_index=(int(pts)-FIRST)//256
        rendered=self.particles.frame(rendered,frame_index,foreground=foreground)
        u=float(np.clip((int(pts)-FIRST)/12800./.5,0.,1.))
        u=u*u*(3.-2.*u)
        if u<1.:
            rendered=np.clip(native.astype(np.float32)*(1.-u)+rendered.astype(np.float32)*u+.5,0,255).astype(np.uint8)
        return rendered


def _encoder():
    ffmpeg=_ffmpeg_executable()
    args,name=_available_encoder(ffmpeg)
    return ffmpeg,_encoder_quality(args,name),name


def render_main(workers=2):
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    # Parent loads only the compact source packet; full Samples live in workers.
    with np.load(BASE_OUT/'source_packet.npz',allow_pickle=False) as z:
        source=str(z['source_path'])
    ffmpeg,args,name=_encoder()
    duration=MAIN_FRAMES/FPS
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','bgr24','-video_size','1920x1080','-framerate','50','-i','pipe:0',
             '-ss',str(FIRST/12800.),'-i',source,'-map','0:v:0','-map','1:a:0?',
             *args,'-pix_fmt','yuv420p','-t',str(duration),'-c:a','aac','-b:a','192k',
             '-af',f'atrim=duration={duration}','-movflags','+faststart',str(_out(MAIN_VIDEO.name))]
    write_json(_out('main_command.json'),command)
    count=0
    stream=None
    stats={}
    with _out('main_encode.stderr.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
        try:
            if int(workers)>1:
                from calibration.program_tracking.demo_pipeline import ordered_frames
                stream=ordered_frames(decode(source,FIRST,LAST),sample_kwargs={},workers=int(workers),
                    slots=int(workers)*3,sample_factory='calibration.program_tracking.render_demo_intro:IntroShotSample',
                    stats=stats)
            else:
                sample=IntroShotSample()
                stream=((pts,sample.draw(native,pts)) for pts,native in decode(source,FIRST,LAST))
            for pts,frame in stream:
                if pts!=FIRST+count*256:
                    raise ValueError('Source native50 PTS order changed')
                process.stdin.write(frame.tobytes())
                count+=1
                if count%100==0:_progress('main',count,MAIN_FRAMES,started)
            if count!=MAIN_FRAMES:raise ValueError('First complete shot endpoint absent')
        finally:
            if stream is not None and hasattr(stream,'close'):stream.close()
            process.stdin.close()
            return_code=process.wait()
        if return_code:raise RuntimeError('ffmpeg: '+_out('main_encode.stderr.log').read_text(errors='replace')[-3000:])
    report=dict(status='complete',video=str(MAIN_VIDEO),native_source_frames=count,encoded_frames=count,
                source=source,source_first_pts=FIRST,source_last_pts=LAST,source_time_base='1/12800',
                source_seconds=[FIRST/12800.,LAST/12800.],fps=FPS,width=WH[0],height=WH[1],
                duration_seconds=duration,first_effect_fade_seconds=.5,
                original_audio=True,original_speed=True,official_native_masks=True,
                style='v2 deep-red heat sigma1.5m plus seeded visible particle billboards',
                encoder=name,encoder_args=args,drawing_pipeline=stats,solver_modified=False,body_fit_run=False,
                parameters={n:dict(path=str(BASE_OUT/n),sha256=digest(BASE_OUT/n)) for n in ('source_packet.npz','contact_packet.npz')},
                implementation=dict(path=str(Path(__file__)),sha256=digest(__file__)),
                wall_seconds=time.perf_counter()-started)
    write_json(_out('main_report.json'),report)
    write_json(_out('main_status.json'),report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def _intro_state():
    # Lazy import keeps --main-only independent from the opening implementation.
    from calibration.program_tracking.demo_intro import IntroTransform
    with np.load(BASE_OUT/'source_packet.npz',allow_pickle=False) as z:
        source=str(z['source_path'])
        scene_path=str(z['scene_path'])
        positions=np.flatnonzero(z['pts']==FIRST)
        if len(positions)!=1:raise ValueError('Official opening source PTS missing')
        camera=z['camera25'][int(positions[0])].copy()
    scene=json.loads(Path(scene_path).read_text(encoding='utf-8-sig'))
    native=next(decode(source,FIRST,FIRST))[1]
    intro=IntroTransform(scene,camera,output_wh=WH,overview_seconds=1.,
                         travel_seconds=3.5,fade_seconds=.7,fps=50)
    if intro.total_frames!=INTRO_FRAMES or abs(intro.total_seconds-INTRO_SECONDS)>1e-8:
        raise ValueError('Intro duration must remain260 frames /5.2 seconds')
    return intro,native,source,scene_path


def _encoded_frames(file,expected_count):
    with av.open(str(file)) as container:
        stream=container.streams.video[0]
        if stream.average_rate!=FPS or (stream.width,stream.height)!=WH:
            raise ValueError('Main encoded stream must be1920x1080 native50 CFR')
        count=0
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None or Fraction(frame.pts)*frame.time_base!=Fraction(index,FPS):
                raise ValueError(f'Encoded native50 clock mismatch at frame{index}')
            yield index,frame.to_ndarray(format='bgr24')
            count+=1
        if count!=expected_count:raise ValueError('Encoded main frame count differs from complete opening shot')


def render_final():
    OUT.mkdir(parents=True,exist_ok=True)
    main_report=json.loads(_out('main_report.json').read_text(encoding='utf8'))
    if main_report.get('status')!='complete' or main_report.get('encoded_frames')!=MAIN_FRAMES:
        raise ValueError('Complete1248-frame main shot must finish before final-only')
    started=time.perf_counter()
    intro,first_native,source,scene_path=_intro_state()
    ffmpeg,args,name=_encoder()
    total_frames=INTRO_FRAMES+MAIN_FRAMES
    duration=total_frames/FPS
    audio_filter=f'atrim=duration={MAIN_FRAMES/FPS},asetpts=PTS-STARTPTS,adelay=5200:all=1,asetnsamples=n=1024:p=1,asetpts=N/SR/TB'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','bgr24','-video_size','1920x1080','-framerate','50','-i','pipe:0',
             '-ss',str(FIRST/12800.),'-i',source,'-map','0:v:0','-map','1:a:0?',
             *args,'-pix_fmt','yuv420p','-t',str(duration),'-c:a','aac','-b:a','192k',
             '-af',audio_filter,'-movflags','+faststart',str(_out(FINAL_VIDEO.name))]
    write_json(_out('final_command.json'),command)
    count=0
    with _out('final_encode.stderr.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
        try:
            for index in range(INTRO_FRAMES):
                frame=intro.frame(index/FPS,first_native)
                if frame.shape!=(WH[1],WH[0],3) or frame.dtype!=np.uint8:
                    raise ValueError('Intro frame must be uint8 BGR1920x1080')
                process.stdin.write(frame.tobytes())
                count+=1
                if count%50==0:_progress('final',count,total_frames,started)
            for index,frame in _encoded_frames(MAIN_VIDEO,MAIN_FRAMES):
                process.stdin.write(frame.tobytes())
                count+=1
                if count%100==0:_progress('final',count,total_frames,started)
            if count!=total_frames:raise ValueError('Final source/intro frame count mismatch')
        finally:
            process.stdin.close()
            return_code=process.wait()
        if return_code:raise RuntimeError('ffmpeg: '+_out('final_encode.stderr.log').read_text(errors='replace')[-3000:])
    files={
        'source_packet':BASE_OUT/'source_packet.npz','contact_packet':BASE_OUT/'contact_packet.npz',
        'scene':Path(scene_path),'main_video':MAIN_VIDEO,'main_report':_out('main_report.json'),
        'render_code':Path(__file__),'intro_code':Path(__file__).with_name('demo_intro.py'),
        'layers_code':Path(__file__).with_name('demo_layers.py'),
        'particle_code':Path(__file__).with_name('demo_particle_pass.py')}
    report=dict(status='complete',video=str(FINAL_VIDEO),fps=FPS,width=WH[0],height=WH[1],
                encoded_frames=count,intro_frames=INTRO_FRAMES,native_source_frames=MAIN_FRAMES,
                duration_seconds=duration,intro_seconds=INTRO_SECONDS,
                source=source,source_first_pts=FIRST,source_last_pts=LAST,source_time_base='1/12800',
                source_seconds=[FIRST/12800.,LAST/12800.],source_speed=1.,
                opening=dict(overview_seconds=1.,travel_seconds=3.5,fade_seconds=.7,
                             full_3d_ice_and_boards=True,advertisements=False,markings=False),
                intro_parameters=intro.parameters(),intro_selfcheck=intro.selfcheck(),
                first_effect_fade_seconds=.5,source_audio_delay_milliseconds=5200,
                intro_audio='silence',new_music=False,new_text=False,ending_added=False,
                original_audio=True,encoder=name,encoder_args=args,
                solver_modified=False,body_fit_run=False,new_source_fitting=False,
                metric_scale_is_conditional=True,
                parameter_sources={key:dict(path=str(file),sha256=digest(file)) for key,file in files.items()},
                video_sha256=digest(FINAL_VIDEO),wall_seconds=time.perf_counter()-started)
    write_json(_out('final_report.json'),report)
    write_json(_out('final_status.json'),report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def previews():
    OUT.mkdir(parents=True,exist_ok=True)
    intro,first_native,_,_=_intro_state()
    elapsed=[0.,1.,1.7,2.5,3.3,4.2,4.8,5.18]
    tiles=[]
    for value in elapsed:
        frame=intro.frame(value,first_native)
        tile=cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'intro +{value:.2f}s',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.58,
                    (245,245,245),1,cv2.LINE_AA)
        tiles.append(tile)
    target=_out('intro_preview.jpg')
    if not cv2.imwrite(str(target),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])])):
        raise IOError('Could not save intro preview')
    end=intro.frame(INTRO_SECONDS,first_native)
    if not np.array_equal(end,first_native):raise ValueError('Intro endpoint is not exact source first frame')
    report=dict(preview=str(target),elapsed_seconds=elapsed,
                endpoint_equals_original_first_frame=True,total_frames=INTRO_FRAMES,
                duration_seconds=INTRO_SECONDS,no_video_generated=True)
    write_json(_out('preview_report.json'),report)
    print(json.dumps(report),flush=True)
    return report


def verify():
    report=json.loads(_out('final_report.json').read_text(encoding='utf8'))
    total=INTRO_FRAMES+MAIN_FRAMES
    review_indices=[0,50,100,160,225,259,260,285,600,1000,1250,total-1]
    pictures={}
    count=0
    for index,frame in _encoded_frames(FINAL_VIDEO,total):
        count+=1
        if index in review_indices:
            tile=cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,f'{index/FPS:.2f}s frame{index}',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,
                        (245,245,245),1,cv2.LINE_AA)
            pictures[index]=tile
    # AAC packet time must progress by1024 samples, not merely claim a long
    # silent packet duration. A5.2s timestamp hole can make audio-clock-driven
    # playback skip the opening even when all video decode PTS are50fps.
    with av.open(str(FINAL_VIDEO)) as container:
        if len(container.streams.audio)!=1:raise ValueError('Final video must contain the original programme audio')
        audio=container.streams.audio[0]
        audio_rate=int(audio.rate)
        if audio_rate!=48000:raise ValueError('Expected original48kHz programme audio')
        audio_packets=[]
        for packet in container.demux(audio):
            if packet.size<=0:continue
            if packet.pts is None or packet.dts is None:
                raise ValueError('AAC packet timestamp missing')
            scaled=[Fraction(value)*audio.time_base*audio_rate
                    for value in (packet.pts,packet.dts,packet.duration)]
            if any(value.denominator!=1 for value in scaled):
                raise ValueError('AAC packet timing is off the integral sample grid')
            audio_packets.append(tuple(int(value) for value in scaled))
    if not audio_packets:raise ValueError('AAC packet stream empty')
    packet_max_gap_samples=0
    packet_max_step_samples=0
    for index,(pts,dts,duration) in enumerate(audio_packets):
        if not 0<duration<=1024 or (index<len(audio_packets)-1 and duration!=1024):
            raise ValueError(f'AAC packet{index} invalid sample duration{duration}')
        if index:
            previous=audio_packets[index-1]
            pts_step,dts_step=pts-previous[0],dts-previous[1]
            packet_max_step_samples=max(packet_max_step_samples,pts_step,dts_step)
            packet_max_gap_samples=max(packet_max_gap_samples,abs(pts_step-1024),abs(dts_step-1024))
            if pts_step!=1024 or dts_step!=1024:
                raise ValueError(f'AAC packet clock gap at{index}: PTS step{pts_step},DTS step{dts_step}')
    with av.open(str(FINAL_VIDEO)) as container:
        audio=container.streams.audio[0]
        samples=0;squares=0.;peak=0.;silent_peak=0.;silent_samples=0;silent_squares=0.;silent_values=0
        decoded_audio_frames=0
        decoded_samples_per_channel=0
        decoded_max_gap_samples=0
        previous_start=None
        previous_samples=None
        for frame in container.decode(audio):
            if frame.pts is None:raise ValueError('Decoded audio frame timestamp missing')
            exact_start=Fraction(frame.pts)*frame.time_base*frame.sample_rate
            if exact_start.denominator!=1:raise ValueError('Decoded audio timestamp is off the sample grid')
            start_sample=int(exact_start)
            expected_start=0 if previous_start is None else previous_start+previous_samples
            gap=start_sample-expected_start
            decoded_max_gap_samples=max(decoded_max_gap_samples,abs(gap))
            if gap!=0:
                raise ValueError(f'Decoded audio sample-clock gap at frame{decoded_audio_frames}: {gap} samples')
            previous_start,previous_samples=start_sample,int(frame.samples)
            decoded_audio_frames+=1
            decoded_samples_per_channel+=int(frame.samples)
            raw_values=frame.to_ndarray()
            values=raw_values.astype(np.float64)
            if np.issubdtype(raw_values.dtype,np.integer):
                values/=np.iinfo(raw_values.dtype).max
            if values.ndim==1:values=values[None,:]
            if values.shape[-1]!=frame.samples:
                values=values.reshape(-1,frame.samples)
            start=float(exact_start/frame.sample_rate)
            times=start+np.arange(frame.samples)/frame.sample_rate
            silence=(times>=0.)&(times<INTRO_SECONDS-.02)
            if silence.any():
                silent_peak=max(silent_peak,float(np.max(np.abs(values[:,silence]))))
                silent_samples+=int(silence.sum())
                silent_squares+=float(np.sum(values[:,silence]**2))
                silent_values+=values[:,silence].size
            audible=(times>=INTRO_SECONDS)&(times<report['duration_seconds'])
            if audible.any():
                actual=values[:,audible]
                samples+=actual.size;squares+=float(np.sum(actual*actual))
                peak=max(peak,float(np.max(np.abs(actual))))
    required_silent_samples=math.floor((INTRO_SECONDS-.02)*audio_rate)
    rms=float(np.sqrt(squares/max(samples,1)))
    silent_rms=float(np.sqrt(silent_squares/max(silent_values,1)))
    # AAC may leave a tiny encoder-priming impulse in otherwise silent frames.
    # Require both a low peak and low integrated level, rather than bit-zero PCM.
    if count!=total or silent_samples<required_silent_samples or silent_peak>1e-3 or silent_rms>1e-5 or rms<1e-5:
        raise ValueError(f'Final audio/video verification failed: frames={count},intropeak={silent_peak},mainrms={rms}')
    target=_out('encoded_review.jpg')
    tiles=[pictures[index] for index in review_indices]
    if not cv2.imwrite(str(target),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])):
        raise IOError('Could not save encoded review')
    result=dict(status='verified',video=str(FINAL_VIDEO),decoded_frames=count,fps=50,
                native_encoded_interval_seconds=.02,first_encoded_seconds=0.,last_encoded_seconds=(count-1)/FPS,
                intro_frames=INTRO_FRAMES,main_native_frames=MAIN_FRAMES,
                official_source_pts_first=FIRST,official_source_pts_last=LAST,official_source_time_base='1/12800',
                audio_streams=1,audio_sample_rate=audio_rate,
                audio_packet_count=len(audio_packets),
                audio_packet_first_pts_samples=audio_packets[0][0],
                audio_packet_last_pts_samples=audio_packets[-1][0],
                audio_packet_max_step_samples=packet_max_step_samples,
                audio_packet_max_gap_samples=packet_max_gap_samples,
                audio_packet_max_gap_seconds=packet_max_gap_samples/audio_rate,
                decoded_audio_frames=decoded_audio_frames,
                decoded_audio_samples_per_channel=decoded_samples_per_channel,
                decoded_audio_max_gap_samples=decoded_max_gap_samples,
                decoded_audio_max_gap_seconds=decoded_max_gap_samples/audio_rate,
                intro_silent_samples_per_channel=silent_samples,
                intro_required_silent_samples_per_channel=required_silent_samples,
                intro_silence_verified_to_seconds=INTRO_SECONDS-.02,
                intro_audio_peak=silent_peak,intro_audio_rms=silent_rms,
                intro_silence_limits=dict(peak=1e-3,rms=1e-5,reason="AAC encoder priming tolerance"),
                programme_audio_rms=rms,programme_audio_peak=peak,
                encoded_review=str(target),video_sha256=digest(FINAL_VIDEO))
    write_json(_out('verification.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-only',action='store_true')
    parser.add_argument('--final-only',action='store_true')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    if args.main_only:return render_main(args.workers)
    if args.final_only:return render_final()
    if args.preview:return previews()
    if args.verify:return verify()
    render_main(args.workers)
    return render_final()


if __name__=='__main__':
    main()
