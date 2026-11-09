"""Re-render exact revised source exposures and patch them into the full review."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np

from compare_s09_repair import programme, use_saved
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from rinkspace.review.render import source_frames, validate_video
from rinkspace.viewer3d.source import SourceVideo

TB=Fraction(1,12800)
FPS=50
W,H=1920,1080

def log(event,**values):
    print(json.dumps(dict(event=event,**values),ensure_ascii=False),flush=True)


def windows(mask):
    f=np.flatnonzero(mask)
    if not len(f):
        return []
    return [(int(a[0]),int(a[-1])+1) for a in np.split(f,np.flatnonzero(np.diff(f)>1)+1)]


def clip_support(old,new,t):
    K=old['absolute_knots']
    np.testing.assert_array_equal(K,new['absolute_knots'])
    for k in ('relative_skate_rotation_controls','camera_binding_json','prepared_input_sha256'):
        np.testing.assert_array_equal(old[k],new[k])
    a=old['absolute_position_controls']
    change=np.any(a!=new['absolute_position_controls'],axis=1)
    change|=np.any(old['absolute_rotation_controls']!=new['absolute_rotation_controls'],axis=(1,2,3))
    e=np.flatnonzero(np.any(old['absolute_rotation_edge_reference_quaternion']!=new['absolute_rotation_edge_reference_quaternion'],axis=(1,2)))
    selected=np.union1d(np.flatnonzero(change),np.r_[e,e+1])
    ix=np.clip(np.searchsorted(K,t,side='right')-4,0,len(a)-4)
    return np.isin(ix[...,None]+np.arange(4),selected).any(axis=-1)


def ffmpeg(args,command):
    return [str(args.ffmpeg),'-hide_banner','-loglevel','info','-stats_period','10','-nostdin','-xerror','-n',*command]


def patch_stream(args,p,segments,file):
    from render_programme_body_com import project_batch,compose,legend
    command=ffmpeg(args,['-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080',
        '-framerate','50','-i','pipe:0','-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-c:v','libx264','-preset','ultrafast','-crf','0','-video_track_timescale','12800',str(file)])
    started=time.perf_counter()
    limit=sum(b-a for a,b in segments)
    panel=legend()
    details=[]
    count=0
    last=started
    with (args.output/'lossless_patch_encoder.log').open('xb') as errors:
        run=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for a,b in segments:
                reader=iter(source_frames(p.source,p.source_spec,p.pts[a:b]))
                try:
                    for i in range(a,b,16):
                        rows=np.arange(i,min(i+16,b))
                        _,_,all_uv=project_batch(p,rows)
                        for f,uv in zip(rows,all_uv):
                            pts,raw=next(reader)
                            if pts!=p.pts[f]:
                                raise ValueError(f'Expected original exposure at index {f}')
                            raw,body,com,grid=compose(raw,p,int(f),uv,panel)
                            if not body or not grid:
                                raise ValueError(f'Missing body or rink overlay at PTS {pts}')
                            run.stdin.write(memoryview(np.ascontiguousarray(raw)).cast('B'))
                            details.append(dict(source_pts=pts,source_index=int(f),patch_index=count,
                                                body_pixels=body,com_on_screen=bool(com),grid_pixels=grid))
                            count+=1
                        now=time.perf_counter()
                        if now-last>=10. or count==limit:
                            log('revised_overlay_progress',frames=count,total=limit,source_seconds=pts/12800.,
                                elapsed_s=round(now-started,2))
                            last=now
                    if next(reader,None) is not None:
                        raise ValueError('Decoded too many source frames for patch')
                finally:
                    reader.close()
            run.stdin.close()
            if run.wait()!=0:
                raise RuntimeError('Patch encoder failed')
        finally:
            if run.poll() is None:
                run.terminate()
                run.wait()
    if count!=limit:
        raise ValueError('Incomplete corrected patch frames')
    return details


def replace_video(args,parts,file,out,n):
    sections=[]
    start=0;offset=0
    for a,b in parts:
        if a>start:
            sections.append((0,start,a))
        sections.append((1,offset,offset+b-a))
        offset+=b-a
        start=b
    if n>start:
        sections.append((0,start,n))
    counts=[sum(s[0]==k for s in sections) for k in (0,1)]
    plan=[]
    for k,num in enumerate(counts):
        names=''.join(f'[i{k}_{j}]' for j in range(num))
        plan.append(f'[{k}:v]split={num}{names}' if num>1 else f'[{k}:v]null{i_fmt(k,0)}')
    counters=[0,0]
    for q,(k,a,b) in enumerate(sections):
        plan.append(f'{i_fmt(k,counters[k])}trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,setsar=1[s{q}]')
        counters[k]+=1
    plan.append(''.join(f'[s{j}]' for j in range(len(sections)))+
                f'concat=n={len(sections)}:v=1:a=0,settb=1/12800,setpts=N*256[v]')
    filters=';'.join(plan)
    command=ffmpeg(args,['-i',str(args.previous_video),'-i',str(file),
        '-filter_complex_threads','2','-filter_complex',filters,
        '-map','[v]','-map','0:a:0','-c:a','copy','-c:v','h264_nvenc','-preset','p4','-cq','17','-b:v','0',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-pix_fmt','yuv420p',
        '-r','50','-enc_time_base','1:50','-fps_mode','cfr','-bf','0','-g','100','-video_track_timescale','12800','-movflags','+faststart',str(out)])
    with (args.output/'full_video_50cfr.log').open('xb') as e:
        p=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
    if p.returncode:
        raise RuntimeError('Full video assembly failed')
    return sections


def i_fmt(k,j):
    return f'[i{k}_{j}]'


def audio_identity(file):
    content=hashlib.sha256()
    timebase=hashlib.sha256()
    count=0
    with av.open(str(file)) as movie:
        if len(movie.streams.audio)!=1:
            raise ValueError('Expected one original AAC audio stream')
        audio=movie.streams.audio[0]
        for packet in movie.demux(audio):
            if packet.pts is None or not packet.size:
                continue
            content.update(bytes(packet))
            record=(str(Fraction(packet.pts)*packet.time_base),
                    str(Fraction(packet.duration)*packet.time_base))
            timebase.update(('|'.join(record)+'\n').encode())
            count+=1
    return [count,content.hexdigest(),timebase.hexdigest()]


def samples(p,sections,old,new,patch):
    readers=[SourceVideo(str(path),(1,12800)) for path in (old,patch,new)]
    result=[]
    try:
        offset=0
        critical=np.searchsorted(p.pts,np.rint(12800*np.array([150.78,150.98,157.24,160.24,161.48,162.74,163.90])).astype(int))
        for k,a,b in sections:
            n=b-a
            ids=np.r_[0,n//2,n-1,critical[(critical>=offset)&(critical<offset+n)]-offset]
            for j in np.unique(ids):
                actual=readers[2].read((offset+int(j))*256)
                expected=readers[k].read((a+int(j))*256)
                err=float(cv2.PSNR(expected,actual))
                if err<30.:
                    raise ValueError(f'Incorrect video frame near source {p.pts[offset+j]/12800.}: {err:.2f} dB')
                result.append(dict(pts=int(p.pts[offset+j]),revised=bool(k),psnr_db=err))
            offset+=n
        if offset!=len(p.pts):
            raise ValueError('Assembly samples miss part of the programme')
    finally:
        for r in readers:r.close()
    return result


def main():
    parser=argparse.ArgumentParser()
    for key in ('project','data','config','prepared','source','renderer','parameter','previous-video','output','ffmpeg'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.renderer/'verification.json').read_text('utf8'))
    if digest(args.previous_video)!=original['video_sha256']:
        raise ValueError('Input review changed')
    p=programme(args.project,args.data,args.config,args.prepared,args.source)
    original_com=p.com.copy()
    old=load(p.parameter);new=load(args.parameter)
    p0=clip_support(old,new,p.clock)
    readout=p.readout
    exposure=(p.clock[:,None]+readout[:,None]*np.linspace(-.5,.5,5))
    display=clip_support(old,new,exposure).any(axis=1)
    # The old video draws 31 prior COM points spaced four source exposures apart.
    for shift in range(4,125,4):
        display[shift:]|=p0[:-shift]
    use_saved(p,args.parameter)
    np.testing.assert_array_equal(original_com[~p0],p.com[~p0])
    segments=windows(display)
    log('partial_render_started',total_frames=len(p.pts),revised_frames=int(display.sum()),
        source_intervals_seconds=[[float(p.pts[a]*TB),float(p.pts[b-1]*TB)] for a,b in segments])
    inter=args.output/'lossless_patch_intermediate.mp4'
    pending=args.output/'okada_mei_sp_9shots_s09_revised.pending.mp4'
    video=args.output/'okada_mei_sp_9shots_s09_revised.mp4'
    details=patch_stream(args,p,segments,inter)
    check_patch=validate_video(inter,np.arange(len(details),dtype=np.int64)*256,TB)
    log('assemble_all_nine_shots',revised_frames=len(details),reused_frames=len(p.pts)-len(details),all_frames=len(p.pts))
    parts=replace_video(args,segments,inter,pending,len(p.pts))
    log('verify_assembled_clock',all_frames=len(p.pts))
    result=validate_video(pending,p.pts,TB)
    old_audio,new_audio=audio_identity(args.previous_video),audio_identity(pending)
    if old_audio!=new_audio:
        raise ValueError(f'Original sound payload or clock changed: {old_audio} versus {new_audio}')
    checks=samples(p,parts,args.previous_video,pending,inter)
    pending.rename(video)
    report=dict(video=str(video),video_sha256=digest(video),previous_video_sha256=original['video_sha256'],
                source=str(p.source),source_sha256=p.source_spec['sha256'],candidate_sha256=digest(args.parameter),
                original_source_time_base=[1,12800],source_first_last_seconds=[p.pts[0]/12800,p.pts[-1]/12800],
                verified_clock=result,frames=len(p.pts),source_shots=list(dict.fromkeys(p.shots.tolist())),
                revised_frames=len(details),outside_patch_image_layers_reused_then_encoded=len(p.pts)-len(details),
                original_aac_payload_and_clock_bitwise_equal=True,original_audio_packet_count=old_audio[0],
                patch_clock=check_patch,source_patch_intervals_seconds=[[float(p.pts[a]*TB),float(p.pts[b-1]*TB)] for a,b in segments],
                encoded_image_check=checks,min_sample_psnr_db=min(r['psnr_db'] for r in checks))
    with (args.output/'verification.json').open('x',encoding='utf8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    inter.unlink()
    log('complete',**report)

if __name__=='__main__':main()
