"""Lossless assembly and continuous audio for the accepted full-programme demo."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import av
from calibration.program_tracking.demo_source import check_output_path
from calibration.program_tracking.render_demo_intro_v2 import (
    _video_packets, _check_video, _sequence_hash, _ffmpeg_executable,
    FPS, TIME_BASE, TICK_STEP,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp')
OUT = DATA/'results/demo_full_20261016'
INTRO = DATA/'results/demo_intro_20261016/v3/intro_only.mp4'
FIRST_SHOT = DATA/'results/demo_intro_20261016/first_shot_v2.mp4'
REST = OUT/'rest_and_outro.mp4'
JOINED = OUT/'joined_video.mp4'
FINAL = OUT/'RinkSpace_full_demo.mp4'
COUNTS = (410, 1248, 7396)
SOURCE_FIRST = 256256
SOURCE_FRAMES = 8434
OUTRO_FRAMES = 210
TOTAL_FRAMES = sum(COUNTS)
TOTAL_SECONDS = TOTAL_FRAMES/FPS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def file_info(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())



def normalize_audio_clock(source_path, target_path):
    """Keep every compressed payload, quantize AAC timestamps to sample integers."""
    from fractions import Fraction
    source_path, target_path = Path(source_path), Path(target_path)
    before_video = _video_packets(source_path)[1]
    audio_hashes=[]; count=0; final_end=None; sample_rate=None
    with av.open(str(source_path)) as source, av.open(str(target_path),'w',format='mp4',
            options={'movflags':'+faststart','video_track_timescale':'12800'}) as target:
        mapping={s.index:target.add_stream_from_template(s) for s in source.streams}
        if len(source.streams.audio)!=1:raise ValueError('Expected one AAC stream')
        audio=source.streams.audio[0]
        if audio.codec_context.name!='aac':raise ValueError('Expected AAC')
        sample_rate=int(audio.codec_context.sample_rate)
        total_samples=round(TOTAL_SECONDS*sample_rate)
        tb=Fraction(1,sample_rate)
        for packet in source.demux():
            if packet.size<=0:continue
            original_stream=packet.stream
            if original_stream.type=='audio':
                if count==0 and int(packet.pts)!=-1024:
                    raise ValueError('Unexpected AAC priming convention')
                audio_hashes.append(hashlib.sha256(bytes(packet)).hexdigest())
                packet.pts=packet.dts=(count-1)*1024
                packet.duration=min(1024,total_samples-packet.pts)
                packet.time_base=tb
                if packet.duration<=0:raise ValueError('AAC extends beyond expected sample endpoint')
                final_end=packet.pts+packet.duration
                count+=1
            packet.stream=mapping[original_stream.index]
            target.mux(packet)
    if final_end!=round(TOTAL_SECONDS*sample_rate):
        raise ValueError('Normalized AAC endpoint differs')
    if _video_packets(target_path)[1]!=before_video:
        raise ValueError('AAC normalization changed video packets')
    after_hashes=[]; prior=None; durations=[]
    with av.open(str(target_path)) as check:
        stream=check.streams.audio[0]
        for packet in check.demux(stream):
            if packet.size<=0:continue
            after_hashes.append(hashlib.sha256(bytes(packet)).hexdigest())
            if prior is not None and packet.pts-prior!=1024:
                raise ValueError('AAC integer sample PTS still discontinuous')
            prior=int(packet.pts);durations.append(int(packet.duration))
    if after_hashes!=audio_hashes or any(x!=1024 for x in durations[:-1]):
        raise ValueError('AAC payload changed or internal duration differs')
    return dict(audio_packets=count,sample_rate=sample_rate,internal_sample_step=1024,
                final_packet_samples=durations[-1],final_sample_endpoint=final_end,
                audio_payload_exact=True,video_payload_pts_dts_exact=True)


def assemble(*,output_dir=OUT,rest_path=None,full_source=False,final_name=None):
    """Join frozen intro and either the complete re-render or the old remainder.

    full_source copies 410 intro packets plus 8644 source/ending packets.
    Separate revision output_dir prevents changing any accepted earlier export.
    """
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rest=Path(rest_path) if rest_path is not None else output_dir/REST.name
    name=final_name or FINAL.name
    if Path(name).name!=name:raise ValueError('final_name must be a single filename')
    joined_path=check_output_path(output_dir/JOINED.name)
    final_path=check_output_path(output_dir/name)
    sources=(INTRO,rest) if full_source else (INTRO,FIRST_SHOT,rest)
    counts=(COUNTS[0],SOURCE_FRAMES+OUTRO_FRAMES) if full_source else COUNTS
    if sum(counts)!=TOTAL_FRAMES:raise AssertionError('Full output frame count changed')
    segments=[]
    for path,count in zip(sources,counts):
        extra,identities=_video_packets(path)
        if len(identities)!=count:
            raise ValueError(f'{path.name}: expected {count}, got {len(identities)}')
        if identities[0][0]!=0 or not identities[0][4]:
            raise ValueError(f'{path.name}: expected independent IDR at display PTS0')
        segments.append((extra,identities))
    if any(extra!=segments[0][0] for extra,_ in segments):
        raise ValueError('Independent segments have different AVC extradata; stop before assembly')
    count=0; previous=None; offset=0
    with av.open(str(INTRO)) as first, av.open(str(joined_path),'w',format='mp4',
            options={'movflags':'+faststart','video_track_timescale':'12800'}) as target:
        output=target.add_stream_from_template(first.streams.video[0])
        output.time_base=TIME_BASE
        for path,expected in zip(sources,counts):
            with av.open(str(path)) as source:
                stream=source.streams.video[0]
                _check_video(stream)
                for packet in source.demux(stream):
                    if packet.size<=0:continue
                    packet.pts=int(packet.pts)+offset
                    packet.dts=int(packet.dts)+offset
                    if previous is not None and packet.dts-previous!=TICK_STEP:
                        raise ValueError('Non-continuous DTS at copied segment boundary')
                    previous=int(packet.dts)
                    packet.stream=output
                    packet.time_base=TIME_BASE
                    target.mux(packet)
                    count+=1
            offset+=expected*TICK_STEP
    _,joined=_video_packets(joined_path)
    if count!=TOTAL_FRAMES or len(joined)!=TOTAL_FRAMES:
        raise ValueError('Full copied video packet count differs')
    start=0; reports=[]
    for path,expected,(_,identities) in zip(sources,counts,segments):
        shift=start*TICK_STEP
        actual=[(a-shift,b-shift,c,d,e) for a,b,c,d,e in joined[start:start+expected]]
        if actual!=identities:
            raise ValueError(f'{path.name}: copied payload or shifted PTS/DTS differs')
        reports.append(dict(path=str(path),frames=expected,
                            output_start_frame=start,pts_dts_shift_ticks=shift,
                            payload_sequence_sha256=_sequence_hash(identities),
                            copied_payload_and_pts_dts_exact=True))
        start+=expected
    if sorted(p[0] for p in joined)!=list(range(0,TOTAL_FRAMES*TICK_STEP,TICK_STEP)):
        raise ValueError('Full display PTS grid is incomplete')
    join_report=dict(schema='rinkspace.demo-full-join.v1',video=file_info(joined_path),
                     frames=TOTAL_FRAMES,duration_seconds=TOTAL_SECONDS,
                     segments=reports,all_source_frames_rerendered=bool(full_source),extradata_identical=True,
                     extradata_sha256=hashlib.sha256(segments[0][0]).hexdigest(),
                     time_base='1/12800',display_interval_ticks=TICK_STEP,
                     source_native_frames=SOURCE_FRAMES,intro_frames=COUNTS[0],
                     outro_frames=OUTRO_FRAMES)
    write_json(output_dir/'join_report.json',join_report)
    packet_path=DATA/'results/demo_last_shot_20261016/source_packet.json'
    # The immutable native packet binds the actual official source path.
    import numpy as np
    with np.load(DATA/'results/demo_last_shot_20261016/source_packet.npz',allow_pickle=False) as p:
        source_path=str(p['source_path'])
    duration=SOURCE_FRAMES/FPS
    fade_start=duration-.45
    audio_filter=(f'atrim=duration={duration:.8f},asetpts=PTS-STARTPTS,'
                  f'afade=t=out:st={fade_start:.8f}:d=0.45,apad=pad_dur={OUTRO_FRAMES/FPS:.8f},'
                  f'adelay={round(COUNTS[0]*1000/FPS)}:all=1,'
                  'asetnsamples=n=1024:p=1,asetpts=N/SR/TB')
    command=[_ffmpeg_executable(),'-hide_banner','-loglevel','error','-y',
             '-i',str(joined_path),'-ss',str(SOURCE_FIRST/12800.),'-i',source_path,
             '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
             '-af',audio_filter,'-t',str(TOTAL_SECONDS),'-video_track_timescale','12800',
             '-movflags','+faststart',str(final_path)]
    write_json(output_dir/'final_mux_command.json',command)
    with (output_dir/'final_mux.stderr.log').open('wb') as log:
        result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
    if result.returncode:
        raise RuntimeError((output_dir/'final_mux.stderr.log').read_text(errors='replace')[-3000:])
    normalized=output_dir/'audio_clock_normalized.mp4'
    normalization=normalize_audio_clock(final_path,normalized)
    normalized.replace(final_path)
    write_json(output_dir/'audio_clock_normalization.json',normalization)
    _,final=_video_packets(final_path)
    if final!=joined:
        raise ValueError('Continuous audio mux changed video payload or clock')
    report=dict(schema='rinkspace.demo-full-assembly.v1',video=file_info(final_path),
                frames=TOTAL_FRAMES,fps=FPS,width=1920,height=1080,
                duration_seconds=TOTAL_SECONDS,intro_seconds=COUNTS[0]/FPS,
                source_seconds=duration,outro_seconds=OUTRO_FRAMES/FPS,
                source_first_pts=SOURCE_FIRST,source_last_pts_inclusive=2415104,
                source_time_base='1/12800',source_path=source_path,
                source_audio_delay_milliseconds=8200,audio_filter=audio_filter,
                video_copy_payload_pts_dts_exact=True,statistics=False,
                join_report=str(output_dir/'join_report.json'))
    write_json(output_dir/'assembly_report.json',report)
    print(json.dumps(dict(video=str(final_path),frames=TOTAL_FRAMES,duration_seconds=TOTAL_SECONDS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--rest-path',type=Path)
    parser.add_argument('--full-source',action='store_true',help='Rest file contains all 8434 source frames plus 210 ending frames')
    parser.add_argument('--final-name')
    args=parser.parse_args()
    assemble(output_dir=args.output_dir,rest_path=args.rest_path,
             full_source=args.full_source,final_name=args.final_name)
