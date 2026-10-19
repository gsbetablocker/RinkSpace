"""Render the new full-programme remainder and confirmed ending only.

Frozen intro/first-shot packet joining, fresh audio muxing and final QA are
separate operations. No optimization or fitted model is changed here.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import cv2
import numpy as np
from calibration.program_tracking.render_demo_last_shot import (
    Sample,DATA,BASE_OUT,WH,FPS,decode,digest,write_json)
from calibration.program_tracking.demo_particle_pass import ParticleOverlay,_encoder_quality
from calibration.program_tracking.demo_pipeline import ordered_frames, _resolve_factory
from calibration.program_tracking.render_demo_intro_v2 import _video_packets
from calibration.program_tracking.render_result import _ffmpeg_executable,_available_encoder
from calibration.program_tracking.demo_source import check_output_path
FIRST,LAST=256256,2415104
REST_FIRST=575744
FIRST_SHOT_LAST=575488
MAIN_FRAMES=(LAST-FIRST)//256+1
REST_FRAMES=(LAST-REST_FIRST)//256+1
OUTRO_FRAMES=210
TOTAL_FRAMES=REST_FRAMES+OUTRO_FRAMES
OUT=DATA/'results/demo_full_20261016'
CONTACT=OUT/'contact_packet.npz'
TARGET=OUT/'rest_and_outro.mp4'
INTRO=DATA/'results/demo_intro_20261016/v3/intro_only.mp4'
FIRST_SHOT=DATA/'results/demo_intro_20261016/first_shot_v2.mp4'


class FullSample(Sample):
    """Same approved v2 effects with full-range decorative particle indexing."""
    def __init__(self,version='v2',contact_path=None):
        if version!='v2':raise ValueError('Full demo uses the confirmed v2 style')
        super().__init__(version=version,contact_path=contact_path or CONTACT)
        expected=np.arange(FIRST,LAST+256,256,dtype=np.int64)
        if not np.array_equal(self.pts,expected):
            raise ValueError('Source packet must cover every full-programme native50 PTS')
        self.particles=ParticleOverlay(self,first_pts=FIRST,last_pts=LAST)

    def draw(self,native,pts):
        frame,base,texture,foreground=super().draw(native,pts)
        index=(int(pts)-FIRST)//256
        frame=self.particles.frame(frame,index,foreground=foreground)
        return frame,base,texture,foreground


def _reference(file):
    file=Path(file)
    return {'path':str(file),'sha256':digest(file)}


def _status(stage,count,started,*,output_dir=OUT,total=TOTAL_FRAMES,**extra):
    value={'status':'running','stage':stage,'frames':count,'total':total,
           'elapsed_seconds':time.perf_counter()-started,**extra}
    write_json(Path(output_dir)/'render_status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def render_rest(workers=4,slots=12,chunk_size=1,contact_path=CONTACT,*,
                source_first_pts=REST_FIRST,output_dir=OUT,target=None,
                sample_factory='calibration.program_tracking.render_demo_full:FullSample',
                sample_kwargs=None):
    """Encode the selected native source suffix and the approved 210-frame ending.

    The default freezes the old intro and first shot. FIRST re-renders all 8434
    programme frames while preserving only the intro; old outputs are untouched
    when output_dir points at a revision directory.
    """
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    target=check_output_path(Path(target) if target is not None else output_dir/TARGET.name)
    source_first_pts=int(source_first_pts)
    if source_first_pts not in (FIRST,REST_FIRST):
        raise ValueError('Only the complete programme or the frozen-first-shot remainder is supported')
    source_frames=(LAST-source_first_pts)//256+1
    total_frames=source_frames+OUTRO_FRAMES
    kwargs=dict(sample_kwargs or {})
    kwargs.setdefault('version','v2')
    kwargs.setdefault('contact_path',str(contact_path))
    def status(stage,count,started,**extra):
        return _status(stage,count,started,output_dir=output_dir,total=total_frames,**extra)
    contact_path=Path(kwargs['contact_path'])
    if not contact_path.is_file():raise FileNotFoundError(contact_path)
    if REST_FIRST!=FIRST_SHOT_LAST+256 or MAIN_FRAMES!=8434 or REST_FRAMES!=7186 or TOTAL_FRAMES!=7396:
        raise AssertionError('Frozen segment boundaries changed')
    started=time.perf_counter()
    sample=_resolve_factory(sample_factory)(**kwargs)
    if round(sample.outro.total_seconds*FPS)!=OUTRO_FRAMES:
        raise ValueError('Confirmed ending must remain4.2seconds/210frames')
    intro_extra,intro_packets=_video_packets(INTRO)
    if len(intro_packets)!=410:
        raise ValueError('Frozen intro frame count differs')
    if source_first_pts==REST_FIRST:
        shot_extra,shot_packets=_video_packets(FIRST_SHOT)
        if intro_extra!=shot_extra or len(shot_packets)!=1248:
            raise ValueError('Frozen intro/first-shot codec or frame counts differ')
    inputs={'source_packet':_reference(BASE_OUT/'source_packet.npz'),
            'contact_packet':_reference(contact_path),'intro':_reference(INTRO),
            'render_code':_reference(__file__),'sample_factory':str(sample_factory),
            'sample_code':_reference(inspect.getfile(type(sample))),'sample_kwargs':kwargs}
    if hasattr(sample,'mask_override_path'):
        inputs['foreground_masks']=_reference(sample.mask_override_path)
    for attribute,label in (('camera_override_path','camera_patch'),('trajectory_override_path','trajectory_patch')):
        if hasattr(sample,attribute):inputs[label]=_reference(getattr(sample,attribute))
    if source_first_pts==REST_FIRST:inputs['first_shot']=_reference(FIRST_SHOT)
    ffmpeg=_ffmpeg_executable();args,name=_available_encoder(ffmpeg)
    args=_encoder_quality(args,name)
    if name!='h264_nvenc':raise ValueError('Confirmed packet-compatible export requires h264_nvenc')
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
             '-pixel_format','bgr24','-video_size','1920x1080','-framerate','50','-i','pipe:0',
             *args,'-pix_fmt','yuv420p','-an','-frames:v',str(total_frames),
             '-video_track_timescale','12800','-movflags','+faststart',str(target)]
    write_json(output_dir/'rest_command.json',command)
    write_json(output_dir/'rest_inputs.json',inputs)
    count=0;source_count=0;source_pts_hash=hashlib.sha256();stats={};iterator=None
    last_rendered=None
    status('rest_prepare',0,started,workers=workers,slots=slots)
    with (output_dir/'rest_encode.stderr.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                                 stderr=log,creationflags=0x08000000)
        try:
            iterator=ordered_frames(decode(str(sample.packet['source_path']),source_first_pts,LAST),
                sample_kwargs=kwargs,
                workers=workers,slots=slots,chunk_size=chunk_size,cv_threads=1,stats=stats,
                sample_factory=sample_factory)
            for pts,frame in iterator:
                expected=source_first_pts+source_count*256
                if int(pts)!=expected:raise ValueError(f'Rest source PTS mismatch:{pts}!={expected}')
                process.stdin.write(frame.tobytes())
                source_pts_hash.update(np.asarray([pts],np.int64).tobytes())
                if int(pts)==LAST:last_rendered=frame.copy()
                count+=1;source_count+=1
                if count%100==0:status('rest_source',count,started,last_source_pts=int(pts))
            if source_count!=source_frames or last_rendered is None:
                raise ValueError(f'Render did not contain all {source_frames} official native frames')
            native_last=next(decode(str(sample.packet['source_path']),LAST,LAST))[1]
            final,base,texture,foreground=sample.draw(native_last,LAST)
            if not np.array_equal(final,last_rendered):
                raise ValueError('Parallel final source frame differs from the serial ending state')
            sample.particles.last_foreground=foreground
            zero=sample.outro.frame(base,texture,0.,foreground_mask=foreground)
            zero=sample.particles.frame(zero,MAIN_FRAMES-1,foreground=foreground)
            if not np.array_equal(zero,final):
                raise ValueError('Ending t0 does not equal the last full demo frame')
            for j in range(OUTRO_FRAMES):
                elapsed=(j+1)/FPS
                frame=sample.outro.frame(base,sample.ending_texture(texture,elapsed),elapsed,
                                         foreground_mask=foreground)
                frame=sample.particles.frame(frame,MAIN_FRAMES+j)
                process.stdin.write(frame.tobytes());count+=1
                if (j+1)%25==0:status('outro',count,started,outro_elapsed_seconds=elapsed)
            if count!=total_frames:raise ValueError('Remainder/ending frame count changed')
        except BaseException as error:
            write_json(output_dir/'render_status.json',{'status':'failed','frames':count,
                       'source_frames':source_count,'error':repr(error),'inputs':inputs})
            raise
        finally:
            if iterator is not None:iterator.close()
            process.stdin.close()
            return_code=process.wait()
        if return_code:raise RuntimeError('NVENC failed:'+ (output_dir/'rest_encode.stderr.log').read_text(errors='replace')[-3000:])
    extra,packets=_video_packets(target)
    if extra!=intro_extra:raise ValueError('Encoded rest SPS/PPS/avcC differs from frozen segments')
    if len(packets)!=total_frames or sorted(p[0] for p in packets)!=list(range(0,total_frames*256,256)):
        raise ValueError('Encoded remainder lost native50 packet count or presentation grid')
    report={'status':'complete','stage':'rest_and_outro_encoded','video':_reference(target),
            'video_only':True,'width':1920,'height':1080,'fps':50,'time_base':'1/12800',
            'source':str(sample.packet['source_path']),'source_first_pts':source_first_pts,
            'source_last_pts':LAST,'source_pts_step':256,'native_source_frames':source_count,
            'source_pts_sequence_sha256':source_pts_hash.hexdigest(),
            'source_programme_frames':MAIN_FRAMES,'reused_first_shot_frames':(source_first_pts-FIRST)//256,
            'outro_frames':OUTRO_FRAMES,'encoded_frames':count,'duration_seconds':count/FPS,
            'outro_t0_pixels_exact':True,'outro_turn_seconds':3.,'outro_hold_seconds':1.2,
            'summary_transition_seconds':.55,'particle_fade_seconds':.55,
            'outro_uniform_trajectory':True,'outro_final_particles':False,
            'no_statistics_or_new_text':True,'original_camera_and_body_fit_unchanged':True,
            'scene_scale_status':'conditional model scale; not independently surveyed metres',
            'metres_per_scene_unit':float(sample.packet['unit']),
            'encoder':name,'encoder_args':args,'codec_extradata_identical_to_frozen_segments':True,
            'codec_extradata_sha256':hashlib.sha256(extra).hexdigest(),
            'drawing_pipeline':stats,'inputs':inputs,'wall_seconds':time.perf_counter()-started}
    write_json(output_dir/'rest_report.json',report);write_json(output_dir/'render_status.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',required=True,
                        help='Encode selected native source range and outro; preserve the frozen intro')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--slots',type=int,default=12)
    parser.add_argument('--chunk-size',type=int,default=1)
    parser.add_argument('--contact-path',type=Path,default=CONTACT)
    parser.add_argument('--source-first-pts',type=int,default=REST_FIRST,choices=(FIRST,REST_FIRST))
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--target',type=Path)
    parser.add_argument('--sample-factory',default='calibration.program_tracking.render_demo_full:FullSample')
    parser.add_argument('--sample-kwargs',type=json.loads,default={},help='Small constructor options as a JSON object')
    args=parser.parse_args();cv2.setNumThreads(1)
    return render_rest(args.workers,args.slots,args.chunk_size,args.contact_path,
        source_first_pts=args.source_first_pts,output_dir=args.output_dir,target=args.target,
        sample_factory=args.sample_factory,sample_kwargs=args.sample_kwargs)


if __name__=='__main__':main()
