"""Assemble and verify one exact-clock review MP4 per Ankara short programme."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import time

import av
import numpy as np

from joint_inputs import OUT, ROOT, STEP, TB, check, lines, numeric_hash, pin, read, write

PROGRAMMES=ROOT/'events/ankara_2026/programmes'
FILES={
    'jin':PROGRAMMES/'jin_shuxian_sp/joint_review_1113.json',
    'okada':PROGRAMMES/'okada_mei_sp/fit_v11/joint_review_1113.json',
}
TARGET=OUT/'full_review'


def windows_escape(path):
    return "'" + path.as_posix().replace("'", "'\\''") + "'"


def verify_clip(variant, shot, full_pts, controls, rigs, geometry, clip_meta, cache):
    clip_path=Path(check(clip_meta['file'])['path'])
    folder=clip_path.parent
    if folder not in cache:
        ref=read(folder/'review.json')
        assert ref['path_count']==922 and ref['unlocalized']==0
        assert ref['geometry_sha256']==geometry
        source=check(ref['numeric_state'])
        with np.load(source['path'],allow_pickle=False) as z:
            state={k:z[k] for k in z.files}
        assert numeric_hash(state['scene_values'])==geometry
        cache[folder]=ref,state
    ref,state=cache[folder]
    name=variant+'_'+shot
    assert name in ref['shots']
    np.testing.assert_array_equal(state[name+'_controls'],controls)
    np.testing.assert_array_equal(state[name+'_rig'],rigs)
    receipt=read(folder/(name+'_receipt.json'))
    assert receipt['video']==clip_meta['file']
    assert receipt['decoded_frames']==len(full_pts)==clip_meta['exact_frames']
    assert receipt['native_PTS_equal'] and receipt['all_922_paths']
    assert (receipt['source_first_pts'],receipt['source_last_pts'])==(int(full_pts[0]),int(full_pts[-1]))
    ledger=check(receipt['source_frames'])
    pixels=list(lines(Path(ledger['path'])))
    np.testing.assert_array_equal([x['pts'] for x in pixels],full_pts)
    assert all(len(x['source_BGR_sha256'])==64 for x in pixels)
    assert sum(bool(x['n_points']) for x in pixels)==receipt['coverage']['frames_with_points']
    assert sum(x['n_points'] for x in pixels)==receipt['coverage']['active_points']
    return dict(shot=shot, file=receipt['video'], source_pts_ledger=ledger,
                source_first_pts=int(full_pts[0]),source_last_pts=int(full_pts[-1]),
                frames=len(full_pts),coverage=receipt['coverage'])


def bind_clips(variant):
    manifest=read(FILES[variant])
    camera_ref=check(manifest['camera'])
    source_ref=check(manifest['source'])
    geometry_ref=check(manifest['geometry'])
    geometry=manifest['geometry']['value_sha256']
    with np.load(geometry_ref['path'],allow_pickle=False) as z:
        assert numeric_hash(z['scene_values'])==geometry
    with np.load(camera_ref['path'],allow_pickle=False) as z:
        pts=z['source_pts']
        ids=z['shot_ids'].tolist()
        spans=z['source_offsets']
        controls=z['controls']
        motion=z['control_offsets']
        rigs=z['rig_by_source_pts']
        assert z['scene_values_sha256'].item()==geometry
    src_pts=manifest['source_native_PTS']
    np.testing.assert_array_equal(pts,np.arange(src_pts['first'],src_pts['last']+STEP,STEP))
    assert src_pts['exposures']==len(pts)
    assert len(ids)==len(set(ids)) and set(ids)==set(manifest['actual_shot_counts'])
    assert spans[0]==motion[0]==0 and spans[-1]==len(pts)==len(rigs) and motion[-1]==len(controls)
    assert np.all(np.diff(spans)>0) and np.all(np.diff(motion)>0)
    cache={}
    clips=[]
    for i,shot in enumerate(ids):
        begin,end=map(int,spans[i:i+2])
        a,b=map(int,motion[i:i+2])
        exact=pts[begin:end]
        assert len(exact)==manifest['actual_shot_counts'][shot]
        selected=manifest['clips'].get(shot)
        if selected is None:
            d=OUT/'full_review_shots'/variant
            receipt=read(d/f'{variant}_{shot}_receipt.json')
            selected={'file':receipt['video'],'exact_frames':receipt['decoded_frames']}
        clips.append(verify_clip(variant,shot,exact,controls[a:b],rigs[begin:end],geometry,selected,cache))
    return manifest,pts,clips


def normalized_frames(path):
    with av.open(str(path)) as c:
        video=c.streams.video[0]
        video.thread_type='AUTO'
        video.codec_context.thread_count=5
        if (video.width,video.height)!=(2560,1144) or video.average_rate!=50:
            raise ValueError('Review stream format differs: '+str(path))
        for f in c.decode(video):
            native_tick=f.pts*f.time_base/TB
            if native_tick.denominator!=1:
                raise ValueError('Non-integral review PTS: '+str(path))
            yield int(native_tick),f


def frame_hash(frame):
    return sha256(frame.to_ndarray(format='yuv420p').tobytes()).hexdigest()


def verify_join(video, pts, clips):
    # Compare decoded pixels across both sides of each edit to certify that
    # stream copy preserved the already rendered overlay. Independently
    # check every output PTS including frames away from edits.
    joined=normalized_frames(video)
    keyframes=0
    cursor=0
    for clip in clips:
        filename=Path(clip['file']['path'])
        local=normalized_frames(filename)
        for j in range(clip['frames']):
            output_pts,frame=next(joined)
            t,reference=next(local)
            expected=int(pts[cursor]-pts[0])
            if output_pts!=expected or t!=j*STEP:
                raise ValueError((clip['shot'],j,output_pts,expected,t))
            if j in (0,clip['frames']//2,clip['frames']-1):
                if frame_hash(frame)!=frame_hash(reference):
                    raise ValueError(('Decoded pixels changed',clip['shot'],j))
                keyframes+=1
            cursor+=1
        if next(local,None) is not None:
            raise ValueError('Extra frames in '+clip['shot'])
    if cursor!=len(pts) or next(joined,None) is not None:
        raise ValueError('The programme has a different duration from its source frames')
    return dict(decoded_frames=cursor,frame_pts_exact=True,pixel_checks_across_cuts=keyframes)


def audio_info(path):
    with av.open(str(path)) as f:
        return [{'codec':s.codec_context.name,'sample_rate':s.codec_context.sample_rate,
                 'start_seconds':float(s.start_time*s.time_base),
                 'duration_seconds':float(s.duration*s.time_base)}
                for s in f.streams.audio]


def verify_audio(source, joined, first, last):
    # Compare the programme soundtrack at spaced positions on the native
    # source clock. This catches an offset that duration alone cannot detect.
    from scipy.signal import correlate

    def samples(path, trim=None):
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i',str(path),'-map','0:a:0']
        if trim is not None:
            cmd+=['-af',trim]
        cmd+=['-ac','1','-ar','24000','-f','f32le','pipe:1']
        return np.frombuffer(subprocess.check_output(cmd),dtype='<f4')

    original=samples(source,f'atrim=start={seconds(first)}:end={seconds(last+STEP)}')
    out=samples(joined)
    checks=[]
    window=4096
    search=360
    for f in (.04,.27,.51,.73,.95):
        i=int(len(original)*f)
        a=original[i:i+window]
        b=out[i-search:i+window+search]
        if len(b)!=window+2*search or np.linalg.norm(a)<.05:
            continue
        signal=a-a.mean()
        score=correlate(b,signal,mode='valid',method='fft')
        j=int(np.argmax(score))
        match=b[j:j+window]
        normal=np.dot(signal,match-match.mean())/(np.linalg.norm(signal)*np.linalg.norm(match-match.mean()))
        if normal<.90 or abs(j-search)>240:
            raise ValueError(('Source audio does not match at the expected time',i,j-search,normal))
        checks.append(dict(output_seconds=i/24000,offset_seconds=(j-search)/24000,correlation=float(normal)))
    if not checks:
        raise ValueError('No audible source window available for audio synchronisation check')
    return checks


def seconds(pts):
    # The source cadence is exactly 20 milliseconds.
    return f'{pts/12800:.6f}'


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('variant',choices=tuple(FILES))
    args=ap.parse_args()
    begin=time.monotonic()
    v=args.variant
    manifest,pts,clips=bind_clips(v)
    source=manifest['source']['path']
    edition='jin_shuxian' if v=='jin' else 'okada_mei'
    name=edition+'_ankara_joint_full_50fps'
    TARGET.mkdir(exist_ok=True)
    video=TARGET/(name+'.mp4')
    temp=TARGET/(name+'.part.mp4')
    if video.exists() or temp.exists():
        raise FileExistsError(video if video.exists() else temp)
    first,last=int(pts[0]),int(pts[-1])
    duration=(last+STEP-first)*TB
    playlist=TARGET/(edition+'.ffconcat')
    body=['ffconcat version 1.0']
    for clip in clips:
        body.extend(['file '+windows_escape(Path(clip['file']['path'])),
                     'duration '+seconds(clip['frames']*STEP)])
    with playlist.open('x',encoding='utf8') as f:
        f.write('\n'.join(body)+'\n')
    aud=audio_info(source)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
         '-f','concat','-safe','0','-i',str(playlist)]
    if aud:
        flt=f"[1:a:0]atrim=start={seconds(first)}:end={seconds(last+STEP)},asetpts=PTS-STARTPTS[a]"
        cmd.extend(['-i',source,'-filter_complex',flt,'-map','0:v:0','-map','[a]','-c:a','aac','-b:a','192k'])
    else:
        cmd.extend(['-map','0:v:0','-an'])
    cmd.extend(['-c:v','copy','-video_track_timescale','12800','-movflags','+faststart',str(temp)])
    print(v,'joining',len(clips),'complete source shots,',len(pts),'frames',flush=True)
    subprocess.run(cmd,check=True)
    validated=verify_join(temp,pts,clips)
    track=audio_info(temp)
    if aud:
        assert len(track)==1 and track[0]['sample_rate']==aud[0]['sample_rate']
        if abs(track[0]['start_seconds'])>1e-8 or abs(track[0]['duration_seconds']-float(duration))>0.023:
            raise ValueError(('The audio is not aligned to the source span',track))
        validated['source_audio_windows']=verify_audio(source,temp,first,last)
    else:
        assert not track
    temp.replace(video)
    write(TARGET/(edition+'_receipt.json'),dict(
        programme=manifest['programme_id'],binding_before_full_review_registration=pin(FILES[v]),scene=manifest['geometry'],
        source=manifest['source'],video=pin(video),source_first_pts=first,source_last_pts=last,
        source_first_seconds=first/12800,source_last_seconds=last/12800,source_time_base=str(TB),
        frame_rate=50,duration_seconds=float(duration),source_has_audio=bool(aud),
        output_audio=track, validation=validated,shots=clips))
    print(v,validated,video,round(time.monotonic()-begin,2),flush=True)


if __name__=='__main__':
    main()
