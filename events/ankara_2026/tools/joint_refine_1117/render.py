"""Complete source-PTS review of the current four-programme joint candidate."""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from itertools import islice
from pathlib import Path
from time import monotonic
import json
import shutil
import subprocess
import av
import cv2
import numpy as np
from shared import prepared_packet_path
from shared import DATA,EVENT,OUT,PREPARED,PROGRAMMES,arrays,read,sha,write,programme_state_sha
from review import camera_state,draw,ji,native_for,sample,source_frames
from source_paint import matte
from rinkspace.review.render import validate_video
from rinkspace.core.gpu import ResidentResidualProblem

TB=Fraction(1,12800)


def graphics(raw,pt,start,mask):
    """Source broadcaster cards, following the existing Chen source renderer."""
    if mask is None:mask=np.zeros((1080,1920),np.uint8)
    if pt<start:return mask
    if pt>=start+256:
        cv2.fillConvexPoly(mask,np.array([[1694,949],[1859,949],[1859,990],[1820,993],
            [1777,995],[1732,994],[1694,991]],np.int32),1)
    def card(top,bottom):
        cyan=raw[top:top+5,83:389].astype(np.int16)
        magenta=raw[bottom-7:bottom-2,83:389].astype(np.int16)
        return np.count_nonzero((np.median(cyan[:,:,0]-cyan[:,:,2],axis=0)>17)&
            (np.median(magenta[:,:,2]-magenta[:,:,1],axis=0)>37))>51
    if card(31,62):
        for box in ((34,30,75,61),(34,69,75,100),
                    (80,30,491 if pt>=start+14*256 else 294,62),
                    (80,69,491 if pt>=start+round(.85*12800) else 346,100)):
            cv2.rectangle(mask,box[:2],box[2:],1,-1)
        rank=raw[33:48,506:537].astype(np.int16)
        if np.median(rank[:,:,2]-rank[:,:,0])>42:cv2.rectangle(mask,(497,30),(550,65),1,-1)
    if card(108,140):
        cv2.rectangle(mask,(80,107),(392,216),1,-1)
        for left in (80,107,134,161,188,215,242):
            chip=raw[227:246,left+1:left+24].astype(np.int16)
            if (np.median(chip[:,:,1]-chip[:,:,2])>27 or
                (np.median(chip.mean(2))>213 and np.median(chip.max(2)-chip.min(2))<22)):
                cv2.rectangle(mask,(left,224),(left+25,250),1,-1)
    return mask


def drawing(native):
    samples=sample(native)
    assert {c['id'] for c in samples['cuts']}==set(native.paths)
    assert not native.unlocalized
    world,paths,overview=draw.draw_config(samples)
    overview[40:64]=22
    draw.text(overview,f'Model inventory: {len(paths)} paths / 26 advertisements',(18,57),.42)
    overview[66:105]=22
    draw.text(overview,'ANKARA | FOUR PROGRAMMES | FREE GEOMETRY',(18,84),.43,(60,210,255))
    overview[990:]=22
    draw.text(overview,'Red: actual fit UV at this source exposure',(18,1038),.42,(100,130,255))
    draw.text(overview,'Geometry coordinates: rink length = 1',(18,1062),.40)
    starts=np.array([p[0] for p in paths],np.int64)
    anchors={}
    for label in {p[4] for p in paths if p[4]}:
        ix=np.concatenate([np.arange(p[0],p[1]) for p in paths if p[4]==label])
        middle=np.median(world[ix],axis=0)
        anchors[label]=int(ix[np.argmin(np.sum((world[ix]-middle)**2,axis=1))])
    return world,paths,overview,starts,anchors,samples['inventory']


def projected(native,rig,controls,s,times,indices,world):
    # The display API certifies the full saved spline support. Use exactly that
    # support on CUDA too, including off-raster continuation; a fit's shortened
    # computational window otherwise sends many off-raster vertices to CPU.
    count=len(world);m=len(indices)
    rigs,camera_id=np.unique(rig[indices],axis=0,return_inverse=True)
    with ResidentResidualProblem(native.core,board=native.scene['board'],
        layout=native.scene['shared_layout'],options=native.scene['shared_geometry_options'],
        rig=rigs if len(rigs)>1 else rigs[0],controls=controls,
        first_control_time=s['first'],control_dt=s['dt'],image_height=1080,
        codes=np.zeros(m*count,np.int32),stations=np.tile(np.linspace(0.,1.,count),m),
        times=np.repeat(times[indices],count),observed=np.zeros((m*count,2)),
        camera_ids=np.repeat(camera_id,count)) as resident:
        resident.configure_geometry([dict(kind=4,vertices=world,parameters=[])])
        result=resident.evaluate('cuda')
    uv=result['values'][:,3:5].reshape(m,count,2);status=result['status'].reshape(m,count)
    return list(zip(uv,status))


def project_check(native,rig,controls,s,time,world,uv,status):
    ix=np.arange(0,len(world),max(1,len(world)//500))
    z=native.core.project_motion(rig,controls,s['first'],s['dt'],1080,float(time),world[ix])
    good=z['status']==0
    gpu_good=status[ix]==0
    inside=lambda xy:(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
    # Compare on-sensor visibility and all commonly certified coordinates;
    # both backends now use the same complete saved spline support.
    cpu_visible=good&inside(z['uv']);gpu_visible=gpu_good&inside(uv[ix])
    if not np.array_equal(cpu_visible,gpu_visible):raise ValueError('CPU/GPU on-sensor projection status differs')
    if not np.array_equal(good,gpu_good):raise ValueError('CPU/GPU full-support certification differs')
    common=good&gpu_good
    difference=np.linalg.norm(z['uv'][common]-uv[ix][common],axis=1)
    scale=np.maximum(1.,np.maximum(np.linalg.norm(z['uv'][common],axis=1),np.linalg.norm(uv[ix][common],axis=1)))
    error=float(np.max(difference))
    visible_error=float(np.max(np.linalg.norm(z['uv'][cpu_visible]-uv[ix][cpu_visible],axis=1))) if cpu_visible.any() else 0.
    # Far outside the raster, a certified ray can approach the horizon and
    # produce coordinates of tens of millions of pixels. Compare floating
    # point agreement relatively there; retain the original absolute on-sensor
    # requirement and identical visibility/status checks.
    if visible_error>1.e-6 or np.any(difference>1.e-6+1.e-12*scale):
        raise ValueError(('CPU/GPU projection differs',error,visible_error,float(np.max(difference/scale))))
    return dict(maximum_common_uv_error_px=error,checked_vertices=len(ix),
        maximum_on_sensor_uv_error_px=visible_error,maximum_relative_uv_error=float(np.max(difference/scale)),
        on_sensor_vertices=int(cpu_visible.sum()),off_sensor_status_differences=int(np.count_nonzero(good!=gpu_good)))


def audio_source(s):
    source=Path(s['source']);audio=source.with_suffix('.m4a')
    if audio.exists():return audio
    with av.open(str(source)) as container:
        if len(container.streams.audio):return source
    return None


def render(candidate,programme,preview=False,version_label='joint_free',source_mattes=False):
    cv2.setNumThreads(1)
    begin=monotonic();setup=read(PREPARED/'setup.json');base=camera_state(candidate)
    shots={n:s for n,s in setup['shots'].items() if s['program']==programme}
    native=native_for(candidate,programme)
    world,paths,overview,starts,anchors,inventory=drawing(native)
    folder=OUT/candidate/('render_preview' if preview else 'review')/programme
    folder.mkdir(parents=True,exist_ok=False)
    first=next(iter(shots.values()));source=Path(first['source'])
    if sha(source)!=first['source_sha256']:raise ValueError('Source identity changed')
    allpts=np.concatenate([base[n+'_pts'] for n in shots])
    if not np.all(np.diff(allpts)==256):raise ValueError('Missing assigned shot exposures')
    scope=read(EVENT/'programmes'/PROGRAMMES[programme]/'rebuild_scope.json')
    cues=scope.get('programme_timer_cues',[])
    start=next((int(c['short_pts'])-3*256 for c in cues if c['display']=='0:00'),int(allpts[0]))
    write(folder/'binding.json',dict(candidate=candidate,programme=programme,
        state=str(OUT/candidate/'state.npz'),state_sha256=sha(OUT/candidate/'state.npz'),
        programme_state_sha256=programme_state_sha(base,shots),
        source=str(source),source_sha256=first['source_sha256'],
        scene=str(native.scene_path),scene_sha256=sha(native.scene_path),inventory=inventory,
        observation_packets={n:sha(prepared_packet_path(n)) for n in shots},
        prepared_directory=str(PREPARED),
        original_source_person_masks=source_mattes,
        source_first_pts=int(allpts[0]),source_last_pts=int(allpts[-1]),native_frames=len(allpts),
        all_paths_drawn_without_error_filter=True,points='only current admitted fit UV at the same source PTS',
        projection_support='complete saved spline support, identical native display and CUDA forward',
        user_visual_acceptance=False,mode='preview' if preview else 'complete'))
    draw.TITLE=programme.upper()+' | ANKARA | FOUR-PROGRAMME JOINT GEOMETRY'
    if not version_label.replace('_','').isalnum():raise ValueError('Invalid video version label')
    partial=folder/'source_grid_silent.mp4';output=folder/(programme+'_'+version_label+'_50fps.mp4')
    mux=None;encoder=None
    if not preview:
        mux=av.open(str(partial),'w',options={'movflags':'+faststart','video_track_timescale':'12800'})
        encoder=mux.add_stream('libx264',rate=50);encoder.width=2560;encoder.height=1144
        encoder.pix_fmt='yuv420p';encoder.time_base=TB;encoder.codec_context.time_base=TB
        encoder.codec_context.thread_count=6;encoder.options={'crf':'19','preset':'veryfast','threads':'6'}
    counts={};total=0
    try:
      with (folder/'source_frames.jsonl').open('x',encoding='utf-8',buffering=1<<19) as ledger:
       for name,s in shots.items():
        pts=base[name+'_pts'];rig=base[name+'_rigs'];controls=base[name+'_controls'];times=base[name+'_times']
        with np.load(prepared_packet_path(name)) as packet:
            point_pts=packet['pts'];order=np.argsort(point_pts,kind='stable')
            a={k:packet[k][order] for k in ('fold','uv')}
        offsets=np.searchsorted(point_pts[order],np.r_[pts,pts[-1]+256])
        masks=matte(s,pts)
        selected=np.unique(np.rint(np.array([0,.25,.5,.75,1])*(len(pts)-1)).astype(int)) if preview else np.arange(len(pts))
        frames=source_frames(source,pts[selected]);seen=0;visible_min=len(world);zero_points=0;mask_missing=0
        checks={};camera_status=Counter();curve_status=Counter()
        # Every exposure has a native camera evaluation, including unmeasured frames.
        unique,inv=np.unique(rig[selected],axis=0,return_inverse=True)
        for i,r in enumerate(unique):
            z=native.core.evaluate_motion_camera(r,controls,s['first'],s['dt'],times[selected[inv==i]])
            camera_status.update(map(int,z['status']))
        if any(k!=0 for k in camera_status):raise ValueError((name,'invalid source cameras',camera_status))
        for lo in range(0,len(selected),8):
            indices=selected[lo:lo+8];chunk=list(islice(frames,len(indices)))
            np.testing.assert_array_equal([p for p,_ in chunk],pts[indices])
            images=projected(native,rig,controls,s,times,indices,world)
            for index,(pt,raw),(uv,status) in zip(indices,chunk,images):
                startrow,stoprow=offsets[index:index+2]
                use=np.isin(a['fold'][startrow:stoprow],base.get(name+'_active_folds',[0]))
                points=a['uv'][startrow:stoprow][use]
                mask=None
                if masks is not None and masks['valid'][index]:
                    mask=cv2.dilate(np.unpackbits(masks['packed'][index],axis=1,count=1920),np.ones((3,3),np.uint8))
                elif source_mattes and masks is None:
                    from lane_end_evidence import display_foreground
                    original=display_foreground(s,index,pt,raw)
                    if original is not None:mask=cv2.dilate((original!=0).astype(np.uint8),np.ones((3,3),np.uint8))
                has_person_mask=mask is not None
                mask_missing+=int(not has_person_mask)
                mask=graphics(raw,pt,start,mask)
                pic,visible,states=draw.image(raw,uv,status,paths,overview,points,pt,s['shot'],int(index),len(pts),None,
                    subpixel_bits=4,occluded=mask,label_anchors=anchors,path_starts=starts)
                checked=index in (0,len(pts)//2,len(pts)-1) or preview
                if checked:
                    checks[str(pt)]=project_check(native,rig[index],controls,s,times[index],world,uv,status)
                    cv2.imwrite(str(folder/(name+'_'+str(pt)+'.jpg')),pic,[cv2.IMWRITE_JPEG_QUALITY,94])
                if encoder is not None:
                    f=av.VideoFrame.from_ndarray(pic,format='bgr24');f.pts=int(pt-allpts[0]);f.time_base=TB
                    for packet in encoder.encode(f):mux.mux(packet)
                ledger.write(json.dumps(dict(shot=name,pts=int(pt),fit_points=len(points),visible_samples=visible,
                    foreground_matte=has_person_mask,projection_status=states))+'\n')
                curve_status.update(states);visible_min=min(visible_min,visible);zero_points+=int(not len(points));seen+=1;total+=1
        if seen!=len(selected):raise ValueError('Incomplete source review '+name)
        counts[name]=dict(frames=seen,first_pts=int(pts[0]),last_pts=int(pts[-1]),
            native_camera_status=dict(camera_status),projected_vertex_status=dict(curve_status),
            visible_samples_min=visible_min,frames_without_fit_points=zero_points,
            frames_without_foreground_matte=mask_missing,cpu_gpu_checks_px=checks)
        write(folder/'progress.json',dict(completed_shots=counts,frames=total,seconds=monotonic()-begin))
        print(name,seen,'frames',round(monotonic()-begin,1),'seconds',flush=True)
       if encoder is not None:
        for packet in encoder.encode():mux.mux(packet)
    finally:
        if mux is not None:mux.close()
    if preview:
        write(folder/'coverage.json',dict(shots=counts,frames=total,mode='selected source previews',user_visual_acceptance=False));return
    aud=audio_source(first)
    if aud is None:shutil.move(partial,output)
    else:
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise ValueError('ffmpeg unavailable for original-source audio')
        cmd=[ffmpeg,'-nostdin','-v','error','-i',str(partial),'-ss',str(allpts[0]/12800.),'-i',str(aud),
            '-map','0:v:0','-map','1:a:0','-t',str(len(allpts)/50.),'-c:v','copy','-c:a','aac','-b:a','192k',
            '-movflags','+faststart','-video_track_timescale','12800',str(output)]
        subprocess.run(cmd,check=True)
        # Only this renderer's verified temporary silent output is removed.
        if not partial.resolve().is_relative_to(OUT.resolve()):raise ValueError('Unexpected temporary path')
        partial.unlink()
    validation=validate_video(output,allpts,TB)
    write(folder/'coverage.json',dict(shots=counts,frames=total,validation=validation,
        video=str(output),source_seconds=[float(allpts[0]/12800.),float(allpts[-1]/12800.)],
        source_audio=str(aud) if aud else None,seconds=monotonic()-begin,user_visual_acceptance=False,
        visual_scope='selected stills inspected separately; every source frame rendered and PTS decoded'))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True)
    ap.add_argument('--programmes',nargs='+',choices=tuple(PROGRAMMES),default=list(PROGRAMMES))
    ap.add_argument('--source-mattes',action='store_true',help='Use registered native-exposure person pixels for all programmes')
    ap.add_argument('--preview',action='store_true');ap.add_argument('--version-label',default='joint_free');args=ap.parse_args()
    for p in args.programmes:render(args.candidate,p,args.preview,args.version_label,args.source_mattes)
