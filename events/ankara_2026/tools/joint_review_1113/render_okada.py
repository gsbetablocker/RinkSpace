
"""Render Okada on the Q874 shared rink using the original nine camera shots."""
from collections import Counter
from fractions import Fraction
import argparse
import re
from time import monotonic

import av
import cv2

from inspect_geometry import *
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.review.draw import COLORS,text,top_down
from rinkspace.review.render import source_frames,validate_video

TB=Fraction(1,12800)
STEP=256
W,H=1920,1080
TITLE='MEI OKADA | Q874 SHARED RINK | 1102 CAMERA'
SAMPLE_FRAMES=(0,.22,.5,.78,1.)
BATCH=8


def shared_scene(a):
    # The two separate Q874 bindings differ in only three installation
    # blocks. Ad18's shot-specific change regresses Okada's source fit.
    state=a['ss'].with_blocks({'installation/Ad18/shared_placement':a['js']['installation/Ad18/shared_placement']})
    native=NativeSceneGPU(a['s14_scene']['path'],a['jc'],parameter_state=state)
    samples=sample_native_review_scene(a['sread'],a['jc'],a['s14_scene']['path'],
                                     parameter_state=state,native=native)
    modified={r['block'] for r in changes(a['js'],state)}
    additions=set(native.paths)-set(a['jn'].paths)
    instances={k.split('::')[0] for k in additions}
    if (modified!={'installation/Ad15/shared_placement','installation/Ad16/shared_placement'} or
        instances!={'print_instance/Ad15','print_instance/Ad16'} or set(a['jn'].paths)-set(native.paths) or
        set(native.paths)!=set(k['id'] for k in samples['cuts']) or native.unlocalized):
        raise ValueError('Unified scene differs from Q874 common geometry outside the two localized ads')
    delta=compare(a['jn'],native)
    if delta['moved_paths'] or delta['removed_paths']:
        raise ValueError('An existing Q874 common path moved in the unified scene')
    return state,native,samples


def draw_config(samples):
    world=samples['xyz']
    cuts=samples['cuts']
    overview=top_down(world,cuts,missing_count=len(samples['inventory']['unlocalized_print_instance_ids']))
    overview[66:105]=22
    text(overview,'Q874 shared rink geometry',(18,84),.47,(60,210,255))
    overview[-59:-23]=22
    text(overview,'Red: saved source UV at this exposure',(18,H-42),.42,(100,130,255))
    paths=[]
    for item in cuts:
        match=re.search(r'Ad\d+',item['label'],re.I)
        label=match.group() if match else item['label'].removeprefix('ice.').split(':')[0]
        paths.append((item['begin'],item['end'],bool(item.get('closed')),
                      COLORS[item['layer']],label if len(label)<=16 else None))
    return world,paths,overview


def image(raw,uv,status,paths,overview,points,pt,shot,i,length,metrics,*,
          subpixel_bits=0,occluded=None,label_anchors=None,path_starts=None):
    canvas=raw.copy()
    valid=(status==0)&np.isfinite(uv).all(axis=1)
    in_image=valid&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    scale=1<<subpixel_bits
    pixels=np.rint(np.clip(np.nan_to_num(uv,nan=0.,posinf=1e7,neginf=-1e7),-1e7,1e7)*scale).astype(np.int32)
    labels={}
    by_color={}
    if path_starts is None:
        projected_paths=paths
    else:
        # Entire polylines outside the sensor cannot affect the image.
        # Reduce the same per-path bounding boxes before the drawing loop.
        limit=np.iinfo(np.int32).max
        visible=np.minimum.reduceat(np.where(valid[:,None],pixels,limit),path_starts,axis=0)
        far=np.maximum.reduceat(np.where(valid[:,None],pixels,-limit),path_starts,axis=0)
        hits=(visible<[W*scale,H*scale]).all(axis=1)&(far>=0).all(axis=1)
        projected_paths=(paths[j] for j in np.flatnonzero(hits))
    for start,stop,closed,color,label in projected_paths:
        p=pixels[start:stop];g=valid[start:stop]
        v=np.flatnonzero(in_image[start:stop])
        # Skip full offscreen polylines only when their bounding boxes
        # cannot cross the image. Preserve paths with both ends offscreen.
        if not len(v):
            p0=p[g]
            if (not len(p0) or p0[:,0].min()>=W*scale or p0[:,0].max()<0 or
                p0[:,1].min()>=H*scale or p0[:,1].max()<0):
                continue
        if g.all():
            if len(p)>1:by_color.setdefault((color,closed),[]).append(p)
        else:
            for a,b in np.flatnonzero(np.diff(np.r_[False,g,False])).reshape(-1,2):
                if b-a>1:by_color.setdefault((color,False),[]).append(p[a:b])
        if label_anchors is None and len(v) and label:
            labels.setdefault(label,[]).append(p[v[len(v)//2]])
    for (color,closed),p in by_color.items():
        cv2.polylines(canvas,p,closed,color,1,cv2.LINE_AA,shift=subpixel_bits)
    for label,v in labels.items():
        text(canvas,label,np.clip(np.median(v,axis=0)/scale,[5,24],[W-90,H-20]),.46)
    if label_anchors is not None:
        for label,index in label_anchors.items():
            if in_image[index]:
                x,y=uv[index]
                if 7<=x<W-52 and 26<=y<H-5:
                    at=(int(x),int(y)-4)
                    if occluded is None or not occluded[int(y),int(x)]:
                        text(canvas,label,at,.44)
    for xy in np.rint(points*scale).astype(np.int32):
        cv2.circle(canvas,tuple(xy),2*scale,(0,0,255),-1,cv2.LINE_AA,shift=subpixel_bits)
    if occluded is not None:
        cv2.copyTo(raw,occluded,canvas)
        visible_indices=np.flatnonzero(in_image)
        xy=np.floor(uv[visible_indices]).astype(np.int64)
        in_image[visible_indices]&=occluded[xy[:,1],xy[:,0]]==0
    side=overview.copy()
    text(side,f'Source PTS {pt}',(18,H-68),.44)
    header=np.full((64,2560,3),18,np.uint8)
    text(header,TITLE,(16,25),.62,(55,215,255))
    text(header,f'{shot.upper()} | Source {pt/12800:.3f}s | PTS {pt} | frame {i+1}/{length} | UV {len(points)}',(16,50),.50)
    if metrics is not None:
        u,v=metrics
        text(header,f'Same camera & source points | RMS original {u:.2f} px  /  shared {v:.2f} px',
             (1938,50),.40)
    return np.vstack((header,np.hstack((canvas,side)))),int(in_image.sum()),dict(Counter(map(int,status)))


def project_batch(native,core,rig,ctrl,om,times,world):
    n=len(world)
    m=len(times)
    with ResidentResidualProblem(core,board=native.scene['board'],layout=native.scene['shared_layout'],
         options=native.scene['shared_geometry_options'],rig=rig,controls=ctrl,
         first_control_time=om['first_control_time'],control_dt=om['control_dt'],image_height=H,
         codes=np.zeros(m*n,np.int32),stations=np.tile(np.linspace(0.,1.,n),m),
         times=np.repeat(times,n),observed=np.zeros((m*n,2))) as resident:
        resident.configure_geometry([dict(kind=4,vertices=world,parameters=[])])
        resident.configure_projection_time_support(half_seconds=om['max_readout']/2)
        z=resident.evaluate('cuda')
    uv=z['values'][:,3:5].reshape(m,n,2)
    status=z['status'].reshape(m,n)
    # The residual backend applies a stricter row-root guard to some offscreen
    # samples. For those use the same saved DLL's native display projection.
    for i,t in enumerate(times):
        missing=status[i]==4
        if missing.any():
            q=core.project_motion(rig,ctrl,om['first_control_time'],om['control_dt'],H,t,world[missing])
            uv[i,missing]=q['uv']
            status[i,missing]=q['status']
    return uv,status


def consistency(native,core,rig,ctrl,om,t,world,uv,status):
    idx=np.arange(0,len(world),max(1,len(world)//350),dtype=int)
    w=world[idx]
    z=core.project_motion(rig,ctrl,om['first_control_time'],om['control_dt'],H,t,w)
    both=(z['status']==0)&(status[idx]==0)
    if not both.any() or not np.array_equal(z['status']==0,status[idx]==0):
        raise ValueError((int(both.sum()),dict(Counter(map(int,z['status']))),dict(Counter(map(int,status[idx]))),len(world)))
    error=np.max(np.linalg.norm(uv[idx[both]]-z['uv'][both],axis=1))
    if error>1.e-6:
        raise ValueError(('Overlay uses a different exposure or motion projection',error))
    return float(error)


def clips(a,output,mode):
    from itertools import islice
    from collections import defaultdict
    from rinkspace.path_policy import guard_project_write

    output=guard_project_write(output)
    output.mkdir(exist_ok=False)
    state,native,samples=shared_scene(a)
    world,paths,overview=draw_config(samples)
    om,orig,source,core=[a[k] for k in ('om','orig','source','old_core')]
    source_ref=pin(source['source']['registered_path'],source['source']['sha256'])
    src={**source['source'],**source['source_clock']}
    clock=ShotMotionTimeline(om['shots'],om['control_dt'],om['first_control_time'])
    scope=source['processing_scope']
    allpts=np.arange(scope['first_pts'],scope['stop_pts_exclusive'],STEP,dtype=np.int64)
    assert len(allpts)==8329 and len(om['shots'])==9
    pfile=Path(a['old_receipt']['frozen_run_root'])/'point_observations.npz'
    pref=pin(pfile,om['point_input_sha256'])
    p=load_arrays(pfile,('pts','uv','path_indices'))
    if not np.array_equal(np.unique(p['pts']),allpts):
        raise ValueError('The original rows no longer cover all nine source shots')
    with np.load(OUT/'impact_errors.npz',allow_pickle=False) as err:
        old,new=err['old'],err['common']
    rows=np.argsort(p['pts'],kind='stable')
    pix=p['uv'][rows]
    ticks=p['pts'][rows]
    row_offsets=np.searchsorted(ticks,np.r_[allpts,allpts[-1]+STEP])
    old,new=old[rows],new[rows]
    t0=monotonic()
    mark_pts=set()
    for shot in om['shots']:
        ab=np.arange(shot['first_pts'],shot['stop_pts_exclusive'],STEP)
        mark_pts.update(ab[round((len(ab)-1)*k)] for k in SAMPLE_FRAMES)
    ns=read(Path(a['old_receipt']['frozen_run_root'])/'mapping.json')['path_ids']
    position=np.argsort(rows,kind='stable')
    for k in ('goal_line_left','Ad10','Ad17','Ad18','Ad23'):
        if k.startswith('Ad'):
            i=np.flatnonzero([s.startswith(f'print_instance/{k}::') for s in ns])
        else:
            i=np.flatnonzero([k in s for s in ns])
        m=np.isin(p['path_indices'],i)
        if m.any():
            x=np.flatnonzero(m)
            rank=np.argsort(np.abs(new[position[x]]-old[position[x]]))
            mark_pts.update(np.unique(p['pts'][x[rank[-120:]]])[-3:].tolist())
    if mode=='preview':
        # One source-frame per selected time, before the complete encode.
        mark_pts=set(sorted(mark_pts)[::2])|{283136,2415104}
    model_info=dict(q874_common=a['base_scene'],q874_localized_installations=a['s14_scene'],
                    q874_installation_values=a['s14_ref'],
                    q874_common_numeric=a['base_ref'],value_sha256=numeric_hash(state.values),
                    displayed_native_paths=len(paths),unlocalized=len(native.unlocalized),
                    numerical_geometry=a['jc_ref'],native_camera_projection=a['old_core_ref'])
    write(output/'render_binding.json',dict(geometry=model_info,source=source_ref,observations=pref,
          camera=a['ok_sref'],q874_index=a['ji_ref'],source_first_pts=int(allpts[0]),source_last_pts=int(allpts[-1]),
          frozen_geometry_no_fit=True,mode=mode))
    if mode=='preview':
        selected=np.asarray(sorted(mark_pts),dtype=np.int64)
        # Read original frames through the registered full clock; no time
        # seek rounding or interpolation can enter the image verification.
        sf=source_frames(source_ref['path'],src,allpts)
        by={s['id']:s for s in om['shots']}
        active=om['shots'][0]
        needed=set(selected.tolist())
        with (output/'preview.jsonl').open('x',encoding='utf8') as log:
            for pt,bgr in sf:
                if pt>selected[-1]:break
                if pt not in needed:continue
                active=next(s for s in by.values() if s['first_pts']<=pt<s['stop_pts_exclusive'])
                j=(pt-int(allpts[0]))//STEP
                c0,c1=row_offsets[j:j+2]
                rig=orig['rig'][active['camera_index']]
                tm=clock.exposure_time(active,pt)
                uv,st=project_batch(native,core,rig,orig['controls'],om,np.asarray([tm]),world)
                eq=consistency(native,core,rig,orig['controls'],om,tm,world,uv[0],st[0])
                img,visible,stages=image(bgr,uv[0],st[0],paths,overview,pix[c0:c1],pt,active['id'],
                    (pt-active['first_pts'])//STEP,(active['stop_pts_exclusive']-active['first_pts'])//STEP,
                    (float(np.sqrt(np.mean(old[c0:c1]**2))),float(np.sqrt(np.mean(new[c0:c1]**2)))))
                cv2.imwrite(str(output/f"{active['id']}_{pt}.jpg"),img,[cv2.IMWRITE_JPEG_QUALITY,92])
                log.write(json.dumps(dict(pts=pt,shot=active['id'],visible=visible,status=stages,projection_check_px=eq))+'\n')
        print('Original preview sample count',len(selected),flush=True)
        return
    partial=output/'partial.mp4'
    video=output/'okada_mei_sp_q874_shared_geometry_50fps.mp4'
    verify={}
    src_path=source_ref['path']
    with av.open(str(partial),'w',options={'movflags':'+faststart','video_track_timescale':'12800'}) as target, \
         (output/'source_frames.jsonl').open('x',encoding='utf8',buffering=1<<19) as led, \
         (output/'render_progress.jsonl').open('x',encoding='utf8',buffering=1) as log:
        stream=target.add_stream('libx264',rate=50)
        stream.width=2560
        stream.height=1144
        stream.pix_fmt='yuv420p'
        stream.time_base=TB
        stream.codec_context.time_base=TB
        stream.codec_context.thread_count=7
        stream.options={'crf':'18','preset':'veryfast','threads':'7'}
        raw=source_frames(src_path,src,allpts)
        done=0
        for shot in om['shots']:
            sid=shot['id']
            p0,p1=shot['first_pts'],shot['stop_pts_exclusive']
            n=(p1-p0)//STEP
            rig=orig['rig'][shot['camera_index']]
            checked=[]
            status_total=Counter()
            vis_min=len(world)
            for start in range(0,n,BATCH):
                chunk=list(islice(raw,min(BATCH,n-start)))
                pp=np.array([t for t,f in chunk],dtype=np.int64)
                want=np.arange(p0+start*STEP,min(p0+(start+BATCH)*STEP,p1),STEP)
                if not np.array_equal(pp,want):
                    raise ValueError('Missing native shot exposures during render')
                t=clock.exposure_time(shot,pp)
                uv,status=project_batch(native,core,rig,orig['controls'],om,t,world)
                for j,(pt,bgr) in enumerate(chunk):
                    fidx=start+j
                    lo,hi=row_offsets[done:done+2]
                    check_frame=pt in mark_pts
                    if check_frame:
                        checked.append(dict(pts=pt,projection_error_px=consistency(
                            native,core,rig,orig['controls'],om,t[j],world,uv[j],status[j])))
                    metrics=(float(np.sqrt(np.mean(old[lo:hi]**2))),float(np.sqrt(np.mean(new[lo:hi]**2))))
                    picture,visible,states=image(bgr,uv[j],status[j],paths,overview,pix[lo:hi],pt,sid,fidx,n,metrics)
                    if not visible:raise ValueError('A source frame has no valid visible rink samples')
                    if check_frame and fidx in (0,n//2,n-1):
                        cv2.imwrite(str(output/f'{sid}_{pt}.jpg'),picture,[cv2.IMWRITE_JPEG_QUALITY,94])
                    vis_min=min(vis_min,visible)
                    status_total.update(states)
                    q=av.VideoFrame.from_ndarray(picture,format='bgr24')
                    q.pts=pt-int(allpts[0])
                    q.time_base=TB
                    for packet in stream.encode(q):target.mux(packet)
                    led.write(json.dumps(dict(shot=sid,source_pts=pt,output_pts=q.pts,
                        source_active_uv=int(hi-lo),visible_samples=visible,status=states),separators=(',',':'))+'\n')
                    done+=1
                    if done%300==0 or fidx==n-1:
                        row=dict(shot=sid,shot_frames=fidx+1,total_frames=done,expected=len(allpts),
                                 elapsed_seconds=round(monotonic()-t0,2),latest_output=str(partial))
                        log.write(json.dumps(row)+'\n')
                        print(row,flush=True)
            verify[sid]=dict(frames=n,source_first_pts=p0,source_last_pts=p1-STEP,
                             min_visible_samples=vis_min,native_camera_check=checked,projection_counts=dict(status_total))
        if done!=len(allpts):raise ValueError('Incomplete source programme')
        for packet in stream.encode():target.mux(packet)
    proof=validate_video(partial,allpts,TB)
    partial.replace(video)
    write(output/'receipt.json',dict(schema='rinkspace.okada_shared_rink_review.v1',geometry=model_info,
          source=source_ref,source_seconds=[int(allpts[0])/12800,int(allpts[-1])/12800],
          frame_rate='50/1',time_base='1/12800',shots=verify,verification=proof,video=pin(video),
          original_camera=a['ok_sref'],q874_index=a['ji_ref'],geometry_changed_during_replay=False,
          no_camera_refit=True,no_new_source_points=True,source_frame_ledger=pin(output/'source_frames.jsonl'),
          measured_impact=pin(OUT/'point_impact.json'),user_visual_acceptance=False))
    print(str(video),proof,flush=True)


if __name__=='__main__':
    cv2.setNumThreads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('preview','full'))
    p.add_argument('out',type=Path)
    args=p.parse_args()
    clips(context(),args.out,args.mode)
