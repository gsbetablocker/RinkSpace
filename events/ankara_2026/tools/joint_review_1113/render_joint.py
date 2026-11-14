"""Render exact-clock full-shot reviews using the accepted-for-review shared candidate."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from time import monotonic
import json

import joint_inputs as ji
from joint_inputs import OUT, np, cv2, TB, STEP
import verify_joint as v
import solve_joint as s
import render_okada as draw
from rinkspace.review.scene import sample_native_review_scene
from rinkspace.review.render import source_frames, validate_video
from qualify import raw_pixels, segment_gap
from prepare_guarded import STOP_RAIL,PAINT_QUALITY,OKADA_WRONG_A4,role

import av

PREP=OUT/'prepared_guarded'
CANDIDATE=OUT/'candidate/review.npz'
AUDIT={}


def raw_selected(native,shot,ledger,decisions,used,raw,pt,idx):
    out=[]
    r=ledger[idx]
    if r['pts']!=pt:raise ValueError('Jin raw frame clock changed')
    h=ji.sha256(raw.tobytes()).hexdigest()
    if r['source_BGR_sha256']!=h:raise ValueError((shot.name,pt,'raw frame hash mismatch'))
    d=decisions.get(pt,{})
    if d.get('source_BGR_sha256') not in (None,h):
        raise ValueError('Candidate pixel ledger came from a different exposure')
    bad=defaultdict(set)
    for g in d.get('excluded',[]):
        if g.get('origin')=='saved':
            bad[g['path']].update(tuple(p) for p in g['uv'])
    for g in r['groups']:
        k=g['path']
        if k not in native.paths or g.get('check_only',False):continue
        f=ji.family(k,native)
        mat=g.get('material','unspecified')
        if shot.id in STOP_RAIL and f.startswith(('goal_line_','trapezoid_')) and mat not in PAINT_QUALITY and mat!='two_sided_original_paint_band':
            continue
        if shot.id=='s19' and f=='F1' and mat=='measured_paired_annulus':continue
        if shot.id=='s14' and f=='F1' and k.endswith('146/path0') and 122.94<=pt/12800<=123.42:
            # The exact correction has actual raw pixel coordinates; outside
            # that checked span no fresh points are displayed.
            if not d or 'F1_ambiguous_lower_shadow_arc' in {q['reason'] for q in d.get('excluded',[])}:
                continue
        a=np.asarray([p for p in g['uv'] if tuple(p) not in bad[k]],dtype=float).reshape(-1,2)
        if len(a):out.append(a)
    # Actual additions on fitted exposures only. Historic points already
    # bound to the earlier camera retain their saved native source times.
    if pt in used:
        for g in d['groups']:
            if g['origin']=='fresh':
                a=np.asarray(g['uv'],float)
                if len(a):out.append(a)
    return np.vstack(out) if out else np.empty((0,2)),h


def exact_projection_check(native,shot,rig,cs,i,world,u,status):
    j=np.arange(0,len(world),max(1,len(world)//550),dtype=int)
    q=native.core.project_motion(rig,cs,shot.first,shot.dt,1080,shot.times[i],world[j])
    p=q['uv']
    visible=(q['status']==0)&(p[:,1]>=0)&(p[:,1]<1080)
    if not visible.any():return 0.
    if (status[j[visible]]!=0).any():raise ValueError('Visible native projection lost: '+shot.name)
    err=float(np.linalg.norm(p[visible]-u[j[visible]],axis=1).max())
    if err>1.e-6:raise ValueError((shot.name,i,err))
    return err


def labels_and_world(native):
    sc=ji.read(native.scene_path)
    samples=sample_native_review_scene(sc,native.core,native.scene_path,
                    parameter_state=native.parameter_state,native=native)
    if {v['id'] for v in samples['cuts']}!=set(native.paths) or native.unlocalized:
        raise ValueError('Review omits native geometry')
    # Use the official sampling on ice/board and print. Batching and visible
    # time halo affect throughput only, never coordinates inside the frame.
    world,paths,side=draw.draw_config(samples)
    side[66:105]=22
    draw.text(side,'ANKARA | COMMON RINK',(18,84),.47,(60,210,255))
    side[-59:-23]=22
    draw.text(side,'Red: active original UV at this PTS',(18,1080-42),.42,(100,130,255))
    return world,paths,side


def latest_points(shot,name,context):
    if shot.program=='jin':
        idx=context['ji']['shots'][shot.id]
        return list(ji.lines(idx['evidence']['path']))
    return None


def okada_uv(ctx,name):
    rr=Path(ctx['ok_mref']['path']).parent
    arrays=ji.load_arrays(rr/'point_observations.npz',('pts','uv','path_indices'))
    mapping=ji.read(rr/'mapping.json')['path_ids']
    ids=[i for i,k in enumerate(mapping) if k=='demo_initial/A4:outer']
    keep=~(np.isin(arrays['pts'],list(OKADA_WRONG_A4))&np.isin(arrays['path_indices'],ids))
    rows=np.flatnonzero(keep)
    rows=rows[np.argsort(arrays['pts'][rows],kind='stable')]
    return arrays['pts'][rows],arrays['uv'][rows]


def clip(shot,native,state,target,world,paths,overview,jledger,ok_points,progress):
    t0=monotonic()
    n=len(shot.pts)
    current=state[shot.name+'_controls']
    rig=state[shot.name+'_rig']
    obs={v['pts']:v for v in ji.lines(PREP/(shot.name+'_decisions.jsonl'))}
    fitted={int(p) for p,g in obs.items() if g['fold']==0}
    # An active review frame cannot borrow another exposure's annotations.
    start,stop=int(shot.pts[0]),int(shot.pts[-1])
    enc=target/(shot.name+'.part.mp4')
    dst=target/(shot.name+'.mp4')
    raw=source_frames(shot.source,dict(time_base_num=1,time_base_den=12800,width=1920,height=1080),shot.pts)
    led=target/(shot.name+'_pts.jsonl')
    selected={0,n-1,round((n-1)*.23),round((n-1)*.51),round((n-1)*.77)}
    for sec in ji.JIN_FOCUS.get(shot.id,()) if shot.program=='jin' else ji.OKADA_FOCUS.get(shot.id,()):
        p=round(sec*12800)
        if start<=p<=stop:selected.add((p-start)//STEP)
    check_rows={}
    counts=Counter()
    with av.open(str(enc),'w',options={'movflags':'+faststart','video_track_timescale':'12800'}) as o,\
         led.open('x',encoding='utf-8',buffering=1<<17) as ledger:
        stream=o.add_stream('libx264',rate=50)
        stream.width=2560;stream.height=1144;stream.pix_fmt='yuv420p'
        stream.time_base=TB;stream.codec_context.time_base=TB
        stream.codec_context.thread_count=6
        stream.options={'crf':'19','preset':'veryfast','threads':'6'}
        for lo in range(0,n,10):
            rr=np.arange(lo,min(lo+10,n))
            vv=ji.project_groups(native,rig,current,shot.first,shot.dt,
                        [(int(i),'viewport',shot.times[i]) for i in rr],{'viewport':world})
            for i,(u,status) in zip(rr,vv,strict=True):
                pt,frame=next(raw)
                if pt!=shot.pts[i]:raise ValueError('Render consumed a different source PTS')
                if jledger is not None:
                    points,h=raw_selected(native,shot,jledger,obs,fitted,frame,pt,i)
                else:
                    source_pts,pixels=ok_points
                    l=np.searchsorted(source_pts,pt,side='left');r=np.searchsorted(source_pts,pt,side='right')
                    points=pixels[l:r];h=ji.sha256(frame.tobytes()).hexdigest()
                in_image=(points>=0).all(1)&(points<[1920,1080]).all(1)
                points=points[in_image]
                pic,visible,stages=draw.image(frame,u,status,paths,overview,points,pt,shot.id,int(i),n,None)
                if i in selected:
                    err=exact_projection_check(native,shot,rig[i],current,i,world,u,status)
                    check_rows[str(pt)]=err
                    # Local stills are source look-through samples of the same fit.
                    cv2.imwrite(str(target/f'{shot.name}_{pt}.jpg'),pic,[cv2.IMWRITE_JPEG_QUALITY,91])
                vf=av.VideoFrame.from_ndarray(pic,format='bgr24')
                vf.pts=pt-start;vf.time_base=TB
                for packet in stream.encode(vf):o.mux(packet)
                counts['active_points']+=len(points)
                counts['frames']+=1
                counts['frames_with_points']+=bool(len(points))
                ledger.write(json.dumps(dict(pts=pt,source_BGR_sha256=h,n_points=len(points),
                                            visible_geometry_samples=visible,status=stages),separators=(',',':'))+'\n')
        for packet in stream.encode():o.mux(packet)
    report=validate_video(enc,shot.pts,TB)
    enc.replace(dst)
    receipt=dict(**report,programme=shot.program,shot=shot.id,
        source_first_seconds=start/12800,source_last_seconds=stop/12800,
        source_first_pts=start,source_last_pts=stop,all_922_paths=True,
        native_projection_checks_px=check_rows,coverage=dict(counts),video=ji.pin(dst),
        source_frames=ji.pin(led),seconds=round(monotonic()-t0,3))
    ji.write(target/(shot.name+'_receipt.json'),receipt)
    progress.write(json.dumps(receipt,ensure_ascii=False,separators=(',',':'))+'\n')
    print(shot.name,n,round(monotonic()-t0,2),flush=True)


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('part',choices=('jin','okada'))
    ap.add_argument('--shots',nargs='+')
    ap.add_argument('--out',default='review_video')
    ap.add_argument('--candidate',default='candidate')
    a=ap.parse_args()
    cv2.setNumThreads(1)
    ctx=ji.context()
    s.INPUT=PREP
    m,core,b,series=s.load()
    candidate=OUT/a.candidate/'review.npz'
    state=v.load_checkpoint(candidate)
    native=s.NativeSceneGPU(m['scene_path'],core,parameter_state=s.ParameterState(m['scene_layout'],state['scene_values']))
    draw.TITLE=('S. JIN' if a.part=='jin' else 'MEI OKADA')+' | ANKARA | JOINT REVIEW'
    world,paths,overview=labels_and_world(native)
    target=OUT/a.out/a.part
    target.mkdir(parents=True,exist_ok=False)
    choices=[k for k,ss in m['shots'].items() if ss['program']==a.part and (not a.shots or ss['shot'] in a.shots)]
    jj=okada_uv(ctx,a.part) if a.part=='okada' else None
    with (target/'progress.jsonl').open('x',encoding='utf-8',buffering=1) as progress:
        for k in choices:
            shot=v.shot_from_inputs(k,m,b)
            z=latest_points(shot,k,ctx)
            clip(shot,native,state,target,world,paths,overview,z,jj,progress)
    ji.write(target/'review.json',dict(geometry_sha256=ji.numeric_hash(state['scene_values']),shots=choices,
                  numeric_state=ji.pin(candidate),path_count=len(paths),unlocalized=len(native.unlocalized)))
if __name__=='__main__':main()
