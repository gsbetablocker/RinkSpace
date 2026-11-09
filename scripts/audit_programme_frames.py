"""Inspect every native source exposure and create a synchronized before/after review."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np

from calibration.program_tracking.body_tracking.programme_projection import required_clock
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
from rinkspace.review.render import source_frames, validate_video

TB = Fraction(1, 12800)
SIZE = (1920, 1080)
STEEL = {'runner_lower_edge', 'runner_upper_edge'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def note(kind, **fields):
    print(json.dumps(dict(event=kind, **fields), ensure_ascii=False), flush=True)


def bound_video(receipt_path, expected, variant):
    receipt = read(receipt_path)
    path = Path(receipt['video'])
    clock = receipt['verified_clock']
    if (digest(path) != receipt['video_sha256'] or receipt['variant'] != variant
            or not clock['native_PTS_equal'] or clock['display_speed'] != 1
            or clock['decoded_frames'] != len(expected)
            or receipt['source_first_last_seconds'] != [float(expected[0]*TB), float(expected[-1]*TB)]):
        raise ValueError('Video does not match the requested exact source and variant: '+str(path))
    return path, receipt


def encoded_frames(path, pts):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 3
        count = 0
        for frame in container.decode(stream):
            if count >= len(pts) or Fraction(frame.pts)*frame.time_base != int(pts[count]-pts[0])*TB:
                raise ValueError('Unexpected encoded presentation timestamp in '+str(path))
            if (frame.width, frame.height) != SIZE:
                raise ValueError('Encoded frame dimensions differ from the registered source')
            yield frame.to_ndarray(format='bgr24')
            count += 1
        if count != len(pts):
            raise ValueError('Encoded frame count differs from the source')


def arrays(path, variant, pts, shots):
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pts'], pts)
        np.testing.assert_array_equal(z['shot_ids'], shots)
        np.testing.assert_array_equal(z['time_base'], [1, 12800])
        prefix = variant+'_'
        return dict(runner=z[prefix+'runner_edges_uv'], boot=z[prefix+'boot_outline_uv'],
                    com=z[prefix+'com_ice_uv'], coco=z[prefix+'COCO_uv'])


def cues_by_frame(path, evidence):
    data = read(path)
    result = [defaultdict(list), defaultdict(list)]
    source_rows = {r['raw_group_id']:r for r in evidence['ordered_contours']}
    for bucket, name in zip(result, ('old','candidate'), strict=True):
        for row in data[name]:
            mark = source_rows[row['id']]
            if (row['pts'],row['foot'],row['role'],row['semantic']) != (
                    mark['pts'],mark['foot'],mark['role'],mark['semantic']):
                raise ValueError('Source role or material identity changed')
            bucket[row['pts'],row['foot']].append(dict(
                row, derived=bool(mark.get('provenance',{}).get('depends_on_track_anchor'))))
    return result


def sample(image, xy):
    xy = np.asarray(xy, dtype=np.float32)
    return cv2.remap(image, xy[...,0], xy[...,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def cuff_cue(lab, uv, use, foot):
    """Read a skin-to-neutral cuff transition using detector position only as a search window."""
    k, a = uv[13+foot], uv[15+foot]
    if not (use[13+foot] and use[15+foot] and np.isfinite([k,a]).all()):
        return None, 'detector_missing'
    d = a-k
    length = float(np.linalg.norm(d))
    if length < 18:
        return None, 'foreshortened_search_window'
    across = np.array([-d[1],d[0]])/length
    if use[16-foot] and np.linalg.norm(a-uv[16-foot]) < max(16.,.10*length):
        return None, 'feet_overlap_in_source_hint'
    along = d/length
    offsets = np.arange(-int(np.clip(.32*length,12,52)),int(np.clip(.36*length,15,55))+1)
    half_width = np.clip(.042*length,3,11)
    sites = a + offsets[:,None,None]*along + np.linspace(-half_width,half_width,9)[None,:,None]*across
    if np.any((sites<1)|(sites>np.asarray(SIZE)-2)):
        return None, 'search_window_cropped'
    patch = sample(lab, sites).astype(np.float32)
    L = patch[:,:,0]
    chroma = np.linalg.norm(patch[:,:,1:]-128.,axis=-1)
    warm = patch[:,:,2]-128.
    pre = warm[:-8]
    post = chroma[8:]
    diff = np.median((warm[:-8]-warm[8:])+.18*(L[8:]-L[:-8]), axis=1)
    valid = ((np.median(pre,axis=1)>9.) & (np.median(post,axis=1)<12.)
             & (np.median(L[8:],axis=1)>105.) & (diff>9.)
             & ((pre-warm[8:]>5.).sum(axis=1)>=6))
    if not np.any(valid):
        return None, 'no_resolved_cuff_in_search_window'
    best = int(np.argmax(np.where(valid,diff,-1.)))
    competitors = np.abs(np.arange(len(diff))-best)>=10
    if np.any(valid & competitors & (diff>.90*diff[best])):
        return None, 'nonunique_cuff_candidate'
    cue = a + float(offsets[best+4])*along
    return cue, 'cuff_candidate'


def runner_profile(gray, runners):
    """Raw image edge support near projected steel. No material identity is inferred."""
    centres = runners[:,8:73:4]
    t = runners[:,9:74:4]-runners[:,7:72:4]
    n = np.stack((-t[...,1],t[...,0]),axis=-1)/np.maximum(np.linalg.norm(t,axis=-1,keepdims=True),1.e-8)
    good = np.isfinite(centres).all(axis=-1) & np.isfinite(n).all(axis=-1)
    good &= ((centres>=22)&(centres<np.asarray(SIZE)-22)).all(axis=-1)
    best = None
    shifts = np.array([-18,-12,-8,-5,-3,0,3,5,8,12,18],dtype=float)
    for p,normal,valid in zip(centres,n,good,strict=True):
        if valid.sum()<8 or np.linalg.norm(p[-1]-p[0])<12.:
            continue
        q=p[valid,None,:] + shifts[None,:,None]*normal[valid,None,:]
        b=normal[valid,None,:]*1.7
        diff=sample(gray,q+b).astype(float)-sample(gray,q-b)
        # A single highlight does not count as coherent support along the runner.
        strength=np.maximum(np.median(diff,axis=0),np.median(-diff,axis=0))
        j=int(np.argmax(strength))
        picked=(float(strength[j]), float(strength[5]), float(shifts[j]))
        if best is None or picked[0]>best[0]:
            best=picked
    return best if best is not None else (None,None,None)


def footprint_gap(point, outline):
    if point is None:
        return None
    uv=outline.reshape(-1,2)
    uv=uv[np.isfinite(uv).all(axis=1)]
    if len(uv)<3:
        return None
    hull=cv2.convexHull(uv.astype(np.float32))
    return max(0.,-float(cv2.pointPolygonTest(hull,tuple(map(float,point)),True)))


def bounds(obs, ok, foot, a, b, i):
    all_points=np.r_[a['boot'][i,foot].reshape(-1,2),b['boot'][i,foot].reshape(-1,2),
                     a['runner'][i,foot].reshape(-1,2),b['runner'][i,foot].reshape(-1,2)]
    if ok[15+foot] and np.isfinite(obs[15+foot]).all():
        all_points=np.r_[all_points,obs[15+foot:16+foot]]
    all_points=all_points[np.isfinite(all_points).all(axis=1)]
    if not len(all_points):
        return (640, 360, 1280, 720)
    low,high=all_points.min(axis=0),all_points.max(axis=0)
    target=np.maximum(high-low + [76,70],[164,116])
    target=np.minimum(np.rint(target).astype(int),SIZE)
    center=.5*(low+high)
    start=np.clip(np.rint(center-.5*target).astype(int),[0,0],np.asarray(SIZE)-target)
    return (*start,*(start+target))


def small(x):
    return None if x is None else round(float(x),3)


def issue(rows):
    present=[x for x in rows if x['semantic'] in STEEL]
    score=[x for x in present if x['p95_px'] is not None]
    if not present:
        return 'no_native_steel_trace',None,0,0,0
    if len(score)<len(present):
        return 'model_path_unavailable',None,len(present),sum(x['role']=='check' for x in present),0
    ratios=[x['p95_px']/x['radius_px'] for x in score]
    faults=sum(q>1. for q in ratios)
    return ('measured_arc_outside_band' if faults else 'measured_arc_inside_band',
            max(ratios),len(score),sum(x['role']=='check' for x in score),faults)


def encode(args, audio, path, error):
    flags=[str(args.ffmpeg),'-hide_banner','-loglevel','warning','-nostdin','-n',
        '-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','50','-i','pipe:0',
        '-i',str(audio),'-map','0:v:0','-map','1:a:0','-map_metadata','-1',
        '-vf','scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-c:v','h264_nvenc','-preset','p4','-cq','18','-b:v','0','-c:a','copy',
        '-fps_mode','passthrough','-video_track_timescale','12800','-movflags','+faststart',str(path)]
    return subprocess.Popen(flags,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                            stderr=error,creationflags=subprocess.CREATE_NO_WINDOW)


def box(image, rect, target):
    a,b,c,d=rect
    crop=image[b:d,a:c]
    height,width=crop.shape[:2]
    factor=min(target.shape[0]/height,target.shape[1]/width)
    h,w=max(1,round(height*factor)),max(1,round(width*factor))
    y,x=(target.shape[0]-h)//2,(target.shape[1]-w)//2
    target[y:y+h,x:x+w]=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)


def text(image, label, xy, scale=.56, color=(233,238,245)):
    cv2.putText(image,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def panel(frames, rows, crops, i, pts, shot):
    page=np.full((1080,1920,3),(24,22,19),np.uint8)
    names=('SOURCE','PREVIOUS','CURRENT')
    for j,(im,name) in enumerate(zip(frames,names,strict=True)):
        x=j*640
        page[34:394,x:x+640]=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
        text(page,name,(x+10,24),.64)
        for k,(row,crop) in enumerate(zip(rows,crops,strict=True)):
            y=418+305*k
            box(im,crop,page[y+20:y+290,x+4:x+636])
            msg=('L' if k==0 else 'R')+' shoe window'
            if j:
                status=row['previous_trace_result' if j==1 else 'current_trace_result']
                msg+=' | '+{'no_native_steel_trace':'steel unverified','model_path_unavailable':'model edge missing',
                            'measured_arc_outside_band':'measured trace outside band',
                            'measured_arc_inside_band':'measured arc inside band'}[status]
            text(page,msg,(x+10,y+14))
    text(page,f'{shot.upper()}  source {pts*float(TB):07.3f} s  native frame {i+1:04d}  PTS {pts}',
         (10,1059),.63)
    return page


FIELDS=('source_frame','source_pts','source_seconds','playback_seconds','shot','foot',
        'previous_trace_result','current_trace_result','fit_or_check_steel_rows','check_steel_rows',
        'derived_steel_rows','previous_trace_max_multiple','current_trace_max_multiple',
        'source_cuff_status','source_cuff_x','source_cuff_y','previous_cuff_to_boot_envelope_px',
        'current_cuff_to_boot_envelope_px','previous_edge_profile_best','previous_edge_profile_center',
        'previous_edge_profile_offset_px','current_edge_profile_best','current_edge_profile_center',
        'current_edge_profile_offset_px','runner_projection_change_p95_px','source_visibility_note',
        'crop_x0','crop_y0','crop_x1','crop_y1')


def run(args):
    args.output.mkdir()
    cv2.setNumThreads(1)
    scope=read(args.scope)
    pts,shots,tb=required_clock(scope)
    np.testing.assert_array_equal(tb,[1,12800])
    source=scope['source']|scope['source_clock']
    raw_path=Path(source['registered_path'])
    if digest(raw_path)!=source['sha256']:
        raise ValueError('Source movie changed')
    old,vr0=bound_video(args.old_video_receipt,pts,args.old_variant)
    new,vr1=bound_video(args.new_video_receipt,pts,args.new_variant)
    if any(q['source_sha256']!=source['sha256'] for q in (vr0,vr1)) or vr0['camera_sha256']!=vr1['camera_sha256']:
        raise ValueError('Source or camera differs between the comparison versions')
    for registration,video in ((args.old_projection,vr0),(args.new_projection,vr1)):
        info=read(registration.with_name('projection_registration.json'))
        if info['output_projection']['sha256'] != digest(registration):
            raise ValueError('Native projection asset changed')
        # The registration and the rendered video must refer to the same state.
        if (digest(Path(info['parameter']))!=video['parameter_sha256']
                or digest(Path(info['result']))!=video['result_sha256']
                or info['camera_state_sha256']!=video['camera_sha256']
                or read(info['source_evidence'])['source_sha256']!=source['sha256']
                or digest(Path(info['source_evidence']))!=info['source_file_sha256'][info['source_evidence']]):
            raise ValueError('Rendered and projected input identity differs')
    oldproj=arrays(args.old_projection,args.old_variant,pts,shots)
    newproj=arrays(args.new_projection,args.new_variant,pts,shots)
    evidence=read(args.source_evidence)['programme_skate_evidence']
    c0,c1=cues_by_frame(args.scores,evidence)
    observed=np.load(args.observations,allow_pickle=False)
    np.testing.assert_array_equal(observed['pts'],pts)
    raw_uv=observed['uv']; raw_use=observed['use']; mapping=observed['model_joint_ids']
    observed_uv=np.full(raw_uv.shape,np.nan,dtype=float)
    observed_use=np.zeros(raw_use.shape,dtype=bool)
    for joint in range(17):
        mask=(mapping==joint)&raw_use
        valid=(mask.sum(axis=1)==1)
        observed_uv[valid,joint]=raw_uv[np.flatnonzero(valid),mask[valid].argmax(axis=1)]
        observed_use[valid,joint]=True
    visibility=defaultdict(list)
    for row in evidence['exposure_review']:
        if 'pts' in row and row.get('reason_zh'):
            visibility[row['pts'],row.get('foot')].append(row['reason_zh'])
    counts=defaultdict(Counter)
    exact=Counter()
    changes=np.linalg.norm(oldproj['runner']-newproj['runner'],axis=-1)
    changed=np.percentile(changes.reshape(len(pts),2,-1),95,axis=-1)
    size=0
    start=time.perf_counter()
    pending=args.output/'frame_by_frame.pending.mp4'
    final=args.output/'frame_by_frame.mp4'
    with (args.output/'per_frame_skates.csv').open('x',encoding='utf-8-sig',newline='',buffering=1) as stream, (
         args.output/'encoder.log').open('xb') as errors:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        p=encode(args,new,pending,errors)
        try:
            for i,((stamp,raw),old_image,new_image) in enumerate(zip(
                    source_frames(raw_path,source,pts),encoded_frames(old,pts),encoded_frames(new,pts),strict=True)):
                if stamp!=int(pts[i]):
                    raise ValueError('Source clock changed during image analysis')
                gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
                lab=cv2.cvtColor(raw,cv2.COLOR_BGR2LAB)
                rows=[]
                crops=[]
                shot=str(shots[i])
                group=counts[shot]
                group['native_frames']+=1
                if np.max(changed[i])<1.:
                    group['both_steel_projections_change_lt_1px']+=1
                if np.max(changed[i])>=3.:
                    group['at_least_one_steel_projection_change_ge_3px']+=1
                frame_status=[]
                for f in (0,1):
                    material0=c0[stamp,f];material1=c1[stamp,f]
                    result0,error0,_,_,_=issue(material0)
                    result1,error1,m,ck,bad=issue(material1)
                    derived=sum(r['derived'] and r['semantic'] in STEEL for r in material1)
                    cuff,reason=cuff_cue(lab,observed_uv[i],observed_use[i],f)
                    a=runner_profile(gray,oldproj['runner'][i,f])
                    b=runner_profile(gray,newproj['runner'][i,f])
                    e0=footprint_gap(cuff,oldproj['boot'][i,f])
                    e1=footprint_gap(cuff,newproj['boot'][i,f])
                    crop=bounds(observed_uv[i],observed_use[i],f,oldproj,newproj,i)
                    desc=' / '.join(visibility[stamp,f]+visibility[stamp,None])
                    row=dict(source_frame=i+1,source_pts=stamp,source_seconds=f'{stamp*float(TB):.3f}',
                        playback_seconds=f'{(stamp-pts[0])*float(TB):.3f}',shot=shot,foot='left' if f==0 else 'right',
                        previous_trace_result=result0,current_trace_result=result1,fit_or_check_steel_rows=m,
                        check_steel_rows=ck,derived_steel_rows=derived,previous_trace_max_multiple=small(error0),
                        current_trace_max_multiple=small(error1),source_cuff_status=reason,
                        source_cuff_x=small(cuff[0]) if cuff is not None else None,
                        source_cuff_y=small(cuff[1]) if cuff is not None else None,
                        previous_cuff_to_boot_envelope_px=small(e0),current_cuff_to_boot_envelope_px=small(e1),
                        previous_edge_profile_best=small(a[0]),previous_edge_profile_center=small(a[1]),
                        previous_edge_profile_offset_px=small(a[2]),current_edge_profile_best=small(b[0]),
                        current_edge_profile_center=small(b[1]),current_edge_profile_offset_px=small(b[2]),
                        runner_projection_change_p95_px=small(changed[i,f]),source_visibility_note=desc,
                        crop_x0=crop[0],crop_y0=crop[1],crop_x1=crop[2],crop_y1=crop[3])
                    writer.writerow(row)
                    rows.append(row);crops.append(crop)
                    group['foot_exposures']+=1
                    group[result1]+=1
                    group['source_cuff_candidate']+=int(cuff is not None)
                    group['raw_edge_profile_computed']+=int(b[0] is not None)
                    group['source_has_observed_shell']+=int(any(r['semantic'] in ('boot_shell','heel_shell','toe_shell') for r in material1))
                    group['has_fit_or_check_steel']+=int(m>0)
                    group['has_check_steel']+=int(ck>0)
                    group['has_nonderived_steel']+=int(m>derived)
                    group['has_complete_observed_steel']+=int(any(x['semantic'] in STEEL and x['complete_visible'] for x in material1))
                    frame_status.append((m,ck,result1))
                    size+=1
                group['frames_with_steel_evidence']+=int(any(r[0]>0 for r in frame_status))
                group['frames_with_two_feet_measured_steel']+=int(all(r[0]>0 for r in frame_status))
                group['frames_with_check_steel']+=int(any(r[1]>0 for r in frame_status))
                group['frames_with_no_steel_evidence']+=int(all(r[0]==0 for r in frame_status))
                group['frames_with_measured_steel_failure']+=int(any(r[2] in ('measured_arc_outside_band','model_path_unavailable') for r in frame_status))
                video=panel((raw,old_image,new_image),rows,crops,i,stamp,shot)
                p.stdin.write(memoryview(np.ascontiguousarray(video)).cast('B'))
                if (i+1)%250==0 or i+1==len(pts):
                    note('native_frames_inspected',frames=i+1,total=len(pts),shot=shot,
                         source_s=float(stamp*TB),elapsed_s=round(time.perf_counter()-start,1))
            p.stdin.close()
            if p.wait()!=0:
                raise RuntimeError('Comparison video encoder failed')
        finally:
            if p.poll() is None:
                p.terminate()
                p.wait()
    if size!=2*len(pts):
        raise ValueError('Did not inspect exactly both feet on every native exposure')
    for group in counts.values():
        exact.update(group)
    note('native_checks_complete',frames=len(pts),two_foot_rows=size,counts=dict(exact))
    receipt=dict(native_frames=len(pts),two_foot_rows=size,time_base=[1,12800],
        source_first_last_pts=pts[[0,-1]].tolist(),source_sha256=source['sha256'],
        old_parameter_sha256=vr0['parameter_sha256'],new_parameter_sha256=vr1['parameter_sha256'],
        camera_sha256=vr1['camera_sha256'],by_shot={k:dict(v) for k,v in counts.items()},whole=dict(exact),
        trace_sources=str(args.scores),same_frame_original_and_overlays=True,
        cuff_and_edge_profile_are_navigation_only=True,actual_material_match_requires_source_trace=True,
        human_review_of_every_frame=False)
    with (args.output/'audit_summary.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    clock=validate_video(pending,pts,TB)
    pending.rename(final)
    with (args.output/'comparison_verification.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(video=str(final),sha256=digest(final),source_first_last_seconds=[float(pts[0]*TB),float(pts[-1]*TB)],
                  verified_clock=clock,source_video_sha256=source['sha256']),stream,indent=2)
    note('completed',video=str(final),rows=size)


def main():
    p=ArgumentParser()
    for name in ('scope','old-video-receipt','new-video-receipt','old-projection','new-projection',
                 'scores','source-evidence','observations','output','ffmpeg'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('old-variant','new-variant'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__':
    main()
