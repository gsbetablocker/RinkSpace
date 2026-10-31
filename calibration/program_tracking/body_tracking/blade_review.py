"""Full native-time blade review video, source checks and explicit failures."""
from pathlib import Path
from fractions import Fraction
import argparse,json,time
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .footwear_fit import point_polyline_distance
from .rebuild_prepare import digest
from .blade_cotracker_review import overlap_error

LABELS={'tracked_candidate':'前后刀线相符（待整段核对）','source_annotation':'原图起点标线','image_candidate_unconfirmed':'候选未确认','not_located':'未定位'}
COLORS=[(70,230,80),(250,190,60)]


def render(root,output,programme,source_index_path=None):
    root,output,programme=Path(root),Path(output),Path(programme);track=json.loads((output/'tracks.json').read_text(encoding='utf8'));index_path=Path(source_index_path) if source_index_path else output/'source_index.json';idx=json.loads(index_path.read_text(encoding='utf8'));rows=track['rows'];
    if [r['pts'] for r in rows]!=[r['pts'] for r in idx['rows']]:raise ValueError('Track and source-index exposure order differs')
    pts=[r['pts'] for r in rows];by={t:i for i,t in enumerate(pts)};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);records=[]
    for role,name in [('source_anchors','footwear_s01_joint_semantics_v2_20261029.json'),('development','footwear_s01_geometry_check_20261029.json'),('earlier_development','footwear_s01_check_20261029.json')]:
        source=json.loads((programme/name).read_text(encoding='utf8'))
        for q in source['blade_curves']:
            r=rows[by[q['pts']]];t=r['feet'][q['foot']];curve=t['curve_uv'];distance=point_polyline_distance(q['uv'],curve) if curve is not None else None;other=r['feet'][1-q['foot']]['curve_uv'];other_error=float(np.median(point_polyline_distance(q['uv'],other))) if other is not None else None;records.append(dict(role=role,pts=q['pts'],source_seconds=q['pts']/12800,foot=q['foot'],status=t['status'],source_vertices=len(q['uv']),median_px=float(np.median(distance)) if distance is not None else None,p95_px=float(np.percentile(distance,95)) if distance is not None else None,other_track_median_px=other_error,points_not_located=distance is None,source_distances_include_uncovered_vertices=True,source_vertices_covered_within_5px=float(np.mean(distance<=5)) if distance is not None else None,visible_fragment_overlap_error_px=overlap_error(q['uv'],curve) if curve is not None else None,tracking_initialization_at_this_source_time=bool(t.get('source_anchor_pts')==q['pts'])))
    (output/'source_curve_checks.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    checks=[];tiles=[];sample_times=np.linspace(24.6,44.96,28);chosen={pts[int(np.argmin(abs(np.asarray(pts)/12800-t)))] for t in sample_times}|{q['pts'] for q in records}|{r['pts'] for r in rows if any(f['status']=='source_annotation' for f in r['feet'])};start=time.perf_counter();video=output/'S01_blades_continuous.mp4'
    if video.exists():raise FileExistsError(video)
    def draw_curves(image,r,offset=np.zeros(2),scale=1):
        for foot,item in enumerate(r['feet']):
            if item['curve_uv'] is None:continue
            curve=np.rint((np.asarray(item['curve_uv'])-offset)*scale).astype(np.int32);color=COLORS[foot]
            if item['status']=='image_candidate_unconfirmed':
                for k in range(0,len(curve)-1,2):cv2.line(image,tuple(curve[k]),tuple(curve[k+1]),color,1,cv2.LINE_AA)
            else:cv2.polylines(image,[curve],False,color,1,cv2.LINE_AA)
            cv2.putText(image,'AB'[foot],tuple(curve[len(curve)//2]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    with (output/'render_progress.jsonl').open('w',encoding='utf8',buffering=1) as log,av.open(str(root/'offline_review/source_s01.mp4')) as source,av.open(str(video),'w') as dest:
        source.streams.video[0].codec_context.thread_count=2;stream=dest.add_stream('libx264',rate=50);stream.width=1920;stream.height=816;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'18','preset':'fast'};stream.time_base=Fraction(1,12800);frame_count=0
        for frame in source.decode(video=0):
            if frame.pts not in by:continue
            i=by[frame.pts];r=rows[i];raw=frame.to_ndarray(format='bgr24');marked=raw.copy();draw_curves(marked,r);canvas=np.full((816,1920,3),22,np.uint8);canvas[80:800,:1280]=cv2.resize(marked,(1280,720),interpolation=cv2.INTER_AREA);crop_row=idx['rows'][i];x0,y0,x1,y1=crop_row['crop_xyxy'];crop=raw[y0:y1,x0:x1].copy();scale=min(620/crop.shape[1],580/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);draw_curves(crop,r,np.array([x0,y0]),scale);canvas[85:85+crop.shape[0],1290:1290+crop.shape[1]]=crop
            rgb=Image.fromarray(canvas[:,:,::-1]);d=ImageDraw.Draw(rgb);d.text((16,12),f'原片 {r["source_seconds"]:.3f} 秒   |   S01 整段冰刀跟踪检查',font=font,fill=(245,245,245));d.text((16,44),'实线：前后刀线相符或原图起点   虚线：候选未确认   无线：未定位；人体和相机不动',font=small,fill=(230,230,230));d.text((1300,690),'A：'+LABELS[r['feet'][0]['status']],font=font,fill=(80,230,70));d.text((1300,725),'B：'+LABELS[r['feet'][1]['status']],font=font,fill=(60,190,250));d.text((1300,762),'A/B 是跟踪编号，物理左右脚仍待核对',font=small,fill=(240,210,100));canvas=np.asarray(rgb)[:,:,::-1].copy();encoded=av.VideoFrame.from_ndarray(canvas,format='bgr24');encoded.pts=i*256;encoded.time_base=Fraction(1,12800)
            for packet in stream.encode(encoded):dest.mux(packet)
            if r['pts'] in chosen:
                tile=np.full((330,440,3),22,np.uint8);im=raw[y0:y1,x0:x1].copy();s=min(420/im.shape[1],275/im.shape[0]);im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST);draw_curves(im,r,np.array([x0,y0]),s);tile[45:45+im.shape[0],10:10+im.shape[1]]=im;cv2.putText(tile,f'{r["source_seconds"]:.3f}s  A:{r["feet"][0]["status"][:7]} B:{r["feet"][1]["status"][:7]}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1,cv2.LINE_AA);tiles.append((r['pts'],tile))
            frame_count+=1
            if frame_count%100==0:log.write(json.dumps(dict(event='render',frames=frame_count,seconds=time.perf_counter()-start))+'\n')
        for packet in stream.encode():dest.mux(packet)
        if frame_count!=1143:raise ValueError('Incomplete source video review')
        log.write(json.dumps(dict(event='complete',frames=frame_count,seconds=time.perf_counter()-start))+'\n')
    for page,begin in enumerate(range(0,len(tiles),12)):
        group=[im for _,im in tiles[begin:begin+12]]
        while len(group)%4:group.append(np.full_like(group[0],22))
        cv2.imwrite(str(output/f'continuous_review_{page}.png'),np.vstack([np.hstack(group[j:j+4]) for j in range(0,len(group),4)]))
    with av.open(str(video)) as c:
        decoded=[f.pts for f in c.decode(video=0)];tb=c.streams.video[0].time_base
    expected=np.arange(1143)/50;np.testing.assert_allclose(np.array(decoded)*float(tb),expected,atol=1e-9)
    gaps=[]
    for foot in [0,1]:
        states=[r['feet'][foot]['status'] for r in rows];a=0
        for b in range(1,len(states)+1):
            if b==len(states) or states[b]!=states[a]:
                if states[a]!='tracked_candidate':gaps.append(dict(track='AB'[foot],status=states[a],first_source_seconds=rows[a]['source_seconds'],last_source_seconds=rows[b-1]['source_seconds'],exposures=b-a))
                a=b
    result=dict(schema='rinkspace.s01-full-blade-review.v1',source_exposures=1143,video=str(video.resolve()),source_time_range_seconds=[22.12,44.96],playback_zero_corresponds_to_source_seconds=22.12,source_time_base=[1,12800],output_native_exposures_verified=True,camera_updated=False,body_updated=False,contact_feedback=False,counts=track['counts'],source_curve_checks=records,unconfirmed_intervals=gaps,accuracy_accepted=False,whole_video_human_acceptance=False,video_sha256=digest(video));(output/'review_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(video=str(video),frames=1143,curve_checks=records)),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--programme',required=True);p.add_argument('--source-index');a=p.parse_args();render(a.root,a.output,a.programme,a.source_index)
