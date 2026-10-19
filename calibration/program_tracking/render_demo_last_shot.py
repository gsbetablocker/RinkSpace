"""Native-50fps presentation sample using a frozen reviewed export.

No camera/body optimization. Spin contact replacement and cosmetic animation
remain separate from the immutable source model. Diagnostics live in Data.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
import av, cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from calibration.program_tracking.demo_layers import IceLayers, Speedometer
from calibration.program_tracking.demo_outro import OutroTransform,PlaneTexture,prepare_texture,plane_homography,_warp_prepared
from calibration.program_tracking.render_result import PersonMasks,_ffmpeg_executable,_available_encoder
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp')
BASE_OUT=DATA/'results/demo_last_shot_20261016'
OUT=BASE_OUT
FIRST,LAST=1838080,2415104
FPS=50
WH=(1920,1080)

def read_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write_json(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf8')
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def decode(source,first,last):
    with av.open(str(source)) as c:
        s=c.streams.video[0]
        assert str(s.time_base)=='1/12800' and s.average_rate==50
        c.seek(int(first),stream=s,backward=True)
        expected=int(first)
        for f in c.decode(s):
            if f.pts is None or f.pts<first:continue
            if f.pts>last:break
            if f.pts!=expected:raise ValueError(f'native PTS mismatch {f.pts} != {expected}')
            yield int(f.pts),f.to_ndarray(format='bgr24')
            expected+=256
        if expected!=last+256:raise ValueError('source endpoint missing')

class Sample:
    def __init__(self,version='v1',contact_path=None):
        self.version=version
        self.packet=read_npz(BASE_OUT/'source_packet.npz')
        p=self.packet
        self.pts=p['pts'];self.times=self.pts/12800.
        contact=Path(contact_path) if contact_path is not None else BASE_OUT/'contact_packet.npz'
        if not contact.exists():raise FileNotFoundError('Spin contact packet must be reviewed before rendering')
        self.contact=read_npz(contact)
        self.xy=self.contact['xy_m']
        assert self.xy.shape==p['xy_m'].shape
        self.scene=json.loads(Path(str(p['scene_path'])).read_text(encoding='utf-8-sig'))
        style=({'sigma_m':1.5,'heat_palette':'deep_red','particle_style':'particles',
                'heat_update_anchor':0} if version=='v2' else {})
        self.layers=IceLayers(self.scene,float(p['unit']),self.xy,self.times,
                             fade_seconds=2.5,old_alpha=.22,**style)
        self._ending_prepared=None
        self.lookup={int(t):i for i,t in enumerate(self.pts)}
        people=DATA/'history/tracking_runs/people'
        self.masks=PersonMasks(people/'masks',people/'boxes.json')
        self.dial=Speedometer(size=220,maximum=40)
        # Pointer animation only: source-native dt, with a documented response filter.
        display_xy=gaussian_filter1d(self.xy,sigma=.16*FPS,axis=0,mode='nearest')
        raw=np.linalg.norm(np.gradient(display_xy,self.times,axis=0),axis=1)*3.6
        self.speed=gaussian_filter1d(raw,sigma=.26*FPS,mode='nearest')
        self.mask_counts={}
        self.outro=OutroTransform(p['camera25'][-1],self.layers.bounds_xy,WH,
                                  turn_seconds=3.0,fade_seconds=.55,hold_seconds=1.2)
    def foreground(self,index):
        used_pts=int(self.packet['legacy_mask_pts'][index])
        mask,status=self.masks.get(used_pts,True)
        original_status=status
        if status!='measured_mask':
            for distance in range(1,5):
                found=False
                for neighbor in (index-distance,index+distance):
                    if not 0<=neighbor<len(self.pts):continue
                    if self.packet['shot_index'][neighbor]!=self.packet['shot_index'][index]:continue
                    candidate=int(self.packet['legacy_mask_pts'][neighbor])
                    if self.masks.lookup.get(candidate,(None,None,None))[2]=='measured_mask':
                        used_pts=candidate
                        mask,status=self.masks.get(used_pts,True)
                        found=True
                        break
                if found:break
        if status=='measured_mask':
            path,row,_=self.masks.lookup[used_pts]
            raw=np.unpackbits(self.masks.active['masks'][row],axis=1)[:,:960].astype(np.float32)
            mask=cv2.resize(raw,WH,interpolation=cv2.INTER_LINEAR)
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
            mask=cv2.GaussianBlur(mask,(0,0),.9)
        else:
            mask=mask.astype(np.float32)
        # Original broadcast scoreboard and lower-right timer retain source pixels.
        mask[0:255,0:555]=1.
        mask[934:1010,1670:1880]=1.
        label=status if status==original_status else 'neighbor_mask_inferred'
        self.last_mask_label=label
        self.mask_counts[label]=self.mask_counts.get(label,0)+1
        return mask
    def draw(self,native,pts):
        i=self.lookup[int(pts)]
        if self.version=='v2':
            prepared=self.layers.texture_prepared(i)
        else:
            prepared=prepare_texture(self.layers.texture(i))
        cam=self.packet['camera25'][i]
        H=plane_homography(cam,self.layers.bounds_xy,self.layers.texture_wh)
        warped=_warp_prepared(prepared,H,WH,cam)
        fg=self.foreground(i)
        warped*=1.-fg[...,None]
        base=native.copy()
        self.dial.draw(base,float(self.speed[i]))
        result=np.clip(warped[...,:3]*255.+base.astype(np.float32)*(1.-warped[...,3:4])+.5,0,255).astype(np.uint8)
        return result,base,prepared,fg

    def ending_texture(self,last_prepared,elapsed):
        if self.version=='v1':return last_prepared
        if self._ending_prepared is None:
            self._ending_prepared=self.layers.texture_prepared(len(self.pts)-1,summary=True)
        u=float(np.clip(elapsed/.55,0,1));u=u*u*(3-2*u)
        if u>=1:return self._ending_prepared
        if u<=0:return last_prepared
        return PlaneTexture(last_prepared.premultiplied*(1-u)+self._ending_prepared.premultiplied*u)


def previews(version='v1'):
    OUT.mkdir(exist_ok=True,parents=True)
    sample=Sample(version);times=[144.2,152.6,161.6,164.6,175.6,178.6,181.6,184.,188.68]
    tiles=[];last=None
    for t in times:
        pts=round(t*50)*256
        native=next(decode(str(sample.packet['source_path']),pts,pts))[1]
        frame,base,tex,fg=sample.draw(native,pts)
        cv2.imwrite(str(OUT/f'preview_{pts}.jpg'),frame)
        tile=cv2.resize(frame,(640,360));cv2.putText(tile,f'source {t:.2f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA);tiles.append(tile)
        last=(frame,base,tex,fg)
    cv2.imwrite(str(OUT/'layout_preview.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,9,3)]))
    frame,base,tex,fg=last
    frames=[]
    for t in [0.,.25,.55,1.,1.7,2.4,3.,4.]:
        a=sample.outro.frame(base,sample.ending_texture(tex,t),t,foreground_mask=fg.astype(np.float32))
        cv2.imwrite(str(OUT/f'outro_{t:.2f}.jpg'),a)
        tile=cv2.resize(a,(480,270));cv2.putText(tile,f'outro +{t:.2f}s',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA);frames.append(tile)
    cv2.imwrite(str(OUT/'outro_preview.jpg'),np.vstack([np.hstack(frames[j:j+4]) for j in range(0,8,4)]))
    start=sample.outro.frame(base,tex,0,foreground_mask=fg.astype(np.float32))
    diff=int(np.abs(start.astype(int)-frame.astype(int)).max())
    report={'outro_start_max_pixel_difference':diff,'outro_geometry':sample.outro.selfcheck(sample.layers.texture_wh),
            'speedometer':{'size':220,'location':'top_right','margin':28},'fade_seconds':2.5,
            'source_first_pts':FIRST,'source_last_pts':LAST,'source_time_base':'1/12800'}
    assert diff==0,report
    write_json(OUT/'preview_report.json',report)
    print(json.dumps(report),flush=True)

def render(version='v1',workers=1):
    OUT.mkdir(exist_ok=True,parents=True)
    started=time.perf_counter();sample=Sample(version);native_frames=(LAST-FIRST)//256+1
    outro_frames=round(sample.outro.total_seconds*FPS)
    ffmpeg=_ffmpeg_executable();args,name=_available_encoder(ffmpeg)
    target=OUT/('RinkSpace_last_shot_demo_v2.mp4' if version=='v2' else 'RinkSpace_last_shot_demo.mp4')
    total=(native_frames+outro_frames)/FPS
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24',
         '-video_size','1920x1080','-framerate','50','-i','pipe:0',
         '-ss',str(FIRST/12800.),'-i',str(sample.packet['source_path']),
         '-map','0:v:0','-map','1:a:0?',*args,'-pix_fmt','yuv420p',
         '-af',f'atrim=duration={native_frames/FPS},afade=t=out:st={native_frames/FPS-.45}:d=0.45,apad',
         '-t',str(total),'-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)]
    write_json(OUT/'render_command.json',cmd)
    count=0
    stream=None
    with (OUT/'encode.stderr.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
        try:
            draw_stats={}
            if workers>1:
                from calibration.program_tracking.demo_pipeline import ordered_frames
                stream=ordered_frames(decode(str(sample.packet['source_path']),FIRST,LAST),
                    sample_kwargs={'version':version},workers=workers,slots=workers*3,stats=draw_stats)
            else:
                def serial_frames():
                    for pts,native in decode(str(sample.packet['source_path']),FIRST,LAST):
                        yield pts,sample.draw(native,pts)[0]
                stream=serial_frames()
            for pts,frame in stream:
                proc.stdin.write(frame.tobytes());count+=1
                if pts==LAST:final_encoded_frame=frame.copy()
                if count%100==0:
                    write_json(OUT/'render_status.json',{'stage':'source','frames':count,'total':native_frames+outro_frames,'elapsed_s':time.perf_counter()-started})
                    print(f'source {count}/{native_frames}',flush=True)
            if workers==1:draw_stats['mask_states']=dict(sample.mask_counts)
            native_last=next(decode(str(sample.packet['source_path']),LAST,LAST))[1]
            final_native,base,tex,fg=sample.draw(native_last,LAST)
            assert np.max(np.abs(final_native.astype(int)-final_encoded_frame.astype(int)))==0
            start=sample.outro.frame(base,tex,0,foreground_mask=fg.astype(np.float32))
            diff=int(np.abs(start.astype(int)-final_native.astype(int)).max())
            assert diff==0
            for j in range(outro_frames):
                t=(j+1)/FPS
                frame=sample.outro.frame(base,sample.ending_texture(tex,t),t,foreground_mask=fg.astype(np.float32))
                proc.stdin.write(frame.tobytes());count+=1
                if j%50==0:print(f'outro {j+1}/{outro_frames}',flush=True)
        finally:
            if stream is not None and hasattr(stream,'close'):stream.close()
            proc.stdin.close()
            ret=proc.wait()
        if ret:raise RuntimeError(f'ffmpeg {ret}: '+(OUT/'encode.stderr.log').read_text()[-3000:])
    report={'status':'complete','video':str(target),'fps':50,'width':1920,'height':1080,
            'source':str(sample.packet['source_path']),'source_first_pts':FIRST,'source_last_pts':LAST,
            'source_time_base':'1/12800','source_first_seconds':FIRST/12800.,'source_last_seconds':LAST/12800.,
            'native_source_frames':native_frames,'outro_frames':outro_frames,'encoded_frames':count,
            'expected_duration_s':total,'outro_start_max_pixel_difference':diff,
            'no_statistics_or_promotional_text':True,'fade_seconds':2.5,'history_alpha':.22,
            'version':version,'drawing_workers':workers,'heat_sigma_m':(1.5 if version=='v2' else .7),
            'heat_palette':('deep_red' if version=='v2' else 'amber'),
            'particle_style':('particles' if version=='v2' else 'markers'),
            'outro_uniform_trajectory':version=='v2',
            'speedometer':{'size':220,'position':'top_right','margin':28,'no_current_numeric_speed':True,
                           'native_clock':True,'position_response_sigma_s':.16,'pointer_response_sigma_s':.26},
            'full_programme_history_seeded':True,'body_camera_solver_modified':False,
            'inputs':{n:{'path':str(BASE_OUT/n),'sha256':digest(BASE_OUT/n)} for n in ['source_packet.npz','contact_packet.npz']},
            'mask_states':draw_stats.get('mask_states',sample.mask_counts),'drawing_pipeline':draw_stats,'encoder':name,'wall_seconds':time.perf_counter()-started}
    write_json(OUT/'sample_report.json',report);write_json(OUT/'render_status.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def verify():
    report=json.loads((OUT/'sample_report.json').read_text())
    target=Path(report['video']);indices=[0,450,900,1500,2000,2254,2255,2280,2325,2370,2405,2464]
    pictures={};stamps=[]
    with av.open(str(target)) as container:
        stream=container.streams.video[0]
        size=(stream.width,stream.height);rate=str(stream.average_rate)
        assert size==WH and stream.average_rate==FPS
        tb=stream.time_base
        for index,frame in enumerate(container.decode(stream)):
            stamps.append(float(frame.pts*tb))
            if index in indices:pictures[index]=frame.to_ndarray(format='bgr24')
    assert len(stamps)==report['encoded_frames']
    assert np.allclose(np.diff(stamps),1/FPS,atol=1e-8)
    expected_first=next(decode(report['source'],FIRST,FIRST))[1]
    expected_last=next(decode(report['source'],LAST,LAST))[1]
    timer=(slice(944,1000),slice(1690,1865))
    roi_diffs=[]
    for index,native in [(0,expected_first),(2254,expected_last)]:
        diff=np.abs(pictures[index][timer].astype(float)-native[timer].astype(float))
        roi_diffs.append({'frame':index,'mean_abs_error':float(diff.mean()),'p95_abs_error':float(np.percentile(diff,95))})
    timer_base_diffs=[];timer_base_source_diffs=[]
    # A copied, re-encoded presentation base adds a second codec generation.
    # Compare that pass against its frozen input while retaining source errors.
    if report.get('base_video'):
        with av.open(str(report['base_video'])) as base_container:
            for index,frame in enumerate(base_container.decode(base_container.streams.video[0])):
                if index in (0,2254):
                    native_base=frame.to_ndarray(format='bgr24')
                    expected_native=expected_first if index==0 else expected_last
                    source_delta=np.abs(native_base[timer].astype(float)-expected_native[timer].astype(float))
                    timer_base_source_diffs.append({'frame':index,'mean_abs_error':float(source_delta.mean()),'p95_abs_error':float(np.percentile(source_delta,95))})
                    delta=np.abs(pictures[index][timer].astype(float)-native_base[timer].astype(float))
                    timer_base_diffs.append({'frame':index,'mean_abs_error':float(delta.mean()),'p95_abs_error':float(np.percentile(delta,95))})
                if index==2254:break
        assert len(timer_base_diffs)==2
        assert all(d['mean_abs_error']<5 for d in timer_base_source_diffs),timer_base_source_diffs
        assert all(d['mean_abs_error']<5 for d in timer_base_diffs),timer_base_diffs
    else:
        assert all(d['mean_abs_error']<5 for d in roi_diffs),roi_diffs
    audio_samples=0;audio_peak=0.;squares=0.
    with av.open(str(target)) as container:
        assert len(container.streams.audio)==1
        ast=container.streams.audio[0]
        audio_rate=ast.rate
        for frame in container.decode(ast):
            values=frame.to_ndarray().astype(np.float64)
            audio_samples+=values.size;squares+=float(np.sum(values*values))
            audio_peak=max(audio_peak,float(np.max(np.abs(values))))
    assert audio_samples>0 and audio_peak>1e-5
    tiles=[]
    for index in indices:
        frame=pictures[index];tile=cv2.resize(frame,(480,270))
        cv2.putText(tile,f'{index/FPS:.2f}s',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);tiles.append(tile)
    cv2.imwrite(str(OUT/'encoded_review.jpg'),np.vstack([np.hstack(tiles[j:j+4]) for j in range(0,len(tiles),4)]))
    verification={'decoded_video_frames':len(stamps),'width':size[0],'height':size[1],'fps':rate,
        'first_encoded_time_s':stamps[0],'last_encoded_time_s':stamps[-1],
        'all_encoded_frame_intervals_seconds':1/FPS,'audio_streams':1,'audio_sample_rate':audio_rate,
        'audio_rms':float(np.sqrt(squares/audio_samples)),'audio_peak':audio_peak,
        'broadcast_timer_source_comparison':roi_diffs,'broadcast_timer_postprocess_base_comparison':timer_base_diffs,
        'broadcast_timer_comparison_reference':report.get('base_video',report['source']),
        'broadcast_timer_base_source_comparison':timer_base_source_diffs,
        'outro_start_max_pixel_difference':report['outro_start_max_pixel_difference'],
        'video_sha256':digest(target),'video_bytes':target.stat().st_size}
    write_json(OUT/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--version',choices=['v1','v2'],default='v1')
    parser.add_argument('--workers',type=int,default=1)
    a=parser.parse_args()
    OUT=BASE_OUT if a.version=='v1' else BASE_OUT/'v2'
    cv2.setNumThreads(2)
    if a.preview:previews(a.version)
    elif a.verify:verify()
    else:render(a.version,a.workers)
