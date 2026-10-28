"""Read-only S02 auxiliary-camera review using the identical native scene."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,io,json,time
import cv2,numpy as np,av
from PIL import Image,ImageDraw,ImageFont
import fit_camera_s02 as fit
from rinkspace.review.scene import sample_review_scene
from rinkspace.review.render import source_frames,validate_video
from rinkspace.vision_native import file_ref

class Review:
    def __init__(self,state_path):
        self.m,self.z,self.mapping,self.g,self.core,self.native=fit.load()
        data=Path(state_path).read_bytes();self.state=dict(np.load(io.BytesIO(data),allow_pickle=False))
        if not np.array_equal(self.state['scene_values'],self.g['scene_values']):raise ValueError('Camera state changed the shared scene')
        self.state_ref=dict(path=str(state_path),sha256=hashlib.sha256(data).hexdigest())
        self.camera=fit.read(fit.OUT/'camera_seed.json');self.seed=dict(rig=np.array(self.camera['rig']),controls=np.load(fit.OUT/'controls_seed.npy'))
        self.samples=sample_review_scene(fit.read(Path(self.m['scene_path'])),self.core,Path(self.m['scene_path']),parameter_state=self.native.parameter_state,observed_path_ids=set())
        self.covered={r['path_id'] for r in fit.read(fit.OUT/'geometry_coverage.json')['paths'] if r['camera_initialization']}
        self.visibility={r['pts']:r for r in fit.read(fit.source.OUT/'evidence/visibility.json')['frames']}
        self.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)

    def frame(self,pts,image,state,*,all_geometry=False):
        if hashlib.sha256(image.tobytes()).hexdigest()!=self.visibility[pts]['source_BGR_sha256']:raise ValueError('Source exposure mismatch')
        canvas=np.zeros((1160,1920,3),np.uint8);canvas[:1080]=image
        result=self.core.project_motion(state['rig'],state['controls'],self.camera['first_control_time'],self.camera['control_dt'],1080,(pts-self.camera['reference_pts'])/12800,self.samples['xyz'])
        shown=0
        for path in self.samples['cuts']:
            known=path['id'] in self.covered
            if not known and not all_geometry:continue
            uv=result['uv'][path['begin']:path['end']];valid=(result['status'][path['begin']:path['end']]==0)&np.isfinite(uv).all(1)
            ids=np.flatnonzero(valid)
            color=(90,245,160) if known else (40,165,255)
            for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                if len(run)>1:
                    cv2.polylines(canvas[:1080], [np.rint(np.clip(uv[run],-1e6,1e6)).astype(np.int32)],False,color,1,cv2.LINE_AA)
            shown+=1
        pic=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pic)
        draw.text((18,1087),f'S02 辅助取点投影   原片 {pts/12800:.3f} 秒   PTS {pts}',font=self.font,fill=(245,245,245))
        note='绿：几何有隔壁观测支持；橙：缺少直接／独立位置约束，仍含初值。' if all_geometry else '绿：仅显示有隔壁观测支持的几何。其余未优化几何不作为相机初始化依据。'
        draw.text((18,1121),note+' 网格不是实测点。',font=self.font,fill=(220,220,220))
        return cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR),shown

    def compare(self,pts):
        src=cv2.imread(self.visibility[pts]['image']);before,_=self.frame(pts,src,self.seed);after,_=self.frame(pts,src,self.state)
        pic=np.hstack([cv2.resize(x,(960,580),interpolation=cv2.INTER_AREA) for x in (before,after)])
        top=np.zeros((42,1920,3),np.uint8);pic=np.vstack([top,pic]);im=Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
        for x,text in [(20,'旧相机 + 同一份当前几何'),(980,'调整后的辅助相机 + 同一份当前几何')]:draw.text((x,6),text,font=self.font,fill=(250,250,250))
        return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,default=fit.OUT/'navigation/accepted_camera_latest.npz');ap.add_argument('--video',action='store_true');ap.add_argument('--pts',type=int,nargs='+',default=[575744,623872,644608,703744,720640]);args=ap.parse_args()
    cv2.setNumThreads(2);r=Review(args.state);out=fit.OUT/'review';out.mkdir(exist_ok=True)
    # Snapshot exact camera arrays used by all rendered frames, never a moving live state.
    np.savez(out/'review_camera_state.npz',**r.state)
    for pt in args.pts:
        src=cv2.imread(r.visibility[pt]['image']);im,_=r.frame(pt,src,r.state);cv2.imwrite(str(out/f'camera_{pt}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,96]);cv2.imwrite(str(out/f'compare_{pt}.jpg'),r.compare(pt),[cv2.IMWRITE_JPEG_QUALITY,96])
    receipt=dict(camera_state=r.state_ref,review_camera_state=file_ref(out/'review_camera_state.npz'),geometry_source=r.m['geometry_origin'],auxiliary_projection_only=True,source_pts=args.pts,geometry_or_observations_changed=False)
    if args.video:
        pts=np.arange(fit.source.FIRST,fit.source.LAST+1,256);path=out/'S02_auxiliary_projection.mp4';temporary=path.with_suffix('.partial.mp4');start=time.monotonic()
        with av.open(str(temporary),'w') as container:
            stream=container.add_stream('libx264',rate=50);stream.width=1920;stream.height=1160;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,12800);stream.options={'crf':'17','preset':'fast'}
            for i,(pt,im) in enumerate(source_frames(Path(r.m['source_video']),dict(time_base_num=1,time_base_den=12800,width=1920,height=1080),pts)):
                im,_=r.frame(pt,im,r.state,all_geometry=True);v=av.VideoFrame.from_ndarray(im,format='bgr24');v.pts=pt-fit.source.FIRST;v.time_base=Fraction(1,12800)
                for packet in stream.encode(v):container.mux(packet)
                if i%100==0:print(json.dumps(dict(frames=i+1,total=567,elapsed_seconds=time.monotonic()-start)),flush=True)
            for packet in stream.encode():container.mux(packet)
        receipt.update(video_validation=validate_video(temporary,pts,Fraction(1,12800)),source_seconds=[44.98,56.3]);temporary.replace(path);receipt['video']=file_ref(path)
    fit.load();fit.write(out/'review.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
