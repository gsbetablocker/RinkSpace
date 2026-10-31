"""Source-only CoTracker3 online inference with exact native crop transforms."""
from pathlib import Path
from rinkspace.storage import AssetStore
import argparse,json,time,sys,os
import cv2,numpy as np
from .rebuild_prepare import load,digest


def prepare(root,output,review):
    root,output=Path(root),Path(output);out=output/'cotracker_input';out.mkdir(exist_ok=False);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));source=json.loads(Path(review).read_text(encoding='utf8'));wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');rows=idx['rows'];video=np.lib.format.open_memmap(out/'video_uint8.npy',mode='w+',dtype=np.uint8,shape=(len(rows),384,512,3));transforms=[];queries=[];metadata=[];by={r['pts']:i for i,r in enumerate(rows)};cv2.setNumThreads(2)
    for i,row in enumerate(rows):
        im=cv2.imread(row['file']);h,w=im.shape[:2];scale=min(512/w,384/h);width=max(1,round(w*scale));height=max(1,round(h*scale));pad=np.array([(512-width)//2,(384-height)//2]);rgb=cv2.cvtColor(cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB);video[i]=0;video[i,pad[1]:pad[1]+height,pad[0]:pad[0]+width]=rgb;transforms.append([row['crop_xyxy'][0],row['crop_xyxy'][1],width/w,height/h,pad[0],pad[1]])
    video.flush();transforms=np.asarray(transforms,float)
    from .blade_temporal import resample_curve
    for si,seed in enumerate(source['blade_curves']):
        i=by[seed['pts']];row=rows[i];curve=resample_curve(seed['uv'],5);im=cv2.imread(row['file']);offset=np.asarray(row['crop_xyxy'][:2]);ankle=wb['kp2d_coco23'][i,15+seed['foot'],:2]-offset;mask=np.zeros(im.shape[:2],np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.r_[curve-offset,ankle[None]].astype(np.float32)).astype(int),255);mask=cv2.dilate(mask,np.ones((11,11),np.uint8));corners=cv2.goodFeaturesToTrack(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),maxCorners=6,qualityLevel=.004,minDistance=4,mask=mask);extra=np.empty((0,2)) if corners is None else corners[:,0]+offset
        for j,xy in enumerate(np.r_[curve,extra]):queries.append([i,*xy]);metadata.append(dict(seed_index=si,source_pts=seed['pts'],foot=seed['foot'],kind='blade_boundary' if j<5 else 'boot_material',source_native_uv=xy.tolist(),human_reviewed=False))
    q=np.asarray(queries,float);model=q.copy()
    for j in range(len(q)):
        t=int(q[j,0]);tr=transforms[t];model[j,1:]=(q[j,1:]-tr[:2]+.5)*tr[2:4]-.5+tr[4:6]
    np.savez_compressed(out/'queries.npz',queries_native=q,queries_model=model,crop_transforms=transforms,pts=np.array([r['pts'] for r in rows]));config=dict(schema='rinkspace.native-cotracker-blade-input.v1',source_index_sha256=digest(output/'source_index.json'),source_anchor_sha256=digest(review),source_time_base=[1,12800],source_exposures=len(rows),model_shape=[384,512],pixel_transform='model=(native-crop_origin+0.5)*resize_scale-0.5+padding; inverse uses each target exposure transform',body_or_camera_projection_used=False,source_original_observations_changed=False,queries=metadata,source_seed_count=len(source['blade_curves']));(out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(frames=len(rows),query_points=len(q),seed_curves=len(source['blade_curves']))),flush=True)


def infer(root,output,source_dir,checkpoint,destination,first=None,last=None):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);cv2.setNumThreads(2);project=Path(__file__).resolve().parents[3];source_dir=AssetStore(project).source(Path(source_dir).resolve());checkpoint=AssetStore(project).source(Path(checkpoint).resolve());sys.path.insert(0,str(source_dir))
    from cotracker.predictor import CoTrackerOnlinePredictor
    root,output,dest=Path(root),Path(output),Path(destination);dest.mkdir(exist_ok=False);inp=output/'cotracker_input';config=json.loads((inp/'config.json').read_text(encoding='utf8'));z=load(inp/'queries.npz');pts=z['pts'];begin=0 if first is None else int(np.searchsorted(pts,round(first*12800)));end=len(pts) if last is None else int(np.searchsorted(pts,round(last*12800),side='right'));indices=np.arange(begin,end);n=len(indices);selected=(z['queries_native'][:,0]>=begin)&(z['queries_native'][:,0]<end);meta=[q for q,s in zip(config['queries'],selected) if s];q=z['queries_model'][selected].copy();q[:,0]-=begin;native_queries=z['queries_native'][selected].copy();cache=np.load(inp/'video_uint8.npy',mmap_mode='r');state=torch.load(checkpoint,map_location='cpu',weights_only=True);state=state['model'] if 'model' in state else state;model=CoTrackerOnlinePredictor(checkpoint=None,window_len=16).eval();model.model.load_state_dict(state,strict=True)
    class OverlapEncoderCache(torch.nn.Module):
        def __init__(self,encoder):
            super().__init__();self.encoder=encoder;self.tail_input=None;self.tail_features=None;self.verified=False;self.verification_error=None;self.reused_frames=0
        def reset(self):self.tail_input=None;self.tail_features=None
        def forward(self,x):
            overlap=8;can_reuse=self.tail_input is not None and len(x)>=overlap and torch.equal(x[:overlap],self.tail_input)
            if can_reuse:
                new=self.encoder(x[overlap:]) if len(x)>overlap else self.tail_features[:0];result=torch.cat((self.tail_features,new),0)
                if not self.verified:
                    exact=self.encoder(x);self.verification_error=float(torch.max(abs(exact-result)));self.verified=True
                    if not torch.allclose(exact,result,atol=2e-5,rtol=2e-5):raise ValueError('Source-frame encoder cache is not numerically equivalent')
                self.reused_frames+=overlap
            else:result=self.encoder(x)
            self.tail_input=x[-overlap:].clone();self.tail_features=result[-overlap:].clone();return result
    encoder_cache=OverlapEncoderCache(model.model.fnet);model.model.fnet=encoder_cache
    start=time.perf_counter();record=dict(schema='rinkspace.source-cotracker3-online-inference.v1',checkpoint_sha256=digest(checkpoint),source_identity=json.loads((source_dir/'source_identity.json').read_text()),input_config_sha256=digest(inp/'config.json'),device='CPU',threads=2,torch=torch.__version__,strict_state_tensors=len(state),source_first_last_pts=[int(pts[begin]),int(pts[end-1])],source_frames=n,queries=meta,model_executed=False,camera_or_body_updated=False,contact_feedback=False)
    (dest/'config.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');results={}
    with (dest/'inference_progress.jsonl').open('w',encoding='utf8',buffering=1) as log,torch.inference_mode():
        for name,reverse in [('forward',False),('backward',True)]:
            encoder_cache.reset()
            order=indices[::-1] if reverse else indices;queries=q.copy()
            if reverse:queries[:,0]=n-1-queries[:,0]
            model(video_chunk=torch.zeros(1,1,3,384,512),queries=torch.tensor(queries[None],dtype=torch.float32),is_first_step=True,add_support_grid=True);step=model.step;track=visibility=None
            for k in range(0,max(n-step,1),step):
                part=np.asarray(cache[order[k:k+2*step]]).copy();video=torch.from_numpy(part).permute(0,3,1,2)[None].float();track,visibility=model(video_chunk=video,add_support_grid=True);score=(torch.sigmoid(model.model.online_vis_predicted[:,:,:len(q)])*torch.sigmoid(model.model.online_conf_predicted[:,:,:len(q)]));log.write(json.dumps(dict(event='model_window',direction=name,processed_frames=min(k+2*step,n),total_frames=n,seconds=time.perf_counter()-start,encoder_reused_frames=encoder_cache.reused_frames,encoder_cache_maximum_feature_error=encoder_cache.verification_error))+'\n')
                np.savez_compressed(dest/'partial.npz',tracks_model=track[0].cpu().numpy(),visibility_score=score[0].cpu().numpy(),direction=np.array(name),source_order=order[:track.shape[1]])
            coordinates=track[0].cpu().numpy();scores=score[0].cpu().numpy()
            if len(coordinates)!=n:raise ValueError('Model omitted source tail exposures')
            if reverse:coordinates=coordinates[::-1].copy();scores=scores[::-1].copy()
            tr=z['crop_transforms'][indices];coordinates=(coordinates+.5-tr[:,None,4:6])/tr[:,None,2:4]-.5+tr[:,None,:2]
            supported=(np.arange(n)[:,None]<=q[:,0][None]) if reverse else (np.arange(n)[:,None]>=q[:,0][None]);scores[~supported]=0
            for j,item in enumerate(native_queries):coordinates[int(item[0])-begin,j]=item[1:]
            results[name+'_native_uv']=coordinates;results[name+'_visibility_score']=scores;results[name+'_supported_time']=supported
        np.savez_compressed(dest/'tracks.npz',**results,pts=pts[indices],queries_native=native_queries,crop_transforms=z['crop_transforms'][indices]);record.update(model_executed=True,seconds=time.perf_counter()-start,complete_native_exposures=n,queries_exact_source_coordinates_restored=True,source_point_visibility_is_not_human_verification=True,encoder_reused_frames=encoder_cache.reused_frames,encoder_cache_maximum_feature_error=encoder_cache.verification_error);(dest/'report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='complete',frames=n,seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=n,queries=len(q),seconds=time.perf_counter()-start,destination=str(dest))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','infer']);p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--review');p.add_argument('--source-dir');p.add_argument('--checkpoint');p.add_argument('--destination');p.add_argument('--first',type=float);p.add_argument('--last',type=float);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.output,a.review)
    else:infer(a.root,a.output,a.source_dir,a.checkpoint,a.destination,a.first,a.last)
