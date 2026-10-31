"""CPU SAM2 footwear candidates, with unchanged native RGB and explicit prompts."""

from rinkspace.storage import resolve_asset
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import cv2
from .rebuild_prepare import digest


def infer(root,project,roles=('fit',),only_pts=None,dataset=None):
    import torch
    from PIL import Image
    from transformers import Sam2VideoModel,Sam2VideoProcessor
    import torch.nn.functional as F
    torch.set_num_threads(2);cv2.setNumThreads(1)
    root,project=Path(root),Path(project);cache=root/'footwear_v2';out=Path(dataset) if dataset else cache;model_dir=resolve_asset(project, project/'models/cache/sam2.1-hiera-tiny')
    index=json.loads((out/'source_index.json').read_text(encoding='utf8'));pre=json.loads((model_dir/'preprocessor_config.json').read_text())
    checkpoint_sha=digest(model_dir/'model.safetensors');model,loading=Sam2VideoModel.from_pretrained(str(model_dir),local_files_only=True,output_loading_info=True)
    errors={k:loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if loading.get(k)}
    if errors:raise ValueError(errors)
    model=model.eval().requires_grad_(False).to('cpu',dtype=torch.float32);processor=Sam2VideoProcessor.from_pretrained(str(model_dir),local_files_only=True)
    mask_dir=cache/'mask_candidates';mask_dir.mkdir(exist_ok=True);start=time.perf_counter()
    selected=[r for r in index['rows'] if r['status']=='source_crop_available' and r['role'] in roles and (not only_pts or r['pts'] in only_pts)]
    with (out/'mask_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        log.write(json.dumps(dict(event='start',frames=len(selected),roles=roles,device='cpu',threads=2,checkpoint_sha256=checkpoint_sha))+'\n')
        for row in selected:
            dest=mask_dir/f'{row["pts"]}.npz';receipt=mask_dir/f'{row["pts"]}.json'
            if dest.exists() and receipt.exists():
                old=json.loads(receipt.read_text(encoding='utf8'))
                if old['crop_sha256']==row['crop_sha256'] and old['checkpoint_sha256']==checkpoint_sha:continue
                raise ValueError('Existing candidate has different source/model: '+str(dest))
            bgr=cv2.imread(row['crop']);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];offset=np.asarray(row['crop_xyxy'][:2]);bh=(np.asarray(row['search_boxes_uv'])[0,3]-row['search_boxes_uv'][0][1])/.245
            centres=np.asarray(row['search_centres_uv'])-offset;ankles=np.asarray(row['ankle_source_uv'])-offset;knees=np.asarray(row['knee_source_uv'])-offset
            boxes=np.asarray(row['search_boxes_uv'])-np.r_[offset,offset]
            session=processor.init_video_session(inference_device='cpu',inference_state_device='cpu',processing_device='cpu',video_storage_device='cpu',dtype=torch.float32)
            prompts=[]
            for foot in range(2):
                positive=centres[foot];calf=ankles[foot]+.35*(knees[foot]-ankles[foot]);other=centres[1-foot]
                points=np.stack((positive,calf,other));labels=[1,0,0]
                # Touching boot prompts may overlap; a negative inside the same
                # apparent boot is ambiguous and is omitted, never relabelled.
                if np.linalg.norm(positive-other)<.055*bh:points=points[:2];labels=labels[:2]
                processor.add_inputs_to_inference_session(session,frame_idx=0,obj_ids=foot+1,original_size=(h,w),
                    input_boxes=[[boxes[foot].tolist()]],input_points=[[points.tolist()]],input_labels=[[labels]])
                prompts.append(dict(foot=foot,points_crop=points.tolist(),labels=labels,box_crop=boxes[foot].tolist()))
            session.obj_with_new_inputs=[1,2]
            resized=np.asarray(Image.fromarray(rgb).resize((pre['size']['width'],pre['size']['height']),Image.Resampling.BILINEAR),dtype=np.float32)
            values=(resized*float(pre['rescale_factor'])-np.asarray(pre['image_mean'],np.float32))/np.asarray(pre['image_std'],np.float32)
            frame=torch.from_numpy(np.ascontiguousarray(values.transpose(2,0,1)))
            tick=time.perf_counter()
            with torch.inference_mode():
                result=model(inference_session=session,frame=frame,frame_idx=0)
                logits=F.interpolate(result.pred_masks,(h,w),mode='bilinear',align_corners=False).numpy()[:,0]
            masks=(logits>0).astype(np.uint8);np.savez_compressed(dest,masks=masks,object_ids=np.asarray(result.object_ids),crop_xyxy=row['crop_xyxy'],pts=row['pts'])
            info=dict(pts=row['pts'],role=row['role'],crop_sha256=row['crop_sha256'],checkpoint_sha256=checkpoint_sha,
                model='facebook/sam2.1-hiera-tiny',device='cpu',inference_seconds=time.perf_counter()-tick,prompts=prompts,
                status='source_segmentation_candidate_unreviewed',physical_foot_ids='COCO candidates, not automatically confirmed',
                boot_vs_blade_boundary='not yet classified; candidate may include blade, leg or ice',areas=masks.sum((1,2)).tolist(),
                mask_file=str(dest),mask_sha256=digest(dest),RGB_model_projection_used_as_measurement=False)
            receipt.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
            shown=bgr.copy()
            for foot,color in enumerate([(70,230,80),(255,150,40)]):
                mask=masks[foot].astype(bool);shown[mask]=(shown[mask]*.72+np.asarray(color)*.28).astype(np.uint8)
                contours,_=cv2.findContours(masks[foot],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(shown,contours,-1,color,1,cv2.LINE_AA)
            scale=min(900/w,600/h);shown=cv2.resize(shown,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(mask_dir/f'{row["pts"]}_qa.jpg'),shown,[cv2.IMWRITE_JPEG_QUALITY,95])
            log.write(json.dumps(dict(event='candidate',pts=row['pts'],areas=info['areas'],seconds=info['inference_seconds'],elapsed_seconds=time.perf_counter()-start,status=info['status']))+'\n')
            print(json.dumps(dict(pts=row['pts'],seconds=info['inference_seconds'],areas=info['areas'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--roles',nargs='+',default=['fit']);ap.add_argument('--only-pts',nargs='+',type=int);ap.add_argument('--dataset',type=Path);a=ap.parse_args();infer(a.root,a.project,a.roles,a.only_pts,a.dataset)
