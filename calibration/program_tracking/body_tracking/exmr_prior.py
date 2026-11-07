"""Pinned ExMR camera-only comparison on existing exact-PTS action inputs.

The network's camera translation is diagnostic. World initialization and skating
constraints are owned by the same existing native body/skate pipeline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

EXMR_COMMIT = 'eac72e47339b404c0443f5fb323412e2ce432410'
SAM_COMMIT = 'b5c765a0d89d789985e186d396315e7590887b94'
HAMER_COMMIT = '3a01849f4148352e9260b69bf28b65d1671a4905'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def runtime(data):
    import subprocess
    base = Path(data)/'runtime_tools/body'
    for folder, expected in [('ExMR', EXMR_COMMIT), ('HaMeR', HAMER_COMMIT),('ExMR/third_party/sam-3d-body',SAM_COMMIT)]:
        got = subprocess.check_output(['git', '-C', str(base/folder), 'rev-parse', 'HEAD'], text=True).strip()
        if got != expected: raise ValueError(f'{folder}: unexpected commit {got}')
    for folder in [base/'exmr_packages', base/'HaMeR', base/'ExMR']:
        sys.path.insert(0, str(folder))
    os.environ['TORCH_HOME'] = str(Path(data)/'cache/torch')
    # No automatic network use in inference. DINOv3 code is prepared separately.
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    return torch


class Progress:
    def __init__(self, root, stage):
        self.path = Path(root)/'comparison_progress.jsonl'
        self.start = time.perf_counter(); self.stage = stage
    def __call__(self, event, **kw):
        row = dict(stage=self.stage, event=event, seconds=time.perf_counter()-self.start, **kw)
        with self.path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)


def transform_pixels(xy, affine):
    xy = np.asarray(xy)
    return np.einsum('ij,...j->...i', affine[:2, :2], xy)+affine[:2, 2]


def prepare(prepared, contract, output, size=1024):
    """Known raw person boxes, with each released model's own crop semantics."""
    import cv2
    from .action_clip_input import _official_frames
    from .camera_bridge import native_to_rectified,rectified_to_native
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame_dir=output/'frames';frame_dir.mkdir(exist_ok=True)
    d=load(prepared);c=json.loads(Path(contract).read_text(encoding='utf8'))
    pts=d['source_pts'];n=len(pts)
    if pts.dtype.kind not in 'iu' or n<2 or not np.all(np.diff(pts)==512):
        raise ValueError('Require original continuous uniform25Hz real source PTS')
    protocol='raw_person_rect_bbox__sam_padding1.25__exmr_aspect0.75_enlarge1.2_v2'
    identity=dict(gvhmr_prepared_sha256=digest(prepared),protocol=protocol,size=size)
    identity_path=output/'input_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf8'))!=identity:raise ValueError('Cached input identity differs')
    else:
        if any(frame_dir.glob('*.png')):raise ValueError('Unbound cached crops: preserve and choose a fresh output')
        write_json(identity_path,identity)
    affines=[];K=[];boxes=[];raw_boxes=[];full_raw=[];full_model=[]
    kp=d['rect_coco17'].copy()
    # Same 9 samples per native bbox edge as the original GVHMR preparation.
    for i,(x0,y0,x1,y1) in enumerate(d['roi_native_xyxy']):
        t=np.linspace(0,1,9)
        border=np.concatenate([np.c_[x0+(x1-x0)*t,np.full(9,y0)],np.c_[x0+(x1-x0)*t,np.full(9,y1)],
            np.c_[np.full(9,x0),y0+(y1-y0)*t],np.c_[np.full(9,x1),y0+(y1-y0)*t]])
        rect,valid=native_to_rectified(border,d['camera25'][i],K_rect=d['K_rect'][i])
        if valid.sum()<18:raise ValueError('Raw person box cannot be rectified')
        lo,hi=rect[valid].min(0),rect[valid].max(0);center=(lo+hi)/2;wh=hi-lo
        # ExMR's released bbx_xys_from_xyxy(...aspect_ratio=192/256).
        model_side=1.2*max(wh[1],wh[0]/.75)
        # Oversampled virtual image includes SAM's full 1.25 raw-bbox padding;
        # it is not itself passed as the SAM person bbox.
        extent=max(1.5*max(wh),model_side)
        scale=(size-1)/extent
        A=np.array([[scale,0,(size-1)/2-scale*center[0]],
                    [0,scale,(size-1)/2-scale*center[1]],[0,0,1.]])
        affines.append(A);K.append(A@d['K_rect'][i])
        boxes.append(np.r_[transform_pixels(center,A),model_side*scale])
        raw_boxes.append(np.r_[transform_pixels(lo,A),transform_pixels(hi,A)])
        full_raw.append(np.r_[lo,hi]);full_model.append(np.r_[center,model_side])
        kp[i,:,:2]=transform_pixels(kp[i,:,:2],A)
    affines=np.array(affines);K=np.array(K);boxes=np.array(boxes,np.float32);raw_boxes=np.array(raw_boxes,np.float32)
    log=Progress(output,'prepare');valid_fraction=[];closure=[]
    video=c['source']['path']
    for i,(stamp,bgr) in enumerate(_official_frames(video,pts)):
        if hashlib.sha256(bgr.tobytes()).hexdigest()!=str(d['source_frame_bgr_sha256'][i]):
            raise ValueError(f'Source content differs at PTS {stamp}')
        path=frame_dir/f'{i:06d}.png';camera=d['camera25'][i]
        Kn=np.array([[np.exp(camera[6]),0,camera[8]],[0,np.exp(camera[7]),camera[9]],[0,0,1.]])
        if camera[10]!=0:raise ValueError('Unsupported camera skew')
        mx,my=cv2.initUndistortRectifyMap(Kn,camera[11:25],np.eye(3),K[i],(size,size),cv2.CV_32FC1)
        if i in (0,n//2,n-1):
            yy,xx=np.meshgrid(np.linspace(0,size-1,17).astype(int),np.linspace(0,size-1,17).astype(int),indexing='ij')
            xy=np.c_[xx.ravel(),yy.ravel()]
            full=transform_pixels(xy,np.linalg.inv(affines[i]));oracle,good=rectified_to_native(full,camera,K_rect=d['K_rect'][i])
            actual=np.c_[mx[yy.ravel(),xx.ravel()],my[yy.ravel(),xx.ravel()]]
            error=np.linalg.norm(actual[good]-oracle[good],axis=1)
            if error.size:closure.append(float(error.max()))
            if error.size and error.max()>2e-4:raise ValueError('OpenCV/native ray-map closure failed')
        if not path.exists():
            image=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if not cv2.imwrite(str(path),image):raise OSError('Cannot save lossless crop')
        else:
            image=cv2.imread(str(path))
            if image is None or image.shape[:2]!=(size,size):raise ValueError('Invalid cached crop')
        valid_fraction.append(float(((mx>=0)&(mx<bgr.shape[1])&(my>=0)&(my<bgr.shape[0])).mean()))
        if i%32==0 or i==n-1:log('source_crops',completed=i+1,total=n)
    np.savez_compressed(output/'prepared.npz',source_pts=pts,source_time_base=np.array([1,12800]),
        bbox_xys=boxes,bbox_xyxy=raw_boxes,K_input=K.astype(np.float32),rect_to_input_affine=affines,
        bbox_raw_rect_xyxy=np.array(full_raw),bbox_model_rect_xys=np.array(full_model),
        coco17=kp,R_w2c=d['R_w2c'],K_rect=d['K_rect'],camera25=d['camera25'],
        bbox_inferred=d['bbox_inferred'],source_mapping_missing=d['source_mapping_missing'])
    write_json(output/'preparation.json',dict(schema='rinkspace.exmr.same-source-input.v2',protocol=protocol,
        exmr_commit=EXMR_COMMIT,gvhmr_prepared=str(Path(prepared).resolve()),
        gvhmr_prepared_sha256=digest(prepared),source_video=video,source_pts=[int(pts[0]),int(pts[-1])],
        frames=n,fps=25,temporal_context='same complete 76-95 second continuous input as original GVHMR',
        virtual_image_size=size,storage='lossless PNG; original source pixels and native Brown rays',
        person_bbox='raw known person bbox mapped through native Brown inverse; not the expanded GVHMR square',
        sam_crop='released raw bbox -> GetBBoxCenterScale padding1.25 -> native512 input',
        temporal_bbox='released ExMR aspect192/256 and enlarge1.2, represented in the virtual image',
        camera_handling='same midpoint camera; original readout/exposure preserved in common downstream solve',
        ray_map_oracle_max_px=max(closure),crop_valid_fraction=valid_fraction,
        bbox_inferred_pts=pts[d['bbox_inferred']].tolist(),
        body_keypoints_control='original GVHMR COCO17 transformed only by known image affine',
        extra_foot_keypoints='network anatomical landmarks; not blade endpoints, shell or contact truth',
        source_protocol_failure_retained='parent directory contains first expanded-square feature input and outputs'))
    log('prepare_complete',frames=n,ray_map_oracle_max_px=max(closure))


def keypoints(data, output, batch_size=8):
    """Additional anatomical foot/hand evidence; original COCO17 stays fixed."""
    torch=runtime(data)
    import cv2
    import onnxruntime as ort
    from exmr.preprocessing.keypoints import rtmw
    from exmr.preprocessing.keypoints.mappings import COCO133_TO_COCO63
    output=Path(output);d=load(output/'prepared.npz');log=Progress(output,'keypoints')
    path=Path(data)/'cache/pretrained/exmr/rtmw/rtmw-x_384x288_fp16.onnx'
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,
        providers=[('CUDAExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    if session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('RTMW CUDA unavailable')
    log('inference_started',frames=len(d['source_pts']),providers=session.get_providers())
    result=[]
    inp=session.get_inputs()[0]; names=[x.name for x in session.get_outputs()]
    for first in range(0,len(d['source_pts']),batch_size):
        tensors=[];centers=[];scales=[]
        for i in range(first,min(first+batch_size,len(d['source_pts']))):
            bgr=cv2.imread(str(output/'frames'/f'{i:06d}.png'))
            center,scale=rtmw._bbox_xyxy2cs(d['bbox_xyxy'][i],rtmw.BOX_PADDING)
            crop,scale=rtmw._top_down_affine(rtmw.INPUT_SIZE,scale,center,bgr)
            normalized=(crop-np.array(rtmw.MEAN))/np.array(rtmw.STD)
            tensors.append(np.ascontiguousarray(normalized.transpose(2,0,1),dtype=np.float32))
            centers.append(center);scales.append(scale)
        count=len(tensors);images=np.stack(tensors)
        images=np.pad(images,[(0,batch_size-count),(0,0),(0,0),(0,0)],mode='edge')
        sx,sy=session.run(names,{inp.name:images})
        xy,score=rtmw._decode(sx[:count],sy[:count],np.array(centers),np.array(scales))
        result.append(np.concatenate([xy,score[...,None]],axis=-1)[:,COCO133_TO_COCO63])
        log('network_anatomical_2d',completed=first+count,total=len(d['source_pts']))
    raw=np.concatenate(result).astype(np.float32)
    from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD
    kp=raw[:,:23].copy();kp[:,17:23,2]=(kp[:,17:23,2]>VISIBILITY_THRESHOLD['rtmw']).astype(np.float32)
    # A source-specific shared body cache may be absent (FS preparation used
    # explicit zero placeholders). Never replace actual detector pixels by it.
    shared=d['coco17'];valid_shared=np.isfinite(shared).all(-1)&(shared[...,2]>0)
    kp[:,:17]=np.where(valid_shared[...,None],shared,kp[:,:17])
    np.savez_compressed(output/'keypoints.npz',source_pts=d['source_pts'],coco63_native_network=raw,
        coco23_shared_body=kp,foot_semantics=np.array('COCO anatomical foot points; not blade or shoe observations'))
    log('complete',model_sha256=digest(path),shared_body_points_preserved=int(valid_shared.sum()),missing_shared_points_keep_native=True)


def sam_features(data, output, batch_size=8):
    torch=runtime(data)
    from exmr.preprocessing import body_features as mod
    output=Path(output);d=load(output/'prepared.npz');log=Progress(output,'sam')
    class PNGDataset(mod._FrameDataset):
        def __init__(self,frame_dir,boxes,K,transform,load_image,prepare_batch):
            def png_loader(path,**kw):return load_image(Path(path).with_suffix('.png'),**kw)
            super().__init__(frame_dir,boxes,K,transform,png_loader,prepare_batch)
    mod._FrameDataset=PNGDataset
    log('loading_model',frames=len(d['source_pts']),batch_size=batch_size)
    original_hub_load=torch.hub.load
    def local_dino(repo,*args,**kwargs):
        if repo!='facebookresearch/dinov3' or kwargs.get('pretrained',False):
            raise RuntimeError('Unexpected automatic torch.hub source/weights request')
        folder=Path(data)/'cache/torch/hub/facebookresearch_dinov3_main'
        if not (folder/'hubconf.py').is_file():raise FileNotFoundError('Prepare the fixed DINOv3 code first')
        kwargs['source']='local'
        return original_hub_load(str(folder),*args,**kwargs)
    torch.hub.load=local_dino
    extractor=mod.SamBodyFeatures(Path(data)/'cache/pretrained/exmr/sam-3d-body-dinov3',
        fp16=False,batch_size=batch_size,num_workers=0,device='cuda')
    original=extractor.model.forward_step
    native_dir=output/'sam3db_body_native';native_dir.mkdir(exist_ok=True)
    cursor=0
    def save_forward(batch,*args,**kw):
        nonlocal cursor
        out=original(batch,*args,**kw)
        arrays={}
        def flatten(obj,prefix=''):
            if isinstance(obj,dict):
                for k,v in obj.items():flatten(v,prefix+('/' if prefix else '')+str(k))
            elif torch.is_tensor(obj):arrays[prefix]=obj.detach().float().cpu().numpy() if obj.is_floating_point() else obj.detach().cpu().numpy()
        flatten(out)
        count=batch['img'].shape[0]
        path=native_dir/f'batch_{cursor:06d}.npz'
        np.savez_compressed(path,source_pts=d['source_pts'][cursor:cursor+count],**arrays)
        cursor+=count;log('native_body_saved',completed=cursor,total=len(d['source_pts']),fields=list(arrays))
        return out
    extractor.model.forward_step=save_forward
    with torch.inference_mode():
        token=extractor(output/'frames',torch.from_numpy(d['bbox_xyxy']),torch.from_numpy(d['K_input']))
    np.savez_compressed(output/'body_features.npz',source_pts=d['source_pts'],body_feat=token.numpy())
    # Native output is MHR, not mislabeled as SMPL-X. Exact original model fields are kept.
    np.savez_compressed(native_dir/'topology.npz',faces=extractor.model.head_pose.faces.detach().cpu().numpy())
    write_json(native_dir/'native_output.json',dict(model='SAM 3D Body DINOv3',model_format='native MHR',
        source_pts=d['source_pts'].tolist(),raw_output='unmodified body decoder return; tensor values saved before discarded by ExMR feature extractor',
        units='model native units and axes; no conversion claimed here',
        world_processing=False,features_reused_without_second_forward=True))
    log('complete',frames=cursor,peak_cuda_bytes=int(torch.cuda.max_memory_allocated()))


def hand_features(data, output, batch_size=8):
    """Exact HaMeR feature path; the unused MANO mesh model is not instantiated."""
    torch=runtime(data)
    import cv2
    if os.name=='nt':os.environ['PYOPENGL_PLATFORM']='win32'
    from hamer.configs import get_config
    from hamer.models.backbones import create_backbone
    from hamer.models.heads import build_mano_head
    from exmr.preprocessing import hand_features as mod
    from exmr.preprocessing.keypoints.mappings import COCO63_LEFT_HAND,COCO63_RIGHT_HAND
    output=Path(output);d=load(output/'prepared.npz');k=load(output/'keypoints.npz');log=Progress(output,'hands')
    root=Path(data)/'cache/pretrained/exmr/hamer'
    cfg=get_config(str(root/'hamer_ckpts/model_config.yaml'));cfg.defrost()
    cfg.MANO.MEAN_PARAMS=str(root/'data/mano_mean_params.npz')
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    if 'BBOX_SHAPE' not in cfg.MODEL:cfg.MODEL.BBOX_SHAPE=[192,256]
    cfg.freeze()
    class TokenNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=create_backbone(cfg);self.mano_head=build_mano_head(cfg)
    log('loading_model',mano_mesh_required=False)
    with torch.device('cuda'):model=TokenNet()
    saved=torch.load(root/'hamer_ckpts/checkpoints/hamer.ckpt',map_location='cpu',weights_only=False)
    state=saved['state_dict'];active={k:v for k,v in state.items() if k.startswith(('backbone.','mano_head.'))}
    model.load_state_dict(active,strict=True);model.eval().to('cuda').requires_grad_(False)
    del saved,state,active
    # Same feature computation, crops, normalization, mirror and token hook as official ExMR.
    extractor=mod.HamerHandFeatures.__new__(mod.HamerHandFeatures)
    extractor.model=model;extractor.cfg=cfg;extractor.fp16=True
    extractor.batch_size=batch_size;extractor.num_workers=0;extractor.device='cuda';extractor._tokens=None
    model.mano_head.transformer.register_forward_hook(extractor._store_tokens)
    from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD
    raw=torch.from_numpy(k['coco63_native_network'])
    threshold=VISIBILITY_THRESHOLD['rtmw']
    boxes=torch.stack([mod.hand_boxes_from_keypoints(raw[:,COCO63_LEFT_HAND],threshold),
        mod.hand_boxes_from_keypoints(raw[:,COCO63_RIGHT_HAND],threshold)],dim=1)
    mean=255*np.array(cfg.MODEL.IMAGE_MEAN);std=255*np.array(cfg.MODEL.IMAGE_STD)
    result=[]
    for first in range(0,len(boxes),batch_size):
        crops=[]
        for i in range(first,min(first+batch_size,len(boxes))):
            image=cv2.imread(str(output/'frames'/f'{i:06d}.png'))
            b=boxes[i].numpy();center=(b[:,2:]+b[:,:2])/2;scale=2*(b[:,2:]-b[:,:2])/200
            for h in range(2):
                if scale[h,0]==0 or scale[h,1]==0:
                    crop=np.zeros((3,cfg.MODEL.IMAGE_SIZE,cfg.MODEL.IMAGE_SIZE),np.float32)
                else:
                    extent=mod._expand_to_aspect_ratio(scale[h]*200,cfg.MODEL.BBOX_SHAPE).max()
                    patch=mod._crop_hand(image.copy(),center[h,0],center[h,1],extent,cfg.MODEL.IMAGE_SIZE,flip=h==0)
                    crop=patch[:,:,::-1].transpose(2,0,1).astype(np.float32)
                    crop=(crop-mean[:,None,None])/std[:,None,None]
                crops.append(crop)
        with torch.inference_mode():
            features=extractor._forward(torch.from_numpy(np.array(crops,np.float32))).reshape(-1,2,1024).cpu().numpy()
        result.append(features);log('hand_tokens',completed=first+len(features),total=len(boxes))
    tokens=np.concatenate(result)
    if not np.isfinite(tokens).all():raise ValueError('Nonfinite HaMeR tokens')
    np.savez_compressed(output/'hand_features.npz',source_pts=d['source_pts'],hand_feat=tokens,hand_boxes=boxes.numpy())
    write_json(output/'hamer_feature_path.json',dict(hamer_commit=HAMER_COMMIT,
        checkpoint_sha256=digest(root/'hamer_ckpts/checkpoints/hamer.ckpt'),
        backbone_and_head_weights='strict loading, identical released modules',
        omitted='MANO mesh, discriminators, training losses and renderer; none are called by official token path',
        mano_mesh_used=False,hand_boxes_source='RTMW anatomical hand keypoints only',
        static_probabilities_used_for_contact=False))
    log('complete',peak_cuda_bytes=int(torch.cuda.max_memory_allocated()))


def infer(data, output):
    torch=runtime(data)
    from exmr.model.exmr import ExMR
    from exmr.geometry.rotation_conversions import matrix_to_rotation_6d
    output=Path(output);d=load(output/'prepared.npz');k=load(output/'keypoints.npz')
    body=load(output/'body_features.npz');hand=load(output/'hand_features.npz');pts=d['source_pts']
    for other in (k,body,hand):np.testing.assert_array_equal(pts,other['source_pts'])
    log=Progress(output,'exmr')
    model_path=Path(data)/'runtime_tools/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz'
    checkpoint=Path(data)/'cache/pretrained/exmr/exmr/exmr.safetensors'
    log('loading_model',frames=len(pts),camera_only=True)
    model=ExMR.from_pretrained(checkpoint,model_path,device='cuda')
    R=torch.from_numpy(d['R_w2c']).float()
    ang=matrix_to_rotation_6d(R[1:]@R[:-1].mT);ang=torch.cat([ang,ang[-1:]],dim=0)
    from exmr.preprocessing.keypoints import VISIBILITY_THRESHOLD
    native23=k['coco63_native_network'][:,:23].copy()
    native23[:,:,2]=(native23[:,:,2]>VISIBILITY_THRESHOLD['rtmw']).astype(np.float32)
    common={name:torch.from_numpy(array).float()[None].cuda() for name,array in dict(
        bbx_xys=d['bbox_xys'],K_fullimg=d['K_input'],cam_angvel=ang.numpy(),
        body_feat=body['body_feat'],hand_feat=hand['hand_feat']).items()}
    stage_checks={}
    # Main ExMR uses its released RTMW visibility protocol. A cheap control keeps
    # the original GVHMR COCO17 exactly, isolating the effect of its 2D input.
    for input_name,points,prefix in [('native23',native23,'exmr'),
            ('shared17',k['coco23_shared_body'],'exmr_shared17')]:
        log('inference_started',frames=len(pts),body_input=input_name,
            source_pts=[int(pts[0]),int(pts[-1])],postprocess=False,
            snap_to_ground=False,smooth_incam=False,smooth_global=False,fps=25)
        with torch.inference_mode():out=model(**common,kp2d=torch.from_numpy(points).float()[None].cuda(),
            postprocess=False,snap_to_ground=False,smooth_incam=False,smooth_global=False,
            camera_only=True,fps=25)
        cpu={key:{k:v[0].cpu().numpy() for k,v in out[key].items()}
            for key in ('smplx_incam_prefoot','smplx_incam_postfoot','smplx_incam')}
        pre,post,final=(cpu[key] for key in ('smplx_incam_prefoot','smplx_incam_postfoot','smplx_incam'))
        for field in ('global_orient','transl','betas'):
            np.testing.assert_array_equal(pre[field],post[field]);np.testing.assert_array_equal(post[field],final[field])
        unaffected=np.ones(63,bool);unaffected[18:24]=False
        np.testing.assert_array_equal(pre['body_pose'][:,unaffected],post['body_pose'][:,unaffected])
        nohand=np.ones(63,bool);nohand[51:63]=False
        np.testing.assert_array_equal(post['body_pose'][:,nohand],final['body_pose'][:,nohand])
        for suffix,key in [('prefoot','smplx_incam_prefoot'),('postfoot','smplx_incam_postfoot'),('final','smplx_incam')]:
            arrays=cpu[key]
            if not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('Nonfinite ExMR output')
            target=output/(prefix+'_'+suffix+'.npz')
            if target.exists():raise FileExistsError('Preserve existing network output: '+str(target))
            np.savez_compressed(target,pts=pts,source_pts=pts,R_w2c=d['R_w2c'],K_rect=d['K_rect'],**arrays)
        np.savez_compressed(output/(prefix+'_static_diagnostic.npz'),source_pts=pts,
            static_conf_logits=out['static_conf_logits'][0].cpu().numpy(),used_for_contact_truth=np.array(False))
        stage_checks[input_name]=dict(root_translation_shape_unchanged_by_refiners=True,
            foot_refiner_changed_only_ankles=True,hand_refiner_changed_only_elbows_wrists_and_fingers=True,
            maximum_ankle_axis_angle_change_rad=float(np.max(np.abs(pre['body_pose'][:,18:24]-post['body_pose'][:,18:24]))))
        log('camera_candidates_saved',body_input=input_name,prefix=prefix)
    write_json(output/'inference.json',dict(exmr_commit=EXMR_COMMIT,checkpoint_sha256=digest(checkpoint),
        smplx_sha256=digest(model_path),frames=len(pts),source_pts=[int(pts[0]),int(pts[-1])],
        source_time_base=[1,12800],fps=25,one_continuous_model_call_per_input=True,input_controls=['native23','shared17'],
        temporal_compatibility='25Hz explicit real source sequence; model trained frame-index cadence is not recalibrated',
        world_branch_executed=False,postprocess=False,snap_to_ground=False,
        smooth_incam=False,smooth_global=False,orientation_normalization=False,
        primary_body_input='released ExMR RTMW COCO23 with >5.5 binary visibility',
        control_body_input='original GVHMR COCO17, extra6 RTMW anatomical foot points with binary visibility',
        extra6='RTMW anatomical foot landmarks, not skate contacts',stage_checks=stage_checks,
        network_static_logits_contact_truth=False,peak_cuda_bytes=int(torch.cuda.max_memory_allocated())))
    log('complete',candidates=['exmr_prefoot','exmr_postfoot','exmr_final','exmr_shared17_final'])


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','keypoints','sam','hands','infer'])
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prepared');p.add_argument('--contract');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args(argv)
    if a.stage=='prepare':prepare(a.prepared,a.contract,a.output)
    elif a.stage=='infer':infer(a.data_root,a.output)
    else:{'keypoints':keypoints,'sam':sam_features,'hands':hand_features}[a.stage](a.data_root,a.output,a.batch_size)

if __name__=='__main__':main()
