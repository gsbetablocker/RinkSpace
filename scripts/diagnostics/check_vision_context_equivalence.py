"""Measure cached/native RGB forward equivalence with actual saved weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rinkspace.vision import configure, load_training_manifest, sha256
from rinkspace.vision_network import RinkVisionNet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--sequence-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pts',type=int,default=229530880)
    args=p.parse_args()
    config,data=configure(args.config)
    manifest,_=load_training_manifest(config,data)
    index=next(i for i,s in enumerate(manifest['samples']) if s['source_pts']==args.pts)
    with np.load(args.sequence_directory/'training_sequences.npz') as stored:
        rgb=stored[f'sample_{index}'][1]
    saved=torch.load(args.model,map_location='cpu',weights_only=True)
    net=RinkVisionNet(**saved['model_config']).cuda().eval()
    net.load_state_dict(saved['state_dict'])
    x=torch.from_numpy(rgb.copy()).cuda().float().permute(2,0,1)[None,None]/255
    def compare(a,b):
        delta=(a.float()-b.float()).abs()
        return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
            argmax_disagreement=int((a.argmax(2)!=b.argmax(2)).sum()),exact_equal=torch.equal(a,b))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        context=net.encode_context(x)
        native=net(x)
        cached=net(x,context_features=context)
        repeated=net(x)
        recomputed=net._pretrained_features(x[:,0])
        feature_rows=[dict(native_stride=list(a.stride()),cached_stride=list(b[:,0].stride()),
            exact_equal=torch.equal(a,b[:,0]),max_abs=float((a-b[:,0]).abs().max()))
            for a,b in zip(recomputed,context)]
        low_hw=native['hidden'].shape[-2:]
        box=torch.tensor([[0,0,rgb.shape[1],rgb.shape[0]]],device='cuda',dtype=torch.float32)
        raw=context[0][:,0]
        interp=torch.nn.functional.interpolate(raw,size=low_hw,mode='bilinear',align_corners=False)
        roi=net._sample_context_roi(raw,box,rgb.shape[:2],low_hw)
    result=dict(model_sha256=sha256(args.model),source_pts=args.pts,
        source_rgb_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
        inference_mode='actual native RGB; reset memory; no geometric prior',
        logits=compare(native['logits'],cached['logits']),
        primitive_logits=compare(native['primitive_logits'],cached['primitive_logits']),
        native_repeat_logits=compare(native['logits'],repeated['logits']),
        features=feature_rows,
        full_frame_fpn_interpolation=dict(interpolate_dtype=str(interp.dtype),roi_dtype=str(roi.dtype),
            max_abs=float((interp.float()-roi.float()).abs().max())))
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
