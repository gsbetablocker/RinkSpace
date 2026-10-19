"""Compare image-only network predictions with registered native source curves.

This is a diagnostic on declared training/development observations, not a
substitute for a frozen independent full-shot reader evaluation. The annotations
are used only after the image forward; no projection or drawing enters it.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from rinkspace.vision import (ROOT,configure,load_training_manifest,RinkVisionModule,
    _annotations_in_crop,read_json,write_json,sha256)
from rinkspace.vision_supervision import annotation_role


def evaluate_prediction(prediction,sample,radius):
    names=prediction['classes'];height,width=prediction['native_hw']
    logits=prediction['class_log_probabilities']
    primitive=prediction['primitive_probabilities']
    rows=[]
    for annotation in _annotations_in_crop(sample,(0,0),(height,width)):
        if annotation_role(annotation,sample)['authority_role']!='direct_user_confirmed_native_UV':continue
        points=np.asarray(annotation['uv']);normals=np.asarray(annotation['normal'])
        category=names.index(annotation['kind'])
        offsets=np.arange(-radius,radius+1,dtype=np.float32)
        query=points[:,None]+normals[:,None]*offsets[None,:,None]
        inside=(query[:,:,0]>=0)&(query[:,:,0]<width)&(query[:,:,1]>=0)&(query[:,:,1]<height)
        grid=torch.as_tensor(query,device=logits.device,dtype=torch.float32)
        grid=grid*grid.new_tensor([2/(width-1),2/(height-1)])-1
        sampled=F.grid_sample(logits,grid[None],align_corners=True)[0]
        other=[i for i in range(len(names)) if i!=category]
        odds=(sampled[category]-torch.logsumexp(sampled[other],dim=0)).detach().cpu().numpy()
        odds[~inside]=-np.inf
        peak_index=odds.argmax(1);normal_offset=offsets[peak_index]
        peak=points+normals*normal_offset[:,None]
        interval=np.asarray(annotation['interval_px'])
        outside=np.maximum.reduce([interval[:,0]-normal_offset,normal_offset-interval[:,1],np.zeros(len(points))])
        point_grid=torch.as_tensor(points,device=logits.device,dtype=torch.float32)
        point_grid=point_grid*point_grid.new_tensor([2/(width-1),2/(height-1)])-1
        cls=F.grid_sample(logits,point_grid[None,:,None],align_corners=True)[0,:,:,0].argmax(0).cpu().numpy()
        prim=F.grid_sample(primitive,point_grid[None,:,None],align_corners=True)[0,:,:,0].argmax(0).cpu().numpy()
        expected=annotation.get('primitive');wrong_primitive=(prim!=('line','circle','other').index(expected)) if expected in ('line','circle','other') else None
        rows.append(dict(group_id=annotation['group_id'],physical_curve_id=annotation.get('physical_curve_id'),
            kind=annotation['kind'],primitive=expected,source_point_indices=annotation['source_point_indices'].tolist(),
            authoritative_uv=points.tolist(),predicted_normal_profile_peak_uv=peak.tolist(),
            predicted_normal_offset_px=normal_offset.tolist(),drawing_support_interval_px=interval.tolist(),
            drawing_support_outside_px=outside.tolist(),semantic_wrong_count=int((cls!=category).sum()),
            primitive_wrong_count=int(wrong_primitive.sum()) if wrong_primitive is not None else None,
            source_position_count=len(points),peak_outside_support_count=int((outside>0).sum()),
            median_outside_support_px=float(np.median(outside)),p95_outside_support_px=float(np.percentile(outside,95)),
            max_outside_support_px=float(outside.max()),search_radius_native_px=radius,
            peak_hits_search_boundary_count=int((abs(normal_offset)==radius).sum()),
            peak_is_predicted_not_RGB_measured=True,own_branch_profile_may_contain_other_real_paint=True))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--sequence-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pts',type=int,action='append')
    parser.add_argument('--profile-radius',type=int,default=24)
    args=parser.parse_args()
    if args.profile_radius<1:parser.error('profile radius must be positive')
    config,data=configure(args.config);manifest,plan=load_training_manifest(config,data)
    metadata=read_json(args.sequence_directory/'sequence_sources.json')
    bundle=args.sequence_directory/'training_sequences.npz'
    if sha256(bundle)!=metadata['bundle_sha256']:raise ValueError('Native sequence bundle changed')
    args.output.mkdir(parents=True,exist_ok=True)
    vision=RinkVisionModule(args.model,ROOT/config['flow_weights'],device=config['device'],exposure_registry=config.get('source_clock_registry'))
    report=dict(schema='rinkspace.native-source-boundary-network-evaluation.v1',model_path=str(args.model),model_sha256=vision.model_sha256,
        config_path=str(args.config),config_sha256=sha256(args.config),source_bundle_sha256=metadata['bundle_sha256'],
        source_drawing_or_projection_supplied_to_network=False,independent_validation=False,
        comparison_role='known source regression diagnosis; sources already used for development or training',
        inferred_profile_peaks_are_not_RGB_observations=True,accuracy_accepted=False,frames=[])
    started=time.monotonic()
    with np.load(bundle,allow_pickle=False) as stored:
        for i,sample in enumerate(manifest['samples']):
            if args.pts and sample['source_pts'] not in args.pts:continue
            if not any(annotation_role(a,sample)['authority_role']=='direct_user_confirmed_native_UV' for a in sample['annotations']):continue
            sequence=stored[f'sample_{i}'];row=metadata['rows'][i];scope=sample['source_shot_scope']
            binding=dict(source_sha256=sample['source_hash'],time_base=sample['time_base'],source_pts=sample['source_pts'],
                shot_id=scope['shot_id'],native_hw=list(sequence.shape[1:3]))
            prior=int(row['native_pts'][0])
            if prior<binding['source_pts']:
                vision.predict(sequence[0],{**binding,'source_pts':prior},reset=True)
                prediction=vision.predict(sequence[1],binding)
            else:prediction=vision.predict(sequence[1],binding,reset=True)
            groups=evaluate_prediction(prediction,sample,args.profile_radius)
            image=Image.fromarray(sequence[1]);draw=ImageDraw.Draw(image)
            for group in groups:
                for x,y in group['predicted_normal_profile_peak_uv']:draw.point((round(x),round(y)),fill=(0,255,255))
                for x,y in group['authoritative_uv']:draw.point((round(x),round(y)),fill=(255,255,0))
            overlay=args.output/f"{sample['source_pts']}_USER_YELLOW_NETWORK_PEAK_CYAN.png";image.save(overlay)
            frame=dict(source_binding=binding,groups=groups,overlay_path=str(overlay),
                original_RGB_source_path=sample['frame_path'],original_RGB_sha256=sample['frame_sha256'])
            report['frames'].append(frame)
            print(json.dumps(dict(source_pts=sample['source_pts'],groups=len(groups),
                semantic_wrong=sum(g['semantic_wrong_count'] for g in groups),
                profile_outside=sum(g['peak_outside_support_count'] for g in groups))),flush=True)
    report['elapsed_seconds']=time.monotonic()-started
    write_json(args.output/'evaluation.json',report)


if __name__=='__main__':main()
