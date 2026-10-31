"""Current original-image RootSIFT print references for navigation only."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json

import av
import cv2
import numpy as np

from rinkspace.camera_reuse_localization import descriptors
from rinkspace.core.native import NativeCore
from rinkspace.path_policy import guard_project_write

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
REFERENCES = [
    ('Ad1','s07',1496064,[1335,450,1915,539]),
    ('Ad2','s07',1496064,[447,465,1018,548]),
    ('Ad3','s07',1601792,[460,324,838,384]),
    ('Ad4','s09',2234880,[100,353,1915,437]),
    ('Ad6','s08',1806080,[935,408,1632,531]),
    ('Ad7','s09',1838080,[766,522,1565,630]),
]
OVERLAY_EXCLUSIONS = [[0,0,570,290],[1580,940,1920,1080]]
EDGE_ORDER = ['top_edge','right_vertical','bottom_edge','left_vertical']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source_image(path,pts):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if stream.time_base!=Fraction(1,12800) or (stream.width,stream.height)!=(1920,1080):
            raise ValueError('Registered original source dimensions/time base changed')
        container.seek(pts,stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Original source exposure has no PTS')
            if frame.pts<pts:continue
            if frame.pts!=pts or frame.time_base!=Fraction(1,12800):
                raise ValueError('Requested original source PTS not decoded exactly')
            return frame.to_ndarray(format='bgr24')
    raise ValueError('Original source ended before requested PTS')


def prepare(root,output):
    cv2.setNumThreads(2)
    root,output=Path(root).resolve(),guard_project_write(output)
    if output.exists():raise FileExistsError('Preserve existing reference bank: '+str(output))
    if output.is_relative_to(PROJECT.resolve()):raise ValueError('Reference image outputs belong outside OneDrive')
    index=read(root/'index.json'); baseline=read(root.parent/'baseline.json')
    source_path=Path(index['source']['resolved_path']).resolve(strict=True)
    source_sha=sha(source_path)
    if source_sha!=index['source']['actual_sha256'] or source_sha!=index['source']['sha256']:
        raise ValueError('Original source SHA256 differs from bootstrap')
    world=dict(np.load(root/'native_world.npz',allow_pickle=False))
    world_metadata=read(root/'native_world.json')
    anchor_path=root/'independent_view_seeds.json'
    anchors=read(anchor_path)['anchors']
    camera_indices={a['shot']:a['camera_index'] for a in anchors}
    library=Path(baseline['inputs']['library']['path']).resolve(strict=True)
    if sha(library)!=baseline['inputs']['library']['sha256']:raise ValueError('Current native library changed')
    core=NativeCore(library)
    shot_lookup={s['shot_id']:s for s in index['shots']}
    protected={str(p):sha(p) for p in [root/'index.json',root/'native_world.npz',root/'native_world.json',
        anchor_path,root.parent/'baseline.json',library]}
    images={};poses={};staged=[]
    for ad,sid,pts,roi in REFERENCES:
        window=shot_lookup[sid]['processing']
        if pts<window['first_pts'] or pts>window['last_pts'] or (pts-window['first_pts'])%256:
            raise ValueError('Reference PTS lies outside its registered native shot')
        key=(sid,pts)
        if key not in images:images[key]=source_image(source_path,pts)
        image=images[key]
        person_path=root/'person_masks'/sid/f'{pts}.png'
        person=cv2.imread(str(person_path),cv2.IMREAD_GRAYSCALE)
        if person is None or person.shape!=(1080,1920):raise ValueError('Current reference person mask missing or wrong size')
        protected[str(person_path)]=sha(person_path)
        roi_mask=np.zeros((1080,1920),np.uint8)
        x0,y0,x1,y1=roi;roi_mask[y0:y1+1,x0:x1+1]=255
        overlay_mask=np.zeros_like(roi_mask)
        for left,top,right,bottom in OVERLAY_EXCLUSIONS:overlay_mask[top:bottom,left:right]=255
        valid=roi_mask.copy();valid[(person>0)|(overlay_mask>0)]=0
        xy,description=descriptors(image,valid)
        if description is None:description=np.empty((0,128),np.float32)
        if len(xy)!=len(description) or description.shape!=(len(xy),128):raise ValueError('RootSIFT output shape differs')
        if len(xy):
            rounded=np.rint(xy).astype(int)
            if np.any(valid[rounded[:,1],rounded[:,0]]==0):raise ValueError('Descriptor centre outside valid source mask')
        pose_path=root/sid/'navigation/poses_reanchored.npz'
        if sid not in poses:
            state=dict(np.load(pose_path,allow_pickle=False))
            poses[sid]=state;protected[str(pose_path)]=sha(pose_path)
        state=poses[sid];where=np.flatnonzero(state['pts']==pts)
        if len(where)!=1:raise ValueError('Exact reference PTS missing from current independent navigation')
        pose=state['poses'][where[0]]
        if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('Invalid current reference navigation pose')
        camera_index=camera_indices[sid]
        candidates=[c for c in world_metadata['cuts'] if c.get('instance_id')=='print_instance/'+ad
                    and c.get('role')=='panel_outer_boundary']
        edges=[]
        for edge in EDGE_ORDER:
            found=[c for c in candidates if c['template_path_id'].split('/')[0]==ad+'.'+edge]
            if len(found)!=1:raise ValueError(f'{ad} needs one current native {edge}')
            edges.append(found[0])
        xyz=np.concatenate([world['xyz'][c['begin']:c['end']] for c in edges])
        offsets=np.r_[0,np.cumsum([c['end']-c['begin'] for c in edges])].astype(np.int32)
        projected=core.project_motion(world['rig'][camera_index],np.tile(pose,(6,1)),-2.,1.,1080,0.,xyz)
        arrays=dict(keypoints=xy,descriptors=description.astype(np.float32,copy=False),
            world_xyz=xyz,projected_uv=projected['uv'],projection_status=projected['status'],
            edge_offsets=offsets,edge_path_ids=np.array([c['id'] for c in edges]),navigation_pose=pose)
        preview=image.copy()
        excluded=(roi_mask>0)&(valid==0)
        preview[excluded]=(preview[excluded]*.35+np.array([0,0,170])*.65).astype(np.uint8)
        cv2.rectangle(preview,(x0,y0),(x1,y1),(0,255,255),2)
        for x,y in xy:cv2.circle(preview,(round(float(x)),round(float(y))),2,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(preview,f'{ad} {sid.upper()} PTS {pts} | RootSIFT {len(xy)} | navigation only',
                    (30,55),cv2.FONT_HERSHEY_SIMPLEX,.85,(0,255,255),2,cv2.LINE_AA)
        record=dict(ad_id=ad,shot_id=sid,pts=pts,source_time_seconds=pts/12800,
            source_video=str(source_path),source_video_sha256=source_sha,
            source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            source_image_size=[1920,1080],pixel_domain='native_distorted',roi_xyxy_inclusive=roi,
            person_mask=str(person_path),person_mask_sha256=protected[str(person_path)],
            descriptor_kind='RootSIFT',descriptor_implementation='rinkspace.camera_reuse_localization.descriptors',
            keypoints=len(xy),descriptor_coordinates='original full-frame native pixels',
            detector_image='full-frame grayscale from original decoded BGR',
            detector_mask='ROI AND not person_mask AND not current broadcast overlay exclusions',
            overlay_exclusions_xyxy=OVERLAY_EXCLUSIONS,
            excluded_person_pixels_in_roi=int(np.sum((roi_mask>0)&(person>0))),
            valid_search_pixels=int(np.count_nonzero(valid)),camera_index=camera_index,
            navigation_pose_file=str(pose_path),navigation_pose_sha256=protected[str(pose_path)],
            navigation_pose=pose.tolist(),native_edge_path_ids=[c['id'] for c in edges],
            native_edge_offsets=offsets.tolist(),native_projection_nonzero_status=int(np.count_nonzero(projected['status'])),
            native_projection_pose_model='constant reference-frame independent navigation pan/tilt/logf',
            physical_instance_assignment='assistant inference from independently viewed current source ROI',
            navigation_only=True,fit_eligible=False,fit_observations_created=0)
        staged.append((ad,arrays,record,preview,valid))
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for ad,arrays,record,preview,valid in staged:
        folder=output/ad;folder.mkdir()
        np.savez_compressed(folder/'reference.npz',**arrays)
        if not cv2.imwrite(str(folder/'reference_overlay.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('Cannot write reference overlay')
        if not cv2.imwrite(str(folder/'descriptor_mask.png'),valid):raise OSError('Cannot write descriptor mask')
        record.update(descriptor_npz=str(folder/'reference.npz'),descriptor_npz_sha256=sha(folder/'reference.npz'),
            overlay_image=str(folder/'reference_overlay.jpg'),descriptor_mask=str(folder/'descriptor_mask.png'),
            descriptor_mask_sha256=sha(folder/'descriptor_mask.png'))
        write(folder/'reference.json',record);records.append(record)
        print(json.dumps(dict(ad_id=ad,keypoints=record['keypoints'],overlay=record['overlay_image'],fit_eligible=False)),flush=True)
    for path,expected in protected.items():
        if sha(path)!=expected:raise RuntimeError('Current reference dependency changed: '+path)
    receipt=dict(schema='rinkspace.independent_print_navigation_references.v1',references=records,
        input_sha256=protected,descriptor_kind='RootSIFT',navigation_only=True,fit_eligible=False,
        fit_observations_created=0,legacy_new_shot_points_or_motion_loaded=False,
        feature_matching_executed=False,user_visual_acceptance=False)
    write(output/'references.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.root,args.output)
