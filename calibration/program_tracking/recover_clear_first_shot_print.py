"""Recover complete clear print components from current native material edges.

The old material chart is only a component identity/search navigator.  Native
connected-component pixels initialize profiles; all observations are then read
again from the current RGB. No projected path point is emitted.
"""
from copy import deepcopy
import cv2
import numpy as np
from .repair_first_shot_board_patterns import *
from .extract_first_shot_board_source import (refined_ad_mapping,transform,
    path_stations,profile_read,segment_records,native_overlay,source_segments,
    print_inside_current_board_band)


def current_clear_outline(raw,mask,path,bright=False,debug=False):
    low=np.floor(path.min(0)-22).astype(int);high=np.ceil(path.max(0)+22).astype(int)
    x0,y0=np.maximum(low,0);x1,y1=np.minimum(high,[1920,1080])
    if min(x1-x0,y1-y0)<12:return None
    gray=cv2.cvtColor(raw[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if not bright:binary=255-binary
    binary[mask[y0:y1,x0:x1]]=0
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    choices=[];seed_area=abs(cv2.contourArea(path.astype('float32')))
    for contour in contours:
        p=contour[:,0].astype(float)+[x0,y0]
        if len(p)<15 or abs(cv2.contourArea(p.astype('float32')))<35:continue
        area=abs(cv2.contourArea(p.astype('float32')))
        if not .45<area/max(seed_area,1)<2.0:continue
        bx,by,bw,bh=cv2.boundingRect(contour)
        if bx<=1 or by<=1 or bx+bw>=x1-x0-1 or by+bh>=y1-y0-1:continue
        center_delta=np.linalg.norm((p.min(0)+p.max(0)-path.min(0)-path.max(0))/2)
        if center_delta>19:continue
        hu=cv2.matchShapes(path.astype('float32'),p.astype('float32'),cv2.CONTOURS_MATCH_I1,0)
        pscale=np.ptp(p,axis=0);sscale=np.ptp(path,axis=0)
        pn=(p-p.min(0))/np.maximum(pscale,1.);sn=(path-path.min(0))/np.maximum(sscale,1.)
        normalized_distances=np.linalg.norm(pn[:,None]-sn[None],axis=2)
        normalized_shape=float(np.median(normalized_distances.min(0))+np.median(normalized_distances.min(1)))
        if debug:print('candidate',len(p),'area',area/max(seed_area,1),'center',center_delta,'hu',hu,'normalized_shape',normalized_shape,flush=True)
        if hu>.35 and normalized_shape>.15:continue
        d=np.linalg.norm(p[:,None]-path[None],axis=2)
        shape=np.median(d.min(1))+np.median(d.min(0))
        if shape>23:continue
        choices.append((shape+20*min(hu,normalized_shape),p))
    if not choices:return None
    _,outline=min(choices,key=lambda r:r[0]);outline=np.vstack([outline,outline[:1]])
    query=path_stations(outline,2.)
    rows,_=profile_read(raw,mask,query,[1/3]*3,6,maximum_blur=4.,minimum_amplitude=25.,
                        navigation_tolerance=3.5,require_clear_end_plateaus=False)
    good=[r for r in rows if r is not None]
    if debug:print('profile',len(good),len(rows),len(good)/len(rows),flush=True)
    if len(good)<.62*len(rows)or len(good)<12:return None
    # A sharp horizontal remnant alone is not an identifiable letter. Require
    # resolved, oppositely placed sides of the current outline as well.
    actual=np.asarray([r[0]for r in good]);extent=np.ptp(outline,axis=0)
    if np.ptp(actual,axis=0).min()<min(8.,extent.min()*.6):return None
    segments=segment_records(rows,spacing=45,closed=True,minimum_span=8.)
    for s in segments:
        s['source_clear_component_identity_verified_by_current_shape']=True
        s['native_connected_component_rgb_profile_support_fraction']=len(good)/len(rows)
    return segments


def recover(indices=None,target_id=None):
    indices=list(indices or range(201,1249));manifest={r['index']:r for r in read(DATA/'frame_manifest.json')['frames']}
    reg=read(MODEL/'feature_registry.json');scene=read(MODEL/'scene.json');templates=read(MODEL/'ad2_material_template.json')
    a=np.load(DATA/'camera_seeds.npz');cameras=dict(zip(a['programme_indices'],a['camera25']));reports=[]
    worlds={tid:feature_world(scene,d,templates)for tid,d in reg['targets'].items()if d['kind']=='ad_print'and d['advertisement_id']in(3,10)}
    for i in indices:
        packet=read(REPAIR/f'board_candidates/{i:06d}.json');raw=cv2.imread(manifest[i]['image_path'])
        mask=cv2.dilate((cv2.imread(str(current_mask_path(i,manifest[i])),0)>0).astype('uint8'),np.ones((5,5),np.uint8))>0
        mask[:120,:570]=True;mask[945:,1660:]=True;changes=[]
        mapping=refined_ad_mapping(packet['features'],cameras[i],scene,reg,3)
        for tid,world in worlds.items():
            if target_id is not None and tid!=target_id:continue
            d=reg['targets'][tid];n=d['advertisement_id']
            if 'inner'in tid:continue
            path,depth=project_fast(cameras[i],world)
            if depth.min()<=0 or path[:,0].max()<-20 or path[:,0].min()>1940:continue
            if n==3 and mapping is not None:path=transform(templates['advertisements']['3']['contours'][d['contour_key']]['material_uv'],mapping)
            if np.ptp(path,axis=0).min()<12:continue
            segments=current_clear_outline(raw,mask,path,bright=n==3)
            if not segments:continue
            for s in segments:
                if not print_inside_current_board_band(packet['features'],np.asarray(s['uv'])).all():segments=[];break
            if not segments:continue
            previous=[f for f in packet['features']if f['target_id']==tid]
            oldcount=sum(len(s['uv'])for f in previous for s in f['segments'])
            feature=deepcopy(previous[0])if previous else dict(id=tid,target_id=tid,label=d['contour_key'],label_zh=tid,
                physical_surface='board',review_layer='ad2_print',allowed_for_fit=False,current_source_only_readout=True,
                model_or_template_generated_observation=False)
            feature['segments']=segments;feature['source_identity_recovery']='current_native_connected_print_component_then_RGB_edge_readout'
            packet['features']=[f for f in packet['features']if f['target_id']!=tid]+[feature]
            changes.append(dict(target_id=tid,old_points=oldcount,new_points=sum(len(s['uv'])for s in segments)))
        if changes:
            packet['point_count']=sum(len(s['uv'])for f in packet['features']for s in f['segments'])
            packet['diagnostics']['printed_point_count']=packet['point_count']-packet['diagnostics']['board_point_count']
            write(REPAIR/f'board_candidates/{i:06d}.json',packet);reports.append(dict(index=i,changes=changes))
        if i in(300,500,800,850,852,900,1000,1200):
            cv2.imwrite(str(REPAIR/f'board_diagnostics/board_{i:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
    receipt_path=REPAIR/'clear_print_component_recovery.json'
    previous=read(receipt_path).get('rows',[])if receipt_path.is_file()else[]
    combined={r['index']:{c['target_id']:c for c in r['changes']}for r in previous}
    for r in reports:
        for c in r['changes']:
            old=combined.setdefault(r['index'],{}).get(c['target_id'])
            if old:c['old_points']=old['old_points']
            combined[r['index']][c['target_id']]=c
    total=[dict(index=i,changes=list(changes.values()))for i,changes in sorted(combined.items())]
    write(receipt_path,dict(rows=total,current_source_RGB_observations_only=True,model_geometry_unchanged=True))
    print('Recovered',len(reports),'frames',sum(len(r['changes'])for r in reports),'components',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--indices');p.add_argument('--target');args=p.parse_args()
    recover([int(i)for i in args.indices.split(',')]if args.indices else None,args.target)
