"""Extract reproducible material boundaries; never invent positional landmarks.

Input is an explicit recipe, output is artwork-only data. No solver imports,
video observations, physical installation dimensions, or user approvals are made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def image(path):
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    result = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if result is None:
        raise ValueError(f'Cannot decode image: {path}')
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGRA)
    elif result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    return result


def mask_for(im, rule):
    bgr, alpha = im[:, :, :3], im[:, :, 3]
    if 'composite_background_bgr' in rule:
        a=alpha.astype(float)[:,:,None]/255.0
        bgr=np.rint(bgr.astype(float)*a+np.array(rule['composite_background_bgr'])*(1-a)).astype(np.uint8)
        alpha=np.full(alpha.shape,255,np.uint8)
    active = alpha >= rule.get('alpha_min', 128)
    mode = rule.get('color', 'opaque')
    if mode == 'white':
        active &= bgr.min(axis=2) >= rule.get('threshold', 200)
    elif mode == 'dark':
        active &= bgr.max(axis=2) <= rule.get('threshold', 170)
    elif mode == 'nonwhite':
        active &= bgr.min(axis=2) <= rule.get('threshold', 170)
    elif mode == 'red':
        active &= (bgr[:, :, 2].astype(int) > bgr[:, :, 1].astype(int)+rule.get('chroma', 45)) & (bgr[:, :, 2].astype(int)>bgr[:, :, 0].astype(int)+rule.get('chroma',45))
    elif mode != 'opaque':
        raise ValueError(f'Unsupported color selection: {mode}')
    h, w = im.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    if 'box' in rule:
        x0,y0,x1,y1 = rule['box']
        active &= (xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    if 'radial' in rule:
        cx,cy,r0,r1 = rule['radial']
        r2 = (xx-cx)**2+(yy-cy)**2
        active &= (r2>=r0*r0)&(r2<=r1*r1)
    if 'inside_filled_color' in rule:
        region=mask_for(im,rule['inside_filled_color'])
        cc,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        envelope=np.zeros(region.shape,np.uint8)
        cv2.drawContours(envelope,cc,-1,255,cv2.FILLED)
        active &= envelope>0
    if rule.get('largest_component_bbox'):
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(active.astype(np.uint8),8)
        if count>1:
            chosen=1+int(np.argmax(stats[1:,cv2.CC_STAT_WIDTH]*stats[1:,cv2.CC_STAT_HEIGHT]))
            active=labels==chosen
    if 'component_centroid_box' in rule:
        x0,y0,x1,y1=rule['component_centroid_box']
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(active.astype(np.uint8),8)
        chosen=[i for i in range(1,count) if x0<=centroids[i,0]<x1 and y0<=centroids[i,1]<y1]
        active=np.isin(labels,chosen)
    return active.astype(np.uint8)*255


def boundaries(mask, prefix):
    h,w=mask.shape
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], {'outer':0,'holes':0,'vertices':0,'raster_roundtrip_iou':1.0}
    hierarchy=hierarchy[0]
    curves=[]
    for i,c in enumerate(contours):
        depth=0;p=int(hierarchy[i,3]);q=p
        while q>=0:
            depth+=1;q=int(hierarchy[q,3])
        # Keep all digital contours. Samples are approximation support, never
        # semantic identities or independently constrained point landmarks.
        xy=c[:,0,:].astype(float)
        curves.append({'id':f'{prefix}.boundary.{i:04d}','kind':'closed_material_boundary','role':'hole' if depth%2 else 'outer','parent_id':None if p<0 else f'{prefix}.boundary.{p:04d}','depth':depth,'coordinate_space':'template_uv','parameter_domain':'cyclic_curve_range','sample_role':'raster_boundary_discretization_not_landmark','uv':(xy/np.array([w,h])).tolist(),'source_pixel_area':float(cv2.contourArea(c))})
    reconstructed=np.zeros_like(mask)
    cv2.drawContours(reconstructed,contours,-1,255,cv2.FILLED)
    union=np.count_nonzero((reconstructed>0)|(mask>0))
    iou=1.0 if not union else np.count_nonzero((reconstructed>0)&(mask>0))/union
    return curves,{'outer':sum(c['role']=='outer' for c in curves),'holes':sum(c['role']=='hole' for c in curves),'vertices':sum(len(c['uv']) for c in curves),'raster_roundtrip_iou':float(iou)}


def vector_info(paths):
    out=[]
    for path in paths:
        p=Path(path); entry={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'format':p.suffix.lstrip('.')}
        if p.suffix.lower()=='.svg':
            root=ET.fromstring(p.read_text(encoding='utf8'))
            entry.update(view_box=root.get('viewBox'),path_count=sum(e.tag.rsplit('}',1)[-1]=='path' for e in root.iter()),native_resolution='resolution_independent_source_vector')
        out.append(entry)
    return out


def build(recipe_path,output_path):
    recipe=json.loads(Path(recipe_path).read_text(encoding='utf8'))
    output=Path(output_path);output.mkdir(parents=True,exist_ok=True)
    templates=[]
    for item in recipe['templates']:
        src=image(item['raster_path'])
        crop=item.get('source_crop_xyxy',[0,0,src.shape[1],src.shape[0]])
        x0,y0,x1,y1=map(int,crop);im=src[y0:y1,x0:x1].copy()
        if not im.size:raise ValueError(f'Empty artwork crop: {item["id"]}')
        h,w=im.shape[:2]; components=[]; union_mask=np.zeros((h,w),np.uint8); summary={'named_components':0,'outer':0,'holes':0,'vertices':0}
        for component in item['components']:
            mask_im=im
            if component.get('raster_mask_source_path'):
                separate=image(component['raster_mask_source_path'])
                mask_im=separate[y0:y1,x0:x1].copy()
                if mask_im.shape != im.shape:raise ValueError('Component raster coordinate mismatch')
            mask=mask_for(mask_im,component['selection'])
            union_mask=cv2.bitwise_or(union_mask,mask)
            curves,stats=boundaries(mask,item['id']+'.'+component['id'])
            mp=output/(item['id']+'_'+component['id']+'_mask.png')
            ok,encoded=cv2.imencode('.png',mask)
            if not ok:raise RuntimeError('PNG encoding failed')
            mp.write_bytes(encoded.tobytes())
            components.append({**component,'id':item['id']+'.'+component['id'],'mask_path':str(mp),'geometry':curves,'boundary_counts':stats,'semantic_landmarks':[],'landmark_status':'none_created_from_contour_sampling','measurement_compatibility':['curve_range'],'selection_authority':'explicit_engineering_segmentation_recipe_not_user_physical_constraint','geometry_status':'threshold_derived_candidate_boundary_not_physical_acceptance','holes_semantic_identity':'topological_holes_only_noise_possible_not_confirmed_semantic_holes'})
            summary['named_components']+=1
            for k in ['outer','holes','vertices']:summary[k]+=stats[k]
        union_check=None
        if item.get('verify_union_selection'):
            expected=mask_for(im,item['verify_union_selection'])
            changed=int(np.count_nonzero(union_mask!=expected))
            union_check={'changed_pixels':changed,'matches_original_selection':changed==0}
            if changed:raise ValueError(f'Original artwork reconstruction differs: {item["id"]}: {changed} pixels')
        png=output/(item['id']+'_artwork.png'); ok,enc=cv2.imencode('.png',im);png.write_bytes(enc.tobytes())
        template={k:v for k,v in item.items() if k not in ['components']}
        template.update(schema='rinkspace.solve2.artwork_template.v1',coordinate_system={'origin':'top_left','axes':['right','down'],'units':'unitless_artwork_uv','pixel_to_uv':[1/w,1/h],'does_not_define_physical_banner_size':True},raster={'path':str(png),'width':w,'height':h,'source_crop_xyxy':crop,'source_sha256':hashlib.sha256(Path(item['raster_path']).read_bytes()).hexdigest()},source_vectors=vector_info(item.get('vector_paths',[])),components=components,summary=summary,component_union_check=union_check,semantic_landmarks=[],installation_mapping=None,physical_dimensions=None,solver_admission=False)
        target=output/(item['id']+'.json');target.write_text(json.dumps(template,ensure_ascii=False,indent=2),encoding='utf8')
        templates.append({'id':item['id'],'path':str(target),'ads':item['ads'],'kind':item['kind'],'match':item['match'],**summary})
    return templates


def self_check():
    m=np.zeros((101,101),np.uint8);cv2.circle(m,(50,50),35,255,-1);cv2.circle(m,(50,50),15,0,-1)
    cs,counts=boundaries(m,'ring');assert counts['outer']==1 and counts['holes']==1 and counts['raster_roundtrip_iou']==1.0
    assert all(x['sample_role'].endswith('not_landmark') for x in cs)
    im=np.zeros((10,10,4),np.uint8);im[2:5,3:8,:]=255
    s=mask_for(im,{'color':'white','box':[3,2,8,5]});assert np.count_nonzero(s)==15
    return {'ring_outer':counts['outer'],'ring_holes':counts['holes'],'digital_roundtrip_iou':counts['raster_roundtrip_iou'],'landmarks_created':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe');p.add_argument('--out');p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:print(json.dumps(self_check()))
    if args.recipe:print(json.dumps(build(args.recipe,args.out),ensure_ascii=True))
