"""Native first-shot board materials with explicit per-advertisement targets.

The inherited camera and continuous physical panel provide search locations
only. Every retained coordinate is a current-image colour transition. Historical
user-labelled exposures initialize print identities and material charts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .extract_opening_frame_batch import ROOT, read, write, sha
from .framewise_model import feature_world
from .projection import project_fast
from .render_single_frame_source_review import native_overlay, source_segments

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps')
OLD_MODEL = ROOT/'events/ankara_2026/models/framewise_8a28_20261013'
MODEL = ROOT/'events/ankara_2026/models/framewise_8a28_first_shot_20261014'
HISTORY = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training')
HIGH = HISTORY/'source_ads_user_inventory_56218880_native.png'
AD5_SOURCE = HISTORY/'continuous_review/Ad4_17_18_19_source_absolute_20261007/raw_native_230338304.png'
QUADS = {
    3:[[1281,300],[1427,327],[1425,357],[1280,329]],
    4:[[794,229],[1184,284],[1183,311],[794,251]],
    6:[[553,203],[611,208],[611,228],[553,221]],
    7:[[396,207],[504,202],[504,219],[397,225]],
    5:[[1319,373],[1518,373],[1518,491],[1319,491]],
}


def transform(uv, matrix):
    return cv2.perspectiveTransform(np.asarray(uv,np.float64)[None],matrix)[0]


def source_template(number):
    """Source material chart retains source ROI/threshold ambiguity explicitly."""
    path=AD5_SOURCE if number==5 else HISTORY/'source_ads_user_inventory_104693760_native.png' if number==7 else HIGH
    raw=cv2.imread(str(path));quad=np.asarray(QUADS[number],np.float32)
    if number==7:quad=np.float32([[271,411],[604,407],[605,469],[272,487]])
    H=cv2.getPerspectiveTransform(quad,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    b,g,r=raw.transpose(2,0,1).astype(float);luma=(b+g+r)/3
    yy,xx=np.indices(raw.shape[:2]);uv=transform(np.column_stack((xx.ravel(),yy.ravel())),H).reshape(1080,1920,2)
    u,v=uv[:,:,0],uv[:,:,1]
    support=(u>.035)&(u<.965)&(v>.14)&(v<.86)
    if number in (3,6,7):ink=(np.minimum.reduce([b,g,r])>135)&support
    elif number==4:
        # White words in the two colour panels; blue/dark federation logo in
        # the intervening white material. The white background is not a glyph.
        panels=(u<.34)|(u>.565)
        ink=((panels&(np.minimum.reduce([b,g,r])>175)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<60))|
             ((~panels)&((luma<135)|(b-g>20))))&support
    else:
        support=np.zeros(raw.shape[:2],bool)
        for box in [(1130,374,1288,465),(1130,468,1295,496),(1332,414,1510,450),(1540,380,1890,483)]:
            x0,y0,x1,y1=box;support[y0:y1,x0:x1]=True
        ink=((luma<140)|(b-g>18)|(g-r>18))&support
    contours,hierarchy=cv2.findContours(ink.astype(np.uint8)*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    items=[]
    for i,c in enumerate(contours):
        x,y,w,h=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
        if area<3 or w<2 or h<3 or len(c)<8:continue
        # Reject paths against a navigation crop. They are not source ink.
        p=c[:,0];neighbors=np.clip(p+[1,0],[0,0],[1919,1079]);neighbors2=np.clip(p-[1,0],[0,0],[1919,1079])
        if not support[p[:,1],p[:,0]].all():continue
        boundary_crop=False
        for delta in ([1,0],[-1,0],[0,1],[0,-1]):
            nearby=np.clip(p+delta,[0,0],[1919,1079])
            if not support[nearby[:,1],nearby[:,0]].all():boundary_crop=True;break
        if boundary_crop:continue
        approx=cv2.approxPolyDP(c,.45,True)[:,0].astype(float)
        if len(approx)<3:continue
        items.append((x,y,w,h,i,approx,bool(hierarchy[0,i,3]>=0)))
    items.sort(key=lambda row:(row[0],row[1],row[-1]))
    result={}
    for j,(x,y,w,h,i,p,hole) in enumerate(items):
        key=f'component{j:03d}'+('_inner' if hole else '')
        m=transform(p,H)
        if number!=5 and ((m<0).any() or (m>1).any()):continue
        pclosed=np.vstack([p,p[:1]])
        # Reference colours establish material polarity, not query precision.
        tang=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0);length=np.linalg.norm(tang,axis=1)
        normal=np.column_stack((-tang[:,1],tang[:,0]))/np.maximum(length[:,None],.01)
        colors=[]
        for sign in (-1,1):
            q=p+normal*sign*2.5
            colors.append(cv2.remap(raw,q[:,0].astype('float32')[:,None],q[:,1].astype('float32')[:,None],cv2.INTER_LINEAR)[:,0].astype(float))
        delta=np.mean(colors[1]-colors[0],axis=0)
        if np.linalg.norm(delta)<8:
            # Closed-path orientation can vary at small concavities. Luma is
            # retained as the source semantic signal for neutral print.
            delta=np.ones(3)
        weights=delta/np.linalg.norm(delta)
        result[key]=dict(id=key,advertisement_id=number,closed=True,
            role='printed_internal_hole_boundary' if hole else 'printed_ink_external_boundary',
            source_native_uv=pclosed.tolist(),material_uv=m.tolist(),
            source_signal_BGR_weights=weights.tolist(),source_native_bbox_xywh=[x,y,w,h],
            material_coordinate_domain='continuous_board_s_h_relative_to_named_internal_graphic' if number==5 else 'ad_rectangle_u_left_to_right_v_top_to_bottom',
            source_native_boundary_uncertainty_px=2.5,
            initialization_accuracy_accepted=False,
            source_annotation_role='user_confirmed_ad_identity_with_current_source_ink_component')
    return dict(schema='rinkspace.source-advertisement-material-chart.v1',advertisement_number=number,
        source=dict(path=str(path),sha256=sha(path),decoded_native_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest()),
        source_search_corners_native_uv=quad.tolist(),source_corners_are_navigation_not_precision_measurements=True,
        source_to_material_homography=H.tolist(),contours=result,
        physical_identity_authority='calibration/inputs/annotation_inventory.json',
        material_chart_follows_continuous_board_arc_and_height=True,
        planar_3d_ad_surface_required=False,source_rectification_is_uncertain_initialization=True)


def build_model():
    MODEL.mkdir(parents=True,exist_ok=True)
    for name in ('scene.json','seed_parameters.json','feature_registry.json','model.json','ad2_material_template.json'):
        if not (MODEL/name).exists():shutil.copy2(OLD_MODEL/name,MODEL/name)
    scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json');templates=read(MODEL/'ad2_material_template.json')
    original=read(OLD_MODEL/'feature_registry.json')['targets']
    templates.setdefault('advertisements',{})
    for number in (3,4,5,6,7):
        if str(number) not in templates['advertisements']:templates['advertisements'][str(number)]=source_template(number)
        for key,record in templates['advertisements'][str(number)]['contours'].items():
            registry['targets'][f'Ad{number}.print.{key}']=dict(kind='ad_print',advertisement_id=number,contour_key=key,
                contour_role=record['role'],residual='point_to_named_finite_print_contour',
                material_coordinate_domain=record['material_coordinate_domain'],template='ad2_material_template.json',
                geometry_status='source_material_chart_initialization_not_fitted_or_accuracy_accepted')
    for ad in scene['advertisements']:
        number=int(ad['user_number'])
        if 8<=number<=13:
            for record in ad.get('external_contours',[]):
                key=record['id']
                registry['targets'][f'Ad{number}.print.{key}']=dict(kind='ad_print',advertisement_id=number,
                    contour_key=key,geometry_source='scene_existing_external_contour',
                    residual='point_to_named_finite_print_contour',
                    geometry_status='inherited_source_print_geometry_unchanged')
    assert all(registry['targets'][k]==v for k,v in original.items())
    write(MODEL/'ad2_material_template.json',templates);write(MODEL/'feature_registry.json',registry)
    manifest=read(MODEL/'model.json')
    for key,name in [('scene','scene.json'),('feature_registry','feature_registry.json'),('material_templates','ad2_material_template.json'),('seed_parameters','seed_parameters.json')]:
        manifest[key]=dict(path=str(MODEL/name),sha256=sha(MODEL/name))
    manifest.update(derived_model_role='first_shot_material_registry_extension',parent_framewise_model=dict(path=str(OLD_MODEL/'model.json'),sha256=sha(OLD_MODEL/'model.json')),
        geometry_values_inherited_unchanged=True,new_material_initialization_accuracy_accepted=False)
    write(MODEL/'model.json',manifest)
    for target in registry['targets'].values():
        if not np.isfinite(feature_world(scene,target,templates)).all():raise ValueError('Nonfinite model material')
    report=dict(model_directory=str(MODEL),scene_bytes_unchanged=sha(MODEL/'scene.json')==sha(OLD_MODEL/'scene.json'),
        previous_targets_preserved_exactly=True,added_target_count=len(registry['targets'])-len(original),
        new_print_counts={k:len(v['contours'])for k,v in templates['advertisements'].items()},
        ad5_existing_geometry_scope='internal_graphic_only_not_whole_advertisement_border')
    write(DATA/'board_model_extension.json',report);return report


def rebuild_ad4_current_atlas():
    """Stitch mutually verified same-print pixels, then initialize real glyphs.

    No atlas coordinate is a source observation. The chart is merely a shared
    physical contour initializer on the unchanged continuous board material.
    """
    scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json');templates=read(MODEL/'ad2_material_template.json')
    archive=np.load(DATA/'camera_seeds.npz');cameras={int(i):c for i,c in zip(archive['programme_indices'],archive['camera25'])}
    manifests={int(r['index']):r for r in read(DATA/'frame_manifest.json')['frames']}
    rows=[]
    ids=[f'Ad4.{k}'for k in ('left_top','right_top','right_bottom','left_bottom')]
    world=np.concatenate([feature_world(scene,registry['targets'][t])for t in ids])
    for index in (340,360,400,440):
        raw=cv2.imread(manifests[index]['image_path']);mask=cv2.imread(manifests[index]['mask_path'],0)>0
        features=read(DATA/f'board_candidates/{index:06d}.json')['features']
        H=refined_ad_mapping(features,cameras[index],scene,registry,4)
        quad=transform([[0,0],[1,0],[1,1],[0,1]],H)
        support=np.zeros((1080,1920),np.uint8);cv2.fillConvexPoly(support,np.rint(quad).astype('int32'),255)
        support[cv2.dilate(mask.astype('uint8'),np.ones((17,17),np.uint8))>0]=0
        support=cv2.erode(support,np.ones((5,5),np.uint8))
        rows.append(dict(index=index,raw=raw,mask=support,H=H))
    base=rows[2];sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.012)
    bk,bd=sift.detectAndCompute(cv2.cvtColor(base['raw'],cv2.COLOR_BGR2GRAY),base['mask'])
    width,height=3000,180;display=np.diag([width,height,1.])@np.linalg.inv(base['H'])
    atlas=np.zeros((height,width,3),np.uint8);owner=np.zeros((height,width),np.int32);quality=np.zeros((height,width),float);evidence=[]
    for row in rows:
        if row is base:H=np.eye(3);count=len(bk);residual=0.
        else:
            k,d=sift.detectAndCompute(cv2.cvtColor(row['raw'],cv2.COLOR_BGR2GRAY),row['mask'])
            pairs=cv2.BFMatcher().knnMatch(d,bd,k=2);seed=base['H']@np.linalg.inv(row['H'])
            matched=[m for m,n in pairs if m.distance<.82*n.distance and
                     np.linalg.norm(transform([k[m.queryIdx].pt],seed)[0]-bk[m.trainIdx].pt)<35]
            p=np.float32([k[m.queryIdx].pt for m in matched]);q=np.float32([bk[m.trainIdx].pt for m in matched])
            H,inliers=cv2.findHomography(p,q,cv2.RANSAC,2.,maxIters=5000,confidence=.999)if len(p)>=4 else(None,None)
            if H is None or int(inliers.sum())<8:
                H=seed;count=0;residual=None
            else:
                good=inliers.ravel().astype(bool);count=int(good.sum());residual=float(np.median(np.linalg.norm(transform(p[good],H)-q[good],axis=1)))
        M=display@H;image=cv2.warpPerspective(row['raw'],M,(width,height));valid=cv2.warpPerspective(row['mask'],M,(width,height),flags=cv2.INTER_NEAREST)>0
        # The base current frame has highest coverage. Alternative frames fill
        # its actual person or image boundary gaps and retain real source RGB.
        score=np.where(valid,2. if row is base else 1.,0.)
        select=score>quality;atlas[select]=image[select];owner[select]=row['index'];quality[select]=score[select]
        evidence.append(dict(frame_index=row['index'],source=manifests[row['index']]['source'],
            same_physical_print_inliers=count,source_RGB_homography_median_px=residual,
            alignment_initializer='source_RGB_mutual_texture_homography'if count>0 else 'current_source_board_edges_with_inherited_width_navigation',
            source_to_atlas_homography=M.tolist(),source_homography_is_navigation_not_model_geometry=True))
    b,g,r=atlas.transpose(2,0,1).astype(float);luma=(b+g+r)/3
    yy,xx=np.indices((height,width));u=xx/width;v=yy/height
    panel=(u<.345)|(u>.57)
    support=(quality>0)&(v>.12)&(v<.86)&(u>.025)&(u<.97)
    ink=((panel&(np.minimum.reduce([b,g,r])>135)&(np.minimum.reduce([b,g,r])/np.maximum(np.maximum.reduce([b,g,r]),1)>.83))|
        ((~panel)&((luma<145)|((b-g)>25))))&support
    ink[:22]=False;ink[-22:]=False
    contours,h=cv2.findContours(ink.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE);accepted=[]
    for i,c in enumerate(contours):
        x,y,w,hh=cv2.boundingRect(c)
        if abs(cv2.contourArea(c))<20 or w<4 or hh<7:continue
        p=c[:,0]
        if any(not support[np.clip(p[:,1]+dy,0,height-1),np.clip(p[:,0]+dx,0,width-1)].all()for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))):continue
        p=cv2.approxPolyDP(c,.7,True)[:,0].astype(float)
        accepted.append((x,y,bool(h[0,i,3]>=0),p))
    accepted.sort(key=lambda a:(a[0],a[1],a[2]));records={}
    for j,(x,y,hole,p) in enumerate(accepted):
        key=f'current_component{j:03d}'+('_inner'if hole else '')
        records[key]=dict(id=key,advertisement_id=4,closed=True,
            role='printed_internal_hole_boundary'if hole else'printed_ink_external_boundary',
            material_uv=(p/[width,height]).tolist(),source_signal_BGR_weights=[0,1,0],
            source_atlas_uv=p.tolist(),material_coordinate_domain='ad_rectangle_u_left_to_right_v_top_to_bottom',
            current_native_source_atlas_only_initialization=True,initialization_accuracy_accepted=False)
    # The panorama may have no mutual texture matches in a blurred overlap.
    # In those intervals retain independently visible native glyph shapes from
    # their own chart, rather than a contour crossing a stitching seam.
    records={k:v for k,v in records.items()if np.mean(np.asarray(v['material_uv'])[:,0])<.835}
    last=rows[0];p=last['raw'];b,g,r=p.transpose(2,0,1).astype(float)
    binary=(np.minimum.reduce([b,g,r])>130)&(np.minimum.reduce([b,g,r])/np.maximum(np.maximum.reduce([b,g,r]),1)>.80)&(last['mask']>0)
    cs,hs=cv2.findContours(binary.astype('uint8')*255,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    tail=[]
    for i,c in enumerate(cs):
        x,y,w,hh=cv2.boundingRect(c)
        if abs(cv2.contourArea(c))<50 or hh<16 or w<5:continue
        path=c[:,0].astype(float);m=transform(path,np.linalg.inv(last['H']))
        if not(.83<np.mean(m[:,0])<.985) or m[:,1].min()<.17 or m[:,1].max()>.86:continue
        if not(2<x and x+w<1918):continue
        q=cv2.approxPolyDP(c,.8,True)[:,0].astype(float);tail.append((float(np.mean(m[:,0])),i,q,hs[0,i,3]>=0))
    tail.sort(key=lambda a:a[0])
    for j,(_,i,path,hole)in enumerate(tail):
        key=f'current_tail{j:03d}'+('_inner'if hole else'')
        records[key]=dict(id=key,advertisement_id=4,closed=True,
            role='printed_internal_hole_boundary'if hole else'printed_ink_external_boundary',
            material_uv=transform(path,np.linalg.inv(last['H'])).tolist(),source_signal_BGR_weights=[0,1,0],
            source_native_uv=path.tolist(),source_frame_index=340,
            material_coordinate_domain='ad_rectangle_u_left_to_right_v_top_to_bottom',
            source_atlas_uv=transform(path,display@(base['H']@np.linalg.inv(last['H']))).tolist(),
            current_native_source_atlas_only_initialization=True,initialization_accuracy_accepted=False)
    out=DATA/'ad4_current_material_atlas.png';cv2.imwrite(str(out),atlas)
    preview=atlas.copy()
    for record in records.values():cv2.polylines(preview,[np.rint(record['source_atlas_uv']).astype('int32')],True,(0,0,255),1)
    cv2.imwrite(str(DATA/'ad4_current_material_atlas_contours.png'),preview)
    templates['advertisements']['4']=dict(advertisement_number=4,contours=records,
        schema='rinkspace.same-physical-print-current-source-atlas.v1',source_evidence=evidence,
        atlas=dict(path=str(out),sha256=sha(out)),native_RGB_only=True,
        atlas_coordinates_are_not_observations=True,existing_ad3D_geometry_unchanged=True,
        rectification_does_not_require_planar_ad3D_surface=True,geometry_initializer_accuracy_accepted=False)
    registry['targets']={k:v for k,v in registry['targets'].items()if not k.startswith('Ad4.print.')}
    for key,record in records.items():
        registry['targets']['Ad4.print.'+key]=dict(kind='ad_print',advertisement_id=4,contour_key=key,contour_role=record['role'],residual='point_to_named_finite_print_contour',template='ad2_material_template.json',material_coordinate_domain=record['material_coordinate_domain'],geometry_status='source_material_chart_initialization_not_fitted_or_accuracy_accepted')
    write(MODEL/'ad2_material_template.json',templates);write(MODEL/'feature_registry.json',registry)
    manifest=read(MODEL/'model.json');manifest['feature_registry']['sha256']=sha(MODEL/'feature_registry.json');manifest['material_templates']['sha256']=sha(MODEL/'ad2_material_template.json');write(MODEL/'model.json',manifest)
    return dict(ad4_contour_count=len(records),source_evidence=evidence,preview=str(DATA/'ad4_current_material_atlas_contours.png'))


def path_stations(path,spacing):
    p=np.asarray(path,float);d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if d[-1]<1:return p
    locations=np.linspace(0,d[-1],max(2,min(400,int(np.ceil(d[-1]/spacing))+1)))
    return np.column_stack([np.interp(locations,d,p[:,k])for k in (0,1)])


def print_inside_current_board_band(features, points):
    """A native colour transition alone does not establish advertisement ink.

    In particular, a partially hidden small Ad5 graphic can make its extended
    search chart touch the upper blue rail. The current source board boundaries
    exclude that rail and the ice even if a strong transition was read there.
    This is an identity rejection only; no coordinate is moved or synthesized.
    """
    p=np.asarray(points,float);boundaries={}
    for role in ('top','contact'):
        samples=[np.asarray(s['uv'],float)for f in features
                 if f['target_id'].startswith('board.'+role+'.')for s in f['segments']]
        if not samples:return np.zeros(len(p),bool)
        q=np.vstack(samples);q=q[np.argsort(q[:,0])]
        x,indices=np.unique(q[:,0],return_index=True)
        boundaries[role]=np.interp(p[:,0],x,q[indices,1])
    return (p[:,1]>boundaries['top']+12.)&(p[:,1]<boundaries['contact']-6.)


def retain_print_band_segments(features, feature):
    retained=[];rejected=[]
    for segment in feature['segments']:
        p=np.asarray(segment['uv'],float);good=print_inside_current_board_band(features,p)
        if good.all():retained.append(segment);continue
        rejected.extend(p[~good].tolist())
        runs=[];current=[]
        for j,accept in enumerate(good):
            if accept:current.append(j)
            else:
                if len(current)>=2:runs.append(current)
                current=[]
        if len(current)>=2:runs.append(current)
        for keep in runs:
            uv=p[keep]
            if np.linalg.norm(uv[-1]-uv[0])<8:continue
            replacement=deepcopy(segment)
            for key,value in segment.items():
                if isinstance(value,list)and len(value)==len(p):replacement[key]=[value[j]for j in keep]
            replacement.update(uv=uv.tolist(),vertex_count=len(keep),closed=False,
                current_source_board_band_checked=True)
            retained.append(replacement)
    feature['segments']=retained
    return rejected


def profile_read(raw,mask,path,weights,radius,*,maximum_blur=16.,minimum_amplitude=12.,navigation_tolerance=None,gradient_direction=None,signal_mode=None,require_clear_end_plateaus=False,minimum_gradient=2.3):
    """Vectorized current source half transitions along an ordered search path."""
    p=np.asarray(path,float)
    if len(p)<2:return [],[]
    tang=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0);tang[0]=p[1]-p[0];tang[-1]=p[-1]-p[-2]
    norm=np.linalg.norm(tang,axis=1);normal=np.column_stack((-tang[:,1],tang[:,0]))/np.maximum(norm[:,None],.01)
    offsets=np.arange(-radius,radius+1);q=p[:,None,:]+offsets[None,:,None]*normal[:,None,:]
    rgb=cv2.remap(raw,q[:,:,0].astype('float32'),q[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
    signal=np.ptp(rgb,axis=2) if signal_mode=='chroma' else rgb@np.asarray(weights,float)
    signal=gaussian_filter1d(signal,.8,axis=1)
    gradient=np.diff(signal,axis=1);amplitude=signal[:,-4:].mean(1)-signal[:,:4].mean(1)
    # Source sign can differ from an approximate template normal. The nearest
    # strong local transition is considered only when both sides disagree.
    magnitude=np.abs(gradient);cost=magnitude/(1+np.abs(offsets[:-1])[None,:]/max(radius*.65,1))
    if gradient_direction is not None:
        wanted=normal@np.asarray(gradient_direction,float)
        cost=np.where(gradient*wanted[:,None]>0,cost,0.)
    peak=cost.argmax(1);rows=[];diagnostics=[]
    tolerance=radius*.7 if navigation_tolerance is None else navigation_tolerance
    for i,k in enumerate(peak):
        if norm[i]<.01 or abs(offsets[k])>tolerance or magnitude[i,k]<minimum_gradient or cost[i,k]<minimum_gradient:rows.append(None);continue
        side=4
        a=float(np.median(signal[i,max(0,k-side-2):max(1,k-1)]));b=float(np.median(signal[i,min(len(offsets)-1,k+2):min(len(offsets),k+side+3)]))
        delta=b-a
        if require_clear_end_plateaus:
            end_a=float(np.median(signal[i,:4]));end_b=float(np.median(signal[i,-4:]));end_delta=end_b-end_a
            if abs(end_delta)<minimum_amplitude or end_delta*delta<=0:rows.append(None);continue
            a,b,delta=end_a,end_b,end_delta
        elif abs(delta)<minimum_amplitude:rows.append(None);continue
        signed=(signal[i]-a)/delta;cross=[]
        for fraction in (.25,.5,.75):
            candidates=np.flatnonzero((signed[:-1]<=fraction)&(signed[1:]>fraction))
            if not len(candidates):break
            j=int(candidates[np.argmin(abs(candidates-k))]);cross.append(j+(fraction-signed[j])/max(1e-9,signed[j+1]-signed[j]))
        if len(cross)!=3 or not cross[0]<=cross[1]<=cross[2] or cross[2]-cross[0]>maximum_blur:rows.append(None);continue
        uv=p[i]+(cross[1]-radius)*normal[i];x,y=np.rint(uv).astype(int)
        if not (0<=uv[0]<=1919 and 0<=uv[1]<=1079):rows.append(None);continue
        qq=uv+normal[i]*np.linspace(-max(2,cross[2]-cross[0]),max(2,cross[2]-cross[0]),5)[:,None]
        xi,yi=np.rint(qq).astype(int).T
        if not(0<=xi.min() and xi.max()<1920 and 0<=yi.min() and yi.max()<1080) or mask[yi,xi].any():rows.append(None);continue
        rows.append((uv,float(cross[2]-cross[0]),float(abs(delta))))
    return rows,diagnostics


def segment_records(rows,*,spacing=240.,closed=False,straight=False,smooth_curve=False,target_id=None,minimum_span=8.):
    """Retain visible runs at common density, with shape-support evidence."""
    if not rows:return []
    full=all(r is not None for r in rows)
    ordered=rows if full else rows+[None]
    if closed and not full:
        cut=next(i for i,r in enumerate(rows)if r is None);ordered=rows[cut+1:]+rows[:cut+1]+[None]
    runs=[];run=[]
    for row in ordered:
        if row is None:
            if len(run)>=2:runs.append(run)
            run=[]
        else:run.append(row)
    if run:runs.append(run)
    output=[]
    for run in runs:
        p=np.asarray([r[0]for r in run]);width=np.asarray([r[1]for r in run]);strength=np.asarray([r[2]for r in run])
        if np.linalg.norm(p[-1]-p[0])<minimum_span and not(full and closed):continue
        cyclic=bool(full and closed)
        if smooth_curve and len(p)>=6:
            from scipy.optimize import least_squares
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            station=distance/max(distance[-1],1.)
            degree=2;design=np.column_stack([station**j for j in range(degree+1)])
            initial=np.linalg.lstsq(design,p,rcond=None)[0]
            fit=least_squares(lambda c:((design@c.reshape(degree+1,2)-p)/np.maximum(width[:,None],1.)).ravel(),initial.ravel(),loss='soft_l1')
            coefficients=fit.x.reshape(degree+1,2)
            locations=np.linspace(0,1,max(2,int(np.ceil(distance[-1]/spacing))+1))
            selected=np.column_stack([locations**j for j in range(degree+1)])@coefficients
            result=dict(uv=selected.tolist(),closed=False,vertex_count=len(selected),
                measurement_origin='current_native_joint_smooth_material_curve_fit',
                representative_points_share_line_estimate=True,pixels_measured_in_this_frame=False,
                source_evidence_measured_in_this_frame=True,
                source_native_readout_uv=np.round(p,3).tolist(),source_native_transition_iqr_px=np.round(width,3).tolist(),
                source_curve_degree=degree,source_curve_fit_max_residual_px=float(np.max(np.linalg.norm(design@coefficients-p,axis=1))),source_query_count=len(run))
        elif straight:
            line=cv2.fitLine(p.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel();tangent=line[:2];normal=np.array([-tangent[1],tangent[0]])
            residual=(p-line[2:])@normal;good=np.abs(residual)<max(3.,np.median(width)*1.5)
            p=p[good];width=width[good];strength=strength[good]
            if len(p)<2:continue
            # Source line representatives share the current source fit. The
            # actual readouts and residual remain explicit, with no world fit.
            line=cv2.fitLine(p.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel();tangent=line[:2]
            station=(p-line[2:])@tangent
            distances=np.linspace(station[0],station[-1],max(2,int(np.ceil(abs(station[-1]-station[0])/spacing))+1))
            selected=line[2:]+distances[:,None]*tangent
            result=dict(uv=selected.tolist(),closed=False,vertex_count=len(selected),
                measurement_origin='current_native_joint_straight_material_edge_fit',
                representative_points_share_line_estimate=True,pixels_measured_in_this_frame=False,
                source_evidence_measured_in_this_frame=True,
                source_native_readout_uv=np.round(p,3).tolist(),source_native_transition_iqr_px=np.round(width,3).tolist(),
                source_line_fit_max_perpendicular_px=float(np.max(np.abs((p-line[2:])@np.array([-tangent[1],tangent[0]])))),
                source_query_count=len(run))
        else:
            # Simplification is in current native pixels. Keep glyph corners
            # and distribute samples along longer arcs without dense glyphs.
            epsilon=max(2.0,float(np.median(width))*.60)
            approx=cv2.approxPolyDP(p.astype('float32')[:,None],epsilon,cyclic)[:,0]
            chosen=sorted({int(np.argmin(np.linalg.norm(p-q,axis=1)))for q in approx})
            if not cyclic:chosen=sorted(set(chosen+[0,len(p)-1]))
            d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            chosen=sorted(set(chosen+[int(np.argmin(abs(d-q)))for q in np.arange(0,d[-1],spacing)]))
            selected=p[chosen];result=dict(uv=selected.tolist(),closed=cyclic,vertex_count=len(selected),
                transition_iqr_px=width[chosen].tolist(),localization_sigma_px=(width[chosen]/1.349).tolist(),
                measurement_origin='current_native_material_normal_half_transition',pixels_measured_in_this_frame=True,
                source_query_count=len(run),native_simplification_tolerance_px=epsilon)
        if target_id:
            result['point_bindings']=[dict(kind='edge_sample',physical_target=target_id,
                required_residual='point_to_named_finite_model_edge',measurement_origin=result['measurement_origin'])for _ in result['uv']]
        output.append(result)
    return output


def visible_projected_runs(camera,world):
    uv,depth=project_fast(camera,world)
    # Dense world engineering sampling only navigates a finite source path;
    # central front rays prevent wrapped behind-camera ads appearing onscreen.
    good=(depth>0)&np.isfinite(uv).all(1);runs=[];current=[]
    for i,(a,b) in enumerate(zip(uv[:-1],uv[1:])):
        if not good[i:i+2].all():
            if len(current)>=2:runs.append(np.asarray(current))
            current=[];continue
        direction=b-a;lo,hi=0.,1.
        for axis,minimum,maximum in ((0,-20.,1939.),(1,-20.,1099.)):
            if abs(direction[axis])<1e-10:
                if not minimum<=a[axis]<=maximum:hi=-1.;break
            else:
                t1=(minimum-a[axis])/direction[axis];t2=(maximum-a[axis])/direction[axis]
                lo=max(lo,min(t1,t2));hi=min(hi,max(t1,t2))
        if hi<lo:
            if len(current)>=2:runs.append(np.asarray(current))
            current=[];continue
        aa=a+direction*lo;bb=a+direction*hi
        if current and np.linalg.norm(current[-1]-aa)>3:
            if len(current)>=2:runs.append(np.asarray(current))
            current=[]
        if not current:current.append(aa)
        current.append(bb)
    if len(current)>=2:runs.append(np.asarray(current))
    return runs


def refined_ad_mapping(features,camera,scene,registry,number):
    ids=[f'Ad{number}.{k}'for k in ('left_top','right_top','right_bottom','left_bottom')]
    world=np.concatenate([feature_world(scene,registry['targets'][t])for t in ids]);quad,depth=project_fast(camera,world)
    if (depth<=0).any():return None
    # A same-frame source edge fit refines the material search chart. Corner
    # positions outside image are navigation extrapolations, never observations.
    by={f['id']:f for f in features};lines={}
    for role in ('top_edge','bottom_edge','left_vertical','right_vertical'):
        f=by.get(f'Ad{number}.{role}')
        if not f:continue
        p=np.vstack([s['uv']for s in f['segments']]);line=cv2.fitLine(p.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
        lines[role]=line
    top=lines.get('top_edge');bottom=lines.get('bottom_edge');left=lines.get('left_vertical');right=lines.get('right_vertical')
    for i,(vertical,horizontal) in enumerate(((left,top),(right,top),(right,bottom),(left,bottom))):
        if horizontal is not None:
            v=horizontal[:2];n=np.array([-v[1],v[0]]);quad[i]-=((quad[i]-horizontal[2:])@n)*n
        if vertical is not None and horizontal is not None:
            try:
                t=np.linalg.solve(np.column_stack((vertical[:2],-horizontal[:2])),horizontal[2:]-vertical[2:])[0]
                quad[i]=vertical[2:]+t*vertical[:2]
            except np.linalg.LinAlgError:pass
    if len(lines)<2:return None
    return cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),quad.astype('float32'))


def extract_frame(index,record,camera,scene,registry,templates,*,overlay=False):
    raw=cv2.imread(record['image_path']);mask=cv2.imread(record['mask_path'],0)
    if mask is None:raise ValueError('Current source person exclusion is not ready')
    mask=cv2.dilate((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
    # Broadcast overlays never become physical board entities.
    mask[:120,:570]=True;mask[945:,1660:]=True
    features=[];diagnostics=[]
    targets=registry['targets']
    for tid,target in targets.items():
        kind=target['kind']
        if kind not in ('board_edge','ad_edge'):continue
        if kind=='board_edge' and target.get('part') is None:continue
        if kind=='ad_edge' and int(target['advertisement_id'])>13:continue
        world=feature_world(scene,target,templates)
        all_segments=[]
        for projected in visible_projected_runs(camera,world):
            path=path_stations(projected,12)
            if np.linalg.norm(path[-1]-path[0])<10:continue
            if kind=='board_edge':
                weights=[1,-.5,-.5]if target['role']=='top'else[-1,.5,.5]
                gradient_direction=[0,1]if target['role']=='top'else[0,-1];mode=None
            else:
                number=int(target['advertisement_id']);weights=([-1,.5,.5]if number==6 else [1,-.5,-.5]if number in (3,7)else [.5,-1,.5]if number in (2,4)else[1,-.5,-.5])
                gradient_direction={'top_edge':[0,1],'bottom_edge':[0,-1],'left_vertical':[1,0],'right_vertical':[-1,0]}[target['role']];mode='chroma'
            rows,_=profile_read(raw,mask,path,weights,36 if kind=='board_edge' else 24,
                maximum_blur=18.,minimum_amplitude=10.,gradient_direction=gradient_direction,signal_mode=mode)
            straight=(kind=='board_edge' and target.get('part')in ('upper_long_side','lower_long_side','left_end','right_end'))or(kind=='ad_edge'and (int(target['advertisement_id'])in (3,4,5,6)or 'vertical'in target['role']))
            all_segments+=segment_records(rows,spacing=200,straight=straight,smooth_curve=not straight,target_id=tid)
        if not all_segments:continue
        if kind=='board_edge':fid=('board_top'if target['role']=='top'else'board_contact')+'.'+str(target['part']);layer='board_top'if target['role']=='top'else'board_contact'
        else:fid=tid;layer='ad2_border'
        features.append(dict(id=fid,label_zh=tid,physical_surface='board',target_id=tid,world_curve=tid,
            review_layer=layer,segments=all_segments,allowed_for_fit=False))
    # One physical vertical material has one current-source line. Short
    # diagonal chroma fragments at a blurred/rounded printed join do not form
    # separate model sides. All visible runs use the jointly read straight side.
    for feature in features:
        if not feature['id'].startswith('Ad') or 'vertical'not in feature['id']:continue
        evidence=np.vstack([s.get('source_native_readout_uv',s['uv'])for s in feature['segments']])
        if len(evidence)<5:continue
        line=cv2.fitLine(evidence.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float)
        perpendicular=np.array([-line[1],line[0]])
        errors=np.abs((evidence-line[2:])@perpendicular)
        accepted=evidence[errors<3.5]
        if len(accepted)<5:continue
        line=cv2.fitLine(accepted.astype('float32'),cv2.DIST_HUBER,0,.001,.001).ravel().astype(float);perpendicular=np.array([-line[1],line[0]])
        replacement=[]
        for segment in feature['segments']:
            native=np.asarray(segment.get('source_native_readout_uv',segment['uv']),float)
            good=np.abs((native-line[2:])@perpendicular)<3.5;native=native[good]
            if len(native)<2 or np.linalg.norm(native[-1]-native[0])<8:continue
            stations=(native[[0,-1]]-line[2:])@line[:2]
            locations=np.linspace(stations[0],stations[-1],max(2,int(np.ceil(abs(stations[1]-stations[0])/200))+1))
            uv=line[2:]+locations[:,None]*line[:2]
            segment.update(uv=uv.tolist(),vertex_count=len(uv),source_native_readout_uv=native.tolist(),
                point_bindings=[dict(kind='edge_sample',physical_target=feature['target_id'],required_residual='point_to_named_finite_model_edge',measurement_origin='current_source_joint_vertical_side_line')for _ in uv],
                shared_current_source_vertical_line=line.tolist(),printed_corner_radius_not_model_feature=True)
            replacement.append(segment)
        feature['segments']=replacement
    features=[f for f in features if f['segments']]
    # Source corners are finite same-source intersections, emitted once at the
    # relevant horizontal edge endpoint only when both materials were measured.
    for number in range(1,8):
        by={f['id']:f for f in features}
        mapping=refined_ad_mapping(features,camera,scene,registry,number)
        if mapping is None:continue
        quad=transform([[0,0],[1,0],[1,1],[0,1]],mapping)
        for point,side,level in zip(quad,['left','right','right','left'],['top','top','bottom','bottom']):
            fid=f'Ad{number}.{level}_edge';vertical=f'Ad{number}.{side}_vertical';f=by.get(fid)
            if not f or vertical not in by:continue
            x,y=np.rint(point).astype(int)
            if not(4<=x<1916 and 4<=y<1076)or mask[y-3:y+4,x-3:x+4].any():continue
            # Only replace an adjacent actual readout. No disconnected edge
            # endpoint or clipped image boundary is promoted to a corner.
            candidates=[(np.linalg.norm(np.asarray(s['uv'][j])-point),s,j)for s in f['segments']for j in (0,len(s['uv'])-1)]
            distance,s,j=min(candidates,key=lambda x:x[0])
            if distance>24:continue
            s['uv'][j]=point.tolist();s['point_bindings'][j]=dict(kind='corner',physical_target=f'Ad{number}.{side}_{level}',required_residual='point_to_corresponding_model_corner_2d',measurement_origin='current_source_measured_adjacent_edge_intersection')
            s.setdefault('corner_vertex_indices',{})[side+'_'+level]=j
    mappings={n:refined_ad_mapping(features,camera,scene,registry,n)for n in range(1,8)}
    for tid,target in targets.items():
        if target['kind']!='ad_print' or int(target['advertisement_id'])>13:continue
        n=int(target['advertisement_id']);world=feature_world(scene,target,templates)
        if target.get('geometry_source')in ('scene_existing_external_contour','scene_existing_board_pattern','scene_shield_anchor_material_chart'):
            paths=visible_projected_runs(camera,world);weights=[1/3,1/3,1/3];closed=True
        else:
            t=templates.get('advertisements',{}).get(str(n),templates);contour=t['contours'][target['contour_key']]
            mapping=mappings.get(n)
            if mapping is None:continue
            p=transform(contour['material_uv'],mapping)
            if n==2:weights=[0,1,0]
            else:weights=contour.get('source_signal_BGR_weights',[1,1,1])
            closed=bool(contour['closed']);paths=[p]
        all_segments=[]
        for p in paths:
            if len(p)<3:continue
            if closed and np.linalg.norm(p[-1]-p[0])>.1:p=np.vstack([p,p[:1]])
            p=path_stations(p,3.5)
            rows,_=profile_read(raw,mask,p,weights,10,maximum_blur=6.,minimum_amplitude=24.,navigation_tolerance=6.5,require_clear_end_plateaus=True)
            all_segments+=segment_records(rows,spacing=45,closed=closed,minimum_span=8.)
        if not all_segments:continue
        features.append(dict(id=tid,target_id=tid,label=target['contour_key'],label_zh=tid,physical_surface='board',review_layer='ad2_print',
            world_correspondence_status='named_same_physical_print_component',segments=all_segments,allowed_for_fit=False,
            current_source_only_readout=True,model_or_template_generated_observation=False))
    for feature in features:
        if '.print.'in feature['id']:retain_print_band_segments(features,feature)
    features=[f for f in features if f['segments']]
    count=sum(len(s['uv'])for f in features for s in f['segments'])
    # Preserve every retained material, then reduce only redundant measured
    # glyph samples if necessary. Global hard ceiling is enforced by merger.
    if count>300:
        from .single_frame_source_review import compact_review_points
        glyphs=[f for f in features if '.print.'in f['id']]
        board_count=sum(len(s['uv'])for f in features if '.print.'not in f['id']for s in f['segments'])
        try:compact_review_points(glyphs,max(1,300-board_count))
        except ValueError:pass
    count=sum(len(s['uv'])for f in features for s in f['segments'])
    for f in features:
        cleaned=[]
        for s in f['segments']:
            p=np.asarray(s['uv']);good=(p[:,0]>=0)&(p[:,0]<=1919)&(p[:,1]>=0)&(p[:,1]<=1079)
            xy=np.rint(np.clip(p,[0,0],[1919,1079])).astype(int);good&=~mask[xy[:,1],xy[:,0]]
            keep=np.flatnonzero(good).tolist()
            if len(keep)<2:continue
            if not good.all():
                for key in ('uv','point_bindings','transition_iqr_px','localization_sigma_px'):
                    if key in s:s[key]=[s[key][j]for j in keep]
                if 'corner_vertex_indices'in s:s['corner_vertex_indices']={k:keep.index(v)for k,v in s['corner_vertex_indices'].items()if v in keep}
                s.update(vertex_count=len(keep),closed=False)
            cleaned.append(s)
        f['segments']=cleaned
    features=[f for f in features if f['segments']]
    count=sum(len(s['uv'])for f in features for s in f['segments'])
    source=deepcopy(record['source']);source.update(mask_path=record['mask_path'],mask_sha256=sha(record['mask_path']),source_points_extracted=True,optimization_performed=False)
    packet=dict(schema='rinkspace.single-frame-source-review.v1',source=source,source_pts=source['native_pts'],
        programme_frame_index_1based=index,batch_sequence_index_1based=index,frame_count=1,features=features,point_count=count,
        status='board_print_current_native_candidate',user_accepted=False,user_hard_point_ceiling=500,
        observation_coordinates_from_projection=False,optimization_performed=False,
        excluded_features_by_user=['blue_strip_lower_edge','yellow_strip_upper_edge','board_panel_seam'],
        unresolved_source_regions=[],source_correlated_groups=[dict(id=f['target_id']+'_current_source_shared_line',feature_id=f['id'],
            source='same_current_native_edge_joint_line_fit',representative_points_share_line_estimate=True)for f in features if any(s.get('representative_points_share_line_estimate')for s in f['segments'])],
        diagnostics=dict(board_point_count=sum(len(s['uv'])for f in features if '.print.'not in f['id']for s in f['segments']),
            printed_point_count=sum(len(s['uv'])for f in features if '.print.'in f['id']for s in f['segments']),
            material_targets=sorted({f['target_id']for f in features}),native_readout_admits_only_clear_profile_pairs=True,
            source_blur_suppresses_unresolved_glyph_detail=True))
    write(DATA/f'board_candidates/{index:06d}.json',packet)
    if overlay:
        (DATA/'board_overlays').mkdir(exist_ok=True);cv2.imwrite(str(DATA/f'board_overlays/{index:06d}.png'),native_overlay(raw,list(source_segments(packet))),[cv2.IMWRITE_PNG_COMPRESSION,1])
    return dict(index=index,points=count,board_points=packet['diagnostics']['board_point_count'],print_points=packet['diagnostics']['printed_point_count'],
        ads=sorted({int(f['target_id'].split('.')[0][2:])for f in features if f['target_id'].startswith('Ad')}),
        board_roles=sorted({f['target_id'].split('.')[1]for f in features if f['target_id'].startswith('board.')}))


def run(indices,*,overlays=False):
    cv2.setNumThreads(2)
    scene=read(MODEL/'scene.json');registry=read(MODEL/'feature_registry.json');templates=read(MODEL/'ad2_material_template.json')
    manifest=read(DATA/'frame_manifest.json');records={int(r['index']):r for r in manifest['frames']}
    a=np.load(DATA/'camera_seeds.npz');cameras={int(i):c for i,c in zip(a['programme_indices'],a['camera25'])}
    result=[]
    failures=[]
    for i in indices:
        try:row=extract_frame(i,records[i],cameras[i],scene,registry,templates,overlay=overlays or i%25==0 or i in (201,1248))
        except Exception as error:
            failures.append(dict(index=i,error=repr(error)));print(json.dumps(failures[-1]),flush=True);continue
        result.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    write(DATA/f'board_counts_{indices[0]:06d}_{indices[-1]:06d}.json',result)
    write(DATA/f'board_failures_{indices[0]:06d}_{indices[-1]:06d}.json',failures);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--indices',default='201:1248');parser.add_argument('--build-model',action='store_true');parser.add_argument('--overlays',action='store_true')
    args=parser.parse_args()
    if args.build_model:print(json.dumps(build_model(),ensure_ascii=False),flush=True)
    else:
        indices=list(range(int(args.indices.split(':')[0]),int(args.indices.split(':')[1])+1))if':'in args.indices else list(map(int,args.indices.split(',')))
        run(indices,overlays=args.overlays)
