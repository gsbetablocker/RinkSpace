"""Qualify visible white boot outer contours from unchanged source RGB.

SAM2 supplies search objects only. Native RGB material evidence selects the
white boot; thin steel, dark mounting hardware, skin and mutual occlusion do
not become boot measurements. Blade lower edges retain separate source roles.
"""
from pathlib import Path
import argparse,json
import cv2
import numpy as np
from .rebuild_prepare import load,digest


def material_model(root):
    root=Path(root);out=root/'footwear_v2';admission=json.loads((out/'admission.json').read_text(encoding='utf8'));index=json.loads((out/'source_index.json').read_text(encoding='utf8'));by={r['pts']:r for r in index['rows']};samples=[];normalized_areas=[];body_input=load(root/'input.npz')
    for row in admission['records']:
        if row['status']!='admitted_visible_boundary':continue
        image=cv2.imread(by[row['pts']]['crop']);lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
        mask=load(row['mask_file'])['masks'][row['foot']];frame=int(np.searchsorted(body_input['pts'],row['pts']));height=body_input['source_bbox'][frame,3]-body_input['source_bbox'][frame,1];normalized_areas.append(float(mask.sum()/height**2));inside=cv2.erode(mask,np.ones((5,5),np.uint8))>0;values=lab[inside]
        if len(values):samples.append(values)
    values=np.concatenate(samples);chroma=np.linalg.norm(values[:,1:]-128,axis=1);neutral=values[chroma<28]
    bright=neutral[neutral[:,0]>=np.quantile(neutral[:,0],.55)]
    minimum_L=float(np.quantile(bright[:,0],.05)-15);maximum_C=float(np.quantile(np.linalg.norm(bright[:,1:]-128,axis=1),.95)+3)
    return dict(schema='rinkspace.source-white-boot-material.v1',minimum_L=minimum_L,maximum_chroma=maximum_C,
        source='native RGB interiors of previously source-reviewed fit masks only',source_admission_sha256=digest(out/'admission.json'),
        colour_space='OpenCV uint8 Lab',maximum_search_object_area_per_body_height_squared=float(np.quantile(normalized_areas,.95)*1.75),size_navigation='source-reviewed footwear image sizes, not fitted body projections',steel_is_not_white_boot=True,thresholds_are_source_material_navigation_not_3d_truth=True)


def prepare(root,dataset,roles=('fit',),model_path=None):
    root,dataset=Path(root),Path(dataset);cache=root/'footwear_v2';index=json.loads((dataset/'source_index.json').read_text(encoding='utf8'));by={r['pts']:r for r in index['rows']}
    model_file=Path(model_path) if model_path else dataset/'white_boot_material.json'
    if model_file.exists():model=json.loads(model_file.read_text(encoding='utf8'))
    else:
        model=material_model(root);model_file.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf8')
    body_input=load(root/'input.npz');records=[];mattes=dataset/'white_boot_mattes';mattes.mkdir(exist_ok=True)
    for row in index['rows']:
        if row['role'] not in roles or row['status']!='source_crop_available':continue
        source_file=cache/'mask_candidates'/f'{row["pts"]}.npz'
        if not source_file.exists():continue
        packet=load(source_file);masks=packet['masks'];image=cv2.imread(row['crop']);lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
        neutral=(lab[:,:,0]>=model['minimum_L'])&(np.linalg.norm(lab[:,:,1:]-128,axis=2)<=model['maximum_chroma'])
        white=[]
        for foot in range(2):
            candidate=(neutral&(masks[foot]>0)).astype(np.uint8)
            candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            count,labels,stats,centres=cv2.connectedComponentsWithStats(candidate)
            selected=np.zeros_like(candidate)
            if count>1:
                k=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);selected=(labels==k).astype(np.uint8)
            white.append(selected)
        white=np.asarray(white);intersection=np.count_nonzero(white[0]&white[1]);union=np.count_nonzero(white[0]|white[1]);iou=intersection/max(union,1)
        matte_file=mattes/f'{row["pts"]}.npz';np.savez_compressed(matte_file,masks=white,crop_xyxy=row['crop_xyxy'],pts=row['pts'])
        decisions=[]
        for foot in range(2):
            record=dict(pts=row['pts'],foot=foot,source_role=row['role'],observed_kind='boot_external_contour',
                mask_file=str(source_file),white_material_mask_file=str(matte_file),source_RGB_crop_sha256=row['crop_sha256'],
                material_model_sha256=digest(model_file),reviewer='source_RGB_material_and_instance_checks',human_confirmed=False,
                body_model_projection_used_as_measurement=False,reverse_boundary=False,partial=True,sigma_px=3.,mask_pair_white_iou=iou)
            native_frame=int(np.searchsorted(body_input['pts'],row['pts']));body_height=body_input['source_bbox'][native_frame,3]-body_input['source_bbox'][native_frame,1]
            parent_normalized_area=float(masks[foot].sum()/body_height**2)
            record['search_object_area_per_body_height_squared']=parent_normalized_area
            if parent_normalized_area>model['maximum_search_object_area_per_body_height_squared']:
                record.update(status='excluded_search_object_spans_background',reason='source object is too large for reviewed footwear appearance; no model-fit error used');records.append(record);continue
            if iou>.4:record.update(status='excluded_instance_identity_merged');records.append(record);continue
            area=int(white[foot].sum());parent_area=int(masks[foot].sum());fraction=area/max(parent_area,1)
            if fraction<.15 or area<20:record.update(status='excluded_no_resolved_white_boot',white_fraction=fraction);records.append(record);continue
            yy,xx=np.nonzero(white[foot]);box=np.array([xx.min(),yy.min(),xx.max(),yy.max()]);offset=np.array(row['crop_xyxy'][:2]);ankle=np.array(row['ankle_source_uv'][foot])-offset
            height=max(row['search_boxes_uv'][foot][3]-row['search_boxes_uv'][foot][1],1)/.245
            outside=np.maximum(box[:2]-ankle,0)+np.minimum(box[2:]-ankle,0)
            if np.linalg.norm(outside)>.10*height or max(box[2:]-box[:2])>.38*height:
                record.update(status='excluded_material_not_ankle_bound',white_box=box.tolist());records.append(record);continue
            contours,_=cv2.findContours(masks[foot],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not contours:record['status']='excluded_empty_sam_object';records.append(record);continue
            contour=max(contours,key=cv2.contourArea)[:,0];near_white=cv2.distanceTransform((1-white[foot]).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            other=cv2.dilate(white[1-foot],np.ones((5,5),np.uint8))>0
            # Keep actual proposed source outer pixels only where the inside
            # is supported by white boot material. Dark/steel boundaries and
            # inferred hidden edges are not completed.
            supported=(near_white[contour[:,1],contour[:,0]]<=3)&(~other[contour[:,1],contour[:,0]])
            # A SAM boundary on uniform ice is not a source edge. Read an
            # actual native-RGB normal profile and refine only within 3 px.
            tangent=np.roll(contour,-3,axis=0)-np.roll(contour,3,axis=0)
            normal=np.c_[-tangent[:,1],tangent[:,0]].astype(float)
            normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-12)[:,None]
            centre=contour.astype(float);offsets=np.arange(-3,4,dtype=float)
            gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.7)
            def sample(xy):
                return cv2.remap(gray,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            locations=centre[:,None]+normal[:,None]*offsets[None,:,None]
            strength=abs(sample(locations+normal[:,None]*1.5)-sample(locations-normal[:,None]*1.5))
            best=strength.argmax(1);contrast=strength[np.arange(len(contour)),best]
            refined=locations[np.arange(len(contour)),best]
            supported&=(contrast>=8)&(centre[:,0]>=5)&(centre[:,0]<image.shape[1]-5)&(centre[:,1]>=5)&(centre[:,1]<image.shape[0]-5)
            visible=refined[supported]
            if len(visible)<8:record['status']='excluded_insufficient_visible_white_outer_boundary';records.append(record);continue
            selection=np.linspace(0,len(visible)-1,48).round().astype(int);chosen=visible[selection]
            record.update(status='admitted_visible_boundary',kind='RGB_qualified_visible_white_upper_boot',source_boundary_crop_uv=chosen.tolist(),
                source_boundary_native_uv=(chosen+offset).tolist(),white_fraction=fraction,visible_candidate_pixels=len(visible),unseen_contours_completed=False,
                source_boundary_definition='native RGB normal-profile edge on visible white upper boot; dark/steel and internal material edges excluded',
                minimum_profile_contrast_gray=8,profile_search_px=3,profile_contrast=contrast[supported][selection].tolist(),
                original_SAM_guide_crop_uv=centre[supported][selection].tolist())
            records.append(record)
        # QC retains rejected candidates and records; this canvas never
        # contains the fitted equipment projection.
        shown=image.copy()
        for foot,color in enumerate([(70,230,80),(255,150,40)]):
            cs,_=cv2.findContours(white[foot],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(shown,cs,-1,color,1,cv2.LINE_AA)
        for rec in records[-2:]:
            if rec['status']=='admitted_visible_boundary':
                for xy in rec['source_boundary_crop_uv']:cv2.circle(shown,tuple(np.rint(xy).astype(int)),1,(30,30,255),-1,cv2.LINE_AA)
        cv2.imwrite(str(mattes/f'{row["pts"]}_qa.jpg'),shown,[cv2.IMWRITE_JPEG_QUALITY,95])
    result=dict(schema='rinkspace.source-white-boot-admission.v1',source_index_sha256=digest(dataset/'source_index.json'),records=records,
        source_material_model=model,role=list(roles),source_review_status='machine material qualification with reference-source visual review; no user confirmation',
        scope='native source white boot outer pixels only; lower blade observations are separate',independent_3d_truth=False)
    name='admission.json' if roles==('fit',) else 'check_admission.json'
    (dataset/name).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(candidates=len(records),admitted=sum(r['status']=='admitted_visible_boundary' for r in records),model=model)),flush=True)

def qa_montage(dataset,pts_values):
    dataset=Path(dataset);tiles=[];records=json.loads((dataset/'admission.json').read_text(encoding='utf8'))['records']
    for pts in pts_values:
        file=dataset/'white_boot_mattes'/f'{pts}_qa.jpg'
        if not file.exists():continue
        image=cv2.imread(str(file));box=np.full((320,420,3),24,np.uint8);scale=min(410/image.shape[1],265/image.shape[0]);image=cv2.resize(image,None,fx=scale,fy=scale)
        box[45:45+image.shape[0],5:5+image.shape[1]]=image;selected=[r for r in records if r['pts']==pts]
        label=' '.join(('LR'[r['foot']]+(' OK' if r['status']=='admitted_visible_boundary' else ' unknown')) for r in selected)
        cv2.putText(box,f'{pts/12800:.3f}s | '+label,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA);tiles.append(box)
    while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
    cv2.imwrite(str(dataset/'source_qa_montage.png'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--roles',nargs='+',default=['fit']);ap.add_argument('--material-model',type=Path);ap.add_argument('--qa-pts',nargs='+',type=int);a=ap.parse_args()
    if a.qa_pts:qa_montage(a.dataset,a.qa_pts)
    else:prepare(a.root,a.dataset,tuple(a.roles),a.material_model)
