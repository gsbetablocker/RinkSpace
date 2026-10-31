"""Explicit source-only review and contour-role preparation for two boots."""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
from .rebuild_prepare import load,digest

# Per-source visual decisions, not selected by model fit error. Boot contours
# and full equipment contours have distinct expected geometry. Unknown and
# crossing identities are preserved and do not enter calibration.



def prepare(root,review_path):
    root=Path(root);review=json.loads(Path(review_path).read_text(encoding='utf8'));out=root/'footwear_v2';index=json.loads((out/'source_index.json').read_text(encoding='utf8'));records=[]
    for row in index['rows']:
        if row['role']!='fit' or row['status']!='source_crop_available':continue
        packet=load(out/'mask_candidates'/f'{row["pts"]}.npz');masks=packet['masks']
        for foot,kind in enumerate(review['mask_review_by_pts'][str(row['pts'])]):
            decision=dict(pts=row['pts'],foot=foot,source_role='fit',reviewer='assistant_source_visual_review',review_image=str(out/('mask_atlas_'+str(len(records)//24)+'.jpg')),
                human_confirmed=False,kind=kind,body_projection_used_to_qualify=False,mask_file=str(out/'mask_candidates'/f'{row["pts"]}.npz'))
            if kind=='unknown':decision.update(status='excluded_identity_occlusion_or_boundary_ambiguity',reason='source mask can merge legs/boots or the visible equipment boundary is not resolved');records.append(decision);continue
            mask=masks[foot].copy();other=masks[1-foot]
            cc,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not cc:decision.update(status='excluded_empty_mask');records.append(decision);continue
            contour=max(cc,key=cv2.contourArea)[:,0]
            # Only real visible contours outside the other shoe may constrain
            # the model. Do not complete the hidden contour by subtraction.
            near_other=cv2.dilate(other,np.ones((5,5),np.uint8))>0
            points=contour[~near_other[contour[:,1],contour[:,0]]]
            if len(points)<8:decision.update(status='excluded_no_resolved_contour');records.append(decision);continue
            chosen=points[np.linspace(0,len(points)-1,48).round().astype(int)]
            decision.update(status='admitted_visible_boundary',observed_kind='full_equipment_external_contour' if kind.startswith('full') else 'boot_external_contour',
                source_boundary_crop_uv=chosen.tolist(),source_boundary_native_uv=(chosen+np.array(row['crop_xyxy'][:2])).tolist(),
                sigma_px=4. if 'blur' in kind else 2.5,partial=('partial' in kind),contour_source='SAM2 proposed and source visually reviewed; approximate pixel uncertainty, not measured 3D',
                mask_overlap_pixels=int(np.count_nonzero(masks[0]&masks[1])),unseen_contours_completed=False)
            records.append(decision)
    report=dict(schema='rinkspace.two-boot-source-admission.v1',source_index_sha256=digest(out/'source_index.json'),source_time_base='1/12800',
        records=records,source_review='fit frames only, check windows remain unseen',confirmed_by_user=False,
        equipment_dimensions_measured=False,scope='development footwear pixel observations, not ground truth 3D contact')
    (out/'admission.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Source pixels with native coordinate grid for precise blade annotation.
    by={r['pts']:r for r in index['rows']}
    for pts in review['blade_grid_pts']:
        row=by[pts];image=cv2.imread(row['crop']);offset=np.array(row['crop_xyxy'][:2]);h,w=image.shape[:2];factor=3;shown=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
        for x in range(((int(offset[0])+19)//20)*20,int(offset[0])+w,20):
            xx=(x-int(offset[0]))*factor;cv2.line(shown,(xx,0),(xx,h*factor-1),(170,150,90),1);cv2.putText(shown,str(x),(xx+2,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        for y in range(((int(offset[1])+19)//20)*20,int(offset[1])+h,20):
            yy=(y-int(offset[1]))*factor;cv2.line(shown,(0,yy),(w*factor-1,yy),(170,150,90),1);cv2.putText(shown,str(y),(2,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        cv2.imwrite(str(out/f'blade_grid_{pts}.png'),shown)
    print(json.dumps(dict(foot_candidates=len(records),admitted=sum(r['status']=='admitted_visible_boundary' for r in records),excluded=sum(r['status']!='admitted_visible_boundary' for r in records))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();prepare(a.root,a.review)
