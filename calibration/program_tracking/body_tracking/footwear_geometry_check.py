"""Check unconsumed blade pixels after boot-only pose inference."""
from pathlib import Path
import argparse,json
import numpy as np
import cv2
from .rebuild_prepare import load,digest
from .footwear_review import statistics
from .footwear_fit import sample_polyline,point_polyline_distance,hull


def check(root,review_path):
    root=Path(root);out=root/'footwear_v2';review=json.loads(Path(review_path).read_text(encoding='utf8'));p=load(root/'input.npz')
    old=load(root/'pose/result.npz');dense=load(out/'dense/articulated/result.npz');observed=load(out/'pose_transfer/observed_pose/result.npz')
    variants={'old_B':old['blade_uv'],'dense_blind_pose':dense['equipment_uv'][:,:,14:31],'frozen_geometry_observed_pose':observed['equipment_uv'][:,:,14:31]}
    report=dict(schema='rinkspace.fixed-equipment-blade-check-result.v1',source_check_sha256=digest(review_path),models={},
        source_curve_count=len(review['blade_curves']),source_annotated_vertices=sum(len(c['uv']) for c in review['blade_curves']),
        same_curve_resampling_is_not_independent_evidence=True,equipment_geometry_refitted=False,
        white_boot_pixels_consumed_for_pose=True,blade_check_pixels_consumed_for_pose_or_geometry=False,
        independent_external_3d_truth=False,pelvis_frozen=True)
    for name,uv in variants.items():
        values=[];rows=[]
        for c in review['blade_curves']:
            i=int(np.searchsorted(p['pts'],c['pts']));observations=sample_polyline(c['uv'],24);distance=point_polyline_distance(observations,uv[i,c['foot']])
            values.extend(distance.tolist());rows.append(dict(pts=c['pts'],source_seconds=c['pts']/12800,foot=c['foot'],**statistics(distance)))
        report['models'][name]=dict(raw_pixel_distance=statistics(values),per_curve=rows)
    report['geometry_unchanged_max']=float(np.max(abs(dense['x'][:14]-observed['x'][:14])))
    report['pelvis_unchanged_max_conditional_m']=float(np.max(abs(observed['P_world_m']-old['P_world_m'])))
    # Retain height as a diagnostic, not a new contact observation.
    evidence=json.loads((root/'contact_evidence.json').read_text(encoding='utf8'));heights=[]
    for c in evidence['records']:
        i=int(np.searchsorted(p['pts'],c['pts']));heights.append(dict(pts=c['pts'],foot=c['foot'],
            old_B_height=float(old['blade_world_m'][i,c['foot'],:,2].min()),
            calibrated_height=float(observed['nominal_blade_minimum_world_m'][i,c['foot'],2]),
            status='apparent near-ice source only; neither equipment nor world height measured'))
    report['apparent_support_height_diagnostics']=heights
    (out/'pose_transfer/blade_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    # Compare actual new source pixels with the frozen shape. No ice-height
    # residual or post-display shift is applied to these images.
    index=json.loads((out/'dense/source_index.json').read_text(encoding='utf8'));by={r['pts']:r for r in index['rows']}
    for pts in sorted(set(c['pts'] for c in review['blade_curves'])):
        row=by[pts];image=cv2.imread(row['crop']);offset=np.array(row['crop_xyxy'][:2]);i=int(np.searchsorted(p['pts'],pts));h,w=image.shape[:2];tiles=[]
        for name,color in [('old_B',(50,180,255)),('dense_blind_pose',(240,180,80)),('frozen_geometry_observed_pose',(60,230,90))]:
            shown=image.copy()
            for foot in range(2):
                cv2.polylines(shown,[np.rint(variants[name][i,foot]-offset).astype(int)],False,color,2,cv2.LINE_AA)
                if name!='old_B':
                    z=dense if name=='dense_blind_pose' else observed
                    cv2.polylines(shown,[np.rint(hull(z['equipment_uv'][i,foot,:14]-offset)).astype(int)],True,color,1,cv2.LINE_AA)
            for c in review['blade_curves']:
                if c['pts']==pts:
                    curve=np.asarray(c['uv'])-offset;cv2.polylines(shown,[np.rint(curve).astype(int)],False,(20,20,240),1,cv2.LINE_AA)
                    for point in curve:cv2.circle(shown,tuple(np.rint(point).astype(int)),2,(20,20,240),-1,cv2.LINE_AA)
            canvas=np.full((720,600,3),22,np.uint8);factor=min(580/w,620/h);im=cv2.resize(shown,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC);canvas[80:80+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(canvas,name,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1,cv2.LINE_AA);cv2.putText(canvas,f'SOURCE {pts/12800:.3f}s | pelvis fixed',(12,54),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Red: blade pixels excluded from fitting',(12,704),cv2.FONT_HERSHEY_SIMPLEX,.4,(50,60,240),1,cv2.LINE_AA);tiles.append(canvas)
        cv2.imwrite(str(out/'pose_transfer'/f'check_{pts}.png'),np.hstack(tiles))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();check(a.root,a.review)
