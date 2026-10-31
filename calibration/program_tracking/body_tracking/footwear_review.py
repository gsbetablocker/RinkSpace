"""Source-only check observations and frozen-calibration image review."""
from pathlib import Path
import argparse,json
import cv2
import numpy as np
from .rebuild_prepare import load,digest
from .footwear_fit import FootwearProblem,hull,sample_polyline,point_polyline_distance,bilinear
from .footwear_geometry import INITIAL


def pages(root,role='check',dataset=None):
    cache=Path(root)/'footwear_v2';out=Path(dataset) if dataset else cache;index=json.loads((out/'source_index.json').read_text(encoding='utf8'));rows=[r for r in index['rows'] if r['role']==role and r['status']=='source_crop_available']
    for page,start in enumerate(range(0,len(rows),21)):
        tiles=[]
        for row in rows[start:start+21]:
            qa=out/'white_boot_mattes'/(str(row['pts'])+'_qa.jpg');qa=qa if qa.exists() else cache/'mask_candidates'/(str(row['pts'])+'_qa.jpg');image=cv2.imread(str(qa));box=np.full((240,360,3),24,np.uint8)
            scale=min(350/image.shape[1],205/image.shape[0]);image=cv2.resize(image,None,fx=scale,fy=scale)
            box[30:30+image.shape[0],5:5+image.shape[1]]=image;cv2.putText(box,str(row['source_seconds'])+'s / '+str(row['pts']),(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA);tiles.append(box)
        while len(tiles)%3:tiles.append(np.full_like(tiles[0],24))
        cv2.imwrite(str(out/(role+'_atlas_'+str(page)+'.jpg')),np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]),[cv2.IMWRITE_JPEG_QUALITY,95])
    print(dict(role=role,frames=len(rows)),flush=True)


def blade_grids(root,pts_values,dataset=None):
    out=Path(dataset) if dataset else Path(root)/'footwear_v2';index=json.loads((out/'source_index.json').read_text(encoding='utf8'));by={r['pts']:r for r in index['rows']}
    for pts in pts_values:
        row=by[int(pts)];image=cv2.imread(row['crop']);offset=np.array(row['crop_xyxy'][:2]);h,w=image.shape[:2];factor=3
        shown=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
        for x in range(((int(offset[0])+19)//20)*20,int(offset[0])+w,20):
            xx=(x-int(offset[0]))*factor;cv2.line(shown,(xx,0),(xx,h*factor-1),(170,150,90),1);cv2.putText(shown,str(x),(xx+2,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        for y in range(((int(offset[1])+19)//20)*20,int(offset[1])+h,20):
            yy=(y-int(offset[1]))*factor;cv2.line(shown,(0,yy),(w*factor-1,yy),(170,150,90),1);cv2.putText(shown,str(y),(2,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        cv2.imwrite(str(out/f'check_blade_grid_{pts}.png'),shown)


def admitted_checks(root,decisions_path,dataset=None):
    out=Path(root)/'footwear_v2';index=json.loads((out/'source_index.json').read_text(encoding='utf8'));decisions=json.loads(Path(decisions_path).read_text(encoding='utf8'));records=[]
    if 'records' in decisions:return decisions
    for row in index['rows']:
        if row['role']!='check' or row['status']!='source_crop_available':continue
        packet=load(out/'mask_candidates'/f'{row["pts"]}.npz');masks=packet['masks']
        for foot,kind in enumerate(decisions['mask_review_by_pts'][str(row['pts'])]):
            record=dict(pts=row['pts'],foot=foot,source_role='check',kind=kind,human_confirmed=False,reviewer='assistant_source_visual_review_without_model_overlay',mask_file=str(out/'mask_candidates'/f'{row["pts"]}.npz'))
            if kind=='unknown':record['status']='excluded_occlusion_identity_or_boundary';records.append(record);continue
            contours,_=cv2.findContours(masks[foot],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not contours:record['status']='excluded_empty_mask';records.append(record);continue
            contour=max(contours,key=cv2.contourArea)[:,0];other=cv2.dilate(masks[1-foot],np.ones((5,5),np.uint8))>0
            visible=contour[~other[contour[:,1],contour[:,0]]]
            if len(visible)<8:record['status']='excluded_no_resolved_contour';records.append(record);continue
            selected=visible[np.linspace(0,len(visible)-1,48).round().astype(int)]
            record.update(status='admitted_visible_boundary',observed_kind='full_equipment_external_contour' if kind.startswith('full') else 'boot_external_contour',
                source_boundary_crop_uv=selected.tolist(),source_boundary_native_uv=(selected+row['crop_xyxy'][:2]).tolist(),sigma_px=4. if 'blur' in kind else 2.5,
                partial=('partial' in kind),mask_overlap_pixels=int(np.count_nonzero(masks[0]&masks[1])),unseen_contours_completed=False)
            records.append(record)
    result=dict(schema='rinkspace.two-boot-check-observations.v1',source_index_sha256=digest(out/'source_index.json'),decisions_sha256=digest(decisions_path),
        model_selection_freeze_sha256=digest(out/'model_selection_freeze.json'),records=records,used_for_fitting=False,independent_3d_truth=False)
    (out/'check_admission.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


def statistics(values):
    values=np.asarray(values,float).ravel()
    return dict(n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max())) if len(values) else dict(n=0)


def audit(root,review_path,checks_path,dataset=None):
    root=Path(root);cache=root/'footwear_v2';out=Path(dataset) if dataset else cache;problem=FootwearProblem(root,review_path,'articulated',dataset=dataset);p=problem.p
    check=admitted_checks(root,checks_path,dataset=dataset);check_details=json.loads(Path(checks_path).read_text(encoding='utf8'));records=problem.admission['records']+check['records'];bypts={r['pts']:r for r in problem.index['rows']}
    frames=np.unique([np.searchsorted(p['pts'],r['pts']) for r in records if r['status']=='admitted_visible_boundary']);lookup={int(p['pts'][f]):i for i,f in enumerate(frames)}
    variants={};initial=np.r_[INITIAL,np.zeros(problem.size-len(INITIAL))]
    variants['initial_rigid_boot']=problem.state(initial,frames)
    shared=load((out if (out/'shared/result.npz').exists() else cache)/'shared/result.npz');xs=initial.copy();xs[:len(INITIAL)]=shared['x'];variants['shared']=problem.state(xs,frames)
    fitted=load(out/'articulated/result.npz');variants['articulated']=problem.state(fitted['x'],frames)
    if dataset is not None:variants['sparse_articulated']=problem.state(load(cache/'articulated/result.npz')['x'],frames)
    report=dict(schema='rinkspace.two-boot-calibration-comparison.v1',camera_scene_pelvis_frozen=True,metric_scale_status='conditional',models={},check_windows=problem.index['detailed_boot_check_windows_s'],
        check_foot_candidates=len(check['records']),check_foot_admitted=sum(r['status']=='admitted_visible_boundary' for r in check['records']),user_visually_accepted=False,independent_3d_truth=False)
    for name,state in variants.items():
        perframe=[];pools={'fit':[],'check':[]};blade=[]
        for r in records:
            if r['status']!='admitted_visible_boundary':continue
            fi=lookup[r['pts']];foot=r['foot'];indices=state['parts']['shell'] if r['observed_kind']=='boot_external_contour' else np.arange(48)
            polygon=hull(state['uv'][fi,foot,indices]);distance=point_polyline_distance(r['source_boundary_native_uv'],polygon,True)
            pools[r['source_role']].extend(distance.tolist());perframe.append(dict(pts=r['pts'],foot=foot,role=r['source_role'],kind=r['observed_kind'],**statistics(distance)))
        for r in problem.blade_rows:
            fi=lookup[r['pts']];pred=state['uv'][fi,r['foot'],state['parts']['blade_left']];observed=sample_polyline(r['uv'],24)
            blade.extend(point_polyline_distance(observed,pred).tolist())
        check_blade=[]
        for rr in check_details.get('blade_curves',[]):
            ii=lookup[rr['pts']];pred=state['uv'][ii,rr['foot'],state['parts']['blade_left']]
            check_blade.extend(point_polyline_distance(sample_polyline(rr['uv'],24),pred).tolist())
        report['models'][name]=dict(fit_visible_boundary_px=statistics(pools['fit']),check_visible_boundary_px=statistics(pools['check']),fit_blade_lower_edge_px=statistics(blade),check_blade_lower_edge_px=statistics(check_blade),per_foot=perframe)
    old=load(root/'pose/result.npz');oldblade=[]
    for r in problem.blade_rows:
        fi=int(np.searchsorted(p['pts'],r['pts']));oldblade.extend(point_polyline_distance(sample_polyline(r['uv'],24),old['blade_uv'][fi,r['foot']]).tolist())
    report['old_B_blade_lower_edge_px']=statistics(oldblade)
    old_check=[]
    for rr in check_details.get('blade_curves',[]):
        ii=int(np.searchsorted(p['pts'],rr['pts']));old_check.extend(point_polyline_distance(sample_polyline(rr['uv'],24),old['blade_uv'][ii,rr['foot']]).tolist())
    report['old_B_check_blade_lower_edge_px']=statistics(old_check)
    full=load(out/'articulated/result.npz');visible=~p['lower_body_inferred'];err=np.linalg.norm(full['COCO_uv']-p['source_uv'],axis=2);baseerr=np.linalg.norm(old['COCO_uv']-p['source_uv'],axis=2)
    for key,ids in [('knees',[13,14]),('upper_body',[0,1,2,3,4,5,6,7,8,9,10,11,12])]:
        mask=p['source_qualified'][:,ids]&visible[:,None];report[key+'_source_px']=dict(base_B=statistics(baseerr[:,ids][mask]),calibrated=statistics(err[:,ids][mask]))
    report['pelvis_world_difference_max_m']=float(np.max(np.abs(full['P_world_m']-old['P_world_m'])))
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='models'},ensure_ascii=False),flush=True)
    print(json.dumps({k:{key:v for key,v in m.items() if key!='per_foot'} for k,m in report['models'].items()},ensure_ascii=False),flush=True)
    return report


def images(root,review_path,dataset=None):
    root=Path(root);cache=root/'footwear_v2';out=Path(dataset) if dataset else cache;problem=FootwearProblem(root,review_path,'articulated',dataset=dataset);p=problem.p;old=load(root/'pose/result.npz')
    shared=load((out if (out/'shared/result.npz').exists() else cache)/'shared/result.npz');new=load(out/'articulated/result.npz');xs=np.r_[shared['x'],np.zeros(problem.size-len(INITIAL))]
    bypts={r['pts']:r for r in problem.index['rows']};wanted=problem.review['blade_grid_pts']
    colors=[(50,180,255),(200,160,80),(70,230,90)];labels=['B original blade proxy','shared boot calibration','articulated boot calibration']
    for pts in wanted:
        row=bypts[pts];fi=int(np.searchsorted(p['pts'],pts));raw=cv2.imread(row['crop']);offset=np.asarray(row['crop_xyxy'][:2]);h,w=raw.shape[:2]
        predictions=[None,problem.state(xs,[fi]),problem.state(new['x'],[fi])];tiles=[]
        for mode in range(3):
            shown=raw.copy();color=colors[mode]
            for foot in range(2):
                if mode==0:
                    curve=old['blade_uv'][fi,foot]-offset;cv2.polylines(shown,[np.rint(curve).astype(int)],False,color,2,cv2.LINE_AA)
                else:
                    state=predictions[mode];uv=state['uv'][0,foot]-offset
                    cv2.polylines(shown,[np.rint(hull(uv[state['parts']['shell']])).astype(int)],True,color,1,cv2.LINE_AA)
                    for part in ['blade_left','blade_right']:cv2.polylines(shown,[np.rint(uv[state['parts'][part]]).astype(int)],False,color,2,cv2.LINE_AA)
                body=old['COCO_uv'][fi] if mode==0 else predictions[mode]['body_uv'][0]
                for a,b in [(11,13),(13,15),(12,14),(14,16)]:cv2.line(shown,tuple(np.rint(body[a]-offset).astype(int)),tuple(np.rint(body[b]-offset).astype(int)),color,1,cv2.LINE_AA)
            for r in problem.review.get('blade_curves',[]):
                if r['pts']==pts:
                    curve=np.asarray(r['uv'])-offset;cv2.polylines(shown,[np.rint(curve).astype(int)],False,(40,40,240),1,cv2.LINE_AA)
                    for at in curve:cv2.circle(shown,tuple(np.rint(at).astype(int)),2,(40,40,240),-1,cv2.LINE_AA)
            canvas=np.full((740,600,3),22,np.uint8);scale=min(580/w,640/h);im=cv2.resize(shown,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC);canvas[75:75+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(canvas,labels[mode],(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1,cv2.LINE_AA)
            cv2.putText(canvas,f'SOURCE {pts/12800:.3f}s | pelvis fixed',(12,52),cv2.FONT_HERSHEY_SIMPLEX,.47,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Red: reviewed lower blade pixels',(12,724),cv2.FONT_HERSHEY_SIMPLEX,.44,(60,70,250),1,cv2.LINE_AA);tiles.append(canvas)
        cv2.imwrite(str(out/f'comparison_{pts}.png'),np.hstack(tiles))
    print(dict(images=len(wanted)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--review',type=Path);ap.add_argument('--checks',type=Path);ap.add_argument('--pages',action='store_true');ap.add_argument('--images',action='store_true');ap.add_argument('--grids',nargs='+',type=int);ap.add_argument('--dataset',type=Path);a=ap.parse_args()
    if a.grids:blade_grids(a.root,a.grids,a.dataset)
    elif a.pages:pages(a.root,dataset=a.dataset)
    elif a.images:images(a.root,a.review,a.dataset)
    else:audit(a.root,a.review,a.checks,a.dataset)
