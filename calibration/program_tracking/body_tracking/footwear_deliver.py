"""Export a source-bound footwear candidate without promoting contact truth.

All numerical products are development diagnostics. The packet retains source
roles, per-exposure inference flags and frozen-body identities. No contact or
camera update is performed by this exporter.
"""
from pathlib import Path
from collections import Counter
import argparse,json
import cv2
import numpy as np
from .rebuild_prepare import load,digest
from .footwear_geometry import INITIAL,NAMES,LEG_DOFS,decode,LOWER,UPPER
from .footwear_fit import point_polyline_distance,sample_polyline,hull
from .footwear_review import statistics


def edge_errors(rows,uv,pts):
    errors=[];details=[]
    for row in rows:
        i=int(np.searchsorted(pts,row['pts']));foot=row['foot'];distances=[]
        for key,pair in [('heel',[0,7]),('toe',[1,8])]:
            edge=uv[i,foot,pair];v=edge[1]-edge[0];point=np.asarray(row[key],float)
            a=np.clip(np.dot(point-edge[0],v)/max(float(v@v),1e-16),0,1)
            distances.append(float(np.linalg.norm(edge[0]+a*v-point)))
        errors.extend(distances);details.append(dict(pts=row['pts'],foot=foot,heel_px=distances[0],toe_px=distances[1],source_sigma_px=row['sigma_px']))
    return dict(raw_pixel_distance=statistics(errors),source_foot_records=len(rows),per_foot=details)


def curve_errors(rows,curves,pts):
    errors=[];details=[]
    for row in rows:
        i=int(np.searchsorted(pts,row['pts']));sample=sample_polyline(row['uv'],24)
        d=point_polyline_distance(sample,curves[i,row['foot']]);errors.extend(d)
        details.append(dict(pts=row['pts'],source_seconds=row['pts']/12800,foot=row['foot'],source_sigma_px=row['sigma_px'],**statistics(d)))
    return dict(raw_pixel_distance=statistics(errors),source_curves=len(rows),source_vertices=sum(len(r['uv']) for r in rows),per_curve=details,
        curve_resampling_adds_no_independent_evidence=True)


def before_after(root,output,index,pts,old,new,fit_rows,check_rows,landmarks):
    by={r['pts']:r for r in index['rows']}
    for point in pts:
        row=by[point];raw=cv2.imread(row['crop']);offset=np.asarray(row['crop_xyxy'][:2]);i=int(np.searchsorted(new['pts'],point));tiles=[];h,w=raw.shape[:2]
        check=any(r['pts']==point for r in check_rows);curves=check_rows if check else fit_rows
        for mode,color,label in [(0,(50,180,255),'Previous B: body-only blade proxy'),(1,(80,225,90),'Calibrated boot and articulated legs')]:
            shown=raw.copy()
            for foot in range(2):
                if mode==0:values=[old['blade_uv'][i,foot]-offset]
                else:
                    uv=new['equipment_uv'][i,foot]-offset
                    cv2.polylines(shown,[np.rint(hull(uv[:14])).astype(int)],True,color,1,cv2.LINE_AA)
                    values=[uv[14:31],uv[31:48]]
                for curve in values:cv2.polylines(shown,[np.rint(curve).astype(int)],False,color,2,cv2.LINE_AA)
            for r in curves:
                if r['pts']!=point:continue
                cv2.polylines(shown,[np.rint(np.asarray(r['uv'])-offset).astype(int)],False,(35,35,245),1,cv2.LINE_AA)
            for r in landmarks:
                if r['pts']==point:
                    for key in ['heel','toe']:
                        p=tuple(np.rint(np.asarray(r[key])-offset).astype(int));cv2.drawMarker(shown,p,(245,200,60),cv2.MARKER_CROSS,7,1)
            canvas=np.full((720,650,3),23,np.uint8);factor=min(630/w,570/h);im=cv2.resize(shown,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
            canvas[94:94+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(canvas,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1,cv2.LINE_AA)
            cv2.putText(canvas,f'Source {point/12800:.3f}s | native PTS {point} | pelvis fixed',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(canvas,'Red: blade development check, excluded from fitting' if check else 'Red: blade pixels used in calibration',(12,683),cv2.FONT_HERSHEY_SIMPLEX,.44,(65,70,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Cyan crosses: source heel / toe pose observations',(12,706),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,200,60),1,cv2.LINE_AA);tiles.append(canvas)
        cv2.imwrite(str(output/f'before_after_{point}.png'),np.hstack(tiles))


def deliver(root,fit_dataset,pose_dataset,review_path,pose_review_path,blade_check_path,programme):
    root,fit_dataset,pose_dataset,programme=map(Path,[root,fit_dataset,pose_dataset,programme]);out=pose_dataset/'delivery';out.mkdir(exist_ok=True)
    p=load(root/'input.npz');old=load(root/'pose/result.npz');fit=load(fit_dataset/'articulated/result.npz');new=load(pose_dataset/'observed_pose/result.npz')
    review=json.loads(Path(review_path).read_text(encoding='utf8'));pr=json.loads(Path(pose_review_path).read_text(encoding='utf8'));check=json.loads(Path(blade_check_path).read_text(encoding='utf8'))
    index=json.loads((pose_dataset/'source_index.json').read_text(encoding='utf8'));admission=json.loads((pose_dataset/'admission.json').read_text(encoding='utf8'))
    fit_run=json.loads((fit_dataset/'articulated/result.json').read_text(encoding='utf8'));pose_run=json.loads((pose_dataset/'observed_pose/result.json').read_text(encoding='utf8'))
    np.testing.assert_array_equal(p['pts'],new['pts']);np.testing.assert_array_equal(new['P_world_m'],old['P_world_m']);np.testing.assert_array_equal(new['Q_world'],old['Q_world'])
    np.testing.assert_array_equal(fit['x'][:len(INITIAL)],new['x'][:len(INITIAL)])
    statuses=np.full((len(p['pts']),2),'spline_inference_without_current_shoe_pixels',dtype='U64')
    statuses[p['lower_body_inferred']]='lower_body_unreliable_or_out_of_frame'
    for row in admission['records']:
        i=int(np.searchsorted(p['pts'],row['pts']));statuses[i,row['foot']]='source_boundary_used_for_pose' if row['status']=='admitted_visible_boundary' else row['status']
    for row in pr.get('boot_landmarks',[]):
        i=int(np.searchsorted(p['pts'],row['pts']));statuses[i,row['foot']]='source_heel_toe_used_for_pose'
    observed_blade=np.zeros(statuses.shape,bool)
    for row in review['blade_curves']:observed_blade[np.searchsorted(p['pts'],row['pts']),row['foot']]=True
    arrays=dict(pts=p['pts'],source_time_base=np.array([1,12800]),shared_parameter_names=np.asarray(NAMES),shared_geometry=new['x'][:len(INITIAL)],
        leg_dofs=np.asarray(LEG_DOFS),leg_correction_controls=new['x'][len(INITIAL):].reshape(-1,len(LEG_DOFS)),correction_knots=new['correction_knots'],
        source_pose_status=statuses,source_blade_pixels_fitted=observed_blade,contact_state=np.full(statuses.shape,'unknown'),contact_allowed_for_root=np.zeros(statuses.shape,bool),
        P_world_m=old['P_world_m'],G_world_m=np.c_[old['P_world_m'][:,:2],np.zeros(len(p['pts']))],Q_world=old['Q_world'],metres_per_scene_unit=p['metres_per_scene_unit'],body_scale=p['body_scale'],lower_body_inferred=p['lower_body_inferred'])
    parameter_file=programme/review['footwear_parameters_filename'];binding=programme/review['footwear_comparison_filename'];image_pts=pr['delivery_image_pts']
    body_binding=json.loads((programme/review['base_body_binding']).read_text(encoding='utf8'));body_parameters=Path(body_binding['parameters_file'])
    for file in [parameter_file,binding]:
        absolute=str(file.resolve())
        if len(absolute.encode('utf-16-le'))//2>=260 or any(len(s.encode('utf-16-le'))//2>255 for s in file.parts):raise ValueError('OneDrive Windows path domain')
    np.savez_compressed(parameter_file,**arrays)
    models={}
    for name,curves in [('previous_B',old['blade_uv']),('calibrated_blind_pose',fit['equipment_uv'][:,:,14:31]),('calibrated_observed_pose',new['equipment_uv'][:,:,14:31])]:
        models[name]=dict(fitted_blade_curves=curve_errors(review['blade_curves'],curves,p['pts']),development_blade_curves=curve_errors(check['blade_curves'],curves,p['pts']))
    boundaries=[];details=[]
    for row in admission['records']:
        if row['status']!='admitted_visible_boundary':continue
        i=int(np.searchsorted(p['pts'],row['pts']));polygon=hull(new['equipment_uv'][i,row['foot'],:14]);d=point_polyline_distance(row['source_boundary_native_uv'],polygon,True)
        boundaries.extend(d);details.append(dict(pts=row['pts'],foot=row['foot'],source_role=row['source_role'],**statistics(d)))
    active=[];x=new['x'][:len(INITIAL)]
    for j,name in enumerate(NAMES):
        u=(x[j]-LOWER[j])/(UPPER[j]-LOWER[j])
        if u<.01 or u>.99:active.append(dict(parameter=name,boundary='lower' if u<.01 else 'upper',normalized_distance_to_nearest_boundary=float(min(u,1-u))))
    errors={};visible=~p['lower_body_inferred']
    for name,ids in [('knees',[13,14]),('upper_body',list(range(13)))]:
        mask=p['source_qualified'][:,ids]&visible[:,None]
        errors[name]=dict(previous_B=statistics(np.linalg.norm(old['COCO_uv'][:,ids]-p['source_uv'][:,ids],axis=2)[mask]),
            calibrated=statistics(np.linalg.norm(new['COCO_uv'][:,ids]-p['source_uv'][:,ids],axis=2)[mask]))
    before_after(root,out,index,image_pts,old,new,review['blade_curves'],check['blade_curves'],pr.get('boot_landmarks',[]))
    sourcecode=Path(__file__).parent
    report=dict(schema='rinkspace.source-bound-footwear-candidate.v1',event=review['event'],programme=review['programme'],shot=review['shot'],
        source=review['source'],source_seconds=[float(p['pts'][0]/12800),float(p['pts'][-1]/12800)],native_exposures=len(p['pts']),
        camera_scene_pelvis_frozen=True,geometry_frozen_during_pose_inference=True,contact_height_fitted=False,contact_for_root_enabled=False,
        pelvis_maximum_difference_conditional_m=float(np.max(abs(new['P_world_m']-old['P_world_m']))),
        equipment_geometry=decode(x),active_numerical_model_boundaries=active,metric_scale_status='conditional',
        models=models,source_boot_contour_pixels=statistics(boundaries),source_boot_contour_per_foot=details,
        heel_toe_semantics=edge_errors(pr.get('boot_landmarks',[]),new['equipment_uv'],p['pts']),body_source_pixel_errors=errors,
        source_boot_candidates=len(admission['records']),source_boot_boundaries_admitted=len(details),
        source_observation_states=dict(Counter(statuses.ravel().tolist())),lower_body_inferred_frames=int(p['lower_body_inferred'].sum()),
        development_windows_seconds=review['check_windows_s'],no_final_independent_holdout=True,external_3d_ground_truth=False,
        shared_fit_termination=fit_run,pose_inference_termination=pose_run,user_visually_accepted=False,
        conclusions_zh=['鞋部源像素、鞋头鞋跟语义与独立刀刃下缘观测已分开登记。模型投影没有替代测量。',
            '鞋体几何全镜头共享，腿部允许受先验约束的局部旋转，没有逐帧自由鞋体平移或尺寸变化。',
            '先前留出失败已用于修正观测与模型，现作为开发诊断，不再称最终独立验证。',
            '仍有几何边界和刀刃检查误差；本轮是可审阅开发候选，不把接冰条件反馈给骨盆。',
            '骨盆及其竖直投影与B组完全相同；三维尺度、人体身高与鞋刀尺寸仍有条件假设。'],
        parameters_file=str(parameter_file.resolve()),parameters_sha256=digest(parameter_file),
        base_body_parameters_file=str(body_parameters.resolve()),base_body_parameters_sha256=digest(body_parameters),
        input_sha256=digest(root/'input.npz'),base_B_sha256=digest(root/'pose/result.npz'),camera_snapshot_sha256=digest(root/'camera_snapshot.npz'),
        objective_review_at_shared_run_sha256=digest(fit_dataset/'articulated/source_review_at_run.json'),
        objective_review_at_pose_run_sha256=digest(pose_dataset/'observed_pose/source_review_at_run.json'),
        objective_review_delivery_fields_added_after_run=True,
        data_result=str((pose_dataset/'observed_pose/result.npz').resolve()),data_result_sha256=digest(pose_dataset/'observed_pose/result.npz'),
        review_sha256=digest(review_path),pose_review_sha256=digest(pose_review_path),blade_check_sha256=digest(blade_check_path),
        source_code_at_run_sha256=dict(shared_solver=digest(fit_dataset/'articulated/source_footwear_fit.py'),pose_solver=digest(pose_dataset/'observed_pose/source_footwear_pose_transfer.py')),
        source_code_hash_role='delivery-time source; post-run changes added PTS validation and metadata snapshots without changing fitted residual mathematics',
        tests_passed=['three rigid geometry and source-distance tests','native footwear full/reduced projection and sparse support test','three moving-body exposure, native derivative and visibility tests'],
        source_code_sha256={f:digest(sourcecode/f) for f in ['footwear_geometry.py','footwear_fit.py','footwear_pose_transfer.py','footwear_rgb_observations.py','footwear_deliver.py']},
        full_video_created=False,review_images=[str((out/f'before_after_{pts}.png').resolve()) for pts in image_pts])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    binding.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(parameters=str(parameter_file),binding=str(binding),images=report['review_images'],models=models,
        active_boundaries=active,body_errors=errors),ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--fit-dataset',type=Path,required=True);ap.add_argument('--pose-dataset',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--pose-review',type=Path,required=True);ap.add_argument('--blade-check',type=Path,required=True);ap.add_argument('--programme',type=Path,required=True);a=ap.parse_args()
    deliver(a.root,a.fit_dataset,a.pose_dataset,a.review,a.pose_review,a.blade_check,a.programme)
