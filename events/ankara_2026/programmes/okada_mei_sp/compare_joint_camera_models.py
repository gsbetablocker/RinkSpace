"""Compare complete S01/S04 hypotheses at their saved native accepted states."""
from pathlib import Path
import argparse, collections, hashlib, html, json
import numpy as np
import prepare_native_joint_shots as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU

RUN=prepared.RUN

def stats(errors):
    a=np.asarray(errors);ok=np.isfinite(a);a=a[ok]
    return dict(rows=int(ok.size),invalid=int((~ok).sum()),cost=float(.5*np.dot(a,a)),rms=float(np.sqrt(np.mean(a*a))),median=float(np.median(a)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max()),above_20px=int((a>20).sum()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-optics',action='store_true');args=parser.parse_args()
    out=RUN/'camera_models';baseline=prepared.read(out/'baseline.json');manifest=prepared.read(RUN/'joint_manifest.json');points=prepared.npz(RUN/'point_observations.npz');mapping=prepared.read(RUN/'mapping.json')
    seed=prepared.npz(baseline['snapshot']);core=NativeCore(manifest['native_library']);scene=prepared.read(manifest['scene_path']);model=SceneParameterModel(scene,Path(manifest['scene_path']),optimize_scene=True)
    native=NativeSceneGPU(manifest['scene_path'],core,parameter_state=ParameterState(model.initial_state.layout,seed['scene_values']))
    scale=prepared.read(prepared.prep.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    paths=np.asarray(mapping['path_ids'])[points['path_indices']];layers=np.array([native.paths[p]['layer'] for p in paths]);entities=np.array([native.paths[p]['label'] for p in paths])
    inputs=[('accepted_baseline',Path(baseline['snapshot']),RUN/'joint_repaired/evaluation_pixels.npz',RUN/'joint_repaired/review/paused_result.json',None)]
    for profile in baseline['profiles']:
        key=Path(profile['path']).stem;directory=out/key
        if (directory/'result.json').exists() and (directory/'result.json').stat().st_mtime_ns>=(directory/'accepted_pixels_latest.npz').stat().st_mtime_ns:inputs.append((key,directory/'accepted_pixels_latest.npz',directory/'evaluation_pixels.npz',directory/'result.json',prepared.read(profile['path'])))
    baseline_uv=None;records=[];per_frame={};per_entity={}
    for key,snapshot,evaluation,result_path,profile in inputs:
        v=prepared.npz(snapshot);ev=prepared.npz(evaluation);result=prepared.read(result_path);rig=v['rig'];values=ev['values'];errors=np.linalg.norm(values[:,9:11],axis=1)
        assert np.array_equal(ev['rows'],np.arange(len(points['pts']))) and not ev['status'].any()
        if key=='accepted_baseline':baseline_uv=values[:,3:5].copy()
        if profile and profile['scene_mode']=='frozen':np.testing.assert_array_equal(v['scene_values'],seed['scene_values'])
        if profile:
            from rinkspace.domain.camera_experiment import apply_camera_experiment
            test,_,metadata=apply_camera_experiment(rig,profile);np.testing.assert_array_equal(test,rig)
            assert str(v['camera_experiment_sha256'])==metadata['profile_sha256']
        metrics=stats(errors);delta=np.linalg.norm(values[:,3:5]-baseline_uv,axis=1)
        physical=dict(pivot_height_conditional_m=float(rig[17]*scale),lever_xyz_conditional_m=(rig[18:21]*scale).tolist(),axis_spacing_conditional_m=float(rig[21]*scale),fx_over_fy=float(np.exp(rig[28])),readout_ms=float(rig[37]*1000),reference_principal_point_px=rig[26:28].tolist(),principal_point_zoom_coefficients=rig[29:31].tolist(),pupil_coefficients_conditional_m=(rig[22:24]*scale).tolist(),pan_axis_tilt_deg=float(np.degrees(np.arccos(np.clip(rig[11],-1,1)))),tilt_axis_camera_right_deviation_deg=float(np.degrees(np.arccos(np.clip(rig[12],-1,1)))),board_height_conditional_m=float(v['board'][5]*scale))
        bound_specs=[('Pz',rig[17]*scale,1.,2.),('lever_x',rig[18]*scale,-.05,.05),('lever_y',rig[19]*scale,-.25,-.08),('lever_z_reference',rig[20]*scale,-.1,.3),('axis_spacing',rig[21]*scale,-.05,.05)]
        physical['box_boundary_hits']=[dict(name=n,side='lower' if abs(x-lo)<1e-7 else 'upper',value_conditional_m=float(x)) for n,x,lo,hi in bound_specs if min(abs(x-lo),abs(x-hi))<1e-7]
        shots={};frames=[];entity_rows=[]
        for shot in manifest['shots']:
            sl=slice(shot['row_begin'],shot['row_stop']);pts=points['pts'][sl];ct=v['controls'][shot['control_begin']:shot['control_stop']];unique=np.unique(pts);t=(unique-shot['reference_pts'])/12800
            pose=core.evaluate_motion_camera(rig,ct,shot['first_control_time'],manifest['control_dt'],t);assert not pose['status'].any()
            q=stats(errors[sl]);q['surfaces']={str(layer):stats(errors[sl][layers[sl]==layer]) for layer in np.unique(layers[sl])};q['cx_range']=[float(pose['cx'].min()),float(pose['cx'].max())];q['cy_range']=[float(pose['cy'].min()),float(pose['cy'].max())];q['optical_height_conditional_m']=[float((pose['centre'][:,2]*scale).min()),float((pose['centre'][:,2]*scale).max())];q['geometric_focal_px']=np.exp([pose['log_f'].min(),pose['log_f'].max()]).tolist();q['optical_elevation_deg']=np.rad2deg([np.arcsin(pose['rotation'][:,2,2]).min(),np.arcsin(pose['rotation'][:,2,2]).max()]).tolist()
            extrema=core.motion_derivative_extrema(ct,shot['first_control_time'],manifest['control_dt'],[float(t.min()-abs(rig[37])/2),float(t.max()+abs(rig[37])/2)])
            q['continuous_motion_extrema']={name:{order:float(extrema['value'][j,i]*(180/np.pi if i<2 else 1.)) for j,order in enumerate(['speed','acceleration','jerk'])} for i,name in enumerate(['pan','tilt','log_f'])}
            if not args.no_optics:
                rt=(pts-shot['reference_pts'])/12800+rig[37]*(values[sl,6]/1079.-.5);rp=core.evaluate_motion_camera(rig,ct,shot['first_control_time'],manifest['control_dt'],rt);assert not rp['status'].any()
                rays=np.einsum('nij,nj->ni',rp['rotation'],values[sl,:3]-rp['centre']);xy=rays[:,:2]/rays[:,2,None];undistorted=xy*np.c_[rp['fx'],rp['fy']]+np.c_[rp['cx'],rp['cy']];distortion=np.linalg.norm(values[sl,3:5]-undistorted,axis=1);q['distortion_shift_px']=stats(distortion)
                pupil=(rig[22]*rp['zoom_log_ratio']+rig[23]*rp['zoom_log_ratio']**2)*scale;q['pupil_shift_conditional_m']=[float(pupil.min()),float(pupil.max())];q['wide_supported_rows']=int((rp['log_f']<rig[38]).sum())
            for pt in unique:
                selected=pts==pt;frames.append(dict(shot=shot['id'],pts=int(pt),seconds=float(pt/12800),**stats(errors[sl][selected]),projection_change_p95=float(np.percentile(delta[sl][selected],95))))
            for entity in np.unique(entities[sl]):entity_rows.append(dict(shot=shot['id'],entity=str(entity),**stats(errors[sl][entities[sl]==entity])))
            shots[shot['id']]=q
        record=dict(model=key,profile=profile,camera_columns=31 if profile is None else len(profile['active_camera_names']),scene_columns=121 if profile is None or profile['scene_mode']=='joint_current' else 0,snapshot=str(snapshot),snapshot_sha256=prepared.digest(snapshot),native_library_provenance_sha256=str(v.get('native_library_provenance_sha256','unknown')),metrics=metrics,shots=shots,physical=physical,scene_values_exactly_equal_baseline=bool(np.array_equal(v['scene_values'],seed['scene_values'])),scene_max_coordinate_delta=float(np.max(np.abs(v['scene_values']-seed['scene_values']))),fitted_observation_change_from_baseline=stats(delta),numerical={k:result.get(k) for k in ['stop_reason','numerically_converged','constraint_projected_gradient_inf','total_trials','total_accepted','wall_ms','stationarity_diagnostics']})
        records.append(record);per_frame[key]=frames;per_entity[key]=entity_rows
        print(key,metrics['rms'],metrics['p95'],metrics['maximum'],physical['fx_over_fy'],physical['pivot_height_conditional_m'],flush=True)
    summary=dict(schema='rinkspace.shared_camera_model_comparison.v1',baseline=baseline,points=len(points['pts']),frames=len(np.unique(points['pts'])),conditional_metres_per_L=scale,all_points_used_for_fitting=True,independent_physical_validation=False,box_boundary_detection_tolerance_conditional_m=1e-7,fitted_observation_change_note='Curve rows optimize their own stations; these differences are fitted observation UV changes, not fixed world-point camera mapping differences.',completed_candidates=len(records)-1,expected_candidates=len(baseline['profiles']),models=records)
    prepared.write(out/'comparison.json',summary);prepared.write(out/'frame_metrics.json',per_frame);prepared.write(out/'entity_metrics.json',per_entity)
    header='<meta charset="utf-8"><title>1＋4 镜头相机模型对照</title><style>body{font:15px system-ui;margin:32px;color:#172a3b;background:#f4f6f8}table{border-collapse:collapse;background:white;margin:18px 0}td,th{padding:9px;border:1px solid #ccd5dd;text-align:right}td:first-child{text-align:left}a{color:#075ca3}p{max-width:1100px;line-height:1.7}img{max-width:100%}</style><h1>1＋4 镜头：简化相机与冻结场景对照</h1>'
    body=f'<p>已完成 {len(records)-1}/{len(baseline["profiles"])} 个候选；运行中的结果不计入本表。</p><p><a href="visual_comparison.html">查看同源画面对照</a></p>'+'<p>所有候选使用同一 569960 条源观测、1936 个曝光、原始无权重像素平方和、同一运动基函数和安装工作范围。当前满意结果保留为 accepted_baseline。以下是开发数据上的完整重拟合对照，不能代替独立物理标定；米制均条件于现有比例。停滞不写成收敛。</p><table><tr><th>模型／场景</th><th>相机列</th><th>RMS px</th><th>P95</th><th>最大</th><th>超过20px</th><th>fx/fy</th><th>Pz 条件m</th><th>readout ms</th><th>触边</th><th>数值停止</th></tr>'
    for rec in records:
        m=rec['metrics'];ph=rec['physical'];bounds=', '.join(x['name']+('↓' if x['side']=='lower' else '↑') for x in ph['box_boundary_hits']);body+=f'<tr><td>{html.escape(rec["model"])}</td><td>{rec["camera_columns"]}</td><td>{m["rms"]:.3f}</td><td>{m["p95"]:.3f}</td><td>{m["maximum"]:.2f}</td><td>{m["above_20px"]}</td><td>{ph["fx_over_fy"]:.4f}</td><td>{ph["pivot_height_conditional_m"]:.3f}</td><td>{ph["readout_ms"]:.3f}</td><td>{bounds}</td><td>{html.escape(rec["numerical"]["stop_reason"])}</td></tr>'
    body+='</table><p>逐镜头、冰面／围挡／广告、主点／光心／畸变位移、共享场景变化以及停止诊断详见 <a href="comparison.json">comparison.json</a>；逐帧与逐实体见 <a href="frame_metrics.json">frame_metrics.json</a> 和 <a href="entity_metrics.json">entity_metrics.json</a>。</p>'
    (out/'comparison.html').write_text(header+body,encoding='utf-8')

if __name__=='__main__':main()
