"""Read-only source/fit diagnostics for the existing S01 foot/boot integration.

Closed and open replay identical old coordinates; their contour operator differs.
Open versus fitted uses the same explicit-topology visible-edge operator and
original source observations. No solver, native camera, body model or viewer is
started. Internal diagnostics are supplied x0/fitted approximate-model quantities.
"""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
from .fullbody_fit import footwear_rows
from .footwear_fit import sample_polyline
from .footwear_pixel_factors import nearest,nearest_segment,pixel_factors
from .footwear_observations import visible_shell_edges
from rinkspace.path_policy import guard_project_write

MODES=('closed','open','fitted')
WINDOWS=((35.2,36.8),(40.,41.5))
KEY_PTS=(336128,461056,521728)
BODY_GROUPS=dict(all=list(range(17)),ankle_proxy=[15,16],upper_limb=[7,8,9,10],lower_limb=[13,14,15,16])


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def serializable(value):
    if isinstance(value,np.ndarray):return serializable(value.tolist())
    if isinstance(value,np.generic):return serializable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    return value


def stats(value):
    value=np.asarray(value,float).ravel();finite=value[np.isfinite(value)]
    return dict(n=len(value),finite_n=len(finite),missing_n=len(value)-len(finite),
        median=float(np.median(finite)) if len(finite) else None,
        p95=float(np.percentile(finite,95)) if len(finite) else None,
        max=float(np.max(finite)) if len(finite) else None)


def rotation_degrees(a,b):
    r=np.asarray(a)@np.swapaxes(np.asarray(b),-1,-2)
    sine=np.linalg.norm(np.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1),axis=-1)/2
    cosine=(np.trace(r,axis1=-2,axis2=-1)-1)/2
    return np.rad2deg(np.arctan2(sine,np.clip(cosine,-1,1)))


def masks_for(pts):
    t=pts/12800.
    windows={f'w{i+1}':(t>=a-1e-9)&(t<=b+1e-9) for i,(a,b) in enumerate(WINDOWS)}
    refined=np.logical_or.reduce(list(windows.values()))
    return dict(all_s01=np.ones(len(pts),bool),body_refined=refined,body_unchanged=~refined,**windows)


def body_metrics(packet,observations,masks,pts):
    error=np.linalg.norm(packet['COCO_uv']-observations['uv'],axis=-1);use=np.asarray(observations['use'],bool)
    result={}
    for name,mask in masks.items():
        result[name]={}
        for group,ids in BODY_GROUPS.items():
            selected=use[:,ids]&mask[:,None];values=error[:,ids][selected];s=stats(values)
            if np.isfinite(values).any():
                fi,ji=np.nonzero(selected);worst=int(np.nanargmax(values));s['worst']=dict(pts=int(pts[fi[worst]]),joint=int(np.asarray(ids)[ji[worst]]))
            result[name][group]=s
    return result


def equipment_metrics(root,baseline,fitted,binding,centres,pts,masks):
    gear,sources=footwear_rows(root,np.arange(len(pts)),pts)
    packets=dict(closed=baseline,open=baseline,fitted=fitted);faces=np.asarray(binding['shell_faces'],int)
    region_ids=np.asarray(binding.get('region_ids',[0]*len(faces)))
    uv={mode:np.concatenate((p['equipment_uv'].reshape(len(pts),96,2),p['COCO_uv']),axis=1) for mode,p in packets.items()}
    edge_sets={mode:{} for mode in ('open','fitted')};visibility={mode:[] for mode in ('open','fitted')}
    for fi,foot in sorted({(int(r['frame']),int(r['foot'])) for r in gear.visible_rows}):
        for mode in edge_sets:
            try:
                edge=visible_shell_edges(packets[mode]['gear_world_mid'][fi,foot,:14],faces,centres[fi],region_ids)
                edge_sets[mode][(fi,foot)]=edge
                visibility[mode].append(dict(pts=int(pts[fi]),foot=foot,edges=edge['edges'].tolist(),region_ids=edge['region_ids'],
                    edge_types=edge['edge_types'],occluded_edges=edge['occluded_edges'].tolist(),method=edge['method'],approximation=edge['approximation']))
            except ValueError as error:
                edge_sets[mode][(fi,foot)]=dict(edges=np.empty((0,2),int))
                visibility[mode].append(dict(pts=int(pts[fi]),foot=foot,edges=[],domain_failure=str(error)))
    normalized={}
    for mode in MODES:
        try:
            residual,_,parts,choices=pixel_factors(uv[mode],gear,body=False,boot_edges=None if mode=='closed' else edge_sets[mode])
            normalized[mode]=dict(status='finite' if np.isfinite(residual).all() else 'nonfinite',residual_rows=len(residual),cost_terms=parts,blade_choices=choices)
        except ValueError as error:normalized[mode]=dict(status='domain_failure',reason=str(error))
    records=[]
    for kind,rows in (('outer_boot_contour',gear.visible_rows),('blade_boundary',gear.blade_rows),('outer_boot_heel_toe',gear.landmark_rows)):
        for row in rows:
            fi,foot=int(row['frame']),int(row['foot']);source=np.asarray(row['observed'] if kind=='outer_boot_contour' else row['uv'] if kind=='blade_boundary' else [row['heel'],row['toe']],float)
            record=dict(kind=kind,pts=int(pts[fi]),foot=foot,sigma_px=float(row['sigma_px']),source_uv=source.tolist(),modes={})
            for mode,packet in packets.items():
                projected=packet['equipment_uv'][fi,foot];details={}
                try:
                    if kind=='outer_boot_contour':
                        if mode=='closed':
                            hull=cv2.convexHull(projected[:14].astype(np.float32),returnPoints=False).ravel()
                            distance=nearest(source,projected[hull],closed=True)[0]
                        else:distance=nearest_segment(source,projected[:14],edge_sets[mode][(fi,foot)]['edges'])[0]
                    elif kind=='blade_boundary':
                        sampled=sample_polyline(source,24);candidates=(projected[14:31],projected[31:48])
                        sampled_dist=[nearest(sampled,line)[0] for line in candidates]
                        which=int(np.argmin([float(d@d) for d in sampled_dist]));distance=nearest(source,candidates[which])[0]
                        details=dict(selected_candidate=('blade_left','blade_right')[which],source_side_confirmed=False,
                            sampled24_summary_px=stats(sampled_dist[which]))
                    else:distance=np.array([nearest_segment(source[i:i+1],projected,np.array([pair]))[0][0] for i,pair in enumerate(([0,7],[1,8]))])
                    if not np.isfinite(distance).all():details['domain_failure']='nonfinite prediction; denominator retained'
                except (ValueError,cv2.error) as error:
                    distance=np.full(len(source),np.nan);details['domain_failure']=str(error)
                record['modes'][mode]=dict(residual_px=distance.tolist(),summary_px=stats(distance),**details)
            records.append(record)
    grouped={}
    for name,mask in masks.items():
        allowed=set(map(int,pts[mask]));grouped[name]={}
        for kind in ('outer_boot_contour','blade_boundary','outer_boot_heel_toe'):
            rows=[r for r in records if r['kind']==kind and r['pts'] in allowed]
            grouped[name][kind]=dict(source_records=len(rows),source_frames=len({r['pts'] for r in rows}),
                modes={mode:stats([v for r in rows for v in r['modes'][mode]['residual_px']]) for mode in MODES})
    return dict(sources=sources,groups=grouped,records=records,visibility=visibility,normalized_source_factor_costs=normalized,
        contour_definition='original admitted source samples to finite explicit shell edges; closed legacy convex hull reported separately',
        blade_definition='original source vertices scored on one finite side per curve; side selected using the unchanged 24-sample rule',
        heel_toe_definition='external boot heel/toe edges 0-7 and 1-8, not bare-foot or bone landmarks',
        zero_observation_groups='n=0 means absent source evidence, never model-filled observations')


def internal_metrics(baseline_diagnostics,fitted,masks):
    names=('cavity_penetration_mid','heel_drift_mid','forefoot_error_mid')
    result={}
    for scope,mask in masks.items():
        result[scope]={}
        for name in names:
            result[scope][name]={mode:dict(combined=stats(packet[name][mask]),
                left=stats(packet[name][mask,0]),right=stats(packet[name][mask,1]))
                for mode,packet in (('x0',baseline_diagnostics),('fitted',fitted))}
    return dict(groups=result,comparison='same newly defined approximate cavity/heel/ball model, x0 versus fitted; old boot did not contain measured inner geometry',
        definitions=dict(cavity_penetration_mid='maximum coordinate violation over sampled foot surface points, not exact Euclidean shell penetration depth',
            heel_drift_mid='heel patch centroid to approximate heel-pocket anchor distance, not temporal foot-sliding measurement',
            forefoot_error_mid='forefoot patch centroid to approximate ball-support anchor distance'),units='current conditional metres')


def state_metrics(baseline,baseline_diagnostics,fitted,binding,masks):
    result={};length_ratio=float(binding['metric_binding']['length_export_ratio'])
    x=np.asarray(binding['fit_parameters_solver_unit'],float);length_scale,width_scale=np.exp(x[12:14])
    template=np.asarray(binding['outer_shell_template']);extent=np.ptp(template,axis=0)
    for scope,mask in masks.items():
        delta=fitted['P_world_mid']-baseline['P_world_mid']
        upper=np.asarray(fitted['upper_deformation_mid'][mask])
        modes={}
        for i,name in enumerate(binding['upper_deformation_modes']['names']):
            values=upper[...,i];maximum=float(binding['deformation_prior']['maximum_mode_displacements_m'][i])
            modes[name]=dict(normalized_activity=stats(values),active_samples=int(np.sum(values>0)),saturated_samples=int(np.sum(values>=1-1e-9)),
                geometric_displacement_m=stats(values*maximum))
        blade={}
        for mode,packet in (('open',baseline),('fitted',fitted)):
            minimum=np.min(packet['gear_world_mid'][mask,:,14:,2],axis=-1)
            blade[mode]=dict(minimum_sample_height_m=stats(minimum),below_z0_samples=int(np.sum(minimum<0)),
                below_z0_depth_m=stats(np.maximum(-minimum,0)))
        group=dict(source_frames=int(mask.sum()),P_change_m=stats(np.linalg.norm(delta[mask],axis=-1)),
            root_Q_change_deg=stats(rotation_degrees(fitted['Q_world'][mask],baseline['Q_world'][mask])),
            local_rotation_change_deg=stats(rotation_degrees(fitted['local_R_mid'][mask],baseline['local_R_mid'][mask])),
            upper_modes=modes,blade_ice_height=blade,blade_world_tilt_deg=stats(np.rad2deg(fitted['blade_tilt_rad_mid'][mask])))
        if 'effective_ankle_relative_rotation_mid' in baseline_diagnostics:
            group['effective_shank_hindfoot_rotation_change_deg']=stats(rotation_degrees(
                fitted['effective_ankle_relative_rotation_mid'][mask],baseline_diagnostics['effective_ankle_relative_rotation_mid'][mask]))
        if 'blade_tilt_rad_mid' in baseline_diagnostics:
            group['blade_world_tilt_change_deg']=stats(np.rad2deg(fitted['blade_tilt_rad_mid'][mask]-baseline_diagnostics['blade_tilt_rad_mid'][mask]))
        result[scope]=group
    unchanged=masks['body_unchanged']
    return dict(groups=result,body_outside_refinement_exact_equal={name:bool(np.array_equal(fitted[name][unchanged],baseline[name][unchanged]))
        for name in ('P_world_mid','Q_world','local_R_mid','native22_world_mid','COCO_world_mid')},
        body_outside_refinement_max_abs={name:float(np.max(abs(fitted[name][unchanged]-baseline[name][unchanged])))
        for name in ('P_world_mid','Q_world','local_R_mid','native22_world_mid','COCO_world_mid')},
        shared_size_change=dict(length_multiplier=float(length_scale),width_multiplier=float(width_scale),
            old_longitudinal_extent_m=float(extent[2]/length_scale),fitted_longitudinal_extent_m=float(extent[2]),
            old_width_extent_m=float(extent[0]/width_scale),fitted_width_extent_m=float(extent[0])),
        shared_registration_delta=dict(translation_m=(x[:12].reshape(2,6)[:,:3]*length_ratio).tolist(),
            rotation_change_deg=np.rad2deg(np.linalg.norm(x[:12].reshape(2,6)[:,3:],axis=1)).tolist()),
        upper_mode_definition='normalized geometric mode activity 0..1, not material strain or measured stiffness',
        ankle_definition='effective shank/hindfoot frame rotation, not anatomically calibrated dorsiflexion/inversion components',
        ice_height_definition='minimum of saved finite blade samples relative to Z=0; not contact classification or continuous-curve minimum')


def fmt(value,scale=1):return '缺失' if value is None else f'{value*scale:.2f}'


def triple(s,scale=1):return '/'.join(fmt(s[k],scale) for k in ('median','p95','max'))


def report_text(report):
    lines=['S01 足—鞋配合结果对照','',
        '共享足鞋配准作用于全S01 1143帧；人体只新增细化35.20–36.80s、40.00–41.50s，共157帧（含端点）。',
        'closed与open使用完全相同的旧坐标：closed为旧投影凸包，open为开放鞋口显式可见边；二者差异属于拓扑/残差语义改变。',
        '主要拟合对照是open→fitted，同一原观测、同一显式可见边算子。下列均为median/P95/max；n保留缺失分母。','',
        '1. 近似内部配合（mm；x0→fitted，同一新增内腔定义）']
    labels=dict(cavity_penetration_mid='足表面采样最大内腔坐标越界',heel_drift_mid='足跟patch—近似跟杯锚点距离',forefoot_error_mid='前掌patch—近似支撑锚点距离')
    for scope in ('all_s01','body_refined'):
        lines.append('  '+('全S01' if scope=='all_s01' else '两个人体细化窗')+':')
        for name,label in labels.items():
            row=report['internal_fit']['groups'][scope][name]
            lines.append(f'    {label}: {triple(row["x0"]["combined"],1000)} → {triple(row["fitted"]["combined"],1000)}; n={row["x0"]["combined"]["n"]}')
    lines += ['  上述内腔、足床和跟杯关系是新增近似几何/先验的诊断，不是旧鞋内部实测。','', '2. 原生图像观测（px）']
    for kind,row in report['equipment_pixels']['groups']['all_s01'].items():
        lines.append(f'  {kind}（{row["source_records"]}条源记录）:')
        for mode in MODES:
            s=row['modes'][mode];lines.append(f'    {mode}: {triple(s)}; n={s["n"]}, missing={s["missing_n"]}')
    lines.append('  两人体窗外固定父版身体的源帧（共享足鞋配准仍生效）：')
    for kind,row in report['equipment_pixels']['groups']['body_unchanged'].items():
        before,after=row['modes']['open'],row['modes']['fitted']
        lines.append(f'    {kind}: {triple(before)} → {triple(after)}; n={before["n"]}, missing={before["missing_n"]}/{after["missing_n"]}')
    for scope in ('all_s01','body_refined','w1','w2'):
        old=report['body_pixels']['open'][scope];new=report['body_pixels']['fitted'][scope]
        lines.append(f'  人体 {scope}: all {triple(old["all"])} → {triple(new["all"])}; n={old["all"]["n"]}')
        lines.append(f'    ankle检测代理 {triple(old["ankle_proxy"])} → {triple(new["ankle_proxy"])}; n={old["ankle_proxy"]["n"]}')
    lines += ['', '3. 尺寸、上帮活动与副作用']
    change=report['state']['shared_size_change']
    lines.append(f'  共享鞋长倍率 {change["length_multiplier"]:.4f}，宽度倍率 {change["width_multiplier"]:.4f}；外壳纵向跨度 {change["old_longitudinal_extent_m"]*1000:.1f}→{change["fitted_longitudinal_extent_m"]*1000:.1f}mm，宽度 {change["old_width_extent_m"]*1000:.1f}→{change["fitted_width_extent_m"]*1000:.1f}mm。')
    for name,row in report['state']['groups']['all_s01']['upper_modes'].items():
        lines.append(f'  {name}: 活动 {row["active_samples"]}/{row["normalized_activity"]["n"]}，到模式上限 {row["saturated_samples"]}；归一幅度 {triple(row["normalized_activity"])}；几何位移P95 {fmt(row["geometric_displacement_m"]["p95"],1000)}mm。')
    state=report['state']['groups']['body_refined']
    lines.append(f'  人体细化窗P改变量 {triple(state["P_change_m"],1000)}mm，根朝向改变量 {triple(state["root_Q_change_deg"])}°。')
    worst=report['state']['body_outside_refinement_max_abs']
    lines.append('  窗外人体最大绝对差：'+', '.join(f'{key}={value:.3g}' for key,value in worst.items()))
    blade=report['state']['groups']['all_s01']['blade_ice_height']
    lines.append(f'  刃样点低于Z=0：open {blade["open"]["below_z0_samples"]} / fitted {blade["fitted"]["below_z0_samples"]}个脚曝光；最大深度 {fmt(blade["open"]["below_z0_depth_m"]["max"],1000)}→{fmt(blade["fitted"]["below_z0_depth_m"]["max"],1000)}mm。穿冰/真实接触未在本轮解决。')
    regressions=report['numeric_regressions']
    lines.append('  open→fitted数值退化（同分母，px）：')
    for r in regressions:
        if r['scope'] in ('all_s01','body_refined','body_unchanged'):
            lines.append(f'    {r["domain"]}/{r["group"]}/{r["scope"]}/{r["statistic"]}: {r["before_px"]:.2f}→{r["after_px"]:.2f}')
    if not regressions:lines.append('    本报告分组统计未发现上升。')
    lines += ['', '4. 代表源时刻']
    for row in report['key_frames']:
        a=row['internal_fit']['x0'];b=row['internal_fit']['fitted']
        lines.append(f'  PTS {row["pts"]} / {row["seconds"]:.2f}s；人体新增细化={row["body_refined"]}')
        for name,label in labels.items():
            lines.append(f'    {label} 左/右 mm: '+ '/'.join(fmt(v,1000) for v in a[name])+' → '+ '/'.join(fmt(v,1000) for v in b[name]))
        source=report['equipment_pixels']['records'];current=[r for r in source if r['pts']==row['pts']]
        lines.append('    此时实际鞋刀源记录 '+str(len(current))+'条；缺项不由模型点补齐。')
    lines += ['', '求解记录：']
    for name,row in report['solver_records'].items():
        lines.append(f'  {name}: {row.get("stopping_reason","未提供")}；optimality/gradient_inf={row.get("optimality",row.get("gradient_inf","未提供"))}')
    lines += ['', '检测器ankle仍是衣物/鞋遮挡下的代理；外鞋heel/toe不作裸足点，刀刃端点不冒充刀齿。',
        '报告只读已保存结果，未新增拟合、视频或图像；查看图由既有viewer程序生成。','']
    return '\n'.join(lines)


def run(root,output,baseline=None):
    root=Path(root).resolve();output=Path(output).resolve();baseline_path=Path(baseline).resolve() if baseline else root/'fullbody_1102/full/result.npz'
    paths=dict(baseline=baseline_path,fitted=output/'fitted/result.npz',observations=output/'observations.npz',ledger=output/'ledger.json',
        binding=output/'binding.json',baseline_diagnostics=output/'baseline_diagnostics.npz',observation_semantics=output/'observation_semantics.json',
        camera_pose=root/'fullbody_1102/camera_pose.npz')
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing:raise FileNotFoundError('Report inputs are not ready; no report written: '+', '.join(missing))
    old=load(paths['baseline']);fitted=load(paths['fitted']);obs=load(paths['observations']);diag=load(paths['baseline_diagnostics']);camera=load(paths['camera_pose'])
    ledger=read(paths['ledger']);binding=read(paths['binding']);pts=obs['pts'];masks=masks_for(pts)
    if len(pts)!=1143 or masks['body_refined'].sum()!=157:raise ValueError('Unexpected S01/body-refinement scope')
    for name,packet in (('baseline',old),('fitted',fitted),('baseline_diagnostics',diag),('camera_pose',camera)):
        clock=packet.get('pts',packet.get('source_pts'))
        if clock is None:raise ValueError('Missing source PTS in '+name)
        np.testing.assert_array_equal(clock,pts)
        if 'time_base' in packet:np.testing.assert_array_equal(packet['time_base'],[1,12800])
    np.testing.assert_array_equal(obs['time_base'],[1,12800]);np.testing.assert_array_equal(fitted['body_refinement_window_mask'],masks['body_refined'])
    if 'footwear_binding_applied_mask' in fitted and not np.all(fitted['footwear_binding_applied_mask']):raise ValueError('Shared binding is not applied over the whole source scope')
    for p in (fitted,camera,diag):
        if 'metres_per_scene_unit' in p:np.testing.assert_equal(p['metres_per_scene_unit'],old['metres_per_scene_unit'])
    if 'binding_x0' not in diag or not np.array_equal(diag['binding_x0'],np.zeros(14)):
        raise ValueError('Baseline diagnostics must explicitly identify zero shared registration')
    if str(diag.get('parent_result_sha256',''))!=digest(paths['baseline']):
        raise ValueError('Baseline diagnostic parent result SHA does not match the compared old geometry')
    if ledger.get('observation_array_sha256')!=digest(paths['observations']):raise ValueError('Observation ledger SHA mismatch')
    if str(fitted['observation_ledger_id'])!=digest(paths['ledger']):raise ValueError('Fitted result has a stale observation ledger')
    for p in (old,fitted):
        info=json.loads(str(p['camera_binding_json']))
        np.testing.assert_equal(info['state_sha256'],binding['metric_binding']['state_sha256'])
    required=('cavity_penetration_mid','heel_drift_mid','forefoot_error_mid')
    for name in required:
        for p in (diag,fitted):
            if p[name].shape!=(len(pts),2):raise ValueError('Unexpected diagnostic shape: '+name)
    centres=camera['centre']*float(old['metres_per_scene_unit'])
    equipment=equipment_metrics(root,old,fitted,binding,centres,pts,masks)
    body=dict(open=body_metrics(old,obs,masks,pts),fitted=body_metrics(fitted,obs,masks,pts))
    regressions=[]
    for scope in masks:
        for group in BODY_GROUPS:
            a,b=body['open'][scope][group],body['fitted'][scope][group]
            for statistic in ('median','p95','max'):
                if a[statistic] is not None and b[statistic] is not None and b[statistic]>a[statistic] and not np.isclose(b[statistic],a[statistic],rtol=0,atol=1e-9):
                    regressions.append(dict(domain='body_pixels',scope=scope,group=group,statistic=statistic,before_px=a[statistic],after_px=b[statistic]))
        for group,row in equipment['groups'][scope].items():
            a,b=row['modes']['open'],row['modes']['fitted']
            for statistic in ('median','p95','max'):
                if a[statistic] is not None and b[statistic] is not None and b[statistic]>a[statistic]:
                    regressions.append(dict(domain='equipment_pixels',scope=scope,group=group,statistic=statistic,before_px=a[statistic],after_px=b[statistic]))
    key=[]
    for point in KEY_PTS:
        fi=int(np.searchsorted(pts,point))
        if fi>=len(pts) or pts[fi]!=point:raise ValueError('Missing representative source PTS')
        key.append(dict(pts=point,seconds=point/12800.,body_refined=bool(masks['body_refined'][fi]),
            internal_fit={mode:{name:p[name][fi].tolist() for name in required} for mode,p in (('x0',diag),('fitted',fitted))},
            body_residual_px={mode:np.linalg.norm(p['COCO_uv'][fi]-obs['uv'][fi],axis=-1).tolist() for mode,p in (('open',old),('fitted',fitted))},
            body_use=obs['use'][fi].tolist(),upper_deformation_normalized=fitted['upper_deformation_mid'][fi].tolist()))
    semantics=read(paths['observation_semantics'])
    solver={name:read(output/(name+'.json')) for name in ('registration','joint_result') if (output/(name+'.json')).is_file()}
    report=dict(schema='rinkspace.boot-integration-review.v1',comparison=dict(closed='old coordinates, historical convex-hull contour operator',
        open='identical old coordinates, explicit open-collar visible-edge contour operator',fitted='shared registration plus two body windows, same explicit visible-edge operator as open'),
        scope=dict(source_frames=len(pts),source_seconds=[float(pts[0]/12800),float(pts[-1]/12800)],shared_binding_frames=len(pts),
            new_body_windows=WINDOWS,new_body_refined_frames=int(masks['body_refined'].sum()),parent_body_frames=int(masks['body_unchanged'].sum())),
        source_time_base=[1,12800],sources={name:dict(path=str(p),sha256=digest(p)) for name,p in paths.items()},
        denominator='same original equipment rows; same selected body observations.use for each scope; nonfinite predictions retained as missing',
        body_pixels=body,equipment_pixels=equipment,internal_fit=internal_metrics(diag,fitted,masks),state=state_metrics(old,diag,fitted,binding,masks),
        key_frames=key,numeric_regressions=regressions,solver_records=solver,
        observation_semantics={key:semantics.get(key) for key in ('schema','counts','missing','coordinate_space','evaluation_role')},
        limitations=['Internal cavity/heel/ball fit is an approximate geometry/registration diagnostic, not an internal shoe measurement.',
            'This round does not resolve true blade contact or below-ice geometry.',
            'Visible edges use the existing low-resolution mesh midpoint self-occlusion approximation.'])
    for filename in ('boot_report.json','results.txt'):guard_project_write(output/filename)
    (output/'boot_report.json').write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    (output/'results.txt').write_text(report_text(report),encoding='utf8')
    return report



def source_figures(root,output):
    """Decode two exact source exposures and draw saved UV only; no fit/inference.

    Optional PyAV/matplotlib imports live here so the ordinary report/export
    entry remains usable in the base Python environment.
    """
    from .fullbody_review import exact_frames,find_source
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,output=Path(root).resolve(),Path(output).resolve()
    baseline_path=root/'fullbody_1102/full/result.npz';fitted_path=output/'fitted/result.npz'
    camera_path=root/'fullbody_1102/camera_pose.npz';binding_path=output/'binding.json'
    for p in (baseline_path,fitted_path,camera_path,binding_path):
        if not p.is_file():raise FileNotFoundError(p)
    old,new=load(baseline_path),load(fitted_path);camera=load(camera_path);binding=read(binding_path)
    pts=old['pts'];targets=(336128,461056);frames=np.asarray([int(np.searchsorted(pts,p)) for p in targets])
    np.testing.assert_array_equal(pts[frames],targets)
    for packet in (new,camera):np.testing.assert_array_equal(packet['pts'],pts)
    for packet in (old,new):np.testing.assert_array_equal(packet['time_base'],[1,12800])
    np.testing.assert_equal(old['metres_per_scene_unit'],new['metres_per_scene_unit'])
    np.testing.assert_equal(camera['metres_per_scene_unit'],old['metres_per_scene_unit'])
    centres=camera['centre']*float(old['metres_per_scene_unit'])
    gear,gear_sources=footwear_rows(root,frames,pts)
    source=find_source(root);rgb=exact_frames(source['path'],targets)
    faces=np.asarray(binding['shell_faces'],int);regions=np.asarray(binding.get('region_ids',[0]*len(faces)))
    fig,axes=plt.subplots(2,3,figsize=(16,11),squeeze=False,constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.12,w_pad=.06,hspace=.10,wspace=.04)
    metadata=dict(schema='rinkspace.boot-source-comparison.v1',source_time_base=[1,12800],source=source,
        input_files={name:dict(path=str(p),sha256=digest(p)) for name,p in
            (('baseline',baseline_path),('fitted',fitted_path),('camera_pose',camera_path),('binding',binding_path))},
        equipment_source=gear_sources,colors=dict(shell='#16b9ff',blade='#ed36a9',source_heel_toe='#ffc400'),
        columns=['raw exact source pixels','old open explicit predicted shell/blade','fitted explicit predicted shell/blade'],
        pixels='native source coordinates; one common crop per row; no geometric/image adjustment',
        visibility='shell topology selected at common midpoint; saved rolling-row equipment_uv drawn without reprojecting or modifying it',frames=[])
    for row,(point,fi) in enumerate(zip(targets,frames)):
        source_image=rgb[point];h,w=source_image.shape[:2]
        original=[np.asarray(r['observed'],float) for r in gear.visible_rows if r['pts']==point]
        original += [np.asarray(r['uv'],float) for r in gear.blade_rows if r['pts']==point]
        landmarks=[r for r in gear.landmark_rows if r['pts']==point]
        original += [np.asarray([r['heel'],r['toe']],float) for r in landmarks]
        projected=[old['equipment_uv'][fi].reshape(-1,2),new['equipment_uv'][fi].reshape(-1,2)]
        framing=np.concatenate(original+projected,axis=0);framing=framing[np.isfinite(framing).all(axis=1)]
        if not len(framing):raise ValueError('No finite original/predicted shoe support for source crop')
        lo=np.maximum(np.floor(framing.min(axis=0)-28),[0,0]).astype(int)
        hi=np.minimum(np.ceil(framing.max(axis=0)+28),[w,h]).astype(int)
        if np.any(hi<=lo):raise ValueError('Shoe crop lies outside the source frame')
        crop=source_image[lo[1]:hi[1],lo[0]:hi[0]]
        record=dict(pts=point,seconds=point/12800.,native_crop_xyxy=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],
            full_rgb_sha256=hashlib.sha256(source_image.tobytes()).hexdigest(),
            cropped_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),source_shape=list(source_image.shape),modes={})
        for col,mode in enumerate(('raw','open','fitted')):
            ax=axes[row,col]
            ax.imshow(crop,extent=(lo[0]-.5,hi[0]-.5,hi[1]-.5,lo[1]-.5))
            subtitle='raw source (no overlay)'
            if mode!='raw':
                packet=old if mode=='open' else new;uv=packet['equipment_uv'][fi]
                edge_packets=[];shell_residual=[];heel_toe_residual=[]
                for foot in (0,1):
                    edge=visible_shell_edges(packet['gear_world_mid'][fi,foot,:14],faces,centres[fi],regions)
                    edge_packets.append(dict(foot=foot,edges=edge['edges'].tolist(),edge_types=edge['edge_types'],region_ids=edge['region_ids']))
                    for a,b in edge['edges']:
                        ax.plot(uv[foot,[a,b],0],uv[foot,[a,b],1],color=metadata['colors']['shell'],lw=1.0)
                    for ids in (slice(14,31),slice(31,48)):
                        ax.plot(uv[foot,ids,0],uv[foot,ids,1],color=metadata['colors']['blade'],lw=.95)
                    for r in gear.visible_rows:
                        if r['pts']==point and r['foot']==foot:
                            shell_residual.extend(nearest_segment(r['observed'],uv[foot,:14],edge['edges'])[0].tolist())
                    for r in landmarks:
                        if r['foot']!=foot:continue
                        for name,pair in (('heel',[0,7]),('toe',[1,8])):
                            p=np.asarray(r[name],float)
                            heel_toe_residual.append(float(nearest_segment(p[None],uv[foot],np.asarray([pair]))[0][0]))
                            ax.scatter(p[0],p[1],s=8,c=metadata['colors']['source_heel_toe'],edgecolors='#4a3900',linewidths=.25,zorder=4)
                shell_stats,heel_stats=stats(shell_residual),stats(heel_toe_residual)
                subtitle=('old open' if mode=='open' else 'fitted')+' | shell med/P95 '+fmt(shell_stats['median'])+'/'+fmt(shell_stats['p95'])+' px'
                subtitle+='\nheel/toe median '+fmt(heel_stats['median'])+f' px (n={heel_stats["n"]})'
                record['modes'][mode]=dict(visible_shell_edges=edge_packets,source_shell_error_px=shell_stats,source_heel_toe_error_px=heel_stats,
                    predicted_points_outside_crop=int(np.sum((uv[...,0]<lo[0])|(uv[...,0]>=hi[0])|(uv[...,1]<lo[1])|(uv[...,1]>=hi[1]))))
            ax.set_title(f'{point/12800:.2f}s | PTS {point}\n'+subtitle,fontsize=10)
            ax.set_xlim(lo[0]-.5,hi[0]-.5);ax.set_ylim(hi[1]-.5,lo[1]-.5);ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            if row==1:ax.set_xlabel('native source u (px)',fontsize=8)
            if col==0:ax.set_ylabel('native source v (px)',fontsize=8)
        metadata['frames'].append(record)
    fig.suptitle('Exact original source | cyan: explicit shell | magenta: both finite blade edges | small gold dots: original outer-boot heel/toe',fontsize=11)
    image_path=guard_project_write(output/'source_shoes.png');record_path=guard_project_write(output/'source_shoes.json')
    fig.savefig(image_path,dpi=160);plt.close(fig)
    metadata['image']=dict(path=str(image_path),sha256=digest(image_path))
    record_path.write_text(json.dumps(serializable(metadata),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return dict(image=str(image_path),source_record=str(record_path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--baseline',help='default: <root>/fullbody_1102/full/result.npz')
    parser.add_argument('--source-figures',action='store_true',help='only draw the two exact source-foot comparisons; requires optional PyAV/matplotlib')
    args=parser.parse_args(argv)
    if args.source_figures:
        print(json.dumps(source_figures(args.root,args.output),ensure_ascii=False),flush=True);return
    result=run(args.root,args.output,args.baseline)
    print(json.dumps(dict(report=str(Path(args.output)/'boot_report.json'),summary=str(Path(args.output)/'results.txt'),
        source_frames=result['scope']['source_frames'],refined_frames=result['scope']['new_body_refined_frames']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
