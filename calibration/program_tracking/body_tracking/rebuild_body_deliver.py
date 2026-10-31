"""Save final body parameters in the programme and factual comparison in Data."""
from pathlib import Path
import argparse,json,ast
import numpy as np
from .rebuild_prepare import load,digest,native_metric_binding
from .rebuild_body_audit import MODES
from .rebuild_body_geometry import DOF_NAMES
from .fullprogramme_body_sequence import SOURCE_INDICES


def checked(path):
    absolute=Path(path).resolve()
    if len(str(absolute).encode('utf-16-le'))//2>=260 or any(len(s.encode('utf-16-le'))//2>255 for s in absolute.parts):raise ValueError('OneDrive path length')
    return absolute


def deliver(root,project):
    root,project=Path(root),Path(project);p=load(root/'input.npz');meta=json.loads((root/'input.json').read_text(encoding='utf8'))
    scale_binding=native_metric_binding(meta['scene_path'],meta['snapshot']);scale=float(p['metres_per_scene_unit'])
    if not np.isclose(scale,scale_binding['metres_per_scene_unit'],rtol=1e-12,atol=0):raise ValueError('Input uses an obsolete metric conversion; preserve it and create a corrected delivery first')
    comparison=json.loads((root/'comparison.json').read_text(encoding='utf8'));z={m:load(root/m/'result.npz') for m in MODES}
    programme=project/'events/ankara_2026/programmes/okada_mei_sp'
    parameters=checked(programme/'body_s01_parameters_20261029.npz');binding=checked(programme/'body_s01_comparison_20261029.json')
    if parameters.exists() or binding.exists():raise FileExistsError('Final delivery already exists; preserve before a new revision')
    arrays=dict(pts=p['pts'],source_time_base=np.array([1,12800]),source_lower_body_inferred=p['lower_body_inferred'],
        pose_prior_axis_angle=p['pose_axis_angle'],root_Q_prior=p['raw_root_prior_Q_world'],shared_betas=p['shared_betas'],body_scale=p['body_scale'],
        metres_per_scene_unit=p['metres_per_scene_unit'],position_knots=z['fixed']['position_knots'],local_dof_names=np.asarray(DOF_NAMES),
        frozen_rig=p['rig'],frozen_controls=p['controls'],first_control_time=p['first_control_time'],control_dt=p['control_dt'])
    for mode in MODES:
        for name in ['x','P_world_m','Q_world','delta_local_rad']:arrays[mode+'__'+name]=z[mode][name]
    np.savez_compressed(parameters,**arrays)
    rows=[]
    for mode in MODES:
        v=comparison['models'][mode];fit=json.loads((root/mode/'result.json').read_text(encoding='utf8'))
        rows.append(dict(variant=mode,body_pixel_median=v['source_body_reprojection_px']['median'],body_pixel_p95=v['source_body_reprojection_px']['p95'],
            diagnostic_body_pixel_median=v['diagnostic_window_body_reprojection_px']['median'],diagnostic_body_pixel_p95=v['diagnostic_window_body_reprojection_px']['p95'],
            nonfit_arm_pixel_median=v['nonfit_elbows_wrists_reprojection_px']['median'],ground_change_median_conditional_m=v['ground_displacement_vs_fixed_conditional_m']['median'],
            ground_change_p95_conditional_m=v['ground_displacement_vs_fixed_conditional_m']['p95'],ground_change_maximum_conditional_m=v['ground_displacement_vs_fixed_conditional_m']['maximum'],
            local_correction_p95_deg=fit['local_correction_absolute_deg']['p95'],local_correction_maximum_deg=fit['local_correction_absolute_deg']['maximum']))
    summary=dict(schema='rinkspace.body-fixed-camera-comparison.v1',event='ankara_2026',programme='okada_mei_sp',shot='s01',
        source_seconds=meta['source_seconds'],native_exposures=len(p['pts']),lower_body_inferred_frames=int(p['lower_body_inferred'].sum()),
        tested_visible_body_frames=int((~p['lower_body_inferred']).sum()),output_definition='SMPL-X pelvis centre P and its world vertical projection G=(P_x,P_y,0)',
        conclusions_zh=[
            '局部姿态可修正组的身体二维重投影有小幅收益；不能据此认定真实三维位置更准确。',
            '加入接冰后，模型刀刃的冰下状态明显减少，但诊断留出窗口尾部和未拟合手臂误差变差；不选为最终准确轨迹。',
            '骨盆竖直投影对身体高度敏感。两组的位移是模型对照差异，不能解释为纠正了同等大小的真实误差。',
            '相机、场景、原始取点和隔壁联合拟合输入均未由本实验修改；没有用历史展示轨迹或网络世界平移。'],
        comparison=rows,metric_status='project_uniform_conversion',metres_per_scene_unit=scale,metric_binding=scale_binding,
        assumptions_zh=['共享体型按既有约1.53米身高审计缩放，未在原片独立量身。','鞋体有效垫高9厘米、五点弧形刀刃端部抬高12毫米为条件几何，未逐鞋测量。'],
        contact_zh='四个粗原图刀刃区域用于接冰拟合，两个时刻及其前后100毫秒身体像素作开发诊断留出。非穿冰先验覆盖可见全身帧；留出接触高度不是独立三维真值。其余时刻接触状态未知，允许接冰滑行，不约束水平速度。',
        time_model='native complete camera predicted-row implicit projection plus quadratic intra-exposure articulated body; exact-row diagnostic in comparison.json',
        validation_status_zh='工程收敛及导数测试通过；画面等待用户审阅；无最终独立三维真值，未生成完整视频。',
        parameters_file=str(parameters),parameters_sha256=digest(parameters),data_root=str(root),data_report=str(root/'comparison.json'),
        source_binding=meta,model_sha256=digest(root/'compact_smplx.npz'),
        files=dict(images=str(root/'review_images.json'),summary_plot=str(root/'diagnostic_summary.png'),
            detail_worst=str(root/'detail_421888.png'),detail_contact=str(root/'detail_461056.png')),
        reproduction=dict(module='calibration.program_tracking.body_tracking.rebuild_body_fit',order=['fixed','pose','pose_contact'],
            warm_start=['none','fixed/result.npz','pose/result.npz'],input='input.npz',input_sha256=digest(root/'input.npz'),body_motion='exposure',spline_spacing_s=.16),
        tests=['test_rebuild_exposure_body: closed-form row solution, explicit out-of-image flag, true native body/contact directional derivative'],
        full_video_created=False)
    binding.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    # Publication-style static plots use actual source exposure times.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,1,figsize=(12,11),gridspec_kw={'height_ratios':[1.6,1,1]},layout='constrained')
    times=p['pts']/12800;visible=~p['lower_body_inferred'];colors=['#d69324','#28a956','#347ed4'];labels=['A 固定局部姿态','B 局部姿态修正','C 修正＋接冰']
    scene=load(root/'review_scene.npz')['xyz']*scale;info=json.loads((root/'review_scene.json').read_text(encoding='utf8'))
    for cut in info['cuts']:
        vv=scene[cut['begin']:cut['end']]
        axes[0].plot(vv[:,0],vv[:,1],color='#d8d8d8',lw=.5,zorder=0)
    for i,mode in enumerate(MODES):
        G=z[mode]['G_world_m'].copy();P=z[mode]['P_world_m'].copy();G[~visible]=np.nan;P[~visible]=np.nan
        axes[0].plot(G[:,0],G[:,1],color=colors[i],lw=1.4,label=labels[i])
        axes[1].plot(times,P[:,2],color=colors[i],lw=1.25)
        distance=np.linalg.norm(z[mode]['G_world_m']-z['fixed']['G_world_m'],axis=1);distance[~visible]=np.nan
        axes[2].plot(times,distance,color=colors[i],lw=1.25)
    axes[0].set(aspect='equal',xlabel='世界 X（米）',ylabel='世界 Y（米）',title='S01 固定相机：三组骨盆竖直投影，灰色为同一冻结场景');axes[0].legend(loc='upper center',ncol=3,fontsize=9)
    axes[1].set(ylabel='骨盆高度（米）',title='全组保持同一体型、骨长、相机及场景')
    axes[2].set(ylabel='相对 A 的水平位移（米）',xlabel='官方原片源时间（秒）',title='位移是模型对照差异，不是真实定位误差')
    for ax in axes[1:]:
        ax.axvspan(times[0],meta['visibility_review']['first_reviewed_full_body_pts']/12800,color='#bbb',alpha=.25,label='下肢推断')
        ax.set_xlim(times[0],times[-1]);ax.grid(alpha=.2)
        for t in [32.06,40.]:ax.axvspan(t-.1,t+.1,color='#9988cc',alpha=.15)
    fig.suptitle('米制采用项目统一换算；人体身高和鞋刀几何仍有近似；没有独立三维真值',fontsize=12)
    fig.savefig(root/'diagnostic_summary.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(binding=str(binding),parameters=str(parameters),parameter_bytes=parameters.stat().st_size),ensure_ascii=False),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);a=ap.parse_args();deliver(a.root,a.project)
