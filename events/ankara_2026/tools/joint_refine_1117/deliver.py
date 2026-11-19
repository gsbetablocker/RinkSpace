"""Export one common event snapshot and four exact-clock camera bindings."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from shared import (EVENT, LIBRARY, OUT, PROGRAMMES, REGISTRATIONS, arrays,
    numeric_sha, read, registration, safe_path, save, sha, write)
from review import camera_state, native_for, sample
from solve import stats


def file_ref(path):
    return dict(path=str(path), sha256=sha(path))


def geometry(native):
    doc=read(native.scene_path); ice={}
    for p in doc['paths']:
        key=p['id']; actual=native.paths[key]
        if actual['layer']!='ice':continue
        code=p.get('generated_code')
        label=key
        if code is not None and 100<=code<132:
            house,ring=divmod(code-100,4)
            label=('F' if house<4 else 'R')+str(house%4+1)+':'+('center','inner','middle','outer')[ring]
        ice[label]=dict(path=key,kind=actual['kind'],parameters=list(actual['parameters']))
    blocks={k:dict(kind=b['kind'],values=native.parameter_state[k].tolist())
            for k,b in native.parameter_state.blocks.items() if b['available']}
    ads={}
    for number in range(1,27):
        key=f'installation/Ad{number}/shared_placement'
        value=native.parameter_state[key]
        ads['Ad'+str(number)]=dict(placement_block=key,board_station=float(value[0]),
            vertical_coordinate=float(value[1]),unwrapped_rotation_radians=float(value[2]))
    size3=native.parameter_state['artwork_pair/Ad3-Ad16/size']
    size16=native.parameter_state['size/Ad16_free_1117']
    return dict(coordinates='rink length L=1; board centre and axes define the gauge; not verified metres',
        board=list(native.scene['board']),ice=ice,advertisements=ads,parameter_blocks=blocks,
        Ad16_to_Ad3_dimension_ratio=np.exp(size16-size3).tolist(),
        interpretation='A fitted dimension ablation, not independent proof of different physical paper sizes')


def camera_export(program,base,digest,revision,*,version='joint_free_1117',prepared=None):
    rec=registration(program);old=Path(rec.get('camera',rec.get('camera_parameters'))['path'])
    a=arrays(old);setup=read((OUT/'prepared' if prepared is None else Path(prepared))/'setup.json')
    shots={n:s for n,s in setup['shots'].items() if s['program']==program}
    for name,s in shots.items():
        sid=s['shot'];pts=base[name+'_pts'];controls=base[name+'_controls'];rig=base[name+'_rigs']
        if 'shot_ids' in a:
            i=a['shot_ids'].tolist().index(sid)
            lo,hi=a['control_offsets'][i:i+2];u,v=a['source_offsets'][i:i+2]
            np.testing.assert_array_equal(a['source_pts'][u:v],pts)
            np.testing.assert_array_equal(a['local_exposure_seconds'][u:v],base[name+'_times'])
            a['controls'][lo:hi]=controls;a['rig_by_source_pts'][u:v]=rig
        else:
            np.testing.assert_array_equal(a[sid+'_pts'],pts)
            np.testing.assert_array_equal(rig,np.broadcast_to(rig[0],rig.shape))
            a[sid+'_controls']=controls;a[sid+'_rig']=rig[0]
            times=a.get(sid+'_exposure_times',(pts-pts[0])/12800.)
            np.testing.assert_array_equal(times,base[name+'_times'])
        np.testing.assert_array_equal(base[name+'_active_folds'],np.arange(5,dtype=np.int8))
    a.update(scene_values_sha256=digest,layout_revision=revision,
        active_observation_folds=np.arange(5,dtype=np.int8))
    target=safe_path(EVENT/'programmes'/PROGRAMMES[program]/('camera_'+version+'.npz'))
    save(target,**a)
    # Verify exact exported values, including legacy exposure-varying rigs.
    saved=arrays(target)
    for key in a:np.testing.assert_array_equal(saved[key],a[key])
    return target


def compare(candidate):
    current=arrays(OUT/candidate/'source_qa/curve_errors.npz')
    original=arrays(OUT/'original/material_source_qa/curve_errors.npz')
    setup=read(OUT/'prepared/setup.json');result={};shot_rows={};tails={}
    for p in PROGRAMMES:
        old=[];new=[];worst=[]
        for name,s in setup['shots'].items():
            if s['program']!=p:continue
            a=arrays(OUT/'prepared'/(name+'.npz'));use=a['fold']<5
            e0=original[name+'_error'][use];e1=current[name+'_error'][use]
            old.extend(e0);new.extend(e1)
            admitted=np.flatnonzero(use)
            for i in admitted[np.argsort(e1)[-5:]]:
                worst.append(dict(shot=name,pts=int(a['pts'][i]),seconds=float(a['pts'][i]/12800.),
                    origin_row=int(a['origin_row'][i]),path=str(a['path'][i]),
                    original_px=float(original[name+'_error'][i]),candidate_px=float(current[name+'_error'][i])))
            shot_rows[name]=dict(original=stats(e0),candidate=stats(e1),
                source_seconds=[float(a['pts'].min()/12800.),float(a['pts'].max()/12800.)])
        result[p]=dict(original={**stats(np.array(old)),'maximum':float(max(old))},
                       candidate={**stats(np.array(new)),'maximum':float(max(new))})
        tails[p]=sorted(worst,key=lambda x:x['candidate_px'],reverse=True)[:5]
    return dict(programmes=result,shots=shot_rows,largest_admitted_residuals=tails,
        metric='nearest native curve distance, original 1920x1080 source pixels',
        evidence_role='same admitted development rows for both states; no independent final accuracy certification')


def markdown_report(candidate,report,programmes,coordinates):
    labels={'jin':'金书贤','okada':'冈田芽衣','chen':'Chen Xi Jin','nayeon':'Nayeon Ko'}
    geo=read(coordinates)
    lines=['# 安卡拉四节目共同几何与相机复核（2026-11-17）','',
        '已保存一套共同的 395 值几何和四套相机。范围为全部 62 段已分配机位镜头、33,688 个原生 50 fps 曝光；Nayeon 的 S01 按原机位确认范围排除。四节目使用完全相同的 922 条路径，不再分别带入 Chen 的局部几何覆盖或 Nayeon 的 pitch 变体。',
        '', '这是供复核和继续使用的改善候选，未替代历史人工确认，`user_visual_acceptance=false`。四节目在同批准入原片点上的综合距离均下降；不能据此宣称所有局部线条、每一帧或真实三维尺寸均已验收。',
        '', '## 参数与完整视频','',
        '[赛事登记](joint_free_1117.json) · [实际几何坐标与全部参数块](joint_free_1117_coordinates.json) · [共同数值快照](field/joint_free_1117.npz) · [结构](field/free_ad16_1117.json)',
        '', '登记同时绑定结构、数值哈希、原生库、原片、观测包和各节目相机。加载结构时必须同时加载数值快照；结构 JSON 中的初始值不是本次拟合的最终值。相机 NPZ 保留原始归档格式、PTS、曝光时间和既有逐曝光 rig 差异。',
        '', '| 节目 | 原片首末帧时间（秒） | 镜头 / 帧 | 完整本地视频 | 参数登记 |',
        '|---|---|---:|---|---|']
    for p,ref in programmes.items():
        r=read(ref['path']);v=r['full_review'];lo,hi=v['source_seconds']
        lines.append(f'| {labels[p]} | {lo:.3f}–{hi:.3f} | {len(r["shots"])} / {v["frames"]:,} | [PotPlayer 回放](<{v["video"]["path"]}>) | [相机及来源](../programmes/{PROGRAMMES[p]}/joint_free_1117.json) |')
    lines+=['','视频本地 0 秒对应各行原片首帧，按原速播放；有原声的节目保留源时钟对齐音频，金书贤原片没有音轨。左侧显示完整共同网格与该曝光当前有效的几何取点，右侧显示同一完整模型。没有几何取点的帧照常渲染并记缺失，不从邻帧补点；帧间跟踪因子另见审计。',
        '', '## 相同原片点上的比较','',
        '两边均使用最终准入的同一批 564,286 行原始像素；基线是三个暂定最终登记及所引用 Nayeon 聊天的 whole_quality_1117。距离为原生投影有限曲线到像素的最近距离，单位为 1920×1080 原片像素。这些旧训练/检查材料已经反复参与调参，均属开发证据，不是独立最终测试集。',
        '', '| 节目 | 原版 P90 → 当前 | 原版 RMS → 当前 | 当前最大值 |',
        '|---|---:|---:|---:|']
    for p,v in report['programmes'].items():
        a,b=v['original'],v['candidate']
        lines.append(f'| {labels[p]} | {a["p90"]:.3f} → {b["p90"]:.3f} | {a["rms"]:.3f} → {b["rms"]:.3f} | {b["maximum"]:.3f} |')
    lines+=['',f'[逐镜头、最大残差及其源行](<{OUT/candidate/"comparison.json"}>)。Nayeon 九段镜头的 P90 均低于原版；其余节目有少数镜头小幅退步，详见逐镜头记录。大残差尾部仍存在，不能把综合分位数下降解释成全部取点身份正确。',
        '', '## Nayeon 的关键变化与未解决处','']
    critical=read(OUT/candidate/'critical_source_evidence.json')['cases']
    for key,title in [('nayeon_s06_Ad16','S06 Ad16'),('nayeon_s04_Ad3','S04 Ad3'),('nayeon_s08_B2','S08 B2 蓝线')]:
        q=critical[key];a,b=q['original'],q['candidate']
        lines.append(f'- {title}：保留稠密原始时间覆盖的 {a["n"]:,} 个源点，P90 {a["p90"]:.3f} → {b["p90"]:.3f} px，RMS {a["rms"]:.3f} → {b["rms"]:.3f} px。')
    lines+=['',
        'Ad16 在原片 109.620 秒的面板右侧从约 20–21 px 偏差缩小到约 2.8–3.5 px，静帧可见面板和字形尺度贴合改善。Ad16 与 Ad3 保留相同归一化图案，但放开实例尺寸后，拟合宽比为 1.016539、高比为 1.005679；这只是联合拟合假设，尚不能证明实物确有该尺寸差。',
        '', 'Ad3 在 61.6–62.1 秒快速摇镜时仍有表观横向不重合和明显拖影，侧边不能确定为单一清晰像素线；原稠密观测的高分位误差仍约 12 px，没有记作已修复。S08 蓝线仅有小幅分位数改善，RMS 仍有退步，不能宣称该处全面改善。没有把宽范围色彩搜索得到的阴影或邻近颜色当成新蓝线点。',
        '', f'S08 原片 143.500 秒的 B2 在无遮挡左侧仍有约 4–7 px 向色带下侧偏的目视现象。另做了仅调整共同 B2 位置/斜率、冻结全部相机的[四节目消融](<{OUT/"blue_shared_ablation/ablation_check.json"}>)：该段稠密 P90 只从 5.42 降到 5.23 px，同时其余三节目 B2 指标小幅退步，因此没有采用该微调，也没有把剩余偏差解释成已经解决。',
        '', f'[关键原片证据](<{OUT/candidate/"critical_source_evidence.json"}>) · [静帧结论及局部退步](<{OUT/"visual_findings.json"}>)。同一原片点的时间覆盖增加不构成独立真值。',
        '', '## 放开的几何与保留的结构','',
        '本次取消旧百分比偏差盒、不同靶圈等半径、等行距、F/R 对齐、基准线平行、Lane 平行/等宽及两端尺寸镜像。八组靶圈中心及各可识别层半径、A1–A4、长线、球门线、球门区和梯形接点均可独立调整；围挡五个既有形状参数参与联合拟合，26 个广告位置参与共同校准。',
        '', '仍保留正尺寸、材料层嵌套、单个靶圈各层同心、线/圆/椭圆类型、Lane 公共端点、梯形连接球门线与围挡、球门区中心在球门线中点及连续围挡。围挡仍属于对称超椭圆族；没有改成任意不对称自由曲面。八个最内白纽扣半径因身份不足而冻结。相机与几何仍耦合，以下数值不能换算为已验证米制。',
        '', '坐标以冰场长度 L=1 为基准。以下是当前原生实际坐标，不是结构初值。',
        '', '| 圈 | 中心 x | 中心 y | 半径 |','|---|---:|---:|---:|']
    for label in ('ice.A1:centreline','ice.A2:centreline','ice.A3:centreline','ice.A4:centreline'):
        actual=next(v for k,v in geo['ice'].items() if v['path'] in native_path_ids(label,candidate))
        x,y,_,radius,*_=actual['parameters']
        lines.append(f'| {label[4:6]} | {x:.8f} | {y:.8f} | {radius:.8f} |')
    lines+=['','| 靶圈 | 中心 x | 中心 y | center / inner / middle / outer 半径 |',
        '|---|---:|---:|---|']
    for house in [f'{side}{i}' for side in ('F','R') for i in range(1,5)]:
        params=[geo['ice'][house+':'+r]['parameters'] for r in ('center','inner','middle','outer')]
        lines.append(f'| {house} | {params[0][0]:.8f} | {params[0][1]:.8f} | '+ ' / '.join(f'{v[3]:.8f}' for v in params)+' |')
    lines+=['', '两端球门线的两个端点、球门区中心/双轴/角度、四条梯形线端点、围挡参数和全部广告安装坐标均见[实际坐标 JSON](joint_free_1117_coordinates.json)。A2、A3 比 A4 直接约束少；Ad15、Ad16 目前主要由金书贤和 Nayeon 提供直接点，Ad18 主要由金书贤和冈田提供。多节目共用模型不表示每个实体都已取得四节目独立约束。',
        '', '## 源取点与相机问题修复','',
        '- 首轮源审核修改 4,800 行：含实际双侧漆边替换旧中点、无法分辨的红漆两侧、同曝光板脚以上的冰面错标签和已审阅梯形线替换。随后先隔离 8 行，再在 4 张原片材料卡中隔离 26 个具体错层或身份未明点；原始 UV、源 PTS、旧角色和原因均保留。没有凭一个错点删除整圈或整镜头。',
        '- 冈田旧帧间跟踪没有使用人体遮挡排除，抽查发现跟踪落在手、裙边或腰臀轮廓。复用全部 8,329 帧的同源遮挡记录，按实际 LK 支撑窗口排除与前景/未知区域相交的 31,860 / 206,202 个原始候选，再重拟合九镜头和三个机位；未用残差作为排除依据。掩码仍是来源候选，并非完整遮挡真值。',
        '- 条件跟踪深度改为从未移动的实际左帧像素，经当前原生畸变和当前围挡/冰面求交重绑；该深度是相机相关辅助量，不作为独立三维观测。',
        '- 金书贤 S14 的旧安装种子含落入围挡内的支点，改用同镜头已有可行安装种子后拟合真实源点。所有最终曝光另查支点在冰场外、高度为正及非零安装杆臂。光学和安装校准共享每个“节目、机位”的增量，保留历史逐曝光差异，未宣称跨节目只有一个已测量的物理相机。',
        '', f'[首轮源审核](<{OUT/"source_admission_red_only/audit.json"}>) · [8 点记录](<{OUT/"source_admission_final/audit.json"}>) · [26 点记录](<{OUT/"source_admission_material/audit.json"}>) · [冈田前景排除](<{OUT/"source_flow_visibility/binding.json"}>) · [帧间源运动审计](<{OUT/candidate/"source_motion_audit.json"}>)',
        '', '## 数值检查与交付边界','',
        '四份完整视频逐帧解码核对 native PTS，无漏帧、无无效源相机。每镜头首中末抽查约 501 个原生顶点，CPU 与 CUDA 同域投影最大差小于 1e-6 px，完整同域状态一致。这证明实现与导出一致，不证明实际场地误差小于该值。播放器视频按完整保存的样条支撑域投影，已保留旧有限支撑渲染尝试及失败原因。',
        '', '自由几何、原生接口与广告实例尺寸相关 15 项测试通过；旧默认盒约束保留，只有显式自由几何选项启用新规则。求解各主要阶段均以 `cost_progress_stagnation_without_convergence` 停止，`numerically_converged=false`，没有将停滞写成收敛。早期漂移、显存不足和被否决的源提案均保留。',
        '', f'[测试记录](<{OUT/"free_geometry_tests.txt"}>) · [最终 Nayeon 求解](<{OUT/candidate/"solve.jsonl"}>) · [冈田源可见性求解](<{OUT/"okada_source_visibility/solve.jsonl"}>) · [共同几何末轮](<{OUT/"ad3_joint_geometry/solve.jsonl"}>)',
        '', '本次完整渲染并核对全部源帧，人工/图像代理检查的是选定静帧和源材料卡，未把它冒充整片连续人工验收。金书贤与冈田的渲染未绑定人物前景 matte（跟踪准入的掩码另用）；Chen 与 Nayeon 使用既有同源 matte。图像弱纹理、模糊侧边、局部广告小幅退步和材料身份未明仍在报告内。独立最终精度、米制尺度及用户整片验收均未通过或未进行。',
        '', f'选定运行：`{candidate}`。未改动旧暂定最终登记、原片或源时钟；未提交或推送 Git。','']
    target=safe_path(EVENT/'scene/joint_free_1117.md')
    temp=safe_path(target.with_suffix('.md.writing'))
    temp.write_text('\n'.join(lines).replace('\\','/'),encoding='utf-8');temp.replace(target)
    return target


def native_path_ids(label,candidate):
    # Labels are serialized in the scene; no extra native instance is needed.
    field=Path(str(camera_state(candidate)['scene_path']))
    return {p['id'] for p in read(field)['paths'] if p.get('label')==label}


def export(candidate,reuse_review=None):
    setup=read(OUT/'prepared/setup.json');base=camera_state(candidate)
    native=native_for(candidate,'nayeon');values=native.parameter_state.values
    for folder in (OUT/candidate/'source_qa',OUT/'original/material_source_qa'):
        for name,digest in read(folder/'metrics_binding.json')['packets'].items():
            assert digest==sha(OUT/'prepared'/(name+'.npz'))
    digest=numeric_sha(values);revision=native.parameter_state.layout['layout_revision']
    reference=sample(native);xyz=reference['xyz'];ids=[p['id'] for p in reference['cuts']]
    for p in PROGRAMMES:
        other=native_for(candidate,p);world=sample(other)
        assert ids==[r['id'] for r in world['cuts']]
        np.testing.assert_array_equal(world['xyz'],xyz)
    snapshot=EVENT/'scene/field/joint_free_1117.npz'
    save(snapshot,scene_values=values,scene_values_sha256=digest,layout_revision=revision)
    coordinates=EVENT/'scene/joint_free_1117_coordinates.json'
    write(coordinates,geometry(native))
    report=compare(candidate);write(OUT/candidate/'comparison.json',report)
    scene_entry=EVENT/'scene/joint_free_1117.json'
    binding=dict(version_id='joint_free_1117',status='four_programme_joint_refinement_review_candidate',
        snapshot=file_ref(snapshot),structure=file_ref(native.scene_path),
        native_library=file_ref(LIBRARY),value_sha256=digest,layout_revision=revision,
        scene_parameters_active=False,coordinate_scale='rink length = 1; no verified metre scale')
    programmes={};mounts={}
    b=np.asarray(native.scene['board'])
    for p,directory in PROGRAMMES.items():
        names=[n for n,s in setup['shots'].items() if s['program']==p]
        folder=OUT/candidate/'review'/p
        if not (folder/'coverage.json').exists() and reuse_review:
            for prior in ([reuse_review] if isinstance(reuse_review,str) else reuse_review):
                alternative=OUT/prior/'review'/p
                if (alternative/'coverage.json').exists():
                    folder=alternative
                    break
        coverage=read(folder/'coverage.json')
        rb=read(folder/'binding.json')
        assert rb['state_sha256']==sha(rb['state'])
        rendered=arrays(rb['state'])
        np.testing.assert_array_equal(rendered['scene_values'],base['scene_values'])
        assert str(rendered['scene_path'])==str(base['scene_path'])
        # A camera-only repair in another programme does not invalidate this
        # programme's identical finished native-PTS rendering.
        for name in names:
            for suffix in ('_pts','_times','_controls','_rigs','_active_folds'):
                np.testing.assert_array_equal(rendered[name+suffix],base[name+suffix])
        assert rb['scene_sha256']==sha(native.scene_path)
        for name,d in rb['observation_packets'].items():assert sha(OUT/'prepared'/(name+'.npz'))==d
        assert coverage['validation']['decoded_frames']==coverage['frames']
        camera=camera_export(p,base,digest,revision)
        for name in names:
            r=base[name+'_rigs'];pivot=r[:,15:18]
            u=np.maximum(0.,(abs(pivot[:,0])-(b[0]-b[2]))/b[2])
            v=np.maximum(0.,(abs(pivot[:,1])-(b[1]-b[3]))/b[3])
            margin=u**b[4]+v**b[4]-1
            valid=(margin>=-1.e-10)&(pivot[:,2]>0)&(np.linalg.norm(r[:,18:21],axis=1)>1.e-12)
            valid&=(pivot[:,0]<0) if setup['shots'][name]['camera_index']==2 else (pivot[:,1]<=-b[1]+1.e-10)
            if not valid.all():raise ValueError((name,'invalid installation',int((~valid).sum())))
            mounts[name]=dict(exposures=len(r),invalid=0,minimum_board_margin=float(margin.min()),
                minimum_pivot_height=float(pivot[:,2].min()),distinct_legacy_rig_rows=len(np.unique(r,axis=0)))
        rec=dict(schema='rinkspace.programme_joint_refinement.v1',event_id='ankara_2026',
            programme_id=directory,status='review_candidate',user_visual_acceptance=False,
            previous=file_ref(REGISTRATIONS[p]),source=registration(p)['source'],
            geometry_binding=binding,camera=file_ref(camera),shots=names,
            fit_observations={n:file_ref(OUT/'prepared'/(n+'.npz')) for n in names},
            fit_roles=[0,1,2,3,4],role_meaning='previously reused development observations',
            full_review=dict(video=file_ref(coverage['video']),coverage=file_ref(folder/'coverage.json'),
                render_state=file_ref(rb['state']),selected_geometry_and_programme_camera_exactly_equal=True,
                source_seconds=coverage['source_seconds'],frames=coverage['frames'],
                native_time_base=[1,12800],fps=50,source_audio=coverage['source_audio']),
            comparison=report['programmes'][p],
            camera_policy='shared calibration increments per assigned camera; legacy per-exposure rig differences retained',
            exclusions='Nayeon s01 remains outside assigned-camera scope; all declared assigned shots included')
        target=EVENT/'programmes'/directory/'joint_free_1117.json';write(target,rec)
        programmes[p]=file_ref(target)
    write(OUT/candidate/'physical_mounts.json',mounts)
    write(scene_entry,dict(schema='rinkspace.event_joint_refinement.v1',event_id='ankara_2026',
        version_id='joint_free_1117',status='source_review_candidate',user_visual_acceptance=False,
        geometry_binding=binding,coordinates=file_ref(coordinates),programmes=programmes,
        selected_run=candidate,checkpoint=file_ref(OUT/candidate/'state.npz'),
        comparison=file_ref(OUT/candidate/'comparison.json'),
        source_admission=file_ref(OUT/'source_admission_material/audit.json'),
        previous_source_admission=file_ref(OUT/'source_admission_final/audit.json'),
        source_flow_visibility=file_ref(OUT/'source_flow_visibility/binding.json'),
        source_motion_audit=file_ref(OUT/candidate/'source_motion_audit.json'),
        critical_nayeon_evidence=file_ref(OUT/candidate/'critical_source_evidence.json'),
        visual_findings=file_ref(OUT/'visual_findings.json'),
        original_comparison_binding=file_ref(OUT/'original/material_source_qa/metrics_binding.json'),
        original_source_admission=file_ref(OUT/'source_admission_red_only/audit.json'),
        original_packets=str(OUT/'prepared_before_source_qa'),
        retained_constraints=read(native.scene_path)['ice_metric_freedom']['retained'],
        removed_constraints=read(native.scene_path)['ice_metric_freedom']['removed'],
        limitations=['No independent frozen final test set; old checks were reused for development',
            'Eight innermost white-button radii remain frozen due unresolved source identity',
            'Ads 15,16,18 have sparse direct constraints; fitted dimensions are hypotheses',
            'Nayeon Ad3 rapid-pan vertical boundaries remain blurred; inherited dense-side errors remain around 12px P90',
            'Some selected ad edges show small local regressions; aggregate improvement does not certify every frame',
            'Numerical fits ended in cost progress stagnation, without stationarity convergence',
            'Source optics and geometry remain coupled; normalized coordinates are not verified metres',
            'Full source frames rendered and decoded; selected stills reviewed, not full human continuous playback'],
        tests=file_ref(OUT/'free_geometry_tests.txt'),physical_mounts=file_ref(OUT/candidate/'physical_mounts.json')))
    report_path=markdown_report(candidate,report,programmes,coordinates)
    entry=read(scene_entry);entry['report']=file_ref(report_path);write(scene_entry,entry)
    print('EXPORTED',scene_entry)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True)
    ap.add_argument('--reuse-review',nargs='+');args=ap.parse_args()
    export(args.candidate,args.reuse_review)
