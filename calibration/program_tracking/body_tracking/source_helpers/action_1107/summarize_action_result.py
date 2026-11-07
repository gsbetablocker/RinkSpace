from pathlib import Path
import json,sys
import numpy as np
from calibration.program_tracking.body_tracking.rebuild_prepare import digest
ROOT=Path(r"D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/trajectory_action_1107/action_completion_1107"); BASE=ROOT.parent
STAGE=ROOT/sys.argv[1]
PARENT=BASE/'axel_axis_joint_1107'
FINAL=STAGE/'final/fitted/result.npz'
PRE=ROOT/'prepared_travel'
def read(path):return json.loads(path.read_text('utf8'))
def shape_report(name,path):
    z=read(path/'final_action_factors.json')['source_and_physics']
    runner=z['runner_support_diagnostics']
    t=np.array(runner['source_pts'])/12800
    ft=np.array(runner['foot'])
    pitch=np.asarray(runner['fore_aft_pitch_degrees'])
    height=np.asarray(runner['runner_height_m'])
    u=np.asarray(runner['actual_u'])
    out={}
    for id,lo,hi,f in [('LSp_left',156.54,165.30,0),('CCoSp_left',172.08,179.02,0),('CCoSp_right',179.26,184.52,1)]:
        v=(t>=lo)&(t<=hi)&(ft==f)
        out[id]=dict(support_foot=['left','right'][f],rows=int(v.sum()),
            actual_u_min_max=[float(np.min(u[v])),float(np.max(u[v]))],
            lowest_point_near_runner_endpoint=int(np.sum((u[v]<.01)|(u[v]>.99))),
            abs_pitch_p95_deg=float(np.percentile(abs(pitch[v]),95)),
            runner_reference_height_m_median=float(np.median(height[v])),
            runner_reference_height_m_min=float(np.min(height[v])))
    q=np.flatnonzero((t==181.84)&(ft==1))[0]
    out['181.84']=dict(pitch_deg=float(pitch[q]),u=float(u[q]),reference_height_m=float(height[q]))
    out['step_factors']={k:dict(samples=len(v['source_pts']),signed_rows=int(np.count_nonzero(v['source_travel_sign'])),
        across_abs_p95_mps=float(np.percentile(np.abs(v['across_runner_mps']),95)),hypotheses=v['hypotheses'])
        for k,v in z.items() if isinstance(v,dict) and 'across_runner_mps' in v}
    out['flight_factors']={k:dict(com_exposures=len(v['source_pts']),momentum_exposures=len(v['momentum_pts']))
        for k,v in z.items() if isinstance(v,dict) and 'momentum_pts' in v}
    out['source_turns']={k:dict(reverse_intervals=v['reverse_intervals'],
        integrated_signed_yaw_proxy_turns=v['integrated_signed_yaw_proxy_turns'])
        for k,v in z.items() if isinstance(v,dict) and 'reverse_intervals' in v}
    return out
before=read(ROOT/'solve_v1/replay_comparison.json')['before']
after=read(STAGE/'replay_comparison.json')['after']
old=shape_report('before',PARENT);new=shape_report('after',STAGE)
d=read(PRE/'source_evidence.json')
ledger=Path(d['source_episode_review']['source_action_ledger'])
episodes=d['source_episode_review']['episodes']
with np.load(PARENT/'final/fitted/result.npz') as a, np.load(FINAL) as b:
    for field in ('pts','time_base','metres_per_scene_unit','shared_body_scale','shared_betas'):
        np.testing.assert_array_equal(a[field],b[field])
    pts=b['pts'];t=pts/12800.
    untouched=(t<=44.96)|((t>=77.30)&(t<=78.76))
    stable={}
    for field in ('P_world_mid','Q_world_mid','native22_world_mid','skate_R_world_mid'):
        error=float(np.max(np.abs(a[field][untouched]-b[field][untouched])))
        assert error<2e-8
        stable[field]=error
    P=np.linalg.norm(a['P_world_mid']-b['P_world_mid'],axis=-1)
    max_change={'m':float(P.max()),'seconds':float(t[np.argmax(P)])}
    body_fields=(a['COCO_uv'],b['COCO_uv'])
    changed_shots=np.asarray(b['source_shot_ids'])
    cut_pts=pts[np.flatnonzero(changed_shots[1:]!=changed_shots[:-1])+1].tolist()
mesh=read(STAGE/'mesh_crossings_fixed_checks.json')['evaluated']
compact=ROOT/'collision_v3'
dense=[]
for key in ('before','after'):
    rows=read(compact/f'collision_timeline_{key}.json')
    dense.append(dict(samples=len(rows),crossing_samples=sum(r['triangles']>0 for r in rows),
                      triangle_pairs=sum(r['triangles'] for r in rows)))
stages={key:read(ROOT/key/'run_summary.json')['additional_accepted_steps']
        for key in ('solve_v1','solve_v2','collision_v3','s09_v4') if (ROOT/key/'run_summary.json').is_file()}
phase_coverage=[]
for row in episodes:
    lo,hi=(row['start_pts'],row['end_pts_inclusive'])
    phase_coverage.append(dict(id=row['episode_id'],kind=row['kind'],native_exposures=int(np.sum((pts>=lo)&(pts<=hi))),
        source_evidence_status=row.get('evidence_status'),explicit_evidence_pts=row.get('evidence_pts',[]),
        foot=row.get('foot'),role=row.get('role','inherited')))
summary=dict(state=dict(frames=len(t),seconds=[float(t[0]),float(t[-1])],time_base=[1,12800],
        cut_pts=cut_pts,unchanged_s01_and_2a=stable,max_root_change_vs_axel=max_change),
    phases=dict(ledger=str(ledger),sha256=digest(ledger),episodes=phase_coverage),
    source_band_median_px=dict(before=before['band_median_px'],after=after['band_median_px']),
    source_band_exposures=dict(before=before['native_bands'],after=after['native_bands']),
    runner_geometry=dict(before=old,after=new),step_phase_input=read(ROOT/'step_source_travel.json'),
    fit_steps_by_stage=stages,exact_mesh_comparison=dict(all_39=mesh,selected_dense=dense),
    support_and_structure=dict(initial_violation=before['gate'],final_violation=after['gate'],
        ice_acceptance='finite soft residual; no ice feasibility gate',
        maximum_violation_scope='foot cavity and collar engineering margin'),
    fs_external=read(BASE/'fs_action_evaluation/metrics.json'),
    playback=read(STAGE/'player_validation.json'),
    accuracy_accepted=False)
(STAGE/'action_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
b=before['band_median_px'];f=after['band_median_px']
key=str(round(181.84*12800))
run0=old['CCoSp_right'];run1=new['CCoSp_right']
line='\n'.join([
    '# 全节目动作联合修正',
    '',
    f'实际保留 {sum(stages.values())} 步，全节目 {len(t)} 曝光，九镜头。使用同一尺度、冰场和原生相机。满意 S01 和 2A 在新候选内通过状态一致性检查。',
    '',
    f'- LSp 支撑带超差中位数 {b["LSp"]:.2f} → {f["LSp"]:.2f} px，CCoSp {b["CCoSp"]:.2f} → {f["CCoSp"]:.2f} px。181.84 秒右刀 {before["native_bands"][key]["outside_px"]:.2f} → {after["native_bands"][key]["outside_px"]:.2f} px。',
    f'- CCoSp 换右足后的刀具纵倾绝对值 P95 为 {run0["abs_pitch_p95_deg"]:.1f}° → {run1["abs_pitch_p95_deg"]:.1f}°。参考面高度中位数 {100*run0["runner_reference_height_m_median"]:.1f} → {100*run1["runner_reference_height_m_median"]:.1f} cm。近端点 {run0["lowest_point_near_runner_endpoint"]} → {run1["lowest_point_near_runner_endpoint"]} / {run1["rows"]}，均为模型诊断。',
    f'- 同一组 39 项网格精查中，相交由 {mesh["axel"]["confirmed_model_crossing_checks"]} 项降到 {mesh["candidate"]["confirmed_model_crossing_checks"]} 项。专项 36 个帧及中点样本由 {dense[0]["crossing_samples"]} 降到 {dense[1]["crossing_samples"]} 项。',
    f'- 步法滑行关系共 {sum(v["samples"] for v in new["step_factors"].values())} 个实际侧滑样本，其中 {sum(v["signed_rows"] for v in new["step_factors"].values())} 个有源约束前后滑。右后滑换左前滑有原片条件支持，内外刃和具体步法名留待核对。',
    f'- 最大足腔／鞋口结构违例 {1000*before["gate"]["max_violation_m"]:.2f} → {1000*after["gate"]["max_violation_m"]:.2f} mm。使用已有有限冰面项。',
    '',
    '全节目原片逐曝光解码核对，动作源读图用全片顺序图与重点密集曝光。最终接冰、深度、未消除的网格相交和多视角世界精度仍待验证。FS 的原实际独立验证保留失败证据。',
    '',
    f'[实际接线、逐动作结果与未通过项]({(STAGE/"action_receipt.json").as_posix()})',
    f'[源动作台账]({ledger.as_posix()})',
    f'[播放器]({(Path(r"D:/OneDrive/codex_projs/RinkSpace")/"launchers/START_3D_PROGRAMME_ACTION_1107.cmd").as_posix()})',
])
(STAGE/'RESULT_ZH.md').write_text(line+'\n',encoding='utf8')
print(line,flush=True)
