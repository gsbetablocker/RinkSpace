"""Publish factual joint-run status at the programme's existing result entry."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,html
from .io_utils import read_json,write_json,file_sha256

def report(project,run):
 project=Path(project).resolve();run=Path(run).resolve();out=project/'events/ankara_2026/programmes/okada_mei_sp/fit_joint9'
 for f in (out/'status.json',out/'review.html'):
  if len(str(f).encode('utf-16-le'))//2>=260 or any(len(p.encode('utf-16-le'))//2>255 for p in f.parts):raise ValueError('OneDrive path exceeds physical Windows policy')
 def load(rel, details=False):
  p=run/rel
  if not p.is_file():return None
  ref={'path':str(p),'sha256':file_sha256(p)}
  if details:ref['data']=read_json(p)
  return ref
 scope=load('programme_scope.json',True);prep=load('prepare_summary.json');eq=load('material_compile_v3/s01_equivalence.json');roles=load('roles/manifest.json');mat=load('material_compile_v3/manifest.json');stats=load('joint_packet_v3/packet.json');s01=load('s01_joint_replay_v8/state/value_report.json')
 candidates=sorted((p for p in run.glob('fit_geometry_v*_quality/quality.json')),key=lambda p:int(p.parent.name.split('_v')[1].split('_')[0]))
 latest=load(str(candidates[-1].relative_to(run)),True) if candidates else None
 admission=load('admission_v4/manifest.json');dense_admission=load('dense_admission_v1/manifest.json');visual=load('fit_geometry_v4_review/visual_observations.json');definition_audit=load('definition_audit_v1/manifest.json');source_audit=load('source_definition_audit_v1/manifest.json');withdrawal=load('inner_board_withdrawal_v1.json');reference=load('reference_8a28.json');aliases=load('s01_print_crosswalk_v1/equivalence.json')
 recall=load('outer_source_recall_v2.json');source_statistics_report=load('joint_source_statistics_report_v3.json');valid_seed=load('fit_phr_v13/initial_audit.json');outer=load('board_outer_full_v1/manifest.json');current_replay=load('accepted_replay_v11_1/state/before/value_report.json');current_visual=load('fit_phr_v11_accepted1_review/visual_observations.json');end_seed=load('end_optical_seed_v3/patch.json');s01_source=load('s01_statistics_source_audit_v7/manifest.json');display=load('joint_display_QA_v1/receipt.json')
 selection_file=out/'selection.json';selection={'path':str(selection_file),'sha256':file_sha256(selection_file),'data':read_json(selection_file)} if selection_file.is_file() else None
 dense=[]
 for name in ('dense_s03_v1','dense_remaining_v1'):
  p=run/name
  if p.is_dir():
   ready=(p/'manifest.json').is_file();ref=load(name+('/manifest.json' if ready else '/progress.json'),True)
   dense.append({'name':name,'complete':ready,'receipt':ref})
 status={'schema':'rinkfit.joint9.delivery_status.v1','updated_utc':datetime.now(timezone.utc).isoformat(),'status':'in_progress','joint_fit_complete':False,'whole_programme_accuracy_verified':False,'project_root':str(project),'run_directory':str(run),'programme_scope':scope,'preparation':prep,'s01_geometry_import_equivalence':eq,'material_compilation':mat,'source_statistics':stats,'latest_actual_fit_quality':latest,'selected_source_admission':admission,'dense_source_admission':dense_admission,'actual_source_visual_review':visual,'navigation_8a28_reference':reference,'s01_added_material_alias_equivalence':aliases,'frame_role_protection':roles,'s01_joint_replay':s01,'dense_runs':dense,'current_user_selection':selection,'actual_current_initializer_native_replay':current_replay,'actual_current_initializer_visual_review':current_visual,'current_physical_outer_full_scan':outer,'outer_source_certificate_recall':recall,'joint_source_statistics_audit_report':source_statistics_report,'numerically_valid_rebased_seed_native_audit':valid_seed,'recalled_outer_candidates_are_not_fit_approved':True,'end_optical_coordinate_seed':end_seed,'s01_full_source_statistics_audit':s01_source,'same_model_dual_view_display_check':display,'source_semantic_definition_audit':definition_audit,'full_source_semantics_audit':source_audit,'withdrawn_source_membership':withdrawal,'pre_audit_admission_and_fit_are_historical_only':True,'all_current_fit_sources_semantically_revalidated':False,'measurement_definitions':{'width_midline':['B1','B2','C','C0','A1','A2','A3','A4'],'target_physical_interfaces':['F1','F2','F3','F4','R1','R2','R3','R4']},'current_limitations':['另两台机位已开始真实源像素优化，仍有明显误差；主机位宽镜头与共同场地尚须完整释放联调','源统计保留原像素量化噪声假设，未宣称独立校准或正式统计最优','保护窗口尚待历史来源审计与冻结模型验证','稠密候选须补全逐帧身份闭合、数值统计及正式准入','完整联合结果、网格叠加和全范围验证尚未完成'],'video_policy':'仅在全范围真实改善后生成同一模型完整视频；本地路径和原片时间供PotPlayer播放'}
 out.mkdir(parents=True,exist_ok=True)
 previous=read_json(out/'status.json') if (out/'status.json').is_file() else {}
 for key in ('current_joint_progress','s01_baseline_boundary','acceleration_progress','storage_progress','current_limitations'):
  if key in previous:status[key]=previous[key]
 # Sealed source statistics and consumed solver statistics have distinct scopes.
 control_file=run/'statistics_latest_control.json'
 if control_file.is_file() and status.get('current_joint_progress',{}).get('statistics_progress'):
  live=read_json(control_file).get('board_unique_development_statistics',{})
  partition_ref=live.get('prior_partition')
  if partition_ref:
   partition_file=Path(partition_ref.get('file',partition_ref.get('path','')))
   if not partition_file.is_absolute():partition_file=control_file.parent/partition_file
   if file_sha256(partition_file)!=partition_ref['sha256']:raise ValueError('Changed sealed statistics partition')
   partition=read_json(partition_file)
   if not partition.get('zero_overlap_and_full_6006_partition'):raise ValueError('Unverified statistics coverage partition')
   sealed=int(partition['numeric_frozen_unique_native_PTS']);pending=int(partition['pending_unique_native_PTS'])
   if sealed+pending!=6006:raise ValueError('Incomplete statistics partition')
   statp=status['current_joint_progress']['statistics_progress']
   statp.update(sealed_exposures=sealed,remaining_exposures=pending,resource_resume_manifest={'path':str(partition_file),'sha256':partition_ref['sha256']})
   # Never copy this sealed count into the active solver's consumed count.
 status['receipt_payloads_inlined']=False
 def compact(value):
  if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if not (k=='data' and 'path' in value and 'sha256' in value)}
  if isinstance(value,list):return [compact(v) for v in value]
  return value
 write_json(out/'status.json',compact(status))
 def a(ref,label):
  return ('<a href="'+html.escape(Path(ref['path']).as_uri(),quote=True)+'">'+html.escape(label)+'</a>') if ref else html.escape(label+'（尚未生成）')
 rows=''.join('<tr><td>'+s['id']+'</td><td>'+s['rig_id']+'</td><td>'+f"{s['source_seconds'][0]:.3f}–{s['source_seconds'][1]:.3f}"+'</td><td>'+str(s['native_frame_count'])+'</td></tr>' for s in scope['data']['shots'])
 drows=''
 for d in dense:
  r=(d['receipt'] or {}).get('data',{});c=r.get('counts',{});drows+='<tr><td>'+d['name']+'</td><td>'+('本分片已完成当前轮读取' if d['complete'] else '运行中')+'</td><td>'+str(c.get('development_frames',0))+'</td><td>'+str(c.get('actual_current_profile_rows',0))+'</td><td>'+str(c.get('development_frames_with_support',0))+'</td></tr>'
 qrows=''
 if latest:
  for sid,r in latest['data']['by_shot'].items():qrows+='<tr><td>'+sid+'</td><td>'+str(r['active_rows'])+'</td><td>'+str(r['invalid_rows'])+'</td><td>'+str(round(r['median_px'],3))+'</td><td>'+str(round(r['max_px'],3))+'</td></tr>'
 fitsection='<section><h2>实际联合后端的历史数值代（语义自查前）</h2><p>'+a(latest,'同一模型完整源误差')+' · '+a(visual,'47张原帧叠加实看记录')+'。当前为分阶段数值启动，首镜头和主机位共享块仍有临时固定，误差不符合交付要求；原始W与初始化几何W分别记账。</p><table><thead><tr><th>镜头</th><th>当前实际求值源数</th><th>局部辅助量重解后的严格无效</th><th>中位像素误差</th><th>最大像素误差</th></tr></thead><tbody>'+qrows+'</tbody></table><p>'+a(admission,'选取源完整准入账本')+' · '+a(dense_admission,'原生视频稠密源完整门禁')+'。稠密源仍在接入实际联合参数问题，未凭源数量宣称全帧拟合通过。</p><p>'+a(reference,'8a28历史状态与相机档案SHA核实')+'：该状态用于初始化参考，首镜头原完整修复导出保留，当前数值候选的变化及回归另行报告。</p></section>'
 title='冈田芽衣短节目：三机位九镜头联合拟合'

 progress=status.get('current_joint_progress',{})
 acceleration=status.get('acceleration_progress',{})
 storage=status.get('storage_progress',{})
 raw=progress.get('current_raw_error',{})
 statp=progress.get('statistics_progress',{})
 statsection='<p>开发曝光的整图条件统计已封存 '+str(statp.get('sealed_exposures',0))+'/6006；剩余 '+str(statp.get('remaining_exposures',6006))+'。'+a(statp.get('resource_resume_manifest'),'无重叠的曝光续跑账本')+'。原始599个自动固定角点已全部建立来源处置：425个通过材料与算子资格，174个因当前可见支撑或算子失败保留失败记录，另2个人工点单列。'+a(progress.get('fixed_entity_disposition_coverage'),'完整角点来源账本')+'。425个合格点中'+str(statp.get('fixed_entities_known_conditional_noise',0))+'个已有条件噪声统计，'+str(statp.get('fixed_entities_unknown_noise',0))+'个尚未知；其中'+str(statp.get('fixed_entities_currently_in_active_solver',0))+'个合格角点已接入当前完整求解。主问题实际消费'+str(statp.get('sealed_exposures_consumed_in_current_solver',0))+'个整曝光统计，后续签证批次仍单独保留；来源与统计资格不能当作拟合已经通过。</p>' if statp else ''
 futureice=progress.get('future_confirmed_ice_sources',{})
 icesource_section=('<p>'+a(futureice.get('whole_exposure_union_plan'),'新冰面来源及整曝光联合统计计划')+'：'+str(futureice.get('source_qualified_rows',0))+'条新读数已通过来源核对，其中宽线两侧界面'+str(futureice.get('paired_flank_rows',0))+'条，R3靶圈实体界面'+str(futureice.get('single_target_interface_rows',0))+'条。涉及'+str(futureice.get('existing_exposures_requiring_joint_replacement',0))+'个曝光的完整联合统计已完成；旧围挡成员的物理证书被召回。已从完整Cov取得新416条及原B2两界面合418条的精确协方差边缘并重新白化；合法冰面资格保留，尚未进入主求解，不增加冰场形状或硬关系。</p>') if futureice else ''
 rawsection='<p>'+html.escape(raw.get('scope_label',''))+' '+a(progress.get('latest_same_model_source_metrics'),'最近已完成复算的逐源误差（以报告内模型SHA为准）')+'：固定当前材料站位、仅局部重解RS的有效投影 '+str(raw.get('valid','?'))+'/'+str(raw.get('active','?'))+'；按实际测量算子计算的原生像素RMS '+str(round(raw.get('RMS_px',0),3))+'、P95 '+str(round(raw.get('p95_px',0),3))+'、最大 '+str(round(raw.get('max_px',0),3))+'。首镜头Ad2全部623个实体角点RMS '+str(round(raw.get('Ad2_entity_all623_RMS_px',0),3))+'。这项诊断不证明完整RS唯一性或物理支持通过，原E回归尚未恢复。</p>' if raw else ''
 identity_recalled=bool(progress.get('active_source_identity_recalled'))
 board_intro=('6006个开发曝光曾自动产出431195条精确去重的围挡候选；原片复查已查实红色广告图案被误签为黄带下沿。当前完整源证书已召回，求解入口拒绝继续消费；全部曝光正按实际材料、实例与边界拓扑逐ID重新核实，不能沿用旧资格布尔。正确边界仍为蓝带上边界、黄带下边界并排除倒影。所有原点、错误签证、统计及失败状态完整保留。' if identity_recalled else '当前输入的逐ID物理资格、来源及统计范围以本页最新输入和回执为准。围挡上沿取蓝带上边界、下沿取黄带下边界并排除倒影；候选数量和历史数值签证不能代替材料身份。')
 identity_banner=('当前完整源的物理资格已召回：原片证实存在红色广告误签为黄带下沿。已阻止继续消费该输入，正在修复整条准入链并复核全部曝光，同时查修Ad4等系统性物理参数问题；主任务继续推进。下列旧代误差与签证不能证明源正确。' if identity_recalled else '已撤回的内沿、栏杆、红色印刷错源及其签证完整保留；当前输入须独立核实材料身份，历史回执不能证明新输入正确。')
 currentsection='<section><h2>当前完整源联合求解</h2><p>'+board_intro+'</p><p>'+a(progress.get('latest_accepted_complete_state'),'最近完整数值态（尚未验收）')+' · '+a(progress.get('latest_full_stage_checkpoint'),'完整续算检查点')+'：'+html.escape(progress.get('phase_description_zh','当前继续真实联合优化；物理、像素及独立KKT尚未通过。'))+'。'+a(progress.get('latest_rejected_global_quality'),'最近全局候选回归诊断')+' · '+a(progress.get('latest_rejected_global_complete_checkpoint'),'保留的全局失败完整检查点')+'。</p><p>'+a(progress.get('next_complete_source_packet'),'当前完整源问题')+' · '+a(progress.get('next_complete_source_QA'),'原UV／W与全部Q一致性检查')+'：'+str(progress.get('active_full_source_rows',613441))+'条活跃源，包含补齐曝光、Ad4 R已有材料的有限可见轮廓和后续固定实体角点；未知噪声源保留实际E、RS、有限域及原片查询约束，尚不能宣称全源统计最优。</p><p>'+a(acceleration.get('joint_actual_end_to_end_timing'),'真实生产求解分阶段计时')+'：GPU原生UV导数、标量导数缓存和CUDA线性求解已接入；当前完整接受步约16秒，线性求解约3.3秒。'+a(acceleration.get('cuda_linear_complete_problem_probe'),'CUDA CGNR完整问题对照')+'已通过同初始导数及实际后端核验：单步线性求解55.32→3.17秒，Ceres总耗时160.96→80.19秒，已接回生产；早期生产接受步约14–15秒，当前完整问题约16秒。上述对照为同一状态的一步，不能据此宣称整轮收敛；当前优先修复错源准入和已定位的真实物理参数漂移。'+a(storage.get('system_memory_and_pagefile_audit'),'系统内存与分页实测')+'记录了主求解及统计作业退出前后的提交内存和分页变化，磁盘变化不能全部归因于本次结果文件。后续使用来源引用、共享数组与流式统计，不逐迭代复制完整状态或将大Jacobian写盘。</p><p>'+a(progress.get('temporal_boundary_comparison'),'首帧及135–140帧原始边界与预测同图')+'：首帧过宽的RS导数证书已修正，并通过相同投影复核；135–140帧广告和材料误差仍未闭合，最终同态仍需完整重算。</p><p>首镜头原导出保存完好；当前数值候选曾修改主机位光学初值及辅助站位，不能宣称与原导出逐值相同，回归尚待关闭。'+a(progress.get('source_segment_audit'),'U／K、Ad4和W／Y逐段可见来源')+'。不迁入等径、等距、对称等53条关系；C0与A1–A4仍为宽线中线，F1–F4与R1–R4为靶圈实体界面。</p></section>'
 cause_section=('<section><h2>当前最大误差与已查实原因</h2><p>'+a(progress.get('actual_error_hotspots'),'全九镜头按逐源、原E及时间／材料排序')+'：同时列最坏点与合计误差，避免少量极值遮住系统性问题。</p><table><thead><tr><th>原片位置</th><th>已查实问题</th><th>当前处置与证据</th></tr></thead><tbody><tr><td>s02 48.54秒</td><td>红色广告印刷被签为黄带下沿</td><td>'+a(progress.get('frozen_false_certificate_source_chain'),'冻结源码、原RGB和签证链')+'；输入已召回，整族逐ID复核</td></tr><tr><td>s01 44.54秒 Ad4</td><td>既有字形仿射参数漂移；最大311.53像素</td><td>'+a(progress.get('Ad4_existing_parameter_root_cause'),'同帧相机／物理参数交叉对照')+'；修复既有块及有限站位，尚未采纳候选</td></tr><tr><td>s04 71.52秒</td><td>所查点为真实黄带；当前物理态明显偏离</td><td>'+a(progress.get('historical_physical_state_same_source_contrast'),'同源早期4–7像素与当前123–126像素对照')+'；保留正确源，修相机／场地状态</td></tr></tbody></table><p>'+a(progress.get('actual_Source_Query_RS_kernel_audit'),'C++／独立投影、RS及导数专项核对')+'排除了所查围挡案例的投影及Query实现错误；该结论不扩展为所有材料后端已验证。</p></section>') if progress.get('actual_error_hotspots') else ''
 ad2factor=progress.get('Ad2_geometry_camera_factorial')
 if ad2factor:
  factor_file=Path(ad2factor['path'])
  if file_sha256(factor_file)!=ad2factor['sha256']:raise ValueError('Changed Ad2 same-source diagnosis')
  factor=read_json(factor_file);factor_rows=''
  names={'current_geometry_current_camera':'当前安装＋当前相机','original_geometry_current_camera':'原安装＋当前相机','current_geometry_original_camera':'当前安装＋原相机','original_geometry_original_camera':'原安装＋原相机'}
  for key,label in names.items():
   v=factor['cases'][key];parts=v['time_partition'];factor_rows+='<tr><td>'+label+'</td><td>'+str(round(v['all623']['RMS_px'],3))+'</td><td>'+str(round(parts['opening_before135']['RMS_px'],3))+'</td><td>'+str(round(parts['pulse135_140']['RMS_px'],3))+'</td><td>'+str(round(parts['later_after140']['RMS_px'],3))+'</td></tr>'
  cause_section+='<section><h2>Ad2全部623角点：安装与相机分离诊断</h2><p>'+a(ad2factor,'同一原始点、独立真实RS四组合回执')+'。开头268点、135–140帧12点、后段343点均保留。单独恢复安装会使全体误差加重；原安装与原相机共同恢复显著较好，仍有388点超原E，只是数值恢复候选。</p><table><thead><tr><th>参数组合</th><th>全部623 RMS</th><th>开头 RMS</th><th>135–140 RMS</th><th>后段 RMS</th></tr></thead><tbody>'+factor_rows+'</tbody></table><p>单位为原生像素。当前和原组合均未验收，未覆盖主状态、原始取点、允许域或正式续算元状态。</p></section>'
 ad4repair=progress.get('Ad4_repair_candidate_metrics')
 if ad4repair:
  cause_section+='<section><h2>Ad4既有参数修复候选</h2><p>'+a(progress.get('Ad4_existing_parameter_repair'),'生产后端／GPU导数对照、参数与有限站位补丁')+'：保持当前相机、广告安装与全部源点，只修复5个既有字形仿射块并重算同一有限路径站位和RS行时，182点RMS从188.693降至3.823像素，P95为7.614、最大16.299，仍80点超原E。未接入已召回输入，未宣称拟合完成。</p><p>历史回执证实早期接入保持原字形，漂移随后续joint接受步累积。当前Phase I共同预算允许约311.445像素，182行中181行无像素罚损，且此阶段尚未采用W精度目标。实际生产print导数、GPU缓存、CPU Jet和独立有限差分相符，未发现该组参数索引或导数误绑。完整正确源范围的目标、允许域及分时段真实误差仍须在采纳前重算；不增加形状、先验或几何硬关系。</p></section>'
 roleprogress=progress.get('board_physical_requalification_progress',{})
 if roleprogress:
  rc=roleprogress.get('canonical_status_counts',{})
  cause_section+='<section><h2>围挡全部源点复查</h2><p>'+a(progress.get('board_physical_requalification_recipe'),'固定材料角色与原49Q算子复核配方')+'：按原生RGB实际蓝带上缘／黄带下缘、相同曝光与材料次序逐ID核实；不改旧均值或Query，不通过历史签证自动准入。当前快照 '+str(roleprogress.get('closed_exposures',0))+' / '+str(roleprogress.get('requested_exposures',6006))+' 曝光；精确去重源中 '+str(rc.get('pass',0))+' 双层通过、'+str(rc.get('wrong_role',0))+' 错误角色、'+str(rc.get('unresolved',0))+' 来源不足待解决。'+a(dict(path=roleprogress['path']),'运行进度')+'会继续变化；尚未完成整个6006范围，不能据此恢复旧packet。</p><p>'+a(progress.get('board_legacy_mask_provenance'),'全部既有遮挡mask来源审计')+'另行区分认证身体框、独立实例和派生反射排除；有材料依赖的57曝光补做只含身体与固定字幕区域的因果核对。</p></section>'
 if progress.get('Ad7_10_diagnostic_metrics'):
  cause_section+='<section><h2>Ad7、Ad10相机运动漂移</h2><p>'+a(progress.get('Ad7_10_source_and_motion_diagnosis'),'全部12,228原实体源诊断')+'：所查原片中实体边界可见；只更换共同相机光学／机头参数几乎不改善，恢复首镜头运动数值后明显下降，说明沿围挡方向运动漂移是主要原因之一。完整源和既有材料候选重算后，Ad7的4,147点RMS约42.33→4.48，Ad10的8,081点约19.56→2.85像素；共4,128点仍超原E，Ad7另镜头31点仍约24.08像素。候选未接入当前完整态。</p><p>'+a(progress.get('full_numeric_donor_invariants'),'完整数值候选的原物理几何与现允许域核对')+' · '+a(progress.get('full_numeric_donor_source_diagnosis'),'首镜头全源及425固定点真实RS交叉检查')+'。全部旧相机辅助坐标不能盲继承，必须在新模型下完整重算；后段固定点仍有明显误差，不作全范围改善结论。</p></section>'
 text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f7f8;color:#243848}main{max-width:1080px;margin:auto;padding:30px}section{background:white;padding:22px;margin:18px 0;border:1px solid #dce5eb;border-radius:10px}h1{font-size:28px}h2{font-size:21px}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #e1e9ec}a{color:#087d7f}code{overflow-wrap:anywhere}.status{background:#fff1d8;padding:13px;border-left:4px solid #bc812a}.muted{color:#657786;font-size:14px}</style><main><p><a href="../../../../../START_HERE.html">RinkSpace 唯一结果入口</a></p><h1>'+title+'</h1><p class="status">正在推进。尚未完成联合拟合、全帧准确性验证与完整叠加视频；下列准备与接入结果不能代替验收。</p><p class="status">'+identity_banner+'</p><p>官方50帧单节目原片，原生PTS 1/12800，处理20.020–188.680秒，共8,434帧。全场回放数值只作为有来源的导航初值。</p><p>按用户最新决定，本次不迁入新提出的共圆径、等距、对称或Lane硬关系，以首镜头最新修复导出和未作这些投影的实际候选继续。<a href="selection.json">当前选择与基准SHA</a>。</p><section><h2>当前可审查结果</h2><p>'+a(definition_audit,'已完成的110350选取源及准入源逐行定义预检')+' · '+a(source_audit,'全部源分支定义审计与撤回账本')+' · '+a(withdrawal,'错误内沿逐ID撤回及实际packet命中核验')+'。定义预检通过不等于RGB物理身份通过；实际重测、统计和求解接入继续进行。</p><ul><li>'+a(prep,'完整数据准备')+'：110,350条新观测，201个原生选取帧，原始UV、人工点击和删除记录保留。</li><li>'+a(aliases,'新增印刷材料的接入等价回执')+'：该次接入保留295个原材料、12,672个世界样本差0及原参数块；这不代表后续候选一直未漂移。</li><li>'+a(mat,'共同材料几何')+'：单一场地和连续围挡chart；新增材料保持有限物理支撑，缺少身份来源的材料继续自动解决。</li><li>'+a(roles,'源家族角色锁定与保护窗口')+'：620个待审保留帧和560个保护帧不进入训练或穿越追踪。</li><li>'+a(stats,'当前原生像素统计')+'：相关W来自同一当前均值，量化噪声假设单列，未宣称独立校准。</li></ul><p class="muted">C0、A1–A4属于有宽度的圆形线，取材料两边的宽度中线；F1–F4、R1–R4才是靶圈，读取已定义的实体材料界面。</p></section>'+currentsection+cause_section+icesource_section+statsection+rawsection+fitsection+'<section><h2>完整处理范围</h2><table><thead><tr><th>镜头</th><th>机位</th><th>源秒</th><th>帧数</th></tr></thead><tbody>'+rows+'</tbody></table></section><section><h2>当前原生帧稠密读取</h2><p>光流只提供搜索位置。当前RGB重新读取边界、两侧材料线或角点；人物由当前原生图像的YOLO和SAM2保守排除。没有源支持的帧记为待重捕，不填预测零误差；保护帧等待冻结后验证。</p><table><thead><tr><th>分片</th><th>状态</th><th>当前开发帧</th><th>真实读取行</th><th>有支撑帧</th></tr></thead><tbody>'+drows+'</tbody></table><p>这些是源候选记录，尚须身份闭合、数值统计和准入，不能当作完成的联合拟合。</p></section><section><h2>运行和来源</h2><p><code>'+html.escape(str(run))+'</code></p><p><a href="status.json">机器可读状态和文件SHA</a> · <a href="../fit_v11/audit_1018.html">首镜头既有修复审阅</a></p><p class="muted">更新UTC：'+html.escape(status['updated_utc'])+'</p></section></main></html>'
 (out/'review.html').write_text(text,encoding='utf8')
 entry=project/'START_HERE.html';s=entry.read_text(encoding='utf8');begin='<!-- OKADA_JOINT9_BEGIN -->';end='<!-- OKADA_JOINT9_END -->';call=begin+'<p class="callout"><strong>2026-10-19：三机位九镜头联合拟合正在推进。</strong>8,434帧完整数据准备、首镜头最新完整状态接入与自动源审查已推进；当前已召回含红色印刷错签黄带的完整源资格，正全面修复取点准入与物理参数漂移；合法冰面证据及全部状态保留，尚未验收。<a href="events/ankara_2026/programmes/okada_mei_sp/fit_joint9/review.html">联合运行结果、来源和当前限制</a>。下方早期暂停及等待人工审核文字保留为历史，当前按最新授权自动推进。</p>'+end
 if begin in s:s=s[:s.index(begin)]+call+s[s.index(end)+len(end):]
 else:s=s.replace('<!-- CURRENT_TASK_BEGIN -->','<!-- CURRENT_TASK_BEGIN -->\n'+call,1)
 entry.write_text(s,encoding='utf8')
 print(json.dumps({'review':str(out/'review.html'),'status':str(out/'status.json'),'run':str(run)},ensure_ascii=False));return status
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',required=True);a=p.parse_args();report(a.project,a.run)
