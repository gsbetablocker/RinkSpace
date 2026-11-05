from pathlib import Path
import json,hashlib,collections,html,subprocess,datetime
from PIL import Image,ImageDraw,ImageFont
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');D=Path(r'D:/CodexData/RinkSpace_Data');O=D/'runs/body_s01_20261029/blade_observability_1105'
def read(name):return json.loads((O/name).read_text(encoding='utf-8-sig'))
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ref(f):
 f=Path(f);return dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)
assembly=read('factor_assembly_audit.json');identity=read('identity_audit.json');contact=read('dual_source_manifest.json');early=read('visual_early.json');late=read('visual_late.json')
# Keep original reviewer records. Normalization never changes foot identity or observation admission.
visual=[];unassigned=[]
for frame in early['rows']+early.get('extra_rows',[]):
 for foot in frame['feet']:
  r={k:v for k,v in frame.items() if k!='feet'};r.update(foot);r['foot']=foot['foot_proposal'];r['review_group']='early';visual.append(r)
for r0 in late['records']+late.get('extra_rows',[]):
 r=dict(r0);r['foot']=r.get('foot_roi_proposal');r['review_group']='late'
 if r['foot'] is None:unassigned.append(r)
 else:visual.append(r)
assert len(visual)==126 and len(unassigned)==2
for r in visual:
 r['source_image_axis_visible']=r['blade_axis_observable']=='visible'
 r['visible_axis_is_undirected_and_not_verified_material_identity']=True
 r['source_review_creates_no_factors']=True
mapped={(r['pts'],r['foot']):r for r in visual};assert len(mapped)==len(visual)
reviewed_pts=sorted({r['pts'] for r in visual+unassigned});assert len(reviewed_pts)==64
contactbypts={r['pts']:r for r in contact['frames']}
# Read-only extra contact sheet: two verified native PNGs and source pane from the existing composite.
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',18)
supp=Image.new('RGB',(1920,400),'white');sd=ImageDraw.Draw(supp)
extras=[]
for i,pt in enumerate([1044480,1044736,1196800]):
 r=next(r for r in visual+unassigned if r['pts']==pt)
 if r.get('source_frame'):
  image_path=Path(r['source_frame']);im=Image.open(image_path).convert('RGB');assert sha(image_path)==r['source_frame_sha256']
  provenance='verified native source PNG'
 else:
  image_path=Path(r['reviewed_image_paths'][0]);im=Image.open(image_path).convert('RGB')
  assert im.size==(7680,1080) and sha(image_path)==r['composite_png_sha256'];im=im.crop((0,0,1920,1080));provenance='existing composite: original_source pane only'
 supp.paste(im.resize((640,360)),(640*i,40));sd.text((640*i+8,3),f'{pt/12800:.2f}s | PTS {pt}',fill='black',font=font)
 sd.text((640*i+8,22),provenance,fill='black',font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12))
 extras.append(dict(pts=pt,source_seconds=pt/12800,image_path=str(image_path),provenance=provenance))
supp_path=O/'key_supplemental_source_contacts.png';supp.save(supp_path)
# Avoid mixing direct source regions, flow selection, and conditional contact projections.
windows=[(77.6,78.2),(81.2,82.0),(84.8,85.1),(87.6,90.0),(92.4,93.6)]
candidate_summaries={};window_rows=[]
for name,c in assembly['candidates'].items():
 active=[r for r in c['rows'] if r['actual_fit_index_0based'] is not None]
 pairs=collections.defaultdict(set)
 for r in active:
  if r['kind']=='blade_endpoint_region':pairs[(r['pts'],r['foot'])].add(r['original_row'].get('blade_endpoint'))
 candidate_summaries[name]={k:c[k] for k in ['total_document_rows','fit_role_rows','review_role_rows','optimization_enabled_explicit_true','optimization_enabled_explicit_missing','optimization_enabled_effective_true','actual_self_rows','actual_report_self_rows','objective_residual_scalar_rows','actual_fit_groups','active_blade_material_identity_flags']}
 candidate_summaries[name]['direct_endpoint_labels_both_present_pts']=sum(s=={'front','rear'} for s in pairs.values())
 candidate_summaries[name]['both_labels_do_not_certify_source_axis_or_material_correspondence']=True
 held=[r for r in c['rows'] if r['actual_report_index_0based'] is not None and r['actual_fit_index_0based'] is None]
 candidate_summaries[name]['review_only_active_rows']=len(held);candidate_summaries[name]['review_only_active_unique_pts']=len({r['pts'] for r in held})
 for lo,hi in windows:
  rows=[r for r in active if lo<=r['seconds']<=hi]
  window_rows.append(dict(candidate=name,window_seconds=[lo,hi],active_shoe_blade_source_rows=len(rows),groups=dict(collections.Counter(f"foot{r['foot']}:{r['kind']}" for r in rows)),source_reviewed_pts=[pt for pt in reviewed_pts if lo<=pt/12800<=hi],meaning='counts are actual SupportImageFactors source rows, not body COCO or physical priors'))
# The source evidence metadata is shown per foot per PTS without inventing image geometry.
exposures=[]
for pt in reviewed_pts:
 feet=[]
 for foot in [0,1]:
  v=mapped.get((pt,foot))
  vis=v if v else dict(pts=pt,foot=foot,anatomical_identity='unknown',identity_confidence='unknown',boot_heel_visible='unknown',boot_toe_visible='unknown',blade_front_visible='unknown',blade_rear_visible='unknown',blade_axis_observable='unknown',end_on='unknown',occluded='unknown',motion_blur='unknown',shadow_reflection_confusion='unknown',orientation_unknown_reason='Two source white boots seen, but cannot assign anatomical side or foot0/foot1. See unassigned_source_objects.',source_only_note='93.50s source-pane review preserves unassigned physical objects.')
  obs={}
  for name,c in assembly['candidates'].items():
   rows=[]
   for r in c['rows']:
    if r['pts']!=pt or r['foot']!=foot:continue
    rr=r['original_row'];kind=r['kind']
    category='A_direct_boot_material_or_contour_region' if kind=='boot_marker_region' else ('B_direct_blade_endpoint_or_midpoint_region' if kind in ['blade_endpoint_region','blade_midpoint_region'] and r['optimization_enabled_explicit'] is not False else ('D_contact_region_distinct_from_material' if kind=='support_contact_region' else 'disabled_or_derived_region_not_direct_material'))
    rows.append(dict(id=r['id'],category=category,semantic=r['semantic'],kind=kind,role=r['role'],optimization_enabled_explicit=r['optimization_enabled_explicit'],optimization_enabled_effective=r['optimization_enabled_effective'],actual_fit_index_0based=r['actual_fit_index_0based'],actual_report_index_0based=r['actual_report_index_0based'],fit_filter_reasons=r['fit_filter_reasons'],region_shape_px2=rr.get('region_shape_px2'),centre_uv=rr.get('centre_uv'),sigma_px=rr.get('sigma_px'),sigma_note='No calibrated Gaussian sigma inferred from finite hand-picked region shape.',reliability=rr.get('reliability'),original_objective_weight=r['original_objective_weight'],runtime_objective_weight=r['runtime_objective_weight'],material_identity_confirmed=rr.get('material_identity_confirmed'),source_semantic=rr.get('source_semantic'),source_uncertainty=rr.get('source_uncertainty'),source_provenance=rr.get('provenance'),image_source=r.get('image_source'),full_original_row_reference=f"factor_assembly.candidates.{name}.rows[{r['document_row_index_0based']}].original_row"))
   obs[name]=rows
  feet.append(dict(foot=foot,anatomical_side_hypothesis='left' if foot==0 else 'right',hypothesis_is_not_per_frame_confirmation=True,visual_review=vis,observations_by_candidate=obs))
 exposures.append(dict(pts=pt,source_seconds=pt/12800,time_base=[1,12800],source_frame=contactbypts.get(pt,{}).get('source_frame',v.get('source_frame') if v else None),source_frame_sha256=contactbypts.get(pt,{}).get('source_frame_sha256',v.get('source_frame_sha256') if v else None),contact_sheet=contactbypts.get(pt,{}).get('contact_sheet',str(supp_path)),feet=feet,unassigned_source_objects=[r for r in unassigned if r['pts']==pt]))
def intervals(values):
 out=[]
 for v in values:
  if out and v==out[-1][1]+256:out[-1][1]=v
  else:out.append([v,v])
 return [dict(first_pts=a,last_pts=b,first_seconds=a/12800,last_seconds=b/12800,exposures=(b-a)//256+1) for a,b in out]
coverage={}
allpts=list(range(972800,1216000+1,256))
for foot in [0,1]:
 rows=[r for r in visual if r['foot']==foot]
 visible=[r['pts'] for r in rows if r['blade_axis_observable']=='visible']
 partial=[r['pts'] for r in rows if r['blade_axis_observable']=='partially_visible']
 unknown=[pt for pt in reviewed_pts if (pt,foot) not in mapped or mapped[(pt,foot)]['blade_axis_observable']=='unknown']
 unreviewed=[pt for pt in allpts if pt not in reviewed_pts]
 assert len(visible)+len(partial)+len(unknown)+len(unreviewed)==951
 coverage[str(foot)]=dict(denominator=951,reviewed_pts=64,mapped_foot_review_rows=len(rows),visible_undirected_axis_pts=visible,partially_visible_not_usable_as_verified_axis_pts=partial,reviewed_orientation_unknown_pts=unknown,unreviewed_exposures=len(unreviewed),unreviewed_ranges=intervals(unreviewed),not_axis_verified_total=951-len(visible),identity_limit='visible axis counts retain conditional foot mapping and never certify template endpoint identities')
identity['archive'].pop('manifest_sha256',None);identity['archive'].pop('manifest_hash_definition',None)
report=dict(schema='rinkspace.blade-observability-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_stage=1,status='completed_readonly_audit_and_stopped_before_solve',scope=identity['scope'],identity=identity,candidate_summaries=candidate_summaries,evidence_classes=dict(A='Direct source-review boot material/contour regions; not blade steel.',B='Direct source-review blade apparent endpoint/midpoint regions; preserve conditional template correspondence and unknown.',C='Saved flow valid/selected without affine/error ellipse or fixed material identity; excluded from blade material factors.',D='Conditional support/contact regions; neither fixed material nor contact ground truth.'),flow_recomputed=assembly['flow'],factor_assembly=assembly,reviewed_source_exposures=exposures,unknown_coverage=coverage,window_coverage=window_rows,source_contacts=dict(manifest=ref(O/'dual_source_manifest.json'),pages=contact['sheets'],supplemental=ref(supp_path),supplemental_sources=extras),visual_review_files=[ref(O/'visual_early.json'),ref(O/'visual_late.json')],historical_scope_correction=dict(old_A_endpoint_rows=10,old_A_endpoint_exposures=7,old_A_midpoint_rows=2,old_A_midpoint_exposures=2,window_seconds=[88,92],metric_phase='original_A == stage_seed, not optimized stage_best',actual_restricted_current_constructor_matches=True,full_segment_A_endpoint_rows=50,full_segment_A_endpoint_exposures=32),findings_zh=[
'旧10/7与2/2是88–92秒窗口计数且旧审核表引用初值阶段，不能当作整段最终因子稀疏度或当前优化结果。',
'A/B实际97条source rows，C6/local实际122条；双端标签同帧均18个PTS，但不等于18帧真实刀轴或固定材料对应已验收。',
'foot0没有任何实际刀端/中点行；源图86.80秒可见抬足无向钢片线，但前后端及固定材料身份未知，尚未接入方向因子。',
'77.6–78.2、81.2–82.0、84.8–85.1秒当前实际鞋刀图像行均为0。81.60/81.62秒源图可辨近冰鞋无向刀线，说明其中至少部分是可见但未装配，而非全部不可观测。',
'C6/local在85.50秒撤出1条front观测；旧source_evidence_rows保留50刀端台账，实际49。当前53条active刀具行material_identity_confirmed=false，条件模板对应并未被证明；构造器不筛此字段。',
'C6/local无额外有效review-only鞋刀行；不能将全部source-region report当成独立留出验收。A/B有16条/5PTS的report-only行，也不自动构成最终独立检查。',
'部分ROI提议重复近冰鞋或未捕获鞋体：84.52秒双ROI重复，93.20秒foot1 ROI为膝/手臂区。已按全幅源图审阅或保留unknown；没有自动换脚。',
'88.94、89.98秒近冰鞋刀端视，92.80、93.20及93.50秒刀轴未知；93.50两处白鞋未分配foot0/foot1。',
'flow当前实际重建仍185 valid、52 selected、3.70秒，材料准入0。'
],missing_or_unverified=[
'尚未执行第二阶段同一保存人体姿态/相机/静态绑定的反事实投影比较，不能归因静态注册、踝姿态、模板语义或身份错误。',
'尚无逐PTS源图刀轴数值几何及其不确定性，视觉可辨连续线不等于已获得轴角误差或固定材料观测。',
'当前已登记source review对77.6–78.2秒无对应鞋刀源几何审阅；不得从相邻曝光插值补成观测。',
'124条observation带可解析源图SHA且匹配；50条历史传播/派生observation本身缺少可解析源图字段，保留缺项。'
],execution_boundaries=dict(no_solve=True,no_network_inference=True,no_new_points=True,no_observation_or_identity_rebinding=True,no_camera_scene_template_scale_time_changes=True,no_candidate_overwrite=True,no_stage2_or_new_DOF=True,no_new_video=True,no_GUI=True,no_commit_or_push=True,first_stage_stop_quote='阶段一交付：blade_observability_audit.json + 一页中文摘要 + 可打开的关键源帧联系表。完成后停止，不启动求解。'),accuracy_accepted=False)
(O/'blade_observability_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Browser-independent local contact index; no video player or remote assets.
def esc(x):return html.escape(str(x if x is not None else 'unknown'))
def link(path,text):return f'<a href="{esc(Path(path).as_uri())}">{esc(text)}</a>'
intro='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S05 鞋刀源图可观测审计</title><style>body{font:16px "Microsoft YaHei",sans-serif;margin:24px;color:#222}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #bbb;padding:6px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eee}summary{cursor:pointer}img{max-width:100%;height:auto}.warn{background:#fff3cf;padding:14px}a{color:#1763a6}</style><h1>S05 双鞋／双刀源图可观测性</h1><p>第一阶段只读审计：76–95秒，951个原生曝光，实际审阅64个PTS。没有求解、取新点、改身份或模型投影。</p><p class="warn">foot0/foot1是保留的条件映射（解剖左/右假设），不是逐帧真值。图上的ROI仅为源审阅或检测器裁区提议，可能重复同一只鞋；以全幅源图与逐行说明为准。“可辨刀线”不等于固定材料对应或准确性验收。partially_visible不足以验证方向。93.50秒只读既有复合图的原片面板，两只鞋保持身份unknown。</p>'''
pages=[]
for sh in contact['sheets']:
 f=Path(sh['path']);pages.append(f'<details><summary>{esc(f.stem)}：'+', '.join(f'{pt/12800:.2f}' for pt in sh['pts'])+f' 秒</summary><a href="{esc(f.name)}"><img loading="lazy" src="{esc(f.name)}"></a></details>')
pages.append(f'<details open><summary>81.60 / 81.62 / 93.50秒关键源图</summary><img src="{supp_path.name}"></details>')
table=['<h2>逐PTS／逐足记录</h2><p>因子数按A / C6列出；完整A/B/C6/local索引、原始说明、区域矩阵、权重与SHA见JSON。</p><table><thead><tr><th>源时间 / PTS</th><th>足提议与身份</th><th>靴跟／靴尖</th><th>刀前／刀后／轴</th><th>端视／遮挡／模糊／影反光</th><th>实际因子 A/C6</th><th>源图说明、unknown原因</th></tr></thead><tbody>']
for e in exposures:
 for f in e['feet']:
  r=f['visual_review'];foot=f['foot'];source=r.get('source_frame');path=source or (r.get('reviewed_image_paths') or [str(supp_path)])[0]
  count={name:sum(q['actual_fit_index_0based'] is not None for q in f['observations_by_candidate'][name]) for name in ['A','C6']}
  table.append('<tr><td>'+link(path,f"{e['source_seconds']:.2f}s / {e['pts']}")+'</td><td>'+esc(f"foot{foot}; {r.get('anatomical_identity')} / {r.get('identity_confidence')}")+'</td><td>'+esc(r.get('boot_heel_visible'))+' / '+esc(r.get('boot_toe_visible'))+'</td><td>'+esc(r.get('blade_front_visible'))+' / '+esc(r.get('blade_rear_visible'))+' / <b>'+esc(r.get('blade_axis_observable'))+'</b></td><td>'+' / '.join(esc(r.get(k)) for k in ['end_on','occluded','motion_blur','shadow_reflection_confusion'])+'</td><td>'+esc(f"{count['A']} / {count['C6']}")+'</td><td>'+esc(r.get('physical_object_description',''))+'<br>'+esc(r.get('source_only_note',''))+'<br>'+esc(r.get('orientation_unknown_reason',''))+'</td></tr>')
table.append('</tbody></table>')
(O/'source_contact_sheet.html').write_text(intro+'\n'.join(pages+table)+'</html>',encoding='utf-8')
summary='''# S05 冰鞋／冰刀：阶段一审计摘要

基线 `0078c8db`。原片 SHA、A/B/C6/local_281 四个父结果 SHA 全匹配；均为 951 个原生曝光，PTS 972800–1216000，time_base 1/12800。实际审阅64个PTS；其余887个未作本轮源图方向判断。未启动求解、推理或取新点，未改已有候选、相机、鞋模板或观测。

| 当前入口 | 原始观察 | fit / review | enabled显式 / 含默认 | 实际fit / report行 |
|---|---:|---:|---:|---:|
| A/B | 174 | 155 / 19 | 54 / 113 | 97 / 113 |
| C6/local | 174 | 173 / 1 | 79 / 122 | 122 / 122 |

A/B实际foot0靴区17行/12PTS；foot1靴区26/12、刀端50/32、中点4/4。C6/local对应22/15、36/16、49/31、4/4，另有11/11条件支持区。**foot0没有刀端/中点因子。** 两端标签同帧均为18PTS，仍不证明模板材料对应。C6/local的53条active刀具行明确保留 `material_identity_confirmed=false`；85.50秒一条front已撤为unknown，原始50条台账未删除。

旧报告“10行/7曝光、2/2”是**88–92秒窗口**，并取初值`original_A/stage_seed`，并非整段最终A。当前构造函数按同窗口重筛仍吻合。flow实际重建仍为185 valid / 52 selected / 3.70秒，材料准入0；与直接源区域分开统计。C6/local没有额外有效review-only鞋刀行，不能视为独立留出验收。

**可修复缺口：**77.6–78.2、81.2–82.0、84.8–85.1秒当前实际鞋刀图像行全部为0。81.60/81.62秒近冰鞋可辨无向刀线；86.80秒抬足也能辨认一段刀线，前后材料端身份未知。这些“可见但未装配”不能和“图像本身不可辨”混为一谈。当前源审阅还缺77.6–78.2秒的鞋刀几何判断。

**保留unknown：**88.94/89.98秒近冰刀端视；92.80/93.20秒刀轴不可辨；93.50秒两只鞋未分配foot0/1。84.52秒双ROI重复近冰鞋，93.20秒foot1 ROI未捕获鞋体；已依全幅图判断或保留unknown，不自动换脚。源图可辨无向线的条件足映射：foot0为1个PTS、foot1为29个PTS；不等于已测量轴角或固定材料点。

**按任务书第一阶段停止。**第二阶段尚未运行，未归因到静态注册、踝姿态或模板，未增加鞋体自由度，也没有新的准确性通过结论。

文件：
- [完整审计 JSON](AUDIT_JSON)
- [可打开的双脚源帧联系表](CONTACT_HTML)
- [因子逐行与当前代码复现](FACTOR_JSON)
'''.replace('AUDIT_JSON',str(O/'blade_observability_audit.json').replace('\\','/')).replace('CONTACT_HTML',str(O/'source_contact_sheet.html').replace('\\','/')).replace('FACTOR_JSON',str(O/'factor_assembly_audit.json').replace('\\','/'))
(O/'BLADE_AUDIT_ZH.md').write_text(summary,encoding='utf-8')
print(json.dumps(dict(output=str(O/'blade_observability_audit.json'),bytes=(O/'blade_observability_audit.json').stat().st_size,reviewed_pts=len(reviewed_pts),mapped_foot_rows=len(visual),unassigned_objects=len(unassigned),axis_visible_counts={f:len(v['visible_undirected_axis_pts']) for f,v in coverage.items()},HTML=str(O/'source_contact_sheet.html'),summary=str(O/'BLADE_AUDIT_ZH.md')),ensure_ascii=False))
