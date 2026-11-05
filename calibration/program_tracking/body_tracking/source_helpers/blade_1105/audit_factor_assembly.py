"""One-off read-only assembly audit. No model fitting, inference or core writes.
Run from any cwd using the existing body_cuda Python, with -B.
Only output: factor_assembly_audit.json alongside this script.
"""
from __future__ import annotations
import copy, hashlib, inspect, json, pathlib, subprocess, sys
from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

PROJECT = pathlib.Path('D:/OneDrive/codex_projs/RinkSpace')
BASE = pathlib.Path('D:/CodexData/RinkSpace_Data/runs/body_s01_20261029')
OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from calibration.program_tracking.body_tracking.support_image import SupportImageFactors
from calibration.program_tracking.body_tracking.support_anchor_factors import NativeRegionObservation, pixel_region_factor
from calibration.program_tracking.body_tracking.source_track_evidence import build_track_region_document
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve
from calibration.program_tracking.body_tracking.skate_binding import SkateBinding
from calibration.program_tracking.body_tracking.skate_template import SkateParameters
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT

def js(path): return json.loads(path.read_text(encoding='utf-8'))
def npz(path):
    with np.load(path, allow_pickle=False) as z: return dict(z)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def file(path): return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
def group(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['foot'],r.get('source_semantic',r['kind']),r['kind'],r['role'])].append(r)
    return [dict(foot=k[0],semantic=k[1],kind=k[2],role=k[3],rows=len(v),unique_pts=len({r['pts']for r in v}),pts=sorted({r['pts']for r in v}),row_ids=[r['id']for r in v]) for k,v in sorted(groups.items(),key=lambda kv:str(kv[0]))]

def lightweight_problem(stage):
    """Use real compact rest joints, frozen static binding and seed curve.
    No CompactBody/SMPL-X/JointContactProblem constructor, skinning or camera.
    Real SkateBinding.packet/parameters/transforms and native22 fk_state execute
    during the unmodified SupportImageFactors.__init__ call.
    """
    case=stage/'common/exmr_hand_nofoot'
    inputs=npz(case/'input.npz'); compact=npz(case/'compact_smplx.npz')
    seed=npz(case/'initial.npz'); saved=npz(stage/'solve/candidate.npz')
    curve=curve_from_packet(seed,seed=True)
    binding=object.__new__(SkateBinding)
    binding.base_parameters=SkateParameters(**json.loads(str(saved['base_skate_parameters_json'])))
    binding.R0=saved['R_FB_initial'];binding.t0=saved['t_FB_initial_solver']
    binding.x=saved['x'][:17].copy();binding._parameter_key=None;binding._packet=None
    binding.ratio=float(json.loads(str(saved['camera_binding_json']))['length_export_ratio'])
    binding.layout=OBSERVATION_LAYOUT
    rest=compact['J_template']+np.einsum('jcb,b->jc',compact['J_shapedirs'],inputs['shared_betas'])
    ref=SimpleNamespace(p={'pts':seed['pts']},times=(seed['pts']-seed['pts'][0])/12800.,binding=binding,
        body=SimpleNamespace(J=rest,parents=compact['parents'],scale=float(inputs['body_scale'])),
        at=lambda times:evaluate_curve(curve,times))
    state=SimpleNamespace(m=len(curve.position_controls),basis=curve.basis)
    return SimpleNamespace(optimize_shared=False,reference=ref,binding=binding,
        body=SimpleNamespace(state=state,tau=float(inputs['rig'][37]))), [case/'input.npz',case/'compact_smplx.npz',case/'initial.npz',stage/'solve/candidate.npz']

def reasons(row,include_review):
    failures=[]
    if not(row['role']=='fit' or include_review): failures.append('role_review_not_in_fit_objective')
    if row['foot'] not in (0,1): failures.append('foot_unassociated')
    if row['visibility']!='visible': failures.append('visibility_'+row['visibility'])
    if row['reliability']<=0: failures.append('nonpositive_reliability')
    if row['kind']=='support_contact_region' and row['contact_state']!='supported': failures.append('contact_not_supported')
    if not(row.get('optimization_enabled',True) or (include_review and row['role']=='review' and not row.get('replaced_in_optimization_by'))): failures.append('optimization_disabled'+('_replaced_historical' if row.get('replaced_in_optimization_by') else ''))
    return failures

def audit_candidate(name,stage,document_path):
    document=js(document_path);runtime=copy.deepcopy(document)
    mult=js(stage/'grasp_relations.json')['factor_parameters']['shoe_region_weight_multiplier']
    for row in runtime['observations']: row['objective_weight']*=mult
    problem,input_paths=lightweight_problem(stage)
    fit=SupportImageFactors(problem,runtime,arc_ids={})
    report=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
    fit_ids=[r[0]['id']for r in fit.rows];report_ids=[r[0]['id']for r in report.rows]
    row_records=[]
    for i,(original,row) in enumerate(zip(document['observations'],runtime['observations'])):
        fit_reasons=reasons(row,False);report_reasons=reasons(row,True)
        assert (not fit_reasons)==(row['id']in fit_ids)
        assert (not report_reasons)==(row['id']in report_ids)
        o=NativeRegionObservation(**{k:row[k]for k in NativeRegionObservation.__dataclass_fields__})
        # Guard execution only: the observation centre is a synthetic query.
        # This is NOT a saved-model projection or accuracy measurement.
        guard=pixel_region_factor(o.centre_uv,o,source_id=document['source_id'],shot_id=document['shot_id'],pts=o.pts,time_base=document['time_base'],predicted_foot=o.foot,predicted_kind=o.kind,robust_delta=document['robust_delta'])
        if row['id'] in fit_ids: assert guard['enabled']
        footprint=original.get('image_footprint',{})
        source_frame=footprint.get('source_frame',original.get('source_frame'))
        source_expected=footprint.get('source_frame_sha256',original.get('source_frame_sha256'))
        source_actual=sha(pathlib.Path(source_frame)) if source_frame and pathlib.Path(source_frame).is_file() else None
        row_records.append(dict(document_row_index_0based=i,id=row['id'],pts=row['pts'],seconds=row['pts']/12800.,foot=row['foot'],
            semantic=row.get('source_semantic',row['kind']),semantic_is_explicit=('source_semantic'in row),kind=row['kind'],role=row['role'],
            optimization_enabled_explicit=row.get('optimization_enabled'),optimization_enabled_effective=row.get('optimization_enabled',True),
            actual_fit_index_0based=fit_ids.index(row['id'])if row['id']in fit_ids else None,
            actual_report_index_0based=report_ids.index(row['id'])if row['id']in report_ids else None,
            fit_filter_reasons=fit_reasons,report_filter_reasons=report_reasons,
            pixel_region_guard_at_synthetic_centre=dict(enabled=guard['enabled'],reason=guard['reason'],accuracy_measurement=False),
            original_objective_weight=original['objective_weight'],runtime_objective_weight=row['objective_weight'],
            image_source=dict(path=source_frame,recorded_sha256=source_expected,actual_sha256=source_actual,sha_matches=(source_actual==source_expected)if source_actual and source_expected else None),
            original_row=original))
    quality=js(stage/'cross_stage_quality.json')
    old={phase:{k:v for k,v in quality[phase]['groups'].items() if k in ['fit_blade_endpoint_foot_1','fit_blade_midpoint_foot_1','fit_boot_foot_1']}for phase in ['original_A','stage_seed','stage_best']}
    return dict(name=name,stage_root=str(stage),document=file(document_path),
        root_source_document=file((stage.parent if name in ['A','B'] else stage.parent)/'source_evidence.json'),
        runtime=dict(production_path='in-memory copy.deepcopy(document) in spin_continuous_refit.make_problem',origin_path=str(document_path),
            persisted_separately_by_production=False,transformation='all observation.objective_weight *= grasp_relations.factor_parameters.shoe_region_weight_multiplier',
            multiplier=mult,canonical_json_sha256=canonical(runtime),hash_encoding='UTF-8, sort_keys=True, ensure_ascii=False, separators=(comma,colon)'),
        input_files=[file(p)for p in input_paths],
        result=file(stage/'common/exmr_hand_nofoot/fitted/result.npz'),
        total_document_rows=len(document['observations']),fit_role_rows=sum(r['role']=='fit'for r in document['observations']),review_role_rows=sum(r['role']=='review'for r in document['observations']),
        optimization_enabled_explicit_true=sum(r.get('optimization_enabled')is True for r in document['observations']),
        optimization_enabled_explicit_missing=sum('optimization_enabled'not in r for r in document['observations']),
        optimization_enabled_effective_true=sum(r.get('optimization_enabled',True)for r in document['observations']),
        actual_self_rows=len(fit.rows),actual_report_self_rows=len(report.rows),objective_residual_scalar_rows=2*len(fit.rows),
        five_nodes_per_source_row=5,five_nodes_are_not_independent_observations=True,
        document_groups=group(document['observations']),actual_fit_groups=group([r[0]for r in fit.rows]),actual_report_groups=group([r[0]for r in report.rows]),
        actual_fit_row_ids=fit_ids,actual_report_row_ids=report_ids,
        actual_fit_document_indices_0based=[i for i,r in enumerate(document['observations'])if r['id']in fit_ids],
        rows=row_records,old_report=dict(file=file(stage/'cross_stage_quality.json'),window_seconds=[88,92],phases=old,
            corresponding_current_fit_window_groups=group([r[0]for r in fit.rows if 88*12800<=r[1].pts<=92*12800])),
        source_evidence_ledger_total_rows=len(document.get('source_evidence_rows',[])),
        source_evidence_ledger_endpoint_rows=[r for r in document.get('source_evidence_rows',[])if r.get('source_semantic')=='blade_endpoint_region'],
        document_source_evidence_admission=document.get('source_evidence_admission'),
        additional_direct_review_unknowns=document.get('additional_direct_review_unknowns',[]))

def flow_admission():
    old=BASE/'spin_entry_motion_1104';root=BASE/'spin_continuous_1105'
    doc_path=old/'support_and_free_shoe_regions.json';track_path=old/'source_support_track.npz'
    anchors_path=PROJECT/'events/ankara_2026/programmes/okada_mei_sp/spin_support_midpoint_anchors_1104.json'
    overrides_path=root/'source_review/frozen_review_overrides.json'
    original=js(doc_path);track=npz(track_path);overrides=js(overrides_path)
    protected={int(r['pts'])+d for r in original['observations']if r['role']=='review'for d in[-256,0,256]}
    rebuilt,prepared=build_track_region_document(original,track,js(anchors_path)['anchors'],review_overrides=overrides,excluded_review_pts=protected,native_step_pts=256)
    original_ids={r['id']for r in original['observations']}
    added=[r for r in rebuilt['observations']if r['id']not in original_ids]
    # Exercise the same current admission with no semantic overrides: valid
    # tracks and confidence alone must not produce material observations.
    flow_only,flow_prepared=build_track_region_document(original,track,js(anchors_path)['anchors'],review_overrides=None,excluded_review_pts=protected,native_step_pts=256)
    newly_enabled_flow=[r['id']for r in flow_only['observations']if r['id']not in original_ids and r.get('optimization_enabled',True)]
    current=js(root/'source_evidence.json')
    assert rebuilt['observations']==current['observations']
    return dict(executed_current_function='source_track_evidence.build_track_region_document',
        inputs=[file(p)for p in[doc_path,track_path,anchors_path,overrides_path]],track_keys=list(track),
        raw_track_exposures=len(track['pts']),raw_valid=int(np.asarray(track['valid'],bool).sum()),
        recomputed_quadrature=prepared['source_exposure_audit'],
        rebuilt_observations_identical_to_current_A_document=True,
        rebuilt_observations_canonical_sha256=canonical(rebuilt['observations']),
        added_direct_rows=len(added),added_direct_endpoint_rows=sum(r['kind']=='blade_endpoint_region'for r in added),added_direct_midpoint_rows=sum(r['kind']=='blade_midpoint_region'for r in added),
        admitted_flow_material_row_ids=newly_enabled_flow,admitted_flow_material_rows=len(newly_enabled_flow),
        flow_only_current_admission=flow_only['source_evidence_admission'],
        reconstructed_admission=rebuilt['source_evidence_admission'],
        no_flow_affine_or_error_ellipse_in_saved_track=not any('affine'in k.lower()or'ellipse'in k.lower()for k in track),
        note='selection and exposure seconds are temporal quadrature; direct source reviews enter separately. No propagation uncertainty inferred from confidence.')

def finalize_metadata(result):
    """Finalize evidence links/index mappings without rerunning assembly."""
    result['script']=file(pathlib.Path(__file__).resolve())
    revisions=BASE/'spin_continuation_1105'
    result['history_revision_sources']=[file(p) for p in [
        revisions/'source_evidence_revision.json',revisions/'source_review/support_semantic_review.json',
        revisions/'source_review/contact_region_review.json',revisions/'source_evidence_v3.json',
        revisions/'source_evidence_v4.json',revisions/'source_evidence.json'] if p.is_file()]
    A={r['id']:r for r in result['candidates']['A']['rows']}
    C={r['id']:r for r in result['candidates']['C6']['rows']}
    result['A_to_C6_observation_changes']=[dict(id=rid,pts=A[rid]['pts'],
        source_document_index_0based=A[rid]['document_row_index_0based'],
        changed_fields={k:[A[rid]['original_row'].get(k),C[rid]['original_row'].get(k)] for k in sorted(set(A[rid]['original_row'])|set(C[rid]['original_row'])) if A[rid]['original_row'].get(k)!=C[rid]['original_row'].get(k)})
        for rid in A if A[rid]['original_row']!=C[rid]['original_row']]
    for candidate in result['candidates'].values():
        rows={r['id']:r for r in candidate['rows']}
        candidate['source_evidence_endpoint_mapping']=[]
        for i,ledger in enumerate(candidate['source_evidence_ledger_endpoint_rows']):
            rid='reviewed_source_seed_'+str(ledger['pts'])+'_'+ledger['blade_endpoint']
            r=rows.get(rid)
            candidate['source_evidence_endpoint_mapping'].append(dict(
                ledger_endpoint_list_index_0based=i,pts=ledger['pts'],foot=ledger['foot'],blade_endpoint=ledger['blade_endpoint'],
                ledger_optimization_enabled=ledger.get('optimization_enabled'),observation_id=rid,
                observation_document_index_0based=r['document_row_index_0based']if r else None,
                actual_fit_index_0based=r['actual_fit_index_0based']if r else None,
                actual_fit_filter_reasons=r['fit_filter_reasons']if r else ['not_created_by_source_admission'],
                observation_material_identity_confirmed=r['original_row'].get('material_identity_confirmed')if r else None))
        candidate['active_blade_material_identity_flags']=dict(Counter(str(r['original_row'].get('material_identity_confirmed','missing'))for r in candidate['rows']if r['actual_fit_index_0based']is not None and r['kind']in['blade_endpoint_region','blade_midpoint_region']))
        candidate['source_frame_hash_check']=dict(matching=sum(r['image_source']['sha_matches']is True for r in candidate['rows']),mismatching=sum(r['image_source']['sha_matches']is False for r in candidate['rows']),not_resolvable_from_observation_fields=sum(r['image_source']['sha_matches']is None for r in candidate['rows']))
    result['findings_zh']=[
        '真实构造结果：A/B各174 document rows=155 fit+19 review，optimization_enabled显式true54、默认True后有效113，实际self.rows97（194标量residual），report113；C6/local各174=173fit+1review，显式true79、有效122，实际self.rows122（244标量residual），report122。',
        'A/B全段入行：foot0 boot17行/12PTS；foot1 boot26/12、endpoint50/32、midpoint4/4。C6/local：foot0 boot22/15；foot1 boot36/16、endpoint49/31、midpoint4/4、conditional support11/11。',
        '旧A的10端点行/7曝光和2中点行/2曝光是support_refinement.measure固定88–92秒窗口统计，不是951曝光整段self.rows；当前用实际self.rows重筛该窗口仍吻合，来源并未变化。',
        '旧AUDIT_ZH表A的1.44px区域外P95、19.22px中心P95与midpoint7.55px中心P95取自cross_stage_quality.original_A/stage_seed，而不是已优化A的stage_best；不可混淆phase。',
        'source_evidence_rows的50 endpoint行是源审阅ledger；A/B全部进入，C6/local在observation修订中将reviewed_source_seed_1094400_front（85.50秒）改为visibility unknown、optimization_enabled false，实际只余49行/31PTS，ledger仍保留50。',
        'C6/local另将53个仍入行blade observation的material_identity_confirmed改为false，注明conditional_template_mapping；当前SupportImageFactors不筛这个字段，因此仍准入有限表观区域。实际入行不能升级为已证实固定材料对应。',
        'pixel_region_factor不读取optimization_enabled；该标志由SupportImageFactors.__init__装配层先过滤。合成中心调用只核对守卫，不能当源图误差或方向准确性。',
        '当前build_track_region_document已实际重跑并重建出与A相同observations；185valid/52selected/3.70秒仍成立。没有语义override的同函数调用产生0条新flow材料观测；没有用confidence补造affine或误差椭圆。',
        'C6/local已消费旧review作为开发拟合，不能再作为独立最终验证；本轮只读确认，不作新的图像判断、原片观测升级或准确性验收。']

def main():
    candidates={
        'A':BASE/'spin_continuous_1105/source_stage','B':BASE/'spin_continuous_1105/flight_stage',
        'C6':BASE/'spin_continuation_1105/fit6/continuous_stage','local_281':BASE/'spin_local_280_1105/fit1/continuous_stage'}
    result=dict(schema='rinkspace.factor-assembly-readonly-audit.v1',
        scope='stage1 only, S05 76-95s, no solve/inference/model projection/new source-image judgement',
        method='Unmodified real SupportImageFactors.__init__ called twice per candidate with real document and minimal real seed/compact/frozen-binding state. Real native22 FK and SkateBinding methods execute. pixel_region_factor guards called at synthetic observation centres only.',
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        script=file(pathlib.Path(__file__).resolve()),
        code_files=[file(PROJECT/'calibration/program_tracking/body_tracking'/p)for p in ['spin_continuous_refit.py','support_image.py','support_anchor_factors.py','source_track_evidence.py','support_refinement.py','skate_binding.py','skate_physics.py','absolute_motion_state.py']],
        candidates={name:audit_candidate(name,stage,stage/'support_and_free_shoe_regions.json')for name,stage in candidates.items()},
        flow=flow_admission(),
        limitations=['No residual accuracy or orientation claim: no saved-model projection or visual judgement performed.','Human confirmations remain as recorded; no unknown/foot identity upgraded.','Production runtime document is in memory; canonical SHA records reconstructed contents, not a historical on-disk runtime file.'],
        findings_zh=['旧A的10端点行/7曝光和2中点行/2曝光是measure固定88–92秒窗口统计，不是951曝光整段self.rows。',
        '旧报告A的1.44px与7.55px等数值对应cross_stage_quality.original_A/stage_seed；stage_best是已优化A结果，应分清phase。',
        'source_evidence_rows保留原审阅层；C6/local的单独observation修订可禁用端点，而ledger仍留50行，不能从ledger推断最终入行。',
        'pixel_region_factor不读取optimization_enabled；该标志在SupportImageFactors.__init__装配层先过滤。'])
    finalize_metadata(result)
    output=OUT/'factor_assembly_audit.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),sha256=sha(output),summary={k:{x:v[x]for x in ['total_document_rows','fit_role_rows','review_role_rows','optimization_enabled_explicit_true','optimization_enabled_effective_true','actual_self_rows','actual_report_self_rows','objective_residual_scalar_rows']}for k,v in result['candidates'].items()},flow={k:result['flow'][k]for k in ['raw_track_exposures','raw_valid','added_direct_endpoint_rows','admitted_flow_material_rows']}),ensure_ascii=False))
if __name__=='__main__':main()
