"""Publish a checked source-index revision through the existing sole entry."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha


from rinkspace.storage import AssetStore

def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def publish(dataset_path, audit_path, receipt_path, parent_summary_path, keep_facts_path, output,
            consumer_report_path):
    refs = [reference(p) for p in (dataset_path, audit_path, receipt_path,
                                  parent_summary_path, keep_facts_path)]
    for ref in refs:
        authenticate_source_reference(ref)
    dataset, audit, receipt, parent, keep_facts = map(read, (
        dataset_path, audit_path, receipt_path, parent_summary_path, keep_facts_path))
    consumer_ref = reference(consumer_report_path)
    authenticate_source_reference(consumer_ref)
    consumer = read(consumer_report_path)
    if (audit['passed'] is not True or audit['revised_dataset_reference'] != refs[0]
            or receipt['revised_dataset_reference'] != refs[0]
            or audit['parent_dataset_reference']['sha256'] != parent['dataset_reference']['sha256']
            or audit['independent_keep_facts_reference'] != refs[4]
            or audit['independently_verified_original_points_kept'] != 157
            or keep_facts['exact_existing_point_keep_count'] != 157
            or consumer['passed'] is not True or consumer['dataset_reference'] != refs[0]
            or sum(audit['counts'].values()) != sum(c['source_points'] for c in dataset['cuts'])):
        raise ValueError('Publish only a checked conservation revision of the current delivery')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Publish to a new project-owned version directory')
    output.mkdir(parents=True)
    target = output / 'fit_dataset.json'
    target.write_bytes(Path(dataset_path).read_bytes())
    if file_sha(target) != refs[0]['sha256']:
        raise ValueError('Published dataset bytes changed')
    summary = deepcopy(parent)
    summary['dataset_reference'] = reference(target)
    summary['previous_delivery_reference'] = refs[3]
    summary['exact_identity_revision_reference'] = refs[2]
    summary['exact_identity_index_audit_reference'] = refs[1]
    summary['independently_confirmed_specific_inner_keep_facts_reference'] = refs[4]
    summary['current_source_public_consumer_checks'] = [
        *parent['current_source_public_consumer_checks'], consumer_ref]
    summary['source_packages'] = len(dataset['cuts'])
    summary['support_index_bytes'] = sum(Path(c['support_index']['path']).stat().st_size for c in dataset['cuts'])
    for field in ('fit_points', 'held_points', 'pending_support_points', 'pending_named_points',
                  'actual_exposures', 'actual_exposures_with_finite_fit', 'actual_exposures_without_finite_fit'):
        summary['counts'][field] = dataset[field]
    quantities = {r['shot_id']: r for r in audit['source_cut_point_coverage']}
    for row in summary['source_cut_coverage']:
        current = quantities.pop(row['shot_id'])
        for field in ('fit_points', 'held_points', 'pending_support_points', 'pending_named_points',
                      'actual_exposures_with_finite_fit'):
            row[field] = current.get(field, 0)
        row['actual_exposures_without_finite_fit'] = (
            row['source_clock_actual_exposures'] - row['actual_exposures_with_finite_fit'])
    if quantities or sum(r['fit_points'] for r in summary['source_cut_coverage']) != dataset['fit_points']:
        raise ValueError('Published nine-cut quantities disagree')
    summary.pop('additional_specific_inner_identity_review_pending', None)
    summary['specific_inner_identity_review_completed'] = dict(
        additional_reviewed_original_finite_points=2193,
        independently_confirmed_original_points_kept=157,
        exact_original_points_quarantined=2036,
        pending_filter_references=dataset['exact_identity_revision_filter_references'],
        keep_facts_reference=refs[4],
        other_points_not_deleted_by_review_flag=True,
        pending_is_neither_confirmed_reflection_nor_training_negative=True)
    summary['changes_from_initial_dataset']['additional_specific_inner_points_quarantined'] = 2036
    summary['changes_from_initial_dataset']['specific_inner_points_independently_confirmed_and_kept'] = 157
    if audit['new_exact_pending_points'] != 2036:
        raise ValueError('Current publication requires the completed exact 2036-point source review')
    summary['vision_training_status'] = 'current_source_admission_frozen_optimizer_launch_pending'
    summary['status'] = 'finite_source_candidate_dataset_frozen_full_solver_graph_adapter_pending'
    summary['publication_implementation_reference'] = reference(__file__)
    summary_path = output / 'delivery_summary.json'
    with summary_path.open('x', encoding='utf8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    entry = AssetStore(ROOT).project_entry()
    original = entry.read_bytes()
    begin = b'<!-- CURRENT_FIT_DATASET_20261013_BEGIN -->'
    end = b'<!-- CURRENT_FIT_DATASET_20261013_END -->'
    if original.count(begin) != 1 or original.count(end) != 1:
        raise ValueError('Existing sole-entry managed block required')
    left, rest = original.split(begin, 1)
    _, right = rest.split(end, 1)
    relative = output.relative_to(ROOT).as_posix()
    counts = summary['counts']
    content = f'''\n<section class="section" id="current-fit-dataset-20261013"><span class="eyebrow">最新源测量拟合候选输入 · 2026-10-13</span><h2>九镜头修订版已冻结，可按公开接口消费</h2><div class="surface">
<p><strong>当前 {counts['fit_points']:,} 个有限支持 FIT 候选点，{counts['held_points']:,} 个 HELD 点独立保留。</strong>{len(dataset['cuts'])} 个原源包保留 {counts['native_frames']:,} 原生帧／{counts['actual_exposures']:,} 实际曝光。{counts['actual_exposures_with_finite_fit']:,} 个曝光有有效 FIT 候选，{counts['actual_exposures_without_finite_fit']:,} 个曝光当前没有；{counts['pending_support_points']:,} 个缺失支持点和 {counts['pending_named_points']:,} 个身份／源轮廓待核点保留证据，不进入 FIT。<a href="{relative}/fit_dataset.json"><strong>打开当前数据集清单 →</strong></a> · <a href="{relative}/delivery_summary.json">九镜头覆盖、消费接口与剩余工作</a></p>
<p>本轮新增 1,097 个原片完整蓝漆双侧 q50 中心，覆盖 99 个 B1 缺点曝光；27 个曝光因遮挡、接口不完整或出画继续保留缺失，先前 12 点小包不重复追加。034 的 190 个具体内层未核实点已隔离；其余四镜头的 2,193 个疑点已逐点复核，保留 157 个原图可独立确认的真实内缘，将其余 2,036 个精确隔离。真实漆线与已确认外圈保留，不把待核点当成已确认倒影或训练负例。</p>
<p>跨 31 包的原像素／目标／算子比较无重复或冲突。修订后全部原点均由独立索引审计逐一计入，原 UV、sigma、支持区间、源时钟、算子与 HELD 不变；原大数组复用，索引共 {summary['support_index_bytes']:,} 字节。sigma 表示未校准的实测支持范围，不代表已验证定位精度。</p>
<p class="small"><strong>剩余工作：</strong>继续核对其他漏点与准确性；本轮准入已冻结，视觉训练将覆盖全部新增 B1 点与曝光。完整求解图接入和拟合后的完整镜头验证尚未完成。本版是可消费的候选数据集，尚未新拟合或声明准确性通过。</p>
</div></section>\n'''.encode('utf8')
    updated = left + begin + content + end + right
    if updated.split(begin, 1)[0] != left or updated.split(end, 1)[1] != right:
        raise ValueError('Publication changed another entry region')
    updated.decode('utf8', errors='strict')
    for filename in ('fit_dataset.json', 'delivery_summary.json'):
        if not (ROOT / relative / filename).is_file():
            raise ValueError('Published entry target missing')
    entry.write_bytes(updated)
    registration = dict(schema='rinkspace.exact-reviewed-source-dataset-entry-registration.v1',
        dataset_reference=reference(target), delivery_reference=reference(summary_path),
        sole_entry_reference=reference(entry), previous_entry_sha256=__import__('hashlib').sha256(original).hexdigest(),
        all_other_entry_bytes_unchanged=True, new_source_points_copied=False,
        audit_reference=refs[1], identity_revision_reference=refs[2], keep_facts_reference=refs[4])
    registration_path = Path(dataset_path).parent / 'entry_registration.json'
    with registration_path.open('x', encoding='utf8') as stream:
        json.dump(registration, stream, ensure_ascii=False, separators=(',', ':'))
    return dict(dataset_reference=registration['dataset_reference'],
                delivery_reference=registration['delivery_reference'],
                registration_reference=reference(registration_path), counts=counts)


def main():
    parser = argparse.ArgumentParser()
    for name in ('dataset', 'audit', 'receipt', 'parent-summary', 'keep-facts', 'output-dir',
                 'consumer-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.dataset, args.audit, args.receipt, args.parent_summary,
                             args.keep_facts, args.output_dir, args.consumer_report)), flush=True)


if __name__ == '__main__':
    main()
