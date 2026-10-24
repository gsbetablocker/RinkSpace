"""Inspect a delivered original-S01 input; never prepare sources or start a solver.

The optional delta/reference checks are explicit development checks for the
changed delivery. This script is not a solver startup or provenance gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array(d):
    return np.fromfile(d['file'], dtype=d['dtype']).reshape(d['shape'])


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def audit(input_dir, deltas=(), immutable_reference=None):
    directory = Path(input_dir).resolve()
    packet = read(directory/'packet/manifest.json')
    semantic = read(directory/'semantic/manifest.json')
    model = read(directory/'model.json')
    tables = read(packet['string_tables']['file'])
    active = array(semantic['active_source_mask']).astype(bool)
    statistical = array(semantic['statistical_active_mask']).astype(bool)
    material = array(semantic['resolved_material_index'])
    branch = np.fromfile(directory/'path.u32', dtype='<u4')
    n = packet['observations']
    issues = []
    def require(ok, message):
        if not bool(ok):
            issues.append(message)
    require(all(len(a) == n for a in [active, statistical, material, branch]), 'Array lengths differ')
    require(np.all(~statistical | active), 'Statistical source outside engineering activity')
    owner = np.zeros(n, np.int16)
    curve = [None]*n
    for row in range(n):
        if material[row] >= len(model['materials']):
            require(not active[row], f'Active row {row} has no material')
            continue
        mat = model['materials'][int(material[row])]
        match = re.match(r'^Ad(\d+)(?:[.:/]|$)', mat['label'])
        owner[row] = 0 if match is None else int(match.group(1))
        if branch[row] >= len(mat['paths']):
            require(not active[row], f'Active row {row} has no finite path')
            continue
        path = mat['paths'][int(branch[row])]
        curve[row] = path.get('current_curve_id', path['id'])
    advertisements = []
    for ad in range(2, 13):
        rows = np.flatnonzero(active & (owner == ad))
        paths = sorted(set((int(material[r]), int(branch[r])) for r in rows))
        free_blocks = set()
        failures = []
        for mi, pi in paths:
            mat, path = model['materials'][mi], model['materials'][mi]['paths'][pi]
            if mat.get('display_only_initializer') or path.get('display_only_initializer'):
                failures.append(f'{mi}/{pi}: demo geometry')
            points = np.asarray(path.get('points', []), float)
            if not points.size or not np.isfinite(points).all():
                failures.append(f'{mi}/{pi}: absent/nonfinite geometry')
            for name in ['advertisement_block', 'geometry_block', 'affine_block']:
                if name not in path:
                    continue
                bi = path[name]
                if not 0 <= bi < len(model['blocks']):
                    failures.append(f'{mi}/{pi}: invalid {name}')
                    continue
                block = model['blocks'][bi]
                if not block.get('constant', False):
                    free_blocks.add(block['name'])
        require(len(rows) > 0, f'Ad{ad} has zero admitted sources')
        require(bool(free_blocks), f'Ad{ad} has no connected free object block')
        require(not failures, f'Ad{ad} path defects: {failures}')
        advertisements.append(dict(advertisement=ad, engineering_sources=len(rows),
            statistical_sources=int(np.sum(statistical & (owner == ad))),
            active_finite_paths=len(paths), connected_free_object_blocks=sorted(free_blocks),
            active_path_defects=failures, structural_connection_correct=not failures and bool(free_blocks),
            whole_shot_visual_acceptance=False))
    prepared_path = packet['current_prepared_export']['prepared_manifest']
    prepared = read(prepared_path)
    refs = {r['id'].removeprefix('preparation.'): r for r in prepared['array_refs']}
    def delivered(name):
        return np.load(refs[name]['path'], allow_pickle=False)
    prefix = prepared['row_count']
    measurements = []
    for field, name, current in [
        ('UV', 'immutable_uv', array(packet['arrays']['native_uv'])),
        ('PTS', 'native_pts', array(packet['arrays']['native_pts'])),
        ('E', 'engineering_budget_px', array(packet['arrays']['engineering_budget_px'])),
        ('sigma', 'source_sigma', array(semantic['arrays']['source_sigma']))]:
        unchanged = same_bits(current[:prefix], delivered(name))
        require(unchanged, field+' current raw prefix differs from delivered measurements')
        measurements.append(dict(field=field, compared_rows=prefix, bitexact=unchanged))
    reference_check = None
    if immutable_reference:
        reference = Path(immutable_reference)
        rp, rs = read(reference/'packet/manifest.json'), read(reference/'semantic/manifest.json')
        rt = read(rp['string_tables']['file'])
        checks = {}
        for field in ['native_uv', 'native_pts', 'engineering_budget_px']:
            checks[field] = same_bits(array(packet['arrays'][field]), array(rp['arrays'][field]))
        checks['source_sigma'] = same_bits(array(semantic['arrays']['source_sigma']), array(rs['arrays']['source_sigma']))
        checks['stable_observation_ids'] = tables['stable_observation_ids'] == rt['stable_observation_ids']
        require(all(checks.values()), 'Immutable reference has changed UV/PTS/E/sigma/source IDs')
        reference_check = dict(input=str(reference), checks=checks)
    group_rows = array(semantic['arrays']['group_rows']).astype(np.int64)
    weights = array(semantic['arrays']['weights'])
    ordinary = array(semantic['arrays']['ordinary_rows']).astype(np.int64)
    source_rows = delivered('group_member_rows')
    source_offsets = delivered('group_member_offsets')
    source_weights = delivered('group_weight_values')
    source_woffsets = delivered('group_weight_offsets')
    group_index = {g['id']: i for i, g in enumerate(prepared['groups'])}
    groups = [json.loads(s) for s in Path(semantic['assets']['groups.jsonl']['file']).read_text(encoding='utf8').splitlines() if s]
    coverage = np.zeros(n, np.int32)
    group_defects = []
    inherited_support_reductions = []
    rank = 0
    for group in groups:
        gid = group['group_id']
        gi = group_index.get(gid)
        if gi is None:
            group_defects.append(gid+': not in delivered complete factors')
            continue
        start, count = group['member_start'], group['member_count']
        members = group_rows[start:start+count]
        source_members = source_rows[source_offsets[gi]:source_offsets[gi+1]]
        ws = group['weight_start']
        matrix = weights[ws:ws+int(np.prod(group['matrix_shape']))]
        original = source_weights[source_woffsets[gi]:source_woffsets[gi+1]]
        sha = hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest()
        meta = prepared['groups'][gi]
        if not np.array_equal(members, source_members):
            group_defects.append(gid+': complete member support changed')
        if not same_bits(matrix, original.astype('<f8')) or sha != group['matrix_sha256']:
            group_defects.append(gid+': delivered W matrix changed')
        if sha != meta['original_matrix_sha256']:
            if meta.get('retained_support_operator_preserved_without_recomputation', False):
                inherited_support_reductions.append(dict(group_id=gid, source_shape=meta['source_original_matrix_shape'], current_shape=group['matrix_shape'], source_rank=meta['source_original_rank'], current_rank=group['rank'], source_matrix_sha256=meta['original_matrix_sha256'], current_matrix_sha256=sha, already_present_in_delivery=True))
            else:
                group_defects.append(gid+': undeclared change from original W')
        if not np.all(active[members]):
            group_defects.append(gid+': inactive member in retained factor')
        coverage[members] += 1
        rank += group['rank']
    coverage[ordinary] += 1
    rank += 2*len(ordinary)
    require(not group_defects, 'Current delivered complete W/support differs')
    require(np.all(coverage[statistical] == 1) and np.all(coverage[~statistical] == 0), 'Statistics not partitioned exactly once')
    require(rank == semantic['current_input_scope']['formal_W_rank'], 'W rank count differs')
    sigma = array(semantic['arrays']['source_sigma'])
    require(np.all(np.isfinite(sigma[ordinary]) & (sigma[ordinary] > 0)), 'Independent sigma invalid')
    delta_checks = []
    expected, pending = {}, set()
    evidence_mismatches = []
    uv = array(packet['arrays']['native_uv'])
    pts = array(packet['arrays']['native_pts'])
    budget = array(packet['arrays']['engineering_budget_px'])
    def delta_row(item):
        return int(item.get('original_row', item.get('row', item.get('full_row'))))
    for delta_path in deltas:
        delta = read(delta_path)
        records = delta.get('records', delta.get('rebind', delta.get('rebindings', [])))
        for item in records:
            row = delta_row(item)
            target = item.get('new_curve_id', item.get('current_native_path_id'))
            for key, value in [('source_id', item.get('source_id', item.get('stable_id'))), ('native_pts', item.get('native_pts')), ('immutable_uv', item.get('immutable_uv')), ('engineering_budget_px', item.get('engineering_budget_px')), ('sigma', item.get('source_sigma_preserved', item.get('original_sigma')))]:
                if value is None:
                    continue
                current = dict(source_id=tables['stable_observation_ids'][row], native_pts=int(pts[row]), immutable_uv=uv[row].tolist(), engineering_budget_px=float(budget[row]), sigma=float(sigma[row]))[key]
                if current != value:
                    evidence_mismatches.append(dict(row=row, field=key))
            if target:
                expected[row] = target
                pending.discard(row)
        detached = delta.get('pending_records', delta.get('pending', delta.get('pending_sources', [])))
        for item in detached:
            pending.add(delta_row(item))
        delta_checks.append(dict(path=str(Path(delta_path).resolve()), sha256=digest(delta_path),
            rebindings=len(records), pending_sources=len(detached)))
    mismatches = [dict(row=r, expected=v, actual=curve[r]) for r, v in expected.items() if curve[r] != v]
    pending_active = [r for r in sorted(pending) if active[r]]
    require(not mismatches, 'Reviewed delta mapping not consumed')
    require(not evidence_mismatches, 'Reviewed delta immutable evidence differs')
    require(not pending_active, 'Reviewed pending source still active')
    return dict(schema='rinkspace.current_original_ad_input_audit.v1', input=str(directory),
        model_sha256=digest(directory/'model.json'), packet_sha256=digest(directory/'packet/manifest.json'),
        semantic_sha256=digest(directory/'semantic/manifest.json'),
        inspected_actual_arrays=True, source_production_or_qualification_replayed=False,
        is_solver_startup_prerequisite=False, advertisements=advertisements,
        measurements=measurements, immutable_reference=reference_check,
        original_W=dict(delivered_current_manifest=prepared_path, groups_checked=len(groups),
            rank=rank, independent_sources=len(ordinary), defects=group_defects,
            complete_current_delivery_support_unchanged=not group_defects,
            original_source_support_unreduced_for_all_groups=not inherited_support_reductions,
            inherited_source_support_reductions=inherited_support_reductions,
            statistical_partition_exactly_once=bool(np.all(coverage[statistical] == 1) and np.all(coverage[~statistical] == 0))),
        changed_delivery_deltas=delta_checks, final_rebinding_mismatches=mismatches,
        delta_immutable_evidence_mismatches=evidence_mismatches,
        pending_sources_still_active=pending_active,
        structural_input_passed=not issues, whole_shot_visual_acceptance=False,
        limitation='结构接入检查覆盖当前活动源和本次显式广告增量；不能替代原片身份复核或整镜头同模型贴合验收。未核实/模糊旧源继续保留退出状态。',
        issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--delta', action='append', default=[], type=Path)
    parser.add_argument('--immutable-reference', type=Path)
    args = parser.parse_args()
    report = audit(args.input, args.delta, args.immutable_reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=report['structural_input_passed'], report=str(args.report),
        issues=report['issues'], advertisements=len(report['advertisements']),
        original_W_groups=report['original_W']['groups_checked']), ensure_ascii=False))
    return 0 if report['structural_input_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
