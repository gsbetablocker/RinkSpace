"""Conservative original-RGB evidence for withdrawing existing print rows.

Only an unchanged actual readout whose entire q25-q75 guard still lies in
unexcluded dark source material is certified interior. Missing replacements,
masking and all other unresolved cases never authorize a withdrawal.
"""
from pathlib import Path
import argparse, hashlib, json
import cv2
import numpy as np


def audit_rows(rgb, topology, exclusion, old_rows, candidate_records, *, source_hash_matches):
    audit = []
    old_candidates = [c for c in candidate_records if c['stage'] == 'registered_navigation_audit']
    for index, row in old_rows:
        uv = np.asarray(row['uv'], float); normal = np.asarray(row['normal'], float)
        qq = np.asarray(row['actual_source_quantiles'], float)
        evidence = dict(old_row_index=index, path_id=row['path_id'], uv=uv.tolist(),
            old_actual_source_quantiles=qq.tolist(), old_normal=normal.tolist(),
            source_hash_matches=bool(source_hash_matches), native_residual_used=False,
            disposition='not_proven_wrong_keep_old_row', withdrawal_authorized_by_this_audit=False)
        exact = [c for c in old_candidates if c['curve'] == row['curve']
            and np.linalg.norm(np.array(c['uv'])-uv) < 1e-8
            and np.allclose(c['actual_source_quantiles'], qq, rtol=0, atol=1e-8)]
        evidence['same_source_readout_reproduced'] = len(exact) == 1
        if len(exact) == 1:
            c = exact[0]
            evidence.update(reproduced_candidate_id=c['candidate_id'],
                candidate_rejection_reasons=c['pending_reasons'], source_topology_probes=c['source_topology_probes'])
        if (topology is not None and source_hash_matches and len(exact) == 1
                and qq.shape == (2, 3, 2) and np.isfinite(qq).all()):
            offsets = (qq[:, [0, 2]].reshape(-1, 2)-uv)@normal
            lo, hi = float(offsets.min()-3.), float(offsets.max()+3.)
            segment = uv+np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)*4))+1))[:, None]*normal
            low, high = np.floor(segment).astype(int), np.ceil(segment).astype(int)
            xy = np.vstack([np.c_[x[:, 0], y[:, 1]] for x in (low, high) for y in (low, high)])
            local = xy-topology['lo']
            inside_image = ((xy >= 0).all(1) & (xy < np.array(exclusion.shape[::-1])).all(1)).all()
            inside_roi = ((local >= 0).all(1) & (local < topology['hi']-topology['lo']).all(1)).all()
            if inside_image and inside_roi:
                unmasked = not exclusion[xy[:, 1], xy[:, 0]].any()
                all_material = bool(topology['actual_material'][local[:, 1], local[:, 0]].all())
                all_envelope = bool(topology['filled_component_envelope'][local[:, 1], local[:, 0]].all())
                probes = np.vstack([qq[0], uv+hi*normal])
                ij = np.rint(probes).astype(int); colours = rgb[ij[:, 1], ij[:, 0]].astype(float)
                blue = (colours[:, 2] > colours[:, 0]+8) & (colours[:, 2] > colours[:, 1]+3)
                evidence.update(actual_guard_normal_span_px=[lo, hi], entire_guard_unexcluded=unmasked,
                    entire_guard_inside_source_material=all_material, entire_guard_inside_source_envelope=all_envelope,
                    q25_q50_q75_and_outer_guard_uv=probes.tolist(),
                    q25_q50_q75_and_outer_guard_RGB=colours.tolist(),
                    q25_q50_q75_and_outer_guard_BGR=colours[:, ::-1].tolist(),
                    source_blue_material_at_all_probes=bool(blue.all()))
                if (unmasked and all_material and all_envelope and blue.all()
                        and 'actual_source_profile_truncated_or_reversed' not in exact[0]['pending_reasons']):
                    evidence.update(disposition='source_verified_interior_blue_transition_not_exterior',
                        withdrawal_authorized_by_this_audit=True,
                        withdrawal_reason='Unchanged RGB q50 and its complete quantile guard, including outward end, remain inside actual blue material; no board-white exterior transition exists at this readout.')
        audit.append(evidence)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--repair', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--shot', default='s09')
    a = p.parse_args(); root = a.root; sid = a.shot
    old_file = root/'qualified_full'/sid/'source_rows.jsonl'
    repair_file = a.repair/sid/'source_rows.jsonl'
    fresh = {r['pts']: r for l in repair_file.open(encoding='utf8') if (r:=json.loads(l))}
    old = {r['pts']: (n, r) for n,l in enumerate(old_file.open(encoding='utf8'),1)
        if (r:=json.loads(l))['pts'] in fresh}
    results = []
    for pts, current in fresh.items():
        line_number, previous = old[pts]
        image = cv2.imread(current['source_image']); rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(current['applied_exclusion'], cv2.IMREAD_GRAYSCALE) > 0
        topology = dict(np.load(current['profile_proofs']['source_topology']['path'], allow_pickle=False))
        rows = [(i, r) for i,r in enumerate(previous['rows']) if r.get('entity') == 'Ad3']
        evidence = audit_rows(rgb, topology, mask, rows, current['pending_rows'],
            source_hash_matches=previous['source_BGR_sha256'] == current['source_BGR_sha256'])
        groups = []
        for path_id in sorted({r['path_id'] for _,r in rows}):
            selected = [e for e in evidence if e['path_id'] == path_id]
            new_count = sum(r['path_id'] == path_id for r in current['rows'])
            groups.append(dict(shot=sid, pts=pts, path_id=path_id, old_rows=len(selected),
                source_verified_bad_rows=sum(e['withdrawal_authorized_by_this_audit'] for e in selected),
                keep_unproven_old_rows=sum(not e['withdrawal_authorized_by_this_audit'] for e in selected),
                new_pending_replacement_rows=new_count,
                empty_group_replacement_source_supported=bool(not new_count and selected
                    and all(e['withdrawal_authorized_by_this_audit'] for e in selected)), evidence=selected))
        results.extend(groups)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(old_capture=str(old_file),
        old_capture_sha256=hashlib.sha256(old_file.read_bytes()).hexdigest(),
        repair_capture=str(repair_file), repair_capture_sha256=hashlib.sha256(repair_file.read_bytes()).hexdigest(),
        source_only_retraction_evidence=True, old_data_modified=False,
        no_replacement_is_not_withdrawal_evidence=True, groups=results), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(a.output), groups=len(results),
        source_verified_bad_rows=sum(g['source_verified_bad_rows'] for g in results),
        kept_unproven_rows=sum(g['keep_unproven_old_rows'] for g in results))), flush=True)


if __name__ == '__main__': main()