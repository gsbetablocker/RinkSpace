"""Resume a complete representative primal through reviewed source revisions.

Retain the original raw denominator and fitting order. Explicit source withdrawals
remove only reviewed fit/inspection members; a canonical rebind retains their
pixels and weights. The original physical review baseline remains pinned.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import numpy as np
from .observed_primal_restart import read, checked, reference, write_new, checkpoint_header, OBJECTIVE
from .joint_metrics import PacketArrays
from .dense_joint_run import representative_identity_ancestry, prepare_run
from .dense_live_review import prepare_anchor_review


def rebind_current_packet_policy(cfg, previous_packet, current_packet):
    """Rebind source policy metadata only after its original source rows match."""
    from .compact_identity_withhold_v2 import _source_identities
    old_ref, new_ref = reference(previous_packet), reference(current_packet)
    stage = cfg.get('runtime_shared_parameter_stage_v2')
    if not stage:
        return dict(source_policy_rows_checked=0)
    if stage.get('source_packet_sha256') not in (old_ref['sha256'], new_ref['sha256']):
        raise ValueError('Shared physical policy binds an unrelated source generation')
    if stage.get('world_values_measured') is not False:
        raise ValueError('Source revision cannot promote unmeasured geometry')
    reader = PacketArrays(current_packet)
    admission, role, uv, pts, material, paths, shot = [reader.load(k) for k in
        ('admission', 'fit_role', 'uv', 'native_pts', 'material_index', 'path', 'shot_index')]
    restarts = stage.get('finite_line_navigation_restarts', [])
    evidence = [r for restart in restarts for r in restart.get('source_rows', [])]
    identities, _ = _source_identities(current_packet, [r['row_index'] for r in evidence])
    for restart in restarts:
        if restart['source_packet_sha256'] not in (old_ref['sha256'], new_ref['sha256']):
            raise ValueError('Finite source coverage binds an unrelated packet')
        for ref in restart['source_evidence']:
            checked(ref)
        for row in restart['source_rows']:
            i = int(row['row_index'])
            if (admission[i] != 1 or role[i] != 1 or identities[i] != row['source_row_id']
                    or uv[i].tolist() != row['source_uv'] or int(pts[i]) != row['native_pts']
                    or reader.packet['shot_ids'][int(shot[i])] != row['shot_id']
                    or int(material[i]) != restart['material_index'] or int(paths[i]) != restart['path']):
                raise ValueError('Original finite coverage source was changed or withdrawn')
            checked(row['native_png']); checked(row['source_report'])
        restart['source_packet_sha256'] = new_ref['sha256']
    stage['source_packet_sha256'] = new_ref['sha256']
    return dict(source_policy_rows_checked=len(evidence), physical_parameter_values_changed=False,
                previous_packet=old_ref, current_packet=new_ref)


def preserve_unobserved_ice(cfg, model, reader):
    """Keep unsupported initial geometry unknown; do not optimize it as observed."""
    stage = cfg['runtime_shared_parameter_stage_v2']
    previous = cfg.pop('source_unsupported_ice_initializers_v1', {}).get('blocks', [])
    releases = {x['block_index']: x for x in stage['blocks']}
    audits = {x['block_index']: x for x in cfg['audit_expected_free_physical_blocks']}
    fixed = list(cfg.get('temporary_runtime_fixed_physical_blocks', []))
    for old in previous:
        i = old['release']['block_index']
        releases[i], audits[i] = old['release'], old['audit']
        if old['release']['block_name'] in fixed:
            fixed.remove(old['release']['block_name'])
    mat, paths, admission, role = [reader.load(k) for k in ('material_index', 'path', 'admission', 'fit_role')]
    active = np.flatnonzero((admission == 1) & (role == 1))
    keys = np.unique((np.asarray(mat[active], dtype=np.uint64) << 32) | np.asarray(paths[active], dtype=np.uint64))
    support = set()
    for key in keys:
        mi, pi = int(key) >> 32, int(key) & 0xffffffff
        material = model['materials'][mi]; path = material['paths'][pi]
        if (material['label'].startswith('ice.') and not path.get('physical_point', False)
                and path.get('kind') in (0, 1, 4, 5) and path.get('geometry_block', -1) >= 0):
            support.add(path['geometry_block'])
    held = []
    for i, release in list(releases.items()):
        if release['block_name'].startswith('ice/') and i not in support:
            held.append(dict(release=release, audit=audits.pop(i)))
            releases.pop(i)
            if release['block_name'] not in fixed:
                fixed.append(release['block_name'])
    stage['blocks'] = list(releases.values())
    cfg['audit_expected_free_physical_blocks'] = list(audits.values())
    cfg['temporary_runtime_fixed_physical_blocks'] = fixed
    cfg['source_unsupported_ice_initializers_v1'] = dict(
        blocks=held, source_packet_sha256=reader.packet['source_packet_sha256'] if 'source_packet_sha256' in reader.packet else stage['source_packet_sha256'],
        source_qualification_is_measurement=False, physical_values_changed=False,
        reason='all observed support withdrawn; initial geometry remains unknown')
    return [x['release']['block_name'] for x in held]


def prepare(*, producer_spec, candidate, packet, qualification, weights, native, prepared, output):
    producer = read(producer_spec)
    cfg = read(checked(producer['configuration']))
    candidate, packet = Path(candidate).resolve(), Path(packet).resolve()
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if prepared.exists() or output.exists():
        raise FileExistsError('Reviewed continuation outputs must be new')
    if cfg.get('source_uv_capture_objective') != OBJECTIVE or cfg.get('representative_full_source_transition_v1'):
        raise ValueError('Expected a complete observed-row representative producer')
    previous_ref = cfg['representative_source_selection_v1']
    previous = read(checked(previous_ref))
    binding_ref = reference(candidate / 'observed_row_initialization_state_binding.json')
    binding = read(checked(binding_ref))
    cp_ref = reference(candidate / 'engineering_checkpoint.json')
    header = checkpoint_header(checked(cp_ref))
    nuisance_ref = reference(candidate / 'explicit_nuisance/manifest.json')
    nuisance = read(checked(nuisance_ref))
    if (header['source_packet_sha256'] != producer['packet']['sha256']
            or binding.get('source_packet_sha256') != producer['packet']['sha256']
            or binding.get('state_token') != header['state_token']
            or binding.get('objective_id') != OBJECTIVE
            or binding.get('true_RS_verified') is not False
            or binding.get('initialization_only') is not True
            or previous['source_packet'] != producer['packet']
            or nuisance.get('packet_sha256') != producer['packet']['sha256']
            or nuisance.get('raw_rows') != previous['source_row_count']):
        raise ValueError('Producer checkpoint/source/selection mismatch')
    n = int(nuisance['raw_rows'])
    for name, width in (('station.f64', 8), ('row.f64', 8), ('alpha.f64', 8), ('path.u32', 4)):
        item = nuisance['files'][name]
        path = candidate / 'explicit_nuisance' / name
        checked(dict(file=str(path), sha256=item['sha256']))
        if path.stat().st_size != n * width or item['size_bytes'] != n * width:
            raise ValueError('Incomplete raw auxiliary column: ' + name)
    packet_ref, qual_ref, weights_ref = reference(packet), reference(qualification), reference(weights)
    document = read(packet)
    key = ('representative_source_withdrawal_v1' if 'native_material_identity_revision' in document
           else 'representative_material_rebind_v1')
    revision_ref = document['native_material_identity_revision' if key.endswith('withdrawal_v1') else 'native_print_identity_rebind']
    chain = representative_identity_ancestry(producer['packet']['sha256'], packet_ref, revision_ref, qual_ref)
    if chain['raw_rows'] != n:
        raise ValueError('Reviewed source changed raw denominator')
    q, w = read(qualification), read(weights)
    if q['source_packet'] != packet_ref or w['source_packet'] != packet_ref or w['dense_source_qualification'] != qual_ref:
        raise ValueError('Reviewed packet qualification/budget mismatch')
    reader = PacketArrays(packet)
    eligible = (reader.load('admission') == 1) & (reader.load('fit_role') == 1)
    if len(eligible) != n or int(eligible.sum()) != int(w['active_rows']):
        raise ValueError('Current qualified source denominator mismatch')
    withdrawn = np.asarray(chain['withheld_rows'], dtype=np.int64)
    if withdrawn.size and np.any(eligible[withdrawn]):
        raise ValueError('An explicitly withdrawn source became eligible again')
    prepared.mkdir(parents=True, exist_ok=False)
    selected = deepcopy(previous)
    selected.update(source_packet=packet_ref, eligible_rows=int(eligible.sum()))
    for name in ('fit', 'inspection'):
        old = np.memmap(checked(previous[name + '_rows']), dtype='<u8', mode='r')
        keep = eligible[old]
        if not np.all(np.isin(old[~keep], withdrawn)):
            raise ValueError('An unreviewed source member disappeared')
        values = np.asarray(old[keep], dtype='<u8')
        path = prepared / (name + '_rows.u64')
        values.tofile(path)
        selected[name + '_rows'] = reference(path)
        selected[name + '_count'] = len(values)
    if selected['fit_count'] + selected['inspection_count'] != selected['eligible_rows']:
        raise ValueError('Fit/inspection partition lost qualified sources')
    shot, pts = reader.load('shot_index'), reader.load('native_pts')
    fit = np.memmap(selected['fit_rows']['file'], dtype='<u8', mode='r')
    inspection = np.memmap(selected['inspection_rows']['file'], dtype='<u8', mode='r')
    per_shot = []
    for sid, name in enumerate(reader.packet['shot_ids']):
        before = (shot == sid) & eligible
        fi, ii = fit[shot[fit] == sid], inspection[shot[inspection] == sid]
        full_frames, fit_frames = np.unique(pts[before]), np.unique(pts[fi])
        if not np.array_equal(full_frames, fit_frames):
            raise ValueError('A native frame lost all fitting support: ' + name)
        per_shot.append(dict(shot=name, before=int(before.sum()), fit=len(fi), inspection=len(ii),
                             frames_before=len(full_frames), frames_after=len(fit_frames)))
    selected.update(per_shot=per_shot, reviewed_source_ancestry=[g for g in chain['generations']],
                    source_rows_deleted=False, retained_membership_order_unchanged=True)
    selection_ref = write_new(prepared / 'selection.json', selected)
    continuation = dict(schema='rinkfit.representative_complete_primal_continuation.v1',
        source_packet_sha256=producer['packet']['sha256'], complete_checkpoint=True,
        model=reference(candidate / 'model.json'), checkpoint=cp_ref, state_token=header['state_token'],
        representative_source_selection_v1=previous_ref, explicit_nuisance_manifest=nuisance_ref,
        source_state_binding=binding_ref, quality_accepted=False)
    cfg['representative_continuation_v1'] = write_new(prepared / 'continuation.json', continuation)
    cfg['representative_source_selection_v1'] = selection_ref
    for field in ('representative_source_withdrawal_v1', 'representative_material_rebind_v1'):
        cfg.pop(field, None)
    cfg[key] = revision_ref
    cfg['dense_source_qualification'], cfg['engineering_dense_source_weights_v1'] = qual_ref, weights_ref
    cfg['engineering_support_budget_v2']['source_packet_sha256'] = packet_ref['sha256']
    rebind_current_packet_policy(cfg, checked(producer['packet']), packet)
    preserve_unobserved_ice(cfg, read(checked(producer['model'])), reader)
    cfg['audit_expected_source_counts'] = {x['shot']: x['fit'] for x in per_shot}
    if cfg.get('accepted_source_anchor_review'):
        prior_review = read(checked(cfg['accepted_source_anchor_review']))
        prior_baseline = read(checked(prior_review['baseline']))
        cfg['accepted_source_anchor_review'] = prepare_anchor_review(
            cfg['accepted_source_anchor_review'], prior_baseline['model'], packet, prepared / 'anchor_review')
    cfg['source_uv_capture_construct_problem_only'] = False
    new_spec = deepcopy(producer)
    new_spec.update(packet=packet_ref, native=reference(native), output=str(output),
                    configuration=write_new(prepared / 'config.json', cfg))
    spec_ref = write_new(prepared / 'spec.json', new_spec)
    info = prepare_run(new_spec)
    report = dict(spec=spec_ref, complete_state_token=header['state_token'],
                  raw_rows=n, active_rows=int(eligible.sum()), fit_rows=selected['fit_count'],
                  inspection_rows=selected['inspection_count'], per_shot=per_shot,
                  rebound_rows=len(chain['rows']), withdrawn_rows=len(withdrawn),
                  native_command=info['command'], solver_launched=False, quality_accepted=False)
    write_new(prepared / 'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('producer-spec', 'candidate', 'packet', 'qualification', 'weights', 'native', 'prepared', 'output'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(prepare(**{k.replace('-', '_'): v for k, v in args.items()}), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

