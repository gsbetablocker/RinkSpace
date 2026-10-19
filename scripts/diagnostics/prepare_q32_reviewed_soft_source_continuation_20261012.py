"""Bind the reviewed source revisions to q32 without building its solver graph."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from calibration.solver_runtime.reviewed_source_continuation import (
    OPTION, prepare_engineering_reference_rebind, prepare_reviewed_source_revision)
from calibration.solver_runtime import truth_source_location_revision as common

CONFIG = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_q32_current_RGB_20261012.json'
SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'q32_reviewed_soft_source_continuation_20261012'


def bind_unchanged_confirmed_circle_rows(options, rebound_files):
    """Rebind a source container only when every protected row is byte exact."""
    options = deepcopy(options)
    before_ref = common.reference(options[common.CONFIRMED])
    before = common._read(common._checked(before_ref)); after = deepcopy(before)
    circle_ref = common.reference(options['--additional-board-circle-source'])
    circle = common._read(common._checked(circle_ref)); checks = []
    for record in after['records']:
        if record.get('family') != 'native_board_circle':continue
        old_ref = record['packet_reference']; old = common._read(common._checked(old_ref))
        index = record['original_row_index']; original = old['rows'][index]; current = circle['rows'][index]
        if (not common._same(original, current)
                or record['source_row_canonical_json_sha256'] != common._row_hash(original)):
            raise ValueError('Circle UV/support revision intersects a protected confirmed row')
        checks.append(dict(id=record['id'], original_row_index=index,
            source_row_canonical_json_sha256=common._row_hash(original), entire_row_unchanged=True))
        record['packet_reference'] = deepcopy(circle_ref)
    if common._same(before, after):return options, rebound_files, []
    reference = common._write(OUT / ('confirmed_authority_circle_reference_rebind_' +
        hashlib.sha256(common.canonical(after)).hexdigest()[:16] + '.json'), after)
    files = deepcopy(rebound_files)
    item = next(r for r in files if r['option_key'] == common.CONFIRMED)
    item['derived_reference'] = reference
    options[common.CONFIRMED] = reference['path']
    receipt = dict(schema='rinkspace.protected-confirmed-circle-container-only-rebind.v1',
        original_confirmed_authority=before_ref, revised_confirmed_authority=reference,
        actual_circle_source=circle_ref, protected_circle_rows=checks,
        only_protected_record_packet_reference_rebound=True,
        every_original_protected_row_UV_sigma_support_world_clock_unchanged=True,
        other_original_confirmation_and_measurement_evidence_refs_preserved=True)
    evidence = common._write(OUT / ('confirmed_circle_reference_rebind_receipt_' +
        hashlib.sha256(common.canonical(receipt)).hexdigest()[:16] + '.json'), receipt)
    return options, files, [evidence]


def main():
    config = common._read(CONFIG)
    receipt_ref = config['native_ice_source_revision_receipt']
    receipt = common._read(common._checked(receipt_ref))
    engineering = prepare_engineering_reference_rebind(config['source_options'], OUT)
    if OPTION in config['source_options']:
        previous = common._read(common._checked(common.reference(config['source_options'][OPTION])))
        rebound = deepcopy(previous['rebound_files'])
    else:
        rebound = receipt['rebound_files'] + engineering['rebound_files']
    options, rebound, circle_evidence = bind_unchanged_confirmed_circle_rows(engineering['source_options'], rebound)
    evidence = [receipt_ref, common.reference(SYS / 'original_source_81records_public_sampling_installation_20261012.json'),
        common.reference(SYS / 'original_motion_four_skate_points_public_source_validation_20261012.json'),
        common.reference(SYS / 'original81_qualification36_saved_measured_location_dependency_check_20261012.json'),
        common.reference(SYS / 'native_board_material_circle_RGB_review_20261012/native_circle16_C001_outer_interface_public_installation_receipt.json')] + circle_evidence
    sampling = evidence[1]
    expected = [dict(manifest_key='original_' + family + '_curve_integration', evidence_reference=sampling,
        value_path=['actual_sampling_integration', family]) for family in ('board', 'ice')]
    expected.append(dict(manifest_key='native_base_curve_integration', evidence_reference=receipt_ref,
        value_path=['native_integration_contract']))
    result = prepare_reviewed_source_revision(config['source_options']['--truth-restoration-state'],
        config['continuation']['original_q32_state_identity']['path'], options, OUT,
        evidence_references=evidence, rebound_files=rebound,
        expected_actual_contracts=expected)
    patch = {key: value for key, value in result['source_options'].items()
        if config['source_options'].get(key) != value}
    output = dict(schema='rinkspace.q32-reviewed-soft-source-continuation-options.v1',
        proof_reference=result['proof_reference'], source_options_patch=patch,
        exact_actual_source_refs_frozen=True, no_full_graph_residual_J_or_solver_started=True,
        real_startup_manifest_and_hard_factor_authentication_still_required=True)
    common._write(OUT / ('reviewed_soft_source_continuation_options_' + result['proof_reference']['sha256'][:16] + '.json'), output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
