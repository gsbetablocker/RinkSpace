"""Prepare only the three OOM source retries, preserving six completed cuts.

Check-only authenticates the predecessor, terminal receipt, exact clocks and
actual RGB role evidence without writing a manifest or launching a reader.
The final manifest uses the same full source ranges and explicit lossless spill.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from calibration.program_tracking.strict_model_scope import authenticate_source_reference, file_sha

WORK = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/'
    'current56_shared_active_set_filter_20261009/systematic_continuation_20261010/'
    'full_programme_strict_source_review_20261012')
PREDECESSOR = WORK/'source_inventory/source_full_precision_continuation5_FROZEN_v1.json'
PREDECESSOR_SHA = 'ba4e0cbb32ddf9231194195a6c67088c4d31961286704364a1c6bc9598a43c80'
BATCH = WORK/'source_full_precision_continuation5_v1'
TERMINAL = BATCH/'batch_receipt.json'
TERMINAL_SHA = '9c9e182fdd6e63d2c94a4a9ccd25c0088e45d5aafc67864616d091ed10a6540b'
ROLE_FACTS = ROOT/'events/ankara_2026/model_constraints/source_material_operator_corrections_straight_connections_20261012.json'
RETRY = {'223403520:R0402_034', '223403520:R0402_038', '230400000:R0410_004'}
SUCCESS = {
    '230400000:R0410_001': '58d13a61af8486667f77f73d1d3698f3ad23db5b8d3dde61e8f0d34fbb228e41',
    '230400000:R0410_003': '5636c34706d3043ec21ad2f7a9bd79e92b48fedc40ce5bca7f59484a8c7d4936',
}
MUTABLE_CODE = {
    ROOT/'scripts/admit_programme_source_inputs.py',
    ROOT/'calibration/program_tracking/compact_source_json.py',
    ROOT/'calibration/program_tracking/strict_absolute_source.py',
}
CLOCK_FIELDS = ('shot_id', 'start_pts', 'stop_pts', 'native_count', 'canonical_count')
ROLE_NAMES = {'curling_centerline_row3:segment0', 'house_connection_F2_R2:segment0',
    'house_connection_F3_R3:segment0'}


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def prepare_retry(output=None):
    output = None if output is None else Path(output).resolve()
    if output is not None and output.exists():
        raise FileExistsError('Preserve an earlier frozen source retry')
    predecessor_ref = dict(path=str(PREDECESSOR), sha256=PREDECESSOR_SHA)
    terminal_ref = dict(path=str(TERMINAL), sha256=TERMINAL_SHA)
    authenticate_source_reference(predecessor_ref)
    authenticate_source_reference(terminal_ref)
    predecessor = json.loads(PREDECESSOR.read_text(encoding='utf8'))
    terminal = json.loads(TERMINAL.read_text(encoding='utf8'))
    if (predecessor.get('schema') != 'rinkspace.exact-source-cut-batch.v1'
            or terminal.get('schema') != 'rinkspace.exact-source-cut-batch-receipt.v1'
            or terminal.get('manifest_reference') != predecessor_ref
            or len(predecessor['cuts']) != 5
            or len(predecessor.get('preserved_completed_source_cuts', ())) != 4):
        raise ValueError('Exact original five-cut predecessor and terminal receipt required')
    terminals = {cut['shot_id']: cut for cut in terminal['cuts']}
    if len(terminals) != 5 or set(terminals) != RETRY | set(SUCCESS):
        raise ValueError('Only the original five terminal source cuts may be considered')
    common = list(predecessor['common_arguments'])
    if '--spill-source-state' in common or str(ROLE_FACTS) in common:
        raise ValueError('Predecessor already includes this successor operation')
    if (common[common.index('--native-alias-review')+1] != 'rinkspace.native_alias_exact_query_review:review_native_alias'
            or common[common.index('--board-reader')+1] != 'rinkspace.vision_board_typed_material_clearance:TypedMaterialBoardSourceReader'
            or predecessor.get('ice_graph_class') != 'rinkspace.vision_ice_final_precision_graph:PreciseSourceIceMaterialIdentityGraph'):
        raise ValueError('Existing final precision, exact alias and typed board interfaces required')
    common += ['--spill-source-state', '--source-role-overrides', str(ROLE_FACTS)]
    references = {}

    def add(reference):
        reference = dict(reference)
        path = Path(reference['path']).resolve()
        key = str(path)
        reference['path'] = key
        prior = references.get(key)
        if prior is not None and prior['sha256'] != reference['sha256']:
            raise ValueError('One source reference has competing byte identities: '+key)
        authenticate_source_reference(reference)
        references[key] = reference
        return reference

    successors = []
    for original in predecessor['references']:
        reference = dict(original)
        path = Path(reference['path']).resolve()
        if path in MUTABLE_CODE:
            reference['sha256'] = file_sha(path)
            if reference['sha256'] != original['sha256']:
                successors.append(dict(path=str(path), predecessor_sha256=original['sha256'],
                    successor_sha256=reference['sha256']))
        # Every other previously bound source and authority keeps its old SHA.
        # Unexpected public implementation changes also fail authentication.
        add(reference)
    for path in (PREDECESSOR, TERMINAL, Path(__file__), ROOT/'src/rinkspace/source_state_spill.py',
            ROOT/'scripts/diagnostics/consolidate_full_precision_source_continuation_20261012.py'):
        add(_reference(path))
    identity_path = Path(common[common.index('--identity')+1]).resolve()
    identity_ref = references[str(identity_path)]
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    camera_ref = add(dict(path=identity['camera_path'], sha256=identity['camera_sha256']))
    with np.load(camera_ref['path'], allow_pickle=False) as camera:
        native = camera['pts'].copy()
        actual = camera['exposure_representative_pts'].copy()
        camera_metadata = json.loads(str(camera['metadata_json'].item()))
    source = dict(video_sha256=camera_metadata['source_sha256'],
        time_base=camera_metadata['source_time_base'], pixel_domain='native1920x1080')
    old_path = Path(common[common.index('--current-points')+1]).resolve()
    with np.load(old_path, allow_pickle=False) as original:
        old_pts = original['actual_pts'].copy()
        old_offsets = original['offsets'].copy()
    if not np.array_equal(old_pts, np.unique(actual)):
        raise ValueError('Exact original retirement clock required')

    roles = json.loads(ROLE_FACTS.read_text(encoding='utf8'))
    if (roles.get('schema') != 'rinkspace.actual-source-material-role-corrections.v1'
            or roles.get('source_sha256') != source['video_sha256']
            or roles.get('time_base') != source['time_base']
            or {r['curve_id'] for r in roles['records']} != ROLE_NAMES
            or len(roles['records']) != len(ROLE_NAMES)
            or set(roles.get('source_observation_overlap_review_required', ())) !=
                {'curling_centerline_row3:segment0', 'house_connection_F3_R3:segment0'}
            or roles.get('physical_curve_alias_authenticated') is not False
            or roles.get('duplicate_FIT_budget_from_curve_names_permitted') is not False):
        raise ValueError('Three actual source roles with explicit overlapping-source review required')

    def evidence_refs(value):
        if isinstance(value, list):
            for item in value: evidence_refs(item)
        elif isinstance(value, dict):
            if value.get('path') and value.get('sha256'): add(value)
            for item in value.values(): evidence_refs(item)

    evidence_refs(roles)
    role_ref = add(_reference(ROLE_FACTS))
    for role in roles['records']:
        if (role.get('original_source_image_actually_viewed') is not True
                or role.get('model_projection_or_fit_residual_used_for_role') is not False
                or role.get('confirmed_measurement_operator') != 'paint_center'
                or role.get('confirmed_material_signal') != 'dark'
                or role.get('cached_old_dark_candidates_can_be_approved_by_relabeling') is not False
                or role.get('current_source_dark_profile_remeasurement_required') is not True
                or role.get('source_operator_role_is_not_a_new_identity_seed') is not True
                or role['actual_pts'] != role['native_pts'] or role['actual_pts'] not in old_pts):
            raise ValueError('New roles require actual native source evidence and fresh current RGB reading')
        image_ref = role['source_role_evidence']['exact_programme_original_RGB']
        import cv2
        import hashlib
        bgr = cv2.imread(image_ref['path'])
        if (bgr is None or bgr.shape != (1080,1920,3)
                or hashlib.sha256(bgr.tobytes()).hexdigest() != role['decoded_native_BGR_sha256']
                or hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest() != role['decoded_native_RGB_sha256']):
            raise ValueError('Role proof does not match its exact original native RGB/BGR')
        del bgr

    def validate_completed(cut, absolute_ref, alias_ref, retirement_ref):
        mask = (native >= cut['start_pts']) & (native < cut['stop_pts'])
        cut_native = native[mask]; cut_actual = np.unique(actual[mask])
        absolute_ref, alias_ref, retirement_ref = map(add, (absolute_ref,alias_ref,retirement_ref))
        with np.load(absolute_ref['path'], allow_pickle=False) as pack:
            metadata = json.loads(str(pack['metadata_json'].item()))
            if (metadata.get('source') != source or metadata.get('legacy_family_fallback') != 'forbidden'
                    or metadata.get('physically_compacted_approved_pixels_only') is not True
                    or not np.array_equal(metadata['expected_actual_pts'],old_pts)
                    or not np.array_equal(metadata['processed_actual_pts'],cut_actual)):
                raise ValueError('Completed source cut has a differing actual source clock or gate')
            frame_pts=pack['source_frame_actual_pts']; frame_native=pack['source_frame_native_pts']; hashes=pack['source_frame_BGR_sha256']
            lookup={int(pts):i for i,pts in enumerate(frame_pts)}
            for pts in cut_actual:
                at=lookup[int(pts)]
                if frame_native[at]!=pts or len(hashes[at])!=64:
                    raise ValueError('Completed source exposure lost its own exact BGR binding')
            for reference in metadata['original_references']: add(reference)
        with np.load(alias_ref['path'], allow_pickle=False) as alias:
            representative = cut_native == actual[mask]
            if (not np.array_equal(alias['native_pts'],cut_native)
                    or not np.array_equal(alias['actual_pts'],actual[mask])
                    or not np.array_equal(alias['representative'],representative)
                    or not np.array_equal(alias['alias_native_pts'],cut_native[~representative])
                    or not np.array_equal(alias['alias_actual_pts'],actual[mask][~representative])
                    or not alias['native_material_checked'].all()):
                raise ValueError('Completed source cut omitted an exact native alias check')
        with np.load(retirement_ref['path'], allow_pickle=False) as retired:
            retirement = json.loads(str(retired['metadata_json'].item()))
            disposition=retired['old_pixel_disposition']; ids=retired['retired_original_pixel_indices']
            if (retirement.get('source') != source
                    or retirement.get('default_legacy_admission') != 'forbidden'
                    or retirement.get('original_pixel_reference') != references[str(old_path)]
                    or retirement['approved_absolute_inputs']['sha256'] != absolute_ref['sha256']
                    or not np.array_equal(retired['exposure_pts'],old_pts)
                    or not np.array_equal(retired['exposure_offsets'],old_offsets)
                    or len(disposition)!=old_offsets[-1]
                    or not np.array_equal(ids,np.flatnonzero(disposition>0))):
                raise ValueError('Completed cut lost the exact original retirement indices/clock')
            for i,pts in enumerate(old_pts):
                if pts not in cut_actual and np.any(disposition[old_offsets[i]:old_offsets[i+1]]):
                    raise ValueError('A completed cut retired pixels outside its original actual clock')
        return dict(cut, absolute_inputs=absolute_ref, native_alias_review=alias_ref,
            old_retirement_indices=retirement_ref, original_path_reused=True,
            processing_complete_is_not_manual_visual_completion=True)

    preserved=[]; retries=[]; seen=set()
    all_cuts=sorted([*predecessor['preserved_completed_source_cuts'],*predecessor['cuts']],key=lambda c:c['start_pts'])
    for cut in all_cuts:
        mask=(native>=cut['start_pts'])&(native<cut['stop_pts']); clock=native[mask]
        if (not len(clock) or len(clock)!=cut['native_count']
                or len(np.unique(actual[mask]))!=cut['canonical_count']
                or clock[0]!=cut['start_pts'] or clock[-1]+1!=cut['stop_pts']
                or seen.intersection(map(int,clock))):
            raise ValueError('Original nine source cuts must keep their complete exact ranges')
        seen.update(map(int,clock))
        if 'absolute_inputs' in cut:
            preserved.append(validate_completed(cut,cut['absolute_inputs'],cut['native_alias_review'],cut['old_retirement_indices']))
            continue
        row=terminals[cut['shot_id']]
        if any(row[key]!=cut[key] for key in CLOCK_FIELDS):
            raise ValueError('Terminal receipt changed the original cut clock')
        if cut['shot_id'] in RETRY:
            if row['exit_code']!=1:
                raise ValueError('Only the original OOM failures may replay')
            log_path=Path(row['log_path'])
            if log_path.parent.resolve()!=BATCH.resolve() or 'MemoryError' not in log_path.read_text(encoding='utf8'):
                raise ValueError('Each retry must preserve its actual OOM terminal evidence')
            add(_reference(log_path)); retries.append(dict(cut))
        else:
            if row.get('exit_code')!=0 or row.get('sha256')!=SUCCESS[cut['shot_id']]:
                raise ValueError('Preserve the exact complete successful source output')
            path=Path(row['output_path']).resolve()
            if path.parent!=BATCH.resolve():raise ValueError('Successful source output moved')
            preserved.append(validate_completed(cut,dict(path=str(path),sha256=row['sha256']),
                _reference(path.with_name(path.stem+'_native_alias_review.npz')),
                _reference(path.with_name(path.stem+'_retired_old_source_indices.npz'))))
    if seen!=set(map(int,native)) or len(preserved)!=6 or {c['shot_id'] for c in retries}!=RETRY:
        raise ValueError('Six completed and three retry cuts must cover the full original programme')
    result=dict(predecessor)
    result.update(common_arguments=common,cuts=retries,references=list(references.values()),
        preserved_completed_source_cuts=preserved,implementation_successors=successors,
        predecessor_manifest_references=[predecessor_ref],terminal_batch_receipt_reference=terminal_ref,
        new_source_role_facts_reference=role_ref,
        closeout_template_reference=predecessor_ref,
        closeout_existing_batch_directory=str(BATCH),
        closeout_uses_original_five_cut_template_and_successful_terminal_receipts=True,
        continuation_policy=dict(completed_source_cuts_reprocessed=False,
            complete_independent_source_replay_for_only_OOM_cuts=True,original_source_ranges_unchanged=True,
            lossless_source_state_spill=True,source_observations_or_temporal_graph_history_removed=False,
            fresh_current_RGB_roles_remeasured=True,old_candidates_relabelled=False,
            overlapping_source_curve_budget_review_still_required=True,
            interrupted_source_journals_preserved=True,native_alias_source_checks_skipped=False,
            solver_started=False,source_processing_is_not_manual_audit_completion=True))
    # A concurrent source change cannot be hidden by either check-only or freeze.
    for reference in references.values():authenticate_source_reference(reference)
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,separators=(',',':'))
    return dict(manifest_reference=None if output is None else _reference(output),frozen=output is not None,
        retry_cuts=retries,preserved_cut_ids=[c['shot_id'] for c in preserved],
        retained_successful_successor_cuts=sorted(SUCCESS),
        native_source_frames_to_replay=sum(c['native_count'] for c in retries),
        canonical_source_frames_to_replay=sum(c['canonical_count'] for c in retries),
        implementation_successors=successors,new_source_role_facts_reference=role_ref,
        authenticated_reference_count=len(references),closeout_template_reference=predecessor_ref,
        source_workers_launched=False,solver_started=False)


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare_retry(args.output),ensure_ascii=False,separators=(',',':')),flush=True)


if __name__=='__main__':main()
