"""Prepare the existing named66 nine-shot continuation inputs without solving.

Derives actual options from the frozen c13434340 launch, preserving every full
source correction and retired row. New source packets are append-only; no
solver graph, array, profile, cache or checkpoint is copied or rebuilt here.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from register_user_native230024192_all_visible import ROOT,DATA,BASE,SUFFIX,ref,write

RUNS=DATA/'calibration/runs/solver_acceleration_20261003'
FROZEN=RUNS/'okada_joint_morning_c134965_cached_fastQP_20261008_192005_957.bootstrap/frozen_input_config.json'
MANIFEST=RUNS/'okada_joint_morning_c134965_cached_fastQP_20261008_192005_957/input_manifest.json'
CONFIG=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user230024192_20261008.json'


def main():
    if CONFIG.exists():raise FileExistsError('Preserve prepared full-source config')
    frozen=json.loads(FROZEN.read_text(encoding='utf8'));manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    options=deepcopy(frozen['source_options'])
    if manifest['full_parameters']!=34340:raise ValueError('Actual complete parent must be 34340')
    old_track_path=Path(options['--additional-native-board-track-source'])
    old_track=json.loads(old_track_path.read_text(encoding='utf8'))
    candidate_path=BASE/'native230024192_actual_RGB_source_candidate/native_six_parent_plus_Ad21_22_23_native230024192_actual_frozen_ink_ADMITTED_v1.json'
    candidate=json.loads(candidate_path.read_text(encoding='utf8'))
    added=[r for r in candidate['source_anchor_observations']if r['id'].startswith(('Ad21_user_identity_native230024192','Ad22_user_identity_native230024192','Ad23_user_identity_native230024192'))]
    if len(added)!=3 or sum(len(r['uv'])for r in added)!=57:raise ValueError('Exact admitted 57-point source append required')
    if old_track['frozen_source_print_templates']!=candidate['frozen_source_print_templates']:raise ValueError('Existing exact template owner differs')
    existing={r['id']for r in old_track['rows']+old_track['source_anchor_observations']}
    if any(r['id']in existing for r in added):raise ValueError('New source already present')
    full_track=deepcopy(old_track);full_track['source_anchor_observations'].extend(deepcopy(added))
    if 'fit_row_order'in full_track:full_track['fit_row_order'].extend(r['id']for r in added)
    full_track['same_frame_user_identity_actual_RGB_append']=deepcopy(candidate['same_frame_user_identity_actual_RGB_append'])
    full_track['same_frame_user_identity_actual_RGB_append'].update(actual_full_parent_packet=ref(old_track_path),
        admitted_source_candidate=ref(candidate_path),full_original_pan_tracks_dedup_and_all_HELD_preserved=True)
    assert full_track['rows']==old_track['rows']and full_track['frozen_source_print_templates']==old_track['frozen_source_print_templates']
    assert full_track['source_anchor_observations'][:len(old_track['source_anchor_observations'])]==old_track['source_anchor_observations']
    track_path=BASE/'native230024192_actual_RGB_source_candidate/native_full_parent_all_pan_plus_Ad21_22_23_native230024192_actual_frozen_ink_ADMITTED_v1.json'
    write(track_path,full_track)
    new_authority=BASE/f'human_registered_hard_truth_authorities_{SUFFIX}.json'
    authority=json.loads(new_authority.read_text(encoding='utf8'))
    old_policy_path=Path(options['--practical-human-location-envelope']);policy=json.loads(old_policy_path.read_text(encoding='utf8'))
    native_old=[r for r in policy['records']if r['family']=='native'];wide_old=[r for r in policy['records']if r['family']=='wide']
    if [r['id']for r in native_old]!=[r['id']for r in authority['records'][:10]]:raise ValueError('Original manual engineering policy groups differ')
    rows=deepcopy(native_old)
    for record in authority['records'][10:]:
        rows.append(dict(family='native',id=record['id'],full_vertices=record['complete_human_stroke']['sample_count'],
            normal_expansion_px=0.,tangent_expansion_px=0.,engineering_uncorrelated_location_scale_px=6.,
            location_scale_is_not_a_user_accuracy_threshold=True,normal_and_tangent_manual_location_already_bound_in_source=True,
            manual_location_uncertainty_native_px=6.,recovery_uncertainty_is_not_additional_support_expansion=True,
            physical_endpoint_or_world_dimension_observed=False))
    rows.extend(deepcopy(wide_old));policy['records']=rows
    policy['authority_references']['native']=ref(new_authority)
    policy['previous_location_policy_reference']=ref(old_policy_path)
    policy['purpose']='Preserve original ten native and two wide drawing engineering envelopes; new22 source records already include6px drawing error, no second expansion.'
    policy['full_source_resume_note']=dict(original_native10_and_wide2_policy_records_exact=True,new22_guard_bands_already_include6px_manual_location=True,
        new_normal_and_tangent_extra_expansion_px=0.,restoration_uses_engineering_uncorrelated_location_scale_px=True)
    policy_path=BASE/'fixed_practical_manual_location_envelope_full_user230024192_v1.json';write(policy_path,policy)
    replacements={
        '--additional-board-source':BASE/f'board_sources_{SUFFIX}.json',
        '--additional-ice-source':BASE/f'ice_sources_{SUFFIX}.json',
        '--direct-human-truth-authority':new_authority,
        '--direct-human-full-stroke-guard':BASE/f'human_registered_full_stroke_hard_guard_{SUFFIX}.json',
        '--direct-human-stroke-cap-support':BASE/f'human_registered_stroke_end_2D_source_footprint_support_{SUFFIX}.json',
        '--user-confirmed-source-truth-authority':BASE/f'user_confirmed_native_source_priority_authorities_{SUFFIX}.json',
        '--additional-native-board-track-source':track_path,
        '--practical-human-location-envelope':policy_path,
    }
    for key,path in replacements.items():options[key]=str(path)
    # A changed q and appended target cannot use the old complete derivative
    # normal. Existing same-run RAM reuse stays enabled after the fresh normal.
    for key in ('--reuse-complete-linearization','--reuse-source-blocks-from','--reuse-wide-linearization','--verify-source-blocks-from','--verified-source-prefix-report'):
        options.pop(key,None)
    baseline=RUNS/'okada_joint_compact_fastcamera_c134_20261008_104825_235/initial_states/965e58d4c3ac40833090d4d9db639dd2870dbb17558d449893d10b72405f4031/complete_state_UNACCEPTED.npz'
    options['--camera-seed']=str(baseline)
    with np.load(baseline,allow_pickle=False)as state:
        if len(state['q'])!=34340:raise ValueError('Complete baseline34340 required')
        profile_names=[k for k in state.files if k.startswith('profile__')];aux_names=[k for k in state.files if k.startswith('aux__')]
        identity=json.loads(str(state['identity_json'].item()))
    previous_board=json.loads(Path(frozen['source_options']['--additional-board-source']).read_text(encoding='utf8'))
    previous_ice=json.loads(Path(frozen['source_options']['--additional-ice-source']).read_text(encoding='utf8'))
    current_board=json.loads(Path(options['--additional-board-source']).read_text(encoding='utf8'))
    current_ice=json.loads(Path(options['--additional-ice-source']).read_text(encoding='utf8'))
    assert current_board['rows'][:len(previous_board['rows'])]==previous_board['rows']
    assert current_ice['rows'][:len(previous_ice['rows'])]==previous_ice['rows']
    added_scalars=sum(2*len(r['uv'])for now,old in [(current_board,previous_board),(current_ice,previous_ice)]for r in now['rows'][len(old['rows']):]if r['split']=='fit')+2*57
    unchanged_options={k:v for k,v in options.items()if k not in replacements and k!='--camera-seed'}
    contract=dict(schema='rinkspace.reusable-solver-inputs.v1',resource_root=frozen['resource_root'],driver=frozen['driver'],source_options=options,
        event_id='ankara_2026',programme_id='okada_mei_sp',planned_parameter_count=34340,planned_world_parameter_count=100,
        planned_source_rows=manifest['source_rows']+added_scalars,actual_full_source_input_manifest_reference=ref(MANIFEST),
        source_options_from_actual_frozen_launch=ref(FROZEN),complete_c134_baseline=ref(baseline),
        default_seed_q_sha256=identity.get('q_sha256'),profile_array_count=len(profile_names),auxiliary_array_count=len(aux_names),
        full_original_source_options_preserved=unchanged_options,new_source_references={k:ref(p)for k,p in replacements.items()},
        new_source_scalar_rows=added_scalars,new_manual_stroke_groups=22,new_actual_RGB_frozen_print_points=57,
        source_clock=dict(time_base='1/15360',native_start_inclusive=229114112,native_stop_exclusive=231705600,native_presentations=10123,shots=9),
        required_seed_override='Use START_FAST_SOLVER.ps1 -Seed with the visually reviewed complete34340 same-q export checkpoint. No solver is started by this config preparation.',
        resume_command=dict(script=str(ROOT/'START_FAST_SOLVER.ps1'),InputConfig=str(CONFIG),Seed='REVIEWED_COMPLETE_34340_CHECKPOINT',Output='NEW_DATA_RUN_DIRECTORY'),
        model_contract=dict(shared_hockey_curling_axis=True,continuous_finite_physical_course=True,print_locations_independent=True,
            manual_stroke_endpoints_are_not_physical_endpoints=True,new_manual_location_not_pixel_exact=True,new_manual_error_not_double_expanded=True,
            actual_RGB_source_identity_and_manual_pixels_separate=True),
        source_preservation=dict(original_full_FIT_HELD_rows_and_all_retirements_preserved=True,old_F2_pixel_correction_preserved=True,
            all_main9_pan_and_source_dedup_roles_preserved=True,all18_profiles_and48_auxiliary_restored_from_requested_checkpoint=True,
            underlying_solver_not_copied_or_rebuilt=True,old_complete_J_cache_not_requested_for_changed_target=True),
        required_existing_driver_compatibility='truth restoration uncertainty reads engineering_uncorrelated_location_scale_px when present; new policy normal_expansion0 must not double-add manual6',
        prepared_only=True,solver_started=False,accuracy_accepted=False)
    if len(profile_names)!=18 or len(aux_names)!=48:raise ValueError('Expected complete18profile48aux baseline')
    write(CONFIG,contract)
    print(json.dumps(dict(config=str(CONFIG),full_track_packet=str(track_path),new_manual_policy=str(policy_path),new_source_scalars=added_scalars,
        old_source_rows=manifest['source_rows'],planned_source_rows=contract['planned_source_rows'],old_track_groups=len(old_track['rows']),old_anchor_groups=len(old_track['source_anchor_observations']),
        new_anchor_groups=len(full_track['source_anchor_observations']),profiles=len(profile_names),aux=len(aux_names),prepared_only=True),ensure_ascii=False))


if __name__=='__main__':main()
