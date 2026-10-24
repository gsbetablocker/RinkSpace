"""Prepare a declared new training objective from a complete representative state.

This prepares and verifies artifacts; it never starts a native solver. The raw
source packet, all four complete auxiliary columns and the old state remain
immutable. Native import must still verify its exact state round trip.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .observed_primal_restart import (OBJECTIVE, checked, checkpoint_header, read,
                                     reference, write_new, semantic_config)
from .joint_metrics import PacketArrays
from .dense_live_review import prepare_anchor_review

STALE_REVISIONS = ('representative_source_withdrawal_v1', 'representative_material_rebind_v1',
                   'representative_full_source_transition_v1')
DELTA_KEYS = {'representative_source_selection_v1', 'source_identity_verdicts_v1',
              'engineering_support_budget_v2', 'audit_expected_source_counts'}


def require(value, message):
    if not value:
        raise ValueError(message)

REPAIR_KEYS = {'engineering_global_phr_v1', 'legacy_line_coordinate_rechart_v1'}


def same_ref(left, right):
    return (left.get('sha256') == right.get('sha256') and
            Path(left['file']).resolve() == Path(right['file']).resolve())


def validate_repair_delta(path, producer, state):
    document = read(path)
    require(document.get('schema') == 'rinkfit.numerical_repair_delta.v1' and
            same_ref(document['source_packet'], producer['packet']) and
            same_ref(document['from_model'], state['model']) and
            document['from_state_token'] == state['state_token'], 'Numerical repair binds another complete source state')
    changes = document['changes']
    require(changes and set(changes) <= REPAIR_KEYS, 'Numerical repair delta contains unreviewed keys')
    native_evidence = False
    for proof in document['evidence']:
        checked(proof)
        report = read(checked(proof))
        native_evidence |= (report.get('schema') == 'rinkfit.actual_legacy_line_rechart_native_regression.v1'
                            and report.get('passed') is True and report.get('solver_started') is False)
    if 'legacy_line_coordinate_rechart_v1' in changes:
        policy = changes['legacy_line_coordinate_rechart_v1']
        require(native_evidence and policy['schema'] == 'rinkfit.legacy_line_coordinate_rechart.v1' and
                policy['allowed_change'] == 'equivalent_affine_chart_clip_to_existing_rink_without_source_clamping' and
                same_ref(policy['from_model'], state['model']) and
                policy['from_state_token'] == state['state_token'] and
                policy['source_packet_sha256'] == producer['packet']['sha256'],
                'Line rechart lacks actual native equivalence and complete import evidence')
        checked(policy['physical_admission'])
    if 'engineering_global_phr_v1' in changes:
        require(changes['engineering_global_phr_v1'] == dict(
            schema='rinkfit.engineering_global_phr_policy.v1', initialization='explicit_new_objective_zero_duals',
            initial_rho=10, rho_growth=5, required_progress_ratio=.5,
            primal_tolerance=1e-8, complementarity_tolerance=1e-5),
            'Global PHR defaults/initialization differ from the reviewed existing mathematics')
    return deepcopy(changes)



def validate_source_append_delta(path, producer, state, selected):
    document=read(path)
    require(document.get('schema')=='rinkfit.source_append_input_delta.v1' and
            same_ref(document['from_source_packet'],producer['packet']) and
            same_ref(document['source_packet'],selected['source_packet']), 'Append delta source binding')
    require(set(document['configuration'])=={'dense_source_qualification','engineering_dense_source_weights_v1','archived_dependency_aliases'},
            'Append configuration contains unreviewed changes')
    extension=selected['source_append_selection_v1'];proof_ref=extension['append_source_proof'];proof=read(checked(proof_ref))
    require(same_ref(extension['original_state_packet'],producer['packet']) and
            same_ref(proof['source_packet'],selected['source_packet']) and
            same_ref(proof['original_state_packet'],producer['packet']) and
            state['raw_rows']==extension['parent_raw_rows']==proof['parent_raw_rows'] and
            selected['source_row_count']==extension['current_raw_rows']==proof['current_raw_rows'], 'Append source/primal prefix denominator')
    current=read(checked(selected['source_packet']))
    require(same_ref(current['dense_parent_packet'],extension['parent_packet']) and
            current['native_ad_source_binding']==proof['source_registry'], 'Append actual parent/registry differs')
    for ref in document['configuration'].values(): checked(ref)
    qualification=read(checked(document['configuration']['dense_source_qualification']))
    weights=read(checked(document['configuration']['engineering_dense_source_weights_v1']))
    require(qualification['schema']=='rinkfit.native_surface_source_append_qualification.v2' and
            same_ref(qualification['source_packet'],selected['source_packet']) and
            same_ref(weights['source_packet'],selected['source_packet']) and
            same_ref(weights['dense_source_qualification'],document['configuration']['dense_source_qualification']), 'Append qualification/budget authority')
    policy=dict(schema='rinkfit.authenticated_source_append_primal.v1',
                from_source_packet=producer['packet'],source_packet=selected['source_packet'],parent_packet=extension['parent_packet'],
                append_source_proof=proof_ref,from_raw_rows=state['raw_rows'],to_raw_rows=selected['source_row_count'],
                from_state_token=state['state_token'],from_runtime_fingerprint=state['runtime_fingerprint'],
                from_model=state['model'],from_checkpoint=state['checkpoint'],from_nuisance_manifest=state['explicit_nuisance_manifest'],
                tail_source_columns={key:current['arrays'][key] for key in ('station','row','alpha','path')},
                old_domain_reseeded=False,tail_initialization='exact_authenticated_packet_station_row_alpha_path',same_objective_resume=False,
                reason='Preserve the complete r9 original physical/raw/motion primal; apply the exact source qualification withdrawals and initialize only independently re-read authentic C0 source rows. The changed training domain and budget initialize a new objective.')
    return policy,deepcopy(document['configuration'])

def inspect_candidate(producer, candidate):
    candidate = Path(candidate).resolve()
    checkpoint_ref = reference(candidate / 'engineering_checkpoint.json')
    header = checkpoint_header(checked(checkpoint_ref))
    model_ref = reference(candidate / 'model.json')
    model = read(checked(model_ref))
    require(all(math.isfinite(float(value)) for block in model['blocks'] for value in block['values']),
            'Candidate contains nonfinite physical parameters')
    binding_ref = reference(candidate / 'observed_row_initialization_state_binding.json')
    binding = read(checked(binding_ref))
    nuisance_ref = reference(candidate / 'explicit_nuisance/manifest.json')
    nuisance = read(checked(nuisance_ref))
    require(header['source_packet_sha256'] == producer['packet']['sha256'] == nuisance['packet_sha256'] == binding['source_packet_sha256'],
            'Complete candidate and producer packet differ')
    require(header['state_token'] == binding['state_token'] and binding['objective_id'] == OBJECTIVE and
            binding['initialization_only'] is True and binding['true_RS_verified'] is False and
            binding['formal_Query_closed'] is False and binding['phase_II_allowed'] is False,
            'Candidate is not the declared unaccepted observed-row state')
    require(nuisance['schema'] == 'rinkfit.joint_explicit_PHR_nuisance_state.v1', 'Unknown complete auxiliary schema')
    n = int(nuisance['raw_rows'])
    files = {}
    for name, width in (('station.f64', 8), ('row.f64', 8), ('alpha.f64', 8), ('path.u32', 4)):
        desc = nuisance['files'][name]
        ref = dict(file=str(candidate / 'explicit_nuisance' / name), sha256=desc['sha256'])
        path = checked(ref)
        require(path.stat().st_size == desc['size_bytes'] == n * width, 'Incomplete raw auxiliary: ' + name)
        files[name] = ref
    return dict(model=model_ref, checkpoint=checkpoint_ref, state_token=header['state_token'],
                runtime_fingerprint=header['fingerprint'], source_packet_sha256=header['source_packet_sha256'],
                explicit_nuisance_manifest=nuisance_ref, source_state_binding=binding_ref,
                raw_rows=n, files=files)


def configuration(producer_cfg, producer, state, delta, continuation, repair_changes=None):
    require(set(delta) == DELTA_KEYS, 'Training delta contains unreviewed changes')
    require(producer_cfg['source_uv_capture_objective'] == OBJECTIVE and
            producer_cfg['reinitialize_native_auxiliaries'] is False and
            producer_cfg['source_uv_capture_reseed_new_sources'] is False,
            'Only complete observed-row primal imports without reseeding are supported')
    cfg = deepcopy(producer_cfg)
    for key in STALE_REVISIONS:
        cfg.pop(key, None)
    cfg.update(deepcopy(delta))
    cfg['representative_continuation_v1'] = continuation
    cfg['representative_objective_transition_v1'] = dict(
        schema='rinkfit.representative_objective_transition.v1', kind='complete_primal_warm_start',
        from_state_token=state['state_token'], from_runtime_fingerprint=state['runtime_fingerprint'],
        from_checkpoint_sha256=state['checkpoint']['sha256'], from_model_sha256=state['model']['sha256'],
        to_objective_id=OBJECTIVE,
        reason='Original protected exposure roles and confirmed wrong-source verdicts remove training members; '
               'preserve original per-exposure budget and complete primal; disable the clipped optical chart; '
               'recover only direct logf controls to the existing certified interval. This is a new objective initialization.')
    cfg['runtime_shared_parameter_stage_v2']['optical_feasible_coordinates'] = False
    cfg['observed_optical_gap_blocks_v1'] = False
    cfg['capture_zoom_feasibility_recovery_v1'] = dict(
        schema='rinkfit.capture_zoom_feasibility_recovery.v1',
        allowed_change='project_only_direct_control_logf_to_existing_certified_interval',
        from_state_token=state['state_token'], from_source_packet_sha256=producer['packet']['sha256'],
        from_model=state['model'], certified_domain=cfg['capture_certified_numeric_domain'],
        reason='After exact complete primal import into the declared new training objective, restore direct '
               'logf controls to their existing interval; retain other physical values, all raw auxiliaries and motion.')
    if repair_changes:
        require(set(repair_changes) <= REPAIR_KEYS, 'Numerical repair delta contains unreviewed keys')
        cfg.update(deepcopy(repair_changes))
    cfg.update(write_accepted_physical_preview=True, accepted_preview_min_seconds_v1=120.,
               write_accepted_complete_checkpoint=True, accepted_checkpoint_min_seconds_v1=120.,
               source_uv_capture_construct_problem_only=False)
    return cfg


def support_audit(model, reader, previous, selected, stage):
    oldfit = np.fromfile(checked(previous['fit_rows']), '<u8')
    fit = np.fromfile(checked(selected['fit_rows']), '<u8')
    material, path, shot = (reader.load(key) for key in ('material_index', 'path', 'shot_index'))
    def counts(rows):
        support = Counter()
        pairs, amounts = np.unique(np.c_[material[rows], path[rows]], axis=0, return_counts=True)
        for (mi, pi), amount in zip(pairs, amounts):
            item = model['materials'][int(mi)]['paths'][int(pi)]
            for key in ('geometry_block', 'advertisement_block'):
                if item.get(key, -1) >= 0:
                    support[int(item[key])] += int(amount)
        return support
    before, after = counts(oldfit), counts(fit)
    same_s01 = np.array_equal(oldfit[shot[oldfit] == 0], fit[shot[fit] == 0])
    records, unsupported = [], []
    for block in stage['blocks']:
        name, index = block['block_name'], block['block_index']
        if name.startswith(('scene/', 'rig/', 'shot/')):
            continue
        entry = dict(block_index=index, block_name=name, previous_direct_material_rows=before[index],
                     current_direct_material_rows=after[index])
        records.append(entry)
        if after[index] == 0:
            unsupported.append(entry)
    require(not unsupported, 'Released physical blocks lost direct source support; explicit review required: ' + str(unsupported))
    require(same_s01, 'S01 control-support domain changed; native basis support must be audited explicitly')
    return dict(schema='rinkfit.training_transition_material_support.v1', records=records,
                zero_direct_support_released_blocks=unsupported, all_old_freedoms_retained=True,
                S01_selected_source_domain_bit_exact=True, support_is_not_observability_certificate=True,
                native_actual_dependency_and_tangent_guards_still_required=True, physical_values_changed=False)


def prepare(*, producer_spec, candidate, input_delta, prepared, output, numerical_repair_delta=None, source_append_delta=None):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    require(not prepared.exists() and not output.exists(), 'Use new immutable preparation and solve directories')
    producer = read(producer_spec)
    for key in ('model', 'packet', 'configuration'):
        checked(producer[key])
    original_cfg = read(checked(producer['configuration']))
    state = inspect_candidate(producer, candidate)
    delta = read(input_delta)
    require(set(delta) == DELTA_KEYS, 'Training delta contains unreviewed changes')
    selected = read(checked(delta['representative_source_selection_v1']))
    previous = read(checked(original_cfg['representative_source_selection_v1']))
    require(selected['schema'] == 'rinkfit.per_frame_representative_source_selection.v2' and
            same_ref(selected['original_selection'], original_cfg['representative_source_selection_v1']),
            'Training selection does not bind the exact producer representatives')
    append_policy, source_changes = None, None
    if source_append_delta:
        append_policy,source_changes=validate_source_append_delta(source_append_delta,producer,state,selected)
    else:
        require(same_ref(selected['source_packet'],producer['packet']) and selected['source_row_count']==state['raw_rows'],
                'Source append requires an explicit authenticated delta')
    target_packet=selected['source_packet']
    repair_producer=deepcopy(producer);repair_producer['packet']=target_packet
    repair = validate_repair_delta(numerical_repair_delta, repair_producer, state) if numerical_repair_delta else None
    require(selected['eligibility_provenance']['source_identity_verdicts'] == delta['source_identity_verdicts_v1'],
            'Native identity verdict differs from selection provenance')
    require(delta['audit_expected_source_counts'] == {row['shot']: row['fit'] for row in selected['per_shot']},
            'New expected fit counts differ from selected domain')
    verify_path = Path(input_delta).parent / 'verification.json'
    verified = read(verify_path)
    require(verified['passed'] is True and verified['manifest'] == delta['representative_source_selection_v1'] and
            same_ref(verified['source_packet'], target_packet) and verified['model'] == state['model'] and
            verified['config'] == producer['configuration'], 'Full current training-input verification is missing or stale')
    reader = PacketArrays(checked(target_packet))
    model = read(checked(state['model']))
    require(reader.packet['row_count'] == selected['source_row_count'], 'Selected and actual raw source denominators differ')
    support = support_audit(model, reader, previous, selected, original_cfg['runtime_shared_parameter_stage_v2'])
    prepared.mkdir(parents=True, exist_ok=False)
    continuation = dict(schema='rinkfit.representative_complete_primal_continuation.v1',
        source_packet_sha256=producer['packet']['sha256'], complete_checkpoint=True,
        model=state['model'], checkpoint=state['checkpoint'], state_token=state['state_token'],
        representative_source_selection_v1=original_cfg['representative_source_selection_v1'],
        explicit_nuisance_manifest=state['explicit_nuisance_manifest'], source_state_binding=state['source_state_binding'],
        quality_accepted=False)
    continuation_ref = write_new(prepared / 'continuation.json', continuation)
    cfg = configuration(original_cfg, producer, state, delta, continuation_ref, repair)
    if append_policy:
        cfg.update(source_changes)
        cfg['representative_source_append_primal_v1']=append_policy
        from .representative_source_resume import rebind_current_packet_policy
        rebind_current_packet_policy(cfg, checked(producer['packet']), checked(target_packet))
        if 'source_unsupported_ice_initializers_v1' in cfg:
            cfg['source_unsupported_ice_initializers_v1']['source_packet_sha256']=target_packet['sha256']
    cfg['accepted_source_anchor_review'] = prepare_anchor_review(
        original_cfg['accepted_source_anchor_review'], state['model'], checked(target_packet), prepared / 'anchor_review')
    require(read(checked(read(checked(cfg['accepted_source_anchor_review']))['baseline']))['observed_row_clock'] == 'source_uv',
            'New review baseline must use original sourceUVy for every shot')
    support_ref = write_new(prepared / 'support_audit.json', support)
    loss = dict(schema='rinkfit.training_loss_budget_transition.v1',
        producer_configuration=producer['configuration'], input_delta=reference(input_delta),
        previous_selection=original_cfg['representative_source_selection_v1'],
        current_selection=delta['representative_source_selection_v1'],
        previous_budget=original_cfg['engineering_support_budget_v2'], current_budget=cfg['engineering_support_budget_v2'],
        source_packet_unchanged=producer['packet'] if append_policy is None else False,
        previous_source_packet=producer['packet'],current_source_packet=target_packet,
        authenticated_source_append=append_policy, same_objective_continuation=False,
        raw_auxiliary_state_reseeded=False, original_W_or_source_sigma_changed=False,
        protected_pixels_history_erased=False, independent_validation_certified=False,
        numerical_repair_delta=reference(numerical_repair_delta) if numerical_repair_delta else None)
    loss_ref = write_new(prepared / 'loss_transition.json', loss)
    cfg_ref = write_new(prepared / 'config.json', cfg)
    template = deepcopy(producer)
    template.update(packet=target_packet, configuration=cfg_ref, output=str(output), native=None)
    template_ref = write_new(prepared / 'spec.template.json', template)
    changed = sorted(key for key in set(semantic_config(original_cfg)) | set(semantic_config(cfg))
                     if semantic_config(original_cfg).get(key) != semantic_config(cfg).get(key))
    report = dict(schema='rinkfit.complete_primal_training_transition_preparation.v1',
        producer_spec=reference(producer_spec), candidate=str(Path(candidate).resolve()),
        source_packet=target_packet, complete_import_state=state,
        authenticated_source_append=append_policy, source_append_delta=reference(source_append_delta) if source_append_delta else None,
        continuation=continuation_ref, configuration=cfg_ref, spec_template=template_ref,
        support_audit=support_ref, loss_transition=loss_ref, input_verification=reference(verify_path),
        semantic_configuration_keys_changed=changed, final_native_binding_required=True,
        native_complete_state_roundtrip_required=True, optimizer_history_restored=False,
        solver_launched=False, quality_accepted=False)
    write_new(prepared / 'preparation.json', report)
    return report



def prepare_variant(*, source_prepared, prepared, shape_delta, gpu_delta, effective_manifest):
    """Extend an unlaunched preparation with independently SHA-bound migrations.

    Arrays and the complete original primal are referenced in place. The shape
    topology is applied by native only after complete old-state restoration.
    """
    source_prepared=Path(source_prepared).resolve();prepared=Path(prepared).resolve()
    require(not prepared.exists(),'Variant preparation already exists')
    previous=read(source_prepared/'preparation.json');template=read(checked(previous['spec_template']))
    require(template.get('native') is None and previous['solver_launched'] is False,'Variant needs an unlaunched preparation')
    cfg=read(checked(previous['configuration']));state=previous['complete_import_state']
    shape_changes=read(shape_delta)
    allowed={'shared_print_shape_transition_v1','do_not_reapply_old_template_restorers_after_migration','quality_accepted',
             'runtime_shared_parameter_stage_v2','audit_expected_free_physical_blocks','temporary_runtime_fixed_physical_blocks'}
    require(set(shape_changes)==allowed and shape_changes['quality_accepted'] is False and
            shape_changes['do_not_reapply_old_template_restorers_after_migration'] is True,'Unreviewed shared-shape delta keys')
    shape=read(checked(shape_changes['shared_print_shape_transition_v1']))
    require(shape['schema']=='rinkfit.shared_print_shape_transition.v1' and same_ref(shape['source_model'],state['model']) and
            same_ref(shape['source_checkpoint'],state['checkpoint']) and shape['source_state_token']==state['state_token'] and
            same_ref(shape['source_packet'],template['packet']) and same_ref(shape['representative_selection'],cfg['representative_source_selection_v1']) and
            shape['original_block_values_unchanged'] is True and shape['source_rows_or_membership_changed'] is False and
            shape['original_model_complete_primal_restore_required'] is True and shape['quality_accepted'] is False,
            'Shape transition lacks exact old complete-primal/current source authority')
    for key in ('mapping','conditional_spectrum','initial_model','request'):checked(shape[key])
    stage=shape_changes['runtime_shared_parameter_stage_v2']
    require(stage['schema']=='rinkfit.explicit_shared_parameter_stage.v2' and stage['source_packet_sha256']==template['packet']['sha256'] and
            stage['world_values_measured'] is False and isinstance(shape_changes['audit_expected_free_physical_blocks'],list) and
            len({x['block_index'] for x in shape_changes['audit_expected_free_physical_blocks']})==len(shape_changes['audit_expected_free_physical_blocks']) and
            {x['block_index'] for x in stage['blocks'] if x.get('free_ambient_indices')} <= {x['block_index'] for x in shape_changes['audit_expected_free_physical_blocks']},
            'Shape stage differs from actual shared-layout migration')
    execution=read(gpu_delta)
    require(set(execution)=={'gpu_raw_uv','observed_source_allowed_path_restart_v1'},'Unreviewed GPU/branch execution delta')
    gpu=execution['gpu_raw_uv']
    require(gpu['kernel_abi']=='shared_print_vertices_v2' and gpu['batch_reuse'] is True,'Shared vertex GPU ABI is required')
    checked(dict(file=gpu['ptx'],sha256=gpu['ptx_sha256']))
    shadow=read(checked(gpu['shadow_receipt']));batch=read(checked(gpu['batch_reuse_qualification']))
    require(shadow['passed'] is True and shadow['kernel_abi']==gpu['kernel_abi'] and shadow['ptx_sha256']==gpu['ptx_sha256'] and
            batch['passed'] is True,'GPU actual shared-vertex qualification changed')
    require(execution['observed_source_allowed_path_restart_v1']==dict(schema='rinkfit.observed_source_allowed_path_restart.v1',
            acceptance='same_complete_objective_and_nonworsened_physical_feasibility'),'Observed branch restart changed acceptance mathematics')
    effective=read(effective_manifest)
    require(effective['schema']=='rinkfit.effective_joint_input.v1' and same_ref(effective['source_packet'],template['packet']) and
            same_ref(effective['source_model'],template['model']) and
            same_ref(effective['qualification']['original_qualification'],cfg['dense_source_qualification']) and
            same_ref(effective['qualification']['selection'],cfg['representative_source_selection_v1']) and
            effective['engineering_support_budget_v2']==cfg['engineering_support_budget_v2'],'Effective ABI does not bind this actual prepared source objective')
    cfg.update(deepcopy(shape_changes));cfg.update(deepcopy(execution))
    cfg.pop('dense_source_qualification');cfg.pop('engineering_dense_source_weights_v1')
    cfg['effective_input_v1']=reference(effective_manifest)
    cfg['write_accepted_physical_preview']=False
    cfg['accepted_checkpoint_min_seconds_v1']=120.
    require(cfg['representative_objective_transition_v1']['from_state_token']==state['state_token'] and
            cfg['representative_source_append_primal_v1']['from_state_token']==state['state_token'],
            'Variant lost the complete original state import')
    prepared.mkdir(parents=True,exist_ok=False)
    cfg['accepted_source_anchor_review']=prepare_anchor_review(cfg['accepted_source_anchor_review'],state['model'],checked(template['packet']),prepared/'anchor_review')
    cfg_ref=write_new(prepared/'config.json',cfg);template['configuration']=cfg_ref
    template_ref=write_new(prepared/'spec.template.json',template)
    loss=write_new(prepared/'loss_transition.json',dict(schema='rinkfit.shared_shape_effective_objective_transition.v1',
        previous_preparation=reference(source_prepared/'preparation.json'),source_packet=template['packet'],
        from_complete_state=state,shape_delta=reference(shape_delta),gpu_delta=reference(gpu_delta),effective_input=reference(effective_manifest),
        original_arrays_or_W_changed=False,old_complete_primal_restored_before_shape_layout=True,
        new_shape_parameters_explicitly_initialized=True,same_objective_resume=False,quality_accepted=False))
    result=deepcopy(previous);result.update(configuration=cfg_ref,spec_template=template_ref,loss_transition=loss,
        source_preparation=reference(source_prepared/'preparation.json'),shape_delta=reference(shape_delta),gpu_delta=reference(gpu_delta),
        effective_input=reference(effective_manifest),final_native_binding_required=True,solver_launched=False,quality_accepted=False,
        new_shape_layout_native_roundtrip_required=True,write_accepted_physical_preview=False)
    write_new(prepared/'preparation.json',result)
    return result


def prepare_graph_variant(*, source_spec, prepared, output):
    """Prepare a complete graph evaluation with the native Solve branch disabled."""
    source=read(source_spec);cfg=read(checked(source['configuration']));prepared=Path(prepared).resolve()
    require(not prepared.exists(),'Graph preparation already exists')
    require(cfg.get('effective_input_v1') and cfg.get('representative_continuation_v1'),'Full graph preflight needs the effective input and complete primal import')
    cfg['source_uv_capture_preflight_only']=True;cfg['source_uv_capture_construct_problem_only']=False
    cfg['write_accepted_physical_preview']=False
    prepared.mkdir(parents=True,exist_ok=False)
    cfg_ref=write_new(prepared/'config.json',cfg);template=deepcopy(source)
    template.update(configuration=cfg_ref,output=str(Path(output).resolve()),native=None)
    template_ref=write_new(prepared/'spec.template.json',template)
    report=dict(schema='rinkfit.effective_complete_graph_preflight_preparation.v1',source_spec=reference(source_spec),
        configuration=cfg_ref,spec_template=template_ref,solver_launched=False,quality_accepted=False,
        native_solve_branch_enabled=False,final_native_binding_required=True)
    write_new(prepared/'preparation.json',report)
    return report

def finalize(prepared, native, preparation_name="preparation.json"):
    """Bind the new build and run every ordinary preflight, without Solve."""
    from .dense_joint_run import prepare_run
    prepared = Path(prepared).resolve()
    preparation = read(prepared / preparation_name)
    spec = read(checked(preparation['spec_template']))
    checked(preparation['configuration'])
    require(spec['configuration'] == preparation['configuration'], 'Prepared configuration binding changed')
    require(spec['native'] is None, 'Unexpected bound native in pending template')
    spec['native'] = reference(native)
    result = prepare_run(spec)
    spec_ref = write_new(prepared / 'spec.json', spec)
    result['spec'] = spec_ref
    result['solver_launched'] = False
    write_new(prepared / 'finalization.json', result)
    return result



def bind_final_native(prepared,native):
    """Keep an earlier unlaunched spec and bind a new implementation separately."""
    from .dense_joint_run import prepare_run
    prepared=Path(prepared).resolve();preparation=read(prepared/'preparation.json')
    original=reference(prepared/'spec.json');previous=read(checked(original))
    template=read(checked(preparation['spec_template']));checked(preparation['configuration'])
    require(previous['configuration']==template['configuration']==preparation['configuration'] and
            template['native'] is None,'Native rebinding may not alter the prepared configuration')
    template['native']=reference(native);result=prepare_run(template)
    final_ref=write_new(prepared/'spec_final.json',template)
    result.update(spec=final_ref,previous_unlaunched_spec=original,configuration_bytes_unchanged=True,solver_launched=False)
    write_new(prepared/'finalization_final.json',result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for key in ('producer-spec', 'candidate', 'input-delta', 'prepared', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--numerical-repair-delta')
    p.add_argument('--source-append-delta')
    p = sub.add_parser('finalize'); p.add_argument('--prepared', required=True); p.add_argument('--native', required=True); p.add_argument('--preparation-name',default='preparation.json')
    args = vars(parser.parse_args()); command = args.pop('command')
    values = {key.replace('-', '_'): value for key, value in args.items()}
    print(json.dumps(prepare(**values) if command == 'prepare' else finalize(**values), indent=2))


if __name__ == '__main__':
    main()
