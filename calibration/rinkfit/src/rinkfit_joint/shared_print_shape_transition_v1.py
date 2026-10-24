"""SHA-bound static shared-print vertex layout transition; no source rows added.

The old inverse-camera contours are numerical initializers. This changes the
physical parameterization and objective, never certifies those initializers as
physical outlines or removes their unobserved directions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .shared_print_shape_mapping_v1 import read_ref, file_ref, normalized_points, shape_inventory


def _write(path, value):
    path = Path(path)
    if len(str(path.resolve()).encode('utf-16-le')) // 2 >= 260 or any(len(x.encode('utf-16-le')) // 2 > 255 for x in path.parts):
        raise ValueError('Windows path limit')
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return file_ref(path)


def produce(model_file, mapping_file, spectrum_file, request_file, checkpoint_file, output, config_file=None):
    output = Path(output)
    if len(str(output.resolve()).encode('utf-16-le')) // 2 >= 240:
        raise ValueError('output path too long')
    output.mkdir(parents=True, exist_ok=True)
    model_ref, mapping_ref, spectrum_ref, request_ref, checkpoint_ref = map(file_ref,
        (model_file, mapping_file, spectrum_file, request_file, checkpoint_file))
    original = json.loads(Path(model_file).read_text(encoding='utf-8'))
    mapping = json.loads(Path(mapping_file).read_text(encoding='utf-8'))
    spectrum = json.loads(Path(spectrum_file).read_text(encoding='utf-8'))
    request = json.loads(Path(request_file).read_text(encoding='utf-8'))
    selection, selection_ref = read_ref(request['manifest'])
    if mapping['model']['sha256'] != model_ref['sha256'] or spectrum['dependencies']['model']['sha256'] != model_ref['sha256']:
        raise ValueError('mapping/spectrum do not describe this exact accepted model')
    source_packet = selection['source_packet']
    if request['raw_rows'] != selection['source_row_count'] or spectrum['actual_fit_rows'] != selection['fit_count']:
        raise ValueError('spectrum/selection current target mismatch')
    # Parse the complete native checkpoint header only for its explicit token;
    # native runtime separately restores every old block and all raw auxiliaries.
    with Path(checkpoint_file).open('r', encoding='utf-8') as f:
        first = [f.readline().strip() for _ in range(7)]
    if first[0] != 'RINKFIT_AL_CHECKPOINT_V1' or first[5] != '1':
        raise ValueError('unknown complete checkpoint encoding')
    # Token is archived with the source model handoff, and verified natively
    # against the already-restored complete state, not inferred from this title.
    model = copy.deepcopy(original)
    old_count = len(model['blocks'])
    inventory = shape_inventory(original)
    if set(inventory) != {(r['material_index'], r['path_index']) for r in mapping['shape_records']}:
        raise ValueError('mapping omitted actual candidate print paths')
    groups, changes, supported_vertices, unsupported_vertices = [], [], 0, 0
    changed_paths = set()
    vertex_indices, envelope_indices = [], []
    for gi, group in enumerate(mapping['shared_shapes']):
        members = [tuple(x) for x in group['members']]
        canonical = tuple(group['canonical_path'])
        cp = original['materials'][canonical[0]]['paths'][canonical[1]]
        q = normalized_points(original, cp)
        source_points = np.asarray(cp['points'], dtype='<f8')
        closed = bool(cp.get('closed', False)) and np.array_equal(source_points[0], source_points[-1])
        if closed:
            q = q[:-1]
        if len(q) < 2:
            raise ValueError('degenerate contour requires its original point representation, not invented split edges')
        active = bool(group['fit_rows'])
        if not active:
            # Preserve an unsupported display initializer, while preventing its
            # inverse-camera numeric shape from constraining physical scene/ad.
            for mi, pi in members:
                changed_paths.add((mi,pi))
                before=original['materials'][mi]['paths'][pi]
                after=copy.deepcopy(before)
                after['display_only_initializer']=True
                after['display_only_initializer_provenance']=dict(mapping=mapping_ref,
                    shared_group_index=gi, actual_training_rows=0,
                    inverse_projection_seed_is_not_physical_truth=True,
                    excluded_from_physical_shape_domains=True,quality_accepted=False)
                model['materials'][mi]['paths'][pi]=after
                changes.append(dict(material_index=mi,path_index=pi,
                    material_label=original['materials'][mi]['label'],before=before,after=after,
                    group_index=gi,display_only_initializer=True,
                    initial_world_curve_changed=False))
            continue
        vertex_blocks = []
        for vi, uv in enumerate(q):
            bi = len(model['blocks'])
            vertex_blocks.append(bi)
            vertex_indices.append(bi)
            model['blocks'].append(dict(name=f'print_vertex/g{gi}/v{vi}', values=uv.tolist(),
                lower=[-float(np.finfo(float).max)]*2, upper=[float(np.finfo(float).max)]*2,
                scales=[0.02, 0.02], constant=not active, fixed_indices=[]))
        envelope = len(model['blocks'])
        envelope_indices.append(envelope)
        model['blocks'].append(dict(name=f'print_envelope/g{gi}', values=[float(q[:,0].min()),float(q[:,0].max())],
            lower=[-float(np.finfo(float).max)]*2, upper=[float(np.finfo(float).max)]*2,
            scales=[0.02,0.02], constant=not active, fixed_indices=[]))
        supported_vertices += len(q)*int(active)
        unsupported_vertices += len(q)*int(not active)
        record = dict(group_index=gi, canonical_path=list(canonical), members=group['members'],
            vertex_blocks=vertex_blocks, course_envelope_block=envelope,
            closed=closed, original_runtime_segments=len(source_points)-1,
            fit_rows=group['fit_rows'], fit_shots=group['fit_shots'], fit_frames=group['fit_frames'],
            supported_by_training_rows=active,
            shape_authority='unmeasured_static_numerical_initialization',
            course_envelope_is_physical_measurement=False,
            numerical_nullspace_fixed_as_truth=False)
        groups.append(record)
        for mi, pi in members:
            key = (mi,pi)
            if key in changed_paths:
                raise ValueError('duplicate path in shared mapping')
            changed_paths.add(key)
            before = original['materials'][mi]['paths'][pi]
            actual = normalized_points(original,before)
            sp = np.asarray(before['points'], dtype='<f8')
            member_closed = bool(before.get('closed',False)) and np.array_equal(sp[0],sp[-1])
            if member_closed != closed or not np.array_equal(actual[:-1] if closed else actual,q):
                raise ValueError('same template migration would change this installation initial world shape')
            after = copy.deepcopy(before)
            for k in ('points','geometry_offset','geometry_indices'):
                after.pop(k,None)
            after.pop('display_only_initializer',None)
            after.update(kind=6, geometry_block=-1, vertex_blocks=vertex_blocks,
                course_envelope_block=envelope, closed=closed)
            closure = inventory[key]['closure_evidence']
            after['provenance'] = json.dumps(dict(schema='rinkfit.shared_print_static_vertex_path.v1',
                original_model=model_ref, original_material_index=mi,original_path_index=pi,
                group_index=gi, original_provenance=before.get('provenance',''),
                closure_evidence=closure, source_closed_flag=bool(before.get('closed',False)),
                runtime_segment_count_preserved=len(sp)-1,
                numerical_initializer_not_physical_truth=True, per_frame_shape_parameters=False),
                ensure_ascii=False,separators=(',',':'))
            model['materials'][mi]['paths'][pi] = after
            changes.append(dict(material_index=mi,path_index=pi,material_label=original['materials'][mi]['label'],
                before=before,after=after,group_index=gi,
                initial_normalized_vertices_exactly_preserved=True,
                initial_world_curve_changed=False, closure_evidence=closure))
    if changed_paths != set(inventory):
        raise ValueError('transition not complete for mapped contour scope')
    # Resolve consumers using the exact accepted raw path column, never the
    # source numerical seed's path or a residual-based nearest association.
    aux,_=read_ref(mapping['accepted_aux'])
    aux_dir=Path(mapping['accepted_aux']['file']).parent
    path_file=aux_dir/'path.u32'
    if file_ref(path_file)['sha256']!=aux['files']['path.u32']['sha256']:
        raise ValueError('accepted complete path column differs')
    accepted_path=np.memmap(path_file,dtype='<u4',mode='r')
    for key in ('material_index','path'):
        if file_ref(request['arrays'][key]['file'])['sha256']!=request['arrays'][key]['sha256']:
            raise ValueError('actual source '+key+' SHA mismatch')
    raw_material=np.memmap(request['arrays']['material_index']['file'],dtype='<u4',mode='r')
    raw_path=np.memmap(request['arrays']['path']['file'],dtype='<u4',mode='r')
    fit_desc=selection['fit_rows']
    if file_ref(fit_desc['file'])['sha256']!=fit_desc['sha256']:
        raise ValueError('actual selected row SHA mismatch')
    actual_fit=np.fromfile(fit_desc['file'],dtype='<u8')
    current_paths=np.asarray(raw_path[actual_fit]).copy();prefix=actual_fit<len(accepted_path)
    current_paths[prefix]=accepted_path[actual_fit[prefix]]
    active_paths={tuple(map(int,x)) for x in np.c_[raw_material[actual_fit],current_paths]}
    old_affine={r['geometry_block'] for r in mapping['shape_records'] if r['geometry_block']>=0}
    for mi,mat in enumerate(model['materials']):
        for pi,path in enumerate(mat['paths']):
            bi=path.get('geometry_block',-1)
            if bi not in old_affine or (mi,pi) in changed_paths or (mi,pi) in active_paths:
                continue
            if path['kind']!=2:
                raise ValueError('unrecorded non-print consumer of old affine')
            before=copy.deepcopy(original['materials'][mi]['paths'][pi]);after=copy.deepcopy(path)
            after['display_only_initializer']=True
            after['display_only_initializer_provenance']=dict(mapping=mapping_ref,
                actual_training_rows=0,legacy_affine_retained_for_display_only=True,
                inverse_projection_seed_is_not_physical_truth=True,
                excluded_from_physical_shape_domains=True,quality_accepted=False)
            model['materials'][mi]['paths'][pi]=after
            changes.append(dict(material_index=mi,path_index=pi,material_label=mat['label'],
                before=before,after=after,display_only_initializer=True,
                initial_world_curve_changed=False))
            changed_paths.add((mi,pi))
    active_geometry={model['materials'][mi]['paths'][pi].get('geometry_block',-1) for mi,pi in active_paths}
    physical_geometry={path.get('geometry_block',-1) for mat in model['materials'] for path in mat['paths'] if not path.get('display_only_initializer')}
    orphan_affine=sorted(old_affine-active_geometry-physical_geometry)
    unused_flag_changes=[]
    for bi in orphan_affine:
        before=bool(model['blocks'][bi].get('constant',False))
        model['blocks'][bi]['constant']=True
        unused_flag_changes.append(dict(block_index=bi,block_name=model['blocks'][bi]['name'],
            before_constant=before,after_constant=True,
            active_row_consumers=0,physical_shape_domain_consumers=0,
            old_values_preserved_bitwise=True,display_only_consumers_retained=True))
    model['shared_print_shape_initialization_v1'] = dict(schema='rinkfit.shared_print_shape_initialization.v1',
        source_model=model_ref,source_packet=source_packet,mapping=mapping_ref,spectrum=spectrum_ref,
        original_block_count=old_count, static_vertex_layout=True, independent_shape_truth_claimed=False,
        coupled_camera_scene_rank_verified=False, no_per_frame_shapes=True)
    initial_model_ref = _write(output/'initial_model.json',model)
    manifest = dict(schema='rinkfit.shared_print_shape_transition.v1',source_model=model_ref,
        source_checkpoint=checkpoint_ref,source_state_token=json.loads(first[6]),source_packet=source_packet,request=request_ref,
        representative_selection=selection_ref,mapping=mapping_ref,conditional_spectrum=spectrum_ref,
        initial_model=initial_model_ref,original_block_count=old_count,final_block_count=len(model['blocks']),
        original_block_values_unchanged=True,source_rows_or_membership_changed=False,
        original_unused_affine_constant_changes=unused_flag_changes,
        original_model_complete_primal_restore_required=True,
        groups=groups,path_changes=changes,vertex_block_indices=vertex_indices,
        course_envelope_block_indices=envelope_indices,orphaned_old_affine_block_indices=orphan_affine,
        no_old_used_affine_block_frozen=True,
        summary=dict(path_count=len(changes),shared_shapes=len(groups),
            fit_shared_shapes=sum(g['supported_by_training_rows'] for g in groups),
            migrated_static_vertex_paths=sum(not c.get('display_only_initializer') for c in changes),
            unverified_display_only_paths=sum(bool(c.get('display_only_initializer')) for c in changes),
            unverified_display_only_groups=sum(not bool(g['fit_rows']) for g in mapping['shared_shapes']),
            training_supported_vertices=supported_vertices,unobserved_display_vertices=unsupported_vertices,
            full_static_shape_scalar_coordinates=2*(supported_vertices+unsupported_vertices),
            maximum_vertex_ambient=2,maximum_pixel_shape_ambient=4,
            numerical_envelope_scalars=2*len(groups)),
        actual_global_coupled_observability_verified=False,
        conditional_spectrum_is_full_objective_rank_certificate=False,
        unknown_shape_directions_are_not_physical_truth=True,
        unsupported_display_shape_policy='retain_old_display_initializer_excluded_from_physical_shape_domains_without_quality_claim',
        native_state_policy='append_after_complete_old_primal_restore_then_full_new_layout_state_on_resume',
        quality_accepted=False)
    manifest_ref = _write(output/'manifest.json',manifest)
    vertex_index_set=set(vertex_indices)
    delta=dict(shared_print_shape_transition_v1=manifest_ref,
        do_not_reapply_old_template_restorers_after_migration=True,quality_accepted=False)
    if config_file:
        config_ref=file_ref(config_file)
        config=json.loads(Path(config_file).read_text(encoding='utf8'))
        stage=copy.deepcopy(config['runtime_shared_parameter_stage_v2'])
        stage['blocks']=[r for r in stage['blocks'] if r['block_index'] not in orphan_affine]
        stage['static_shared_print_vertex_stage_v1']=dict(schema='rinkfit.explicit_static_print_vertex_stage.v1',
            transition=manifest_ref,source_packet_sha256=source_packet['sha256'],
            scalar_coordinates_are_physical_measurements=False,
            no_pixel_connection_directions_remain_unverified=True)
        gauge=stage.get('print_only_coordinate_gauge')
        if gauge:
            migrated_ads={model['materials'][c['material_index']]['paths'][c['path_index']]['advertisement_block'] for c in changes if not c.get('display_only_initializer')}
            gauge['groups']=[r for r in gauge['groups'] if r['advertisement_block'] not in migrated_ads and r['print_block'] not in orphan_affine]
            if not gauge['groups']:stage.pop('print_only_coordinate_gauge')
        free=[r for r in config['audit_expected_free_physical_blocks'] if r['block_index'] not in orphan_affine]
        seen={r['block_index'] for r in free}
        for bi in vertex_indices+envelope_indices:
            if bi in seen:raise ValueError('new static shape block duplicates old freedom registry')
            seen.add(bi)
            free.append(dict(block_index=bi,block_name=model['blocks'][bi]['name'],ambient_size=2,
                expected_tangent_size=2,free_ambient_indices=[0,1],
                parameter_role='static_unmeasured_print_vertex' if bi in vertex_index_set else 'numerical_course_envelope'))
        fixed=set(config['temporary_runtime_fixed_physical_blocks'])
        fixed.update(model['blocks'][bi]['name'] for bi in orphan_affine)
        added_names={model['blocks'][bi]['name'] for bi in vertex_indices+envelope_indices}
        if fixed&added_names:raise ValueError('new static shape block is explicitly frozen by old stage')
        for prefix in config.get('temporary_runtime_fixed_physical_prefixes',[]):
            if any(name.startswith(prefix) for name in added_names):raise ValueError('old prefix freezes new static shape variables')
        delta.update(runtime_shared_parameter_stage_v2=stage,
            audit_expected_free_physical_blocks=free,
            temporary_runtime_fixed_physical_blocks=sorted(fixed))
        _write(output/'production_stage_update.json',dict(schema='rinkfit.static_vertex_production_stage_update.v1',
            original_config=config_ref,transition=manifest_ref,
            removed_old_affine_blocks=unused_flag_changes,
            added_static_vertex_blocks=len(vertex_indices),added_numeric_envelope_blocks=len(envelope_indices),
            old_free_blocks=len(config['audit_expected_free_physical_blocks']),new_free_blocks=len(free),
            old_fixed_blocks=len(config['temporary_runtime_fixed_physical_blocks']),new_fixed_blocks=len(fixed),
            original_source_selection_and_weights_unchanged=True,
            actual_Ceres_stage_must_verify_all_new_two_coordinate_blocks_variable=True))
    _write(output/'config_delta.json',delta)
    return dict(manifest=manifest_ref,**manifest['summary'])


def main():
    p=argparse.ArgumentParser()
    for key in ('model','mapping','spectrum','request','checkpoint','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--config')
    a=p.parse_args()
    print(json.dumps(produce(a.model,a.mapping,a.spectrum,a.request,a.checkpoint,a.output,a.config),ensure_ascii=False))
if __name__=='__main__': main()
