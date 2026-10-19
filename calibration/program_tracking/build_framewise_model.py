"""Bind accepted source observations to a new model seeded from their 8a28 state.

Only the accepted packet is read. No next frame is decoded, historical fit
observations imported, or optimizer invoked. Original source UV stay immutable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'events/ankara_2026/models/framewise_8a28_20261013'
DATASET = ROOT / ('events/ankara_2026/programmes/okada_mei_sp/source_datasets/'
                  'frame_by_frame_review_20261013')
DIAGNOSTICS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/'
                   'okada_mei_sp/results/frame_by_frame_review_20261013/model_binding')
MODEL_ID = 'ankara_2026.framewise_8a28_20261013'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def point_key(feature, segment_index, vertex_index):
    return f'{feature}/segment/{segment_index}/vertex/{vertex_index}'


def make_registry(scene):
    """Stable physical identities, independent of visibility or source density."""
    from .render_result import advertisement_boundary_paths
    actual_ad_edges={(p['number'],p['role'].removesuffix('_xyz'))
                     for p in advertisement_boundary_paths(scene)}
    targets = {}
    for part in [None] + [p['id'] for p in scene['ice_boundary']['parts_counterclockwise']]:
        for role in ('top', 'contact'):
            name = f'board.{role}' + ('.'+part if part else '')
            targets[name] = dict(kind='board_edge', role=role, part=part,
                                 residual='point_to_named_finite_curve')
    for ad in scene['advertisements']:
        number = int(ad['user_number'])
        for role in ad.get('corners_xyz_scene_units',{}):
            targets[f'Ad{number}.{role}'] = dict(kind='ad_corner',
                advertisement_id=number, role=role, residual='point_to_corresponding_corner_2d')
        for role in ('top_edge', 'bottom_edge', 'left_vertical', 'right_vertical'):
            if (number,role) not in actual_ad_edges: continue
            targets[f'Ad{number}.{role}'] = dict(kind='ad_edge',
                advertisement_id=number, role=role, residual='point_to_named_finite_curve')
    for key in ('ISU_S', 'U', 'hyphen', 'SKATING_S', 'K'):
        targets['Ad2.print.'+key] = dict(kind='ad_print', advertisement_id=2,
            contour_key=key, residual='point_to_named_finite_print_contour',
            material_coordinate_domain='ad_rectangle_u_left_to_right_v_top_to_bottom',
            template='ad2_material_template.json',
            geometry_status='independent_source_template_initialization_not_accuracy_accepted')
    from rinkspace.vision_programme_ice_review import ice_curve_inventory
    for curve in ice_curve_inventory(scene):
        targets['ice.'+curve['curve_id']]=dict(kind='ice_curve',curve_id=curve['curve_id'],
            residual='point_to_named_finite_ice_curve')
    return dict(schema='rinkspace.framewise-feature-registry.v1', model_id=MODEL_ID,
        targets=targets,
        inherited_ice_features=[dict(id=f['id'], kind=f['kind'],
            scene_pointer='/ice_features/'+str(i)) for i, f in enumerate(scene['ice_features'])],
        no_generic_nearest_advertisement_fallback=True)


def make_binding(packet, packet_path, registry):
    grouped = defaultdict(list)
    glyphs = {'hyphen':'hyphen', 'U':'U', 'S':'SKATING_S',
              'left_S_visible_fragment':'ISU_S', 'right_cropped_letter':'K'}
    for feature in packet['features']:
        fid = feature['id']
        for si, segment in enumerate(feature['segments']):
            for vi, uv in enumerate(segment['uv']):
                if fid == 'board_top': target = 'board.top.upper_long_side'
                elif fid == 'board_contact': target = 'board.contact.upper_long_side'
                elif '.print.' in fid: target = 'Ad2.print.'+glyphs[feature['label']]
                else:
                    target = segment['point_bindings'][vi]['physical_target']
                    target = {'Ad2.top_left':'Ad2.left_top',
                              'Ad2.bottom_left':'Ad2.left_bottom'}.get(target, target)
                if target not in registry['targets']: raise ValueError('Unregistered target '+target)
                grouped[(fid, si, target)].append(vi)
    observations = []
    for (fid, si, target), indices in grouped.items():
        observations.append(dict(id=f'{fid}/segment/{si}/{target}',
            source_feature_id=fid, source_segment_index=si, source_vertex_indices=indices,
            source_point_ids=[point_key(fid, si, i) for i in indices],
            target_id=target,
            point_correspondence=registry['targets'][target]['kind']=='ad_corner',
            observation_uncertainty='retained_in_accepted_source_packet',
            dependence_group=('board_top_shared_line' if fid=='board_top' else target),
            source_coordinates_copied_or_changed=False))
    ids = [i for row in observations for i in row['source_point_ids']]
    if len(ids)!=packet['point_count'] or len(set(ids))!=len(ids):
        raise ValueError('Every accepted point must have exactly one model target')
    return dict(schema='rinkspace.framewise-source-model-binding.v1', model_id=MODEL_ID,
        source_packet=reference(packet_path), source_pts=packet['source']['native_pts'],
        source_time_base=packet['source']['time_base'], frame_count=1,
        source_user_accepted=packet['user_accepted'], point_count=len(ids), observations=observations,
        status='all_points_have_typed_model_entries',
        new_model_accuracy_accepted=False, optimization_performed=False,
        formal_fit_weighting_status='not_selected_by_model_entry_diagnostic',
        shared_line_weighting_requirement='board_top_samples_share_one_source_line_estimate',
        next_frame_processed=False)


def build(*, model_dir=MODEL, dataset_dir=DATASET, diagnostics_dir=DIAGNOSTICS):
    from .framewise_model import build_row, feature_world, evaluate_rows

    packet_path = dataset_dir/'frame_000001_points.json'
    original_packet_sha = sha(packet_path)
    packet = json.loads(packet_path.read_text(encoding='utf8'))
    if not packet['user_accepted'] or packet['frame_count']!=1 or packet['next_frame_processed']:
        raise ValueError('Only the explicitly accepted first-frame packet is admitted')
    source = packet['source']
    scene_path, camera_path = Path(source['scene_path']), Path(source['camera_path'])
    if sha(scene_path)!=source['scene_sha256'] or sha(camera_path)!=source['camera_sha256']:
        raise ValueError('Seed scene/camera no longer match the accepted 8a28 provenance')
    identity_path = scene_path.with_name('state_identity.json')
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    if identity['q_sha256']!=source['model_q_sha256']: raise ValueError('Seed Q mismatch')
    state_path = scene_path.with_name('complete_state_UNACCEPTED.npz')
    if sha(state_path)!=identity['state_npz_sha256']: raise ValueError('Complete seed state mismatch')
    run_dir = scene_path.parents[2]
    input_path = run_dir/'input_manifest.json'
    if sha(input_path)!=identity['input_manifest_sha256']: raise ValueError('8a28 input identity mismatch')
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    base_geometry = deepcopy({k:scene[k] for k in ('ice_boundary','board_surface','ice_features','advertisements')})
    scene['model_id'] = MODEL_ID
    scene['status'] = '8a28_geometry_seed_for_new_framewise_dataset_not_refitted'
    scene['framewise_model'] = dict(parent_q_sha256=source['model_q_sha256'],
        feature_registry='feature_registry.json', material_template='ad2_material_template.json',
        dataset_model_binding=str(dataset_dir/'frame_000001_model_binding.json'),
        additional_material_world_producer='calibration.program_tracking.framewise_model.feature_world',
        geometry_values_inherited_unchanged=True, accuracy_accepted=False)
    with np.load(state_path, allow_pickle=False) as state:
        q = state['q']; indices=state['aux__current_shared_world_geometry_parameter_indices']
        names=state['aux__current_named66_parameter_names']
        if hashlib.sha256(q.tobytes()).hexdigest()!=source['model_q_sha256']:
            raise ValueError('Complete Q array identity mismatch')
        parameters=[dict(name=str(name), complete_q_index=int(index), seed_value=float(q[index]))
                    for index, name in zip(indices,names)]
        if len(parameters)!=100: raise ValueError('Unexpected inherited geometry chart')
    with np.load(camera_path, allow_pickle=False) as cameras:
        matches=np.flatnonzero(cameras['pts']==source['native_pts'])
        if len(matches)!=1: raise ValueError('Exactly one matching native camera is required')
        index=int(matches[0]); camera=cameras['camera25'][index].copy()
        installation=str(cameras['installation_id'][index])
        metadata=json.loads(str(cameras['metadata_json'].item()))
        camera_seed=dict(schema='rinkspace.framewise-camera-seed.v1',
            native_pts=int(cameras['pts'][index]),
            exposure_representative_pts=int(cameras['exposure_representative_pts'][index]),
            source_time_base=source['time_base'], installation_id=installation,
            camera25=camera.tolist(), camera25_float64_sha256=hashlib.sha256(camera.tobytes()).hexdigest(),
            view_state4=cameras['view_state4'][index].tolist(),
            effective_pp_offset_px=cameras['effective_pp_offset_px'][index].tolist(),
            shared_calibration=metadata['calibrations'][installation],
            inherited_camera_archive=reference(camera_path), source_q_sha256=source['model_q_sha256'])
    templates_path=model_dir/'ad2_material_template.json'
    templates=json.loads(templates_path.read_text(encoding='utf8'))
    registry=make_registry(scene)
    for descriptor in registry['targets'].values():
        world=feature_world(scene,descriptor,templates)
        if not np.isfinite(world).all(): raise ValueError('Registered target is not finite')
    binding=make_binding(packet,packet_path,registry)
    features={f['id']:f for f in packet['features']}
    rows=[]; summaries=[]
    for observation in binding['observations']:
        segment=features[observation['source_feature_id']]['segments'][observation['source_segment_index']]
        uv=np.asarray(segment['uv'],float)[observation['source_vertex_indices']]
        descriptor=registry['targets'][observation['target_id']]
        row=build_row(scene,descriptor,uv,np.ones(len(uv)),templates,
                      diagnostic_unit_weights=True)
        row['id']=observation['id']
        if bool(row['point_correspondence'])!=observation['point_correspondence']:
            raise ValueError('Corner/curve route changed')
        rows.append(row)
        summaries.append(dict(target_id=observation['target_id'],source_points=len(uv),
            world_vertices=len(feature_world(scene,descriptor,templates)),
            residual_kind=descriptor['residual']))
    residual,jacobian=evaluate_rows(camera,rows,jacobian=True)
    if residual.shape!=(2*packet['point_count'],) or jacobian.shape!=(len(residual),15):
        raise ValueError('Every source point must reach the actual residual/Jacobian interface')
    if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
        raise ValueError('Nonfinite model entry diagnostic')
    offset=0
    for row, summary in zip(rows,summaries):
        length=2*len(row['uv']); error=np.linalg.norm(residual[offset:offset+length].reshape(-1,2),axis=1)
        summary.update(initial_raw_pixel_error_median=float(np.median(error)),
                       initial_raw_pixel_error_max=float(np.max(error)))
        offset+=length
    if base_geometry!={k:scene[k] for k in base_geometry}: raise ValueError('Inherited geometry changed')
    if sha(packet_path)!=original_packet_sha: raise ValueError('Accepted source packet was modified')
    model_dir.mkdir(parents=True,exist_ok=True)
    write(model_dir/'scene.json',scene)
    write(model_dir/'feature_registry.json',registry)
    write(model_dir/'seed_parameters.json',dict(schema='rinkspace.framewise-seed-parameters.v1',
        complete_state=reference(state_path), source_q_sha256=source['model_q_sha256'],
        complete_q_parameter_count=len(q), inherited_geometry_parameters=parameters,
        names_with_m_suffix_are_legacy_names_not_verified_metric_units=True))
    write(dataset_dir/'frame_000001_camera_seed.json',camera_seed)
    write(dataset_dir/'frame_000001_model_binding.json',binding)
    manifest=dict(schema='rinkspace.framewise-reconstruction-model.v1',model_id=MODEL_ID,
        event_id='ankara_2026', created_date_local='2026-10-13',
        parent=dict(q_sha256=source['model_q_sha256'],scene=reference(scene_path),
                    complete_state=reference(state_path),input_manifest=reference(input_path),
                    state_identity=reference(identity_path)),
        scene=reference(model_dir/'scene.json'), feature_registry=reference(model_dir/'feature_registry.json'),
        material_templates=reference(templates_path), seed_parameters=reference(model_dir/'seed_parameters.json'),
        programme_binding=reference(dataset_dir/'frame_000001_model_binding.json'),
        camera_seed=reference(dataset_dir/'frame_000001_camera_seed.json'),
        implementation='calibration.program_tracking.framewise_model',
        residual_entry='z190_source_camera_diagnostic.absolute_value_jac15',
        strict_v6_used=False, historical_observation_rows_imported=0,
        source_point_count=packet['point_count'],accepted_source_frame_count=1,
        model_accuracy_accepted=False, optimization_performed=False,next_frame_processed=False,
        coordinate_system=scene['coordinate_system'])
    write(model_dir/'model.json',manifest)
    diagnostic=dict(schema='rinkspace.framewise-model-entry-validation.v1',
        model=reference(model_dir/'model.json'), source_packet=reference(packet_path),
        point_count=packet['point_count'],source_points_covered_once=True,
        residual_shape=list(residual.shape),jacobian_shape=list(jacobian.shape),
        finite_residual_and_jacobian=True, diagnostic_weighting='explicit_unit_sigma_raw_pixel_errors_only',
        uncertainty_and_group_covariance_not_replaced_by_diagnostic_unit_weights=True,
        inherited_geometry_unchanged=True, accepted_source_packet_unchanged=True,
        original_seed_files_unchanged=True, strict_v6_used=False,
        old_observation_rows_imported=0, optimization_performed=False,
        next_frame_processed=False,model_accuracy_accepted=False,groups=summaries)
    write(diagnostics_dir/'model_binding_validation.json',diagnostic)
    return diagnostic


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    result=build()
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},ensure_ascii=False))
