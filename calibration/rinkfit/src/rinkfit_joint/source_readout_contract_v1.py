"""Verify actual fixed-birth source readouts, retaining every original row."""
import hashlib
import numpy as np
from .io_utils import read_json, read_jsonl


def source_readout_rows(reference, lineage_reference, verify):
    generation = read_json(verify(reference))
    if generation['schema'] != 'rinkfit.source_current_fixed_birth_affine_observation_generation.v1':
        raise ValueError('Actual fixed-birth observation generation required')
    for key in ('input_contract', 'original_generation', 'original_source_qualification', 'reader', 'helper',
                'source_rows', 'readout_records', 'correction_patch'):
        verify(generation[key])
    contract = read_json(verify(generation['input_contract']))
    for key in ('reader', 'helper', 'source_generation', 'source_qualification', 'source_candidate_rows',
                'original_typed_source_rows', 'source_roles', 'source_configuration'):
        verify(contract[key])
    for item in contract['actor_ledgers']: verify(item)
    if (contract['reader'] != generation['reader'] or contract['helper'] != generation['helper']
        or contract['source_video_sha256'] != 'e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
        or contract['no_model_used'] is not True):
        raise ValueError('Readout current source, strategy, or helper input identity differs')
    lineage = read_json(verify(lineage_reference))
    if (lineage['new_manifest'] != reference or lineage['full_original_source_rows'] != 139
        or lineage['new_source_mean_rows'] != 135 or lineage['uncertain_old_mean_rows'] != 3):
        # Reference metadata may carry size_bytes without changing identity.
        if (lineage['new_manifest']['sha256'] != reference['sha256']
            or lineage['full_original_source_rows'] != 139 or lineage['new_source_mean_rows'] != 135
            or lineage['uncertain_old_mean_rows'] != 3):
            raise ValueError('Full actual readout lineage QA required')
    patch = read_json(verify(generation['correction_patch']))
    records = list(read_jsonl(verify(generation['readout_records'])))
    rows = list(read_jsonl(verify(generation['source_rows'])))
    old = {r['source_row_id']: r for r in read_jsonl(verify(contract['original_typed_source_rows']))}
    if len(rows) != 139 or len(records) != 139 or len(patch['rows']) != 139:
        raise ValueError('Readout must retain the complete track denominator')
    mean_hash = lambda v: hashlib.sha256(np.asarray(v, dtype='<f8').tobytes()).hexdigest()
    seen = set(); changed = 0; uncertain = 0
    parameters = contract['parameters']
    for row, rec, delta in zip(rows, records, patch['rows']):
        rid = row['source_row_id']; prior = old[rid]
        if (rid in seen or rec['source_row_id'] != rid or delta['source_row_id'] != rid
            or row['physical_track_id'] != prior['physical_track_id']
            or row['native_pts'] != prior['native_pts'] or row['shot_id'] != prior['shot_id']
            or row['current_rgb_sha256'] != prior['current_rgb_sha256']
            or rec['current_RGB_sha256'] != prior['current_rgb_sha256']
            or row['surface_instance_id'] != prior['surface_instance_id']
            or row['joint_material_id'] != prior['joint_material_id']
            or delta['original_source_uv_f64_sha256'] != mean_hash(prior['source_uv'])
            or row['original_source_uv_f64_sha256'] != mean_hash(prior['source_uv'])
            or row['original_source_uv'] != prior['source_uv']
            or rec['source_geometry_model_used'] is not False or rec['ice_or_global_H_used'] is not False
            or row['source_precision_or_covariance_issued'] is not False):
            raise ValueError('Readout changed original UV lineage, clock, RGB, or physical instance')
        seen.add(rid)
        if delta['source_uv_changed']:
            direct = rec['direct']; reverse = rec['reverse']
            A = np.asarray(direct['warp']); B = np.asarray(reverse['warp'])
            uv = np.asarray(rec['current_search_patch_origin_native']) + A @ np.array([16., 16., 1.])
            bias = B @ np.array([16., 16., 1.]) - 32.
            corners = np.array([[-16., -16.], [-16., 16.], [16., -16.], [16., 16.]])
            error = corners @ (B[:, :2] @ A[:, :2] - np.eye(2)).T + bias
            cycle = float(np.linalg.norm(error, axis=1).max()); fb = float(np.linalg.norm(bias))
            if (rec['status'] != 'new_current_source_UV_candidate'
                or min(direct['registered_NCC'], reverse['registered_NCC']) < parameters['NCC_min']
                or fb > parameters['centre_FB_max_px'] or cycle > parameters['all_four_template_corner_cycle_max_px']
                or abs(cycle - rec['affine_patch_cycle_max_px']) > 1e-12
                or uv.tolist() != delta['next_source_uv'] or uv.tolist() != row['current_uv']):
                raise ValueError('New current UV lacks actual forward/reverse affine centre and whole-patch evidence')
            changed += 1
        else:
            if mean_hash(row['current_uv']) != mean_hash(prior['source_uv']):
                raise ValueError('Unknown readout must retain the exact original UV')
            if rec['readout_status'] == 'uncertain_original_UV_retained': uncertain += 1
        if (row['source_uv'] != row['current_uv'] or row['current_uv'] != delta['next_source_uv']
            or mean_hash(row['current_uv']) != delta['next_source_uv_f64_sha256']
            or row['observation_generation_id'] != generation['observation_generation_id']):
            raise ValueError('Applied mean is not this exact source observation generation')
    if changed != 135 or uncertain != 3:
        raise ValueError('Actual readout effect or retained unknown denominator differs')
    return {r['source_row_id']: r for r in rows}
