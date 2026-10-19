"""Value-only replay of immutable original FIT board correspondences.

Uses the production finite-board binding and camera reconstruction. No source
admission, optimization, global source Jacobian, or source edits. The binding
retains its normal local geometry metadata. This does not accept accuracy.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]
sys.dont_write_bytecode = True
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from verify_current_native_primitives_20261007 import state_camera_adapter
from z190_motion_native import native_motion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, float)
    return dict(count=len(x), median=float(np.median(x)), p95=float(np.percentile(x, 95)), maximum=float(x.max()))


def actual_profile_indices(binding, block_name, profile, manifest):
    """Authenticate the saved chart before selecting its existing weights."""
    compaction = manifest['original_motion_input_compaction']
    actual_block = next(row for row in compaction['blocks'] if row['block'] == block_name)
    if len(profile) != actual_block['after_points']:
        raise ValueError('Saved profile length differs from actual compact source block')
    reference = binding.packet.get('original_input_compaction_map')
    if reference is None:
        if (actual_block['before_points'] != actual_block['after_points'] or
                actual_block['removed_points'] != 0):
            raise ValueError('Nonidentity profile chart needs an authenticated original compaction map')
        indices = binding.indices.copy()
        proof = dict(kind='actual_manifest_identity_map', actual_profile_points=len(profile),
                     original_and_runtime_indices_bitexact=True)
    else:
        parent = compaction['main_candidate']
        if sha(parent['path']) != parent['sha256']:
            raise ValueError('Actual manifest compaction candidate changed')
        candidate = json.loads(Path(parent['path']).read_text(encoding='utf8'))
        actual = next(row for row in candidate['classification'] if row['block'] == block_name)
        if (reference != actual['map'] or sha(reference['path']) != reference['sha256'] or
                actual['before_points'] != actual_block['before_points'] or
                actual['after_points'] != actual_block['after_points']):
            raise ValueError('Admitted role map differs from actual saved-state manifest map')
        with np.load(reference['path'], allow_pickle=False) as mapping:
            keep = mapping['keep_old_point_indices'].copy()
            inverse = mapping['old_point_to_compact_point'].copy()
            duplicate_keeper = dict(zip(map(int, mapping['removed_exact_duplicate_old_point_indices']),
                                        map(int, mapping['preserved_equal_source_point_for_each_removed'])))
        if (len(inverse) != actual_block['before_points'] or len(keep) != len(profile) or
                not np.array_equal(inverse[keep], np.arange(len(keep)))):
            raise ValueError('Exact original-to-compact profile bijection changed')
        selected = []
        for row in binding.records:
            old = int(row['original_global_bundle_index'])
            declared = int(row['current_compact_bundle_index'])
            if old < 0 or old >= len(inverse) or int(inverse[old]) != declared:
                raise ValueError('Role certificate original/compact profile index differs')
            if declared < 0:
                keeper = duplicate_keeper.get(old)
                if (declared != -1 or keeper is None or
                        row.get('current_runtime_duplicate_keeper_precompaction_bundle_index') != keeper or
                        row.get('current_runtime_equivalent_compact_bundle_index') != int(inverse[keeper]) or
                        int(inverse[keeper]) < 0):
                    raise ValueError('Removed evidence lacks its exact authenticated runtime keeper')
                declared = int(inverse[keeper])
            elif int(keep[declared]) != old:
                raise ValueError('Retained source index differs from actual compaction map')
            selected.append(declared)
        indices = np.asarray(selected, dtype=int)
        proof = dict(kind='actual_manifest_exact_compaction_map', map=reference,
                     actual_candidate=parent, actual_profile_points=len(profile),
                     role_map_equals_actual_manifest_map=True,
                     removed_evidence_mapped_only_to_certified_exact_duplicate_keeper=True)
    if np.any(indices < 0) or np.any(indices >= len(profile)):
        raise ValueError('Role profile index is outside the actual saved array')
    unique, first = np.unique(indices, return_index=True)
    first = np.sort(first)
    representatives = {int(indices[i]): int(i) for i in first}
    for i, row in enumerate(binding.records):
        peer = binding.records[representatives[int(indices[i])]]
        for field in ('source_pts', 'target_pts', 'source_uv', 'target_uv', 'source_sigma_px'):
            if not np.array_equal(row[field], peer[field]):
                raise ValueError('Two evidence rows assigned to one runtime source differ: ' + field)
    proof.update(original_evidence_points=len(indices), distinct_runtime_points=len(unique),
                 historical_duplicate_evidence_points=len(indices)-len(unique),
                 profile_array_sha256=hashlib.sha256(np.asarray(profile).tobytes()).hexdigest(),
                 original_indices_sha256=hashlib.sha256(binding.indices.tobytes()).hexdigest(),
                 actual_runtime_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                 original_runtime_identity_map=bool(np.array_equal(indices, binding.indices)))
    return indices, first, proof


def main():
    cli = argparse.ArgumentParser()
    for name in ('state-directory', 'admitted', 'auxiliary', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--source-manifest', type=Path,
                     help='Saved-state input manifest; defaults to the containing run')
    a = cli.parse_args()
    identity = json.loads((a.state_directory / 'state_identity.json').read_text(encoding='utf8'))
    state_path = a.state_directory / 'complete_state_UNACCEPTED.npz'
    if sha(state_path) != identity['state_npz_sha256'] or sha(identity['camera_path']) != identity['camera_sha256']:
        raise ValueError('Immutable accepted state or camera bytes changed')
    manifest_path = a.source_manifest or a.state_directory.parents[1] / 'input_manifest.json'
    if sha(manifest_path) != identity['input_manifest_sha256']:
        raise ValueError('Actual saved-state source manifest bytes changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    binding = OriginalMotionFarBoardRoleBinding(a.admitted)
    keys = ('q', 'aux__canonical_pts', 'aux__block_lengths', 'aux__block_start', 'aux__block_names',
            'aux__installation_xyz_slots', 'aux__current_shared_mount_parameter_indices',
            'aux__current_all_installation_optics_parameter_indices',
            'aux__correlated_roll_prior_contract_json', 'aux__current_complete_parameter_layout_json')
    with np.load(state_path, allow_pickle=False) as archive:
        small = {key: archive[key].copy() for key in keys}
        names = small['aux__block_names'].tolist()
        blocks = {row['source_role_block'] for row in binding.records}
        if len(blocks) != 1:
            raise ValueError('One immutable physical source block per replay required')
        block_name = next(iter(blocks))
        profile_name = 'profile__weights_' + str(names.index(block_name))
        profile = archive[profile_name].copy()
        if json.loads(str(archive['aux__current_original_motion_input_compaction_json'])) != manifest['original_motion_input_compaction']:
            raise ValueError('Saved-state auxiliary and manifest compaction contracts differ')
        runtime_indices, distinct_rows, mapping_proof = actual_profile_indices(binding, block_name, profile, manifest)
        weights = profile[runtime_indices].copy()
    if hashlib.sha256(small['q'].tobytes()).hexdigest() != identity['q_sha256']:
        raise ValueError('Canonical complete q identity differs')
    provider, world, proof = current_complete_source_geometry(state_path, a.auxiliary)
    cameras = dict(np.load(identity['camera_path'], allow_pickle=False))
    state, camera_error = state_camera_adapter(small, cameras, provider)
    q = state.evaluation_q(small['q'])
    source_pts = np.asarray([row['source_pts'] for row in binding.records], np.int64)
    target_pts = np.asarray([row['target_pts'] for row in binding.records], np.int64)
    stamps = np.unique(np.r_[source_pts, target_pts])
    block = state.bypts[int(source_pts[0])]
    if any(state.bypts[int(stamp)] is not block for stamp in stamps):
        raise ValueError('Source points cross physical camera blocks')
    local = block.local(q[:state.structured_programme_coordinate_chart.core_count])
    camera = {int(stamp): block.bundle.camera(local, int(stamp)) for stamp in stamps}
    ca = np.asarray([camera[int(stamp)] for stamp in source_pts])
    cb = np.asarray([camera[int(stamp)] for stamp in target_pts])
    rotations = {stamp: cv2.Rodrigues(value[3:6])[0] for stamp, value in camera.items()}
    binding.source_cameras = {int(stamp): camera[int(stamp)] for stamp in np.unique(source_pts)}
    bound = binding.current_binding(provider, q[state.geometry_parameter_indices])
    observed = np.asarray([row['target_uv'] for row in binding.records])
    residual, _, trace = native_motion(
        a=np.asarray([row['source_uv'] for row in binding.records]), b=observed,
        fa=np.exp(ca[:, 6]), fb=np.exp(cb[:, 6]),
        ra=np.asarray([rotations[int(stamp)] for stamp in source_pts]),
        rb=np.asarray([rotations[int(stamp)] for stamp in target_pts]),
        ca=ca[:, :3], cb=cb[:, :3], pp=ca[0, 8:10], k1=ca[0, 11],
        planes=bound['planes'], sigma=np.asarray([row['source_sigma_px'] for row in binding.records]),
        profile_weight=weights, da=None, db=None, dca_shared=None, dcb_shared=None,
        dca_view=None, dcb_view=None, jacobian=False,
        minimum_source_depth_scene_units=1. / provider.scene_unit_m, schur=True, return_trace=True)
    vector = residual.reshape(-1, 2)
    for i, runtime in enumerate(runtime_indices):
        representative = int(np.flatnonzero(runtime_indices == runtime)[0])
        if not np.array_equal(vector[i], vector[representative]) or not np.array_equal(trace[i], trace[representative]):
            raise ValueError('Exact duplicate evidence has different native runtime value/trace')
    distinct = np.zeros(len(vector), dtype=bool)
    distinct[distinct_rows] = True
    pair_rows = []
    for pa, pb in sorted(set(zip(source_pts.tolist(), target_pts.tolist()))):
        mask = (source_pts == pa) & (target_pts == pb)
        selected = mask & distinct
        pair_rows.append(dict(source_pts=pa, target_pts=pb, points=int(mask.sum()),
                              distinct_runtime_points=int(selected.sum()),
                              native_error_px=stats(np.linalg.norm(vector[selected], axis=1)),
                              signed_median_xy_px=np.median(vector[selected], axis=0).tolist()))
    result = dict(schema='rinkspace.immutable-original-finite-board-value-only-replay.v1',
                  q_sha256=identity['q_sha256'], state_path=str(state_path), state_sha256=sha(state_path),
                  admitted_packet=dict(path=str(a.admitted), sha256=sha(a.admitted)),
                  source_points=len(vector), actual_camera_archive_reproduce_max=camera_error,
                  distinct_runtime_points=len(distinct_rows),
                  native_error_px=stats(np.linalg.norm(vector[distinct], axis=1)),
                  actual_profile_weight_mapping=mapping_proof,
                  source_input_manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
                  actual_selected_profile_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                  original_source_UV_sigma_clock_verified_by_public_role_constructor=True,
                  evidence_records=[dict(original_global_bundle_index=int(row['original_global_bundle_index']),
                      actual_runtime_profile_index=int(runtime_indices[i]),
                      distinct_runtime_representative=bool(distinct[i]),
                      source_pts=int(row['source_pts']),target_pts=int(row['target_pts']),
                      source_uv=row['source_uv'],target_uv=row['target_uv'],source_sigma_px=row['source_sigma_px'],
                      exact_saved_profile_weights=weights[i].tolist(),
                      native_error_xy_px=vector[i].tolist(),finite=bool(trace[i,12]),unknown_depth=bool(trace[i,9]))
                      for i,row in enumerate(binding.records)],
                  all_points_finite=bool(np.all(trace[:, 12])),
                  unknown_depth_profile_absent=bool(not np.any(trace[:, 9])), pairs=pair_rows,
                  source_admission_or_FIT_UV_weights_modified=False, full_graph_or_J_constructed=False,
                  absolute_alignment_or_whole_video_accuracy_accepted=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: result[key] for key in ('q_sha256', 'source_points', 'native_error_px',
                     'all_points_finite', 'unknown_depth_profile_absent')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
