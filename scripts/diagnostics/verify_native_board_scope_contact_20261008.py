"""Check actual admitted board observations on an archived complete state.

Reuse the owned material snapshot and exact physical-camera archive. No
motion graph, source actors, normal matrix or synthetic observations are built.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(Path(__file__).resolve().parent)]
from calibration.solver_runtime import install
install()
import numpy as np
from prepared_cache_helpers import install_mappingproxy_pickle, install_geometry_unpickle
from verify_current_native_primitives_20261007 import state_camera_adapter
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from rinkspace.vision_camera_initialization import frozen_scene_curve


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True, type=Path)
    ap.add_argument('--owner-runtime-report', required=True, type=Path)
    ap.add_argument('--board-packet', required=True, type=Path)
    ap.add_argument('--parent-board-packet', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    started = time.perf_counter()
    identity = json.loads((args.state.parent/'state_identity.json').read_text(encoding='utf8'))
    assert digest(args.state) == identity['state_npz_sha256']
    assert digest(identity['camera_path']) == identity['camera_sha256']
    owner = json.loads(args.owner_runtime_report.read_text(encoding='utf8'))['material_owner']
    assert digest(owner['path']) == owner['sha256']
    install_mappingproxy_pickle()
    install_geometry_unpickle()
    provider = pickle.loads(Path(owner['path']).read_bytes())
    arrays = dict(np.load(args.state, allow_pickle=False))
    cameras = dict(np.load(identity['camera_path'], allow_pickle=False))
    state, camera_error = state_camera_adapter(arrays, cameras, provider)
    q = state.evaluation_q(arrays['q']).copy()
    provider.validate_parameters(q[state.geometry_parameter_indices])
    scene_path = args.state.parent/'scene_UNACCEPTED.json'
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    assert digest(scene_path) == identity['scene_sha256']
    factor = NativeBoardEdgeFactor(state, args.board_packet, len(q))
    world_checks = []
    for row in factor.rows+factor.held:
        if row['curve']['kind'] not in ('ice_board_contact', 'advertisement_boundary'):
            continue
        actual = factor._world(q, row['curve'])
        curve = dict(row['curve'])
        if row['curve']['kind'] == 'advertisement_boundary':
            curve['extent_scope'] = row['actual_modeled_extent_scope']
        exported = frozen_scene_curve(scene, curve)
        assert actual.shape == exported.shape, (row['id'], actual.shape, exported.shape)
        error = float(np.max(np.abs(actual-exported)))
        assert error < 1e-9, (row['id'], error)
        world_checks.append(dict(id=row['id'], curve=row['curve'], samples=len(actual),
                                 maximum_exported_world_difference=error))
    block = factor.block(q, True)
    retained = None
    if args.parent_board_packet:
        parent = NativeBoardEdgeFactor(state, args.parent_board_packet, len(q))
        parent_block = parent.block(q, True)
        offsets, offset = {}, 0
        for row in factor.rows:
            offsets[row['id']] = (offset, row)
            offset += 2*len(row['uv'])
        slots = []
        for old in parent.rows:
            start, row = offsets[old['id']]
            for field in ('uv', 'sigma', 'normal', 'normal_interval'):
                assert np.array_equal(old[field], row[field]), (old['id'], field)
            assert old['curve'] == row['curve'] and old['actual_pts'] == row['actual_pts']
            slots.extend(range(start, start+2*len(row['uv'])))
        slots = np.asarray(slots, int)
        assert np.array_equal(parent_block.r, block.r[slots])
        delta = parent_block.J-block.J[slots]
        old_error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        assert old_error < 2e-3, old_error
        retained = dict(parent_packet=str(args.parent_board_packet),
            parent_packet_sha256=digest(args.parent_board_packet),
            FIT_scalar_rows=len(parent_block.r), retained_source_UV_sigma_support_roles_exact=True,
            retained_r_bitexact=True, retained_J_max_difference=old_error)
    # An arbitrary direction must not leave the existing active Ad17 Z face.
    # Keep its four independent Z/height columns fixed for this small check.
    direction = np.zeros(len(q))
    columns = factor.active.copy()
    excluded = state.geometry_parameter_indices[[47, 56, 57, 71]]
    columns = columns[~np.isin(columns, excluded)]
    direction[columns] = np.random.default_rng(230024192).normal(size=len(columns))
    direction /= max(float(np.max(np.abs(block.J@direction))), 1.)
    h, support = factor.local_support_directional_step(q, direction, 1e-4)
    assert support['same_support_branch_step_found'], support
    provider.validate_parameters((q+h*direction)[state.geometry_parameter_indices])
    provider.validate_parameters((q-h*direction)[state.geometry_parameter_indices])
    finite = (factor.block(q+h*direction, False).r-factor.block(q-h*direction, False).r)/(2*h)
    error = float(np.max(np.abs(finite-block.J@direction)))
    assert error < 2e-3, (error, h)
    provider.validate_parameters(q[state.geometry_parameter_indices])
    report = dict(schema='rinkspace.actual-native-board-scope-contact-value-J-check.v1',
        complete_state=str(args.state), q_sha256=identity['q_sha256'],
        board_packet=str(args.board_packet), packet_sha256=digest(args.board_packet),
        material_owner=owner, physical_camera_archive_reproduce_max=camera_error,
        parameter_count=len(q), world_parameter_count=provider.parameter_count,
        FIT_scalar_rows=len(block.r), J_shape=list(block.J.shape), J_nnz=block.J.nnz,
        coupled_directional_max_error=error, validation_step=h,
        original_J_tolerance=2e-3, support_branch_selection=support,
        modeled_world_checks=world_checks, factor_contract=factor.contract,
        unchanged_parent_source_check=retained,
        actual_review=factor.review(q), full_physical_guard_passed=True,
        source_UV_sigma_roles_and_state_modified=False, motion_graph_loaded=False,
        normal_or_solver_started=False, accuracy_accepted=False,
        elapsed_seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(report_path=str(args.output), scalar_rows=len(block.r),
                         directional_max_error=error, seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
