"""Exact current-role cache replay on two authenticated numeric producers."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from verify_original68_motion_role_production_20261008 import (
    producer_state, full_motion_values, SecondaryOpticsFixedXYZRinkBlock)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    for key in ('state', 'auxiliary', 'packet', 'output'):
        ap.add_argument('--' + key, required=True, type=Path)
    args = ap.parse_args()
    begin = time.perf_counter()
    state, q, saved, old, camera_error, production_error, contract, proof = producer_state(
        args.state, args.auxiliary, args.packet)
    role = state.original_motion_far_board_roles
    core_count = state.structured_programme_coordinate_chart.core_count
    world_indices = state.geometry_parameter_indices
    cases = [('nominal', q.copy())]
    ice = q.copy(); ice[world_indices[9]] += 1e-6
    course = q.copy(); course[world_indices[46]] += 1e-7
    cases.extend((('ice_radius', ice), ('course_radius', course)))
    for block in state.blocks:
        for name, column in (('view', block.start),
                             ('mount', block.optics_chart.mount_indices[0]),
                             ('lens', block.optics_chart.indices[0])):
            value = q.copy(); value[int(column)] += 1e-6
            cases.append((block.name + '_' + name, value))
    results = []
    for label, value in cases:
        start = time.perf_counter()
        state.geometry.validate_parameters(value[world_indices])
        camera_difference = 0.
        for block in state.blocks:
            block.local(value[:core_count])
            local = SecondaryOpticsFixedXYZRinkBlock.local(block, value[:core_count])
            for stamp in role.source_stamps_by_block[block.name]:
                camera_difference = max(camera_difference, float(np.abs(
                    role.source_cameras[stamp] - block.bundle.camera(local, stamp)).max()))
        actual = role.current_binding(state.geometry, value[world_indices])
        uncached_start = time.perf_counter()
        expected = role._build_current_binding(state.geometry, value[world_indices])
        uncached_seconds = time.perf_counter() - uncached_start
        assert same(actual, expected), label
        assert camera_difference == 0., (label, camera_difference)
        hit_start = time.perf_counter()
        assert same(role.current_binding(state.geometry, value[world_indices]), expected)
        hit_seconds = time.perf_counter() - hit_start
        results.append(dict(case=label, planes_J_extents_records_exact=True,
                            source_camera25_max_difference=camera_difference,
                            uncached_seconds=uncached_seconds, cache_hit_seconds=hit_seconds,
                            wall_seconds=time.perf_counter() - start))
        print(json.dumps(results[-1]), flush=True)
    # Real native kernel and finite/positive guards consume both producers.
    # Switch only this diagnostic instance between the one cached wrapper and
    # its unchanged builder, with identical arrays/calibration/profiles.
    cached_method = role.current_binding
    native = {}
    for name, method in (('cached', cached_method), ('uncached', role._build_current_binding)):
        role.current_binding = method
        native[name] = {}
        for block in state.blocks:
            block._course_current_geometry_key = None
            native[name][block.name] = full_motion_values(block, q, core_count)
    role.current_binding = cached_method
    assert same(native['cached'], native['uncached'])
    for block in state.blocks:
        bi = {'secondary': 2, 'low': 3}[block.name]
        assert np.array_equal(saved['profile__weights_' + str(bi)], block.bundle.profile_weights)
        assert all(np.array_equal(getattr(block.bundle, key), old[block.name][key])
                   for key in ('a', 'b', 'ia', 'ib', 'motion_sigma'))
    report = dict(schema='rinkspace.actual-original-motion-role-exact-cache-check.v1',
                  state=str(args.state), state_sha256=sha(args.state),
                  canonical_q_sha256=hashlib.sha256(saved['q'].tobytes()).hexdigest(),
                  packet=str(args.packet), packet_sha256=sha(args.packet),
                  parameter_count=len(q), world_count=len(world_indices),
                  roles=len(role.records), camera_reproduction=camera_error,
                  production_camera_reproduction=production_error,
                  cases=results, actual_full_saved_profile_native_kernel_bitexact=True,
                  actual_positive_and_finite_extent_guards_passed=True,
                  original_source_UV_sigma_PTS_profiles_and_rows_changed=False,
                  exact_cache_statistics=role.exact_cache_statistics,
                  full_solver_or_all_world_FD_evaluated=False,
                  wall_seconds=time.perf_counter() - begin, accuracy_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(report=str(args.output), sha256=sha(args.output),
                          wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
