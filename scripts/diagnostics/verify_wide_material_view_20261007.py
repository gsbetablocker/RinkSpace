"""Check exact wide source materials without building unrelated print scenes."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from rinkspace.vision_camera_initialization import current_complete_source_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True)
    parser.add_argument('--auxiliary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    provider, q, proof = current_complete_source_geometry(args.state, args.auxiliary)
    from wide_anchor_candidate.native_wide_source import load_native_wide_source_packet
    from wide_anchor_candidate.wide_user_ad_and_secondary_source import load_wide_user_ad_source
    from structured_programme_source_candidate.current_ice_material_scene import CurrentIceMaterialView
    rows = load_native_wide_source_packet().observations
    ads = load_wide_user_ad_source()['rows']
    geometry = provider.current
    view = CurrentIceMaterialView(geometry)

    def bundle(world, fast):
        if fast:
            geometry.validate_parameters(world)
            course, cq, _ = geometry.print_course_binding(world)
            scene = course.scene(cq, rebuild_grid=False)
            scene['ice_features'] = list(view.features(world).values())
        else:
            scene = geometry.source_scene(world)
        scene = provider.qualified_scene(scene)
        curves = geometry.one_scene_source_world_bundle(world, scene, rows)
        top = np.asarray(scene['board_surface']['top_edge_xyz'], float)
        if fast:
            sampled_top = course.sample_perimeter(cq, with_jacobian=False)[1]
            assert np.array_equal(top, sampled_top), 'Direct current course top differs from its scene export'
        corners = np.asarray([geometry.corner(r['user_ad_number'], r['role'], world) for r in ads])
        return [*curves, top, corners]

    states = [q]
    for col in range(66):
        for sign in (1, -1):
            value = q.copy()
            value[col] += sign * 1e-6
            states.append(value)
    old_seconds = new_seconds = maximum = 0.
    bitexact = True
    for index, value in enumerate(states):
        begin = time.perf_counter()
        original = bundle(value, False)
        old_seconds += time.perf_counter() - begin
        begin = time.perf_counter()
        candidate = bundle(value, True)
        new_seconds += time.perf_counter() - begin
        for old, new in zip(original, candidate):
            assert old.shape == new.shape
            maximum = max(maximum, float(np.max(np.abs(old - new))))
            bitexact &= np.array_equal(old, new)
        if index % 22 == 0:
            print(json.dumps(dict(states=index + 1, maximum_world_difference=maximum,
                old_seconds=old_seconds, new_seconds=new_seconds)), flush=True)
    report = dict(schema='rinkspace.exact-wide-material-view-equivalence.v1',
        complete_state_path=args.state, world_identity=proof,
        states_checked=len(states), world_columns_checked=66, finite_difference_step=1e-6,
        original_curve_count=len(rows), advertisement_corners=len(ads),
        board_top_included=True, direct_sampled_course_top_bitexact=True,
        maximum_world_difference=maximum, bitexact=bool(bitexact),
        original_seconds=old_seconds, material_view_seconds=new_seconds,
        unchanged_source_uv_sigma_operator_and_trial_guard=True)
    assert maximum <= 1e-12, report
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
