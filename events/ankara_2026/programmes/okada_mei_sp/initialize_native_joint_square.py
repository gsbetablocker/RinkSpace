"""Initialize square-pixel refinement from an already fitted joint image state.

This near-horizontal lift preserves the accepted horizontal rink layout and
starts the physical camera/height decomposition near its previous projection.
It is an initial guess, not a claim of exact projective equivalence. The normal
joint runner remains responsible for camera, scene and station refinement.
"""
from pathlib import Path
import argparse
import numpy as np
import prepare_native_joint_shots as prepared
from rinkspace.domain.parameters import SceneParameterModel, ParameterState


def square_initial_state(snapshot, layout):
    rig = snapshot['rig'].copy()
    controls = snapshot['controls'].copy()
    values = snapshot['scene_values'].copy()
    eta = float(rig[28])
    vertical_scale = float(np.exp(-eta))
    focal_shift = eta / 2
    rig[17] *= vertical_scale
    rig[19] *= vertical_scale
    rig[9:11] *= vertical_scale
    rig[9:12] /= np.linalg.norm(rig[9:12])
    rotation = rig[:9].reshape(3, 3).copy()
    forward = rotation[2].copy()
    forward[2] *= vertical_scale
    forward /= np.linalg.norm(forward)
    right = rotation[0] - np.dot(rotation[0], forward) * forward
    right /= np.linalg.norm(right)
    rig[:9] = np.array([right, np.cross(forward, right), forward]).ravel()
    axis = rig[:9].reshape(3, 3) @ rig[9:12]
    tilt = rig[12:15] - np.dot(rig[12:15], axis) * axis
    rig[12:15] = tilt / np.linalg.norm(tilt)
    controls[:, 1] *= vertical_scale
    controls[:, 2] += focal_shift
    # Preserve the zoom reference and wide basis, rather than changing their
    # pupil/principal-point/distortion functions during the focal-chart shift.
    rig[24] *= np.exp(focal_shift)
    rig[38] += focal_shift
    rig[28] = 0.
    values[5] *= vertical_scale
    for block in layout['blocks']:
        if not block['available']:
            continue
        offset = block['offset']
        if block['kind'] == 'print_size':
            values[offset + 1] += np.log(vertical_scale)
        elif block['kind'] == 'print_placement':
            values[offset + 1] *= vertical_scale
    ParameterState(layout, values)
    result = {**snapshot, 'rig': rig, 'controls': controls,
              'scene_values': values, 'board': values[:6].copy(),
              'layout': values[6:41].copy()}
    # This is an explicitly changed numerical initial state, never a seamless
    # optimizer resume under an old experiment identity.
    result.pop('camera_experiment_sha256', None)
    result.pop('camera_experiment_json', None)
    return result, dict(previous_fx_over_fy=float(np.exp(eta)),
                        initialized_fx_over_fy=1., vertical_scale=vertical_scale,
                        joint_scene_refinement_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = prepared.read(prepared.RUN / 'joint_manifest.json')
    mapping = prepared.read(prepared.RUN / 'mapping.json')
    model = SceneParameterModel(prepared.read(manifest['scene_path']),
        Path(manifest['scene_path']), optimize_scene=True)
    snapshot = prepared.npz(args.resume)
    points = prepared.npz(prepared.RUN / 'point_observations.npz')
    prepared.validate_joint_snapshot(snapshot, model.initial_state.layout,
        mapping, manifest, points)
    if args.output.exists():
        raise FileExistsError('Choose an unused initial-state output: ' + str(args.output))
    result, record = square_initial_state(snapshot, model.initial_state.layout)
    prepared.bind_joint_snapshot(result, model.initial_state.layout, mapping, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    prepared.write(args.output.with_suffix('.json'), dict(**record,
        source_snapshot=str(args.resume), source_sha256=prepared.digest(args.resume),
        source_observations_changed=False, source_grid_is_the_starting_reference=True,
        exact_projection_equivalence=False, initial_state=str(args.output)))
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
