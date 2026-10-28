"""Carry an accepted image fit into a constant-principal-point camera.

The native camera supplies every instantaneous rotation. The initializer
changes the existing pan/tilt coefficients to preserve each original optical
axis ray, instead of clearing principal-point terms and keeping old motion.
The regular joint solver subsequently refines the complete scene and camera.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import prepare_native_joint_shots as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel


def constant_principal_initial_state(core, snapshot, principal_point):
    base = snapshot['rig']
    rig = base.copy()
    controls = snapshot['controls'].copy()
    rig[26:28] = principal_point
    rig[29:31] = 0.
    maximum_ray_error = 0.
    def instant(physical, channels):
        pose = core.evaluate_motion_camera(physical, np.tile(channels, (4, 1)), 0., 1., [1.])
        if pose['status'][0]:
            raise ValueError('Invalid native instantaneous camera during initialization')
        return pose
    for i, original in enumerate(snapshot['controls']):
        old = instant(base, original)
        f = float(old['fx'][0]); fy = float(old['fy'][0])
        camera_matrix = np.array([[f, 0, rig[26]], [0, fy, rig[27]], [0, 0, 1.]])
        distortion = np.array([old[k][0] for k in ['k1', 'k2', 'p1', 'p2']] + [0.])
        target = cv2.undistortPoints(np.array([[[old['cx'][0], old['cy'][0]]]]),
                                    camera_matrix, distortion).reshape(2)
        original_axis = old['rotation'][0, 2]
        trial = original.copy()
        for iteration in range(12):
            rotation = instant(rig, trial)['rotation'][0]
            ray = rotation @ original_axis
            normalized = ray[:2] / ray[2]
            error = normalized - target
            if np.max(np.abs(error)) <= 1e-13:
                break
            columns = []
            for axis in [rotation @ rig[9:12], rig[12:15]]:
                derivative = np.cross(axis, ray)
                columns.append((derivative[:2] - normalized * derivative[2]) / ray[2])
            trial[:2] -= np.linalg.solve(np.column_stack(columns), error)
        else:
            raise RuntimeError('Native principal-point ray initialization did not settle')
        maximum_ray_error = max(maximum_ray_error, float(np.max(np.abs(error))))
        controls[i] = trial
    result = {**snapshot, 'rig': rig, 'controls': controls}
    result.pop('camera_experiment_sha256', None)
    result.pop('camera_experiment_json', None)
    return result, dict(principal_point_initial=list(map(float, principal_point)),
        constant_point_remains_fitted=True, maximum_axis_ray_error=maximum_ray_error,
        scene_initial_values_preserved=True, scene_refinement_is_joint=True,
        focal_distortion_readout_mount_initial_values_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cx', type=float)
    parser.add_argument('--cy', type=float)
    args = parser.parse_args()
    manifest = prepared.read(prepared.RUN / 'joint_manifest.json')
    mapping = prepared.read(prepared.RUN / 'mapping.json')
    points = prepared.npz(prepared.RUN / 'point_observations.npz')
    model = SceneParameterModel(prepared.read(manifest['scene_path']),
        Path(manifest['scene_path']), optimize_scene=True)
    snapshot = prepared.npz(args.resume)
    prepared.validate_joint_snapshot(snapshot, model.initial_state.layout,
        mapping, manifest, points)
    if args.output.exists():
        raise FileExistsError('Initial state output already exists: ' + str(args.output))
    camera = prepared.read(manifest['shared_camera_definition'])
    principal = [(camera['image_width']-1)/2 if args.cx is None else args.cx,
                 (camera['image_height']-1)/2 if args.cy is None else args.cy]
    result, receipt = constant_principal_initial_state(
        NativeCore(manifest['native_library']), snapshot, principal)
    prepared.bind_joint_snapshot(result, model.initial_state.layout, mapping, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    prepared.write(args.output.with_suffix('.json'), dict(**receipt,
        source_snapshot=str(args.resume), source_sha256=prepared.digest(args.resume),
        source_observations_changed=False, no_factory_principal_point_measurement=True))
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
