"""Query the saved ankle-attached steel runner trajectories at source time."""
from argparse import ArgumentParser
from copy import copy
import json
from pathlib import Path

import numpy as np

from .absolute_motion_state import absolute_reference, curve_from_packet
from .body_clock import source_seconds, time_base
from .programme_skate_evidence import programme_observation_path
from .rebuild_prepare import digest, load
from .sequence_skate_state import shoe_curve_from_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state
from .trajectory_backend import load_reference


class ProgrammeBladeTrajectory:
    """The existing continuous body's two attached steel trajectories in rink metres."""

    def __init__(self, project, prepared, source, parameter):
        prepared = Path(prepared)
        saved = load(parameter)
        camera_binding = json.loads(str(saved['camera_binding_json']))
        shared, _, camera, document, _ = load_reference(
            project, prepared, source, native_library=camera_binding.get('native_library'))
        validate_checkpoint_identity(saved, shared.fixed_checkpoint_identity)
        self.source_sha256 = digest(source)
        for field, expected in (
            ('source_evidence_sha256', self.source_sha256),
            ('prepared_input_sha256', digest(prepared / 'input.npz')),
            ('prepared_observations_sha256', digest(programme_observation_path(prepared, document))),
        ):
            if str(saved[field]) != expected:
                raise ValueError('Trajectory input changed: ' + field)
        self.parameter_sha256 = digest(parameter)
        self.camera_sha256 = camera['state_sha256']
        self.reference = absolute_reference(shared, curve_from_packet(saved), root=prepared)
        self.reference.relative_skate_curve = shoe_curve_from_packet(saved)
        binding = self.reference.binding = copy(shared.binding)
        binding.reference = self.reference
        binding.x = saved['x'][:17].copy()
        binding._parameter_key = binding._packet = None
        self.ratio = binding.ratio
        self.runner_local_m = binding.packet()['edges'] * self.ratio
        self.source_pts = shared.p['pts']
        self.source_time_s = source_seconds(shared.p)
        self.source_time_base = time_base(shared.p)
        self.shot_ids = shared.p['shot_ids']
        self.origin = float(self.source_time_s[0] - shared.times[0])
        np.testing.assert_allclose(self.source_time_s - self.origin, shared.times, rtol=0, atol=5e-10)
        self.document = document

    def at(self, source_time_s):
        """Return R(t), O(t) and both fixed material edges B(t) at broadcast times."""
        t = np.atleast_1d(np.asarray(source_time_s, dtype=float))
        if (t.ndim != 1 or not len(t) or not np.isfinite(t).all()
                or t.min() < self.source_time_s[0] or t.max() > self.source_time_s[-1]):
            raise ValueError('Query a finite time inside the fitted source programme')
        rotation, position = [], []
        for start in range(0, len(t), 640):
            clock = t[start:start + 640] - self.origin
            pose = fk_state(self.reference, self.reference.at(clock), np.zeros((len(clock), 69)))
            rotation.append(pose['R'])
            position.append(pose['O'] * self.ratio)
        R, O = np.concatenate(rotation), np.concatenate(position)
        edge = O[:, :, None, None, :] + np.einsum('tfij,esj->tfesi', R, self.runner_local_m)
        return dict(source_time_s=t, skate_R_world=R, skate_origin_world_m=O,
                    runner_edge_world_m=edge, runner_heel_world_m=edge[:, :, :, 0],
                    runner_toe_world_m=edge[:, :, :, -1])

    def packet(self, source_time_s=None):
        result = self.at(self.source_time_s if source_time_s is None else source_time_s)
        result.update(schema=np.array('rinkspace.programme-blade-trajectory.v1'),
                      source_time_base=np.asarray(self.source_time_base), foot_names=np.array(['left', 'right']),
                      lateral_edge_names=np.array(['negative_local_y', 'positive_local_y']),
                      runner_local_m=self.runner_local_m, parameter_sha256=np.array(self.parameter_sha256),
                      camera_sha256=np.array(self.camera_sha256), source_evidence_sha256=np.array(self.source_sha256))
        if source_time_s is None:
            result.update(source_pts=self.source_pts, shot_ids=self.shot_ids)
            role = np.full((len(self.source_pts), 2), 'no_native_steel_trace', dtype='U24')
            rows = self.document['programme_skate_evidence']['ordered_contours']
            for row in sorted(rows, key=lambda row: row['role'] == 'fit'):
                if row['role'] in ('fit', 'check') and row['semantic'] in ('runner_lower_edge', 'runner_upper_edge'):
                    role[np.searchsorted(self.source_pts, row['pts']), row['foot']] = row['role']
            result['native_steel_trace_role'] = role
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ('project', 'prepared', 'source', 'parameter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--at', type=float, nargs='+', help='Broadcast source seconds, including between native exposures')
    args = parser.parse_args()
    trajectory = ProgrammeBladeTrajectory(args.project, args.prepared, args.source, args.parameter)
    with args.output.open('xb') as output:
        np.savez_compressed(output, **trajectory.packet(args.at))


if __name__ == '__main__':
    main()
