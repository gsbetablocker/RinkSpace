"""Same-state shared-parameter replay through the production residual assembly."""
import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import skate_integration as skate
from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT
from calibration.program_tracking.body_tracking.programme_camera import StaticCropCamera


class SyntheticBinding:
    layout = OBSERVATION_LAYOUT
    collar_constraints_enabled = False
    compatibility_width = 32
    adapter_faces = np.array([[0, 1, 2]])
    sample_ids = np.array([0])
    neutral_native = np.zeros((1, 22, 3))
    neutral_frames = np.broadcast_to(np.eye(3), (1, 22, 3, 3))
    neutral_samples = np.zeros((1, 1, 3))

    def __init__(self):
        self.x = np.zeros(17)
        self.surface = SimpleNamespace(sample_vertices=self.sample_vertices)
        self.sensitivity = np.linspace(.2, .8, 17)
        angle = np.linspace(0, 2*np.pi, self.layout.shell_count, endpoint=False)
        shell = np.c_[.12*np.cos(angle), .08*np.sin(angle), np.zeros(len(angle))]
        edge = np.c_[np.linspace(-.1, .1, self.layout.edge_count),
                     np.full(self.layout.edge_count, -.02), np.zeros(self.layout.edge_count)]
        self.points = np.r_[shell, edge, edge + [0, .04, 0]]

    def evaluate(self, P, Q, joints, frames, *, skate_relative_R=None):
        relative = np.eye(3) if skate_relative_R is None else skate_relative_R
        axes = Q[:, None] @ frames[:, [7, 8]] @ relative
        shifted = self.points + [0, self.x @ self.sensitivity, 0]
        origins = P[:, None] + np.einsum('nij,nfj->nfi', Q, joints[:, [7, 8]])
        return dict(gear=origins[:, :, None] + np.einsum('nfij,kj->nfki', axes, shifted))

    def sample_vertices(self, joints, frames, vertex_ids, *, double_precision, hand_local_R=None):
        samples = joints[:, :1]
        if hand_local_R is not None:
            samples = samples + hand_local_R[:, :1, 0, :, 0]
        return samples

    def compatibility_rows(self, joints, frames, samples=None, *, hand_local_R=None, skate_relative_R=None):
        current = self.sample_vertices(joints, frames, self.sample_ids, double_precision=True,
                                       hand_local_R=hand_local_R)
        if samples is not None:
            np.testing.assert_allclose(samples, current, atol=1e-14, rtol=0)
        relative = np.eye(3) if skate_relative_R is None else skate_relative_R
        response = (.4 + .1*relative[..., 0, 0]) * (self.x @ self.sensitivity)
        rows = np.zeros((len(joints), self.compatibility_width))
        rows[:, [12, 20]] = .3 + .08*relative[..., 1, 0] + response + .01*current[:, :, 1]
        return rows


class SyntheticReference:
    layout = OBSERVATION_LAYOUT
    source_document = {'generic_context': True}
    new_refinements = ()

    def __init__(self, mode):
        self.mode = mode
        self.times = np.arange(3)*.02
        self.p = dict(pts=np.arange(3)*256, time_base=np.array([1, 12800]))
        self.body = object()
        self.binding = SyntheticBinding()
        self.footwear_observations = dict(
            visible_rows=[dict(pts=int(self.p['pts'][0]), foot=0, observed=[[445., 355.], [455., 357.]],
                               sigma_px=2., reverse_boundary=False)],
            blade_rows=[dict(pts=int(self.p['pts'][-1]), foot=1, uv=[[500., 355.], [513., 353.]],
                             sigma_px=2., partial=True)])

    def at(self, times):
        n = len(times)
        state = dict(P=np.tile([.1, -.1, 4.], (n, 1)), times=times,
                     Q=np.broadcast_to(np.eye(3), (n, 3, 3)),
                     local_R=np.broadcast_to(np.eye(3), (n, 21, 3, 3)))
        if self.mode != 'absent':
            angles = np.zeros((n, 2, 3))
            if self.mode == 'dynamic':
                angles[:, 0, 2] = .3 + 6*times
                angles[:, 1, 2] = -.5 + 4*times
                state['hand_local_R'] = np.broadcast_to(skate.matrices(angles[:, :, None]), (n, 2, 15, 3, 3))
            state['skate_relative_R'] = skate.matrices(angles)
        return state

    def world(self, ref, delta):
        n = len(delta)
        P = ref['P'] + delta[:, :3]
        Q = ref['Q'] @ skate.matrices(delta[:, 3:6])
        local = ref['local_R'] @ skate.matrices(delta[:, 6:].reshape(n, 21, 3))
        frames = np.concatenate((ref['Q'][:, None], local), axis=1)
        joints = .02*frames[..., 0]
        joints[:, [7, 8], 0] += [-.25, .25]
        native = P[:, None] + np.einsum('nij,nkj->nki', Q, joints)
        coco = native[:, :17]
        gear = self.binding.evaluate(P, Q, joints, frames, skate_relative_R=ref.get('skate_relative_R'))['gear']
        state = dict(P=P, Q=Q, native=native, coco=coco, frames=frames,
                     world=np.concatenate((gear.reshape(n, -1, 3), coco), axis=1))
        state.update({k: ref[k] for k in ('skate_relative_R', 'hand_local_R') if k in ref})
        return state


class SyntheticCamera(StaticCropCamera):
    def readout_times(self, frames):
        return np.full(len(frames), .008)


@pytest.mark.parametrize('mode', ['absent', 'identity', 'dynamic'])
def test_shared_replay_matches_full_residual_and_all_shared_columns(tmp_path, monkeypatch, mode):
    reference = SyntheticReference(mode)
    camera = SyntheticCamera(reference.p, np.tile([[400., 0, 480], [0, 400., 320], [0, 0, 1]], (3, 1, 1)),
                             np.eye(3), np.zeros(3))
    ledger = tmp_path / 'ledger.npz'
    np.savez(ledger, pts=reference.p['pts'], use=np.ones((3, 17), bool),
             uv=np.full((3, 17, 2), 420.), sigma_px=np.full((3, 17), 2.))

    # Only the unrelated ice geometry provider is bypassed. Keep the image
    # factors, compatibility, shared replay, cavity AL and sparse assembly.
    monkeypatch.setattr(skate, 'fk_state', lambda *args: {})
    monkeypatch.setattr(skate, 'constraints', lambda *args:
                        dict(g=np.empty(0), margin_m=np.empty(0), weights=np.empty(0),
                             values=np.empty(0), labels=[]))
    monkeypatch.setattr(skate, 'body_constraint_jacobian', lambda *args: csr_matrix((0, problem.body.size)))
    execution = BodyExecution(BodyExecutionConfig(local_derivatives='finite_difference'))
    execution.attach(reference)
    try:
        problem = skate.JointContactProblem(
            reference, camera, ledger, (0., .04), [], tmp_path, io.StringIO(), optimize_shared=True,
            ice_policy=skate.IceSoftPolicy(.005, .05, 1., .025, .08, 1.))
        x = np.linspace(-.004, .005, problem.size)
        linear = problem.evaluate(x, jacobian=True)
        base = problem.body.evaluate(x[17:])
        replay = problem.shared_replay(x[:17], x[17:], base, binding=copy.copy(reference.binding))
        np.testing.assert_allclose(replay['residual'], base['residual'], atol=2e-12, rtol=0)
        np.testing.assert_allclose(replay['compatibility_residual'], base['compatibility_residual'], atol=2e-12, rtol=0)

        blocks = {b.name: b for b in linear['weighted_blocks']}
        gear = blocks['old_gear_pixels']
        cavity = blocks['dynamic_foot_compatibility']
        assert [(choice['pts'], choice['foot']) for choice in gear.provenance['choices']] == [(512, 1)]
        np.testing.assert_array_equal(cavity.provenance['source_pts'], reference.p['pts'])
        expected = linear['jac'][:, :17].toarray()
        start = len(blocks['body_pixels'].residual)
        shell = slice(start, start + 2)
        assert np.all(np.linalg.norm(expected[shell], axis=0) > 0)
        assert np.all(np.linalg.norm(expected[base['block_slices']['foot_compatibility']], axis=0) > 0)
        for column in range(17):
            offset = np.zeros_like(x)
            offset[column] = 1e-5
            plus, minus = (problem.evaluate(x + sign*offset) for sign in (1, -1))
            for value in (plus, minus):
                assert [(b.name, b.residual.shape) for b in value['weighted_blocks']] == [
                    (b.name, b.residual.shape) for b in linear['weighted_blocks']]
                assert value['contact']['labels'] == linear['contact']['labels']
            numeric = (plus['residual'] - minus['residual']) / (2*offset[column])
            np.testing.assert_allclose(expected[:, column], numeric, atol=2e-8, rtol=2e-8)
    finally:
        execution.close()
