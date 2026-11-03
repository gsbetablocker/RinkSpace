"""Named compatibility block regressions without models or a native camera."""
import io
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import boot_integration as boot
from calibration.program_tracking.body_tracking import skate_integration as skate
from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.contact_feasibility import ContactAL


class TwoFrameState:
    active = np.array([6])
    d = 1
    size = 2

    def basis(self, times):
        t = np.asarray(times)
        return np.c_[1-t, t]

    def increments(self, controls, times):
        delta = np.zeros((len(times), 69))
        delta[:, 6] = self.basis(times)@controls
        return delta


class CompatibilityReference:
    def world(self, ref, delta):
        n = len(delta)
        native = np.zeros((n, 1, 3))
        native[:, 0, :2] = delta[:, 6, None]*[1., 2.]
        return dict(P=np.zeros((n, 3)), Q=np.tile(np.eye(3), (n, 1, 1)),
                    native=native, frames=np.tile(np.eye(3), (n, 1, 1, 1)))


class NamedBlocksTests(unittest.TestCase):
    def test_boot_named_residual_jacobian_and_slice_survive_appended_rows(self):
        body = boot.BootWindowProblem.__new__(boot.BootWindowProblem)
        body.binding = SimpleNamespace(x=np.zeros(1), compatibility_rows=lambda local, frames: local[:, 0, :2])
        body.binding_key = None
        body.reference = CompatibilityReference()
        body.state = TwoFrameState()
        body.size = 2
        body.mid_times = np.array([0., 1.])
        body.mid_ref = {}
        body.execution = BodyExecution(BodyExecutionConfig())
        base = dict(residual=np.array([7., 8.]), jac=csr_matrix(np.eye(2)), parts={}, phases={})
        with patch.object(boot.FullBodyProblem, 'evaluate', return_value=base):
            result = body.evaluate(np.array([3., 5.]), True)
        expected = np.array([3., 6., 5., 10.])/np.sqrt(2)
        expected_jac = np.array([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])/np.sqrt(2)
        result['residual'] = np.r_[result['residual'], [-900., -800.]]
        result['jac'] = boot.vstack((result['jac'], csr_matrix(np.full((2, 2), 99.))), format='csr')
        block = result['block_slices']['foot_compatibility']
        self.assertEqual((block.start, block.stop), (2, 6))
        np.testing.assert_allclose(result['compatibility_residual'], expected)
        np.testing.assert_allclose(result['residual'][block], expected)
        np.testing.assert_allclose(result['compatibility_jacobian'].toarray(), expected_jac, atol=1e-10)
        np.testing.assert_allclose(result['jac'][block].toarray(), expected_jac, atol=1e-10)

    def make_joint(self, tail):
        selected = np.array([12, 20, 44, 52])
        compat = np.zeros(64)
        compat[selected] = [.2, .3, .4, .5]
        named_jac = np.zeros((64, 2))
        named_jac[selected] = [[1., 0.], [0., 2.], [3., 0.], [0., 4.]]
        packet = dict(residual=np.r_[7., compat, tail],
                      compatibility_residual=compat, compatibility_jacobian=csr_matrix(named_jac),
                      block_slices={'foot_compatibility': slice(1, 65)}, parts={},
                      jac=csr_matrix(np.r_[np.zeros((1, 2)), named_jac, np.full((len(tail), 2), 999.)]))
        n = 10
        packet['world'] = dict(P=np.zeros((n, 3)), Q=np.tile(np.eye(3), (n, 1, 1)),
                               native=np.zeros((n, 1, 3)), frames=np.tile(np.eye(3), (n, 1, 1, 1)))
        binding = SimpleNamespace(x=np.zeros(17), collar_constraints_enabled=False,
            neutral_native=None, neutral_frames=None, neutral_samples=None, sample_ids=np.array([0]),
            compatibility_rows=lambda *args: np.zeros(32),
            surface=SimpleNamespace(sample_vertices=lambda *args, **kwargs: np.zeros((2, 1, 3))))
        joint = skate.JointContactProblem.__new__(skate.JointContactProblem)
        joint.binding = binding
        joint.ice_policy = skate.IceSoftPolicy(.005, .05, 1., .025, .08, 1.)
        joint.body = SimpleNamespace(n=2, size=2, evaluate=lambda x, jacobian=False: packet,
            execution=BodyExecution(BodyExecutionConfig()))
        joint.source_motion = SimpleNamespace(evaluate=lambda x, jacobian=False:
            dict(residual=np.empty(0), jac=csr_matrix((0, 2)), parts={}, metrics={}))
        joint.size = 19
        joint.cache = joint.cache_x = None
        joint.compatibility_width = 32
        joint.cavity_rows_per_frame = 2
        joint.foot_row_labels = [('neutral', 'left'), ('neutral', 'right'),
                                (0, 'left'), (0, 'right'), (1, 'left'), (1, 'right')]
        joint.row_labels = joint.foot_row_labels
        joint.physics_times = np.array([0., 1.])
        joint.contact = lambda x, jacobian=False: dict(margin_m=np.empty(0), g=np.empty(0),
            weights=np.empty(0), values=np.empty(0), labels=[], body_jac=csr_matrix((0, 2)))
        joint.shared_column = lambda c, x, base: (np.zeros(len(packet['residual'])+32+17), np.zeros(6))
        joint.al = ContactAL.zeros(6, penalty=10.)
        joint.start = time.perf_counter()
        joint.outer = 0
        joint.log = io.StringIO()
        return joint, named_jac[selected]

    def test_joint_cavity_value_and_body_jacobian_use_named_block_with_tail(self):
        ordinary, selected_jac = self.make_joint([])
        appended, _ = self.make_joint([900., 800., 700.])
        first = ordinary.evaluate(np.zeros(19), True)
        extra = appended.evaluate(np.zeros(19), True)
        expected = np.r_[0., 0., -.003*np.sqrt(2)*np.array([.2, .3, .4, .5])]
        np.testing.assert_allclose(extra['contact']['margin_m'], expected)
        np.testing.assert_allclose(extra['contact']['margin_m'], first['contact']['margin_m'])
        scale = np.sqrt(10.*.01)*.003*np.sqrt(2)/skate.NORMALIZATION
        np.testing.assert_allclose(extra['jac'][-6:, 17:].toarray(),
                                   np.r_[np.zeros((2, 2)), scale*selected_jac])
        np.testing.assert_allclose(extra['jac'][-6:, 17:].toarray(), first['jac'][-6:, 17:].toarray())

    def test_shared_column_reads_named_replay_compatibility_after_tail(self):
        joint, _ = self.make_joint([])
        joint.reference = CompatibilityReference()
        joint.body.state = TwoFrameState()
        joint.physics_ref = {}
        joint.origin = 0.
        joint.evidence = []
        selected = np.array([12, 20, 44, 52])
        rates = np.array([1., 2., 3., 4.])
        def replay(shared, body_x, packet, binding=None):
            compat = np.zeros(64)
            compat[selected] = .3+rates*shared[0]
            return dict(compatibility_residual=compat,
                        residual=np.r_[7., compat, 999.+1000*shared[0]])
        joint.shared_replay = replay
        # Exercise the production method, rather than make_joint's unrelated
        # shared-column stand-in used by the full evaluate() test above.
        with patch.object(skate, 'fk_state', return_value={}), patch.object(skate, 'constraints', return_value=dict(g=np.empty(0), margin_m=np.empty(0), values=np.empty(0), weights=np.empty(0), labels=[])):
            residual_column, guard_column = skate.JointContactProblem.shared_column(joint, 0, np.zeros(19), {})
        np.testing.assert_allclose(guard_column,
            np.r_[0., 0., -.003*np.sqrt(2)*rates/skate.NORMALIZATION], rtol=1e-9, atol=1e-10)
        self.assertGreater(abs(residual_column[65]), 100.)


if __name__ == '__main__':
    unittest.main()