from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
from scipy.sparse import csr_matrix, hstack

from calibration.program_tracking.body_tracking import blade_boot_refit as driver


class GroupedFactor:
    block_indices = {'normal_fan': [0, 1], 'gap': [2], 'u': [3], 'edge': [4]}

    def evaluate(self, x, jacobian=False):
        A = np.array([[2., 0.], [0., 3.], [1., -1.], [0., 4.], [0., 0.]])
        self.last = dict(residual=A @ x, jac=hstack((csr_matrix((5, 17)), csr_matrix(A)), format='csr')
                         if jacobian else None, parts={}, metrics={})
        return self.last


class DriverHelperTests(unittest.TestCase):
    def test_group_directional_check_preserves_fullwidth_and_unknown_edge(self):
        factor = GroupedFactor()
        x = np.r_[np.zeros(17), [.4, -.2]]
        d = np.r_[np.zeros(17), [.6, .8]]
        value, report = driver._factor_directional_check(factor, x, d, tuple(factor.block_indices))
        self.assertEqual(value['jac'].shape, (5, 19))
        self.assertFalse(driver._derivative_failed(report))
        self.assertEqual(report['groups']['u']['status'], 'checked')
        self.assertEqual(report['groups']['edge']['status'], 'unexercised')
        self.assertIsNone(report['groups']['edge']['relative_directional_error'])
        self.assertIs(factor.last, value)
        report['groups']['u']['relative_directional_error'] = .1
        self.assertTrue(driver._derivative_failed(report))


    def test_source_groups_keep_scope_foot_semantic_and_role(self):
        def row(seconds, foot, role, semantic, error):
            return dict(pts=round(seconds*12800), foot=foot, kind='boot_center_region',
                        source_semantic=semantic, role=role, centre_error_px=error,
                        outside_region_px=error/2, actual_geometry_centre_distance_px=error+1,
                        actual_geometry_outside_region_px=error/3)
        metrics = {'observations': [row(81.6, 0, 'fit', 'boot', 4.),
                                    row(82.2, 0, 'review', 'steel', 9.),
                                    row(90., 1, 'fit', 'boot', 100.)]}
        groups = driver._source_quality_groups(metrics, (81.2, 82.), (80.8, 82.4))
        self.assertEqual(len(groups['core']), 1)
        self.assertEqual(len(groups['context']), 2)
        self.assertEqual(groups['core'][0]['actual_geometry_centre_distance_px']['median'], 5.)
        self.assertEqual({r['role'] for r in groups['context']}, {'fit', 'review'})


    def test_preflight_rejects_existing_candidate_and_inconsistent_backup_without_writes(self):
        with TemporaryDirectory() as directory, patch.dict(driver.os.environ, {'RINKSPACE_BODY_DRIVER_BACKUP': ''}):
            tmp_path = Path(directory)
            project = tmp_path/'project'
            programme = project/driver.PROGRAMME
            programme.mkdir(parents=True)
            stage = tmp_path/'parent'
            stage.mkdir()
            parameter = programme/'parent.npz'
            parameter.write_bytes(b'parent')
            document = stage/'source.json'
            document.write_text('{}')
            output = tmp_path/'new_run'
            archive, backup = driver._preflight_targets(project, stage, parameter, document, output, 'candidate')
            self.assertEqual(backup, tmp_path/'implementation_before/driver.py')
            self.assertFalse(output.exists() or backup.exists())
            archive.write_bytes(b'preserved')
            with self.assertRaisesRegex(FileExistsError, 'final parameter'):
                driver._preflight_targets(project, stage, parameter, document, output, 'candidate')
            self.assertEqual(archive.read_bytes(), b'preserved')
            self.assertFalse(output.exists())
            backup.parent.mkdir()
            backup.write_bytes(b'earlier incompatible driver')
            with self.assertRaisesRegex(FileExistsError, 'inconsistent'):
                driver._preflight_targets(project, stage, parameter, document, output, 'other_candidate')
            self.assertEqual(backup.read_bytes(), b'earlier incompatible driver')
            self.assertFalse(output.exists())


    def test_active_support_outside_context_fails_before_creating_run(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Context must contain'):
                driver.run(tmp_path, tmp_path/'parent', tmp_path/'parent.npz', tmp_path/'source.json',
                           tmp_path/'new_run', (92.4, 93.6), (92., 94.), max_accepted=1000000,
                           wall_seconds=600., active_control_seconds=(91.8, 94.))
            self.assertFalse((tmp_path/'new_run').exists())


    def test_pose_prior_is_reported_separately_from_disabled_monitor(self):
        actions = SimpleNamespace(last={'parts': {'boot_axis': 2., 'runner_support_mode': 3.}},
                                  motion=SimpleNamespace(last={'parts': {'finite_network_pose_prior': 7.}}),
                                  runner_support=None)
        quality = driver._action_quality(actions)
        self.assertEqual(quality['pose_prior_cost'], 7.)
        self.assertEqual(quality['action_parts']['boot_axis'], 2.)
        self.assertTrue(quality['pose_prior_report_only'])
        self.assertFalse(quality['pose_quality_monitor_enabled'])
