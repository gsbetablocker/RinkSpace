"""Bounded checks for the current complete-objective performance probe."""
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/profile_contact_solver.py'
spec = importlib.util.spec_from_file_location('contact_profile_test_subject', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ContactProfileTests(unittest.TestCase):
    def args(self, directory, *extra):
        return probe.parser().parse_args([
            '--kind', 'network', '--project', directory, '--base-root', directory,
            '--case-root', directory, '--state', directory+'/initial_attempt.npz',
            '--initial', directory+'/initial.npz', '--evidence', directory+'/evidence.json',
            '--support', directory+'/support.json', '--clip', 'spin_s05',
            '--output', directory+'/new_profile', *extra])

    def test_default_is_one_complete_evaluation_and_no_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            probe.validate_args(args)
            self.assertEqual(args.operation, 'evaluate')
            self.assertEqual(args.repeats, 1)
            self.assertFalse(args.include_fun_jac)
            self.assertFalse(args.optimize_shared)

    def test_no_implicit_solve_budget_or_existing_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, '--run-policy'):
                probe.validate_args(self.args(tmp, '--operation', 'solve'))
            Path(tmp, 'new_profile').mkdir()
            sentinel = Path(tmp, 'new_profile', 'candidate.npz')
            sentinel.write_bytes(b'old candidate')
            with self.assertRaises(FileExistsError):
                probe.validate_args(self.args(tmp))
            self.assertEqual(sentinel.read_bytes(), b'old candidate')

    def test_support_requires_frozen_arc_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, '--kind', 'support')
            with self.assertRaisesRegex(ValueError, 'arc IDs'):
                probe.validate_args(args)
            args.support_profile = Path(tmp)/'profile.json'
            probe.validate_args(args)

    def test_network_solve_requires_fixed_quality_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, '--operation', 'solve', '--run-policy', tmp+'/run_policy.json')
            with self.assertRaisesRegex(ValueError, '--quality-baseline'):
                probe.validate_args(args)
            args.quality_baseline = Path(tmp)/'gvhmr_profile.json'
            with self.assertRaisesRegex(ValueError, '--jacobian-seconds-hint'):
                probe.validate_args(args)
            args.jacobian_seconds_hint = 103.755842
            probe.validate_args(args)

    def test_current_spin_requires_existing_grasp_inputs_frozen_shoes_and_solve_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, '--kind', 'current_spin')
            with self.assertRaisesRegex(ValueError, '--grasp-relations'):
                probe.validate_args(args)
            args.grasp_relations = Path(tmp)/'grasp_relations.json'
            args.grasp_geometry = Path(tmp)/'palm_grip_geometry.json'
            probe.validate_args(args)
            self.assertFalse(args.optimize_shared)
            args.optimize_shared = True
            with self.assertRaisesRegex(ValueError, 'frozen shared'):
                probe.validate_args(args)
            args.optimize_shared = False
            args.operation, args.run_policy = 'solve', Path(tmp)/'run_policy.json'
            with self.assertRaisesRegex(ValueError, '--quality-baseline'):
                probe.validate_args(args)
            args.quality_baseline = Path(tmp)/'profile.json'
            with self.assertRaisesRegex(ValueError, '--jacobian-seconds-hint'):
                probe.validate_args(args)
            args.jacobian_seconds_hint = 25.
            probe.validate_args(args)

    def test_current_spin_runtime_weights_preserve_immutable_source_and_disabled_rows(self):
        document = {'observations': [dict(id='middle', objective_weight=2.),
            dict(id='old_contact', objective_weight=3., optimization_enabled=False)]}
        before = deepcopy(document)
        runtime = probe.current_spin_runtime_document(document,
            {'factor_parameters': {'shoe_region_weight_multiplier': 160.}})
        self.assertEqual(document, before)
        self.assertEqual([r['objective_weight'] for r in runtime['observations']], [320., 480.])
        self.assertFalse(runtime['observations'][1]['optimization_enabled'])

    def test_current_spin_identity_checks_prepared_inputs_root_hands_and_all_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [
                ('case_input_sha256', 'case_input'), ('initial_reference_sha256', 'initial'),
                ('observations_sha256', 'observation'), ('compact_smplx_sha256', 'case_compact'),
                ('export_source_sha256', 'case_export_source'), ('shared_s01_parameters_sha256', 'shared_s01'),
                ('support_document_sha256', 'support'), ('action_evidence_sha256', 'evidence'),
                ('relation_document_sha256', 'grasp_relations'), ('geometry_semantics_sha256', 'grasp_geometry')]
            files, case = {}, dict(label='exmr_hand_nofoot', schema='rinkspace.network-comparison-case.v2',
                source_frames=2, source_time_base=[1, 12800], camera={'fixed_camera': 1})
            for field, key in pairs:
                files[key] = Path(tmp)/key
                files[key].write_bytes(key.encode())
                case[field] = probe.H.sha256_file(files[key])
            pts = np.array([100, 356], dtype='<i8')
            case['source_pts_sha256'] = probe.hashlib.sha256(pts.tobytes()).hexdigest()
            saved = dict(state_kind=np.array('spin_support_v2'), active=np.arange(63),
                grasp_relations_sha256=np.array(case['relation_document_sha256']),
                grasp_geometry_sha256=np.array(case['geometry_semantics_sha256']))
            initial = dict(root_curve_schema='saved', root_curve_knots=np.arange(8),
                root_curve_controls=np.zeros((4, 3)), root_curve_degree=3, hand_local_R=np.zeros((2, 2, 15, 3, 3)))
            modules = {
                'calibration.program_tracking.body_tracking.exmr_integration': SimpleNamespace(read_case_identity=lambda _: case),
                'calibration.program_tracking.body_tracking.skate_checkpoint': SimpleNamespace(CAMERA_LOCATION_FIELDS=set())}
            with patch.dict(sys.modules, modules):
                self.assertEqual(probe.validate_current_spin_inputs(saved, initial, {'pts': pts}, files,
                    {'fixed_camera': 1}), case)
                saved['active'] = np.arange(57)
                with self.assertRaisesRegex(ValueError, 'all 63'):
                    probe.validate_current_spin_inputs(saved, initial, {'pts': pts}, files, {'fixed_camera': 1})
                saved['active'] = np.arange(63)
                del initial['hand_local_R']
                with self.assertRaisesRegex(ValueError, 'both hand'):
                    probe.validate_current_spin_inputs(saved, initial, {'pts': pts}, files, {'fixed_camera': 1})
                files['grasp_relations'].write_bytes(b'changed source')
                with self.assertRaisesRegex(ValueError, 'relation_document_sha256'):
                    probe.validate_current_spin_inputs(saved, initial, {'pts': pts}, files, {'fixed_camera': 1})

    def test_current_mask_quality_reads_fixed_original_G_only_during_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path, profile_path = Path(tmp)/'result.npz', Path(tmp)/'profile.json'
            result_path.write_bytes(b'existing original G')
            case = {'label': 'gvhmr'}
            profile_path.write_text(json.dumps({'comparison_case_identity_json': json.dumps(case)}), encoding='utf8')
            pts = np.array([88*12800, 93*12800])
            original = dict(pts=pts[::-1], COCO_uv=np.array([np.full((17, 2), 9.), np.full((17, 2), 2.)]),
                comparison_stage=np.array('common_initial'))
            use = np.zeros((2, 17), bool); use[0, :3] = True
            observation = dict(pts=pts, uv=np.zeros((2, 17, 2)), use=use)
            baseline = {'groups': {'body': {'count': 999}}}
            trace = SimpleNamespace(identity={}, original_A=deepcopy(baseline), measure=lambda x, v: {'candidate': v})
            problem = SimpleNamespace(stage_quality=trace, body=SimpleNamespace(obs=observation, frames=np.arange(2)))
            calls = []
            def load(path):
                calls.append(path)
                return original
            def groups(predicted, observed, mask, clock):
                np.testing.assert_array_equal(clock, pts[:1])
                np.testing.assert_array_equal(predicted, np.full((1, 17, 2), 2.))
                np.testing.assert_array_equal(mask, use[:1])
                return {'body': {'count': int(mask.sum())}}
            modules = {
                'calibration.program_tracking.body_tracking.rebuild_prepare': SimpleNamespace(load=load),
                'calibration.program_tracking.body_tracking.stage_quality': SimpleNamespace(body_groups=groups),
                'calibration.program_tracking.body_tracking.exmr_integration': SimpleNamespace(read_case_identity=lambda _: case)}
            with patch.dict(sys.modules, modules):
                probe.attach_current_mask_baseline(problem, result_path, profile_path)
                self.assertEqual(calls, [])
                self.assertEqual(trace.original_A, baseline)
                self.assertEqual(trace.measure(None, 7), {'candidate': 7})
                self.assertEqual(trace.original_A['groups']['body']['count'], 3)
                self.assertEqual(trace.original_A['original_profile_metrics'], baseline)
                trace.measure(None, 8)
                self.assertEqual(calls, [result_path])

    def test_quality_baseline_preserves_original_instead_of_current_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json'
            baseline = {'groups': {'body': {'p95_px': 12.}}}
            path.write_text(json.dumps({'label': 'gvhmr', 'metrics': baseline}), encoding='utf8')
            self.assertEqual(probe.read_quality_baseline('network', path), baseline)
            self.assertEqual(probe.read_quality_baseline('current_spin', path), baseline)
            path.write_text(json.dumps({'label': 'exmr_final', 'metrics': baseline}), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'fixed gvhmr'):
                probe.read_quality_baseline('network', path)
            path.write_text(json.dumps({'candidates': {'original_A': {'measurement': baseline},
                'stage1_retained': {'measurement': {'groups': {'body': {'p95_px': .1}}}}}}), encoding='utf8')
            self.assertEqual(probe.read_quality_baseline('support', path), baseline)

    def test_cuda_policy_cannot_claim_cpu_analytic_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'one owner'):
                probe.validate_args(self.args(tmp, '--backend', 'cuda', '--workers', '4'))
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                probe.validate_args(self.args(tmp, '--backend', 'cuda', '--local-derivatives', 'analytic'))

    def test_recursive_clear_removes_inner_jacobian_but_keeps_model_cache(self):
        immutable = np.ones((2, 3))
        body = SimpleNamespace(cache={'jac': object()}, cache_x=np.zeros(3),
                               binding_key=b'old', _sample_double_cache=immutable)
        support = SimpleNamespace(cache={'jac': object()}, cache_x=np.zeros(3))
        problem = SimpleNamespace(cache={'jac': object()}, cache_x=np.zeros(20),
                                  body=body, support_image=support)
        support.body = body
        probe.clear_result_caches(problem)
        for obj in (problem, body, support):
            self.assertIsNone(obj.cache)
            self.assertIsNone(obj.cache_x)
        self.assertIsNone(body.binding_key)
        self.assertIs(body._sample_double_cache, immutable)

    def test_class_trace_preserves_copied_binding_ownership_and_restores(self):
        class Binding:
            def __init__(self, x):
                self.x = x
            def rows(self):
                return self.x * 7
        original = Binding.rows
        a = Binding(3)
        with probe.PhaseTrace() as trace:
            trace.patch(Binding, 'rows', 'rows')
            b = copy(a)
            b.x = 5
            self.assertEqual(a.rows(), 21)
            self.assertEqual(b.rows(), 35)
            self.assertEqual(trace.snapshot()['phases']['rows']['calls'], 2)
        self.assertIs(Binding.rows, original)

    def test_trace_restores_inherited_attribute_after_exception(self):
        class Base:
            def call(self):
                raise RuntimeError('expected failure')
        class Child(Base):
            pass
        with self.assertRaisesRegex(RuntimeError, 'expected failure'):
            with probe.PhaseTrace() as trace:
                trace.patch(Child, 'call', 'failing')
                Child().call()
        self.assertNotIn('call', vars(Child))
        row = trace.snapshot()['phases']['failing']
        self.assertEqual(row['failures'], 1)
        self.assertGreaterEqual(row['inclusive_call_seconds'], 0)

    def test_nested_trace_separates_inclusive_and_exclusive(self):
        class Work:
            def inner(self):
                return sum(range(500))
            def outer(self):
                return self.inner()+self.inner()
        with probe.PhaseTrace() as trace:
            trace.patch(Work, 'outer', 'outer')
            trace.patch(Work, 'inner', 'inner')
            self.assertEqual(Work().outer(), 249500)
            rows = trace.snapshot()['phases']
            self.assertEqual(rows['inner']['calls'], 2)
            self.assertGreaterEqual(rows['outer']['inclusive_call_seconds'], rows['inner']['inclusive_call_seconds'])
            self.assertAlmostEqual(rows['outer']['inclusive_call_seconds']-rows['outer']['exclusive_call_seconds'],
                                   rows['inner']['inclusive_call_seconds'], delta=1e-9)

    def test_lsmr_probe_keeps_solution_and_reports_iterations(self):
        owner = SimpleNamespace(lsmr=lambda: (np.array([2.]), 2, 5, .3, .04, 3., 4., 2.))
        original = owner.lsmr
        with probe.PhaseTrace() as trace:
            trace.patch(owner, 'lsmr', 'lsmr', lsmr=True)
            result = owner.lsmr()
            np.testing.assert_array_equal(result[0], [2.])
            self.assertEqual(trace.snapshot()['lsmr_calls'][0]['iterations'], 5)
        self.assertIs(owner.lsmr, original)

    def test_explicit_local_derivative_config_on_current_and_baseline(self):
        @dataclass
        class Baseline:
            backend: str = 'cpu'
            workers: int = 1
            blas_threads: int = 1
            gpu_batch_size: int = 512
            perturbation_batch: int = 4
            max_batch_samples: int = 4096
        @dataclass
        class Current(Baseline):
            local_derivatives: str = 'analytic'
        @dataclass
        class Resident(Current):
            compatibility_backend: str = 'cpu'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            args = self.args(tmp, '--local-derivatives', 'finite_difference')
            self.assertEqual(probe.execution_config(Current, args).local_derivatives, 'finite_difference')
            self.assertEqual(probe.execution_config(Baseline, args).workers, 1)
            self.assertEqual(os.environ['RINKSPACE_BODY_LOCAL_DERIVATIVES'], 'finite_difference')
            args.local_derivatives = 'analytic'
            with self.assertRaisesRegex(ValueError, 'no analytic'):
                probe.execution_config(Baseline, args)
            args.local_derivatives = None
            self.assertEqual(probe.execution_config(Current, args).local_derivatives, 'analytic')
            self.assertEqual(probe.execution_config(Resident, args).compatibility_backend, 'cpu')
            args.compatibility_backend = 'cuda_resident'
            self.assertEqual(probe.execution_config(Resident, args).compatibility_backend, 'cuda_resident')
            self.assertEqual(os.environ['RINKSPACE_BODY_COMPATIBILITY_BACKEND'], 'cuda_resident')
            with self.assertRaisesRegex(ValueError, 'no resident CUDA compatibility'):
                probe.execution_config(Current, args)

    def test_resident_compatibility_is_a_separate_gpu_scope_with_cpu_body_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp, '--backend', 'cpu', '--workers', '4',
                '--local-derivatives', 'analytic', '--compatibility-backend', 'cuda_resident')
            probe.validate_args(args)
            self.assertEqual(args.backend, 'cpu')
            self.assertEqual(args.compatibility_backend, 'cuda_resident')
            self.assertEqual(args.workers, 4)

    def test_resident_measurement_sync_reset_events_and_shared_allocator_without_cuda(self):
        calls = []
        class Cuda:
            def synchronize(self, device): calls.append(('sync', device))
            def reset_peak_memory_stats(self, device): calls.append(('reset_peak', device))
            def max_memory_allocated(self, device): return 123
            def max_memory_reserved(self, device): return 456
        device = SimpleNamespace(type='cuda', index=0)
        torch = SimpleNamespace(cuda=Cuda())
        measured = dict(calls=17, linearize_calls=2, trial_calls=15,
            h2d_bytes=100, d2h_bytes=200, h2d_seconds=.01, d2h_seconds=.02, compute_seconds_cuda_events=.3)
        resident = SimpleNamespace(torch=torch, device=device,
            record=lambda: dict(scope='selected skin compatibility', static_h2d_bytes=999),
            reset_measurement=lambda: calls.append(('reset_measurement', None)),
            measurement=lambda: measured)
        execution = SimpleNamespace(_body_backend=SimpleNamespace(torch=torch, device=device),
                                    _compatibility_backend=resident)
        self.assertEqual(set(probe.cuda_execution_owners(execution)), {'compact_body', 'compatibility'})
        self.assertEqual(set(probe.synchronize_execution(execution)), {'compact_body', 'compatibility'})
        probe.gpu_allocator_records(execution, reset=True)
        self.assertEqual(sum(name == 'reset_peak' for name, _ in calls), 1)
        memory = probe.gpu_allocator_records(execution)
        self.assertEqual(memory['peak_allocated'], 123)
        self.assertEqual(memory['peak_reserved'], 456)
        self.assertEqual(memory['devices'][0]['implementation_scopes'], ['compact_body', 'compatibility'])
        probe.reset_compatibility_measurement(execution)
        self.assertIn(('reset_measurement', None), calls)
        result = probe.compatibility_measurement(execution)
        self.assertEqual(result['measurement'], measured)
        self.assertEqual(result['implementation']['static_h2d_bytes'], 999)
        measured['calls'] = 99
        self.assertEqual(result['measurement']['calls'], 17)
        self.assertIn('only', result['implementation_scope'])

    def test_before_without_compatibility_owner_keeps_cpu_measurement_valid(self):
        execution = SimpleNamespace(_body_backend=None)
        self.assertEqual(probe.cuda_execution_owners(execution), {})
        self.assertEqual(probe.synchronize_execution(execution), {})
        probe.reset_compatibility_measurement(execution)
        self.assertIsNone(probe.compatibility_measurement(execution))
        self.assertEqual(probe.gpu_allocator_records(execution)['devices'], [])

    def test_fun_jac_equivalence_checks_full_csr_not_only_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            original = dict(residual=np.array([1., 2.]), jac=csr_matrix([[1., 0.], [0., 2.]]))
            snapshot = probe.H.snapshot_arrays(original, np.zeros(2), {'input_fingerprint': 'same'})
            result = probe.compare_cached_evaluation(snapshot, original, args)
            self.assertTrue(result['jacobian_within_tolerance'])
            wrong = dict(residual=original['residual'].copy(), jac=csr_matrix([[1., 0.], [0., 20.]]))
            with self.assertRaisesRegex(ValueError, 'complete evaluations differ'):
                probe.compare_cached_evaluation(snapshot, wrong, args)

    def test_sparse_entry_gate_catches_lost_small_row_in_both_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            original = dict(residual=np.zeros(2), jac=csr_matrix([[1e6, 0.], [0., 1.]]))
            wrong = dict(residual=np.zeros(2), jac=csr_matrix([[1e6, 0.], [0., 0.]]))
            identity = {'input_fingerprint': 'same'}
            old = probe.H.snapshot_arrays(original, np.zeros(2), identity)
            current = probe.H.snapshot_arrays(wrong, np.zeros(2), identity)
            path = Path(tmp)/'baseline.npz'
            np.savez(path, **old)
            result = probe.compare_snapshot(current, path, args)
            self.assertLess(result['jacobian_relative_frobenius'], args.jacobian_rtol)
            self.assertFalse(result['jacobian_within_tolerance'])
            self.assertEqual(result['jacobian_entry_violation_count'], 1)
            self.assertEqual(result['jacobian_worst_entry']['row'], 1)
            self.assertEqual(result['jacobian_worst_entry']['column'], 1)
            with self.assertRaisesRegex(ValueError, 'complete evaluations differ'):
                probe.compare_cached_evaluation(old, wrong, args)

    def test_equivalence_rejects_nonfinite_old_or_new_residual_and_jacobian(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(tmp)
            for index in range(4):
                values = [np.zeros(2), csr_matrix(np.eye(2)), np.zeros(2), csr_matrix(np.eye(2))]
                if index in (0, 2):
                    values[index][0] = np.inf
                else:
                    values[index].data[0] = np.inf
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    probe.compare_evaluations(*values, args)

    def test_frozen_support_arcs_cannot_be_silently_reselected(self):
        fit = dict(id='fit_contact', kind='support_contact_region', role='fit', foot=1,
                   visibility='visible', reliability=.35, contact_state='supported')
        review = dict(fit, id='review_contact', role='review')
        document = {'observations': [fit, review]}
        arcs = [{'id': 'runner_arc', 'semantic': 'runner'}, {'id': 'toe_pick', 'semantic': 'tooth'}]
        with self.assertRaisesRegex(ValueError, 'omits admitted fit rows'):
            probe.validate_frozen_arc_ids(document, {}, arcs)
        with self.assertRaisesRegex(ValueError, 'non-runner'):
            probe.validate_frozen_arc_ids(document, {'fit_contact': 'toe_pick'}, arcs)
        probe.validate_frozen_arc_ids(document, {'fit_contact': 'runner_arc'}, arcs)

    def test_disabled_current_spin_contact_pixels_do_not_reselect_an_arc(self):
        document = {'observations': [dict(id='old_contact', kind='support_contact_region', role='fit',
            foot=1, visibility='visible', reliability=.35, contact_state='supported', optimization_enabled=False)]}
        probe.validate_frozen_arc_ids(document, {}, [{'id': 'runner', 'semantic': 'runner'}])
        document['observations'][0]['optimization_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'omits admitted fit rows'):
            probe.validate_frozen_arc_ids(document, {}, [{'id': 'runner', 'semantic': 'runner'}])

    def test_source_selection_rejects_a_mixed_parent_package_before_loading_model(self):
        project = Path(__file__).resolve().parents[1]
        foreign = SimpleNamespace(__file__=str(project.parent/'other_checkout/calibration/__init__.py'))
        original_path = sys.path.copy()
        try:
            with patch.dict(sys.modules, {'calibration.foreign_probe_test': foreign}):
                with self.assertRaisesRegex(ValueError, 'Mixed checkout module'):
                    probe.select_project_sources(project)
        finally:
            sys.path[:] = original_path

    def test_actual_arrays_not_backend_settings_define_input_identity(self):
        compact = SimpleNamespace(J=np.ones((2, 3)), scale=1.)
        surface = SimpleNamespace(shaped=np.ones((3, 3)), weights=np.eye(3))
        reference = SimpleNamespace(body=compact, binding=SimpleNamespace(surface=surface))
        problem = SimpleNamespace(body=SimpleNamespace(state=SimpleNamespace(active=np.arange(3), knots=np.arange(8)),
            ref={'P': np.zeros((2, 3))}), physics_ref={'P': np.zeros((2, 3))},
            physics_times=np.array([0., 1.]), al=SimpleNamespace(multiplier=np.zeros(2), penalty=10.))
        setup = SimpleNamespace(problem=problem, reference=reference, camera=SimpleNamespace(p={'rig': np.ones(43)}),
                                objective={'kind': 'network'}, x=np.zeros(5))
        first = probe.numerical_identity(setup, {'state': {'sha256': 'abc'}})
        setup.requested_workers = 20
        setup.compatibility_backend = 'cuda_resident'
        second = probe.numerical_identity(setup, {'state': {'sha256': 'abc'}})
        self.assertEqual(first, second)
        compact.J[0, 0] += .01
        self.assertNotEqual(first, probe.numerical_identity(setup, {'state': {'sha256': 'abc'}}))

    def test_hands_and_exact_root_curve_are_in_the_numerical_fingerprint(self):
        reference = SimpleNamespace(body={}, binding=SimpleNamespace(surface={}),
            p_curve=SimpleNamespace(t=np.arange(8.), c=np.zeros((4, 3)), k=3))
        problem = SimpleNamespace(body=SimpleNamespace(state={}, ref={'hand_local_R': np.eye(3)[None]}),
            physics_ref={}, physics_times=np.array([0.]), al={})
        setup = SimpleNamespace(problem=problem, reference=reference, camera=SimpleNamespace(p={}),
                                objective={'kind': 'current_spin'}, x=np.zeros(5))
        first = probe.numerical_identity(setup, {})
        problem.body.ref['hand_local_R'][0, 0, 0] = .99
        self.assertNotEqual(first, probe.numerical_identity(setup, {}))
        problem.body.ref['hand_local_R'][0, 0, 0] = 1.
        reference.p_curve.c[0, 0] = .01
        self.assertNotEqual(first, probe.numerical_identity(setup, {}))


if __name__ == '__main__':
    unittest.main()
