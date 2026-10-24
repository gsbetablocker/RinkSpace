"""Declared objective transitions keep the complete import and refuse reseeds."""
from pathlib import Path
import tempfile
import unittest
from rinkfit_joint import training_source_resume_v1 as resume
from rinkfit_joint.io_utils import write_json


class TrainingResumeTests(unittest.TestCase):
    def fixture(self):
        producer = dict(packet=dict(file='packet', sha256='packet-sha'))
        state = dict(state_token='original-token', runtime_fingerprint='old-objective',
                     model=dict(file='model', sha256='model-sha'), checkpoint=dict(file='checkpoint', sha256='checkpoint-sha'))
        cfg = dict(source_uv_capture_objective=resume.OBJECTIVE, reinitialize_native_auxiliaries=False,
                   source_uv_capture_reseed_new_sources=False,
                   runtime_shared_parameter_stage_v2=dict(optical_feasible_coordinates=True, blocks=[]),
                   observed_optical_gap_blocks_v1=True, capture_certified_numeric_domain=dict(file='domain', sha256='domain-sha'),
                   representative_source_withdrawal_v1=dict(sha256='stale-revision'),
                   representative_material_rebind_v1=dict(sha256='stale-rebind'),
                   observed_primal_warm_start_v1=dict(sha256='superseded-historical-input'))
        delta = dict(representative_source_selection_v1=dict(sha256='training-v2'),
                     source_identity_verdicts_v1=dict(sha256='current-verdict'),
                     engineering_support_budget_v2=dict(total_budget=3.), audit_expected_source_counts=dict(s01=2))
        return producer, state, cfg, delta

    def test_new_objective_declares_exact_import_and_bounds_recovery(self):
        producer, state, cfg, delta = self.fixture()
        actual = resume.configuration(cfg, producer, state, delta, dict(sha256='continuation'))
        transition = actual['representative_objective_transition_v1']
        self.assertEqual(transition['from_state_token'], state['state_token'])
        self.assertEqual(transition['from_runtime_fingerprint'], state['runtime_fingerprint'])
        self.assertEqual(transition['from_checkpoint_sha256'], state['checkpoint']['sha256'])
        self.assertEqual(transition['from_model_sha256'], state['model']['sha256'])
        self.assertEqual(transition['kind'], 'complete_primal_warm_start')
        self.assertEqual(actual['capture_zoom_feasibility_recovery_v1']['certified_domain'], cfg['capture_certified_numeric_domain'])
        self.assertFalse(actual['runtime_shared_parameter_stage_v2']['optical_feasible_coordinates'])
        self.assertFalse(actual['observed_optical_gap_blocks_v1'])
        self.assertEqual(actual['accepted_preview_min_seconds_v1'], 120.)
        self.assertEqual(actual['accepted_checkpoint_min_seconds_v1'], 120.)
        self.assertEqual(actual['audit_expected_source_counts'], {'s01': 2})
        self.assertNotIn('representative_source_withdrawal_v1', actual)
        self.assertNotIn('representative_material_rebind_v1', actual)
        self.assertEqual(actual['observed_primal_warm_start_v1'], cfg['observed_primal_warm_start_v1'])
        self.assertTrue(cfg['runtime_shared_parameter_stage_v2']['optical_feasible_coordinates'])

    def test_reseed_or_unreviewed_delta_is_refused(self):
        producer, state, cfg, delta = self.fixture()
        cfg['reinitialize_native_auxiliaries'] = True
        with self.assertRaisesRegex(ValueError, 'without reseeding'):
            resume.configuration(cfg, producer, state, delta, {})
        cfg['reinitialize_native_auxiliaries'] = False
        delta['arbitrary_new_geometry'] = {}
        with self.assertRaisesRegex(ValueError, 'unreviewed changes'):
            resume.configuration(cfg, producer, state, delta, {})

    def test_candidate_header_and_raw_arrays_must_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'explicit_nuisance').mkdir()
            cp = ('RINKFIT_AL_CHECKPOINT_V1'+chr(10)+'"fingerprint"'+chr(10)+'0 0 0 0 0 0 0'+chr(10)+
                  '0'+chr(10)+'0'+chr(10)+'1'+chr(10)+'"token"'+chr(10)+
                  '"packet-sha" "a" "b" "c" "d" "'+resume.OBJECTIVE+'"'+chr(10))
            (root/'engineering_checkpoint.json').write_text(cp,'utf8')
            write_json(root/'model.json', dict(blocks=[dict(values=[1.])]))
            write_json(root/'observed_row_initialization_state_binding.json',
                       dict(source_packet_sha256='packet-sha',state_token='token',objective_id=resume.OBJECTIVE,
                            initialization_only=True,true_RS_verified=False,formal_Query_closed=False,phase_II_allowed=False))
            files = {}
            for name,width in (('station.f64',8),('row.f64',8),('alpha.f64',8),('path.u32',4)):
                file=root/'explicit_nuisance'/name;file.write_bytes(bytes(width))
                files[name]=dict(sha256=resume.reference(file)['sha256'],size_bytes=width)
            write_json(root/'explicit_nuisance/manifest.json',dict(schema='rinkfit.joint_explicit_PHR_nuisance_state.v1',
                       packet_sha256='packet-sha',raw_rows=1,files=files))
            state=resume.inspect_candidate(dict(packet=dict(sha256='packet-sha')),root)
            self.assertEqual(state['state_token'],'token')
            (root/'explicit_nuisance/row.f64').write_bytes(bytes(16))
            with self.assertRaisesRegex(ValueError,'Missing or changed'):
                resume.inspect_candidate(dict(packet=dict(sha256='packet-sha')),root)


if __name__ == '__main__':
    unittest.main()
