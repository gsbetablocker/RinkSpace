"""Synthetic case-identity/archive checks; no networks, native projection or fitting."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from calibration.program_tracking.body_tracking import exmr_integration as subject


class ComparisonReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.project=root/'project';self.base=root/'data'
        self.output=self.base/'compare';self.label='exmr_final'
        self.paths=subject.case_paths(self.project,self.base,self.output,self.label)
        self.case=self.paths['case'];self.case.mkdir(parents=True)
        self.pts=np.array([972800,973056,973312],dtype=np.int64)
        for key in ('native','prior','shared'):
            path=self.paths[key];path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,pts=self.pts)
        for key in ('support','evidence'):
            self.paths[key].write_text('{}',encoding='utf8')
        for name in ('input','initial','compact_smplx','export_source','observations'):
            np.savez(self.case/(name+'.npz'),pts=self.pts)
        self.info=dict(state_sha256='camera-state',native_library_sha256='native-binary',
                       record='/old/path',native_library='/old/library')
        self.identity=subject.case_identity(self.project,self.base,self.output,self.label,self.info)
        self.saved={subject.CASE_FIELD:np.array(json.dumps(self.identity,sort_keys=True))}

    def archive(self):
        target=self.project/subject.PROGRAMME/'test_initial.npz'
        target.write_bytes((self.case/'initial.npz').read_bytes())
        self.saved.update(initial_reference=np.array(target.relative_to(self.project).as_posix()),
                          initial_reference_sha256=np.array(subject.digest(target)))
        return target

    def test_case_identity_is_path_independent_and_requires_all_fields(self):
        moved=dict(self.info,record='/new/path',native_library='/new/library')
        expected=subject.case_identity(self.project,self.base,self.output,self.label,moved)
        self.assertEqual(expected,self.identity)
        subject.validate_case_identity(self.saved,expected)
        with self.assertRaisesRegex(ValueError,'Missing comparison case identity'):
            subject.validate_case_identity({},expected)
        incomplete=dict(self.identity);incomplete.pop('native_input_sha256')
        with self.assertRaisesRegex(ValueError,'Incomplete comparison case identity'):
            subject.validate_case_identity({subject.CASE_FIELD:np.array(json.dumps(incomplete))},expected)

    def test_wrong_case_or_different_own_prior_is_rejected(self):
        for key in ('label','camera_prior_sha256','initial_reference_sha256','native_input_sha256'):
            wrong=dict(self.identity);wrong[key]='different'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'identity mismatch'):
                subject.validate_case_identity({subject.CASE_FIELD:np.array(json.dumps(wrong))},self.identity)

    def test_archive_initial_path_is_consumed_and_hash_is_checked(self):
        target=self.archive()
        self.assertEqual(subject.resolve_initial_reference(self.project,self.saved),target.resolve())
        target.write_bytes(target.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'Initial reference SHA256 mismatch'):
            subject.resolve_initial_reference(self.project,self.saved)

    def test_archive_declared_sha_must_agree_with_case_identity(self):
        self.archive();self.saved['initial_reference_sha256']=np.array('wrong')
        with self.assertRaisesRegex(ValueError,'Initial reference SHA256 mismatch'):
            subject.resolve_initial_reference(self.project,self.saved)

    def test_initial_reference_cannot_escape_project(self):
        self.archive();self.saved['initial_reference']=np.array('../outside.npz')
        with self.assertRaisesRegex(ValueError,'project-relative'):
            subject.resolve_initial_reference(self.project,self.saved)

    def test_real_projection_fields_are_required_and_error_is_reported(self):
        packet=dict(pts=self.pts,source_pts=self.pts)
        for field in ('P_world_mid','Q_world_mid','local_R_mid','native22_world_mid','COCO_world_mid','COCO_uv'):
            packet[field]=np.zeros((3,3))
        packet['equipment_uv']=np.zeros((3,2,20,2))
        self.assertTrue(subject.replay_errors(packet,packet)['passed'])
        changed={k:v.copy() for k,v in packet.items()};changed['equipment_uv'][1,1,18,0]=.01
        report=subject.replay_errors(changed,packet)
        self.assertFalse(report['passed'])
        self.assertEqual(report['maximum_absolute_errors']['blade_native_uv'],.01)
        changed['pts']=self.pts+1
        with self.assertRaisesRegex(ValueError,'source PTS mismatch'):
            subject.replay_errors(changed,packet)


if __name__=='__main__':unittest.main()
