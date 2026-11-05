"""Targeted checks for source-region preservation and unknown time gaps."""
import unittest
import numpy as np
from calibration.program_tracking.body_tracking.source_track_evidence import (exposure_cell_weights, propagated_region_shape, source_region_shape, summarize_source_regions, source_ledger_export, prepare_track_evidence, apply_seed_review_overrides, build_track_region_document)
from calibration.program_tracking.body_tracking.actual_observation_ledger import update_actual_observation_ledger,admitted_support_image_mask


class SourceEvidenceTests(unittest.TestCase):
    def test_unknown_seconds_cannot_become_an_anchor_weight(self):
        pts=np.arange(0,1100,100,dtype=np.int64)
        valid=np.zeros(len(pts),bool);valid[[0,1,9,10]]=True
        selected=np.zeros(len(pts),bool);selected[[0,10]]=True
        weights,audit=exposure_cell_weights(pts,valid,selected,time_base=(1,1000),native_step_pts=100)
        np.testing.assert_allclose(weights,[.2,.2])
        self.assertAlmostEqual(audit['assigned_seconds'],.4)
        self.assertEqual(len(audit['runs']),2)

    def test_removed_review_exposure_splits_time_support(self):
        pts=np.array([0,100,300,400],dtype=np.int64)
        weights,audit=exposure_cell_weights(pts,np.ones(4,bool),np.array([1,0,0,1],bool),time_base=(1,1000),native_step_pts=100)
        np.testing.assert_allclose(weights,[.2,.2])
        self.assertEqual(len(audit['runs']),2)
        self.assertFalse(audit['unknown_gaps_integrated'])

    def test_unrepresented_valid_run_stays_explicit(self):
        pts=np.arange(5,dtype=np.int64)
        weights,audit=exposure_cell_weights(pts,[True,False,True,False,True],[True,False,False,False,True],time_base=(1,50),native_step_pts=1)
        self.assertEqual(audit['unrepresented_valid_pts'],[2])
        self.assertAlmostEqual(sum(weights),.04)

    def test_source_rectangle_corners_survive_ellipse_conversion(self):
        shape=source_region_shape({'region_half_extent_uv':[25,10]})
        for point in ([25,10],[-25,10],[25,-10],[-25,-10]):
            self.assertLessEqual(float(np.array(point)@np.linalg.solve(shape,point)),1.+1e-12)
        explicit=source_region_shape({'region_shape_px2':[[3528,0],[0,98]]})
        np.testing.assert_array_equal(explicit,[[3528,0],[0,98]])

    def test_propagated_outer_envelope_contains_correlated_error(self):
        shape=np.diag([625.,100.]);affine=np.array([[1,.2],[-.1,.8]]);flow=np.diag([16.,9.])
        outer=propagated_region_shape(shape,affine,flow)
        angles=np.linspace(0,2*np.pi,37)
        for a in angles:
            for b in angles:
                delta=affine@np.array([25*np.cos(a),10*np.sin(a)])+np.array([4*np.cos(b),3*np.sin(b)])
                self.assertLessEqual(float(delta@np.linalg.solve(outer,delta)),1.+1e-12)

    def test_new_blade_types_are_reported_without_relabeling(self):
        rows=[dict(role='fit',kind='blade_endpoint_region',foot=1,pts=100,outside_region_px=3.,optimization_enabled=True),
              dict(role='fit',kind='support_contact_region',foot=1,pts=200,outside_region_px=17.,optimization_enabled=False)]
        result=summarize_source_regions(rows)['groups']
        self.assertEqual(result['fit/blade_endpoint_region/foot1']['objective_enabled'],1)
        self.assertEqual(result['fit/support_contact_region/foot1']['objective_enabled'],0)

    def test_export_uses_loaded_visibility_mask_and_objective_frames(self):
        pts=np.array([0,256,512],np.int64)
        use=np.ones((3,17),bool);use[1,9]=False
        obs=dict(pts=pts,use=use,uv=np.zeros((3,17,2)),sigma_px=np.ones((3,17)))
        exported=source_ledger_export(obs,pts,np.array([1],int),ledger_sha256='a'*64)
        self.assertFalse(exported['source_observation_use'][1,9])
        self.assertEqual(int(exported['source_observation_use'].sum()),50)
        self.assertEqual(int(exported['source_observation_objective_use'].sum()),16)
        self.assertFalse(exported['source_observation_objective_use'][0].any())
        self.assertIsNot(exported['source_observation_use'],obs['use'])



class SourceAdmissionTests(unittest.TestCase):
    def seed(self, pts=0):
        return dict(id='historical_'+str(pts),pts=pts,foot=1,role='fit',
            source_sha256='a'*64,centre_uv=[100.,200.],region_half_extent_uv=[25.,10.],
            visibility='partially_visible',source_frame='native.png',source_frame_sha256='b'*64)

    def document(self):
        return dict(source_id='original',source_sha256='a'*64,shot_id='s05',time_base=[1,12800],
            observations=[],supports=[])

    def track(self):
        return dict(pts=np.arange(6,dtype=np.int64)*256,uv=np.ones((6,2)),
            valid=np.array([1,0,1,0,1,1],bool),optimization_use=np.array([1,0,0,0,0,1],bool))

    def review(self,**extra):
        return dict(self.seed(),source_semantic='blade_midpoint_region',
            material_identity_confirmed=True,optimization_enabled=True,
            provenance='direct source image review',**extra)

    def test_every_short_valid_run_gets_a_real_representative(self):
        prepared=prepare_track_evidence(self.track())
        self.assertTrue(prepared['optimization_use'][2])
        self.assertEqual(prepared['source_exposure_audit']['added_representative_pts'],[512])
        self.assertEqual(prepared['source_exposure_audit']['unrepresented_valid_pts'],[])
        self.assertAlmostEqual(float(prepared['exposure_weight_seconds'].sum()),.08)

    def test_review_exposure_splits_runs_and_gets_no_duration(self):
        track=self.track();track['valid'][:]=True
        prepared=prepare_track_evidence(track,excluded_review_pts=[512,768])
        self.assertEqual(len(prepared['source_exposure_audit']['runs']),2)
        self.assertAlmostEqual(float(prepared['exposure_weight_seconds'].sum()),.08)
        self.assertFalse(prepared['optimization_use'][2:4].any())

    def test_unreviewed_seed_stays_coarse_and_keeps_original_extent(self):
        seed=self.seed();reviewed=apply_seed_review_overrides([seed],source_sha256='a'*64)[0]
        self.assertFalse(reviewed['optimization_enabled'])
        self.assertEqual(reviewed['source_semantic'],'coarse_steel_shadow_region')
        np.testing.assert_array_equal(reviewed['region_shape_px2'],np.diag([1250.,200.]))
        self.assertNotIn('region_shape_px2',seed)

    def test_confidence_one_does_not_admit_material_midpoint(self):
        document,track=build_track_region_document(self.document(),dict(self.track(),confidence=np.ones(6)),[self.seed()])
        self.assertEqual(document['observations'],[])
        self.assertEqual(document['source_evidence_admission']['saved_flow_material_rows_admitted'],0)

    def test_two_reviewed_endpoints_share_exposure_without_becoming_midpoint(self):
        overrides=[]
        for endpoint in ('front','rear'):
            row=self.review();row.update(source_semantic='blade_endpoint_region',blade_endpoint=endpoint)
            overrides.append(row)
        document,_=build_track_region_document(self.document(),self.track(),[self.seed()],review_overrides=overrides)
        self.assertEqual(len(document['observations']),2)
        self.assertEqual({r['blade_endpoint'] for r in document['observations']},{'front','rear'})
        self.assertTrue(all(r['kind']=='blade_endpoint_region' for r in document['observations']))
        self.assertTrue(all(r['objective_weight']==.02 for r in document['observations']))

    def test_direct_review_can_add_a_real_unknown_flow_exposure(self):
        row=self.review();row['pts']=256
        document,prepared=build_track_region_document(self.document(),self.track(),[self.seed()],review_overrides=[row])
        self.assertEqual(document['observations'][0]['pts'],256)
        self.assertFalse(prepared['valid'][1])
        self.assertEqual(document['observations'][0]['dependency_seed_pts'],[256])

    def test_protected_review_role_cannot_be_promoted(self):
        seed=self.seed();seed['role']='review'
        with self.assertRaisesRegex(ValueError,'Protected review'):
            apply_seed_review_overrides([seed],[self.review()],source_sha256='a'*64)

    def test_actual_ledger_export_keeps_original_file_hash_and_admission_edits(self):
        import hashlib,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.npz';path.write_bytes(b'original immutable file')
            pts=np.array([0,256],np.int64);use=np.ones((2,17),bool);use[1,2]=False
            obs=dict(pts=pts,use=use,uv=np.zeros((2,17,2)),sigma_px=np.ones((2,17)))
            body=SimpleNamespace(obs=obs,frames=np.array([1]),of=np.array([0,0]),oj=np.array([3,4]),ledger_path=path)
            packet=dict(pts=pts,source_observation_use=np.ones((2,17),bool))
            update_actual_observation_ledger(packet,body)
            self.assertEqual(int(packet['source_observation_objective_use'].sum()),2)
            self.assertFalse(packet['source_observation_use'][1,2])
            self.assertTrue(packet['source_observation_initial_use'][1,2])
            self.assertEqual(str(packet['source_observation_ledger_sha256']),hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(str(packet['source_observation_loaded_arrays_sha256'])),64)


if __name__=='__main__':
    unittest.main()
