"""Current source-group updates keep unrelated measurements and fitted state."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROGRAMME=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0,str(PROGRAMME))
spec=importlib.util.spec_from_file_location('native_source_update',PROGRAMME/'prepare_native_source_update.py')
update=importlib.util.module_from_spec(spec);spec.loader.exec_module(update)


class NativeSourceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='rink_source_update_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.parent=self.root/'current';self.parent.mkdir()
        self.layout=dict(size=10,layout_revision='layout',scene_content_revision='scene')
        self.model=SimpleNamespace(initial_state=SimpleNamespace(layout=self.layout))
        self.scene,self.library=self.root/'scene.json',self.root/'native.dll'
        self.scene.write_text('{}',encoding='utf8');self.library.write_bytes(b'No native library is loaded by these tests')
        profile=self.root/'profile.json';profile.write_text('{}',encoding='utf8')
        self.parameterization=dict(profile_path=str(profile),profile_sha256=update.prepared.digest(profile),stage='pupil')
        self.mapping=dict(path_ids=['pathA','pathB'],scene_revision='scene',source_video_sha256='a'*64)
        self.points=dict(pts=np.array([1000,1000,1256,1256,1512,1512,2000,2000,2256,2256],np.int64),
            observation_ids=np.array([f'old-{i}'for i in range(10)],dtype='U96'),
            path_indices=np.array([0,1]*5,np.int32),uv=np.arange(20,dtype=float).reshape(10,2)+20,
            kind=np.array([1,0]*5,np.int64),stations=np.array([.5,.25]*5),source_rows=np.arange(100,110,dtype=np.int64))
        self.shots=[dict(id=sid,row_begin=a,row_stop=b,control_begin=8*i,control_stop=8*i+8,camera_index=i,
            reference_pts=first,time_base_num=1,time_base_den=12800,first_control_time=-.06,knot_stride=1,
            motion_reference_time=0.,first_pts=first,last_pts=last,stop_pts_exclusive=last+256)
            for i,(sid,a,b,first,last)in enumerate([('s01',0,6,1000,1512),('s07',6,10,2000,2256)])]
        rig=np.arange(86,dtype=float).reshape(2,43)/100;rig[:,26:28]=[959.5,539.5];rig[:,[28,29,30]]=0
        self.state=dict(rig=rig,controls=np.arange(48,dtype=float).reshape(16,3)/100,
            scene_values=np.arange(10,dtype=float),board=np.arange(6,dtype=float),layout=np.arange(146,dtype=float),
            stations=np.where(self.points['kind']==1,.61+np.arange(10)*.01,.25),source_rows=self.points['source_rows'].copy())
        self.manifest=dict(schema='rinkspace.multicamera_disjoint_shots.v1',shots=copy.deepcopy(self.shots),
            cameras=[dict(id='main'),dict(id='other')],control_dt=.02,first_control_time=0.,max_readout=.02,
            scene_path=str(self.scene),scene_file_sha256=update.prepared.digest(self.scene),scene_revision='scene',
            native_library=str(self.library),native_library_sha256=update.prepared.digest(self.library),
            camera_parameterization=self.parameterization,spatial_refinement_preparation=dict(existing='preserve'))
        self.resume=self.parent/'accepted.npz';self.save_baseline()

    def save_baseline(self):
        np.savez(self.parent/'point_observations.npz',**self.points)
        self.manifest['point_input_sha256']=update.prepared.digest(self.parent/'point_observations.npz')
        update.prepared.write(self.parent/'joint_manifest.json',self.manifest)
        update.prepared.write(self.parent/'mapping.json',self.mapping)
        update.prepared.bind_snapshot(self.state,self.layout,self.mapping,self.manifest)
        np.savez(self.resume,**self.state)

    def packet(self,name='packet',fit=True):
        folder=self.root/name;folder.mkdir()
        paths=['pathA','pathC']
        z=dict(pts=np.array([1256,1256,1512],np.int64),
            observation_ids=np.array([f'{name}-{i}'for i in range(3)],dtype='U96'),
            path_indices=np.array([0,0,1],np.int32),uv=np.array([self.points['uv'][2],[100.,200.],[150.,250.]]),
            kind=np.ones(3,np.int64),stations=np.full(3,.5),source_rows=np.arange(900,903,dtype=np.int64))
        capture=folder/'source_rows.jsonl';raw=b'';locations=[]
        for number,pts in enumerate(np.unique(z['pts']),1):
            ids=np.flatnonzero(z['pts']==pts);actual=[dict(path_id=paths[z['path_indices'][i]],uv=z['uv'][i].tolist(),
                kind=int(z['kind'][i]),measurement_operator='actual_RGB_quantiles')for i in ids]
            frame=dict(pts=int(pts),source_BGR_sha256='b'*64,fit_eligible=fit,rows=actual)
            line=(json.dumps(frame)+'\n').encode();raw+=line
            for row_index,i in enumerate(ids):
                locations.append(dict(packet_row=int(i),observation_id=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),
                    pts=int(pts),path_id=paths[z['path_indices'][i]],uv=z['uv'][i].tolist(),source_BGR_sha256='b'*64,
                    source_jsonl=str(capture),source_line=number,source_line_sha256=hashlib.sha256(line).hexdigest(),row_index=row_index))
        capture.write_bytes(raw);capture_hash=update.prepared.digest(capture)
        for row in locations:row['source_jsonl_sha256']=capture_hash
        row_manifest=folder/'source_row_manifest.jsonl'
        row_manifest.write_text(''.join(json.dumps(row)+'\n'for row in locations),encoding='utf8')
        np.savez(folder/'point_observations.npz',**z)
        mapping=dict(path_ids=paths,source_video_sha256='a'*64,independent_source_capture=True,
            pixel_domain='native_distorted',image_size=[1920,1080],point_input_sha256=update.prepared.digest(folder/'point_observations.npz'),
            source_row_manifest='source_row_manifest.jsonl',source_row_manifest_sha256=update.prepared.digest(row_manifest),
            source_capture_jsonl=str(capture),source_capture_sha256=capture_hash)
        update.prepared.write(folder/'mapping.json',mapping)
        return folder,z,mapping

    def corrections(self,indices=(0,),name='corrections'):
        evidence=self.root/'source_review.json'
        evidence.write_text('{"review":"test-only source review evidence fixture"}',encoding='utf8')
        topology=self.root/'source_topology.txt'
        topology.write_text('test-only source topology evidence fixture',encoding='utf8')
        rows=[]
        for index in indices:
            shot=next(s for s in self.manifest['shots']if s['row_begin']<=index<s['row_stop'])
            rows.append(dict(observation_id=str(self.points['observation_ids'][index]),shot_id=shot['id'],
                pts=int(self.points['pts'][index]),path_id=self.mapping['path_ids'][self.points['path_indices'][index]],
                uv=self.points['uv'][index].tolist(),source_BGR_sha256='b'*64,
                reason='Source review shows the sampled pixels belong to a different printed contour.',
                source_review_disposition='confirmed_not_target_material_boundary',
                source_evidence=[dict(path=evidence.name,sha256=update.prepared.digest(evidence)),
                                 dict(path=str(topology),sha256=update.prepared.digest(topology))]))
        document=dict(schema=update.CORRECTIONS_SCHEMA,parent_input_sha256=self.manifest['point_input_sha256'],
                      source_video_sha256='a'*64,rows=rows)
        path=self.root/(name+'.json');update.prepared.write(path,document)
        return path,document

    def run_update(self,folder=None,sparse=False,native_library=None,source_corrections=None,initialized_rows=2):
        native=SimpleNamespace(paths={'pathA':{},'pathB':{},'pathC':{}},scene_revision='scene')
        def initialize(native_arg,**kw):
            self.assertTrue(np.all(kw['kinds']==1))
            self.assertEqual(len(kw['pts']),initialized_rows)
            np.testing.assert_array_equal(kw['controls'],self.state['controls'][:8])
            return np.full(len(kw['pts']),.91),dict(unresolved=[],groups=2)
        args=SimpleNamespace(run_root=self.parent,resume=self.resume,point_packets=[]if folder is None else[folder],
            output=self.root/'updated',allow_sparse_bootstrap=sparse,native_library=native_library,
            source_corrections=source_corrections)
        with patch.object(update,'SceneParameterModel',return_value=self.model), \
             patch.object(update,'ParameterState',return_value=SimpleNamespace()),patch.object(update,'NativeCore'), \
             patch.object(update,'NativeSceneGPU',return_value=native), \
             patch.object(update,'initialize_curve_stations',side_effect=initialize):
            return update.prepare(args)

    def test_update_preserves_unprovided_groups_and_every_fitted_numeric_global(self):
        folder,incoming,_=self.packet()
        before={str(p):update.prepared.digest(p)for p in self.parent.iterdir()}
        receipt=self.run_update(folder);out=self.root/'updated'
        z=update.prepared.npz(out/'point_observations.npz');v=update.prepared.npz(out/'initial_pixels.npz')
        mapping=update.prepared.read(out/'mapping.json');manifest=update.prepared.read(out/'joint_manifest.json')
        origins=update.prepared.npz(out/'source_origins.npz')
        retained=np.flatnonzero(origins['parent_row_indices']>=0);parents=origins['parent_row_indices'][retained]
        for key in self.points:
            self.assertEqual(z[key][retained].tobytes(),self.points[key][parents].tobytes())
            self.assertEqual(z[key].dtype,self.points[key].dtype)
        self.assertEqual(v['stations'][retained].tobytes(),self.state['stations'][parents].tobytes())
        for key in self.state.keys()-{'stations','source_rows','joint_numeric_dependencies_json','point_input_sha256'}:
            self.assertEqual(v[key].tobytes(),self.state[key].tobytes())
        self.assertEqual(mapping['path_ids'],['pathA','pathB','pathC'])
        self.assertEqual(manifest['camera_parameterization'],self.parameterization)
        self.assertEqual(manifest['spatial_refinement_preparation'],dict(existing='preserve'))
        self.assertEqual([(s['first_pts'],s['last_pts'],s['control_begin'],s['control_stop'])for s in manifest['shots']],
                         [(s['first_pts'],s['last_pts'],s['control_begin'],s['control_stop'])for s in self.shots])
        self.assertEqual((manifest['shots'][1]['row_begin'],manifest['shots'][1]['row_stop']),(8,12))
        added=np.flatnonzero(origins['packet_indices']==0)
        np.testing.assert_array_equal(z['uv'][added],incoming['uv'])
        np.testing.assert_array_equal(z['pts'][added],incoming['pts'])
        np.testing.assert_array_equal(v['stations'][added],[self.state['stations'][2],.91,.91])
        self.assertEqual(receipt['reused_resume_stations_for_same_source_pixels'],1)
        self.assertEqual(receipt['native_initialized_new_kind1_rows'],2)
        self.assertEqual(receipt['replaced_groups'],1);self.assertEqual(receipt['appended_groups'],1)
        self.assertEqual(receipt['withdrawn_rows'],0)
        self.assertEqual(receipt['packet_replaced_rows'],1)
        self.assertIsNone(receipt['source_corrections'])
        archive=update.prepared.npz(out/'replaced_source_observations.npz')
        np.testing.assert_array_equal(archive['parent_row_indices'],[2])
        update.prepared.validate_snapshot(v,self.layout,mapping,manifest,z)
        for path,sha in before.items():self.assertEqual(update.prepared.digest(path),sha)
        self.assertFalse(receipt['fit_executed'])

    def test_partial_line_updates_preserve_unread_ends_and_gaps(self):
        old = copy.deepcopy(self.points)
        old['pts'][:6] = 1000
        old['path_indices'][:6] = 0
        old['kind'][:6] = 1
        old['uv'][:6] = [[0,10],[20,10],[40,10],[60,10],[80,10],[100,10]]
        incoming = {key:value[:4].copy() for key,value in old.items()}
        incoming['observation_ids'] = np.array(['new-a','new-b','new-c','new-d'],dtype='U96')
        incoming['uv'] = np.array([[20.,11.],[40.,11.],[80.,11.],[100.,11.]])
        declared = dict(policy='partial_line_spans', groups=[dict(pts=1000,path_id='pathA',
            spans=[[[20.,11.],[40.,11.]],[[80.,11.],[100.,11.]]])])
        mapping = dict(self.mapping,source_replacement=declared)
        spans = update.authenticate_replacement_spans(mapping,incoming)
        packet = dict(shot_id='s01',points=incoming,mapping=mapping,replacement_spans=spans)
        merged = update.replace_source_groups(old,self.state,self.manifest,self.mapping,[packet])
        np.testing.assert_array_equal(merged['replaced'],[1,2,4,5])
        for parent in [0,3,6,7,8,9]:
            row = np.flatnonzero(merged['parent_rows']==parent).item()
            for key in old:
                self.assertEqual(merged['points'][key][row].tobytes(),old[key][parent].tobytes())
            self.assertEqual(merged['numeric'][row].tobytes(),self.state['stations'][parent].tobytes())
        self.assertEqual(merged['changes'][0]['action'],'replace_remeasured_line_spans')

    def test_partial_span_endpoints_must_be_actual_incoming_pixels(self):
        _,points,mapping = self.packet()
        mapping['source_replacement'] = dict(policy='partial_line_spans',groups=[
            dict(pts=1256,path_id='pathA',spans=[[[24.,25.],[101.,200.]]])])
        with self.assertRaisesRegex(ValueError,'endpoint is not an actual'):
            update.authenticate_replacement_spans(mapping,points)
        self.assertTrue(update.in_remeasured_line_spans([24.,25.],[np.array([[24.,25.],[24.,25.]])]))
        self.assertFalse(update.in_remeasured_line_spans([24.,26.],[np.array([[24.,25.],[24.,25.]])]))

    def test_occluded_support_withdrawal_preserves_its_distinct_review_reason(self):
        path,document=self.corrections()
        document['rows'][0]['source_review_disposition']='source_material_support_occluded'
        document['rows'][0]['reason']='Same-exposure person foreground covers the measured material interval.'
        path.write_text(json.dumps(document),encoding='utf8')
        receipt=self.run_update(source_corrections=path)
        review=update.prepared.read(self.root/'updated/source_corrections_receipt.json')
        self.assertEqual(receipt['withdrawn_rows'],1)
        self.assertEqual(review['rows'][0]['source_review_disposition'],'source_material_support_occluded')
        saved=update.prepared.npz(self.root/'updated/initial_pixels.npz')
        self.assertEqual(saved['controls'].tobytes(),self.state['controls'].tobytes())
        self.assertEqual(saved['stations'].tobytes(),self.state['stations'][1:].tobytes())

    def test_unresolved_source_identity_is_archived_without_claiming_wrong_pixels(self):
        path,document=self.corrections()
        document['rows'][0]['source_review_disposition']='source_entity_identity_unresolved'
        document['rows'][0]['reason']='Current source merges different printed letters into one band; the individual letter identity is not observable.'
        path.write_text(json.dumps(document),encoding='utf8')
        receipt=self.run_update(source_corrections=path)
        review=update.prepared.read(self.root/'updated/source_corrections_receipt.json')
        self.assertEqual(receipt['withdrawn_rows'],1)
        self.assertEqual(review['rows'][0]['source_review_disposition'],'source_entity_identity_unresolved')
        saved=update.prepared.npz(self.root/'updated/initial_pixels.npz')
        self.assertEqual(saved['stations'].tobytes(),self.state['stations'][1:].tobytes())
        document['rows'][0]['reason']='Large residual requires removal.'
        path.write_text(json.dumps(document),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'not fitted residuals'):
            update.authenticate_source_corrections(path,self.manifest,self.mapping,self.points)

    def test_explicit_withdrawal_keeps_same_group_other_shots_and_fitted_stations(self):
        # Two old rows share the exact shot/PTS/path group. Only one is wrong.
        self.points['path_indices'][1]=0;self.save_baseline()
        path,document=self.corrections()
        before={str(p):update.prepared.digest(p)for p in self.parent.iterdir()}
        receipt=self.run_update(source_corrections=path);out=self.root/'updated'
        z=update.prepared.npz(out/'point_observations.npz')
        saved=update.prepared.npz(out/'initial_pixels.npz')
        origins=update.prepared.npz(out/'source_origins.npz')
        kept=np.arange(1,10,dtype=np.int64)
        np.testing.assert_array_equal(origins['parent_row_indices'],kept)
        np.testing.assert_array_equal(origins['numeric_station_parent_rows'],kept)
        self.assertTrue(np.all(origins['packet_indices']==-1))
        for key in self.points:self.assertEqual(z[key].tobytes(),self.points[key][kept].tobytes())
        self.assertEqual(saved['stations'].tobytes(),self.state['stations'][kept].tobytes())
        for key in ('rig','controls','scene_values','board','layout'):
            self.assertEqual(saved[key].tobytes(),self.state[key].tobytes())
        archive=update.prepared.npz(out/'replaced_source_observations.npz')
        np.testing.assert_array_equal(archive['parent_row_indices'],[0])
        np.testing.assert_array_equal(archive['explicitly_withdrawn'],[True])
        np.testing.assert_array_equal(archive['replaced_by_packet'],[False])
        for key in self.points:self.assertEqual(archive[key].tobytes(),self.points[key][[0]].tobytes())
        self.assertEqual(archive['numeric_stations'].tobytes(),self.state['stations'][[0]].tobytes())
        self.assertEqual((out/'source_corrections.json').read_bytes(),path.read_bytes())
        review=update.prepared.read(out/'source_corrections_receipt.json')
        self.assertEqual(review['rows'][0]['parent_row_index'],0)
        self.assertEqual(review['rows'][0]['reason'],document['rows'][0]['reason'])
        self.assertEqual(review['rows'][0]['source_BGR_sha256'],'b'*64)
        self.assertEqual(review['source_BGR_sha256_basis'],
                         'source_review_declaration_not_parent_point_array_authentication')
        self.assertFalse(review['source_evidence_content_automatically_verified'])
        for item in review['rows'][0]['source_evidence']:
            self.assertEqual(receipt['protected_current_files_sha256'][item['path']],item['sha256'])
        self.assertEqual(receipt['withdrawn_rows'],1);self.assertEqual(receipt['retired_rows'],1)
        self.assertEqual(receipt['packet_replaced_rows'],0)
        self.assertEqual(receipt['withdrawn_rows_also_replaced_by_packets'],0)
        self.assertEqual(receipt['retained_parent_rows'],9)
        self.assertEqual(receipt['native_initialized_new_kind1_rows'],0)
        self.assertEqual(receipt['changed_groups'],0)
        self.assertFalse(receipt['fit_executed'])
        for file,sha in before.items():self.assertEqual(update.prepared.digest(file),sha)

    def test_withdrawals_and_packet_replacements_archive_the_union_once(self):
        folder,_,_=self.packet()
        path,_=self.corrections(indices=(0,2))
        receipt=self.run_update(folder,source_corrections=path,initialized_rows=3)
        out=self.root/'updated';archive=update.prepared.npz(out/'replaced_source_observations.npz')
        np.testing.assert_array_equal(archive['parent_row_indices'],[0,2])
        np.testing.assert_array_equal(archive['explicitly_withdrawn'],[True,True])
        np.testing.assert_array_equal(archive['replaced_by_packet'],[False,True])
        np.testing.assert_array_equal(archive['numeric_stations'],self.state['stations'][[0,2]])
        self.assertEqual(receipt['withdrawn_rows'],2)
        self.assertEqual(receipt['packet_replaced_rows'],1)
        self.assertEqual(receipt['withdrawn_rows_also_replaced_by_packets'],1)
        self.assertEqual(receipt['retired_rows'],2)
        self.assertEqual(receipt['reused_resume_stations_for_same_source_pixels'],0)
        review=update.prepared.read(out/'source_corrections_receipt.json')
        self.assertEqual([row['parent_row_index']for row in review['rows']],[0,2])
        saved=update.prepared.npz(out/'initial_pixels.npz')
        origins=update.prepared.npz(out/'source_origins.npz')
        np.testing.assert_array_equal(saved['stations'][origins['packet_indices']==0],[.91,.91,.91])

    def test_source_correction_authentication_rejects_mismatches_and_non_source_reasons(self):
        path,baseline=self.corrections()
        def row_change(key,value):
            candidate=copy.deepcopy(baseline);candidate['rows'][0][key]=value;return candidate
        cases=[('parent hash',dict(baseline,parent_input_sha256='c'*64),'parent point input SHA256'),
               ('video hash',dict(baseline,source_video_sha256='c'*64),'source video SHA256'),
               ('schema',dict(baseline,schema='unknown'),'Unsupported source correction schema'),
               ('unknown id',row_change('observation_id','missing'),'exactly one current observation_id'),
               ('shot',row_change('shot_id','s07'),'shot identity'),
               ('pts',row_change('pts',1256),'PTS/path/UV'),
               ('path',row_change('path_id','pathB'),'PTS/path/UV'),
               ('uv',row_change('uv',[20.,21.01]),'PTS/path/UV'),
               ('boolean uv',row_change('uv',[True,21.]),'finite UV'),
               ('nonfinite uv',row_change('uv',[float('nan'),21.]),'finite UV'),
               ('BGR digest',row_change('source_BGR_sha256','bad'),'SHA256'),
               ('disposition',row_change('source_review_disposition','large_residual'),'source review'),
               ('blank reason',row_change('reason','  '),'reason is required'),
               ('fit residual reason',row_change('reason','High reprojection residual'),'not fitted residuals'),
               ('Chinese residual reason',row_change('reason','拟合误差过大'),'not fitted residuals'),
               ('no evidence',row_change('source_evidence',[]),'nonempty source evidence'),
               ('no evidence hash',row_change('source_evidence',[dict(path='source_review.json')]),'fields must be exactly'),
               ('wrong evidence hash',row_change('source_evidence',[dict(path='source_review.json',sha256='c'*64)]),'file SHA256'),
               ('self evidence',row_change('source_evidence',[dict(path=path.name,sha256='c'*64)]),'external source evidence'),
               ('empty rows',dict(baseline,rows=[]),'nonempty explicit rows')]
        duplicate=copy.deepcopy(baseline);duplicate['rows']*=2
        cases.append(('duplicate',duplicate,'Duplicate source correction'))
        residual_field=copy.deepcopy(baseline);residual_field['rows'][0]['residual_px']=99
        cases.append(('residual field',residual_field,'fields must be exactly'))
        missing=copy.deepcopy(baseline);del missing['rows'][0]['source_evidence']
        cases.append(('missing evidence field',missing,'fields must be exactly'))
        for name,candidate,message in cases:
            with self.subTest(name=name):
                # Deliberately allow nonfinite input here to exercise the reader's guard.
                path.write_text(json.dumps(candidate,ensure_ascii=False),encoding='utf8')
                with self.assertRaisesRegex(ValueError,message):self.run_update(source_corrections=path)
                self.assertFalse((self.root/'updated').exists())

    def test_source_correction_missing_evidence_file_and_ambiguous_parent_are_rejected(self):
        path,document=self.corrections()
        evidence=self.root/document['rows'][0]['source_evidence'][0]['path'];evidence.unlink()
        with self.assertRaises(FileNotFoundError):
            update.authenticate_source_corrections(path,self.manifest,self.mapping,self.points)
        self.corrections()
        ambiguous={key:value.copy()for key,value in self.points.items()}
        ambiguous['observation_ids'][1]=ambiguous['observation_ids'][0]
        with self.assertRaisesRegex(ValueError,'exactly one current observation_id'):
            update.authenticate_source_corrections(path,self.manifest,self.mapping,ambiguous)

    def test_no_packets_or_corrections_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Provide point packets or explicit source corrections'):
            self.run_update()
        self.assertFalse((self.root/'updated').exists())

    def test_explicit_resume_library_override_updates_metadata_without_rewriting_state(self):
        folder,_,_=self.packet()
        new_library=self.root/'dynamic.dll';new_library.write_bytes(b'dynamic-cut native revision fixture')
        new_hash=update.prepared.digest(new_library)
        self.state['native_library_provenance_sha256']=np.array(new_hash)
        np.savez(self.resume,**self.state)
        with self.assertRaisesRegex(ValueError,'provide --native-library'):self.run_update(folder)
        self.assertFalse((self.root/'updated').exists())
        receipt=self.run_update(folder,native_library=new_library)
        manifest=update.prepared.read(self.root/'updated/joint_manifest.json')
        saved=update.prepared.npz(self.root/'updated/initial_pixels.npz')
        self.assertEqual(manifest['native_library'],str(new_library))
        self.assertEqual(manifest['native_library_sha256'],new_hash)
        self.assertEqual(saved['native_library_provenance_sha256'].tobytes(),self.state['native_library_provenance_sha256'].tobytes())
        for name in ('rig','scene_values','controls','board','layout'):
            self.assertEqual(saved[name].tobytes(),self.state[name].tobytes())
        revision=receipt['native_library_revision']
        self.assertTrue(revision['explicit_override']);self.assertTrue(revision['manifest_library_revision_changed'])
        self.assertEqual(revision['parent_manifest_library_sha256'],self.manifest['native_library_sha256'])
        self.assertFalse(revision['numeric_state_provenance_rewritten'])

    def test_override_with_a_different_actual_library_cannot_forge_resume_provenance(self):
        folder,_,_=self.packet()
        wrong=self.root/'wrong.dll';wrong.write_bytes(b'not the resume native revision')
        with self.assertRaisesRegex(ValueError,'differs from resume.native_library_provenance'):
            self.run_update(folder,native_library=wrong)
        self.assertFalse((self.root/'updated').exists())

    def test_non_fit_eligible_and_unknown_fields_are_rejected_before_output(self):
        folder,_,_=self.packet(fit=False)
        with self.assertRaisesRegex(ValueError,'explicitly fit_eligible'):self.run_update(folder)
        self.assertFalse((self.root/'updated').exists())
        other,z,m=self.packet('weighted')
        np.savez(other/'point_observations.npz',**z,weights=np.ones(3))
        m['point_input_sha256']=update.prepared.digest(other/'point_observations.npz');update.prepared.write(other/'mapping.json',m)
        with self.assertRaisesRegex(ValueError,'seven unweighted'):self.run_update(other)
        self.assertFalse((self.root/'updated').exists())

    def test_empty_packets_and_changed_source_provenance_cannot_delete_old_groups(self):
        folder,z,m=self.packet()
        np.savez(folder/'point_observations.npz',**{k:v[:0]for k,v in z.items()})
        m['point_input_sha256']=update.prepared.digest(folder/'point_observations.npz');update.prepared.write(folder/'mapping.json',m)
        with self.assertRaisesRegex(ValueError,'Empty point packets'):self.run_update(folder)
        path,_=self.corrections()
        with self.assertRaisesRegex(ValueError,'Empty point packets'):
            self.run_update(folder,source_corrections=path)
        other,_,_=self.packet('changed')
        (other/'source_rows.jsonl').write_text('{}\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'qualification dependency changed'):self.run_update(other)
        self.assertFalse((self.root/'updated').exists())

    def test_one_actual_source_row_cannot_be_imported_twice(self):
        folder,z,m=self.packet()
        z['uv'][1]=z['uv'][0]
        np.savez(folder/'point_observations.npz',**z)
        path=folder/'source_row_manifest.jsonl'
        locations=[json.loads(line)for line in path.read_text().splitlines()]
        locations[1]['uv']=z['uv'][0].tolist();locations[1]['row_index']=0
        path.write_text(''.join(json.dumps(row)+'\n'for row in locations),encoding='utf8')
        m.update(point_input_sha256=update.prepared.digest(folder/'point_observations.npz'),
                 source_row_manifest_sha256=update.prepared.digest(path))
        update.prepared.write(folder/'mapping.json',m)
        with self.assertRaisesRegex(ValueError,'actual source row is repeated'):self.run_update(folder)
        self.assertFalse((self.root/'updated').exists())

    def test_sparse_bootstrap_is_explicit_and_keeps_complete_shot_clock(self):
        keep=self.points['pts']!=1256
        self.points={k:v[keep]for k,v in self.points.items()}
        self.state['stations']=self.state['stations'][keep];self.state['source_rows']=self.points['source_rows']
        self.manifest['shots'][0]['row_stop']=4
        self.manifest['shots'][1].update(row_begin=4,row_stop=8)
        self.save_baseline()
        folder,z,m=self.packet()
        # Update only the already-covered pathC exposure; missing 1256 stays missing.
        z={k:v[2:3]for k,v in z.items()}
        np.savez(folder/'point_observations.npz',**z)
        # Use a separately authenticated one-row packet from a fresh fixture.
        capture=folder/'source_rows.jsonl'
        frame=dict(pts=1512,fit_eligible=True,source_BGR_sha256='b'*64,rows=[dict(path_id='pathC',uv=[150.,250.],measurement_operator='actual_RGB_quantiles')])
        raw=(json.dumps(frame)+'\n').encode();capture.write_bytes(raw);capture_hash=update.prepared.digest(capture)
        location=dict(packet_row=0,observation_id=str(z['observation_ids'][0]),source_row=int(z['source_rows'][0]),pts=1512,
            path_id='pathC',uv=[150.,250.],source_BGR_sha256='b'*64,source_jsonl=str(capture),source_jsonl_sha256=capture_hash,
            source_line=1,source_line_sha256=hashlib.sha256(raw).hexdigest(),row_index=0)
        source_manifest=folder/'source_row_manifest.jsonl';source_manifest.write_text(json.dumps(location)+'\n')
        m.update(point_input_sha256=update.prepared.digest(folder/'point_observations.npz'),source_capture_sha256=capture_hash,
                 source_row_manifest_sha256=update.prepared.digest(source_manifest));update.prepared.write(folder/'mapping.json',m)
        with self.assertRaisesRegex(ValueError,'Incomplete shot exposure'):self.run_update(folder)
        # This case initializes one new row, so use the real existing mock hook with a smaller return.
        native=SimpleNamespace(paths={'pathA':{},'pathB':{},'pathC':{}},scene_revision='scene')
        args=SimpleNamespace(run_root=self.parent,resume=self.resume,point_packets=[folder],output=self.root/'updated',allow_sparse_bootstrap=True)
        with patch.object(update,'SceneParameterModel',return_value=self.model),patch.object(update,'ParameterState',return_value=SimpleNamespace()), \
             patch.object(update,'NativeCore'),patch.object(update,'NativeSceneGPU',return_value=native), \
             patch.object(update,'initialize_curve_stations',return_value=(np.array([.91]),dict(unresolved=[],groups=1))):
            receipt=update.prepare(args)
        manifest=update.prepared.read(self.root/'updated/joint_manifest.json')
        self.assertTrue(receipt['partial_source_coverage'])
        self.assertEqual(manifest['shots'][0]['source_coverage']['missing_ranges'],[dict(first_pts=1256,last_pts=1256,count=1)])
        self.assertEqual((manifest['shots'][0]['first_pts'],manifest['shots'][0]['last_pts']),(1000,1512))


if __name__=='__main__':unittest.main()
