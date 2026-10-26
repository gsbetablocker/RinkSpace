"""Physical role reporting must preserve evidence without promoting stray logits."""
import unittest
import json
import gzip
import hashlib
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from rinkspace.vision_review import (_pixel_runs, _run_points, source_board_top_roles,
    source_shots, clip_diagnostic_shots, _canonical_json_sha256, sha256, _write_source_record,
    source_region_constraints,load_saved_source_record,compact_instances,reread_source_advertisements,
    retain_same_source_refinement_evidence, compare_confirmed_source,bootstrap_source_print_materials,
    SourceRecordMemberStream, load_committed_source_record, application_source_record,
    retain_same_shot_installation_identity)


class SourceRecordStorageTests(unittest.TestCase):
    def test_tracking_installation_alias_preserves_current_native_source_and_rejects_cuts(self):
        previous=dict(source_sha256='a'*64,time_base='1/60',shot_id='s',native_hw=[10,20],
            source_pts=7,installation_id='exact_source_camera_0')
        current=dict(previous,source_pts=8,installation_id='main_long_side')
        carried=retain_same_shot_installation_identity(current,previous)
        self.assertEqual(carried['installation_id'],'exact_source_camera_0')
        self.assertEqual(carried['source_classification_installation_id'],'main_long_side')
        self.assertEqual(carried['source_pts'],8)
        self.assertEqual(current['installation_id'],'main_long_side')
        with self.assertRaises(ValueError):
            retain_same_shot_installation_identity(dict(current,shot_id='other'),previous)
        with self.assertRaises(ValueError):
            retain_same_shot_installation_identity(dict(current,source_pts=7),previous)

    def test_application_keeps_actual_readings_and_identity_without_inference_workspaces(self):
        uv=np.array([[1.25,2.75],[4.125,5.5]],np.float64)
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        reading=dict(expected_id='B1',physical_id=None,kind='ice_paint',physical_role='painted_stroke_center_axis',
            actual_source_uv=uv,source_indices=np.array([1,3],np.int64),
            coherent_support=dict(normal_interval_px=np.array([[-2.,2.],[-3.,3.]],np.float32)),
            measured_runs=[dict(actual_source_uv=uv,source_indices=np.array([1,3],np.int64))],
            metadata=dict(source_point_count=5),flags=dict(blur=np.array([False,True,False,False,False])),
            source_peak_candidates=dict(profile=np.ones((5,1000),np.float32)))
        row=dict(source_binding=binding,RGB_sha256='b'*64,model_sha256='c'*64,
            named_readings=[reading],source_components=[dict(mask=np.ones((100,100),bool))],
            source_board_foot_material_reading=dict(original_RGB_profiles=np.ones((20,1000))),issues=[dict(reasons=['unconfirmed'])])
        compact=application_source_record(row)
        self.assertEqual(compact['measurement_groups'][0]['physical_id'],None)
        self.assertEqual(compact['measurement_groups'][0]['expected_id'],'B1')
        self.assertEqual(compact['measurement_groups'][0]['actual_source_uv'].dtype,np.float64)
        self.assertEqual(compact['measurement_groups'][0]['actual_source_uv'].tobytes(),uv.tobytes())
        np.testing.assert_array_equal(compact['measurement_groups'][0]['source_flags']['blur'],[True,False])
        self.assertNotIn('source_components',compact)
        self.assertNotIn('source_board_foot_material_reading',compact)
        self.assertEqual(compact['issues'],row['issues'])
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'readings.jsonl.gz'
            with SourceRecordMemberStream(path)as records:
                _write_source_record(records,compact,hashlib.sha256(),array_sidecar_path=Path(directory)/'actual.npz',
                    minimum_array_bytes=1,storage_layout='byte_pool')
            entry=json.loads(path.with_name('readings_committed_frames.jsonl').read_text())
            restored=load_committed_source_record(path,entry)
            self.assertEqual(restored['measurement_groups'][0]['actual_source_uv'].tobytes(),uv.tobytes())
            self.assertEqual(restored['measurement_groups'][0]['actual_source_uv'].dtype,np.float64)

    def test_closed_frame_members_are_readable_before_programme_stream_closes(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        actual=np.array([[1.25,2.5],[3.75,4.125]],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'readings.jsonl.gz'
            digest=hashlib.sha256()
            with SourceRecordMemberStream(path) as records:
                first=dict(source_binding=binding,actual_source_uv=actual,source_missing_mask=np.array([False,True]))
                _write_source_record(records,first,digest,array_sidecar_path=Path(directory)/'source_7_arrays.npz',
                    minimum_array_bytes=1,storage_layout='byte_pool')
                index=path.with_name('readings_committed_frames.jsonl')
                entry=json.loads(index.read_text(encoding='utf-8'))
                recovered=load_committed_source_record(path,entry,expected_source_binding=binding)
                np.testing.assert_array_equal(recovered['actual_source_uv'],actual)
                self.assertEqual(recovered['actual_source_uv'].dtype,np.float32)
                later=dict(source_binding=dict(binding,source_pts=8),missing_reason='source occluded',
                    actual_source_uv=np.empty((0,2),np.float32))
                _write_source_record(records,later,digest,array_sidecar_path=Path(directory)/'source_8_arrays.npz',
                    minimum_array_bytes=1,storage_layout='byte_pool')
                entries=[json.loads(line) for line in index.read_text(encoding='utf-8').splitlines()]
                self.assertEqual([row['frame_ordinal'] for row in entries],[1,2])
                self.assertEqual(entries[1]['record_member_offset'],entry['record_member_compressed_bytes'])
                final=load_committed_source_record(path,entries[1])
                self.assertEqual(final['actual_source_uv'].shape,(0,2))
                self.assertEqual(final['actual_source_uv'].dtype,np.float32)
                with gzip.open(path,'rt',encoding='utf-8') as legacy_reader:
                    self.assertEqual([json.loads(line)['source_binding']['source_pts'] for line in legacy_reader],[7,8])
                changed=dict(entry,record_member_sha256='0'*64)
                with self.assertRaisesRegex(ValueError,'changed or is incomplete'):
                    load_committed_source_record(path,changed)
                with self.assertRaises(ValueError):
                    load_committed_source_record(path,entry,expected_source_binding=dict(binding,source_pts=8))
            decoded=gzip.decompress(path.read_bytes())
            self.assertEqual(hashlib.sha256(decoded).hexdigest(),digest.hexdigest())

    def test_writer_byte_pool_preserves_actual_UV_empty_arrays_and_nonfinite_source_values(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        uv=np.array([[1.25,2.5],[3.75,4.125]],np.float64)
        original=dict(source_binding=binding,actual_source_uv=uv,
            empty_source_intervals=np.empty((0,2,2),np.float32),missing=np.array([False,True]),
            source_signed_RGB=np.array([np.nan,np.inf,-np.inf],np.float32),unresolved_python_score=float('nan'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as records:
                _write_source_record(records,original,hashlib.sha256(),array_sidecar_path=Path(directory)/'source.npz',
                    minimum_array_bytes=1,storage_layout='byte_pool')
            recovered=load_saved_source_record(path,7,expected_source_binding=binding)
            with np.load(Path(directory)/'source.npz',allow_pickle=False)as archive:
                self.assertEqual(len(archive.files),1,'Small evidence arrays share one lossless byte pool')
        for key in('actual_source_uv','empty_source_intervals','missing','source_signed_RGB'):
            self.assertEqual(recovered[key].dtype,original[key].dtype)
            self.assertEqual(recovered[key].shape,original[key].shape)
            self.assertEqual(recovered[key].tobytes(),original[key].tobytes())
        self.assertTrue(np.isnan(recovered['unresolved_python_score']))

    def test_material_navigation_refines_same_exposure_and_rereads_original_RGB(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60',native_hw=[4,5],shot_id='s')
        rgb=np.zeros((4,5,3),np.uint8)
        mask=np.ones((4,5),bool)
        navigation=dict(source_binding=binding,search_mask=mask,mask_is_navigation_only=True,RGB_hard_veto_allowed=False)
        known=dict(object_id=2000,kind='advertisement_panel',source_binding=binding)
        original=dict(source_binding=binding,memory_update='shot_reset',instances=[known],
            source_person_reidentification=dict(source_binding=binding,current_source_boxes=[]))
        prompt=dict(object_id=300000,kind='advertisement_panel',source_binding=binding,
            source_material_proposal_unverified=True)
        refined=dict(source_binding=binding,memory_update='source_frame_prompt_refinement',instances=[known,prompt])
        provider=SimpleNamespace(frame_index=0,read=lambda *args,**kwargs:refined)
        actual_uv=np.array([[2.25,3.5]],np.float32)
        selected_evidence=dict(gray_history=dict(path="selected",sha256="b"*64))
        def reread(source,packet,instance_provider,*,near_board_evidence):
            self.assertIs(near_board_evidence,selected_evidence)
            self.assertIs(source,rgb)
            packet['instances'][-1]=dict(packet['instances'][-1],actual_source_uv=actual_uv)
            return packet
        with patch('rinkspace.vision_instances.source_print_material_search_proposals',return_value=dict(
            source_binding=binding,prompts=[prompt],material_candidates=[]))as proposal,patch(
            'rinkspace.vision_review.reread_source_advertisements',side_effect=reread):
            result,evidence=bootstrap_source_print_materials(rgb,original,
                dict(source_board_material_search_navigation=navigation),provider,near_board_evidence=selected_evidence)
        self.assertIs(proposal.call_args.kwargs['board_search'],navigation)
        self.assertEqual(result['initial_source_read_memory_update'],'shot_reset')
        self.assertEqual(result['source_person_reidentification'],original['source_person_reidentification'])
        self.assertEqual(evidence['SAM_source_frame_index_after'],0)
        np.testing.assert_array_equal(result['instances'][-1]['actual_source_uv'],actual_uv)
        self.assertFalse(evidence['navigation_boxes_or_masks_used_as_measured_UV'])
        self.assertEqual(len(original['instances']),1,'Refinement cannot erase the pre-refinement evidence')

    def test_same_PTS_material_navigation_cannot_advance_source_index_or_hard_veto(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60',native_hw=[4,5],shot_id='s')
        original=dict(source_binding=binding,instances=[])
        navigation=dict(source_binding=binding,search_mask=np.ones((4,5),bool),
            mask_is_navigation_only=True,RGB_hard_veto_allowed=False)
        provider=SimpleNamespace(frame_index=0)
        def read(*args,**kwargs):
            provider.frame_index+=1
            return original
        provider.read=read
        with patch('rinkspace.vision_instances.source_print_material_search_proposals',return_value=dict(prompts=[dict(
            object_id=300000,kind='advertisement_panel',source_binding=binding,source_material_proposal_unverified=True)])):
            with self.assertRaisesRegex(ValueError,'exposure index'):
                bootstrap_source_print_materials(np.zeros((4,5,3),np.uint8),original,
                    dict(source_board_material_search_navigation=navigation),provider)
        with self.assertRaisesRegex(ValueError,'navigation-only'):
            bootstrap_source_print_materials(np.zeros((4,5,3),np.uint8),original,
                dict(source_board_material_search_navigation=dict(navigation,RGB_hard_veto_allowed=True)),provider)

    def test_human_comparison_reads_saved_subpixel_upper_and_legacy_pixel_runs(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        sample=dict(source_pts=7,source_hash='a'*64,time_base='1/60',annotations=[dict(
            state='user_confirmed_visual_supervision',kind='board_top',uv=[[4.25,5.75]],
            feature_id='upper',group_id='visible_upper',authority={'kind':'direct_user_drawing'})])
        primary_packets=[dict(actual_source_uv=np.array([[4.25,5.75]],np.float64),
            source_pixel_runs_y_x_length=[]),dict(source_pixel_runs_y_x_length=[[6,4,1]])]
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for index,primary in enumerate(primary_packets):
                row=dict(source_binding=binding,RGB_sha256='b'*64,model_sha256=str(index),
                    source_components=[dict(kind='board_top',source_pixel_runs_y_x_length=[[20,4,1]]),
                        dict(kind='ice_contact',source_pixel_runs_y_x_length=[[6,4,1]])],
                    source_board_top_role_assignment=dict(primary_source_top=primary),source_region_rejections={})
                path=Path(directory)/f'record_{index}.jsonl.gz'
                with gzip.open(path,'wt',encoding='utf-8') as records:
                    _write_source_record(records,row,hashlib.sha256(),
                        array_sidecar_path=Path(directory)/f'arrays_{index}.npz',minimum_array_bytes=1)
                paths.append(path)
            comparison=compare_confirmed_source(paths,sample)
        for model in comparison['models']:
            group=model['groups'][0]
            self.assertEqual(group['primary_same_board_structure_top_coverage_by_native_px']['2'],1.)
            self.assertEqual(group['correct_class_coverage_by_native_px']['2'],0.)
            self.assertLess(group['primary_same_board_structure_top_distance_px_p50'],1.)
        self.assertEqual(comparison['models'][0]['groups'][0]['primary_same_board_structure_top_distance_px_p50'],0.)
        self.assertFalse(comparison['full_visual_accuracy_accepted'])

    def test_complete_provider_measurements_preserve_small_and_empty_typed_evidence(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        uv=np.array([[1.125,2.25],[3.5,4.75]],np.float64)
        indices=np.array([11,14],np.int32)
        reading=dict(actual_source_uv=uv,source_indices=indices,
            measured_runs=[dict(actual_source_uv=uv,source_indices=indices,closed=False)],
            flags=dict(wide_source_material_RGB_interval_retained=np.array([True,False])),
            observed_RGB_transition_intervals=dict(actual_RGB_interval_uv=np.array([[[1.,2.],[1.,3.]]],np.float32)),
            observed_RGB_signed_peak_pool=dict(actual_source_peak_uv=np.empty((0,2),np.float32),
                source_indices=np.empty(0,np.int64)),physical_boundary_verified=False)
        packet=dict(source_binding=binding,instances=[dict(object_id=1,kind='advertisement_panel',
            boundary_readings=[reading],source_print_raw_boundary_readings=[reading],
            source_print_role_conflict_readings=[reading],region_mask=np.ones((4,5),bool),
            source_pattern_material_component_links=[dict(source_component_id='blue_material')])])
        row=dict(source_binding=binding,instances=compact_instances(packet))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as records:
                _write_source_record(records,row,hashlib.sha256(),
                    array_sidecar_path=Path(directory)/'source_7.npz',minimum_array_bytes=1)
            restored=load_saved_source_record(path,7,expected_source_binding=binding)
        for collection in ('boundary_readings','source_print_raw_boundary_readings','source_print_role_conflict_readings'):
            result=restored['instances'][0][collection][0]
            self.assertEqual(result['actual_source_uv'].dtype,uv.dtype)
            self.assertEqual(result['actual_source_uv'].tobytes(),uv.tobytes())
            self.assertEqual(result['source_measured_runs'][0]['source_indices'].dtype,indices.dtype)
            self.assertEqual(result['source_measured_runs'][0]['source_indices'].tobytes(),indices.tobytes())
            empty=result['observed_RGB_signed_peak_pool']['actual_source_peak_uv']
            self.assertEqual(empty.shape,(0,2));self.assertEqual(empty.dtype,np.dtype('float32'))
        self.assertIsInstance(reading['measured_runs'][0],dict,'Compaction must not mutate provider data')

    def test_legacy_inline_provider_UV_does_not_acquire_an_invented_dtype(self):
        packet=dict(instances=[dict(kind='advertisement_panel',boundary_readings=[dict(
            actual_source_uv=[[1.,2.]],measured_runs=[dict(actual_source_uv=[[1.,2.]])])])])
        result=compact_instances(packet)[0]['boundary_readings'][0]
        self.assertIsInstance(result['actual_source_uv'],list)
        self.assertIsInstance(result['measured_runs'][0],list)

    def test_refined_same_native_exposure_keeps_person_refresh_and_initial_reset_evidence(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        previous=dict(source_binding=binding,memory_update='shot_reset',
            source_person_reidentification=dict(source_binding=binding,current_source_boxes=[dict(box_xyxy=[1,2,3,4])]),
            bootstrap_source_proposals=dict(source_binding=binding),instances=[dict(object_id=10)])
        refined=dict(source_binding=binding,memory_update='source_frame_prompt_refinement',instances=[dict(object_id=20)])
        result=retain_same_source_refinement_evidence(previous,refined)
        self.assertEqual(result['source_person_reidentification'],previous['source_person_reidentification'])
        self.assertEqual(result['instances'],refined['instances'])
        self.assertEqual(result['initial_source_read_memory_update'],'shot_reset')
        with self.assertRaisesRegex(ValueError,'another native exposure'):
            retain_same_source_refinement_evidence(previous,dict(refined,source_binding=dict(binding,source_pts=8)))

    def test_false_provider_reread_routes_unfiltered_contours_to_actual_source_reader(self):
        binding=dict(source_sha256='a'*64,source_pts=7,time_base='1/60')
        raw=[dict(uv=np.array([[50.,50.]]),role='unfiltered_source_search')]
        packet=dict(source_binding=binding,instances=[dict(kind='advertisement_panel',
            printed_pattern_contours=[],source_print_role_conflict_contours=[],source_print_unfiltered_contours=raw)])
        measured=dict(actual_source_uv=np.array([[4.25,5.75]],np.float32),measured_runs=[],boundary_role='source_RGB_candidate')
        provider=SimpleNamespace(reread_rgb=False,device='cuda',search_radius_px=12.,printed_contour_search_radius_px=3.)
        with patch('rinkspace.vision_instances.reread_region_edges',side_effect=[[],[measured]]) as reader:
            result=reread_source_advertisements(np.zeros((60,60,3),np.uint8),packet,provider)
        self.assertIs(reader.call_args_list[-1].args[1],raw)
        np.testing.assert_array_equal(result['instances'][0]['source_print_raw_boundary_readings'][0]['actual_source_uv'],measured['actual_source_uv'])
        self.assertFalse(result['instances'][0]['source_print_unfiltered_contours_are_measured_UV'])

    def test_large_typed_arrays_roundtrip_source_bound_while_legacy_stays_readable(self):
        from rinkspace.vision_storage import ARCHIVE_FIELD
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);path=directory/'readings.jsonl.gz'
            binding=dict(source_sha256='a'*64,source_pts=1,time_base='1/1000')
            large=np.arange(16384,dtype=np.float32).reshape(128,128)
            large[0,:3]=[np.nan,np.inf,-np.inf]
            row=dict(source_binding=binding,profile=large,repeated_profile=large.copy(),
                actual_source_uv=[[1.,2.]],missing=['未读到'],conflict=dict(source_points=[1,4]))
            metadata={};digest=hashlib.sha256()
            with gzip.open(path,'wt',encoding='utf-8',newline='\n') as records:
                count=_write_source_record(records,row,digest,
                    array_sidecar_path=directory/'source_1_arrays.npz',storage_metadata=metadata)
                _write_source_record(records,dict(source_binding=dict(binding,source_pts=2),profile=[[3.,4.]]),digest)
            payload=gzip.decompress(path.read_bytes());packed=json.loads(payload.splitlines()[0])
            self.assertIn(ARCHIVE_FIELD,packed)
            self.assertEqual(count,len(payload.splitlines(keepends=True)[0]))
            self.assertEqual(digest.hexdigest(),hashlib.sha256(payload).hexdigest())
            self.assertEqual(metadata['sidecar_count'],1)
            self.assertEqual(metadata['duplicate_array_references'],1)
            restored=load_saved_source_record(path,1,expected_source_binding=binding)
            np.testing.assert_array_equal(restored['profile'],large)
            self.assertEqual(restored['profile'].dtype,large.dtype)
            self.assertEqual(restored['profile'].tobytes(),large.tobytes())
            self.assertEqual(restored['missing'],row['missing'])
            self.assertEqual(restored['conflict'],row['conflict'])
            self.assertIs(restored['profile'],restored['repeated_profile'])
            self.assertEqual(load_saved_source_record(path,2)['profile'],[[3.,4.]])
            with self.assertRaisesRegex(ValueError,'requested source frame'):
                load_saved_source_record(path,1,expected_source_binding=dict(binding,source_pts=2))
            with self.assertRaises(FileExistsError):
                with gzip.open(directory/'other.jsonl.gz','wt',encoding='utf-8') as records:
                    _write_source_record(records,row,hashlib.sha256(),array_sidecar_path=directory/'source_1_arrays.npz')

    def test_record_hash_matches_actual_decompressed_UTF8_bytes_including_LF(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"readings.jsonl.gz"
            content_hash = hashlib.sha256()
            row = dict(actual_source_uv=np.array([[1., 2.], [3., np.nan]]),
                       missing=["广告缺测"], conflict=dict(source_points=[1, 4, 7]))
            with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=1) as records:
                count = _write_source_record(records, row, content_hash)
            payload = gzip.decompress(path.read_bytes())
            self.assertEqual(content_hash.hexdigest(), hashlib.sha256(payload).hexdigest())
            self.assertEqual(count, len(payload))
            self.assertTrue(payload.endswith(b"\n"))
            self.assertFalse(payload.endswith(b"\r\n"))
            stored = json.loads(payload)
            self.assertEqual(stored["actual_source_uv"], [[1., 2.], [3., None]])
            self.assertEqual(stored["missing"], row["missing"])
            self.assertEqual(stored["conflict"], row["conflict"])


class SourceBoardRegionRoleTests(unittest.TestCase):
    def test_named_SAM_print_region_does_not_hard_veto_contact_when_structure_unverified(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("native source region masks run on CUDA")
        ice = torch.zeros((40, 60), device="cuda", dtype=torch.bool)
        ice[25:] = True
        print_region = torch.zeros_like(ice)
        print_region[5:12, 10:24] = True
        packet = dict(source_binding=dict(source_pts=1), instances=[
            dict(kind="ice_region", region_mask=ice),
            dict(kind="board_region", region_mask=print_region)])
        candidate = source_region_constraints(packet, native_hw=(40, 60), device="cuda", radius_px=3)
        self.assertNotIn("ice_contact", candidate["masks"])
        self.assertNotIn("ad_boundary", candidate["masks"])
        self.assertIn("ice_contact", candidate["soft_search_masks"])
        self.assertFalse(candidate["metadata"]["model_prompt_region_name_establishes_physical_structure"])
        self.assertTrue(candidate["metadata"]["unverified_board_region_hard_veto_disabled"])
        self.assertTrue(candidate["missing_roles"])
        packet["same_board_structure_verified"] = True
        verified = source_region_constraints(packet, native_hw=(40, 60), device="cuda", radius_px=3)
        self.assertIn("ice_contact", verified["masks"])
        self.assertIn("ad_boundary", verified["masks"])


class NativeBoundaryOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.timeline = self.directory / "timeline.json"
        self.overlay_path = self.directory / "overlay.json"
        self.digest = "a"*64
        self.rows = [dict(id=name, start_pts=index*10, end_pts=(index+1)*10,
            interval_type="ignored" if index == 2 else "shot",
            camera_id=None if index == 2 else "main" if index == 0 else "low_end",
            authority="user_confirmation" if index == 0 else "assistant_review",
            review_status="source_review") for index, name in enumerate(("a", "b", "c"))]
        self.timeline.write_text(json.dumps(dict(source_sha256=self.digest, time_base="1/1000",
                                                 intervals=self.rows)), encoding="utf-8")
        changes = [(0, 9), (9, 19), (19, 30)]
        overrides = []
        for original, (start, end) in zip(self.rows, changes):
            new = dict(original, start_pts=start, end_pts=end)
            overrides.append(dict(shot_id=original["id"], original_start_pts=original["start_pts"],
                original_end_pts_exclusive=original["end_pts"], start_pts=start, end_pts_exclusive=end,
                camera_id=original["camera_id"], camera_authority=original["authority"],
                camera_review_status=original["review_status"], original_row=original, corrected_row=new,
                original_row_json_sha256=_canonical_json_sha256(original),
                corrected_row_json_sha256=_canonical_json_sha256(new)))
        self.overlay = dict(schema="rinkspace.native-shot-boundary-correction.v1",
            source_sha256=self.digest, time_base="1/1000", original_timeline_sha256=sha256(self.timeline),
            shot_overrides=overrides)

    def read(self):
        self.overlay_path.write_text(json.dumps(self.overlay), encoding="utf-8")
        return source_shots(self.timeline, self.digest, boundary_overlay_path=self.overlay_path)

    def test_atomic_adjacent_boundary_preserves_assistant_and_ignored_taxonomy(self):
        shots = self.read()
        self.assertEqual([(r["start_pts"], r["end_pts_exclusive"]) for r in shots], [(0, 9), (9, 19)])
        self.assertEqual(shots[1]["camera_authority"], "assistant_review")
        self.assertEqual(shots[0]["source_boundary_overlay"]["ranges"][-1]["camera_id"], None)
        clipped = clip_diagnostic_shots(shots, 8, 11)
        self.assertEqual([(r["start_pts"], r["end_pts_exclusive"]) for r in clipped], [(8, 9), (9, 11)])

    def test_wrong_video_timeline_and_original_range_are_rejected(self):
        for field, value in (("source_sha256", "b"*64), ("original_timeline_sha256", "b"*64)):
            prior = self.overlay[field]
            self.overlay[field] = value
            with self.assertRaises(ValueError):
                self.read()
            self.overlay[field] = prior
        self.overlay["shot_overrides"][1]["original_start_pts"] = 11
        with self.assertRaisesRegex(ValueError, "original row/range"):
            self.read()

    def test_one_sided_cut_or_promoted_camera_authority_is_rejected(self):
        prior = self.overlay["shot_overrides"]
        self.overlay["shot_overrides"] = prior[:-1]
        with self.assertRaisesRegex(ValueError, "adjacent"):
            self.read()
        self.overlay["shot_overrides"] = prior
        correction = prior[1]
        correction["camera_authority"] = "user_confirmation"
        correction["corrected_row"]["authority"] = "user_confirmation"
        correction["corrected_row_json_sha256"] = _canonical_json_sha256(correction["corrected_row"])
        with self.assertRaisesRegex(ValueError, "taxonomy or authority"):
            self.read()


class SourceBoardRoleTests(unittest.TestCase):
    def test_missing_bootstrap_pair_reads_RGB_upper_material_entry_and_renders_exit_as_internal(self):
        from rinkspace.vision_review import draw_review
        rgb=np.full((140,200,3),160,np.uint8)
        rgb[42:108]=225;rgb[30:42]=[45,65,130];rgb[70:95,40:160]=[60,100,180]
        binding=dict(source_sha256='a'*64,source_pts=8,time_base='1/15360',native_hw=[140,200],shot_id='synthetic_source')
        x=np.arange(200)
        components=[dict(typed_class_candidate='board_top',candidate_kind='straight_edge_fragment',
            source_support_uv=np.column_stack((x,np.full(200,42))))]
        foot=dict(source_binding=binding,readings=[dict(physical_role='ice_contact_lower_material_boundary',
            primary_source_uv=np.column_stack((np.arange(0,200,4),np.full(50,116))).astype(np.float32))])
        printed=np.concatenate((np.column_stack((np.arange(40,160),np.full(120,70))),
            np.column_stack((np.arange(40,160),np.full(120,94)))))
        ads=[dict(kind='advertisement_panel',object_id=2,source_binding=binding,
            boundary_readings=[dict(boundary_role='printed_pattern_external_contour',actual_source_uv=printed)])]
        roles=source_board_top_roles(rgb,components,dict(board_contact_pair_hypotheses=[]),
            source_binding=binding,foot_material_packet=foot,source_instances=ads)
        self.assertEqual(roles['status'],'source_RGB_upper_rail_entry_and_internal_exit_separated')
        material=roles['actual_source_rail_material_reading']
        self.assertTrue((material['actual_source_blue_rail_entry_uv'][:,1]<material['actual_source_blue_to_white_exit_uv'][:,1]).all())
        self.assertTrue(roles['all_original_typed_board_top_pixels_retained'])
        self.assertFalse(roles['navigation_fit_pixels_used_as_output'])
        self.assertEqual(roles['blue_to_white_exit_role'],'board_internal_material_interface_not_board_upper_rim')
        result=dict(source_binding=binding,source=dict(components=components),guided=dict(readings=[]))
        canvas=draw_review(rgb,result,dict(source_board_top_role_assignment=roles,issues=[]),ads)
        entry=roles['primary_source_top']['actual_source_uv'][10]
        exit=material['actual_source_blue_to_white_exit_uv'][10]
        entry_xy=np.rint(entry).astype(int);exit_xy=np.rint(exit).astype(int)
        self.assertEqual(tuple(canvas[entry_xy[1],entry_xy[0]]),(255,235,40))
        self.assertEqual(tuple(canvas[exit_xy[1],exit_xy[0]]),(170,150,210))

    def test_original_rgb_rail_and_unrelated_wall_have_separate_roles(self):
        rgb = np.full((108, 192, 3), 70, np.uint8)
        rgb[18:] = 125
        rgb[40:] = 230
        rail = np.column_stack((np.arange(192), np.full(192, 40)))
        wall = np.column_stack((np.arange(192), np.full(192, 18)))
        actual = np.concatenate((rail, wall))
        components = [dict(typed_class_candidate="board_top", source_support_uv=actual)]
        proposal = dict(board_contact_pair_hypotheses=[dict(candidate_pair_score=1.,
            top_hypothesis=dict(preferred_region_search_hypothesis=True,
                source_bbox_xyxy=[128, 40, 192, 41],
                source_gradient_pixel_runs_y_x_length=_pixel_runs(rail[128:])),
            contact_hypothesis=dict(mean_source_y=60), board_print_source_gradient_pixels=20)])
        roles = source_board_top_roles(rgb, components, proposal)
        np.testing.assert_array_equal(_run_points(roles["primary_source_top"]["source_pixel_runs_y_x_length"]), rail)
        np.testing.assert_array_equal(_run_points(roles["unresolved_typed_edges"]["source_pixel_runs_y_x_length"]), wall)
        self.assertFalse(roles["navigation_fit_pixels_used_as_output"])
        self.assertFalse(roles["physical_structure_accuracy_verified"])

    def test_missing_same_board_pair_keeps_all_actual_edges_unresolved(self):
        rgb = np.full((108, 192, 3), 80, np.uint8)
        actual = np.array([[3, 4], [4, 4], [5, 4]])
        roles = source_board_top_roles(rgb, [dict(typed_class_candidate="board_top", source_support_uv=actual)], {})
        self.assertEqual(roles["primary_source_top"]["source_pixel_count"], 0)
        np.testing.assert_array_equal(_run_points(roles["unresolved_typed_edges"]["source_pixel_runs_y_x_length"]), actual)
        self.assertEqual(roles["status"], "missing_same_contact_board_source_pair")


if __name__ == "__main__":
    unittest.main()
