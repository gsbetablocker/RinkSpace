"""Boundary tests for observed restarts and intermediate-review policy."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from rinkfit_joint import observed_primal_restart as restart
from rinkfit_joint import dense_live_review as review
from rinkfit_joint import capture_restart


def report(invalid=0):
    return {
        "same_source_comparison": {"by_shot": {"s02": {
            "newly_invalid": invalid, "baseline_true_RS": {"rms_px": 6, "p95_px": 12},
            "candidate_true_RS": {"rms_px": 8, "p95_px": 16},
            "sum_squared_error_delta_on_common_valid": 10,
        }}},
        "worst_worsened_subject_by_entity": [{"shot_id": "s02", "feature_id": "Ad7.top",
            "candidate_error_px": 22, "baseline_error_px": 6, "original_E": 2}],
    }


class PolicyTests(unittest.TestCase):
    def bound_review(self, root):
        packet=root / "packet.json"; packet.write_text("{}")
        distances=root / "distances.npz"; distances.write_bytes(b"immutable residual fixture")
        baseline=root / "baseline.json"
        baseline.write_text(json.dumps({
            "schema":"rinkfit.full_existing_fixed2D_source_anchor_guard.v1",
            "packet":review.reference(packet),
            "implementation":review.reference(Path(review.__file__).with_name("dense_fixed_anchor_guard.py")),
            "source_visibility_operator":review.reference(Path(review.__file__).with_name("joint_replay.py")),
            "rows":3,"cohort_row_ID_sha256":"all-original-row-IDs",
            "distances":review.reference(distances)}))
        cfg=root / "review.json"
        cfg.write_text(json.dumps({"schema":"rinkfit.dense6.accepted_source_anchor_review.v1",
            "baseline":review.reference(baseline),"intermediate_policy":"diagnostic"}))
        return packet,baseline,cfg

    def test_stale_packet_baseline_rejected_before_any_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);packet,baseline,cfg=self.bound_review(root)
            self.assertEqual(review.validate_anchor_review_binding(review.reference(cfg),packet)["intermediate_policy"],"diagnostic")
            packet.write_text('{"new_generation":true}')
            with self.assertRaisesRegex(ValueError,"current packet before launch"):
                review.AcceptedSourceReview(root,packet,review.reference(cfg))
            self.assertFalse((root / "STOP_REQUESTED").exists())

    def test_changed_operator_and_guard_rejected_before_launch(self):
        for key in ("source_visibility_operator","implementation"):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);packet,baseline,cfg=self.bound_review(root)
                data=json.loads(baseline.read_text());data[key]["sha256"]="previous-operator"
                baseline.write_text(json.dumps(data))
                configuration=json.loads(cfg.read_text());configuration["baseline"]=review.reference(baseline)
                cfg.write_text(json.dumps(configuration))
                with self.assertRaisesRegex(ValueError,"current "+key+" before launch"):
                    review.AcceptedSourceReview(root,packet,review.reference(cfg))

    def test_extended_real_invalid_rows_do_not_crash_on_validity_improvement(self):
        value=report();comparison=value["same_source_comparison"]["by_shot"]["s02"]
        comparison["baseline_true_RS"]["rms_px"]="+Infinity"
        result=review.decide_anchor_review(value,"strict")
        self.assertFalse(result["request_stop"])
        comparison["candidate_true_RS"]["rms_px"]="+Infinity"
        comparison["newly_invalid"]=1
        self.assertTrue(review.decide_anchor_review(value,"strict")["request_stop"])

    def test_transient_quality_regression_is_recorded_without_stopping(self):
        value = review.decide_anchor_review(report(), "diagnostic")
        self.assertTrue(value["requires_source_or_geometry_review"])
        self.assertFalse(value["request_stop"])
        self.assertFalse(value["quality_accepted"])
        self.assertTrue(value["final_quality_gate_unchanged"])

    def test_existing_strict_policy_remains_explicitly_available(self):
        self.assertTrue(review.decide_anchor_review(report())["request_stop"])

    def test_new_true_rs_invalidity_not_hidden_by_diagnostic_mode(self):
        value = review.decide_anchor_review(report(3), "diagnostic")
        self.assertEqual(value["reasons"][0]["rows"], 3)
        self.assertFalse(value["quality_accepted"])

    def test_invalid_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            review.decide_anchor_review(report(), "off")

    def test_nonfinite_candidate_raises_and_user_stop_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "working_physical").mkdir()
            packet, baseline, cfg = self.bound_review(root)
            observer = review.AcceptedSourceReview(root, packet, restart.reference(cfg))
            model = root / "working_physical/model.json"
            model.write_text('{"blocks":[{"values":[NaN]}]}')
            (model.parent / "receipt.json").write_text(json.dumps({"state_kind": "physical_working_candidate",
                "complete_checkpoint": False, "global_iteration": 1, "model_sha256": restart.sha256(model)}))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                observer.poll()
            (root / "STOP_REQUESTED").write_text("USER STOP")
            observer.fail_closed(ValueError("nonfinite"))
            self.assertEqual((root / "STOP_REQUESTED").read_text(), "USER STOP")
            self.assertTrue((root / "accepted_source_review/error.json").is_file())


class RestartTests(unittest.TestCase):
    def test_true_rs_and_reseed_cannot_consume_observed_state(self):
        for cfg in ({"source_uv_capture_objective": "rinkfit.sourceUV_trueRS_finite_capture.v3"},
                    {"source_uv_capture_objective": restart.OBJECTIVE,
                     "dense_source_fresh_capture_v1": True, "reinitialize_native_auxiliaries": True}):
            with self.assertRaises(ValueError):
                restart.validate_restart_manifest(cfg, {}, {})

    def test_gauge_and_weights_are_semantic_changes(self):
        first = {"temporary_runtime_fixed_physical_blocks": ["a"], "num_threads": 8,
                 "engineering_dense_source_weights_v1": {"sha256": "old"}}
        second = dict(first, num_threads=12)
        self.assertEqual(restart.semantic_config(first), restart.semantic_config(second))
        self.assertNotEqual(restart.semantic_config(first),
                            restart.semantic_config(dict(second, temporary_runtime_fixed_physical_blocks=["b"])))
        self.assertNotEqual(restart.semantic_config(first),
                            restart.semantic_config(dict(second, engineering_dense_source_weights_v1={"sha256": "new"})))

    def test_only_explicit_atlas_delta_is_allowed_as_new_problem(self):
        first = {"pixel_total_weight": 2.0, "runtime_atlas_gauge_v1_bad": False,
                 "audit_expected_free_physical_blocks": ["old"]}
        second = dict(first, runtime_atlas_gauge_v1={"schema": "rinkfit.runtime_atlas_gauge.v1"},
                      audit_expected_free_physical_blocks=["new"], joint6_atlas_gauge_transition={"source": "receipt"})
        restart.validate_problem_change(first, second, True, "runtime_atlas_coordinate_gauge_only")
        with self.assertRaisesRegex(ValueError, "differs"):
            restart.validate_problem_change(first, second, False)
        with self.assertRaisesRegex(ValueError, "differs"):
            restart.validate_problem_change(first, dict(second, pixel_total_weight=3.0), True,
                                            "runtime_atlas_coordinate_gauge_only")
        with self.assertRaisesRegex(ValueError, "first explicit"):
            restart.validate_problem_change(second, second, True, "runtime_atlas_coordinate_gauge_only")
        restart.validate_problem_change(second, dict(second, num_threads=8), False)

    def test_auxiliary_scheduling_can_resume_without_changing_the_problem(self):
        cfg = {"source_uv_capture_objective": restart.OBJECTIVE,
               "pixel_total_weight": 2.0,
               "engineering_dense_source_weights_v1": {"sha256": "weights"},
               "dense_source_qualification": {"sha256": "source"},
               "capture_certified_numeric_domain": {"sha256": "domain"},
               "temporary_runtime_fixed_physical_blocks": ["protected"]}
        flags = ("defer_complete_continuation_auxiliary_v1",
                 "defer_auxiliary_minimization_to_joint_phase_v1")
        deferred = dict(cfg, **dict.fromkeys(flags, True))
        immediate = dict(cfg, **dict.fromkeys(flags, False))
        restart.validate_problem_change(deferred, immediate, False)
        restart.validate_problem_change(cfg, immediate, False)
        for key, value in (("pixel_total_weight", 3.0),
                           ("engineering_dense_source_weights_v1", {"sha256": "changed"}),
                           ("dense_source_qualification", {"sha256": "changed"}),
                           ("capture_certified_numeric_domain", {"sha256": "changed"}),
                           ("temporary_runtime_fixed_physical_blocks", []),
                           ("source_uv_capture_objective", capture_restart.TRUE_RS)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differs"):
                restart.validate_problem_change(deferred, dict(immediate, **{key: value}), False)

    def test_explicit_zero_solver_tolerances_are_execution_settings(self):
        cfg={"pixel_total_weight":2.0,"parameter_tolerance":1e-10,"gradient_tolerance":1e-10}
        restart.validate_problem_change(cfg,dict(cfg,parameter_tolerance=0,gradient_tolerance=0),False)
        with self.assertRaises(ValueError):
            restart.validate_problem_change(cfg,dict(cfg,parameter_tolerance=0,pixel_total_weight=3.0),False)

    def test_graph_verification_does_not_change_problem_semantics(self):
        cfg = {"pixel_total_weight": 2.0, "source_uv_capture_construct_problem_only": False}
        restart.validate_problem_change(cfg, dict(cfg, source_uv_capture_construct_problem_only=True), False)
        with self.assertRaisesRegex(ValueError, "differs"):
            restart.validate_problem_change(cfg, dict(cfg, source_uv_capture_preflight_only=True), False)

    def test_exact_python_archive_metadata_extension_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "old.py"; code.write_bytes(b"historical issuer\n")
            old = root / "aliases_old.json"
            old.write_text(json.dumps({"schema": "rinkfit.exact_dependency_aliases.v1", "aliases": [], "scope": "code only"}))
            new = root / "aliases_new.json"
            alias = {"legacy_path": str(root / "producer.py"), "actual_file": str(code), "sha256": restart.sha256(code)}
            value = {"schema": "rinkfit.exact_dependency_aliases.v1", "aliases": [alias], "scope": "code only"}
            new.write_text(json.dumps(value))
            first = {"archived_dependency_aliases": restart.reference(old), "pixel_total_weight": 2.0}
            second = dict(first, archived_dependency_aliases=restart.reference(new))
            restart.validate_problem_change(first, second, False)
            with self.assertRaisesRegex(ValueError, "differs"):
                restart.validate_problem_change(first, dict(second, pixel_total_weight=3.0), False)
            value["aliases"][0]["legacy_path"] = str(root / "packet.json")
            new.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Python code"):
                restart.validate_problem_change(first, dict(second, archived_dependency_aliases=restart.reference(new)), False)
            value["aliases"][0]["legacy_path"] = str(root / "producer.py")
            new.write_text(json.dumps(value))
            current_ref = restart.reference(new)
            code.write_text("modified issuer")
            with self.assertRaisesRegex(ValueError, "changed"):
                restart.validate_problem_change(first, dict(second, archived_dependency_aliases=current_ref), False)

    def test_overlay_packet_denominator_inherits_pinned_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root / "parent.json"; child = root / "child.json"
            parent.write_text(json.dumps({"schema": "rinkfit.joint_packet.v1", "row_count": 123}))
            child.write_text(json.dumps({"schema": "rinkfit.joint_packet.v1",
                                         "dense_parent_packet": restart.reference(parent)}))
            ref = restart.reference(child)
            self.assertEqual(restart.packet_row_count(ref), 123)
            parent.write_text(json.dumps({"schema": "rinkfit.joint_packet.v1", "row_count": 124}))
            with self.assertRaises(ValueError):
                restart.packet_row_count(ref)


class RepresentativeAncestryTests(unittest.TestCase):
    def chain(self, root, *, changed_array=False, membership=False, deleted=False, discontinuous=False):
        def put(name, value):
            path=root/name; path.write_text(json.dumps(value),encoding="utf-8")
            return restart.reference(path)
        arrays={"uv":{"sha256":"UV"},"E":{"sha256":"E"},"admission":{"sha256":"A"}}
        base=put("base.json",dict(schema="rinkfit.joint_packet.v1",row_count=3,active_rows=2,admitted_rows=2,arrays=arrays))
        packet=base; qual=put("base_qual.json",{"schema":"rinkfit.native_named_print_append_qualification.v3","source_packet":base})
        for generation in (1,2):
            row=dict(raw_row=1,source_row_id="actual/1",native_pts=256,uv=[1.,2.],old_material_index=generation-1,
                     new_material_index=generation,old_path=0,new_path=0,new_station=.5)
            if generation==2 and discontinuous: row["old_material_index"]=9
            review=put(f"review{generation}.json",dict(source_pixels_changed=False,source_model_projection_used=False,rows=[row]))
            revision=put(f"revision{generation}.json",dict(schema="rinkfit.native_print_identity_rebind.v1",parent_packet=packet,
                parent_qualification=qual,review=review,rebound_rows=[1],raw_rows=3,
                source_rows_deleted=1 if deleted and generation==2 else 0,source_UV_ID_PTS_and_membership_unchanged=True))
            fields=dict(schema="rinkfit.joint_packet.v1",dense_parent_packet=packet,row_count=3,active_rows=1 if membership and generation==2 else 2,
                        admitted_rows=2,native_print_identity_rebind=revision,arrays={})
            if changed_array and generation==2: fields["arrays"]={"uv":{"sha256":"changed"}}
            child=put(f"packet{generation}.json",fields)
            qual=put(f"qual{generation}.json",dict(schema="rinkfit.native_print_identity_rebind_qualification.v1",source_packet=child,
                    parent_packet=packet,parent_qualification=qual,revision=revision))
            packet=child
        return base,packet,revision,qual

    def test_two_generations_merge_only_reviewed_rows_from_original_to_final_identity(self):
        from rinkfit_joint.dense_joint_run import representative_identity_ancestry
        with tempfile.TemporaryDirectory() as tmp:
            base,packet,revision,qual=self.chain(Path(tmp))
            result=representative_identity_ancestry(base["sha256"],packet,revision,qual)
            self.assertEqual((len(result["generations"]),result["raw_rows"],len(result["rows"])),(2,3,1))
            self.assertEqual((result["rows"][0]["old_material_index"],result["rows"][0]["new_material_index"]),(0,2))

    def test_source_measurement_membership_delete_and_discontinuous_identity_are_rejected(self):
        from rinkfit_joint.dense_joint_run import representative_identity_ancestry
        for key in ("changed_array","membership","deleted","discontinuous"):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                base,packet,revision,qual=self.chain(Path(tmp),**{key:True})
                with self.assertRaises(ValueError): representative_identity_ancestry(base["sha256"],packet,revision,qual)

    def test_bad_hash_and_unreachable_ancestor_are_rejected(self):
        from rinkfit_joint.dense_joint_run import representative_identity_ancestry
        with tempfile.TemporaryDirectory() as tmp:
            base,packet,revision,qual=self.chain(Path(tmp))
            with self.assertRaises(ValueError): representative_identity_ancestry(base["sha256"],packet,dict(revision,sha256="0"*64),qual)
            with self.assertRaises(ValueError): representative_identity_ancestry("not-an-ancestor",packet,revision,qual)

    def mixed_chain(self, root, *, altered_uv=False, wrong_count=False, duplicate=False):
        base, packet, revision, qual = self.chain(root)
        def put(name,value):
            path=root/name;path.write_text(json.dumps(value),encoding="utf-8");return restart.reference(path)
        image=put("actual_image.json",{"original_RGB_fixture":True});extractor=put("extractor.json",{"source_only":True})
        for generation,raw in enumerate((1,1 if duplicate else 0),3):
            row=dict(raw_row=raw,source_row_id="actual/"+str(raw),native_pts=256 if raw==1 else 512,uv=[1.,2.] if raw==1 else [3.,4.],qualification="unknown",finding="Original source belongs to a different entity",native_images=[image])
            review=put(f"mixed_review{generation}.json",dict(schema="rinkfit.native_material_identity_review.v1",qualification_uses_model_residual=False,native_frame_extractor=extractor,rows=[row]))
            revision=put(f"mixed_revision{generation}.json",dict(schema="rinkfit.native_material_identity_revision.v1",parent_packet=packet,parent_qualification=qual,review=review,withheld_rows=[raw],raw_rows=3,source_rows_deleted=0,source_UV_ID_PTS_and_operators_unchanged=True))
            arrays={"admission":{"sha256":str(generation)},"fit_role":{"sha256":str(generation)}}
            if altered_uv and generation==4:arrays["uv"]={"sha256":"forbidden"}
            child=put(f"mixed_packet{generation}.json",dict(schema="rinkfit.joint_packet.v1",dense_parent_packet=packet,row_count=3,active_rows=(0 if wrong_count else 1)if generation==3 else 0,admitted_rows=1 if generation==3 else 0,native_material_identity_revision=revision,arrays=arrays))
            qual=put(f"mixed_qual{generation}.json",dict(schema="rinkfit.native_material_identity_revision_qualification.v1",source_packet=child,parent_packet=packet,parent_qualification=qual,revision=revision));packet=child
        return base,packet,revision,qual

    def test_mixed_rebind_then_two_withdrawals_preserve_rebind_and_exact_union(self):
        from rinkfit_joint.dense_joint_run import representative_identity_ancestry
        with tempfile.TemporaryDirectory() as tmp:
            base,packet,revision,qual=self.mixed_chain(Path(tmp));result=representative_identity_ancestry(base["sha256"],packet,revision,qual)
            self.assertEqual([g["kind"]for g in result["generations"]],["rebind","rebind","withdrawal","withdrawal"])
            self.assertEqual(result["withheld_rows"],[0,1]);self.assertEqual(len(result["rows"]),1)
            self.assertEqual(result["rows"][0]["new_material_index"],2)

    def test_mixed_chain_rejects_changed_uv_wrong_count_and_repeated_withdrawal(self):
        from rinkfit_joint.dense_joint_run import representative_identity_ancestry
        for key in ("altered_uv","wrong_count","duplicate"):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                base,packet,revision,qual=self.mixed_chain(Path(tmp),**{key:True})
                with self.assertRaises(ValueError):representative_identity_ancestry(base["sha256"],packet,revision,qual)

    def test_prepare_dispatch_reads_current_raw_packet_without_inherited_rebind(self):
        from unittest.mock import patch
        from rinkfit_joint import dense_joint_run as run
        class CurrentRevisionReached(RuntimeError): pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def put(name,value):
                path=root/name;path.write_text(json.dumps(value),encoding="utf-8");return restart.reference(path)
            stub=put("stub.json",{});old_revision=put("old_revision.json",{});latest=put("latest.json",{})
            parent=put("parent.json",dict(schema="rinkfit.joint_packet.v1",native_print_identity_rebind=old_revision))
            child=dict(schema="rinkfit.joint_packet.v1",dense_parent_packet=parent,native_material_identity_revision=latest)
            packet=put("packet.json",child);native=root/"native.exe";native.write_bytes(b"fixture")
            put("recipe.json",dict(executable_sha256=restart.sha256(native),source_files={}))
            saved=put("continuation.json",dict(schema="rinkfit.representative_complete_primal_continuation.v1",complete_checkpoint=True,source_packet_sha256=parent["sha256"],model=stub,checkpoint=stub,explicit_nuisance_manifest=stub,representative_source_selection_v1=stub))
            cfg=put("config.json",dict(engineering_active_shot_ids=run.SHOTS,linear_solver_backend="cuda_cgnr",gpu_raw_uv=dict(batch_reuse=True),dense_source_fresh_capture_v1=True,source_uv_capture_objective=restart.OBJECTIVE,representative_continuation_v1=saved,representative_source_withdrawal_v1=latest,dense_source_qualification=stub))
            spec=dict(output=str(root/"unused_fit"),model=stub,packet=packet,native=restart.reference(native),configuration=cfg)
            normal=run.read_json
            def expanded_reader(path):
                # io_utils expands compact packets; inherited metadata is not
                # evidence of the current generation's actual revision kind.
                if Path(path)==Path(packet["file"]):return dict(child,native_print_identity_rebind=old_revision)
                return normal(path)
            with patch.object(run,"read_json",side_effect=expanded_reader),patch.object(run,"representative_identity_ancestry",side_effect=CurrentRevisionReached)as ancestry:
                with self.assertRaises(CurrentRevisionReached):run.prepare_run(spec)
                self.assertEqual(ancestry.call_args.args[2],latest)


class StageV2Tests(unittest.TestCase):
    def test_budget_transition_is_explicit_and_does_not_allow_source_changes(self):
        old={'source_uv_capture_objective':restart.OBJECTIVE,'dense_source_qualification':{'sha256':'source'},'pixel_total_weight':9}
        new=dict(old,engineering_support_budget_v2={'schema':'rinkfit.native_exposure_material_budget.v2'})
        restart.validate_problem_change(old,new,True,'explicit_engineering_stage_v2')
        with self.assertRaises(ValueError):restart.validate_problem_change(old,new,False)
        with self.assertRaises(ValueError):restart.validate_problem_change(old,dict(new,dense_source_qualification={'sha256':'different'}),True,'explicit_engineering_stage_v2')
        with self.assertRaises(ValueError):restart.validate_problem_change(old,dict(new,temporary_runtime_fixed_physical_blocks=[]),True,'explicit_engineering_stage_v2')

    def test_existing_motion_source_set_cannot_be_replaced_during_stage_change(self):
        old={'engineering_support_budget_v2':{'total_budget':9},'static_surface_motion_v1':{'sha256':'a'}}
        with self.assertRaisesRegex(ValueError,'replace'):restart.validate_problem_change(old,dict(old,static_surface_motion_v1={'sha256':'b'}),True,'explicit_engineering_stage_v2')

    def test_true_rs_and_source_identity_remain_semantic(self):
        old={'source_uv_capture_objective':restart.OBJECTIVE,'dense_source_qualification':{'sha256':'a'}}
        self.assertNotEqual(capture_restart.comparable(old),capture_restart.comparable(dict(old,source_uv_capture_objective=capture_restart.TRUE_RS)))
        self.assertNotEqual(capture_restart.comparable(old),capture_restart.comparable(dict(old,dense_source_qualification={'sha256':'b'})))


class SharedStageTests(unittest.TestCase):
    def test_shared_stage_preserves_gauge_source_and_primal(self):
        from rinkfit_joint.shared_parameter_stage import build
        model={'blocks':[{'name':'rig/main/head15','values':[0.]*15,'fixed_indices':[3,4,5,9,10,11,12,13,14]}]}
        old={'engineering_support_budget_v2':{'source_packet_sha256':'packet'},'audit_expected_free_physical_blocks':[],
             'temporary_runtime_fixed_physical_blocks':['rig/main/head15'],'temporary_runtime_fixed_nuisance_shots':['s01']}
        new=build(old,model,['rig/main/head15'])
        self.assertEqual(new['runtime_shared_parameter_stage_v2']['blocks'][0]['free_ambient_indices'],[0,1,2,6,7,8])
        self.assertEqual(old['temporary_runtime_fixed_nuisance_shots'],['s01'])
        self.assertEqual(model['blocks'][0]['values'],[0.]*15)
        restart.validate_problem_change(old,new,True,'explicit_engineering_stage_v2')
        with self.assertRaises(ValueError):restart.validate_problem_change(old,new,False)
        with self.assertRaises(ValueError):restart.validate_problem_change(old,dict(new,dense_source_qualification={'sha256':'changed'}),True,'explicit_engineering_stage_v2')

    def test_shared_optics_preserves_structural_indices_and_requires_release_provenance(self):
        from rinkfit_joint.shared_parameter_stage import build
        old={'engineering_support_budget_v2':{'source_packet_sha256':'packet'},'audit_expected_free_physical_blocks':[],'temporary_runtime_fixed_physical_blocks':[]}
        block={'name':'rig/main/optics12','values':[0.]*12,'fixed_indices':[3,4,6,7,8]}
        model={'blocks':[block]}
        current=build(old,model,[block['name']])
        self.assertEqual(current['runtime_shared_parameter_stage_v2']['blocks'][0]['free_ambient_indices'],[0,1,2,5,9,10,11])
        with self.assertRaisesRegex(ValueError,'temporary-initialization provenance'):
            build(old,model,[block['name']],release_camera_initializers=True)
        provenance=dict(model,camera_bootstrap_rebase_v24={'all_shared_parameters_must_be_released_after_initialization':True})
        released=build(old,provenance,[block['name']],release_camera_initializers=True)
        record=released['runtime_shared_parameter_stage_v2']['blocks'][0]
        self.assertEqual(record['free_ambient_indices'],list(range(12)))
        self.assertTrue(record['release_initialization_camera_indices'])
        with self.assertRaisesRegex(ValueError,'frozen prefix'):
            build(dict(old,temporary_runtime_fixed_physical_prefixes=['rig/main/']),provenance,[block['name']],release_camera_initializers=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_true_rs_rejection_is_not_reported_as_missing_solver_receipt(tmp_path):
    from rinkfit_joint.dense_joint_run import record_initial_capture_rejection
    import json
    p=tmp_path/'rejection.json'
    p.write_text(json.dumps(dict(solver_started=False,quality_accepted=False,complete_denominator_retained=True,seeds=dict(active_rows=100,failed_RS_rows_retained=3))))
    r=record_initial_capture_rejection(dict(source_active_rows=100,missing_final_native_receipt=True,missing_native_graph_receipt=True),p)
    assert r['solver_started'] is False and r['initialization_ready'] is False
    assert r['failed_RS_rows_retained']==3 and r['requires_feasible_observed_initialization'] is True
    assert 'missing_final_native_receipt' not in r and 'missing_native_graph_receipt' not in r


def test_true_rs_rejection_cannot_drop_source_denominator(tmp_path):
    from rinkfit_joint.dense_joint_run import record_initial_capture_rejection
    import json,pytest
    p=tmp_path/'rejection.json'
    p.write_text(json.dumps(dict(solver_started=False,quality_accepted=False,complete_denominator_retained=True,seeds=dict(active_rows=99,failed_RS_rows_retained=3))))
    with pytest.raises(ValueError,match='denominator'):
        record_initial_capture_rejection(dict(source_active_rows=100),p)
