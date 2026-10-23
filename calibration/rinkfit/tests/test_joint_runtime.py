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
            baseline = root / "baseline.json"; baseline.write_text("{}")
            cfg = root / "review.json"
            cfg.write_text(json.dumps({"schema": "rinkfit.dense6.accepted_source_anchor_review.v1",
                "baseline": restart.reference(baseline), "intermediate_policy": "diagnostic"}))
            observer = review.AcceptedSourceReview(root, root / "packet.json", restart.reference(cfg))
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

    def test_shared_stage_rejects_unrestricted_optics_release(self):
        from rinkfit_joint.shared_parameter_stage import build
        old={'engineering_support_budget_v2':{'source_packet_sha256':'packet'},'audit_expected_free_physical_blocks':[],'temporary_runtime_fixed_physical_blocks':[]}
        with self.assertRaisesRegex(ValueError,'Only'):
            build(old,{'blocks':[{'name':'rig/main/optics12','values':[0.]*12}]},['rig/main/optics12'])


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
