#pragma once
#include "rinkfit/event_io.hpp"
#include <cmath>
#include "rinkfit/joint_global_phr_epoch_v1.hpp"
namespace rinkfit {
// One definition for production recovery, strict restart and mode transition.
// Transport, logging and numerical execution controls are deliberately absent.
// Source/selection, actual stage and initialization transitions remain explicit.
inline eventio::Json joint_engineering_objective_configuration_v1(const eventio::Json&cfg,
 double default_pixel_total_weight=std::numeric_limits<double>::quiet_NaN()){
 using Json=eventio::Json;const double weight=cfg.contains("pixel_total_weight")?cfg.at("pixel_total_weight").get<double>():default_pixel_total_weight;
 if(!std::isfinite(weight)||weight<0)throw std::runtime_error("engineering objective identity requires a finite actual pixel budget");
 Json out={{"objective_id",cfg.value("source_uv_capture_objective",std::string("engineering_recovery_native_pixel_v1"))},{"physical_constraint_penalty",cfg.value("physical_constraint_penalty",10.)},{"pixel_total_weight",weight}};
 for(const auto*key:{"engineering_active_shot_ids","engineering_pixel_dependency_policy","visible_ice_source_constraints","source_query_branch_registry","board_current_source_qualification","board_print_source_qualification","ice_shape_source_qualification","dense_source_qualification","runtime_atlas_gauge_v1","parameter_stage","temporary_runtime_fixed_physical_prefixes","temporary_runtime_fixed_physical_blocks","temporary_runtime_fixed_nuisance_shots","source_identity_verdicts_v1","capture_zoom_feasibility_recovery_v1","legacy_line_coordinate_rechart_v1","representative_source_selection_v1","representative_full_source_transition_v1","engineering_support_budget_v2","static_surface_motion_v1","runtime_shared_parameter_stage_v2","engineering_global_phr_v1","shared_print_shape_transition_v1","effective_input_v1","current_input_v1","runtime_current_primitive_release_v1"})if(cfg.contains(key))out[key]=cfg.at(key);
 if(cfg.contains("current_input_v1")){const auto&ref=cfg.at("current_input_v1");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto m=eventio::read_json(ref.at("file").get<std::string>());if(m.at("schema")!="rinkfit.current_joint_input.v1"||m.at("version")!=1)throw std::runtime_error("current data schema/version");out["current_input_v1"]={{"schema",m.at("schema")},{"version",m.at("version")},{"logical_source_id",m.at("logical_source_id")},{"raw_rows",m.at("raw_rows")},{"packet_values_sha256",m.at("packet").at("sha256")},{"source_model_static_authority_sha256",m.at("source_model").at("sha256")},{"qualification",m.at("qualification")},{"motion",m.at("motion")}};}
 if(cfg.contains("effective_input_v1")){if(cfg.contains("dense_source_qualification")||cfg.contains("engineering_dense_source_weights_v1"))throw std::runtime_error("effective input identity is mutually exclusive with legacy dense qualification/weight authority");const auto&ref=cfg.at("effective_input_v1");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto e=eventio::read_json(ref.at("file").get<std::string>());if(e.at("abi")!="rinkfit.effective_input_abi.v1")throw std::runtime_error("effective input semantic ABI differs");out["effective_input_v1"]={{"input_validation_mode","effective_input_v1"},{"abi",e.at("abi")},{"manifest_sha256",ref.at("sha256")},{"resolved_metadata_sha256",e.at("resolved_metadata_sha256")},{"validator_code_sha256",e.at("validator_code_sha256")},{"compiled_source_validator_sha256",e.at("compiled_source_validator_sha256")}};}
 if(cfg.contains("source_uv_capture_objective")){
  out["capture_certified_numeric_domain"]=cfg.contains("current_input_v1")?Json{{"schema","rinkfit.current_capture_numeric_domain.v1"},{"sha256",cfg.at("capture_certified_numeric_domain").at("sha256")}}:cfg.at("capture_certified_numeric_domain");const char*chosen=nullptr;for(const auto*key:{"engineering_source_weights_v3","engineering_source_weights_v4","engineering_source_weights_v5","engineering_dense_source_weights_v1","effective_input_v1","current_input_v1"})if(cfg.contains(key)){if(chosen)throw std::runtime_error("engineering objective identity has ambiguous source weights");chosen=key;}if(!chosen)throw std::runtime_error("engineering objective identity lacks its actual source weights");if(std::string(chosen)!="effective_input_v1"&&std::string(chosen)!="current_input_v1")out[chosen]=cfg.at(chosen);if(cfg.contains("capture_station_cells_v20"))out["capture_station_cells_v20"]=cfg.at("capture_station_cells_v20");
 }
 return out;
}
inline std::string joint_engineering_semantics_sha256_v1(const eventio::Json&semantics){const auto text=semantics.dump();return eventio::sha256(std::vector<std::uint8_t>(text.begin(),text.end()));}
inline std::string joint_engineering_objective_fingerprint_v1(const eventio::Json&cfg,double default_pixel_total_weight=std::numeric_limits<double>::quiet_NaN()){
 return joint_engineering_semantics_sha256_v1(joint_engineering_objective_configuration_v1(cfg,default_pixel_total_weight));
}
// Old producer binaries used narrower semantic key sets.  Their real runtime
// fingerprint is an archived fact, not something a new whitelist can recreate.
// Import one only through its SHA-bound actual run specification/native binary
// plus the already verified complete checkpoint and original configuration.
template<class Pin>inline std::string verified_joint_engineering_producer_fingerprint_v1(
 const eventio::Json&producer_cfg,const eventio::Json&manifest,const std::string&actual_fingerprint,
 const std::string&actual_token,const std::string&producer_source_sha,Pin pin){
 const auto canonical=joint_engineering_objective_fingerprint_v1(producer_cfg);if(!manifest.contains("historical_producer_objective_identity_v1")){if(canonical!=actual_fingerprint)throw std::runtime_error("producer uses a historical semantic identity; explicit SHA-bound actual producer specification is required");return canonical;}
 const auto&binding=manifest.at("historical_producer_objective_identity_v1");auto require=[](bool ok,const std::string&why){if(!ok)throw std::runtime_error("historical producer objective identity: "+why);};
 require(binding.at("schema")=="rinkfit.historical_producer_objective_identity.v1"&&binding.at("runtime_fingerprint")==actual_fingerprint&&binding.at("expected_state_token")==actual_token,"actual checkpoint fingerprint/token differs");require(binding.at("original_configuration")==manifest.at("original_configuration")&&binding.at("checkpoint")==manifest.at("checkpoint"),"producer configuration/checkpoint reference differs");
 const auto spec=eventio::read_json(pin(binding.at("actual_run_specification")));require(spec.at("configuration")==manifest.at("original_configuration")&&spec.at("native")==binding.at("native_artifact")&&spec.at("packet").at("sha256")==producer_source_sha,"actual producer specification/configuration/native/source differs");require(eventio::read_json(pin(spec.at("configuration")))==producer_cfg,"SHA-bound producer configuration content differs");pin(spec.at("native"));
 if(manifest.contains("input_model"))require(spec.at("model").at("sha256")==manifest.at("input_model").at("sha256"),"actual producer input model differs");require(!binding.at("reason").get<std::string>().empty(),"historical identity reason missing");return actual_fingerprint;
}
// AL multipliers belong to an explicit constraint problem. A mode transition
// may warm-start the identical global constraints, or deliberately initialize
// a different full formal AL. Neither is an exact optimizer continuation.
inline eventio::Json transition_joint_engineering_global_phr_v1(eventio::Json&state,
 const eventio::Json&cfg,const std::string&from_token,const std::string&target_mode,
 const eventio::Json&registry=eventio::Json()){
 using Json=eventio::Json;if(state.is_null())return {{"schema","rinkfit.global_phr_mode_transition_receipt.v1"},{"parent_global_state_present",false},{"target_objective_id",target_mode},{"global_duals_silently_dropped",false}};
 if(!cfg.contains("engineering_global_phr_mode_transition_v1"))throw std::runtime_error("mode transition with persisted engineering global PHR state requires an explicit SHA-bound preserve/reset policy");
 const auto&policy=cfg.at("engineering_global_phr_mode_transition_v1");auto require=[](bool ok,const std::string&why){if(!ok)throw std::runtime_error("global PHR mode transition: "+why);};
 require(state.at("schema")=="rinkfit.engineering_global_phr_state.v1"&&policy.at("schema")=="rinkfit.engineering_global_phr_mode_transition.v1","unsupported schema");require(policy.at("from_state_token")==from_token&&policy.at("from_global_state_sha256")==joint_engineering_semantics_sha256_v1(state)&&policy.at("target_objective_id")==target_mode,"parent complete state/global state or target mode differs");require(!policy.at("reason").get<std::string>().empty(),"reason missing");
 const auto before=state;const auto action=policy.at("action").get<std::string>();if(action=="preserve_same_global_constraints"){
  require(target_mode=="rinkfit.sourceUV_trueRS_finite_capture.v3"||target_mode=="rinkfit.sourceUV_trueRS_finite_capture.v4","global warm start only supports the same actual capture constraints");require(cfg.contains("engineering_global_phr_v1")&&state.at("policy_sha256")==joint_engineering_semantics_sha256_v1(cfg.at("engineering_global_phr_v1")),"target actual global PHR consumer/policy differs");require(!registry.is_null()&&state.at("inequality_ids")==registry.at("inequality_ids")&&state.at("equality_ids")==registry.at("equality_ids"),"target ordered original global constraints differ");
 }else if(action=="reset_for_new_formal_AL"){
  require(target_mode=="rinkfit.hybrid_sourceUV_localIFT_budgeted_full_W.v1","reset is only for the explicitly new full formal AL");require(!cfg.contains("engineering_global_phr_v1"),"formal transition cannot leave an unused engineering-global consumer configured");state=nullptr;
 }else throw std::runtime_error("unsupported explicit global PHR mode-transition action");
 return {{"schema","rinkfit.global_phr_mode_transition_receipt.v1"},{"policy",policy},{"parent_global_state_present",true},{"parent_global_state",before},{"target_global_state",state},{"target_objective_id",target_mode},{"global_duals_silently_dropped",false},{"exact_optimizer_continuation_claimed",false},{"quality_accepted",false}};
}
inline eventio::Json transition_joint_engineering_global_phr_v1(JointGlobalPHRStateEpochV1&state,const eventio::Json&cfg,const std::string&from_token,const std::string&target_mode,const eventio::Json&registry=eventio::Json()){eventio::Json next=state.json();auto receipt=transition_joint_engineering_global_phr_v1(next,cfg,from_token,target_mode,registry);if(next!=state.json())state.publish(std::move(next));return receipt;}

}
