#pragma once
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/event_io.hpp"
#include "rinkfit/joint_global_phr_state_v1.hpp"
namespace rinkfit {
// Complete primal state import and continuation of the same objective are
// distinct contracts. A configuration change must be declared and pinned.
inline eventio::Json validate_joint_representative_resume_identity_v1(
    const RunCheckpoint& state, const eventio::Json& saved,
    const eventio::Json& cfg, const std::string& target_fingerprint,
    bool source_revision) {
  if(state.outer.rho!=0 || !state.outer.lambda.empty() || !state.outer.nu.empty())
    throw std::runtime_error("representative observed-row import cannot discard AL dual/penalty state");
  const std::string objective=cfg.at("source_uv_capture_objective");
  if(state.state.data().identity.objective_id!=objective)
    throw std::runtime_error("representative import objective mode differs; use an explicit mode transition");
  if(state.state.token()!=saved.at("state_token").get<std::string>())
    throw std::runtime_error("representative checkpoint state token differs from manifest");
  const bool fingerprint_changed=state.runtime_fingerprint!=target_fingerprint;
  const bool changed=fingerprint_changed||source_revision;
  if(changed) {
    if(!cfg.contains("representative_objective_transition_v1"))
      throw std::runtime_error("representative objective fingerprint changed: explicit complete-primal warm-start declaration required");
    const auto& t=cfg.at("representative_objective_transition_v1");
    if(t.at("schema")!="rinkfit.representative_objective_transition.v1" ||
       t.at("kind")!="complete_primal_warm_start" ||
       t.at("from_state_token")!=state.state.token() ||
       t.at("from_runtime_fingerprint")!=state.runtime_fingerprint ||
       t.at("from_checkpoint_sha256")!=saved.at("checkpoint").at("sha256") ||
       t.at("from_model_sha256")!=saved.at("model").at("sha256") ||
       t.at("to_objective_id")!=objective ||
       t.at("reason").get<std::string>().empty())
      throw std::runtime_error("unbound or incomplete representative objective transition");
  }
  const auto global=read_joint_global_phr_small_state_v1(state.state.data().discrete);
  const auto global_action=validate_joint_global_phr_transition_v1(cfg,global,state.state.token(),changed);
  return {{"schema","rinkfit.representative_resume_identity_check.v1"},
    {"previous_state_token",state.state.token()},
    {"previous_runtime_fingerprint",state.runtime_fingerprint},
    {"target_runtime_fingerprint",target_fingerprint},
    {"restored_runtime_model_sha256",saved.at("model").at("sha256")},
    {"source_revision",source_revision},
    {"objective_changed",changed},
    {"runtime_fingerprint_changed",fingerprint_changed},
    {"saved_global_phr_state_present",!global.is_null()},
    {"global_phr_migration_action",global_action},
    {"global_phr_source_state_sha256",global.is_null()?eventio::Json(nullptr):eventio::Json(joint_global_phr_digest_v1(global))},
    {"exact_objective_continuation",!changed&&!source_revision},
    {"optimizer_history_restored",false},
    {"primal_restore_must_still_pass_original_roundtrip",true}};
}
} // namespace rinkfit
