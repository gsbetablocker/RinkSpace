#pragma once
#include "rinkfit/joint_measurement_operator.hpp"

namespace rinkfit {
inline eventio::Json joint_frozen_target_binding(const eventio::Json&packet) {
 const auto arrays=packet.value("arrays",eventio::Json::object());
 eventio::Json selected=eventio::Json::object();
 for(const auto*key:{"E","source_sigma"})if(arrays.contains(key))selected[key]=arrays.at(key);
 const eventio::Json value={{"schema","rinkfit.joint_frozen_W_E_target.v1"},
  {"measurement_operator",joint_measurement_operator_binding(packet)},
  {"shot_ids",packet.value("shot_ids",eventio::Json::array())},
  {"source_clock",packet.value("source_clock",eventio::Json(nullptr))},
  {"source_manifests",packet.value("source_manifests",eventio::Json::array())},
  {"arrays",selected},{"groups",packet.value("groups",eventio::Json::array())},
  {"independent_rows",packet.value("independent_rows",eventio::Json::array())}};
 const auto hash=joint_canonical_tree_sha256(joint_dependency_identity(value));
 return {{"schema","rinkfit.joint_frozen_W_E_target.v1"},{"sha256",hash},
  {"objective_id","original_frozen_W_E_v1:"+hash}};
}
inline eventio::Json joint_replay_objective_binding(const eventio::Json&packet,
 const std::string&source_hash,const eventio::Json&report,const eventio::Json&exclusions,
 const eventio::Json*frozen_target=nullptr) {
 const eventio::Json state={{"schema","rinkfit.joint_profiled_replay_state.v1"},
  {"model_sha256",report.at("model_sha256")},{"files",report.at("files")},
  {"replay_semantics",report.value("profiled_native_export_semantics",eventio::Json(nullptr))},
  {"complete_snapshot_attachment",report.value("complete_snapshot_attachment",eventio::Json(nullptr))}};
 return {{"schema","rinkfit.joint_replay_objective_binding.v1"},
  {"source_packet_sha256",source_hash},{"frozen_target",frozen_target?*frozen_target:joint_frozen_target_binding(packet)},
  {"evaluation_state_sha256",joint_canonical_tree_sha256(state)},
  {"evaluation_state_kind","rinkfit.joint_profiled_replay_state.v1"},
  {"exclusion_ledger",exclusions},{"explicit_checkpoint_state_claimed",false}};
}
} // namespace rinkfit
