#pragma once
#include "rinkfit/joint_effective_input_v1.hpp"
#include "rinkfit/joint_model.hpp"
namespace rinkfit {
// This is the immediate load(mp) -> packet-load boundary. It checks exact raw
// source bytes plus the block/runtime metadata view emitted by exact_json().
// It is deliberately not an attestation of arbitrary later material/rig edits.
inline eventio::Json assert_joint_effective_model_load_boundary_v1(const JointEventModel&live,const std::string&cli_model,const eventio::Json&ref){
 const auto original=JointEffectiveInputV1::pin(ref);const auto actual=std::filesystem::u8path(cli_model);
 eventio::verify(actual,ref.at("sha256"));if(!std::filesystem::equivalent(actual,original))throw std::runtime_error("effective input original CLI model address differs");
 const auto source=JointEffectiveInputV1::strict_json(original);
 if(live.source_json!=source)throw std::runtime_error("effective input original model source JSON differs from supplied load");
 const auto expected=JointEventModel::load(original.u8string());const auto normalized=expected.exact_json();
 if(live.exact_json()!=normalized)throw std::runtime_error("effective input live model blocks/runtime metadata differ from canonical same-source load");
 return {{"schema","rinkfit.actual_effective_model_load_identity.v1"},{"original_model",ref},{"original_raw_SHA_and_strict_JSON_verified",true},
  {"canonical_typed_default_normalization_verified",true},{"raw_document_required_default_normalization",source!=normalized},
  {"live_block_and_runtime_metadata_view_checked",true},{"arbitrary_later_material_or_rig_mutations_proven",false},
  {"validation_boundary","immediately_after_JointEventModel_load_before_authorized_runtime_transform"}};
}
}
