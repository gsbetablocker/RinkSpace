#pragma once
#include "rinkfit/joint_enriched_aux_prefix.hpp"
namespace rinkfit {
inline JointAcceptedAuxiliaryPreservation preserve_joint_accepted_auxiliary_prefix_current_v5(JointProblemBuilder&core,const std::filesystem::path&model,const std::filesystem::path&packet,std::shared_ptr<JointSourceQueryBranchRegistry>branch_registry=nullptr){
 if(core.packet.source_json.contains("accepted_explicit_state_binding")&&core.packet.source_json.at("accepted_explicit_state_binding").value("schema",std::string{})=="rinkfit.accepted_source_enriched_state_packet_binding.v1")return preserve_joint_source_enriched_auxiliary_prefix_v1(core,model,packet,std::move(branch_registry));
 return preserve_joint_accepted_auxiliary_prefix_current_v4(core,model,packet);
}
}
