#pragma once
#include "rinkfit/joint_aux_prefix_current_v3.hpp"
#include "rinkfit/joint_R_inner_aux_prefix.hpp"
namespace rinkfit {
inline JointAcceptedAuxiliaryPreservation preserve_joint_accepted_auxiliary_prefix_current_v4(JointProblemBuilder&core,const std::filesystem::path&model,const std::filesystem::path&packet){
 if(core.packet.source_json.contains("accepted_explicit_state_binding")&&core.packet.source_json.at("accepted_explicit_state_binding").value("schema",std::string{})=="rinkfit.accepted_R_inner_source_append_state_packet_binding.v1")return preserve_joint_R_inner_inherited_auxiliary_prefix(core,model,packet);
 return preserve_joint_accepted_auxiliary_prefix_current_v3(core,model,packet);
}
}
