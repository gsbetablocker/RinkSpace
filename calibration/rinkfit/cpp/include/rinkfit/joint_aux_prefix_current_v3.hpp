#pragma once
#include "rinkfit/joint_aux_prefix_current_v2.hpp"
#include "rinkfit/joint_paint_operator_aux_prefix.hpp"
namespace rinkfit {
inline JointAcceptedAuxiliaryPreservation preserve_joint_accepted_auxiliary_prefix_current_v3(JointProblemBuilder&core,const std::filesystem::path&model,const std::filesystem::path&packet){
 if(core.packet.source_json.contains("accepted_explicit_state_binding")&&core.packet.source_json.at("accepted_explicit_state_binding").value("schema",std::string{})=="rinkfit.accepted_operator_transition_state_packet_binding.v1")return preserve_joint_accepted_auxiliary_prefix_operator_transition(core,model,packet);
 return preserve_joint_accepted_auxiliary_prefix_current_v2(core,model,packet);
}
}
