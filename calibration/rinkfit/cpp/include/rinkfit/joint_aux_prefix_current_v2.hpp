#pragma once
#include "rinkfit/joint_successor_aux_prefix.hpp"
#include "rinkfit/joint_paint_width_aux_prefix.hpp"
namespace rinkfit {
inline JointAcceptedAuxiliaryPreservation preserve_joint_accepted_auxiliary_prefix_current_v2(JointProblemBuilder&core,const std::filesystem::path&input_model,const std::filesystem::path&input_packet){
 if(core.packet.source_json.contains("accepted_explicit_state_binding")){const auto&binding=core.packet.source_json.at("accepted_explicit_state_binding");const auto contract=eventio::read_json(accepted_aux_detail::verified(binding.at("prefix_contract")));if(contract.contains("qualified_physical_suffix")&&contract.at("qualified_physical_suffix").value("schema",std::string{})=="rinkfit.qualified_physical_suffix_for_accepted_aux.v3_local_paint_width")return preserve_joint_accepted_auxiliary_prefix_paint_width(core,input_model,input_packet);}
 return preserve_joint_accepted_auxiliary_prefix_current(core,input_model,input_packet);
}
}
