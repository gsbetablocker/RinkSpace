#pragma once
#include "rinkfit/joint_source_guard.hpp"
#include "rinkfit/joint_source_math_identity_v1.hpp"
namespace rinkfit {
inline eventio::Json joint_source_math_wrapper_view(eventio::Json x,bool enriched){
 x.erase("source_math_source_authority");x.erase("source_query_branch_registry");
 if(enriched){x.erase("accepted_explicit_state_binding");for(const char*k:{"station","path","row","alpha"})x["arrays"].erase(k);}return x;
}
template<class Packet,class Model>inline void assert_joint_source_identity_current_v2(const Packet&p,const std::filesystem::path&path,const Model&model){
 const auto&d=p.source_json;if(!d.contains("source_math_source_authority")){assert_joint_board_source_qualification(p,path,model);return;}
 const auto&a=d.at("source_math_source_authority");auto check=[](bool b,const char*s){if(!b)throw std::runtime_error(std::string("source mathematics wrapper: ")+s);};
 check(a.at("schema")=="rinkfit.source_math_numerical_wrapper_authority.v1","unknown authority schema");joint_qualification_ref(a.at("issuer_module_ref"),path.parent_path());
 const auto pp=joint_qualification_ref(a.at("canonical_source_packet"),path.parent_path()),mp=joint_qualification_ref(a.at("canonical_source_model"),path.parent_path());
 const auto canonical_model=Model::load(mp.u8string());const auto canonical=Packet::load(pp.u8string(),canonical_model);check(!canonical.source_json.contains("source_math_source_authority"),"recursive numerical authority is forbidden");
 check(joint_source_recall_marker(canonical.source_json,pp,true).empty(),"canonical source recalled");assert_joint_board_source_qualification(canonical,pp,canonical_model);
 const std::string kind=a.at("numerical_state_kind");const bool enriched=kind=="source_enriched_complete_primal";check(enriched||kind=="frozen_source_initialization","unknown numerical-state kind");
 check(joint_source_math_wrapper_view(d,enriched)==joint_source_math_wrapper_view(canonical.source_json,enriched),"canonical non-auxiliary source fields changed");
 auto cm=canonical_model.source_json,am=model.source_json;for(auto&b:cm["blocks"])b.erase("values");for(auto&b:am["blocks"])b.erase("values");check(cm==am,"canonical physical model metadata changed");
 validate_joint_source_math_bindings(a.at("source_math_bindings"),model,p,path);
}
}
