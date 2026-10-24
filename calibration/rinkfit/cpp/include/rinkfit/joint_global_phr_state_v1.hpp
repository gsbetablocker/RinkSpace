#pragma once
#include "rinkfit/event_io.hpp"
#include <cmath>
#include <set>
namespace rinkfit {
inline std::string joint_global_phr_digest_v1(const eventio::Json&state){const auto text=state.dump();return eventio::sha256(std::vector<std::uint8_t>(text.begin(),text.end()));}
inline void validate_joint_global_phr_payload_v1(const eventio::Json&s){
 auto need=[](bool ok,const char*why){if(!ok)throw std::runtime_error(why);};
 need(s.is_object()&&s.at("schema")=="rinkfit.engineering_global_phr_state.v1","global PHR checkpoint schema mismatch");
 need(s.at("source_packet_sha256").is_string()&&s.at("source_packet_sha256").get<std::string>().size()==64&&s.at("policy_sha256").is_string()&&s.at("policy_sha256").get<std::string>().size()==64,"global PHR checkpoint source/policy SHA missing");
 std::set<std::string>ids;for(const char*name:{"inequality_ids","equality_ids"}){need(s.at(name).is_array(),"global PHR checkpoint constraint registry is not an array");for(const auto&id:s.at(name)){need(id.is_string()&&!id.get<std::string>().empty()&&ids.insert(id.get<std::string>()).second,"global PHR checkpoint duplicate or invalid constraint ID");}}
 const double rho=s.at("rho");need(std::isfinite(rho)&&rho>0,"global PHR checkpoint penalty invalid");
 for(const auto&names:std::vector<std::pair<std::string,std::string>>{{"lambda","inequality_ids"},{"nu","equality_ids"}}){const auto&a=s.at(names.first);need(a.is_array()&&a.size()==s.at(names.second).size(),"global PHR checkpoint multiplier shape mismatch");for(const auto&v:a){const double x=v;need(std::isfinite(x)&&(names.first!="lambda"||x>=0),"global PHR checkpoint multiplier invalid");}}
 need(s.at("outer_iteration").is_number_integer()&&s.at("outer_iteration").get<long long>()>=0,"global PHR checkpoint outer iteration invalid");
 if(!s.at("previous_primal_inf").is_null()){const double x=s.at("previous_primal_inf");need(std::isfinite(x)&&x>=0,"global PHR checkpoint progress invalid");}
}
template<class Discrete>eventio::Json read_joint_global_phr_small_state_v1(const Discrete&discrete){
 const auto i=discrete.find("engineering_global_phr_state_v1");if(i==discrete.end())return nullptr;auto state=eventio::Json::parse(i->second);validate_joint_global_phr_payload_v1(state);
 const auto source=discrete.find("joint_packet_sha256");if(source==discrete.end()||state.at("source_packet_sha256")!=source->second)throw std::runtime_error("global PHR checkpoint differs from saved source identity");return state;
}
inline void assert_joint_global_phr_compatible_v1(const eventio::Json&saved,const eventio::Json&current,const eventio::Json&registry,const std::string&source){
 if(saved.is_null()){if(!current.is_null())throw std::runtime_error("checkpoint omits active global PHR dual state; explicit new-objective initialization required");return;}
 validate_joint_global_phr_payload_v1(saved);
 if(saved.at("source_packet_sha256")!=source||saved.at("inequality_ids")!=registry.at("inequality_ids")||saved.at("equality_ids")!=registry.at("equality_ids"))throw std::runtime_error("global PHR checkpoint source or ordered constraint IDs changed");
 if(!current.is_null()){validate_joint_global_phr_payload_v1(current);if(saved.at("policy_sha256")!=current.at("policy_sha256"))throw std::runtime_error("global PHR continuation changed policy");}
}
inline void validate_joint_current_objective_transition_sources_v1(const eventio::Json&cfg,const std::string&parent,const std::string&target){
 if(!cfg.contains("current_input_v1")||cfg.contains("representative_objective_transition_v1")||!cfg.contains("current_objective_transition_v1"))throw std::runtime_error("current source append requires its current objective transition, without historical import keys");
 const auto&t=cfg.at("current_objective_transition_v1");if(t.at("parent_logical_source_id")!=parent||t.at("target_logical_source_id")!=target)throw std::runtime_error("current objective transition source IDs differ from the explicit current append");
}
inline std::string validate_joint_global_phr_transition_v1(const eventio::Json&cfg,const eventio::Json&saved,const std::string&source_token,bool objective_changed){
 const bool current=cfg.contains("current_input_v1");if(current&&cfg.contains("representative_objective_transition_v1"))throw std::runtime_error("current input rejects historical objective migration keys");if(!current&&cfg.contains("current_objective_transition_v1"))throw std::runtime_error("current objective migration requires prepared current input");
 if(saved.is_null()){
  if(cfg.contains("engineering_global_phr_v1")&&(!objective_changed||cfg.at("engineering_global_phr_v1").at("initialization")!="explicit_new_objective_zero_duals"))throw std::runtime_error("new global PHR policy requires explicit new-objective zero-dual initialization");return "absent";
 }
 validate_joint_global_phr_payload_v1(saved);
 if(!cfg.empty()&&!cfg.contains("engineering_global_phr_v1"))throw std::runtime_error("global PHR cannot disappear from a continued objective");
 std::string action="preserve_exact";
 if(objective_changed){
  const char*key=current?"current_objective_transition_v1":"representative_objective_transition_v1";
  if(!cfg.contains(key))throw std::runtime_error("new objective requires explicit global PHR state migration");
  const auto&t=cfg.at(key);const char*schema=current?"rinkfit.current_objective_transition.v1":"rinkfit.representative_objective_transition.v1";if(t.at("schema")!=schema||t.at("kind")!="complete_primal_warm_start"||t.at("from_state_token")!=source_token||!t.contains("global_phr_state"))throw std::runtime_error("new objective omits bound global PHR state migration");const auto&g=t.at("global_phr_state");
  if(current&&(t.at("parent_logical_source_id")!=saved.at("source_packet_sha256")||!t.at("target_logical_source_id").is_string()||t.at("target_logical_source_id").get<std::string>().empty()))throw std::runtime_error("current global PHR migration source binding differs");
  if(g.at("from_state_sha256")!=joint_global_phr_digest_v1(saved)||g.at("reason").get<std::string>().empty())throw std::runtime_error("global PHR migration source digest/reason mismatch");action=g.at("action");
  if(current&&action!="preserve_exact")throw std::runtime_error("current source append must preserve exact global PHR state");
  if(action!="preserve_exact"&&action!="reset_for_new_objective")throw std::runtime_error("unsupported global PHR state migration action");
 }
 if(action=="reset_for_new_objective"){
  if(!objective_changed||!cfg.contains("engineering_global_phr_v1")||cfg.at("engineering_global_phr_v1").at("initialization")!="explicit_new_objective_zero_duals")throw std::runtime_error("global PHR reset is not an explicit new-objective initialization");
 }else if(cfg.contains("engineering_global_phr_v1")&&saved.at("policy_sha256")!=joint_global_phr_digest_v1(cfg.at("engineering_global_phr_v1")))throw std::runtime_error("global PHR preserve changed the policy");
 return action;
}
// A deliberately changed current physical model may add confirmed constraints.
// This is a one-time current-data transport, never a strict-resume fallback.
// Preserve the old ordered subsequence and every existing multiplier/rho; only
// explicitly declared new IDs receive zero multipliers. No historical file is read.
inline eventio::Json extend_joint_current_physical_constraints_v1(const eventio::Json&saved,const eventio::Json&registry,const eventio::Json&extension,const std::string&source_token,const std::string&source){
 using J=eventio::Json;auto need=[](bool ok,const char*why){if(!ok)throw std::runtime_error(std::string("current physical constraint extension: ")+why);};
 validate_joint_global_phr_payload_v1(saved);
 need(extension.at("schema")=="rinkfit.current_physical_constraint_extension.v1"&&extension.at("new_objective_identity_required")==true,"explicit new target required");
 need(extension.at("from_state_token")==source_token&&extension.at("from_global_state_sha256")==joint_global_phr_digest_v1(saved),"source complete state binding differs");
 need(saved.at("source_packet_sha256")==source&&extension.at("target_registry_sha256")==joint_global_phr_digest_v1(registry),"source or target registry differs");
 need(extension.at("reason").is_string()&&!extension.at("reason").get<std::string>().empty(),"missing actual physical-model reason");
 J result=saved;std::size_t added=0;
 for(const auto&pair:std::vector<std::pair<std::string,std::string>>{{"inequality_ids","lambda"},{"equality_ids","nu"}}){
  const auto&old_ids=saved.at(pair.first);const auto&target=registry.at(pair.first);const auto&declared=extension.at("added_"+pair.first);need(declared.is_array()&&target.is_array(),"new registry/ID declaration shape");
  std::set<std::string>wanted,seen;for(const auto&v:declared)need(v.is_string()&&!v.get<std::string>().empty()&&wanted.insert(v.get<std::string>()).second,"duplicate/empty added ID");
  J values=J::array();std::size_t previous=0;
  for(const auto&id:target){need(id.is_string()&&seen.insert(id.get<std::string>()).second,"duplicate target ID");if(previous<old_ids.size()&&id==old_ids.at(previous)){need(!wanted.count(id.get<std::string>()),"old ID declared new");values.push_back(saved.at(pair.second).at(previous++));}else{need(wanted.erase(id.get<std::string>())==1,"old constraint reordered, removed or undeclared addition");values.push_back(0.);++added;}}
  need(previous==old_ids.size()&&wanted.empty(),"old constraint missing or declared new ID absent");result[pair.first]=target;result[pair.second]=std::move(values);
 }
 need(added>0,"empty constraint extension is not a new target");
 // Progress against the old constraint denominator is not a new-target ratio.
 // Retain it explicitly, while leaving every old primal/dual/penalty intact.
 result["current_constraint_extension_v1"]={{"source_state_token",source_token},{"source_global_state_sha256",joint_global_phr_digest_v1(saved)},{"added_constraints",added},{"previous_primal_inf_before_extension",saved.at("previous_primal_inf")},{"last_progress_before_extension",saved.value("last_progress",J(nullptr))},{"all_previous_multipliers_rho_outer_iteration_preserved",true}};
 result["previous_primal_inf"]=nullptr;result["last_progress"]=nullptr;validate_joint_global_phr_payload_v1(result);return result;
}
template<class Builder,class Snapshot>eventio::Json restore_joint_global_phr_small_state_v1(Builder&b,const Snapshot&s,const eventio::Json&cfg=eventio::Json::object(),bool objective_changed=false){
 const auto saved=read_joint_global_phr_small_state_v1(s.data().discrete);const auto action=validate_joint_global_phr_transition_v1(cfg,saved,s.token(),objective_changed);
 if(saved.is_null()){assert_joint_global_phr_compatible_v1(saved,b.engineering_global_phr_state_v1,b.engineering_global_constraint_registry_v1(),b.core.packet.source_hash);}
 else if(action=="reset_for_new_objective")b.engineering_global_phr_state_v1=nullptr;
 else{assert_joint_global_phr_compatible_v1(saved,b.engineering_global_phr_state_v1,b.engineering_global_constraint_registry_v1(),b.core.packet.source_hash);b.engineering_global_phr_state_v1=saved;}
 return {{"schema","rinkfit.global_phr_small_state_restore.v1"},{"source_state_token",s.token()},{"source_global_state_present",!saved.is_null()},{"source_global_state_sha256",saved.is_null()?eventio::Json(nullptr):eventio::Json(joint_global_phr_digest_v1(saved))},{"action",action},{"multipliers_restored_exactly",!saved.is_null()&&action=="preserve_exact"},{"new_objective_zero_duals_required",action=="reset_for_new_objective"||saved.is_null()},{"source_global_state_retained_in_original_checkpoint",true}};
}
// Small numerical metadata participates in the same accepted snapshot as the
// large parameter array. Copying it does not rebuild the parameter registry.
class JointCompactSmallStateV1 {
 eventio::Json*live_;eventio::Json saved_;
 public:explicit JointCompactSmallStateV1(eventio::Json&state):live_(&state){capture();}
 void capture(){saved_=*live_;}void restore()const{*live_=saved_;}bool matches()const{return *live_==saved_;}
 void append_token(std::uint64_t&h)const{if(saved_.is_null())return;const auto bytes=saved_.dump();for(const unsigned char c:bytes){h^=c;h*=1099511628211ull;}}
};
}
