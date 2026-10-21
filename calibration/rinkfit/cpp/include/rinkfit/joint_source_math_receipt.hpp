#pragma once
#include "rinkfit/joint_al.hpp"
namespace rinkfit {
inline eventio::Json joint_source_math_bindings(const JointALBuilder&b){
 eventio::Json result=eventio::Json::array();
 if(const auto source=b.visible_ice_source_support())result.push_back({{"kind","visible_ice_source_constraints"},{"consumer_config",source->receipt().at("consumer_config")}});
 if(const auto policy=b.source_query_branch_registry())result.push_back({{"kind","source_query_branches"},{"binding",policy->binding()}});
 return result;
}
inline void add_joint_source_math_receipt(const JointALBuilder&b,eventio::Json&report){
 b.assert_source_math_dependencies();report["mathematical_source_id"]=b.identity().source_id;
 report["source_math_bindings"]=joint_source_math_bindings(b);
 if(const auto policy=b.source_query_branch_registry())report["source_query_branches"]={{"identity",policy->identity()},{"source_rows",JointSourceQueryBranchRegistry::count},{"binding",policy->binding()},{"raw_full_Query_source_ledger_preserved",true},{"typed_Query_only_own_source_component",true},{"capture_or_quality_accepted",false}};
}
inline void assert_joint_source_math_configuration(const eventio::Json&cfg,const JointPacket&packet){
 if(!packet.source_json.contains("source_math_source_authority"))return;
 eventio::Json actual=eventio::Json::array();
 if(cfg.contains("visible_ice_source_constraints"))actual.push_back({{"kind","visible_ice_source_constraints"},{"consumer_config",cfg.at("visible_ice_source_constraints")}});
 if(cfg.contains("source_query_branch_registry"))actual.push_back({{"kind","source_query_branches"},{"binding",cfg.at("source_query_branch_registry")}});
 if(actual!=packet.source_json.at("source_math_source_authority").at("source_math_bindings"))throw std::runtime_error("run config omitted or changed packet-bound source mathematical consumers");
}
inline std::shared_ptr<JointSourceQueryBranchRegistry> load_joint_source_query_policy(const eventio::Json&cfg,const JointEventModel&model,const JointPacket&packet){
 const bool requested=cfg.contains("source_query_branch_registry"),bound=packet.source_json.contains("source_query_branch_registry");
 if(requested!=bound)throw std::runtime_error("source Query branch config and actual packet binding must both be present");
 if(!requested)return {};
 if(cfg.at("source_query_branch_registry")!=packet.source_json.at("source_query_branch_registry"))throw std::runtime_error("source Query branch config does not equal complete packet binding");
 return JointSourceQueryBranchRegistry::load(cfg.at("source_query_branch_registry"),model,packet);
}
}
