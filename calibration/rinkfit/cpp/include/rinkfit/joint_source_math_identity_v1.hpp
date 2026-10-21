#pragma once
#include "rinkfit/joint_visible_ice_source_support.hpp"
#include "rinkfit/joint_source_query_branch_registry_v2.hpp"
#include <map>
namespace rinkfit {
struct JointSourceMathIdentityV1 {std::string identity;std::map<std::string,std::string>discrete;eventio::Json report=eventio::Json::array();};
inline std::string joint_source_math_hash(const std::string&s){eventio::SHA256Stream h;h.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());return h.finish();}
inline JointSourceMathIdentityV1 validate_joint_source_math_bindings(const eventio::Json&bindings,const JointEventModel&model,const JointPacket&packet,const std::filesystem::path&path){
 auto require=[](bool b,const char*s){if(!b)throw std::runtime_error(std::string("source mathematics identity: ")+s);};require(bindings.is_array()&&!bindings.empty()&&bindings.size()<=2,"missing/invalid ordered attachments");
 JointSourceMathIdentityV1 out;out.identity=packet.source_hash;int last=-1;bool branch=false;
 for(const auto&b:bindings){const std::string kind=b.at("kind");const int position=kind=="visible_ice_source_constraints"?0:kind=="source_query_branches"?1:-1;require(position>last&&position>=0,"unknown, duplicate or unordered attachment");last=position;std::string identity;
  if(position==0){require(b.size()==2&&b.contains("consumer_config"),"visible attachment fields differ");auto support=JointVisibleIceSourceSupport::load(model,b.at("consumer_config"));support.assert_dependencies();identity=support.identity();require(support.rows().size()==4&&support.inequality_count()==16,"visible source inventory differs");out.discrete["visible_ice_source_identity"]=identity;out.discrete["visible_ice_source_count"]="4";out.discrete["visible_ice_constraint_count"]="16";out.report.push_back(support.receipt());}
  else{require(b.size()==2&&b.contains("binding")&&packet.source_json.contains("source_query_branch_registry")&&packet.source_json.at("source_query_branch_registry")==b.at("binding"),"top-level branch policy differs");auto registry=JointSourceQueryBranchRegistry::load(b.at("binding"),model,packet);registry->assert_dependencies();identity=registry->identity();out.discrete["source_query_branch_identity"]=identity;out.discrete["source_query_branch_count"]="6328";out.report.push_back(registry->report());branch=true;}
  out.identity=joint_source_math_hash(out.identity+"|"+kind+"|"+identity);
 }
 require(branch==packet.source_json.contains("source_query_branch_registry"),"branch policy is outside ordered source attachments");return out;
}
}
