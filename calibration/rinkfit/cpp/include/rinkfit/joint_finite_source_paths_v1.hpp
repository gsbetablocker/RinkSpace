#pragma once
#include "rinkfit/joint_packet.hpp"
#include <algorithm>
namespace rinkfit {
// The explicit first-four source generation binds finite search domains. This
// is numerical navigation permission, never a new observation or root proof.
inline bool joint_first4_source_path_policy_v1(const JointPacket&p){
 if(!p.source_json.contains("engineering_scope")||!p.source_json.contains("allowed_paths"))return false;
 const auto&s=p.source_json.at("engineering_scope");return s.value("schema",std::string{})=="rinkfit.engineering_active_shots.v1"&&(s.at("shot_ids")==eventio::Json::array({"s01","s02","s03","s04"})||(p.source_json.contains("dense_source_binding")&&s.at("shot_ids")==eventio::Json::array({"s01","s02","s03","s04","s05","s06"})));
}
inline std::vector<int> joint_source_allowed_finite_paths_v1(const JointPacket&p,const JointEventModel&m,std::size_t i){
 const auto&r=p.rows.at(i);std::vector<int>out={r.path};
 if(joint_first4_source_path_policy_v1(p)){
  const auto&table=p.source_json.at("allowed_paths");auto it=table.find(std::to_string(i));if(it!=table.end())out=it->get<std::vector<int>>();
 }
 if(p.source_json.contains("dense_source_binding")){const auto&reg=p.source_json.at("dense_source_binding");const auto start=reg.at("row_start").get<std::size_t>(),count=reg.at("count").get<std::size_t>();if(i>=start){if(i-start>=count)throw std::runtime_error("dense finite path suffix index outside source");const auto&ref=reg.at("allowed_path_mask");static std::map<std::string,std::vector<std::uint8_t>>cache;const auto key=ref.at("file").get<std::string>()+":"+ref.at("sha256").get<std::string>();auto it=cache.find(key);if(it==cache.end())it=cache.emplace(key,eventio::read_array<std::uint8_t>({},ref,"|u1")).first;if(it->second.size()!=count)throw std::runtime_error("dense allowed path mask size");const auto mask=it->second[i-start];out.clear();for(int k=0;k<8;++k)if(mask&(1u<<k))out.push_back(k);}}
 if(out.empty())throw std::runtime_error("empty source-bound finite path domain");std::set<int>seen;
 for(int pi:out)if(pi<0||std::size_t(pi)>=m.materials.at(r.material).paths.size()||!seen.insert(pi).second||(r.kind==1&&!m.materials.at(r.material).paths.at(pi).physical_point))throw std::runtime_error("invalid source-bound finite entity path domain");
 return out;
}
inline bool joint_source_finite_path_allowed_v1(const JointPacket&p,const JointEventModel&m,std::size_t i,int path){const auto v=joint_source_allowed_finite_paths_v1(p,m,i);return std::find(v.begin(),v.end(),path)!=v.end();}
}
