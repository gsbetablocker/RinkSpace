#pragma once
#include "rinkfit/joint_packet.hpp"
#include <algorithm>
#include <mutex>
namespace rinkfit {
// The explicit first-four source generation binds finite search domains. This
// is numerical navigation permission, never a new observation or root proof.
inline bool joint_first4_source_path_policy_v1(const JointPacket&p){
 if(!p.source_json.contains("engineering_scope")||!p.source_json.contains("allowed_paths"))return false;
 const auto&s=p.source_json.at("engineering_scope");return s.value("schema",std::string{})=="rinkfit.engineering_active_shots.v1"&&(s.at("shot_ids")==eventio::Json::array({"s01","s02","s03","s04"})||(p.source_json.contains("dense_source_binding")&&s.at("shot_ids")==eventio::Json::array({"s01","s02","s03","s04","s05","s06"})));
}
inline std::vector<int> joint_source_allowed_finite_paths_v1(const JointPacket&p,const JointEventModel&m,std::size_t i){
 const auto&r=p.rows.at(i);if(r.source_print_identity_reviewed){if(!r.source_finite_paths||r.source_finite_paths->empty())throw std::runtime_error("missing reviewed finite print path");return *r.source_finite_paths;}std::vector<int>out=r.source_finite_paths?*r.source_finite_paths:std::vector<int>{r.path};
 if(joint_first4_source_path_policy_v1(p)){
  const auto&table=p.source_json.at("allowed_paths");auto it=table.find(std::to_string(i));if(it!=table.end())out=it->get<std::vector<int>>();
  else if(r.shot_id=="s01"&&r.kind!=1){
   // Original S01 board measurements already declare a continuous material
   // association. The successor packet's table covers appended rows; absence
   // of a prefix entry must not turn the old numerical part into a hard match.
   const auto&material=m.materials.at(r.material);const auto&current=material.paths.at(r.path);
   if(current.kind==3&&material.paths.size()>1){
    const auto provenance=eventio::Json::parse(current.provenance,nullptr,false);
    if(provenance.is_object()&&provenance.contains("descriptor")){
     const auto policy=provenance.at("descriptor").value("association_policy",eventio::Json::object());
     if(policy.value("schema",std::string())=="rinkspace.continuous-board-association.v1"&&policy.value("original_part_is_context_not_a_hard_correspondence",false)&&policy.value("edge_role_is_preserved",false)){
      const auto parts=policy.at("candidate_parts").get<std::vector<std::string>>();std::vector<int>declared;
      for(std::size_t pi=0;pi<material.paths.size();++pi){const auto&q=material.paths[pi];const auto slash=q.id.rfind('/');const auto part=q.id.substr(slash==std::string::npos?0:slash+1);
       if(std::find(parts.begin(),parts.end(),part)!=parts.end()){
        if(q.kind!=3||q.fixed_height_fraction!=current.fixed_height_fraction||q.geometry_block!=current.geometry_block||q.provenance!=current.provenance)throw std::runtime_error("original continuous board association changed physical role");
        declared.push_back(int(pi));
       }
      }
      if(declared.size()!=parts.size())throw std::runtime_error("incomplete original continuous board candidate set");out=std::move(declared);
     }
    }
   }
  }
 }
 if(p.source_json.contains("dense_source_binding")){const auto&reg=p.source_json.at("dense_source_binding");const auto start=reg.at("row_start").get<std::size_t>(),count=reg.at("count").get<std::size_t>();if(i>=start&&i-start<count){const auto&ref=reg.at("allowed_path_mask");static std::map<std::string,std::vector<std::uint8_t>>cache;static std::mutex cache_mutex;std::lock_guard<std::mutex> lock(cache_mutex);const auto key=ref.at("file").get<std::string>()+":"+ref.at("sha256").get<std::string>();auto it=cache.find(key);if(it==cache.end())it=cache.emplace(key,eventio::read_array<std::uint8_t>({},ref,"|u1")).first;if(it->second.size()!=count)throw std::runtime_error("dense allowed path mask size");const auto mask=it->second[i-start];out.clear();for(int k=0;k<8;++k)if(mask&(1u<<k))out.push_back(k);}}
 const auto& refmeta=m.runtime_model_metadata.contains("printed_material_path_refinements")?m.runtime_model_metadata:m.source_json;
 if(r.kind!=1&&refmeta.contains("printed_material_path_refinements")){
  const auto&table=refmeta.at("printed_material_path_refinements");auto it=table.find(std::to_string(r.material));if(it!=table.end()){
   const auto&material=m.materials.at(r.material);const auto&ref=*it;if(ref.at("label")!=material.label||ref.at("original_path_count")!=1||!r.source_station_intervals.empty())throw std::runtime_error("print topology refinement conflicts with an explicit original finite station domain");
   const auto children=ref.at("children").get<std::vector<int>>();for(int pi:out)if(pi!=0&&std::find(children.begin(),children.end(),pi)==children.end())throw std::runtime_error("print topology refinement changed another source entity");out=children;
  }
 }
 if(out.empty())throw std::runtime_error("empty source-bound finite path domain");std::set<int>seen;
 for(int pi:out)if(pi<0||std::size_t(pi)>=m.materials.at(r.material).paths.size()||!seen.insert(pi).second||(r.kind==1&&!m.materials.at(r.material).paths.at(pi).physical_point))throw std::runtime_error("invalid source-bound finite entity path domain");
 return out;
}
inline bool joint_source_finite_path_allowed_v1(const JointPacket&p,const JointEventModel&m,std::size_t i,int path){const auto v=joint_source_allowed_finite_paths_v1(p,m,i);return std::find(v.begin(),v.end(),path)!=v.end();}
}
