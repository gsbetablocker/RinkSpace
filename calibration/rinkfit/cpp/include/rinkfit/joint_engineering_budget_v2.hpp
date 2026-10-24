#pragma once
#include "rinkfit/joint_builder.hpp"
#include <tuple>
#include <numeric>
namespace rinkfit {
// Engineering aggregation, never a covariance or calibrated independence claim.
// Parent and appended observations use ONE budget for each native exposure.
inline eventio::Json apply_joint_engineering_budget_v2(const eventio::Json& cfg,
    const JointProblemBuilder& core,std::vector<double>& weights,
    const std::map<std::pair<std::string,std::int64_t>,std::size_t>& motion,
    std::map<std::pair<std::string,std::int64_t>,double>& motion_weights) {
 if(!cfg.contains("engineering_support_budget_v2"))return eventio::Json();
 const auto& policy=cfg.at("engineering_support_budget_v2");
 if(policy.at("schema")!="rinkfit.native_exposure_material_budget.v2" ||
    policy.at("scheme")!="equal_native_exposure" ||
    policy.at("calibrated_covariance")!=false ||
    policy.at("source_packet_sha256")!=core.packet.source_hash)
   throw std::runtime_error("invalid explicit engineering support budget v2");
 const double total=policy.at("total_budget");
 if(!std::isfinite(total)||total<=0)throw std::runtime_error("invalid total support budget");
 std::vector<int> canonical(core.model.materials.size());std::iota(canonical.begin(),canonical.end(),0);
 eventio::Json canonical_groups=eventio::Json::array();std::set<int> assigned;
 if(policy.contains("canonical_material_entities")){
  const auto&ref=policy.at("canonical_material_entities");const bool current=core.packet.source_json.contains("runtime_current_input_v1");eventio::Json rules;if(current){rules=ref;if(rules.at("schema")!="rinkfit.current_canonical_material_entities.v1")throw std::runtime_error("current canonical entity schema");}else{eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));rules=eventio::read_json(ref.at("file").get<std::string>());if(rules.at("schema")!="rinkfit.reviewed_canonical_board_entities.v1"||rules.at("source_UV_ID_PTS_and_membership_changed")!=false)throw std::runtime_error("unsupported canonical engineering entities");const auto&review=rules.at("source_review");eventio::verify(review.at("file").get<std::string>(),review.at("sha256"));}
  for(const auto&group:rules.at("groups")){
   if(group.value("physical_entity_kind",std::string{})=="printed_panel"){
    const int number=group.at("advertisement_number");const std::string prefix="Ad"+std::to_string(number)+".";
    if(number<1||group.at("materials").size()<2||group.at("canonical_physical_entity_id")!="board.advertisement."+std::to_string(number))throw std::runtime_error("invalid actual printed panel budget");
    const int representative=group.at("materials").at(0).at("material_index");int owner=-1,offset=-1;
    for(const auto&item:group.at("materials")){const int mi=item.at("material_index");if(mi<0||mi>=int(canonical.size())||!assigned.insert(mi).second)throw std::runtime_error("overlapping printed panel budget");const auto& material=core.model.materials.at(mi);if(material.label!=item.at("material_label")||material.label.rfind(prefix,0)!=0||material.paths.empty())throw std::runtime_error("printed panel source identity changed");for(const auto&p:material.paths){if((p.kind!=2&&p.kind!=6)||p.advertisement_block<0||std::size_t(p.advertisement_block)>=core.model.blocks.size()||p.advertisement_offset<0||std::size_t(p.advertisement_offset)+4>core.model.blocks.at(p.advertisement_block).values.size())throw std::runtime_error("printed panel contains non-board source or invalid installation slots");if(owner<0){owner=p.advertisement_block;offset=p.advertisement_offset;}if(owner!=p.advertisement_block||offset!=p.advertisement_offset)throw std::runtime_error("different physical installations cannot share one source budget");}canonical[mi]=representative;}
    canonical_groups.push_back(group);continue;
   }
   const std::string id=group.at("canonical_physical_entity_id");const double height=group.at("fixed_height_fraction");
   if(!((id=="board.top"&&height==1.)||(id=="board.contact"&&height==0.))||group.at("materials").size()<2)throw std::runtime_error("unreviewed physical edge merge");
   const int representative=group.at("materials").at(0).at("material_index");
   for(const auto&item:group.at("materials")){const int mi=item.at("material_index");if(mi<0||mi>=int(canonical.size())||!assigned.insert(mi).second)throw std::runtime_error("overlapping canonical physical entity");
    const auto&material=core.model.materials.at(mi);if(material.label!=item.at("material_label")||(material.label!=id&&material.label.rfind(id+".",0)!=0)||material.paths.empty())throw std::runtime_error("canonical entity changed source material");
    for(const auto&p:material.paths)if(p.kind!=3||p.boundary_segment!=-2||p.geometry_block>=0||p.fixed_height_fraction!=height)throw std::runtime_error("different physical board boundary cannot share a budget");
    canonical[mi]=representative;
   }canonical_groups.push_back(group);
  }
 }

 using Exposure=std::pair<std::string,std::int64_t>;
 using Component=std::tuple<std::string,std::int64_t,int,int>;
 std::map<Component,std::size_t> counts;std::map<Exposure,std::size_t> components;
 for(const auto&r:core.packet.rows)if(r.active)++counts[{r.shot_id,r.pts,canonical.at(r.material),r.kind}];
 for(const auto&v:counts)++components[{std::get<0>(v.first),std::get<1>(v.first)}];
 for(const auto&item:motion)++components[item.first];
 if(components.empty())throw std::runtime_error("empty qualified support budget");
 weights.assign(core.packet.rows.size(),0.);std::map<std::string,double> shot_weight;
 double sum=0;std::size_t active=0;
 for(std::size_t i=0;i<core.packet.rows.size();++i){const auto&r=core.packet.rows[i];if(!r.active)continue;
  const auto n=counts.at({r.shot_id,r.pts,canonical.at(r.material),r.kind});
  const double w=total/double(components.size())/double(components.at({r.shot_id,r.pts}))/double(n);
  weights[i]=w;sum+=w;shot_weight[r.shot_id]+=w;++active;
 }
 std::map<std::string,double> motion_shot_weight;double motion_sum=0;for(const auto&v:motion){const double each=total/double(components.size())/double(components.at(v.first))/double(v.second);motion_weights[v.first]=each;motion_sum+=each*v.second;motion_shot_weight[v.first.first]+=each*v.second;}
 auto combined_shot_weight=shot_weight;for(const auto&v:motion_shot_weight)combined_shot_weight[v.first]+=v.second;
 if(std::abs(sum+motion_sum-total)>1e-9*total)throw std::runtime_error("support budget conservation failure");
 return {{"schema","rinkfit.actual_engineering_support_budget.v2"},{"policy",policy},
  {"actual_total_engineering_budget",sum},{"motion_endpoint_budget",motion_sum},{"common_total_budget",sum+motion_sum},{"active_rows",active},{"native_exposures",components.size()},
  {"material_kind_components",counts.size()},{"canonical_physical_entity_groups",canonical_groups},{"source_material_aliases_preserved",true},{"weight_by_shot",shot_weight},{"motion_weight_by_shot",motion_shot_weight},{"combined_weight_by_shot",combined_shot_weight},
  {"parent_and_dense_merged",true},{"legacy_rig_rates_consumed",false},
  {"E_changed",false},{"source_UV_or_membership_changed",false},
  {"calibrated_covariance",false},{"pixel_support_independence_claimed",false},
  {"temporal_independence_claimed",false},
  {"correlation_policy","one engineering budget per exposure; all rows of the same reviewed physical entity and observation kind share one component; original material/source identities remain intact"}};
}
}
