#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include <functional>
#include <chrono>
namespace rinkfit {
// Source-evidenced navigation guesses only. No continuity residual, fixed
// trajectory, final smoothing, changed source, or reduced physical freedom.
inline eventio::Json refine_joint_motion_navigation(
 JointALBuilder&b,ceres::Problem&problem,const eventio::Json&spec,
 const std::filesystem::path&out,int threads,
 const std::function<eventio::Json()>&constraint_debt){
 std::filesystem::create_directories(out);
 if(spec.at("source_packet_sha256").get<std::string>()!=b.core.packet.source_hash)
  throw std::runtime_error("motion navigation source mismatch");
 for(const auto&ref:spec.at("source_evidence"))
  eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
 const auto start=std::chrono::steady_clock::now();
 JointCompactAcceptedState before(b);
 auto evaluate=[&](){
  if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))cache->PrepareForEvaluation(false,true);
  ceres::Problem::EvaluateOptions o;o.num_threads=threads;double cost=NAN;
  if(!problem.Evaluate(o,&cost,nullptr,nullptr,nullptr)||!std::isfinite(cost))
   throw std::runtime_error("motion navigation complete objective evaluation failed");
  return cost;
 };
 auto roots=[&](){eventio::Json result=eventio::Json::array();
  for(auto index:spec.at("diagnostic_source_rows")){
   const auto i=index.get<std::size_t>();const auto&r=b.core.packet.rows.at(i);
   if(!r.active)throw std::runtime_error("motion navigation diagnostic source inactive");
   const auto l=r.layout(b.core.model);const auto ptr=b.core.model.pointers(l);
   const auto p=b.core.model.predict_for_al(l,b.core.station[i],b.core.row[i],ptr.data());
   const auto root=b.core.model.project_rs(l,b.core.station[i],r.uv[1]);
   result.push_back({{"row_index",i},{"shot_id",r.shot_id},{"native_pts",r.pts},
    {"source_row_id",r.source_row_id},{"source_uv",r.uv},
    {"CP_ray_domain_valid",p.physical_valid()},{"native_RS_valid",root.valid},
    {"native_RS_error_px",root.valid?eventio::Json(std::hypot(root.uv[0]-r.uv[0],root.uv[1]-r.uv[1])):eventio::Json()}});
  }return result;
 };
 const double cost_before=evaluate();const auto debt_before=constraint_debt(),roots_before=roots();
 eventio::Json proposals=eventio::Json::array();
 for(const auto&p:spec.at("proposals")){
  const auto name=p.at("block").get<std::string>();const int axis=p.at("axis");
  auto it=std::find_if(b.core.model.blocks.begin(),b.core.model.blocks.end(),[&](const auto&v){return v.name==name;});
  if(it==b.core.model.blocks.end()||axis<0||axis>=int(it->values.size())||
     !problem.HasParameterBlock(it->values.data())||problem.IsParameterBlockConstant(it->values.data())||
     it->constant||std::find(it->fixed_indices.begin(),it->fixed_indices.end(),axis)!=it->fixed_indices.end())
   throw std::runtime_error("motion navigation target is not an existing free coordinate");
  double value=0,weight=0;
  for(const auto&n:p.at("navigation_controls")){
   const auto nn=n.at("block").get<std::string>();const double w=n.at("weight");
   auto ni=std::find_if(b.core.model.blocks.begin(),b.core.model.blocks.end(),[&](const auto&v){return v.name==nn;});
   if(ni==b.core.model.blocks.end()||axis>=int(ni->values.size())||!(w>=0&&std::isfinite(w)))
    throw std::runtime_error("invalid motion navigation control");
   value+=w*ni->values[axis];weight+=w;
  }
  if(std::abs(weight-1)>1e-12||!std::isfinite(value)||value<it->lower.at(axis)||value>it->upper.at(axis))
   throw std::runtime_error("invalid motion navigation value");
  const double previous=it->values[axis];it->values[axis]=value;
  proposals.push_back({{"block",name},{"axis",axis},{"before",previous},{"proposed",value}});
 }
 const double proposed_cost=evaluate();const auto debt_proposed=constraint_debt(),roots_proposed=roots();
 std::size_t valid_before=0,valid_proposed=0;for(const auto&r:roots_before)valid_before+=r.at("native_RS_valid").get<bool>();for(const auto&r:roots_proposed)valid_proposed+=r.at("native_RS_valid").get<bool>();
 const bool accepted=proposed_cost<cost_before&&valid_proposed>valid_before;
 if(!accepted){before.restore(b);evaluate();}
 eventio::Json report={{"schema","rinkfit.source_evidenced_motion_navigation.v1"},
  {"source_packet_sha256",b.core.packet.source_hash},{"source_evidence",spec.at("source_evidence")},
  {"proposals",proposals},{"full_current_PHR_cost_before",cost_before},{"full_current_PHR_cost_proposed",proposed_cost},
  {"original_constraint_debt_before",debt_before},{"original_constraint_debt_proposed",debt_proposed},
  {"diagnostic_source_roots_before",roots_before},{"diagnostic_source_roots_proposed",roots_proposed},
  {"accepted",accepted},{"source_UV_ID_Query_W_E_changed",false},{"same_complete_problem_and_free_coordinates_retained",true},
  {"continuity_residual_or_post_smoothing_added",false},
  {"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
 eventio::atomic_json(out/"receipt.json",report);return report;
}
}
