#pragma once
#include "rinkfit/joint_al.hpp"
namespace rinkfit {
// A supported finite carrier has two physical DOF (offset and direction).
// When every real observation is interior and neither endpoint is observed,
// two endpoint abscissas are a numerical coordinate convention. The ordinate
// of each endpoint remains free: this fixes neither the line nor its angle.
inline eventio::Json install_joint_line_coordinate_gauge_v1(JointALBuilder& b,
 ceres::Problem& problem,const eventio::Json& cfg){
 const auto stage=cfg.value("runtime_shared_parameter_stage_v2",eventio::Json::object());
 if(!stage.contains("finite_line_coordinate_gauges"))return {{"enabled",false}};
 eventio::Json report=eventio::Json::array();
 for(const auto& item:stage.at("finite_line_coordinate_gauges")){
  const bool current=cfg.contains("current_input_v1");if(item.at("schema")!=(current?"rinkfit.current_line_coordinate_gauge.v1":"rinkfit.source_supported_line_coordinate_gauge.v1")||item.at("endpoints_are_physical_measurements")!=false)throw std::runtime_error("untyped numerical line coordinate policy");
  if(!current){const auto ref=item.at("source_review");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));}
  const int bi=item.at("geometry_block"),axis=item.at("coordinate_axis");auto& block=b.core.model.blocks.at(bi);
  if(block.name!=item.at("block_name")||block.values.size()!=4||axis<0||axis>1||problem.IsParameterBlockConstant(block.values.data())||problem.ParameterBlockTangentSize(block.values.data())!=4)throw std::runtime_error("numerical line gauge requires existing four-free-coordinate carrier");
  const double span=block.values[axis+2]-block.values[axis];if(std::abs(span)<1e-8)throw std::runtime_error("numerical line coordinate chart is singular");
  std::size_t rows=0;double lo=1,hi=0;
  for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;const auto&p=b.core.model.materials.at(r.material).paths.at(r.path);if(p.geometry_block!=bi)continue;
   const auto& indices=b.core.model.path_geometry_indices.at({r.material,r.path});
   if(p.kind!=0||p.physical_point||r.kind==1||!indices.empty()||b.core.model.path_geometry_offsets.at({r.material,r.path})!=0||!problem.HasParameterBlock(&b.core.station[i])||problem.IsParameterBlockConstant(&b.core.station[i]))throw std::runtime_error("observed endpoint or restricted consumer invalidates numerical line gauge");
   const double s=b.core.station[i];if(!(s>0&&s<1))throw std::runtime_error("real source lies at numerical support end; source-bound extension is required before this chart");
   if(r.station_intervals(b.core.model,r.path)!=std::vector<std::array<double,2>>{{0.,1.}})throw std::runtime_error("numerical line gauge cannot overwrite source-specific parameter intervals");
   lo=std::min(lo,s);hi=std::max(hi,s);++rows;
  }
  if(!rows)throw std::runtime_error("empty source support for numerical line chart");
  problem.SetManifold(block.values.data(),new ceres::SubsetManifold(4,{axis,axis+2}));
  report.push_back({{"block",bi},{"name",block.name},{"source_rows",rows},{"station_range",{lo,hi}},{"actual_tangent",2},{"removed_along_line_coordinate_directions",2},{"line_position_and_direction_remain_free",true},{"physical_endpoints_measured",false},{"source_points_and_world_values_changed",false},{"finite_support_remains_explicit",true}});
 }
 return {{"enabled",true},{"schema","rinkfit.actual_line_coordinate_gauge.v1"},{"groups",report}};
}
}
