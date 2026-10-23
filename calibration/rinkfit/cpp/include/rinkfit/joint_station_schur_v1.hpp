#pragma once
#include "rinkfit/joint_al.hpp"
#include <ceres/ordered_groups.h>
#include <cstdint>
#include <unordered_set>
namespace rinkfit {
// Exact Schur elimination of the LINEARIZED independent scalar stations.
// The pixel/Jacobian cache remains the existing GPU path. Eigen factorization
// is on the CPU; this is not a nonlinear closest-point or IFT implementation.
inline std::shared_ptr<ceres::ParameterBlockOrdering> make_joint_station_schur_ordering_v1(const JointALBuilder&b,const ceres::Problem&problem,const std::filesystem::path&out){
 auto require=[](bool ok,const std::string&why){if(!ok)throw std::runtime_error("station Schur ordering: "+why);};
 require(b.source_uv_capture_v1()||b.source_uv_observed_row_initialization(),"only the explicit sourceUV capture/initialization graph is eligible");
 const auto begin=reinterpret_cast<std::uintptr_t>(b.core.station.data()),end=begin+b.core.station.size()*sizeof(double);
 auto station_index=[&](const double*x,std::size_t*index){const auto p=reinterpret_cast<std::uintptr_t>(x);if(p<begin||p>=end||(p-begin)%sizeof(double))return false;*index=(p-begin)/sizeof(double);return true;};
 std::unordered_set<const double*>physical;for(const auto&block:b.core.model.blocks)physical.insert(block.values.data());
 std::unordered_set<const double*>motion_latents;if(b.static_motion)for(const auto&r:b.static_motion->rows)motion_latents.insert(r.latent.data());std::size_t motion_blocks=0;
 auto ordering=std::make_shared<ceres::ParameterBlockOrdering>();std::vector<double*>all;problem.GetParameterBlocks(&all);std::vector<std::uint8_t>eliminated(b.core.station.size(),0);std::size_t stations=0,remaining=0,fixed=0,at_lower=0,at_upper=0;std::size_t physical_tangent=0;
 for(double*x:all){const int tangent=problem.ParameterBlockTangentSize(x);if(problem.IsParameterBlockConstant(x)||tangent<=0){ordering->AddElementToGroup(x,2);++fixed;continue;}std::size_t i=0;if(station_index(x,&i)){const auto&r=b.core.packet.rows.at(i);require(r.active&&r.kind!=1&&problem.ParameterBlockSize(x)==1&&tangent==1,"invalid free station block");require(std::isfinite(*x),"nonfinite original station");const double lo=problem.GetParameterLowerBound(x,0),hi=problem.GetParameterUpperBound(x,0);require(lo<=*x&&*x<=hi,"station is outside its original Ceres bounds");at_lower+=*x==lo;at_upper+=*x==hi;eliminated.at(i)=1;ordering->AddElementToGroup(x,0);++stations;}else if(motion_latents.count(x)){require(problem.ParameterBlockSize(x)==2&&tangent==2,"invalid shared surface latent");ordering->AddElementToGroup(x,0);++motion_blocks;}else{require(physical.count(x)>0,"a nonphysical row/alpha/budget/enclosure is unexpectedly free");ordering->AddElementToGroup(x,1);++remaining;physical_tangent+=tangent;}}
 require(stations>0&&remaining>0,"nonempty independent station and remaining physical groups are required");
 std::vector<ceres::ResidualBlockId>residuals;problem.GetResidualBlocks(&residuals);std::vector<double*>parameters;parameters.reserve(64);
 // A shared board point has unknown material UV, not a measured absolute
 // anchor. Eliminate these small independent blocks with the stations so the
 // reduced camera/scene system retains their full linearized coupling.
 std::unordered_set<const double*>surface_points;
 for(const auto&block:b.core.model.blocks){const auto*x=block.values.data();if(block.name.rfind("board_texture_shared_uv/",0)==0&&problem.HasParameterBlock(x)&&!problem.IsParameterBlockConstant(x)&&problem.ParameterBlockTangentSize(x)>0)surface_points.insert(x);}
 const auto proposed_surface_points=surface_points.size();
 for(auto id:residuals){parameters.clear();problem.GetParameterBlocksForResidualBlock(id,&parameters);bool occupied=false;for(const double*x:parameters){std::size_t potential=0;occupied|=ordering->GroupId(const_cast<double*>(x))==0||station_index(x,&potential);}
  for(const double*x:parameters)if(surface_points.count(x)){if(occupied)surface_points.erase(x);else occupied=true;}}
 std::size_t surface_tangent=0;for(const double*x:surface_points){auto*p=const_cast<double*>(x);require(ordering->GroupId(p)==1,"surface point was not a remaining physical block");const auto tangent=problem.ParameterBlockTangentSize(x);ordering->Remove(p);ordering->AddElementToGroup(p,0);--remaining;physical_tangent-=tangent;surface_tangent+=tangent;}
 std::size_t no_station=0,one_station=0,maximum=0;
 for(auto id:residuals){parameters.clear();problem.GetParameterBlocksForResidualBlock(id,&parameters);std::size_t count=0;for(const double*x:parameters){std::size_t i=0;if(station_index(x,&i)||motion_latents.count(x)||surface_points.count(x))++count;}maximum=std::max(maximum,count);require(count<=1,"two group0 stations share one actual residual block; exact independent elimination is invalid");count?++one_station:++no_station;}
 eventio::atomic_json(out/"actual_station_schur_ordering.json",{{"schema","rinkfit.actual_linearized_station_Schur_independent_set.v1"},{"all_actual_parameter_blocks",all.size()},{"group0_independent_free_scalar_stations",stations},{"group0_shared_surface_latent_blocks",motion_blocks},{"group0_board_material_UV_blocks",surface_points.size()},{"group0_board_material_UV_tangent_dimensions",surface_tangent},{"board_material_UV_blocks_retained_due_to_actual_coupling",proposed_surface_points-surface_points.size()},{"group1_remaining_physical_blocks",remaining},{"group1_physical_tangent_dimensions",physical_tangent},{"group2_constant_blocks_removed_by_Ceres",fixed},{"original_station_at_lower_bound",at_lower},{"original_station_at_upper_bound",at_upper},{"all_actual_residual_blocks_inspected",residuals.size()},{"residuals_with_one_group0_station",one_station},{"residuals_with_no_group0_station",no_station},{"maximum_group0_stations_in_any_actual_residual",maximum},{"original_station_bounds_or_values_changed",false},{"fixed_stations_preserved",true},{"GPU_raw_UV_J_cache_reused",true},{"linear_solver","SPARSE_SCHUR"},{"sparse_library","EIGEN_SPARSE"},{"linear_factorization_on_GPU",false},{"nonlinear_closest_points_eliminated",false},{"all_present_station_blocks_including_current_KKT_constants_are_independent",true},{"passed",true}});
 return ordering;
}
// The full constructor above proves independence for every present station,
// including currently constant KKT vertices. Only their constant/free status
// changes here: physical columns, residual incidence and source membership do not.
inline void update_joint_station_schur_kkt_v1(const JointALBuilder& b,
 const ceres::Problem& p,ceres::ParameterBlockOrdering& order,const eventio::Json& rows){
 for(const auto& item:rows){const auto i=item.get<std::size_t>();auto* x=const_cast<double*>(&b.core.station.at(i));const auto& r=b.core.packet.rows.at(i);
  if(!r.active||r.kind==1||!p.HasParameterBlock(x)||p.ParameterBlockSize(x)!=1)throw std::runtime_error("incremental Schur station identity changed");
  const int previous=order.GroupId(x);if(previous!=0&&previous!=2)throw std::runtime_error("incremental Schur station not in proved independent family");
  const int group=p.IsParameterBlockConstant(x)?2:0;
  if(group==0&&p.ParameterBlockTangentSize(x)!=1)throw std::runtime_error("incremental Schur station tangent changed");
  if(group!=previous){order.Remove(x);order.AddElementToGroup(x,group);}
 }
}

}
