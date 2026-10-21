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
 auto ordering=std::make_shared<ceres::ParameterBlockOrdering>();std::vector<double*>all;problem.GetParameterBlocks(&all);std::vector<std::uint8_t>eliminated(b.core.station.size(),0);std::size_t stations=0,remaining=0,fixed=0,at_lower=0,at_upper=0;std::size_t physical_tangent=0;
 for(double*x:all){const int tangent=problem.ParameterBlockTangentSize(x);if(problem.IsParameterBlockConstant(x)||tangent<=0){ordering->AddElementToGroup(x,2);++fixed;continue;}std::size_t i=0;if(station_index(x,&i)){const auto&r=b.core.packet.rows.at(i);require(r.active&&r.kind!=1&&problem.ParameterBlockSize(x)==1&&tangent==1,"invalid free station block");require(std::isfinite(*x),"nonfinite original station");const double lo=problem.GetParameterLowerBound(x,0),hi=problem.GetParameterUpperBound(x,0);require(lo<=*x&&*x<=hi,"station is outside its original Ceres bounds");at_lower+=*x==lo;at_upper+=*x==hi;eliminated.at(i)=1;ordering->AddElementToGroup(x,0);++stations;}else{require(physical.count(x)>0,"a nonphysical row/alpha/budget/enclosure is unexpectedly free");ordering->AddElementToGroup(x,1);++remaining;physical_tangent+=tangent;}}
 require(stations>0&&remaining>0,"nonempty independent station and remaining physical groups are required");
 std::vector<ceres::ResidualBlockId>residuals;problem.GetResidualBlocks(&residuals);std::vector<double*>parameters;parameters.reserve(64);std::size_t no_station=0,one_station=0,maximum=0;
 for(auto id:residuals){parameters.clear();problem.GetParameterBlocksForResidualBlock(id,&parameters);std::size_t count=0;for(const double*x:parameters){std::size_t i=0;if(station_index(x,&i)&&eliminated.at(i))++count;}maximum=std::max(maximum,count);require(count<=1,"two group0 stations share one actual residual block; exact independent elimination is invalid");count?++one_station:++no_station;}
 eventio::atomic_json(out/"actual_station_schur_ordering.json",{{"schema","rinkfit.actual_linearized_station_Schur_independent_set.v1"},{"all_actual_parameter_blocks",all.size()},{"group0_independent_free_scalar_stations",stations},{"group1_remaining_physical_blocks",remaining},{"group1_physical_tangent_dimensions",physical_tangent},{"group2_constant_blocks_removed_by_Ceres",fixed},{"original_station_at_lower_bound",at_lower},{"original_station_at_upper_bound",at_upper},{"all_actual_residual_blocks_inspected",residuals.size()},{"residuals_with_one_group0_station",one_station},{"residuals_with_no_group0_station",no_station},{"maximum_group0_stations_in_any_actual_residual",maximum},{"original_station_bounds_or_values_changed",false},{"fixed_stations_preserved",true},{"GPU_raw_UV_J_cache_reused",true},{"linear_solver","SPARSE_SCHUR"},{"sparse_library","EIGEN_SPARSE"},{"linear_factorization_on_GPU",false},{"nonlinear_closest_points_eliminated",false},{"passed",true}});
 return ordering;
}
}
