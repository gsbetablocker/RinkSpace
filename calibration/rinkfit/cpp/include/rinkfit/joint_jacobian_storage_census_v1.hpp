#pragma once
#include "rinkfit/event_io.hpp"
#include <ceres/problem.h>
#include <ceres/cost_function.h>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <map>
#include <limits>
namespace rinkfit {
// Required construction-time index-capacity check. Count the declared tangent
// block pattern, including currently constant blocks that working sets can
// release. No names/JSON, PHR state serialization, timing probe or dense J.
inline void assert_joint_ceres_int32_storage_capacity_v1(const ceres::Problem&p){
 using U=std::uint64_t;const U limit=std::numeric_limits<int>::max();U columns=0,rows=0,slots=0;std::vector<double*>parameters;p.GetParameterBlocks(&parameters);for(const auto*x:parameters){const int width=p.ParameterBlockTangentSize(x);if(width<0)throw std::runtime_error("invalid Ceres tangent block size");columns+=U(width);if(columns>limit)throw std::runtime_error("actual Ceres tangent column upper bound exceeds int32 index capacity");}
 std::vector<ceres::ResidualBlockId>residuals;p.GetResidualBlocks(&residuals);for(auto id:residuals){const auto*cost=p.GetCostFunctionForResidualBlock(id);const U m=U(cost->num_residuals());rows+=m;if(rows>limit)throw std::runtime_error("actual Ceres residual rows exceed int32 index capacity");parameters.clear();p.GetParameterBlocksForResidualBlock(id,&parameters);if(parameters.size()!=cost->parameter_block_sizes().size())throw std::runtime_error("Ceres factor block layout differs");for(std::size_t k=0;k<parameters.size();++k){const auto*x=parameters[k];if(p.ParameterBlockSize(x)!=cost->parameter_block_sizes()[k])throw std::runtime_error("Ceres factor ambient size differs");slots+=m*U(p.ParameterBlockTangentSize(x));if(slots>limit)throw std::runtime_error("actual Ceres tangent Jacobian pattern upper bound exceeds int32 index capacity");}}
}
// Count the actual declared block Jacobian, including structural zeros. GPU
// derivative sparsity does not alter Ceres' dense Jacobian block allocation.
// This does not allocate J, form a normal matrix, or certify solver performance.
inline eventio::Json joint_jacobian_storage_census_v1(const ceres::Problem&p,const std::map<const double*,std::string>&physical_names={}){
 using U=std::uint64_t;std::vector<double*>all;p.GetParameterBlocks(&all);U ambient=0,tangent=0,free_blocks=0,fixed_blocks=0,diagonal_slots=0;
 eventio::Json large=eventio::Json::array();std::vector<const double*>large_keys;for(const auto*x:all){const U a=p.ParameterBlockSize(x),t=p.ParameterBlockTangentSize(x);if(p.IsParameterBlockConstant(x)||!t){++fixed_blocks;continue;}++free_blocks;ambient+=a;tangent+=t;diagonal_slots+=t*t;const auto name=physical_names.find(x);if(name!=physical_names.end()&&a>64){large_keys.push_back(x);large.push_back({{"name",name->second},{"ambient",a},{"tangent",t},{"dense_diagonal_bytes",8*t*t}});}}
 U ambient_slots=0,tangent_slots=0,residual_dimensions=0,block_cells=0,max_ambient_row=0,max_tangent_row=0,normal_outer_multiplications=0;std::map<const double*,U>named_ambient,named_tangent;
 std::vector<ceres::ResidualBlockId>ids;p.GetResidualBlocks(&ids);std::vector<double*>blocks;blocks.reserve(64);
 for(auto id:ids){const auto*cost=p.GetCostFunctionForResidualBlock(id);const U m=cost->num_residuals();residual_dimensions+=m;blocks.clear();p.GetParameterBlocksForResidualBlock(id,&blocks);if(blocks.size()!=cost->parameter_block_sizes().size())throw std::runtime_error("actual Jacobian census parameter layout differs");U row_a=0,row_t=0;
  for(std::size_t k=0;k<blocks.size();++k){const auto*x=blocks[k];const U a=p.ParameterBlockSize(x),t=p.ParameterBlockTangentSize(x);if(a!=U(cost->parameter_block_sizes()[k]))throw std::runtime_error("actual Jacobian census ambient dimension differs");if(p.IsParameterBlockConstant(x)||!t)continue;++block_cells;row_a+=a;row_t+=t;ambient_slots+=m*a;tangent_slots+=m*t;if(physical_names.count(x)){named_ambient[x]+=m*a;named_tangent[x]+=m*t;}}
  max_ambient_row=std::max(max_ambient_row,row_a);max_tangent_row=std::max(max_tangent_row,row_t);normal_outer_multiplications+=m*row_t*row_t;
 }
 if(residual_dimensions!=U(p.NumResiduals()))throw std::runtime_error("actual Jacobian census residual denominator differs");
 U large_a=0,large_t=0;for(std::size_t i=0;i<large_keys.size();++i){auto&entry=large[i];const auto*x=large_keys[i];entry["declared_ambient_J_slots"]=named_ambient[x];entry["actual_tangent_J_slots"]=named_tangent[x];large_a+=named_ambient[x];large_t+=named_tangent[x];}
 const U limit=std::numeric_limits<int>::max();const bool safe=residual_dimensions<=limit&&tangent<=limit&&tangent_slots<=limit;
 return {{"schema","rinkfit.actual_Ceres_Jacobian_storage_census.v1"},{"actual_graph_inspected",true},{"residual_blocks",ids.size()},{"residual_dimensions",residual_dimensions},{"free_parameter_blocks",free_blocks},{"constant_parameter_blocks",fixed_blocks},{"free_ambient_dimensions",ambient},{"free_tangent_dimensions",tangent},{"dense_block_J_cells",block_cells},{"declared_ambient_J_scalar_slots",ambient_slots},{"actual_tangent_J_scalar_slots",tangent_slots},{"actual_tangent_J_values_bytes",8*tangent_slots},{"actual_tangent_J_CRS_indices_bytes_if_expanded",4*tangent_slots+4*(residual_dimensions+1)},{"two_J_value_copies_bytes",16*tangent_slots},{"large_physical_blocks_over_64_ambient",large},{"large_physical_ambient_J_slots",large_a},{"large_physical_tangent_J_slots",large_t},{"max_free_ambient_columns_in_one_residual_block",max_ambient_row},{"max_free_tangent_columns_in_one_residual_block",max_tangent_row},{"dense_parameter_block_diagonal_scalar_slots",diagonal_slots},{"dense_parameter_block_diagonal_bytes",8*diagonal_slots},{"potential_full_dense_row_outerproduct_multiplications",normal_outer_multiplications},{"ceres_int32_sparse_index_max",limit},{"ceres_int32_storage_dimensions_safe",safe},{"structural_zero_entries_are_still_counted",true},{"GPU_sparse_derivative_jobs_do_not_change_Ceres_block_pattern",true},{"Schur_fill_factor_and_workspace_bytes_not_included",true},{"numeric_J_allocated_or_profiled",false},{"performance_passed",false},{"quality_accepted",false}};
}
}
