#pragma once
#include <ceres/problem.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rinkfit {
// Ceres rejects every zero-width box, including ambient coordinates removed by
// SubsetManifold. An existing fixed coordinate already enforces that equality.
// Keep the serialized physical bounds unchanged; verify the exact initial value
// and install only genuinely nonempty Ceres box intervals.
template<class Block>std::size_t install_joint_ceres_parameter_bounds(ceres::Problem&problem,Block&block){
 if(block.lower.size()!=block.values.size()||block.upper.size()!=block.values.size())throw std::runtime_error("physical box shape differs: "+block.name);
 std::size_t fixed_equalities=0;
 for(std::size_t j=0;j<block.values.size();++j){const double lo=block.lower[j],hi=block.upper[j],value=block.values[j];const bool fixed=std::find(block.fixed_indices.begin(),block.fixed_indices.end(),int(j))!=block.fixed_indices.end();
  if(std::isnan(lo)||std::isnan(hi)||lo>hi)throw std::runtime_error("invalid physical box: "+block.name+"/"+std::to_string(j));
  if(lo==hi){if(!fixed||!std::isfinite(lo)||value!=lo)throw std::runtime_error("zero-width physical box requires an exactly matching SubsetManifold fixed coordinate: "+block.name+"/"+std::to_string(j));++fixed_equalities;continue;}
  problem.SetParameterLowerBound(block.values.data(),int(j),lo);problem.SetParameterUpperBound(block.values.data(),int(j),hi);
 }
 return fixed_equalities;
}
} // namespace rinkfit
