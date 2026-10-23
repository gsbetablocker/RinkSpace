#pragma once
#include <ceres/cost_function.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>
namespace rinkfit {
// Concatenate scalar costs with identical parameter blocks. Their arithmetic,
// row ordering and PHR branches are unchanged; Ceres traverses one graph node.
class JointStackedScalarCost final:public ceres::CostFunction {
 std::vector<std::unique_ptr<ceres::CostFunction>> costs_;
 public:
 explicit JointStackedScalarCost(std::vector<std::unique_ptr<ceres::CostFunction>> costs):costs_(std::move(costs)){
  if(costs_.empty()||!costs_[0])throw std::invalid_argument("empty scalar stack");
  *mutable_parameter_block_sizes()=costs_[0]->parameter_block_sizes();
  for(const auto& c:costs_)if(!c||c->num_residuals()!=1||c->parameter_block_sizes()!=parameter_block_sizes())throw std::invalid_argument("scalar stack parameter incidence differs");
  set_num_residuals(int(costs_.size()));
 }
 bool Evaluate(double const*const* p,double* r,double** j)const override{
  if(!j){for(std::size_t i=0;i<costs_.size();++i)if(!costs_[i]->Evaluate(p,r+i,nullptr))return false;return true;}
  std::array<double*,32> local{};std::vector<double*> large;double** rows=local.data();
  const auto n=parameter_block_sizes().size();if(n>local.size()){large.resize(n);rows=large.data();}
  for(std::size_t i=0;i<costs_.size();++i){for(std::size_t k=0;k<n;++k)rows[k]=j[k]?j[k]+i*parameter_block_sizes()[k]:nullptr;if(!costs_[i]->Evaluate(p,r+i,rows))return false;}return true;
 }
};
}
