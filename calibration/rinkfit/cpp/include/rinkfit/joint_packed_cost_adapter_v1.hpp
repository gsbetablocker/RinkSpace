#pragma once
#include "rinkfit/joint_nuisance_component_pack_v1.hpp"
#include <atomic>
namespace rinkfit {
struct JointPackedTrialStatusV1 {std::atomic<bool>valid{false};std::string reason="trial has not been prepared";std::size_t prepared=0,rejected=0;};
// The original cost is called unchanged. Only parameter pointer views and dense
// block Jacobian column placement change; all W rows and residuals remain whole.
class JointPackedCostAdapterV1 final:public ceres::CostFunction {
 public:struct Slot {int target=0,offset=0,size=0;};
 private:const ceres::CostFunction*source_;std::shared_ptr<const ceres::CostFunction>owner_;std::shared_ptr<JointPackedTrialStatusV1>status_;std::vector<Slot>slots_;std::vector<double*>parameters_;
 public:
 JointPackedCostAdapterV1(const ceres::CostFunction*source,const std::vector<double*>&original,JointNuisanceComponentPackV1&pack,std::shared_ptr<JointPackedTrialStatusV1>status={},std::shared_ptr<const ceres::CostFunction>owner={}):source_(source),owner_(std::move(owner)),status_(std::move(status)){
  if(!source_||original.size()!=source_->parameter_block_sizes().size())throw std::runtime_error("packed cost original parameter inventory mismatch");set_num_residuals(source_->num_residuals());std::unordered_map<double*,int>targets;int eliminated_component=-1;
  for(std::size_t i=0;i<original.size();++i){JointNuisanceComponentPackV1::Location l;double*ptr=original[i];int offset=0,size=source_->parameter_block_sizes()[i],target_size=size;if(pack.locate(ptr,&l)){if(eliminated_component>=0&&eliminated_component!=l.component)throw std::runtime_error("new residual crosses two packed elimination components; rebuild complete incidence including W");eliminated_component=l.component;if(l.size!=size)throw std::runtime_error("packed original cost block size mismatch");ptr=pack.component_data(l.component);offset=l.offset;target_size=int(pack.components().at(l.component).values.size());}auto it=targets.find(ptr);if(it==targets.end()){int t=int(parameters_.size());targets[ptr]=t;parameters_.push_back(ptr);mutable_parameter_block_sizes()->push_back(target_size);it=targets.find(ptr);}else if(parameter_block_sizes()[it->second]!=target_size)throw std::runtime_error("packed cost duplicate target block has inconsistent size");slots_.push_back({it->second,offset,size});}
 }
 const auto&parameters()const{return parameters_;}const auto&slots()const{return slots_;}const ceres::CostFunction&original_cost()const{return *source_;}
 bool Evaluate(double const*const*p,double*r,double**J)const override{
  if(status_&&!status_->valid.load(std::memory_order_acquire))return false;std::vector<const double*>original;original.reserve(slots_.size());for(auto s:slots_)original.push_back(p[s.target]+s.offset);bool derivative=false;if(J)for(std::size_t i=0;i<parameter_block_sizes().size();++i)derivative|=J[i]!=nullptr;if(!derivative)return source_->Evaluate(original.data(),r,nullptr);
  const int m=num_residuals();std::size_t total=0;for(auto s:slots_)if(J[s.target])total+=std::size_t(m)*s.size;std::vector<double>storage(total);std::vector<double*>old_J(slots_.size(),nullptr);std::size_t at=0;for(std::size_t i=0;i<slots_.size();++i)if(J[slots_[i].target]){old_J[i]=storage.data()+at;at+=std::size_t(m)*slots_[i].size;}
  if(!source_->Evaluate(original.data(),r,old_J.data()))return false;for(std::size_t i=0;i<parameter_block_sizes().size();++i)if(J[i])std::fill_n(J[i],std::size_t(m)*parameter_block_sizes()[i],0.);for(std::size_t i=0;i<slots_.size();++i){auto s=slots_[i];if(!old_J[i])continue;const int width=parameter_block_sizes()[s.target];for(int a=0;a<m;++a)for(int j=0;j<s.size;++j)J[s.target][a*width+s.offset+j]+=old_J[i][a*s.size+j];}return true;
 }
};
// Explicit ownership assertion: the source Problem was constructed with
// cost_function_ownership=DO_NOT_TAKE_OWNERSHIP. It must outlive target because
// its original physical/manifold objects remain the authority. target owns the
// adapters; shared owners delete each transferred original cost exactly once.
struct JointSourceCostsAreUnownedV1 {};
inline eventio::Json transfer_joint_packed_costs_v1(ceres::Problem&source,ceres::Problem&target,JointNuisanceComponentPackV1&pack,const std::shared_ptr<JointPackedTrialStatusV1>&status,JointSourceCostsAreUnownedV1){
 if(!status)throw std::runtime_error("production packed graph requires an explicit whole-trial validity callback");pack.assert_incidence(source);pack.register_parameters(target);std::vector<ceres::ResidualBlockId>ids;source.GetResidualBlocks(&ids);const int before=source.NumResiduals();std::unordered_map<const ceres::CostFunction*,std::shared_ptr<const ceres::CostFunction>>owners;std::vector<double*>parameters;
 // All rejection checks finish before ownership is transferred or graph mutated.
 for(auto id:ids)if(source.GetLossFunctionForResidualBlock(id))throw std::runtime_error("same-objective PHR pack does not permit robust loss");
 for(auto id:ids){auto*cost=source.GetCostFunctionForResidualBlock(id);auto it=owners.find(cost);if(it==owners.end())it=owners.emplace(cost,std::shared_ptr<const ceres::CostFunction>(cost)).first;parameters.clear();source.GetParameterBlocksForResidualBlock(id,&parameters);auto adapter=std::make_unique<JointPackedCostAdapterV1>(cost,parameters,pack,status,it->second);auto target_parameters=adapter->parameters();target.AddResidualBlock(adapter.release(),nullptr,target_parameters);source.RemoveResidualBlock(id);}
 if(target.NumResiduals()!=before||source.NumResidualBlocks()!=0)throw std::runtime_error("packed graph transfer omitted an original residual");return {{"schema","rinkfit.same_objective_original_cost_transfer.v1"},{"residual_blocks",ids.size()},{"residual_scalars",before},{"original_cost_functions_transferred",owners.size()},{"W_or_residual_or_physical_math_changed",false},{"source_problem_retained_for_manifold_ownership",true}};
}
} // namespace rinkfit
