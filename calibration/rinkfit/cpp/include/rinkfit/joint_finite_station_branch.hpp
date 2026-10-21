#pragma once
#include "rinkfit/joint_builder.hpp"
#include "rinkfit/joint_factors.hpp"
#include <ceres/ceres.h>
#include <memory>
namespace rinkfit {
// A numerical component of the unchanged source/material station union.
// This is a trial chart, never a new source correspondence or a clamp.
struct JointFiniteStationBranch {
 std::size_t raw=0;int path=0;double lower=0,upper=1,seed=0;
 std::vector<std::array<double,2>> source_components;
};
struct JointFiniteStationBranchPlan {
 std::vector<JointFiniteStationBranch> branches;
 eventio::Json failures=eventio::Json::array();
 std::size_t protected_rows=0,whole_unit_rows=0;
 bool valid()const{return failures.empty();}
 eventio::Json receipt()const{eventio::Json rows=eventio::Json::array();for(const auto&b:branches)if(b.lower!=0||b.upper!=1)rows.push_back({{"raw_row",b.raw},{"path",b.path},{"initial_station",b.seed},{"selected_component",{b.lower,b.upper}},{"all_original_components",b.source_components}});return {{"schema","rinkfit.same_source_finite_station_trial_component_plan.v1"},{"valid",valid()},{"planned_rows",branches.size()},{"protected_rows_not_mutated",protected_rows},{"whole_unit_rows",whole_unit_rows},{"restricted_rows",rows},{"failures",failures},{"source_intervals_or_UV_or_W_changed",false},{"invalid_seed_is_not_clamped",true},{"component_selection_only_from_current_legal_station",true}};}
};
inline std::vector<std::array<double,2>> joint_source_station_components(const JointObservation&o,const JointEventModel&m,int path){
 auto a=o.station_intervals(m,path);std::sort(a.begin(),a.end());std::vector<std::array<double,2>>u;
 for(const auto&v:a){if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[0]<0||v[1]>1||v[0]>v[1])throw std::runtime_error("invalid source/material station interval");
  if(u.empty()||v[0]>u.back()[1])u.push_back(v);else u.back()[1]=std::max(u.back()[1],v[1]);}
 return u;
}
inline JointFiniteStationBranchPlan inspect_joint_finite_station_branches(const JointProblemBuilder&core,const std::vector<std::size_t>&station_rows,const std::set<std::string>&protected_shots={}){
 JointFiniteStationBranchPlan result;std::set<std::size_t>seen;
 for(const auto raw:station_rows){if(!seen.insert(raw).second)throw std::runtime_error("duplicate station chart row");const auto&o=core.packet.rows.at(raw);
  if(!o.active||o.kind==1)throw std::runtime_error("finite station plan requires real active curve auxiliaries");
  auto components=joint_source_station_components(o,core.model,o.path);const double s=core.station.at(raw);std::vector<std::array<double,2>>containing;
  for(auto v:components)if(std::isfinite(s)&&s>=v[0]&&s<=v[1])containing.push_back(v);
  if(containing.size()!=1){result.failures.push_back({{"raw_row",raw},{"shot",o.shot_id},{"native_pts",o.pts},{"material",o.material},{"path",o.path},{"station",s},{"source_row_id",o.source_row_id},{"source_components",components},{"reason","no unique connected source component contains current station; genuine broad source/RS capture required before graph registration; no clamp or nearest-box substitution"}});continue;}
  if(protected_shots.count(o.shot_id)){++result.protected_rows;continue;}
  const auto v=containing.front();result.whole_unit_rows+=v[0]==0&&v[1]==1;result.branches.push_back({raw,o.path,v[0],v[1],s,components});
 }return result;
}
// Call after generic box registration in both existing graph builders.
// All inputs are validated before any Ceres bound is mutated.
inline void install_joint_finite_station_branches(ceres::Problem&problem,JointProblemBuilder&core,const JointFiniteStationBranchPlan&plan){
 if(!plan.valid())throw std::runtime_error("finite station component installation rejected: unresolved source capture rows");
 for(const auto&b:plan.branches){const auto&o=core.packet.rows.at(b.raw);const auto components=joint_source_station_components(o,core.model,o.path);if(o.path!=b.path||core.station[b.raw]!=b.seed||components!=b.source_components||!problem.HasParameterBlock(&core.station[b.raw]))throw std::runtime_error("finite station component plan/input changed");if(b.seed<b.lower||b.seed>b.upper)throw std::runtime_error("invalid finite station component seed");}
 for(const auto&b:plan.branches){auto*x=&core.station[b.raw];if(b.lower==b.upper){if(b.seed!=b.lower)throw std::runtime_error("zero-width source station mismatch");problem.SetParameterBlockConstant(x);}else{problem.SetParameterLowerBound(x,0,b.lower);problem.SetParameterUpperBound(x,0,b.upper);}}
}
struct JointFiniteStationTrialGuard {
 const JointEventModel*model=nullptr;const JointObservation*source=nullptr;JointFiniteStationBranch branch;
 std::vector<int>physical_slots;int station_slot=-1,row_slot=-1,alpha_slot=-1;bool require_query_closure=true;
 bool check(double const*const*p,std::array<double,3>*solution=nullptr)const{
  if(!model||!source||source->path!=branch.path)return false;const double s=station_slot<0?source->station:p[station_slot][0],r=p[row_slot][0];
  if(!std::isfinite(s)||s<branch.lower||s>branch.upper||!std::isfinite(r)||r<0||r>1079||!source->station_allowed(*model,branch.path,s))return false;
  JointObservation o=*source;o.station=s;o.row=r;o.path=branch.path;o.source_station_intervals={{branch.lower,branch.upper}};o.station_domain_path=branch.path;
  if(alpha_slot>=0)o.alpha=p[alpha_slot][0];if(!require_query_closure||o.kind==1)o.query.reset();
  JointRowEvaluation q;q.model=model;q.layout=o.layout(*model);q.observation=&o;q.require_physical_visibility=true;
  if(physical_slots.size()!=q.layout.block_ids.size())return false;std::vector<const double*>pp;for(std::size_t j=0;j<physical_slots.size();++j){if(physical_slots[j]<0)return false;pp.push_back(p[physical_slots[j]]);q.parameter_sizes[int(j)]=int(model->blocks.at(q.layout.block_ids[j]).values.size());}
  std::array<double,3>x;Eigen::Matrix3d J;if(!q.local_solution_uncached(pp.data(),s,&x,&J)||x[0]<branch.lower||x[0]>branch.upper)return false;
  if(solution)*solution=x;return true;
 }
};
// The existing residual/Jacobian is unchanged on a valid chart. Invalid
// stations, local RS/Query roots or visibility reject a Ceres trial via false;
// no row is dropped and no residual is replaced with zero.
class JointFiniteStationGuardedCost final:public ceres::CostFunction {
 std::unique_ptr<ceres::CostFunction>inner_;std::vector<JointFiniteStationTrialGuard>guards_;
 public:
 JointFiniteStationGuardedCost(std::unique_ptr<ceres::CostFunction>inner,std::vector<JointFiniteStationTrialGuard>guards):inner_(std::move(inner)),guards_(std::move(guards)){if(!inner_)throw std::runtime_error("empty finite-station guarded cost");set_num_residuals(inner_->num_residuals());*mutable_parameter_block_sizes()=inner_->parameter_block_sizes();}
 bool Evaluate(double const*const*p,double*r,double**J)const override{for(const auto&g:guards_)if(!g.check(p))return false;return inner_->Evaluate(p,r,J);}
};
inline std::unique_ptr<ceres::CostFunction> guard_joint_finite_station_trial(std::unique_ptr<ceres::CostFunction>cost,std::vector<JointFiniteStationTrialGuard>guards){return std::make_unique<JointFiniteStationGuardedCost>(std::move(cost),std::move(guards));}
} // namespace rinkfit
