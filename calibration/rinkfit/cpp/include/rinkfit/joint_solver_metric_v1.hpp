#pragma once
#include "rinkfit/joint_manifold.hpp"
#include <ceres/ceres.h>
#include <Eigen/Cholesky>
#include <map>
#include <memory>
#include <utility>
namespace rinkfit {
// These units change only the trust-region metric. No residual, parameter,
// source value, physical bound, or original manifold direction is added.
class JointSolverMetricManifoldV1 final:public ceres::Manifold {
 const ceres::Manifold*original_;std::vector<double>units_;int n_,k_;bool affine_;Eigen::MatrixXd B_,inverse_B_;
 void transforms(const double*x,Eigen::MatrixXd&B,Eigen::MatrixXd&inverse)const{
  std::vector<double>raw(std::size_t(n_)*k_);if(!original_->PlusJacobian(x,raw.data()))throw std::runtime_error("original numerical chart Jacobian failed");
  Eigen::MatrixXd normalized(n_,k_);for(int a=0;a<n_;++a)for(int t=0;t<k_;++t)normalized(a,t)=raw[a*k_+t]/units_[a];
  const Eigen::MatrixXd G=normalized.transpose()*normalized;Eigen::LLT<Eigen::MatrixXd>factor(G);if(factor.info()!=Eigen::Success)throw std::runtime_error("physical metric original tangent is not independent");
  inverse=factor.matrixU();B=factor.matrixU().solve(Eigen::MatrixXd::Identity(k_,k_));if(!B.allFinite()||!inverse.allFinite())throw std::runtime_error("physical metric has nonfinite units");
 }
 void at(const double*x,Eigen::MatrixXd&B,Eigen::MatrixXd&inverse)const{if(affine_){B=B_;inverse=inverse_B_;}else transforms(x,B,inverse);}
 public:
 JointSolverMetricManifoldV1(const ceres::Manifold*original,std::vector<double>units,const double*x):original_(original),units_(std::move(units)),n_(int(units_.size())),k_(original?original->TangentSize():n_),affine_(!original||dynamic_cast<const ceres::SubsetManifold*>(original)||dynamic_cast<const JointLinearControlManifold*>(original)){
  for(double u:units_)if(!std::isfinite(u)||u<=0)throw std::runtime_error("physical numerical units must be positive finite");if(n_<1||k_<1||(original_&&original_->AmbientSize()!=n_))throw std::runtime_error("physical metric chart dimension");if(original_&&affine_)transforms(x,B_,inverse_B_);
 }
 const ceres::Manifold*original()const{return original_;}int AmbientSize()const override{return n_;}int TangentSize()const override{return k_;}
 bool Plus(const double*x,const double*d,double*y)const override{try{if(!original_){for(int a=0;a<n_;++a){y[a]=x[a]+units_[a]*d[a];if(!std::isfinite(y[a]))return false;}return true;}Eigen::MatrixXd B,inverse;at(x,B,inverse);const Eigen::VectorXd delta=B*Eigen::Map<const Eigen::VectorXd>(d,k_);return original_->Plus(x,delta.data(),y);}catch(...){return false;}}
 bool PlusJacobian(const double*x,double*j)const override{try{if(!original_){std::fill_n(j,std::size_t(n_)*n_,0.);for(int a=0;a<n_;++a)j[a*n_+a]=units_[a];return true;}Eigen::MatrixXd B,inverse;at(x,B,inverse);std::vector<double>raw(std::size_t(n_)*k_);if(!original_->PlusJacobian(x,raw.data()))return false;for(int a=0;a<n_;++a)for(int t=0;t<k_;++t){double v=0;for(int c=0;c<k_;++c)v+=raw[a*k_+c]*B(c,t);j[a*k_+t]=v;}return true;}catch(...){return false;}}
 bool Minus(const double*y,const double*x,double*d)const override{try{if(!original_){for(int a=0;a<n_;++a){d[a]=(y[a]-x[a])/units_[a];if(!std::isfinite(d[a]))return false;}return true;}Eigen::MatrixXd B,inverse;at(x,B,inverse);Eigen::VectorXd base(k_);if(!original_->Minus(y,x,base.data()))return false;Eigen::Map<Eigen::VectorXd>result(d,k_);result=inverse*base;return result.allFinite();}catch(...){return false;}}
 bool MinusJacobian(const double*x,double*j)const override{try{if(!original_){std::fill_n(j,std::size_t(n_)*n_,0.);for(int a=0;a<n_;++a){j[a*n_+a]=1./units_[a];if(!std::isfinite(j[a*n_+a]))return false;}return true;}Eigen::MatrixXd B,inverse;at(x,B,inverse);std::vector<double>raw(std::size_t(k_)*n_);if(!original_->MinusJacobian(x,raw.data()))return false;for(int t=0;t<k_;++t)for(int a=0;a<n_;++a){double v=0;for(int c=0;c<k_;++c)v+=inverse(t,c)*raw[c*n_+a];j[t*n_+a]=v;if(!std::isfinite(v))return false;}return true;}catch(...){return false;}}
};
inline const ceres::Manifold*joint_solver_original_manifold_v1(const ceres::Manifold*m){if(const auto*units=dynamic_cast<const JointSolverMetricManifoldV1*>(m))return units->original();return m;}
class JointSolverMetricV1 {
 // Problem owns every installed manifold. Share Euclidean unit maps across all
 // identical blocks; retain chart-specific wrappers until this graph is retired.
 std::map<std::vector<double>,JointSolverMetricManifoldV1*>euclidean_;std::map<std::pair<double*,const ceres::Manifold*>,JointSolverMetricManifoldV1*>charts_;
 std::vector<std::pair<double*,const ceres::Manifold*>>installed_;
 void install(ceres::Problem&p,double*x,const std::vector<double>&units){if(!p.HasParameterBlock(x)||p.IsParameterBlockConstant(x))return;const auto*original=p.GetManifold(x);if(dynamic_cast<const JointSolverMetricManifoldV1*>(original))throw std::runtime_error("physical metric was not restored before preparing another face");if(!original&&std::all_of(units.begin(),units.end(),[](double u){return u==1.;}))return;
  JointSolverMetricManifoldV1*map=nullptr;if(!original){auto it=euclidean_.find(units);if(it!=euclidean_.end())map=it->second;}else{auto it=charts_.find({x,original});if(it!=charts_.end())map=it->second;}
  if(!map){auto pending=std::make_unique<JointSolverMetricManifoldV1>(original,units,x);map=pending.get();installed_.push_back({x,original});p.SetManifold(x,map);pending.release();if(!original)euclidean_.emplace(units,map);else charts_.emplace(std::make_pair(x,original),map);}else{installed_.push_back({x,original});p.SetManifold(x,map);}
 }
 public:
 class ScopeRestoreV1{JointSolverMetricV1*metric_;ceres::Problem*p_;public:ScopeRestoreV1(JointSolverMetricV1*metric,ceres::Problem*p):metric_(metric),p_(p){}~ScopeRestoreV1(){if(metric_)metric_->restore(*p_);}void release()noexcept{metric_=nullptr;}ScopeRestoreV1(const ScopeRestoreV1&)=delete;ScopeRestoreV1&operator=(const ScopeRestoreV1&)=delete;};
 void restore(ceres::Problem&p){for(const auto&v:installed_)p.SetManifold(v.first,const_cast<ceres::Manifold*>(v.second));installed_.clear();}
 void retire_graph(){if(!installed_.empty())throw std::runtime_error("retiring graph with live physical metric");euclidean_.clear();charts_.clear();}
 template<class Builder>eventio::Json prepare(Builder&b,ceres::Problem&p){if(!installed_.empty())throw std::runtime_error("physical metric prepare before restore");try{for(auto&block:b.core.model.blocks)install(p,block.values.data(),block.scales);if(!b.source_uv_observed_row_initialization())for(std::size_t i=0;i<b.core.packet.rows.size();++i)if(b.core.packet.rows[i].active)install(p,&b.core.row[i],{1079.});}catch(...){restore(p);throw;}
  return {{"schema","rinkfit.physical_parameter_trust_metric.v1"},{"model_units","existing_parameter_block_scales"},{"station_alpha_budget_logzoom_units",1.},{"motion_units",{1.,1.}},{"existing_auxiliary_manifolds_retained",true},{"sensor_row_units",1079.},{"jacobi_scaling",false},{"fixed_LM_diagonal",1.},{"scaled_parameter_blocks",installed_.size()},{"cached_Euclidean_unit_maps",euclidean_.size()},{"cached_original_chart_maps",charts_.size()},{"original_tangent_spaces_and_boxes_preserved",true},{"prior_residuals_added",0},{"objective_and_complete_primal_modified",false},{"bad_geometry_restored_or_observability_certified",false}};
 }
 static void configure(ceres::Solver::Options&options){options.jacobi_scaling=false;options.min_lm_diagonal=options.max_lm_diagonal=1.;}
};
}
