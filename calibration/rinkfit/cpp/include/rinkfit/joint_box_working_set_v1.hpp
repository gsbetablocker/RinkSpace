#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_observed_branch_restart_v1.hpp"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <set>
#include <map>
#include <unordered_set>
namespace rinkfit {
// Solve on the current face of the existing parameter boxes. These are
// reversible numerical active constraints, never new physical equalities.
// Only affine, axis-aligned charts are changed; nonlinear optical charts and
// periodic curve stations retain their original manifold implementations.
class JointBoxFaceManifoldV1 final:public ceres::Manifold {
 Eigen::MatrixXd A_;
 public:explicit JointBoxFaceManifoldV1(Eigen::MatrixXd A):A_(std::move(A)){}
 int AmbientSize()const override{return int(A_.rows());}
 int TangentSize()const override{return int(A_.cols());}
 bool Plus(const double*x,const double*d,double*y)const override{Eigen::Map<Eigen::VectorXd>z(y,A_.rows());z=Eigen::Map<const Eigen::VectorXd>(x,A_.rows())+A_*Eigen::Map<const Eigen::VectorXd>(d,A_.cols());return z.allFinite();}
 bool PlusJacobian(const double*,double*j)const override{for(int i=0;i<A_.rows();++i)for(int k=0;k<A_.cols();++k)j[i*A_.cols()+k]=A_(i,k);return true;}
 bool Minus(const double*y,const double*x,double*d)const override{const auto v=Eigen::Map<const Eigen::VectorXd>(y,A_.rows())-Eigen::Map<const Eigen::VectorXd>(x,A_.rows());for(int k=0;k<A_.cols();++k)d[k]=A_.col(k).dot(v)/A_.col(k).squaredNorm();return true;}
 bool MinusJacobian(const double*,double*j)const override{for(int k=0;k<A_.cols();++k)for(int i=0;i<A_.rows();++i)j[k*A_.rows()+i]=A_(i,k)/A_.col(k).squaredNorm();return true;}
};
class JointBoxWorkingSetV1 {
 struct Held{double*p;const ceres::Manifold*original;bool made_constant;std::size_t station_raw;};
 std::vector<Held>held_;std::unordered_set<std::size_t>held_station_rows_;
 public:
 class ScopeRestoreV1{JointBoxWorkingSetV1*box_;ceres::Problem*problem_;public:ScopeRestoreV1(JointBoxWorkingSetV1*box,ceres::Problem*problem):box_(box),problem_(problem){}ScopeRestoreV1(const ScopeRestoreV1&)=delete;ScopeRestoreV1&operator=(const ScopeRestoreV1&)=delete;~ScopeRestoreV1(){if(box_)box_->restore(*problem_);}void release()noexcept{box_=nullptr;}};
 using Signature=std::set<std::pair<std::string,int>>;
 Signature signature;
 void restore(ceres::Problem&p){for(const auto&h:held_){if(h.made_constant)p.SetParameterBlockVariable(h.p);else p.SetManifold(h.p,const_cast<ceres::Manifold*>(h.original));}held_.clear();held_station_rows_.clear();}
 template<class Builder>eventio::Json prepare(Builder&b,ceres::Problem&p,int threads,bool removal_incidence_enabled=false){
  if(!held_.empty())throw std::runtime_error("box working set was not restored");
  try{
  struct Candidate{double*p;std::string name;int n,k;std::size_t offset;const ceres::Manifold*original;Eigen::MatrixXd A;std::vector<int>at_bound;std::size_t station_raw;};
  std::vector<Candidate>candidates;ceres::Problem::EvaluateOptions options;options.num_threads=threads;std::size_t offset=0;
  std::set<ceres::ResidualBlockId>residuals;
  auto consider=[&](double*x,const std::string&name,std::size_t station_raw=std::size_t(-1)){
   if(!p.HasParameterBlock(x)||p.IsParameterBlockConstant(x))return;
   const int n=p.ParameterBlockSize(x),k=p.ParameterBlockTangentSize(x);auto*original=p.GetManifold(x);
   if(original&&!dynamic_cast<const ceres::SubsetManifold*>(original))return;
   std::vector<int>at;for(int i=0;i<n;++i)if(x[i]<=p.GetParameterLowerBound(x,i)||x[i]>=p.GetParameterUpperBound(x,i))at.push_back(i);if(at.empty())return;
   Eigen::MatrixXd A=Eigen::MatrixXd::Identity(n,k);if(original){std::vector<double>j(n*k);if(!original->PlusJacobian(x,j.data()))throw std::runtime_error("box chart Jacobian failed");for(int i=0;i<n;++i)for(int c=0;c<k;++c)A(i,c)=j[i*k+c];}
   options.parameter_blocks.push_back(x);candidates.push_back({x,name,n,k,offset,original,std::move(A),std::move(at),station_raw});offset+=k;

  };
  for(auto&v:b.core.model.blocks)consider(v.values.data(),v.name);
  if(b.static_motion)for(std::size_t i=0;i<b.static_motion->rows.size();++i)consider(b.static_motion->rows[i].latent.data(),"static_motion/"+std::to_string(i));
  for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(r.active&&r.kind!=1&&joint_row_shape_vertex_path_v1(b.core.model,r))consider(&b.core.station[i],"station/"+std::to_string(i),i);}
  signature.clear();eventio::Json entries=eventio::Json::array();if(candidates.empty())return {{"active_coordinates",entries},{"original_parameter_boxes_changed",false}};
  std::size_t full_graph_scans=0,incidence_queries=0;
  if(removal_incidence_enabled){std::vector<ceres::ResidualBlockId>connected;for(const auto&v:candidates){connected.clear();p.GetResidualBlocksForParameterBlock(v.p,&connected);++incidence_queries;residuals.insert(connected.begin(),connected.end());}}
  else{std::unordered_set<const double*>wanted;for(const auto&v:candidates)wanted.insert(v.p);std::vector<ceres::ResidualBlockId>all;std::vector<double*>blocks;p.GetResidualBlocks(&all);++full_graph_scans;for(auto id:all){p.GetParameterBlocksForResidualBlock(id,&blocks);for(auto*x:blocks)if(wanted.count(x)){residuals.insert(id);break;}}}
  // Keep the previous residual pointer ordering and parameter tangent ordering.
  options.residual_blocks.assign(residuals.begin(),residuals.end());double cost=0;std::vector<double>gradient;
  if(!p.Evaluate(options,&cost,nullptr,&gradient,nullptr)||gradient.size()!=offset)throw std::runtime_error("box working set gradient evaluation failed");
  for(auto&v:candidates){std::set<int>remove;for(int i:v.at_bound){int column=-1;for(int c=0;c<v.k;++c)if(v.A(i,c)!=0){if(column!=-1)throw std::runtime_error("non-axis-aligned box chart");column=c;}if(column<0)continue;
    const double derivative=gradient[v.offset+column]/v.A(i,column),tol=128*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(derivative));const bool low=v.p[i]<=p.GetParameterLowerBound(v.p,i),high=v.p[i]>=p.GetParameterUpperBound(v.p,i);
    if((low&&high)||(low&&derivative>tol)||(high&&derivative<-tol)){remove.insert(column);signature.emplace(v.name,i);entries.push_back({{"block",v.name},{"coordinate",i},{"value",v.p[i]},{"side",low&&high?"fixed_interval":(low?"lower":"upper")},{"outward_objective_gradient",derivative}});}
   }
   if(remove.empty())continue;
   const int kept=v.k-int(remove.size());held_.push_back({v.p,v.original,kept==0,v.station_raw});if(v.station_raw!=std::size_t(-1))held_station_rows_.insert(v.station_raw);if(kept==0)p.SetParameterBlockConstant(v.p);else{Eigen::MatrixXd A(v.n,kept);int dest=0;for(int c=0;c<v.k;++c)if(!remove.count(c))A.col(dest++)=v.A.col(c);p.SetManifold(v.p,new JointBoxFaceManifoldV1(std::move(A)));}
  }
  return {{"schema","rinkfit.original_box_numerical_working_set.v1"},{"active_coordinates",entries},{"temporary_kind6_station_faces",held_station_rows_.size()},{"full_graph_incidence_scans",full_graph_scans},{"indexed_parameter_incidence_queries",incidence_queries},{"connected_residual_blocks",residuals.size()},{"original_parameter_boxes_changed",false},{"source_pixels_weights_changed",false},{"all_original_freedoms_restored_after_inner_solve",true},{"release_rechecked_after_active_face_convergence",true}};
  }catch(...){restore(p);throw;}
 }
 bool temporarily_holds_station_v1(std::size_t raw)const{return held_station_rows_.count(raw)!=0;}
 template<class Builder>bool station_face_should_activate_v1(Builder&b,ceres::Problem&p,std::size_t raw,const JointObservedActualStationSlopeV1&slope)const{auto*x=&b.core.station.at(raw);if(!slope.valid||temporarily_holds_station_v1(raw)||!p.HasParameterBlock(x)||p.IsParameterBlockConstant(x))return false;return (*x<=p.GetParameterLowerBound(x,0)&&slope.gradient>slope.tolerance)||(*x>=p.GetParameterUpperBound(x,0)&&slope.gradient< -slope.tolerance); }
 template<class Builder>bool station_face_invalidated_v1(Builder&b,ceres::Problem&p,bool removal_incidence_enabled){
  if(held_station_rows_.empty())return false;if(auto*cache=b.gpu_evaluation_callback())cache->PrepareForEvaluation(true,false);
  for(const auto&h:held_){if(h.station_raw==std::size_t(-1))continue;const auto&r=b.core.packet.rows.at(h.station_raw);const int direction=joint_shape_cell_inward_direction_v1(b.core.model,r,*h.p,r.numerical_shape_segment);if(direction==0&&p.GetParameterLowerBound(h.p,0)==p.GetParameterUpperBound(h.p,0))continue;const auto slope=joint_observed_actual_station_slope_v1(p,h.p,direction,removal_incidence_enabled,true);if(!slope.valid)throw std::runtime_error("temporary station box face has no actual complete-objective derivative");if(slope.descent)return true;}
  return false;
 }
};
}
