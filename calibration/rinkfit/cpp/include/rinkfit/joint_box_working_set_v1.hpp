#pragma once
#include "rinkfit/joint_al.hpp"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <set>
#include <map>
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
 struct Held{double*p;const ceres::Manifold*original;bool made_constant;};
 std::vector<Held>held_;
 public:
 using Signature=std::set<std::pair<std::string,int>>;
 Signature signature;
 void restore(ceres::Problem&p){for(const auto&h:held_){if(h.made_constant)p.SetParameterBlockVariable(h.p);else p.SetManifold(h.p,const_cast<ceres::Manifold*>(h.original));}held_.clear();}
 eventio::Json prepare(JointALBuilder&b,ceres::Problem&p,int threads){
  if(!held_.empty())throw std::runtime_error("box working set was not restored");
  struct Candidate{double*p;std::string name;int n,k;std::size_t offset;const ceres::Manifold*original;Eigen::MatrixXd A;std::vector<int>at_bound;};
  std::vector<Candidate>candidates;ceres::Problem::EvaluateOptions options;options.num_threads=threads;std::size_t offset=0;
  std::set<ceres::ResidualBlockId>residuals;
  auto consider=[&](double*x,const std::string&name){
   if(!p.HasParameterBlock(x)||p.IsParameterBlockConstant(x))return;
   const int n=p.ParameterBlockSize(x),k=p.ParameterBlockTangentSize(x);auto*original=p.GetManifold(x);
   if(original&&!dynamic_cast<const ceres::SubsetManifold*>(original))return;
   std::vector<int>at;for(int i=0;i<n;++i)if(x[i]<=p.GetParameterLowerBound(x,i)||x[i]>=p.GetParameterUpperBound(x,i))at.push_back(i);if(at.empty())return;
   Eigen::MatrixXd A=Eigen::MatrixXd::Identity(n,k);if(original){std::vector<double>j(n*k);if(!original->PlusJacobian(x,j.data()))throw std::runtime_error("box chart Jacobian failed");for(int i=0;i<n;++i)for(int c=0;c<k;++c)A(i,c)=j[i*k+c];}
   options.parameter_blocks.push_back(x);candidates.push_back({x,name,n,k,offset,original,std::move(A),std::move(at)});offset+=k;
   std::vector<ceres::ResidualBlockId>ids;p.GetResidualBlocksForParameterBlock(x,&ids);residuals.insert(ids.begin(),ids.end());
  };
  for(auto&v:b.core.model.blocks)consider(v.values.data(),v.name);
  if(b.static_motion)for(std::size_t i=0;i<b.static_motion->rows.size();++i)consider(b.static_motion->rows[i].latent.data(),"static_motion/"+std::to_string(i));
  signature.clear();eventio::Json entries=eventio::Json::array();if(candidates.empty())return {{"active_coordinates",entries},{"original_parameter_boxes_changed",false}};
  options.residual_blocks.assign(residuals.begin(),residuals.end());double cost=0;std::vector<double>gradient;
  if(!p.Evaluate(options,&cost,nullptr,&gradient,nullptr)||gradient.size()!=offset)throw std::runtime_error("box working set gradient evaluation failed");
  for(auto&v:candidates){std::set<int>remove;for(int i:v.at_bound){int column=-1;for(int c=0;c<v.k;++c)if(v.A(i,c)!=0){if(column!=-1)throw std::runtime_error("non-axis-aligned box chart");column=c;}if(column<0)continue;
    const double derivative=gradient[v.offset+column]/v.A(i,column),tol=128*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(derivative));const bool low=v.p[i]<=p.GetParameterLowerBound(v.p,i),high=v.p[i]>=p.GetParameterUpperBound(v.p,i);
    if((low&&derivative>tol)||(high&&derivative<-tol)){remove.insert(column);signature.emplace(v.name,i);entries.push_back({{"block",v.name},{"coordinate",i},{"value",v.p[i]},{"side",low?"lower":"upper"},{"outward_objective_gradient",derivative}});}
   }
   if(remove.empty())continue;
   const int kept=v.k-int(remove.size());held_.push_back({v.p,v.original,kept==0});if(kept==0)p.SetParameterBlockConstant(v.p);else{Eigen::MatrixXd A(v.n,kept);int dest=0;for(int c=0;c<v.k;++c)if(!remove.count(c))A.col(dest++)=v.A.col(c);p.SetManifold(v.p,new JointBoxFaceManifoldV1(std::move(A)));}
  }
  return {{"schema","rinkfit.original_box_numerical_working_set.v1"},{"active_coordinates",entries},{"original_parameter_boxes_changed",false},{"source_pixels_weights_changed",false},{"all_original_freedoms_restored_after_inner_solve",true},{"release_rechecked_after_active_face_convergence",true}};
 }
};
}
