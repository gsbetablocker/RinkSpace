#pragma once
#include "rinkfit/joint_al.hpp"
#include <unordered_map>
#include "rinkfit/joint_factor_failure_v1.hpp"
namespace rinkfit {
// Three points on one carrier. The fifth independent coordinate is C.x;
// C.y is derived exactly. Numerical box faces are a reversible working set,
// never a permanent physical lock or an invalid-residual-only constraint.
class JointCollinearCarrierV1 final:public ceres::Manifold {
 std::vector<int> free_;
 std::vector<double> lower_,upper_;
 bool project_box(double*q)const{
  if(lower_.empty())return true;
  // Project the independent chart before reconstructing C.y. Clipping the
  // six ambient coordinates independently would bend the physical carrier.
  for(int i=0;i<5;++i)q[i]=std::clamp(q[i],lower_[i],upper_[i]);
  const double dx=q[2]-q[0],dy=q[3]-q[1];
  if(dx==0)return false;
  // C must remain on AB and in its original rectangle. Its x interval is
  // the intersection of the original x bounds and the two y-face crossings.
  double lo=lower_[4],hi=upper_[4];
  if(dy!=0){double a=q[0]+(lower_[5]-q[1])*dx/dy,b=q[0]+(upper_[5]-q[1])*dx/dy;if(a>b)std::swap(a,b);lo=std::max(lo,a);hi=std::min(hi,b);}
  else if(q[1]<lower_[5]||q[1]>upper_[5])return false;
  if(lo>hi)return false;
  q[4]=std::clamp(q[4],lo,hi);
  // At a derived y face, round toward the feasible interval rather than
  // rejecting an otherwise legal step for an arithmetic ulp.
  for(int i=0;i<4;++i){double cy=q[1]+(q[4]-q[0])*dy/dx;if(cy>=lower_[5]&&cy<=upper_[5])return true;q[4]=std::nextafter(q[4],cy<lower_[5]?(dy/dx>0?hi:lo):(dy/dx>0?lo:hi));}
  return false;
 }
 template<class T>static bool chart(const T*x,T*q){for(int i=0;i<5;++i)q[i]=x[i];return finite_scalar(q[0])&&finite_scalar(q[1])&&finite_scalar(q[2])&&finite_scalar(q[3])&&finite_scalar(q[4])&&q[2]!=q[0];}
 template<class T>static bool ambient(const T*q,T*x){for(int i=0;i<5;++i)x[i]=q[i];if(q[2]==q[0])return false;x[5]=q[1]+(q[4]-q[0])*(q[3]-q[1])/(q[2]-q[0]);for(int i=0;i<6;++i)if(!finite_scalar(x[i]))return false;return true;}
 public:
 explicit JointCollinearCarrierV1(const std::vector<int>&held={},const EventParameterBlock*box=nullptr){if(box){lower_=box->lower;upper_=box->upper;}for(int i=0;i<5;++i)if(std::find(held.begin(),held.end(),i)==held.end())free_.push_back(i);if(free_.empty())throw std::runtime_error("empty carrier face");}
 int AmbientSize()const override{return 6;}int TangentSize()const override{return int(free_.size());}
 static bool collinear(const double*x){const double vx=x[2]-x[0],vy=x[3]-x[1],wx=x[4]-x[0],wy=x[5]-x[1];return vx*vx+vy*vy>0&&std::abs(vx*wy-vy*wx)<=128*std::numeric_limits<double>::epsilon()*std::max(1.,std::hypot(vx,vy)*std::hypot(wx,wy));}
 bool Plus(const double*x,const double*d,double*y)const override{double q[5];if(!chart(x,q))return false;bool zero=true;for(int i=0;i<TangentSize();++i){zero=zero&&d[i]==0;q[free_[i]]+=d[i];}if(zero){std::copy(x,x+6,y);return true;}if(project_box(q)&&ambient(q,y))return true;
  // Ceres also calls Plus(x,-gradient) to measure a projected gradient.
  // That vector is not a trust-region proposal and can be enormous. Do not
  // collapse A and B to the same box corner and report a residual failure.
  double magnitude=0;for(int j=0;j<TangentSize();++j){if(!std::isfinite(d[j]))return false;magnitude=std::max(magnitude,std::abs(d[j]));}
  double width=1;if(!lower_.empty()){width=0;for(int j=0;j<5;++j)width=std::max(width,upper_[j]-lower_[j]);}
  double scale=std::min(.5,.25*width/std::max(magnitude,std::numeric_limits<double>::min()));
  for(;;){double next[5];if(!chart(x,next))return false;bool moved=false;for(int j=0;j<TangentSize();++j){next[free_[j]]+=scale*d[j];moved=moved||next[free_[j]]!=x[free_[j]];}if(!moved){std::copy(x,x+6,y);return true;}if(project_box(next)&&ambient(next,y))return true;scale*=.5;}
 }
 bool Minus(const double*y,const double*x,double*d)const override{double a[5],b[5];if(!chart(x,a)||!chart(y,b))return false;for(int i=0;i<TangentSize();++i)d[i]=b[free_[i]]-a[free_[i]];return true;}
 bool PlusJacobian(const double*x,double*J)const override{double q[5];if(!chart(x,q))return false;using T=ceres::Jet<double,5>;T a[5],b[6];for(int j=0;j<5;++j){a[j].a=q[j];a[j].v.setZero();a[j].v[j]=1;}if(!ambient(a,b))return false;for(int i=0;i<6;++i)for(int j=0;j<TangentSize();++j)J[TangentSize()*i+j]=b[i].v[free_[j]];return true;}
 bool MinusJacobian(const double*,double*J)const override{std::fill(J,J+6*TangentSize(),0.);for(int j=0;j<TangentSize();++j)J[6*j+free_[j]]=1;return true;}
};
class JointCollinearCarrierBoxV1 final:public ceres::SizedCostFunction<1,6>{std::vector<double>lo_,hi_;const EventParameterBlock*origin_;public:explicit JointCollinearCarrierBoxV1(const EventParameterBlock&b):lo_(b.lower),hi_(b.upper),origin_(&b){}bool Evaluate(double const*const*x,double*r,double**J)const override{for(int i=0;i<6;++i)if(!std::isfinite(x[0][i])||x[0][i]<lo_.at(i)||x[0][i]>hi_.at(i))return joint_actual_factor_failure_v1([&](eventio::Json&record){record["factor"]="collinear_carrier_box";record["component"]=i;record["lower"]=lo_.at(i);record["upper"]=hi_.at(i);append_joint_actual_failure_parameter_v1(record,origin_->values.data(),x[0],6,origin_->name);});r[0]=0;if(J&&J[0])std::fill(J[0],J[0]+6,0.);return true;}};
class JointCollinearBoxWorkingSetV1 {
 struct Item{int block;std::vector<int> held,pending;std::vector<ceres::ResidualBlockId> residuals;std::array<double,5> gradient{};};
 std::vector<Item> items_;
 static std::array<double,5> gradient(ceres::Problem&p,double*x,const std::vector<ceres::ResidualBlockId>&ids){std::array<double,6>g{};
  for(auto id:ids){const auto*f=p.GetCostFunctionForResidualBlock(id);std::vector<double*>blocks;p.GetParameterBlocksForResidualBlock(id,&blocks);std::vector<const double*>ptr(blocks.begin(),blocks.end());std::vector<double*>jac(blocks.size(),nullptr);const auto it=std::find(blocks.begin(),blocks.end(),x);if(it==blocks.end())throw std::runtime_error("carrier gradient consumer changed");std::vector<double>r(f->num_residuals()),J(6*f->num_residuals());jac[it-blocks.begin()]=J.data();if(!f->Evaluate(ptr.data(),r.data(),jac.data()))throw std::runtime_error("carrier face gradient failed at accepted state");double w=1;if(const auto*loss=p.GetLossFunctionForResidualBlock(id)){double rr=0,rho[3];for(auto v:r)rr+=v*v;loss->Evaluate(rr,rho);w=rho[1];}for(int k=0;k<f->num_residuals();++k)for(int j=0;j<6;++j)g[j]+=w*r[k]*J[6*k+j];}
  JointCollinearCarrierV1 full;double M[30];if(!full.PlusJacobian(x,M))throw std::runtime_error("singular carrier coordinate");std::array<double,5>q{};for(int j=0;j<5;++j)for(int k=0;k<6;++k)q[j]+=g[k]*M[5*k+j];return q;
 }
 public:
 JointCollinearBoxWorkingSetV1(JointALBuilder&b,ceres::Problem&p){
  std::unordered_map<const double*,std::size_t> wanted;
  for(int i=0;i<int(b.core.model.blocks.size());++i){auto*x=b.core.model.blocks[i].values.data();if(p.HasParameterBlock(x)&&!p.IsParameterBlockConstant(x)&&dynamic_cast<const JointCollinearCarrierV1*>(p.GetManifold(x))){wanted.emplace(x,items_.size());Item a;a.block=i;items_.push_back(std::move(a));}}
  if(wanted.empty())return;
  std::vector<ceres::ResidualBlockId> ids;std::vector<double*>blocks;p.GetResidualBlocks(&ids);
  for(auto id:ids){p.GetParameterBlocksForResidualBlock(id,&blocks);for(auto*x:blocks){const auto it=wanted.find(x);if(it!=wanted.end())items_[it->second].residuals.push_back(id);}}
 }
 // Local factor replacement keeps all parameter addresses and carrier charts.
 // Preserve incidence ordering while replacing the exact retired factor IDs.
 struct PreparedResidualRemapV1{std::vector<std::vector<ceres::ResidualBlockId>> residuals;};
 PreparedResidualRemapV1 prepare_remap_residuals_v1(const std::vector<std::pair<ceres::ResidualBlockId,ceres::ResidualBlockId>>& replacements)const{
  std::unordered_map<ceres::ResidualBlockId,ceres::ResidualBlockId> ids;
  for(const auto&r:replacements){if(!r.first||!r.second)throw std::runtime_error("null carrier residual replacement");const auto inserted=ids.emplace(r.first,r.second);if(!inserted.second&&inserted.first->second!=r.second)throw std::runtime_error("conflicting carrier residual replacement");}
  PreparedResidualRemapV1 result;result.residuals.reserve(items_.size());
  for(const auto&a:items_){result.residuals.push_back(a.residuals);for(auto&id:result.residuals.back()){const auto it=ids.find(id);if(it!=ids.end())id=it->second;}}
  return result;
 }
 void commit_remap_residuals_v1(PreparedResidualRemapV1&&prepared)noexcept{for(std::size_t i=0;i<items_.size();++i)items_[i].residuals.swap(prepared.residuals[i]);}
 void remap_residuals_v1(const std::vector<std::pair<ceres::ResidualBlockId,ceres::ResidualBlockId>>& replacements){commit_remap_residuals_v1(prepare_remap_residuals_v1(replacements));}
 bool empty()const{return items_.empty();}
 bool changed(JointALBuilder&b,ceres::Problem&p,bool new_point=false){bool any=false,prepared=false;for(auto&a:items_){const auto&v=b.core.model.blocks[a.block];bool near_boundary=false;for(int j=0;j<5;++j)near_boundary=near_boundary||v.values[j]==v.lower[j]||v.values[j]==v.upper[j];a.pending.clear();if(near_boundary){if(!prepared){if(auto*c=b.gpu_evaluation_callback())c->PrepareForEvaluation(true,new_point);prepared=true;}a.gradient=gradient(p,const_cast<double*>(v.values.data()),a.residuals);for(int j=0;j<5;++j){const double g=a.gradient[j];if(!std::isfinite(g))throw std::runtime_error("nonfinite carrier KKT gradient");if((v.values[j]==v.upper[j]&&g<0)||(v.values[j]==v.lower[j]&&g>0))a.pending.push_back(j);}}any=any||a.pending!=a.held;}return any;}
 eventio::Json install(JointALBuilder&b,ceres::Problem&p){eventio::Json rows=eventio::Json::array();for(auto&a:items_){auto&v=b.core.model.blocks[a.block];const auto previous=a.held;if(a.pending!=a.held){p.SetManifold(v.values.data(),new JointCollinearCarrierV1(a.pending,&v));a.held=a.pending;}rows.push_back({{"block",a.block},{"name",v.name},{"active_box_coordinates",a.held},{"previous_active",previous},{"full_chart_gradient",a.gradient},{"tangent",p.ParameterBlockTangentSize(v.values.data())},{"physical_position_and_direction_free",true},{"source_or_primal_values_changed",false},{"reconsidered_at_each_accepted_step",true}});}return {{"schema","rinkfit.collinear_box_working_set.v1"},{"groups",rows}};}
};
}
