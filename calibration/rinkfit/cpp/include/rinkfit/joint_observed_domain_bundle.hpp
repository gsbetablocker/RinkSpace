#pragma once
#include "rinkfit/joint_explicit.hpp"
#include "rinkfit/joint_factor_failure_v1.hpp"
namespace rinkfit {
// Same scalar constraints and PHR branches, with one physical row layout per
// observation. All scalar residuals remain present and keep their original order.
class JointObservedDomainBundleCost final:public ceres::CostFunction {
 std::unique_ptr<JointExplicitRow>owned_row_;const JointExplicitRow*row_ptr_=nullptr;std::vector<int>kinds_;std::vector<double>lambda_;double rho_,sqrt_rho_;
 struct Workspace {std::vector<ceres::Jet<double,4>>zero;std::vector<const ceres::Jet<double,4>*>zp;std::vector<ceres::Jet<double,64>>jet;std::vector<const ceres::Jet<double,64>*>jp;std::vector<std::size_t>offset;std::vector<std::pair<int,int>>columns;std::vector<unsigned char>active;};
 public:
 JointObservedDomainBundleCost(JointExplicitRow row,const std::vector<int>&sizes,std::vector<int>kinds,std::vector<double>lambda,double rho):JointObservedDomainBundleCost(static_cast<const JointExplicitRow*>(nullptr),sizes,std::move(kinds),std::move(lambda),rho){owned_row_=std::make_unique<JointExplicitRow>(std::move(row));row_ptr_=owned_row_.get();}
 JointObservedDomainBundleCost(const JointExplicitRow*row,const std::vector<int>&sizes,std::vector<int>kinds,std::vector<double>lambda,double rho):row_ptr_(row),kinds_(std::move(kinds)),lambda_(std::move(lambda)),rho_(rho),sqrt_rho_(std::sqrt(rho)){
  if(kinds_.empty()||kinds_.size()!=lambda_.size()||!std::isfinite(rho)||rho<=0)throw std::invalid_argument("native constraint bundle shape/rho");for(double v:lambda_)if(!std::isfinite(v)||v<0)throw std::invalid_argument("native constraint multiplier");for(int n:sizes)if(n<1)throw std::invalid_argument("native constraint parameter size");*mutable_parameter_block_sizes()=sizes;set_num_residuals(int(kinds_.size()));
 }
 bool Evaluate(double const*const*p,double*r,double**J)const override{
  if(!row_ptr_)return joint_factor_failure_v1(row_ptr_,p,"native_domain_bundle","missing_source_row");const auto&row_=*row_ptr_;
  const auto&sizes=parameter_block_sizes();bool requested=false;if(J)for(std::size_t b=0;b<sizes.size();++b)requested|=J[b]!=nullptr;
  if(!requested){for(std::size_t k=0;k<kinds_.size();++k){double raw=0.;if(!JointExplicitDomain::evaluate_kind(row_,kinds_[k],p,&raw))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","scalar_value",std::isfinite(raw)?"evaluation_false":"nonfinite",kinds_[k]);if(!std::isfinite(raw))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","scalar_value","nonfinite",kinds_[k]);const double a=lambda_[k]+rho_*raw;if(!std::isfinite(a))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","PHR_value","nonfinite",kinds_[k]);r[k]=std::max(0.,a)/sqrt_rho_;}return true;}
  thread_local Workspace w;w.offset.resize(sizes.size());w.zp.resize(sizes.size());w.jp.resize(sizes.size());w.columns.clear();w.active.assign(kinds_.size(),0);std::size_t total=0;
  for(std::size_t b=0;b<sizes.size();++b){w.offset[b]=total;total+=sizes[b];if(J[b]){std::fill_n(J[b],num_residuals()*sizes[b],0.);for(int c=0;c<sizes[b];++c)w.columns.emplace_back(int(b),c);}}
  bool zero_ready=false;auto zero_parameters=[&](){if(zero_ready)return;w.zero.resize(total);for(std::size_t b=0;b<sizes.size();++b){w.zp[b]=w.zero.data()+w.offset[b];for(int c=0;c<sizes[b];++c){auto&x=w.zero[w.offset[b]+c];x.a=p[b][c];x.v.setZero();}}zero_ready=true;};
  bool any_active=false;
  for(std::size_t k=0;k<kinds_.size();++k){double original=0.;if(!JointExplicitDomain::evaluate_kind(row_,kinds_[k],p,&original))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","original_value",std::isfinite(original)?"evaluation_false":"nonfinite",kinds_[k]);if(!std::isfinite(original))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","original_value","nonfinite",kinds_[k]);double raw;bool cached=false;
   if(kinds_[k]==7){using S=ceres::Jet<double,1>;S station(row_.station_slot<0?row_.source->station:p[row_.station_slot][0]);station.v[0]=row_.station_slot<0?0.:1.;const S v=row_.source->station_domain(*row_.model,row_.layout.path,station);if(!std::isfinite(v.a)||!v.v.allFinite())return joint_factor_failure_v1(&row_,p,"native_domain_bundle","station_domain_Jet1","nonfinite",kinds_[k]);const double a=lambda_[k]+rho_*v.a;if(!std::isfinite(a))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","PHR_value","nonfinite",kinds_[k]);r[k]=std::max(0.,a)/sqrt_rho_;if(a>=0.&&row_.station_slot>=0&&J[row_.station_slot])J[row_.station_slot][k]=sqrt_rho_*v.v[0];continue;}
   if(kinds_[k]>=5&&row_.layout.width_slot<0){const double a=lambda_[k]-rho_;if(!std::isfinite(a))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","PHR_value","nonfinite",kinds_[k]);r[k]=std::max(0.,a)/sqrt_rho_;continue;}
   if(kinds_[k]>=0&&kinds_[k]<5&&row_.uv_cache){JointUVCacheView view;if(row_.uv_cache->lookup(row_.cache_row,&view)&&view.native_domain_available[kinds_[k]]){raw=view.canonical_AD_native_domains[kinds_[k]];row_.uv_cache->note_native_value_hit(true);cached=true;}}
   if(!cached){zero_parameters();ceres::Jet<double,4>value;if(!JointExplicitDomain::evaluate_kind(row_,kinds_[k],w.zp.data(),&value))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","zero_Jet4","evaluation_false",kinds_[k]);if(!std::isfinite(value.a)||!value.v.allFinite())return joint_factor_failure_v1(&row_,p,"native_domain_bundle","zero_Jet4","nonfinite",kinds_[k]);raw=value.a;}
   const double branch=lambda_[k]+rho_*raw;if(!std::isfinite(branch))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","PHR_branch","nonfinite",kinds_[k]);w.active[k]=branch>=0.;any_active|=w.active[k]!=0;r[k]=0.;
  }
  if(!any_active)return true;
  w.jet.resize(total);for(std::size_t b=0;b<sizes.size();++b)w.jp[b]=w.jet.data()+w.offset[b];
  for(std::size_t begin=0;begin<w.columns.size();begin+=64){for(std::size_t b=0;b<sizes.size();++b)for(int c=0;c<sizes[b];++c){auto&x=w.jet[w.offset[b]+c];x.a=p[b][c];x.v.setZero();}const auto count=std::min<std::size_t>(64,w.columns.size()-begin);for(std::size_t k=0;k<count;++k){const auto bc=w.columns[begin+k];w.jet[w.offset[bc.first]+bc.second].v[int(k)]=1.;}
   for(std::size_t k=0;k<kinds_.size();++k)if(w.active[k]){ceres::Jet<double,64>value;if(!JointExplicitDomain::evaluate_kind(row_,kinds_[k],w.jp.data(),&value))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","Jet64_Jacobian","evaluation_false",kinds_[k]);if(!std::isfinite(value.a)||!value.v.allFinite())return joint_factor_failure_v1(&row_,p,"native_domain_bundle","Jet64_Jacobian","nonfinite",kinds_[k]);const double a=lambda_[k]+rho_*value.a;if(!std::isfinite(a))return joint_factor_failure_v1(&row_,p,"native_domain_bundle","PHR_value","nonfinite",kinds_[k]);r[k]=std::max(0.,a)/sqrt_rho_;const double scale=a>=0.?sqrt_rho_:0.;for(std::size_t q=0;q<count;++q){const auto bc=w.columns[begin+q];J[bc.first][k*sizes[bc.first]+bc.second]=value.v[int(q)]*scale;}}
  }
  return true;
 }
};
}
