#pragma once
#include "rinkfit/joint_explicit.hpp"
#include <ceres/manifold.h>
namespace rinkfit {
// Coordinates for the interior of the EXISTING closed-disk Brown certificate.
// No lens parameter is fixed, no pixel term or physical bound is changed.
// Endpoint gaps use a monotone local retraction with unit derivative at zero.
// Squaring a signed step reflects negative proposals and explodes positive ones.
// k2 at each zoom endpoint is represented by its positive gap above the exact
// radial/tangential quadratic lower envelope. Tangential coefficients use an
// open disk. This gives Ceres derivatives along the feasible region instead
// of repeatedly rejecting an unconstrained direction at the global gate.
class JointOpticalFeasibleCoordinates final : public ceres::Manifold {
 double radius_,z0_,z1_,pmax_,margin_=1e-8;bool common_lower_;std::vector<int> free_;
 static double scalar(double x){return x;}template<int N>static double scalar(const ceres::Jet<double,N>&x){return x.a;}
 template<class T>T k2_lower(const T& k1,const T& c)const{
  const T q=T(radius_*radius_);
  auto bound=[&](const T&a){
   if(a<T(0)&&-T(2)*c/a<q)return a*a/(T(4)*c);
   return (-c-a*q)/(q*q);
  };
  const T tangential=bound(k1),radial=bound(T(3)*k1)/T(5);
  return tangential>radial?tangential:radial;
 }
 template<class T>bool lower_at_margin(const T*x,const T&c,T*lo,T*hi)const{
  if(!(c>T(0)))return false;
  const T a=x[5]+T(z0_)*x[6],b=x[5]+T(z1_)*x[6];
  if(common_lower_){*lo=*hi=k2_lower(a<b?a:b,c);}
  else{*lo=k2_lower(a,c);*hi=k2_lower(b,c);}
  return finite_scalar(*lo)&&finite_scalar(*hi);
 }
 template<class T>bool lower(const T*x,T*lo,T*hi)const{
  using std::sqrt;const T p2=x[9]*x[9]+x[10]*x[10];
  const T pn=p2>T(0)?sqrt(p2):T(0);
  return lower_at_margin(x,T(1.-margin_)-T(6*radius_)*pn,lo,hi);
 }
 static double gap_plus(double gap,double delta){return std::max(0.,gap+delta); }
 template<class T>bool chart(const T*x,T*q)const{
  for(int i=0;i<12;++i)q[i]=x[i];
  using std::sqrt;using std::log;
  const T d=T(pmax_*pmax_)-x[9]*x[9]-x[10]*x[10];if(!(d>T(0)))return false;
  q[9]=x[9]/sqrt(d);q[10]=x[10]/sqrt(d);
  T lo,hi;if(!lower(x,&lo,&hi))return false;
  T a=x[7]+T(z0_)*x[8]-lo,b=x[7]+T(z1_)*x[8]-hi;
  const double tol=64*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(scalar(lo)),std::abs(scalar(hi))});
  if(scalar(a)<-tol||scalar(b)<-tol)return false;
  if(scalar(a)<0)a-=T(scalar(a));if(scalar(b)<0)b-=T(scalar(b));
  q[7]=a;q[8]=b;return true;
 }
 template<class T>bool ambient(const T*q,T*x)const{
  for(int i=0;i<12;++i)x[i]=q[i];
  using std::sqrt;using std::exp;
  const T d=sqrt(T(1)+q[9]*q[9]+q[10]*q[10]);x[9]=T(pmax_)*q[9]/d;x[10]=T(pmax_)*q[10]/d;
  const T qn2=q[9]*q[9]+q[10]*q[10],qn=qn2>T(0)?sqrt(qn2):T(0);
  const T c=T(1.-margin_)/(d*(d+qn));
  T lo,hi;if(!lower_at_margin(x,c,&lo,&hi))return false;
  if(q[7]<T(0)||q[8]<T(0))return false;
  const T a=lo+q[7],b=hi+q[8];
  x[8]=(b-a)/T(z1_-z0_);x[7]=a-T(z0_)*x[8];
  for(int i=0;i<12;++i)if(!finite_scalar(x[i]))return false;return true;
 }
 public:
 JointOpticalFeasibleCoordinates(const JointRig&r,const std::array<double,2>&range,const EventParameterBlock& block,std::array<bool,2> active={false,false})
 :radius_(r.math.radius_max),z0_(range[0]-std::log(r.math.f_ref)),z1_(range[1]-std::log(r.math.f_ref)),
  pmax_((1.-1e-8)/(6*radius_)),common_lower_(false){
  for(int i=0;i<12;++i)if(i<7||i>8||!active[i-7])free_.push_back(i);
  const bool extension=std::any_of(r.math.wide_coefficients.begin(),r.math.wide_coefficients.end(),[](double v){return v!=0.;});
  if(!(radius_>0)||!(z1_>z0_)||(extension&&range[0]<r.math.wide_log_f_boundary))throw std::runtime_error("feasible optical coordinates require a nondegenerate affine zoom interval");
  // Preserve the pre-existing conservative mixed-coefficient enclosure for
  // the inactive wide-extension branch, including its floating padding.
  common_lower_=extension;
  {
   const double z=std::max(std::abs(z0_),std::abs(z1_)),q=radius_*radius_;
   auto magnitude=[&](int i){return std::max(std::abs(block.lower.at(i)),std::abs(block.upper.at(i)));};
   const double k1=magnitude(5)+z*magnitude(6),k2=magnitude(7)+z*magnitude(8);
   // Enclose floating cancellation at the CLOSED Brown boundary. The
   // original predicate and physical boxes are still evaluated unchanged.
   margin_+=64*std::numeric_limits<double>::epsilon()*(2.+k1*q+k2*q*q);
   if(common_lower_)margin_+=1e-10*(2.+k1*q+k2*q*q);
   pmax_=(1.-margin_)/(6*radius_);
  }
 }
 int AmbientSize()const override{return 12;}int TangentSize()const override{return int(free_.size());}
 std::array<bool,2> active_endpoint_gaps()const{return {std::find(free_.begin(),free_.end(),7)==free_.end(),std::find(free_.begin(),free_.end(),8)==free_.end()};}
 bool endpoint_gaps(const double*x,std::array<double,2>*g)const{double q[12];if(!chart(x,q))return false;*g={q[7],q[8]};return true;}
 bool with_endpoint_gaps(const double*x,const std::array<double,2>&g,double*y)const{double q[12];if(!chart(x,q))return false;q[7]=g[0];q[8]=g[1];return ambient(q,y);}
 std::array<std::array<double,2>,2> gap_coefficient_jacobian()const{return {{{z1_/(z1_-z0_),-z0_/(z1_-z0_)},{-1./(z1_-z0_),1./(z1_-z0_)}}};}
 bool Plus(const double*x,const double*d,double*y)const override{
  double q[12];if(!chart(x,q))return false;bool zero=true;for(int i=0;i<TangentSize();++i)zero=zero&&d[i]==0.;if(zero){std::copy(x,x+12,y);return true;}for(int j=0;j<TangentSize();++j){int i=free_[j];q[i]=(i==7||i==8)?gap_plus(q[i],d[j]):q[i]+d[j];}return ambient(q,y);
 }
 bool Minus(const double*y,const double*x,double*d)const override{
  double a[12],b[12];if(!chart(x,a)||!chart(y,b))return false;for(int i=0;i<TangentSize();++i)d[i]=b[free_[i]]-a[free_[i]];return true;
 }
 bool PlusJacobian(const double*x,double*out)const override{
  double q[12];if(!chart(x,q))return false;using J=ceres::Jet<double,12>;J a[12],b[12];
  for(int i=0;i<12;++i){a[i].a=q[i];a[i].v.setZero();a[i].v[i]=1.;}if(!ambient(a,b))return false;
  for(int i=0;i<12;++i)for(int j=0;j<TangentSize();++j){out[TangentSize()*i+j]=b[i].v[free_[j]];if(!std::isfinite(out[TangentSize()*i+j]))return false;}return true;
 }
 bool MinusJacobian(const double*x,double*out)const override{
  using J=ceres::Jet<double,12>;J a[12],b[12];for(int i=0;i<12;++i){a[i].a=x[i];a[i].v.setZero();a[i].v[i]=1.;}if(!chart(a,b))return false;
  for(int i=0;i<TangentSize();++i)for(int j=0;j<12;++j){out[12*i+j]=b[free_[i]].v[j];if(!std::isfinite(out[12*i+j]))return false;}return true;
 }
};
}
