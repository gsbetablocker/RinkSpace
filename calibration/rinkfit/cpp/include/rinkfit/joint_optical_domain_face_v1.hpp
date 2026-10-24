#pragma once
#include "rinkfit/joint_capture_certified_zoom_v1.hpp"
#include "rinkfit/joint_solver_metric_v1.hpp"
#include <ceres/jet.h>
#include <cstring>
namespace rinkfit {
// One original closed Brown inequality, solved exactly in its dependent k1
// coordinate. This is a local active face, not a global lens parameterization.
// A trial without a root in the original boxes returns false; no coordinate
// clipping, fake gap, or softened certificate is used.
class JointOpticalDomainFaceManifoldV1 final:public ceres::Manifold {
 public:
  static constexpr int dependent=5;
  using V=std::array<double,12>;
 private:
  std::array<JointZoomCertificate,2>tests_;std::array<double,2>range_;
  V lower_,upper_;double level_;int endpoint_;
  std::vector<int>axes_;std::vector<double>axis_units_;
  bool original_box(const double*x)const{for(int i=0;i<12;++i)if(!std::isfinite(x[i])||(i>=5&&i<=10&&(x[i]<lower_[i]||x[i]>upper_[i])))return false;return true;}
 public:
  JointOpticalDomainFaceManifoldV1(const JointRig&rig,const std::array<double,2>&range,const EventParameterBlock&box,const ceres::Manifold*original,int endpoint,const double*base):range_(range),endpoint_(endpoint){
   if(endpoint<0||endpoint>1||box.values.size()!=12)throw std::runtime_error("optical face shape");
   std::copy_n(box.lower.data(),12,lower_.data());std::copy_n(box.upper.data(),12,upper_.data());
   for(int e=0;e<2;++e){auto&c=tests_[e];c.f_ref=rig.math.f_ref;c.radius=rig.math.radius_max;c.endpoint=e;c.wide_boundary=rig.math.wide_log_f_boundary;c.wide_coefficients=rig.math.wide_coefficients;}
   const int k=original?original->TangentSize():12;std::vector<double>A(12*k,0.);if(original){if(original->AmbientSize()!=12||!original->PlusJacobian(base,A.data()))throw std::runtime_error("optical original chart Jacobian");}else for(int i=0;i<12;++i)A[i*k+i]=1.;
   bool dep=false;for(int c=0;c<k;++c){int axis=-1;double unit=0;for(int a=0;a<12;++a)if(A[a*k+c]!=0){if(axis>=0)throw std::runtime_error("optical face requires original axis-aligned chart");axis=a;unit=A[a*k+c];}if(axis<0)throw std::runtime_error("optical original chart lost rank");if(axis==dependent){if(dep)throw std::runtime_error("duplicate optical dependent axis");dep=true;}else{axes_.push_back(axis);axis_units_.push_back(unit);}}
   if(!dep||axes_.empty()||!values(base,&level_)||level_>0||!original_box(base))throw std::runtime_error("optical face requires a feasible independent k1 coordinate");
   V outward;std::copy_n(base,12,outward.data());outward[dependent]=std::nextafter(base[dependent],lower_[dependent]);double outside;
   if(!(outward[dependent]<base[dependent])||!values(outward.data(),&outside)||outside<=0)throw std::runtime_error("optical face requires a real representable closed-domain boundary");
   std::array<double,12>gradient;if(!derivative(base,&gradient))throw std::runtime_error("optical face dependent derivative lost rank");
  }
  int endpoint()const{return endpoint_;}double level()const{return level_;}
  int AmbientSize()const override{return 12;}int TangentSize()const override{return int(axes_.size());}
  bool certificate(const double*x,int endpoint,double*out)const{const double*p[]={x,range_.data()};return tests_[endpoint](p,out)&&std::isfinite(*out);}
  bool values(const double*x,double*out)const{return certificate(x,endpoint_,out);}
  bool feasible(const double*x)const{if(!original_box(x))return false;for(int e=0;e<2;++e){double g;if(!certificate(x,e,&g)||g>0)return false;}return true;}
  bool derivative(const double*x,V*out)const{using J=ceres::Jet<double,12>;std::array<J,12>v;std::array<J,2>range;for(int i=0;i<12;++i){v[i].a=x[i];v[i].v.setZero();v[i].v[i]=1.;}for(int i=0;i<2;++i)range[i]=J(range_[i]);const J*p[]={v.data(),range.data()};J g;if(!tests_[endpoint_](p,&g)||!finite_scalar(g)||!g.v.allFinite())return false;double largest=g.v.cwiseAbs().maxCoeff();for(int i=0;i<12;++i)(*out)[i]=g.v[i];return g.v[dependent]< -128*std::numeric_limits<double>::epsilon()*std::max(1.,largest);}
  bool Plus(const double*x,const double*d,double*y)const override{
   bool zero=true;for(int c=0;c<TangentSize();++c){if(!std::isfinite(d[c]))return false;zero=zero&&d[c]==0.;}if(zero){std::memcpy(y,x,12*sizeof(double));return feasible(y);}
   std::copy_n(x,12,y);for(int c=0;c<TangentSize();++c)y[axes_[c]]+=axis_units_[c]*d[c];for(int i=0;i<12;++i)if(i!=dependent&&(!std::isfinite(y[i])||(i>=5&&i<=10&&(y[i]<lower_[i]||y[i]>upper_[i]))))return false;
   double lo=lower_[dependent],hi=upper_[dependent],glo,ghi;y[dependent]=lo;if(!values(y,&glo))return false;y[dependent]=hi;if(!values(y,&ghi)||glo<level_||ghi>level_)return false;
   if(glo==level_)hi=lo;else while(std::nextafter(lo,hi)<hi){const double mid=lo+(hi-lo)*.5;if(mid==lo||mid==hi)break;y[dependent]=mid;double g;if(!values(y,&g))return false;if(g>level_)lo=mid;else hi=mid;}
   y[dependent]=hi;V grad;return feasible(y)&&derivative(y,&grad);
  }
  bool PlusJacobian(const double*x,double*j)const override{V grad;if(!derivative(x,&grad))return false;const int k=TangentSize();std::fill_n(j,12*k,0.);for(int c=0;c<k;++c){j[axes_[c]*k+c]=axis_units_[c];j[dependent*k+c]=-grad[axes_[c]]*axis_units_[c]/grad[dependent];if(!std::isfinite(j[dependent*k+c]))return false;}return true;}
  bool Minus(const double*y,const double*x,double*d)const override{if(!feasible(y)||!feasible(x))return false;for(int c=0;c<TangentSize();++c){d[c]=(y[axes_[c]]-x[axes_[c]])/axis_units_[c];if(!std::isfinite(d[c]))return false;}V round;if(!Plus(x,d,round.data()))return false;for(int a=0;a<12;++a)if(std::abs(round[a]-y[a])>32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(round[a]),std::abs(y[a])}))return false;return true;}
  bool MinusJacobian(const double*,double*j)const override{std::fill_n(j,TangentSize()*12,0.);for(int c=0;c<TangentSize();++c){j[c*12+axes_[c]]=1./axis_units_[c];if(!std::isfinite(j[c*12+axes_[c]]))return false;}return true;}
};
}
