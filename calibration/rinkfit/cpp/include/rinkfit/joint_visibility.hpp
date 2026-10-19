#pragma once
#include "rinkfit/event_model.hpp"
namespace rinkfit {
// Same finite convex board and interior-face visibility used by native review.
// The near-board intersection uses an IFT derivative of its exact scalar root;
// the binary search is a numerical root finder, never a differentiated branch.
template<class T>struct JointVisibility {T front=T(-1),occlusion=T(-1);bool arithmetic_valid=true;bool visible(double tolerance=1e-8)const{return arithmetic_valid&&event_scalar(front)<=tolerance&&event_scalar(occlusion)<=tolerance;}};
template<class T>JointVisibility<T>joint_visibility(const T*g,const Vec3<T>&C,const Vec3<T>&X,bool board){
 JointVisibility<T>out;using std::abs;using std::sqrt;const auto sg=[](const T&x){return event_scalar(x)<0?T(-1):T(1);};
 const T ux=(abs(X[0])-(T(.5)-g[1]))/g[1],uy=(abs(X[1])-(g[0]/T(2)-g[2]))/g[2];
 const T nx=-sg(X[0])*event_pow_nonnegative(ux,g[3]-T(1))/g[1],ny=-sg(X[1])*event_pow_nonnegative(uy,g[3]-T(1))/g[2];
 if(board){const T n2=nx*nx+ny*ny;if(event_scalar(n2)<=1e-24){out.arithmetic_valid=false;return out;}out.front=-(nx*(C[0]-X[0])+ny*(C[1]-X[1]))/(sqrt(n2)*T(.5));}
 if(board&&event_scalar(out.front)>1e-8)return out;
 if(event_scalar(EventModel::rink_membership(C[0],C[1],g))>1e-10){
  double gd[5];for(int i=0;i<5;++i)gd[i]=event_scalar(g[i]);Vec3<double>cd,xd;for(int i=0;i<3;++i){cd[i]=event_scalar(C[i]);xd[i]=event_scalar(X[i]);}double a=0,b=1;for(int k=0;k<48;++k){double t=(a+b)*.5;double x=cd[0]+t*(xd[0]-cd[0]),y=cd[1]+t*(xd[1]-cd[1]);if(EventModel::rink_membership(x,y,gd)<=0)b=t;else a=t;}const double t0=(a+b)*.5;
  if(t0<1-1e-7){Vec3<T>entry;for(int i=0;i<3;++i)entry[i]=C[i]+T(t0)*(X[i]-C[i]);double u=std::max(0.,(std::abs(event_scalar(entry[0]))-(.5-gd[1]))/gd[1]),v=std::max(0.,(std::abs(event_scalar(entry[1]))-(gd[0]/2-gd[2]))/gd[2]);double dx=gd[3]*(u>0?std::pow(u,gd[3]-1):0)/gd[1]*(event_scalar(entry[0])<0?-1:1),dy=gd[3]*(v>0?std::pow(v,gd[3]-1):0)/gd[2]*(event_scalar(entry[1])<0?-1:1);double ft=dx*(xd[0]-cd[0])+dy*(xd[1]-cd[1]);if(!std::isfinite(ft)||std::abs(ft)<1e-12){out.arithmetic_valid=false;return out;}const T t=T(t0)-EventModel::rink_membership(entry[0],entry[1],g)/T(ft);const T z=C[2]+t*(X[2]-C[2]);out.occlusion=(g[4]-z)/T(.02);}
 }
 out.arithmetic_valid=finite_scalar(out.front)&&finite_scalar(out.occlusion);return out;
}
}
