#pragma once
#include "core_types.hpp"
namespace rinkspace {
/* Scalar directional Jet reused from calibration/rinkfit/cpp/gpu_camera/exact_math.cuh.
 * No old geometry, camera model, guards or runtime are imported. */
struct LocalJet {double a,v;RS_HD LocalJet(double x=0):a(x),v(0){}RS_HD LocalJet(double x,double d):a(x),v(d){}
RS_HD LocalJet& operator+=(LocalJet b){a+=b.a;v+=b.v;return *this;}
RS_HD LocalJet& operator-=(LocalJet b){a-=b.a;v-=b.v;return *this;}
RS_HD LocalJet& operator*=(LocalJet b){v=v*b.a+a*b.v;a*=b.a;return *this;}
RS_HD LocalJet& operator/=(LocalJet b){const double inverse=1./b.a,ratio=a*inverse;v=(v-ratio*b.v)*inverse;a=ratio;return *this;}};
RS_HD inline LocalJet operator+(LocalJet a,LocalJet b){return a+=b;}RS_HD inline LocalJet operator-(LocalJet a,LocalJet b){return a-=b;}
RS_HD inline LocalJet operator*(LocalJet a,LocalJet b){return a*=b;}RS_HD inline LocalJet operator/(LocalJet a,LocalJet b){return a/=b;}
RS_HD inline LocalJet operator-(LocalJet a){return LocalJet(-a.a,-a.v);}
RS_HD inline bool operator<(LocalJet a,LocalJet b){return a.a<b.a;}RS_HD inline bool operator>(LocalJet a,LocalJet b){return a.a>b.a;}
RS_HD inline bool operator<=(LocalJet a,LocalJet b){return a.a<=b.a;}RS_HD inline bool operator>=(LocalJet a,LocalJet b){return a.a>=b.a;}
RS_HD inline LocalJet sin(LocalJet x){return LocalJet(::sin(x.a),::cos(x.a)*x.v);}RS_HD inline LocalJet cos(LocalJet x){return LocalJet(::cos(x.a),-::sin(x.a)*x.v);}
RS_HD inline LocalJet sqrt(LocalJet x){double y=::sqrt(x.a);return LocalJet(y,x.v==0?0.:x.v*.5/y);}RS_HD inline LocalJet exp(LocalJet x){double y=::exp(x.a);return LocalJet(y,y*x.v);}
RS_HD inline LocalJet pow(LocalJet x,LocalJet p){double y=::pow(x.a,p.a),dx=p.a*::pow(x.a,p.a-1.),dp=(x.a==0&&p.a>0)?0.:y*::log(x.a);return LocalJet(y,(x.v==0?0.:dx*x.v)+(p.v==0?0.:dp*p.v));}
RS_HD inline bool operator==(LocalJet a,LocalJet b){return a.a==b.a;}
RS_HD inline bool operator!=(LocalJet a,LocalJet b){return a.a!=b.a;}
RS_HD inline double primal(LocalJet x){return x.a;}
RS_HD inline LocalJet fmod(LocalJet x,LocalJet y){return {::fmod(x.a,y.a),x.v-::trunc(x.a/y.a)*y.v};}
RS_HD inline LocalJet expm1(LocalJet x){return {::expm1(x.a),::exp(x.a)*x.v};}
RS_HD inline LocalJet log(LocalJet x){return LocalJet(::log(x.a),x.v/x.a);}
RS_HD inline LocalJet atan(LocalJet x){return LocalJet(::atan(x.a),x.v/(1+x.a*x.a));}
RS_HD inline LocalJet abs(LocalJet x){return x.a<0?-x:x;}RS_HD inline bool isfinite(LocalJet x){return finite_scalar(x.a)&&finite_scalar(x.v);}
RS_HD inline bool finite_scalar(LocalJet x){return finite_scalar(x.a)&&finite_scalar(x.v);}
}
