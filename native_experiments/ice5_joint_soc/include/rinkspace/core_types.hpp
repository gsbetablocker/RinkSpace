#pragma once
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
# define RS_HD __host__ __device__
#else
# define RS_HD
#endif

namespace rinkspace {
constexpr double pi=3.141592653589793238462643383279502884;
enum class Status:int { ok=0,invalid_argument=1,degenerate=2,outside_support=3,
    no_intersection=4,behind_camera=5,nonfinite=6,not_coplanar=7 };
template<class T> struct Vec3Value { T x,y,z; };
template<class T> struct Vec2Value { T x,y; };
template<class T> struct Mat3Value { T a[9]; };
using Vec3=Vec3Value<double>;using Vec2=Vec2Value<double>;using Mat3=Mat3Value<double>;
template<class T,class U> RS_HD inline Vec3Value<T> scalar_cast(Vec3Value<U> p){return {T(p.x),T(p.y),T(p.z)};}
template<class T,class U> RS_HD inline Mat3Value<T> scalar_cast(const Mat3Value<U>& p){Mat3Value<T> r{};for(int i=0;i<9;++i)r.a[i]=T(p.a[i]);return r;}
template<class T> RS_HD inline Vec3Value<T> add(Vec3Value<T> a,Vec3Value<T> b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
template<class T> RS_HD inline Vec3Value<T> sub(Vec3Value<T> a,Vec3Value<T> b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
template<class S,class T> RS_HD inline Vec3Value<T> mul(S s,Vec3Value<T> a){return {s*a.x,s*a.y,s*a.z};}
template<class T> RS_HD inline T dot(Vec3Value<T> a,Vec3Value<T> b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<class T> RS_HD inline Vec3Value<T> apply(const Mat3Value<T>& r,Vec3Value<T> p){return {
    r.a[0]*p.x+r.a[1]*p.y+r.a[2]*p.z,
    r.a[3]*p.x+r.a[4]*p.y+r.a[5]*p.z,
    r.a[6]*p.x+r.a[7]*p.y+r.a[8]*p.z};}
template<class T> RS_HD inline Vec3Value<T> transpose_apply(const Mat3Value<T>& r,Vec3Value<T> p){return {
    r.a[0]*p.x+r.a[3]*p.y+r.a[6]*p.z,
    r.a[1]*p.x+r.a[4]*p.y+r.a[7]*p.z,
    r.a[2]*p.x+r.a[5]*p.y+r.a[8]*p.z};}
/* MSVC's std::isfinite template is host-only under NVCC. Preserve the host
 * predicate and use its exact IEEE finite-range equivalent on the device. */
RS_HD inline bool finite_scalar(double value){
#if defined(__CUDA_ARCH__)
    return value==value && value<=std::numeric_limits<double>::max() && value>=-std::numeric_limits<double>::max();
#else
    return std::isfinite(value);
#endif
}
template<class T> RS_HD inline bool finite(Vec3Value<T> p){return finite_scalar(p.x)&&finite_scalar(p.y)&&finite_scalar(p.z);}
template<class T> RS_HD inline bool finite_array(const T* p,std::size_t n){
    if(!p)return false;
    for(std::size_t i=0;i<n;++i)if(!finite_scalar(p[i]))return false;
    return true;
}
template<class T> RS_HD inline Vec3Value<T> load3(const T* p){return {p[0],p[1],p[2]};}
template<class T> RS_HD inline void store3(T* p,Vec3Value<T> v){p[0]=v.x;p[1]=v.y;p[2]=v.z;}
RS_HD inline double primal(double x){return x;}
inline bool valid_rotation(const Mat3& r){
    if(!finite_array(r.a,9))return false;
    for(int i=0;i<3;++i)for(int j=i;j<3;++j){
        double v=0;for(int k=0;k<3;++k)v+=r.a[3*i+k]*r.a[3*j+k];
        if(std::abs(v-(i==j?1.:0.))>1e-6)return false;
    }
    const double det=r.a[0]*(r.a[4]*r.a[8]-r.a[5]*r.a[7])-
        r.a[1]*(r.a[3]*r.a[8]-r.a[5]*r.a[6])+r.a[2]*(r.a[3]*r.a[7]-r.a[4]*r.a[6]);
    return std::abs(det-1)<1e-6;
}
} // namespace rinkspace
