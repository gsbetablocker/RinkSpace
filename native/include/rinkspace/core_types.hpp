#pragma once
#include <cmath>
#include <cstddef>

#if defined(__CUDACC__)
# define RS_HD __host__ __device__
#else
# define RS_HD
#endif

namespace rinkspace {
constexpr double pi=3.141592653589793238462643383279502884;
enum class Status:int { ok=0,invalid_argument=1,degenerate=2,outside_support=3,
    no_intersection=4,behind_camera=5,nonfinite=6,not_coplanar=7 };
struct Vec3 { double x,y,z; };
struct Vec2 { double x,y; };
struct Mat3 { double a[9]; };
RS_HD inline Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
RS_HD inline Vec3 sub(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
RS_HD inline Vec3 mul(double s,Vec3 a){return {s*a.x,s*a.y,s*a.z};}
RS_HD inline double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
RS_HD inline Vec3 apply(const Mat3& r,Vec3 p){return {
    r.a[0]*p.x+r.a[1]*p.y+r.a[2]*p.z,
    r.a[3]*p.x+r.a[4]*p.y+r.a[5]*p.z,
    r.a[6]*p.x+r.a[7]*p.y+r.a[8]*p.z};}
RS_HD inline Vec3 transpose_apply(const Mat3& r,Vec3 p){return {
    r.a[0]*p.x+r.a[3]*p.y+r.a[6]*p.z,
    r.a[1]*p.x+r.a[4]*p.y+r.a[7]*p.z,
    r.a[2]*p.x+r.a[5]*p.y+r.a[8]*p.z};}
RS_HD inline bool finite(Vec3 p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
inline bool finite_array(const double* p,std::size_t n){
    if(!p)return false;
    for(std::size_t i=0;i<n;++i)if(!std::isfinite(p[i]))return false;
    return true;
}
inline Vec3 load3(const double* p){return {p[0],p[1],p[2]};}
inline void store3(double* p,Vec3 v){p[0]=v.x;p[1]=v.y;p[2]=v.z;}
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
