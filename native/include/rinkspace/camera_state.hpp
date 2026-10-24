#pragma once
#include "projection.hpp"
#include <algorithm>
#include <limits>

namespace rinkspace {
RS_HD inline Vec3 cross3(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
RS_HD inline Mat3 multiply3(const Mat3& a,const Mat3& b){Mat3 c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c.a[3*i+j]+=a.a[3*i+k]*b.a[3*k+j];return c;}
RS_HD inline Vec3 unit3(Vec3 a){return mul(1/std::sqrt(dot(a,a)),a);}
RS_HD inline Mat3 axis_rotation(Vec3 a,double t){
    const double x=a.x,y=a.y,z=a.z,s=std::sin(t),c=std::cos(t),d=1-c;
    return {{c+d*x*x,d*x*y-s*z,d*x*z+s*y,d*x*y+s*z,c+d*y*y,d*y*z-s*x,d*x*z-s*y,d*y*z+s*x,c+d*z*z}};
}
RS_HD inline Mat3 exp_rotation(Vec3 v){const double q=dot(v,v);if(q==0)return {{1,0,0,0,1,0,0,0,1}};return axis_rotation(mul(1/std::sqrt(q),v),std::sqrt(q));}
/* Explicit physical rig, 38 values. Units: world scene gauge, radians,
 * seconds and original source pixels. This is also the sole moving-camera
 * forward used by review; no Python pose or distortion evaluator is called.
 */
struct MotionRig {
    Mat3 R0;Vec3 pan,tilt,pivot,lever;double spacing,pupil1,pupil2;
    double fref,radius,cx,cy,eta,cxz,cyz,k10,k11,k20,k21,p1,p2,tau;
    bool valid()const{
        const double p[]={spacing,pupil1,pupil2,fref,radius,cx,cy,eta,cxz,cyz,k10,k11,k20,k21,p1,p2,tau};
        return valid_rotation(R0)&&finite(pan)&&finite(tilt)&&finite(pivot)&&finite(lever)&&finite_array(p,17)&&
            std::abs(dot(pan,pan)-1)<1e-8&&std::abs(dot(tilt,tilt)-1)<1e-8&&std::abs(dot(tilt,apply(R0,pan)))<1e-8&&fref>0&&radius>0;
    }
};
inline MotionRig motion_rig(const double* p){MotionRig r{};for(int i=0;i<9;++i)r.R0.a[i]=p[i];r.pan=load3(p+9);r.tilt=load3(p+12);r.pivot=load3(p+15);r.lever=load3(p+18);r.spacing=p[21];r.pupil1=p[22];r.pupil2=p[23];r.fref=p[24];r.radius=p[25];r.cx=p[26];r.cy=p[27];r.eta=p[28];r.cxz=p[29];r.cyz=p[30];r.k10=p[31];r.k11=p[32];r.k20=p[33];r.k21=p[34];r.p1=p[35];r.p2=p[36];r.tau=p[37];return r;}
inline Status decode_d78_camera(const double* p,double* out){
    // p = global32, R0_seed9, n_seed3, b_seed3, n_tangent1/2 (3 each), f_ref, radius.
    Mat3 seed{};for(int i=0;i<9;++i)seed.a[i]=p[32+i];if(!valid_rotation(seed))return Status::invalid_argument;
    const auto R=multiply3(exp_rotation(load3(p+5)),seed);
    Vec3 n=add(load3(p+41),add(mul(p[8],load3(p+47)),mul(p[9],load3(p+50))));
    if(dot(n,n)<1e-20)return Status::degenerate;n=unit3(n);const Vec3 q=apply(R,n),bseed=load3(p+44);
    const Vec3 e0=sub(bseed,mul(dot(bseed,q),q));if(dot(e0,e0)<1e-20)return Status::degenerate;const Vec3 e=unit3(e0);
    const Vec3 b=add(mul(std::cos(p[10]),e),mul(std::sin(p[10]),cross3(q,e)));
    for(int i=0;i<9;++i)out[i]=R.a[i];store3(out+9,n);store3(out+12,b);store3(out+15,load3(p+11));store3(out+18,load3(p+14));
    out[21]=p[17];out[22]=p[18];out[23]=p[19];out[24]=p[53];out[25]=p[54];
    for(int i=0;i<12;++i)out[26+i]=p[20+i];return motion_rig(out).valid()?Status::ok:Status::invalid_argument;
}
struct MotionSpline {
    const double* controls;size_t count;double first,dt;
    bool valid()const{return controls&&count>=4&&count<=size_t(std::numeric_limits<int>::max())&&std::isfinite(first)&&std::isfinite(dt)&&dt>0&&finite_array(controls,3*count);}
    RS_HD Status evaluate(double time,double* value)const{
        if(!std::isfinite(time))return Status::invalid_argument;
        if(time<first+dt-1e-12||time>first+(count-2)*dt+1e-12)return Status::outside_support;
        const double x=(time-first)/dt;int cell=int(std::floor(x));
        if(cell==int(count)-2&&std::abs(x-cell)<1e-10)--cell;
        if(cell<1||cell+2>=int(count))return Status::outside_support;
        const double u=x-cell,v=1-u,u2=u*u,u3=u2*u,w[]={v*v*v/6,(3*u3-6*u2+4)/6,(-3*u3+3*u2+3*u+1)/6,u3/6};
        for(int c=0;c<3;++c){value[c]=0;for(int j=0;j<4;++j)value[c]+=w[j]*controls[3*(cell-1+j)+c];}
        return finite_array(value,3)?Status::ok:Status::nonfinite;
    }
    Status rates(double lo,double hi,double* rates)const{
        double q[3];if(evaluate(lo,q)!=Status::ok||evaluate(hi,q)!=Status::ok)return Status::outside_support;
        const int begin=std::max(1,int(std::floor((lo-first)/dt))),end=std::min(int(count)-3,int(std::floor((hi-first)/dt)));
        for(int c=0;c<3;++c){rates[c]=0;
            for(int cell=begin;cell<=end;++cell){
                const double u0=std::max(0.,(lo-first)/dt-cell),u1=std::min(1.,(hi-first)/dt-cell);
                const double c0=controls[3*(cell-1)+c],c1=controls[3*cell+c],c2=controls[3*(cell+1)+c],c3=controls[3*(cell+2)+c];
                const double a=(-c0+3*c1-3*c2+c3)/2,b=c0-2*c1+c2,d=(c2-c0)/2;
                auto derivative=[&](double u){return std::abs((a*u*u+b*u+d)/dt);};
                rates[c]=std::max(rates[c],std::max(derivative(u0),derivative(u1)));
                if(a!=0){const double vertex=-b/(2*a);if(vertex>u0&&vertex<u1)rates[c]=std::max(rates[c],derivative(vertex));}
            }rates[c]=std::nextafter(rates[c]+2e-11*(1+rates[c]),std::numeric_limits<double>::infinity());}return Status::ok;
    }
};
struct MotionInstant {Camera camera;Vec3 ray;double logf,z,depth;};
RS_HD inline double quadratic_min(double c1,double c2,double r2){double v=std::fmin(1.,1+c1*r2+c2*r2*r2);if(c2>0){const double t=-c1/(2*c2);if(t>0&&t<r2)v=std::fmin(v,1+c1*t+c2*t*t);}return v;}
RS_HD inline Status motion_instant(const MotionRig& r,const MotionSpline& s,Vec3 X,double time,MotionInstant& out,Vec2& uv){
    double tr[3];auto status=s.evaluate(time,tr);if(status!=Status::ok)return status;
    out.logf=tr[2];out.z=tr[2]-std::log(r.fref);const auto R=multiply3(multiply3(axis_rotation(r.tilt,tr[1]),r.R0),axis_rotation(r.pan,tr[0]));
    Vec3 h=cross3(r.pan,transpose_apply(R,r.tilt));if(dot(h,h)<1e-20)return Status::degenerate;h=unit3(h);
    Vec3 lever=r.lever;lever.z+=r.pupil1*out.z+r.pupil2*out.z*out.z;
    Camera c{};c.rotation=R;c.centre=pivot_centre(R,r.pivot,lever,h,r.spacing);c.fx=std::exp(tr[2]+r.eta/2);c.fy=std::exp(tr[2]-r.eta/2);
    c.cx=r.cx+r.cxz*out.z;c.cy=r.cy+r.cyz*out.z;c.k1=r.k10+r.k11*out.z;c.k2=r.k20+r.k21*out.z;c.p1=r.p1;c.p2=r.p2;
    out.camera=c;out.ray=apply(R,sub(X,c.centre));out.depth=out.ray.z;
    if(!finite(out.ray))return Status::nonfinite;if(out.depth<=0)return Status::behind_camera;
    const double x=out.ray.x/out.depth,y=out.ray.y/out.depth,r2=x*x+y*y;
    if(r2>r.radius*r.radius)return static_cast<Status>(8);
    const double lower=std::fmin(quadratic_min(c.k1,c.k2,r2),quadratic_min(3*c.k1,5*c.k2,r2))-6*std::hypot(c.p1,c.p2)*std::sqrt(r2);
    if(!(lower>1e-10))return static_cast<Status>(8);
    return project(c,X,uv);
}
/* Sufficient analytic bound over the ENTIRE source row-time interval. Cubic
 * derivative extrema and rigid-body/Brown norm bounds exclude multiple row
 * roots. This is a floating-point padded bound, not interval arithmetic.
 */
inline Status contraction_bound(const MotionRig& r,const MotionSpline& s,Vec3 X,double t,int height,double& bound){
    const double half=std::abs(r.tau)/2;double rates[3];auto status=s.rates(t-half,t+half,rates);if(status!=Status::ok)return status;
    MotionInstant m{};Vec2 uv{};status=motion_instant(r,s,X,t,m,uv);if(status!=Status::ok)return status;
    const double omega=rates[0]+rates[1],zd=rates[2],zabs=std::abs(m.z)+zd*half;
    const double speed3=omega*(std::sqrt(dot(sub(X,r.pivot),sub(X,r.pivot)))+std::abs(r.spacing))+(std::abs(r.pupil1)+2*std::abs(r.pupil2)*zabs)*zd;
    const double span=speed3*half,zmin=m.depth-span;if(!(zmin>0))return static_cast<Status>(9);
    const double x=(std::abs(m.ray.x)+span)/zmin,y=(std::abs(m.ray.y)+span)/zmin,r2=x*x+y*y,radius=std::sqrt(r2);
    const double xp=speed3*(zmin+std::abs(m.ray.x)+span)/(zmin*zmin),yp=speed3*(zmin+std::abs(m.ray.y)+span)/(zmin*zmin);
    const double k1=std::abs(r.k10)+std::abs(r.k11)*zabs,k2=std::abs(r.k20)+std::abs(r.k21)*zabs,p1=std::abs(r.p1),p2=std::abs(r.p2);
    const double L=1+k1*r2+k2*r2*r2,jyx=2*x*y*(k1+2*k2*r2)+2*p1*x+2*p2*y,jyy=L+2*y*y*(k1+2*k2*r2)+6*p1*y+2*p2*x;
    const double ybrown=y*L+p1*(r2+2*y*y)+2*p2*x*y,direct=y*(std::abs(r.k11)*r2+std::abs(r.k21)*r2*r2)*zd;
    const double fymax=std::exp(m.logf+zd*half-r.eta/2),speed=std::nextafter((fymax*(ybrown*zd+jyx*xp+jyy*yp+direct)+std::abs(r.cyz)*zd)*(1+2e-10)+1e-8,std::numeric_limits<double>::infinity());
    bound=std::abs(r.tau)*speed/(height-1);
    const double k1lo=m.camera.k1-std::abs(r.k11)*zd*half,k2lo=m.camera.k2-std::abs(r.k21)*zd*half;
    const double fold=std::fmin(quadratic_min(k1lo,k2lo,r2),quadratic_min(3*k1lo,5*k2lo,r2))-6*std::hypot(p1,p2)*radius;
    return radius<=r.radius&&fold>1e-10&&std::isfinite(bound)&&bound<1-1e-9?Status::ok:static_cast<Status>(9);
}
struct MotionProjection {Vec2 uv;double depth,row,equality,contraction;};
inline Status project_motion(const MotionRig& r,const MotionSpline& s,int height,Vec3 X,double time,MotionProjection& out){
    if(!finite(X)||!std::isfinite(time))return Status::invalid_argument;
    auto status=contraction_bound(r,s,X,time,height,out.contraction);if(status!=Status::ok)return status;
    MotionInstant a{},b{};Vec2 ua{},ub{};const double last=height-1;
    status=motion_instant(r,s,X,time-r.tau/2,a,ua);if(status!=Status::ok)return status;
    status=motion_instant(r,s,X,time+r.tau/2,b,ub);if(status!=Status::ok)return status;
    double lo=0,hi=last,flo=-ua.y,fhi=last-ub.y;
    if(flo>1e-8||fhi< -1e-8)return Status::no_intersection;
    if(std::abs(flo)<=1e-8){out.uv=ua;out.depth=a.depth;out.row=0;out.equality=flo;return Status::ok;}
    if(std::abs(fhi)<=1e-8){out.uv=ub;out.depth=b.depth;out.row=last;out.equality=fhi;return Status::ok;}
    for(int iteration=0;iteration<64;++iteration){
        double row=lo-flo*(hi-lo)/(fhi-flo);if(!(row>lo&&row<hi)||iteration>16)row=(lo+hi)/2;
        MotionInstant current{};Vec2 uv{};status=motion_instant(r,s,X,time+r.tau*(row/last-.5),current,uv);if(status!=Status::ok)return status;
        const double f=row-uv.y;if(std::abs(f)<1e-8){out.uv=uv;out.row=row;out.equality=f;out.depth=current.depth;return Status::ok;}
        if(f>0){hi=row;fhi=f;}else{lo=row;flo=f;}
    }return static_cast<Status>(10);
}
} // namespace rinkspace
