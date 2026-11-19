#pragma once
#include "projection.hpp"
#include <algorithm>
#include <limits>

namespace rinkspace {
template<class T> RS_HD inline Vec3Value<T> cross3(Vec3Value<T> a,Vec3Value<T> b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
template<class T> RS_HD inline Mat3Value<T> multiply3(const Mat3Value<T>& a,const Mat3Value<T>& b){Mat3Value<T> c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c.a[3*i+j]+=a.a[3*i+k]*b.a[3*k+j];return c;}
template<class T> RS_HD inline Vec3Value<T> unit3(Vec3Value<T> a){using std::sqrt;return mul(1/sqrt(dot(a,a)),a);}
template<class T> RS_HD inline Mat3Value<T> axis_rotation(Vec3Value<T> a,T t){
    using std::sin;using std::cos;
    const T x=a.x,y=a.y,z=a.z,s=sin(t),c=cos(t),d=1-c;
    return {{c+d*x*x,d*x*y-s*z,d*x*z+s*y,d*x*y+s*z,c+d*y*y,d*y*z-s*x,d*x*z-s*y,d*y*z+s*x,c+d*z*z}};
}
RS_HD inline Mat3 exp_rotation(Vec3 v){const double q=dot(v,v);if(q==0)return {{1,0,0,0,1,0,0,0,1}};return axis_rotation(mul(1/std::sqrt(q),v),std::sqrt(q));}
/* Explicit physical rig, 38 values. Units: world scene gauge, radians,
 * seconds and original source pixels. This is also the sole moving-camera
 * forward used by review; no Python pose or distortion evaluator is called.
 */
template<class T> struct MotionRigValue {
    Mat3Value<T> R0;Vec3Value<T> pan,tilt,pivot,lever;T spacing,pupil1,pupil2;
    T fref,radius,cx,cy,eta,cxz,cyz,k10,k11,k20,k21,p1,p2,tau;
    T wide_boundary{};T wide_coefficients[4]{};
    bool valid()const{
        const double p[]={spacing,pupil1,pupil2,fref,radius,cx,cy,eta,cxz,cyz,k10,k11,k20,k21,p1,p2,tau};
        return valid_rotation(R0)&&finite(pan)&&finite(tilt)&&finite(pivot)&&finite(lever)&&finite_array(p,17)&&
            std::abs(dot(pan,pan)-1)<1e-8&&std::abs(dot(tilt,tilt)-1)<1e-8&&std::abs(dot(tilt,apply(R0,pan)))<1e-8&&fref>0&&radius>0&&finite_scalar(wide_boundary)&&finite_array(wide_coefficients,4);
    }
};
using MotionRig=MotionRigValue<double>;
RS_HD inline MotionRig motion_rig(const double* p,const double* wide=nullptr){MotionRig r{};for(int i=0;i<9;++i)r.R0.a[i]=p[i];r.pan=load3(p+9);r.tilt=load3(p+12);r.pivot=load3(p+15);r.lever=load3(p+18);r.spacing=p[21];r.pupil1=p[22];r.pupil2=p[23];r.fref=p[24];r.radius=p[25];r.cx=p[26];r.cy=p[27];r.eta=p[28];r.cxz=p[29];r.cyz=p[30];r.k10=p[31];r.k11=p[32];r.k20=p[33];r.k21=p[34];r.p1=p[35];r.p2=p[36];r.tau=p[37];if(wide){r.wide_boundary=wide[0];for(int i=0;i<4;++i)r.wide_coefficients[i]=wide[i+1];}return r;}
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
// Derivatives of the same uniform cubic used below. First differences remove
// the constant offset before dividing by small time steps (notably log(f)).
// Available for both ordinary values and the native analytic direction type.
template<class T> RS_HD inline T camera_spline_derivative(const T* c,int channel,double u,double dt,int order){
    const T p=c[3+channel]-c[channel],q=c[6+channel]-c[3+channel],r=c[9+channel]-c[6+channel];
    if(order==1)return ((.5*(1-u)*(1-u))*p+(.5+u-u*u)*q+(.5*u*u)*r)/dt;
    if(order==2)return ((1-u)*(q-p)+u*(r-q))/(dt*dt);
    return (r-2*q+p)/(dt*dt*dt);
}
RS_HD inline int camera_velocity_extremum_times(const double* c,int channel,double lo,double hi,double* times){
    times[0]=lo;times[1]=hi;int count=2;
    const double p=c[3+channel]-c[channel],q=c[6+channel]-c[3+channel],r=c[9+channel]-c[6+channel];
    const double bend=r-2*q+p;
    if(bend!=0){const double u=(p-q)/bend;if(u>lo&&u<hi)times[count++]=u;}
    return count;
}
struct MotionSpline {
    const double* controls;size_t count;double first,dt;
    bool valid()const{return controls&&count>=4&&count<=size_t(std::numeric_limits<int>::max())&&finite_scalar(first)&&finite_scalar(dt)&&dt>0&&finite_array(controls,3*count);}
    RS_HD Status basis(double time,int& cell,double* weights,double* derivatives=nullptr)const{
        if(!finite_scalar(time))return Status::invalid_argument;
        if(time<first+dt-1e-12||time>first+(count-2)*dt+1e-12)return Status::outside_support;
        const double x=(time-first)/dt;cell=int(std::floor(x));
        if(cell==0&&std::abs(x-1)<1e-10)cell=1;
        if(cell==int(count)-2&&std::abs(x-cell)<1e-10)--cell;
        if(cell<1||cell+2>=int(count))return Status::outside_support;
        const double u=x-cell,v=1-u,u2=u*u,u3=u2*u,w[]={v*v*v/6,(3*u3-6*u2+4)/6,(-3*u3+3*u2+3*u+1)/6,u3/6};
        for(int j=0;j<4;++j)weights[j]=w[j];
        if(derivatives){const double dw[]={-v*v/2,(9*u2-12*u)/6,(-9*u2+6*u+3)/6,u2/2};for(int j=0;j<4;++j)derivatives[j]=dw[j]/dt;}
        return Status::ok;
    }
    RS_HD Status evaluate(double time,double* value)const{
        int cell;double w[4];const auto status=basis(time,cell,w);if(status!=Status::ok)return status;
        for(int c=0;c<3;++c){value[c]=0;for(int j=0;j<4;++j)value[c]+=w[j]*controls[3*size_t(cell-1+j)+c];}
        return finite_array(value,3)?Status::ok:Status::nonfinite;
    }
    // Diagnostic derivatives of the SAME trajectory, in seconds. The third
    // derivative is one-sided at a knot; cubic splines guarantee C2, not C3.
    RS_HD Status kinematics(double time,double* value)const{
        int cell;double w[4];const auto status=basis(time,cell,w);if(status!=Status::ok)return status;
        const double u=(time-first)/dt-cell;const double* local=controls+3*size_t(cell-1);
        for(int c=0;c<3;++c){
            value[c]=0;for(int j=0;j<4;++j)value[c]+=w[j]*controls[3*size_t(cell-1+j)+c];
        }
        for(int order=1;order<=3;++order)for(int channel=0;channel<3;++channel)
            value[3*order+channel]=camera_spline_derivative(local,channel,u,dt,order);
        return finite_array(value,12)?Status::ok:Status::nonfinite;
    }
    // Exact absolute extrema on a requested continuous interval. Nine entries
    // (derivative order, channel), each signed value / time / one-sided cell.
    RS_HD Status derivative_extrema(double lo,double hi,double* out)const{
        if(!(lo<=hi))return Status::invalid_argument;
        double value[3];auto status=evaluate(lo,value);if(status!=Status::ok)return status;
        status=evaluate(hi,value);if(status!=Status::ok)return status;
        for(int k=0;k<27;++k)out[k]=std::numeric_limits<double>::quiet_NaN();
        const int begin=std::max(1,int(std::floor((lo-first)/dt))),end=std::min(int(count)-3,int(std::floor((hi-first)/dt)));
        for(int cell=std::min(begin,int(count)-3);cell<=end;++cell){
            const double u0=std::max(0.,(lo-first)/dt-cell),u1=std::min(1.,(hi-first)/dt-cell);
            const double* c=controls+3*size_t(cell-1);
            for(int channel=0;channel<3;++channel){
                double times[3];const int ntime=camera_velocity_extremum_times(c,channel,u0,u1,times);
                for(int order=1;order<=3;++order){
                    const int n=order==1?ntime:(order==2?2:1),index=3*(3*(order-1)+channel);
                    for(int i=0;i<n;++i){
                        const double v=camera_spline_derivative(c,channel,times[i],dt,order);
                        if(!finite_scalar(v))return Status::nonfinite;
                        if(!finite_scalar(out[index])||std::abs(v)>std::abs(out[index])){
                            out[index]=v;out[index+1]=first+(cell+times[i])*dt;out[index+2]=cell;}
                    }
                }
            }
        }return finite_array(out,27)?Status::ok:Status::nonfinite;
    }
    RS_HD Status rates(double lo,double hi,double* rates)const{
        if(!(lo<=hi))return Status::invalid_argument;
        double value[3];
        if(evaluate(lo,value)!=Status::ok||evaluate(hi,value)!=Status::ok)return Status::outside_support;

        // Match basis() at the closed endpoints, including accepted roundoff.
        int begin,end;double weights[4];
        if(basis(lo,begin,weights)!=Status::ok||basis(hi,end,weights)!=Status::ok)return Status::outside_support;
        const double start=(lo-first)/dt,stop=(hi-first)/dt;
        for(int channel=0;channel<3;++channel)rates[channel]=0.;
        for(int cell=begin;cell<=end;++cell){
            const double u0=cell==begin?start-cell:0.,u1=cell==end?stop-cell:1.;
            const double* c=controls+3*size_t(cell-1);
            for(int channel=0;channel<3;++channel){
                double times[3];const int n=camera_velocity_extremum_times(c,channel,u0,u1,times);
                for(int i=0;i<n;++i){
                    // Adjacent differences remove the offset before expansion.
                    const double velocity=camera_spline_derivative(c,channel,times[i],dt,1);
                    if(!finite_scalar(velocity))return Status::nonfinite;
                    rates[channel]=std::max(rates[channel],std::abs(velocity));
                }
            }
        }
        for(int channel=0;channel<3;++channel){
            const double rate=rates[channel];
            rates[channel]=std::nextafter(rate+2e-11*(1+rate),std::numeric_limits<double>::infinity());
            if(!finite_scalar(rates[channel]))return Status::nonfinite;
        }
        return Status::ok;
    }
};
template<class T> struct MotionCameraStateValue {CameraValue<T> camera;T pan,tilt,logf,z;};
using MotionCameraState=MotionCameraStateValue<double>;
struct MotionInstant : MotionCameraState {Vec3 ray;double depth;};
RS_HD inline double quadratic_min(double c1,double c2,double r2){double v=std::fmin(1.,1+c1*r2+c2*r2*r2);if(c2>0){const double t=-c1/(2*c2);if(t>0&&t<r2)v=std::fmin(v,1+c1*t+c2*t*t);}return v;}
// Minimum of 1+linear*r+a*r^2+b*r^4 on [0,radius]. The derivative is a
// depressed cubic, monotone on at most two positive-radius intervals. Bracket
// every stationary point instead of combining extrema at different radii.
RS_HD inline double radial_quartic_lower(double a,double b,double linear,double radius){
    auto value=[&](double r){return 1+r*(linear+r*(a+b*r*r));};
    auto derivative=[&](double r){return linear+r*(2*a+4*b*r*r);};
    double points[3]={0.,radius,radius};int count=2;
    if(b!=0.&&a*b<0.){const double split=std::sqrt(-a/(6*b));
        if(split>0.&&split<radius){points[1]=split;points[2]=radius;count=3;}}
    double lower=std::fmin(1.,value(radius));
    for(int i=0;i+1<count;++i){
        double lo=points[i],hi=points[i+1],flo=derivative(lo),fhi=derivative(hi);
        lower=std::fmin(lower,std::fmin(value(lo),value(hi)));
        if(!((flo<0.&&fhi>0.)||(flo>0.&&fhi<0.)))continue;
        for(int iteration=0;iteration<64;++iteration){
            const double mid=lo+(hi-lo)/2;if(!(mid>lo&&mid<hi))break;
            const double fmid=derivative(mid);
            if((flo<0.)==(fmid<0.)){lo=mid;flo=fmid;}else{hi=mid;fhi=fmid;}}
        // A derivative bound on this final bracket also covers rounding of
        // its stationary point. The original positive optical margin stays.
        const double slope=std::abs(linear)+2*std::abs(a)*hi+4*std::abs(b)*hi*hi*hi;
        lower=std::fmin(lower,std::fmin(value(lo),value(hi))-slope*(hi-lo));
    }
    const double scale=1+std::abs(linear)*radius+std::abs(a)*radius*radius+std::abs(b)*radius*radius*radius*radius;
    return lower-64*std::numeric_limits<double>::epsilon()*scale;
}
RS_HD inline double brown_disk_lower(double k1,double k2,double tangential_norm,double radius){
    const double r2=radius*radius;
    const double old=std::fmin(quadratic_min(k1,k2,r2),quadratic_min(3*k1,5*k2,r2))-6*tangential_norm*radius;
    if(old>1e-10)return old;
    // At EACH radius, the radial Jacobian eigenvalues are L and L+2*r^2*L'.
    // The tangential Jacobian is bounded below by -6*|p|*r. Weyl's bound is
    // therefore minimized at one common radius, over the COMPLETE disk.
    // This is still a sufficient positive-definiteness certificate, never a
    // waiver for an actual fold or for any unexamined radius/direction.
    return std::fmax(old,std::fmin(radial_quartic_lower(k1,k2,-6*tangential_norm,radius),
                                 radial_quartic_lower(3*k1,5*k2,-6*tangential_norm,radius)));
}
// Outward-rounded intervals used only to tighten a failed sufficient row-rate
// certificate. They enclose the same camera function over the whole window.
struct CameraRange {double lo,hi;};
RS_HD inline double camera_down(double x){return std::nextafter(x,-std::numeric_limits<double>::infinity());}
RS_HD inline double camera_up(double x){return std::nextafter(x,std::numeric_limits<double>::infinity());}
RS_HD inline CameraRange camera_range(double x,double span=0.){return {camera_down(x-span),camera_up(x+span)};}
RS_HD inline CameraRange operator+(CameraRange a,CameraRange b){return {camera_down(a.lo+b.lo),camera_up(a.hi+b.hi)};}
RS_HD inline CameraRange operator-(CameraRange a,CameraRange b){return {camera_down(a.lo-b.hi),camera_up(a.hi-b.lo)};}
RS_HD inline CameraRange operator*(CameraRange a,CameraRange b){
    const double x=a.lo*b.lo,y=a.lo*b.hi,z=a.hi*b.lo,w=a.hi*b.hi;
    return {camera_down(std::fmin(std::fmin(x,y),std::fmin(z,w))),camera_up(std::fmax(std::fmax(x,y),std::fmax(z,w)))};
}
RS_HD inline CameraRange operator/(CameraRange a,CameraRange b){return a*CameraRange{camera_down(1/b.hi),camera_up(1/b.lo)};}
RS_HD inline CameraRange camera_square(CameraRange a){
    return {a.lo<=0&&a.hi>=0?0.:camera_down(std::fmin(a.lo*a.lo,a.hi*a.hi)),camera_up(std::fmax(a.lo*a.lo,a.hi*a.hi))};
}
RS_HD inline double camera_abs_upper(CameraRange a){return finite_scalar(a.lo)&&finite_scalar(a.hi)?std::fmax(std::abs(a.lo),std::abs(a.hi)):std::numeric_limits<double>::infinity();}
RS_HD inline double signed_row_speed_bound(const MotionRig& r,const MotionInstant& m,double half,double zd,
        Vec3 velocity,double acceleration,double zoom_rate,double zoom_acceleration,Vec3 span,
        double k1span,double k2span,double p1span,double p2span){
    const auto one=camera_range(1.),two=camera_range(2.),six=camera_range(6.);
    const auto rx=camera_range(m.ray.x,span.x),ry=camera_range(m.ray.y,span.y),rz=camera_range(m.ray.z,span.z);
    const auto vx=camera_range(velocity.x,acceleration*half),vy=camera_range(velocity.y,acceleration*half),vz=camera_range(velocity.z,acceleration*half);
    const auto x=rx/rz,y=ry/rz,xp=(vx-x*vz)/rz,yp=(vy-y*vz)/rz;
    const auto x2=camera_square(x),y2=camera_square(y),r2=x2+y2,r4=camera_square(r2),xy=x*y;
    const auto k1=camera_range(m.camera.k1,k1span),k2=camera_range(m.camera.k2,k2span);
    const auto p1=camera_range(m.camera.p1,p1span),p2=camera_range(m.camera.p2,p2span);
    const auto zoom=camera_range(zoom_rate,zoom_acceleration*half);
    const CameraRange hinge{std::fmax(0.,camera_down(r.wide_boundary-m.logf-zd*half)),std::fmax(0.,camera_up(r.wide_boundary-m.logf+zd*half))};
    const auto k1dot=(camera_range(r.k11)-two*camera_range(r.wide_coefficients[0])*hinge)*zoom;
    const auto k2dot=(camera_range(r.k21)-two*camera_range(r.wide_coefficients[1])*hinge)*zoom;
    const auto p1dot=(camera_range(0.)-two*camera_range(r.wide_coefficients[2])*hinge)*zoom;
    const auto p2dot=(camera_range(0.)-two*camera_range(r.wide_coefficients[3])*hinge)*zoom;
    const auto L=one+k1*r2+k2*r4;
    const auto jyx=two*xy*(k1+two*k2*r2)+two*p1*x+two*p2*y;
    const auto jyy=L+two*y2*(k1+two*k2*r2)+six*p1*y+two*p2*x;
    const auto brown=y*L+p1*(r2+two*y2)+two*p2*xy;
    const auto direct=y*(k1dot*r2+k2dot*r4)+p1dot*(r2+two*y2)+two*p2dot*xy;
    const CameraRange fy{camera_down(std::exp(m.logf-zd*half-r.eta/2)),camera_up(std::exp(m.logf+zd*half-r.eta/2))};
    return camera_abs_upper(fy*(zoom*brown+jyx*xp+jyy*yp+direct)+camera_range(r.cyz)*zoom);
}
template<class T,class U> RS_HD inline Status motion_camera_from_values(const MotionRigValue<U>& r,const T* tr,MotionCameraStateValue<T>& out){
    using std::exp;using std::log;
    const auto pan=scalar_cast<T>(r.pan),tilt=scalar_cast<T>(r.tilt),pivot=scalar_cast<T>(r.pivot);
    const auto R0=scalar_cast<T>(r.R0);
    out.pan=tr[0];out.tilt=tr[1];
    out.logf=tr[2];out.z=tr[2]-log(r.fref);const auto R=multiply3(multiply3(axis_rotation(tilt,tr[1]),R0),axis_rotation(pan,tr[0]));
    // P is the pan-axis endpoint of the shortest connector to the tilt axis.
    // Q=P+spacing*h is its other endpoint; h is perpendicular to both axes.
    // For a vertical pan axis P and Q have equal world height. There is no
    // independent P-to-Q axial height: adding one duplicates translation of P.
    // P is not asserted to be a particular screw or the bottom of the head.
    Vec3Value<T> h=cross3(pan,transpose_apply(R,tilt));if(dot(h,h)<1e-20)return Status::degenerate;h=unit3(h);
    Vec3Value<T> lever=scalar_cast<T>(r.lever);lever.z+=r.pupil1*out.z+r.pupil2*out.z*out.z;
    CameraValue<T> c{};c.rotation=R;c.centre=pivot_centre(R,pivot,lever,h,T(r.spacing));c.fx=exp(tr[2]+r.eta/2);c.fy=exp(tr[2]-r.eta/2);
    c.cx=r.cx+r.cxz*out.z;c.cy=r.cy+r.cyz*out.z;c.k1=r.k10+r.k11*out.z;c.k2=r.k20+r.k21*out.z;c.p1=r.p1;c.p2=r.p2;
    const T hinge=tr[2]<r.wide_boundary?T(r.wide_boundary)-tr[2]:T(0),basis=hinge*hinge;
    c.k1+=r.wide_coefficients[0]*basis;c.k2+=r.wide_coefficients[1]*basis;c.p1+=r.wide_coefficients[2]*basis;c.p2+=r.wide_coefficients[3]*basis;
    out.camera=c;
    const T optics[]={c.fx,c.fy,c.cx,c.cy,c.k1,c.k2,c.p1,c.p2};
    for(int i=0;i<8;++i)if(!finite_scalar(optics[i]))return Status::nonfinite;
    return finite(c.centre)&&c.fx>0&&c.fy>0?Status::ok:Status::nonfinite;
}
RS_HD inline Status evaluate_motion_camera(const MotionRig& r,const MotionSpline& s,double time,MotionCameraState& out){
    double tr[3];const auto status=s.evaluate(time,tr);if(status!=Status::ok)return status;
    return motion_camera_from_values(r,tr,out);
}
RS_HD inline Status motion_instant_from_camera(const MotionRig& r,const MotionCameraState& camera,Vec3 X,MotionInstant& out,Vec2& uv){
    static_cast<MotionCameraState&>(out)=camera;
    const auto& c=out.camera;out.ray=apply(c.rotation,sub(X,c.centre));out.depth=out.ray.z;
    if(!finite(out.ray))return Status::nonfinite;if(out.depth<=0)return Status::behind_camera;
    const double x=out.ray.x/out.depth,y=out.ray.y/out.depth,r2=x*x+y*y;
    if(r2>r.radius*r.radius)return static_cast<Status>(8);
    const double lower=brown_disk_lower(c.k1,c.k2,std::hypot(c.p1,c.p2),std::sqrt(r2));
    if(!(lower>1e-10))return static_cast<Status>(8);
    return project(c,X,uv);
}
RS_HD inline Status motion_instant(const MotionRig& r,const MotionSpline& s,Vec3 X,double time,MotionInstant& out,Vec2& uv){
    MotionCameraState camera{};auto status=evaluate_motion_camera(r,s,time,camera);if(status!=Status::ok)return status;
    return motion_instant_from_camera(r,camera,X,out,uv);
}
// Per-camera/exposure quantities are independent of the candidate 3-D point.
// They may be reused only while the current rig and motion controls are fixed.
// Point-dependent depth, optical validity and contraction bounds remain exact.
struct MotionProjectionContext {
    MotionCameraState centre{},lower{},upper{};double source_time=0.,time=0.,half=0.,rates[3]{};
    Status rates_status=Status::ok,centre_status=Status::ok,lower_status=Status::ok,upper_status=Status::ok;
};
RS_HD inline void prepare_motion_projection_context(const MotionRig& r,const MotionSpline& s,double time,MotionProjectionContext& context){
    context.source_time=time;const double half=std::abs(r.tau)/2,lo=time-half,hi=time+half;
    context.half=(hi-lo)/2;context.time=lo+context.half;
    context.rates_status=r.tau==0.?Status::ok:s.rates(context.time-context.half,context.time+context.half,context.rates);
    context.centre_status=evaluate_motion_camera(r,s,context.time,context.centre);
    if(r.tau==0.){context.lower=context.upper=context.centre;context.lower_status=context.upper_status=context.centre_status;}
    else{context.lower_status=evaluate_motion_camera(r,s,time-r.tau/2,context.lower);
        context.upper_status=evaluate_motion_camera(r,s,time+r.tau/2,context.upper);}
}
/* Sufficient analytic bound over the ENTIRE source row-time interval. Cubic
 * derivative extrema and rigid-body/Brown norm bounds exclude multiple row
 * roots. This is a floating-point padded bound, not interval arithmetic.
 */
RS_HD inline Status contraction_bound_window(const MotionRig& r,const MotionSpline& s,Vec3 X,double t,double half,int height,double& bound,bool directional=false,const MotionProjectionContext* context=nullptr){
    const bool cached=context&&context->time==t&&context->half==half;
    double rates[3];auto status=Status::ok;
    if(cached){status=context->rates_status;for(int i=0;i<3;++i)rates[i]=context->rates[i];}
    else status=s.rates(t-half,t+half,rates);if(status!=Status::ok)return status;
    MotionInstant m{};Vec2 uv{};
    if(cached){status=context->centre_status;if(status==Status::ok)status=motion_instant_from_camera(r,context->centre,X,m,uv);}
    else status=motion_instant(r,s,X,t,m,uv);if(status!=Status::ok)return status;
    const double omega=rates[0]+rates[1],zd=rates[2],zabs=std::abs(m.z)+zd*half;
    const double speed3=omega*(std::sqrt(dot(sub(X,r.pivot),sub(X,r.pivot)))+std::abs(r.spacing))+(std::abs(r.pupil1)+2*std::abs(r.pupil2)*zabs)*zd;
    Vec3 velocity_bound{speed3,speed3,speed3},span{speed3*half,speed3*half,speed3*half},signed_velocity{};
    double acceleration_bound=0.,signed_zoom_rate=0.,zoom_acceleration_bound=0.;
    if(directional){
        double k[12],extrema[27];status=s.kinematics(t,k);if(status!=Status::ok)return status;
        status=s.derivative_extrema(t-half,t+half,extrema);if(status!=Status::ok)return status;
        // In body coordinates ray = R*(X-P) - spacing*R*h - lever - e_z*pupil.
        // R*h is independent of pan: both axis-connector endpoints rotate
        // together. Differentiate this same rigid-body expression, retaining
        // each signed component instead of assigning full 3-D speed to y/z.
        Vec3 tilted=add(m.ray,r.lever);tilted.z+=r.pupil1*m.z+r.pupil2*m.z*m.z;
        Vec3 velocity=add(mul(k[4],cross3(r.tilt,tilted)),
            mul(k[3],apply(m.camera.rotation,cross3(r.pan,sub(X,r.pivot)))));
        velocity.z-=(r.pupil1+2*r.pupil2*m.z)*k[5];
        signed_velocity=velocity;signed_zoom_rate=k[5];
        const double acceleration_angles=std::abs(extrema[9])+std::abs(extrema[12]);
        const double acceleration_zoom=std::abs(extrema[15]);
        const double distance=std::sqrt(dot(sub(X,r.pivot),sub(X,r.pivot)))+std::abs(r.spacing);
        // ||ray''|| <= (|pan''|+|tilt''|+(|pan'|+|tilt'|)^2)*distance
        //              + |pupil_z|*|z''| + |pupil_zz|*|z'|^2.
        // Thus component velocity and displacement are bounded throughout
        // each window, not merely checked at sampled times.
        const double acceleration=std::nextafter(((acceleration_angles+omega*omega)*distance+
            (std::abs(r.pupil1)+2*std::abs(r.pupil2)*zabs)*acceleration_zoom+
            2*std::abs(r.pupil2)*zd*zd)*(1+2e-10)+1e-8,std::numeric_limits<double>::infinity());
        acceleration_bound=acceleration;zoom_acceleration_bound=acceleration_zoom;
        const Vec3 v{std::abs(velocity.x)*(1+2e-10)+1e-8,std::abs(velocity.y)*(1+2e-10)+1e-8,std::abs(velocity.z)*(1+2e-10)+1e-8};
        velocity_bound={v.x+acceleration*half,v.y+acceleration*half,v.z+acceleration*half};
        span={half*(v.x+acceleration*half/2),half*(v.y+acceleration*half/2),half*(v.z+acceleration*half/2)};
    }
    const double zmin=m.depth-span.z;if(!(zmin>0))return static_cast<Status>(9);
    const double x=(std::abs(m.ray.x)+span.x)/zmin,y=(std::abs(m.ray.y)+span.y)/zmin,r2=x*x+y*y,radius=std::sqrt(r2);
    const double xp=(velocity_bound.x+x*velocity_bound.z)/zmin,yp=(velocity_bound.y+y*velocity_bound.z)/zmin;
    const double hinge=std::max(0.,r.wide_boundary-m.logf+zd*half),hinge_rate=2*hinge*zd;
    // Bound the actual coefficient over this window. Bounding each zoom
    // basis separately loses cancellations in the SAME lens polynomial and
    // does not tighten as a rejected window is subdivided. The centre value
    // plus a derivative bound covers the complete interval, including a
    // crossing of the C1 wide-zoom hinge, without changing the lens model.
    const double k1span=(std::abs(r.k11)*zd+std::abs(r.wide_coefficients[0])*hinge_rate)*half;
    const double k2span=(std::abs(r.k21)*zd+std::abs(r.wide_coefficients[1])*hinge_rate)*half;
    const double k1=std::abs(m.camera.k1)+k1span,k2=std::abs(m.camera.k2)+k2span;
    const double p1=std::abs(m.camera.p1)+std::abs(r.wide_coefficients[2])*hinge_rate*half;
    const double p2=std::abs(m.camera.p2)+std::abs(r.wide_coefficients[3])*hinge_rate*half;
    const double L=1+k1*r2+k2*r2*r2,jyx=2*x*y*(k1+2*k2*r2)+2*p1*x+2*p2*y,jyy=L+2*y*y*(k1+2*k2*r2)+6*p1*y+2*p2*x;
    const double ybrown=y*L+p1*(r2+2*y*y)+2*p2*x*y,direct=y*(std::abs(r.k11)*r2+std::abs(r.k21)*r2*r2)*zd+hinge_rate*(y*(std::abs(r.wide_coefficients[0])*r2+std::abs(r.wide_coefficients[1])*r2*r2)+std::abs(r.wide_coefficients[2])*(r2+2*y*y)+2*std::abs(r.wide_coefficients[3])*x*y);
    const double fymax=std::exp(m.logf+zd*half-r.eta/2),speed=std::nextafter((fymax*(ybrown*zd+jyx*xp+jyy*yp+direct)+std::abs(r.cyz)*zd)*(1+2e-10)+1e-8,std::numeric_limits<double>::infinity());
    bound=std::abs(r.tau)*speed/(height-1);
    if(directional&&bound>=1-1e-9){
        // Keeping signed products lets this enclosure converge to the actual
        // derivative as the certified covering partition is subdivided. The
        // absolute-term sum above need not converge when terms cancel.
        const double tight=signed_row_speed_bound(r,m,half,zd,signed_velocity,acceleration_bound,
            signed_zoom_rate,zoom_acceleration_bound,span,k1span,k2span,
            std::abs(r.wide_coefficients[2])*hinge_rate*half,std::abs(r.wide_coefficients[3])*hinge_rate*half);
        bound=std::fmin(bound,std::abs(r.tau)*tight/(height-1));
    }
    const double k1lo=m.camera.k1-k1span,k2lo=m.camera.k2-k2span;
    const double fold=brown_disk_lower(k1lo,k2lo,std::hypot(p1,p2),radius);
    return radius<=r.radius&&fold>1e-10&&finite_scalar(bound)&&bound<1-1e-9?Status::ok:static_cast<Status>(9);
}
RS_HD inline Status contraction_bound_interval(const MotionRig& r,const MotionSpline& s,Vec3 X,double lo_time,double hi_time,int height,double& bound,const MotionProjectionContext* context=nullptr){
    const double half=(hi_time-lo_time)/2,t=lo_time+half;
    auto status=contraction_bound_window(r,s,X,t,half,height,bound,false,context);
    if(status!=static_cast<Status>(9))return status;
    // A failed sufficient whole-interval estimate is not a physical fold.
    // Certify a covering partition with the SAME depth, optical-domain and
    // row-contraction inequalities. The derivative still uses the complete
    // tau/(height-1), not the shorter subinterval duration. Successful leaves
    // jointly prove uniqueness on the entire requested interval, including
    // any continuation beyond the visible image. No failed leaf is accepted.
    constexpr int max_depth=8;double left[max_depth+1],right[max_depth+1];int depth[max_depth+1];
    int pending=1;left[0]=t-half;right[0]=t+half;depth[0]=0;double maximum=0.;
    while(pending){--pending;const double lo=left[pending],hi=right[pending],mid=lo+(hi-lo)/2;const int d=depth[pending];double part=0.;
        status=contraction_bound_window(r,s,X,mid,(hi-lo)/2,height,part,true);
        if(status==Status::ok){maximum=std::max(maximum,part);continue;}
        if(status!=static_cast<Status>(9))return status;
        // This bounds certification work only; no failed leaf is accepted.
        if(d==max_depth||!(mid>lo&&mid<hi))return static_cast<Status>(9);
        left[pending]=mid;right[pending]=hi;depth[pending++]=d+1;
        left[pending]=lo;right[pending]=mid;depth[pending++]=d+1;
    }
    bound=maximum;return Status::ok;
}
RS_HD inline Status contraction_bound(const MotionRig& r,const MotionSpline& s,Vec3 X,double t,int height,double& bound,const MotionProjectionContext* context=nullptr){
    const double half=std::abs(r.tau)/2;
    return contraction_bound_interval(r,s,X,t-half,t+half,height,bound,context);
}
struct MotionProjection {Vec2 uv;double depth,row,equality,contraction;};
// The pixel plane is not clipped to the raster. A finite prediction just beyond
// an image edge still has its ordinary pixel residual. Continue the SAME
// predicted-row equation, certifying depth, optical domain and uniqueness over
// every added interval. Never substitute the observed row or clamp row time.
// time_half is an optional declared computational support in seconds; a joint
// plan uses its compiled motion support. A negative value uses spline support.
RS_HD inline Status project_motion(const MotionRig& r,const MotionSpline& s,int height,Vec3 X,double time,MotionProjection& out,double time_half=-1.,const MotionProjectionContext* context=nullptr){
    if(!finite(X)||!finite_scalar(time))return Status::invalid_argument;
    if(context&&context->source_time!=time)context=nullptr;
    if(r.tau==0.){
        MotionInstant current{};auto status=context?context->centre_status:Status::ok;
        if(status==Status::ok)status=context?motion_instant_from_camera(r,context->centre,X,current,out.uv):motion_instant(r,s,X,time,current,out.uv);
        if(status!=Status::ok)return status;
        out.depth=current.depth;out.row=out.uv.y;out.equality=out.contraction=0.;return Status::ok;
    }
    auto status=contraction_bound(r,s,X,time,height,out.contraction,context);if(status!=Status::ok)return status;
    MotionInstant a{},b{};Vec2 ua{},ub{};const double last=height-1;
    status=context?context->lower_status:Status::ok;
    if(status==Status::ok)status=context?motion_instant_from_camera(r,context->lower,X,a,ua):motion_instant(r,s,X,time-r.tau/2,a,ua);if(status!=Status::ok)return status;
    status=context?context->upper_status:Status::ok;
    if(status==Status::ok)status=context?motion_instant_from_camera(r,context->upper,X,b,ub):motion_instant(r,s,X,time+r.tau/2,b,ub);if(status!=Status::ok)return status;
    double lo=0,hi=last,flo=-ua.y,fhi=last-ub.y;
    double first=s.first+s.dt,stop=s.first+(s.count-2)*s.dt;
    if(time_half>=0.){first=std::max(first,time-time_half);stop=std::min(stop,time+time_half);}
    if(time-std::abs(r.tau)/2<first-1e-12||time+std::abs(r.tau)/2>stop+1e-12)return Status::outside_support;
    const double ra=last*(.5+(first-time)/r.tau),rb=last*(.5+(stop-time)/r.tau);
    const double lower=std::min(ra,rb),upper=std::max(ra,rb);
    for(int extension=0;flo>1e-8||fhi< -1e-8;++extension){
        // Numerical work guard only: exhaustion is failure, never acceptance.
        if(extension==64)return static_cast<Status>(10);
        const bool below=flo>1e-8;const double edge=below?lo:hi,error=below?flo:-fhi;
        const double distance=error/(1-out.contraction),limit=below?lower:upper;
        const double proposed=below?edge-distance:edge+distance;
        const double next=below?std::max(limit,std::nextafter(proposed,-std::numeric_limits<double>::infinity())):
            std::min(limit,std::nextafter(proposed,std::numeric_limits<double>::infinity()));
        if(!finite_scalar(next)||(below?next>=edge:next<=edge))return Status::no_intersection;
        const double ta=time+r.tau*(edge/last-.5),tb=time+r.tau*(next/last-.5);
        double bound=0.;status=contraction_bound_interval(r,s,X,std::min(ta,tb),std::max(ta,tb),height,bound);
        if(status!=Status::ok)return status;out.contraction=std::max(out.contraction,bound);
        MotionInstant current{};Vec2 uv{};status=motion_instant(r,s,X,tb,current,uv);if(status!=Status::ok)return status;
        if(below){lo=next;flo=next-uv.y;a=current;ua=uv;}
        else{hi=next;fhi=next-uv.y;b=current;ub=uv;}
    }
    if(std::abs(flo)<=1e-8){out.uv=ua;out.depth=a.depth;out.row=lo;out.equality=flo;return Status::ok;}
    if(std::abs(fhi)<=1e-8){out.uv=ub;out.depth=b.depth;out.row=hi;out.equality=fhi;return Status::ok;}
    for(int iteration=0;iteration<64;++iteration){
        double row=lo-flo*(hi-lo)/(fhi-flo);if(!(row>lo&&row<hi)||iteration>16)row=(lo+hi)/2;
        MotionInstant current{};Vec2 uv{};status=motion_instant(r,s,X,time+r.tau*(row/last-.5),current,uv);if(status!=Status::ok)return status;
        const double f=row-uv.y;if(std::abs(f)<1e-8){out.uv=uv;out.row=row;out.equality=f;out.depth=current.depth;return Status::ok;}
        if(f>0){hi=row;fhi=f;}else{lo=row;flo=f;}
    }return static_cast<Status>(10);
}
} // namespace rinkspace
