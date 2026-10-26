#pragma once
#include "camera_chart.hpp"

namespace rinkspace {
// Explicit installation working envelope. No dimensions or numerical ranges
// are inferred here. The caller supplies scene units and radians.
// limits: complete forward lo/hi, pupil shift lo/hi, pan-from-vertical max,
// tilt-from-camera-right max, optical elevation lo/hi, body roll lo/hi.
// Optional next nine: absolute speed, acceleration, jerk limits, each in
// pan/tilt/log(f) order, per second. +infinity explicitly leaves a limit unset.
struct CameraEnvelope {
    bool enabled=false;double forward_lo=0,forward_hi=0,pupil_lo=0,pupil_hi=0;
    double pan_cos=0,tilt_cos=0,pitch_lo=0,pitch_hi=0,roll_lo=0,roll_hi=0;
    double motion_max[3][3]={{INFINITY,INFINITY,INFINITY},{INFINITY,INFINITY,INFINITY},{INFINITY,INFINITY,INFINITY}};
    // Optional selected exposure-midpoint interval in native spline-grid
    // coordinates. Each candidate's own readout expands it to all sensor rows.
    bool windowed=false;double frame_lo=0,frame_hi=0;
};
inline bool camera_envelope(const double* p,size_t n,CameraEnvelope& e){
    if(!p||(n!=10&&n!=19&&n!=21)||!finite_array(p,10)||p[0]>=p[1]||p[2]>=p[3]||p[2]>0||p[3]<0||
       p[4]<=0||p[4]>=1.5707963267948966||p[5]<=0||p[5]>=1.5707963267948966||
       p[6]>=p[7]||p[6]<=-1.5707963267948966||p[7]>=1.5707963267948966||
       p[8]>=p[9]||p[8]<=-1.5707963267948966||p[9]>=1.5707963267948966)return false;
    e={true,p[0],p[1],p[2],p[3],std::cos(p[4]),std::cos(p[5]),std::sin(p[6]),std::sin(p[7]),std::tan(p[8]),std::tan(p[9])};
    if(n>=19)for(int order=0;order<3;++order)for(int channel=0;channel<3;++channel){
        const double limit=p[10+3*order+channel];if(!(limit>0))return false;e.motion_max[order][channel]=limit;
    }
    if(n==21){if(!finite_array(p+19,2)||p[19]<1||p[20]<p[19])return false;
        e.windowed=true;e.frame_lo=p[19];e.frame_hi=p[20];}
    return true;
}
struct CameraConstraint {
    // 0 pan direction, 1 tilt direction, 2 continuous attitude certificate,
    // 3 pupil at a zoom endpoint, 4 full forward at a zoom endpoint,
    // 5 pupil at its quadratic vertex, 6 full forward at that vertex,
    // 7 speed, 8 acceleration, 9 jerk (absolute derivative limits).
    int kind=-1,cell=1,channel=-1;double u=.5,sign=1,minimum=0,nx=0,ny=0,nz=0,margin=0,limit=0;
};
RS_HD inline double camera_envelope_tolerance(){return 128*std::numeric_limits<double>::epsilon();}
template<class T> RS_HD inline T camera_spline_value(const T* c,int channel,double u){
    const double v=1-u,w[4]={v*v*v/6,(3*u*u*u-6*u*u+4)/6,(-3*u*u*u+3*u*u+3*u+1)/6,u*u*u/6};
    T x{};for(int i=0;i<4;++i)x+=c[3*i+channel]*w[i];return x;
}
template<class T> RS_HD inline T camera_angle_rate(const T* c,int channel,double dt){
    using std::abs;
    const T a=(-c[channel]+3*c[3+channel]-3*c[6+channel]+c[9+channel])/2;
    const T b=c[channel]-2*c[3+channel]+c[6+channel],d=(c[6+channel]-c[channel])/2;
    T result=abs(d);const T end=abs(a+b+d);if(end>result)result=end;
    if(primal(a)!=0){const double u=-primal(b)/(2*primal(a));if(u>0&&u<1){const T vertex=abs((a*u+b)*u+d);if(vertex>result)result=vertex;}}
    return result/dt;
}
template<class T> RS_HD inline T camera_constraint_value(const MotionRigValue<T>& r,const T* c,double dt,const CameraConstraint& a){
    using std::log;
    if(a.kind==0)return r.pan.z-a.minimum;
    if(a.kind==1)return r.tilt.x-a.minimum;
    // A dimensionless inequality, not a residual or a weight on source UV.
    // The extremum time is held while differentiating: its time derivative
    // vanishes at an interior speed extremum (envelope theorem).
    if(a.kind>=7)return 1-a.sign*camera_spline_derivative(c,a.channel,a.u,dt,a.kind-6)/a.limit;
    if(a.kind==2){
        const T alpha=camera_spline_value(c,0,.5),beta=camera_spline_value(c,1,.5);
        const auto R=multiply3(multiply3(axis_rotation(r.tilt,beta),r.R0),axis_rotation(r.pan,alpha));
        // |d(v^T R e_z)/dt| <= ||v|| (|pan'|+|tilt'|). The exact maxima
        // of both cubic derivatives certify the entire spline cell, including
        // rolling-row exposure times. This is a sufficient (padded) bound.
        const double length=std::sqrt(a.nx*a.nx+a.ny*a.ny+a.nz*a.nz);
        return a.nx*R.a[2]+a.ny*R.a[5]+a.nz*R.a[8]-a.minimum-
            length*(camera_angle_rate(c,0,dt)+camera_angle_rate(c,1,dt))*(dt/2);
    }
    const T z=a.kind>=5?-r.pupil1/(2*r.pupil2):camera_spline_value(c,2,a.u)-log(r.fref);
    T value=r.pupil1*z+r.pupil2*z*z;if(a.kind==4||a.kind==6)value+=r.lever.z;
    return a.sign*value-a.minimum;
}
RS_HD inline int camera_quadratic_roots(double a,double b,double c,double* roots){
    if(a==0){if(b==0)return 0;roots[0]=-c/b;return 1;}
    const double d=b*b-4*a*c;if(d<0)return 0;const double q=-.5*(b+std::copysign(std::sqrt(d),b));
    if(q==0){roots[0]=-b/(2*a);return 1;}roots[0]=q/a;roots[1]=c/q;return 2;
}
RS_HD inline CameraConstraint camera_cell_constraint(const MotionRig& r,const MotionSpline& s,const CameraEnvelope& e,int cell){
    CameraConstraint worst{};worst.margin=std::numeric_limits<double>::infinity();const double* c=s.controls+3*size_t(cell-1);
    if(e.windowed){
        const double half=std::abs(r.tau)/(2*s.dt),lo=e.frame_lo-half,hi=e.frame_hi+half;
        // Keep every complete cell touching the required row-exposure range.
        // The certificate may conservatively extend by at most one storage
        // cell at each end; allocated but unused extrapolation is not a shot.
        if(cell+1<lo||cell>hi){worst.cell=cell;worst.margin=std::numeric_limits<double>::max();return worst;}
    }
    auto offer=[&](CameraConstraint a){a.cell=cell;a.margin=camera_constraint_value(r,c,s.dt,a);if(!finite_scalar(a.margin)||a.margin<worst.margin)worst=a;};
    if(cell==1||e.windowed){CameraConstraint a{};a.kind=0;a.minimum=e.pan_cos;offer(a);a.kind=1;a.minimum=e.tilt_cos;offer(a);}
    CameraConstraint a{};a.kind=2;a.nz=1;a.minimum=e.pitch_lo;offer(a);
    a.nz=-1;a.minimum=-e.pitch_hi;offer(a);
    a.nz=0;a.nx=1;a.ny=e.roll_lo;a.minimum=0;offer(a);
    a.nx=-1;a.ny=-e.roll_hi;offer(a);
    a.nx=0;a.ny=-1;offer(a); // avoid an upside-down body satisfying a tangent ratio
    double us[4]={0,1,0,0};int count=2;
    const double da=(-c[2]+3*c[5]-3*c[8]+c[11])/2,db=c[2]-2*c[5]+c[8],dc=(c[8]-c[2])/2;
    double roots[2];int nr=camera_quadratic_roots(da,db,dc,roots);
    for(int i=0;i<nr;++i)if(roots[i]>0&&roots[i]<1)us[count++]=roots[i];
    double zlo=std::numeric_limits<double>::infinity(),zhi=-zlo;
    for(int i=0;i<count;++i){
        const double z=camera_spline_value(c,2,us[i])-std::log(r.fref);zlo=std::fmin(zlo,z);zhi=std::fmax(zhi,z);
        a={};a.kind=3;a.u=us[i];a.sign=1;a.minimum=e.pupil_lo;offer(a);a.sign=-1;a.minimum=-e.pupil_hi;offer(a);
        a.kind=4;a.sign=1;a.minimum=e.forward_lo;offer(a);a.sign=-1;a.minimum=-e.forward_hi;offer(a);
    }
    if(r.pupil2!=0){const double z=-r.pupil1/(2*r.pupil2);if(z>zlo&&z<zhi){
        a={};a.kind=5;a.sign=1;a.minimum=e.pupil_lo;offer(a);a.sign=-1;a.minimum=-e.pupil_hi;offer(a);
        a.kind=6;a.sign=1;a.minimum=e.forward_lo;offer(a);a.sign=-1;a.minimum=-e.forward_hi;offer(a);
    }}
    for(int channel=0;channel<3;++channel){
        double times[3];const int ntime=camera_velocity_extremum_times(c,channel,0,1,times);
        for(int order=1;order<=3;++order){
            const double limit=e.motion_max[order-1][channel];if(!finite_scalar(limit))continue;
            // Velocity is quadratic; acceleration is linear; jerk is constant.
            // Endpoints and the interior velocity vertex exhaust all extrema.
            const int checks=order==1?ntime:(order==2?2:1);
            for(int i=0;i<checks;++i){
                a={};a.kind=6+order;a.channel=channel;a.u=times[i];a.limit=limit;
                const double value=camera_spline_derivative(c,channel,a.u,s.dt,order);
                a.sign=value<0?-1:1;offer(a);
            }
        }
    }
    return worst;
}
inline CameraConstraint camera_envelope_check(const MotionRig& r,const MotionSpline& s,const CameraEnvelope& e){
    CameraConstraint worst{};worst.margin=std::numeric_limits<double>::infinity();
    if(e.windowed&&(e.frame_hi>s.count-2||e.frame_lo-std::abs(r.tau)/(2*s.dt)<1||e.frame_hi+std::abs(r.tau)/(2*s.dt)>s.count-2)){
        worst.kind=10;worst.margin=-std::numeric_limits<double>::infinity();return worst;}
    if(e.enabled)for(int cell=1;cell<int(s.count)-2;++cell){auto a=camera_cell_constraint(r,s,e,cell);if(!finite_scalar(a.margin)||a.margin<worst.margin)worst=a;}
    return worst;
}
} // namespace rinkspace
