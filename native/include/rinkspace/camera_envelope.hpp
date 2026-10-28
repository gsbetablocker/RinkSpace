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
    const double* frame_windows=nullptr;size_t frame_window_count=0;
};
inline bool camera_envelope(const double* p,size_t n,CameraEnvelope& e){
    if(!p||(n!=10&&n!=19&&(n<21||(n-19)%2))||!finite_array(p,10)||p[0]>=p[1]||p[2]>=p[3]||p[2]>0||p[3]<0||
       p[4]<=0||p[4]>=1.5707963267948966||p[5]<=0||p[5]>=1.5707963267948966||
       p[6]>=p[7]||p[6]<=-1.5707963267948966||p[7]>=1.5707963267948966||
       p[8]>=p[9]||p[8]<=-1.5707963267948966||p[9]>=1.5707963267948966)return false;
    e={true,p[0],p[1],p[2],p[3],std::cos(p[4]),std::cos(p[5]),std::sin(p[6]),std::sin(p[7]),std::tan(p[8]),std::tan(p[9])};
    if(n>=19)for(int order=0;order<3;++order)for(int channel=0;channel<3;++channel){
        const double limit=p[10+3*order+channel];if(!(limit>0))return false;e.motion_max[order][channel]=limit;
    }
    if(n>=21){
        if(!finite_array(p+19,n-19))return false;
        for(size_t i=19;i<n;i+=2)if(p[i]<1||p[i+1]<p[i]||(i>19&&p[i]<=p[i-1]))return false;
        e.windowed=true;e.frame_lo=p[19];e.frame_hi=p[n-1];
        e.frame_windows=p+19;e.frame_window_count=(n-19)/2;
    }
    return true;
}
struct CameraConstraint {
    // 0 pan direction, 1 tilt direction, 2 continuous attitude certificate,
    // 3 pupil at a zoom endpoint, 4 full forward at a zoom endpoint,
    // 5 pupil at its quadratic vertex, 6 full forward at that vertex,
    // 7 speed, 8 acceleration, 9 jerk (absolute derivative limits).
    int kind=-1,cell=1,channel=-1;double u=.5,sign=1,minimum=0,nx=0,ny=0,nz=0,margin=0,limit=0;
    // The attitude certificate is a maximum of signed derivative branches.
    // Keep the selected face when differentiating and distinguish it from
    // opposing signs/endpoints in the coupled active set.
    int rate_branch[2]={0,0};double rate_u[2]={0,0},rate_sign[2]={1,1};
    int camera=0; // identity of the physical rig in a joint retained system
    // Interior extrema of the zoom cubic move when controls change. Their
    // physical face identity is minimum/maximum, not the floating-point time.
    // 0 keeps fixed endpoint identity; 1 minimum, 2 maximum, 3 double root.
    int zoom_branch=0;
};
RS_HD inline bool same_camera_constraint_face(const CameraConstraint& a,const CameraConstraint& b){
    if(a.camera!=b.camera||a.kind!=b.kind||a.sign!=b.sign||a.minimum!=b.minimum)return false;
    // The quadratic pupil vertex is rig-wide; its cell is only a witness that
    // the vertex lies in an exposed zoom interval.
    if(a.kind==5||a.kind==6)return true;
    if(a.cell!=b.cell||a.channel!=b.channel||a.nx!=b.nx||a.ny!=b.ny||a.nz!=b.nz||a.limit!=b.limit)return false;
    if(a.kind==2)for(int i=0;i<2;++i)
        if(a.rate_branch[i]!=b.rate_branch[i]||a.rate_sign[i]!=b.rate_sign[i])return false;
    if(a.kind==3||a.kind==4){
        if(a.zoom_branch!=b.zoom_branch)return false;
        if(a.zoom_branch)return true;
    }
    return a.u==b.u;
}
RS_HD inline double camera_envelope_tolerance(){return 128*std::numeric_limits<double>::epsilon();}
template<class T> RS_HD inline T camera_spline_value(const T* c,int channel,double u){
    const double v=1-u,w[4]={v*v*v/6,(3*u*u*u-6*u*u+4)/6,(-3*u*u*u+3*u*u+3*u+1)/6,u*u*u/6};
    T x{};for(int i=0;i<4;++i)x+=c[3*i+channel]*w[i];return x;
}
RS_HD inline void camera_attitude_rate_faces(const double* c,double dt,CameraConstraint& a){
    for(int channel=0;channel<2;++channel){
        double times[3];const int count=camera_velocity_extremum_times(c,channel,0,1,times);
        double maximum=-1;
        for(int i=0;i<count;++i){
            const double value=camera_spline_derivative(c,channel,times[i],dt,1);
            if(std::abs(value)>maximum){maximum=std::abs(value);a.rate_branch[channel]=i;
                a.rate_u[channel]=times[i];a.rate_sign[channel]=value<0?-1:1;}
        }
    }
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
            length*(a.rate_sign[0]*camera_spline_derivative(c,0,a.rate_u[0],dt,1)+
                    a.rate_sign[1]*camera_spline_derivative(c,1,a.rate_u[1],dt,1))*(dt/2);
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
template<class Visit> RS_HD inline void camera_cell_constraints(const MotionRig& r,const MotionSpline& s,const CameraEnvelope& e,int cell,Visit visit){
    const double* c=s.controls+3*size_t(cell-1);
    if(e.windowed){
        const double half=std::abs(r.tau)/(2*s.dt);
        // Keep every complete cell touching the required row-exposure range.
        // The certificate may conservatively extend by at most one storage
        // cell at each end; allocated but unused extrapolation is not a shot.
        bool selected=false;
        for(size_t i=0;i<e.frame_window_count;++i){
            const double lo=e.frame_windows[2*i]-half,hi=e.frame_windows[2*i+1]+half;
            selected|=cell+1>=lo&&cell<=hi;
        }
        if(!selected)return;
    }
    auto offer=[&](CameraConstraint a){a.cell=cell;a.margin=camera_constraint_value(r,c,s.dt,a);visit(a);};
    if(cell==1||e.windowed){CameraConstraint a{};a.kind=0;a.minimum=e.pan_cos;offer(a);a.kind=1;a.minimum=e.tilt_cos;offer(a);}
    CameraConstraint a{};a.kind=2;camera_attitude_rate_faces(c,s.dt,a);a.nz=1;a.minimum=e.pitch_lo;offer(a);
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
        a={};a.kind=3;a.u=us[i];
        if(i>=2){const double curvature=2*da*us[i]+db;a.zoom_branch=curvature>0?1:(curvature<0?2:3);}
        a.sign=1;a.minimum=e.pupil_lo;offer(a);a.sign=-1;a.minimum=-e.pupil_hi;offer(a);
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
}
RS_HD inline CameraConstraint camera_cell_constraint(const MotionRig& r,const MotionSpline& s,const CameraEnvelope& e,int cell){
    CameraConstraint worst{};worst.cell=cell;worst.margin=std::numeric_limits<double>::max();
    camera_cell_constraints(r,s,e,cell,[&](CameraConstraint a){if(!finite_scalar(a.margin)||a.margin<worst.margin)worst=a;});
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
