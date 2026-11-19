#pragma once
#include "rinkspace/camera_state.hpp"

namespace motion_spline_rates_test {
using namespace rinkspace;

inline constexpr const char* case_names[] = {
    "first endpoint", "last endpoint", "accepted first-endpoint roundoff",
    "accepted last-endpoint roundoff", "large common offset", "reversed interval",
    "NaN start", "NaN end", "outside support", "nonfinite derivative",
    "nonfinite padded bound", "across a knot", "interior velocity extremum",
    "ordinary interval", "rolling readout at support edge", "zero readout at endpoint",
    "zero readout outside support"
};
inline constexpr int case_count = sizeof(case_names)/sizeof(*case_names);

struct Result {
    Status status, reference_status;
    double rates[3], sampled[3];
    MotionProjection projection;
    Vec2 instant;
    double instant_depth;
};

RS_HD inline Result run_case(int index){
    constexpr double dt=.1;
    double controls[15]{};
    for(int i=0;i<5;++i){
        controls[3*i]=.125*i+(index==4?1e15:0.);
        controls[3*i+1]=-.25*i;
        controls[3*i+2]=.375*i;
    }
    MotionSpline spline{controls,4,0.,dt};
    double lo=.1,hi=.2;
    const double inf=std::numeric_limits<double>::infinity();
    switch(index){
    case 0: hi=lo;break;
    case 1: lo=hi;break;
    case 2: lo=hi=std::nextafter(lo,-inf);break;
    case 3: lo=hi=std::nextafter(hi,inf);break;
    case 4: lo=.19;break;
    case 5: lo=.19;hi=.11;break;
    case 6: lo=std::numeric_limits<double>::quiet_NaN();break;
    case 7: hi=std::numeric_limits<double>::quiet_NaN();break;
    case 8: hi=.201;break;
    case 9:
        for(int i=0;i<4;++i)controls[3*i]=i*3e307;
        break;
    case 10:
        spline.dt=1.;
        lo=hi=1.;
        controls[0]=-std::numeric_limits<double>::max();
        controls[3]=0.;
        controls[6]=controls[9]=std::numeric_limits<double>::max();
        break;
    }
    if(index>=11&&index<=13){
        const double ordinary[15]={0.,.1,.4, 0.,.2,.1, 1.,.1,.3, 1.,0.,.5, .8,.4,0.};
        for(int i=0;i<15;++i)controls[i]=ordinary[i];
        spline.count=index==12?4:5;
        lo=index==11?.17:.11;
        hi=index==11?.26:(index==12?.19:.15);
    }
    Result result{};
    if(index>=14){
        MotionRig rig{};
        rig.R0={{1,0,0,0,1,0,0,0,1}};
        rig.pan={0,0,1};rig.tilt={1,0,0};
        rig.fref=100.;rig.radius=1.;rig.cx=rig.cy=50.;
        rig.tau=index==14?.02:0.;
        for(int i=0;i<4;++i){
            controls[3*i]=.0125*i;controls[3*i+1]=.00625*i;controls[3*i+2]=std::log(100.);
        }
        const double t=index==14?.19:(index==15?.2:.201);
        const Vec3 point{.1,.2,3.};
        MotionProjectionContext context{};
        prepare_motion_projection_context(rig,spline,t,context);
        // A zero-readout projection must not consume a cached rate failure.
        if(rig.tau==0.)context.rates_status=Status::nonfinite;
        result.status=project_motion(rig,spline,101,point,t,result.projection,-1.,&context);
        if(rig.tau==0.){
            MotionInstant instant{};
            result.reference_status=motion_instant(rig,spline,point,t,instant,result.instant);
            result.instant_depth=instant.depth;
        }
        return result;
    }
    result.status=spline.rates(lo,hi,result.rates);
    if(index<5||(index>=11&&index<=13)){
        for(int i=0;i<=100;++i){
            const double time=i==100?hi:lo+(hi-lo)*i/100.;
            double kinematics[12];
            result.reference_status=spline.kinematics(time,kinematics);
            if(result.reference_status!=Status::ok)break;
            for(int c=0;c<3;++c)result.sampled[c]=std::max(result.sampled[c],std::abs(kinematics[3+c]));
        }
    }
    return result;
}

inline Status expected_status(int index){
    if(index>=5&&index<=7)return Status::invalid_argument;
    if(index==8||index==16)return Status::outside_support;
    if(index==9||index==10)return Status::nonfinite;
    return Status::ok;
}

inline bool passes(int index,const Result& result){
    if(result.status!=expected_status(index))return false;
    if(index==16)return result.reference_status==Status::outside_support;
    if(result.status!=Status::ok)return true;
    if(index==14){
        return finite_scalar(result.projection.uv.x)&&finite_scalar(result.projection.uv.y)&&
            std::abs(result.projection.equality)<=1e-8&&
            result.projection.contraction>0.&&result.projection.contraction<1.-1e-9;
    }
    if(result.reference_status!=Status::ok)return false;
    if(index==15){
        return result.projection.uv.x==result.instant.x&&result.projection.uv.y==result.instant.y&&
            result.projection.depth==result.instant_depth&&result.projection.row==result.instant.y&&
            result.projection.equality==0.&&result.projection.contraction==0.;
    }
    for(int c=0;c<3;++c){
        if(!finite_scalar(result.rates[c])||result.rates[c]<result.sampled[c])return false;
    }
    if(index<5){
        for(int c=0;c<3;++c){
            const double expected=1.25*(c+1);
            const double padded=std::nextafter(expected+2e-11*(1+expected),
                std::numeric_limits<double>::infinity());
            if(std::abs(result.rates[c]-padded)>1e-12)return false;
        }
    }
    // The ordinary controls give first differences [0, 1, 0] in channel 0.
    // The exact maximum is at u=.5, strictly between the queried endpoints.
    if(index==12&&!(result.rates[0]>=7.5&&result.rates[0]<7.5+2e-10))return false;
    return true;
}

inline bool agrees(int index,const Result& host,const Result& device){
    if(host.status!=Status::ok)return true;
    if(index==14||index==15){
        return std::abs(host.projection.uv.x-device.projection.uv.x)<=1e-9&&
            std::abs(host.projection.uv.y-device.projection.uv.y)<=1e-9&&
            std::abs(host.projection.depth-device.projection.depth)<=1e-12&&
            std::abs(host.projection.row-device.projection.row)<=1e-9&&
            std::abs(host.projection.contraction-device.projection.contraction)<=1e-12;
    }
    for(int c=0;c<3;++c){
        if(std::abs(host.rates[c]-device.rates[c])>1e-12*std::max(1.,host.rates[c]))return false;
    }
    return true;
}
}
