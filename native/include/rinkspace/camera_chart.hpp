#pragma once
#include "local_jet.hpp"
#include "camera_state.hpp"

namespace rinkspace {
/* Local 27-coordinate physical chart. No free per-frame centre, roll, skew or
 * k3 is introduced. fref and the validated ray domain are conventions.
 * [R0 rotation3, pan tangent2, tilt around R0*pan1, pivot3, lever3,
 * axis spacing1, entrance pupil2, cx/cy/eta/cxz/cyz/k10/k11/k20/k21/p1/p2/tau12]. */
constexpr int motion_rig_chart_size=27;
template<class T> RS_HD inline Mat3Value<T> chart_exp_rotation(Vec3Value<T> v){
    using std::sin;using std::cos;using std::sqrt;
    T q=dot(v,v),a,b;
    if(q<1e-12){a=1-q/6+q*q/120;b=.5-q/24+q*q/720;}
    else{T t=sqrt(q);a=sin(t)/t;b=(1-cos(t))/q;}
    const T x=v.x,y=v.y,z=v.z;
    return {{1-b*(y*y+z*z),b*x*y-a*z,b*x*z+a*y,b*x*y+a*z,1-b*(x*x+z*z),b*y*z-a*x,b*x*z-a*y,b*y*z+a*x,1-b*(x*x+y*y)}};
}
template<class T> RS_HD inline MotionRigValue<T> rig_from_values(const T* p){
    MotionRigValue<T> r{};for(int i=0;i<9;++i)r.R0.a[i]=p[i];
    r.pan={p[9],p[10],p[11]};r.tilt={p[12],p[13],p[14]};r.pivot={p[15],p[16],p[17]};r.lever={p[18],p[19],p[20]};
    r.spacing=p[21];r.pupil1=p[22];r.pupil2=p[23];r.fref=p[24];r.radius=p[25];
    r.cx=p[26];r.cy=p[27];r.eta=p[28];r.cxz=p[29];r.cyz=p[30];r.k10=p[31];r.k11=p[32];r.k20=p[33];r.k21=p[34];r.p1=p[35];r.p2=p[36];r.tau=p[37];return r;
}
template<class T> RS_HD inline Status retract_motion_rig(const double* base,const T* d,T* out){
    const auto r=motion_rig(base);
    const Vec3 seed=std::abs(r.pan.x)<.8?Vec3{1,0,0}:Vec3{0,1,0};
    const auto e1=unit3(sub(seed,mul(dot(seed,r.pan),r.pan))),e2=cross3(r.pan,e1);
    auto pan=unit3(add(scalar_cast<T>(r.pan),add(mul(d[3],scalar_cast<T>(e1)),mul(d[4],scalar_cast<T>(e2)))));
    auto R=multiply3(chart_exp_rotation(Vec3Value<T>{d[0],d[1],d[2]}),scalar_cast<T>(r.R0));
    auto q=apply(R,pan),old=scalar_cast<T>(r.tilt),b=sub(old,mul(dot(old,q),q));
    if(dot(b,b)<1e-16)return Status::degenerate;
    b=apply(axis_rotation(q,d[5]),unit3(b));
    for(int i=0;i<38;++i)out[i]=T(base[i]);
    for(int i=0;i<9;++i)out[i]=R.a[i];
    out[9]=pan.x;out[10]=pan.y;out[11]=pan.z;out[12]=b.x;out[13]=b.y;out[14]=b.z;
    for(int i=0;i<9;++i)out[15+i]+=d[6+i];
    for(int i=0;i<12;++i)out[26+i]+=d[15+i];
    for(int i=0;i<38;++i)if(!finite_scalar(out[i]))return Status::nonfinite;
    return Status::ok;
}
RS_HD inline Status motion_rig_tangent(const double* rig,const double* directions,int q,int column,double* derivatives){
    LocalJet d[27],out[38];for(int i=0;i<27;++i)d[i]={0.,directions[size_t(i)*q+column]};
    auto st=retract_motion_rig(rig,d,out);if(st!=Status::ok)return st;
    for(int i=0;i<38;++i)derivatives[size_t(i)*q+column]=out[i].v;return Status::ok;
}
/* Implicit rolling-row pullback for any physical rig direction. The trajectory
 * time derivative includes entrance-pupil motion, zoom and Brown zoom terms. */
RS_HD inline Status motion_rig_projection_direction(const double* base,const double* derivatives,int q,int column,
    const MotionSpline& spline,int height,Vec3 xyz,double frame_time,const MotionProjection& root,
    const double* motion_partials,double* out,Vec3 xyz_derivative={0,0,0},const double* wide=nullptr,const double* wide_directions=nullptr){
    const double fraction=root.row/(height-1)-.5,t=frame_time+base[37]*fraction;
    double tr[3],w[4],dw[4],rate[3]{};int cell;
    auto st=spline.evaluate(t,tr);if(st!=Status::ok)return st;
    st=spline.basis(t,cell,w,dw);if(st!=Status::ok)return st;
    for(int c=0;c<3;++c)for(int j=0;j<4;++j)rate[c]+=dw[j]*spline.controls[3*size_t(cell-1+j)+c];
    double total_u=0.,total_v=0.;
    for(int c=0;c<3;++c){total_u+=motion_partials[c]*rate[c];total_v+=motion_partials[4+c]*rate[c];}
    LocalJet rig[38],angles[3];for(int i=0;i<38;++i)rig[i]={base[i],derivatives?derivatives[size_t(i)*q+column]:0.};
    for(int c=0;c<3;++c)angles[c]=LocalJet(tr[c]);
    auto physical=rig_from_values(rig);if(wide){physical.wide_boundary=LocalJet(wide[0]);for(int i=0;i<4;++i)physical.wide_coefficients[i]={wide[1+i],wide_directions?wide_directions[size_t(i)*q+column]:0.};}
    MotionCameraStateValue<LocalJet> camera{};st=motion_camera_from_values(physical,angles,camera);if(st!=Status::ok)return st;
    Vec2Value<LocalJet> uv{};st=project_uv(camera.camera,Vec3Value<LocalJet>{{xyz.x,xyz_derivative.x},{xyz.y,xyz_derivative.y},{xyz.z,xyz_derivative.z}},uv);if(st!=Status::ok)return st;
    // total_u/v are d(pixel)/d(time) AFTER implicit row elimination.
    double dt=base[37]/(height-1)*uv.y.v+fraction*rig[37].v;
    out[0]=uv.x.v+total_u*dt;out[1]=uv.y.v+total_v*dt;
    return finite_array(out,2)?Status::ok:Status::nonfinite;
}
} // namespace rinkspace
