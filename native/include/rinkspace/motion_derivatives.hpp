#pragma once
#include "local_jet.hpp"
#include "entities.hpp"
#include "camera_state.hpp"

namespace rinkspace {
/* At an already-certified implicit row root: directional AD through the SAME
 * instantaneous camera/geometry, then the exact implicit-function pullback.
 * Output two rows by [instant pan,tilt,logf,station]. No global finite differences. */
RS_HD inline Status motion_point_partials(const MotionRig& rig,const MotionSpline& spline,int height,
    Vec3 xyz,Vec3 station_tangent,double frame_time,const MotionProjection& root,
    double* jac,int& cell,double* weights){
    const double t=frame_time+rig.tau*(root.row/(height-1)-.5);
    double tr[3],dw[4],rate[3]{};
    auto status=spline.evaluate(t,tr);if(status!=Status::ok)return status;
    status=spline.basis(t,cell,weights,dw);if(status!=Status::ok)return status;
    for(int c=0;c<3;++c)for(int j=0;j<4;++j)rate[c]+=dw[j]*spline.controls[3*size_t(cell-1+j)+c];
    double instant[8];
    for(int direction=0;direction<4;++direction){
        LocalJet angles[3];for(int c=0;c<3;++c)angles[c]=LocalJet(tr[c],direction==c?1.:0.);
        const double ds=direction==3?1.:0.;
        const Vec3Value<LocalJet> X{LocalJet(xyz.x,ds*station_tangent.x),
            LocalJet(xyz.y,ds*station_tangent.y),LocalJet(xyz.z,ds*station_tangent.z)};
        MotionCameraStateValue<LocalJet> camera{};
        status=motion_camera_from_values(rig,angles,camera);if(status!=Status::ok)return status;
        Vec2Value<LocalJet> uv{};
        status=project_uv(camera.camera,X,uv);if(status!=Status::ok)return status;
        instant[direction]=uv.x.v;instant[4+direction]=uv.y.v;
    }
    double du_dt=0,dv_dt=0;
    for(int c=0;c<3;++c){du_dt+=instant[c]*rate[c];dv_dt+=instant[4+c]*rate[c];}
    const double k=rig.tau/(height-1),den=1-k*dv_dt;
    if(!(den>1e-9)||!finite_scalar(den))return static_cast<Status>(9);
    for(int c=0;c<4;++c){jac[4+c]=instant[4+c]/den;jac[c]=instant[c]+du_dt*k*jac[4+c];}
    return finite_array(jac,8)?Status::ok:Status::nonfinite;
}
/* Curve search needs only the station derivative. Contract the three motion
 * directions with their time rates before AD, rather than rebuilding the same
 * instantaneous camera once per pan/tilt/zoom direction. The implicit row root,
 * denominator and optical model are unchanged (including off-image roots). */
RS_HD inline Status motion_station_partial(const MotionRig& rig,const MotionSpline& spline,int height,
    Vec3 xyz,Vec3 station_tangent,double frame_time,const MotionProjection& root,Vec2& jac){
    const double t=frame_time+rig.tau*(root.row/(height-1)-.5);
    double tr[3],rate[3]{};auto status=spline.evaluate(t,tr);if(status!=Status::ok)return status;
    if(rig.tau!=0.){int cell;double weights[4],dw[4];status=spline.basis(t,cell,weights,dw);if(status!=Status::ok)return status;
        for(int c=0;c<3;++c)for(int j=0;j<4;++j)rate[c]+=dw[j]*spline.controls[3*size_t(cell-1+j)+c];}
    Vec2 derivatives[2]{};
    for(int direction=0;direction<(rig.tau==0.?1:2);++direction){
        LocalJet angles[3];for(int c=0;c<3;++c)angles[c]=LocalJet(tr[c],direction?rate[c]:0.);
        const Vec3Value<LocalJet> X{LocalJet(xyz.x,direction?0.:station_tangent.x),
            LocalJet(xyz.y,direction?0.:station_tangent.y),LocalJet(xyz.z,direction?0.:station_tangent.z)};
        MotionCameraStateValue<LocalJet> camera{};status=motion_camera_from_values(rig,angles,camera);if(status!=Status::ok)return status;
        Vec2Value<LocalJet> uv{};status=project_uv(camera.camera,X,uv);if(status!=Status::ok)return status;
        derivatives[direction]={uv.x.v,uv.y.v};
    }
    const double k=rig.tau/(height-1),den=1-k*derivatives[1].y;
    if(!(den>1e-9)||!finite_scalar(den))return static_cast<Status>(9);
    jac.y=derivatives[0].y/den;jac.x=derivatives[0].x+derivatives[1].x*k*jac.y;
    return finite_scalar(jac.x)&&finite_scalar(jac.y)?Status::ok:Status::nonfinite;
}
RS_HD inline Status motion_curve_partials(const MotionRig& rig,const MotionSpline& spline,int height,
    int kind,const double* parameters,double station,double frame_time,const MotionProjection& root,
    double* jac,int& cell,double* weights){
    auto x=sample_planar_primitive(kind,parameters,LocalJet(station,1.));
    return motion_point_partials(rig,spline,height,{x.x.a,x.y.a,x.z.a},{x.x.v,x.y.v,x.z.v},
        frame_time,root,jac,cell,weights);
}

}
