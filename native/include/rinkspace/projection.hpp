#pragma once
#include "core_types.hpp"

namespace rinkspace {
struct Camera {
    Mat3 rotation;Vec3 centre;
    double fx,fy,cx,cy,skew,k1,k2,p1,p2,k3;
    bool valid()const{
        const double p[]={fx,fy,cx,cy,skew,k1,k2,p1,p2,k3};
        return valid_rotation(rotation)&&finite(centre)&&finite_array(p,10)&&fx>0&&fy>0;
    }
};
RS_HD inline Vec3 pivot_centre(const Mat3& rotation,Vec3 pivot,Vec3 lever,Vec3 axis_world,double spacing){
    return add(add(pivot,transpose_apply(rotation,lever)),mul(spacing,axis_world));
}
/* Shared static exposure primitive. Optional jac is d(pixel)/d(world point).
 * The moving physical head, native cubic trajectory and implicit row exposure
 * in camera_state.hpp call this same projection primitive.
 */
RS_HD inline Status project(const Camera& c,Vec3 world,Vec2& uv,double* jac=nullptr){
    if(!finite(world))return Status::invalid_argument;
    const Vec3 q=apply(c.rotation,sub(world,c.centre));
    if(!finite(q))return Status::nonfinite;
    if(q.z<=0)return Status::behind_camera;
    const double x=q.x/q.z,y=q.y/q.z,r2=x*x+y*y,r4=r2*r2;
    const double rad=1+c.k1*r2+c.k2*r4+c.k3*r4*r2;
    const double xd=x*rad+2*c.p1*x*y+c.p2*(r2+2*x*x);
    const double yd=y*rad+c.p1*(r2+2*y*y)+2*c.p2*x*y;
    uv={c.fx*xd+c.skew*yd+c.cx,c.fy*yd+c.cy};
    if(!std::isfinite(uv.x)||!std::isfinite(uv.y))return Status::nonfinite;
    if(jac){
        const double dr=2*(c.k1+2*c.k2*r2+3*c.k3*r4),rx=dr*x,ry=dr*y;
        const double xx=rad+x*rx+2*c.p1*y+6*c.p2*x;
        const double xy=x*ry+2*c.p1*x+2*c.p2*y;
        const double yx=y*rx+2*c.p1*x+2*c.p2*y;
        const double yy=rad+y*ry+6*c.p1*y+2*c.p2*x;
        const double a[]={c.fx*xx+c.skew*yx,c.fx*xy+c.skew*yy,c.fy*yx,c.fy*yy};
        for(int row=0;row<2;++row){
            const double j0=a[2*row]/q.z,j1=a[2*row+1]/q.z,j2=-(a[2*row]*x+a[2*row+1]*y)/q.z;
            for(int col=0;col<3;++col){
                jac[3*row+col]=j0*c.rotation.a[col]+j1*c.rotation.a[3+col]+j2*c.rotation.a[6+col];
                if(!std::isfinite(jac[3*row+col]))return Status::nonfinite;
            }
        }
    }
    return Status::ok;
}
} // namespace rinkspace
