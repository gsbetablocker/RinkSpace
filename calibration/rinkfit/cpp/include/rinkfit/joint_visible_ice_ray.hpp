#pragma once
#include "rinkfit/joint_model.hpp"
#include <string>
#include <algorithm>
namespace rinkfit {
// Source-only free-space witness. No material station/world point is read.
// The numerical inverse is lifted with the Brown implicit derivative, rather
// than differentiating Newton iteration/line-search decisions. Global positive
// Brown Jacobian lower bound certifies uniqueness on the retained finite disk.
struct JointBrownInverseValue {Vec2<double>xy{};double pixel_error=INFINITY;int iterations=0;bool valid=false;std::string reason="not_evaluated";};
inline JointBrownInverseValue joint_brown_inverse_value(const Vec2<double>&target,const Brown<double>&d,double fx,double fy){
 JointBrownInverseValue out;if(!finite_array(target)||!finite_scalar(d.radius_max)||d.radius_max<=0||!(fx>0&&fy>0)){out.reason="nonfinite_intrinsic_or_disk";return out;}
 if(!(brown_no_fold_lower_bound(d)>0)){out.reason="uncertified_whole_Brown_disk";return out;}
 Vec2<double>x=target;const double rr=std::hypot(x[0],x[1]);if(rr>d.radius_max){const double f=d.radius_max/(2*rr);x[0]*=f;x[1]*=f;}
 auto err=[&](const Vec2<double>&q){const auto y=brown_distort(q,d);return std::hypot((y[0]-target[0])*fx,(y[1]-target[1])*fy);};
 for(;;){
  const auto y=brown_distort(x,d);const double e=err(x),eps=32*std::numeric_limits<double>::epsilon()*std::max({1.,fx,fy})*(1+std::hypot(target[0],target[1]));
  if(e<=eps){out.xy=x;out.pixel_error=e;out.valid=true;out.reason="unique_in_certified_disk";return out;}
  const double r2=x[0]*x[0]+x[1]*x[1],L=1+d.k1*r2+d.k2*r2*r2,A=d.k1+2*d.k2*r2;
  const double a=L+2*x[0]*x[0]*A+2*d.p1*x[1]+6*d.p2*x[0],b=2*x[0]*x[1]*A+2*d.p1*x[0]+2*d.p2*x[1],c=L+2*x[1]*x[1]*A+6*d.p1*x[1]+2*d.p2*x[0],det=a*c-b*b;
  if(!std::isfinite(det)||a<=0||det<=0){out.reason="inverse_local_Jacobian_not_positive";return out;}
  const Vec2<double>r={y[0]-target[0],y[1]-target[1]},step={(c*r[0]-b*r[1])/det,(a*r[1]-b*r[0])/det};bool accepted=false;
  for(double f=1.;f>0;f*=.5){const Vec2<double>q={x[0]-f*step[0],x[1]-f*step[1]};if(q==x)break;if(std::hypot(q[0],q[1])<=d.radius_max&&err(q)<e){x=q;accepted=true;break;}}
  if(!accepted){out.reason="inverse_no_representable_decreasing_in_disk_step";return out;}++out.iterations;
 }
}
template<class T>struct JointVisibleIceRayResult {
 bool arithmetic_valid=false,inverse_valid=false,finite_ice=false,visibility_evaluated=false;std::string reason="not_evaluated";
 Vec2<T>inverse_xy{};Vec3<T>C{},direction{},ice_intersection{};T source_row_time=T(0),depth=T(0),rink_g=T(0),camera_above_ice_g=T(0),forward_depth_g=T(0),occlusion_g=T(0),inverse_ray_g=T(0),Brown_disk_lower=T(0),forward_error_px=T(0);
 bool feasible(double tolerance=1e-8)const{return arithmetic_valid&&inverse_valid&&finite_ice&&visibility_evaluated&&event_scalar(camera_above_ice_g)<0&&event_scalar(forward_depth_g)<0&&event_scalar(occlusion_g)<=tolerance;}
};
template<class T>JointVisibleIceRayResult<T>joint_visible_ice_sensor_ray(const JointEventModel&model,const JointRowLayout&layout,const Vec2<double>&uv,T const*const*pars){
 JointVisibleIceRayResult<T>out;
 if(!finite_array(uv)||uv[0]<0||uv[0]>1919||uv[1]<0||uv[1]>1079){out.reason="source_UV_outside_native_sensor";return out;}
 const auto g=model.global(layout,pars);std::vector<const T*>p;std::vector<T>unused_geometry;const auto e=model.legacy_layout(layout,p,g,pars,unused_geometry);const auto&math=model.shots.at(layout.shot_index).math;
 out.source_row_time=row_time(T(layout.frame_time),g[31],T(uv[1]));auto tr=math.trajectory(out.source_row_time,p.data(),e);tr[0]+=pars[layout.reference_angles_slot][0];tr[1]+=pars[layout.reference_angles_slot][1];const auto o=zoom_optics(math.optics(g.data()),tr[2]);const auto pose=camera_pose(math.head(g.data()),tr[0],tr[1],o.z);
 if(!pose.valid()||!valid_rotation(pose.R)){out.reason="invalid_actual_source_row_pose";return out;}out.C=pose.C;out.Brown_disk_lower=brown_no_fold_lower_bound(o.brown);
 const Vec2<T>target={(T(uv[0])-o.intr.cx)/o.intr.fx,(T(uv[1])-o.intr.cy)/o.intr.fy};Brown<double>bd{event_scalar(o.brown.k1),event_scalar(o.brown.k2),event_scalar(o.brown.p1),event_scalar(o.brown.p2),event_scalar(o.brown.radius_max)};
 const auto inv=joint_brown_inverse_value({event_scalar(target[0]),event_scalar(target[1])},bd,event_scalar(o.intr.fx),event_scalar(o.intr.fy));if(!inv.valid){out.reason=inv.reason;return out;}out.inverse_valid=true;
 const double x=inv.xy[0],y=inv.xy[1],r2=x*x+y*y,L=1+bd.k1*r2+bd.k2*r2*r2,A=bd.k1+2*bd.k2*r2,a=L+2*x*x*A+2*bd.p1*y+6*bd.p2*x,b=2*x*y*A+2*bd.p1*x+2*bd.p2*y,c=L+2*y*y*A+6*bd.p1*y+2*bd.p2*x,det=a*c-b*b;
 const auto y0=brown_distort(Vec2<T>{T(x),T(y)},o.brown);const Vec2<T>r={y0[0]-target[0],y0[1]-target[1]};out.inverse_xy={T(x)-(T(c)*r[0]-T(b)*r[1])/T(det),T(y)-(T(a)*r[1]-T(b)*r[0])/T(det)};
 out.inverse_ray_g=out.inverse_xy[0]*out.inverse_xy[0]+out.inverse_xy[1]*out.inverse_xy[1]-o.brown.radius_max*o.brown.radius_max;
 if(event_scalar(out.inverse_ray_g)>1e-14){out.reason="inverse_root_outside_retained_Brown_disk";return out;}
 out.direction=matvec(transpose(pose.R),Vec3<T>{out.inverse_xy[0],out.inverse_xy[1],T(1)});out.camera_above_ice_g=-pose.C[2];
 if(!finite_array(out.direction)||std::abs(event_scalar(out.direction[2]))<=1e-14){out.reason="ray_parallel_to_ice_or_singular";return out;}
 out.depth=-pose.C[2]/out.direction[2];out.forward_depth_g=-out.depth;out.ice_intersection=add(pose.C,scale(out.direction,out.depth));out.ice_intersection[2]=T(0);out.rink_g=EventModel::rink_membership(out.ice_intersection[0],out.ice_intersection[1],g.data());out.finite_ice=event_scalar(out.rink_g)<=0;
 const auto replay=project(out.ice_intersection,pose,o.intr,o.brown);if(!replay.valid()){out.reason="inverse_ray_forward_replay_invalid";return out;}
 using std::sqrt;const T dx=replay.uv[0]-T(uv[0]),dy=replay.uv[1]-T(uv[1]);out.forward_error_px=sqrt(dx*dx+dy*dy+T(1e-60));
 if(!finite_array(out.ice_intersection)||!finite_scalar(out.depth)||!finite_scalar(out.rink_g)){out.reason="nonfinite_ice_intersection";return out;}
 if(!out.finite_ice){out.reason="ray_ice_intersection_outside_finite_rink_visibility_not_attempted";return out;}
 const auto vis=joint_visibility(g.data(),pose.C,out.ice_intersection,false);if(!vis.arithmetic_valid){out.reason="near_board_intersection_not_differentiable_or_invalid";return out;}
 out.visibility_evaluated=true;out.occlusion_g=vis.occlusion;out.arithmetic_valid=true;out.reason=out.feasible()?"source_ice_ray_physically_visible":"source_ice_ray_physical_inequality_violation";return out;
}
}
