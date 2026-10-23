#pragma once
#include "rinkfit/joint_finite_source_paths_v1.hpp"
#include <ceres/jet.h>
namespace rinkfit {
struct JointObservedCurveFootV1{bool valid=false;int path=-1;double station=0,squared_error=INFINITY,gradient=NAN,tangent_error=NAN,normal_error=NAN,tangent_speed=NAN;std::array<double,2>prediction{};std::size_t evaluations=0,local_basins=0;};
// Initialization-only: physical parameters and original source_y are fixed.
// Reuse the existing 32-cell chart navigation, refining EVERY sampled basin.
// This is broad finite-chart navigation, not a mathematical global lower-bound certificate.
inline JointObservedCurveFootV1 profile_joint_observed_curve_v1(const JointEventModel&m,const JointPacket&p,std::size_t index,bool current_path_only=false,bool fast_polyline=false){
 const auto&r=p.rows.at(index);JointObservedCurveFootV1 best;best.station=r.station;best.path=r.path;std::vector<int>paths=r.kind==1?std::vector<int>{r.path}:joint_source_allowed_finite_paths_v1(p,m,index);
 if(current_path_only)paths.erase(std::remove_if(paths.begin(),paths.end(),[&](int path){return path!=r.path;}),paths.end());
 for(int path:paths){const auto l=m.layout(m.shot_map.at(r.shot_id),r.material,path,r.pts,r.width_block,r.width_sign);const auto ptr=m.pointers(l);// Source row time and physical model are unchanged while profiling s.
 // Cache their exact camera/optical calculation once per finite path.
 const auto g=m.global(l,ptr.data());std::vector<const double*>frozen;std::vector<double>geometry;
 const auto layout=m.legacy_layout(l,frozen,g,ptr.data(),geometry);const auto&math=m.shots.at(l.shot_index).math;
 auto tr=math.trajectory(row_time(l.frame_time,g[31],r.uv[1]),frozen.data(),layout);
 tr[0]+=ptr[l.reference_angles_slot][0];tr[1]+=ptr[l.reference_angles_slot][1];
 const auto optical=zoom_optics(math.optics(g.data()),tr[2]);const auto pose=camera_pose(math.head(g.data()),tr[0],tr[1],optical.z);
 const double brown_lower=brown_no_fold_lower_bound(optical.brown);const bool camera_valid=pose.valid()&&valid_rotation(pose.R)&&std::isfinite(brown_lower)&&brown_lower>0;
 const auto&material=m.materials.at(r.material);const auto&shape=material.paths.at(path);
 const bool top=(material.label=="board.top"||material.label.rfind("board.top.",0)==0)&&shape.kind==3&&shape.geometry_block<0&&shape.fixed_height_fraction==1.;
 const bool board=(shape.kind==2||shape.kind==3||shape.kind==6||shape.kind==7)&&!top;
 double radius_width_domain=-1.;if(l.width_slot>=0&&(shape.kind==1||shape.kind==4)){const int ri=shape.kind==1?shape.radius_index:2;radius_width_domain=(1e-12-geometry.at(ri)+double(l.width_sign+1)*ptr[l.width_slot][0])/.05;}
 Vec2<double> prediction{};
 auto cost=[&](double s){++best.evaluations;if(!camera_valid||!r.station_allowed(m,path,s)||radius_width_domain>0)return double(INFINITY);
  const auto world=m.world_point(l,s,frozen.data(),layout,math);const auto xc=matvec(pose.R,sub(world,pose.C));if(!(xc[2]>0)||!finite_array(xc))return double(INFINITY);
  const Vec2<double>xy={xc[0]/xc[2],xc[1]/xc[2]};if(xy[0]*xy[0]+xy[1]*xy[1]>optical.brown.radius_max*optical.brown.radius_max)return double(INFINITY);
  if(!joint_visibility(g.data(),pose.C,world,board).visible()||(l.width_slot>=0&&EventModel::rink_membership(world[0],world[1],g.data())>0))return double(INFINITY);
  const auto d=brown_distort(xy,optical.brown);prediction={optical.intr.fx*d[0]+optical.intr.cx,optical.intr.fy*d[1]+optical.intr.cy};if(!finite_array(prediction))return double(INFINITY);
  const double a=prediction[0]-r.uv[0],b=prediction[1]-r.uv[1];return a*a+b*b;
 };
 auto record=[&](double s,double f){if(f<best.squared_error){best.valid=true;best.path=path;best.station=s;best.squared_error=f;best.prediction=prediction;}};
 auto consider=[&](double s){const double f=cost(s);record(s,f);};
 if(r.kind==1){consider(r.station);continue;}for(auto interval:r.station_intervals(m,path)){std::vector<double>breaks{interval[0],interval[1]};const auto&shape=m.materials.at(r.material).paths.at(path);int segments=0;if(shape.kind==2)segments=int(shape.points.size())-1;else if(shape.kind==6){const int n=l.geometry_indices.empty()?(int(m.blocks.at(shape.geometry_block).values.size())-l.geometry_offset)/2:int(l.geometry_indices.size()/2);segments=shape.closed?n:n-1;}for(int k=1;k<segments;++k){double s=double(k)/segments;if(s>interval[0]&&s<interval[1])breaks.push_back(s);}std::sort(breaks.begin(),breaks.end());if(path==r.path&&r.station>=interval[0]&&r.station<=interval[1])consider(r.station);if(fast_polyline&&(shape.kind==2||shape.kind==6)){
  // The projection of a short source polyline edge supplies a navigation
  // seed only. Every candidate and the final refinement use the exact
  // original curve, visibility and native camera, with unchanged identity.
  double seed_cost=INFINITY,seed_lo=0.,seed_hi=0.;
  auto seed=[&](double s,double a,double b){double f=cost(s);record(s,f);if(f<seed_cost){seed_cost=f;seed_lo=a;seed_hi=b;}};
  double fa=cost(breaks.front());auto ua=prediction;record(breaks.front(),fa);
  for(std::size_t segment=1;segment<breaks.size();++segment){double a=breaks[segment-1],b=breaks[segment],fb=cost(b);auto ub=prediction;record(b,fb);
   if(std::isfinite(fa)&&std::isfinite(fb)){double dx=ub[0]-ua[0],dy=ub[1]-ua[1],norm=dx*dx+dy*dy;double t=norm>0?std::clamp(((r.uv[0]-ua[0])*dx+(r.uv[1]-ua[1])*dy)/norm,0.,1.):0.;seed(a+(b-a)*t,a,b);}
   else for(int k=0;k<=32;++k)seed(a+(b-a)*k/32.,a,b);
   fa=fb;ua=ub;
  }
  if(std::isfinite(seed_cost)){++best.local_basins;double lo=seed_lo,hi=seed_hi;while(true){double u=lo+(hi-lo)*.3819660112501051,v=lo+(hi-lo)*.6180339887498949;if(!(u>lo&&v>u&&v<hi))break;double fu=cost(u);record(u,fu);double fv=cost(v);record(v,fv);if(fu<fv)hi=v;else lo=u;}
   consider(lo);consider(hi);consider((lo+hi)*.5);
  }
  continue;
 }
 for(std::size_t segment=1;segment<breaks.size();++segment){const double a=breaks[segment-1],b=breaks[segment];std::array<double,33>x{},f{};for(int k=0;k<=32;++k){x[k]=a+(b-a)*k/32.;f[k]=cost(x[k]);if(std::isfinite(f[k]))record(x[k],f[k]);}for(int k=0;k<=32;++k)if(std::isfinite(f[k])&&(k==0||f[k]<=f[k-1])&&(k==32||f[k]<=f[k+1])){++best.local_basins;double lo=x[std::max(0,k-1)],hi=x[std::min(32,k+1)];while(true){double u=lo+(hi-lo)*.3819660112501051,v=lo+(hi-lo)*.6180339887498949;if(!(u>lo&&v>u&&v<hi))break;const double fu=cost(u),fv=cost(v);if(fu<fv)hi=v;else lo=u;}consider(lo);consider(hi);consider((lo+hi)*.5);}}}
 }
 if(best.valid){const auto l=m.layout(m.shot_map.at(r.shot_id),r.material,best.path,r.pts,r.width_block,r.width_sign);using K=ceres::Jet<double,1>;std::vector<std::vector<K>>v;std::vector<const K*>ptr;for(int id:l.block_ids){v.emplace_back();for(double x:m.blocks.at(id).values)v.back().emplace_back(x);}for(const auto&x:v)ptr.push_back(x.data());K station(best.station);station.v[0]=1;const auto uv=m.predict_explicit(l,station,K(r.uv[1]),ptr.data());if(uv.valid()){const double dx=uv.uv[0].a-r.uv[0],dy=uv.uv[1].a-r.uv[1],tx=uv.uv[0].v[0],ty=uv.uv[1].v[0];best.gradient=2*(dx*tx+dy*ty);best.tangent_speed=std::hypot(tx,ty);if(best.tangent_speed>0){best.tangent_error=(dx*tx+dy*ty)/best.tangent_speed;best.normal_error=(dx*ty-dy*tx)/best.tangent_speed;}}}return best;
}
}
