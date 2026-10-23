#pragma once
#include "rinkfit/joint_observed_curve_profile_v1.hpp"
namespace rinkfit {
// Cheap numerical reinitialization on the actual current finite print curves.
// Original source pixels, identities, weights and physical parameters stay intact.
inline JointObservedCurveFootV1 profile_joint_observed_print_v1(const JointEventModel&m,const JointPacket&p,std::size_t index,double observed_row){
 const auto&r=p.rows.at(index);JointObservedCurveFootV1 best;best.station=r.station;best.path=r.path;std::vector<int>paths=r.kind==1?std::vector<int>{r.path}:joint_source_allowed_finite_paths_v1(p,m,index);
 for(int path:paths){const auto l=m.layout(m.shot_map.at(r.shot_id),r.material,path,r.pts,r.width_block,r.width_sign);const auto ptr=m.pointers(l);// Source row time and physical model are unchanged while profiling s.
 // Cache their exact camera/optical calculation once per finite path.
 const auto g=m.global(l,ptr.data());std::vector<const double*>frozen;std::vector<double>geometry;
 const auto layout=m.legacy_layout(l,frozen,g,ptr.data(),geometry);const auto&math=m.shots.at(l.shot_index).math;
 auto tr=math.trajectory(row_time(l.frame_time,g[31],observed_row),frozen.data(),layout);
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
 if(shape.kind!=2||shape.points.size()<2)continue;
 std::vector<std::array<double,3>> seeds;const int segments=int(shape.points.size())-1;
 for(const auto& interval:r.station_intervals(m,path)){
  std::vector<double>breaks{interval[0],interval[1]};for(int k=1;k<segments;++k){const double s=double(k)/segments;if(s>interval[0]&&s<interval[1])breaks.push_back(s);}std::sort(breaks.begin(),breaks.end());
  for(std::size_t k=1;k<breaks.size();++k){const double a=breaks[k-1],b=breaks[k];const double fa=cost(a);const auto pa=prediction;record(a,fa);const double fb=cost(b);const auto pb=prediction;record(b,fb);if(!std::isfinite(fa)||!std::isfinite(fb))continue;const double dx=pb[0]-pa[0],dy=pb[1]-pa[1],den=dx*dx+dy*dy;const double u=den>0?std::clamp(((r.uv[0]-pa[0])*dx+(r.uv[1]-pa[1])*dy)/den,0.,1.):0.;const double s=a+(b-a)*u,f=cost(s);record(s,f);if(std::isfinite(f))seeds.push_back({f,a,b});}
 }
 const std::size_t n=std::min<std::size_t>(3,seeds.size());std::partial_sort(seeds.begin(),seeds.begin()+n,seeds.end());for(std::size_t k=0;k<n;++k){double lo=seeds[k][1],hi=seeds[k][2];for(int j=0;j<18;++j){const double u=lo+(hi-lo)*.3819660112501051,v=lo+(hi-lo)*.6180339887498949;const double fu=cost(u);record(u,fu);const double fv=cost(v);record(v,fv);if(fu<fv)hi=v;else lo=u;}consider((lo+hi)*.5);}

 }
 return best;
}
}
