#pragma once
#include "rinkfit/joint_al.hpp"
#include <ceres/manifold.h>
namespace rinkfit {
// A complete source-admitted circle has no physical endpoint at s=0/1.
// Keep the saved coordinate in [0,1], but let an optimizer step cross that seam.
// Partial arcs and source-limited station intervals retain their original bounds.
class JointPeriodicStationManifold final : public ceres::Manifold {
 public:
 int AmbientSize() const override { return 1; }
 int TangentSize() const override { return 1; }
 bool Plus(const double* x,const double* d,double* y) const override {
  if(!std::isfinite(x[0])||!std::isfinite(d[0]))return false;
  if(d[0]==0.){y[0]=x[0];return true;}
  const double t=x[0]+d[0];if(!std::isfinite(t))return false;
  y[0]=t-std::floor(t);return true;
 }
 bool Minus(const double* y,const double* x,double* d) const override {
  if(!std::isfinite(y[0])||!std::isfinite(x[0]))return false;
  d[0]=std::remainder(y[0]-x[0],1.);return true;
 }
 bool PlusJacobian(const double*,double* j) const override {j[0]=1.;return true;}
 bool MinusJacobian(const double*,double* j) const override {j[0]=1.;return true;}
};
inline bool joint_full_circle_station(const JointEventModel& m,const JointObservation& r){
 if(r.kind==1)return false;const auto& p=m.materials.at(r.material).paths.at(r.path);
 const bool circle=p.kind==1&&p.closed&&std::abs(std::abs(p.end_angle-p.start_angle)-2*std::acos(-1.))<=16*std::numeric_limits<double>::epsilon();
 const bool printed=p.kind==2&&p.closed&&p.points.size()>2&&p.points.front()==p.points.back();
 const bool shared=p.kind==6&&p.closed&&p.vertex_blocks.empty();
 const bool board=p.kind==3&&p.boundary_segment==-2;
 if(p.physical_point||!(circle||printed||shared||board))return false;
 const auto intervals=r.station_intervals(m,r.path);return intervals.size()==1&&intervals[0][0]==0.&&intervals[0][1]==1.;
}
inline eventio::Json install_joint_periodic_stations(JointALBuilder& b,ceres::Problem& p){
 std::size_t count=0,seam=0;eventio::Json shots=eventio::Json::object();
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto& r=b.core.packet.rows[i];auto* s=&b.core.station[i];
  if(!r.active||!p.HasParameterBlock(s)||p.IsParameterBlockConstant(s)||!joint_full_circle_station(b.core.model,r))continue;
  if(p.GetManifold(s))throw std::runtime_error("periodic station cannot replace an existing source manifold");
  p.SetManifold(s,new JointPeriodicStationManifold);
  p.SetParameterLowerBound(s,0,-std::numeric_limits<double>::max());p.SetParameterUpperBound(s,0,std::numeric_limits<double>::max());
  ++count;if(*s==0.||*s==1.)++seam;shots[r.shot_id]=shots.value(r.shot_id,std::size_t(0))+1;
 }
 return {{"schema","rinkfit.complete_material_periodic_station.v2"},{"rows",count},{"initial_seam_rows",seam},{"shots",shots},{"original_source_interval_and_physical_shape_unchanged",true},{"physical_parameter_values_changed",false}};
}
}
