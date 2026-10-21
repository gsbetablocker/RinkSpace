#pragma once
// Independent probe operator. Does not alter or register production factors.
#include "rinkfit/joint_factors.hpp"
namespace rinkfit::sourcebranch {
struct Branch {
 double lower=0,upper=0,measured_alpha=0;
 std::string source_row_id;
};
struct Root {
 bool valid=false;
 std::array<double,3>x{};
 Eigen::Matrix3d J=Eigen::Matrix3d::Zero();
 Vec2<double>uv{};
 std::string status="not_evaluated";
};
class CurveOperator {
 public:
 JointObservation observation;
 JointRowEvaluation q;
 Branch branch;
 CurveOperator(const JointRowEvaluation&base,const Branch&b):observation(*base.observation),q(base),branch(b){
  if(!observation.query||observation.kind==1||b.source_row_id!=observation.source_row_id||!(b.lower<b.upper)||b.measured_alpha<b.lower||b.measured_alpha>b.upper)throw std::runtime_error("invalid source material branch");
  auto local=std::make_shared<JointQuery>(*observation.query);
  if(!local->den.empty()){
   // Rational curve coefficients preserve the original source chart.
   local->alpha={b.lower,b.upper};
  }else{
   Vec2<double>a,z;if(!observation.query->point(b.lower,&a)||!observation.query->point(b.upper,&z))throw std::runtime_error("source branch outside original Query");
   if(local->alpha.size()!=2)throw std::runtime_error("nonlinear piecewise Query branch must retain internal source knots");
   local->alpha={b.lower,b.upper};local->uv={a,z};
  }
  observation.query=local;q.observation=&observation;q.cache=std::make_shared<JointRowEvaluation::RootCache>();
 }
 CurveOperator(const CurveOperator&)=delete;
 CurveOperator&operator=(const CurveOperator&)=delete;
 Root resolve(const std::vector<const double*>&p,double station,double sensor_seed,double alpha_seed){
  Root out;observation.row=sensor_seed;observation.alpha=alpha_seed;
  if(q.layout.width_slot>=0&&(!std::isfinite(p[q.layout.width_slot][0])||p[q.layout.width_slot][0]<=0)){out.status="nonpositive_local_width_reject";return out;}
  if(!observation.station_allowed(*q.model,q.layout.path,station)){out.status="finite_source_station_reject";return out;}
  if(alpha_seed<branch.lower||alpha_seed>branch.upper){out.status="source_branch_seed_reject";return out;}
  if(!q.local_solution_uncached(p.data(),station,&out.x,&out.J)){out.status="local_station_row_alpha_root_reject";return out;}
  if(!out.J.allFinite()||out.J.fullPivLu().rank()!=3){out.status="local_root_rank_reject";return out;}
  if(out.x[2]<branch.lower||out.x[2]>branch.upper||out.x[1]<0||out.x[1]>1079){out.status="source_or_sensor_root_domain_reject";return out;}
  auto uv=q.model->predict_explicit(q.layout,out.x[0],out.x[1],p.data());
  if(!uv.valid()){out.status="projection_domain_reject";return out;}
  if(!q.model->visibility(q.layout,out.x[0],out.x[1],p.data()).visible()){out.status="physical_visibility_reject";return out;}
  for(int kind:{5,6}){double g=q.model->measurement_domain(q.layout,out.x[0],p.data(),kind);if(!std::isfinite(g)||g>1e-8){out.status="width_or_finite_rink_reject";return out;}}
  out.uv=uv.uv;out.valid=true;out.status="valid_local_source_branch_root";return out;
 }
 template<class T>bool native_error_at_root(T const*const*p,const Root&root,Vec2<T>*error)const{
  if(!root.valid)return false;
  auto uv=q.model->predict_explicit(q.layout,T(root.x[0]),T(root.x[1]),p);Vec2<T>srcq;
  if(!uv.valid()||!observation.query->point(T(root.x[2]),&srcq))return false;
  const std::array<T,3>F={T(root.x[1])-uv.uv[1],uv.uv[0]-srcq[0],uv.uv[1]-srcq[1]};
  for(const auto&f:F)if(!std::isfinite(event_scalar(f))||std::abs(event_scalar(f))>1e-8)return false; // A stale physical root is never an IFT certificate.
  const Eigen::Matrix3d inv=root.J.inverse();std::array<T,3>y={T(root.x[0]),T(root.x[1]),T(root.x[2])};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)y[i]-=T(inv(i,j))*F[j];
  const auto resolved=q.model->predict_explicit(q.layout,y[0],y[1],p);
  if(!resolved.valid())return false;
  *error={resolved.uv[0]-T(observation.uv[0]),resolved.uv[1]-T(observation.uv[1])};return finite_array(*error);
 }
};
} // namespace rinkfit::sourcebranch
