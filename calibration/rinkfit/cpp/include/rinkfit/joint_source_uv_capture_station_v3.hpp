#pragma once
#include "rinkfit/joint_source_uv_capture_station_v2.hpp"
namespace rinkfit {
// A numerical active-set chart of the SAME qualified finite curve. A fixed entity
// is never retracted. No sensor/finite interval is enlarged, no residual omitted.
inline JointCaptureSolutionV2 joint_capture_solution_v3(const JointRowEvaluation&q,const double*const*p,double raw_station,double reference_station){
 JointCaptureSolutionV2 r;r.station=raw_station;std::array<double,3>x{};Eigen::Matrix3d J;bool ordinary=q.local_solution_uncached(p,raw_station,&x,&J);
 const auto strict_bracket=[&](double s){try{const auto lo=q.model->predict_explicit(q.layout,s,0.,p),hi=q.model->predict_explicit(q.layout,s,1079.,p);return lo.valid()&&hi.valid()&&-lo.uv[1]<=0.&&1079.-hi.uv[1]>=0.;}catch(...){return false;}};
 if(ordinary&&strict_bracket(raw_station)){r.valid=true;r.row=x[1];r.RS_denominator=J(0,0);r.reason="ordinary_true_RS_strict_sensor_bracket";return r;}
 r.reason="original_RS_root_failed";const auto&src=*q.observation;const auto&m=*q.model;const auto&l=q.layout;if(src.kind==1||!src.station_allowed(m,l.path,raw_station))return r;
 try{
  auto low=m.predict_explicit(l,raw_station,0.,p),high=m.predict_explicit(l,raw_station,1079.,p);if(!low.valid()||!high.valid())return r;double F0=-low.uv[1],FH=1079-high.uv[1];int active=F0<0&&FH<0?1:(F0>0&&FH>0?-1:0);if(!active){r.reason="no_sensor_same_sign_bracket";return r;}const double row=active>0?1079.:0.;
  std::array<double,2>interval{};bool span=false;for(auto v:src.station_intervals(m,l.path))if(raw_station>=v[0]&&raw_station<=v[1]){interval=v;span=true;break;}if(!span||interval[0]>=interval[1]){r.reason="no_finite_station_interval";return r;}
  auto f=[&](double s,double*value){if(!src.station_allowed(m,l.path,s))return false;const auto v=m.predict_explicit(l,s,row,p);if(!v.valid())return false;*value=row-v.uv[1];return std::isfinite(*value);};double base;if(!f(raw_station,&base))return r;double partner=0,fp=0,distance=INFINITY;auto consider=[&](double s){double v;if(s<interval[0]||s>interval[1]||!f(s,&v)||base*v>=0)return;if(std::abs(s-raw_station)<distance){partner=s;fp=v;distance=std::abs(s-raw_station);}};(void)reference_station;for(int k=0;k<=64;++k)consider(interval[0]+(interval[1]-interval[0])*k/64.);if(!std::isfinite(distance)){r.reason="no_same_finite_branch_sensor_crossing";return r;}
  // The margin moves the numerical station INSIDE the unchanged sensor domain;
  // it never relaxes F(0)<=0<=F(1079). Keep a verified feasible-side point
  // so independent double implementations do not disagree on a rounded zero.
  const double inward_margin=64*std::numeric_limits<double>::epsilon()*1079.;
  double a=raw_station,b=partner,fa=base,fb=fp;for(int it=0;it<80;++it){double mid=(a+b)/2;if(mid==a||mid==b)break;double fm;if(!f(mid,&fm)){r.reason="invalid_physical_continuation";return r;}if(fa*fm>0){a=mid;fa=fm;}else{b=mid;fb=fm;}}
  // B is always the verified feasible side, including a rounded exact zero.
  // Walk representable stations inward rather than picking a coarse bisection
  // sample: this keeps the numerical root smooth to the last few ULPs.
  double s=b,fs=fb;const double toward=partner>raw_station?INFINITY:-INFINITY;
  for(int it=0;it<4096&&std::abs(fs)<inward_margin;++it){double candidate=std::nextafter(s,toward);if(candidate==s||candidate<interval[0]||candidate>interval[1]||!f(candidate,&fs))break;s=candidate;}
  if(!strict_bracket(s)){r.reason="strict_sensor_bracket_not_closed";return r;}
  const double endpoint_tolerance=64*std::numeric_limits<double>::epsilon()*std::max(1.,interval[1]-interval[0]);if(s-interval[0]<=endpoint_tolerance||interval[1]-s<=endpoint_tolerance){r.reason="finite_endpoint_sensor_intersection_nondifferentiable";return r;}
  const auto&path=m.materials.at(l.material).paths.at(l.path);if((path.kind==2||path.kind==6||path.kind==7)&&path.points.size()>1){double z=s*(path.points.size()-1);if(std::abs(z-std::round(z))<=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(z))){r.reason="polyline_vertex_sensor_intersection_nondifferentiable";return r;}}
  using K=ceres::Jet<double,1>;std::vector<std::vector<K>>store;std::vector<const K*>kp;for(std::size_t k=0;k<l.block_ids.size();++k){store.emplace_back();for(std::size_t j=0;j<m.blocks.at(l.block_ids[k]).values.size();++j)store.back().emplace_back(p[k][j]);kp.push_back(store.back().data());}K st(s);st.v[0]=1;auto vs=m.predict_explicit(l,st,K(row),kp.data());K rw(row);rw.v[0]=1;auto vr=m.predict_explicit(l,K(s),rw,kp.data());const auto plain=m.predict_explicit(l,s,row,p);if(!vs.valid()||!vr.valid()||!plain.valid()||std::abs(row-plain.uv[1])>=1e-8||!std::isfinite(vs.uv[1].v[0])||std::abs(vs.uv[1].v[0])<1e-8||!std::isfinite(vr.uv[1].v[0])||std::abs(1-vr.uv[1].v[0])<1e-6){r.reason="sensor_station_IFT_singular_or_unclosed";return r;}
  r.valid=true;r.sensor_active=active;r.station=s;r.row=row;r.dv_ds=vs.uv[1].v[0];r.RS_denominator=1-vr.uv[1].v[0];r.reason="same_finite_curve_strict_sensor_feasible_station_IFT";return r;
 }catch(...){r.reason="sensor_station_retraction_arithmetic_rejected";return r;}
}
}
