#pragma once
#include "rinkfit/joint_explicit.hpp"
#include "rinkfit/joint_source_uv_capture_station_v3.hpp"
namespace rinkfit {
// Coarse source-UV objective only. The immutable packet Query is never changed.
// A row root is eliminated; a curve station remains a bounded finite-entity variable.
struct JointCaptureStationCellV20 {bool enabled=false,kkt=false,left_exists=false,right_exists=false;int cell=-1,vertex=-1;unsigned char left_status=0,right_status=0,active_kind=0,sensor_active=0;double lower=0,upper=1,numeric_lower=0,numeric_upper=1,anchor=0,left_gradient=0,right_gradient=0;};
struct JointSourceUVCaptureRowV1 {
 JointCaptureStationCellV20 numerical_cell;
 JointRowEvaluation implicit; std::shared_ptr<const JointObservation> source;
 bool feasible_sensor_station=false;
 const JointUVCache* cache=nullptr; std::size_t cache_row=0;
 template<class T>bool nuisance_unchecked(T const*const*p,std::array<T,3>*out)const{
  JointUVCacheView v;
  if(cache&&cache->lookup(cache_row,&v)){
   // An explicit-row cache is a different function and cannot qualify this factor.
   if(!v.implicit_RS_capture || !v.implicit_RS_valid)return false;
   T station=implicit.station_slot<0?T(source->station):p[implicit.station_slot][0];
   if(v.sensor_station_active){if(!feasible_sensor_station||source->kind==1||implicit.station_slot<0)return false;station=T(v.implicit_station);if constexpr(!std::is_arithmetic<T>::value){if(!v.jacobians_ready||!v.partials||!v.block_column_starts)return false;const double ds=v.partials[v.station_column].v;if(!std::isfinite(ds)||std::abs(ds)<1e-8||std::abs(ds-v.sensor_station_dv_ds)>1e-8*std::max(1.,std::abs(ds)))return false;T dv=T(0);for(std::size_t k=0;k<implicit.layout.block_ids.size();++k){const int start=v.block_column_starts->at(k);for(std::size_t j=0;j<implicit.model->blocks.at(implicit.layout.block_ids[k]).values.size();++j)dv+=T(v.partials[start+int(j)].v)*(p[k][j]-T(event_scalar(p[k][j])));}station-=dv/T(ds);}if(!source->station_allowed(*implicit.model,implicit.layout.path,event_scalar(station)))return false;*out={station,T(v.implicit_RS_row),T(source->alpha)};return finite_array(*out); }
   if(!source->station_allowed(*implicit.model,implicit.layout.path,event_scalar(station)))return false;
   if constexpr(std::is_arithmetic<T>::value){*out={station,T(v.implicit_RS_row),T(source->alpha)};return true;}
   else if(v.jacobians_ready&&v.partials&&v.block_column_starts){
    const auto&dr=v.partials[v.row_column];const double den=1-dr.v;
    if(!std::isfinite(den)||std::abs(den)<1e-6||std::abs(den-v.implicit_RS_denominator)>1e-8*std::max(1.,std::abs(den)))return false;
    T rhs=T(0);for(std::size_t k=0;k<implicit.layout.block_ids.size();++k){const int start=v.block_column_starts->at(k);const auto&b=implicit.model->blocks.at(implicit.layout.block_ids[k]);for(std::size_t j=0;j<b.values.size();++j)rhs+=T(v.partials[start+int(j)].v)*(p[k][j]-T(event_scalar(p[k][j])));}
    if(implicit.station_slot>=0)rhs+=T(v.partials[v.station_column].v)*(station-T(event_scalar(station)));
    *out={station,T(v.implicit_RS_row)+rhs/T(den),T(source->alpha)};return finite_array(*out);
   }
  }
  if(feasible_sensor_station){std::vector<std::vector<double>>storage;std::vector<const double*>dp;for(const auto&item:implicit.parameter_sizes){if(storage.size()<=std::size_t(item.first))storage.resize(item.first+1);storage[item.first].resize(item.second);for(int j=0;j<item.second;++j)storage[item.first][j]=event_scalar(p[item.first][j]);}for(const auto&v:storage)dp.push_back(v.data());const double raw=implicit.station_slot<0?source->station:event_scalar(p[implicit.station_slot][0]);auto sol=joint_capture_solution_v3(implicit,dp.data(),raw,source->station);if(!sol.valid)return false;if(sol.sensor_active){T s=T(sol.station);if constexpr(!std::is_arithmetic<T>::value){const auto v=implicit.model->predict_explicit(implicit.layout,T(sol.station),T(sol.row),p);if(!v.valid())return false;s-=(v.uv[1]-T(event_scalar(v.uv[1])))/T(sol.dv_ds);}*out={s,T(sol.row),T(source->alpha)};return finite_array(*out);}}
  return implicit.nuisance(p,out);
 }
 template<class T>bool nuisance(T const*const*p,std::array<T,3>*out)const{if(!nuisance_unchecked(p,out))return false;if(numerical_cell.enabled){const double s=event_scalar((*out)[0]);if(s<numerical_cell.lower||s>numerical_cell.upper)return false;if(numerical_cell.kkt&&s!=numerical_cell.anchor)return false;}return true;}
 template<class T>bool raw_uv(T const*const*p,Vec2<T>*out)const{
  std::array<T,3>y;if(!nuisance(p,&y))return false;JointUVCacheView v;
  if(cache&&cache->lookup(cache_row,&v)){
   if(!v.implicit_RS_capture||!v.implicit_RS_valid)return false;
   if constexpr(std::is_arithmetic<T>::value){*out={T(v.canonical_uv[0]),T(v.canonical_uv[1])};return finite_array(*out);}
   else if(v.jacobians_ready&&v.partials&&v.block_column_starts){
    *out={T(v.canonical_AD_uv[0]),T(v.canonical_AD_uv[1])};
    auto lift=[&](const T&x,int column){const T d=x-T(event_scalar(x));(*out)[0]+=T(v.partials[column].u)*d;(*out)[1]+=T(v.partials[column].v)*d;};
    for(std::size_t k=0;k<implicit.layout.block_ids.size();++k){const auto&b=implicit.model->blocks.at(implicit.layout.block_ids[k]);for(std::size_t j=0;j<b.values.size();++j)lift(p[k][j],v.block_column_starts->at(k)+int(j));}
    if(implicit.station_slot>=0)lift(y[0],v.station_column);lift(y[1],v.row_column);cache->note_direct_chain_hit();return finite_array(*out);
   }
  }
  try{const auto q=implicit.model->predict_explicit(implicit.layout,y[0],y[1],p);if(!q.valid())return false;*out=q.uv;return finite_array(*out);}catch(...){return false;}
 }
};
struct JointSourceUVCapturePixelV1 {JointSourceUVCaptureRowV1 row;double scale=1;template<class T>bool operator()(T const*const*p,T*r)const{Vec2<T>v;if(!row.raw_uv(p,&v))return false;r[0]=T(scale)*(v[0]-T(row.source->uv[0]));r[1]=T(scale)*(v[1]-T(row.source->uv[1]));return finite_scalar(r[0])&&finite_scalar(r[1]);}};
struct JointSourceUVCaptureDomainV1 {JointSourceUVCaptureRowV1 row;int kind=0;template<class T>bool operator()(T const*const*p,T*r)const{std::array<T,3>y;if(!row.nuisance(p,&y))return false;const auto&m=*row.implicit.model;const auto&l=row.implicit.layout;try{if(kind==7)r[0]=row.source->station_domain(m,l.path,y[0]);else if(kind>=5)r[0]=m.measurement_domain(l,y[0],p,kind);else if(kind>=3){const auto v=m.visibility(l,y[0],y[1],p);if(!v.arithmetic_valid)return false;r[0]=kind==3?v.front:v.occlusion;}else{const auto v=m.predict_for_al(l,y[0],y[1],p);if(!v.arithmetic_valid)return false;const double R=m.shots.at(l.shot_index).math.radius_max;r[0]=kind==0?v.ray_radius_sq/T(R*R)-T(1):(kind==1?-v.depth:-v.brown_lower);}return finite_scalar(r[0]);}catch(...){return false;}}};
}
