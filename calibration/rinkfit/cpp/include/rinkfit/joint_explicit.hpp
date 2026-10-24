#pragma once
#include "rinkfit/joint_builder.hpp"
#include "rinkfit/event_factors.hpp"
#include "rinkfit/joint_uv_cache.hpp"
namespace rinkfit {
// Exact AL chart: the same finite material, explicit sensor row and frozen query.
// Arithmetic outside a ray/no-fold inequality is retained for PHR restoration.
struct JointPhysicalEqualityTerm {int slot=0,index=0;double coefficient=0;};
struct JointPhysicalEquality {std::vector<JointPhysicalEqualityTerm>terms;double rhs=0,evaluation_scale=1;template<class T>bool operator()(T const*const*p,T*r)const{T v=-T(rhs);for(const auto&t:terms)v+=T(t.coefficient)*p[t.slot][t.index];r[0]=T(evaluation_scale)*v;return finite_scalar(r[0]);}};
struct JointZoomControlBound {int offset=0,upper=0;template<class T>bool operator()(T const*const*p,T*r)const{r[0]=upper?p[1][offset+2]-p[0][1]:p[0][0]-p[1][offset+2];return finite_scalar(r[0]);}};
struct JointZoomCertificate {
 double f_ref=1,radius=1;int endpoint=0;double wide_boundary=0;std::array<double,4>wide_coefficients{};
 template<class T>std::array<T,2> coefficient_interval(const T&base,const T&slope,double coefficient,const T&lo,const T&hi)const{
  auto value=[&](const T&x){const T hinge=x<T(wide_boundary)?T(wide_boundary)-x:T(0);return base+slope*(x-T(std::log(f_ref)))+T(coefficient)*hinge*hinge;};
  T low=value(lo),high=low;auto include=[&](const T&x){const T v=value(x);if(v<low)low=v;if(v>high)high=v;};include(hi);
  if(lo<T(wide_boundary)&&T(wide_boundary)<hi)include(T(wide_boundary));
  if(coefficient!=0){const T vertex=T(wide_boundary)-slope/T(2*coefficient);if(lo<vertex&&vertex<hi&&vertex<T(wide_boundary))include(vertex);}
  return {low,high};
 }
 template<class T>bool operator()(T const*const*p,T*r)const{
  const bool extension=std::any_of(wide_coefficients.begin(),wide_coefficients.end(),[](double x){return x!=0;});
  const T log_f=p[1][endpoint],z=log_f-T(std::log(f_ref));const T hinge=log_f<T(wide_boundary)?T(wide_boundary)-log_f:T(0),basis=hinge*hinge;
  Brown<T>b;b.radius_max=T(radius);b.k1=p[0][5]+p[0][6]*z+T(wide_coefficients[0])*basis;b.k2=p[0][7]+p[0][8]*z+T(wide_coefficients[1])*basis;b.p1=p[0][9]+T(wide_coefficients[2])*basis;b.p2=p[0][10]+T(wide_coefficients[3])*basis;
  // The old affine zoom has a concave no-fold lower bound in log-f, so its
  // endpoints suffice. A wide quadratic can attain an interior extremum.
  // Keep two original schema rows: the first encloses the complete interval;
  // the second retains its exact upper-end disk predicate.
  if(extension&&endpoint==0){
   const auto k1=coefficient_interval(p[0][5],p[0][6],wide_coefficients[0],p[1][0],p[1][1]);
   const auto k2=coefficient_interval(p[0][7],p[0][8],wide_coefficients[1],p[1][0],p[1][1]);
   const auto p1=coefficient_interval(p[0][9],T(0),wide_coefficients[2],p[1][0],p[1][1]);
   const auto p2=coefficient_interval(p[0][10],T(0),wide_coefficients[3],p[1][0],p[1][1]);
   using std::abs;using std::sqrt;b.k1=k1[0];b.k2=k2[0];b.p1=abs(p1[0])>abs(p1[1])?abs(p1[0]):abs(p1[1]);b.p2=abs(p2[0])>abs(p2[1])?abs(p2[0]):abs(p2[1]);
   const T pnorm2=b.p1*b.p1+b.p2*b.p2,pnorm=pnorm2>T(0)?sqrt(pnorm2):T(0),q=T(radius*radius);
   const T pad=T(1e-10)*(T(1)+abs(b.k1)*q+abs(b.k2)*q*q+T(6*radius)*pnorm);
   r[0]=T(1e-8)-brown_no_fold_lower_bound(b)+pad;
  }else r[0]=T(1e-8)-brown_no_fold_lower_bound(b);
  return finite_scalar(r[0]);
 }
};
struct JointExplicitRow {
 const JointEventModel* model=nullptr;JointRowLayout layout;const JointObservation* source=nullptr;
 int station_slot=-1,row_slot=-1,alpha_slot=-1;double fixed_row_value=0;
 const JointUVCache*uv_cache=nullptr;std::size_t cache_row=0;std::vector<int>physical_slots;
 JointExplicitRow remap(const std::vector<int>&map)const{auto q=*this;auto re=[&](int i){return i<0?i:map.at(i);};q.layout.scene_slot=re(layout.scene_slot);q.layout.rig_slot=re(layout.rig_slot);q.layout.optics_slot=re(layout.optics_slot);q.layout.reference_angles_slot=re(layout.reference_angles_slot);q.layout.geometry_slot=re(layout.geometry_slot);for(auto&i:q.layout.shape_vertex_slots)i=re(i);q.layout.advertisement_slot=re(layout.advertisement_slot);q.layout.width_slot=re(layout.width_slot);for(auto&i:q.layout.control_slots)i=re(i);q.station_slot=re(station_slot);q.row_slot=re(row_slot);q.alpha_slot=re(alpha_slot);for(auto&i:q.physical_slots)i=re(i);return q;}
 template<class T>T sensor_row(T const*const*p)const{return row_slot<0?T(fixed_row_value):p[row_slot][0];}
 template<class T>bool predict(T const*const*p,EventALProjection<T>*out)const{try{const T s=station_slot<0?T(source->station):p[station_slot][0];*out=model->predict_for_al(layout,s,sensor_row(p),p);return out->arithmetic_valid;}catch(...){return false;}}
 template<class T>bool cached_native_value(T const*const*p,int kind,T*value)const{if(kind<0||kind>4||!uv_cache)return false;JointUVCacheView cached;if(!uv_cache->lookup(cache_row,&cached)||!cached.native_domain_available[kind])return false;if constexpr(std::is_arithmetic<T>::value){*value=T(cached.canonical_native_domains[kind]);uv_cache->note_native_value_hit(false);return true;}else if constexpr(T::DIMENSION==4){if(physical_slots.size()!=layout.block_ids.size())return false;for(std::size_t k=0;k<layout.block_ids.size();++k)for(std::size_t j=0;j<model->blocks.at(layout.block_ids[k]).values.size();++j)if(!p[physical_slots[k]][j].v.isZero(0))return false;if(station_slot>=0&&!p[station_slot][0].v.isZero(0))return false;if(row_slot>=0&&!p[row_slot][0].v.isZero(0))return false;*value=T(cached.canonical_AD_native_domains[kind]);uv_cache->note_native_value_hit(true);return true;}return false;}
 template<class T>bool raw_uv(T const*const*p,Vec2<T>*uv)const{JointUVCacheView cached;if(uv_cache&&uv_cache->lookup(cache_row,&cached)){if constexpr(std::is_arithmetic<T>::value){*uv={T(cached.canonical_uv[0]),T(cached.canonical_uv[1])};return finite_array(*uv);}else if(cached.jacobians_ready&&cached.partials&&cached.block_column_starts&&physical_slots.size()==layout.block_ids.size()){*uv={T(cached.canonical_AD_uv[0]),T(cached.canonical_AD_uv[1])};auto lift=[&](const T&value,int column){const T delta=value-T(event_scalar(value));(*uv)[0]+=T(cached.partial(column).u)*delta;(*uv)[1]+=T(cached.partial(column).v)*delta;};for(std::size_t k=0;k<layout.block_ids.size();++k){const auto&block=model->blocks.at(layout.block_ids[k]);const int start=cached.block_column_starts->at(k);cached.for_each_block_partial(start,int(block.values.size()),[&](int j,const JointUVPartial&d){const T delta=p[physical_slots[k]][j]-T(event_scalar(p[physical_slots[k]][j]));(*uv)[0]+=T(d.u)*delta;(*uv)[1]+=T(d.v)*delta;});}if(station_slot>=0)lift(p[station_slot][0],cached.station_column);if(row_slot>=0)lift(p[row_slot][0],cached.row_column);return finite_array(*uv);}}EventALProjection<T>q;if(!predict(p,&q))return false;*uv=q.uv;return finite_array(*uv);}
 template<class T>bool native_error(T const*const*p,Vec2<T>*e)const{Vec2<T>uv;if(!raw_uv(p,&uv))return false;*e={uv[0]-T(source->uv[0]),uv[1]-T(source->uv[1])};return finite_array(*e);}
};
// Direct native pixel operator. E is an engineering scale, never a sigma.
struct JointEngineeringPixel {
 JointExplicitRow row;double scale=1;
 template<class T>bool operator()(T const*const*p,T*r)const{
  Vec2<T>e;if(!row.native_error(p,&e))return false;
  if(row.source->query){r[0]=T(scale)*(T(row.source->normal[0])*e[0]+T(row.source->normal[1])*e[1]);return finite_scalar(r[0]);}
  r[0]=T(scale)*e[0];r[1]=T(scale)*e[1];return finite_scalar(r[0])&&finite_scalar(r[1]);
 }
};
struct JointExplicitPixel {JointExplicitRow row;int budget_slot=-1;template<class T>bool operator()(T const*const*p,T*r)const{Vec2<T>e;if(!row.native_error(p,&e))return false;const auto&o=*row.source;T v=o.query?T(o.normal[0])*e[0]+T(o.normal[1])*e[1]:T(0);r[0]=(o.query?v*v:e[0]*e[0]+e[1]*e[1])/T(o.E*o.E)-p[budget_slot][0];return finite_scalar(r[0]);}};
struct JointExplicitEquality {JointExplicitRow row;int kind=-1;template<class T>bool operator()(T const*const*p,T*r)const{Vec2<T>uv;if(!row.raw_uv(p,&uv))return false;if(kind<0)r[0]=p[row.row_slot][0]-uv[1];else{Vec2<T>q;if(!row.source->query||!row.source->query->point(p[row.alpha_slot][0],&q))return false;r[0]=uv[kind]-q[kind];}return finite_scalar(r[0]);}};
struct JointExplicitDomain {JointExplicitRow row;int kind=0;
 std::optional<bool> cached_ad_value(double const*const*,double* residual)const{
  if(kind<0||kind>=5||!row.uv_cache)return std::nullopt;
  JointUVCacheView cached;
  if(!row.uv_cache->lookup(row.cache_row,&cached)||!cached.native_domain_available[kind])return std::nullopt;
  // Same branch as cached_native_value<Jet<double,4>> with every seed zero.
  residual[0]=cached.canonical_AD_native_domains[kind];row.uv_cache->note_native_value_hit(true);
  return std::isfinite(residual[0]);
 }
template<class T>static bool evaluate_kind(const JointExplicitRow&row,int kind,T const*const*p,T*r){if(kind<5&&row.cached_native_value(p,kind,r))return finite_scalar(r[0]);if(kind==7){const T s=row.station_slot<0?T(row.source->station):p[row.station_slot][0];r[0]=row.source->station_domain(*row.model,row.layout.path,s);return finite_scalar(r[0]);}if(kind>=5){const T s=row.station_slot<0?T(row.source->station):p[row.station_slot][0];r[0]=row.model->measurement_domain(row.layout,s,p,kind);return finite_scalar(r[0]);}if(kind>=3){const T s=row.station_slot<0?T(row.source->station):p[row.station_slot][0];const auto v=row.model->visibility(row.layout,s,row.sensor_row(p),p);if(!v.arithmetic_valid)return false;r[0]=kind==3?v.front:v.occlusion;return finite_scalar(r[0]);}EventALProjection<T>v;if(!row.predict(p,&v))return false;double R=row.model->shots.at(row.layout.shot_index).math.radius_max;r[0]=kind==0?v.ray_radius_sq/T(R*R)-T(1):(kind==1?-v.depth:-v.brown_lower);return finite_scalar(r[0]);}
 template<class T>bool operator()(T const*const*p,T*r)const{return evaluate_kind(row,kind,p,r);}
};
struct JointExplicitStatistic {const JointStatisticalGroup*group=nullptr;std::vector<JointExplicitRow>rows;template<class T>bool operator()(T const*const*p,T*r)const{std::vector<T>e(2*rows.size());for(std::size_t k=0;k<rows.size();++k){Vec2<T>v;if(!rows[k].native_error(p,&v))return false;e[2*k]=v[0];e[2*k+1]=v[1];}for(std::size_t k=0;k<group->rank;++k){T v=T(0);for(std::size_t j=0;j<e.size();++j)if(group->W[k*e.size()+j]!=0)v+=T(group->W[k*e.size()+j])*e[j];r[k]=v;if(!finite_scalar(v))return false;}return true;}};
}
