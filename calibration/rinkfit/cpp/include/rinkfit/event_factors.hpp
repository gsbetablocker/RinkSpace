#pragma once
#include <optional>
#include "rinkfit/event_model.hpp"
#include "rinkfit/event_packet.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <memory>

namespace rinkfit {

// A genuine frozen query has its original alpha coordinate. In particular,
// alpha is NOT assumed to be a native pixel and Q(alpha) need not be a line.
struct EventQueryCurve {
  std::string provenance;
  std::vector<double> alpha;
  std::vector<Vec2<double>> native_uv;
  // Optional exact rational polynomial of z=(alpha-alpha_origin)/alpha_scale.
  // Coefficients are ascending powers, all derived from frozen source camera
  // and original physical section, never the currently optimized camera.
  std::vector<double> numerator_x,numerator_y,denominator;
  double alpha_origin=0.,alpha_scale=1.,chord_error_bound_px=0.;
  template<class T> static T polynomial(const std::vector<double>& a,const T& x){T y=T(0);for(auto i=a.rbegin();i!=a.rend();++i)y=y*x+T(*i);return y;}
  template<class T> bool point(const T& a,Vec2<T>* uv) const {
    if(alpha.size()<2||native_uv.size()!=alpha.size()||event_scalar(a)<alpha.front()||event_scalar(a)>alpha.back())return false;
    if(!denominator.empty()){
      if(numerator_x.empty()||numerator_y.empty()||!(alpha_scale>0))return false;
      const T z=(a-T(alpha_origin))/T(alpha_scale),d=polynomial(denominator,z);
      if(std::abs(event_scalar(d))<1e-14)return false;
      *uv={polynomial(numerator_x,z)/d,polynomial(numerator_y,z)/d};return finite_array(*uv);
    }
    auto it=std::upper_bound(alpha.begin(),alpha.end(),event_scalar(a));const std::size_t i=it==alpha.end()?alpha.size()-2:std::size_t(it-alpha.begin()-1);
    if(!(alpha[i+1]>alpha[i]))return false;
    const T u=(a-T(alpha[i]))/T(alpha[i+1]-alpha[i]);
    *uv={T(native_uv[i][0])*(T(1)-u)+T(native_uv[i+1][0])*u,T(native_uv[i][1])*(T(1)-u)+T(native_uv[i+1][1])*u};return finite_array(*uv);
  }
};

// Slots refer to THIS functor's supplied parameter array. A group functor can
// use remap() to embed several row dependency lists in one exact sparse union.
struct EventRowEvaluation {
  const EventModel* model=nullptr;
  EventRowLayout layout;
  Vec2<double> source_uv{},source_normal{};
  double engineering_budget_px=2.,fixed_station=0.;
  bool genuine_profile=false;
  int station_slot=-1,row_slot=-1,alpha_slot=-1;
  std::shared_ptr<const EventQueryCurve> query;
  EventRowEvaluation remap(const std::vector<int>& map) const {
    EventRowEvaluation e=*this;auto re=[&](int slot){return slot<0?slot:map.at(std::size_t(slot));};
    e.layout.global_slot=re(layout.global_slot);e.layout.geometry_slot=re(layout.geometry_slot);e.layout.advertisement_slot=re(layout.advertisement_slot);
    for(auto& slot:e.layout.control_slots)slot=re(slot);
    e.station_slot=re(station_slot);e.row_slot=re(row_slot);e.alpha_slot=re(alpha_slot);return e;
  }
  template<class T> bool predict(T const*const* parameters,Projection<T>* prediction) const {
    if(model==nullptr||row_slot<0)return false;
    try {const T s=station_slot<0?T(fixed_station):parameters[station_slot][0];
      *prediction=model->predict_explicit(layout,s,parameters[row_slot][0],parameters);return prediction->valid();}
    catch(const std::exception&){return false;}
  }
  template<class T> bool native_error(T const*const* parameters,Vec2<T>* error) const {
    EventALProjection<T> prediction;if(!predict_for_al(parameters,&prediction))return false;
    *error={prediction.uv[0]-T(source_uv[0]),prediction.uv[1]-T(source_uv[1])};return true;
  }
  template<class T> bool predict_for_al(T const*const* parameters,EventALProjection<T>* prediction) const {
    if(model==nullptr||row_slot<0)return false;
    try {const T s=station_slot<0?T(fixed_station):parameters[station_slot][0];
      *prediction=model->predict_for_al(layout,s,parameters[row_slot][0],parameters);return prediction->arithmetic_valid;}
    catch(const std::exception&){return false;}
  }
  template<class T> T q(const Vec2<T>& e) const {
    const T budget=T(engineering_budget_px);
    if(genuine_profile){const T en=T(source_normal[0])*e[0]+T(source_normal[1])*e[1];return en*en/(budget*budget);}
    return (e[0]*e[0]+e[1]*e[1])/(budget*budget);
  }
};

// All constraint costs below return UNWEIGHTED g<=0 or h=0. Wrap using
// add_phr_inequality/equality with the frozen outer lambda/nu/rho. No robust
// loss, no hidden zero residual on failed RS/material/arithmetic domains.
struct EventPixelConstraint {
  EventRowEvaluation row;
  int budget_slot=-1;
  template<class T>bool operator()(T const*const* p,T* residual)const{
    Vec2<T> error;if(budget_slot<0||!row.native_error(p,&error))return false;
    residual[0]=row.q(error)-p[budget_slot][0];return finite_scalar(residual[0]);
  }
};
struct EventRSEquality {
  EventRowEvaluation row;
  template<class T>bool operator()(T const*const* p,T* residual)const{
    EventALProjection<T> prediction;if(!row.predict_for_al(p,&prediction))return false;
    residual[0]=p[row.row_slot][0]-prediction.uv[1];return finite_scalar(residual[0]);
  }
};
struct EventQueryEquality {
  EventRowEvaluation row;
  int component=0;
  template<class T>bool operator()(T const*const* p,T* residual)const{
    if(!row.genuine_profile||row.alpha_slot<0||!row.query||component<0||component>1)return false;
    EventALProjection<T> prediction;Vec2<T> query;if(!row.predict_for_al(p,&prediction)||!row.query->point(p[row.alpha_slot][0],&query))return false;
    residual[0]=prediction.uv[component]-query[component];return finite_scalar(residual[0]);
  }
};
// The SAME frozen physical domain as strict project(): R is immutable, no
// source or engineering budget enters these inequalities. AL uses closure;
// strict positive depth/no-fold remain required by native acceptance.
struct EventRowDomainInequality {
  EventRowEvaluation row;
  int kind=0; // 0: r^2/R^2-1; 1: -depth in L=1; 2: -Brown disk lower bound
  double scale=1.; // frozen positive numerical scale, stored in state schema
  template<class T>bool operator()(T const*const* p,T* residual)const{
    if(!(scale>0)||kind<0||kind>2)return false;
    EventALProjection<T> prediction;if(!row.predict_for_al(p,&prediction))return false;
    residual[0]=(kind==0?prediction.ray_radius_sq/T(row.model->radius_max*row.model->radius_max)-T(1):
      (kind==1?-prediction.depth:-prediction.brown_lower))/T(scale);
    return finite_scalar(residual[0]);
  }
};
struct EventPhysicalInequality {
  const EventModel* model=nullptr;
  EventPhysicalConstraint constraint;
  template<class T>bool operator()(T const*const* p,T* residual)const{
    try{residual[0]=model->evaluate_physical(constraint,p);return finite_scalar(residual[0]);}catch(const std::exception&){return false;}
  }
};

// Each verified run uses one explicit direction/seam candidate, saved in the
// snapshot. Board stations on distinct but adjacent allowed parts are mapped
// to the SAME continuous perimeter coordinate before testing order.
struct EventRunOrderConstraint {
  EventPath left_path,right_path;
  int global_slot=0,left_slot=1,right_slot=2,direction=1;
  double left_unwrap=0.,right_unwrap=0.,upper_span=0.;
  bool span_limit=false;
  template<class T>static T coordinate(const EventPath& path,const T& s,const T* g){
    if(path.kind!=3||path.boundary_segment<0)return s;
    const T arc=EventModel::quarter_arclength(T(1.5707963267948966),g),ll=T(1)-T(2)*g[1],sl=g[0]-T(2)*g[2];
    const T lengths[8]={ll,arc,sl,arc,ll,arc,sl,arc};T x=s*lengths[path.boundary_segment];for(int j=0;j<path.boundary_segment;++j)x+=lengths[j];
    return x/(T(2)*(ll+sl)+T(4)*arc);
  }
  template<class T>bool operator()(T const*const* p,T* residual)const{
    const T a=coordinate(left_path,p[left_slot][0],p[global_slot])+T(left_unwrap),b=coordinate(right_path,p[right_slot][0],p[global_slot])+T(right_unwrap);
    residual[0]=span_limit?T(direction)*(b-a)-T(upper_span):T(direction)*(a-b);return finite_scalar(residual[0]);
  }
};

// Frozen W has exactly the original row order and rank. Every source appears
// in one original group. Profile equality/normal engineering coverage does
// not replace, rotate, diagonalize or re-estimate this raw pixel statistic.
struct EventGroupedStatistic {
  const EventStatisticalGroup* group=nullptr;
  std::vector<EventRowEvaluation> rows;
  template<class T>bool operator()(T const*const* p,T* residual)const{
    if(group==nullptr||rows.size()!=group->rows.size()||group->W.size()!=group->rank*2*rows.size())return false;
    std::vector<T> native(2*rows.size());for(std::size_t i=0;i<rows.size();++i){Vec2<T> e;if(!rows[i].native_error(p,&e))return false;native[2*i]=e[0];native[2*i+1]=e[1];}
    for(std::size_t i=0;i<group->rank;++i){T sum=T(0);const std::size_t begin=i*native.size();for(std::size_t j=0;j<native.size();++j)if(group->W[begin+j]!=0.)sum+=T(group->W[begin+j])*native[j];residual[i]=sum;if(!finite_scalar(sum))return false;}return true;
  }
};

// Native domain probes can already have the exact zero-tangent AD scalar.
// Return no value when the ordinary Jet probe is still required.
template<class Functor>auto event_cached_ad_value(const Functor& f,double const*const* p,double* r,int)
 ->decltype(f.cached_ad_value(p,r)){return f.cached_ad_value(p,r);}
template<class Functor>std::optional<bool> event_cached_ad_value(const Functor&,double const*const*,double*,long){return std::nullopt;}

template<class Functor>class EventADCost final : public ceres::CostFunction, public PHRADValueProbe {
 public:
  EventADCost(Functor functor,const std::vector<int>& sizes,int residuals)
    :functor_(new Functor(std::move(functor))),cost_(functor_.get(),ceres::DO_NOT_TAKE_OWNERSHIP){
    if(residuals<1)throw std::invalid_argument("event cost needs at least one residual");
    for(int size:sizes){if(size<1)throw std::invalid_argument("empty event parameter block");mutable_parameter_block_sizes()->push_back(size);cost_.AddParameterBlock(size);}set_num_residuals(residuals);cost_.SetNumResiduals(residuals);
  }
  const Functor& functor_view()const{return *functor_;}
  bool Evaluate(double const*const* p,double* residuals,double** jacobians)const override{return cost_.Evaluate(p,residuals,jacobians);}
  bool EvaluateADValue(double const*const* p,double* residuals)const override{
    if(const auto cached=event_cached_ad_value(*functor_,p,residuals,0);cached.has_value())return *cached;
    using Jet=ceres::Jet<double,4>;std::vector<std::vector<Jet>> storage;std::vector<const Jet*> ptr;
    storage.reserve(parameter_block_sizes().size());ptr.reserve(parameter_block_sizes().size());
    for(std::size_t b=0;b<parameter_block_sizes().size();++b){storage.emplace_back(parameter_block_sizes()[b]);for(int j=0;j<parameter_block_sizes()[b];++j){storage.back()[j].a=p[b][j];storage.back()[j].v.setZero();}ptr.push_back(storage.back().data());}
    std::vector<Jet> output(num_residuals());if(!(*functor_)(ptr.data(),output.data()))return false;
    for(int r=0;r<num_residuals();++r){if(!std::isfinite(output[r].a)||!output[r].v.allFinite())return false;residuals[r]=output[r].a;}return true;
  }
 private:
  std::unique_ptr<Functor> functor_;
  ceres::DynamicAutoDiffCostFunction<Functor,64> cost_;
};
template<class Functor>std::unique_ptr<ceres::CostFunction> event_autodiff_cost(Functor functor,const std::vector<int>& sizes,int residuals=1){
  return std::make_unique<EventADCost<Functor>>(std::move(functor),sizes,residuals);
}
} // namespace rinkfit
