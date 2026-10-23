#pragma once
#include "rinkfit/joint_explicit.hpp"
namespace rinkfit {
// Reuse the already-shadowed ambient GPU UV columns directly for scalar chains.
// Original CPU functors still supply both double and zero-Jet4 scalar arithmetic;
// unsupported layouts and independent/fresh CPU builders retain full Jet64 AD.
inline bool joint_uv_chain(const JointExplicitPixel&f,const JointUVCacheView&v,double const*const*,double*du,double*dv,int*extra,double*coefficient){const auto&o=*f.row.source;const double x=v.canonical_AD_uv[0]-o.uv[0],y=v.canonical_AD_uv[1]-o.uv[1],E2=o.E*o.E;if(o.query){const double z=o.normal[0]*x+o.normal[1]*y;*du=2*z*o.normal[0]/E2;*dv=2*z*o.normal[1]/E2;}else{*du=2*x/E2;*dv=2*y/E2;}*extra=f.budget_slot;*coefficient=-1;return std::isfinite(*du)&&std::isfinite(*dv);}
inline bool joint_uv_chain(const JointExplicitEquality&f,const JointUVCacheView&,double const*const*p,double*du,double*dv,int*extra,double*coefficient){if(f.kind<0){*du=0;*dv=-1;*extra=f.row.row_slot;*coefficient=1;return true;}using J=ceres::Jet<double,1>;J alpha(p[f.row.alpha_slot][0]);alpha.v[0]=1;Vec2<J>q;if(!f.row.source->query||!f.row.source->query->point(alpha,&q))return false;*du=f.kind==0?1:0;*dv=f.kind==1?1:0;*extra=f.row.alpha_slot;*coefficient=-q[f.kind].v[0];return finite_array(q)&&std::isfinite(*coefficient);}
template<class Functor>class JointUVChainCost final:public ceres::CostFunction,public PHRADValueProbe{
 public:
 JointUVChainCost(Functor f,const std::vector<int>&sizes):functor_(std::move(f)),fallback_(&functor_,ceres::DO_NOT_TAKE_OWNERSHIP){set_num_residuals(1);for(int n:sizes){if(n<1)throw std::invalid_argument("empty UV chain parameter");mutable_parameter_block_sizes()->push_back(n);fallback_.AddParameterBlock(n);}fallback_.SetNumResiduals(1);}
 bool EvaluateADValue(double const*const*p,double*r)const override{using J=ceres::Jet<double,4>;std::vector<std::vector<J>>store;std::vector<const J*>jp;store.reserve(parameter_block_sizes().size());jp.reserve(parameter_block_sizes().size());for(std::size_t k=0;k<parameter_block_sizes().size();++k){store.emplace_back(parameter_block_sizes()[k]);for(int j=0;j<parameter_block_sizes()[k];++j){store.back()[j].a=p[k][j];store.back()[j].v.setZero();}jp.push_back(store.back().data());}J result;if(!functor_(jp.data(),&result)||!finite_scalar(result)||!result.v.allFinite())return false;r[0]=result.a;return true;}
 bool Evaluate(double const*const*p,double*r,double**J)const override{bool requested=false;if(J)for(std::size_t i=0;i<parameter_block_sizes().size();++i)requested|=J[i]!=nullptr;if(!requested)return fallback_.Evaluate(p,r,J);const auto&row=functor_.row;JointUVCacheView view;if(!row.uv_cache||!row.uv_cache->lookup(row.cache_row,&view)||!view.jacobians_ready||!view.partials||!view.block_column_starts||row.physical_slots.size()!=row.layout.block_ids.size())return fallback_.Evaluate(p,r,J);if(!EvaluateADValue(p,r))return false;double du=0,dv=0,coefficient=0;int extra=-1;if(!joint_uv_chain(functor_,view,p,&du,&dv,&extra,&coefficient))return false;for(std::size_t k=0;k<parameter_block_sizes().size();++k)if(J[k])std::fill_n(J[k],parameter_block_sizes()[k],0.);auto add=[&](int slot,int component,int column){if(slot<0||!J[slot])return;const auto&d=view.partial(column);J[slot][component]+=du*d.u+dv*d.v;};for(std::size_t k=0;k<row.layout.block_ids.size();++k){int slot=row.physical_slots[k],start=view.block_column_starts->at(k);const auto n=row.model->blocks.at(row.layout.block_ids[k]).values.size();for(std::size_t j=0;j<n;++j)add(slot,int(j),start+int(j));}if(row.station_slot>=0)add(row.station_slot,0,view.station_column);add(row.row_slot,0,view.row_column);if(extra>=0&&J[extra])J[extra][0]+=coefficient;for(std::size_t k=0;k<parameter_block_sizes().size();++k)if(J[k])for(int j=0;j<parameter_block_sizes()[k];++j)if(!std::isfinite(J[k][j]))return false;row.uv_cache->note_direct_chain_hit();return true;}
 private:Functor functor_;ceres::DynamicAutoDiffCostFunction<Functor,64>fallback_;
};
template<class F>std::unique_ptr<ceres::CostFunction>joint_uv_scalar_cost(F f,const std::vector<int>&sizes){if(!f.row.uv_cache)return event_autodiff_cost(std::move(f),sizes,1);return std::make_unique<JointUVChainCost<F>>(std::move(f),sizes);}
// The engineering objective has two UV residuals. Consume the same shadowed
// ambient columns as the scalar chains instead of lifting them into Jet<64>.
struct JointEngineeringPixelView {const JointEngineeringPixel*pixel;template<class T>bool operator()(T const*const*p,T*r)const{return (*pixel)(p,r);}};
class JointEngineeringUVChainCost final:public ceres::CostFunction {
 public:
 const JointExplicitRow& native_domain_row()const{return functor_.row;}
 JointEngineeringUVChainCost(JointEngineeringPixel f,const std::vector<int>&sizes)
 :functor_(std::move(f)),fallback_(event_autodiff_cost(JointEngineeringPixelView{&functor_},sizes,functor_.row.source->query?1:2)){
  set_num_residuals(functor_.row.source->query?1:2);
  for(int n:sizes)mutable_parameter_block_sizes()->push_back(n);
 }
 bool Evaluate(double const*const*p,double*r,double**J)const override{
  bool requested=false;if(J)for(std::size_t k=0;k<parameter_block_sizes().size();++k)requested|=J[k]!=nullptr;
  if(!requested)return fallback_->Evaluate(p,r,J);
  const auto&row=functor_.row;JointUVCacheView view;
  if(!row.uv_cache||!row.uv_cache->lookup(row.cache_row,&view)||!view.jacobians_ready||!view.partials||!view.block_column_starts||row.physical_slots.size()!=row.layout.block_ids.size())return fallback_->Evaluate(p,r,J);
  const auto&o=*row.source;const double ex=view.canonical_AD_uv[0]-o.uv[0],ey=view.canonical_AD_uv[1]-o.uv[1];
  if(o.query)r[0]=functor_.scale*(o.normal[0]*ex+o.normal[1]*ey);
  else{r[0]=functor_.scale*ex;r[1]=functor_.scale*ey;}
  for(int d=0;d<num_residuals();++d)if(!std::isfinite(r[d]))return false;
  for(std::size_t k=0;k<parameter_block_sizes().size();++k)if(J[k])std::fill_n(J[k],num_residuals()*parameter_block_sizes()[k],0.);
  auto add=[&](int slot,int component,int column){if(slot<0||!J[slot])return;const auto&d=view.partial(column);const int n=parameter_block_sizes()[slot];
   if(o.query)J[slot][component]+=functor_.scale*(o.normal[0]*d.u+o.normal[1]*d.v);
   else{J[slot][component]+=functor_.scale*d.u;J[slot][n+component]+=functor_.scale*d.v;}
  };
  for(std::size_t k=0;k<row.layout.block_ids.size();++k){const int slot=row.physical_slots[k];if(!J[slot])continue;const int start=view.block_column_starts->at(k);for(int j=0;j<parameter_block_sizes()[slot];++j)add(slot,j,start+j);}
  if(row.station_slot>=0)add(row.station_slot,0,view.station_column);add(row.row_slot,0,view.row_column);
  for(std::size_t k=0;k<parameter_block_sizes().size();++k)if(J[k])for(int j=0;j<num_residuals()*parameter_block_sizes()[k];++j)if(!std::isfinite(J[k][j]))return false;
  row.uv_cache->note_direct_chain_hit();return true;
 }
 private:JointEngineeringPixel functor_;std::unique_ptr<ceres::CostFunction>fallback_;
};
inline std::unique_ptr<ceres::CostFunction>joint_engineering_uv_cost(JointEngineeringPixel f,const std::vector<int>&sizes,const JointExplicitRow**shared_row=nullptr){
 if(!f.row.uv_cache){auto cost=std::make_unique<EventADCost<JointEngineeringPixel>>(std::move(f),sizes,f.row.source->query?1:2);if(shared_row)*shared_row=&cost->functor_view().row;return cost;}
 auto cost=std::make_unique<JointEngineeringUVChainCost>(std::move(f),sizes);if(shared_row)*shared_row=&cost->native_domain_row();return cost;
}

}
