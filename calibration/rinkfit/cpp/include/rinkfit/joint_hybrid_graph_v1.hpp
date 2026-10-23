#pragma once
// Existing explicit factors are retained. Only legal local RS/Query auxiliaries
// are eliminated, by the same 3x3 IFT; all original W cross terms stay together.
#include "rinkfit/joint_hybrid_formal_v1.hpp"
namespace rinkfit {
inline ceres::EvaluationCallback*joint_hybrid_raw_callback_v1(JointALBuilder&base){
#ifdef RINKFIT_JOINT_FORMAL_HYBRID_HOOK_V1
 return base.native_gpu_evaluation_callback_v1();
#else
 return base.gpu_evaluation_callback();
#endif
}
inline const ConstraintSchema&joint_hybrid_original_schema_v1(JointALBuilder&base){
#ifdef RINKFIT_JOINT_FORMAL_HYBRID_HOOK_V1
 return base.native_schema_v1();
#else
 return base.schema();
#endif
}

struct JointHybridDerivedSlotV1 {std::size_t raw=0;int coordinate=0;};
class JointHybridImplicitCostV1 final:public ceres::CostFunction {
 const ceres::CostFunction*source_;std::shared_ptr<const ceres::CostFunction>owner_;JointHybridLocalStateV1*state_;const JointUVCache*gpu_;bool guard_;
 struct Slot{int target=-1;std::size_t raw=0;int coordinate=-1;};std::vector<Slot>slots_;std::vector<double*>parameters_;std::map<std::size_t,JointRowEvaluation>local_;
 public:
 JointHybridImplicitCostV1(const ceres::CostFunction*source,const std::vector<double*>&original,JointHybridLocalStateV1&state,std::shared_ptr<const ceres::CostFunction>owner={},const JointUVCache*gpu=nullptr,bool whole_trial_guard=true):source_(source),owner_(std::move(owner)),state_(&state),gpu_(gpu),guard_(whole_trial_guard){
  if(!source||original.size()!=source->parameter_block_sizes().size())throw std::runtime_error("hybrid implicit original cost inventory mismatch");set_num_residuals(source->num_residuals());auto locate=[&](double*p,JointHybridDerivedSlotV1*out){for(int coordinate:{0,1,2}){auto&values=coordinate==0?state.core.station:(coordinate==1?state.core.row:state.core.alpha);auto start=reinterpret_cast<std::uintptr_t>(values.data()),address=reinterpret_cast<std::uintptr_t>(p);if(address<start||address>=start+values.size()*sizeof(double)||(address-start)%sizeof(double))continue;auto raw=std::size_t((address-start)/sizeof(double));const auto&o=state.core.packet.rows[raw];if(o.active&&(coordinate==1||o.query)){*out={raw,coordinate};return true;}}return false;};
  auto add=[&](double*p,int size){auto it=std::find(parameters_.begin(),parameters_.end(),p);if(it!=parameters_.end()){int k=int(it-parameters_.begin());if(parameter_block_sizes()[k]!=size)throw std::runtime_error("hybrid target block size mismatch");return k;}int k=int(parameters_.size());parameters_.push_back(p);mutable_parameter_block_sizes()->push_back(size);return k;};
  for(std::size_t k=0;k<original.size();++k){JointHybridDerivedSlotV1 d;if(!locate(original[k],&d))slots_.push_back({add(original[k],source->parameter_block_sizes()[k]),0,-1});else{if(source->parameter_block_sizes()[k]!=1)throw std::runtime_error("hybrid derived auxiliary must be scalar");slots_.push_back({-1,d.raw,d.coordinate});local_.emplace(d.raw,state.rows.at(d.raw));}}
  for(auto&entry:local_){auto raw=entry.first;auto old=state.mutable_pointers(raw);std::vector<int>map;for(std::size_t j=0;j<old.size();++j)map.push_back(add(old[j],state.rows[raw].parameter_sizes.at(int(j))));entry.second=entry.second.remap(map);}
 }
 const std::vector<double*>&parameters()const{return parameters_;}
 bool Evaluate(double const*const*p,double*r,double**J)const override{
  if(guard_&&!state_->status->valid.load())return false;std::map<std::size_t,std::array<double,3>>x;std::map<std::size_t,Eigen::Matrix3d>root_J;
  for(const auto&entry:local_){const auto&q=entry.second;const double station=q.station_slot<0?q.observation->station:p[q.station_slot][0];std::array<double,3>z;Eigen::Matrix3d h;if(!q.local_solution(p,station,&z,&h))return false;x[entry.first]=z;root_J[entry.first]=h;}
  std::vector<const double*>old;for(const auto&s:slots_)old.push_back(s.coordinate<0?p[s.target]:&x.at(s.raw)[s.coordinate]);bool derivatives=false;if(J)for(std::size_t k=0;k<parameters_.size();++k)derivatives|=J[k]!=nullptr;if(!derivatives)return source_->Evaluate(old.data(),r,nullptr);
  const int m=num_residuals();std::vector<std::vector<double>>old_storage(slots_.size());std::vector<double*>old_J(slots_.size());for(std::size_t k=0;k<slots_.size();++k){old_storage[k].resize(std::size_t(m)*source_->parameter_block_sizes()[k]);old_J[k]=old_storage[k].data();}if(!source_->Evaluate(old.data(),r,old_J.data()))return false;
  for(std::size_t k=0;k<parameters_.size();++k)if(J[k])std::fill_n(J[k],std::size_t(m)*parameter_block_sizes()[k],0.);
  for(std::size_t k=0;k<slots_.size();++k)if(slots_[k].coordinate<0){int dst=slots_[k].target;if(J[dst])for(std::size_t j=0;j<old_storage[k].size();++j)J[dst][j]+=old_storage[k][j];}
  // Use qualified resident raw-UV partials when present. The CPU fallback is
  // the original predict_explicit Jet path, evaluated at the exact local root.
  // Both feed the same J_h inverse, not an independent row-wise normal equation.
  for(const auto&entry:local_){const auto raw=entry.first;const auto&q=entry.second;const auto&z=x.at(raw);const auto&h=root_J.at(raw);JointUVCacheView view;bool cached=gpu_&&gpu_->lookup(raw,&view)&&view.jacobians_ready&&view.partials&&view.block_column_starts&&!view.implicit_RS_capture;
   if(cached){for(std::size_t k=0;k<parameters_.size();++k)for(int j=0;j<parameter_block_sizes()[k];++j)if(p[k][j]!=parameters_[k][j])cached=false;cached&=std::abs(z[0]-(q.observation->kind==1?q.observation->station:state_->core.station[raw]))<=1e-12&&std::abs(z[1]-state_->core.row[raw])<=1e-7;}
   for(std::size_t block=0;block<parameters_.size();++block){if(!J[block])continue;const int width=parameter_block_sizes()[block];for(int offset=0;offset<width;offset+=4){using T=ceres::Jet<double,4>;std::vector<std::vector<T>>store(parameters_.size());std::vector<const T*>pp(parameters_.size());for(std::size_t k=0;k<parameters_.size();++k){store[k].resize(parameter_block_sizes()[k]);for(int j=0;j<parameter_block_sizes()[k];++j)store[k][j]=T(p[k][j]);pp[k]=store[k].data();}for(int c=0;c<4&&offset+c<width;++c)store[block][offset+c].v[c]=1.;Vec2<T>uv;
     if(!cached){auto projection=q.model->predict_explicit(q.layout,T(z[0]),T(z[1]),pp.data());if(!projection.valid())return false;uv=projection.uv;}
     for(int c=0;c<4&&offset+c<width;++c){double du=0,dv=0;if(cached){for(std::size_t physical=0;physical<q.layout.block_ids.size();++physical){int slot=-1;const auto*ptr=state_->core.model.blocks.at(q.layout.block_ids[physical]).values.data();auto it=std::find(parameters_.begin(),parameters_.end(),ptr);if(it!=parameters_.end())slot=int(it-parameters_.begin());if(slot==int(block)){int column=view.block_column_starts->at(physical)+offset+c;du+=view.partial(column).u;dv+=view.partial(column).v;}}}else{du=uv[0].v[c];dv=uv[1].v[c];}
      Eigen::Vector3d dz=Eigen::Vector3d::Zero();if(q.observation->query){Eigen::Vector3d f;f<<-dv,du,dv;dz=-h.inverse()*f;}else{const double ds=q.station_slot==int(block)&&offset+c==0?1.:0.;if(ds){if(cached){du+=view.partial(view.station_column).u;dv+=view.partial(view.station_column).v;}else{using K=ceres::Jet<double,1>;std::vector<std::vector<K>>kk(parameters_.size());std::vector<const K*>kp(parameters_.size());for(std::size_t k=0;k<parameters_.size();++k){for(int j=0;j<parameter_block_sizes()[k];++j)kk[k].push_back(K(p[k][j]));kp[k]=kk[k].data();}K s(z[0]);s.v[0]=1;auto pv=q.model->predict_explicit(q.layout,s,K(z[1]),kp.data());if(!pv.valid())return false;dv+=pv.uv[1].v[0];}}dz[0]=ds;dz[1]=dv/h(0,0);}
      for(std::size_t original=0;original<slots_.size();++original)if(slots_[original].coordinate>=0&&slots_[original].raw==raw){const double a=dz[slots_[original].coordinate];for(int row=0;row<m;++row)J[block][row*width+offset+c]+=old_storage[original][row]*a;}
     }
    }}
  }return true;
 }
};

inline JointHybridDataCostV1 make_joint_hybrid_explicit_data_cost_v1(const JointHybridUnitV1&unit,JointHybridLocalStateV1&state,const JointUVCache*gpu=nullptr,bool guard=true){
 JointExplicitStatistic f;f.group=&unit.group;std::vector<double*>original;std::vector<int>sizes;auto add=[&](double*p,int n){auto it=std::find(original.begin(),original.end(),p);if(it!=original.end())return int(it-original.begin());int k=int(original.size());original.push_back(p);sizes.push_back(n);return k;};
 for(auto raw:unit.group.rows){const auto&o=state.core.packet.rows[raw];JointExplicitRow row;row.model=&state.core.model;row.source=&o;row.layout=o.layout(state.core.model);row.uv_cache=gpu;row.cache_row=raw;std::vector<int>map;for(auto bi:row.layout.block_ids){const auto&b=state.core.model.blocks[bi];map.push_back(add(state.core.model.blocks[bi].values.data(),int(b.values.size())));}row=row.remap(map);row.physical_slots=map;if(o.kind!=1)row.station_slot=add(&state.core.station[raw],1);row.row_slot=add(&state.core.row[raw],1);f.rows.push_back(std::move(row));}
 auto owned=std::shared_ptr<const ceres::CostFunction>(event_autodiff_cost(std::move(f),sizes,int(unit.group.rank)).release());auto adapter=std::make_unique<JointHybridImplicitCostV1>(owned.get(),original,state,owned,gpu,guard);JointHybridDataCostV1 out;out.parameters=adapter->parameters();out.cost=std::move(adapter);return out;
}

class JointHybridPrepareCallbackV1 final:public ceres::EvaluationCallback {
 JointALBuilder*base_;JointHybridLocalStateV1*state_;
 public:JointHybridPrepareCallbackV1(JointALBuilder&base,JointHybridLocalStateV1&state):base_(&base),state_(&state){}
 void PrepareForEvaluation(bool derivatives,bool new_point)override{if(new_point||!state_->status->valid.load()){if(!state_->prepare(false))return;}if(auto*gpu=joint_hybrid_raw_callback_v1(*base_))gpu->PrepareForEvaluation(derivatives,new_point);}
};
struct JointHybridFullGraphV1 {
 // The target does not own manifolds. Their unchanged originals live in source.
 // Target owns adapters, and adapters share exactly one owner per original cost.
 std::unique_ptr<ceres::Problem>source,owned_target;ceres::Problem*target=nullptr;std::unique_ptr<JointHybridPrepareCallbackV1>callback;BuildInventory inventory;eventio::Json receipt;
};
inline JointHybridFullGraphV1 build_joint_hybrid_full_graph_v1(JointALBuilder&base,JointHybridLocalStateV1&state,const std::vector<JointHybridUnitV1>&units,const ObjectiveContext&objective,ceres::Problem*external=nullptr){
 if(!state.status->valid.load())throw std::runtime_error("hybrid production graph rejects whole source: "+state.status->reason);const auto&schema=joint_hybrid_original_schema_v1(base);std::size_t local_equalities=0;for(const auto&id:schema.equality_ids){if(id.rfind("RS/",0)==0||id.rfind("frozen_query_",0)==0)++local_equalities;else break;}if(objective.lambda.size()!=schema.inequality_ids.size()||objective.nu.size()!=schema.equality_ids.size()-local_equalities)throw std::runtime_error("hybrid original full pixel/physical multipliers or remaining hard equality inventory mismatch");ObjectiveContext expanded=objective;expanded.nu.assign(local_equalities,0.);expanded.nu.insert(expanded.nu.end(),objective.nu.begin(),objective.nu.end());
 JointHybridFullGraphV1 graph;ceres::Problem::Options so;so.cost_function_ownership=ceres::DO_NOT_TAKE_OWNERSHIP;so.manifold_ownership=ceres::DO_NOT_TAKE_OWNERSHIP;graph.source=std::make_unique<ceres::Problem>(so);BuildInventory old;
#ifdef RINKFIT_JOINT_FORMAL_HYBRID_HOOK_V1
 old=base.native_build_ceres_problem_v1(*graph.source,expanded);
#else
 old=base.build_ceres_problem(*graph.source,expanded);
#endif
graph.callback=std::make_unique<JointHybridPrepareCallbackV1>(base,state);ceres::Problem::Options to;to.evaluation_callback=graph.callback.get();if(external)graph.target=external;else{graph.owned_target=std::make_unique<ceres::Problem>(to);graph.target=graph.owned_target.get();}
 std::vector<double*>parameters;graph.source->GetParameterBlocks(&parameters);for(auto*p:parameters){const int n=graph.source->ParameterBlockSize(p);graph.target->AddParameterBlock(p,n);if(auto*m=graph.source->GetManifold(p))graph.target->SetManifold(p,const_cast<ceres::Manifold*>(m));if(graph.source->IsParameterBlockConstant(p))graph.target->SetParameterBlockConstant(p);for(int j=0;j<n;++j){graph.target->SetParameterLowerBound(p,j,graph.source->GetParameterLowerBound(p,j));graph.target->SetParameterUpperBound(p,j,graph.source->GetParameterUpperBound(p,j));}}
 graph.target->SetParameterLowerBound(base.budget_parameter(),0,0.);
 for(std::size_t i=0;i<base.core.packet.rows.size();++i){const auto&o=base.core.packet.rows[i];if(!o.active)continue;graph.target->SetParameterBlockConstant(&base.core.row[i]);if(o.query){graph.target->SetParameterBlockConstant(&base.core.station[i]);graph.target->SetParameterBlockConstant(&base.core.alpha[i]);}else if(o.kind!=1&&base.core.fixed_nuisance_shots.count(o.shot_id))graph.target->SetParameterBlockConstant(&base.core.station[i]);}
 std::map<void*,int>kind;for(auto id:old.inequality_blocks)kind[id]=1;for(std::size_t i=local_equalities;i<old.equality_blocks.size();++i)kind[old.equality_blocks[i]]=2;for(auto id:old.measured_prior_blocks)kind[id]=3;for(auto id:old.smoothness_blocks)kind[id]=4;
 std::vector<ceres::ResidualBlockId>ids;graph.source->GetResidualBlocks(&ids);std::map<const ceres::CostFunction*,std::shared_ptr<const ceres::CostFunction>>owners;std::size_t discarded_statistics=0,eliminated_equalities=0;
 for(auto id:ids){if(graph.source->GetLossFunctionForResidualBlock(id))throw std::runtime_error("hybrid full graph rejects robust loss on original anchors");auto*cost=graph.source->GetCostFunctionForResidualBlock(id);auto it=owners.find(cost);if(it==owners.end())it=owners.emplace(cost,std::shared_ptr<const ceres::CostFunction>(cost)).first;auto k=kind.find(id);if(k!=kind.end()){std::vector<double*>original;graph.source->GetParameterBlocksForResidualBlock(id,&original);auto adapter=std::make_unique<JointHybridImplicitCostV1>(cost,original,state,it->second,dynamic_cast<const JointUVCache*>(joint_hybrid_raw_callback_v1(base)));auto target_parameters=adapter->parameters();auto fresh=graph.target->AddResidualBlock(adapter.release(),nullptr,target_parameters);if(k->second==1)graph.inventory.inequality_blocks.push_back(fresh);else if(k->second==2)graph.inventory.equality_blocks.push_back(fresh);else if(k->second==3)graph.inventory.measured_prior_blocks.push_back(fresh);else graph.inventory.smoothness_blocks.push_back(fresh);}else{discarded_statistics+=std::find(old.statistical_blocks.begin(),old.statistical_blocks.end(),id)!=old.statistical_blocks.end();eliminated_equalities+=std::find(old.equality_blocks.begin(),old.equality_blocks.begin()+local_equalities,id)!=old.equality_blocks.begin()+local_equalities;}graph.source->RemoveResidualBlock(id);}
 std::size_t active=0;for(const auto&u:units){auto f=make_joint_hybrid_explicit_data_cost_v1(u,state,dynamic_cast<const JointUVCache*>(joint_hybrid_raw_callback_v1(base)));graph.inventory.statistical_blocks.push_back(graph.target->AddResidualBlock(f.cost.release(),nullptr,f.parameters));active+=u.group.rows.size();}graph.inventory.exact_final_physical_graph=true;
 if(graph.inventory.inequality_blocks.size()!=schema.inequality_ids.size()||graph.inventory.equality_blocks.size()!=objective.nu.size()||eliminated_equalities!=local_equalities||discarded_statistics!=old.statistical_blocks.size()||active!=std::count_if(base.core.packet.rows.begin(),base.core.packet.rows.end(),[](const auto&o){return o.active;}))throw std::runtime_error("hybrid full graph omitted original constraints or raw source");
 graph.receipt={{"schema","rinkfit.actual_hybrid_full_original_factor_graph.v1"},{"pixel_native_physical_zoom_guard_factors",graph.inventory.inequality_blocks.size()},{"local_RS_Query_equalities_exactly_IFT_eliminated",local_equalities},{"remaining_original_hard_equalities",graph.inventory.equality_blocks.size()},{"original_statistics_replaced_whole",discarded_statistics},{"all_source_hybrid_units",units.size()},{"all_active_original_UV_rows",active},{"complete_parameter_blocks_including_constant_witnesses",graph.target->NumParameterBlocks()},{"resident_raw_UV_GPU_partials_reused",joint_hybrid_raw_callback_v1(base)!=nullptr},{"original_GPU_backend_or_PTX_modified",false},{"rank0_removed_or_zeroed",false},{"original_W_modified",false},{"robust_loss",nullptr}};return graph;
}
} // namespace rinkfit
