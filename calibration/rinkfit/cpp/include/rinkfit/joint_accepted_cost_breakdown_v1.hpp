#pragma once
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_uv_cache.hpp"
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>
#include <typeinfo>
#include <set>

namespace rinkfit {
// An exact residual-only census of the live accepted problem. No independent
// replay, finite station reprofile, alternate weights, Jacobian, or row sample.
// Direct CostFunction evaluation is intentional: Problem::Evaluate cannot be
// called recursively while Ceres is executing an IterationCallback.
class JointAcceptedCostBreakdownV1 {
 struct Pixel {std::size_t row;double weight;};
 struct Stats {
  std::size_t rows=0,witness=0;long double cost=0,raw2=0,weights=0;
  double max_raw=0,max_cost=-1,min_weight=INFINITY,max_weight=0;
  void add(std::size_t row,double c,double e2,double w){++rows;cost+=c;raw2+=e2;weights+=w;max_raw=std::max(max_raw,std::sqrt(e2));min_weight=std::min(min_weight,w);max_weight=std::max(max_weight,w);if(c>max_cost){max_cost=c;witness=row;}}
  void merge(const Stats&v){rows+=v.rows;cost+=v.cost;raw2+=v.raw2;weights+=v.weights;max_raw=std::max(max_raw,v.max_raw);min_weight=std::min(min_weight,v.min_weight);max_weight=std::max(max_weight,v.max_weight);if(v.max_cost>max_cost){max_cost=v.max_cost;witness=v.witness;}}
 };
 struct Accumulator {std::map<std::pair<int,int>,Stats>pixel;std::map<std::string,long double>motion;long double domain=0,relation=0;std::size_t factors=0;std::vector<std::pair<double,std::size_t>>domain_top;};
 std::vector<Pixel>pixels_;std::vector<std::string>entities_;
 int threads_;bool completed_=false;std::filesystem::file_time_type request_time_{};
 static void keep_top(std::vector<std::pair<double,std::size_t>>&v,std::pair<double,std::size_t>item){v.push_back(item);std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return a.first>b.first;});if(v.size()>12)v.resize(12);}
 static double value(const ceres::Problem&p,ceres::ResidualBlockId id,std::vector<double>&residual,std::vector<double*>&parameters,std::vector<const double*>&constant){
  
  const auto*cost=p.GetCostFunctionForResidualBlock(id);p.GetParameterBlocksForResidualBlock(id,&parameters);constant.assign(parameters.begin(),parameters.end());residual.resize(cost->num_residuals());
  if(!cost->Evaluate(constant.data(),residual.data(),nullptr))throw std::runtime_error("accepted cost census residual evaluation failed");
  long double sum=0;for(double x:residual){if(!std::isfinite(x))throw std::runtime_error("nonfinite actual accepted residual in cost census");sum+=static_cast<long double>(x)*x;}if(const auto*loss=p.GetLossFunctionForResidualBlock(id)){double rho[3];loss->Evaluate(double(sum),rho);if(!std::isfinite(rho[0]))throw std::runtime_error("nonfinite actual robust cost in census");return .5*rho[0];}return double(.5L*sum);
 }
 public:
 JointAcceptedCostBreakdownV1(const JointALBuilder&b,const eventio::Json&ledger,int threads):threads_(std::max(1,threads)){
  if(!b.source_uv_observed_row_initialization())throw std::runtime_error("accepted cost census currently requires the observed-row 2D objective");
  std::vector<unsigned char>seen(b.core.packet.rows.size(),0);
  for(const auto&g:ledger.at("groups")){const auto&rows=g.at("actual_row_indices");const auto&weights=g.at("actual_per_row_weight");if(rows.size()!=weights.size())throw std::runtime_error("cost census original weight inventory differs");for(std::size_t k=0;k<rows.size();++k){const auto i=rows.at(k).get<std::size_t>();const double w=weights.at(k);if(i>=seen.size()||seen[i]++||!b.core.packet.rows[i].active||!std::isfinite(w)||!(w>0))throw std::runtime_error("cost census source/weight membership differs");pixels_.push_back({i,w});}}
  for(std::size_t i=0;i<seen.size();++i)if(bool(seen[i])!=b.core.packet.rows[i].active)throw std::runtime_error("cost census omitted or added a live source row");
  for(const auto&m:b.core.model.materials)entities_.push_back(m.label);
  if(ledger.contains("actual_pixel_weights_v2"))for(const auto&g:ledger.at("actual_pixel_weights_v2").value("canonical_physical_entity_groups",eventio::Json::array()))for(const auto&m:g.at("materials")){const auto mi=m.at("material_index").get<std::size_t>();if(b.core.model.materials.at(mi).label!=m.at("material_label"))throw std::runtime_error("cost census canonical entity material differs");entities_.at(mi)=g.at("canonical_physical_entity_id").get<std::string>();}
 }
 bool requested(const std::filesystem::path&out)const{const auto file=out/"COST_BREAKDOWN_REQUESTED";return !completed_||(std::filesystem::exists(file)&&std::filesystem::last_write_time(file)!=request_time_);}
 void write(JointALBuilder&b,ceres::Problem&p,const BuildInventory&inv,const std::filesystem::path&out,std::size_t iteration,double expected,const std::string&token,const std::string&state_origin="accepted_solver_iteration"){
  const auto begin=std::chrono::steady_clock::now();
  if(inv.statistical_blocks.size()<pixels_.size())throw std::runtime_error("cost census residual inventory shrank");
  const std::size_t motion=inv.statistical_blocks.size()-pixels_.size();
  if(motion!=(b.static_motion?b.static_motion->rows.size():0))throw std::runtime_error("cost census unrecognized statistical residual family");
  if(auto*cache=b.gpu_evaluation_callback())cache->PrepareForEvaluation(false,true);
  std::vector<Accumulator>acc(threads_);std::atomic<std::size_t>next{0};std::exception_ptr failure;std::mutex failure_mutex;
  const auto nstat=inv.statistical_blocks.size(),ng=inv.inequality_blocks.size(),nh=inv.equality_blocks.size(),total=nstat+ng+nh;
  std::vector<ceres::ResidualBlockId>live;p.GetResidualBlocks(&live);
  std::vector<void*>live_sorted(live.begin(),live.end()),known;known.reserve(total);
  known.insert(known.end(),inv.statistical_blocks.begin(),inv.statistical_blocks.end());known.insert(known.end(),inv.inequality_blocks.begin(),inv.inequality_blocks.end());known.insert(known.end(),inv.equality_blocks.begin(),inv.equality_blocks.end());
  const auto less=std::less<void*>{};std::sort(live_sorted.begin(),live_sorted.end(),less);std::sort(known.begin(),known.end(),less);
  const auto unique_end=std::unique(known.begin(),known.end());const auto duplicates=std::size_t(known.end()-unique_end);known.erase(unique_end,known.end());
  std::size_t inventory_missing=0;for(void*id:known)if(!std::binary_search(live_sorted.begin(),live_sorted.end(),id,less))++inventory_missing;
  std::vector<ceres::ResidualBlockId>unclassified;for(auto id:live)if(!std::binary_search(known.begin(),known.end(),static_cast<void*>(id),less))unclassified.push_back(id);
  auto work=[&](int worker){try{auto&a=acc[worker];std::vector<double>residual;std::vector<double*>parameters;std::vector<const double*>constant;for(;;){const auto first=next.fetch_add(128);if(first>=total)break;for(auto k=first;k<std::min(first+128,total);++k){
   const auto id=k<nstat?inv.statistical_blocks[k]:(k<nstat+ng?inv.inequality_blocks[k-nstat]:inv.equality_blocks[k-nstat-ng]);if(!std::binary_search(live_sorted.begin(),live_sorted.end(),id,less))continue;const double cost=value(p,static_cast<ceres::ResidualBlockId>(id),residual,parameters,constant);++a.factors;
   if(k<pixels_.size()){if(residual.size()!=2)throw std::runtime_error("cost census pixel residual is not actual source UV 2D");const auto&meta=pixels_[k];const auto&r=b.core.packet.rows[meta.row];const double raw2=2*cost*r.E*r.E/meta.weight;a.pixel[{b.core.model.shot_map.at(r.shot_id),r.material}].add(meta.row,cost,raw2,meta.weight);}
   else if(k<nstat){const auto&r=b.static_motion->rows[k-pixels_.size()];a.motion[b.core.model.shots.at(r.shot).id]+=cost;}
   else if(k<nstat+ng){a.domain+=cost;keep_top(a.domain_top,{cost,k-nstat});}
   else a.relation+=cost;
  }}}catch(...){std::lock_guard<std::mutex>lock(failure_mutex);if(!failure)failure=std::current_exception();}};
  std::vector<std::thread>pool;for(int k=1;k<threads_;++k)pool.emplace_back(work,k);work(0);for(auto&t:pool)t.join();
  std::string evaluation_error;if(failure)try{std::rethrow_exception(failure);}catch(const std::exception&e){evaluation_error=e.what();}
  Accumulator sum;for(const auto&a:acc){for(const auto&v:a.pixel)sum.pixel[v.first].merge(v.second);for(const auto&v:a.motion)sum.motion[v.first]+=v.second;sum.domain+=a.domain;sum.relation+=a.relation;sum.factors+=a.factors;for(const auto&v:a.domain_top)keep_top(sum.domain_top,v);}
  auto witness=[&](std::size_t i){const auto&r=b.core.packet.rows.at(i);const auto l=r.layout(b.core.model);auto ptr=b.core.model.pointers(l);const double station=r.kind==1?r.station:b.core.station[i];const auto q=b.core.model.predict_for_al(l,station,b.core.row[i],ptr.data());if(!q.arithmetic_valid||!finite_array(q.uv))throw std::runtime_error("cost census witness native projection failed");return eventio::Json{{"packet_row",i},{"source_row_id",r.source_row_id},{"shot",r.shot_id},{"native_pts",r.pts},{"material",b.core.model.materials.at(r.material).label},{"path",r.path},{"station",station},{"explicit_row",b.core.row[i]},{"source_UV",r.uv},{"predicted_UV",q.uv},{"raw_error_px",std::hypot(q.uv[0]-r.uv[0],q.uv[1]-r.uv[1])},{"original_E",r.E},{"width_block",r.width_block},{"width_sign",r.width_sign}};};
  auto statistics=[&](const Stats&s){return eventio::Json{{"rows",s.rows},{"weighted_half_cost",double(s.cost)},{"raw_pixel_sum_squared",double(s.raw2)},{"raw_pixel_RMS",s.rows?std::sqrt(double(s.raw2/s.rows)):0.},{"raw_pixel_max",s.max_raw},{"original_weight_sum",double(s.weights)},{"original_weight_min",s.min_weight},{"original_weight_max",s.max_weight},{"largest_weighted_row_half_cost",s.max_cost},{"largest_weighted_row_witness",witness(s.witness)}};};
  eventio::Json by_material=eventio::Json::array(),by_entity=eventio::Json::array(),by_shot=eventio::Json::array();std::map<std::pair<int,std::string>,Stats>entities;std::map<int,Stats>shots;long double pixel_cost=0,motion_cost=0;
  for(const auto&v:sum.pixel){auto d=statistics(v.second);d["shot"]=b.core.model.shots.at(v.first.first).id;d["material_index"]=v.first.second;d["material"]=b.core.model.materials.at(v.first.second).label;d["canonical_entity"]=entities_.at(v.first.second);by_material.push_back(std::move(d));entities[{v.first.first,entities_.at(v.first.second)}].merge(v.second);shots[v.first.first].merge(v.second);pixel_cost+=v.second.cost;}
  for(const auto&v:entities){auto d=statistics(v.second);d["shot"]=b.core.model.shots.at(v.first.first).id;d["canonical_entity"]=v.first.second;by_entity.push_back(std::move(d));}
  for(const auto&v:shots){auto d=statistics(v.second);d["shot"]=b.core.model.shots.at(v.first).id;d["static_motion_half_cost"]=double(sum.motion[b.core.model.shots.at(v.first).id]);by_shot.push_back(std::move(d));}
  auto descending=[](const auto&a,const auto&z){return a.at("weighted_half_cost").template get<double>()>z.at("weighted_half_cost").template get<double>();};std::sort(by_material.begin(),by_material.end(),descending);std::sort(by_entity.begin(),by_entity.end(),descending);std::sort(by_shot.begin(),by_shot.end(),descending);
  for(const auto&v:sum.motion)motion_cost+=v.second;
  eventio::Json extra=eventio::Json::array();long double extra_cost=0;std::size_t extra_evaluated=0;
  std::map<const double*,std::string>physical_names;for(const auto&block:b.core.model.blocks)physical_names.emplace(block.values.data(),block.name);
  for(std::size_t k=0;k<unclassified.size();++k){const auto id=unclassified[k];std::vector<double>residual;std::vector<double*>parameters;std::vector<const double*>constant;eventio::Json item={{"representative_live_id",k},{"actual_cost_function_type",typeid(*p.GetCostFunctionForResidualBlock(id)).name()}};
   try{const double c=value(p,id,residual,parameters,constant);extra_cost+=c;++extra_evaluated;item["half_cost"]=c;item["actual_residuals"]=residual;item["parameters"]=eventio::Json::array();for(auto*ptr:parameters){auto it=physical_names.find(ptr);item["parameters"].push_back({{"name",it==physical_names.end()?"nonphysical_or_unregistered_parameter":it->second},{"ambient_size",p.ParameterBlockSize(ptr)},{"tangent_size",p.ParameterBlockTangentSize(ptr)},{"constant",p.IsParameterBlockConstant(ptr)}});}}
   catch(const std::exception&e){item["evaluation_error"]=e.what();evaluation_error+="; unclassified: "+std::string(e.what());}extra.push_back(std::move(item));
  }
  const long double actual=pixel_cost+motion_cost+sum.domain+sum.relation+extra_cost;const double closure=double(actual)-expected,roundoff=2048*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(expected));
  const bool closed=duplicates==0&&inventory_missing==0&&sum.factors+extra_evaluated==live.size()&&std::isfinite(expected)&&std::abs(closure)<=roundoff&&evaluation_error.empty();
  eventio::Json domain_top=eventio::Json::array();for(const auto&v:sum.domain_top){const auto id=inv.inequality_blocks.at(v.second);domain_top.push_back({{"inventory_index",v.second},{"half_cost",v.first},{"actual_cost_function_type",typeid(*p.GetCostFunctionForResidualBlock(static_cast<ceres::ResidualBlockId>(id))).name()}});}
  const auto folder=out/"accepted_cost_breakdown";std::filesystem::create_directories(folder);const auto file=folder/("iteration_"+std::to_string(iteration)+".json");
  eventio::atomic_json(file,{{"schema","rinkfit.exact_accepted_cost_breakdown.v1"},{"global_iteration",iteration},{"compact_primal_token",token},{"packet_sha256",b.core.packet.source_hash},{"objective_id",b.identity().objective_id},{"source_UV_pixel_half_cost",double(pixel_cost)},{"static_surface_motion_half_cost",double(motion_cost)},{"physical_domain_g_half_cost",double(sum.domain)},{"physical_relation_h_half_cost",double(sum.relation)},{"actual_total_half_cost",double(actual)},{"reference_half_cost",expected},{"reference_kind",state_origin},{"closure_error",closure},{"closure_roundoff_bound",roundoff},{"pixel_rows",pixels_.size()},{"motion_factors",motion},{"all_residual_factors",live.size()},{"known_inventory_residual_factors",total},{"unclassified_residual_factors",unclassified.size()},{"unclassified_half_cost",double(extra_cost)},{"unclassified_live_residuals",extra},{"inventory_missing_live_factors",inventory_missing},{"duplicate_inventory_handles",duplicates},{"actually_evaluated_factors",sum.factors+extra_evaluated},{"evaluation_error",evaluation_error},{"closure_passed",closed},{"by_shot",by_shot},{"by_shot_entity",by_entity},{"by_shot_material",by_material},{"largest_domain_factors",domain_top},{"common_budget_consumed",false},{"standalone_optics_residual",false},{"optics_enter_actual_projection_and_original_physical_domains",true},{"source_E_UV_weights_or_parameters_changed",false},{"all_current_active_residuals_counted_once",closed},{"Jacobians_or_large_row_arrays_written",false},{"quality_accepted",false},{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}});
  eventio::atomic_json(folder/"latest.json",{{"global_iteration",iteration},{"report",{{"file",file.u8string()},{"sha256",eventio::file_sha256(file)}}},{"actual_total_half_cost",double(actual)}});
  completed_=true;const auto request=out/"COST_BREAKDOWN_REQUESTED";if(std::filesystem::exists(request))request_time_=std::filesystem::last_write_time(request);
 }
};
}
