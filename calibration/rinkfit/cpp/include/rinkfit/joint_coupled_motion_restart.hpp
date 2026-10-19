#pragma once
#include "rinkfit/joint_al.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <sstream>

namespace rinkfit {
namespace joint_coupled_motion_detail {
// Values-only rollback: do not synchronize untouched packet mirrors. Registry
// storage is fixed for this call and all accepted/trial doubles own their data.
class Primal {
 std::vector<ParameterBlockBinding> bindings_;
 std::vector<std::size_t> offsets_;
 std::vector<double> values_;
 public:
 explicit Primal(JointALBuilder&b):bindings_(b.parameter_blocks()){
  std::size_t n=0;for(const auto&r:bindings_){offsets_.push_back(n);n+=r.size;}
  values_.resize(n);for(std::size_t i=0;i<bindings_.size();++i)
   std::memcpy(values_.data()+offsets_[i],bindings_[i].data,bindings_[i].size*sizeof(double));
 }
 void restore()const{for(std::size_t i=0;i<bindings_.size();++i)
  std::memcpy(bindings_[i].data,values_.data()+offsets_[i],bindings_[i].size*sizeof(double));}
 bool matches()const{for(std::size_t i=0;i<bindings_.size();++i)
  if(std::memcmp(values_.data()+offsets_[i],bindings_[i].data,bindings_[i].size*sizeof(double)))return false;return true;}
 std::size_t bytes()const{return values_.capacity()*sizeof(double);}
};
struct Row {
 std::size_t i;int path;
 double source_station,source_row,source_alpha,station,row,alpha;
};
// A local mutation/freshness guard, not source/statistical qualification.
struct Token {
 std::uint64_t h=1469598103934665603ull;
 void bytes(const void*p,std::size_t n){const auto*q=static_cast<const unsigned char*>(p);for(std::size_t i=0;i<n;++i){h^=q[i];h*=1099511628211ull;}}
 template<class T>void add(const T&v){bytes(&v,sizeof(v));}
 void add(const std::string&s){add(s.size());bytes(s.data(),s.size());}
 std::string text()const{std::ostringstream out;out<<std::hex<<h;return out.str();}
};
inline std::pair<std::string,std::string> row_tokens(const JointALBuilder&b,const std::vector<std::uint8_t>&preserved){
 Token immutable,untouched;
 immutable.add(b.core.packet.source_hash);immutable.add(b.core.packet.rows.size());
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];
  immutable.add(r.source_dataset_id);immutable.add(r.source_row_id);immutable.add(r.shot_id);immutable.add(r.source_group_id);
  immutable.add(r.pts);immutable.add(r.uv);immutable.add(r.normal);immutable.add(r.material);immutable.add(r.kind);
  immutable.add(r.width_block);immutable.add(r.width_sign);immutable.add(r.stroke_pair_index);immutable.add(r.stroke_source_side);
  immutable.add(r.E);immutable.add(r.active);immutable.add(r.fit_role);immutable.add(r.station_domain_path);
  immutable.add(r.source_station_intervals.size());for(const auto&v:r.source_station_intervals)immutable.add(v);
  const auto query=r.query.get();immutable.add(query); // JointQuery itself is shared_ptr<const>.
  if(preserved[i]||!r.active){untouched.add(i);untouched.add(r.path);untouched.add(r.station);untouched.add(r.row);untouched.add(r.alpha);
   untouched.add(b.core.station[i]);untouched.add(b.core.row[i]);untouched.add(b.core.alpha[i]);}
 }
 return {immutable.text(),untouched.text()};
}
inline eventio::Json roots(const JointALBuilder&b,const eventio::Json&indices){
 eventio::Json out=eventio::Json::array();
 for(const auto&index:indices){const auto i=index.get<std::size_t>();const auto&r=b.core.packet.rows.at(i);
  const auto layout=r.layout(b.core.model);const auto ptr=b.core.model.pointers(layout);
  const double station=r.kind==1?r.station:b.core.station[i];
  const auto cp=b.core.model.predict_for_al(layout,station,b.core.row[i],ptr.data());
  const auto root=b.core.model.project_rs(layout,station,r.uv[1]);
  out.push_back({{"row_index",i},{"source_row_id",r.source_row_id},{"shot_id",r.shot_id},{"native_pts",r.pts},
   {"source_uv",r.uv},{"path",r.path},{"station",station},{"stored_row",b.core.row[i]},
   {"CP_ray_domain_valid",cp.physical_valid()},{"native_RS_valid",root.valid},
   {"native_RS_error_px",root.valid?eventio::Json(std::hypot(root.uv[0]-r.uv[0],root.uv[1]-r.uv[1])):eventio::Json()}});
 }return out;
}
inline std::size_t valid_roots(const eventio::Json&v){std::size_t n=0;for(const auto&r:v)n+=r.at("native_RS_valid").get<bool>();return n;}
inline int effective_parameters(const ceres::Problem&problem){
 std::vector<double*>blocks;problem.GetParameterBlocks(&blocks);int n=0;
 for(const auto*p:blocks)if(!problem.IsParameterBlockConstant(p))n+=problem.ParameterBlockTangentSize(p);return n;
}
} // namespace joint_coupled_motion_detail

// Call only at a normal graph/checkpoint boundary, before numerical manifolds
// and Ceres Solve. Rebuild must refresh GPU cache, the COMPLETE same-rho/dual
// engineering graph and current stage; it must not alter b or its source/state.
// initialize is the existing initialize_recovery_auxiliaries(core,out,threads,
// preserved) callback. Only selected active rows have a zero preserved byte.
// Both callbacks must return synchronously. No graph may survive path mutation.
inline eventio::Json refine_joint_coupled_motion_navigation(
 JointALBuilder&b,std::unique_ptr<ceres::Problem>&problem,const eventio::Json&spec,
 const std::filesystem::path&out,int threads,
 const std::function<void()>&destroy_graph,
 const std::function<void(const std::filesystem::path&)>&rebuild_graph,
 const std::function<eventio::Json(const std::vector<std::uint8_t>&,const std::filesystem::path&)>&initialize,
 const std::function<eventio::Json()>&constraint_debt){
 using namespace joint_coupled_motion_detail;
 if(!problem)throw std::runtime_error("coupled motion navigation requires the complete current graph");
 if(spec.at("source_packet_sha256").get<std::string>()!=b.core.packet.source_hash)
  throw std::runtime_error("coupled motion navigation source mismatch");
 if(spec.at("source_evidence").empty()||spec.at("proposals").empty()||spec.at("diagnostic_source_rows").empty())
  throw std::runtime_error("coupled motion navigation requires actual source evidence, controls and diagnostic rows");
 for(const auto&ref:spec.at("source_evidence"))eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));
 const auto started=std::chrono::steady_clock::now();std::filesystem::create_directories(out);
 struct Proposal{int block,axis;double before,value;};std::vector<Proposal>proposals;std::set<int>targets;std::set<std::pair<int,int>>coordinates;
 for(const auto&p:spec.at("proposals")){
  const auto name=p.at("block").get<std::string>();const int axis=p.at("axis");
  const auto it=std::find_if(b.core.model.blocks.begin(),b.core.model.blocks.end(),[&](const auto&v){return v.name==name;});
  if(it==b.core.model.blocks.end()||axis<0||axis>=3||it->values.size()!=3||it->constant||!it->fixed_indices.empty()||
     !problem->HasParameterBlock(it->values.data())||problem->IsParameterBlockConstant(it->values.data())||problem->ParameterBlockTangentSize(it->values.data())!=3)
   throw std::runtime_error("coupled motion target must be an existing free three-coordinate control");
  const int id=int(it-b.core.model.blocks.begin());bool actual_control=false;
  for(const auto&shot:b.core.model.shots)actual_control=actual_control||std::find(shot.control_blocks.begin(),shot.control_blocks.end(),id)!=shot.control_blocks.end();
  if(!actual_control||!coordinates.insert({id,axis}).second)throw std::runtime_error("coupled motion target is not a unique existing shot control coordinate");
  double value=0,weight=0;
  for(const auto&n:p.at("navigation_controls")){
   const auto nn=n.at("block").get<std::string>();const double w=n.at("weight");
   const auto ni=std::find_if(b.core.model.blocks.begin(),b.core.model.blocks.end(),[&](const auto&v){return v.name==nn;});
   if(ni==b.core.model.blocks.end()||ni->values.size()!=3||!(w>=0&&std::isfinite(w)))throw std::runtime_error("invalid coupled motion navigation control");
   const int nid=int(ni-b.core.model.blocks.begin());bool same_shot=false;
   for(const auto&shot:b.core.model.shots)if(std::find(shot.control_blocks.begin(),shot.control_blocks.end(),id)!=shot.control_blocks.end())
    same_shot=same_shot||std::find(shot.control_blocks.begin(),shot.control_blocks.end(),nid)!=shot.control_blocks.end();
   if(!same_shot||nid==id)throw std::runtime_error("motion navigation must use other controls of the same shot");
   value+=w*ni->values[axis];weight+=w;
  }
  if(std::abs(weight-1)>1e-12||!std::isfinite(value)||value<it->lower.at(axis)||value>it->upper.at(axis))throw std::runtime_error("invalid coupled motion navigation value");
  targets.insert(id);proposals.push_back({id,axis,it->values[axis],value});
 }
 for(int id:targets)for(int axis=0;axis<3;++axis)if(!coordinates.count({id,axis}))
  throw std::runtime_error("coupled motion navigation requires all three free control coordinates as a starting proposal");
 std::vector<std::uint8_t>preserved(b.core.packet.rows.size(),1);std::vector<Row>rows;
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;
  const auto layout=r.layout(b.core.model);bool selected=false;for(int id:layout.block_ids)selected=selected||targets.count(id);
  if(!selected)continue;
  const auto&material=b.core.model.materials.at(r.material);
  if((material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty()){
   if(material.paths.size()!=8)throw std::runtime_error("coupled motion whole-board source requires original eight charts");
   const bool domain=b.core.model.station_intervals(r.material,r.path)!=std::vector<std::array<double,2>>{{0.,1.}};
   for(std::size_t k=0;k<material.paths.size();++k)if(material.paths[k].kind!=3||material.paths[k].physical_point||
    (b.core.model.station_intervals(r.material,int(k))!=std::vector<std::array<double,2>>{{0.,1.}})!=domain)
     throw std::runtime_error("coupled motion whole-board chart would change finite native-domain schema");
  }
  preserved[i]=0;rows.push_back({i,r.path,r.station,r.row,r.alpha,b.core.station[i],b.core.row[i],b.core.alpha[i]});
 }
 if(rows.empty())throw std::runtime_error("coupled motion target has no active source consumers");
 std::set<std::size_t>diagnostics;
 for(const auto&v:spec.at("diagnostic_source_rows")){const auto i=v.get<std::size_t>();
  if(i>=preserved.size()||preserved[i]||!diagnostics.insert(i).second)throw std::runtime_error("coupled motion diagnostic must be a unique selected active source row");}
 const auto tokens_before=row_tokens(b,preserved);Primal before(b);
 const int residual_count=problem->NumResiduals(),parameter_count=problem->NumParameters(),effective_count=effective_parameters(*problem);
 auto evaluate=[&](){
  if(!problem)throw std::runtime_error("coupled motion objective has no current graph");
  if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))cache->PrepareForEvaluation(false,true);
  ceres::Problem::EvaluateOptions options;options.num_threads=threads;double value=NAN;
  if(!problem->Evaluate(options,&value,nullptr,nullptr,nullptr)||!std::isfinite(value))throw std::runtime_error("coupled motion complete current PHR evaluation failed");return value;
 };
 auto destroy=[&](){destroy_graph();if(problem)throw std::runtime_error("coupled motion callback retained a live graph before mutation");};
 auto rebuild=[&](const char*name){if(problem)throw std::runtime_error("coupled motion rebuild requires destroyed graph");rebuild_graph(out/name);
  if(!problem||problem->NumResiduals()!=residual_count||problem->NumParameters()!=parameter_count||effective_parameters(*problem)!=effective_count)
   throw std::runtime_error("coupled motion changed complete residual/parameter denominator or stage");};
 const double cost_before=evaluate();const auto debt_before=constraint_debt(),roots_before=roots(b,spec.at("diagnostic_source_rows"));
 double candidate_cost=NAN,restored_cost=NAN;eventio::Json candidate_debt,roots_candidate,initialization;
 bool accepted=false,mutated=false,rolled_back=false;std::string candidate_error;
 auto rollback=[&](){
  if(problem)destroy();before.restore();
  for(const auto&s:rows){auto&r=b.core.packet.rows[s.i];r.path=s.path;r.station=s.source_station;r.row=s.source_row;r.alpha=s.source_alpha;
   b.core.station[s.i]=s.station;b.core.row[s.i]=s.row;b.core.alpha[s.i]=s.alpha;}
  if(!before.matches()||row_tokens(b,preserved)!=tokens_before)throw std::runtime_error("coupled motion rollback did not restore exact primal/source mirrors");
  rebuild("rollback_gpu_raw_uv_cache.jsonl");restored_cost=evaluate();
  if(std::abs(restored_cost-cost_before)>1e-8*std::max(1.,std::abs(cost_before)))throw std::runtime_error("coupled motion rollback objective mismatch");
  rolled_back=true;mutated=false;
 };
 try{
  mutated=true;destroy();for(const auto&p:proposals)b.core.model.blocks[p.block].values[p.axis]=p.value;
  initialization=initialize(preserved,out/"native_auxiliary_navigation");
  if(row_tokens(b,preserved)!=tokens_before)throw std::runtime_error("coupled motion initializer changed immutable source or unselected auxiliaries");
  for(const auto&s:rows){const auto&r=b.core.packet.rows[s.i];
   const auto&material=b.core.model.materials.at(r.material);
   const bool whole_board=(material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty();
   if((r.path!=s.path&&!whole_board)||!r.station_allowed(b.core.model,r.path,b.core.station[s.i],1e-12)||
      !std::isfinite(b.core.row[s.i])||b.core.row[s.i]<0||b.core.row[s.i]>1079||
      (r.kind==1&&std::memcmp(&r.station,&s.source_station,sizeof(double)))||
      (r.query&&(!std::isfinite(b.core.alpha[s.i])||b.core.alpha[s.i]<r.query->alpha.front()||b.core.alpha[s.i]>r.query->alpha.back())))
    throw std::runtime_error("coupled motion initializer changed finite source/Query support");
  }
  rebuild("candidate_gpu_raw_uv_cache.jsonl");candidate_cost=evaluate();candidate_debt=constraint_debt();roots_candidate=roots(b,spec.at("diagnostic_source_rows"));
  accepted=candidate_cost<cost_before&&valid_roots(roots_candidate)>valid_roots(roots_before);
 }catch(const std::exception&e){candidate_error=e.what();accepted=false;}
 // Record the actual proposal before any rollback; stream rows, not a giant JSON array.
 std::ofstream ledger(out/"proposed_rows.jsonl");
 for(const auto&s:rows){const auto&r=b.core.packet.rows[s.i];ledger<<eventio::Json({{"row_index",s.i},{"source_row_id",r.source_row_id},{"shot_id",r.shot_id},{"native_pts",r.pts},
  {"source_uv",r.uv},{"old_path",s.path},{"path",r.path},{"old_station",s.station},{"station",b.core.station[s.i]},
  {"old_row",s.row},{"row",b.core.row[s.i]},{"old_alpha",s.alpha},{"alpha",b.core.alpha[s.i]}}).dump()<<'\n';}
 ledger.close();if(!ledger){if(mutated)rollback();throw std::runtime_error("coupled motion proposal ledger write failed");}
 if(!accepted&&mutated)rollback();
 eventio::Json pjson=eventio::Json::array();for(const auto&p:proposals)pjson.push_back({{"block",b.core.model.blocks[p.block].name},{"axis",p.axis},{"before",p.before},{"navigation_proposal",p.value}});
 eventio::Json report={{"schema","rinkfit.source_evidenced_coupled_motion_aux_navigation.v1"},{"source_packet_sha256",b.core.packet.source_hash},
  {"source_evidence",spec.at("source_evidence")},{"proposals",pjson},{"selected_active_layout_consumers",rows.size()},{"raw_rows",b.core.packet.rows.size()},
  {"full_residual_count",residual_count},{"full_parameter_count",parameter_count},{"full_effective_parameter_count",effective_count},
  {"compact_primal_capacity_bytes",before.bytes()},{"full_current_PHR_cost_before",cost_before},
  {"full_current_PHR_cost_candidate",std::isfinite(candidate_cost)?eventio::Json(candidate_cost):eventio::Json()},
  {"original_constraint_debt_before",debt_before},{"original_constraint_debt_candidate",candidate_debt},
  {"diagnostic_source_roots_before",roots_before},{"diagnostic_source_roots_candidate",roots_candidate},
  {"accepted",accepted},{"exact_rollback_performed",rolled_back},{"rollback_full_current_PHR_cost",std::isfinite(restored_cost)?eventio::Json(restored_cost):eventio::Json()},
  {"candidate_error",candidate_error},{"auxiliary_initializer_receipt",initialization},{"source_UV_ID_Query_W_E_or_precision_changed",false},
  {"same_complete_problem_and_all_original_free_coordinates_retained",true},{"continuity_residual_or_post_smoothing_added",false},
  {"navigation_controls_are_only_starting_guesses",true},{"graph_and_GPU_layout_rebuilt_after_apply_or_rollback",true},
  {"source_or_statistics_qualification_granted",false},{"proposed_rows_file_sha256",eventio::file_sha256(out/"proposed_rows.jsonl")},
  {"complete_checkpoint_required_from_caller",true},{"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
 eventio::atomic_json(out/"receipt.json",report);return report;
}
} // namespace rinkfit
