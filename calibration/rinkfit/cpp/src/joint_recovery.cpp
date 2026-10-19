#include "rinkfit/joint_recovery.hpp"
#include "rinkfit/joint_memory.hpp"
#include "rinkfit/joint_station_restart.hpp"
#include "rinkfit/joint_motion_restart.hpp"
#include "rinkfit/joint_coupled_motion_restart.hpp"
#include "rinkfit/joint_finite_line_restart.hpp"
#include "rinkfit/joint_source_camera_mode.hpp"
#include "rinkfit/joint_screen_grid_navigation.hpp"
#include "rinkfit/joint_query_restart.hpp"
#include "rinkfit/joint_compact_state.hpp"
#include "rinkfit/joint_baseline_aux.hpp"
#include "rinkfit/joint_accepted_aux_prefix.hpp"
#include "rinkfit/joint_component_rollback.hpp"
#include "rinkfit/joint_engineering_outer.hpp"
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_source_guard.hpp"
#include "rinkfit/joint_replay_artifacts.hpp"
#include "rinkfit/joint_seed_inner.hpp"
#include "rinkfit/joint_recovery_metrics_export.hpp"
#include <iostream>
#include <atomic>
#include <thread>
namespace rinkfit {namespace {

eventio::Json initialize_recovery_auxiliaries(JointProblemBuilder&b,const std::filesystem::path&out,int threads,const std::vector<std::uint8_t>&preserved){
 std::vector<std::uint8_t>status(b.packet.rows.size(),0);std::atomic<std::size_t>next{0},valid{0},failed{0},recharted{0},no_finite{0},weak_visibility{0},weak_ray{0},preserved_count{0};
 const auto work=[&](){for(;;){const auto i=next.fetch_add(1);if(i>=b.packet.rows.size())break;auto&r=b.packet.rows[i];if(!r.active)continue;
  if(!preserved.empty()&&preserved.at(i)){status[i]=3;++preserved_count;continue;}
  const double observed_row=r.uv[1];double station=r.station;int chosen_path=r.path;
  const auto&material=b.model.materials.at(r.material);
  const bool whole_board=(material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty();
  std::vector<int>paths={r.path};if(whole_board){paths.clear();for(std::size_t k=0;k<material.paths.size();++k){const auto&path=material.paths[k];if(path.kind!=3||path.physical_point)throw std::runtime_error("continuous board source has an incompatible finite chart");paths.push_back(int(k));}}
  double best=INFINITY;
  for(int seed_tier=0;seed_tier<(whole_board?3:1)&&!std::isfinite(best);++seed_tier){
  for(int path:paths){const auto l=b.model.layout(b.model.shot_map.at(r.shot_id),r.material,path,r.pts,r.width_block,r.width_sign);const auto ptr=b.model.pointers(l);
   auto error=[&](double st){const auto pred=b.model.predict_for_al(l,st,observed_row,ptr.data());
    if(!pred.arithmetic_valid||!finite_array(pred.uv))return double(INFINITY);
    if(whole_board&&seed_tier<2&&pred.physical_status!=DomainStatus::kValid)return double(INFINITY);
    if(whole_board&&seed_tier==0&&!b.model.visibility(l,st,observed_row,ptr.data()).visible())return double(INFINITY);
    return std::hypot(pred.uv[0]-r.uv[0],pred.uv[1]-r.uv[1]);
   };
   if(r.kind==1){const auto value=error(r.station);if(value<best){best=value;station=r.station;chosen_path=path;}continue;}
   for(const auto&interval:r.station_intervals(b.model,path)){
    double local=INFINITY;int kbest=-1;for(int k=0;k<=32;++k){const double st=interval[0]+(interval[1]-interval[0])*k/32.;const double value=error(st);if(value<local){local=value;kbest=k;}}
    if(kbest<0||!std::isfinite(local))continue;double candidate=interval[0]+(interval[1]-interval[0])*kbest/32.;
    double lo=interval[0]+(interval[1]-interval[0])*std::max(0,kbest-1)/32.,hi=interval[0]+(interval[1]-interval[0])*std::min(32,kbest+1)/32.;
    for(int k=0;k<28;++k){const double x=lo+(hi-lo)*.3819660112501051,z=lo+(hi-lo)*.6180339887498949;if(error(x)<error(z))hi=z;else lo=x;}
    const double st=(lo+hi)*.5,value=error(st);if(value<local){local=value;candidate=st;}
    if(local<best){best=local;station=candidate;chosen_path=path;}
   }
  }
  if(seed_tier==1&&std::isfinite(best))++weak_visibility;
  if(seed_tier==2&&std::isfinite(best))++weak_ray;
  }
  if(!std::isfinite(best))++no_finite;
  if(chosen_path!=r.path){++recharted;r.path=chosen_path;}
  const auto l=r.layout(b.model);const auto ptr=b.model.pointers(l);const auto root=b.model.project_rs(l,station,observed_row);b.station[i]=r.station=station;
  b.row[i]=r.row=root.valid?root.row:observed_row;
  if(r.query){const auto prediction=b.model.predict_for_al(l,station,b.row[i],ptr.data());double best_alpha=INFINITY,alpha=r.alpha;
   if(prediction.arithmetic_valid)for(int k=0;k<=64;++k){const double x=r.query->alpha.front()+(r.query->alpha.back()-r.query->alpha.front())*k/64.;Vec2<double>q;if(r.query->point(x,&q)){const double v=std::hypot(q[0]-prediction.uv[0],q[1]-prediction.uv[1]);if(v<best_alpha){best_alpha=v;alpha=x;}}}
   b.alpha[i]=r.alpha=alpha;
  }
  status[i]=root.valid?1:2;if(root.valid)++valid;else ++failed;
 }};
 std::vector<std::thread>pool;for(int i=0;i<std::max(1,threads);++i)pool.emplace_back(work);for(auto&t:pool)t.join();
 std::filesystem::create_directories(out);std::ofstream f(out/"initialization_status.u8",std::ios::binary);f.write(reinterpret_cast<const char*>(status.data()),status.size());f.close();
 eventio::Json receipt={{"schema","rinkfit.recovery_fresh_auxiliary_initialization.v2"},{"source_packet_sha256",b.packet.source_hash},{"raw_rows",b.packet.rows.size()},{"fresh_RS_valid_rows",valid.load()},{"fresh_RS_failed_rows",failed.load()},{"preserved_auxiliary_rows_not_RS_recertified",preserved_count.load()},{"no_arithmetic_finite_chart_rows",no_finite.load()},{"no_visible_chart_used_ray_finite_seed_rows",weak_visibility.load()},{"no_ray_valid_chart_used_arithmetic_seed_rows",weak_ray.load()},{"continuous_board_numerical_rechart_rows",recharted.load()},{"source_UV_ID_material_operator_unchanged",true},{"chart_policy","Only declared whole continuous board.top/contact materials without source-restricted station intervals may change finite perimeter chart; all fixed entities and other finite material paths remain fixed. If no visible chart seed exists, closest ray-valid then arithmetic-finite seeds are retained with all physical failures still counted; a default unrelated chart is never treated as valid."},{"failed_RS_policy","retain active source and explicit observed-row numerical seed; independent failure remains counted"},{"status_file_sha256",eventio::file_sha256(out/"initialization_status.u8")}};
 eventio::atomic_json(out/"receipt.json",receipt);return receipt;
}

void save_recovery_state(JointALBuilder&b,const std::filesystem::path&out,const std::string&fingerprint){
 std::filesystem::create_directories(out);eventio::atomic_json(out/"model.json",b.core.model.exact_json());
 b.export_explicit_nuisance(out/"explicit_nuisance");
 OuterState unused;unused.phase=Phase::kPrecision;unused.rho=0;
 std::ofstream stream(out/"engineering_checkpoint.json");write_checkpoint(stream,{b.capture_state(),{},unused,fingerprint});
}
class RecoveryCallback final:public ceres::IterationCallback {
 JointALBuilder&b_;std::ofstream stream_;std::filesystem::path out_;std::function<bool()>recalled_;
 public:JointCompactAcceptedState accepted;std::string error;bool stopped=false;std::size_t outer_index=0,global_iteration=0;bool numerical_navigation_active=false,write_physical_preview=false;
 void begin_outer(std::size_t index){outer_index=index;accepted.capture();}
 RecoveryCallback(JointALBuilder&b,const std::filesystem::path&out,std::function<bool()> recalled):b_(b),stream_(out/"iterations.jsonl"),out_(out),recalled_(std::move(recalled)),accepted(b){}
 ceres::CallbackReturnType operator()(const ceres::IterationSummary&s)override{try{
  if(s.step_is_successful){accepted.capture();if(write_physical_preview){std::filesystem::create_directories(out_/"working_physical");eventio::atomic_json(out_/"working_physical/model.json",b_.core.model.exact_json());eventio::atomic_json(out_/"working_physical/receipt.json",{{"state_kind","physical_working_candidate"},{"global_iteration",global_iteration},{"original_screen_grid_overwritten",false},{"numerical_navigation_active",numerical_navigation_active},{"complete_checkpoint",false},{"source_accuracy_accepted",false}});}}
  stream_<<eventio::Json({{"numerical_grid_navigation_active",numerical_navigation_active},{"outer_iteration",outer_index},{"global_iteration",global_iteration++},{"iteration",s.iteration},{"accepted",s.step_is_successful},{"cost",s.cost},{"cost_change",s.cost_change},{"gradient_max_norm",s.gradient_max_norm},{"trust_region_radius",s.trust_region_radius},{"iteration_seconds",s.iteration_time_in_seconds},{"linear_seconds",s.step_solver_time_in_seconds},{"cumulative_seconds",s.cumulative_time_in_seconds},{"compact_primal_token",accepted.token()}}).dump()<<std::endl;
  joint_memory_marker(out_,"Ceres_iteration",{{"outer_iteration",outer_index},{"iteration",s.iteration},{"accepted",s.step_is_successful},{"cost",s.cost},{"iteration_seconds",s.iteration_time_in_seconds},{"linear_seconds",s.step_solver_time_in_seconds}});
  if(recalled_()||std::filesystem::exists(out_/"STOP_REQUESTED")){stopped=true;return ceres::SOLVER_TERMINATE_SUCCESSFULLY;}
  return ceres::SOLVER_CONTINUE;
 }catch(const std::exception&e){error=e.what();return ceres::SOLVER_ABORT;}}
};

eventio::Json native_interior_pixel_derivative_probe(JointALBuilder&b){
 std::size_t selected=b.core.packet.rows.size();double measured=INFINITY;
 for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active||r.kind!=1||r.query)continue;const auto l=r.layout(b.core.model);const auto ptr=b.core.model.pointers(l);const auto v=b.core.model.predict_for_al(l,b.core.station[i],b.core.row[i],ptr.data());
  if(!v.arithmetic_valid)continue;const double e=std::hypot(v.uv[0]-r.uv[0],v.uv[1]-r.uv[1]);if(e>1e-6&&e<r.E){selected=i;measured=e;break;}
 }
 if(selected==b.core.packet.rows.size())return {{"status","no_natural_E_interior_fixed_corner"},{"passed",false}};
 const auto&r=b.core.packet.rows[selected];JointExplicitRow q;q.model=&b.core.model;q.source=&r;q.layout=r.layout(b.core.model);
 std::vector<std::vector<double>>storage;std::vector<int>sizes;for(auto id:q.layout.block_ids){storage.push_back(b.core.model.blocks[id].values);sizes.push_back(int(storage.back().size()));}
 q.row_slot=int(storage.size());storage.push_back({b.core.row[selected]});sizes.push_back(1);
 std::vector<const double*>ptr;for(const auto&v:storage)ptr.push_back(v.data());
 auto cost=event_autodiff_cost(JointEngineeringPixel{q,1./r.E},sizes,2);std::vector<std::vector<double>>jac;std::vector<double*>jp;for(auto n:sizes){jac.emplace_back(2*n);jp.push_back(jac.back().data());}
 double residual[2];if(!cost->Evaluate(ptr.data(),residual,jp.data()))throw std::runtime_error("interior native AD probe failed");
 const int slot=q.layout.reference_angles_slot;const double analytic=residual[0]*jac[slot][0]+residual[1]*jac[slot][sizes[slot]];
 const double original=storage[slot][0],h=1e-7;auto halfcost=[&](double x){storage[slot][0]=x;double v[2];if(!cost->Evaluate(ptr.data(),v,nullptr))throw std::runtime_error("interior native finite difference failed");return .5*(v[0]*v[0]+v[1]*v[1]);};
 const double finite=(halfcost(original+h)-halfcost(original-h))/(2*h);storage[slot][0]=original;
 const double relative=std::abs(analytic-finite)/std::max(1.,std::abs(finite));
 return {{"status","actual_qualified_native_corner_inside_original_E"},{"source_row_index",selected},{"source_dataset_id",r.source_dataset_id},{"source_row_id",r.source_row_id},{"native_pts",r.pts},{"error_px",measured},{"original_E",r.E},{"native_pixel_half_cost",.5*(residual[0]*residual[0]+residual[1]*residual[1])},{"native_camera_yaw_gradient",analytic},{"central_difference",finite},{"relative_difference",relative},{"passed",std::isfinite(relative)&&relative<1e-5&&std::abs(analytic)>1e-12},{"original_UV_or_model_mutated",false}};
}

void apply_native_station_units(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg,const std::filesystem::path&out){
 if(cfg.value("native_station_step_units",true)){
  const auto initial_token=b.capture_state().token();
  auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback());if(cache)cache->PrepareForEvaluation(true,true);
  std::size_t scaled=0,CPU=0,invalid=0;double minimum=INFINITY,maximum=0;
  for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];double*st=&b.core.station[i];if(!r.active||r.kind==1||!problem.HasParameterBlock(st)||problem.IsParameterBlockConstant(st))continue;
   double du=NAN,dv=NAN;JointUVCacheView view;if(cache&&cache->lookup(i,&view)&&view.jacobians_ready&&view.partials&&view.station_column>=0){du=view.partials[view.station_column].u;dv=view.partials[view.station_column].v;}
   else{++CPU;using J=ceres::Jet<double,1>;const auto l=r.layout(b.core.model);std::vector<std::vector<J>>storage;for(int id:l.block_ids){storage.emplace_back();for(double x:b.core.model.blocks[id].values)storage.back().emplace_back(x);}std::vector<const J*>ptr;for(const auto&v:storage)ptr.push_back(v.data());J station(*st);station.v[0]=1;const auto q=b.core.model.predict_for_al(l,station,J(b.core.row[i]),ptr.data());if(q.arithmetic_valid){du=q.uv[0].v[0];dv=q.uv[1].v[0];}}
   const double speed=std::hypot(du,dv);if(!std::isfinite(speed)){++invalid;continue;}problem.SetManifold(st,new joint_seed_detail::JointScalarStepUnits(1./std::max(1.,speed)));++scaled;minimum=std::min(minimum,speed);maximum=std::max(maximum,speed);
  }
  if(b.capture_state().token()!=initial_token)throw std::runtime_error("numerical station units changed physical/auxiliary state");
  eventio::atomic_json(out/"native_station_units.json",{{"scaled_stations",scaled},{"CPU_derivative_fallback",CPU},{"invalid_derivative_rows",invalid},{"speed_min_px",minimum},{"speed_max_px",maximum},{"ambient_values_or_original_bounds_changed",false}});
 }
}
std::size_t apply_recovery_stage(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg){
 const auto stage=cfg.value("parameter_stage",std::string("motion"));
 if(stage!="motion"&&stage!="shared_geometry_rig"&&stage!="optics")throw std::runtime_error("unknown recovery parameter stage");
 std::size_t fixed=0;for(auto&block:b.core.model.blocks){bool free=block.name.rfind("shot/",0)==0;
  if(stage=="shared_geometry_rig")free=block.name.find("/optics")==std::string::npos;
  if(stage=="optics")free=true;
  for(const auto&prefix:cfg.value("temporary_runtime_fixed_physical_prefixes",std::vector<std::string>{}))if(block.name.rfind(prefix,0)==0)free=false;
  if(!free){problem.SetParameterBlockConstant(block.values.data());++fixed;}
 }return fixed;
}
eventio::Json engineering_constraint_debt(const ConstraintValues&v){
 double gmax=0,hmax=0;long double gsum=0,hsum=0;std::size_t violated=0;
 for(double x:v.g){if(!std::isfinite(x))throw std::runtime_error("nonfinite original engineering g");const double z=std::max(0.,x);gmax=std::max(gmax,z);gsum+=z*z;if(z>1e-8)++violated;}
 for(double x:v.h){if(!std::isfinite(x))throw std::runtime_error("nonfinite original engineering h");hmax=std::max(hmax,std::abs(x));hsum+=x*x;}
 return {{"original_g_count",v.g.size()},{"original_h_count",v.h.size()},{"g_max_positive",gmax},{"h_max_abs",hmax},{"g_positive_l2",std::sqrt(double(gsum))},{"h_l2",std::sqrt(double(hsum))},{"g_over_1e_minus8",violated},{"state_token",v.state_token}};
}
double contribution(ceres::Problem&p,const std::vector<void*>&ids){ceres::Problem::EvaluateOptions o;for(void*i:ids)o.residual_blocks.push_back(static_cast<ceres::ResidualBlockId>(i));if(ids.empty())return 0;double c=NAN;if(!p.Evaluate(o,&c,nullptr,nullptr,nullptr)||!std::isfinite(c))throw std::runtime_error("engineering contribution evaluation failed");return c;}
}
int run_joint_engineering_recovery(const std::string&mp,const std::string&pp,const std::string&op,const std::string&config){
 const auto cfg=config.empty()?eventio::Json::object():eventio::read_json(config);
 if(cfg.contains("resume_checkpoint"))throw std::runtime_error("engineering entry rejects PHR checkpoint; use resume_engineering_checkpoint with matching objective identity");
 if(std::filesystem::exists(op))throw std::runtime_error("immutable recovery output exists");
 joint_memory_marker(op,"before_model_load");auto model=JointEventModel::load(mp);joint_memory_marker(op,"before_packet_load");auto packet=JointPacket::load(pp,model);joint_memory_marker(op,"packet_loaded",{{"raw_array_cache_bytes",joint_packet_raw_cache_bytes()},{"typed_rows",packet.rows.size()},{"typed_groups",packet.groups.size()}});
 assert_joint_board_source_qualification(packet,pp);
 auto guard=joint_source_guard_snapshot(packet.source_json);
 if(packet.source_json.contains("accepted_explicit_state_binding"))guard["accepted_explicit_state_binding"]=packet.source_json.at("accepted_explicit_state_binding");
 const auto released_raw=joint_packet_raw_cache_bytes();release_joint_packet_raw_cache();packet.source_json=guard;packet.source_json["schema"]=packet.schema;
 joint_memory_marker(op,"source_verified_and_runtime_compacted",{{"released_raw_array_cache_bytes",released_raw},{"full_source_document_retained_on_disk",pp},{"typed_rows_queries_groups_W_retained",true}});
 auto recalled=[&](){return !joint_source_recall_marker(guard,pp,true).empty();};
 if(recalled())throw std::runtime_error("source qualification recalled");
 const std::filesystem::path out=op;
 // Stage, initialization and execution settings do not alter the frozen mathematical objective.
 const eventio::Json objective_cfg={{"objective_id","engineering_recovery_native_pixel_v1"},{"physical_constraint_penalty",cfg.value("physical_constraint_penalty",10.)},{"pixel_total_weight",cfg.value("pixel_total_weight",double(std::count_if(packet.rows.begin(),packet.rows.end(),[](const auto&r){return r.active;})))}};const auto config_text=objective_cfg.dump();const auto config_identity=eventio::sha256(std::vector<std::uint8_t>(config_text.begin(),config_text.end()));
 JointALBuilder b(std::move(model),std::move(packet),false,false);b.set_engineering_identity(config_identity);
 if(cfg.contains("resume_engineering_checkpoint")){std::ifstream stream(cfg.at("resume_engineering_checkpoint").get<std::string>());auto state=read_checkpoint(stream);if(!state.outer.lambda.empty()||!state.outer.nu.empty())throw std::runtime_error("engineering checkpoint contains old AL multipliers");b.restore_state(state.state);}
 const int threads=cfg.value("num_threads",8);
 joint_memory_marker(out,"before_auxiliary_initialization");
 if(cfg.value("reinitialize_native_auxiliaries",true)){if(cfg.contains("resume_engineering_checkpoint"))throw std::runtime_error("cannot reinitialize an engineering checkpoint");
  JointBaselineAuxiliaryPreservation preserved;
  if(b.core.packet.source_json.contains("accepted_explicit_state_binding")){auto inherited=preserve_joint_accepted_auxiliary_prefix(b.core,mp,pp);preserved.preserved=std::move(inherited.preserved);eventio::atomic_json(out/"accepted_auxiliary_prefix_restore.json",inherited.receipt); }
  if(cfg.contains("baseline_auxiliary_manifest")){if(!preserved.preserved.empty())throw std::runtime_error("cannot mix baseline import and accepted-prefix import");preserved=restore_joint_baseline_auxiliaries(b.core,cfg.at("baseline_auxiliary_manifest"),mp,pp);eventio::atomic_json(out/"baseline_auxiliary_restore.json",preserved.receipt);}
  if(cfg.contains("s01_component_rollback")){if(preserved.preserved.empty())throw std::runtime_error("component rollback requires verified accepted prefix");auto rollback=rollback_joint_s01_component(b.core,cfg.at("s01_component_rollback"),mp,pp);for(std::size_t i=0;i<rollback.preserved.size();++i)if(rollback.preserved[i])preserved.preserved.at(i)=1;eventio::atomic_json(out/"s01_component_rollback.json",rollback.receipt);}
  initialize_recovery_auxiliaries(b.core,out/"fresh_auxiliary_initialization",threads,preserved.preserved);
 }
 joint_memory_marker(out,"after_auxiliary_initialization");save_recovery_state(b,out/"initial_state",config_identity);joint_memory_marker(out,"initial_checkpoint_written");

 if(cfg.value("initialization_only",false)){
  const auto metrics=export_joint_recovery_explicit_metrics(b,out/"initial_state/CP_explicit_metrics",threads);
  JointReplayArtifactMemo memo;const auto replay=export_joint_replay_artifacts(b,b.capture_state(),out/"initial_independent_RS_state",&memo,false,true);
  eventio::atomic_json(out/"initialization_only_receipt.json",{{"schema","rinkfit.recovery_initialization_preflight.v1"},{"objective_id",b.identity().objective_id},{"source_packet_sha256",b.core.packet.source_hash},{"configuration_sha256",config_identity},{"explicit_metrics",metrics},{"independent_RS_replay",replay},{"solver_started",false},{"initial_checkpoint",(out/"initial_state/engineering_checkpoint.json").string()}});
  return 0;
 }
 if(cfg.contains("gpu_raw_uv")){const auto&gpu=cfg.at("gpu_raw_uv");eventio::verify(gpu.at("ptx").get<std::string>(),gpu.at("ptx_sha256"));const auto&shadow=gpu.at("shadow_receipt");eventio::verify(shadow.at("file").get<std::string>(),shadow.at("sha256"));if(!eventio::read_json(shadow.at("file").get<std::string>()).value("passed",false))throw std::runtime_error("GPU native Jacobian shadow failed");b.enable_gpu_raw_uv(gpu.at("ptx"),threads,out/"gpu_raw_uv_cache.jsonl");}
 const auto stage=cfg.value("parameter_stage",std::string("motion"));
 auto outer=fresh_engineering_outer(b,cfg);
 if(cfg.contains("resume_engineering_outer_state")){
  if(!cfg.contains("resume_engineering_checkpoint"))throw std::runtime_error("engineering outer state requires its complete checkpoint");
  outer=load_engineering_outer(b,cfg.at("resume_engineering_outer_state"),cfg.at("resume_engineering_checkpoint").get<std::string>(),config_identity);
 }
 const double gtol=cfg.value("engineering_primal_tolerance",1e-8),htol=cfg.value("engineering_equality_tolerance",1e-8);
 if(!(gtol>0&&gtol<=1e-8&&htol>0&&htol<=1e-8))throw std::runtime_error("engineering original constraints cannot be relaxed");
 const double rho_max=cfg.value("engineering_rho_max",RuntimeLimits{}.rho_max);
 const double ratio=cfg.value("engineering_improvement_ratio",RuntimeLimits{}.improvement_ratio),factor=cfg.value("engineering_penalty_factor",RuntimeLimits{}.penalty_factor);
 const auto derivative_probe=native_interior_pixel_derivative_probe(b);eventio::atomic_json(out/"interior_pixel_derivative_probe.json",derivative_probe);
 if(!derivative_probe.value("passed",false))throw std::runtime_error("direct engineering interior native pixel gradient not verified");
 const auto initial_token=b.capture_state().token();
 const auto initial_explicit_metrics=export_joint_recovery_explicit_metrics(b,out/"initial_state/CP_explicit_metrics",threads);
 save_engineering_outer(b,outer,out/"initial_state/engineering_checkpoint.json",out/"initial_state/engineering_outer.json",config_identity,stage,"ready_to_solve");
 RecoveryCallback callback(b,out,recalled);callback.write_physical_preview=cfg.value("write_accepted_physical_preview",false);
 joint_memory_marker(out,"compact_callback_ready",{{"accepted_value_bytes",callback.accepted.value_capacity_bytes()},{"registry_blocks",callback.accepted.registry_blocks()}});
 ceres::Solver::Options options;
 const auto backend=cfg.value("linear_solver_backend",std::string("cuda_cgnr"));const bool cuda=backend=="cuda_cgnr";
 if(!cuda&&backend!="cpu_suite_sparse")throw std::runtime_error("unsupported engineering linear backend");
 if(!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(cuda?ceres::CUDA_SPARSE:ceres::SUITE_SPARSE))throw std::runtime_error("explicit requested sparse linear backend unavailable");
 options.linear_solver_type=cuda?ceres::CGNR:ceres::SPARSE_NORMAL_CHOLESKY;options.sparse_linear_algebra_library_type=cuda?ceres::CUDA_SPARSE:ceres::SUITE_SPARSE;options.preconditioner_type=ceres::JACOBI;
 options.initial_trust_region_radius=cfg.value("initial_trust_region_radius",100.);options.min_lm_diagonal=cfg.value("min_lm_diagonal",1e-6);options.eta=cfg.value("inexact_linear_step_eta",.1);
 options.max_num_iterations=cfg.value("max_inner_iterations",50);options.num_threads=threads;options.update_state_every_iteration=true;options.callbacks.push_back(&callback);options.minimizer_progress_to_stdout=true;options.function_tolerance=cfg.value("function_tolerance",1e-8);options.parameter_tolerance=1e-10;options.gradient_tolerance=1e-10;options.use_mixed_precision_solves=false;
 bool motion_navigation_done=false,query_navigation_done=false,coupled_motion_navigation_done=false,finite_line_navigation_done=false,source_camera_mode_done=false;
 bool screen_grid_navigation_done=false,last_summary_included_grid_navigation=false;
 bool first=true,closed=false,raw_closed=false,independent_closed=false,failed=false,limited=false;std::size_t fixed=0,iterations=0,completed_outers=0;
 double residual_seconds=0,jacobian_seconds=0,linear_seconds=0,total_seconds=0;
 eventio::Json initial_costs,final_costs,initial_debt,final_debt;ceres::Solver::Summary summary;
 std::ofstream outer_log(out/"outer_iterations.jsonl");
 for(;;){
  // Destroy each complete old graph before allocating the next one. The typed
  // immutable source and native GPU cache remain shared by this one builder.
  const auto round_dir=out/("outer_"+std::to_string(outer.outer_iteration));std::filesystem::create_directories(round_dir);
  eventio::Json inner_cfg=cfg;inner_cfg["physical_constraint_penalty"]=outer.rho;
  ceres::Problem::Options po;po.evaluation_callback=b.gpu_evaluation_callback();auto problem=std::make_unique<ceres::Problem>(po);
  eventio::Json ledger;
  joint_memory_marker(out,"before_graph_build",{{"outer_iteration",outer.outer_iteration},{"rho",outer.rho}});
  auto inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg);
  joint_memory_marker(out,"after_graph_build",{{"outer_iteration",outer.outer_iteration},{"pixel_factors",inventory.statistical_blocks.size()},{"physical_factors",inventory.inequality_blocks.size()},{"RS_Query_factors",inventory.equality_blocks.size()}});
  auto costs=[&](){return eventio::Json({{"pixel_half_cost",contribution(*problem,inventory.statistical_blocks)},{"physical_half_cost",contribution(*problem,inventory.inequality_blocks)},{"RS_Query_half_cost",contribution(*problem,inventory.equality_blocks)},{"rho",outer.rho},{"outer_iteration",outer.outer_iteration}});};
  if(first){
   initial_costs=costs();eventio::atomic_json(out/"initial_contributions.json",initial_costs);eventio::atomic_json(out/"pixel_groups.json",ledger);
   initial_debt=engineering_constraint_debt(b.evaluate_current_engineering_constraints());eventio::atomic_json(out/"initial_original_constraint_debt.json",initial_debt);
   ceres::Problem::EvaluateOptions probe;for(void*i:inventory.statistical_blocks)probe.residual_blocks.push_back(static_cast<ceres::ResidualBlockId>(i));
   joint_memory_marker(out,"before_pixel_linearization");double pixel_cost=0;std::vector<double>gradient;
   if(!problem->Evaluate(probe,&pixel_cost,nullptr,&gradient,nullptr))throw std::runtime_error("actual native pixel gradient evaluation failed");
   double norm=0,maxnorm=0;for(double x:gradient){norm+=x*x;maxnorm=std::max(maxnorm,std::abs(x));}
   eventio::atomic_json(out/"pixel_gradient_probe.json",{{"objective_id",b.identity().objective_id},{"half_cost",pixel_cost},{"gradient_l2",std::sqrt(norm)},{"gradient_max_abs",maxnorm},{"tangent_scalars",gradient.size()},{"old_budget_or_full_W_consumed",false}});
   JointReplayArtifactMemo initial_memo;export_joint_replay_artifacts(b,b.capture_state(),out/"initial_independent_RS_state",&initial_memo,false,true);
   joint_memory_marker(out,"after_initial_independent_replay");first=false;
  }
  if(!source_camera_mode_done&&cfg.contains("source_camera_mode_navigation")){
   const auto restart_dir=round_dir/"source_camera_mode_navigation";
   auto destroy_graph=[&](){problem.reset();inventory=BuildInventory{};};
   auto rebuild_graph=[&](const std::filesystem::path&log){
    if(problem)throw std::runtime_error("source camera mode graph was not destroyed");
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,log);
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg);
   };
   const auto restart=refine_joint_source_camera_mode_navigation(b,problem,cfg.at("source_camera_mode_navigation"),inner_cfg,outer.nu,options,restart_dir,threads,destroy_graph,rebuild_graph,
    [&](const std::vector<std::uint8_t>&mask,const std::filesystem::path&dir){return initialize_recovery_auxiliaries(b.core,dir,threads,mask);},
    [&](){return engineering_constraint_debt(b.evaluate_current_engineering_constraints());});
   source_camera_mode_done=true;
   if(restart.value("accepted",false)){save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");}
  }
  if(!motion_navigation_done&&cfg.contains("motion_navigation_restart")){
   const auto restart_dir=round_dir/"motion_navigation_restart";
   const auto restart=refine_joint_motion_navigation(b,*problem,cfg.at("motion_navigation_restart"),restart_dir,threads,[&](){return engineering_constraint_debt(b.evaluate_current_engineering_constraints());});
   motion_navigation_done=true;
   if(restart.value("accepted",false)){
    save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");
   }
  }
  if(!finite_line_navigation_done&&cfg.contains("finite_line_navigation_restart")){
   const auto restart_dir=round_dir/"finite_line_navigation_restart";
   const auto restart=refine_joint_finite_line_navigation(b,*problem,cfg.at("finite_line_navigation_restart"),restart_dir,threads,[&](){return engineering_constraint_debt(b.evaluate_current_engineering_constraints());});
   finite_line_navigation_done=true;
   if(restart.value("accepted",false)){save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");}
  }
  if(!coupled_motion_navigation_done&&cfg.contains("coupled_motion_navigation_restart")){
   const auto restart_dir=round_dir/"coupled_motion_navigation_restart";
   auto destroy_graph=[&](){problem.reset();inventory=BuildInventory{};};
   auto rebuild_graph=[&](const std::filesystem::path&log){
    if(problem)throw std::runtime_error("coupled motion graph was not destroyed");
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,log);
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg);
   };
   const auto restart=refine_joint_coupled_motion_navigation(b,problem,cfg.at("coupled_motion_navigation_restart"),restart_dir,threads,destroy_graph,rebuild_graph,
    [&](const std::vector<std::uint8_t>&mask,const std::filesystem::path&dir){return initialize_recovery_auxiliaries(b.core,dir,threads,mask);},
    [&](){return engineering_constraint_debt(b.evaluate_current_engineering_constraints());});
   coupled_motion_navigation_done=true;
   if(restart.value("accepted",false)){save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");}
  }
  if(!query_navigation_done&&cfg.value("refine_query_navigation_seeds",false)){
   const auto restart_dir=round_dir/"query_navigation_restart";
   auto plan=propose_joint_query_restart(b,inner_cfg,outer.nu,restart_dir,threads);
   query_navigation_done=true;
   auto complete_cost=[&](){
    if(auto*cache=dynamic_cast<JointUVCache*>(b.gpu_evaluation_callback()))cache->PrepareForEvaluation(false,true);
    ceres::Problem::EvaluateOptions eo;eo.num_threads=threads;double value=NAN;
    if(!problem->Evaluate(eo,&value,nullptr,nullptr,nullptr)||!std::isfinite(value))throw std::runtime_error("Query navigation full current objective failed");
    return value;
   };
   const auto debt_before=engineering_constraint_debt(b.evaluate_current_engineering_constraints());
   const double cost_before=complete_cost();double candidate_cost=cost_before;auto candidate_debt=debt_before;bool accepted=false;
   const auto residual_count=problem->NumResiduals();
   auto rebuild=[&](const char*log_name){
    // No old graph can retain a callback or layout while paths/cache change.
    problem.reset();inventory=BuildInventory{};
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,restart_dir/log_name);
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg);
    if(problem->NumResiduals()!=residual_count)throw std::runtime_error("Query navigation changed full residual denominator");
   };
   if(!plan.rows.empty()){
    problem.reset();inventory=BuildInventory{};
    apply_joint_query_restart_plan(b,plan,true);rebuild("candidate_gpu_raw_uv_cache.jsonl");
    candidate_cost=complete_cost();candidate_debt=engineering_constraint_debt(b.evaluate_current_engineering_constraints());
    accepted=candidate_cost<cost_before;
    if(!accepted){problem.reset();inventory=BuildInventory{};rollback_joint_query_restart_plan(b,plan,true);rebuild("rollback_gpu_raw_uv_cache.jsonl");
     const double restored_cost=complete_cost();
     if(std::abs(restored_cost-cost_before)>1e-8*std::max(1.,std::abs(cost_before)))throw std::runtime_error("Query navigation rollback did not restore objective");
    }
   }
   write_joint_query_restart_result(b,plan,accepted,cost_before,candidate_cost,debt_before,candidate_debt,restart_dir);
   if(accepted){save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");}
  }
  if(cfg.value("refine_ordinary_station_seeds",true)){
   const auto restart_dir=round_dir/"ordinary_station_restart";
   const auto restart=refine_joint_ordinary_stations(b,*problem,restart_dir,threads,[&](){return engineering_constraint_debt(b.evaluate_current_engineering_constraints());});
   if(restart.value("accepted",false)){
    save_recovery_state(b,restart_dir/"accepted_state",config_identity);
    save_engineering_outer(b,outer,restart_dir/"accepted_state/engineering_checkpoint.json",restart_dir/"accepted_state/engineering_outer.json",config_identity,stage,"ready_to_solve");
   }
  }
  apply_native_station_units(b,*problem,cfg,round_dir);
  JointScreenGridNavigation screen_navigation;
  const bool used_screen_navigation=!screen_grid_navigation_done&&cfg.contains("direct_screen_grid_navigation_spec");
  if(used_screen_navigation){
   const auto spec=eventio::read_json(cfg.at("direct_screen_grid_navigation_spec").get<std::string>());
   screen_navigation=add_joint_screen_grid_numerical_navigation(b,*problem,spec);
   screen_navigation.receipt["original_objective_before"]=costs();
   screen_navigation.receipt["navigation_half_cost_before"]=screen_navigation.half_cost(*problem);
   eventio::atomic_json(round_dir/"direct_grid_navigation_started.json",screen_navigation.receipt);
  }
  callback.numerical_navigation_active=used_screen_navigation;last_summary_included_grid_navigation=used_screen_navigation;
  callback.begin_outer(outer.outer_iteration);joint_memory_marker(out,"before_Ceres_solve",{{"outer_iteration",outer.outer_iteration},{"rho",outer.rho}});
  ceres::Solve(options,problem.get(),&summary);callback.accepted.restore(b);
  if(used_screen_navigation){
   screen_navigation.receipt["navigation_half_cost_after"]=screen_navigation.half_cost(*problem);
   screen_navigation.remove_from(*problem);screen_grid_navigation_done=true;
   screen_navigation.receipt["original_objective_after_removing_navigation"]=costs();
   screen_navigation.receipt["optimizer_summary_cost_includes_navigation"]=true;
   screen_navigation.receipt["screen_initial_grid_modified"]=false;
   eventio::atomic_json(round_dir/"direct_grid_navigation_completed.json",screen_navigation.receipt);
  }
  joint_memory_marker(out,"after_Ceres_solve",{{"outer_iteration",outer.outer_iteration}});
  iterations+=summary.iterations.size();residual_seconds+=summary.residual_evaluation_time_in_seconds;jacobian_seconds+=summary.jacobian_evaluation_time_in_seconds;linear_seconds+=summary.linear_solver_time_in_seconds;total_seconds+=summary.total_time_in_seconds;
  final_costs=costs();auto constraints=b.evaluate_current_engineering_constraints();final_debt=engineering_constraint_debt(constraints);
  failed=summary.termination_type==ceres::FAILURE||!callback.error.empty();const bool stop=callback.stopped||recalled()||(used_screen_navigation&&cfg.value("navigation_stage_only",false));
  raw_closed=!used_screen_navigation&&final_debt.at("g_max_positive").get<double>()<=gtol&&final_debt.at("h_max_abs").get<double>()<=htol&&summary.termination_type==ceres::CONVERGENCE;
  independent_closed=false;
  if(raw_closed){
   JointReplayArtifactMemo closure_memo;const auto proof=export_joint_replay_artifacts(b,b.capture_state(),round_dir/"independent_RS_closure",&closure_memo,false,true);
   const auto&counts=proof.at("projection_classification_counts");const auto n=proof.at("active_rows").get<std::size_t>();
   independent_closed=counts.at("fixed_current_station_local_native_RS_root_valid").get<std::size_t>()==n&&counts.at("local_RS_root_and_physical_visibility_valid").get<std::size_t>()==n&&counts.at("local_RS_root_and_source_station_domain_valid").get<std::size_t>()==n&&counts.at("explicit_current_aux_ray_domain_valid").get<std::size_t>()==n;
  }
  closed=raw_closed&&independent_closed;
  eventio::Json record={{"outer_iteration",outer.outer_iteration},{"rho",outer.rho},{"pixel_and_augmented_contributions",final_costs},{"original_constraint_debt",final_debt},{"inner_termination",summary.BriefReport()},{"inner_converged",summary.termination_type==ceres::CONVERGENCE},{"stopped",stop},{"failed",failed},{"raw_constraints_closed",raw_closed},{"independent_fixed_station_RS_domain_closed",independent_closed},{"augmented_cost_comparable_across_outer_contexts",false}};
  // No iteration cap terminates this recovery task. An exhausted Ceres inner
  // batch continues this same original objective in the next AL context.
  outer.previous_inner_incomplete=failed||stop;
  if(!failed&&!stop){
   auto next=update_phr(constraints.g,constraints.h,outer.lambda,outer.nu,outer.rho,outer.previous_progress,rho_max,ratio,factor);
   outer.lambda=std::move(next.lambda);outer.nu=std::move(next.nu);outer.previous_progress=next.progress;outer.rho=next.next_rho;++outer.outer_iteration;++completed_outers;
   limited=next.penalty_limit_reached&&!closed;record["next_rho"]=outer.rho;record["next_original_constraint_progress"]=outer.previous_progress;record["penalty_conditioning_limit_reached"]=limited;record["dual_update_applied"]=true;
  }else record["dual_update_applied"]=false;
  outer_log<<record.dump()<<std::endl;eventio::atomic_json(round_dir/"receipt.json",record);
  // Residual IDs and manifold ownership end here, before checkpoint/rebuild.
  problem.reset();inventory=BuildInventory{};
  joint_memory_marker(out,"before_outer_checkpoint",{{"next_outer_iteration",outer.outer_iteration}});
  save_recovery_state(b,round_dir/"state",config_identity);
  save_engineering_outer(b,outer,round_dir/"state/engineering_checkpoint.json",round_dir/"state/engineering_outer.json",config_identity,stage,"ready_to_solve");
  eventio::atomic_json(out/"current_outer_state.json",{{"schema","rinkfit.engineering_outer_pointer.v1"},{"outer_iteration",outer.outer_iteration},{"incomplete_inner",failed||stop},{"checkpoint",{{"file",(round_dir/"state/engineering_checkpoint.json").string()},{"sha256",eventio::file_sha256(round_dir/"state/engineering_checkpoint.json")}}},{"engineering_outer",{{"file",(round_dir/"state/engineering_outer.json").string()},{"sha256",eventio::file_sha256(round_dir/"state/engineering_outer.json")}}}});
  joint_memory_marker(out,"after_outer_checkpoint");
  if(closed||failed||stop||limited)break;
 }
 joint_memory_marker(out,"before_candidate_checkpoint");save_recovery_state(b,out/"candidate",config_identity);
 save_engineering_outer(b,outer,out/"candidate/engineering_checkpoint.json",out/"candidate/engineering_outer.json",config_identity,stage,"ready_to_solve");joint_memory_marker(out,"after_candidate_checkpoint");
 const auto candidate_explicit_metrics=export_joint_recovery_explicit_metrics(b,out/"candidate/CP_explicit_metrics",threads);
 JointReplayArtifactMemo memo;const auto replay=export_joint_replay_artifacts(b,b.capture_state(),out/"independent_RS_state",&memo,false,true);
 const auto token=b.capture_state().token();eventio::atomic_json(out/"independent_RS_state/engineering_checkpoint_receipt.json",{{"objective_id",b.identity().objective_id},{"source_packet_sha256",b.core.packet.source_hash},{"state_token",token},{"formal_acceptance_claimed",false}});
 eventio::Json receipt={{"schema","rinkfit.joint_engineering_recovery_run.v1"},{"last_solver_summary_included_numerical_grid_navigation",last_summary_included_grid_navigation},{"original_objective_contributions_exclude_numerical_grid_navigation",true},{"objective_id",b.identity().objective_id},{"builder","JointALBuilder::build_engineering_problem"},{"original_constraint_enforcement","engineering_PHR_outer_v1"},{"common_worst_budget_consumed",false},{"original_W_likelihood_consumed",false},{"source_packet",pp},{"source_packet_sha256",b.core.packet.source_hash},{"input_model_sha256",eventio::file_sha256(mp)},{"configuration_sha256",config_identity},{"model_sha256",eventio::file_sha256(out/"candidate/model.json")},{"initial_state_token",initial_token},{"candidate_state_token",token},{"initial_CP_explicit_metrics",initial_explicit_metrics},{"candidate_CP_explicit_metrics",candidate_explicit_metrics},{"initial_contributions",initial_costs},{"final_contributions",final_costs},{"initial_original_constraint_debt",initial_debt},{"final_original_constraint_debt",final_debt},{"original_constraint_tolerances",{{"g",gtol},{"h",htol}}},{"raw_original_constraints_closed",raw_closed},{"independent_fixed_station_RS_domain_closed",independent_closed},{"parameter_stage",stage},{"temporary_fixed_physical_blocks",fixed},{"linear_backend",backend},{"GPU_raw_uv",b.gpu_report()},{"iterations",iterations},{"completed_engineering_outer_updates",completed_outers},{"termination",summary.BriefReport()},{"callback_error",callback.error},{"stopped",callback.stopped},{"penalty_conditioning_limit_reached",limited},{"source_recalled",recalled()},{"independent_RS_replay",replay},{"candidate_publication_state","candidate_pending_same_source_comparison"},{"formal_Phase_II_KKT_passed",false},{"metric_scale_certified",false}};
 receipt["timing_seconds"]={{"residual",residual_seconds},{"Jacobian",jacobian_seconds},{"linear",linear_seconds},{"total",total_seconds}};
 eventio::atomic_json(out/"engineering_receipt.json",receipt);std::cout<<receipt.dump(2)<<std::endl;
 return failed||recalled()||limited?2:0;
}
}
