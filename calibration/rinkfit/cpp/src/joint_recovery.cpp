#include "rinkfit/joint_capture_station_cells_v20.hpp"
#include "rinkfit/joint_recovery.hpp"
#include "rinkfit/joint_finite_source_paths_v1.hpp"
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
#include "rinkfit/joint_aux_prefix_current_v5.hpp"
#include "rinkfit/joint_component_rollback.hpp"
#include "rinkfit/joint_engineering_outer.hpp"
#include "rinkfit/joint_al.hpp"
#include "rinkfit/joint_source_enriched_guard_v2.hpp"
#include "rinkfit/joint_replay_artifacts.hpp"
#include "rinkfit/joint_source_math_receipt.hpp"
#include "rinkfit/joint_seed_inner.hpp"
#include "rinkfit/joint_recovery_source_capture.hpp"
#include "rinkfit/joint_recovery_scalar_protection_v3.hpp"
#include "rinkfit/joint_recovery_metrics_export.hpp"
#include "rinkfit/joint_source_uv_capture_boundary_v1.hpp"
#include "rinkfit/joint_capture_certified_zoom_v1.hpp"
#include "rinkfit/joint_capture_primal_prefix_v1.hpp"
#include "rinkfit/joint_capture_primal_prefix_v5.hpp"
#include "rinkfit/joint_current_board_source_guard_v1.hpp"
#include "rinkfit/joint_printed_board_source_guard_v1.hpp"
#include "rinkfit/joint_ice_shape_source_guard_v1.hpp"
#include "rinkfit/joint_dense_source_guard_v1.hpp"
#include "rinkfit/joint_observed_numeric_overlay_v1.hpp"
#include "rinkfit/joint_segmented_numeric_overlay_v1.hpp"
#include "rinkfit/joint_capture_stage_guard_v1.hpp"
#include "rinkfit/joint_runtime_primitive_release_v1.hpp"
#include "rinkfit/joint_station_schur_v1.hpp"
#include "rinkfit/joint_cap168_runtime_v1.hpp"
#include <iostream>
#include <atomic>
#include <optional>
#include <thread>
namespace rinkfit {namespace {

eventio::Json initialize_recovery_auxiliaries(JointProblemBuilder&b,const std::filesystem::path&out,int threads,const std::vector<std::uint8_t>&preserved){
 std::vector<std::uint8_t>status(b.packet.rows.size(),0);std::atomic<std::size_t>next{0},valid{0},failed{0},recharted{0},no_finite{0},weak_visibility{0},weak_ray{0},preserved_count{0},source_paths_count{0},source_multi_paths_count{0};
 const auto work=[&](){for(;;){const auto i=next.fetch_add(1);if(i>=b.packet.rows.size())break;auto&r=b.packet.rows[i];if(!r.active)continue;
  if(!preserved.empty()&&preserved.at(i)){status[i]=3;++preserved_count;continue;}
  const double observed_row=r.uv[1];double station=r.station;int chosen_path=r.path;
  const auto&material=b.model.materials.at(r.material);
  const bool whole_board=(material.label=="board.top"||material.label=="board.contact")&&r.kind!=1&&r.source_station_intervals.empty();
  const bool source_paths=r.kind!=1&&joint_first4_source_path_policy_v1(b.packet)&&(b.packet.source_json.at("allowed_paths").contains(std::to_string(i))||(b.packet.source_json.contains("dense_source_binding")&&i>=b.packet.source_json.at("dense_source_binding").at("row_start").get<std::size_t>()));
  std::vector<int>paths=source_paths?joint_source_allowed_finite_paths_v1(b.packet,b.model,i):std::vector<int>{r.path};if(source_paths){++source_paths_count;if(paths.size()>1)++source_multi_paths_count;}
  const bool chart_search=whole_board||(source_paths&&paths.size()>1);
  if(whole_board&&!source_paths){paths.clear();for(std::size_t k=0;k<material.paths.size();++k){const auto&path=material.paths[k];if(path.kind!=3||path.physical_point)throw std::runtime_error("continuous board source has an incompatible finite chart");paths.push_back(int(k));}}
  double best=INFINITY;
  for(int seed_tier=0;seed_tier<(chart_search?3:1)&&!std::isfinite(best);++seed_tier){
  for(int path:paths){const auto l=b.model.layout(b.model.shot_map.at(r.shot_id),r.material,path,r.pts,r.width_block,r.width_sign);const auto ptr=b.model.pointers(l);
   auto error=[&](double st){const auto pred=b.model.predict_for_al(l,st,observed_row,ptr.data());
    if(!pred.arithmetic_valid||!finite_array(pred.uv))return double(INFINITY);
    if(chart_search&&seed_tier<2&&pred.physical_status!=DomainStatus::kValid)return double(INFINITY);
    if(chart_search&&seed_tier==0&&!b.model.visibility(l,st,observed_row,ptr.data()).visible())return double(INFINITY);
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
 eventio::Json receipt={{"schema","rinkfit.recovery_fresh_auxiliary_initialization.v3"},{"source_packet_sha256",b.packet.source_hash},{"raw_rows",b.packet.rows.size()},{"fresh_RS_valid_rows",valid.load()},{"fresh_RS_failed_rows",failed.load()},{"preserved_auxiliary_rows_not_RS_recertified",preserved_count.load()},{"no_arithmetic_finite_chart_rows",no_finite.load()},{"no_visible_chart_used_ray_finite_seed_rows",weak_visibility.load()},{"no_ray_valid_chart_used_arithmetic_seed_rows",weak_ray.load()},{"continuous_board_numerical_rechart_rows",recharted.load()},{"source_allowed_finite_path_rows",source_paths_count.load()},{"source_allowed_multi_path_rows",source_multi_paths_count.load()},{"source_bound_finite_path_domains_consumed",true},{"source_UV_ID_material_operator_unchanged",true},{"chart_policy","Scoped first4 rows enumerate only their original source-bound allowed finite paths and station intervals; fixed entities stay fixed. Legacy whole continuous board.top/contact navigation is retained. If no visible chart seed exists, closest ray-valid then arithmetic-finite seeds are retained with all physical failures still counted; a default unrelated chart is never treated as valid."},{"failed_RS_policy","retain active source and explicit observed-row numerical seed; independent failure remains counted"},{"status_file_sha256",eventio::file_sha256(out/"initialization_status.u8")}};
 eventio::atomic_json(out/"receipt.json",receipt);return receipt;
}

void save_recovery_state(JointALBuilder&b,const std::filesystem::path&out,const std::string&fingerprint,std::size_t preserve_prefix=0){
 std::filesystem::create_directories(out);
 const auto before_materialize=b.capture_state().token();eventio::Json capture_binding;if(b.source_uv_capture_v1())capture_binding=materialize_joint_source_uv_capture_v1(b.core,out/"derived_auxiliary_binding",b.source_uv_capture_v2(),preserve_prefix);
 if(b.source_uv_capture_v4()&&before_materialize!=b.capture_state().token())throw std::runtime_error("v4 save attempted to change actual graph primal during serialization");
 if(b.visible_ice_source_support()||b.source_query_branch_registry()){eventio::Json report={{"state_token",b.capture_state().token()}};add_joint_source_math_receipt(b,report);eventio::atomic_json(out/"source_math.json",report); }
 if(const auto source=b.visible_ice_source_support()){source->assert_dependencies();eventio::atomic_json(out/"source_visible_ice.json",{{"state_token",b.capture_state().token()},{"source_set",source->receipt()},{"current_physical_report",source->evaluate(b.core.model)},{"statistical_rank_added",0}});}
 eventio::atomic_json(out/"model.json",b.core.model.exact_json());
 b.export_explicit_nuisance(out/"explicit_nuisance");
 if(b.source_uv_observed_row_initialization())eventio::atomic_json(out/"observed_row_initialization_state_binding.json",{{"schema","rinkfit.provisional_measured_row_initialization_state_binding.v1"},{"state_token",b.capture_state().token()},{"source_packet_sha256",b.core.packet.source_hash},{"objective_id",b.identity().objective_id},{"true_RS_verified",false},{"stored_rows_are_measured_source_y_except_frozen_S01",true},{"initialization_only",true},{"fresh_trueRS_v3_transition_required",true},{"formal_Query_closed",false},{"phase_II_allowed",false}});
 if(b.source_uv_capture_v1())eventio::atomic_json(out/"source_uv_capture_state_binding.json",{{"schema","rinkfit.sourceUV_trueRS_capture_state_binding.v1"},{"state_token",b.capture_state().token()},{"model_sha256",eventio::file_sha256(out/"model.json")},{"packet_sha256",b.core.packet.source_hash},{"objective_id",b.identity().objective_id},{"stored_valid_rows_are_derived_true_RS_roots",true},{"failed_rows_retain_seed_and_status",true},{"actual_derived_row_status",capture_binding},{"row_alpha_optimization_dof",0},{"formal_Query_bounds_and_UV_retained",true},{"formal_Query_closed",false},{"formal_entry_requires_explicit_mode_transition",true}});
 OuterState unused;unused.phase=Phase::kPrecision;unused.rho=0;
 std::ofstream stream(out/"engineering_checkpoint.json");write_checkpoint(stream,{b.capture_state(),{},unused,fingerprint});stream.close();if(b.source_uv_capture_v4())write_capture_cell_binding_v20(b,out);
}
eventio::Json actual_joint_linear_solver_options_v1(const ceres::Solver::Options&options){
 return {{"schema","rinkfit.actual_joint_linear_solver_options.v1"},{"linear_solver_type",ceres::LinearSolverTypeToString(options.linear_solver_type)},{"sparse_linear_algebra_library_type",ceres::SparseLinearAlgebraLibraryTypeToString(options.sparse_linear_algebra_library_type)},{"preconditioner_type",ceres::PreconditionerTypeToString(options.preconditioner_type)},{"inexact_linear_step_eta",options.eta},{"min_linear_solver_iterations",options.min_linear_solver_iterations},{"max_linear_solver_iterations",options.max_linear_solver_iterations},{"num_threads",options.num_threads},{"max_num_iterations",options.max_num_iterations},{"mixed_precision",options.use_mixed_precision_solves},{"linear_solver_ordering_supplied",bool(options.linear_solver_ordering)},{"actual_library_available",ceres::IsSparseLinearAlgebraLibraryTypeAvailable(options.sparse_linear_algebra_library_type)},{"per_shot_descent_guaranteed",false},{"source_or_physical_values_modified",false}};
}
class RecoveryCallback final:public ceres::IterationCallback {
 JointALBuilder&b_;std::ofstream stream_;std::filesystem::path out_;std::function<bool()>recalled_;
 public:JointCompactAcceptedState accepted;std::string error;bool stopped=false;std::size_t outer_index=0,global_iteration=0;bool numerical_navigation_active=false,write_physical_preview=false,capture_sensor_boundary_enabled=false,capture_sensor_boundary_requested=false,capture_cell_boundary_enabled=false,capture_review_boundary_enabled=false,capture_review_boundary_requested=false,capture_review_completed=false;std::vector<std::vector<double>>capture_initial_physical;
 void begin_outer(std::size_t index){outer_index=index;accepted.capture();}
 RecoveryCallback(JointALBuilder&b,const std::filesystem::path&out,std::function<bool()> recalled):b_(b),stream_(out/"iterations.jsonl"),out_(out),recalled_(std::move(recalled)),accepted(b){for(const auto&block:b.core.model.blocks)capture_initial_physical.push_back(block.values);}
 ceres::CallbackReturnType operator()(const ceres::IterationSummary&s)override{try{
   b_.assert_source_math_dependencies();assert_joint_runtime_primitive_release_dependencies_v1();
  if(s.step_is_successful){accepted.capture();if(write_physical_preview){std::filesystem::create_directories(out_/"working_physical");eventio::atomic_json(out_/"working_physical/model.json",b_.core.model.exact_json());eventio::atomic_json(out_/"working_physical/receipt.json",{{"state_kind","physical_working_candidate"},{"model_sha256",eventio::file_sha256(out_/"working_physical/model.json")},{"global_iteration",global_iteration},{"original_screen_grid_overwritten",false},{"numerical_navigation_active",numerical_navigation_active},{"complete_checkpoint",false},{"source_accuracy_accepted",false}});}}
  stream_<<eventio::Json({{"numerical_grid_navigation_active",numerical_navigation_active},{"outer_iteration",outer_index},{"global_iteration",global_iteration++},{"iteration",s.iteration},{"accepted",s.step_is_successful},{"cost",s.cost},{"cost_change",s.cost_change},{"gradient_max_norm",s.gradient_max_norm},{"trust_region_radius",s.trust_region_radius},{"iteration_seconds",s.iteration_time_in_seconds},{"linear_seconds",s.step_solver_time_in_seconds},{"linear_solver_iterations",s.linear_solver_iterations},{"step_norm",s.step_norm},{"relative_decrease",s.relative_decrease},{"cumulative_seconds",s.cumulative_time_in_seconds},{"compact_primal_token",accepted.token()}}).dump()<<std::endl;
  joint_memory_marker(out_,"Ceres_iteration",{{"outer_iteration",outer_index},{"iteration",s.iteration},{"accepted",s.step_is_successful},{"cost",s.cost},{"iteration_seconds",s.iteration_time_in_seconds},{"linear_seconds",s.step_solver_time_in_seconds}});
  if(recalled_()||std::filesystem::exists(out_/"STOP_REQUESTED")){stopped=true;return ceres::SOLVER_TERMINATE_SUCCESSFULLY;}
  if(capture_review_boundary_enabled&&!capture_review_completed&&s.iteration>0&&s.step_is_successful){bool changed=false;for(std::size_t k=0;k<b_.core.model.blocks.size()&&!changed;++k)for(std::size_t j=0;j<b_.core.model.blocks[k].values.size();++j)if(b_.core.model.blocks[k].values[j]!=capture_initial_physical[k][j]){changed=true;break;}if(changed){capture_review_boundary_requested=true;eventio::atomic_json(out_/"first_accepted_review_request.json",{{"accepted_iteration",s.iteration},{"global_iteration",global_iteration-1},{"physical_model_changed",true},{"parameters_mutated_in_callback",false},{"complete_checkpoint_written_outside_Ceres",true}});return ceres::SOLVER_TERMINATE_SUCCESSFULLY;}}
  if(capture_cell_boundary_enabled&&s.iteration>0&&s.step_is_successful){capture_sensor_boundary_requested=true;return ceres::SOLVER_TERMINATE_SUCCESSFULLY;}
  if(capture_sensor_boundary_enabled&&s.iteration>0&&s.step_is_successful){std::size_t active=0;auto*cache=dynamic_cast<JointUVCache*>(b_.gpu_evaluation_callback());for(std::size_t i=0;i<b_.core.packet.rows.size();++i){const auto&r=b_.core.packet.rows[i];if(!r.active||r.kind==1)continue;JointUVCacheView view;if(cache&&cache->lookup(i,&view)){if(view.sensor_station_active)++active;continue;}auto src=r;src.query.reset();src.row=b_.core.row[i];JointRowEvaluation q;q.model=&b_.core.model;q.observation=&src;q.layout=src.layout(b_.core.model);for(std::size_t k=0;k<q.layout.block_ids.size();++k)q.parameter_sizes[int(k)]=int(b_.core.model.blocks[q.layout.block_ids[k]].values.size());auto p=b_.core.model.pointers(q.layout);auto sol=joint_capture_solution_v3(q,p.data(),b_.core.station[i],src.station);if(sol.valid&&sol.sensor_active)++active;}if(active){capture_sensor_boundary_requested=true;eventio::atomic_json(out_/("sensor_boundary_request_"+std::to_string(outer_index)+".json"),{{"accepted_iteration",s.iteration},{"sensor_active_station_rows",active},{"parameters_mutated_in_callback",false},{"reason","materialize derived station outside Ceres and relinearize so visible inward descent remains available"}});return ceres::SOLVER_TERMINATE_SUCCESSFULLY;}}
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
std::size_t apply_recovery_stage(JointALBuilder&b,ceres::Problem&problem,const eventio::Json&cfg,const std::filesystem::path&out){
 const auto stage=cfg.value("parameter_stage",std::string("motion"));
 if(stage!="motion"&&stage!="shared_geometry_rig"&&stage!="optics")throw std::runtime_error("unknown recovery parameter stage");
 std::set<std::string> exact_fixed;for(const auto&name:cfg.value("temporary_runtime_fixed_physical_blocks",std::vector<std::string>{}))if(!exact_fixed.insert(name).second)throw std::runtime_error("duplicate exact physical freeze");
 std::set<std::string> matched_exact;std::size_t fixed=0;for(auto&block:b.core.model.blocks){bool free=block.name.rfind("shot/",0)==0;
  if(stage=="shared_geometry_rig")free=block.name.find("/optics")==std::string::npos;
  if(stage=="optics")free=true;
  for(const auto&prefix:cfg.value("temporary_runtime_fixed_physical_prefixes",std::vector<std::string>{}))if(block.name.rfind(prefix,0)==0)free=false;
  if(exact_fixed.count(block.name)){free=false;matched_exact.insert(block.name);}
  for(const auto&rec:cfg.value("protected_S01_scalar_prefixes",eventio::Json::array()))if(block.name==rec.at("block_name").get<std::string>())free=true;
  if(!free){problem.SetParameterBlockConstant(block.values.data());++fixed;}
 }if(matched_exact!=exact_fixed)throw std::runtime_error("exact physical freeze names differ from actual registry");eventio::atomic_json(out/"runtime_primitive_release.json",apply_joint_runtime_primitive_release_v1(b,problem,cfg));install_joint_recovery_scalar_prefix_manifolds_v3(b,problem,cfg);const auto protection=protect_joint_recovery_s01_scalar_prefix_v3(b,problem,cfg);eventio::atomic_json(out/"S01_problem_protection.json",protection);return fixed;
}
eventio::Json engineering_constraint_debt(const ConstraintValues&v){
 double gmax=0,hmax=0;long double gsum=0,hsum=0;std::size_t violated=0;
 for(double x:v.g){if(!std::isfinite(x))throw std::runtime_error("nonfinite original engineering g");const double z=std::max(0.,x);gmax=std::max(gmax,z);gsum+=z*z;if(z>1e-8)++violated;}
 for(double x:v.h){if(!std::isfinite(x))throw std::runtime_error("nonfinite original engineering h");hmax=std::max(hmax,std::abs(x));hsum+=x*x;}
 return {{"original_g_count",v.g.size()},{"original_h_count",v.h.size()},{"g_max_positive",gmax},{"h_max_abs",hmax},{"g_positive_l2",std::sqrt(double(gsum))},{"h_l2",std::sqrt(double(hsum))},{"g_over_1e_minus8",violated},{"state_token",v.state_token}};
}
double contribution(ceres::Problem&p,const std::vector<void*>&ids){ceres::Problem::EvaluateOptions o;for(void*i:ids)o.residual_blocks.push_back(static_cast<ceres::ResidualBlockId>(i));if(ids.empty())return 0;double c=NAN;if(!p.Evaluate(o,&c,nullptr,nullptr,nullptr)||!std::isfinite(c))throw std::runtime_error("engineering contribution evaluation failed");return c;}

int run_source_uv_capture_native_v1(JointALBuilder&b,const eventio::Json&cfg,const std::filesystem::path&out,const std::string&fingerprint,const std::function<bool()>&recalled,int threads){
 if((!b.source_uv_capture_v1()&&!b.source_uv_observed_row_initialization())||!b.gpu_evaluation_callback())throw std::runtime_error("sourceUV trueRS capture requires its actual implicit-row GPU cache");
 ceres::Problem::Options po;po.evaluation_callback=b.gpu_evaluation_callback();std::unique_ptr<ceres::Problem>problem;eventio::Json ledger;BuildInventory inv;std::size_t fixed=0;
 const bool station_schur=cfg.value("linear_solver_backend",std::string("cuda_cgnr"))=="cpu_eigen_sparse_schur";if(station_schur&&!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))throw std::runtime_error("explicit Eigen sparse Schur backend unavailable");std::shared_ptr<ceres::ParameterBlockOrdering>station_ordering;
 const bool provisional=b.source_uv_observed_row_initialization();const auto certified_domain=JointCaptureCertifiedDomainV1::load(b.core.model,cfg.at("capture_certified_numeric_domain"));std::optional<JointCaptureStationCellsV20>cells;if(b.source_uv_capture_v4()){cells=JointCapturePrimalPrefixV5::load_policy(b,cfg.at("capture_station_cells_v20"));if(cfg.value("source_uv_capture_transition_from_v3",false))cells->prepare(b,out/"initial_station_cell_rebind",true);else validate_capture_cell_primal_v20(b);}
 auto construct=[&](const std::filesystem::path&where){std::filesystem::create_directories(where);problem=std::make_unique<ceres::Problem>(po);inv=b.build_engineering_problem(*problem,cfg,&ledger);fixed=apply_recovery_stage(b,*problem,cfg,where);if(provisional)for(std::size_t i=0;i<b.core.packet.rows.size();++i){if(problem->HasParameterBlock(&b.core.row[i]))problem->SetParameterBlockConstant(&b.core.row[i]);if(problem->HasParameterBlock(&b.core.alpha[i]))problem->SetParameterBlockConstant(&b.core.alpha[i]);}eventio::atomic_json(where/"actual_certified_numeric_domain.json",certified_domain.install(b,*problem));if(cells)cells->install(b,*problem,where);eventio::atomic_json(where/"actual_stage_guard.json",assert_joint_capture_stage_v1(b,*problem,cfg));if(station_schur)station_ordering=make_joint_station_schur_ordering_v1(b,*problem,where);};construct(out);if(cfg.value("source_uv_capture_construct_problem_only",false)){auto*cache=b.gpu_evaluation_callback();cache->PrepareForEvaluation(true,true);eventio::atomic_json(out/"actual_production_graph_preflight.json",{{"schema","rinkfit.actual_production_sourceUV_graph_preflight.v1"},{"objective_id",b.identity().objective_id},{"raw_rows",b.core.packet.rows.size()},{"all_original_sources_retained",true},{"GPU",b.gpu_report()},{"pixel_residual_blocks",inv.statistical_blocks.size()},{"inequality_blocks",inv.inequality_blocks.size()},{"equality_blocks",inv.equality_blocks.size()},{"provisional_measured_row",provisional},{"true_RS_verified",false},{"graph_and_GPU_callback_exercised",true},{"forward_geometry_must_be_checked_separately",true},{"Solve_started",false},{"quality_accepted",false}});return 0;}if(cells)save_recovery_state(b,out/"initial_state",fingerprint);eventio::atomic_json(out/"pixel_groups.json",ledger);
 auto costs=[&](){return eventio::Json{{"source_UV_2D_pixel_half_cost",contribution(*problem,inv.statistical_blocks)},{provisional?"physical_domain_g_half_cost":"true_RS_physical_g_half_cost",contribution(*problem,inv.inequality_blocks)},{"physical_relation_h_half_cost",contribution(*problem,inv.equality_blocks)}};};
 const auto initial=costs();eventio::atomic_json(out/"initial_contributions.json",initial);eventio::atomic_json(out/(provisional?"initial_provisional_physical_constraint_debt.json":"initial_true_RS_constraint_debt.json"),engineering_constraint_debt(b.evaluate_current_engineering_constraints()));
 eventio::Json registry=eventio::Json::array();std::size_t free=0,tangent=0;for(const auto&bl:b.core.model.blocks){const auto*x=bl.values.data();const int t=problem->ParameterBlockTangentSize(x);if(!problem->IsParameterBlockConstant(x)){++free;tangent+=t;registry.push_back({{"name",bl.name},{"ambient",bl.values.size()},{"tangent",t}});}}eventio::atomic_json(out/"actual_physical_stage.json",{{"free_blocks",free},{"tangent_dimensions",tangent},{"blocks",registry},{"fixed_blocks",fixed},{"source_station_joint_aux_not_Schur",!station_schur},{"linearized_station_Schur_elimination",station_schur},{"nonlinear_closest_point_elimination",false},{"row_alpha_are_not_optimization_variables",true}});
 RecoveryCallback callback(b,out,recalled);callback.write_physical_preview=cfg.value("write_accepted_physical_preview",true);callback.capture_sensor_boundary_enabled=b.source_uv_capture_v2();callback.capture_cell_boundary_enabled=b.source_uv_capture_v4();callback.capture_review_boundary_enabled=b.source_uv_capture_v2()&&cfg.value("source_uv_capture_first_accepted_review_checkpoint",true);callback.begin_outer(0);ceres::Solver::Summary summary;bool ran=false;eventio::Json batches=eventio::Json::array();double trust_radius=cfg.value("initial_trust_region_radius",100.);
 if(!cfg.value("source_uv_capture_preflight_only",false)){
  const auto backend=cfg.value("linear_solver_backend",std::string("cuda_cgnr"));const bool cuda=backend=="cuda_cgnr";if(!cuda&&!station_schur&&backend!="cpu_suite_sparse")throw std::runtime_error("unsupported trueRS capture sparse backend");const auto sparse_library=station_schur?ceres::EIGEN_SPARSE:(cuda?ceres::CUDA_SPARSE:ceres::SUITE_SPARSE);if(!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(sparse_library))throw std::runtime_error("explicit capture sparse backend unavailable");
  for(std::size_t batch=0;;++batch){callback.capture_sensor_boundary_requested=false;callback.capture_review_boundary_requested=false;callback.begin_outer(batch);ceres::Solver::Options options;options.linear_solver_type=station_schur?ceres::SPARSE_SCHUR:(cuda?ceres::CGNR:ceres::SPARSE_NORMAL_CHOLESKY);options.sparse_linear_algebra_library_type=sparse_library;options.preconditioner_type=ceres::JACOBI;if(station_schur)options.linear_solver_ordering=station_ordering;options.max_num_iterations=cfg.value("max_inner_iterations",50);options.num_threads=threads;options.update_state_every_iteration=true;options.callbacks.push_back(&callback);options.minimizer_progress_to_stdout=true;options.function_tolerance=cfg.value("function_tolerance",1e-8);options.parameter_tolerance=1e-10;options.gradient_tolerance=1e-10;options.initial_trust_region_radius=trust_radius;options.min_lm_diagonal=cfg.value("min_lm_diagonal",1e-6);options.eta=cfg.value("inexact_linear_step_eta",.1);eventio::atomic_json(out/("actual_linear_solver_options_"+std::to_string(batch)+".json"),actual_joint_linear_solver_options_v1(options));joint_memory_marker(out,"before_true_RS_capture_Ceres_solve",{{"sensor_station_batch",batch}});ceres::Solve(options,problem.get(),&summary);callback.accepted.restore(b);ran=true;batches.push_back({{"batch",batch},{"summary",summary.FullReport()},{"sensor_boundary_requested",callback.capture_sensor_boundary_requested},{"first_review_boundary_requested",callback.capture_review_boundary_requested},{"callback_error",callback.error}});
   if((!callback.capture_sensor_boundary_requested&&!callback.capture_review_boundary_requested)||callback.stopped||!callback.error.empty()||!summary.IsSolutionUsable())break;
   const auto before=costs();const auto where=out/"sensor_station_working";problem.reset();const auto boundary=materialize_joint_source_uv_capture_v1(b.core,where/"derived_auxiliary_binding",true);if(boundary.at("failed_RS_rows_retained").get<std::size_t>())throw std::runtime_error("accepted sensor station materialization failed; complete source denominator retained");if(cells)cells->prepare(b,where/"station_cell_merit_rebind",false);construct(where);const auto after=costs();double discrepancy=0,total=0;for(const char*key:{"source_UV_2D_pixel_half_cost","true_RS_physical_g_half_cost","physical_relation_h_half_cost"}){discrepancy=std::max(discrepancy,std::abs(before.at(key).get<double>()-after.at(key).get<double>()));total+=std::abs(before.at(key).get<double>());}eventio::atomic_json(out/"sensor_station_batches"/std::to_string(batch)/"relinearization.json",{{"before_materialization",before},{"after_relinearization",after},{"max_cost_discrepancy",discrepancy},{"actual_derived_station_and_row_bound",true},{"parameters_mutated_in_callback",false},{"Ceres_problem_rebuilt",true},{"same_sourceUV_objective",true},{"state_token",b.capture_state().token()}});if(cells){if(after.at("source_UV_2D_pixel_half_cost").get<double>()>before.at("source_UV_2D_pixel_half_cost").get<double>()+2e-9*std::max(1.,total))throw std::runtime_error("cell merit boundary increased actual weighted source objective");}else if(discrepancy>2e-9*std::max(1.,total))throw std::runtime_error("sensor station boundary changed the evaluated objective");if(callback.capture_review_boundary_requested&&!callback.capture_review_completed){save_recovery_state(b,out/"first_accepted_review",fingerprint);callback.capture_review_completed=true;}save_recovery_state(b,out/"recoverable_sensor_boundary",fingerprint);if(!summary.iterations.empty())trust_radius=summary.iterations.back().trust_region_radius;
  }
 }
 if(provisional){const auto final=costs();save_recovery_state(b,out/"candidate",fingerprint);eventio::atomic_json(out/"source_uv_observed_row_initialization_receipt.json",{{"schema","rinkfit.provisional_sourceUV_measured_row_initialization_receipt.v1"},{"objective_id",b.identity().objective_id},{"solver_started",ran},{"solution_usable",ran?summary.IsSolutionUsable():true},{"initial",initial},{"final",final},{"GPU",b.gpu_report()},{"summary",summary.FullReport()},{"complete_state_token",b.capture_state().token()},{"true_RS_verified",false},{"fresh_trueRS_v3_required",true},{"phase_II_allowed",false},{"quality_accepted",false}});return callback.error.empty()?0:2;}problem.reset();const std::size_t preflight_prefix=cfg.value("source_uv_capture_preflight_only",false)&&!cfg.value("dense_source_fresh_capture_v1",false)&&!cfg.contains("resume_engineering_checkpoint")&&b.core.packet.source_json.contains("source_capture_complete_state_binding")?b.core.packet.source_json.at("source_capture_complete_state_binding").at("old_raw_rows").get<std::size_t>():0;const auto boundary=materialize_joint_source_uv_capture_v1(b.core,out/"candidate/derived_auxiliary_binding",b.source_uv_capture_v2(),preflight_prefix);if(cells&&!(cfg.value("source_uv_capture_preflight_only",false)&&(cfg.value("source_uv_capture_source_append_v4",false)||cfg.value("source_uv_capture_source_append_v5",false))))cells->prepare(b,out/"final_station_cell_merit_rebind",false);construct(out/"final_materialized_graph");const auto final=costs();const auto debt=engineering_constraint_debt(b.evaluate_current_engineering_constraints());save_recovery_state(b,out/"candidate",fingerprint,preflight_prefix);if(debt.at("state_token")!=b.capture_state().token())throw std::runtime_error("saved capture CP differs from evaluated materialized debt state");eventio::atomic_json(out/"final_contributions.json",final);eventio::atomic_json(out/"final_true_RS_constraint_debt.json",debt);
 eventio::Json receipt={{"schema","rinkfit.sourceUV_trueRS_finite_capture_native_receipt.v1"},{"objective_id",b.identity().objective_id},{"state_kind","coarse_sourceUV_capture_with_derived_true_RS_rows"},{"source_packet_sha256",b.core.packet.source_hash},{"model_sha256",eventio::file_sha256(out/"candidate/model.json")},{"checkpoint_sha256",eventio::file_sha256(out/"candidate/engineering_checkpoint.json")},{"complete_state_token",b.capture_state().token()},{"configuration_sha256",fingerprint},{"solver_started",ran},{"stopped",callback.stopped},{"callback_error",callback.error},{"initial",initial},{"final",final},{"true_RS_constraint_debt",debt},{"derived_auxiliary_binding",boundary},{"GPU",b.gpu_report()},{"sensor_station_Ceres_batches",batches},{"first_accepted_review_checkpoint",callback.capture_review_completed?eventio::Json((out/"first_accepted_review/engineering_checkpoint.json").string()):eventio::Json(nullptr)},{"original_W_likelihood_consumed",false},{"formal_local_Query_consumed",false},{"all_original_Queries_UV_bounds_rows_retained",true},{"formal_Query_rebind_required",true},{"quality_accepted",false},{"phase_II_allowed",false}};if(ran){receipt["summary"]=summary.FullReport();receipt["solution_usable"]=summary.IsSolutionUsable();}eventio::atomic_json(out/"source_uv_capture_receipt.json",receipt);return callback.error.empty()?0:2;
}

}
int run_joint_engineering_recovery(const std::string&mp,const std::string&pp,const std::string&op,const std::string&config){
 const auto cfg=config.empty()?eventio::Json::object():eventio::read_json(config);
 const bool source_uv_capture=cfg.contains("source_uv_capture_objective");const bool observed_init=source_uv_capture&&cfg.at("source_uv_capture_objective")=="rinkfit.sourceUV_observed_row_finite_initialization.v1";if(cfg.value("source_uv_capture_transition_from_v1",false))throw std::runtime_error("legacy capture transition requires frozen v18; v19 uses explicit v3 objective");if(source_uv_capture){if((!observed_init&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v3"&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4")||!cfg.contains("capture_certified_numeric_domain")||(int(cfg.contains("engineering_source_weights_v3"))+int(cfg.contains("engineering_source_weights_v4"))+int(cfg.contains("engineering_source_weights_v5"))+int(cfg.contains("engineering_dense_source_weights_v1"))!=1)||!cfg.contains("gpu_raw_uv")||cfg.value("initialization_only",false)||cfg.contains("source_capture_init")||cfg.contains("protected_S01_scalar_prefixes"))throw std::runtime_error("v19 capture requires strict v3 and immutable certified numeric domain");if(cfg.value("resume_source_uv_capture_checkpoint",false)&&!cfg.contains("resume_engineering_checkpoint"))throw std::runtime_error("capture resume mode requires its complete v3 checkpoint");if(cfg.contains("resume_engineering_outer_state"))throw std::runtime_error("sourceUV capture cannot import old AL duals");if(cfg.at("source_uv_capture_objective")=="rinkfit.sourceUV_trueRS_finite_capture.v4"&&(!cfg.contains("capture_station_cells_v20")||(!cfg.value("resume_source_uv_capture_checkpoint",false)&&!cfg.value("source_uv_capture_source_append_v4",false)&&!cfg.value("source_uv_capture_source_append_v5",false))))throw std::runtime_error("v4 requires explicit cell policy and complete primal restore");if(cfg.value("source_uv_capture_transition_from_v3",false)&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4")throw std::runtime_error("v3 transition is v4-only");}
 if(cfg.contains("resume_checkpoint"))throw std::runtime_error("engineering entry rejects PHR checkpoint; use resume_engineering_checkpoint with matching objective identity");
 if(std::filesystem::exists(op))throw std::runtime_error("immutable recovery output exists");
 if(cfg.contains("archived_dependency_aliases"))JointExactDependencyAliasesV1::install(cfg.at("archived_dependency_aliases"));joint_memory_marker(op,"before_model_load");auto model=JointEventModel::load(mp);joint_memory_marker(op,"before_packet_load");auto packet=JointPacket::load(pp,model);joint_memory_marker(op,"packet_loaded",{{"raw_array_cache_bytes",joint_packet_raw_cache_bytes()},{"typed_rows",packet.rows.size()},{"typed_groups",packet.groups.size()}});
 assert_joint_source_math_configuration(cfg,packet);
 std::shared_ptr<JointCurrentBoardSourceQualificationV1> current_board_qualification;
 std::shared_ptr<JointPrintedBoardSourceQualificationV1> printed_board_qualification;std::vector<std::filesystem::path> printed_board_recall_dirs;
 std::shared_ptr<JointIceShapeSourceQualificationV1> ice_shape_qualification;
 std::shared_ptr<JointDenseSourceQualificationV1> dense_source_qualification;
 if(int(cfg.contains("board_print_source_qualification"))+int(cfg.contains("board_current_source_qualification"))+int(cfg.contains("ice_shape_source_qualification"))+int(cfg.contains("dense_source_qualification"))>1)throw std::runtime_error("current/printed/mixed source proofs are mutually exclusive");
 if(cfg.contains("dense_source_qualification")){if(!source_uv_capture||(!observed_init&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v3"))throw std::runtime_error("dense source proof requires fresh trueRS v3 or provisional measured-row initializer");dense_source_qualification=std::make_shared<JointDenseSourceQualificationV1>(JointDenseSourceQualificationV1::load(packet,pp,model,cfg.at("dense_source_qualification")));eventio::atomic_json(std::filesystem::path(op)/"dense_source_qualification.json",dense_source_qualification->report());}
 else if(cfg.contains("ice_shape_source_qualification")){if(!source_uv_capture||cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4")throw std::runtime_error("mixed ice/shape source proof requires explicit v4 capture identity");ice_shape_qualification=std::make_shared<JointIceShapeSourceQualificationV1>(JointIceShapeSourceQualificationV1::load(packet,pp,model,cfg.at("ice_shape_source_qualification")));eventio::atomic_json(std::filesystem::path(op)/"ice_shape_source_qualification.json",ice_shape_qualification->report());}
 else if(cfg.contains("board_print_source_qualification")){if(!source_uv_capture||(cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v3"&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4"))throw std::runtime_error("printed board source proof requires explicit capture v3/v4 entry");printed_board_qualification=std::make_shared<JointPrintedBoardSourceQualificationV1>(JointPrintedBoardSourceQualificationV1::load(packet,pp,model,cfg.at("board_print_source_qualification")));eventio::atomic_json(std::filesystem::path(op)/"printed_board_source_qualification.json",printed_board_qualification->report());auto proofpath=std::filesystem::u8path(cfg.at("board_print_source_qualification").at("file").get<std::string>());if(proofpath.is_relative())proofpath=std::filesystem::path(pp).parent_path()/proofpath;const auto proof=eventio::read_json(proofpath);auto inherited=std::filesystem::u8path(proof.at("inherited_native_qualification").at("file").get<std::string>());if(inherited.is_relative())inherited=proofpath.parent_path()/inherited;printed_board_recall_dirs={proofpath.parent_path(),inherited.parent_path()};}
 else if(cfg.contains("board_current_source_qualification")){if(!source_uv_capture||(cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v3"&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4"))throw std::runtime_error("current board source proof requires explicit capture v3 entry");current_board_qualification=std::make_shared<JointCurrentBoardSourceQualificationV1>(JointCurrentBoardSourceQualificationV1::load(packet,pp,model,cfg.at("board_current_source_qualification")));eventio::atomic_json(std::filesystem::path(op)/"current_board_source_qualification.json",current_board_qualification->report());}else assert_joint_source_identity_current_v3(packet,pp,model);
 std::shared_ptr<JointObservedNumericOverlayV1> numeric_overlay;std::shared_ptr<JointSegmentedNumericOverlayV1> segmented_numeric_overlay;if(packet.source_json.contains("dense_numeric_auxiliary_overlay")){if(!dense_source_qualification)throw std::runtime_error("numeric overlay requires actual qualified dense source gate");const auto&nr=packet.source_json.at("dense_numeric_auxiliary_overlay");eventio::verify(nr.at("file").get<std::string>(),nr.at("sha256"));const auto mode=eventio::read_json(nr.at("file").get<std::string>()).at("schema");if(mode=="rinkfit.segmented_native_numeric_auxiliary_overlay.v1"){segmented_numeric_overlay=std::make_shared<JointSegmentedNumericOverlayV1>(JointSegmentedNumericOverlayV1::consume(model,packet,mp,cfg));eventio::atomic_json(std::filesystem::path(op)/"segmented_numeric_overlay_consumption.json",segmented_numeric_overlay->report());}else{numeric_overlay=std::make_shared<JointObservedNumericOverlayV1>(JointObservedNumericOverlayV1::consume(model,packet,mp,cfg));eventio::atomic_json(std::filesystem::path(op)/"observed_numeric_overlay_consumption.json",numeric_overlay->report());}}
 if(cfg.contains("runtime_source_curve_coverage_v1")){if(!dense_source_qualification||!segmented_numeric_overlay)throw std::runtime_error("cap coverage transform requires actual qualified source and complete fitted-prefix numeric consumption first");apply_joint_cap168_runtime_v1(model,packet,mp,pp,cfg,std::filesystem::path(op)/"runtime_source_curve_coverage");}
 const auto source_capture_initialization=initialize_joint_recovery_source_capture(model,packet,cfg,mp,pp,op,config.empty()?std::filesystem::current_path():std::filesystem::path(config).parent_path());
 std::shared_ptr<const JointVisibleIceSourceSupport> visible_ice_support;
 if(cfg.contains("visible_ice_source_constraints"))visible_ice_support=std::make_shared<JointVisibleIceSourceSupport>(JointVisibleIceSourceSupport::load(model,cfg.at("visible_ice_source_constraints")));
 const auto source_query_policy=load_joint_source_query_policy(cfg,model,packet);
 auto guard=joint_source_math_guard_snapshot_v1(packet.source_json,pp);if(ice_shape_qualification)guard["board_material_source_qualification"]=packet.source_json.at("source_only_ice_shape_successor").at("qualification");if(printed_board_qualification)guard["board_material_source_qualification"]=packet.source_json.at("source_only_printed_board_successor").at("qualification");if(current_board_qualification)guard["board_material_source_qualification"]=packet.source_json.at("source_only_board_successor").at("qualification");
 const auto source_recall_pin=JointSourceMathRecallPinV1::capture(packet.source_json,pp);
 const bool inherited_source_enriched_CP=packet.source_json.contains("accepted_explicit_state_binding")&&packet.source_json.at("accepted_explicit_state_binding").value("schema",std::string{})=="rinkfit.accepted_source_enriched_state_packet_binding.v1";
 if(packet.source_json.contains("accepted_explicit_state_binding"))guard["accepted_explicit_state_binding"]=packet.source_json.at("accepted_explicit_state_binding");
 // Complete accepted-state width/operator contracts consume the real pair
 // identities after packet compaction; row.width_block alone is insufficient.
 if(packet.source_json.contains("stroke_pairs"))guard["stroke_pairs"]=packet.source_json.at("stroke_pairs");
 for(const char*key:{"engineering_scope","shot_ids","engineering_pixel_dependency_groups","allowed_paths","source_capture_complete_state_binding","dense_numeric_auxiliary_overlay","dense_source_binding","dense_joint_source_append","dense_qualification_binding","source_qualification_delta"})if(packet.source_json.contains(key))guard[key]=packet.source_json.at(key);
 // Complete primal import compares these small SHA/dtype/shape descriptors after
 // runtime compaction; the full arrays and packet remain on disk.
 if(packet.source_json.contains("source_capture_complete_state_binding"))for(const char*key:{"station","row","alpha","path"})guard["arrays"][key]=packet.source_json.at("arrays").at(key);
 // v4 source budgets validate the original per-row family identity after compaction.
 if(cfg.contains("engineering_source_weights_v4")||cfg.contains("engineering_source_weights_v5"))guard["source_family_ids"]=packet.source_json.at("source_family_ids");
 const auto released_raw=joint_packet_raw_cache_bytes();release_joint_packet_raw_cache();packet.source_json=guard;packet.source_json["schema"]=packet.schema;
 joint_memory_marker(op,"source_verified_and_runtime_compacted",{{"released_raw_array_cache_bytes",released_raw},{"full_source_document_retained_on_disk",pp},{"typed_rows_queries_groups_W_retained",true}});
 auto recalled=[&](){if(segmented_numeric_overlay)segmented_numeric_overlay->assert_dependencies();if(numeric_overlay)numeric_overlay->assert_dependencies();if(dense_source_qualification)dense_source_qualification->assert_dependencies();if(ice_shape_qualification)ice_shape_qualification->assert_dependencies();if(printed_board_qualification){printed_board_qualification->assert_dependencies();for(const auto&dir:printed_board_recall_dirs){for(const char*name:{"SOURCE_IDENTITY_CERTIFICATE_RECALLED.json","SOURCE_IDENTITY_EVIDENCE_RECALL.json"})if(std::filesystem::exists(dir/name))return true;}}if(current_board_qualification){current_board_qualification->assert_dependencies();const auto dir=std::filesystem::u8path(cfg.at("board_current_source_qualification").at("file").get<std::string>()).parent_path();for(const char*name:{"SOURCE_IDENTITY_CERTIFICATE_RECALLED.json","SOURCE_IDENTITY_EVIDENCE_RECALL.json"})if(std::filesystem::exists(dir/name))return true;}if(visible_ice_support)visible_ice_support->assert_dependencies();if(source_query_policy)source_query_policy->assert_dependencies();return !source_recall_pin.marker(guard,pp,true).empty();};
 if(recalled())throw std::runtime_error("source qualification recalled");
 const std::filesystem::path out=op;
 eventio::atomic_json(out/"canonical_source_recall_pin.json",source_recall_pin.report());
 // Stage, initialization and execution settings do not alter the frozen mathematical objective.
 eventio::Json objective_cfg={{"objective_id","engineering_recovery_native_pixel_v1"},{"physical_constraint_penalty",cfg.value("physical_constraint_penalty",10.)},{"pixel_total_weight",cfg.value("pixel_total_weight",double(std::count_if(packet.rows.begin(),packet.rows.end(),[](const auto&r){return r.active;})))}};if(cfg.contains("engineering_active_shot_ids"))objective_cfg["engineering_active_shot_ids"]=cfg.at("engineering_active_shot_ids");if(cfg.contains("engineering_pixel_dependency_policy"))objective_cfg["engineering_pixel_dependency_policy"]=cfg.at("engineering_pixel_dependency_policy");if(visible_ice_support)objective_cfg["visible_ice_source_constraints"]=cfg.at("visible_ice_source_constraints");if(source_query_policy)objective_cfg["source_query_branch_registry"]=source_query_policy->binding();if(cfg.contains("board_current_source_qualification"))objective_cfg["board_current_source_qualification"]=cfg.at("board_current_source_qualification");if(cfg.contains("board_print_source_qualification"))objective_cfg["board_print_source_qualification"]=cfg.at("board_print_source_qualification");if(cfg.contains("ice_shape_source_qualification"))objective_cfg["ice_shape_source_qualification"]=cfg.at("ice_shape_source_qualification");if(cfg.contains("dense_source_qualification"))objective_cfg["dense_source_qualification"]=cfg.at("dense_source_qualification");if(source_uv_capture){objective_cfg["capture_certified_numeric_domain"]=cfg.at("capture_certified_numeric_domain");if(cfg.contains("engineering_dense_source_weights_v1"))objective_cfg["engineering_dense_source_weights_v1"]=cfg.at("engineering_dense_source_weights_v1");else if(cfg.contains("engineering_source_weights_v5"))objective_cfg["engineering_source_weights_v5"]=cfg.at("engineering_source_weights_v5");else if(cfg.contains("engineering_source_weights_v4"))objective_cfg["engineering_source_weights_v4"]=cfg.at("engineering_source_weights_v4");else objective_cfg["engineering_source_weights_v3"]=cfg.at("engineering_source_weights_v3");if(cfg.contains("capture_station_cells_v20"))objective_cfg["capture_station_cells_v20"]=cfg.at("capture_station_cells_v20");}const auto config_text=objective_cfg.dump();auto config_identity=eventio::sha256(std::vector<std::uint8_t>(config_text.begin(),config_text.end()));
 JointALBuilder b(std::move(model),std::move(packet),false,false);
 const bool dense_fresh_capture=cfg.value("dense_source_fresh_capture_v1",false);if(dense_fresh_capture&&(!source_uv_capture||!dense_source_qualification||(!observed_init&&cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v3")||cfg.contains("resume_engineering_checkpoint")||cfg.value("source_uv_capture_source_append_v4",false)||cfg.value("source_uv_capture_source_append_v5",false)))throw std::runtime_error("fresh dense capture requires qualified sources and current six-shot v3 graph, no historical primal resume");auto capture_prefix=(cfg.contains("resume_engineering_checkpoint")||dense_fresh_capture)?JointCapturePrimalPrefixV5{}:JointCapturePrimalPrefixV5::load(b.core);if(dense_fresh_capture)eventio::atomic_json(out/"fresh_dense_current_model_binding.json",{{"qualified_dense_sources",true},{"current_model_sha256",eventio::file_sha256(mp)},{"source_packet_sha256",b.core.packet.source_hash},{"inherited_source_capture_complete_state_binding_retained_as_source_history",b.core.packet.source_json.contains("source_capture_complete_state_binding")},{"historical_whole_physical_primal_imported",false},{"current_packet_four_auxiliaries_retained",true},{"true_RS_finite_capture_preflight_follows",true}});if(cfg.value("source_uv_capture_source_append_v4",false)&&cfg.value("source_uv_capture_source_append_v5",false))throw std::runtime_error("v4/v5 source-append entry flags are mutually exclusive");if(cfg.value("source_uv_capture_source_append_v5",false)){if(!source_uv_capture||cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4"||cfg.contains("resume_engineering_checkpoint")||!capture_prefix.selected_v5||!ice_shape_qualification||cfg.value("source_uv_capture_transition_from_v3",false))throw std::runtime_error("explicit mixed v5 source append requires its selected full prefix, source gate, and fresh graph");}else if(cfg.value("source_uv_capture_source_append_v4",false)){if(!source_uv_capture||cfg.at("source_uv_capture_objective")!="rinkfit.sourceUV_trueRS_finite_capture.v4"||cfg.contains("resume_engineering_checkpoint")||!capture_prefix.selected_v4||capture_prefix.selected_v5||cfg.value("source_uv_capture_transition_from_v3",false))throw std::runtime_error("explicit v4 source append requires its selected complete prefix and a fresh graph");}else if(capture_prefix.selected_v4)throw std::runtime_error("source prefix requires its explicit source-append entry");if(capture_prefix.count){if(!source_uv_capture||cfg.contains("baseline_auxiliary_manifest"))throw std::runtime_error("capture primal prefix is fresh v3-only, not legacy restore or baseline import");eventio::atomic_json(out/"selected_capture_primal_prefix.json",capture_prefix.report());}
 if(visible_ice_support){b.attach_visible_ice_source_support(visible_ice_support);const auto report=visible_ice_support->evaluate(b.core.model);if(report.at("invalid_sources_retained").get<std::size_t>()!=0)throw std::runtime_error("source-visible ice ray initial inverse/finite domain failed; complete source denominator retained");eventio::atomic_json(out/"source_visible_ice_initial.json",{{"source_set",visible_ice_support->receipt()},{"current_physical_report",report},{"quality_accepted",false}});}
 if(source_uv_capture&&cfg.value("resume_source_uv_capture_checkpoint",false)){auto j=objective_cfg;j["objective_id"]=cfg.at("source_uv_capture_objective");auto s=j.dump();config_identity=eventio::sha256(std::vector<std::uint8_t>(s.begin(),s.end()));if(cfg.at("source_uv_capture_objective")=="rinkfit.sourceUV_trueRS_finite_capture.v4"&&!cfg.value("source_uv_capture_transition_from_v3",false)){b.enable_source_uv_capture_v4();b.capture_cell_policy_ref_v20=cfg.at("capture_station_cells_v20");}else b.enable_source_uv_capture_v3();}
 if(cfg.contains("resume_engineering_checkpoint")){if(source_query_policy)b.attach_source_query_branch_registry(source_query_policy);std::ifstream stream(cfg.at("resume_engineering_checkpoint").get<std::string>());auto state=read_checkpoint(stream);if(!state.outer.lambda.empty()||!state.outer.nu.empty())throw std::runtime_error("engineering checkpoint contains old AL multipliers");const bool transition=cfg.value("source_uv_capture_transition_from_v3",false);if(transition&&state.state.data().identity.objective_id!="rinkfit.sourceUV_trueRS_finite_capture.v3")throw std::runtime_error("explicit v3->v4 transition requires complete v3 source state");b.set_engineering_identity(transition?state.runtime_fingerprint:config_identity);if(b.source_uv_capture_v4())load_capture_cell_binding_v20(b,cfg.at("resume_engineering_checkpoint").get<std::string>(),state.state);b.restore_state(state.state);if(b.capture_state().token()!=state.state.token())throw std::runtime_error("engineering full checkpoint failed exact physical/continuous/discrete roundtrip");if(transition){const auto before=b.capture_state().token();b.enable_source_uv_capture_v4();b.capture_cell_policy_ref_v20=cfg.at("capture_station_cells_v20");b.set_engineering_identity(config_identity);eventio::atomic_json(out/"sourceUV_capture_v3_to_v4_complete_primal_transition.json",{{"previous_complete_token",before},{"previous_checkpoint_sha256",eventio::file_sha256(cfg.at("resume_engineering_checkpoint").get<std::string>())},{"new_mode_token_before_cell_rebind",b.capture_state().token()},{"physical_source_and_all_four_aux_changed",false},{"numerical_cell_rebind_follows",true}});}else if(b.source_uv_capture_v4())validate_capture_cell_primal_v20(b);}
 const int threads=cfg.value("num_threads",8);
 joint_memory_marker(out,"before_auxiliary_initialization");
 if(cfg.value("reinitialize_native_auxiliaries",true)&&!capture_prefix.count){if(cfg.contains("resume_engineering_checkpoint"))throw std::runtime_error("cannot reinitialize an engineering checkpoint");
  JointBaselineAuxiliaryPreservation preserved;
  if(b.core.packet.source_json.contains("accepted_explicit_state_binding")){auto inherited=preserve_joint_accepted_auxiliary_prefix_current_v5(b.core,mp,pp,source_query_policy);preserved.preserved=std::move(inherited.preserved);eventio::atomic_json(out/"accepted_auxiliary_prefix_restore.json",inherited.receipt); }
  if(cfg.contains("baseline_auxiliary_manifest")){if(!preserved.preserved.empty())throw std::runtime_error("cannot mix baseline import and accepted-prefix import");preserved=restore_joint_baseline_auxiliaries(b.core,cfg.at("baseline_auxiliary_manifest"),mp,pp);eventio::atomic_json(out/"baseline_auxiliary_restore.json",preserved.receipt);}
  if(cfg.contains("s01_component_rollback")){if(preserved.preserved.empty())throw std::runtime_error("component rollback requires verified accepted prefix");auto rollback=rollback_joint_s01_component(b.core,cfg.at("s01_component_rollback"),mp,pp);for(std::size_t i=0;i<rollback.preserved.size();++i)if(rollback.preserved[i])preserved.preserved.at(i)=1;eventio::atomic_json(out/"s01_component_rollback.json",rollback.receipt);}
  initialize_recovery_auxiliaries(b.core,out/"fresh_auxiliary_initialization",threads,preserved.preserved);
 }
 if(!cfg.contains("resume_engineering_checkpoint")){if(source_query_policy){if(!cfg.value("reinitialize_native_auxiliaries",true))throw std::runtime_error("cold source Query branch installation requires complete accepted four-aux import before C-only rebind");if(!inherited_source_enriched_CP){const auto receipt=source_query_policy->rebind_C_seeds_after_complete_CP(b.core.model,b.core.packet,b.core.station,b.core.row,b.core.alpha);eventio::atomic_json(out/"C_source_branch_numerical_seed_rebind.json",receipt);}else eventio::atomic_json(out/"C_source_branch_existing_complete_state.json",{{"inherited_true_complete_CP",true},{"base_seed_overlay_reapplied",false},{"typed_query_only_idempotent_attachment",true},{"quality_accepted",false}});b.attach_source_query_branch_registry(source_query_policy);}b.set_engineering_identity(config_identity); }
 if(source_uv_capture){if(observed_init){if(cfg.contains("resume_engineering_checkpoint"))throw std::runtime_error("measured-row initializer does not resume an implicit/formal checkpoint");b.enable_source_uv_observed_row_initialization();auto j=objective_cfg;j["objective_id"]=cfg.at("source_uv_capture_objective");auto s=j.dump();config_identity=eventio::sha256(std::vector<std::uint8_t>(s.begin(),s.end()));b.set_engineering_identity(config_identity);const auto domain=JointCaptureCertifiedDomainV1::load(b.core.model,cfg.at("capture_certified_numeric_domain"));eventio::atomic_json(out/"capture_certified_domain_binding.json",domain.bind_auxiliary(b));std::vector<std::uint8_t>preserved(b.core.packet.rows.size(),1);std::size_t invalid=0,active=0;for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;++active;const auto l=r.layout(b.core.model);const auto p=b.core.model.pointers(l);const auto v=b.core.model.predict_explicit(l,r.kind==1?r.station:b.core.station[i],b.core.row[i],p.data());if(!v.valid()){preserved[i]=r.shot_id=="s01"?1:0;++invalid;}}if(invalid){initialize_recovery_auxiliaries(b.core,out/"measured_row_finite_seed",threads,preserved);b.enable_source_uv_observed_row_initialization();}std::size_t valid=0;eventio::Json failures=eventio::Json::array();for(std::size_t i=0;i<b.core.packet.rows.size();++i){const auto&r=b.core.packet.rows[i];if(!r.active)continue;const auto l=r.layout(b.core.model);const auto p=b.core.model.pointers(l);const auto v=b.core.model.predict_explicit(l,r.kind==1?r.station:b.core.station[i],b.core.row[i],p.data());if(v.valid())++valid;else failures.push_back({{"row",i},{"shot",r.shot_id},{"native_pts",r.pts},{"source_row_id",r.source_row_id}});}eventio::atomic_json(out/"measured_row_initial_forward.json",{{"schema","rinkfit.provisional_measured_row_forward_capture.v1"},{"raw_rows",b.core.packet.rows.size()},{"active_rows",active},{"valid_forward_rows",valid},{"invalid_forward_rows_retained",active-valid},{"failed_sources",failures},{"true_RS_verified",false},{"source_UV_Query_E_W_or_qualification_changed",false}});if(valid!=active&&!cfg.value("source_uv_capture_construct_problem_only",false))return 2;}else{if(!b.source_uv_capture_v3()){const auto previous=b.capture_state().token();if(capture_prefix.selected_v4){b.enable_source_uv_capture_v4();b.capture_cell_policy_ref_v20=cfg.at("capture_station_cells_v20");eventio::atomic_json(out/"selected_capture_cell_prefix_import.json",capture_prefix.inherit_cells(b));}else b.enable_source_uv_capture_v3();auto j=objective_cfg;j["objective_id"]=cfg.at("source_uv_capture_objective");auto s=j.dump();config_identity=eventio::sha256(std::vector<std::uint8_t>(s.begin(),s.end()));b.set_engineering_identity(config_identity);eventio::atomic_json(out/"sourceUV_capture_mode_transition.json",{{"previous_mode_state_token",previous},{"new_mode_state_token",b.capture_state().token()},{"objective_id",b.identity().objective_id},{"physical_or_source_changed",false},{"numeric_aux_rebinding_after_this_record",true}});}const auto numeric_domain=JointCaptureCertifiedDomainV1::load(b.core.model,cfg.at("capture_certified_numeric_domain"));eventio::atomic_json(out/"capture_certified_domain_binding.json",numeric_domain.bind_auxiliary(b));const auto seeds=prepare_joint_source_uv_capture_v1(b.core,out/"true_RS_finite_seed",threads,cfg.value("source_uv_capture_reseed_new_sources",!cfg.value("resume_source_uv_capture_checkpoint",false)),capture_prefix.count);capture_prefix.assert_unchanged(b.core);if(capture_prefix.count&&!capture_prefix.selected_v4){const auto before=b.capture_state().token();auto numerical_rebind=rebind_joint_capture_prefix_strict_sensor_v1(b.core,capture_prefix);numerical_rebind["state_token_before"]=before;numerical_rebind["state_token_after"]=b.capture_state().token();eventio::atomic_json(out/"selected_primal_strict_sensor_numerical_rebind.json",numerical_rebind);}const auto boundary=materialize_joint_source_uv_capture_v1(b.core,out/"initial_derived_auxiliary_binding",b.source_uv_capture_v2(),capture_prefix.count);capture_prefix.assert_unchanged(b.core);if(capture_prefix.selected_v5)eventio::atomic_json(out/"mixed_new_polyline_cell_initialization.json",capture_prefix.initialize_new_polyline_cells(b,out/"mixed_new_polyline_cells"));if(boundary.at("failed_RS_rows_retained").get<std::size_t>()!=0){if(!b.source_uv_capture_v4())save_recovery_state(b,out/"rejected_initial_state",config_identity);eventio::atomic_json(out/"source_uv_capture_initial_rejection.json",{{"complete_denominator_retained",true},{"seeds",seeds},{"derived_auxiliary_binding",boundary},{"solver_started",false},{"quality_accepted",false}});return 2;}}}else polish_joint_recovery_protected_RS_rows(b,cfg,out);
 joint_memory_marker(out,"after_auxiliary_initialization");const bool graph_only_capture=source_uv_capture&&cfg.value("source_uv_capture_construct_problem_only",false)&&b.core.packet.source_json.contains("dense_numeric_auxiliary_overlay");if(!b.source_uv_capture_v4()&&!graph_only_capture)save_recovery_state(b,out/"initial_state",config_identity,capture_prefix.count);capture_prefix.assert_unchanged(b.core);joint_memory_marker(out,graph_only_capture?"initial_checkpoint_not_duplicated_for_graph_only":"initial_checkpoint_written");if(graph_only_capture)eventio::atomic_json(out/"graph_only_recoverable_input_binding.json",{{"schema","rinkfit.graph_only_existing_recoverable_input_binding.v1"},{"model_file",mp},{"model_sha256",eventio::file_sha256(mp)},{"packet_file",pp},{"packet_sha256",b.core.packet.source_hash},{"numerical_overlay",b.core.packet.source_json.at("dense_numeric_auxiliary_overlay")},{"configuration_sha256",config_identity},{"complete_state_duplicated",false},{"Solve_started",false}});

 if(cfg.value("initialization_only",false)){
  const auto metrics=export_joint_recovery_explicit_metrics(b,out/"initial_state/CP_explicit_metrics",threads);
  JointReplayArtifactMemo memo;const auto replay=export_joint_replay_artifacts(b,b.capture_state(),out/"initial_independent_RS_state",&memo,false,true);
  eventio::atomic_json(out/"initialization_only_receipt.json",{{"schema","rinkfit.recovery_initialization_preflight.v1"},{"objective_id",b.identity().objective_id},{"source_packet_sha256",b.core.packet.source_hash},{"mathematical_source_id",b.identity().source_id},{"source_math_bindings",joint_source_math_bindings(b)},{"configuration_sha256",config_identity},{"explicit_metrics",metrics},{"independent_RS_replay",replay},{"solver_started",false},{"initial_checkpoint",(out/"initial_state/engineering_checkpoint.json").string()}});
  return 0;
 }
 if(cfg.contains("gpu_raw_uv")){
  const auto&gpu=cfg.at("gpu_raw_uv");eventio::verify(gpu.at("ptx").get<std::string>(),gpu.at("ptx_sha256"));
  const auto&shadow=gpu.at("shadow_receipt");eventio::verify(shadow.at("file").get<std::string>(),shadow.at("sha256"));const auto qa=eventio::read_json(shadow.at("file").get<std::string>());
  if(!qa.value("passed",false))throw std::runtime_error("GPU native Jacobian shadow failed");
  const auto abi=gpu.value("kernel_abi",std::string("legacy_raw_uv_v1"));
  if(abi=="ice_line_width_v1"){
   if(qa.value("kernel_abi",std::string())!=abi||qa.value("kernel_symbol",std::string())!="joint_raw_uv_width_v1"||qa.value("UVLayoutWidthV1_bytes",0)!=56||qa.value("GPUMeta_bytes",0)!=304)throw std::runtime_error("width raw UV shadow ABI does not match actual consumer");
   const auto&ref=qa.at("kernel_contract");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto contract=eventio::read_json(ref.at("file").get<std::string>());
   if(contract.value("schema",std::string())!="rinkfit.raw_UV_GPU_ice_line_width_kernel_contract.v1"||contract.value("kernel_abi",std::string())!=abi||contract.at("ptx_sha256")!=gpu.at("ptx_sha256"))throw std::runtime_error("width raw UV kernel contract/PTX differs");
  }
  if(gpu.value("batch_reuse",false)){
   const auto&ref=gpu.at("batch_reuse_qualification");eventio::verify(ref.at("file").get<std::string>(),ref.at("sha256"));const auto reuse=eventio::read_json(ref.at("file").get<std::string>());
   if(reuse.value("schema",std::string())!="rinkfit.complete_dependency_per_shot_cache_handoff.v1"||!reuse.value("passed",false)||!reuse.value("PTX_and_GPU_metadata_layout_bytes_unchanged",false)||reuse.value("physical_model_source_UV_or_actual_packet_modified",true))throw std::runtime_error("batch cache reuse lacks actual immutable qualification");
   for(const auto*key:{"source","new_opt_in_header","actual_QA"}){const auto&r=reuse.at(key);eventio::verify(r.at("file").get<std::string>(),r.at("sha256"));}
   const auto qa=eventio::read_json(reuse.at("actual_QA").at("file").get<std::string>());if(!qa.value("passed",false))throw std::runtime_error("actual complete-dependency cache QA failed");
  }
  b.enable_gpu_raw_uv(gpu.at("ptx"),threads,out/"gpu_raw_uv_cache.jsonl",abi,gpu.value("batch_reuse",false));
 }
 if(source_uv_capture)return run_source_uv_capture_native_v1(b,cfg,out,config_identity,recalled,threads);
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
 options.max_num_iterations=cfg.value("max_inner_iterations",50);options.num_threads=threads;options.update_state_every_iteration=true;options.callbacks.push_back(&callback);options.minimizer_progress_to_stdout=true;options.function_tolerance=cfg.value("function_tolerance",1e-8);options.parameter_tolerance=1e-10;options.gradient_tolerance=1e-10;options.use_mixed_precision_solves=false;eventio::atomic_json(out/"actual_linear_solver_options.json",actual_joint_linear_solver_options_v1(options));
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
  auto inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg,out);
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
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,log,cfg.at("gpu_raw_uv").value("kernel_abi",std::string("legacy_raw_uv_v1")),cfg.at("gpu_raw_uv").value("batch_reuse",false));
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg,out);
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
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,log,cfg.at("gpu_raw_uv").value("kernel_abi",std::string("legacy_raw_uv_v1")),cfg.at("gpu_raw_uv").value("batch_reuse",false));
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg,out);
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
    if(cfg.contains("gpu_raw_uv"))b.enable_gpu_raw_uv(cfg.at("gpu_raw_uv").at("ptx"),threads,restart_dir/log_name,cfg.at("gpu_raw_uv").value("kernel_abi",std::string("legacy_raw_uv_v1")),cfg.at("gpu_raw_uv").value("batch_reuse",false));
    ceres::Problem::Options ro;ro.evaluation_callback=b.gpu_evaluation_callback();problem=std::make_unique<ceres::Problem>(ro);
    inventory=b.build_engineering_problem(*problem,inner_cfg,&ledger,outer.lambda,outer.nu);fixed=apply_recovery_stage(b,*problem,cfg,out);
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
   eventio::atomic_json(round_dir/"solver_summary.json",{{"message",summary.message},{"full_report",summary.FullReport()},{"brief_report",summary.BriefReport()},{"reported_iterations",summary.iterations.size()},{"initial_cost",summary.initial_cost},{"final_cost",summary.final_cost},{"source_accuracy_accepted",false}});
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
 if(visible_ice_support){visible_ice_support->assert_dependencies();receipt["source_visible_ice_constraints"]={{"source_set",visible_ice_support->receipt()},{"current_physical_report",visible_ice_support->evaluate(b.core.model)},{"source_world_points_issued",0},{"statistical_rank_added",0}};}
 receipt["canonical_source_recall_pin"]=source_recall_pin.report();
 add_joint_source_math_receipt(b,receipt);
 receipt["timing_seconds"]={{"residual",residual_seconds},{"Jacobian",jacobian_seconds},{"linear",linear_seconds},{"total",total_seconds}};
 eventio::atomic_json(out/"engineering_receipt.json",receipt);std::cout<<receipt.dump(2)<<std::endl;
 return failed||recalled()||limited?2:0;
}
}
