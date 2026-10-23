#pragma once
#include "rinkfit/joint_al.hpp"
#include <chrono>
#include <climits>
#include <unordered_map>
// Initialization-only inner adapter. Original canonical CeresInnerSolver remains untouched.
// Reuses original budget, snapshot, exact inventory and sparse solve implementation.
namespace rinkfit {namespace joint_seed_detail {
struct JointSeedBudgetFactor {
  template <typename T> bool operator()(const T* b, T* residual) const {
    if (b[0] < T(0.)) return false;
    using std::sqrt;
    residual[0] = sqrt(T(2.) * b[0] + T(1.));
    return true;
  }
};


// Reversible numerical units only. Ambient state and its original box remain unchanged.
class JointScalarStepUnits final:public ceres::Manifold {
 double scale_;
 public:explicit JointScalarStepUnits(double scale):scale_(scale){if(!std::isfinite(scale_)||scale_<=0)throw std::invalid_argument("invalid positive scalar numerical units");}
 int AmbientSize()const override{return 1;}int TangentSize()const override{return 1;}
 bool Plus(const double*x,const double*d,double*y)const override{y[0]=x[0]+scale_*d[0];return std::isfinite(y[0]);}
 bool PlusJacobian(const double*,double*j)const override{j[0]=scale_;return true;}
 bool Minus(const double*y,const double*x,double*d)const override{d[0]=(y[0]-x[0])/scale_;return std::isfinite(d[0]);}
 bool MinusJacobian(const double*,double*j)const override{j[0]=1./scale_;return true;}
};

inline void validate_problem(ceres::Problem& problem, ProblemBuilder& builder,
                       const BuildInventory& inventory, const ObjectiveContext& context,
                       void* budget_residual) {
  const auto* joint=dynamic_cast<const JointALBuilder*>(&builder);
  if(!joint)throw std::invalid_argument("joint inner adapter requires the exact joint physical builder");
  if(!joint->initialization_geometry_guidance() && context.phase==Phase::kFeasibility && !inventory.statistical_blocks.empty())throw std::invalid_argument("canonical joint Phase I cannot contain statistical guidance");
  const auto& schema = builder.schema();
  if (!inventory.exact_final_physical_graph)
    throw std::invalid_argument("builder did not attest an exact final physical graph without delta/image warp");
  if (inventory.inequality_blocks.size() != schema.inequality_ids.size()
      || inventory.equality_blocks.size() != schema.equality_ids.size())
    throw std::invalid_argument("AL factor inventory differs from complete constraint schema");
  if (context.phase == Phase::kFeasibility
      && (!inventory.measured_prior_blocks.empty()
          || !inventory.smoothness_blocks.empty() || inventory.common_mode_active))
    throw std::invalid_argument("guided initialization forbids measured priors, smoothness and common-mode alternatives");
  std::vector<ceres::ResidualBlockId> residuals;
  problem.GetResidualBlocks(&residuals);
  const std::set<void*> actual_residuals(residuals.begin(), residuals.end());
  std::set<void*> accounted;
  for (const auto* factors : {&inventory.inequality_blocks, &inventory.equality_blocks}) {
    for (void* factor : *factors) {
      if (!factor || !actual_residuals.count(factor) || !accounted.insert(factor).second) throw std::invalid_argument("unknown/duplicate AL residual handle");
      const auto handle = static_cast<ceres::ResidualBlockId>(factor);
      if (problem.GetLossFunctionForResidualBlock(handle) != nullptr
          || problem.GetCostFunctionForResidualBlock(handle)->num_residuals() != 1)
        throw std::invalid_argument("AL/support factors must be scalar with no robust loss");
    }
  }
  for (const auto* factors : {&inventory.statistical_blocks, &inventory.measured_prior_blocks, &inventory.smoothness_blocks})
    for (void* factor : *factors)
      if (!factor || !accounted.insert(factor).second) throw std::invalid_argument("duplicate/unclassified base factor");
  if (budget_residual) accounted.insert(budget_residual);
  if (residuals.size() != accounted.size()) throw std::invalid_argument("unclassified residuals in Ceres problem");
  for (auto* residual : residuals)
    if (!accounted.count(residual)) throw std::invalid_argument("Ceres residual missing from factor inventory");

  std::vector<double*> actual_blocks;
  problem.GetParameterBlocks(&actual_blocks);
  const auto bindings = builder.parameter_blocks();
  const auto snapshot = builder.capture_state();
  if (bindings.size() != actual_blocks.size() || bindings.size() != snapshot.data().parameters.size())
    throw std::invalid_argument("Ceres blocks and complete named snapshot blocks disagree");
  std::set<double*> pointers; std::set<std::string> names;
  std::unordered_map<std::string,const ParameterValues*> snapshot_names;
  for(const auto& p:snapshot.data().parameters) snapshot_names.emplace(p.name,&p);
  bool found_budget = false;
  for (const auto& binding : bindings) {
    if (!binding.data || !binding.size || binding.size > INT_MAX || !pointers.insert(binding.data).second
        || !names.insert(binding.name).second || !problem.HasParameterBlock(binding.data)
        || problem.ParameterBlockSize(binding.data) != static_cast<int>(binding.size))
      throw std::invalid_argument("invalid/missing stable parameter-block registry entry");
    const auto found = snapshot_names.find(binding.name);
    const auto* p = found==snapshot_names.end()?nullptr:found->second;
    if (!p || p->values.size() != binding.size || p->equality_auxiliary != binding.equality_auxiliary
        || !std::equal(p->values.begin(), p->values.end(), binding.data))
      throw std::invalid_argument("snapshot omitted/aliased or stale parameter values");
    if (binding.data == builder.budget_parameter())
      found_budget = binding.name == "budget_sq" && binding.size == 1 && !binding.equality_auxiliary;
    for (const char* forbidden : {"delta_xi", "image_warp", "frame_translation_correction"})
      if (binding.name.find(forbidden) != std::string::npos)
        throw std::invalid_argument("forbidden final correction block: " + binding.name);
  }
  if (!found_budget) throw std::invalid_argument("budget_sq must be a registered snapshotted scalar");
}

class JointSeedSnapshotCallback final : public ceres::IterationCallback {
 public:
  JointSeedSnapshotCallback(ProblemBuilder& builder, const ObjectiveContext& context, const OuterState& outer,
                   const RunHooks& hooks)
      : accepted_(builder.capture_state()), builder_(builder), context_(context), outer_(outer), hooks_(hooks),
        started_(std::chrono::steady_clock::now()) {}
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
    try {
      if (summary.step_is_successful) accepted_ = builder_.capture_state();
      if (hooks_.on_iteration) {
        IterationRecord record;
        record.iteration = summary.iteration; record.step_accepted = summary.step_is_successful;
        record.inner_cost = summary.cost; record.actual_cost_change = summary.cost_change;
        record.relative_decrease = summary.relative_decrease; record.trust_region_radius = summary.trust_region_radius;
        record.objective_token = context_.objective_token;
        record.inequality_kink_active = context_.inequality_kink_active;
        record.iteration_time_in_seconds = summary.iteration_time_in_seconds;
        record.step_solver_time_in_seconds = summary.step_solver_time_in_seconds;
        record.cumulative_time_in_seconds = summary.cumulative_time_in_seconds;
        OuterState timed_outer = outer_;
        timed_outer.elapsed_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
        hooks_.on_iteration(accepted_, timed_outer, record);
      }
      if (builder_.root_chart_restart_requested()) { restart_ = true; return ceres::SOLVER_TERMINATE_SUCCESSFULLY; }
      if (hooks_.stop_requested && hooks_.stop_requested()) { stopped_ = true; return ceres::SOLVER_TERMINATE_SUCCESSFULLY; }
      return ceres::SOLVER_CONTINUE;
    } catch (const std::exception& error) {
      callback_error_ = error.what(); return ceres::SOLVER_ABORT;
    } catch (...) {
      callback_error_ = "unknown exception from accepted-state checkpoint callback"; return ceres::SOLVER_ABORT;
    }
  }
  StateSnapshot accepted_;
  bool restart_ = false, stopped_ = false;
  std::string callback_error_;
 private:
  ProblemBuilder& builder_;
  const ObjectiveContext& context_;
  const OuterState& outer_;
  const RunHooks& hooks_;
  std::chrono::steady_clock::time_point started_;
};

}
class JointGeometrySeedInner final:public InnerSolver{
 double initial_radius_=1e4;int line_search_steps_=20;bool jacobi_scaling_=true;bool gpu_linear_=false;std::vector<std::string>temporary_physical_fixed_prefixes_;std::set<std::string>temporary_nuisance_fixed_shots_;double min_lm_diagonal_=1e-6;double linear_solver_eta_=.1;bool native_station_step_units_=false,normalize_budget_step_=false;bool temporary_exact_allowlist_active_=false;std::set<std::string>temporary_exact_allowed_parameters_;
 public:explicit JointGeometrySeedInner(double initial_radius=1e4,int line_search_steps=20,bool jacobi_scaling=true,bool gpu_linear=false):initial_radius_(initial_radius),line_search_steps_(line_search_steps),jacobi_scaling_(jacobi_scaling),gpu_linear_(gpu_linear){if(line_search_steps<0)throw std::invalid_argument("negative bounded line search iteration setting");if(!std::isfinite(initial_radius_)||initial_radius_<=0||initial_radius_>1e16)throw std::invalid_argument("invalid joint initial trust region radius");}
 void set_temporary_physical_fixed_prefixes(std::vector<std::string>prefixes){for(const auto&p:prefixes)if(p.empty())throw std::invalid_argument("empty temporary physical parameter prefix");temporary_physical_fixed_prefixes_=std::move(prefixes);}
 void set_temporary_nuisance_fixed_shots(const std::vector<std::string>&shots){temporary_nuisance_fixed_shots_=std::set<std::string>(shots.begin(),shots.end());}
 void set_explicit_initialization_parameter_allowlist(std::set<std::string>names){if(names.empty())throw std::invalid_argument("empty explicit initialization parameter allowlist");for(const auto&name:names)if(name.empty())throw std::invalid_argument("empty initialization parameter name");temporary_exact_allowed_parameters_=std::move(names);temporary_exact_allowlist_active_=true;}
 void set_min_lm_diagonal(double value){if(!std::isfinite(value)||value<=0||value>1e32)throw std::invalid_argument("invalid numerical LM diagonal lower bound");min_lm_diagonal_=value;}
 void set_linear_solver_eta(double value){if(!std::isfinite(value)||value<=0||value>=1)throw std::invalid_argument("invalid inexact CGNR step eta");linear_solver_eta_=value;}
 void set_native_numerical_step_units(bool station,bool budget){native_station_step_units_=station;normalize_budget_step_=budget;}
 InnerResult solve(ProblemBuilder&,const ObjectiveContext&,const OuterState&,const RuntimeLimits&,const RunHooks&,double)override;};
inline InnerResult JointGeometrySeedInner::solve(ProblemBuilder& builder, const ObjectiveContext& objective,
                                    const OuterState& outer, const RuntimeLimits& limits,
                                    const RunHooks& hooks, double remaining_seconds) {
  InnerResult result;
  result.accepted_state = builder.capture_state();
  try {
    limits.validate();
    if (!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(gpu_linear_?ceres::CUDA_SPARSE:ceres::SUITE_SPARSE))
      throw std::runtime_error(gpu_linear_?"requested CUDA CGNR backend unavailable in loaded Ceres library":"production CPU backend requires Ceres built with SuiteSparse; no dense/SciPy fallback");
    if (limits.max_inner_iterations > INT_MAX || remaining_seconds <= 0.)
      throw std::invalid_argument("invalid Ceres inner iteration/time budget");
    ceres::Problem::Options problem_options;
    if(auto*joint=dynamic_cast<JointALBuilder*>(&builder))problem_options.evaluation_callback=joint->gpu_evaluation_callback();
    ceres::Problem problem(problem_options);
    double* b = builder.budget_parameter();
    problem.AddParameterBlock(b, 1);
    problem.SetParameterLowerBound(b, 0, 0.);
    const BuildInventory inventory = builder.build_ceres_problem(problem, objective);
    std::size_t temporary_fixed_blocks=0,temporary_fixed_scalars=0;
    if(!temporary_physical_fixed_prefixes_.empty()){auto*joint=dynamic_cast<JointALBuilder*>(&builder);if(!joint)throw std::runtime_error("temporary physical stage requires exact joint builder");for(auto&block:joint->core.model.blocks){bool selected=false;for(const auto&prefix:temporary_physical_fixed_prefixes_)selected|=block.name.rfind(prefix,0)==0;if(selected&&problem.HasParameterBlock(block.values.data())&&!problem.IsParameterBlockConstant(block.values.data())){problem.SetParameterBlockConstant(block.values.data());++temporary_fixed_blocks;temporary_fixed_scalars+=block.values.size();}}}
    std::size_t temporary_fixed_nuisance_blocks=0;
    if(!temporary_nuisance_fixed_shots_.empty()){auto*joint=dynamic_cast<JointALBuilder*>(&builder);if(!joint)throw std::runtime_error("temporary nuisance stage requires exact joint builder");for(const auto&shot:temporary_nuisance_fixed_shots_)if(!joint->core.model.shot_map.count(shot))throw std::invalid_argument("unknown temporary nuisance shot");for(std::size_t i=0;i<joint->core.packet.rows.size();++i){const auto&r=joint->core.packet.rows[i];if(!r.active||!temporary_nuisance_fixed_shots_.count(r.shot_id))continue;for(double*value:{&joint->core.station[i],&joint->core.row[i],&joint->core.alpha[i]})if(problem.HasParameterBlock(value)&&!problem.IsParameterBlockConstant(value)){problem.SetParameterBlockConstant(value);++temporary_fixed_nuisance_blocks;}}}
    std::size_t exact_initialization_fixed_blocks=0;
    void* budget_residual = nullptr;
    if (objective.phase == Phase::kFeasibility) {
      problem.SetParameterBlockVariable(b);
      budget_residual = problem.AddResidualBlock(new ceres::AutoDiffCostFunction<joint_seed_detail::JointSeedBudgetFactor, 1, 1>(new joint_seed_detail::JointSeedBudgetFactor), nullptr, b);
    } else {
      if (*b != 1.) throw std::invalid_argument("phase II must already have exactly b=1 before building factors");
      problem.SetParameterBlockConstant(b);
    }
    if(temporary_exact_allowlist_active_){std::set<std::string>found;for(const auto&binding:builder.parameter_blocks()){if(temporary_exact_allowed_parameters_.count(binding.name)){if(problem.IsParameterBlockConstant(binding.data))throw std::runtime_error("initialization allowlist names an original constant: "+binding.name);found.insert(binding.name);}else if(!problem.IsParameterBlockConstant(binding.data)){problem.SetParameterBlockConstant(binding.data);++exact_initialization_fixed_blocks;}}if(found!=temporary_exact_allowed_parameters_)throw std::runtime_error("unregistered explicit initialization parameter name");}
    if(temporary_exact_allowlist_active_&&!temporary_exact_allowed_parameters_.count("budget_sq")&&!problem.IsParameterBlockConstant(b))throw std::runtime_error("explicit initialization failed to keep budget constant");
    joint_seed_detail::validate_problem(problem, builder, inventory, objective, budget_residual);

    std::size_t native_unit_stations=0,native_unit_CPU_derivatives=0,native_unit_invalid_derivatives=0;
    double station_speed_min=INFINITY,station_speed_max=0,budget_step_scale=1;
    const auto state_before_numerical_units=builder.capture_state().token();
    if(native_station_step_units_){
      auto*joint=dynamic_cast<JointALBuilder*>(&builder);if(!joint)throw std::runtime_error("native station units require exact joint builder");
      auto*cache=dynamic_cast<JointUVCache*>(joint->gpu_evaluation_callback());
      if(cache)cache->PrepareForEvaluation(true,true);
      for(std::size_t i=0;i<joint->core.packet.rows.size();++i){const auto&r=joint->core.packet.rows[i];double*s=&joint->core.station[i];
        if(!r.active||r.kind==1||!problem.HasParameterBlock(s)||problem.IsParameterBlockConstant(s))continue;
        if(problem.GetManifold(s))throw std::runtime_error("unexpected station manifold before numerical units");
        double du=NAN,dv=NAN;JointUVCacheView view;
        if(cache&&cache->lookup(i,&view)&&view.jacobians_ready&&view.partials&&view.station_column>=0){const auto&d=view.partial(view.station_column);du=d.u;dv=d.v;}
        else{++native_unit_CPU_derivatives;using J=ceres::Jet<double,1>;const auto l=r.layout(joint->core.model);std::vector<std::vector<J>>storage;std::vector<const J*>ptrs;
          for(int id:l.block_ids){storage.emplace_back();auto&v=storage.back();for(double x:joint->core.model.blocks[id].values)v.emplace_back(x);ptrs.push_back(v.data());}
          J station(*s);station.v[0]=1;const auto q=joint->core.model.predict_for_al(l,station,J(joint->core.row[i]),ptrs.data());if(q.arithmetic_valid){du=q.uv[0].v[0];dv=q.uv[1].v[0];}
        }
        const double speed=std::hypot(du,dv);if(!std::isfinite(speed)){++native_unit_invalid_derivatives;continue;}
        const double scale=1./std::max(1.,speed);problem.SetManifold(s,new joint_seed_detail::JointScalarStepUnits(scale));++native_unit_stations;
        station_speed_min=std::min(station_speed_min,speed);station_speed_max=std::max(station_speed_max,speed);
      }
    }
    if(normalize_budget_step_&&objective.phase==Phase::kFeasibility){budget_step_scale=std::sqrt(2.*(*b)+1.);problem.SetManifold(b,new joint_seed_detail::JointScalarStepUnits(budget_step_scale));}
    if(builder.capture_state().token()!=state_before_numerical_units)throw std::runtime_error("reversible numerical units changed ambient checkpoint values");
    if(auto*joint=dynamic_cast<JointALBuilder*>(&builder))if(auto*cache=dynamic_cast<JointUVCache*>(joint->gpu_evaluation_callback()))cache->configure_for_problem(problem);

    joint_seed_detail::JointSeedSnapshotCallback callback(builder, objective, outer, hooks);
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius=initial_radius_;options.max_num_line_search_step_size_iterations=line_search_steps_;options.jacobi_scaling=jacobi_scaling_;options.min_lm_diagonal=min_lm_diagonal_;options.eta=linear_solver_eta_;
    options.linear_solver_type = gpu_linear_?ceres::CGNR:ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = gpu_linear_?ceres::CUDA_SPARSE:ceres::SUITE_SPARSE;
    options.preconditioner_type=ceres::JACOBI;options.use_mixed_precision_solves=false;
    options.use_inner_iterations = false; options.use_nonmonotonic_steps = false;
    options.max_num_iterations = static_cast<int>(limits.max_inner_iterations);
    options.max_solver_time_in_seconds = remaining_seconds;
    options.num_threads = limits.num_threads;
    options.minimizer_progress_to_stdout = limits.log_ceres_progress;
    // Ceres gradient metrics are not the adapter's scaled tangent KKT test.
    // Permit a tight solve, then independently check the required AL accuracy.
    options.function_tolerance = 0.; options.parameter_tolerance = 0.; options.gradient_tolerance = 1e-12;
    options.update_state_every_iteration = true;
    options.callbacks.push_back(&callback);
    std::string invalid_options;
    if (!options.IsValid(&invalid_options)) throw std::invalid_argument(invalid_options);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(summary.linear_solver_type_used!=options.linear_solver_type || summary.mixed_precision_solves_used || (gpu_linear_&&summary.preconditioner_type_used!=ceres::JACOBI))throw std::runtime_error("actual Ceres linear backend/precision differs from explicitly requested production backend");
    if(hooks.on_solver_summary) hooks.on_solver_summary(outer,objective.objective_token,{
      {"residual_evaluation_time_in_seconds",summary.residual_evaluation_time_in_seconds},
      {"jacobian_evaluation_time_in_seconds",summary.jacobian_evaluation_time_in_seconds},
      {"linear_solver_time_in_seconds",summary.linear_solver_time_in_seconds},
      {"preprocessor_time_in_seconds",summary.preprocessor_time_in_seconds},
      {"postprocessor_time_in_seconds",summary.postprocessor_time_in_seconds},
      {"minimizer_time_in_seconds",summary.minimizer_time_in_seconds},
      {"total_time_in_seconds",summary.total_time_in_seconds},
      {"num_residual_evaluations",double(summary.num_residual_evaluations)},
      {"num_jacobian_evaluations",double(summary.num_jacobian_evaluations)},
      {"num_linear_solves",double(summary.num_linear_solves)},{"num_line_search_steps",double(summary.num_line_search_steps)},{"initial_cost",summary.initial_cost},{"final_cost",summary.final_cost},{"initial_gradient_max_norm",summary.iterations.empty()?NAN:summary.iterations.front().gradient_max_norm},{"initial_gradient_norm",summary.iterations.empty()?NAN:summary.iterations.front().gradient_norm},{"line_search_cost_evaluation_time_in_seconds",summary.line_search_cost_evaluation_time_in_seconds},{"line_search_gradient_evaluation_time_in_seconds",summary.line_search_gradient_evaluation_time_in_seconds},{"line_search_polynomial_minimization_time_in_seconds",summary.line_search_polynomial_minimization_time_in_seconds},{"line_search_total_time_in_seconds",summary.line_search_total_time_in_seconds},
      {"linear_solver_type_given",double(summary.linear_solver_type_given)},{"linear_solver_type_used",double(summary.linear_solver_type_used)},{"preconditioner_type_used",double(summary.preconditioner_type_used)},{"sparse_linear_algebra_library_type",double(summary.sparse_linear_algebra_library_type)},{"mixed_precision_solves_used",double(summary.mixed_precision_solves_used)},{"jacobi_column_scaling",double(jacobi_scaling_)},{"temporary_runtime_fixed_physical_blocks",double(temporary_fixed_blocks)},{"temporary_runtime_fixed_physical_scalars",double(temporary_fixed_scalars)},{"temporary_runtime_fixed_nuisance_blocks",double(temporary_fixed_nuisance_blocks)},{"explicit_initialization_allowlist_active",double(temporary_exact_allowlist_active_)},{"actual_budget_parameter_is_constant",double(problem.IsParameterBlockConstant(b))},{"explicit_initialization_free_blocks",double(temporary_exact_allowed_parameters_.size())},{"explicit_initialization_fixed_blocks",double(exact_initialization_fixed_blocks)},{"min_lm_diagonal",min_lm_diagonal_},{"inexact_linear_step_eta",options.eta},{"native_station_step_units",double(native_station_step_units_)},{"native_unit_station_blocks",double(native_unit_stations)},{"native_unit_CPU_derivative_rows",double(native_unit_CPU_derivatives)},{"native_unit_invalid_derivative_rows",double(native_unit_invalid_derivatives)},{"native_station_speed_min_px",station_speed_min},{"native_station_speed_max_px",station_speed_max},{"budget_positive_step_scale",budget_step_scale},{"num_threads_given",double(summary.num_threads_given)},{"num_threads_used",double(summary.num_threads_used)}});
    result.iterations = summary.iterations.size();
    result.accepted_state = callback.accepted_;
    // The last checkpoint is a complete accepted Ceres state.  Callback stop or
    // failure cannot expose an unaccepted trial or partially modified block set.
    builder.restore_state(result.accepted_state);
    result.message = summary.BriefReport();
    if (!callback.callback_error_.empty()) {
      result.status = InnerStatus::kFailed; result.message = "checkpoint callback failed: " + callback.callback_error_;
    } else if (callback.restart_) result.status = InnerStatus::kRootChartRestart;
    else if (callback.stopped_) result.status = InnerStatus::kStopped;
    else if (summary.termination_type == ceres::FAILURE || summary.termination_type == ceres::USER_FAILURE)
      result.status = InnerStatus::kFailed;
    else result.status = InnerStatus::kFinished; // Fresh outer diagnostics decide numerical success.
  } catch (const std::exception& error) {
    builder.restore_state(result.accepted_state);
    result.status = InnerStatus::kFailed; result.message = error.what();
  }
  return result;
}

}
