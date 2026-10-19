#include "rinkfit/augmented_lagrangian.hpp"

#include <ceres/ceres.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <set>
#include <unordered_map>
#include <stdexcept>

namespace rinkfit {
namespace {

class PHRFactor final : public ceres::CostFunction {
 public:
  PHRFactor(std::unique_ptr<ceres::CostFunction> source, double multiplier, double rho, bool equality, bool active_at_kink = false)
      : source_(std::move(source)), multiplier_(multiplier), rho_(rho), equality_(equality), active_at_kink_(active_at_kink) {
    if (!source_ || source_->num_residuals() != 1 || !std::isfinite(multiplier_)
        || (!equality_ && multiplier_ < 0.) || !std::isfinite(rho_) || rho_ <= 0.)
      throw std::invalid_argument("PHR requires one true scalar constraint, finite multiplier and rho>0");
    set_num_residuals(1);
    *mutable_parameter_block_sizes() = source_->parameter_block_sizes();
  }
  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
    // Always evaluate the original constraint's arithmetic, including inactive
    // inequalities. On an inactive PHR branch its composite residual is exactly
    // zero, so its exact Jacobian is zero without differentiating the source.
    // Active constraints/equalities retain the original AD and finite-J checks.
    bool any_jacobian = false;
    if (jacobians) for (std::size_t i = 0; i < parameter_block_sizes().size(); ++i) any_jacobian = any_jacobian || jacobians[i] != nullptr;
    const auto* probe = dynamic_cast<const PHRADValueProbe*>(source_.get());
    if (!equality_ && any_jacobian && probe) {
      double raw;
      // Jet scalar arithmetic matches the old full-AD scalar path; a separate
      // double guard still evaluates the original value-only arithmetic.
      double original_value;
      if (!source_->Evaluate(parameters, &original_value, nullptr) || !std::isfinite(original_value)
          || !probe->EvaluateADValue(parameters, &raw) || !std::isfinite(raw)) return false;
      const double branch = multiplier_ + rho_ * raw;
      if (!std::isfinite(branch)) return false;
      if (branch < 0. || (branch == 0. && !active_at_kink_)) {
        residuals[0] = 0.;
        for (std::size_t i = 0; i < parameter_block_sizes().size(); ++i)
          if (jacobians[i]) std::fill_n(jacobians[i], parameter_block_sizes()[i], 0.);
        return true;
      }
    }
    if (!source_->Evaluate(parameters, residuals, jacobians) || !std::isfinite(residuals[0])) return false;
    const double a = multiplier_ + rho_ * residuals[0];
    if (!std::isfinite(a)) return false;
    // At a==0 this selects a declared Clarke residual Jacobian.
    // Residual values and the exact C1 AL objective gradient are unchanged.
    const double scale = equality_ || a > 0. || (active_at_kink_ && a == 0.) ? std::sqrt(rho_) : 0.;
    residuals[0] = (equality_ ? a : std::max(0., a)) / std::sqrt(rho_);
    if (jacobians) {
      for (std::size_t i = 0; i < parameter_block_sizes().size(); ++i) {
        if (!jacobians[i]) continue;
        for (int j = 0; j < parameter_block_sizes()[i]; ++j) {
          if (!std::isfinite(jacobians[i][j])) return false;
          jacobians[i][j] *= scale;
        }
      }
    }
    return std::isfinite(residuals[0]);
  }
 private:
  std::unique_ptr<ceres::CostFunction> source_;
  double multiplier_, rho_;
  bool equality_, active_at_kink_;
};

struct BudgetFactor {
  template <typename T> bool operator()(const T* b, T* residual) const {
    if (b[0] < T(0.)) return false;
    using std::sqrt;
    residual[0] = sqrt(T(2.) * b[0] + T(1.));
    return true;
  }
};

void validate_problem(ceres::Problem& problem, ProblemBuilder& builder,
                       const BuildInventory& inventory, const ObjectiveContext& context,
                       void* budget_residual) {
  const auto& schema = builder.schema();
  if (!inventory.exact_final_physical_graph)
    throw std::invalid_argument("builder did not attest an exact final physical graph without delta/image warp");
  if (inventory.inequality_blocks.size() != schema.inequality_ids.size()
      || inventory.equality_blocks.size() != schema.equality_ids.size())
    throw std::invalid_argument("AL factor inventory differs from complete constraint schema");
  if (context.phase == Phase::kFeasibility
      && (!inventory.statistical_blocks.empty() || !inventory.measured_prior_blocks.empty()
          || !inventory.smoothness_blocks.empty() || inventory.common_mode_active))
    throw std::invalid_argument("phase I must contain only budget objective and required constraints");
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

class SnapshotCallback final : public ceres::IterationCallback {
 public:
  SnapshotCallback(ProblemBuilder& builder, const ObjectiveContext& context, const OuterState& outer,
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

} // namespace

std::unique_ptr<ceres::CostFunction> make_phr_cost(std::unique_ptr<ceres::CostFunction> constraint,
                                                    double multiplier, double rho, bool equality, bool active_at_kink) {
  return std::make_unique<PHRFactor>(std::move(constraint), multiplier, rho, equality, active_at_kink);
}
void* add_phr_inequality(ceres::Problem& problem, std::unique_ptr<ceres::CostFunction> constraint,
                        const std::vector<double*>& blocks, double lambda, double rho, bool active_at_kink) {
  return problem.AddResidualBlock(make_phr_cost(std::move(constraint), lambda, rho, false, active_at_kink).release(), nullptr, blocks);
}
void* add_phr_equality(ceres::Problem& problem, std::unique_ptr<ceres::CostFunction> constraint,
                      const std::vector<double*>& blocks, double nu, double rho) {
  return problem.AddResidualBlock(make_phr_cost(std::move(constraint), nu, rho, true).release(), nullptr, blocks);
}

InnerResult CeresInnerSolver::solve(ProblemBuilder& builder, const ObjectiveContext& objective,
                                    const OuterState& outer, const RuntimeLimits& limits,
                                    const RunHooks& hooks, double remaining_seconds) {
  InnerResult result;
  result.accepted_state = builder.capture_state();
  try {
    limits.validate();
    if (!ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
      throw std::runtime_error("production backend requires Ceres built with SuiteSparse; no dense/SciPy fallback");
    if (limits.max_inner_iterations > INT_MAX || remaining_seconds <= 0.)
      throw std::invalid_argument("invalid Ceres inner iteration/time budget");
    ceres::Problem problem;
    double* b = builder.budget_parameter();
    problem.AddParameterBlock(b, 1);
    problem.SetParameterLowerBound(b, 0, 0.);
    const BuildInventory inventory = builder.build_ceres_problem(problem, objective);
    void* budget_residual = nullptr;
    if (objective.phase == Phase::kFeasibility) {
      problem.SetParameterBlockVariable(b);
      budget_residual = problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BudgetFactor, 1, 1>(new BudgetFactor), nullptr, b);
    } else {
      if (*b != 1.) throw std::invalid_argument("phase II must already have exactly b=1 before building factors");
      problem.SetParameterBlockConstant(b);
    }
    validate_problem(problem, builder, inventory, objective, budget_residual);
    SnapshotCallback callback(builder, objective, outer, hooks);
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
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
      {"num_linear_solves",double(summary.num_linear_solves)},
      {"num_threads_given",double(summary.num_threads_given)},{"num_threads_used",double(summary.num_threads_used)}});
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

} // namespace rinkfit
