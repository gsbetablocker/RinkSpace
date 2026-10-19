#pragma once

// Independent v1.1 PHR driver contract.  This header has no Ceres/Eigen include;
// the reference tests compile with a C++17 compiler and the standard library.
// Ceres owns LM and box/manifold steps; THIS library owns general constraints.
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ceres { class Problem; class CostFunction; }

namespace rinkfit {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

double inequality_residual(double g, double lambda, double rho);
double equality_residual(double h, double nu, double rho);
double phase_one_budget_residual(double budget_sq);
double projected_box_gradient(double x, double gradient, double scale,
                              double lower, double upper);
double infinity_norm(const std::vector<double>& values);

struct PHRUpdate {
  std::vector<double> lambda, nu;
  double progress = 0.0;
  double next_rho = 1.0;
  bool penalty_limit_reached = false;
};
PHRUpdate update_phr(const std::vector<double>& g, const std::vector<double>& h,
                    const std::vector<double>& lambda_old, const std::vector<double>& nu_old,
                    double rho, double previous_progress, double rho_max,
                    double improvement_ratio = 0.5, double penalty_factor = 5.0);

enum class Phase { kFeasibility = 1, kPrecision = 2 };
enum class RunStatus {
  kUnfinished, kStopped, kNumericallyLimited, kInnerAccuracyUnfinished,
  kLocalStationaryOutsideBudget, kInvalidInitialState, kContractViolation,
  kReplayFailed, kPrecisionReady
};
const char* status_name(RunStatus status);

struct FrozenIdentity {
  std::string source_id;     // Full observation-array/key manifest SHA-256.
  std::string model_id;      // Material graph, optics and state-layout version.
  std::string budget_id;     // Frozen native E/profile/query provenance hash.
  std::string statistics_id; // Frozen W/sigma/common-mode choice.
  std::string scales_id;     // All variable/constraint scales.
  std::string objective_id;  // v1.1 objective semantics and configuration ID.
  bool operator==(const FrozenIdentity& rhs) const;
};

struct ParameterValues {
  std::string name;
  std::vector<double> values;
  bool equality_auxiliary = false; // Only explicit profile/RS witnesses.
};
struct StatePayload {
  FrozenIdentity identity;
  std::vector<ParameterValues> parameters;
  // Must cover all stations, active chart/branch IDs, auxiliary-root witnesses,
  // optics choice, shared shape topology and every other non-parameter state.
  std::map<std::string, std::string> discrete;
  std::map<std::string, std::string> root_charts; // May change in equivalent restart.
};

class StateSnapshot {
 public:
  StateSnapshot() = default;
  explicit StateSnapshot(StatePayload data); // Deep ownership, finite doubles.
  const StatePayload& data() const;
  const std::string& token() const;
  bool empty() const { return !data_; }
 private:
  std::shared_ptr<const StatePayload> data_;
  std::string token_; // Deterministic local corruption/freshness token, NOT SHA-256.
};

enum class ObservationSemantics { kOrdinary2D, kGenuineProfile };
struct ObservationSpec {
  std::string key; // shot/source clock + native PTS + original source point ID.
  std::array<double, 2> source_uv{};
  ObservationSemantics semantics = ObservationSemantics::kOrdinary2D;
  std::array<double, 4> E{4., 0., 0., 4.}; // Row-major, frozen native px^2.
  std::array<double, 2> profile_normal{0., 0.};
  bool original_curved_query = false;
  bool explicit_profile_query = false;
  std::array<double, 2> profile_query_uv{0., 0.}; // Origin differs from measured source_uv.
  double profile_radius_px = 0.;
  double profile_alpha_lower_px = 0., profile_alpha_upper_px = 0.;
  bool active = true; // Raw source identity remains in the full ledger.
  std::string exclusion_reason, exclusion_id;
};

struct NativeRow {
  std::string key;
  std::array<double, 2> predicted_uv{kInfinity, kInfinity};
  double station = 0.;
  std::string branch_id;
  bool projection_valid = false;
  bool finite_material_and_order_valid = false;
  bool true_root_valid = false; // Actual non-singular implicit/query replay.
  double root_witness_error_px = 0.; // Lifted RS/query witness versus true replay.
  double profile_query_error_px = 0.; // Exact immutable curved Q(alpha) replay.
  bool has_profile_witness = false;
  double profile_witness_alpha_px = 0.;
  std::string failure_reason;
  bool excluded = false;
  std::string exclusion_reason;
};

struct NativeReplay {
  std::string state_token;
  std::vector<NativeRow> rows;
  bool exact_physical_graph = false;
  bool physical_domain_valid = false;
  bool all_original_support_relations_valid = false;
  double statistical_pixel_cost = kInfinity;
  double measured_prior_cost = 0.;
  double smoothness_cost = 0.;
  std::vector<std::string> physical_failures;
};

struct NativeAudit {
  std::vector<double> q;
  std::vector<std::array<double, 2>> raw_errors;
  std::vector<double> query_error_px;
  std::vector<std::size_t> invalid_rows, over_budget_rows, excluded_rows;
  std::size_t expected_active_rows = 0, accounted_active_rows = 0;
  std::size_t expected_rows = 0, accounted_rows = 0;
  double t_hat = kInfinity;
  bool identity_complete = false;
  bool pixel_pass = false, support_pass = false, physical_pass = false;
  bool passed() const { return identity_complete && pixel_pass && support_pass && physical_pass; }
};

NativeAudit audit_native(const StateSnapshot& snapshot,
                         const std::vector<ObservationSpec>& observations,
                         const NativeReplay& replay, double q_tolerance,
                         double equality_tolerance_px);

struct ConstraintSchema {
  // First pixel_count inequalities are active q_i-b in original source order.
  // Later inequalities/equalities use separately frozen scales, never b.
  std::vector<std::string> inequality_ids, equality_ids;
  std::size_t pixel_count = 0;
};
struct ConstraintValues {
  std::string state_token;
  std::vector<double> g, h; // Current true AL-chart values, not weighted residuals.
};

struct GradientReport {
  std::string state_token;
  std::string scales_id;
  // Fresh gradient of F_phase + lambda*g + nu*h at this exact state, projected
  // in the actual scaled tangent/active box cone.  Fixed variables contribute 0.
  std::vector<double> scaled_projected_components;
  bool complete_current_derivatives = false;
};

struct RuntimeLimits {
  double rho_initial = 1., rho_max = 1e6;
  double penalty_factor = 5., improvement_ratio = 0.5;
  std::size_t max_outer_iterations_per_phase = 24;
  std::size_t max_inner_iterations = 100;
  std::size_t max_inner_attempts_per_outer = 2;
  double max_total_seconds = 3600.;
  double inner_stationarity_early = 1e-3, inner_stationarity_late = 1e-6;
  std::size_t tighten_after_outer = 4;
  double primal_tolerance = 1e-4, equality_tolerance = 0.01;
  double complementarity_tolerance = 1e-5, stationarity_tolerance = 1e-5;
  double native_q_tolerance = 1e-4, native_equality_tolerance_px = 0.01;
  std::size_t max_chart_restarts_per_outer = 4;
  int num_threads = 1;
  bool log_ceres_progress = false;
  bool inequality_kink_active = false; // Clarke residual Jacobian sqrt(rho) at exactly lambda+rho*g==0.
  void validate() const;
  std::string fingerprint() const;
};

struct OuterState {
  Phase phase = Phase::kFeasibility;
  std::vector<double> lambda, nu;
  double rho = 1., previous_progress = kInfinity;
  std::size_t outer_iteration = 0;
  std::size_t inner_attempt = 0;
  std::size_t chart_restarts = 0;
  double elapsed_seconds = 0.;
};

struct ObjectiveContext {
  Phase phase;
  std::vector<double> lambda, nu; // Immutable values copied per inner solve.
  double rho;
  std::string objective_token;
  bool inequality_kink_active = false;
};

// Build inventory lets the backend reject missing/robustified AL factors.
// Handles are ceres::ResidualBlockId, represented here without Ceres headers.
struct BuildInventory {
  std::vector<void*> inequality_blocks, equality_blocks;
  std::vector<void*> statistical_blocks, measured_prior_blocks, smoothness_blocks;
  bool exact_final_physical_graph = false;
  bool common_mode_active = false;
};

struct ParameterBlockBinding {
  std::string name;
  double* data = nullptr;
  std::size_t size = 0;
  bool equality_auxiliary = false;
};

class ProblemBuilder {
 public:
  virtual ~ProblemBuilder() = default;
  virtual const FrozenIdentity& identity() const = 0;
  virtual const std::vector<ObservationSpec>& observations() const = 0;
  virtual const ConstraintSchema& schema() const = 0;
  // Stable address; included in capture_state() as budget_sq; size one, b >= 0.
  virtual double* budget_parameter() = 0;
  // Every Ceres block (free or fixed) is registered and deep-snapshotted by name.
  virtual std::vector<ParameterBlockBinding> parameter_blocks() = 0;
  virtual StateSnapshot capture_state() const = 0;
  virtual void restore_state(const StateSnapshot& snapshot) = 0;
  // Must replay the supplied frozen state, not mutable current buffers or W/cost.
  virtual NativeReplay replay_native(const StateSnapshot& snapshot) const = 0;
  virtual ConstraintValues evaluate_constraints(const StateSnapshot& snapshot) const = 0;
  virtual GradientReport lagrangian_gradient(const StateSnapshot& snapshot, Phase phase,
      const std::vector<double>& lambda, const std::vector<double>& nu) const = 0;
  // All current phase variables, bounds, manifolds and full original factors.
  // No cost term is added for phase I here; the backend adds sqrt(2*b+1).
  virtual BuildInventory build_ceres_problem(ceres::Problem& problem,
                                           const ObjectiveContext& objective) = 0;
  // Optional chart restart: construct equivalent explicit query/RS witnesses at
  // the same physical state.  No camera/geometry/S recovery edit is permitted.
  virtual bool root_chart_restart_requested() const { return false; }
  virtual bool rebuild_equivalent_root_chart() { return false; }
};

struct IterationRecord {
  int iteration = 0;
  bool step_accepted = false;
  double inner_cost = kInfinity;
  double actual_cost_change = 0.;
  double relative_decrease = 0.; // Ceres actual/model decrease for same objective.
  double trust_region_radius = 0.;
  std::string objective_token;
  bool inequality_kink_active = false;
  double iteration_time_in_seconds = 0.;
  double step_solver_time_in_seconds = 0.;
  double cumulative_time_in_seconds = 0.;
};
struct RunHooks {
  std::function<bool()> stop_requested;
  // Receives a deep immutable accepted state; useful for atomic disk checkpoints.
  std::function<void(const StateSnapshot&, const OuterState&, const IterationRecord&)> on_iteration;
  std::function<void(const StateSnapshot&, const OuterState&, const NativeReplay&,
                     const NativeAudit&, const std::string&)> on_outer_checkpoint;
  std::function<void(const OuterState&, const std::string&, const std::map<std::string,double>&)> on_solver_summary;
};

enum class InnerStatus { kFinished, kStopped, kFailed, kRootChartRestart };
struct InnerResult {
  InnerStatus status = InnerStatus::kFailed;
  StateSnapshot accepted_state;
  std::size_t iterations = 0;
  std::string message;
};
class InnerSolver {
 public:
  virtual ~InnerSolver() = default;
  virtual InnerResult solve(ProblemBuilder& builder, const ObjectiveContext& objective,
                           const OuterState& outer, const RuntimeLimits& limits,
                           const RunHooks& hooks, double remaining_seconds) = 0;
};

// Concrete implementation is linked only in the Ceres target.
class CeresInnerSolver final : public InnerSolver {
 public:
  InnerResult solve(ProblemBuilder&, const ObjectiveContext&, const OuterState&,
                    const RuntimeLimits&, const RunHooks&, double remaining_seconds) override;
};

// A scalar CostFunction supplied here must evaluate the *unweighted* g/h and
// its full block Jacobians.  Wrapper owns it; Ceres owns wrapper.  Loss=nullptr.
// Optional value evaluation on the same Jet scalar path as full AD.
// Generic costs without this interface always retain the complete old AD.
class PHRADValueProbe {
 public:
  virtual ~PHRADValueProbe() = default;
  virtual bool EvaluateADValue(double const* const* parameters, double* residuals) const = 0;
};
std::unique_ptr<ceres::CostFunction> make_phr_cost(std::unique_ptr<ceres::CostFunction> constraint, double multiplier, double rho, bool equality, bool active_at_kink = false);
void* add_phr_inequality(ceres::Problem& problem,
                        std::unique_ptr<ceres::CostFunction> constraint,
                        const std::vector<double*>& parameter_blocks,
                        double lambda, double rho, bool active_at_kink = false);
void* add_phr_equality(ceres::Problem& problem,
                      std::unique_ptr<ceres::CostFunction> constraint,
                      const std::vector<double*>& parameter_blocks,
                      double nu, double rho);

struct KKTReport {
  double primal_violation = kInfinity, equality_violation = kInfinity;
  double negative_multiplier = kInfinity, complementarity = kInfinity;
  double scaled_stationarity = kInfinity, budget_stationarity = kInfinity;
  bool fresh = false;
  bool passed(const RuntimeLimits& limits, Phase phase) const;
};

struct RunCheckpoint {
  StateSnapshot state;
  StateSnapshot best_physical_state; // Replayed on resume; never spliced by frame.
  OuterState outer; // Resume this objective; no multiplier update was assumed.
  std::string runtime_fingerprint;
};
void write_checkpoint(std::ostream& stream, const RunCheckpoint& checkpoint);
RunCheckpoint read_checkpoint(std::istream& stream);

struct RunResult {
  RunStatus status = RunStatus::kUnfinished;
  std::string detail;
  RunCheckpoint checkpoint;
  StateSnapshot best_physical_state;
  NativeReplay final_replay;
  NativeAudit final_audit;
  KKTReport numerical;
  bool phase_two_completed = false;
  // S6 profiles/grid/overlay/export are separate runner obligations, even when
  // this constrained solve returns kPrecisionReady.
};

RunResult solve_two_phase(ProblemBuilder& builder, InnerSolver& inner,
                         const RuntimeLimits& limits, const RunHooks& hooks = {},
                         const RunCheckpoint* resume = nullptr);

} // namespace rinkfit
