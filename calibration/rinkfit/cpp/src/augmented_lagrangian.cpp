#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/state_text_codec.hpp"
#include <unordered_set>
#include <string_view>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace rinkfit {
namespace {

void require(bool condition, const std::string& reason) {
  if (!condition) throw std::invalid_argument(reason);
}
void positive(double x, const char* name) {
  require(std::isfinite(x) && x > 0., std::string(name) + " must be finite and positive");
}
void finite_vector(const std::vector<double>& x, const char* name) {
  for (double v : x) require(std::isfinite(v), std::string(name) + " is non-finite");
}
std::string hash_text(const std::string& text) {
  // FNV-1a is only a deterministic freshness/corruption check.  Source and model
  // SHA-256 identities are supplied by the frozen manifest/adapter.
  std::uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : text) { h ^= c; h *= 1099511628211ULL; }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << h;
  return out.str();
}
void put_identity(std::ostream& out, const FrozenIdentity& id) {
  out << std::quoted(id.source_id) << ' ' << std::quoted(id.model_id) << ' '
      << std::quoted(id.budget_id) << ' ' << std::quoted(id.statistics_id) << ' '
      << std::quoted(id.scales_id) << ' ' << std::quoted(id.objective_id) << '\n';
}
std::string canonical(const StatePayload& data) {
 state_text::Writer out;out.text.reserve(data.parameters.size()*64);
 const auto&id=data.identity;const std::string* ids[]={&id.source_id,&id.model_id,&id.budget_id,&id.statistics_id,&id.scales_id,&id.objective_id};
 for(int i=0;i<6;++i){out.quoted(*ids[i]);out.ch(i==5?'\n':' ');}out.integer(data.parameters.size());out.ch('\n');
 for(const auto&p:data.parameters){out.quoted(p.name);out.ch(' ');out.integer(int(p.equality_auxiliary));out.ch(' ');out.integer(p.values.size());for(double v:p.values){out.ch(' ');out.number(v,true);}out.ch('\n');}
 for(const auto* map:{&data.discrete,&data.root_charts}){out.integer(map->size());out.ch('\n');for(const auto&item:*map){out.quoted(item.first);out.ch(' ');out.quoted(item.second);out.ch('\n');}}
 return std::move(out.text);
}

double max_positive(const std::vector<double>& values) {
  double result = 0.;
  for (double v : values) result = std::max(result, v);
  return result;
}
void check_schema(const ProblemBuilder& model) {
  const auto& schema = model.schema();
  require(!model.observations().empty(), "an empty observation set cannot be fitted");
  std::size_t active_count = 0;
  for (const auto& o : model.observations()) {
    if (o.active) ++active_count;
    else require(!o.exclusion_reason.empty() && !o.exclusion_id.empty(), "excluded source lacks immutable authorization identity/reason");
  }
  require(active_count > 0 && schema.pixel_count == active_count, "schema must retain every active original pixel obligation");
  require(schema.inequality_ids.size() >= schema.pixel_count, "missing pixel constraints");
  std::set<std::string> keys;
  for (const auto& o : model.observations())
    require(!o.key.empty() && keys.insert(o.key).second, "original composite observation key is duplicated/empty");
  for (const auto* ids : {&schema.inequality_ids, &schema.equality_ids}) {
    keys.clear();
    for (const auto& id : *ids) require(!id.empty() && keys.insert(id).second, "constraint IDs are duplicated/empty");
  }
}
void check_values(const ConstraintValues& values, const StateSnapshot& state,
                  const ConstraintSchema& schema) {
  require(values.state_token == state.token(), "constraint values are not bound to the current complete snapshot");
  require(values.g.size() == schema.inequality_ids.size() && values.h.size() == schema.equality_ids.size(),
          "constraint cardinality changed without an explicit chart restart");
  finite_vector(values.g, "g"); finite_vector(values.h, "h");
}
double check_gradient(const GradientReport& gradient, const StateSnapshot& state) {
  require(gradient.complete_current_derivatives && gradient.state_token == state.token()
          && gradient.scales_id == state.data().identity.scales_id,
          "missing/stale/incompletely scaled true tangent gradient");
  require(!gradient.scaled_projected_components.empty(), "empty gradient is not a stationarity certificate");
  finite_vector(gradient.scaled_projected_components, "scaled tangent gradient");
  return infinity_norm(gradient.scaled_projected_components);
}
bool same_physical_state(const StateSnapshot& old_state, const StateSnapshot& new_state) {
  const auto& a = old_state.data(); const auto& b = new_state.data();
  if (!(a.identity == b.identity) || a.discrete != b.discrete) return false;
  std::map<std::string, std::vector<double>> pa, pb;
  for (const auto& p : a.parameters) if (!p.equality_auxiliary) pa[p.name] = p.values;
  for (const auto& p : b.parameters) if (!p.equality_auxiliary) pb[p.name] = p.values;
  return pa == pb;
}
std::vector<double> remap(const std::vector<std::string>& old_ids, const std::vector<double>& old_values,
                          const std::vector<std::string>& new_ids) {
  std::map<std::string, double> old;
  for (std::size_t i = 0; i < old_ids.size(); ++i) old[old_ids[i]] = old_values.at(i);
  std::vector<double> result;
  for (const auto& id : new_ids) result.push_back(old.count(id) ? old.at(id) : 0.);
  return result;
}

} // namespace

double inequality_residual(double g, double lambda, double rho) {
  positive(rho, "rho");
  require(std::isfinite(g) && std::isfinite(lambda) && lambda >= 0., "invalid inequality value/multiplier");
  return std::max(0., lambda + rho * g) / std::sqrt(rho);
}
double equality_residual(double h, double nu, double rho) {
  positive(rho, "rho"); require(std::isfinite(h) && std::isfinite(nu), "invalid equality value/multiplier");
  return std::sqrt(rho) * (h + nu / rho);
}
double phase_one_budget_residual(double b) {
  require(std::isfinite(b) && b >= 0., "budget_sq must be finite and nonnegative");
  return std::sqrt(2. * b + 1.);
}
double projected_box_gradient(double x, double grad, double scale, double lo, double hi) {
  positive(scale, "variable scale");
  require(std::isfinite(x) && std::isfinite(grad) && !std::isnan(lo) && !std::isnan(hi)
          && lo <= x && x <= hi, "invalid current variable, gradient or box");
  const double z = x / scale;
  return z - std::clamp(z - scale * grad, lo / scale, hi / scale);
}
double infinity_norm(const std::vector<double>& values) {
  double norm = 0.;
  for (double value : values) {
    if (!std::isfinite(value)) return kInfinity;
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}
PHRUpdate update_phr(const std::vector<double>& g, const std::vector<double>& h,
                    const std::vector<double>& old_lam, const std::vector<double>& old_nu,
                    double rho, double previous, double rho_max,
                    double improvement, double factor) {
  positive(rho, "rho"); positive(rho_max, "rho_max");
  require(rho <= rho_max && !std::isnan(previous) && previous >= 0., "invalid PHR history/cap");
  require(improvement > 0. && improvement < 1. && std::isfinite(factor) && factor > 1., "invalid PHR schedule");
  require(g.size() == old_lam.size() && h.size() == old_nu.size(), "PHR dimensions disagree");
  finite_vector(g, "g"); finite_vector(h, "h");
  finite_vector(old_lam, "lambda"); finite_vector(old_nu, "nu");
  PHRUpdate out;
  out.next_rho = rho;
  for (std::size_t i = 0; i < g.size(); ++i) {
    require(old_lam[i] >= 0., "negative inequality multiplier");
    out.progress = std::max(out.progress, std::abs(std::max(g[i], -old_lam[i] / rho)));
    out.lambda.push_back(std::max(0., old_lam[i] + rho * g[i]));
  }
  for (std::size_t j = 0; j < h.size(); ++j) {
    out.progress = std::max(out.progress, std::abs(h[j]));
    out.nu.push_back(old_nu[j] + rho * h[j]);
  }
  if (out.progress > improvement * previous) {
    out.penalty_limit_reached = rho > rho_max / factor;
    if (!out.penalty_limit_reached) out.next_rho = rho * factor;
  }
  finite_vector(out.lambda, "updated lambda"); finite_vector(out.nu, "updated nu");
  return out;
}

const char* status_name(RunStatus s) {
  switch (s) {
    case RunStatus::kUnfinished: return "UNFINISHED";
    case RunStatus::kStopped: return "STOPPED_WITH_COMPLETE_CHECKPOINT";
    case RunStatus::kNumericallyLimited: return "NUMERICALLY_LIMITED";
    case RunStatus::kInnerAccuracyUnfinished: return "INNER_ACCURACY_UNFINISHED";
    case RunStatus::kLocalStationaryOutsideBudget: return "LOCAL_STATIONARY_OUTSIDE_UNIT_BUDGET";
    case RunStatus::kInvalidInitialState: return "INVALID_INITIAL_STATE";
    case RunStatus::kContractViolation: return "CONTRACT_VIOLATION";
    case RunStatus::kReplayFailed: return "INDEPENDENT_NATIVE_REPLAY_FAILED";
    case RunStatus::kPrecisionReady: return "PRECISION_READY_FOR_S6_EXPORTS";
  }
  return "UNKNOWN";
}
bool FrozenIdentity::operator==(const FrozenIdentity& b) const {
  return source_id == b.source_id && model_id == b.model_id && budget_id == b.budget_id
      && statistics_id == b.statistics_id && scales_id == b.scales_id && objective_id == b.objective_id;
}
StateSnapshot::StateSnapshot(StatePayload data) {
  const auto& id = data.identity;
  require(!id.source_id.empty() && !id.model_id.empty() && !id.budget_id.empty()
          && !id.statistics_id.empty() && !id.scales_id.empty() && !id.objective_id.empty(),
          "complete frozen problem identity is required");
  require(!data.parameters.empty(), "empty state cannot be snapshotted");
  std::set<std::string> names;
  for (const auto& p : data.parameters) {
    require(!p.name.empty() && names.insert(p.name).second && !p.values.empty(), "invalid/duplicate parameter block");
    finite_vector(p.values, "snapshot parameters");
  }
  token_ = hash_text(canonical(data));
  data_ = std::make_shared<const StatePayload>(std::move(data));
}
const StatePayload& StateSnapshot::data() const {
  require(!empty(), "empty state snapshot"); return *data_;
}
const std::string& StateSnapshot::token() const {
  require(!empty(), "empty state snapshot"); return token_;
}

NativeAudit audit_native(const StateSnapshot& state, const std::vector<ObservationSpec>& observations,
                         const NativeReplay& replay, double q_tol, double eq_tol) {
  require(std::isfinite(q_tol) && q_tol >= 0. && std::isfinite(eq_tol) && eq_tol >= 0., "invalid audit tolerances");
  NativeAudit out;
  out.expected_rows = observations.size(); out.accounted_rows = replay.rows.size();
  out.q.assign(observations.size(), kInfinity);
  out.raw_errors.assign(observations.size(), {kInfinity, kInfinity});
  out.query_error_px.assign(observations.size(), 0.);
  out.identity_complete = !observations.empty() && replay.state_token == state.token()
                       && observations.size() == replay.rows.size();
  out.support_pass = out.identity_complete && replay.all_original_support_relations_valid;
  out.physical_pass = replay.exact_physical_graph && replay.physical_domain_valid;
  std::set<std::string> seen;
  double max_q = 0.;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const auto& spec = observations[i];
    require(!spec.key.empty() && seen.insert(spec.key).second, "duplicate/empty composite source key");
    require(std::isfinite(spec.source_uv[0]) && std::isfinite(spec.source_uv[1]), "non-finite original UV");
    if (!spec.active) {
      out.excluded_rows.push_back(i);
      if (spec.exclusion_reason.empty() || spec.exclusion_id.empty() || i >= replay.rows.size()
          || replay.rows[i].key != spec.key || !replay.rows[i].excluded
          || replay.rows[i].exclusion_reason != spec.exclusion_reason) out.identity_complete = false;
      continue; // Not a fabricated zero residual or an invalid active source.
    }
    ++out.expected_active_rows;
    if (i < replay.rows.size() && replay.rows[i].key == spec.key && !replay.rows[i].excluded)
      ++out.accounted_active_rows;
    bool valid = i < replay.rows.size();
    bool support = valid;
    if (valid) {
      const auto& row = replay.rows[i];
      if (row.key != spec.key || row.excluded) out.identity_complete = false;
      valid = row.key == spec.key && row.projection_valid && row.true_root_valid
           && std::isfinite(row.predicted_uv[0]) && std::isfinite(row.predicted_uv[1]);
      support = row.finite_material_and_order_valid && std::isfinite(row.root_witness_error_px)
             && row.root_witness_error_px >= 0. && row.root_witness_error_px <= eq_tol;
      const double ex = row.predicted_uv[0] - spec.source_uv[0];
      const double ey = row.predicted_uv[1] - spec.source_uv[1];
      out.raw_errors[i] = {ex, ey};
      if (spec.semantics == ObservationSemantics::kOrdinary2D) {
        for (double v : spec.E) require(std::isfinite(v), "non-finite frozen E");
        require(std::abs(spec.E[1] - spec.E[2]) <= 1e-12, "E must be symmetric");
        const double det = spec.E[0] * spec.E[3] - spec.E[1] * spec.E[2];
        require(spec.E[0] > 0. && spec.E[3] > 0. && det > 0. && std::isfinite(det), "E must be positive definite");
        if (valid) out.q[i] = (spec.E[3] * ex * ex - 2. * spec.E[1] * ex * ey + spec.E[0] * ey * ey) / det;
      } else {
        const double nx = spec.profile_normal[0], ny = spec.profile_normal[1];
        require(std::isfinite(nx) && std::isfinite(ny) && std::abs(nx * nx + ny * ny - 1.) <= 1e-10,
                "profile normal must be an already frozen unit vector");
        positive(spec.profile_radius_px, "profile pixel radius");
        require(std::isfinite(spec.profile_alpha_lower_px) && std::isfinite(spec.profile_alpha_upper_px)
                && spec.profile_alpha_lower_px <= spec.profile_alpha_upper_px, "finite original profile query extent required");
        if (valid) {
          const double en = nx * ex + ny * ey;
          const auto& query_uv = spec.explicit_profile_query ? spec.profile_query_uv : spec.source_uv;
          require(std::isfinite(query_uv[0]) && std::isfinite(query_uv[1]), "non-finite original profile query origin");
          const double qx = row.predicted_uv[0] - query_uv[0], qy = row.predicted_uv[1] - query_uv[1];
          const double query_alpha = nx*qx + ny*qy;
          double query = spec.original_curved_query ? row.profile_query_error_px : std::hypot(qx - query_alpha * nx, qy - query_alpha * ny);
          if (!spec.original_curved_query) support = support && query_alpha >= spec.profile_alpha_lower_px - eq_tol && query_alpha <= spec.profile_alpha_upper_px + eq_tol;
          if (row.has_profile_witness) {
            const double alpha = row.profile_witness_alpha_px;
            const double witness_error = spec.original_curved_query ? row.profile_query_error_px : std::hypot(qx - alpha * nx, qy - alpha * ny);
            query = std::max(query, witness_error);
            support = support && std::isfinite(alpha) && std::isfinite(witness_error)
                    && alpha >= spec.profile_alpha_lower_px - eq_tol && alpha <= spec.profile_alpha_upper_px + eq_tol;
          }
          out.query_error_px[i] = query;
          support = support && std::isfinite(query) && query <= eq_tol;
          out.q[i] = (en / spec.profile_radius_px) * (en / spec.profile_radius_px);
        }
      }
    }
    if (!valid || !std::isfinite(out.q[i]) || out.q[i] < 0.) out.q[i] = kInfinity;
    if (!valid || !support) { out.invalid_rows.push_back(i); out.support_pass = false; }
    if (out.q[i] > 1. + q_tol) out.over_budget_rows.push_back(i);
    max_q = std::max(max_q, out.q[i]);
  }
  out.identity_complete = out.identity_complete && out.expected_active_rows > 0
      && out.accounted_active_rows == out.expected_active_rows;
  out.t_hat = out.identity_complete ? std::sqrt(max_q) : kInfinity;
  out.pixel_pass = out.identity_complete && out.over_budget_rows.empty();
  out.support_pass = out.support_pass && out.identity_complete;
  return out;
}

void RuntimeLimits::validate() const {
  positive(rho_initial, "rho_initial"); positive(rho_max, "rho_max");
  require(rho_initial <= rho_max && penalty_factor > 1. && std::isfinite(penalty_factor)
          && improvement_ratio > 0. && improvement_ratio < 1., "invalid frozen PHR schedule");
  require(max_outer_iterations_per_phase && max_inner_iterations && max_inner_attempts_per_outer
          && max_chart_restarts_per_outer && num_threads > 0, "run budgets/thread count must be positive");
  positive(max_total_seconds, "max_total_seconds");
  for (double t : {inner_stationarity_early, inner_stationarity_late, primal_tolerance,
                  equality_tolerance, complementarity_tolerance, stationarity_tolerance,
                  native_q_tolerance, native_equality_tolerance_px}) positive(t, "arithmetic tolerance");
  require(inner_stationarity_late <= inner_stationarity_early, "late inner tolerance must tighten");
}
std::string RuntimeLimits::fingerprint() const {
  std::ostringstream s; s.imbue(std::locale::classic()); s << std::hexfloat;
  s << rho_initial << ' ' << rho_max << ' ' << penalty_factor << ' ' << improvement_ratio << ' '
    << max_outer_iterations_per_phase << ' ' << max_inner_iterations << ' ' << max_inner_attempts_per_outer << ' '
    << max_total_seconds << ' ' << inner_stationarity_early << ' ' << inner_stationarity_late << ' '
    << tighten_after_outer << ' ' << primal_tolerance << ' ' << equality_tolerance << ' '
    << complementarity_tolerance << ' ' << stationarity_tolerance << ' ' << native_q_tolerance << ' '
    << native_equality_tolerance_px << ' ' << max_chart_restarts_per_outer << ' ' << num_threads << ' ' << log_ceres_progress;
  if (inequality_kink_active) s << " inequality_kink_active=1"; // Preserve existing zero-kink checkpoint fingerprints.
  return hash_text(s.str());
}
bool KKTReport::passed(const RuntimeLimits& l, Phase phase) const {
  return fresh && primal_violation <= l.primal_tolerance && equality_violation <= l.equality_tolerance
      && negative_multiplier <= l.complementarity_tolerance && complementarity <= l.complementarity_tolerance
      && scaled_stationarity <= l.stationarity_tolerance
      && (phase == Phase::kPrecision || budget_stationarity <= l.stationarity_tolerance);
}

RunResult solve_two_phase(ProblemBuilder& builder, InnerSolver& inner, const RuntimeLimits& limits,
                         const RunHooks& hooks, const RunCheckpoint* resume) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  RunResult result;
  OuterState outer;
  StateSnapshot state, best;
  NativeReplay replay;
  NativeAudit audit;
  double baseline_elapsed = 0., best_t = kInfinity;
  auto clock_update = [&]() {
    outer.elapsed_seconds = baseline_elapsed + std::chrono::duration<double>(Clock::now() - start).count();
  };
  auto finish = [&](RunStatus status, const std::string& reason) {
    clock_update();
    result.status = status; result.detail = reason;
    result.checkpoint = {state, best, outer, limits.fingerprint()};
    result.best_physical_state = best; result.final_replay = replay; result.final_audit = audit;
    return result;
  };
  auto full_replay = [&](const std::string& reason, bool metadata_only_reuse = false) {
    const bool reused = metadata_only_reuse && replay.state_token == state.token();
    if (!reused) replay = builder.replay_native(state);
    audit = audit_native(state, builder.observations(), replay,
                         limits.native_q_tolerance, limits.native_equality_tolerance_px);
    if (audit.identity_complete && audit.physical_pass && audit.support_pass && audit.t_hat < best_t) {
      best_t = audit.t_hat; best = state;
    }
    clock_update();
    if (hooks.on_outer_checkpoint) hooks.on_outer_checkpoint(state, outer, replay, audit, reused ? reason + "; exact unchanged-state native replay reused" : reason);
  };
  auto reset_phase = [&](Phase phase) {
    const double elapsed = outer.elapsed_seconds;
    outer = OuterState{}; outer.phase = phase; outer.rho = limits.rho_initial; outer.elapsed_seconds = elapsed;
    outer.lambda.assign(builder.schema().inequality_ids.size(), 0.);
    outer.nu.assign(builder.schema().equality_ids.size(), 0.);
    if (phase == Phase::kPrecision) *builder.budget_parameter() = 1.;
    state = builder.capture_state();
  };

  try {
    limits.validate(); check_schema(builder);
    require(builder.budget_parameter() != nullptr, "missing budget_sq parameter");
    if (resume) {
      require(resume->runtime_fingerprint == limits.fingerprint(), "resume settings differ from frozen runtime limits");
      require(!resume->state.empty() && resume->state.data().identity == builder.identity(), "resume problem identity mismatch");
      state = resume->state; builder.restore_state(state); outer = resume->outer;
      baseline_elapsed = outer.elapsed_seconds;
      require(outer.phase == Phase::kFeasibility || outer.phase == Phase::kPrecision, "invalid resumed phase");
      require(outer.lambda.size() == builder.schema().inequality_ids.size() && outer.nu.size() == builder.schema().equality_ids.size(),
              "resume AL dimensions do not match the restored root chart");
      finite_vector(outer.lambda, "resume lambda"); finite_vector(outer.nu, "resume nu");
      for (double v : outer.lambda) require(v >= 0., "negative resumed multiplier");
      require(std::isfinite(outer.rho) && outer.rho > 0. && outer.rho <= limits.rho_max
              && std::isfinite(baseline_elapsed) && baseline_elapsed >= 0.
              && !std::isnan(outer.previous_progress) && outer.previous_progress >= 0., "invalid resumed AL history");
      require(outer.phase != Phase::kPrecision || *builder.budget_parameter() == 1., "precision resume must keep b=1");
      phase_one_budget_residual(*builder.budget_parameter());
      if (!resume->best_physical_state.empty()) {
        require(resume->best_physical_state.data().identity == builder.identity(), "best-state identity mismatch");
        auto br = builder.replay_native(resume->best_physical_state);
        auto ba = audit_native(resume->best_physical_state, builder.observations(), br,
                               limits.native_q_tolerance, limits.native_equality_tolerance_px);
        require(ba.identity_complete && ba.physical_pass && ba.support_pass, "saved best physical state does not replay");
        best = resume->best_physical_state; best_t = ba.t_hat;
      }
      full_replay("resume: complete independent replay; continue same outer objective");
    } else {
      state = builder.capture_state();
      require(state.data().identity == builder.identity(), "captured state identity mismatch");
      const double old_b = *builder.budget_parameter();
      phase_one_budget_residual(old_b);
      const auto initial = builder.evaluate_constraints(state);
      check_values(initial, state, builder.schema());
      double max_q = 1.;
      for (std::size_t i = 0; i < builder.schema().pixel_count; ++i) {
        require(initial.g[i] + old_b >= -1e-12, "pixel g is not q-b with nonnegative q");
        max_q = std::max(max_q, initial.g[i] + old_b);
      }
      *builder.budget_parameter() = max_q;
      reset_phase(Phase::kFeasibility);
      full_replay("initial full source replay");
      if (!audit.identity_complete) return finish(RunStatus::kContractViolation, "initial native replay omitted or reordered original observations");
      // evaluate_constraints/check_values above already established finite,
      // complete arithmetic in the exact AL chart. A finite positive scene or
      // source-domain inequality, or a nonzero RS/query/order residual, is an
      // infeasibility for PHR to restore, not an arithmetic failure. Strict
      // native physical/support validity still gates phase II and every best.
      if (!replay.exact_physical_graph)
        return finish(RunStatus::kInvalidInitialState, "initial model is not the exact required physical graph");
      if (audit.passed()) { reset_phase(Phase::kPrecision); full_replay("native unit budget already passed; reset AL for phase II"); }
    }

    while (true) {
      clock_update();
      if (hooks.stop_requested && hooks.stop_requested()) return finish(RunStatus::kStopped, "stop requested at a complete-state boundary");
      if (outer.elapsed_seconds >= limits.max_total_seconds) return finish(RunStatus::kUnfinished, "frozen active execution time budget exhausted");
      if (outer.outer_iteration >= limits.max_outer_iterations_per_phase) return finish(RunStatus::kUnfinished, "frozen outer-iteration budget exhausted");
      if (outer.inner_attempt >= limits.max_inner_attempts_per_outer)
        return finish(RunStatus::kInnerAccuracyUnfinished, "same-objective inner continuation budget exhausted before multiplier update");
      std::ostringstream objective_identity;
      objective_identity << std::hexfloat << outer.rho;
      for (double v : outer.lambda) objective_identity << ' ' << v;
      objective_identity << '/';
      for (double v : outer.nu) objective_identity << ' ' << v;
      const ObjectiveContext context{outer.phase, outer.lambda, outer.nu, outer.rho,
          builder.identity().objective_id + "/" + std::to_string(static_cast<int>(outer.phase)) + "/"
          + std::to_string(outer.outer_iteration) + "/" + std::to_string(outer.chart_restarts)
          + "/" + hash_text(objective_identity.str()), limits.inequality_kink_active};
      const InnerResult solved = inner.solve(builder, context, outer, limits, hooks,
                                             limits.max_total_seconds - outer.elapsed_seconds);
      require(!solved.accepted_state.empty(), "inner solver did not return an immutable accepted state");
      require(solved.accepted_state.data().identity == builder.identity(), "inner state changed frozen problem identity");
      state = solved.accepted_state;
      builder.restore_state(state); // Restore exactly, never post-step recover.
      full_replay("after inner attempt: " + solved.message);
      if (!audit.identity_complete) return finish(RunStatus::kContractViolation, "native replay lost/reordered original source obligations");
      if (solved.status == InnerStatus::kStopped || (hooks.stop_requested && hooks.stop_requested()))
        return finish(RunStatus::kStopped, "inner stopped; multipliers unchanged and accepted full state retained");
      if (solved.status == InnerStatus::kRootChartRestart) {
        if (outer.chart_restarts >= limits.max_chart_restarts_per_outer)
          return finish(RunStatus::kNumericallyLimited, "equivalent root-chart restart budget exhausted");
        const ConstraintSchema old_schema = builder.schema();
        const auto old_state = state;
        require(builder.rebuild_equivalent_root_chart(), "requested root chart could not be restored as equivalent equality auxiliaries");
        state = builder.capture_state();
        require(same_physical_state(old_state, state), "root-chart restart changed physical parameters/discrete scene; recovery edit rejected");
        check_schema(builder);
        require(old_schema.pixel_count == builder.schema().pixel_count, "root chart altered native pixel obligations");
        outer.lambda = remap(old_schema.inequality_ids, outer.lambda, builder.schema().inequality_ids);
        outer.nu = remap(old_schema.equality_ids, outer.nu, builder.schema().equality_ids);
        ++outer.chart_restarts; outer.inner_attempt = 0; outer.previous_progress = kInfinity;
        full_replay("equivalent explicit-root chart rebuilt at the same physical state; all derivatives invalidated");
        continue;
      }
      if (solved.status == InnerStatus::kFailed) return finish(RunStatus::kNumericallyLimited, solved.message);
      if (outer.phase == Phase::kFeasibility && audit.passed()) {
        reset_phase(Phase::kPrecision);
        full_replay("phase I native feasibility passed; b=1 and all multipliers/rho reset for phase II");
        continue;
      }
      const auto values = builder.evaluate_constraints(state);
      check_values(values, state, builder.schema());
      std::vector<double> effective_lambda, effective_nu;
      for (std::size_t i = 0; i < values.g.size(); ++i) effective_lambda.push_back(std::max(0., outer.lambda[i] + outer.rho * values.g[i]));
      for (std::size_t j = 0; j < values.h.size(); ++j) effective_nu.push_back(outer.nu[j] + outer.rho * values.h[j]);
      const double inner_gradient = check_gradient(builder.lagrangian_gradient(state, outer.phase, effective_lambda, effective_nu), state);
      const double required_accuracy = outer.outer_iteration < limits.tighten_after_outer
          ? limits.inner_stationarity_early : limits.inner_stationarity_late;
      if (hooks.on_solver_summary) hooks.on_solver_summary(outer, state.token(), {
          {"fresh_original_tangent_gradient_norm", inner_gradient},
          {"required_original_tangent_tolerance", required_accuracy},
          {"fresh_original_tangent_accuracy_passed", double(inner_gradient <= required_accuracy)},
          {"fresh_original_tangent_phase", double(static_cast<int>(outer.phase))},
          {"fresh_original_tangent_outer", double(outer.outer_iteration)},
          {"fresh_original_tangent_inner_attempt", double(outer.inner_attempt)},
          {"additional_gradient_evaluations", 0.}});
      if (inner_gradient > required_accuracy) {
        ++outer.inner_attempt;
        if (outer.inner_attempt >= limits.max_inner_attempts_per_outer)
          return finish(RunStatus::kInnerAccuracyUnfinished, "Ceres stopping is insufficient: fresh scaled AL gradient=" + std::to_string(inner_gradient));
        full_replay("inner accuracy insufficient; continue with identical lambda/rho/objective", true);
        continue;
      }
      const auto updated = update_phr(values.g, values.h, outer.lambda, outer.nu, outer.rho,
                                      outer.previous_progress, limits.rho_max, limits.improvement_ratio, limits.penalty_factor);
      outer.lambda = updated.lambda; outer.nu = updated.nu;
      KKTReport kkt;
      kkt.primal_violation = max_positive(values.g); kkt.equality_violation = infinity_norm(values.h);
      kkt.negative_multiplier = 0.; kkt.complementarity = 0.;
      for (std::size_t i = 0; i < values.g.size(); ++i) {
        kkt.negative_multiplier = std::max(kkt.negative_multiplier, -outer.lambda[i]);
        kkt.complementarity = std::max(kkt.complementarity, std::abs(outer.lambda[i] * values.g[i]));
      }
      kkt.scaled_stationarity = check_gradient(builder.lagrangian_gradient(state, outer.phase, outer.lambda, outer.nu), state);
      kkt.budget_stationarity = 0.;
      if (outer.phase == Phase::kFeasibility) {
        double sum = 0.;
        for (std::size_t i = 0; i < builder.schema().pixel_count; ++i) sum += outer.lambda[i];
        kkt.budget_stationarity = std::abs(projected_box_gradient(*builder.budget_parameter(), 1. - sum, 1., 0., kInfinity));
      }
      kkt.fresh = true; result.numerical = kkt;
      // Checkpoint below is the NEXT outer objective, not a half-updated state.
      ++outer.outer_iteration; outer.inner_attempt = 0; outer.chart_restarts = 0;
      outer.previous_progress = updated.progress;
      outer.rho = updated.next_rho;
      full_replay("outer multipliers updated after fresh inner accuracy and KKT evaluation", true);
      if (kkt.passed(limits, outer.phase)) {
        if (outer.phase == Phase::kPrecision && audit.passed()) {
          // Separate complete replay after freezing the final state; do not use
          // an old cost, an accepted subset or best-frame parameter splicing.
          const auto final_state = builder.capture_state();
          require(final_state.token() == state.token(), "final state changed outside an accepted inner step");
          state = final_state; full_replay("final independent frozen-state native acceptance replay");
          if (!audit.passed()) return finish(RunStatus::kReplayFailed, "final independent native replay did not reproduce unit-budget physical support");
          result.phase_two_completed = true;
          return finish(RunStatus::kPrecisionReady, "phase II full native support and fresh constrained numerical termination passed; S6 uncertainty/exports remain runner obligations");
        }
        if (outer.phase == Phase::kFeasibility)
          return finish(RunStatus::kLocalStationaryOutsideBudget, "current basin reached local KKT outside the fixed native unit budget; no global infeasibility claim");
      }
      if (updated.penalty_limit_reached) return finish(RunStatus::kNumericallyLimited, "required rho growth would exceed the frozen cap");
    }
  } catch (const std::exception& error) {
    // Never convert missing data/gradient/adapter failures into a low-cost pass.
    if (!state.empty()) { try { builder.restore_state(state); } catch (...) {} }
    return finish(RunStatus::kContractViolation, error.what());
  }
}

} // namespace rinkfit
