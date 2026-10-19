#include "rinkfit/augmented_lagrangian.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace rinkfit;
namespace {
void check(bool condition, const char* reason) { if (!condition) throw std::runtime_error(reason); }
void close(double a, double b, double tolerance = 1e-12) { check(std::abs(a - b) <= tolerance, "numeric comparison failed"); }

StateSnapshot snapshot(double value = 2.) {
  StatePayload p;
  p.identity = {"source", "model", "E", "W", "scales", "objective"};
  p.parameters = {{"x", {value}}, {"budget_sq", {1.}}};
  p.discrete = {{"scene", "shared finite material"}};
  p.root_charts = {{"profile/0", "exact"}};
  return StateSnapshot(p);
}

class ScalarFixture final : public ProblemBuilder {
 public:
  ScalarFixture() {
    id_ = snapshot().data().identity;
    ObservationSpec o; o.key = "clock:256256:1"; o.E = {1., 0., 0., 1.}; observations_ = {o};
    schema_.pixel_count = 1; schema_.inequality_ids = {"pixel/clock:256256:1"};
  }
  const FrozenIdentity& identity() const override { return id_; }
  const std::vector<ObservationSpec>& observations() const override { return observations_; }
  const ConstraintSchema& schema() const override { return schema_; }
  double* budget_parameter() override { return &b; }
  std::vector<ParameterBlockBinding> parameter_blocks() override { return {{"x", &x, 1}, {"budget_sq", &b, 1}}; }
  StateSnapshot capture_state() const override {
    StatePayload p; p.identity = id_; p.parameters = {{"x", {x}}, {"budget_sq", {b}}};
    p.discrete = {{"synthetic", "scalar algebra test only"}}; return StateSnapshot(std::move(p));
  }
  void restore_state(const StateSnapshot& state) override {
    if (!(state.data().identity == id_)) throw std::invalid_argument("identity mismatch");
    x = state.data().parameters.at(0).values.at(0); b = state.data().parameters.at(1).values.at(0);
  }
  NativeReplay replay_native(const StateSnapshot& state) const override {
    ++replay_calls; NativeReplay r; r.state_token = state.token(); r.exact_physical_graph = true;
    r.physical_domain_valid = native_physical_valid; r.all_original_support_relations_valid = true;
    const double px = state.data().parameters[0].values[0];
    NativeRow row; row.key = observations_[0].key; row.predicted_uv = {px, 0.};
    row.projection_valid = row.true_root_valid = row.finite_material_and_order_valid = true;
    r.rows = {row}; r.statistical_pixel_cost = .5 * px * px; return r;
  }
  ConstraintValues evaluate_constraints(const StateSnapshot& state) const override {
    if(!arithmetic_valid)throw std::domain_error("fixture true factor arithmetic failure");
    const double px = state.data().parameters[0].values[0], pb = state.data().parameters[1].values[0];
    return {state.token(), {px * px - pb}, {}};
  }
  GradientReport lagrangian_gradient(const StateSnapshot& state, Phase phase,
      const std::vector<double>& lambda, const std::vector<double>&) const override {
    const double px = state.data().parameters[0].values[0], pb = state.data().parameters[1].values[0];
    const double gx = (phase == Phase::kPrecision ? px : 0.) + 2. * lambda.at(0) * px;
    const double gb = (phase == Phase::kFeasibility ? 1. : 0.) - lambda.at(0);
    return {state.token(), id_.scales_id,
      {projected_box_gradient(px, gx, 1., -5., 5.),
       projected_box_gradient(pb, gb, 1., phase == Phase::kPrecision ? 1. : 0., phase == Phase::kPrecision ? 1. : kInfinity)}, true};
  }
  BuildInventory build_ceres_problem(ceres::Problem&, const ObjectiveContext&) override { throw std::runtime_error("unused in std-only algebra fixture"); }
  mutable int replay_calls=0;
  double x = 2., b = 1.;
  bool native_physical_valid=true,arithmetic_valid=true;
 private:
  FrozenIdentity id_;
  std::vector<ObservationSpec> observations_;
  ConstraintSchema schema_;
};
class DeclaredInner final : public InnerSolver {
 public:
  explicit DeclaredInner(bool accurate) : accurate_(accurate) {}
  InnerResult solve(ProblemBuilder& builder, const ObjectiveContext& objective, const OuterState&,
                   const RuntimeLimits&, const RunHooks&, double) override {
    if (accurate_) {
      auto& model = dynamic_cast<ScalarFixture&>(builder);
      model.x = objective.phase == Phase::kFeasibility ? .25 : 0.;
      if (objective.phase == Phase::kFeasibility) model.b = .25;
    }
    ++calls;
    return {InnerStatus::kFinished, builder.capture_state(), 1, "declared low inner cost (not a physical certificate)"};
  }
  int calls = 0;
 private:
  bool accurate_;
};

void math_test() {
  close(.5 * std::pow(phase_one_budget_residual(.3), 2), .8);
  const auto updated = update_phr({-3., -.2, .4}, {.1}, {1., 1., 0.}, {-.4}, 2., .5, 1e6);
  close(updated.progress, .5); close(updated.next_rho, 10.);
  close(updated.lambda[0], 0.); close(updated.lambda[1], .6); close(updated.lambda[2], .8); close(updated.nu[0], -.2);
  close(inequality_residual(-3., 1., 2.), 0.);
  close(equality_residual(.1, -.4, 2.), -.1 * std::sqrt(2.));
  check(update_phr({1.}, {}, {0.}, {}, 1e6, 1., 1e6).penalty_limit_reached, "rho cap must report a numerical limit");
  close(projected_box_gradient(0., 1., 1., 0., kInfinity), 0.);
  close(projected_box_gradient(2., .7, 1., 0., kInfinity), .7);
  close(projected_box_gradient(.2, -.1, 2., .2, .2), 0.);
  bool rejected = false;
  try { inequality_residual(-1., -1., 1.); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "negative inequality multiplier was silently accepted");
}

void native_test() {
  const auto state = snapshot();
  ObservationSpec ordinary; ordinary.key = "clock:0:0";
  NativeReplay r; r.state_token = state.token(); r.exact_physical_graph = true;
  r.physical_domain_valid = true; r.all_original_support_relations_valid = true;
  NativeRow row; row.key = ordinary.key; row.predicted_uv = {5., .1};
  row.projection_valid = row.true_root_valid = row.finite_material_and_order_valid = true; r.rows = {row};
  auto a = audit_native(state, {ordinary}, r, 1e-4, .01);
  close(a.q[0], 25.01 / 4.); check(!a.passed(), "finite-material tangent error disappeared");
  r.rows[0].projection_valid = false; r.rows[0].predicted_uv = {0., 0.};
  a = audit_native(state, {ordinary}, r, 1e-4, .01);
  check(std::isinf(a.q[0]) && !a.invalid_rows.empty(), "invalid projection was zeroed or deleted");
  r.rows[0] = row; r.rows[0].key = "another:key";
  check(!audit_native(state, {ordinary}, r, 1e-4, .01).identity_complete, "wrong source key accepted");
  r.rows.clear(); check(!audit_native(state, {ordinary}, r, 1e-4, .01).passed(), "missing row passed");
  ObservationSpec profile = ordinary; profile.semantics = ObservationSemantics::kGenuineProfile;
  profile.profile_normal = {0., 1.}; profile.profile_radius_px = 2.;
  profile.profile_alpha_lower_px = -2.; profile.profile_alpha_upper_px = 2.;
  row.predicted_uv = {5., 1.}; r.rows = {row};
  a = audit_native(state, {profile}, r, 1e-4, .01);
  close(a.q[0], .25); close(a.query_error_px[0], 5.);
  check(a.pixel_pass && !a.support_pass && !a.passed(), "profile query geometry was replaced by normal error alone");
  profile.explicit_profile_query=true;profile.profile_query_uv={5.,-20.};
  profile.profile_alpha_lower_px=20.;profile.profile_alpha_upper_px=22.;
  row.has_profile_witness=true;row.profile_witness_alpha_px=21.;r.rows={row};
  a=audit_native(state,{profile},r,1e-4,.01);
  check(a.passed(),"query-origin alpha was confused with measured normal error");close(a.q[0],.25);
  profile.original_curved_query=true;profile.profile_alpha_lower_px=.2;profile.profile_alpha_upper_px=.4;
  row.profile_witness_alpha_px=.3;row.profile_query_error_px=.002;r.rows={row};
  a=audit_native(state,{profile},r,1e-4,.01);check(a.passed(),"exact curved-query alpha units were relabeled as native pixels");
  r.rows[0].profile_query_error_px=.02;check(!audit_native(state,{profile},r,1e-4,.01).support_pass,"curved query mismatch passed");

}

void snapshot_and_scheduler_test() {
  StatePayload mutable_data = snapshot().data();
  StateSnapshot frozen(mutable_data); mutable_data.parameters[0].values[0] = 99.;
  close(frozen.data().parameters[0].values[0], 2.);
  RuntimeLimits limits; limits.max_total_seconds = 30.;
  ScalarFixture cost_only; DeclaredInner insufficient(false);
  auto bad = solve_two_phase(cost_only, insufficient, limits);
  check(bad.status == RunStatus::kInnerAccuracyUnfinished && !bad.phase_two_completed,
        "one low-cost inner iteration was accepted as completion");
  check(std::all_of(bad.checkpoint.outer.lambda.begin(), bad.checkpoint.outer.lambda.end(), [](double v) { return v == 0.; }),
        "multipliers updated before configured inner accuracy");
  check(cost_only.replay_calls==1+insufficient.calls,"metadata-only fresh continuation repeated physical replay or skipped a new inner replay");
  ScalarFixture finite_infeasible;finite_infeasible.native_physical_valid=false;DeclaredInner recovery(false);
  const auto recovering=solve_two_phase(finite_infeasible,recovery,limits);
  check(recovery.calls>0&&recovering.status==RunStatus::kInnerAccuracyUnfinished&&!recovering.final_audit.physical_pass&&recovering.best_physical_state.empty(),"finite AL infeasibility was rejected as arithmetic or admitted as best physical");
  ScalarFixture arithmetic_failure;arithmetic_failure.arithmetic_valid=false;DeclaredInner forbidden(false);
  const auto rejected_arithmetic=solve_two_phase(arithmetic_failure,forbidden,limits);
  check(forbidden.calls==0&&rejected_arithmetic.status==RunStatus::kContractViolation,"true arithmetic failure reached an inner solve");
  ScalarFixture model; DeclaredInner accurate(true);
  RunHooks stop; stop.stop_requested = []() { return true; };
  auto paused = solve_two_phase(model, accurate, limits, stop);
  check(paused.status == RunStatus::kStopped && accurate.calls == 0, "boundary stop changed state");
  std::stringstream stream; write_checkpoint(stream, paused.checkpoint);
  const auto restored = read_checkpoint(stream);
  check(restored.state.token() == paused.checkpoint.state.token(), "round-trip snapshot lost state identity");
  close(restored.outer.rho, 1.); check(std::isinf(restored.outer.previous_progress), "infinite initial progress did not round-trip");
  ScalarFixture resumed_model;
  RunHooks final_check;int last_replay_count=0;bool final_recomputed=false;final_check.on_outer_checkpoint=[&](const StateSnapshot&,const OuterState&,const NativeReplay&,const NativeAudit&,const std::string&reason){if(reason.find("final independent frozen-state")!=std::string::npos){check(resumed_model.replay_calls==last_replay_count+1,"final frozen acceptance reused an old native replay");final_recomputed=true;}last_replay_count=resumed_model.replay_calls;};
  auto done = solve_two_phase(resumed_model, accurate, limits, final_check, &restored);
  check(final_recomputed,"final independent frozen replay was omitted");
  check(done.status == RunStatus::kPrecisionReady && done.phase_two_completed && done.final_audit.passed(),
        "two-phase continuation did not finish native/numerical acceptance");
  close(resumed_model.b, 1.); check(accurate.calls == 2, "phase II was skipped after feasibility");
  RuntimeLimits kink_limits=limits;kink_limits.inequality_kink_active=true;check(kink_limits.fingerprint()!=limits.fingerprint(),"kink numerical branch missing from frozen runtime fingerprint");kink_limits.inequality_kink_active=false;check(kink_limits.fingerprint()==limits.fingerprint(),"default kink branch altered compatible runtime fingerprint");
  RuntimeLimits changed = limits; changed.rho_max = 1e5;
  auto mismatch = solve_two_phase(resumed_model, accurate, changed, {}, &restored);
  check(mismatch.status == RunStatus::kContractViolation, "resume silently changed frozen run limits");
}
} // namespace

int main() {
  try {
    math_test(); native_test(); snapshot_and_scheduler_test();
    std::cout << "PHR math, full native audit, immutable checkpoint, numerical gate and two-phase resume: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
