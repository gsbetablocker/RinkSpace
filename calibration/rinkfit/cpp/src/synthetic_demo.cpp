// RUNNABLE NUMERICAL FIXTURE, NOT THE EVENT ADAPTER OR A FIT OF THE SOURCE SHOT.
// Two finite 3-D materials, independently declared endpoints and verified order,
// one shared physical pinhole camera, closed station bounds, two AL phases.
#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/physical.hpp"

#include <ceres/ceres.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace rinkfit;

struct Datum {
  Vec3<double> a, b;
  Vec2<double> uv;
  bool endpoint = false;
  double true_station = 0.;
  int material = 0;
};

template <typename T>
bool predict(const Datum& datum, const T* camera, const T* station, T* uv) {
  if (!(station[0] >= T(0.) && station[0] <= T(1.))) return false;
  const Vec3<T> a{T(datum.a[0]), T(datum.a[1]), T(datum.a[2])};
  const Vec3<T> b{T(datum.b[0]), T(datum.b[1]), T(datum.b[2])};
  const Vec3<T> world = finite_line_point(a, b, station[0]);
  Pose<T> pose; pose.C = {camera[1], T(0.), T(0.)};
  using std::exp;
  const T f = exp(camera[0]);
  const auto projection = project(world, pose, Intrinsics<T>{f, f, T(320.), T(240.)}, Brown<T>{});
  if (!projection.valid()) return false;
  uv[0] = projection.uv[0]; uv[1] = projection.uv[1];
  return true;
}
struct PixelConstraint {
  Datum datum;
  template <typename T> bool operator()(const T* camera, const T* station, const T* budget, T* g) const {
    T uv[2]; if (!predict(datum, camera, station, uv)) return false;
    const T ex = uv[0] - T(datum.uv[0]), ey = uv[1] - T(datum.uv[1]);
    g[0] = (ex * ex + ey * ey) / T(4.) - budget[0]; return true;
  }
};
struct StatisticalResidual {
  Datum datum;
  template <typename T> bool operator()(const T* camera, const T* station, T* e) const {
    if (!predict(datum, camera, station, e)) return false;
    e[0] -= T(datum.uv[0]); e[1] -= T(datum.uv[1]); return true; // frozen engineering sigma=1 px
  }
};
struct OrderConstraint {
  template <typename T> bool operator()(const T* left, const T* right, T* g) const {
    g[0] = left[0] - right[0]; return true;
  }
};
struct EqualityWitness {
  template <typename T> bool operator()(const T* x, T* h) const { h[0] = x[0] - T(.3); return true; }
};
struct InvalidDomainConstraint {
  template <typename T> bool operator()(const T*, T*) const { return false; }
};

void check_phr_ceres_wrappers() {
  double x = .6;
  ceres::Problem problem;
  auto equality = std::make_unique<ceres::AutoDiffCostFunction<EqualityWitness, 1, 1>>(new EqualityWitness);
  auto eq_id = static_cast<ceres::ResidualBlockId>(add_phr_equality(problem, std::move(equality), {&x}, -.4, 2.));
  double const* blocks[] = {&x}; double residual, jacobian; double* jac[] = {&jacobian};
  if (!problem.GetCostFunctionForResidualBlock(eq_id)->Evaluate(blocks, &residual, jac)
      || std::abs(residual - .2 / std::sqrt(2.)) > 1e-12 || std::abs(jacobian - std::sqrt(2.)) > 1e-12)
    throw std::runtime_error("Ceres equality PHR residual/Jacobian scaling failed");
  auto invalid = std::make_unique<ceres::AutoDiffCostFunction<InvalidDomainConstraint, 1, 1>>(new InvalidDomainConstraint);
  auto bad_id = static_cast<ceres::ResidualBlockId>(add_phr_inequality(problem, std::move(invalid), {&x}, 0., 1.));
  if (problem.GetCostFunctionForResidualBlock(bad_id)->Evaluate(blocks, &residual, nullptr))
    throw std::runtime_error("an invalid-domain constraint was silently replaced with zero AL cost");
}

class FiniteLineFixture final : public ProblemBuilder {
 public:
  FiniteLineFixture() {
    id_ = {"synthetic-only-12-original-rows-v1", "two-finite-lines-shared-static-camera-v1",
           "engineering-E-4I-v1", "engineering-sigma-1px-no-common-mode", "logf-1-Cx-0.1-S-1-b-1", "phr-v1.1-fixture"};
    const double truth[2] = {std::log(600.), .08};
    const double stations[] = {0., .18, .38, .60, .80, 1.};
    for (int material = 0; material < 2; ++material) {
      const Vec3<double> a = material == 0 ? Vec3<double>{-1., -.5, 5.} : Vec3<double>{.8, -.6, 6.};
      const Vec3<double> b = material == 0 ? Vec3<double>{1., -.5, 5.} : Vec3<double>{.8, .8, 6.};
      for (int j = 0; j < 6; ++j) {
        Datum datum{a, b, {}, j == 0 || j == 5, stations[j], material};
        double uv[2]; if (!predict(datum, truth, &stations[j], uv)) throw std::runtime_error("invalid synthetic truth");
        const std::size_t i = data_.size();
        datum.uv = {uv[0] + .15 * std::sin(double(i + 1)), uv[1] + .12 * std::cos(double(i + 1))};
        data_.push_back(datum);
        s_.push_back(datum.endpoint ? datum.true_station : datum.true_station + .035);
        ObservationSpec spec; spec.key = "fixture_clock:0:" + std::to_string(i); spec.source_uv = datum.uv;
        observations_.push_back(spec); schema_.inequality_ids.push_back("pixel/" + spec.key);
        if (j > 0) orders_.push_back({i - 1, i});
      }
    }
    schema_.pixel_count = observations_.size();
    for (const auto& pair : orders_) schema_.inequality_ids.push_back("verified_order/" + std::to_string(pair[0]) + "/" + std::to_string(pair[1]));
  }
  const FrozenIdentity& identity() const override { return id_; }
  const std::vector<ObservationSpec>& observations() const override { return observations_; }
  const ConstraintSchema& schema() const override { return schema_; }
  double* budget_parameter() override { return &budget_; }
  std::vector<ParameterBlockBinding> parameter_blocks() override {
    std::vector<ParameterBlockBinding> out{{"camera/log_f_and_shared_Cx", camera_.data(), 2}, {"budget_sq", &budget_, 1}};
    for (std::size_t i = 0; i < s_.size(); ++i) out.push_back({"material_station/" + std::to_string(i), &s_[i], 1});
    return out;
  }
  StateSnapshot capture_state() const override {
    StatePayload p; p.identity = id_;
    p.parameters.push_back({"camera/log_f_and_shared_Cx", {camera_[0], camera_[1]}});
    p.parameters.push_back({"budget_sq", {budget_}});
    for (std::size_t i = 0; i < s_.size(); ++i) p.parameters.push_back({"material_station/" + std::to_string(i), {s_[i]}});
    p.discrete = {{"fixture", "not-event-data"}, {"materials", "fixed-horizontal-z5;fixed-vertical-z6"},
                  {"endpoints", "0,5,6,11 are genuine material endpoints"}, {"root_model", "global-shutter explicit projection"}};
    return StateSnapshot(std::move(p));
  }
  void restore_state(const StateSnapshot& state) override {
    if (!(state.data().identity == id_)) throw std::invalid_argument("fixture snapshot belongs to another problem");
    const auto values = decode(state);
    camera_ = values.camera; budget_ = values.budget;
    std::copy(values.s.begin(), values.s.end(), s_.begin()); // Stable Ceres addresses.
  }
  NativeReplay replay_native(const StateSnapshot& state) const override {
    const auto x = decode(state);
    NativeReplay replay; replay.state_token = state.token();
    replay.exact_physical_graph = true; replay.physical_domain_valid = true;
    replay.all_original_support_relations_valid = true; replay.statistical_pixel_cost = 0.;
    for (std::size_t i = 0; i < data_.size(); ++i) {
      NativeRow row; row.key = observations_[i].key; row.station = x.s[i]; row.branch_id = "segment/" + std::to_string(data_[i].material);
      double uv[2]; row.projection_valid = predict(data_[i], x.camera.data(), &x.s[i], uv);
      row.true_root_valid = row.projection_valid; // GS fixture: no implicit root or query auxiliary.
      row.finite_material_and_order_valid = x.s[i] >= 0. && x.s[i] <= 1.
          && (!data_[i].endpoint || x.s[i] == data_[i].true_station);
      for (const auto& order : orders_)
        if ((order[0] == i || order[1] == i) && x.s[order[0]] > x.s[order[1]] + 1e-12)
          row.finite_material_and_order_valid = false;
      if (row.projection_valid) {
        row.predicted_uv = {uv[0], uv[1]};
        const double ex = uv[0] - data_[i].uv[0], ey = uv[1] - data_[i].uv[1];
        replay.statistical_pixel_cost += .5 * (ex * ex + ey * ey);
      } else { row.failure_reason = "true projection/material arithmetic domain failure"; replay.physical_domain_valid = false; }
      if (!row.finite_material_and_order_valid) replay.all_original_support_relations_valid = false;
      replay.rows.push_back(std::move(row));
    }
    return replay;
  }
  ConstraintValues evaluate_constraints(const StateSnapshot& state) const override {
    const auto x = decode(state); ConstraintValues out; out.state_token = state.token();
    for (std::size_t i = 0; i < data_.size(); ++i) {
      double g;
      if (!PixelConstraint{data_[i]}(x.camera.data(), &x.s[i], &x.budget, &g)) throw std::domain_error("invalid constraint projection");
      out.g.push_back(g);
    }
    for (const auto& pair : orders_) out.g.push_back(x.s[pair[0]] - x.s[pair[1]]);
    return out;
  }
  GradientReport lagrangian_gradient(const StateSnapshot& state, Phase phase,
      const std::vector<double>& lambda, const std::vector<double>& nu) const override {
    if (lambda.size() != schema_.inequality_ids.size() || !nu.empty()) throw std::invalid_argument("fixture multiplier dimensions");
    const auto x = decode(state);
    double glogf = 0., gcx = 0., gb = phase == Phase::kFeasibility ? 1. : 0.;
    std::vector<double> gs(s_.size(), 0.);
    const double f = std::exp(x.camera[0]);
    for (std::size_t i = 0; i < data_.size(); ++i) {
      double uv[2]; if (!predict(data_[i], x.camera.data(), &x.s[i], uv)) throw std::domain_error("invalid derivative state");
      const double ex = uv[0] - data_[i].uv[0], ey = uv[1] - data_[i].uv[1];
      const double coefficient = (phase == Phase::kPrecision ? 1. : 0.) + .5 * lambda[i];
      glogf += coefficient * (ex * (uv[0] - 320.) + ey * (uv[1] - 240.));
      gcx += coefficient * ex * (-f / data_[i].a[2]);
      gs[i] += coefficient * f / data_[i].a[2]
          * (ex * (data_[i].b[0] - data_[i].a[0]) + ey * (data_[i].b[1] - data_[i].a[1]));
      gb -= lambda[i];
    }
    for (std::size_t k = 0; k < orders_.size(); ++k) {
      gs[orders_[k][0]] += lambda[data_.size() + k];
      gs[orders_[k][1]] -= lambda[data_.size() + k];
    }
    GradientReport out; out.state_token = state.token(); out.scales_id = id_.scales_id;
    out.complete_current_derivatives = true;
    out.scaled_projected_components = {
      projected_box_gradient(x.camera[0], glogf, 1., std::log(100.), std::log(2000.)),
      projected_box_gradient(x.camera[1], gcx, .1, -2., 2.),
      projected_box_gradient(x.budget, gb, 1., phase == Phase::kPrecision ? 1. : 0., phase == Phase::kPrecision ? 1. : kInfinity)};
    for (std::size_t i = 0; i < gs.size(); ++i) {
      const double lo = data_[i].endpoint ? data_[i].true_station : 0.;
      const double hi = data_[i].endpoint ? data_[i].true_station : 1.;
      out.scaled_projected_components.push_back(projected_box_gradient(x.s[i], gs[i], 1., lo, hi));
    }
    return out;
  }
  BuildInventory build_ceres_problem(ceres::Problem& problem, const ObjectiveContext& context) override {
    BuildInventory inventory; inventory.exact_final_physical_graph = true;
    problem.AddParameterBlock(camera_.data(), 2); problem.AddParameterBlock(&budget_, 1);
    problem.SetParameterLowerBound(camera_.data(), 0, std::log(100.));
    problem.SetParameterUpperBound(camera_.data(), 0, std::log(2000.));
    problem.SetParameterLowerBound(camera_.data(), 1, -2.); problem.SetParameterUpperBound(camera_.data(), 1, 2.);
    for (std::size_t i = 0; i < data_.size(); ++i) {
      problem.AddParameterBlock(&s_[i], 1);
      problem.SetParameterLowerBound(&s_[i], 0, 0.); problem.SetParameterUpperBound(&s_[i], 0, 1.);
      if (data_[i].endpoint) problem.SetParameterBlockConstant(&s_[i]);
      auto g = std::make_unique<ceres::AutoDiffCostFunction<PixelConstraint, 1, 2, 1, 1>>(new PixelConstraint{data_[i]});
      inventory.inequality_blocks.push_back(add_phr_inequality(problem, std::move(g), {camera_.data(), &s_[i], &budget_}, context.lambda.at(i), context.rho));
      if (context.phase == Phase::kPrecision)
        inventory.statistical_blocks.push_back(problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<StatisticalResidual, 2, 2, 1>(new StatisticalResidual{data_[i]}), nullptr, camera_.data(), &s_[i]));
    }
    for (std::size_t k = 0; k < orders_.size(); ++k) {
      auto g = std::make_unique<ceres::AutoDiffCostFunction<OrderConstraint, 1, 1, 1>>(new OrderConstraint);
      inventory.inequality_blocks.push_back(add_phr_inequality(problem, std::move(g), {&s_[orders_[k][0]], &s_[orders_[k][1]]}, context.lambda.at(data_.size() + k), context.rho));
    }
    return inventory;
  }
  void check_autodiff() const {
    const Datum& datum = data_[2];
    ceres::AutoDiffCostFunction<PixelConstraint, 1, 2, 1, 1> cost(new PixelConstraint{datum});
    double camera[2] = {std::log(570.), .11}, s = .37, b = 1.;
    double const* blocks[] = {camera, &s, &b};
    double jc[2], js[1], jb[1], residual;
    double* jac[] = {jc, js, jb};
    if (!cost.Evaluate(blocks, &residual, jac)) throw std::runtime_error("actual Ceres Jet evaluation failed");
    double* variables[] = {&camera[0], &camera[1], &s, &b};
    const double analytic[] = {jc[0], jc[1], js[0], jb[0]};
    for (int j = 0; j < 4; ++j) {
      const double old = *variables[j], h = 1e-6;
      *variables[j] = old + h; double plus; cost.Evaluate(blocks, &plus, nullptr);
      *variables[j] = old - h; double minus; cost.Evaluate(blocks, &minus, nullptr);
      *variables[j] = old;
      const double numeric = (plus - minus) / (2. * h);
      if (std::abs(numeric - analytic[j]) > 2e-6 * (1. + std::abs(numeric)))
        throw std::runtime_error("Ceres Jet derivative disagrees with feasible finite difference");
    }
  }
 private:
  struct Decoded { std::array<double, 2> camera; double budget; std::vector<double> s; };
  Decoded decode(const StateSnapshot& state) const {
    if (!(state.data().identity == id_) || state.data().parameters.size() != s_.size() + 2)
      throw std::invalid_argument("fixture complete state layout mismatch");
    const auto& p = state.data().parameters;
    if (p[0].name != "camera/log_f_and_shared_Cx" || p[0].values.size() != 2
        || p[1].name != "budget_sq" || p[1].values.size() != 1)
      throw std::invalid_argument("fixture camera/budget registry mismatch");
    Decoded x{{p[0].values[0], p[0].values[1]}, p[1].values[0], {}};
    for (std::size_t i = 0; i < s_.size(); ++i) {
      if (p[i + 2].name != "material_station/" + std::to_string(i) || p[i + 2].values.size() != 1)
        throw std::invalid_argument("fixture station registry mismatch");
      x.s.push_back(p[i + 2].values[0]);
    }
    return x;
  }
  FrozenIdentity id_;
  std::vector<Datum> data_;
  std::vector<ObservationSpec> observations_;
  ConstraintSchema schema_;
  std::vector<std::array<std::size_t, 2>> orders_;
  std::array<double, 2> camera_{std::log(520.), .30};
  std::vector<double> s_;
  double budget_ = 1.;
};

void print_result(const RunResult& result) {
  std::cout << "fixture_only=true\nfit_of_original_shot=false\nstatus=" << status_name(result.status)
            << "\nrows=" << result.final_audit.accounted_rows << '/' << result.final_audit.expected_rows
            << "\nnative_t_hat=" << std::setprecision(12) << result.final_audit.t_hat
            << "\nF_px=" << result.final_replay.statistical_pixel_cost
            << "\nphase_II_completed=" << result.phase_two_completed
            << "\nscaled_stationarity=" << result.numerical.scaled_stationarity
            << "\ncomplementarity=" << result.numerical.complementarity
            << "\nstate_token=" << result.checkpoint.state.token()
            << "\ndetail=" << result.detail << '\n';
}

int self_test() {
  RuntimeLimits limits; limits.max_total_seconds = 60.;
  CeresInnerSolver inner;
  FiniteLineFixture continuous;
  continuous.check_autodiff();
  check_phr_ceres_wrappers();
  auto initial_state = continuous.capture_state();
  auto initial = audit_native(initial_state, continuous.observations(), continuous.replay_native(initial_state), 1e-4, .01);
  if (!(initial.t_hat > 1.)) throw std::runtime_error("fixture did not start outside native budget");
  auto full = solve_two_phase(continuous, inner, limits);
  print_result(full);
  if (full.status != RunStatus::kPrecisionReady || !full.phase_two_completed || !full.final_audit.passed())
    throw std::runtime_error("continuous Ceres solve failed");
  FiniteLineFixture interrupted;
  bool stop = false;
  RunHooks hooks;
  hooks.on_iteration = [&](const StateSnapshot&, const OuterState&, const IterationRecord& i) {
    if (i.iteration > 0 && i.step_accepted) stop = true;
  };
  hooks.stop_requested = [&]() { return stop; };
  auto paused = solve_two_phase(interrupted, inner, limits, hooks);
  if (paused.status != RunStatus::kStopped) throw std::runtime_error("stop was mistaken for successful precision completion");
  std::stringstream file; write_checkpoint(file, paused.checkpoint);
  const auto restored = read_checkpoint(file);
  if (restored.state.token() != paused.checkpoint.state.token()) throw std::runtime_error("checkpoint changed accepted complete state");
  FiniteLineFixture fresh_process_model;
  auto resumed = solve_two_phase(fresh_process_model, inner, limits, {}, &restored);
  if (resumed.status != RunStatus::kPrecisionReady || !resumed.final_audit.passed())
    throw std::runtime_error("serialized accepted-state resume failed: " + resumed.detail);
  if (std::abs(resumed.final_replay.statistical_pixel_cost - full.final_replay.statistical_pixel_cost) > 1e-7)
    throw std::runtime_error("resume reached a different native fixture objective");
  std::cout << "initial_t_hat=" << initial.t_hat << "\nautodiff_checked=log_f,Cx,S,b\nserialized_stop_resume=passed\n";
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::string checkpoint_path, resume_path, ledger_path;
    bool test = false, verbose = false, capabilities = false;
    int stop_after = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") test = true;
      else if (arg == "--log") verbose = true;
      else if (arg == "--capabilities-json") capabilities = true;
      else if ((arg == "--checkpoint" || arg == "--resume" || arg == "--ledger" || arg == "--stop-after") && i + 1 < argc) {
        const std::string value = argv[++i];
        if (arg == "--checkpoint") checkpoint_path = value;
        if (arg == "--resume") resume_path = value;
        if (arg == "--ledger") ledger_path = value;
        if (arg == "--stop-after") stop_after = std::stoi(value);
      } else throw std::invalid_argument("usage: rinkfit_synthetic_demo [--self-test] [--log] [--capabilities-json] [--checkpoint PATH] [--resume PATH] [--ledger CSV] [--stop-after N]");
    }
    if (capabilities) {
      std::cout << "{\"schema\":\"rinkfit.synthetic_backend.v1\",\"backend_kind\":\"synthetic_fixture\",\"solver\":\"Ceres\",\"linear_solver\":\"SPARSE_NORMAL_CHOLESKY\",\"sparse_library\":\"SuiteSparse\",\"whole_shot_rows\":12,\"synthetic_demo\":true,\"independent_native_replay\":true,\"fixed_shared_camera_head\":false}\n";
      return 0;
    }
    if (test) return self_test();
    RuntimeLimits limits; limits.max_total_seconds = 60.; limits.log_ceres_progress = verbose;
    FiniteLineFixture builder; CeresInnerSolver inner;
    RunCheckpoint checkpoint;
    if (!resume_path.empty()) {
      std::ifstream file(resume_path); if (!file) throw std::runtime_error("cannot open resume checkpoint");
      checkpoint = read_checkpoint(file);
    }
    bool stop = false; int accepted = 0;
    RunHooks hooks;
    hooks.stop_requested = [&]() { return stop; };
    hooks.on_iteration = [&](const StateSnapshot&, const OuterState&, const IterationRecord& record) {
      if (record.iteration > 0 && record.step_accepted && stop_after > 0 && ++accepted >= stop_after) stop = true;
    };
    auto result = solve_two_phase(builder, inner, limits, hooks, resume_path.empty() ? nullptr : &checkpoint);
    if (!checkpoint_path.empty()) {
      std::ofstream file(checkpoint_path); if (!file) throw std::runtime_error("cannot write checkpoint");
      write_checkpoint(file, result.checkpoint);
    }
    if (!ledger_path.empty()) {
      std::ofstream file(ledger_path); if (!file) throw std::runtime_error("cannot write ledger");
      file << "key,source_u,source_v,predicted_u,predicted_v,station,q,native_support_valid\n" << std::setprecision(17);
      for (std::size_t i = 0; i < result.final_replay.rows.size(); ++i) {
        const auto& row = result.final_replay.rows[i]; const auto& spec = builder.observations()[i];
        file << row.key << ',' << spec.source_uv[0] << ',' << spec.source_uv[1] << ',' << row.predicted_uv[0] << ',' << row.predicted_uv[1]
             << ',' << row.station << ',' << result.final_audit.q[i] << ',' << row.finite_material_and_order_valid << '\n';
      }
    }
    print_result(result);
    return result.status == RunStatus::kPrecisionReady || result.status == RunStatus::kStopped ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "synthetic fixture error: " << error.what() << '\n'; return 2;
  }
}
