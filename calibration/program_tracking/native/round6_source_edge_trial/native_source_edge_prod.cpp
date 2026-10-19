// Isolated GTSAM 4.2.2 factors. Construction crosses pybind once; evaluation is
// entirely C++ and contains no Python objects, callbacks, or GIL acquisition.
#include <gtsam/config.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <boost/make_shared.hpp>
#include <oneapi/tbb/global_control.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "source_edge_native.hpp"
#include "brown15_projection.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);
namespace py = pybind11;
namespace rs = rinkspace_native;

namespace {
using Vec11 = Eigen::Matrix<double, 11, 1>;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Chain = Eigen::Matrix<double, 15, 11>;
using RowMatrix15 = Eigen::Matrix<double, Eigen::Dynamic, 15, Eigen::RowMajor>;
std::atomic<std::uint64_t> edge_calls{0}, edge_cache_hits{0};
std::atomic<std::uint64_t> edge_cache_payload_bytes{0};
std::atomic<std::uint64_t> edge_unsupported_calls{0}, edge_domain_calls{0};
std::mutex tbb_mutex;
std::unique_ptr<oneapi::tbb::global_control> tbb_limit;
struct NativeBoardDomainError : std::runtime_error {
  std::size_t observations;
  std::int64_t factor_id, source_pts;
  bool jacobian;
  NativeBoardDomainError(std::size_t count, std::int64_t id,
                         std::int64_t pts, bool with_jacobian)
      : std::runtime_error("source_edge_no_central_segment"),
        observations(count), factor_id(id), source_pts(pts),
        jacobian(with_jacobian) {}
};
struct NativeUnsupportedError : std::runtime_error {
  std::int64_t factor_id, source_pts;
  bool jacobian;
  // The complete GTSAM trial is copied in C++ on the residual path. This
  // exception can cross a TBB worker boundary without touching Python.
  std::shared_ptr<const gtsam::Values> trial_values;
  NativeUnsupportedError(std::int64_t id, std::int64_t pts, bool with_jacobian,
                         std::shared_ptr<const gtsam::Values> trial = nullptr)
      : std::runtime_error("rs_edge_eval status 2"),
        factor_id(id), source_pts(pts), jacobian(with_jacobian),
        trial_values(std::move(trial)) {}
};

py::object required(const py::dict& d, const char* key) {
  if (!d.contains(key)) throw std::invalid_argument(std::string("missing spec field: ") + key);
  return d[key].cast<py::object>();
}

template <typename T>
std::vector<T> flat_array(py::handle object, int ndim,
                          const std::vector<py::ssize_t>& trailing,
                          const char* label) {
  using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
  auto array = Array::ensure(object);
  if (!array || array.ndim() != ndim)
    throw std::invalid_argument(std::string(label) + ": wrong array rank");
  for (int j = 0; j < static_cast<int>(trailing.size()); ++j) {
    if (array.shape(ndim - static_cast<int>(trailing.size()) + j) != trailing[j])
      throw std::invalid_argument(std::string(label) + ": wrong trailing shape");
  }
  auto* start = array.data();
  return {start, start + array.size()};
}

struct RigData {
  std::vector<gtsam::Key> deps, active_keys;
  std::vector<int> active_position;
  std::vector<Vec11> frozen;
  int frame = -1, lens_base = -1, lens_slope = -1;
  int mount = -1, offset = -1, offset_slope = -1;
  int brown = -1, brown_slope = -1;
  std::vector<std::pair<int, double>> controls;
  double zref = 0.0;
};

struct EdgeData {
  RigData rig;
  std::int64_t factor_id = -1, source_pts = -1;
  std::vector<double> world, observed, scales;
  std::vector<std::int64_t> frozen_segments;
  double sigma = 0.0;
  int dynamic = 0, fixed_mount = 0, central_domain = 0;
  std::array<double, 3> fixed_center{};
  bool cache_enabled = false;
  std::size_t rows() const { return observed.size(); }
  EdgeFactorDesc descriptor() const {
    EdgeFactorDesc out{};
    out.world_xyz = world.data();
    out.world_points = static_cast<std::int64_t>(world.size() / 3);
    out.observed_uv = observed.data();
    out.quality_scales = scales.data();
    out.observations = static_cast<std::int64_t>(observed.size() / 2);
    out.frozen_segments = frozen_segments.data();
    out.frozen_count = static_cast<std::int64_t>(frozen_segments.size());
    out.sigma = sigma;
    out.dynamic_candidates = dynamic;
    out.fixed_mount = fixed_mount;
    out.central_domain = central_domain;
    std::copy(fixed_center.begin(), fixed_center.end(), out.fixed_center_xyz);
    return out;
  }
};

int role(const py::dict& roles, const char* name, bool optional, std::size_t n) {
  if (!roles.contains(name)) {
    if (optional) return -1;
    throw std::invalid_argument(std::string("missing rig role: ") + name);
  }
  int value = py::cast<int>(roles[name]);
  if (value == -1 && optional) return value;
  if (value < 0 || static_cast<std::size_t>(value) >= n)
    throw std::invalid_argument(std::string("invalid rig role: ") + name);
  return value;
}

std::shared_ptr<const EdgeData> parse_edge_data(const py::dict& spec,
                                                const gtsam::SharedNoiseModel& noise) {
  auto d = std::make_shared<EdgeData>();
  d->factor_id = required(spec, "factor_id").cast<std::int64_t>();
  d->source_pts = required(spec, "source_pts").cast<std::int64_t>();
  if (d->factor_id < 0 || d->source_pts < 0)
    throw std::invalid_argument("factor ID and source PTS must be nonnegative");
  auto& rig = d->rig;
  rig.deps = required(spec, "dependency_keys").cast<std::vector<gtsam::Key>>();
  rig.active_keys = required(spec, "active_keys").cast<std::vector<gtsam::Key>>();
  const std::size_t n = rig.deps.size();
  if (n == 0 || rig.active_keys.empty())
    throw std::invalid_argument("source-edge factor needs dependencies and active keys");
  std::unordered_map<gtsam::Key, int> index;
  for (std::size_t i = 0; i < n; ++i) {
    if (!index.emplace(rig.deps[i], static_cast<int>(i)).second)
      throw std::invalid_argument("duplicate dependency key");
  }
  rig.active_position.assign(n, -1);
  for (std::size_t i = 0; i < rig.active_keys.size(); ++i) {
    const auto it = index.find(rig.active_keys[i]);
    if (it == index.end() || rig.active_position[it->second] != -1)
      throw std::invalid_argument("active keys must be a unique dependency subset");
    rig.active_position[it->second] = static_cast<int>(i);
  }
  auto states = flat_array<double>(required(spec, "frozen_states"), 2, {11}, "frozen_states");
  if (states.size() != n * 11) throw std::invalid_argument("frozen_states row count mismatch");
  rig.frozen.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int j = 0; j < 11; ++j) {
      const double value = states[11 * i + j];
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite frozen state");
      rig.frozen[i][j] = value;
    }
  }
  auto roles = required(spec, "roles").cast<py::dict>();
  rig.frame = role(roles, "frame", false, n);
  rig.lens_base = role(roles, "lens_base", false, n);
  rig.lens_slope = role(roles, "lens_slope", true, n);
  rig.mount = role(roles, "mount", false, n);
  rig.offset = role(roles, "offset", false, n);
  rig.offset_slope = role(roles, "offset_slope", false, n);
  rig.brown = role(roles, "brown", false, n);
  rig.brown_slope = role(roles, "brown_slope", true, n);
  for (auto item : required(roles, "controls").cast<py::iterable>()) {
    py::tuple pair = py::cast<py::tuple>(item);
    if (pair.size() != 2) throw std::invalid_argument("control must be (index, weight)");
    int pos = pair[0].cast<int>();
    double weight = pair[1].cast<double>();
    if (pos < 0 || static_cast<std::size_t>(pos) >= n || !std::isfinite(weight))
      throw std::invalid_argument("invalid spline control");
    rig.controls.emplace_back(pos, weight);
  }
  if (rig.controls.empty()) throw std::invalid_argument("spline controls missing");
  rig.zref = required(spec, "zoom_reference").cast<double>();
  if (!std::isfinite(rig.zref)) throw std::invalid_argument("nonfinite zoom reference");

  d->world = flat_array<double>(required(spec, "world_xyz"), 2, {3}, "world_xyz");
  d->observed = flat_array<double>(required(spec, "observed_uv"), 2, {2}, "observed_uv");
  d->scales = flat_array<double>(required(spec, "quality_scales"), 1, {}, "quality_scales");
  d->frozen_segments = flat_array<std::int64_t>(
      required(spec, "frozen_segments"), 1, {}, "frozen_segments");
  d->sigma = required(spec, "sigma").cast<double>();
  d->dynamic = required(spec, "dynamic_candidates").cast<bool>() ? 1 : 0;
  d->fixed_mount = required(spec, "fixed_mount").cast<bool>() ? 1 : 0;
  d->central_domain = required(spec, "central_domain").cast<bool>() ? 1 : 0;
  d->cache_enabled = spec.contains("cache") && py::cast<bool>(spec["cache"]);
  auto fixed_center = flat_array<double>(required(spec, "fixed_center_xyz"), 1, {},
                                         "fixed_center_xyz");
  if (fixed_center.size() != 3) throw std::invalid_argument("fixed_center_xyz must have 3 values");
  std::copy(fixed_center.begin(), fixed_center.end(), d->fixed_center.begin());
  if (d->world.size() < 6 || d->observed.empty() ||
      d->scales.size() != d->observed.size() / 2 || !(d->sigma > 0.0) ||
      !std::isfinite(d->sigma) || noise->dim() != d->rows())
    throw std::invalid_argument("source-edge dimensions or sigma mismatch");
  for (double x : d->world) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite world");
  for (double x : d->observed) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite observation");
  for (double x : d->scales) if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("invalid scale");
  for (double x : d->fixed_center) if (!std::isfinite(x)) throw std::invalid_argument("invalid fixed center");
  for (auto x : d->frozen_segments)
    if (x < 0 || x >= static_cast<std::int64_t>(d->world.size() / 3 - 1))
      throw std::invalid_argument("frozen segment outside world curve");
  return d;
}

void physical_and_chain(const RigData& rig, const gtsam::Values& values,
                        Vec15& q, std::vector<Chain>* chain) {
  std::vector<Vec11> x = rig.frozen;
  for (std::size_t i = 0; i < rig.deps.size(); ++i) {
    if (rig.active_position[i] >= 0) {
      const gtsam::Vector& state = values.at<gtsam::Vector>(rig.deps[i]);
      if (state.size() != 11 || !state.allFinite())
        throw std::invalid_argument("active graph state must be finite Vector11");
      x[i] = state;
    }
  }
  if (chain) chain->assign(rig.deps.size(), Chain::Zero());
  const Vec11& frame = x[rig.frame];
  const Vec11& base = x[rig.lens_base];
  const Vec11 slope = rig.lens_slope >= 0 ? x[rig.lens_slope] : Vec11::Zero().eval();
  const Vec11& mount = x[rig.mount];
  const Vec11& offset0 = x[rig.offset];
  const Vec11& offset1 = x[rig.offset_slope];
  const Vec11& brown0 = x[rig.brown];
  const Vec11 brown1 = rig.brown_slope >= 0 ? x[rig.brown_slope] : Vec11::Zero().eval();
  const double z = frame[6], dz = z - rig.zref;
  Eigen::Vector3d shift = offset0.head<3>() + dz * offset1.head<3>();

  // Reuse the exact Rodrigues implementation used by the native projection.
  double rotation_state[15] = {};
  for (int i = 0; i < 3; ++i) rotation_state[3 + i] = frame[3 + i];
  rs::Brown15Projection projection;
  if (!projection.prepare(rotation_state))
    throw std::runtime_error("Rodrigues preparation failed");
  using Row3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
  const Eigen::Map<const Row3> R(projection.rotation_matrix());

  q.setZero();
  q.head<3>() = mount.head<3>() + R.transpose() * shift;
  for (auto [pos, weight] : rig.controls) {
    q.head<3>() += weight * x[pos].head<3>();
    if (chain) (*chain)[pos].block<3, 3>(0, 0) += weight * Eigen::Matrix3d::Identity();
  }
  q.segment<3>(3) = frame.segment<3>(3);
  const double ratio = base[6] + slope[6] * dz;
  q[6] = z + 0.5 * ratio;
  q[7] = z - 0.5 * ratio;
  q.segment<3>(8) = base.segment<3>(8) + dz * slope.segment<3>(8);
  q.segment<4>(11) = brown0.head<4>() + dz * brown1.head<4>();
  if (!q.allFinite()) throw std::runtime_error("nonfinite physical15");
  if (!chain) return;

  Chain& jf = (*chain)[rig.frame];
  for (int axis = 0; axis < 3; ++axis) {
    const Eigen::Map<const Row3> dR(projection.rotation_derivative(axis));
    jf.block<3, 1>(0, 3 + axis) += dR.transpose() * shift;
    jf(3 + axis, 3 + axis) += 1.0;
  }
  jf.block<3, 1>(0, 6) += R.transpose() * offset1.head<3>();
  jf(6, 6) += 1.0 + 0.5 * slope[6];
  jf(7, 6) += 1.0 - 0.5 * slope[6];
  jf.block<3, 1>(8, 6) += slope.segment<3>(8);
  jf.block<4, 1>(11, 6) += brown1.head<4>();

  Chain& jb = (*chain)[rig.lens_base];
  jb(6, 6) += 0.5; jb(7, 6) -= 0.5;
  jb.block<3, 3>(8, 8) += Eigen::Matrix3d::Identity();
  if (rig.lens_slope >= 0) {
    Chain& js = (*chain)[rig.lens_slope];
    js(6, 6) += 0.5 * dz; js(7, 6) -= 0.5 * dz;
    js.block<3, 3>(8, 8) += dz * Eigen::Matrix3d::Identity();
  }
  (*chain)[rig.mount].block<3, 3>(0, 0) += Eigen::Matrix3d::Identity();
  (*chain)[rig.offset].block<3, 3>(0, 0) += R.transpose();
  (*chain)[rig.offset_slope].block<3, 3>(0, 0) += dz * R.transpose();
  (*chain)[rig.brown].block<4, 4>(11, 0) += Eigen::Matrix4d::Identity();
  if (rig.brown_slope >= 0)
    (*chain)[rig.brown_slope].block<4, 4>(11, 0) += dz * Eigen::Matrix4d::Identity();
}

class NativeEdgeFactor final : public gtsam::NoiseModelFactor {
  std::shared_ptr<const EdgeData> data_;
  struct Cache {
    std::mutex lock;
    // Keep the accepted-state Jacobian when a rejected trial probes error().
    bool jac_valid = false, residual_valid = false;
    std::array<double, 15> jac_state{}, residual_state{};
    std::vector<double> jac_residual, jacobian, residual;
    std::uint64_t payload_bytes() const {
      return static_cast<std::uint64_t>(sizeof(double) *
          ((jac_valid ? 15 : 0) + (residual_valid ? 15 : 0) +
           jac_residual.capacity() + jacobian.capacity() + residual.capacity()));
    }
  };
  mutable Cache cache_;
 public:
  NativeEdgeFactor(const gtsam::SharedNoiseModel& noise,
                   std::shared_ptr<const EdgeData> data)
      : gtsam::NoiseModelFactor(noise, data->rig.active_keys), data_(std::move(data)) {}
  ~NativeEdgeFactor() override {
    if (data_->cache_enabled) edge_cache_payload_bytes -= cache_.payload_bytes();
  }
  bool sendable() const override { return true; }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<NativeEdgeFactor>(noiseModel_, data_);
  }
  gtsam::Vector unwhitenedError(
      const gtsam::Values& values,
      boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
    const bool with_jac = static_cast<bool>(H);
    Vec15 q;
    std::vector<Chain> chain;
    physical_and_chain(data_->rig, values, q, with_jac ? &chain : nullptr);
    std::vector<double> residual, jacobian;
    bool hit = false;
    if (data_->cache_enabled) {
      std::lock_guard<std::mutex> guard(cache_.lock);
      if (cache_.jac_valid &&
          std::memcmp(cache_.jac_state.data(), q.data(), 15 * sizeof(double)) == 0) {
        residual = cache_.jac_residual;
        if (with_jac) jacobian = cache_.jacobian;
        hit = true;
      } else if (!with_jac && cache_.residual_valid &&
                 std::memcmp(cache_.residual_state.data(), q.data(),
                             15 * sizeof(double)) == 0) {
        residual = cache_.residual;
        hit = true;
      }
    }
    if (hit) {
      ++edge_cache_hits;
    } else {
      residual.resize(data_->rows());
      if (with_jac) jacobian.resize(data_->rows() * 15);
      auto desc = data_->descriptor();
      const int code = rs_edge_eval(&desc, q.data(), residual.data(),
                                    with_jac ? jacobian.data() : nullptr,
                                    with_jac ? 0 : 1);
      ++edge_calls;
      if (code == 4) {
        ++edge_domain_calls;
        throw NativeBoardDomainError(data_->rows() / 2, data_->factor_id,
                                     data_->source_pts, with_jac);
      }
      if (code == 2) {
        ++edge_unsupported_calls;
        throw NativeUnsupportedError(data_->factor_id, data_->source_pts,
                                     with_jac,
                                     with_jac ? nullptr :
                                       std::make_shared<gtsam::Values>(values));
      }
      if (code != 0)
        throw std::runtime_error("rs_edge_eval status " + std::to_string(code) +
            " factor_id=" + std::to_string(data_->factor_id) +
            " source_pts=" + std::to_string(data_->source_pts) +
            " stage=" + (with_jac ? "jacobian" : "residual"));
      if (data_->cache_enabled) {
        std::lock_guard<std::mutex> guard(cache_.lock);
        const std::uint64_t old_bytes = cache_.payload_bytes();
        if (with_jac) {
          cache_.jac_valid = true;
          std::copy(q.data(), q.data() + 15, cache_.jac_state.begin());
          cache_.jac_residual = residual;
          cache_.jacobian = jacobian;
        } else {
          cache_.residual_valid = true;
          std::copy(q.data(), q.data() + 15, cache_.residual_state.begin());
          cache_.residual = residual;
        }
        const std::uint64_t new_bytes = cache_.payload_bytes();
        if (new_bytes >= old_bytes) edge_cache_payload_bytes += new_bytes - old_bytes;
        else edge_cache_payload_bytes -= old_bytes - new_bytes;
      }
    }
    gtsam::Vector result(static_cast<Eigen::Index>(residual.size()));
    std::copy(residual.begin(), residual.end(), result.data());
    if (with_jac) {
      H->assign(data_->rig.active_keys.size(),
                gtsam::Matrix::Zero(result.size(), 11));
      Eigen::Map<const RowMatrix15> J(jacobian.data(), result.size(), 15);
      for (std::size_t i = 0; i < chain.size(); ++i) {
        const int slot = data_->rig.active_position[i];
        if (slot >= 0) (*H)[slot].noalias() += J * chain[i];
      }
    }
    return result;
  }
};

struct LinearData {
  std::vector<gtsam::Key> keys;
  std::vector<gtsam::Matrix> matrices;
  gtsam::Vector target;
};
class NativeLinearFactor final : public gtsam::NoiseModelFactor {
  std::shared_ptr<const LinearData> data_;
 public:
  NativeLinearFactor(const gtsam::SharedNoiseModel& noise,
                     std::shared_ptr<const LinearData> data)
      : gtsam::NoiseModelFactor(noise, data->keys), data_(std::move(data)) {}
  bool sendable() const override { return true; }
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<NativeLinearFactor>(noiseModel_, data_);
  }
  gtsam::Vector unwhitenedError(
      const gtsam::Values& values,
      boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
    gtsam::Vector r = -data_->target;
    if (H) *H = data_->matrices;
    for (std::size_t i = 0; i < data_->keys.size(); ++i) {
      const gtsam::Vector& x = values.at<gtsam::Vector>(data_->keys[i]);
      if (x.size() != 11 || !x.allFinite())
        throw std::invalid_argument("linear factor expects finite Vector11");
      r.noalias() += data_->matrices[i] * x;
    }
    return r;
  }
};

std::shared_ptr<const LinearData> parse_linear_data(
    py::iterable keys, py::iterable matrices, const gtsam::Vector& target,
    const gtsam::SharedNoiseModel& noise) {
  auto d = std::make_shared<LinearData>();
  for (py::handle item : keys) d->keys.push_back(py::cast<gtsam::Key>(item));
  for (py::handle item : matrices) d->matrices.push_back(py::cast<gtsam::Matrix>(item));
  d->target = target;
  if (d->keys.empty() || d->keys.size() != d->matrices.size() ||
      target.size() == 0 || noise->dim() != static_cast<std::size_t>(target.size()))
    throw std::invalid_argument("linear factor size mismatch");
  for (const auto& A : d->matrices)
    if (A.rows() != target.size() || A.cols() != 11 || !A.allFinite())
      throw std::invalid_argument("linear factor matrix must be finite rows x 11");
  if (!target.allFinite()) throw std::invalid_argument("nonfinite linear target");
  return d;
}
}  // namespace

PYBIND11_MODULE(native_source_edge_trial_prod, m) {
  m.doc() = "Native GTSAM 4.2.2 source-edge and fixed linear Vector11 factors";
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const NativeUnsupportedError& error) {
      try {
        py::object cls = py::module_::import("continuous_native_edge_optin")
                             .attr("NativeEdgeUnsupported");
        py::object instance = cls(error.factor_id, error.source_pts,
                                  error.jacobian ? "jacobian" : "residual", 2);
        if (error.trial_values) {
          // py::cast(copy) gives Python an independent, owning Values object;
          // the C++ exception may be destroyed immediately after translation.
          instance.attr("trial_values") = py::cast(
              *error.trial_values, py::return_value_policy::copy);
        }
        PyErr_SetObject(cls.ptr(), instance.ptr());
      } catch (py::error_already_set& import_error) {
        import_error.restore();
      }
    } catch (const NativeBoardDomainError& error) {
      try {
        py::object cls = py::module_::import("reference_board_pair")
                             .attr("BoardIntersectionError");
        py::list indices;
        for (std::size_t i = 0; i < error.observations; ++i)
          indices.append(py::int_(i));
        py::object instance = cls("source_edge_no_central_segment", indices);
        instance.attr("factor_id") = py::int_(error.factor_id);
        instance.attr("source_pts") = py::int_(error.source_pts);
        instance.attr("evaluation_stage") = py::str(
            error.jacobian ? "jacobian" : "residual");
        instance.attr("factor_family") = py::str("source_edge");
        instance.attr("source_keys") = py::make_tuple(error.source_pts);
        PyErr_SetObject(cls.ptr(), instance.ptr());
      } catch (py::error_already_set& import_error) {
        import_error.restore();
      }
    }
  });
  m.def("add_factor", [](gtsam::NonlinearFactorGraph& graph,
                          const gtsam::SharedNoiseModel& noise, py::dict spec) {
    auto data = parse_edge_data(spec, noise);
    graph.push_back(boost::make_shared<NativeEdgeFactor>(noise, data));
    return graph.size() - 1;
  }, py::arg("graph"), py::arg("noise"), py::arg("spec"));
  m.def("evaluate_factor", [](const gtsam::SharedNoiseModel& noise, py::dict spec,
                               const gtsam::Values& values, bool jacobian) {
    NativeEdgeFactor factor(noise, parse_edge_data(spec, noise));
    std::vector<gtsam::Matrix> H;
    gtsam::Vector r = jacobian ? factor.unwhitenedError(values, H) :
                                factor.unwhitenedError(values);
    return py::make_tuple(r, H);
  }, py::arg("noise"), py::arg("spec"), py::arg("values"),
     py::arg("jacobian") = true);
  m.def("project_probe", [](const Eigen::Matrix<double, 15, 1>& q,
                            const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                                Eigen::RowMajor>& world) {
    rs::Brown15Projection camera;
    if (!camera.prepare(q.data()) || !camera.numerically_supported_camera(q.data()))
      throw std::invalid_argument("projection probe outside native numeric range");
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> uv(world.rows(), 2);
    Eigen::VectorXd depth(world.rows());
    Eigen::VectorXi status(world.rows());
    for (Eigen::Index i = 0; i < world.rows(); ++i) {
      double pixel[2], z = NAN;
      const bool ok = camera.project(world.row(i).data(), pixel, &z, nullptr);
      uv(i, 0) = ok ? pixel[0] : NAN;
      uv(i, 1) = ok ? pixel[1] : NAN;
      depth(i) = z;
      status(i) = ok ? camera.central_status_from_projection(
                             world.row(i).data(), pixel, z) : 2;
    }
    return py::make_tuple(uv, depth, status);
  }, py::arg("physical15"), py::arg("world_xyz"));
  m.def("add_linear_factor", [](gtsam::NonlinearFactorGraph& graph,
       const gtsam::SharedNoiseModel& noise, py::iterable keys,
       py::iterable matrices, const gtsam::Vector& target) {
    auto data = parse_linear_data(keys, matrices, target, noise);
    graph.push_back(boost::make_shared<NativeLinearFactor>(noise, data));
    return graph.size() - 1;
  }, py::arg("graph"), py::arg("noise"), py::arg("keys"),
     py::arg("matrices"), py::arg("target"));
  m.def("add_linear_factor", [](gtsam::NonlinearFactorGraph& graph,
       py::iterable keys, py::iterable matrices,
       const gtsam::Vector& target, double sigma) {
    if (!std::isfinite(sigma) || sigma <= 0.0)
      throw std::invalid_argument("linear factor sigma must be positive");
    gtsam::SharedNoiseModel noise = gtsam::noiseModel::Isotropic::Sigma(
        static_cast<std::size_t>(target.size()), sigma);
    auto data = parse_linear_data(keys, matrices, target, noise);
    graph.push_back(boost::make_shared<NativeLinearFactor>(noise, data));
    return graph.size() - 1;
  }, py::arg("graph"), py::arg("keys"), py::arg("matrices"),
     py::arg("target"), py::arg("sigma"));
  m.def("set_threads", [](int count) {
    if (count < 1) throw std::invalid_argument("thread count must be positive");
    std::lock_guard<std::mutex> guard(tbb_mutex);
    tbb_limit.reset(new oneapi::tbb::global_control(
        oneapi::tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(count)));
    return count;
  }, py::arg("count"));
  m.def("clear_threads", []() {
    std::lock_guard<std::mutex> guard(tbb_mutex);
    tbb_limit.reset();
    return oneapi::tbb::global_control::active_value(
        oneapi::tbb::global_control::max_allowed_parallelism);
  });
  m.def("stats", []() {
    py::dict out;
    out["edge_kernel_calls"] = edge_calls.load();
    out["edge_unsupported_calls"] = edge_unsupported_calls.load();
    out["edge_domain_calls"] = edge_domain_calls.load();
    out["edge_cache_hits"] = edge_cache_hits.load();
    out["edge_cache_payload_bytes"] = edge_cache_payload_bytes.load();
    out["gtsam_version"] = GTSAM_VERSION_STRING;
    out["tbb_active_max_allowed_parallelism"] =
        oneapi::tbb::global_control::active_value(
            oneapi::tbb::global_control::max_allowed_parallelism);
#ifdef GTSAM_USE_TBB
    out["gtsam_tbb_enabled"] = true;
#else
    out["gtsam_tbb_enabled"] = false;
#endif
    return out;
  });
  m.attr("metadata") = m.attr("stats");
  m.def("reset_stats", []() {
    edge_calls = 0;
    edge_cache_hits = 0;
    edge_unsupported_calls = 0;
    edge_domain_calls = 0;
  });
}
