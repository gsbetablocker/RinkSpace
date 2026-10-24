#pragma once
#include "rinkfit/joint_manifold.hpp"
#include "rinkfit/joint_solver_metric_v1.hpp"
#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace rinkfit {
// Only constant-Jacobian manifolds are pruned coordinate by coordinate. A zero
// row of a nonlinear manifold at one point need not stay zero at the next point.
inline std::vector<unsigned char> joint_gpu_ambient_activity(
    const ceres::Problem& problem, const double* values, int ambient) {
  std::vector<unsigned char> active(ambient, 0);
  if (!problem.HasParameterBlock(values) || problem.IsParameterBlockConstant(values))
    return active;
  if (problem.ParameterBlockSize(values) != ambient)
    throw std::runtime_error("GPU active-column physical block shape mismatch");
  const auto* manifold = joint_solver_original_manifold_v1(problem.GetManifold(values));
  if (!manifold || (!dynamic_cast<const ceres::SubsetManifold*>(manifold) &&
                    !dynamic_cast<const JointLinearControlManifold*>(manifold))) {
    std::fill(active.begin(), active.end(), 1);
    return active;
  }
  const int tangent = problem.ParameterBlockTangentSize(values);
  if (!tangent) return active;
  std::vector<double> plus(std::size_t(ambient) * tangent);
  if (!manifold->PlusJacobian(values, plus.data()))
    throw std::runtime_error("GPU active-column PlusJacobian failed");
  for (int a = 0; a < ambient; ++a)
    for (int t = 0; t < tangent; ++t) {
      const double value = plus[std::size_t(a) * tangent + t];
      if (!std::isfinite(value))
        throw std::runtime_error("GPU active-column nonfinite tangent basis");
      active[a] |= value != 0.;
    }
  return active;
}
}
