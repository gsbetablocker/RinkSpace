#ifndef RINKSPACE_BROWN15_PROJECTION_HPP
#define RINKSPACE_BROWN15_PROJECTION_HPP

#include <cmath>

// Physical15 projection in scene coordinates, with the four active Brown terms.
// Header-only and independent of OpenCV. All Jacobians are with respect to q15.
namespace rinkspace_native {

class Brown15Projection {
 public:
  // q15 = [C_xyz / 100, Rodrigues rvec, log(fx), log(fy),
  //        cx / 1000, cy / 1000, skew / 1000, k1, k2, p1, p2].
  // The remaining ten Brown14 coefficients are fixed at zero.
  bool prepare(const double q15[15]) noexcept {
    ready_ = false;
    if (q15 == nullptr) return false;
    for (int i = 0; i < 15; ++i) {
      if (!std::isfinite(q15[i])) return false;
    }
    for (int i = 0; i < 3; ++i) center_[i] = 100.0 * q15[i];
    fx_ = std::exp(q15[6]);
    fy_ = std::exp(q15[7]);
    cx_ = 1000.0 * q15[8];
    cy_ = 1000.0 * q15[9];
    skew_ = 1000.0 * q15[10];
    for (int i = 0; i < 4; ++i) brown_[i] = q15[11 + i];
    has_brown_ = false;
    for (double coefficient : brown_) {
      if (std::abs(coefficient) > 1e-14) has_brown_ = true;
    }
    if (!(std::isfinite(fx_) && std::isfinite(fy_) && fx_ > 0.0 && fy_ > 0.0 &&
          std::isfinite(cx_) && std::isfinite(cy_) && std::isfinite(skew_))) {
      return false;
    }
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(center_[i])) return false;
    }

    const double w[3] = {q15[3], q15[4], q15[5]};
    const double s = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    double a, b, da_ds, db_ds;
    if (s < 0.01) {
      // Taylor form avoids cancellation in d[(1-cos(theta))/theta^2]/ds.
      a = 1.0 + s * (-1.0 / 6.0 + s * (1.0 / 120.0 +
          s * (-1.0 / 5040.0 + s * (1.0 / 362880.0 - s / 39916800.0))));
      b = 0.5 + s * (-1.0 / 24.0 + s * (1.0 / 720.0 +
          s * (-1.0 / 40320.0 + s * (1.0 / 3628800.0 - s / 479001600.0))));
      da_ds = -1.0 / 6.0 + s * (1.0 / 60.0 + s * (-1.0 / 1680.0 +
          s * (1.0 / 90720.0 - s / 7983360.0)));
      db_ds = -1.0 / 24.0 + s * (1.0 / 360.0 + s * (-1.0 / 13440.0 +
          s * (1.0 / 907200.0 - s / 95800320.0)));
    } else {
      const double theta = std::sqrt(s);
      const double sn = std::sin(theta);
      const double cs = std::cos(theta);
      a = sn / theta;
      b = (1.0 - cs) / s;
      da_ds = (theta * cs - sn) / (2.0 * s * theta);
      db_ds = (theta * sn - 2.0 * (1.0 - cs)) / (2.0 * s * s);
    }
    const double W[9] = {0.0, -w[2], w[1], w[2], 0.0, -w[0],
                         -w[1], w[0], 0.0};
    const double dW[3][9] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    double W2[9];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        const int at = 3 * row + col;
        double ww = 0.0;
        for (int m = 0; m < 3; ++m) ww += W[3 * row + m] * W[3 * m + col];
        W2[at] = ww;
        rotation_[at] = (row == col ? 1.0 : 0.0) + a * W[at] + b * ww;
      }
    }
    for (int k = 0; k < 3; ++k) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          const int at = 3 * row + col;
          double dw2 = 0.0;
          for (int m = 0; m < 3; ++m) {
            dw2 += dW[k][3 * row + m] * W[3 * m + col] +
                   W[3 * row + m] * dW[k][3 * m + col];
          }
          drotation_[k][at] = 2.0 * w[k] * da_ds * W[at] +
                               a * dW[k][at] +
                               2.0 * w[k] * db_ds * W2[at] + b * dw2;
        }
      }
    }
    for (double x : rotation_) if (!std::isfinite(x)) return false;
    for (const auto& dr : drotation_) {
      for (double x : dr) if (!std::isfinite(x)) return false;
    }
    ready_ = true;
    return true;
  }

  // Apply this only to a complete physical camera. Rig assembly also uses
  // prepare() for a rotation-only helper with dummy focal parameters.
  bool numerically_supported_camera(const double q15[15]) const noexcept {
    const double rnorm2 = q15[3]*q15[3] + q15[4]*q15[4] + q15[5]*q15[5];
    if (!(rnorm2 >= 1e-4 && rnorm2 <= 9.0) ||
        std::abs(center_[0]) > 1e4 || std::abs(center_[1]) > 1e4 ||
        std::abs(center_[2]) > 1e4 ||
        fx_ < 100.0 || fy_ < 100.0 || fx_ > 2e5 || fy_ > 2e5 ||
        std::abs(cx_) > 1e4 || std::abs(cy_) > 1e4 ||
        std::abs(skew_) > 1e4) return false;
    for (double coefficient : brown_)
      if (std::abs(coefficient) > 0.5) return false;
    return true;
  }

  // uv[2] in native pixels; depth is camera Z; J[30] is row-major (u,v)xq15.
  // J may be nullptr. A negative finite depth is returned for caller filtering.
  // false means unprepared, nonfinite input/output, or numerically singular Z.
  bool normalized(const double xyz[3], double xy[2], double* depth) const noexcept {
    if (!ready_ || xyz == nullptr || xy == nullptr) return false;
    double relative[3];
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(xyz[i])) return false;
      relative[i] = xyz[i] - center_[i];
    }
    double camera[3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        camera[row] += rotation_[3 * row + col] * relative[col];
      }
    }
    const double z = camera[2];
    if (!std::isfinite(z) || std::abs(z) < 1e-12) return false;
    xy[0] = camera[0] / z;
    xy[1] = camera[1] / z;
    if (depth != nullptr) *depth = z;
    return std::isfinite(xy[0]) && std::isfinite(xy[1]);
  }

  // Row-major world-to-camera Rodrigues rotation and its rvec derivative.
  // Pointers remain valid until the next prepare() or destruction; use only
  // after prepare() succeeds. Axis must be 0, 1, or 2.
  const double* rotation_matrix() const noexcept { return rotation_; }
  const double* rotation_derivative(int axis) const noexcept {
    return (axis >= 0 && axis < 3) ? drotation_[axis] : nullptr;
  }

  // Match physical_projection.safe_project's nine station symmetric-Jacobian
  // test for Brown4 (the rational denominator is identically one). Input is
  // an undistorted normalized camera ray. Returns NaN for nonfinite input.
  double axis_minimum_eigenvalue(const double xy[2]) const noexcept {
    if (!ready_ || xy == nullptr || !std::isfinite(xy[0]) ||
        !std::isfinite(xy[1])) return NAN;
    const double k1 = brown_[0], k2 = brown_[1];
    const double p1 = brown_[2], p2 = brown_[3];
    double minimum = INFINITY;
    for (int station = 0; station < 9; ++station) {
      const double alpha = static_cast<double>(station) / 8.0;
      const double x = xy[0] * alpha;
      const double y = xy[1] * alpha;
      const double r2 = x * x + y * y;
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
      const double slope = k1 + 2.0 * k2 * r2;
      const double j11 = radial + 2.0 * x * x * slope + 2.0 * p1 * y + 6.0 * p2 * x;
      const double j12 = 2.0 * x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y;
      const double j21 = 2.0 * x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y;
      const double j22 = radial + 2.0 * y * y * slope + 6.0 * p1 * y + 2.0 * p2 * x;
      const double gap = j11 - j22;
      const double off = j12 + j21;
      const double eigen = 0.5 * (j11 + j22 - std::sqrt(gap * gap + off * off));
      if (!std::isfinite(eigen)) return NAN;
      if (eigen < minimum) minimum = eigen;
    }
    return minimum;
  }

  // Use after a successful project() when its UV/depth are already cached.
  // 0: invalid; 1: valid; 2: threshold/extreme case requiring Python
  // safe_project fallback for strict branch parity. Non-Brown uses only the
  // finite UV and depth>.05 tests, as project_central_valid does.
  int central_status_from_projection(const double xyz[3], const double uv[2],
                                     double depth) const noexcept {
    if (!ready_ || xyz == nullptr || uv == nullptr ||
        !std::isfinite(uv[0]) || !std::isfinite(uv[1]) ||
        !(std::isfinite(depth) && depth > 0.05)) return 0;
    if (!has_brown_) return 1;
    double xy[2];
    if (!normalized(xyz, xy, nullptr)) return 0;
    const double eigen = axis_minimum_eigenvalue(xy);
    if (!std::isfinite(eigen)) return 2;
    // A strongly folded ray is invalid in safe_project regardless of the
    // depth, off-screen magnitude, or OpenCV roundoff checks below. Test it
    // first so a clearly invalid endpoint cannot turn its entire factor into
    // status 2 merely because it lies far outside the source image.
    if (eigen < 1e-8 - 1e-5) return 0;
    if (depth < 0.5 ||
        std::abs(xy[0]) > 4.0 || std::abs(xy[1]) > 4.0 ||
        std::abs(eigen - 1e-8) <= 1e-5) return 2;
    if (eigen <= 1e-8) return 0;
    // project_central_valid additionally requires <1e-5 px agreement with
    // OpenCV's projectPoints, but only on rays that passed the optical-domain
    // test. At huge off-screen magnitudes, different valid floating-point
    // evaluation orders can exceed that absolute tolerance.
    if (std::abs(uv[0]) > 1e5 || std::abs(uv[1]) > 1e5) return 2;
    return 1;
  }

  int central_status(const double xyz[3]) const noexcept {
    double uv[2], depth;
    if (!project(xyz, uv, &depth, nullptr)) return 0;
    return central_status_from_projection(xyz, uv, depth);
  }

  bool project(const double xyz[3], double uv[2], double* depth,
               double J[30]) const noexcept {
    if (!ready_ || xyz == nullptr || uv == nullptr) return false;
    double relative[3];
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(xyz[i])) return false;
      relative[i] = xyz[i] - center_[i];
    }
    double camera[3] = {};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        camera[row] += rotation_[3 * row + col] * relative[col];
      }
    }
    const double z = camera[2];
    if (!std::isfinite(z) || std::abs(z) < 1e-12) return false;
    const double x = camera[0] / z;
    const double y = camera[1] / z;
    const double xx = x * x, yy = y * y, xy = x * y;
    const double r2 = xx + yy, r4 = r2 * r2;
    const double k1 = brown_[0], k2 = brown_[1];
    const double p1 = brown_[2], p2 = brown_[3];
    const double radial = 1.0 + k1 * r2 + k2 * r4;
    const double xd = x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * xx);
    const double yd = y * radial + p1 * (r2 + 2.0 * yy) + 2.0 * p2 * xy;
    uv[0] = fx_ * xd + skew_ * yd + cx_;
    uv[1] = fy_ * yd + cy_;
    if (depth != nullptr) *depth = z;
    if (!(std::isfinite(uv[0]) && std::isfinite(uv[1]))) return false;
    if (J == nullptr) return true;

    for (int i = 0; i < 30; ++i) J[i] = 0.0;
    const double dr_dx = 2.0 * x * (k1 + 2.0 * k2 * r2);
    const double dr_dy = 2.0 * y * (k1 + 2.0 * k2 * r2);
    const double dxd_dx = radial + x * dr_dx + 2.0 * p1 * y + 6.0 * p2 * x;
    const double dxd_dy = x * dr_dy + 2.0 * p1 * x + 2.0 * p2 * y;
    const double dyd_dx = y * dr_dx + 2.0 * p1 * x + 2.0 * p2 * y;
    const double dyd_dy = radial + y * dr_dy + 6.0 * p1 * y + 2.0 * p2 * x;
    const double du_dx = fx_ * dxd_dx + skew_ * dyd_dx;
    const double du_dy = fx_ * dxd_dy + skew_ * dyd_dy;
    const double dv_dx = fy_ * dyd_dx;
    const double dv_dy = fy_ * dyd_dy;
    const double camera_j[2][3] = {
        {du_dx / z, du_dy / z, -(du_dx * x + du_dy * y) / z},
        {dv_dx / z, dv_dy / z, -(dv_dx * x + dv_dy * y) / z}};

    for (int dim = 0; dim < 2; ++dim) {
      double* row_j = J + 15 * dim;
      for (int axis = 0; axis < 3; ++axis) {
        for (int row = 0; row < 3; ++row) {
          row_j[axis] -= 100.0 * camera_j[dim][row] *
                         rotation_[3 * row + axis];
        }
      }
      for (int axis = 0; axis < 3; ++axis) {
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            row_j[3 + axis] += camera_j[dim][row] *
                               drotation_[axis][3 * row + col] * relative[col];
          }
        }
      }
    }
    J[6] = fx_ * xd;
    J[15 + 7] = fy_ * yd;
    J[8] = 1000.0;
    J[15 + 9] = 1000.0;
    J[10] = 1000.0 * yd;

    const double brown_x[4] = {x * r2, x * r4, 2.0 * xy, r2 + 2.0 * xx};
    const double brown_y[4] = {y * r2, y * r4, r2 + 2.0 * yy, 2.0 * xy};
    for (int i = 0; i < 4; ++i) {
      J[11 + i] = fx_ * brown_x[i] + skew_ * brown_y[i];
      J[15 + 11 + i] = fy_ * brown_y[i];
    }
    for (int i = 0; i < 30; ++i) if (!std::isfinite(J[i])) return false;
    return true;
  }

 private:
  double center_[3] = {};
  double rotation_[9] = {};
  double drotation_[3][9] = {};
  double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0, skew_ = 0.0;
  double brown_[4] = {};
  bool has_brown_ = false;
  bool ready_ = false;
};

}  // namespace rinkspace_native

#endif  // RINKSPACE_BROWN15_PROJECTION_HPP
