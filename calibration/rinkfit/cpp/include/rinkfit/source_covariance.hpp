#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rinkfit {
struct SourceCovarianceRankProof {
  double source_rank_threshold = 0.0;
  std::size_t retained_rank = 0;
};
struct CovarianceMarginalAudit {
  std::size_t computed_retained_rank = 0;
  double source_rank_threshold = 0.0;
  double backward_error_threshold = 0.0;
  double effective_rank_threshold = 0.0;
  double retained_factor_frobenius_norm = 0.0;
  double accepted_min_pivot = 0.0;
  double discarded_max_pivot = 0.0;
  double whitening_max_abs_error = 0.0;
  double cgc_max_abs_error = 0.0;
  double gcg_max_abs_error = 0.0;
  double gc_symmetry_max_abs_error = 0.0;
};
namespace sourcecov_detail {
inline void require(bool pass, const char* message) {
  if (!pass) throw std::runtime_error(std::string("supported covariance marginal: ") + message);
}
inline std::size_t product(std::size_t a, std::size_t b) {
  require(a == 0 || b <= std::numeric_limits<std::size_t>::max() / a, "matrix size overflow");
  return a * b;
}
inline double dot(const double* a, const double* b, std::size_t n) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<long double>(a[i]) * b[i];
  const double result = static_cast<double>(sum);
  require(std::isfinite(result), "nonfinite matrix arithmetic");
  return result;
}
inline double norm(const std::vector<double>& x) {
  double result = 0.0;
  for (double value : x) result = std::hypot(result, value);
  require(std::isfinite(result), "nonfinite vector norm");
  return result;
}
inline std::vector<double> transpose(const std::vector<double>& x, std::size_t rows, std::size_t cols) {
  std::vector<double> result(product(rows, cols));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) result[j * rows + i] = x[i * cols + j];
  return result;
}
inline std::vector<double> multiply(const std::vector<double>& a, std::size_t rows,
                                    std::size_t inner, const std::vector<double>& b, std::size_t cols) {
  require(a.size() == product(rows, inner) && b.size() == product(inner, cols), "matrix product shape");
  std::vector<double> result(product(rows, cols));
  const auto bt = transpose(b, inner, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      result[i * cols + j] = dot(a.data() + i * inner, bt.data() + j * inner, inner);
  return result;
}
inline double check_close(const std::vector<double>& actual, const std::vector<double>& expected,
                          double rtol, double atol, const char* message) {
  require(actual.size() == expected.size(), "identity comparison shape");
  double maximum = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite identity comparison");
    const double error = std::abs(actual[i] - expected[i]);
    maximum = std::max(maximum, error);
    require(error <= atol + rtol * std::abs(expected[i]), message);
  }
  return maximum;
}
// Column-pivoted, twice-reorthogonalized MGS. The numerical support is measured
// from F_keep itself; no declared rank or proof singular value is trusted.
inline void support_rank(const std::vector<double>& factor, std::size_t rows,
                         std::size_t cols, CovarianceMarginalAudit& audit) {
  std::vector<std::vector<double>> work(cols, std::vector<double>(rows));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) work[j][i] = factor[i * cols + j];
  std::vector<std::vector<double>> basis;
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < cols; ++k) {
    std::size_t pivot = k; double maximum = 0.0;
    for (std::size_t j = k; j < cols; ++j) {
      const double value = norm(work[j]);
      if (value > maximum) { maximum = value; pivot = j; }
    }
    if (maximum <= audit.effective_rank_threshold) {
      audit.discarded_max_pivot = maximum; break;
    }
    std::swap(work[k], work[pivot]);
    auto q = work[k];
    for (int pass = 0; pass < 2; ++pass)
      for (const auto& previous : basis) {
        const double value = dot(q.data(), previous.data(), rows);
        for (std::size_t i = 0; i < rows; ++i) q[i] -= value * previous[i];
      }
    const double magnitude = norm(q);
    require(magnitude > audit.effective_rank_threshold, "support rank is unresolved at its declared source threshold");
    minimum = std::min(minimum, magnitude);
    for (double& value : q) value /= magnitude;
    basis.push_back(q);
    for (std::size_t j = k + 1; j < cols; ++j)
      for (int pass = 0; pass < 2; ++pass) {
        const double value = dot(work[j].data(), q.data(), rows);
        for (std::size_t i = 0; i < rows; ++i) work[j][i] -= value * q[i];
      }
  }
  audit.computed_retained_rank = basis.size();
  audit.accepted_min_pivot = basis.empty() ? 0.0 : minimum;
}
} // namespace sourcecov_detail

// The caller binds the rank proof threshold, original W hash, source rows,
// original UV/IDs and retained-coordinate order to their immutable source.
// This function independently verifies the covariance and numerical support.
// All matrices are row major. No eigenvalue floor, diagonal jitter or ridge
// is added. A failed Cholesky is a rejection, never a regularization request.
inline CovarianceMarginalAudit validate_supported_covariance_marginal(
    const std::vector<double>& original_whitener, std::size_t original_rank,
    std::size_t original_coordinate_count, const std::vector<std::size_t>& kept,
    const std::vector<double>& new_whitener, std::size_t new_rank,
    const SourceCovarianceRankProof& rank_proof, double rtol = 1e-10, double atol = 1e-11) {
  using namespace sourcecov_detail;
  const std::size_t count = kept.size();
  require(original_rank > 0 && original_rank <= original_coordinate_count, "original full-row-rank shape");
  require(original_whitener.size() == product(original_rank, original_coordinate_count), "original whitening shape");
  require(new_rank <= count && new_whitener.size() == product(new_rank, count), "new whitening shape");
  require(std::isfinite(rtol) && std::isfinite(atol) && rtol >= 0.0 && atol >= 0.0, "invalid identity tolerances");
  require(std::isfinite(rank_proof.source_rank_threshold) && rank_proof.source_rank_threshold >= 0.0, "invalid source rank threshold");
  for (double value : original_whitener) require(std::isfinite(value), "nonfinite original W");
  for (double value : new_whitener) require(std::isfinite(value), "nonfinite new W");
  std::vector<bool> seen(original_coordinate_count, false);
  for (std::size_t index : kept) {
    require(index < original_coordinate_count && !seen[index], "duplicate or invalid retained coordinate");
    seen[index] = true;
  }
  // A=W W^T and L L^T=A. Strictly positive Cholesky pivots only.
  std::vector<double> lower(product(original_rank, original_rank), 0.0);
  for (std::size_t i = 0; i < original_rank; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      long double value = dot(original_whitener.data() + i * original_coordinate_count,
                              original_whitener.data() + j * original_coordinate_count, original_coordinate_count);
      for (std::size_t k = 0; k < j; ++k)
        value -= static_cast<long double>(lower[i * original_rank + k]) * lower[j * original_rank + k];
      const double residual = static_cast<double>(value);
      require(std::isfinite(residual), "nonfinite Cholesky residual");
      if (i == j) {
        require(residual > 0.0, "original W is not numerically full row rank");
        lower[i * original_rank + j] = std::sqrt(residual);
      } else lower[i * original_rank + j] = residual / lower[j * original_rank + j];
    }
  // F=W^T A^-1, selecting only the required rows after exact triangular solves.
  std::vector<double> factor(product(count, original_rank));
  for (std::size_t row = 0; row < count; ++row) {
    std::vector<double> solution(original_rank);
    for (std::size_t i = 0; i < original_rank; ++i) {
      long double value = original_whitener[i * original_coordinate_count + kept[row]];
      for (std::size_t j = 0; j < i; ++j) value -= static_cast<long double>(lower[i * original_rank + j]) * solution[j];
      solution[i] = static_cast<double>(value) / lower[i * original_rank + i];
    }
    for (std::size_t ii = original_rank; ii-- > 0;) {
      long double value = solution[ii];
      for (std::size_t j = ii + 1; j < original_rank; ++j) value -= static_cast<long double>(lower[j * original_rank + ii]) * solution[j];
      solution[ii] = static_cast<double>(value) / lower[ii * original_rank + ii];
    }
    for (std::size_t j = 0; j < original_rank; ++j) factor[row * original_rank + j] = solution[j];
  }
  CovarianceMarginalAudit audit;
  audit.source_rank_threshold = rank_proof.source_rank_threshold;
  audit.retained_factor_frobenius_norm = norm(factor);
  audit.backward_error_threshold = std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(count, original_rank)) * audit.retained_factor_frobenius_norm;
  audit.effective_rank_threshold = std::max(audit.source_rank_threshold, audit.backward_error_threshold);
  support_rank(factor, count, original_rank, audit);
  require(audit.computed_retained_rank == new_rank && rank_proof.retained_rank == new_rank,
          "retained covariance support rank differs from new W or the proof");
  const auto factor_t = transpose(factor, count, original_rank);
  const auto covariance = multiply(factor, count, original_rank, factor_t, count);
  const auto new_w_t = transpose(new_whitener, new_rank, count);
  const auto precision = multiply(new_w_t, count, new_rank, new_whitener, count);
  const auto whitened = multiply(multiply(new_whitener, new_rank, count, covariance, count),
                                  new_rank, count, new_w_t, new_rank);
  std::vector<double> identity(product(new_rank, new_rank), 0.0);
  for (std::size_t i = 0; i < new_rank; ++i) identity[i * new_rank + i] = 1.0;
  audit.whitening_max_abs_error = check_close(whitened, identity, rtol, atol, "new W does not whiten the original covariance marginal");
  const auto cg = multiply(covariance, count, count, precision, count);
  const auto gc = multiply(precision, count, count, covariance, count);
  audit.cgc_max_abs_error = check_close(multiply(cg, count, count, covariance, count), covariance, rtol, atol, "C G C differs from C");
  audit.gcg_max_abs_error = check_close(multiply(gc, count, count, precision, count), precision, rtol, atol, "G C G differs from G");
  audit.gc_symmetry_max_abs_error = check_close(gc, transpose(gc, count, count), rtol, atol, "G C adds information outside original covariance support");
  return audit;
}
} // namespace rinkfit
