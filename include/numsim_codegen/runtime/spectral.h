#ifndef NUMSIM_CODEGEN_RUNTIME_SPECTRAL_H
#define NUMSIM_CODEGEN_RUNTIME_SPECTRAL_H

// Runtime support header SHIPPED WITH GENERATED CODE. Depends only on tmech and
// the standard library — never on numsim-cas (which is a codegen-TIME
// dependency, not a runtime dependency of the emitted material).
//
// Provides the one spectral primitive the generated code needs:
// `spectral_decompose(A)` — the ascending eigendecomposition of sym(A). The
// eigenvalue / eigenprojection / eigenvector emitters all read from its result,
// and the codegen shares a single call per distinct tensor argument, so the
// decomposition is computed once per material evaluation.
//
// Ascending order is REQUIRED to match numsim-cas's evaluator: the CAS
// `value(i)/basis(i)/normal(i)` facade numbers eigenpairs by increasing
// eigenvalue (numsim_cas/.../spectral_decomposition_cache.h). Generated and
// runtime code must agree on that ordering or `eig_i`/`E_i` mean different
// things at codegen and run time.

#include <tmech/tmech.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace numsim::codegen::rt {

// Which scalar function the isotropic tensor function / divided difference
// applies to the eigenvalues. Mirrors numsim-cas's isotropic_kind but is
// codegen-local so generated code carries no numsim-cas dependency.
enum class scalar_fn { log, exp, sqrt };

// n-th derivative f^{(n)}(x). Verbatim mirror of numsim-cas's
// iso_detail::apply_fderiv — the two MUST agree so codegen ≡ evaluator.
inline double apply_fderiv(scalar_fn k, double x, std::size_t n) {
  switch (k) {
  case scalar_fn::exp:
    return std::exp(x); // all orders
  case scalar_fn::log: {
    if (n == 0)
      return std::log(x);
    double fact = 1.0;
    for (std::size_t j = 2; j < n; ++j)
      fact *= static_cast<double>(j);
    const double sign = ((n - 1) % 2 == 0) ? 1.0 : -1.0;
    return sign * fact / std::pow(x, static_cast<double>(n));
  }
  case scalar_fn::sqrt: {
    if (n == 0)
      return std::sqrt(x);
    double coef = 1.0;
    for (std::size_t j = 0; j < n; ++j)
      coef *= (0.5 - static_cast<double>(j));
    return coef * std::pow(x, 0.5 - static_cast<double>(n));
  }
  }
  return 0.0;
}

// Confluent (Hermite) divided difference over ascending points p[lo..hi]. A
// coincident span (endpoints equal within a sqrt(eps)-relative band) collapses
// to f^{(hi-lo)}(p_lo)/(hi-lo)!, so coalesced eigenvalues take the analytic
// limit instead of 0/0. Verbatim mirror of numsim-cas's dd_range.
inline double dd_range(scalar_fn k, double const *p, std::size_t lo,
                       std::size_t hi, double rel) {
  const double span = p[hi] - p[lo];
  const double scale = std::max(std::abs(p[lo]), std::abs(p[hi]));
  if (span == 0.0 || std::abs(span) <= rel * scale) {
    double fact = 1.0;
    for (std::size_t j = 2; j <= hi - lo; ++j)
      fact *= static_cast<double>(j);
    return apply_fderiv(k, p[lo], hi - lo) / fact;
  }
  return (dd_range(k, p, lo + 1, hi, rel) - dd_range(k, p, lo, hi - 1, rel)) /
         span;
}

// [f; p_0, ..., p_{N-1}], the divided difference of f over the eigenvalue
// multiset. The single sqrt(eps) tolerance lives here — the one shared
// constant with the evaluator, so generated and runtime never drift.
template <std::size_t N>
double divided_difference(scalar_fn k, std::array<double, N> points) {
  static_assert(N >= 1, "divided_difference needs at least one point");
  std::sort(points.begin(), points.end());
  const double rel = std::sqrt(std::numeric_limits<double>::epsilon());
  return dd_range(k, points.data(), std::size_t{0}, N - 1, rel);
}

// Ascending-sorted eigenvalues and matching eigenvectors of a symmetric
// rank-2 tensor.
template <typename V, std::size_t Dim> struct decomposition {
  std::array<V, Dim> eigenvalues{};
  std::array<tmech::tensor<V, Dim, 1>, Dim> eigenvectors{};
};

// Eigendecomposition of sym(in), ascending. Mirror of numsim-cas's
// spectral::cached_decompose minus the content cache (the codegen already
// interns one call per distinct argument, so a runtime cache would be
// redundant here).
template <typename V, std::size_t Dim>
decomposition<V, Dim> spectral_decompose(tmech::tensor<V, Dim, 2> const &in) {
  decomposition<V, Dim> out;
  auto decomp = tmech::eigen_decomposition(tmech::sym(in));
  auto const [vals, vecs] = decomp.decompose();

  std::array<std::size_t, Dim> order{};
  for (std::size_t i = 0; i < Dim; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return vals[a] < vals[b]; });

  for (std::size_t i = 0; i < Dim; ++i) {
    out.eigenvalues[i] = vals[order[i]];
    out.eigenvectors[i] = vecs[order[i]];
  }
  return out;
}

} // namespace numsim::codegen::rt

#endif // NUMSIM_CODEGEN_RUNTIME_SPECTRAL_H
