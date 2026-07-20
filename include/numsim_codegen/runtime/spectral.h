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
#include <cstddef>

namespace numsim::codegen::rt {

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
