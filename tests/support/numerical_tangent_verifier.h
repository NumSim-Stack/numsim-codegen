#ifndef NUMSIM_CODEGEN_TESTS_NUMERICAL_TANGENT_VERIFIER_H
#define NUMSIM_CODEGEN_TESTS_NUMERICAL_TANGENT_VERIFIER_H

// Verification spine (numsim-codegen#90, item 1): a parameterized finite-
// difference check of an emitted rank-4 consistent tangent dσ/dε against central
// differences of the stress, via tmech's built-in symmetric numerical
// differentiation. ALL tolerances + the FD step are caller-supplied — no baked-
// in literals (per the parameterized-verifier directive). Returns the worst-
// deviating component for actionable diagnostics.
//
// Use the SYMMETRIC variant (`num_diff_sym_central`): the plain
// `num_diff_central` with a default Position fails to compile for tensor→tensor,
// and the minor-symmetry tuples below match the minor-symmetric tangents codegen
// emits via tmech::otimesu/otimesl.

#include <tmech/tmech.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace numsim::codegen::verify {

template <std::size_t Dim>
class NumericalTangentVerifier {
public:
  using T2 = tmech::tensor<double, Dim, 2>;
  using T4 = tmech::tensor<double, Dim, 4>;

  struct Params {
    double abs_tol;  // per-component absolute tolerance
    double rel_tol;  // per-component relative tolerance (OR'd with abs)
    double fd_step;  // central-difference step h
  };

  struct Result {
    double max_abs_dev{0.0};
    double max_rel_dev{0.0};
    std::array<std::size_t, 4> worst_index{{0, 0, 0, 0}};
    bool passed{false};
  };

  explicit NumericalTangentVerifier(Params p) : m_p(p) {}

  // stress_fn: T2 -> T2 (wraps the generated compute, returns stress only).
  // analytic:  the emitted rank-4 tangent evaluated at `point`.
  template <typename StressFn>
  [[nodiscard]] auto verify(StressFn &&stress_fn, T2 const &point,
                            T4 const &analytic) const -> Result {
    using Sym2 = std::tuple<tmech::sequence<1, 2>, tmech::sequence<2, 1>>;
    using Sym4 =
        std::tuple<tmech::sequence<1, 2, 3, 4>, tmech::sequence<2, 1, 3, 4>,
                   tmech::sequence<1, 2, 4, 3>, tmech::sequence<2, 1, 4, 3>>;

    T4 const fd = tmech::num_diff_sym_central<Sym2, Sym4>(
        std::forward<StressFn>(stress_fn), point, m_p.fd_step);

    Result r;
    bool all_ok = true;
    double worst_fail_abs = -1.0; // largest abs_dev among FAILING components
    for (std::size_t i = 0; i < Dim; ++i)
      for (std::size_t j = 0; j < Dim; ++j)
        for (std::size_t k = 0; k < Dim; ++k)
          for (std::size_t l = 0; l < Dim; ++l) {
            double const a = analytic(i, j, k, l);
            double const f = fd(i, j, k, l);
            double const abs_dev = std::abs(a - f);
            double const scale = std::max(std::abs(a), std::abs(f));
            double const rel_dev = scale > 0.0 ? abs_dev / scale : 0.0;
            r.max_abs_dev = std::max(r.max_abs_dev, abs_dev);
            r.max_rel_dev = std::max(r.max_rel_dev, rel_dev);
            // Per-component pass: within abs OR rel tolerance.
            bool const ok = (abs_dev <= m_p.abs_tol || rel_dev <= m_p.rel_tol);
            if (!ok) {
              all_ok = false;
              // worst_index points at the worst FAILING component (the one a
              // caller needs to debug), not the global abs-worst (which may pass
              // on relative tolerance for a large-magnitude entry).
              if (abs_dev > worst_fail_abs) {
                worst_fail_abs = abs_dev;
                r.worst_index = {{i, j, k, l}};
              }
            }
          }
    r.passed = all_ok;
    return r;
  }

private:
  Params m_p;
};

} // namespace numsim::codegen::verify

#endif // NUMSIM_CODEGEN_TESTS_NUMERICAL_TANGENT_VERIFIER_H
