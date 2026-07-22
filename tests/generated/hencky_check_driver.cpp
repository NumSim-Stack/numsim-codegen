// Slice-3 driver: compile the generated Hencky material and verify its stress
// and consistent tangent numerically. dstress/dC : δC is checked against
// central finite differences of stress(C) where eigenvalues are DISTINCT (FD is
// unreliable at exact coalescence — log(C±tδC) carries an ill-conditioned
// E_i−E_j term, the cas #329 lesson), and finiteness is asserted at exact
// degeneracy (where the divided-difference guard must take the analytic-limit
// branch; correctness there is validated against the evaluator in the spectral
// e2e). The material is emitted by generate_hencky_check at build time.

#include "HenckyCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;

// dstress/dC : δC vs central FD of stress along a symmetric δC.
void expect_tangent_matches_fd(T2 const &C) {
  T2 sig;
  T4 dsig;
  HenckyHyperelastic_compute(kLambda, kMu, C, sig, dsig);

  T2 dC;
  dC = {0.1, 0.2, 0.05, 0.2, 0.3, 0.1, 0.05, 0.1, 0.15}; // symmetric
  const double t = 1e-6;
  T2 sp, sm;
  T4 scratch;
  HenckyHyperelastic_compute(kLambda, kMu, T2(tmech::eval(C + t * dC)), sp,
                             scratch);
  HenckyHyperelastic_compute(kLambda, kMu, T2(tmech::eval(C - t * dC)), sm,
                             scratch);
  auto fd = tmech::eval((sp - sm) / (2.0 * t));
  auto analytic = tmech::eval(tmech::dcontract(dsig, dC));
  EXPECT_TRUE(tmech::almost_equal(analytic, fd, 1e-6));
}

void expect_finite(T2 const &C) {
  T2 sig;
  T4 dsig;
  HenckyHyperelastic_compute(kLambda, kMu, C, sig, dsig);
  for (int i = 0; i < 9; ++i)
    ASSERT_TRUE(std::isfinite(sig.raw_data()[i])) << "stress[" << i << "]";
  for (int i = 0; i < 81; ++i)
    ASSERT_TRUE(std::isfinite(dsig.raw_data()[i])) << "tangent[" << i << "]";
}

// Distinct eigenvalues — FD is well-conditioned here.
TEST(HenckyE2E, TangentMatchesFDDistinct) {
  expect_tangent_matches_fd(T2{4, 1, 0, 1, 3, 0, 0, 0, 2});
}

// Near-undeformed: eigenvalues close but distinct (~{1.005, 1.01, 1.02}) — the
// realistic FE state near C=I, still FD-verifiable.
TEST(HenckyE2E, TangentMatchesFDNearIdentity) {
  expect_tangent_matches_fd(
      T2{1.02, 0.01, 0.0, 0.01, 1.01, 0.005, 0.0, 0.005, 1.005});
}

// Exact degeneracy: {2,2,5} and {1,1,1}. FD breaks, so assert finiteness — the
// tangent must not blow up.
TEST(HenckyE2E, FiniteAtCoalescedEigenvalues) {
  expect_finite(T2{3, 1, 1, 1, 3, 1, 1, 1, 3}); // {2,2,5}
  expect_finite(T2{1, 0, 0, 0, 1, 0, 0, 0, 1}); // {1,1,1} = C=I
}

} // namespace
