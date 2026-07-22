// Slice-5 end-to-end driver (#105): compile the generated spectral header and
// assert the generated isotropic value and tangent match the CAS evaluator
// across kinds (log/exp/sqrt), dimensions (2 and 3), and coalescence states.
// The header is emitted by generate_spectral_check at build time.

#include "SpectralCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

TEST(SpectralE2E, GeneratedMatchesEvaluator) {
  // 0 mismatches across every baked case: log/exp/sqrt, dim 2 and 3, distinct /
  // coalesced-pair / b=I. The b=I cases are the payoff — fully coalesced
  // eigenvalues, where the divided-difference guard must give the finite
  // analytic limit rather than NaN.
  EXPECT_EQ(spectral_check::run_spectral_check(), 0);
}

// Independent (evaluator-free) check of the b=I tangent: d log(A)/dA at A=I is
// the minor-symmetric identity, so contracting it with ANY H yields sym(H).
// This catches an emitter+evaluator co-bug that run_spectral_check (which
// compares two evaluations of the same tree) structurally cannot.
TEST(SpectralE2E, IdentityLogTangentIsSymmetrizer) {
  tmech::tensor<double, 3, 2> I;
  I = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  tmech::tensor<double, 3, 2> L;
  tmech::tensor<double, 3, 4> dL;
  spectral_check::hencky_log_3(I, L, dL);

  // A deliberately non-symmetric H.
  tmech::tensor<double, 3, 2> H;
  H = {0.1, 0.3, 0.2, 0.0, 0.4, 0.5, 0.6, 0.1, 0.2};
  auto got = tmech::eval(tmech::dcontract(dL, H));
  auto symH = tmech::eval(0.5 * (H + tmech::trans(H)));
  EXPECT_TRUE(tmech::almost_equal(got, symH, 1e-12));

  // And log(I) = 0.
  tmech::tensor<double, 3, 2> zero; // default-constructed tmech tensor is zero
  EXPECT_TRUE(tmech::almost_equal(L, zero, 1e-14));
}
