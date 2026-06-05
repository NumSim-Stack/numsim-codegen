// Compile-check + runtime-check driver. Includes the codegen-generated
// header and invokes the emitted function with concrete tmech::tensor and
// tmech::adaptor arguments. Verifies:
//   1. The generated header compiles standalone (no MOOSE / Abaqus / etc.).
//   2. Templated tensor arguments accept both tmech::tensor and
//      tmech::adaptor without copying.
//   3. The math is correct (verified against hand-computed values).

#include "CompileCheck.h"
#include "HardeningCheck.h"
#include <gtest/gtest.h>
#include <tmech/tmech.h>

TEST(CompileCheckGenerated, ScalarOutputMatchesExpected) {
  double const a = 3.0;
  double const x = 0.5;
  double const mu = 0.7;
  tmech::tensor<double, 3, 2> eps;  // zero-initialised
  // Set a single component so the tensor output is non-trivial.
  eps(1, 1) = 1.0;

  double y_out = 0.0;
  tmech::tensor<double, 3, 2> sigma_out;

  CompileCheck_compute(a, x, mu, eps, y_out, sigma_out);

  // y = a*x + sin(x) = 3*0.5 + sin(0.5)
  EXPECT_NEAR(y_out, a * x + std::sin(x), 1e-12);

  // sigma = 2*mu*eps — only the (1,1) component is non-zero.
  EXPECT_NEAR(sigma_out(1, 1), 2.0 * mu * 1.0, 1e-12);
  EXPECT_NEAR(sigma_out(2, 2), 0.0, 1e-12);
}

TEST(CompileCheckGenerated, AdaptorArgumentsAvoidCopy) {
  // The compute function is templated on tensor types, so passing
  // tmech::adaptor over a raw buffer compiles and works without an
  // intermediate tmech::tensor materialisation. This is the test that
  // guards the MOOSE boundary's zero-copy claim.
  double const a = 1.0;
  double const x = 0.0;
  double const mu = 1.0;

  double eps_buf[9] = {2.0, 0.0, 0.0,
                       0.0, 3.0, 0.0,
                       0.0, 0.0, 5.0};
  double sigma_buf[9] = {0.0};

  tmech::adaptor<double const, 3, 2, tmech::full<3>> eps_ad(eps_buf);
  tmech::adaptor<double, 3, 2, tmech::full<3>> sigma_ad(sigma_buf);

  double y_out = 0.0;
  CompileCheck_compute(a, x, mu, eps_ad, y_out, sigma_ad);

  // sigma = 2*1*eps. The buffer should now hold 2*eps_buf.
  EXPECT_NEAR(sigma_buf[0], 4.0, 1e-12);
  EXPECT_NEAR(sigma_buf[4], 6.0, 1e-12);
  EXPECT_NEAR(sigma_buf[8], 10.0, 1e-12);
}

// End-to-end check that `LocalJacobianPass` emits a numerically-correct
// Jacobian via `cas::diff`. Substring-matching in
// `tests/LocalJacobianTest.cpp` cannot catch a regression that emits a
// plausible-but-wrong expression — this test compiles + calls the
// generated hardening function and verifies both the residual and the
// Jacobian against hand-computed values.
TEST(CompileCheckGenerated, HardeningResidualAndJacobianAreNumericallyCorrect) {
  double const K = 2.0;
  double const alpha = 1.5;
  double const alpha_old = 1.0;
  double const dt = 0.1;
  double residual_out = 0.0;
  double jacobian_out = 0.0;

  // Generated signature (parameters in registration order):
  //   K (param), alpha (state-var current), alpha_old (state-var old),
  //   dt (auto-registered param),
  //   alpha_residual_out, alpha_jacobian_out.
  HardeningCheck_compute(K, alpha, alpha_old, dt, residual_out, jacobian_out);

  // R = (alpha - alpha_old)/dt - K*alpha
  //   = (1.5 - 1.0)/0.1 - 2.0*1.5
  //   = 5.0 - 3.0 = 2.0
  EXPECT_NEAR(residual_out, (alpha - alpha_old) / dt - K * alpha, 1e-12);
  EXPECT_NEAR(residual_out, 2.0, 1e-12);

  // J = ∂R/∂alpha = 1/dt - K
  //   = 1.0/0.1 - 2.0
  //   = 10.0 - 2.0 = 8.0
  EXPECT_NEAR(jacobian_out, 1.0 / dt - K, 1e-12);
  EXPECT_NEAR(jacobian_out, 8.0, 1e-12);
}

TEST(CompileCheckGenerated, HardeningJacobianIndependentOfAlphaOld) {
  // PR #71 round-1 #2 invariant: cas::diff must treat the `alpha_old`
  // leaf as independent of `alpha`. The Jacobian must be unchanged
  // when alpha_old varies (only the residual changes).
  double const K = 3.0;
  double const alpha = 2.0;
  double const dt = 0.5;

  double r1 = 0.0, j1 = 0.0;
  HardeningCheck_compute(K, alpha, /*alpha_old=*/0.0, dt, r1, j1);

  double r2 = 0.0, j2 = 0.0;
  HardeningCheck_compute(K, alpha, /*alpha_old=*/5.0, dt, r2, j2);

  EXPECT_NE(r1, r2) << "residual must depend on alpha_old";
  EXPECT_NEAR(j1, j2, 1e-12)
      << "Jacobian must be independent of alpha_old";
  EXPECT_NEAR(j1, 1.0 / dt - K, 1e-12);
}
