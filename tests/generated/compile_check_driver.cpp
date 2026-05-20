// Compile-check + runtime-check driver. Includes the codegen-generated
// header and invokes the emitted function with concrete tmech::tensor and
// tmech::adaptor arguments. Verifies:
//   1. The generated header compiles standalone (no MOOSE / Abaqus / etc.).
//   2. Templated tensor arguments accept both tmech::tensor and
//      tmech::adaptor without copying.
//   3. The math is correct (verified against hand-computed values).

#include "CompileCheck.h"
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
