// Compile-check + runtime-check driver. Includes the codegen-generated
// header and invokes the emitted function with concrete tmech::tensor and
// tmech::adaptor arguments. Verifies:
//   1. The generated header compiles standalone (no MOOSE / Abaqus / etc.).
//   2. Templated tensor arguments accept both tmech::tensor and
//      tmech::adaptor without copying.
//   3. The math is correct (verified against hand-computed values).

#include "AutocatalyticCheck.h"
#include "CompileCheck.h"
#include "CoupledCheck.h"
#include "HardeningCheck.h"
#include "NewtonCheck.h"
#include "PiecewiseCheck.h"
#include "PiecewiseT2sCheck.h"
#include <cmath>
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

// Phase 3a-2 (issue #75) — the realised Phase 1.3 de-risk spike. The
// NewtonCheck recipe enables in-function local Newton solving, so the
// generated function solves for the converged state variable internally
// and exposes it as `alpha_out` (no R/J outputs). We compile + call it
// and verify convergence to the analytic fixed point.
//
// Linear hardening residual R = (alpha - alpha_old)/dt - K*alpha = 0
//   ⇒ alpha* = alpha_old / (1 - K*dt)
// The residual is linear in alpha, so Newton reaches the root in one step.
TEST(CompileCheckGenerated, NewtonSolveConvergesToAnalyticFixedPoint) {
  double const K = 2.0;
  double const alpha_old = 1.0;
  double const dt = 0.1;            // 1 - K*dt = 0.8 > 0, well-posed
  double sigma_y_out = 0.0;
  double alpha_out = 0.0;

  // Generated signature: (K, alpha_old, dt, sigma_y_out, alpha_out).
  NewtonCheck_compute(K, alpha_old, dt, sigma_y_out, alpha_out);

  double const alpha_star = alpha_old / (1.0 - K * dt);  // = 1.25
  EXPECT_NEAR(alpha_out, alpha_star, 1e-10);
  EXPECT_NEAR(alpha_out, 1.25, 1e-10);

  // The downstream output sees the CONVERGED state, not the initial guess.
  EXPECT_NEAR(sigma_y_out, K * alpha_star, 1e-10);  // = 2.5
  EXPECT_NEAR(sigma_y_out, 2.5, 1e-10);
}

TEST(CompileCheckGenerated, NewtonSolveHandlesZeroOldState) {
  // alpha_old = 0 ⇒ residual root is alpha* = 0; the initial guess IS the
  // root, so the convergence check breaks on iteration 0.
  double const K = 5.0;
  double const dt = 0.01;
  double sigma_y_out = 1.0, alpha_out = 1.0;
  NewtonCheck_compute(K, /*alpha_old=*/0.0, dt, sigma_y_out, alpha_out);
  EXPECT_NEAR(alpha_out, 0.0, 1e-12);
  EXPECT_NEAR(sigma_y_out, 0.0, 1e-12);
}

// Phase 3a-2: NONLINEAR Newton convergence. The autocatalytic Kamal rate
// dα/dt = (K1 + K2·α)·(1−α)^1.5 makes the per-step residual nonlinear, so
// the generated Newton loop must genuinely ITERATE (the linear NewtonCheck
// converges in one step and would not catch a broken loop). We step
// backward-Euler through a cure history and verify, at every converged
// point, that the residual is ~0 — re-derived here independently of the
// generated code.
namespace {
double autocat_residual(double c, double c_old, double dt, double K1,
                        double K2) {
  return (c - c_old) / dt - (K1 + K2 * c) * std::pow(1.0 - c, 1.5);
}
} // namespace

TEST(CompileCheckGenerated, AutocatalyticNewtonConvergesOverCureHistory) {
  double const K1 = 0.2, K2 = 2.0, dt = 0.05;
  double c_old = 0.0;
  double prev = 0.0;
  for (int step = 0; step < 40; ++step) {
    double c_out = 0.0;
    AutocatalyticCheck_compute(K1, K2, c_old, dt, c_out);

    // The generated Newton solve drove the residual to ~0.
    EXPECT_LT(std::abs(autocat_residual(c_out, c_old, dt, K1, K2)), 1e-9)
        << "step " << step << " residual not converged (c=" << c_out << ")";
    // Physically admissible + monotonic cure.
    EXPECT_GE(c_out, prev) << "cure must not decrease at step " << step;
    EXPECT_GE(c_out, 0.0);
    EXPECT_LE(c_out, 1.0);

    prev = c_out;
    c_old = c_out;
  }
  // After 40 steps of dt=0.05 (t=2.0) the autocatalytic cure is well
  // advanced but not yet complete.
  EXPECT_GT(c_old, 0.7) << "cure should be well advanced by t=2.0";
  EXPECT_LT(c_old, 1.0);
}

// Phase 3b-2b (issue #35), PR #83 review F3: NUMERICAL lock for the coupled
// vector-Newton dense (Eigen) solve. Two mutually-referencing equations
// a' = K1·b, b' = K2·a with DISTINCT coefficients (asymmetric, so a transposed
// Jacobian would give a different — wrong — answer). Backward-Euler is linear,
// so the converged (a,b) solve
//     [1, −dt·K1; −dt·K2] [a; b] = [a_old; b_old]
// whose closed form we compare against. This is the compile-AND-RUN check the
// structural string tests cannot provide.
TEST(CompileCheckGenerated, CoupledNewtonMatchesAnalyticLinearSolve) {
  double const K1 = 1.0, K2 = 2.0, dt = 0.1;
  double const a_old = 1.0, b_old = 0.3;
  double a_out = 0.0, b_out = 0.0;
  CoupledCheck_compute(K1, K2, a_old, b_old, dt, a_out, b_out);

  double const det = 1.0 - dt * dt * K1 * K2;
  double const a_ref = (a_old + dt * K1 * b_old) / det;
  double const b_ref = (b_old + dt * K2 * a_old) / det;
  EXPECT_NEAR(a_out, a_ref, 1e-9) << "a_out=" << a_out << " ref=" << a_ref;
  EXPECT_NEAR(b_out, b_ref, 1e-9) << "b_out=" << b_out << " ref=" << b_ref;

  // The coupled backward-Euler residuals are driven to ~0 at the solution.
  EXPECT_LT(std::abs((a_out - a_old) / dt - K1 * b_out), 1e-9);
  EXPECT_LT(std::abs((b_out - b_old) / dt - K2 * a_out), 1e-9);

  // Asymmetry sanity: with these inputs a transposed Jacobian (swapping the
  // off-diagonals) would NOT satisfy the residuals — a_ref != b_ref so the
  // answer is genuinely orientation-sensitive.
  EXPECT_GT(std::abs(a_ref - b_ref), 0.1);
}

// CAS f3e799e catch-up (#275/#285): the `tensor_if_then_else` node — a SCALAR
// condition selecting between two TENSOR branches — emits as a materialized
// ternary `(cond != 0.0 ? tmech::tensor<…>(then) : tmech::tensor<…>(else))`.
// The move_to_virtual refactor turned codegen's tensor/t2s visitor overrides
// into pure-virtuals; this compile-AND-RUN check is the lock that the emitted
// construct is a valid tmech expression AND that both branches are selected
// correctly — neither of which a string test can prove. PiecewiseCheck emits
// `sigma = (x != 0 ? 2*eps : -eps)`.
TEST(CompileCheckGenerated, PiecewiseTensorSelectsBranchAndCompilesVsTmech) {
  tmech::tensor<double, 3, 2> eps;  // zero-initialised
  eps(0, 0) = 1.0;
  eps(1, 1) = 2.0;
  eps(0, 1) = eps(1, 0) = 0.5;
  tmech::tensor<double, 3, 2> sigma;

  // x != 0 ⇒ then branch (2*eps)
  PiecewiseCheck_compute(1.0, eps, sigma);
  EXPECT_NEAR(sigma(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(sigma(1, 1), 4.0, 1e-12);
  EXPECT_NEAR(sigma(0, 1), 1.0, 1e-12);

  // x == 0 ⇒ else branch (-eps)
  PiecewiseCheck_compute(0.0, eps, sigma);
  EXPECT_NEAR(sigma(0, 0), -1.0, 1e-12);
  EXPECT_NEAR(sigma(1, 1), -2.0, 1e-12);
  EXPECT_NEAR(sigma(0, 1), -0.5, 1e-12);
}

// Review #87 t2s coverage: the SIBLING node tensor_to_scalar_if_then_else —
// cond/then/else ALL t2s, with the condition itself in the t2s domain. Emits
// `sigma = (trace(eps) != 0 ? trace(eps) : norm(eps)) * eps`. This is the
// end-to-end lock the original PR #87 lacked for the t2s override (a t2s
// output is inexpressible, so it's lifted into a tensor via with_tensor_mul).
TEST(CompileCheckGenerated, PiecewiseT2sSelectsBranchAndCompilesVsTmech) {
  // trace ≠ 0 ⇒ then branch: pick = trace(eps) = 3, sigma = 3*eps.
  {
    tmech::tensor<double, 3, 2> eps;  // zero-initialised
    eps(0, 0) = 1.0;
    eps(1, 1) = 2.0;                  // trace = 3
    tmech::tensor<double, 3, 2> sigma;
    PiecewiseT2sCheck_compute(eps, sigma);
    EXPECT_NEAR(sigma(0, 0), 3.0 * 1.0, 1e-12);
    EXPECT_NEAR(sigma(1, 1), 3.0 * 2.0, 1e-12);
  }
  // traceless ⇒ else branch: pick = norm(eps) = sqrt(0.5), sigma = norm*eps.
  {
    tmech::tensor<double, 3, 2> eps;  // zero-initialised, trace = 0
    eps(0, 1) = eps(1, 0) = 0.5;
    tmech::tensor<double, 3, 2> sigma;
    PiecewiseT2sCheck_compute(eps, sigma);
    double const norm = std::sqrt(0.5 * 0.5 + 0.5 * 0.5);  // = sqrt(0.5)
    EXPECT_NEAR(sigma(0, 1), norm * 0.5, 1e-12);
    EXPECT_NEAR(sigma(1, 0), norm * 0.5, 1e-12);
    EXPECT_NEAR(sigma(0, 0), 0.0, 1e-12);
  }
}
