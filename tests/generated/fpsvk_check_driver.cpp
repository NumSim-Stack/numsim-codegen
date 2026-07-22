// e2e driver: compile the generated first-PK SVK material and FD-verify its
// consistent tangent dP/dF. P and F are both NON-symmetric (P is a two-point
// tensor, F the deformation gradient), so the tangent is checked against central
// finite differences of P(F) along GENERAL (non-symmetric) directions — 12 of
// them (> 9 F-DOF), which pins the whole F-index space of the rank-4. This is
// also the end-to-end regression for the tmech-v1.1.1 lazy-operand fix: the
// strain enters through C = FᵀF, the aliased-leaf case that was half-correct
// before the fix.

#include "FpSvkCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;

// 12 general (non-symmetric) directions.
std::array<T2, 12> directions() {
  std::array<T2, 12> d;
  double v[12][9] = {
      {1,0,0,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0,0}, {0,0,1,0,0,0,0,0,0},
      {0,0,0,1,0,0,0,0,0}, {0,0,0,0,1,0,0,0,0}, {0,0,0,0,0,1,0,0,0},
      {0,0,0,0,0,0,1,0,0}, {0,0,0,0,0,0,0,1,0}, {0,0,0,0,0,0,0,0,1},
      {1,0.4,-0.2,0.3,1,0.1,-0.2,0.1,1}, {0.5,-0.7,0.2,0.9,0.3,-0.4,0.1,0.6,0.8},
      {-0.3,0.8,0.5,-0.6,0.2,0.7,0.4,-0.9,0.1}};
  for (int k = 0; k < 12; ++k)
    d[k] = T2{v[k][0],v[k][1],v[k][2],v[k][3],v[k][4],v[k][5],v[k][6],v[k][7],v[k][8]};
  return d;
}

void expect_tangent_matches_fd(T2 const &F) {
  T2 P; T4 dP;
  FirstPiolaSvk_compute(kLambda, kMu, F, P, dP);
  const double t = 1e-6;
  for (auto const &dF : directions()) {
    T2 pp, pm; T4 scratch;
    FirstPiolaSvk_compute(kLambda, kMu, T2(tmech::eval(F + t * dF)), pp, scratch);
    FirstPiolaSvk_compute(kLambda, kMu, T2(tmech::eval(F - t * dF)), pm, scratch);
    auto fd = tmech::eval((pp - pm) / (2.0 * t));
    auto an = tmech::eval(tmech::dcontract(dP, dF));
    EXPECT_TRUE(tmech::almost_equal(an, fd, 1e-6)) << "direction mismatch";
  }
}

} // namespace

// P = F·S with S = λ tr(E) I + 2μ E, E = ½(FᵀF − I) — closed-form spot check.
TEST(FirstPiolaSvkE2E, StressMatchesClosedForm) {
  T2 F{1.1, 0.2, 0.0, 0.05, 0.95, 0.1, 0.0, 0.08, 1.04};
  T2 P; T4 dP;
  FirstPiolaSvk_compute(kLambda, kMu, F, P, dP);
  auto I = tmech::eye<double, 3, 2>();
  auto C = tmech::eval(tmech::trans(F) * F);
  auto E = tmech::eval(0.5 * (C - I));
  auto S = tmech::eval(kLambda * tmech::trace(E) * I + 2.0 * kMu * E);
  auto ref = tmech::eval(F * S);
  EXPECT_TRUE(tmech::almost_equal(P, ref, 1e-12));
}

// dP/dF matches FD in every general direction, across several deformation states
// — including large stretch + shear where the FᵀF coupling is strongest.
TEST(FirstPiolaSvkE2E, TangentMatchesFD) {
  expect_tangent_matches_fd(T2{1.1, 0.2, 0.0, 0.05, 0.95, 0.1, 0.0, 0.08, 1.04});
  expect_tangent_matches_fd(T2{1.3, 0.2, 0.0, 0.1, 0.9, 0.05, 0.0, 0.1, 1.1});
  expect_tangent_matches_fd(T2{1.5, 0.4, 0.1, -0.2, 1.2, 0.3, 0.05, -0.1, 0.8});
}
