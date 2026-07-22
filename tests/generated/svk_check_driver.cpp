// e2e driver: compile the generated St. Venant–Kirchhoff material and verify its
// 2nd-PK stress S and consistent tangent dS/dE. The material is emitted by
// generate_svk_check at build time. This is the (S, Green–Lagrange) conjugate
// pair — the tangent MOOSE's total-Lagrangian kernels consume — obtained by
// making E the differentiation input (see stvenant_kirchhoff_recipe.h).

#include "SvkCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;

// 12 linearly-independent symmetric directions (> 6 symmetric DOF ⇒ contracting
// against all of them pins the whole minor-symmetric rank-4 tangent).
std::array<T2, 12> directions() {
  std::array<T2, 12> d;
  double b[6][9] = {{1,0,0, 0,0,0, 0,0,0}, {0,0,0, 0,1,0, 0,0,0},
                    {0,0,0, 0,0,0, 0,0,1}, {0,1,0, 1,0,0, 0,0,0},
                    {0,0,1, 0,0,0, 1,0,0}, {0,0,0, 0,0,1, 0,1,0}};
  double m[6][9] = {{1,0.3,0.2, 0.3,1,0.1, 0.2,0.1,1}, {2,-0.5,0, -0.5,1,0.4, 0,0.4,-1},
                    {0.1,0.7,-0.2, 0.7,0.3,0.5, -0.2,0.5,0.9}, {-1,0.2,0.6, 0.2,2,-0.3, 0.6,-0.3,0.4},
                    {0.5,-0.8,0.1, -0.8,-0.5,0.9, 0.1,0.9,1.2}, {1.5,0.4,-0.6, 0.4,-1,0.2, -0.6,0.2,0.7}};
  int k = 0;
  for (; k < 6; ++k) d[k] = T2{b[k][0],b[k][1],b[k][2],b[k][3],b[k][4],b[k][5],b[k][6],b[k][7],b[k][8]};
  for (int j = 0; j < 6; ++j, ++k) d[k] = T2{m[j][0],m[j][1],m[j][2],m[j][3],m[j][4],m[j][5],m[j][6],m[j][7],m[j][8]};
  return d;
}

// dS/dE : δE vs central FD of S along δE, for many symmetric directions.
void expect_tangent_matches_fd(T2 const &E) {
  T2 S; T4 dS;
  StVenantKirchhoff_compute(kLambda, kMu, E, S, dS);
  const double t = 1e-6;
  for (auto const &dE : directions()) {
    T2 sp, sm; T4 scratch;
    StVenantKirchhoff_compute(kLambda, kMu, T2(tmech::eval(E + t * dE)), sp, scratch);
    StVenantKirchhoff_compute(kLambda, kMu, T2(tmech::eval(E - t * dE)), sm, scratch);
    auto fd = tmech::eval((sp - sm) / (2.0 * t));
    auto an = tmech::eval(tmech::dcontract(dS, dE));
    EXPECT_TRUE(tmech::almost_equal(an, fd, 1e-6)) << "direction mismatch";
  }
}

} // namespace

// The 2nd-PK stress is the exact closed form S = λ tr(E) I + 2μ E.
TEST(StVenantKirchhoffE2E, StressMatchesClosedForm) {
  T2 E{0.03, 0.01, 0.0, 0.01, -0.02, 0.005, 0.0, 0.005, 0.04};
  T2 S; T4 dS;
  StVenantKirchhoff_compute(kLambda, kMu, E, S, dS);
  auto ref = tmech::eval(kLambda * tmech::trace(E) * tmech::eye<double, 3, 2>() + 2.0 * kMu * E);
  EXPECT_TRUE(tmech::almost_equal(S, ref, 1e-12));
}

// The consistent tangent dS/dE matches FD in every symmetric direction, across
// several strain states (small, moderate, compressive).
TEST(StVenantKirchhoffE2E, TangentMatchesFD) {
  expect_tangent_matches_fd(T2{0.03, 0.01, 0.0, 0.01, -0.02, 0.005, 0.0, 0.005, 0.04});
  expect_tangent_matches_fd(T2{0.2, 0.1, -0.05, 0.1, 0.15, 0.08, -0.05, 0.08, -0.12});
  expect_tangent_matches_fd(T2{-0.1, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -0.1});
}

// dS/dE is constant (linear stress) — equals λ (I⊗I) + 2μ I⁴ˢ independent of E.
TEST(StVenantKirchhoffE2E, TangentIsConstantAndAnalytic) {
  auto I = tmech::eye<double, 3, 2>();
  auto ref = tmech::eval(
      kLambda * tmech::otimes(I, I) +
      kMu * (tmech::otimesu(I, I) + tmech::otimesl(I, I)));
  for (T2 const &E : {T2{0.05,0,0, 0,0,0, 0,0,0}, T2{0.3,0.2,0.1, 0.2,-0.1,0.05, 0.1,0.05,0.2}}) {
    T2 S; T4 dS;
    StVenantKirchhoff_compute(kLambda, kMu, E, S, dS);
    EXPECT_TRUE(tmech::almost_equal(dS, ref, 1e-12)) << "tangent not the analytic constant";
  }
}
