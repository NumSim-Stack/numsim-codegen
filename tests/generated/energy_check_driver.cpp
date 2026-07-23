// #108 e2e driver: compile the ENERGY-DERIVED SVK material and verify it two
// ways — (1) its stress S = ∂ψ/∂E matches the INDEPENDENTLY hand-written closed
// form λ tr(E) I + 2μ E (per #108's "self-FD is insufficient" caveat), and
// (2) its derived tangent dS/dE = ∂²ψ/∂E² matches central finite differences of
// that stress. This proves the material compiler produced the same material a
// human would have written by hand.

#include "EnergyCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;

// Independent, hand-written St. Venant–Kirchhoff 2nd-PK stress.
T2 hand_written_S(T2 const &E) {
  auto I = tmech::eye<double, 3, 2>();
  return tmech::eval(kLambda * tmech::trace(E) * I + 2.0 * kMu * E);
}

std::array<T2, 12> directions() {
  std::array<T2, 12> d;
  double b[6][9] = {{1,0,0,0,0,0,0,0,0}, {0,0,0,0,1,0,0,0,0}, {0,0,0,0,0,0,0,0,1},
                    {0,1,0,1,0,0,0,0,0}, {0,0,1,0,0,0,1,0,0}, {0,0,0,0,0,1,0,1,0}};
  double m[6][9] = {{1,0.3,0.2,0.3,1,0.1,0.2,0.1,1}, {2,-0.5,0,-0.5,1,0.4,0,0.4,-1},
                    {0.1,0.7,-0.2,0.7,0.3,0.5,-0.2,0.5,0.9}, {-1,0.2,0.6,0.2,2,-0.3,0.6,-0.3,0.4},
                    {0.5,-0.8,0.1,-0.8,-0.5,0.9,0.1,0.9,1.2}, {1.5,0.4,-0.6,0.4,-1,0.2,-0.6,0.2,0.7}};
  int k = 0;
  for (; k < 6; ++k) d[k] = T2{b[k][0],b[k][1],b[k][2],b[k][3],b[k][4],b[k][5],b[k][6],b[k][7],b[k][8]};
  for (int j = 0; j < 6; ++j, ++k) d[k] = T2{m[j][0],m[j][1],m[j][2],m[j][3],m[j][4],m[j][5],m[j][6],m[j][7],m[j][8]};
  return d;
}

} // namespace

// The DERIVED stress equals the independently hand-written one.
TEST(MaterialCompilerE2E, DerivedStressMatchesHandWritten) {
  for (T2 const &E : {T2{0.03,0.01,0.0, 0.01,-0.02,0.005, 0.0,0.005,0.04},
                      T2{0.2,0.1,-0.05, 0.1,0.15,0.08, -0.05,0.08,-0.12}}) {
    T2 S; T4 dS;
    SvkFromEnergy_compute(kLambda, kMu, E, S, dS);
    EXPECT_TRUE(tmech::almost_equal(S, hand_written_S(E), 1e-12))
        << "derived ∂ψ/∂E must equal the hand-written closed form";
  }
}

// The DERIVED tangent dS/dE = ∂²ψ/∂E² matches central FD of the derived stress,
// across strain states and every symmetric direction.
TEST(MaterialCompilerE2E, DerivedTangentMatchesFD) {
  for (T2 const &E : {T2{0.03,0.01,0.0, 0.01,-0.02,0.005, 0.0,0.005,0.04},
                      T2{0.2,0.1,-0.05, 0.1,0.15,0.08, -0.05,0.08,-0.12}}) {
    T2 S; T4 dS;
    SvkFromEnergy_compute(kLambda, kMu, E, S, dS);
    const double t = 1e-6;
    for (auto const &dE : directions()) {
      T2 sp, sm; T4 scratch;
      SvkFromEnergy_compute(kLambda, kMu, T2(tmech::eval(E + t * dE)), sp, scratch);
      SvkFromEnergy_compute(kLambda, kMu, T2(tmech::eval(E - t * dE)), sm, scratch);
      auto fd = tmech::eval((sp - sm) / (2.0 * t));
      auto an = tmech::eval(tmech::dcontract(dS, dE));
      EXPECT_TRUE(tmech::almost_equal(an, fd, 1e-6)) << "derived tangent vs FD";
    }
  }
}
