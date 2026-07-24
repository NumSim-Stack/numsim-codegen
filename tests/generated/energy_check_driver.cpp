// #108 e2e driver: compile the ENERGY-DERIVED materials and verify the material
// compiler. Two materials:
//   * SvkFromEnergy (quadratic ψ) — the DERIVED stress ∂ψ/∂E matches the
//     INDEPENDENTLY hand-written closed form (per #108's "self-FD insufficient"
//     caveat); its tangent is constant.
//   * NonlinearFromEnergy (quartic ψ) — a genuinely NON-CONSTANT derived tangent,
//     which exercises the second-differentiation machinery a linear-stress
//     material cannot; checked stress-vs-closed-form AND tangent-vs-FD.
// Both are additionally checked for MAJOR symmetry (C_ijkl = C_klij) — the
// Hessian property any energy-derived tangent ∂²ψ/∂E² must have, and the property
// that most distinguishes a derived tangent from an arbitrary hand-written one.

#include "NlEnergyCheck.h"
#include "SvkEnergyCheck.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;
constexpr double kC = 0.4; // quartic coefficient of NonlinearFromEnergy

// Independent, hand-written St. Venant–Kirchhoff 2nd-PK stress (ψ = ½λ(trE)²+μE:E).
T2 hand_written_S(T2 const &E) {
  auto I = tmech::eye<double, 3, 2>();
  return tmech::eval(kLambda * tmech::trace(E) * I + 2.0 * kMu * E);
}

// Independent, hand-written stress for ψ = ½λ(trE)² + μ(E:E) + c(E:E)²:
//   S = λ tr(E) I + 2μ E + 4c (E:E) E.
T2 hand_written_S_nonlinear(T2 const &E) {
  auto I = tmech::eye<double, 3, 2>();
  const double EE = tmech::dcontract(E, E); // E : E
  return tmech::eval(kLambda * tmech::trace(E) * I + 2.0 * kMu * E +
                     4.0 * kC * EE * E);
}

// True if the rank-4 tangent is MAJOR-symmetric: C_ijkl = C_klij.
bool major_symmetric(T4 const &C, double tol = 1e-12) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          if (std::abs(C(i, j, k, l) - C(k, l, i, j)) > tol) return false;
  return true;
}

// 12 linearly-independent symmetric directions — more than the 6 symmetric DOF,
// so contracting the tangent against all of them pins the whole minor-symmetric
// rank-4.
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

// ── NonlinearFromEnergy: quartic ψ → NON-CONSTANT tangent ────────────────────

// The derived stress of the quartic potential matches its hand-written form.
TEST(MaterialCompilerE2E, NonlinearStressMatchesHandWritten) {
  for (T2 const &E : {T2{0.03,0.01,0.0, 0.01,-0.02,0.005, 0.0,0.005,0.04},
                      T2{0.2,0.1,-0.05, 0.1,0.15,0.08, -0.05,0.08,-0.12}}) {
    T2 S; T4 dS;
    NonlinearFromEnergy_compute(kLambda, kMu, kC, E, S, dS);
    EXPECT_TRUE(tmech::almost_equal(S, hand_written_S_nonlinear(E), 1e-12));
  }
}

// The NON-CONSTANT derived tangent matches FD — the real test of the
// second-differentiation machinery (SVK's constant tangent cannot exercise it,
// since a linear stress is FD-exact regardless).
TEST(MaterialCompilerE2E, NonlinearTangentMatchesFD) {
  for (T2 const &E : {T2{0.05,0.02,0.0, 0.02,-0.03,0.01, 0.0,0.01,0.06},
                      T2{0.25,0.12,-0.06, 0.12,0.18,0.09, -0.06,0.09,-0.14}}) {
    T2 S; T4 dS;
    NonlinearFromEnergy_compute(kLambda, kMu, kC, E, S, dS);
    const double t = 1e-6;
    for (auto const &dE : directions()) {
      T2 sp, sm; T4 scratch;
      NonlinearFromEnergy_compute(kLambda, kMu, kC, T2(tmech::eval(E + t * dE)), sp, scratch);
      NonlinearFromEnergy_compute(kLambda, kMu, kC, T2(tmech::eval(E - t * dE)), sm, scratch);
      auto fd = tmech::eval((sp - sm) / (2.0 * t));
      auto an = tmech::eval(tmech::dcontract(dS, dE));
      EXPECT_TRUE(tmech::almost_equal(an, fd, 1e-6)) << "non-constant tangent vs FD";
    }
  }
}

// Both derived tangents are MAJOR-symmetric (C_ijkl = C_klij) — the Hessian
// property ∂²ψ/∂E² must satisfy. This is the assertion that most justifies
// deriving the tangent from a potential rather than writing it by hand.
TEST(MaterialCompilerE2E, DerivedTangentIsMajorSymmetric) {
  for (T2 const &E : {T2{0.03,0.01,0.0, 0.01,-0.02,0.005, 0.0,0.005,0.04},
                      T2{0.25,0.12,-0.06, 0.12,0.18,0.09, -0.06,0.09,-0.14}}) {
    T2 S; T4 dS;
    SvkFromEnergy_compute(kLambda, kMu, E, S, dS);
    EXPECT_TRUE(major_symmetric(dS)) << "SVK-from-energy tangent not major-symmetric";
    NonlinearFromEnergy_compute(kLambda, kMu, kC, E, S, dS);
    EXPECT_TRUE(major_symmetric(dS)) << "nonlinear-from-energy tangent not major-symmetric";
  }
}
