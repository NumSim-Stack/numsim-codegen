// #108 e2e driver: compile the ENERGY-DERIVED materials and verify the material
// compiler. Three materials:
//   * SvkFromEnergy (quadratic ψ) — the DERIVED stress ∂ψ/∂E matches the
//     INDEPENDENTLY hand-written closed form (per #108's "self-FD insufficient"
//     caveat); its tangent is constant.
//   * NonlinearFromEnergy (quartic ψ) — a genuinely NON-CONSTANT derived tangent,
//     which exercises the second-differentiation machinery a linear-stress
//     material cannot; checked stress-vs-closed-form AND tangent-vs-FD.
//   * NonsymmetricFromEnergy (ψ of a non-symmetric leaf F) — exercises the
//     non-symmetric-leaf path; its tangent is major- but NOT minor-symmetric,
//     which also proves minor_symmetric() discriminates.
// Symmetry is checked two ways: MAJOR (C_ijkl=C_klij) is automatic for a Hessian
// so it only guards against emit/diff corruption; MINOR (C_ijkl=C_jikl=C_ijlk) is
// the load-bearing property the symmetric roles::Strain leaf confers.

#include "NlEnergyCheck.h"
#include "NsEnergyCheck.h"
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

// True if the rank-4 tangent is MAJOR-symmetric: C_ijkl = C_klij. The pair-swap
// is a tmech::basis_change with the (ij)↔(kl) permutation <3,4,1,2>; it's an
// involution, so equality with the original is convention-independent.
bool major_symmetric(T4 const &C, double tol = 1e-12) {
  return tmech::almost_equal(
      C, tmech::eval(tmech::basis_change<tmech::sequence<3, 4, 1, 2>>(C)), tol);
}

// True if the tangent is MINOR-symmetric: C_ijkl = C_jikl = C_ijlk — the two
// intra-pair swaps <2,1,3,4> and <1,2,4,3>. This is the property the symmetric
// (roles::Strain) leaf confers; unlike major symmetry (automatic for any Hessian)
// it can genuinely fail if the leaf's symmetry space is dropped, so it's the
// load-bearing symmetry check here.
bool minor_symmetric(T4 const &C, double tol = 1e-12) {
  return tmech::almost_equal(
             C, tmech::eval(tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(C)),
             tol) &&
         tmech::almost_equal(
             C, tmech::eval(tmech::basis_change<tmech::sequence<1, 2, 4, 3>>(C)),
             tol);
}

// 12 symmetric directions (a redundant, over-determined set — the symmetric
// space is only 6-D). Contracting the tangent against all of them exercises its
// action on symmetric strains; note this is blind to any minor-ASYMMETRIC
// component (checked separately by minor_symmetric()).
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

// Both derived tangents are symmetric. Two distinct checks:
//   * MAJOR symmetry (C_ijkl = C_klij) is automatic for any correct Hessian
//     ∂²ψ/∂E² (Clairaut), so this mainly guards against a diff-non-commutation
//     or an emit/CSE corruption of the rank-4 — NOT a value error (FD covers that).
//   * MINOR symmetry (C_ijkl = C_jikl = C_ijlk) is the property the symmetric
//     roles::Strain leaf actually confers; it is NOT automatic and would fail if
//     the leaf's symmetry space were dropped — this is the load-bearing check.
TEST(MaterialCompilerE2E, DerivedTangentIsSymmetric) {
  for (T2 const &E : {T2{0.03,0.01,0.0, 0.01,-0.02,0.005, 0.0,0.005,0.04},
                      T2{0.25,0.12,-0.06, 0.12,0.18,0.09, -0.06,0.09,-0.14}}) {
    T2 S; T4 dS;
    SvkFromEnergy_compute(kLambda, kMu, E, S, dS);
    EXPECT_TRUE(major_symmetric(dS)) << "SVK tangent not major-symmetric";
    EXPECT_TRUE(minor_symmetric(dS)) << "SVK tangent not minor-symmetric";
    NonlinearFromEnergy_compute(kLambda, kMu, kC, E, S, dS);
    EXPECT_TRUE(major_symmetric(dS)) << "nonlinear tangent not major-symmetric";
    EXPECT_TRUE(minor_symmetric(dS)) << "nonlinear tangent not minor-symmetric";
  }
}

// ── NonsymmetricFromEnergy: ψ(F) with a NON-SYMMETRIC leaf ───────────────────

namespace {
// 12 GENERAL (non-symmetric) directions — F is a deformation gradient (9 DOF).
std::array<T2, 12> general_directions() {
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
} // namespace

// The derived P = ∂ψ/∂F for a NON-symmetric leaf matches FD in general (9-DOF)
// directions — proving the non-symmetric-leaf path works, not just the symmetric.
TEST(MaterialCompilerE2E, NonsymmetricTangentMatchesFD) {
  constexpr double mu = 0.5, c = 0.4;
  for (T2 const &F : {T2{1.1,0.2,0.0, 0.05,0.95,0.1, 0.0,0.08,1.04},
                      T2{1.3,0.2,0.1, -0.2,1.1,0.3, 0.05,-0.1,0.9}}) {
    T2 P; T4 dP;
    NonsymmetricFromEnergy_compute(mu, c, F, P, dP);
    const double t = 1e-6;
    for (auto const &dF : general_directions()) {
      T2 pp, pm; T4 scratch;
      NonsymmetricFromEnergy_compute(mu, c, T2(tmech::eval(F + t * dF)), pp, scratch);
      NonsymmetricFromEnergy_compute(mu, c, T2(tmech::eval(F - t * dF)), pm, scratch);
      auto fd = tmech::eval((pp - pm) / (2.0 * t));
      auto an = tmech::eval(tmech::dcontract(dP, dF));
      EXPECT_TRUE(tmech::almost_equal(an, fd, 1e-6)) << "non-symmetric-leaf tangent vs FD";
    }
  }
}

// ∂²ψ/∂F² is MAJOR-symmetric (Hessian) but NOT minor-symmetric (F is not a
// symmetric leaf). This confirms both that the non-symmetric path is correct AND
// that minor_symmetric() genuinely discriminates (does not pass vacuously).
TEST(MaterialCompilerE2E, NonsymmetricTangentIsMajorButNotMinorSymmetric) {
  constexpr double mu = 0.5, c = 0.4;
  T2 F{1.3,0.2,0.1, -0.2,1.1,0.3, 0.05,-0.1,0.9};
  T2 P; T4 dP;
  NonsymmetricFromEnergy_compute(mu, c, F, P, dP);
  EXPECT_TRUE(major_symmetric(dP)) << "Hessian must be major-symmetric";
  EXPECT_FALSE(minor_symmetric(dP)) << "a non-symmetric leaf must NOT give a minor-symmetric tangent";
}
