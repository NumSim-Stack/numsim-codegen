// CalculiX end-to-end gate driver.
//
// Two layers of verification, no external dependencies:
//   Recipe: FD-verify the emitted consistent tangent through the StandaloneCxx
//     `LinearElastic_compute`, and anchor the stress to the closed-form
//     isotropic law.
//   ABI: call the emitted external `NCG_UMAT` exactly as CalculiX would (via its
//     external/dlopen ABI) for a single 3D integration point, and check the
//     returned stre(6)/stiff(21) against an INDEPENDENT isotropic oracle —
//     proving the Voigt boundary and the column-major stiff packing before `ccx`
//     is ever built. A packing-order negative control confirms the oracle
//     discriminates; the real dlopen/@-name path is exercised end-to-end by
//     examples/calculix/.

#include "LinearElastic.h" // StandaloneCxx: LinearElastic_compute(...)

#include "numerical_tangent_verifier.h"

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

// ── The emitted external-behaviour ABI (LinearElastic_ext.cpp) — the exact
// `calculixptr` signature ccx dlsym's and calls. ─────────────────────────────
extern "C" void NCG_UMAT(
    char const *amat, int const *iel, int const *iint, int const *NPROPS,
    double const *MPROPS, double const *STRAN1, double const *STRAN0,
    double const *beta, double const *F0, double const *voj, double const *F1,
    double const *vj, int const *ithermal, double const *TEMP1,
    double const *DTIME, double const *time, double const *ttime,
    int const *icmd, int const *ielas, int const *mi, int const *NSTATV,
    double const *STATEV0, double *STATEV1, double *STRESS, double *DDSDDE,
    int const *iorien, double const *pgauss, double const *orab, double *PNEWDT,
    int const *ipkon, int size);

namespace {

constexpr double kLambda = 1.3;
constexpr double kMu = 0.7;

using T2 = tmech::tensor<double, 3, 2>;
using T4 = tmech::tensor<double, 3, 4>;

// Build the full symmetric strain tensor from a CalculiX emec(6) laid out in
// abq_std order {11,22,33,12,13,23} (tensorial shear — no ×0.5).
auto strain_tensor_from_emec(std::array<double, 6> const &e) -> T2 {
  T2 eps;
  eps(0, 0) = e[0];
  eps(1, 1) = e[1];
  eps(2, 2) = e[2];
  eps(0, 1) = eps(1, 0) = e[3];
  eps(0, 2) = eps(2, 0) = e[4];
  eps(1, 2) = eps(2, 1) = e[5];
  return eps;
}

// Closed-form isotropic stress σ = λ·tr(ε)·I + 2μ·ε.
auto isotropic_stress(T2 const &eps, double lambda, double mu) -> T2 {
  double const tr = eps(0, 0) + eps(1, 1) + eps(2, 2);
  T2 I = tmech::eye<double, 3, 2>();
  return T2(lambda * tr * I + 2.0 * mu * eps);
}
auto isotropic_stress(T2 const &eps) -> T2 {
  return isotropic_stress(eps, kLambda, kMu);
}

// Independent closed-form isotropic tangent packed into CalculiX's stiff(21):
// symmetric 6×6 (engineering) in column-major upper-triangular order,
// k = i + j*(j+1)/2 for 0-based i<=j. Built WITHOUT touching the emitted code.
auto expected_stiff_column_major() -> std::array<double, 21> {
  // Engineering 6×6 isotropic D matrix (abq_std order {11,22,33,12,13,23}).
  double const a = kLambda + 2.0 * kMu; // diagonal normal
  double const b = kLambda;             // off-diagonal normal
  double D[6][6] = {{a, b, b, 0, 0, 0}, {b, a, b, 0, 0, 0},
                    {b, b, a, 0, 0, 0}, {0, 0, 0, kMu, 0, 0},
                    {0, 0, 0, 0, kMu, 0}, {0, 0, 0, 0, 0, kMu}};
  std::array<double, 21> s{};
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i <= j; ++i) s[static_cast<std::size_t>(i + j * (j + 1) / 2)] = D[i][j];
  return s;
}

// Call the emitted external behaviour NCG_UMAT for one integration point,
// exactly as ccx's dlopen path does. Native STANDARD interface: STRAN1=emec
// tensorial, MPROPS=constants, DDSDDE=stiff(21).
void call_ext(std::array<double, 6> const &emec_in, std::array<double, 6> &stre,
              std::array<double, 21> &stiff, double lambda = kLambda,
              double mu = kMu, int icmd_val = 1) {
  std::array<double, 2> mprops{lambda, mu};
  std::array<double, 6> stran1 = emec_in, stran0{}, beta{};
  int iel = 1, iint = 1, nprops = -102, icmd = icmd_val, ielas = 0, mi = 1,
      nstatv = 0, iorien = 0, ipkon = 0, ithermal = 0;
  double f0[9] = {}, voj = 1.0, f1[9] = {}, vj = 1.0, temp1 = 0.0, dtime = 1.0,
         time_[2] = {}, ttime = 0.0, statev0 = 0.0, statev1 = 0.0, pgauss[3] = {},
         orab[7] = {}, pnewdt = 1.0;
  char amat[81] = "@LINEARELASTIC_NCG_UMAT";
  stre.fill(0.0);
  stiff.fill(0.0);
  NCG_UMAT(amat, &iel, &iint, &nprops, mprops.data(), stran1.data(),
           stran0.data(), beta.data(), f0, &voj, f1, &vj, &ithermal, &temp1,
           &dtime, time_, &ttime, &icmd, &ielas, &mi, &nstatv, &statev0,
           &statev1, stre.data(), stiff.data(), &iorien, pgauss, orab, &pnewdt,
           &ipkon, 80);
}

// A spread of strain states: uniaxial, pure shear, and a general symmetric one.
auto sample_strains() -> std::vector<std::array<double, 6>> {
  return {
      {{0.01, 0.0, 0.0, 0.0, 0.0, 0.0}},         // uniaxial ε11
      {{0.0, 0.0, 0.0, 0.02, 0.0, 0.0}},         // pure shear ε12
      {{0.005, -0.003, 0.002, 0.004, -0.001, 0.006}}, // general
  };
}

} // namespace

// ── Phase 0: recipe correctness through the standalone _compute ──────────────

TEST(CalculiXGate, StandaloneStressMatchesIsotropicLaw) {
  for (auto const &e : sample_strains()) {
    T2 const eps = strain_tensor_from_emec(e);
    T2 sigma;
    T4 tangent;
    LinearElastic_compute(kLambda, kMu, eps, sigma, tangent);
    T2 const expected = isotropic_stress(eps);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        EXPECT_NEAR(sigma(i, j), expected(i, j), 1e-12)
            << "stress mismatch at (" << i << "," << j << ")";
  }
}

TEST(CalculiXGate, ConsistentTangentMatchesFiniteDifference) {
  auto stress_only = [&](auto const &e) -> T2 {
    T2 s;
    T4 t;
    LinearElastic_compute(kLambda, kMu, e, s, t);
    return s;
  };
  numsim::codegen::verify::NumericalTangentVerifier<3> const verifier(
      {.abs_tol = 1e-7, .rel_tol = 1e-6, .fd_step = 1e-6});
  for (auto const &e : sample_strains()) {
    T2 const eps = strain_tensor_from_emec(e);
    T2 s;
    T4 emitted_tangent;
    LinearElastic_compute(kLambda, kMu, eps, s, emitted_tangent);
    auto const r = verifier.verify(stress_only, eps, emitted_tangent);
    EXPECT_TRUE(r.passed) << "tangent FD mismatch: max_abs=" << r.max_abs_dev
                          << " max_rel=" << r.max_rel_dev;
  }
}

// ── External NCG_UMAT ABI boundary + stiff packing ───────────────────────────

TEST(CalculiXGate, ExternalUmatStressMatchesIsotropicOracle) {
  for (auto const &e : sample_strains()) {
    std::array<double, 6> stre{};
    std::array<double, 21> stiff{};
    call_ext(e, stre, stiff);
    T2 const sigma = isotropic_stress(strain_tensor_from_emec(e));
    // stre is abq_std order {11,22,33,12,13,23}.
    EXPECT_NEAR(stre[0], sigma(0, 0), 1e-12);
    EXPECT_NEAR(stre[1], sigma(1, 1), 1e-12);
    EXPECT_NEAR(stre[2], sigma(2, 2), 1e-12);
    EXPECT_NEAR(stre[3], sigma(0, 1), 1e-12);
    EXPECT_NEAR(stre[4], sigma(0, 2), 1e-12);
    EXPECT_NEAR(stre[5], sigma(1, 2), 1e-12);
  }
}

TEST(CalculiXGate, ExternalUmatStiffMatchesColumnMajorPackedOracle) {
  auto const expected = expected_stiff_column_major();
  // The tangent is strain-independent for linear elasticity; check on a
  // general state so every column participates.
  std::array<double, 6> stre{};
  std::array<double, 21> stiff{};
  call_ext(sample_strains().back(), stre, stiff);
  for (std::size_t k = 0; k < 21; ++k)
    EXPECT_NEAR(stiff[k], expected[k], 1e-12) << "stiff mismatch at index " << k;
}

// Negative control: the packing ORDER is observable. Row-major-upper packing of
// the same symmetric D differs from column-major-upper (e.g. index 2 holds
// D(0,2)=λ vs D(1,1)=λ+2μ). The emitted stiff must match column-major, NOT
// row-major — so the oracle genuinely pins the order, not just the value set.
TEST(CalculiXGate, StiffPackingIsColumnMajorNotRowMajor) {
  double const a = kLambda + 2.0 * kMu, b = kLambda;
  double D[6][6] = {{a, b, b, 0, 0, 0}, {b, a, b, 0, 0, 0},
                    {b, b, a, 0, 0, 0}, {0, 0, 0, kMu, 0, 0},
                    {0, 0, 0, 0, kMu, 0}, {0, 0, 0, 0, 0, kMu}};
  std::array<double, 21> row_major{};
  std::size_t k = 0;
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) row_major[k++] = D[i][j];

  std::array<double, 6> stre{};
  std::array<double, 21> stiff{};
  call_ext(sample_strains().back(), stre, stiff);

  // stiff equals the column-major oracle but NOT the row-major one.
  auto const col_major = expected_stiff_column_major();
  bool differs_from_row_major = false;
  for (std::size_t idx = 0; idx < 21; ++idx) {
    EXPECT_NEAR(stiff[idx], col_major[idx], 1e-12);
    if (std::abs(stiff[idx] - row_major[idx]) > 1e-9) differs_from_row_major = true;
  }
  EXPECT_TRUE(differs_from_row_major)
      << "column-major and row-major packings are indistinguishable here — "
         "the negative control has no discriminating power";
}

// Regression (review CRITICAL): the external plugin must read MPROPS on EVERY
// call. The earlier thread_local material cached the first call's constants, so
// a second call on the same thread with different λ/μ silently returned the
// first answer. Two back-to-back calls with different constants must both be
// correct. (CalculiX legitimately varies the constants by temperature.)
TEST(CalculiXGate, ExternalReadsConstantsEveryCall) {
  std::array<double, 6> const e{{0.01, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> s1{}, s2{};
  std::array<double, 21> k1{}, k2{};
  call_ext(e, s1, k1, /*lambda=*/1.3, /*mu=*/0.7);
  call_ext(e, s2, k2, /*lambda=*/0.5, /*mu=*/0.2); // same thread, new constants
  // S11 = (λ+2μ)·0.01: 0.027 then 0.009 — must differ and both be correct.
  EXPECT_NEAR(s1[0], (1.3 + 2 * 0.7) * 0.01, 1e-12);
  EXPECT_NEAR(s2[0], (0.5 + 2 * 0.2) * 0.01, 1e-12);
  EXPECT_GT(std::abs(s1[0] - s2[0]), 1e-6) << "constants not re-read per call";
}

// icmd==3 requests stress only: stress must still be correct, and stiff must be
// left untouched. The call helpers pre-fill stiff with 0.0, so an untouched
// stiff stays all-zero — whereas the icmd=1 path writes nonzero packed entries
// (e.g. λ+2μ). Assert both to make "untouched" meaningful.
TEST(CalculiXGate, StressOnlyIcmd3LeavesStiffUntouched) {
  auto const &e = sample_strains().back();
  T2 const sigma = isotropic_stress(strain_tensor_from_emec(e));
  std::array<double, 6> stre{};
  std::array<double, 21> stiff3{}, stiff1{};
  call_ext(e, stre, stiff3, kLambda, kMu, /*icmd=*/3);
  call_ext(e, stre, stiff1, kLambda, kMu, /*icmd=*/1);
  EXPECT_NEAR(stre[0], sigma(0, 0), 1e-12); // stress still correct
  for (std::size_t k = 0; k < 21; ++k)
    EXPECT_DOUBLE_EQ(stiff3[k], 0.0) << "stiff written on icmd==3 at " << k;
  EXPECT_GT(stiff1[0], 0.0) << "icmd==1 must fill stiff (control)";
}
