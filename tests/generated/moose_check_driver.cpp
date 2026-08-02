// Driver for the MOOSE-stub compile gate (#132). The gate itself is the
// BUILD of this target: every MooseMaterialTarget-emitted .C (see
// generate_moose_check.cpp for the recipe list) is compiled as its own TU
// against the minimal MOOSE API stubs in tests/moose_stub/, under the same
// -Wall -Wextra (-Werror in Debug) set as the other generated-code drivers.
//
// On top of the compile gate, the stubs are just functional enough to run one
// quadrature-point evaluation (issue #12 stretch): construct a material from
// its own validParams(), seed its coupled inputs through the stub
// PropertyStore, call the compute hook, and check outputs against closed
// forms.

#include "Material.h" // moose_stub — PropertyStore access

#include "MooseNewtonCheck.h"
#include "MooseShearCheck.h"

#include <gtest/gtest.h>

namespace {

// Emitted MOOSE boilerplate for the shear material evaluates
// stress = 2·mu·eps at the current quadrature point, reading eps and writing
// stress through the RankTwoTensor dataPointer() adaptors. Layout-agnostic
// check: compare raw storage entrywise.
TEST(MooseStubGate, ShearStressMatchesClosedForm) {
  auto params = MooseShearCheck::validParams();
  MooseShearCheck material(params);

  auto &eps =
      moose_stub::PropertyStore::global().get_or_create<RankTwoTensor>("eps");
  Real *eps_data = eps[0].dataPointer();
  // Symmetric strain (roles::Strain marks eps symmetric).
  Real const values[9] = {0.10, 0.02, -0.03, 0.02, -0.20,
                          0.05, -0.03, 0.05, 0.30};
  for (int k = 0; k < 9; ++k) {
    eps_data[k] = values[k];
  }

  material.stubEvaluateQp();

  auto const &stress =
      moose_stub::PropertyStore::global().get_or_create<RankTwoTensor>(
          "MooseShearCheck_stress");
  Real const *stress_data = stress[0].dataPointer();
  Real const mu = 0.7; // validParams default
  for (int k = 0; k < 9; ++k) {
    EXPECT_NEAR(stress_data[k], 2.0 * mu * values[k], 1e-14) << "entry " << k;
  }
}

// The local-Newton material solves the backward-Euler residual
// R = (alpha − alpha_old)/dt − K·alpha in computeQpProperties. With the stub
// timestep _dt = 1 the analytic fixed point is alpha* = alpha_old/(1 − K·dt),
// and the downstream output sees sigma_y = K·alpha*.
TEST(MooseStubGate, NewtonStateConvergesToFixedPoint) {
  Real const alpha_old = 0.3;
  auto &old_prop = moose_stub::PropertyStore::global().get_or_create<Real>(
      moose_stub::old_property_key("MooseNewtonCheck_alpha"));
  old_prop[0] = alpha_old;

  auto params = MooseNewtonCheck::validParams();
  MooseNewtonCheck material(params);
  material.stubEvaluateQp();

  Real const K = 0.5; // validParams default
  Real const dt = 1.0; // moose_stub Material::_dt
  Real const alpha_star = alpha_old / (1.0 - K * dt);

  auto const &alpha = moose_stub::PropertyStore::global().get_or_create<Real>(
      "MooseNewtonCheck_alpha");
  auto const &sigma_y =
      moose_stub::PropertyStore::global().get_or_create<Real>(
          "MooseNewtonCheck_sigma_y");
  EXPECT_NEAR(alpha[0], alpha_star, 1e-10);
  EXPECT_NEAR(sigma_y[0], K * alpha_star, 1e-10);
}

} // namespace
