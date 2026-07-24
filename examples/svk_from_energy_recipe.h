#ifndef NUMSIM_CODEGEN_EXAMPLES_SVK_FROM_ENERGY_RECIPE_H
#define NUMSIM_CODEGEN_EXAMPLES_SVK_FROM_ENERGY_RECIPE_H

// St. Venant–Kirchhoff DERIVED FROM ITS ENERGY POTENTIAL — the #108
// material-compiler front-end. The human writes only the strain-energy density
//
//   ψ(E) = ½λ (tr E)² + μ (E : E)
//
// and `add_hyperelastic_potential` derives the 2nd-Piola–Kirchhoff stress and
// the consistent tangent by differentiation:
//
//   S      = ∂ψ/∂E = λ tr(E) I + 2μ E     (NOT written by hand)
//   dS/dE  = ∂²ψ/∂E²                        (NOT written by hand)
//
// This is the same material as the hand-written St. Venant–Kirchhoff example —
// the e2e checks the derived S against that closed form — but here the stress
// and tangent fall out of `cas::diff` of ψ, the AceGen "differentiate the
// potential" paradigm.
//
// STRESS/STRAIN MEASURE: (S, Green–Lagrange E), 2nd Piola–Kirchhoff conjugate
// pair — same as the hand-written SVK example.
// CONSTRAINTS: `E` must be a declared INPUT leaf (add_hyperelastic_potential
// recovers its name from the handle and rejects a derived measure), and it is
// roles::Strain (symmetric), which is what makes the derived tangent
// minor-symmetric — a non-symmetric leaf would not.
// VERIFICATION BOUNDARY: the e2e compiles the StandaloneCxx form and checks the
// derived S against the independent closed form + the derived tangent vs FD +
// major symmetry (the Hessian property an energy-derived tangent must have).

#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

namespace numsim::codegen::examples {

inline ConstitutiveModel make_svk_from_energy() {
  using namespace numsim::cas;

  ConstitutiveModel model("SvkFromEnergy");

  auto lambda =
      model.add_parameter("lambda", /*default=*/1.0, "Lame first parameter");
  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  auto E = model.add_tensor_input("E", /*dim=*/3, /*rank=*/2, roles::Strain);

  // The ONLY physics the human writes: the strain-energy density ψ(E).
  auto const trE = trace(E);
  auto const psi = 0.5 * lambda * trE * trE + mu * dot(E); // dot(E) = E : E

  // Derive S = ∂ψ/∂E and the consistent tangent dS/dE = ∂²ψ/∂E².
  model.add_hyperelastic_potential("S", psi, E, "dS_dE");

  return model;
}

// A second, deliberately NON-QUADRATIC potential so the derived tangent is a
// genuine function of E (not the constant SVK tangent) — this exercises the
// second-differentiation machinery on a curvature-bearing Hessian, which a
// linear-stress material cannot.
//
//   ψ = ½λ (tr E)² + μ (E:E) + c (E:E)²
//   S = ∂ψ/∂E = λ tr(E) I + 2μ E + 4c (E:E) E      (cubic in E)
//   dS/dE = ∂²ψ/∂E²                                 (non-constant)
inline ConstitutiveModel make_nonlinear_from_energy() {
  using namespace numsim::cas;

  ConstitutiveModel model("NonlinearFromEnergy");

  auto lambda = model.add_parameter("lambda", 1.0, "Lame first parameter");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto c = model.add_parameter("c", 0.4, "Quartic stiffening coefficient");
  auto E = model.add_tensor_input("E", 3, 2, roles::Strain);

  auto const trE = trace(E);
  auto const EE = dot(E); // E : E
  auto const psi =
      0.5 * lambda * trE * trE + mu * EE + c * EE * EE;

  model.add_hyperelastic_potential("S", psi, E, "dS_dE");

  return model;
}

} // namespace numsim::codegen::examples

#endif // NUMSIM_CODEGEN_EXAMPLES_SVK_FROM_ENERGY_RECIPE_H
