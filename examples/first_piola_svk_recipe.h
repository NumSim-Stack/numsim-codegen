#ifndef NUMSIM_CODEGEN_EXAMPLES_FIRST_PIOLA_SVK_RECIPE_H
#define NUMSIM_CODEGEN_EXAMPLES_FIRST_PIOLA_SVK_RECIPE_H

// St. Venant–Kirchhoff in FIRST-Piola–Kirchhoff form — the (P, F) work-conjugate
// pair, driven by the deformation gradient F.
//
//   C = Fᵀ F                     (right Cauchy–Green)
//   E = ½ (C − I)                (Green–Lagrange strain)
//   S = λ tr(E) I + 2μ E         (2nd Piola–Kirchhoff stress)
//   P = F · S                    (FIRST Piola–Kirchhoff stress, ∂ψ/∂F)
//   consistent tangent  dP/dF    (rank-4, the first elasticity tensor A)
//
// WHY THIS EXAMPLE — the first-PK conjugate pair + the C = FᵀF path.
// The stress work-conjugate to the deformation gradient F is the first Piola–
// Kirchhoff stress P, and the total-Lagrangian consistent tangent is dP/dF. This
// is the companion to the SVK (S, Green–Lagrange E) example; here the strain
// enters through C = FᵀF, so the tangent differentiates through a QUADRATIC
// kinematic map in F — the aliased-leaf case (F appears twice in FᵀF).
//
// F IS NOT SYMMETRIC. It is declared roles::DeformationGradient (a General leaf),
// so dF/dF lowers to the full identity δ_ik δ_jl — NOT the minor-symmetric 𝕀ˢ a
// symmetric strain role would give (which would drop F's antisymmetric part and
// silently halve the tangent).
//
// Requires tmech ≥ v1.1.1: earlier tmech zeroed a lazy rank-4 operand fed into
// inner_product, which made exactly this dP/dF (through FᵀF) half-correct. This
// example is the end-to-end regression for that fix (numsim-cas#332).

#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_functions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/sequence.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <cstddef>

namespace numsim::codegen::examples {

inline ConstitutiveModel make_first_piola_svk() {
  using namespace numsim::cas;

  ConstitutiveModel model("FirstPiolaSvk");

  auto lambda =
      model.add_parameter("lambda", /*default=*/1.0, "Lame first parameter");
  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  // The differentiation input is the (non-symmetric) deformation gradient F.
  auto F = model.add_tensor_input("F", /*dim=*/3, /*rank=*/2,
                                  roles::DeformationGradient);

  auto const I = make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  auto const C = inner_product(F, sequence{1}, F, sequence{1}); // Fᵀ F
  auto const E = 0.5 * (C - I);                                 // Green–Lagrange
  auto const S = lambda * trace(E) * I + 2.0 * mu * E;          // 2nd Piola
  auto const P = inner_product(F, sequence{2}, S, sequence{1}); // P = F·S (1st Piola)

  model.add_output("P", P, roles::Stress);
  model.add_algorithmic_tangent("dP_dF", "P", "F");

  return model;
}

} // namespace numsim::codegen::examples

#endif // NUMSIM_CODEGEN_EXAMPLES_FIRST_PIOLA_SVK_RECIPE_H
