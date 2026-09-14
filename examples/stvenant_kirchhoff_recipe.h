#ifndef NUMSIM_CODEGEN_EXAMPLES_STVENANT_KIRCHHOFF_RECIPE_H
#define NUMSIM_CODEGEN_EXAMPLES_STVENANT_KIRCHHOFF_RECIPE_H

// St. Venant–Kirchhoff hyperelastic material — the (2nd-Piola–Kirchhoff stress
// S, Green–Lagrange strain E) work-conjugate pair.
//
//   energy    ψ(E) = ½λ (tr E)² + μ (E : E)
//   stress    S = ∂ψ/∂E = λ tr(E) I + 2μ E          (2nd Piola–Kirchhoff)
//   tangent   dS/dE = λ (I ⊗ I) + 2μ I⁴ˢ            (constant, minor-symmetric)
//
// WHY THIS EXAMPLE — choosing the stress–strain pair for the tangent.
// The consistent tangent MOOSE's total-Lagrangian StressDivergence kernels want
// is dS/dE_gl: the 2nd-PK stress differentiated w.r.t. the Green–Lagrange strain.
// `add_algorithmic_tangent(name, of_output, wrt_input)` differentiates w.r.t. a
// declared INPUT leaf, so the pair is selected simply by making E the input and
// expressing the stress through it — here `add_algorithmic_tangent("dS_dE", "S",
// "E")`. Contrast the Hencky example, which differentiates w.r.t. C (the right
// Cauchy–Green tensor) and therefore yields dσ/dC, NOT dS/dE.
//
// SCOPE — why S is written in CLOSED FORM here, not derived from ψ.
// For St. Venant–Kirchhoff the 2nd-PK stress has the exact closed form above, so
// the recipe states S directly and the tangent is `cas::diff(tensor, tensor)`.
// The general hyperelastic route S = ∂ψ/∂E (differentiate a scalar potential
// w.r.t. a tensor) needs `diff(tensor_to_scalar, tensor)`, which is the deferred
// material-compiler capability (#108) and the differentiation-seam feature (#111
// — selecting a conjugate pair when the strain measure is a DERIVED quantity such
// as the log-strain ½logC). This example uses only the primary-path
// `diff(tensor, tensor)` and pulls in no spectral machinery.

#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <cstddef>

namespace numsim::codegen::examples {

inline ConstitutiveModel make_stvenant_kirchhoff() {
  using namespace numsim::cas;

  ConstitutiveModel model("StVenantKirchhoff");

  auto lambda =
      model.add_parameter("lambda", /*default=*/1.0, "Lame first parameter");
  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  // The differentiation variable IS the strain measure: Green–Lagrange E.
  auto E = model.add_tensor_input("E", /*dim=*/3, /*rank=*/2, roles::Strain);

  auto const I =
      make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  // 2nd Piola–Kirchhoff stress S = λ tr(E) I + 2μ E (closed form of ∂ψ/∂E).
  auto const S = lambda * trace(E) * I + 2.0 * mu * E;

  model.add_output("S", S, roles::Stress);
  // The (S, E) conjugate tangent — this is the pair selection.
  model.add_algorithmic_tangent("dS_dE", "S", "E");

  return model;
}

} // namespace numsim::codegen::examples

#endif // NUMSIM_CODEGEN_EXAMPLES_STVENANT_KIRCHHOFF_RECIPE_H
