#ifndef NUMSIM_CODEGEN_EXAMPLES_HENCKY_RECIPE_H
#define NUMSIM_CODEGEN_EXAMPLES_HENCKY_RECIPE_H

// Hencky (logarithmic-strain) hyperelastic material — the spectral constitutive
// demo (#105 / numsim-cas #326). The stress is a spectral tensor function of
// the right Cauchy–Green tensor C, and the consistent tangent is its exact
// derivative, differentiated through log(C) into the eigenvalue /
// eigenprojection / divided-difference nodes that numsim-codegen now lowers.
//
//   Hencky strain    E = ½ log(C)
//   energy           ψ(E) = ½λ (tr E)² + μ E:E
//   stress           σ = λ tr(E) I + 2μ E
//   consistent tangent  dσ/dC   (rank-4, via cas::diff(tensor, tensor))

#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_isotropic_functions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <cstddef>

namespace numsim::codegen::examples {

inline ConstitutiveModel make_hencky_hyperelastic() {
  using namespace numsim::cas;

  ConstitutiveModel model("HenckyHyperelastic");

  auto lambda =
      model.add_parameter("lambda", /*default=*/1.0, "Lame first parameter");
  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  auto C = model.add_tensor_input("C", /*dim=*/3, /*rank=*/2, roles::Strain);

  auto const E = 0.5 * log(C); // Hencky strain ½ log C
  auto const I =
      make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  auto const stress = lambda * trace(E) * I + 2.0 * mu * E;

  model.add_output("stress", stress, roles::Stress);
  model.add_algorithmic_tangent("dstress_dC", "stress", "C");

  return model;
}

} // namespace numsim::codegen::examples

#endif // NUMSIM_CODEGEN_EXAMPLES_HENCKY_RECIPE_H
