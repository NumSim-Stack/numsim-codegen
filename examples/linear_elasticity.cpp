// Phase-A smoke example: σ = 2μ ε (shear term of linear elasticity).
// Demonstrates the numsim-codegen recipe API end-to-end.
//
// The full linear-elasticity stress σ = λ tr(ε) I + 2μ ε requires a
// `t2s * tensor` operator that numsim-cas doesn't currently expose
// (trace(ε) is tensor-to-scalar; multiplying by the rank-2 identity I
// needs a `tensor_to_scalar_with_tensor_mul`-shaped construction with no
// user-facing operator). Tracked as a follow-up.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <iostream>

int main() {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("LinearElasticShear");

  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", /*dim=*/3, /*rank=*/2);

  auto sigma = 2 * mu * eps;

  model.add_output("stress", sigma);

  std::cout << model.emit_compute_function();
  return 0;
}
