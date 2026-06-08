// Phase 3b-1 (issue #35): consistent-tangent emission.
//
// For the shear stress σ = 2μ ε, the consistent (algorithmic) tangent is
//
//     dσ/dε = ∂σ/∂ε = 2μ I⁴ˢ
//
// `add_algorithmic_tangent(name, of_output, wrt_input)` requests it; the
// AlgorithmicTangentPass synthesises it as a rank-4 tensor output via
// `cas::diff(tensor, tensor)` — no upstream dependency. (The strain-coupled
// case, where σ depends on a solved internal variable, additionally needs
// numsim-cas#275's diff(tensor, scalar); see the pass header.)

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <iostream>

int main() {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("ElasticTangent");

  auto mu = model.add_parameter("mu", /*default=*/0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", /*dim=*/3, /*rank=*/2, roles::Strain);

  model.add_output("stress", 2 * mu * eps, roles::Stress);

  // Request the consistent tangent dstress/deps as a rank-4 output.
  model.add_algorithmic_tangent("dstress_deps", "stress", "eps");

  std::cout << model.emit_compute_function();
  return 0;
}
