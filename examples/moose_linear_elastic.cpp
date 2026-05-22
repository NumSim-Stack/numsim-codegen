// Generate a MOOSE Material class for a linear elastic shear stress.
// Writes the .h and .C files to stdout in sequence.

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

  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);

  auto sigma = 2 * mu * eps;
  model.add_output("stress", sigma, roles::Stress);

  MooseMaterialTarget target("ConstitutiveApp");
  for (auto const &file : target.emit(model)) {
    std::cout << "// ==== " << file.filename << " ====\n";
    std::cout << file.contents << "\n";
  }
  return 0;
}
