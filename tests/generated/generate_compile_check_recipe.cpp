// Generator program for the compile-check test. CMake runs this once at
// build time with the destination header path as argv[1]. The output is a
// self-contained header that the compile-check driver then #include's.
//
// The recipe exercises:
//   - Scalar input (x)
//   - Scalar parameter (a)
//   - Tensor input (eps)
//   - Tensor parameter (lam — well, scalar parameter; tensors are inputs)
//   - Scalar output (y)
//   - Tensor output (sigma)
//   - Templated tensor argument types (so an adaptor at the call site
//     compiles without copying through an intermediate tmech::tensor).

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <output-header-path>\n";
    return 1;
  }

  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("CompileCheck");

  auto a = model.add_parameter("a", 2.0);
  auto x = model.add_scalar_input("x");
  auto mu = model.add_parameter("mu", 0.5);
  auto eps = model.add_tensor_input("eps", 3, 2);

  // Scalar output: y = a*x + sin(x)
  model.add_output("y", a * x + sin(x));

  // Tensor output: sigma = 2*mu*eps
  model.add_output("sigma", 2 * mu * eps);

  StandaloneCxxTarget target;
  auto files = target.emit(model);
  if (files.size() != 1) {
    std::cerr << "expected single emitted file, got " << files.size() << "\n";
    return 1;
  }

  std::ofstream out(argv[1]);
  if (!out) {
    std::cerr << "could not open '" << argv[1] << "' for writing\n";
    return 1;
  }
  out << files[0].contents;
  return 0;
}
