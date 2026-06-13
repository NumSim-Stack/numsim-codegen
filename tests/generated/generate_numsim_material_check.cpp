// Phase B end-to-end gate — generator half.
//
// Emits a NumSimMaterialTarget rate-function material for the canonical
// dα/dt = K·α recipe (with K = -1 this is exponential decay, y(1) = e^-1) and
// writes the header to argv[1]. The companion driver
// (numsim_material_check_driver.cpp) compiles this generated header against the
// REAL numsim-materials / numsim-core headers and runs it through the real
// rk_integrator — proving the emitted code conforms to the solver contract,
// which the string-asserting NumSimMaterialTargetTest cannot.
#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>

#include <fstream>
#include <iostream>

using namespace numsim::cas;
using namespace numsim::codegen;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: generate_numsim_material_check <out-header>\n";
    return 2;
  }

  // dα/dt = K·α — rate = K·α, rate_derivative = K (via cas::diff).
  ConstitutiveModel m("LinearHardening");
  auto K = m.add_parameter("K", -1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);

  for (auto const& f : NumSimMaterialTarget{}.emit(m)) {
    if (f.kind == EmittedFile::Kind::Header) {
      std::ofstream os(argv[1]);
      if (!os) {
        std::cerr << "could not open " << argv[1] << " for writing\n";
        return 1;
      }
      os << f.contents;
      return 0;
    }
  }
  std::cerr << "NumSimMaterialTarget emitted no header\n";
  return 1;
}
