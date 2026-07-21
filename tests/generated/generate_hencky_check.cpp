// Slice-3 generator for the MOOSE Hencky material e2e (numsim-codegen #105
// follow-up): emit the Hencky hyperelastic material (spectral stress + exact
// consistent tangent) via the StandaloneCxx target to a header the driver then
// compiles and FD-verifies. Reuses the shared recipe in
// examples/hencky_recipe.h so the tested material is exactly the shipped
// example.

#include "hencky_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: generate_hencky_check <out.h>\n";
    return 2;
  }
  auto const model = numsim::codegen::examples::make_hencky_hyperelastic();
  numsim::codegen::StandaloneCxxTarget target;

  std::ofstream f(argv[1]);
  for (auto const &e : target.emit(model)) {
    f << e.contents;
  }
  return f ? 0 : 1;
}
