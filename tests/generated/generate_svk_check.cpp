// e2e generator for the St. Venant–Kirchhoff (S, E) example: emit the material
// (2nd-PK stress + consistent tangent dS/dE) via the StandaloneCxx target to a
// header the driver then compiles and FD-verifies. Reuses the shipped recipe in
// examples/stvenant_kirchhoff_recipe.h so the tested material is exactly the
// example.

#include "stvenant_kirchhoff_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: generate_svk_check <out.h>\n";
    return 2;
  }
  auto const model = numsim::codegen::examples::make_stvenant_kirchhoff();
  numsim::codegen::StandaloneCxxTarget target;

  std::ofstream f(argv[1]);
  for (auto const &e : target.emit(model)) {
    f << e.contents;
  }
  return f ? 0 : 1;
}
