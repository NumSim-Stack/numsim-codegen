// e2e generator for the first-PK SVK (P, F) example: emit the material (P = F·S
// + consistent tangent dP/dF) via the StandaloneCxx target to a header the driver
// then compiles and FD-verifies. Reuses the shipped recipe so the tested material
// is exactly the example.

#include "first_piola_svk_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: generate_fpsvk_check <out.h>\n";
    return 2;
  }
  auto const model = numsim::codegen::examples::make_first_piola_svk();
  numsim::codegen::StandaloneCxxTarget target;

  std::ofstream f(argv[1]);
  for (auto const &e : target.emit(model)) {
    f << e.contents;
  }
  return f ? 0 : 1;
}
