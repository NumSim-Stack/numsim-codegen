// #108 e2e generator: emit the energy-derived SVK material to a header the driver
// compiles and verifies against the hand-written closed form.
#include "svk_from_energy_recipe.h"
#include <numsim_codegen/targets/standalone_cxx.h>
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
  if (argc < 2) { std::cerr << "usage: generate_energy_check <out.h>\n"; return 2; }
  auto const model = numsim::codegen::examples::make_svk_from_energy();
  numsim::codegen::StandaloneCxxTarget target;
  std::ofstream f(argv[1]);
  for (auto const &e : target.emit(model)) f << e.contents;
  return f ? 0 : 1;
}
