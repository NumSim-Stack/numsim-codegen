// #108 e2e generator: emit the energy-derived materials to headers the driver
// compiles and verifies. Two materials: SvkFromEnergy (quadratic ψ → constant
// tangent, checked against the hand-written closed form) and NonlinearFromEnergy
// (quartic ψ → non-constant tangent, exercising the second-diff machinery).
#include "svk_from_energy_recipe.h"
#include <numsim_codegen/targets/standalone_cxx.h>
#include <fstream>
#include <iostream>

namespace {
bool write_header(numsim::codegen::ConstitutiveModel const &m, char const *path) {
  std::ofstream f(path);
  for (auto const &e : numsim::codegen::StandaloneCxxTarget{}.emit(m))
    f << e.contents;
  return static_cast<bool>(f);
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: generate_energy_check <svk.h> <nonlinear.h>\n";
    return 2;
  }
  using namespace numsim::codegen::examples;
  if (!write_header(make_svk_from_energy(), argv[1])) return 1;
  if (!write_header(make_nonlinear_from_energy(), argv[2])) return 1;
  return 0;
}
