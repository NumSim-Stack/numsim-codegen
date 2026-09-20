// Emits the energy-derived St. Venant–Kirchhoff material (#108 front-end) via the
// StandaloneCxx target. See svk_from_energy_recipe.h.
#include "svk_from_energy_recipe.h"
#include <numsim_codegen/targets/standalone_cxx.h>
#include <iostream>
int main() {
  using namespace numsim::codegen;
  auto const model = examples::make_svk_from_energy();
  StandaloneCxxTarget target;
  for (auto const &f : target.emit(model)) std::cout << f.contents;
  return 0;
}
