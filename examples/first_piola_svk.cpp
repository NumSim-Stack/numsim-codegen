// Emits the first-Piola–Kirchhoff St. Venant–Kirchhoff material (P = F·S and the
// consistent tangent dP/dF) via the StandaloneCxx target. See
// first_piola_svk_recipe.h for the formulation.

#include "first_piola_svk_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <iostream>

int main() {
  using namespace numsim::codegen;
  auto const model = examples::make_first_piola_svk();
  StandaloneCxxTarget target;
  for (auto const &f : target.emit(model)) {
    std::cout << f.contents;
  }
  return 0;
}
