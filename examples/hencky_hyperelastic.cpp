// Emits the Hencky hyperelastic material (stress + consistent tangent) via the
// StandaloneCxx target. See hencky_recipe.h for the formulation.

#include "hencky_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <iostream>

int main() {
  using namespace numsim::codegen;
  auto const model = examples::make_hencky_hyperelastic();
  StandaloneCxxTarget target;
  for (auto const &f : target.emit(model)) {
    std::cout << f.contents;
  }
  return 0;
}
