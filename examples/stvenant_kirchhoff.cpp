// Emits the St. Venant–Kirchhoff hyperelastic material (2nd-PK stress S +
// consistent tangent dS/dE) via the StandaloneCxx target. See
// stvenant_kirchhoff_recipe.h for the formulation and the pair-selection notes.

#include "stvenant_kirchhoff_recipe.h"

#include <numsim_codegen/targets/standalone_cxx.h>

#include <iostream>

int main() {
  using namespace numsim::codegen;
  auto const model = examples::make_stvenant_kirchhoff();
  StandaloneCxxTarget target;
  for (auto const &f : target.emit(model)) {
    std::cout << f.contents;
  }
  return 0;
}
