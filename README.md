# numsim-codegen

C++ code generation from [numsim-cas](https://github.com/NumSim-Stack/numsim-cas) symbolic
expressions, targeting MOOSE constitutive Material classes.

## What it does

Takes a constitutive model declared symbolically via numsim-cas (strain energy potential,
yield function, stress recipe, etc.) and emits a C++ source file containing:

- A self-contained function that computes the model outputs (stress, tangent, history
  evolution, etc.) from inputs (strain, history, parameters).
- All operations use [tmech](https://github.com/petlenz/tmech) tensors internally.
- A `MOOSE` Material class wrapper that handles `RankTwoTensor ↔ tmech::tensor`
  conversions at the boundary.

## Status

**Phase A scaffold.** Working: scalar codegen visitor with pointer-based CSE, recipe
class registry, and a linear-elasticity smoke test. Not yet working: full tensor /
tensor-to-scalar visitor coverage, MOOSE Material text templating, history / internal
variables.

See [issue #139 in numsim-cas](https://github.com/NumSim-Stack/numsim-cas/issues/139)
for the umbrella tracking the phase plan.

## Example

```cpp
#include <numsim_codegen/numsim_codegen.h>
using namespace numsim::cas;
using namespace numsim::codegen;

int main() {
  ConstitutiveModel model("LinearElastic");

  // Parameters become getParam<Real>(...) calls in the MOOSE wrapper.
  auto lam = model.add_parameter("lam", 1.0);
  auto mu  = model.add_parameter("mu",  0.5);

  // Inputs become MaterialProperty<...> reads at the boundary.
  auto eps = model.add_tensor_input("eps", /*dim=*/3, /*rank=*/2);

  // Build the constitutive expression with the standard numsim-cas API.
  auto I     = make_expression<kronecker_delta>(std::size_t{3});
  auto sigma = lam * trace(eps) * I + 2 * mu * eps;

  // Outputs become MaterialProperty<...> writes.
  model.add_output("stress",  sigma);
  model.add_output("tangent", diff(sigma, eps));

  // Emit the generated C++ source.
  std::cout << model.emit() << std::endl;
}
```

The generated body uses common-subexpression elimination — each unique subterm of the
DAG is emitted exactly once as `auto tN = ...;`.

## Build

CMake 3.20+, C++23, GCC 14 or Clang 18 or MSVC 19.30+.

```bash
git clone https://github.com/NumSim-Stack/numsim-codegen.git
cd numsim-codegen
cmake -B build
cmake --build build -j
ctest --test-dir build
```

The build pulls numsim-cas via `FetchContent`. No manual install needed.

## License

GPL-3.0. Matches numsim-cas.
