# numsim-codegen

C++ code generation from [numsim-cas](https://github.com/NumSim-Stack/numsim-cas) symbolic
expressions, targeting MOOSE constitutive Material classes.

## What it does

Takes a constitutive model declared symbolically via numsim-cas (strain energy potential,
yield function, stress recipe, etc.) and emits target-specific C++ source files. The
architecture is layered:

```
Layer 3: Target wrapper (MOOSE / Abaqus UMAT / ANSYS USERMAT / ...)
                          ↓
Layer 2: Generic compute function (target-agnostic, tmech tensors)
                          ↓
Layer 1: ConstitutiveModel recipe + codegen visitors
```

Layers 1 and 2 are shared across every target. Only Layer 3 differs per framework.
tmech is the internal tensor type throughout; the target wrapper converts at the
boundary using tmech's adaptors (`full`, `voigt`, `abq_std`).

### Targets

| Target | Status | Output |
|--------|--------|--------|
| `StandaloneCxxTarget` | ✓ Phase A | Single inline header with the generic compute function |
| `MooseMaterialTarget` | ✓ Phase A | `.h` + `.C` pair: Material class with `validParams`, constructor, `computeQpProperties` |
| `AbaqusUMATTarget`    | planned | Fortran-callable `extern "C"` UMAT with Voigt boundary |
| `AnsysUSERMATTarget`  | planned | Fortran-callable USERMAT |
| `LSDynaUMATTarget`    | planned | LS-DYNA convention |

## Status

**Phase A scaffold.** Working: scalar codegen visitor with pointer-based CSE, recipe
class registry, and a linear-elasticity smoke test. Not yet working: full tensor /
tensor-to-scalar visitor coverage, MOOSE Material text templating, history / internal
variables.

See [issue #139 in numsim-cas](https://github.com/NumSim-Stack/numsim-cas/issues/139)
for the umbrella tracking the phase plan.

## Example (MOOSE target)

```cpp
#include <numsim_codegen/numsim_codegen.h>
using namespace numsim::cas;
using namespace numsim::codegen;

int main() {
  ConstitutiveModel model("LinearElasticShear");

  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, InputRole::Strain);
  auto sigma = 2 * mu * eps;
  model.add_output("stress", sigma, OutputRole::Stress);

  MooseMaterialTarget target("MyApp");
  for (auto const& file : target.emit(model)) {
    write_to_disk(file.filename, file.contents);
  }
}
```

This generates two files (`LinearElasticShear.h` and `LinearElasticShear.C`) containing
a complete MOOSE `Material` subclass with `validParams`, constructor, and
`computeQpProperties` body.

The generated Layer-2 compute function is **templated on tensor argument types**, so
the MOOSE boundary passes `tmech::adaptor<double, 3, 2, tmech::full<3>>` directly to
the compute function — no intermediate `tmech::tensor` materialisation in either
direction. The only data motion is the unavoidable read-from-strain and
write-to-stress at the FEM/constitutive boundary. A compile-check test
(`tests/generated/`) verifies this end-to-end by including the generated header,
calling the function with both `tmech::tensor` and `tmech::adaptor` arguments, and
asserting numerical results.

The generated body uses common-subexpression elimination — each unique subterm of the
DAG is emitted exactly once as `auto tN = ...;`.

## InputRole and OutputRole

Recipe declarations carry semantic tags (`InputRole::Strain`, `OutputRole::Stress`, etc.)
that backends interpret to wire the recipe to framework-specific inputs and outputs.
The same recipe works across MOOSE, Abaqus, ANSYS without modification — only the
target choice changes.

## Build

CMake 3.20+, C++23, GCC 14 or Clang 18 or MSVC 19.30+.

```bash
git clone https://github.com/NumSim-Stack/numsim-codegen.git
cd numsim-codegen
cmake -B build
cmake --build build -j
ctest --test-dir build
```

The build pulls numsim-cas via [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
(vendored at `cmake/CPM.cmake`, pinned to an explicit SHA). No manual install
needed. To share downloads across builds, point CPM at a cache directory:

```bash
cmake -B build -DCPM_SOURCE_CACHE=$HOME/.cache/CPM
# or set CPM_SOURCE_CACHE in the environment.
```

## License

GPL-3.0. Matches numsim-cas.

### Third-party

`cmake/CPM.cmake` is a vendored release of [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
by Lars Melchior, distributed under the MIT License (see the SPDX header at the
top of that file). Bump by replacing the file with a newer release from
<https://github.com/cpm-cmake/CPM.cmake/releases>.
