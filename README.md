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

## Multi-recipe generator

For projects that ship several constitutive models, `examples/recipe_registry.h`
shows the pattern: each recipe is a factory function returning a
`ConstitutiveModel`; a `registry()` vector maps human-readable names to
factories; one generator binary iterates the catalogue and emits source
files for every recipe through a chosen target.

```bash
./build/examples/recipe_registry_gen <out-dir> <target: numsim_material|standalone|moose>
```

Six worked recipes are shipped (linear-elastic shear, thermo-elastic shear,
phase-coupled shear, K/G elasticity, strain-based damage, J2 trial). Each
includes a comment block describing its Phase A limitations and the
upstream change needed to lift them. Add a new recipe by appending a
factory + a registry entry — no other code changes required.

## Build

CMake 3.20+, C++23, GCC 14 or Clang 19 or MSVC 19.30+.

Clang 18 is **not in the CI matrix** — ubuntu-24.04 defaults to pairing
it with libstdc++-13, which lacks the C++23 `<expected>` header. The
workaround of manually installing `libstdc++-14-dev` alongside clang-18
is untested and unsupported here. The supported clang path is clang-19
from the LLVM toolchain repo paired with `libstdc++-14-dev` — see
`docs/workflow.md` §6.2 and `.github/workflows/build.yml`.

To verify your toolchain: `clang++ --version` should report ≥19, and the
libstdc++ it picks up should be ≥14 (check with
`echo "#include <expected>" | $CXX -std=c++23 -x c++ -E - > /dev/null`).

```bash
git clone https://github.com/NumSim-Stack/numsim-codegen.git
cd numsim-codegen
cmake -B build
cmake --build build -j
ctest --test-dir build
```

The build uses [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) (downloaded
once at configure time and SHA256-pinned in `CMakeLists.txt`) to pull
numsim-cas. The first `cmake -B build` needs network access; subsequent
configures reuse the cached copy. To share that cache across projects,
point CPM at a local cache directory — either via env var:

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/CPM
cmake -B build
```

…or per-invocation via `-D`:

```bash
cmake -B build -DCPM_SOURCE_CACHE=/path/to/cache
```

CMake does not expand `~` in paths supplied to `-D`, so use `$HOME` or an
absolute path.

## License

GPL-3.0. Matches numsim-cas.
