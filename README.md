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
| `StandaloneCxxTarget` | ✓ | Single inline header with the generic compute function |
| `MooseMaterialTarget` | ✓ | `.h` + `.C` pair: Material class with `validParams`, constructor, `computeQpProperties`, optional `Jacobian_mult` consistent tangent |
| `NumSimMaterialTarget` | ✓ | numsim-materials rate / return-map material header + JSON config (one scalar rate variable / Newton unknown per material; additional scalar or tensor history state supported) |
| `AbaqusUMATTarget`    | planned | Fortran-callable `extern "C"` UMAT with Voigt boundary |
| `AnsysUSERMATTarget`  | planned | Fortran-callable USERMAT |
| `LSDynaUMATTarget`    | planned | LS-DYNA convention |

## Status

Working today: scalar / tensor / tensor-to-scalar emission with pointer-based
CSE; semantic roles (open set); scalar and tensor state variables with
`_old` pairing; backward-Euler time integration lowered to in-function
Newton solves (scalar and coupled N×N via Eigen); algorithmic (consistent)
tangents via symbolic differentiation, FD-verified in compile-and-run test
gates; spectral decomposition lowering (`log`/`exp`/`sqrt` of symmetric
tensors) with exact tangents including eigenvalue coalescence; an
end-to-end gate that runs generated materials through the real
numsim-materials solver.

Not yet: MOOSE stateful (history) properties, Kuhn-Tucker/inequality
switches in lowered return maps, tensor-valued Newton unknowns, the
Abaqus/ANSYS/LS-DYNA targets.

## Example (MOOSE target)

```cpp
// Adapted from examples/moose_linear_elastic.cpp (compiled in CI).
#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

int main() {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("LinearElasticShear");

  auto mu  = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", /*dim=*/3, /*rank=*/2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);

  MooseMaterialTarget target("MyApp");
  for (auto const& file : target.emit(model)) {
    write_to_disk(file.filename, file.contents);  // your I/O
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

## Semantic roles

Recipe declarations carry semantic `Role` tags (`roles::Strain`, `roles::Stress`,
`roles::ConsistentTangent`, …) that backends interpret to wire the recipe to
framework-specific inputs and outputs. The role set is open: construct a custom
`Role{.name = "phase_field", .is_driving = true, .expected_rank = 0}` and backends
route it by its attributes — no library change needed. The same recipe works across
targets without modification — only the target choice changes.

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

A recipe outside a target's supported scope is reported as
`SKIPPED (<reason>)` and the run continues. Note: the six shipped recipes
all target `standalone`/`moose`; none currently fits `numsim_material`'s
single-scalar-state-variable contract, so that target's worked examples
live in `tests/generated/generate_numsim_material_check.cpp` instead.

## Build

CMake 3.25+, C++23, GCC 14 or Clang 19. (MSVC is untested — no Windows CI;
on paper the code needs ≥ 19.33 for `std::expected`.)

Supported consumption is **`add_subdirectory` / FetchContent / CPM** — the
example pattern is exactly how the tests and examples link
(`target_link_libraries(app PRIVATE numsim::codegen)`). `cmake --install` +
`find_package` is **not supported yet**: no CMake package config is
generated, and the dependency headers (numsim-cas, tmech) are not installed
alongside — blocked on numsim-cas exporting installable targets.

Clang 18 (and older) **cannot build this project at all**: libstdc++'s
`std::expected` is guarded by `__cpp_concepts >= 202002L`, which clang
only defines from clang-19 — so `<expected>` stays empty under clang ≤ 18
*regardless of the libstdc++ version installed* (verified: clang-18
selecting the GCC-14 toolchain still fails). The supported clang path is
clang-19 from the LLVM toolchain repo paired with `libstdc++-14-dev` —
see `docs/workflow.md` §6.2 and `.github/workflows/build.yml`.

To verify your toolchain: `clang++ --version` should report ≥19, and the
following should compile (note: merely `#include`-ing `<expected>` is NOT
a valid check — the header preprocesses fine on clang ≤ 18, it just leaves
`std::expected` undefined):

```bash
printf '#include <expected>\nstd::expected<int,int> e{1};\n' | \
  $CXX -std=c++23 -x c++ -fsyntax-only -
```

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
