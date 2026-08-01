# CalculiX target — running a codegen'd material in a real FE solver

`CalculiXExternalTarget` (`make_target("calculix")`) emits `<Model>_ext.cpp`,
compiled to a shared library `lib<MODEL>.so` that CalculiX loads at **runtime via
`dlopen`**. Build `ccx` **once** with external-behaviour support, then each new
material is just a new `.so` selected by name in the input deck — **no `ccx`
recompile per material**.

CalculiX's external mechanism (`external.c`) parses a material name
`*MATERIAL, NAME=@<LIB>_<FUNC>` (it uppercases the name), `dlopen`s `lib<LIB>.so`
and `dlsym`s `<FUNC>`. The target emits an `extern "C" void NCG_UMAT(...)`
matching CalculiX's external `calculixptr` ABI and names the library after the
model, so the deck is `NAME=@<MODEL>_NCG_UMAT`.

The emitted file embeds the target-agnostic `<Model>_compute` (full dense tmech
tensors, no Voigt) and wraps it in a Voigt boundary built from tmech's `abq_std`
adaptor, whose ordering `{11,22,33,12,13,23}` is exactly CalculiX's `emec` /
`stre` / tangent order (the external STANDARD interface passes NATIVE quantities):

* `emec` (tensorial Lagrange strain) → strain tensor via `abq_std<3,false>`
  (no engineering-shear scaling — CalculiX passes the tensorial components);
* stress tensor → `stre(6)` via the same adaptor;
* the **minor-symmetric** rank-4 tangent → a 6×6 via `abq_std<3,false>` which,
  for a minor-symmetric `C`, *is* CalculiX's engineering `stiff` D-matrix
  directly (no ×2 factors), then packed into `stiff(21)` column-major
  upper-triangular (`k = i + j*(j+1)/2`, `0`-based `i<=j`), symmetrized.

Because ccx runs the element loop multi-threaded, the plugin holds a stateless
evaluator in a `thread_local` (one per thread) and reads the material constants
from `MPROPS` on **every** call (ccx interpolates them by temperature).

## Files

Each test comes as a **pair** of decks that differ ONLY in the material — the
`@`-material (our `.so`) and CalculiX's own built-in `*ELASTIC` (the gold):

* `uniaxial_c3d8_external.inp` / `uniaxial_c3d8_builtin.inp` — single C3D8,
  laterally confined, x1-face pulled by 0.01 → uniaxial strain `ε=diag(0.01,0,0)`.
* `simpleshear_c3d8_external.inp` / `simpleshear_c3d8_builtin.inp` — simple shear
  (tensorial ε₁₂=0.005); also pins the shear convention (see below).
* the diff is done by `tests/calculix/compare_dat.py` (single source of truth,
  shared with the test harness).
* `build_and_run_external.sh` — builds SPOOLES + `ccx` **once** with external
  support, compiles the generated `.so`, and runs each gold/test pair through ccx,
  comparing field-by-field.

The `*ELASTIC` gold uses `E=1.855, ν=0.325`, the exact equivalent of the codegen
material's `λ=1.3, μ=0.7`.

## Run it

```sh
# generate the material (part of the calculix_check_driver gate build)
cmake --build build --target calculix_check_driver
# build ccx once, run our .so vs CalculiX's built-in *ELASTIC, diff the fields
examples/calculix/build_and_run_external.sh
```

Prereqs (Debian/Ubuntu): `gcc g++ gfortran make wget perl`, plus
`libarpack2-dev liblapack-dev libblas-dev`. SPOOLES is fetched and built by the
script (SPOOLES 2.2 predates C99, so it's compiled with `-fcommon -std=gnu89`
and the gcc-14 implicit-int/-function errors downgraded). `ccx` is built with
`-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT -ldl`; `-ldl` goes on the link command,
not the Makefile's prerequisite `LIBS` (GNU make can't resolve `-l...` as a
target file).

## Verified result

The codegen'd `.so` reproduces CalculiX's own built-in `*ELASTIC` **bit-for-bit**
— every field of the `.dat` (stresses, strains, displacements) matches to `0.0`
for both the uniaxial and shear tests:

```
=== uniaxial: gold (*ELASTIC) vs test (@ codegen .so) ===
160 fields compared; max_abs=0.00e+00 max_rel=0.00e+00 ... RESULT: PASS
=== simpleshear: gold (*ELASTIC) vs test (@ codegen .so) ===
128 fields compared; max_abs=0.00e+00 max_rel=0.00e+00 ... RESULT: PASS
```

(For reference the uniaxial stress is `S₁₁=(λ+2μ)·ε₁₁=0.027`, `S₂₂=S₃₃=0.013`.)

## Shear-convention validation

`uniaxial_c3d8_external.inp` has zero shear, so it cannot tell tensorial from
engineering shear. `simpleshear_c3d8_external.inp` applies `u_x = 0.01·y`
(tensorial `ε₁₂ = 0.005`); a real `ccx` run gives **S₁₂ = 0.007 = 2μ·ε₁₂**,
confirming the boundary reads/writes *tensorial* shear (`abq_std<3,false>`). Had
it treated the strain as engineering, S₁₂ would be 0.014.

## What's tested where

* **CI lock (no external deps):** `tests/generated/calculix_check_driver.cpp`
  calls the emitted `NCG_UMAT` exactly as CalculiX does (its `calculixptr` ABI)
  for one integration point, checks `stre`/`stiff` against an independent
  isotropic oracle, a packing-order negative control, the
  read-constants-every-call regression, and the `icmd==3` stress-only path.
* **Manual (not CI):** `build_and_run_external.sh` is the end-to-end
  confirmation against a real `ccx` built from source — it diffs our `.so`
  against CalculiX's own `*ELASTIC` (independent ground truth, not a self-derived
  oracle). Run by hand (building ccx is too heavy for CI).

## Scope

First cut is **stateless** materials: one symmetric strain input, one stress
output, one consistent tangent, plus scalar parameters (→ the `*USER MATERIAL`
constants, `MPROPS`). Scalar inputs, state variables (the `STATEV` round-trip),
and rate/implicit forms are rejected at emit time — the stateful path (e.g. J2
plasticity with equivalent-plastic-strain history) is a tracked follow-up.
