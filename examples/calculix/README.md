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

* `uniaxial_c3d8_external.inp` — single C3D8, laterally confined, x1-face pulled
  by 0.01 on a unit cube → homogeneous uniaxial strain `ε = diag(0.01,0,0)`.
  Material `@LINEARELASTIC_NCG_UMAT`, `*USER MATERIAL, CONSTANTS=2` = `λ, μ`.
* `simpleshear_c3d8_external.inp` — simple shear (tensorial ε₁₂=0.005); validates
  the shear convention (see below).
* `build_and_run_external.sh` — builds SPOOLES + `ccx` **once** with external
  support, compiles the generated `.so`, runs the deck by `@`-name, checks stress.

## Run it

```sh
# generate the material (part of the calculix_check_driver gate build)
cmake --build build --target calculix_check_driver
# build ccx once with external support, emit lib<MODEL>.so, run by @-name
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

For `λ=1.3, μ=0.7, ε₁₁=0.01` the closed-form uniaxial-strain stress is
`S₁₁=(λ+2μ)·ε₁₁=0.027`, `S₂₂=S₃₃=λ·ε₁₁=0.013`. A real `ccx 2.22` run of the
codegen'd `.so` reproduces this exactly at every integration point:

```
 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL
         1   1  2.700000E-02  1.300000E-02  1.300000E-02  0.0  0.0  0.0
         ...   (all 8 integration points identical)
```

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
* **Manual (not CI):** `build_and_run_external.sh` + the decks are the
  end-to-end confirmation against a real `ccx` built from source; run by hand
  (building ccx is too heavy for CI).

## Scope

First cut is **stateless** materials: one symmetric strain input, one stress
output, one consistent tangent, plus scalar parameters (→ the `*USER MATERIAL`
constants, `MPROPS`). Scalar inputs, state variables (the `STATEV` round-trip),
and rate/implicit forms are rejected at emit time — the stateful path (e.g. J2
plasticity with equivalent-plastic-strain history) is a tracked follow-up.
