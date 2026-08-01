# CalculiX target — running a codegen'd material in a real FE solver

`CalculiXUMATTarget` (`make_target("calculix")`) emits `<Model>_umat.cpp`: an
`extern "C" void umat_user_(...)` that matches CalculiX's **native** `umat_user`
ABI. CalculiX's `umat_main.f` dispatches to `umat_user` for any `*MATERIAL` whose
name begins with `USER`, so the generated object can be linked straight into
`ccx` built from source, in place of the stock `umat_user.f`.

The emitted file embeds the target-agnostic `<Model>_compute` (full dense tmech
tensors, no Voigt) and wraps it in a Voigt boundary built from tmech's `abq_std`
adaptor, whose ordering `{11,22,33,12,13,23}` is exactly CalculiX's `emec` /
`stre` / tangent order:

* `emec` (tensorial Lagrange strain) → strain tensor via `abq_std<3,false>`
  (no engineering-shear scaling — CalculiX passes the tensorial components);
* stress tensor → `stre(6)` via the same adaptor;
* the **minor-symmetric** rank-4 tangent → a 6×6 via `abq_std<3,false>` which,
  for a minor-symmetric `C`, *is* CalculiX's engineering `stiff` D-matrix
  directly (no ×2 factors), then packed into `stiff(21)` column-major
  upper-triangular (`k = i + j*(j+1)/2`, `0`-based `i<=j`), symmetrized.

## Files

* `uniaxial_c3d8.inp` — a single C3D8 element, laterally confined, x1-face pulled
  by 0.01 on a unit cube → homogeneous uniaxial strain `ε = diag(0.01,0,0)`. The
  material is `USERELAS` with `*USER MATERIAL, CONSTANTS=2` = `λ, μ` (1.3, 0.7).
* `build_and_run.sh` — fetches + builds SPOOLES and `ccx` from source with the
  generated material linked in, runs the deck, and checks the stress.

## Two ways to run: compiled-in vs. external (dlopen)

There are two CalculiX targets, both reusing the same `abq_std` boundary:

| | `CalculiXUMATTarget` (`calculix`) | `CalculiXExternalTarget` (`calculix_external`) |
|---|---|---|
| Output | `<Model>_umat.cpp` (`umat_user_`) | `<Model>_ext.cpp` → `lib<MODEL>.so` (`NCG_UMAT`) |
| Linkage | compiled **into** ccx | `dlopen`'d at **runtime** |
| New material | **relink ccx** | just a new `.so` — **no ccx recompile** |
| Deck name | `NAME=USER<Model>` | `NAME=@<MODEL>_NCG_UMAT` |
| Build ccx | with the material linked | **once**, with external support |

The external path is preferred for a material library: build ccx once, then emit a
`.so` per material and select it by name in the deck. It relies on CalculiX's
external-behaviour mechanism (`external.c`, `-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT`,
`-ldl`); ccx uppercases the material name, so the target names the library and the
`NCG_UMAT` symbol accordingly.

## Run it

```sh
# generate both materials (part of the calculix_check_driver gate build)
cmake --build build --target calculix_check_driver

# (A) compiled-in: build ccx from source WITH the material linked + run
examples/calculix/build_and_run.sh

# (B) external (dlopen): build ccx ONCE with external support, emit lib<MODEL>.so,
#     run the deck by @-name — no ccx recompile for future materials
examples/calculix/build_and_run_external.sh
```

Prereqs (Debian/Ubuntu): `gcc g++ gfortran make wget perl`, plus
`libarpack2-dev liblapack-dev libblas-dev`. SPOOLES is fetched and built by the
script (SPOOLES 2.2 predates C99, so the script compiles it with
`-fcommon -std=gnu89` and downgrades the gcc-14 implicit-int/-function errors).
`-lstdc++` is placed on the ccx link command, not in the Makefile's prerequisite
`LIBS`, because GNU make cannot resolve `-lstdc++` as a target file.

## Verified result

For `λ=1.3, μ=0.7, ε₁₁=0.01` the closed-form uniaxial-strain stress is
`S₁₁=(λ+2μ)·ε₁₁=0.027`, `S₂₂=S₃₃=λ·ε₁₁=0.013`. A real `ccx 2.22` run reproduces
this exactly at every integration point:

```
 stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL
         1   1  2.700000E-02  1.300000E-02  1.300000E-02  0.0  0.0  0.0
         ...   (all 8 integration points identical)
```

## What's tested where

* **CI lock (no external deps):** `tests/generated/calculix_check_driver.cpp`
  calls the emitted `umat_user_` exactly as CalculiX does for one integration
  point and checks `stre`/`stiff` against an independent isotropic oracle, plus a
  packing-order negative control. This is what proves the boundary on every
  build; the real `ccx` run above is the end-to-end confirmation.

## Scope

First cut is **stateless** materials: one strain input, one stress output, one
consistent tangent, plus scalar parameters (→ `elconloc`). Scalar inputs, state
variables (`xstate`), and rate/implicit forms are rejected at emit time — the
stateful path (e.g. J2 plasticity with equivalent-plastic-strain history) is a
tracked follow-up.
