# CalculiX material test harness (issue #128)

Golden-file validation of codegen'd CalculiX materials against CalculiX's OWN
built-in reference materials, run through a real `ccx`. Each material is generated
on the fly, run through `ccx`, and diffed field-by-field against a committed gold
`.dat` produced by CalculiX's built-in material — independent ground truth, not a
self-derived oracle. A per-material `tests.json` manifest declares the cases.

> Status: **WIP** for issue #128. `linear_elastic` works end-to-end (uniaxial,
> shear, and a negative control). CMake/ctest integration and more materials are
> the issue's remaining scope.

## Layout

The harness lives one level up (`tests/run_harness.py`) so it can span multiple
solver families; `tests/calculix/` is the CalculiX family.

```
tests/
  run_harness.py          # Python harness: loops families/materials → generate → run → diff
  calculix/
    compare_dat.py        # field-by-field .dat diff (single source of truth)
    <material>/
      tests.json          # manifest: which cases to run + per-case config
      recipe.cpp          # program that emits <Model>_ext.cpp (CalculiXExternalTarget)
      <case>.inp          # deck using the @-codegen material
      gold/
        <case>.dat        # COMMITTED reference output from ccx's built-in material
        gen_gold_<case>.inp # the built-in deck (*ELASTIC etc.) that produced the gold
```

## The manifest (`tests.json`)

Each material folder declares its cases — which tests run, and each test's
parameter set (its tolerances). A test is always the same operation: run the
generated material via its input deck and compare to gold.

```json
{
  "cases": [
    {"name": "uniaxial", "deck": "uniaxial.inp", "gold": "gold/uniaxial.dat",
     "gold_deck": "gold/gen_gold_uniaxial.inp", "abs_tol": 1e-8, "rel_tol": 1e-6},
    {"name": "shear", "deck": "shear.inp", "gold": "gold/shear.dat",
     "gold_deck": "gold/gen_gold_shear.inp", "abs_tol": 1e-8, "rel_tol": 1e-6}
  ]
}
```

Per case: `deck` (the @-material input deck), `gold` (committed reference),
`gold_deck` (the built-in deck that produced the gold — run by `--regen-gold`),
and `abs_tol`/`rel_tol` (this test's tolerance parameter set; manifest-level
values, if given, are the fallback). A declared case that can't run (missing
deck/gold) is a **failure**, not a silent skip; a run that executes zero cases
exits nonzero.

Adding a material = a new folder with `tests.json` + `recipe.cpp` + decks + gold.
No harness changes.

## Run it

```sh
# 1. build numsim-codegen (provides libnumsim_codegen.a + the recipe deps)
cmake -S . -B build && cmake --build build --target numsim_codegen
# 2. build an external-enabled ccx once (see examples/calculix/build_and_run_external.sh)
# 3. run the harness (from the repo root)
CCX=/path/to/ccx_2.22 python3 tests/run_harness.py
```

Env: `CCX` (required; `--require-ccx` makes an unset `CCX` a failure instead of a
skip), `CODEGEN_BUILD` (default `./build`), and optional overrides `CXX`,
`TMECH_INC`, `CAS_INC`, `CODEGEN_LIB`, `CAS_LIB`, `EIGEN_INC`.

## Gold provenance

Each `gold/<case>.dat` comes from running `gold/gen_gold_<case>.inp` (CalculiX's
built-in material) through `ccx`. Regenerate + lock provenance with:

```sh
CCX=/path/to/ccx_2.22 python3 tests/run_harness.py --regen-gold
```

For `linear_elastic` the built-in `*ELASTIC E=1.855, ν=0.325` is the exact
equivalent of the codegen material's `λ=1.3, μ=0.7`, and the codegen `.so`
reproduces it bit-for-bit. (Near-zero entries like `6e-19` in the gold are ccx
solver noise, well under `abs_tol`.)

## Remaining work (issue #128)

- CMake/ctest integration (build each `recipe.cpp` as a target instead of the
  harness shelling `g++`; opt-in ctest gated on `$CCX`).
- Per-block/row-shape comparison in `compare_dat.py` (today it checks aggregate
  field count + per-field tolerance).
- More materials (`j2_plasticity`, …) once the stateful path lands; a shared
  `generate`-stage + per-family strategy once a 2nd solver family arrives.
- Retire the ad-hoc `examples/calculix/` scripts once the ccx-builder is promoted
  into the harness.
