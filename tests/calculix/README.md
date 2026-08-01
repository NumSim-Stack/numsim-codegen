# CalculiX material test harness (issue #128)

Golden-file validation of codegen'd CalculiX materials against CalculiX's OWN
built-in reference materials, run through a real `ccx`. Each material is
generated on the fly, run through `ccx`, and diffed field-by-field against a
committed gold `.dat` produced by CalculiX's built-in material — independent
ground truth, not a self-derived oracle.

> Status: **WIP skeleton** for issue #128. The first material (`linear_elastic`)
> works end-to-end; CMake/ctest integration and more materials are the issue's
> remaining scope.

## Layout

The harness lives one level up (`tests/run_harness.py`) so it can span multiple
solver families; `tests/calculix/` is the CalculiX family.

```
tests/
  run_harness.py          # Python harness: loops families/materials → generate → run → diff
  calculix/
    compare_dat.py        # field-by-field .dat diff (CalculiX-specific)
    <material>/
      recipe.cpp          # program that emits <Model>_ext.cpp (CalculiXExternalTarget)
      <case>.inp          # deck using the @-codegen material
      gold/
        <case>.dat        # COMMITTED reference output from ccx's built-in material
        gen_gold_<case>.inp # the built-in deck (*ELASTIC etc.) that produced the gold
```

Adding a material = a new folder with `recipe.cpp`, one or more `<case>.inp`, and
matching `gold/<case>.dat`. No harness changes.

## Run it

```sh
# 1. build numsim-codegen (provides libnumsim_codegen.a + the recipe deps)
cmake -S . -B build && cmake --build build --target numsim_codegen
# 2. build an external-enabled ccx once (see examples/calculix/build_and_run_external.sh)
# 3. run the harness (from the repo root)
CCX=/path/to/ccx_2.22 python3 tests/run_harness.py
```

Env: `CCX` (required, external-enabled), `CODEGEN_BUILD` (default `./build`),
`TMECH_INC` (default from the build).

## Gold provenance

Each `gold/<case>.dat` is produced once by running `gold/gen_gold_<case>.inp`
(CalculiX's built-in material) through `ccx` and committing the output. For
`linear_elastic` the built-in `*ELASTIC E=1.855, ν=0.325` is the exact equivalent
of the codegen material's `λ=1.3, μ=0.7`, and the codegen `.so` reproduces it
bit-for-bit.

## Remaining work (issue #128)

- CMake/ctest integration (build each `recipe.cpp` as a target; opt-in ctest
  gated on `$CCX`).
- A negative control (perturb constants → harness FAILs) proving the diff
  discriminates.
- More materials (`j2_plasticity`, …) once the stateful path lands.
- Migrate/retire the ad-hoc `examples/calculix/` scripts in favour of this suite.
