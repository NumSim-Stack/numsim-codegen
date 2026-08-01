#!/usr/bin/env bash
#
# CalculiX material test harness (issue #128).
#
# Loops over every material under tests/calculix/<material>/, GENERATES the
# material on the fly via its recipe.cpp (using numsim::codegen), compiles it to
# a shared library, runs each <case>.inp through a real external-enabled `ccx`,
# and diffs the resulting .dat against the committed gold/<case>.dat (produced by
# CalculiX's OWN built-in material — independent ground truth). Reports per-case
# PASS/FAIL; exits nonzero if any case fails.
#
# A material folder is:
#   <material>/recipe.cpp           program emitting <Model>_ext.cpp (the .so source)
#   <material>/<case>.inp           deck using the @-codegen material
#   <material>/gold/<case>.dat      committed reference output (ccx built-in)
#   <material>/gold/gen_gold_<case>.inp   the built-in deck that produced the gold
#
# Requirements (env):
#   CCX            path to a ccx built with -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT
#                  (see examples/calculix/build_and_run_external.sh to build one)
#   CODEGEN_BUILD  a configured numsim-codegen CMake build dir (default: ../../build)
#   TMECH_INC      tmech include dir (default: from CODEGEN_BUILD)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CODEGEN_BUILD="${CODEGEN_BUILD:-$REPO/build}"
TMECH_INC="${TMECH_INC:-$CODEGEN_BUILD/_deps/tmech-src/include}"
CAS_INC="$CODEGEN_BUILD/_deps/numsim_cas-src/include"
CODEGEN_LIB="$CODEGEN_BUILD/libnumsim_codegen.a"
CAS_LIB="$(find "$CODEGEN_BUILD" -name 'libNumSim_CAS.a' 2>/dev/null | head -1)"

: "${CCX:?set CCX to an external-enabled ccx binary (see examples/calculix/build_and_run_external.sh)}"
for f in "$CODEGEN_LIB" "$CAS_LIB" "$TMECH_INC/tmech/tmech.h" "$CAS_INC"; do
  [[ -e "$f" ]] || { echo "ERROR: missing '$f' — build numsim-codegen first (CODEGEN_BUILD=$CODEGEN_BUILD)"; exit 1; }
done

status=0
for matdir in "$HERE"/*/; do
  recipe="$matdir/recipe.cpp"
  [[ -f "$recipe" ]] || continue
  mat="$(basename "$matdir")"
  echo "── material: $mat ─────────────────────────────────────────────"
  work="$(mktemp -d)"

  # 1. generate the material source, 2. compile it to a shared library.
  g++ -std=c++23 -O2 "$recipe" -I "$REPO/include" -I "$CAS_INC" -I "$TMECH_INC" \
      "$CODEGEN_LIB" "$CAS_LIB" -o "$work/generate"
  "$work/generate" "$work/material_ext.cpp"

  for deck in "$matdir"/*.inp; do
    case_name="$(basename "$deck" .inp)"
    gold="$matdir/gold/$case_name.dat"
    [[ -f "$gold" ]] || { echo "  $case_name: no gold/$case_name.dat — skipped"; continue; }
    # library name = the @<LIB>_NCG_UMAT from the deck (uppercased by ccx).
    lib="$(grep -oE '@[A-Za-z0-9]+_NCG_UMAT' "$deck" | head -1 | sed 's/^@//; s/_NCG_UMAT$//')"
    g++ -std=c++23 -O2 -fPIC -shared "$work/material_ext.cpp" -I "$TMECH_INC" \
        -o "$work/lib$lib.so"
    cp "$deck" "$work/job.inp"
    ( cd "$work" && LD_LIBRARY_PATH="$work" "$CCX" job >ccx.log 2>&1 ) \
      || { echo "  $case_name: ccx FAILED"; tail -20 "$work/ccx.log"; status=1; continue; }
    if python3 "$HERE/compare_dat.py" "$gold" "$work/job.dat"; then
      echo "  $case_name: PASS"
    else
      echo "  $case_name: FAIL"; status=1
    fi
  done
  rm -rf "$work"
done

echo "─────────────────────────────────────────────────────────────"
[[ $status -eq 0 ]] && echo "ALL MATERIALS PASS" || echo "SOME MATERIALS FAILED"
exit $status
