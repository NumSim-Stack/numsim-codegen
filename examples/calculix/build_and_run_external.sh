#!/usr/bin/env bash
#
# Build CalculiX (ccx) from source ONCE with external-behaviour support, then run
# the codegen'd material as a runtime-loaded shared library — NO ccx recompile
# per material. This is the preferred workflow: adding a new material only means
# emitting a new lib<MODEL>.so and referencing it by name in the input deck.
#
# How ccx finds it: a *MATERIAL, NAME=@<LIB>_<FUNC> deck entry makes ccx
# (which uppercases the name) dlopen lib<LIB>.so and dlsym <FUNC>. Our target
# emits an extern "C" NCG_UMAT, so the deck uses NAME=@<MODEL>_NCG_UMAT.
#
# Prereqs (Debian/Ubuntu): gcc g++ gfortran make wget perl, plus
# libarpack2-dev liblapack-dev libblas-dev. SPOOLES is fetched + built here.
#
# Usage:
#   examples/calculix/build_and_run_external.sh [WORKDIR]
# Env overrides:
#   EXT_CPP=<path>   generated LinearElastic_ext.cpp (default: repo build tree)
#   TMECH_INC=<dir>  directory containing tmech/tmech.h (default: repo build tree)
#
set -euo pipefail

CCX_VER=2.22
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
INP="$HERE/uniaxial_c3d8_external.inp"
WORK="${1:-$REPO/build/ccx_external}"

EXT_CPP="${EXT_CPP:-$REPO/build/tests/generated/LinearElastic_ext.cpp}"
if [[ ! -f "$EXT_CPP" ]]; then
  echo "ERROR: generated external source not found at '$EXT_CPP'."
  echo "       Build it first:  cmake --build build --target calculix_check_driver"
  exit 1
fi
TMECH_INC="${TMECH_INC:-$REPO/build/_deps/tmech-src/include}"
[[ -f "$TMECH_INC/tmech/tmech.h" ]] || { echo "ERROR: set TMECH_INC to the dir with tmech/tmech.h"; exit 1; }

echo "workdir : $WORK"
echo "ext src : $EXT_CPP"
mkdir -p "$WORK"; cd "$WORK"

# ── Fetch + build ccx ONCE, with external-behaviour support ───────────────────
[[ -f "ccx_${CCX_VER}.src.tar.bz2" ]] || wget -q "http://www.dhondt.de/ccx_${CCX_VER}.src.tar.bz2"
[[ -f "spooles.2.2.tgz" ]] || wget -q "http://www.netlib.org/linalg/spooles/spooles.2.2.tgz"
[[ -d "CalculiX" ]]    || tar xjf "ccx_${CCX_VER}.src.tar.bz2"
[[ -d "SPOOLES.2.2" ]] || { mkdir -p SPOOLES.2.2 && tar xzf spooles.2.2.tgz -C SPOOLES.2.2; }

if [[ ! -f "SPOOLES.2.2/spooles.a" ]]; then
  echo "building SPOOLES ..."
  pushd SPOOLES.2.2 >/dev/null
  sed -i 's#^  CC = .*#  CC = gcc -fcommon -std=gnu89 -Wno-error=implicit-int -Wno-error=implicit-function-declaration -Wno-error=int-conversion -w#' Make.inc
  make lib >spooles_build.log 2>&1 || { echo "SPOOLES build FAILED:"; tail -25 spooles_build.log; exit 1; }
  popd >/dev/null
fi

SRC="CalculiX/ccx_${CCX_VER}/src"
CCX="$WORK/$SRC/ccx_${CCX_VER}"

# assert each sed actually changed the file — a ccx version bump would otherwise
# silently no-op the patch (e.g. leave out -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT,
# giving a ccx that can't find the .so with a confusing error).
patch_or_die() { local before; before="$(md5sum "$2")"; sed -i "$1" "$2"; \
  [[ "$(md5sum "$2")" != "$before" ]] || { echo "ERROR: patch had no effect ($3) — ccx layout changed?"; exit 1; }; }

if [[ ! -x "$CCX" ]]; then
  echo "building ccx with external-behaviour support (one-time) ..."
  #  * enable the dlopen external-material path (external.c / call_external_umat_user.c);
  #  * link -ldl (dlopen) + system ARPACK/LAPACK/BLAS. The stock umat_user.f is kept.
  patch_or_die 's#^CC=cc#CC=gcc#' "$SRC/Makefile" "CC=gcc"
  patch_or_die 's#-DMATRIXSTORAGE -DNETWORKOUT#-DMATRIXSTORAGE -DNETWORKOUT -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT#' "$SRC/Makefile" "enable external behaviours"
  patch_or_die 's#\.\./\.\./\.\./ARPACK/libarpack_INTEL\.a#-larpack -llapack -lblas#' "$SRC/Makefile" "ARPACK->system"
  patch_or_die "s#ccx_${CCX_VER}.a \$(LIBS) -fopenmp#ccx_${CCX_VER}.a \$(LIBS) -fopenmp -ldl#" "$SRC/Makefile" "add -ldl"
  pushd "$SRC" >/dev/null
  make "ccx_${CCX_VER}" >ccx_build.log 2>&1 || { echo "ccx build FAILED:"; tail -30 ccx_build.log; exit 1; }
  popd >/dev/null
fi
echo "ccx (external-enabled): $CCX"

# ── Compile the codegen'd material to a shared library (NO ccx recompile) ─────
# ccx uppercases the deck name @<LIB>_NCG_UMAT → dlopen lib<LIB>.so, dlsym NCG_UMAT.
# LIB must match library_name(model) in calculix_external.cpp (uppercase alnum).
LIB="${LIB:-LINEARELASTIC}"
echo "compiling lib${LIB}.so from generated source ..."
g++ -std=c++23 -O2 -fPIC -shared "$EXT_CPP" -I"$TMECH_INC" -o "lib${LIB}.so"

# ── Run the deck (ccx dlopen's the .so at runtime) ───────────────────────────
cp "$INP" job.inp
echo "running ccx (external .so) ..."
LD_LIBRARY_PATH="$WORK" "$CCX" job >ccx_run.log 2>&1 || { echo "ccx run FAILED:"; tail -30 ccx_run.log; exit 1; }

echo
echo "=== stresses (job.dat) ==="
awk '/stress/{f=1} f&&NF>=8{print} /displacements/{f=0}' job.dat | head -9 || true
echo
python3 - "$WORK/job.dat" <<'PY'
import re, sys, statistics as st
rows=[]; cap=False
for line in open(sys.argv[1]):
    if 'stress' in line.lower(): cap=True; continue
    if cap:
        n=re.findall(r'[-+0-9.eE]+', line)
        if len(n)>=8: rows.append([float(x) for x in n[-6:]])
        elif rows: break
if not rows: print("FAIL: no stress rows"); sys.exit(1)
sxx=st.mean(r[0] for r in rows); syy=st.mean(r[1] for r in rows); szz=st.mean(r[2] for r in rows)
ok = abs(sxx-0.027)<1e-4 and abs(syy-0.013)<1e-4 and abs(szz-0.013)<1e-4
print(f"S11={sxx:.6f} (expect 0.027)  S22={syy:.6f} S33={szz:.6f} (expect 0.013)")
print("RESULT:", "PASS - codegen'd .so ran in CalculiX via dlopen (no recompile)" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
