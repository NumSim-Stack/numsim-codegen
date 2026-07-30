#!/usr/bin/env bash
#
# Build CalculiX (ccx) from source with numsim-codegen's generated USER material
# linked in, then run the single-element uniaxial-strain deck and check the
# stress against the closed-form isotropic answer.
#
# The generated material is CalculiXUMATTarget's output (LinearElastic_umat.cpp):
# an `extern "C" void umat_user_(...)` matching CalculiX's native umat_user ABI.
# We drop the stock umat_user.f from the ccx build and link our object instead;
# ccx's umat_main.f dispatches to umat_user for any *MATERIAL named USER*.
#
# Prereqs (Debian/Ubuntu): gcc g++ gfortran make wget perl, plus system
# libarpack2-dev liblapack-dev libblas-dev. SPOOLES is fetched + built here.
#
# Usage:
#   examples/calculix/build_and_run.sh [WORKDIR]
# Env overrides:
#   UMAT_CPP=<path>   generated LinearElastic_umat.cpp (default: repo build tree)
#   TMECH_INC=<dir>   directory containing tmech/tmech.h (default: repo build tree)
#
set -euo pipefail

CCX_VER=2.22
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
INP="$HERE/uniaxial_c3d8.inp"
WORK="${1:-$REPO/build/ccx_from_source}"

# ── Locate the generated material + tmech headers ────────────────────────────
UMAT_CPP="${UMAT_CPP:-$REPO/build/tests/generated/LinearElastic_umat.cpp}"
if [[ ! -f "$UMAT_CPP" ]]; then
  echo "ERROR: generated umat not found at '$UMAT_CPP'."
  echo "       Build it first:  cmake --build build --target calculix_check_driver"
  echo "       or set UMAT_CPP=<path to LinearElastic_umat.cpp>."
  exit 1
fi
TMECH_INC="${TMECH_INC:-$REPO/build/_deps/tmech-src/include}"
if [[ ! -f "$TMECH_INC/tmech/tmech.h" ]]; then
  echo "ERROR: tmech headers not found under '$TMECH_INC'. Set TMECH_INC."
  exit 1
fi

echo "workdir : $WORK"
echo "umat    : $UMAT_CPP"
echo "tmech   : $TMECH_INC"
mkdir -p "$WORK"; cd "$WORK"

# ── Fetch sources ────────────────────────────────────────────────────────────
[[ -f "ccx_${CCX_VER}.src.tar.bz2" ]] || \
  wget -q "http://www.dhondt.de/ccx_${CCX_VER}.src.tar.bz2"
[[ -f "spooles.2.2.tgz" ]] || \
  wget -q "http://www.netlib.org/linalg/spooles/spooles.2.2.tgz"

[[ -d "CalculiX" ]]     || tar xjf "ccx_${CCX_VER}.src.tar.bz2"
[[ -d "SPOOLES.2.2" ]]  || { mkdir -p SPOOLES.2.2 && tar xzf spooles.2.2.tgz -C SPOOLES.2.2; }

# ── Build SPOOLES (serial; ccx's MT paths are #ifdef USE_MT and stay off) ────
if [[ ! -f "SPOOLES.2.2/spooles.a" ]]; then
  echo "building SPOOLES ..."
  pushd SPOOLES.2.2 >/dev/null
  # SPOOLES 2.2 predates C99: modern gcc needs -fcommon + K&R leniency.
  sed -i 's#^  CC = .*#  CC = gcc -fcommon -std=gnu89 -Wno-error=implicit-int -Wno-error=implicit-function-declaration -Wno-error=int-conversion -w#' Make.inc
  make lib >spooles_build.log 2>&1 || { echo "SPOOLES build FAILED:"; tail -25 spooles_build.log; exit 1; }
  popd >/dev/null
fi

# ── Patch the ccx Makefile ───────────────────────────────────────────────────
SRC="CalculiX/ccx_${CCX_VER}/src"
#  * use system ARPACK/LAPACK/BLAS instead of the bundled ARPACK static lib, and
#    link our generated umat object (the umat is C++/tmech);
#  * drop the stock umat_user.f so our umat_user_ is the one that resolves.
sed -i 's#^CC=cc#CC=gcc#' "$SRC/Makefile"
sed -i 's#\.\./\.\./\.\./ARPACK/libarpack_INTEL\.a#umat_codegen.o -larpack -llapack -lblas#' "$SRC/Makefile"
sed -i '/^umat_user\.f *\\$/d' "$SRC/Makefile.inc"
# libstdc++ (needed by the C++ umat object) goes on the link COMMAND, not the
# prerequisite LIBS: GNU make would try to resolve `-lstdc++` as a target file
# and fail (no libstdc++.so under /usr/lib without -dev), while the linker
# resolves it fine.
sed -i 's#ccx_2.22.a $(LIBS) -fopenmp#ccx_2.22.a $(LIBS) -fopenmp -lstdc++#' "$SRC/Makefile"

# ── Compile the generated umat_user_ (C++/tmech) into the ccx src dir ─────────
echo "compiling generated umat ..."
g++ -std=c++23 -O2 -c "$UMAT_CPP" -I"$TMECH_INC" -o "$SRC/umat_codegen.o"

# ── Build ccx ────────────────────────────────────────────────────────────────
echo "building ccx (this takes a while) ..."
pushd "$SRC" >/dev/null
make "ccx_${CCX_VER}" >ccx_build.log 2>&1 || { echo "ccx build FAILED:"; tail -30 ccx_build.log; exit 1; }
popd >/dev/null
CCX="$WORK/$SRC/ccx_${CCX_VER}"
[[ -x "$CCX" ]] || { echo "ERROR: ccx binary not produced"; exit 1; }
echo "ccx built: $CCX"

# ── Run the single-element job ───────────────────────────────────────────────
cp "$INP" job.inp
echo "running ccx ..."
"$CCX" job >ccx_run.log 2>&1 || { echo "ccx run FAILED:"; tail -30 ccx_run.log; exit 1; }

# ── Verify against the closed-form isotropic answer ──────────────────────────
# lambda=1.3, mu=0.7, eps11=0.01  ->  S11=0.027, S22=S33=0.013.
echo
echo "=== stresses (job.dat) ==="
awk '/stress/{f=1} f&&NF>=8{print} /displacements/{f=0}' job.dat | head -12 || true
echo
python3 - "$WORK/job.dat" <<'PY'
import re, sys
dat = open(sys.argv[1]).read()
# grab the stress table rows: elem intpnt sxx syy szz sxy sxz syz
rows = []
capture = False
for line in dat.splitlines():
    if 'stress' in line.lower():
        capture = True; continue
    if capture:
        nums = re.findall(r'[-+0-9.eE]+', line)
        if len(nums) >= 8:
            rows.append([float(x) for x in nums[-6:]])  # sxx..syz
        elif rows:
            break
if not rows:
    print("FAIL: no stress rows parsed"); sys.exit(1)
import statistics as st
sxx = st.mean(r[0] for r in rows); syy = st.mean(r[1] for r in rows)
szz = st.mean(r[2] for r in rows)
exp11, exp22 = 0.027, 0.013
ok = abs(sxx-exp11) < 1e-4 and abs(syy-exp22) < 1e-4 and abs(szz-exp22) < 1e-4
print(f"S11={sxx:.6f} (expect {exp11})  S22={syy:.6f} S33={szz:.6f} (expect {exp22})")
print("RESULT:", "PASS ✓  codegen'd material ran in CalculiX" if ok else "FAIL ✗")
sys.exit(0 if ok else 1)
PY
