#!/usr/bin/env bash
#
# Build CalculiX (ccx) ONCE from source with external-behaviour support — the
# prerequisite for the test harness (tests/run_harness.py needs $CCX). Fetches +
# builds SPOOLES and ccx, patching the ccx Makefile to enable the dlopen external
# material path (-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT + -ldl) and to use the
# system ARPACK/LAPACK/BLAS. Progress goes to stderr; the ccx binary path is
# printed to stdout, so:
#
#   CCX=$(tests/calculix/build_ccx.sh) python3 tests/run_harness.py
#
# Prereqs (Debian/Ubuntu): gcc g++ gfortran make wget perl, plus
# libarpack2-dev liblapack-dev libblas-dev.
#
set -euo pipefail

CCX_VER=2.22
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-$REPO/build/ccx_external}"

log() { echo "$@" >&2; }

log "workdir : $WORK"
mkdir -p "$WORK"; cd "$WORK"

# ── Fetch sources ────────────────────────────────────────────────────────────
[[ -f "ccx_${CCX_VER}.src.tar.bz2" ]] || wget -q "http://www.dhondt.de/ccx_${CCX_VER}.src.tar.bz2"
[[ -f "spooles.2.2.tgz" ]] || wget -q "http://www.netlib.org/linalg/spooles/spooles.2.2.tgz"
[[ -d "CalculiX" ]]    || tar xjf "ccx_${CCX_VER}.src.tar.bz2"
[[ -d "SPOOLES.2.2" ]] || { mkdir -p SPOOLES.2.2 && tar xzf spooles.2.2.tgz -C SPOOLES.2.2; }

# ── Build SPOOLES (serial; ccx's MT paths are #ifdef USE_MT and stay off) ────
if [[ ! -f "SPOOLES.2.2/spooles.a" ]]; then
  log "building SPOOLES ..."
  pushd SPOOLES.2.2 >/dev/null
  # SPOOLES 2.2 predates C99: modern gcc needs -fcommon + K&R leniency.
  sed -i 's#^  CC = .*#  CC = gcc -fcommon -std=gnu89 -Wno-error=implicit-int -Wno-error=implicit-function-declaration -Wno-error=int-conversion -w#' Make.inc
  make lib >spooles_build.log 2>&1 || { log "SPOOLES build FAILED:"; tail -25 spooles_build.log >&2; exit 1; }
  popd >/dev/null
fi

SRC="CalculiX/ccx_${CCX_VER}/src"
CCX="$WORK/$SRC/ccx_${CCX_VER}"

# assert each sed actually changed the file — a ccx version bump would otherwise
# silently no-op a patch (e.g. omit -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT).
patch_or_die() { local before; before="$(md5sum "$2")"; sed -i "$1" "$2"; \
  [[ "$(md5sum "$2")" != "$before" ]] || { log "ERROR: patch had no effect ($3) — ccx layout changed?"; exit 1; }; }

if [[ ! -x "$CCX" ]]; then
  log "building ccx with external-behaviour support (one-time) ..."
  patch_or_die 's#^CC=cc#CC=gcc#' "$SRC/Makefile" "CC=gcc"
  patch_or_die 's#-DMATRIXSTORAGE -DNETWORKOUT#-DMATRIXSTORAGE -DNETWORKOUT -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT#' "$SRC/Makefile" "enable external behaviours"
  patch_or_die 's#\.\./\.\./\.\./ARPACK/libarpack_INTEL\.a#-larpack -llapack -lblas#' "$SRC/Makefile" "ARPACK->system"
  patch_or_die "s#ccx_${CCX_VER}.a \$(LIBS) -fopenmp#ccx_${CCX_VER}.a \$(LIBS) -fopenmp -ldl#" "$SRC/Makefile" "add -ldl"
  pushd "$SRC" >/dev/null
  make "ccx_${CCX_VER}" >ccx_build.log 2>&1 || { log "ccx build FAILED:"; tail -30 ccx_build.log >&2; exit 1; }
  popd >/dev/null
fi
[[ -x "$CCX" ]] || { log "ERROR: ccx binary not produced"; exit 1; }
log "ccx (external-enabled): $CCX"

echo "$CCX"   # stdout: the binary path, for  CCX=$(build_ccx.sh)
