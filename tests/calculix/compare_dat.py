#!/usr/bin/env python3
"""Compare two CalculiX .dat result files field-by-field.

Usage: compare_dat.py <gold.dat> <test.dat> [--abs-tol A] [--rel-tol R]

Validates a codegen'd material against CalculiX's OWN built-in material: run the
same deck with `*ELASTIC` (gold) and with the `@`-material (test), then diff the
numeric fields (stresses, strains, displacements). A pass proves the generated
material reproduces ccx's reference — independent ground truth, not a self-oracle.

Exit code 0 = match within tolerance, 1 = mismatch / non-finite / parse error.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# CalculiX/Fortran G-format can drop the 'E' on 3-digit exponents
# (e.g. "6.071532-319"); restore it before float() so the row isn't discarded.
_MISSING_E = re.compile(r"(\d)([+-]\d{2,3})$")


def data_values(path: Path) -> list[float]:
    """Flatten every purely-numeric data row into one list of floats.

    Header lines ("stresses (elem, ...", "... for set EALL ...") contain letters
    and are skipped; data rows are whitespace-separated numbers. Gold and test
    share identical geometry + output requests, so the two sequences align
    element-wise (leading integer elem/node/ipnt indices included). A field-COUNT
    mismatch (see main) catches gross layout divergence.
    """
    vals: list[float] = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if len(toks) < 2:
                continue
            row: list[float] = []
            for t in toks:
                try:
                    row.append(float(_MISSING_E.sub(r"\1E\2", t)))
                except ValueError:
                    row = []
                    break  # header/label line -> skip whole row
            vals.extend(row)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Field-by-field CalculiX .dat diff")
    ap.add_argument("gold")
    ap.add_argument("test")
    ap.add_argument("--abs-tol", type=float, default=1e-8)
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    a = ap.parse_args()

    try:
        gold = data_values(Path(a.gold))
        test = data_values(Path(a.test))
    except OSError as e:
        print(f"FAIL: cannot read {getattr(e, 'filename', None) or e}")
        return 1

    if not gold or not test:
        print(f"FAIL: no numeric data parsed (gold={len(gold)}, test={len(test)})")
        return 1
    if len(gold) != len(test):
        print(f"FAIL: field count differs (gold={len(gold)}, test={len(test)}) "
              f"— decks must share geometry + output requests")
        return 1

    worst_abs = worst_rel = 0.0
    fails = 0
    for i, (g, t) in enumerate(zip(gold, test)):
        if not (math.isfinite(g) and math.isfinite(t)):
            fails += 1
            if fails <= 5:
                print(f"  non-finite @field {i}: gold={g} test={t}")
            continue
        ad = abs(g - t)
        scale = max(abs(g), abs(t))
        rd = ad / scale if scale > 0 else 0.0
        worst_abs = max(worst_abs, ad)
        worst_rel = max(worst_rel, rd)
        if ad > a.abs_tol and rd > a.rel_tol:
            fails += 1
            if fails <= 5:
                print(f"  mismatch @field {i}: gold={g:.9g} test={t:.9g} "
                      f"(abs={ad:.2e} rel={rd:.2e})")

    print(f"{len(gold)} fields compared; max_abs={worst_abs:.2e} "
          f"max_rel={worst_rel:.2e}; tol abs={a.abs_tol:g} rel={a.rel_tol:g}")
    if fails:
        print(f"RESULT: FAIL ({fails} field(s) differ or non-finite)")
        return 1
    print("RESULT: PASS — codegen material reproduces CalculiX built-in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
