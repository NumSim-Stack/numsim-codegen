#!/usr/bin/env python3
"""Compare two CalculiX .dat result files field-by-field.

Usage: compare_dat.py <gold.dat> <test.dat> [abs_tol] [rel_tol]

Used to validate a codegen'd material against CalculiX's OWN built-in material:
run the same deck with `*ELASTIC` (gold) and with the `@`-material (test), then
diff the numeric fields (stresses, strains, displacements). A pass proves the
generated material reproduces ccx's reference elasticity — independent ground
truth, not a self-derived oracle.

Exit code 0 = match within tolerance, 1 = mismatch / parse error.
"""
import sys


def data_values(path):
    """Flatten every purely-numeric data row into one list of floats.

    Header lines ("stresses (elem, ...", "for set EALL ...") contain letters and
    are skipped; data rows are whitespace-separated numbers. The two files share
    identical geometry + output requests, so their sequences align element-wise
    (the leading integer elem/node/ipnt indices included).
    """
    vals = []
    for line in open(path):
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            row = [float(t) for t in toks]
        except ValueError:
            continue  # header / label line
        vals.extend(row)
    return vals


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    gold_p, test_p = sys.argv[1], sys.argv[2]
    abs_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    rel_tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    gold, test = data_values(gold_p), data_values(test_p)
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
        ad = abs(g - t)
        rd = ad / max(abs(g), abs(t)) if max(abs(g), abs(t)) > 0 else 0.0
        worst_abs = max(worst_abs, ad)
        worst_rel = max(worst_rel, rd)
        if ad > abs_tol and rd > rel_tol:
            fails += 1
            if fails <= 5:
                print(f"  mismatch @field {i}: gold={g:.9g} test={t:.9g} "
                      f"(abs={ad:.2e} rel={rd:.2e})")

    print(f"{len(gold)} fields compared; max_abs={worst_abs:.2e} "
          f"max_rel={worst_rel:.2e}; tol abs={abs_tol:g} rel={rel_tol:g}")
    if fails:
        print(f"RESULT: FAIL ({fails} field(s) differ) — codegen material does "
              f"NOT match CalculiX built-in")
        return 1
    print("RESULT: PASS — codegen material reproduces CalculiX built-in elasticity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
