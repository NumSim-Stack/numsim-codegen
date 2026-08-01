#!/usr/bin/env python3
"""numsim-codegen material test harness (issue #128).

Validates codegen'd materials against a solver's OWN built-in reference
materials, run through the real solver, by diffing against committed gold files.
Lives at tests/ level so it can span multiple solver families — currently
`tests/calculix/`, with room for more subfolders (e.g. tests/abaqus/) later.

For CalculiX, each material is `tests/calculix/<material>/`:
    recipe.cpp           program emitting <Model>_ext.cpp (CalculiXExternalTarget)
    <case>.inp           deck using the @-codegen material
    gold/<case>.dat      committed reference output from ccx's built-in material
The harness generates the material on the fly, compiles it to lib<LIB>.so, runs
each <case>.inp through ccx, and diffs the .dat against gold/<case>.dat.

Env:
    CCX            path to a ccx built with -DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT
                   (required for the calculix family; see
                    examples/calculix/build_and_run_external.sh to build one)
    CODEGEN_BUILD  configured numsim-codegen CMake build dir (default: ../build)
    TMECH_INC      tmech include dir (default: derived from CODEGEN_BUILD)

Exit code 0 iff every discovered case passes; nonzero otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def run_calculix_family(root: Path) -> tuple[int, int]:
    """Run every material under tests/calculix/. Returns (passed, failed)."""
    ccx = os.environ.get("CCX")
    if not ccx:
        print("calculix: SKIP — set CCX to an external-enabled ccx binary "
              "(see examples/calculix/build_and_run_external.sh)")
        return (0, 0)

    build = Path(os.environ.get("CODEGEN_BUILD", REPO / "build"))
    tmech = Path(os.environ.get("TMECH_INC", build / "_deps/tmech-src/include"))
    cas_inc = build / "_deps/numsim_cas-src/include"
    codegen_lib = build / "libnumsim_codegen.a"
    cas_libs = list(build.glob("**/libNumSim_CAS.a"))
    compare = root / "compare_dat.py"

    for need in (codegen_lib, tmech / "tmech/tmech.h", cas_inc, compare):
        if not need.exists():
            print(f"calculix: ERROR — missing {need} "
                  f"(build numsim-codegen; CODEGEN_BUILD={build})")
            return (0, 1)
    if not cas_libs:
        print(f"calculix: ERROR — libNumSim_CAS.a not found under {build}")
        return (0, 1)
    cas_lib = cas_libs[0]

    passed = failed = 0
    for matdir in sorted(p for p in root.iterdir() if (p / "recipe.cpp").is_file()):
        print(f"── material: {matdir.name} "
              f"{'─' * max(0, 44 - len(matdir.name))}")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            gen = work / "generate"
            r = sh(["g++", "-std=c++23", "-O2", str(matdir / "recipe.cpp"),
                    "-I", str(REPO / "include"), "-I", str(cas_inc),
                    "-I", str(tmech), str(codegen_lib), str(cas_lib),
                    "-o", str(gen)])
            if r.returncode:
                print(f"  compile recipe.cpp FAILED:\n{r.stderr[-2000:]}")
                failed += 1
                continue
            ext_cpp = work / "material_ext.cpp"
            r = sh([str(gen), str(ext_cpp)])
            if r.returncode:
                print(f"  generate FAILED:\n{r.stderr}")
                failed += 1
                continue

            for deck in sorted(matdir.glob("*.inp")):
                case = deck.stem
                gold = matdir / "gold" / f"{case}.dat"
                if not gold.exists():
                    print(f"  {case}: no gold/{case}.dat — skipped")
                    continue
                m = re.search(r"@([A-Za-z0-9]+)_NCG_UMAT", deck.read_text())
                if not m:
                    print(f"  {case}: no @<LIB>_NCG_UMAT in deck — skipped")
                    continue
                lib = m.group(1)
                r = sh(["g++", "-std=c++23", "-O2", "-fPIC", "-shared",
                        str(ext_cpp), "-I", str(tmech),
                        "-o", str(work / f"lib{lib}.so")])
                if r.returncode:
                    print(f"  {case}: compile .so FAILED:\n{r.stderr[-2000:]}")
                    failed += 1
                    continue
                (work / "job.inp").write_text(deck.read_text())
                env = {**os.environ, "LD_LIBRARY_PATH": str(work)}
                r = sh([ccx, "job"], cwd=work, env=env)
                if r.returncode:
                    print(f"  {case}: ccx FAILED:\n{r.stdout[-1500:]}")
                    failed += 1
                    continue
                c = sh([sys.executable, str(compare), str(gold),
                        str(work / "job.dat")])
                sys.stdout.write("    " + c.stdout.replace("\n", "\n    ").rstrip()
                                 + "\n")
                if c.returncode == 0:
                    print(f"  {case}: PASS")
                    passed += 1
                else:
                    print(f"  {case}: FAIL")
                    failed += 1
    return (passed, failed)


FAMILIES = {"calculix": run_calculix_family}


def main() -> int:
    total_pass = total_fail = 0
    ran_any = False
    for name, runner in FAMILIES.items():
        root = TESTS_DIR / name
        if not root.is_dir():
            continue
        ran_any = True
        print(f"═══ {name} family ═══════════════════════════════════════════")
        p, f = runner(root)
        total_pass += p
        total_fail += f
    if not ran_any:
        print("no test families found under tests/")
        return 1
    print("─────────────────────────────────────────────────────────────")
    print(f"{total_pass} passed, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
