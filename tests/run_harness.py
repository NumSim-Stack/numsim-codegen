#!/usr/bin/env python3
"""numsim-codegen material test harness (issue #128).

Validates codegen'd materials against a solver's OWN built-in reference
materials, run through the real solver, by diffing against committed gold files.
Lives at tests/ level so it can span multiple solver families — currently
`tests/calculix/`, with room for more subfolders later.

Each material folder carries a `tests.json` manifest. A test is always the same
operation — run the generated material via its input deck and compare to gold —
and each test defines its own tolerance parameter set (abs_tol, rel_tol):

    {
      "cases": [
        {"name": "uniaxial", "deck": "uniaxial.inp", "gold": "gold/uniaxial.dat",
         "gold_deck": "gold/gen_gold_uniaxial.inp"},
        {"name": "shear", "deck": "shear.inp", "gold": "gold/shear.dat",
         "gold_deck": "gold/gen_gold_shear.inp"}
      ]
    }

Per case: `deck` (the @-material input deck), `gold` (committed reference),
`gold_deck` (the built-in deck that produced the gold — run by --regen-gold), and
optional `abs_tol`/`rel_tol`. Tolerances default to DEFAULT_ABS_TOL /
DEFAULT_REL_TOL; a manifest-level `abs_tol`/`rel_tol` overrides them for a whole
material, and a per-case value overrides that.

The CalculiX family generates each material on the fly (compiles `recipe.cpp`
against numsim::codegen), builds one lib<LIB>.so per material, runs each case deck
through `ccx`, and diffs the .dat against the gold via compare_dat.py.

Env: CCX (required for the calculix family — a ccx built with
-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT), CODEGEN_BUILD (default ../build), and
optional overrides CXX, TMECH_INC, CAS_INC, CODEGEN_LIB, CAS_LIB, EIGEN_INC.

    python3 tests/run_harness.py                # run all families
    CCX=/path/to/ccx python3 tests/run_harness.py
    CCX=/path/to/ccx python3 tests/run_harness.py --regen-gold   # refresh gold from gold_deck

Exit 0 iff every discovered case passed AND at least one case ran; nonzero
otherwise (a discovered-but-unrunnable case is a failure, never a silent skip).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
LIB_RE = re.compile(r"@([A-Za-z0-9]+)_NCG_UMAT", re.IGNORECASE)

# Default comparison tolerances. Cases inherit these unless they set their own;
# a manifest-level "abs_tol"/"rel_tol" overrides these for a whole material.
DEFAULT_ABS_TOL = 1e-8
DEFAULT_REL_TOL = 1e-6


class Tally:
    def __init__(self) -> None:
        self.passed = self.failed = self.executed = 0

    def add(self, ok: bool) -> None:
        self.executed += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1


def sh(cmd: list[str], timeout: int = 900, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def have(exe: str) -> bool:
    return shutil.which(exe) is not None or Path(exe).is_file()


def ccx_run(ccx: str, work: Path, deck_text: str, so_dir: Path | None) -> str | None:
    """Run ccx on `deck_text` in `work`. Returns None on success, else an error."""
    (work / "job.inp").write_text(deck_text)
    env = dict(os.environ)
    if so_dir is not None:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(so_dir), os.environ.get("LD_LIBRARY_PATH")]))
    try:
        r = sh([ccx, "job"], cwd=work, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return "ccx timed out"
    out = r.stdout + r.stderr
    if r.returncode != 0:
        return f"ccx exit {r.returncode}:\n{out[-1500:]}"
    if "*ERROR" in out:  # ccx often prints *ERROR yet still exits 0
        return f"ccx reported *ERROR:\n{out[-1500:]}"
    if not (work / "job.dat").is_file():
        return "ccx wrote no job.dat"
    return None


def resolve_build() -> tuple[Path, list[Path], Path, list[Path], list[Path]]:
    build = Path(os.environ.get("CODEGEN_BUILD", REPO / "build"))
    tmech = Path(os.environ.get("TMECH_INC", build / "_deps/tmech-src/include"))
    cas_inc = Path(os.environ.get("CAS_INC", build / "_deps/numsim_cas-src/include"))
    gen_inc = [REPO / "include", cas_inc, tmech]
    so_inc = [tmech]
    eigen = os.environ.get("EIGEN_INC") or str(build / "_deps/eigen3-src")
    if Path(eigen).is_dir():
        so_inc.append(Path(eigen))
    codegen_lib = Path(os.environ.get("CODEGEN_LIB", build / "libnumsim_codegen.a"))
    cas_env = os.environ.get("CAS_LIB")
    cas_libs = [Path(cas_env)] if cas_env else sorted(build.glob("**/libNumSim_CAS.a"))
    return build, gen_inc, codegen_lib, cas_libs, so_inc


def compile_generator(recipe: Path, out: Path, cxx: str, inc: list[Path],
                      codegen_lib: Path, cas_lib: Path) -> str | None:
    cmd = [cxx, "-std=c++23", "-O2", str(recipe)]
    for d in inc:
        cmd += ["-I", str(d)]
    cmd += [str(codegen_lib), str(cas_lib), "-o", str(out)]
    r = sh(cmd)
    return None if r.returncode == 0 else r.stderr[-2000:]


def regen_gold(root: Path, ccx: str) -> int:
    """Re-run each case's gold_deck (built-in material, with `gold_constants`) →
    overwrite its committed gold. This is what locks gold provenance."""
    failed = 0
    for matdir in sorted(p for p in root.iterdir() if (p / "tests.json").is_file()):
        manifest = json.loads((matdir / "tests.json").read_text())
        for case in manifest["cases"]:
            gd = case.get("gold_deck")
            gold = case.get("gold")
            if not gd or not gold:
                print(f"  {matdir.name}/{case['name']}: no gold_deck/gold — skipped")
                continue
            gdeck = matdir / gd
            if not gdeck.is_file():
                print(f"  {matdir.name}/{case['name']}: gold_deck {gd} missing")
                failed += 1
                continue
            with tempfile.TemporaryDirectory() as tmp:
                err = ccx_run(ccx, Path(tmp), gdeck.read_text(), so_dir=None)
                if err:
                    print(f"  {matdir.name}/{case['name']}: regen FAILED: {err}")
                    failed += 1
                    continue
                (matdir / gold).write_text((Path(tmp) / "job.dat").read_text())
                print(f"  {matdir.name}/{case['name']}: regenerated {gold}")
    return failed


def run_calculix_family(root: Path, args: argparse.Namespace) -> Tally:
    t = Tally()
    ccx = os.environ.get("CCX")
    if not ccx:
        if args.require_ccx or args.regen_gold:
            print("calculix: ERROR — CCX not set (required)")
            t.failed += 1
        else:
            print("calculix: SKIP — set CCX to an external-enabled ccx binary "
                  "(or pass --require-ccx to make this a failure)")
        return t

    cxx = os.environ.get("CXX", "g++")
    for exe in (cxx, ccx):
        if not have(exe):
            print(f"calculix: ERROR — '{exe}' not found on PATH")
            t.failed += 1
            return t

    if args.regen_gold:
        t.failed += regen_gold(root, ccx)
        return t

    build, gen_inc, codegen_lib, cas_libs, so_inc = resolve_build()
    if not codegen_lib.is_file():
        print(f"calculix: ERROR — {codegen_lib} not found (build numsim-codegen; "
              f"CODEGEN_BUILD={build})")
        t.failed += 1
        return t
    if not cas_libs:
        print(f"calculix: ERROR — libNumSim_CAS.a not found under {build}")
        t.failed += 1
        return t
    if len(cas_libs) > 1:
        print(f"calculix: WARNING — multiple libNumSim_CAS.a, using {cas_libs[0]}")
    cas_lib = cas_libs[0]

    materials = sorted(p for p in root.iterdir() if (p / "tests.json").is_file())
    for matdir in materials:
        print(f"── material: {matdir.name} "
              f"{'─' * max(0, 44 - len(matdir.name))}")
        manifest = json.loads((matdir / "tests.json").read_text())
        recipe = matdir / manifest.get("recipe", "recipe.cpp")
        d_abs = manifest.get("abs_tol", DEFAULT_ABS_TOL)
        d_rel = manifest.get("rel_tol", DEFAULT_REL_TOL)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            if not recipe.is_file():
                print(f"  ERROR — {recipe} missing")
                for _ in manifest.get("cases", []):
                    t.add(False)
                continue
            err = compile_generator(recipe, work / "generate", cxx, gen_inc,
                                    codegen_lib, cas_lib)
            if err:
                print(f"  compile {recipe.name} FAILED:\n{err}")
                for _ in manifest.get("cases", []):
                    t.add(False)
                continue
            ext_cpp = work / "material_ext.cpp"
            g = sh([str(work / "generate"), str(ext_cpp)])
            if g.returncode:
                print(f"  generate FAILED:\n{g.stderr}")
                for _ in manifest.get("cases", []):
                    t.add(False)
                continue

            libs: dict[str, Path] = {}  # lib name -> built .so, one per material

            def ensure_lib(name: str) -> Path | None:
                if name in libs:
                    return libs[name]
                so = work / f"lib{name}.so"
                cmd = [cxx, "-std=c++23", "-O2", "-fPIC", "-shared", str(ext_cpp)]
                for d in so_inc:
                    cmd += ["-I", str(d)]
                cmd += ["-o", str(so)]
                r = sh(cmd)
                if r.returncode:
                    print(f"  build lib{name}.so FAILED:\n{r.stderr[-2000:]}")
                    return None
                libs[name] = so
                return so

            for case in manifest["cases"]:
                name = case["name"]
                deck = matdir / case["deck"]
                gold = matdir / case["gold"]
                abs_tol = case.get("abs_tol", d_abs)
                rel_tol = case.get("rel_tol", d_rel)
                if not deck.is_file():
                    print(f"  {name}: FAIL — deck {case['deck']} missing")
                    t.add(False)
                    continue
                if not gold.is_file():
                    print(f"  {name}: FAIL — gold {case['gold']} missing "
                          f"(run --regen-gold)")
                    t.add(False)
                    continue
                text = deck.read_text()
                m = LIB_RE.search(text)
                if not m:
                    print(f"  {name}: FAIL — deck has no @<LIB>_NCG_UMAT name")
                    t.add(False)
                    continue
                so = ensure_lib(m.group(1).upper())
                if so is None:
                    t.add(False)
                    continue
                err = ccx_run(ccx, work, text, so_dir=work)
                if err:
                    print(f"  {name}: FAIL — {err}")
                    t.add(False)
                    continue
                # a test is always: generated-via-input vs gold.
                c = sh([sys.executable, str(root / "compare_dat.py"),
                        str(gold), str(work / "job.dat"),
                        "--abs-tol", repr(abs_tol), "--rel-tol", repr(rel_tol)])
                ok = c.returncode == 0
                print(f"  {name}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    sys.stdout.write("    " +
                                     (c.stdout + c.stderr).strip().replace("\n", "\n    ")
                                     + "\n")
                t.add(ok)
    return t


FAMILIES = {"calculix": run_calculix_family}


def main() -> int:
    ap = argparse.ArgumentParser(description="numsim-codegen material test harness")
    ap.add_argument("--require-ccx", action="store_true",
                    help="fail (not skip) a family whose solver binary is unset")
    ap.add_argument("--regen-gold", action="store_true",
                    help="re-run each case's gold_deck and overwrite its gold file")
    args = ap.parse_args()

    total = Tally()
    ran_any = False
    for name, runner in FAMILIES.items():
        root = TESTS_DIR / name
        if not root.is_dir():
            continue
        ran_any = True
        print(f"═══ {name} family ═══════════════════════════════════════════")
        fam = runner(root, args)
        total.passed += fam.passed
        total.failed += fam.failed
        total.executed += fam.executed

    if not ran_any:
        print("no test families found under tests/")
        return 1
    if args.regen_gold:
        print("─────────────────────────────────────────────────────────────")
        print("gold regenerated" if total.failed == 0 else "gold regen had errors")
        return 1 if total.failed else 0

    print("─────────────────────────────────────────────────────────────")
    print(f"{total.passed} passed, {total.failed} failed, {total.executed} executed")
    if total.executed == 0:
        print("ERROR — zero cases executed (nothing was validated)")
        return 1
    return 1 if total.failed else 0


if __name__ == "__main__":
    sys.exit(main())
