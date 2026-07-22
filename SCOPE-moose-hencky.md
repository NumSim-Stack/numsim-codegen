# Scope — MOOSE Hencky material via numsim-codegen (spectral stress + tangent)

## Goal
Generate a **Hencky (logarithmic-strain) hyperelastic material** end to end from a
numsim-codegen recipe: the stress is a spectral tensor function of the strain and
the consistent tangent is its exact derivative. This is the constitutive demo for
the CMAME case study (numsim-cas+codegen vs AceGen) and rides on the spectral
lowering from PR #107 (numsim-codegen #105) / numsim-cas #326.

**Acceptance:** the Hencky recipe generates via both the StandaloneCxx and MOOSE
targets; the standalone form compiles (tmech + the shipped spectral runtime
header, no numsim-cas) and its stress AND consistent tangent match finite
differences at ~100 deformation states including near-undeformed `C ≈ I`.

---

## Key finding from the source audit — this is SMALL
Two facts collapse most of the anticipated work:

1. **The explicit consistent tangent already works.**
   `ConstitutiveModel::add_algorithmic_tangent(name, of_output, wrt_input)`
   emits `∂σ/∂ε` via `cas::diff(tensor, tensor)` (Phase 3b-1, `elastic_tangent.cpp`).
   For Hencky, `add_algorithmic_tangent("dstress_dC", "stress", "C")` differentiates
   a stress that depends on `C` through `log(C)` → the symbolic spectral tangent →
   which **now lowers** (PR #107). No new pass, no `AlgorithmicTangentPass` change.

2. **A Hencky material is therefore "just a recipe"** — an explicit tensor output
   plus a tangent request. No state variables, no local Newton.

So the ONLY genuine codegen gap is the include wiring below.

---

## The one blocker (Slice 1): spectral runtime include wiring
No target emits `<numsim_codegen/runtime/spectral.h>`, so generated spectral
material references `numsim::codegen::rt::spectral_decompose` /
`divided_difference` / `confluent_derivative` with no declaration → won't compile.

Fix mirrors the existing linalg-include mechanism (`standalone_cxx.cpp`): emit the
body first, then key the include on **actual emitted usage** so it can't drift —

```cpp
const bool needs_spectral =
    body.find("numsim::codegen::rt::") != std::string::npos;
...
if (needs_spectral)
  os << "#include <numsim_codegen/runtime/spectral.h>\n";
```

Apply to both `StandaloneCxxTarget::emit` and `MooseMaterialTarget`'s header
emit. Build-integration note (documented, not code): the consuming material's
build must have numsim-codegen's `include/` on its path — the runtime header is
header-only and ships with codegen, same contract as tmech.

---

## Slice 2: the Hencky recipe
A new example + registry entry. Formulation (unambiguous):
- input `C` — right Cauchy–Green, symmetric rank-2, SPD (the driving strain).
- parameters `lambda`, `mu` (Lamé).
- Hencky strain `E = ½ log(C)` (uses the isotropic `log`).
- stress `σ = λ tr(E) I + 2μ E = ½λ tr(log C) I + μ log C`.
- `add_output("stress", σ, roles::Stress)`.
- `add_algorithmic_tangent("dstress_dC", "stress", "C")`.

Exercises the full surface: isotropic `log` (spectral value), `trace` (t2s),
identity, `t2s × I` (`tensor_to_scalar_with_tensor_mul`), scalar·tensor, add — and
the tangent differentiates through `log(C)` into eigenvalue / eigenprojection /
divided-difference nodes.

## Slice 3: numerical e2e gate
Reuse the slice-5 spectral pattern: generate the Hencky material via
StandaloneCxx, compile it (tmech + runtime header), and FD-verify:
- `dstress/dC : δC` vs central finite differences of `stress(C)` along `δC`,
- at distinct-eigenvalue C, a coalesced-pair C, and **C ≈ I** (near-undeformed —
  the FE-critical state where the divided-difference guard must stay finite).
- plus a string-assert (`MooseTargetTest` style) that the MOOSE `.h/.C` emit has
  the expected structure and the spectral include.

FD is reliable here (stress is smooth in C; unlike the tangent-at-exact-degeneracy
pitfall, we FD the stress, whose derivative the tangent *is*).

## Slice 4 (separate, env-dependent): real MOOSE packaging + demo
The generated MOOSE `Material` subclass (computeQpProperties, getParam, coupled
strain, RankTwo/RankFour ↔ tmech adaptors) — the string-level emit is in scope
here, but compiling/running inside an actual MOOSE app needs a MOOSE build and is
tracked as the case-study integration, likely outside this branch.

## Slice 5 (paper): AceGen comparison
Same Hencky model in AceGen; compare generated-code size, derivation effort, and
robustness at coalescence. Methods-paper deliverable, downstream of Slice 4.

---

## Out of scope / risks
- **Kinematic wrapping.** We emit `σ(C)` and `dσ/dC`; the finite-strain
  push-forward to MOOSE's spatial tangent (and the F→C map) is a modeling wrapper
  documented separately, not a codegen concern. The e2e verifies `dσ/dC` against
  FD of `σ(C)` — self-consistent regardless of the wrapping.
- **`C ≈ I` vs exact `C = I`.** At exact `C = I` all eigenvalues coincide and the
  tangent is the invariant `I_sym` (already proven in #107's e2e). Near `C = I`
  (tiny distinct eigenvalues) is the harder FD case — include it explicitly.
- **tmech-version invariant** (from #107) applies: the material's tmech must match
  what codegen targeted.
