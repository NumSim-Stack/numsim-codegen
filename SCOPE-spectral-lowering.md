# Spectral lowering scope — eigenvalue / eigenprojection / isotropic / divided-difference (#105)

## Goal
Lower the CAS **spectral node set** to generated tmech C++ so finite-strain,
spectrally-defined constitutive models (`log(A)`, `sqrt(A)`, `exp(A)` of a
symmetric tensor and their derivatives to arbitrary order) generate as MOOSE
and standalone materials. This is the codegen half of numsim-cas #326 (the
symbolic spectral tangent + divided-difference primitive).

**Acceptance:** a Hencky (log-strain) elasticity material — `σ = ℂ : log(b)` /
energy `ψ(log λ_i)` — generates, compiles, and its stress **and** consistent
tangent match the CAS runtime evaluator and finite differences within tolerance
at ~100 deformation states, **including coalesced eigenvalues** (undeformed
`b = I`), where the generated code must take the analytic-limit branch, not NaN.

---

## Current state (post Phase C, #90)
- Codegen is a per-domain visitor over the **same** CAS nodes, emitting
  **tmech-typed** C++ (`tmech::tensor<double,Dim,Rank>`, `tmech::inner_product`,
  `tmech::otimesu/otimesl`, `tmech::inv`, …). The local-Newton *solve* uses a
  fixed-size Eigen system (`linear_algebra_emitter.h`); tensor algebra is tmech.
- `CodeGenContext` interns temporaries: `emit_temporary(ptr, rhs)` binds a named
  temp; `find(ptr)` / `find_named(ptr)` return an existing name — this is the CSE
  layer. Nodes call `register_temp(&v, "tmech::…")`.
- **The spectral nodes are unhandled.** None of the five below has an
  `operator()` in the emitters; today they hit the `throw std::runtime_error("…
  not yet implemented …")` stub (`tensor_to_scalar_code_emit.h:181`,
  `tensor_code_emit.h`). So any spectral model fails loudly at codegen — no
  silent wrong output.

### What already lowers (so is NOT in scope)
The differentiated tangent numsim-cas emits is
`I : Σ_{i,j} ½·dd_f[i,j](A)·(otimesu(E_i,E_j)+otimesl(E_i,E_j))`. Of that tree:
- `otimesu`/`otimesl` → `outer_product_wrapper` — **already emitted**
  (`tensor_code_emit.h:286`, `→ tmech::outer_product<seq,seq>`).
- the `Σ`, scalar `½`, `t2s · tensor` product, and outer `I :` contraction are
  ordinary add / scalar-mul / inner_product nodes — **already emitted**.

So once the five leaf/spectral nodes lower, the entire tangent — **and its
higher derivatives**, since they are the same primitives with deeper
divided-difference indices — lowers with no further work. This is the whole
payoff of the symbolic-tangent rework: no bespoke rank-4 spectral kernel to
emit.

---

## The five node handlers (the actual surface)

| CAS node | domain | emits (tmech) | notes |
|---|---|---|---|
| `tensor_to_scalar_eigenvalue` (`eig_i(A)`) | t2s→scalar | `<decomp>.first[perm[i]]` | reads shared decomposition |
| `tensor_eigenprojection` (`E_i(A)`) | tensor | `tmech::otimes(v_i, v_i)` | `v_i = <decomp>.second[perm[i]]` |
| `tensor_eigenvector` (`n_i(A)`) | tensor | `<decomp>.second[perm[i]]` | rank-1; only if a model uses it directly |
| `tensor_to_scalar_divided_difference` (`dd_f[i…](A)`) | t2s→scalar | ternary guard tree (below) | the coincidence-safe scalar |
| `tensor_isotropic_function` (`f(A)`) | tensor | `Σ_i f(λ_i)·otimes(v_i,v_i)` | the *primal* value f(A) |

All five read one shared decomposition of their tensor argument.

### 1. Shared decomposition (the one new piece of infrastructure)
`eig_i`, `E_i`, `n_i`, `dd_f[…]` referencing the **same** tensor argument `A`
must share **one** `tmech::eigen_decomposition`. CSE keyed by node pointer does
*not* achieve this (each spectral node is a distinct pointer). Introduce a
decomposition cache keyed by the **emitted name of the argument** `A`:

```cpp
// emitted once per distinct argument, before first use:
auto _decomp_A = tmech::eigen_decomposition(tmech::sym(<A>)).decompose();
// _decomp_A.first  = eigenvalues (unsorted)
// _decomp_A.second = eigenvectors (unsorted)
```

Realise as a small `SpectralDecompositionCache` on `CodeGenContext` (or a
pre-pass over the recipe collecting spectral arguments), mirroring how
`leaf_collector` interns leaves. `emit_temporary` already gives the binding
mechanism; this only adds an argument-keyed lookup in front of it.

### 2. Sorted-index semantics (must match the CAS evaluator exactly)
CAS numbers eigenvalues **ascending** (`iso_detail::decompose_sorted`,
`eigen_decomposition.h:15`): `value(i)/basis(i)` refer to the i-th *smallest*.
tmech's `decompose()` is **unsorted**. The generated code must emit the same
ascending permutation `perm` so `eig_i`/`E_i` mean the same thing at runtime
and codegen. Emit a tiny sort of `_decomp_A.first` into an index array `perm`
(3 entries, dim 3), once per decomposition. Candidate: a one-line generated
helper `numsim_codegen::spectral_perm(vals)` in a runtime header shipped with
generated output, rather than inlining a bubble sort per material.

### 3. Divided-difference → nested ternary with the coincidence guard
The index multiset is **compile-time constant** in the node, so the confluent
recursion (`dd_range` in `tensor_data_isotropic.h`) unrolls at codegen time into
a straight-line ternary that transcribes the runtime tolerance verbatim:

```cpp
// dd_log[i,j]  (distinct pair or coincident) →
(std::abs(l_i - l_j) <= NUMSIM_DD_REL * std::max(std::abs(l_i), std::abs(l_j)))
    ? (1.0 / l_i)                                  // log'(λ) — analytic limit
    : (std::log(l_j) - std::log(l_i)) / (l_j - l_i)
```

- `l_i`, `l_j` are the shared-decomposition eigenvalue temps.
- `NUMSIM_DD_REL = std::sqrt(eps)` — **same constant** as `confluent_dd`.
- Triple/higher index (from 2nd+ derivatives) unrolls to a nested guard
  bottoming out in `f''(λ)/2!` etc., via the same `apply_fderiv` closed forms
  (`exp→exp`, `log→(−1)ⁿ⁻¹(n−1)!/xⁿ`, `sqrt→∏(½−j)x^{½−n}`) emitted inline.
- Emit per-`kind` `f`, `f'`, `f''` helpers once (`numsim_codegen_isotropic.h`
  runtime header) so the ternaries stay readable.

**This is the FE-critical branch:** at `b = I` (undeformed, `λ_i` all equal) the
generated `dd` takes the limit branch and the tangent is finite — matching the
runtime evaluator bit-for-bit. Silent-cap rule: if a model hits an unsupported
`kind`, throw at codegen, never emit a wrong closed form.

---

## Gating dependencies
- **numsim-cas #326 merged** (the dd node + symbolic tangent + facade). On the
  `326-isotropic-node` branch now; CI green. Bump the codegen CAS pin to the
  merge commit before starting.
- No new CAS upstream work — verified: every node in the emitted tangent is
  either one of the five above or already-lowered arithmetic.
- tmech is already a generated-code dependency (`eigen_decomposition`,
  `otimes`, `otimesu/otimesl`, `sym` all present) — no new third-party dep.

---

## Task breakdown (suggested PR slices)
1. **Shared-decomposition cache + sorted perm** — infra + the two runtime
   helper headers (`spectral_perm`, isotropic `f/f'/f''`). Golden: `eig_0(A)`
   and `E_0(A)` on one `A` emit a single decomposition. *(No model yet.)*
2. **`eigenvalue` + `eigenprojection` + `eigenvector` handlers** — read the
   cache; golden test vs CAS runtime on a diagonal and a full SPD `A`.
3. **`isotropic_function` value handler** — `Σ f(λ_i) E_i`; golden: generated
   `log(A)` vs runtime, incl. `exp(log(A)) ≈ A` round-trip.
4. **`divided_difference` handler** — the ternary guard tree; unit golden
   `dd_log[i,j]` distinct + coincident + triple, vs `confluent_dd`.
5. **End-to-end Hencky material** — generate, compile, FD-verify stress +
   consistent tangent at ~100 states incl. `b=I`. The acceptance gate. Wire the
   CMAME case-study comparison harness (numsim-cas+codegen vs AceGen) here.

## Out of scope
- Non-symmetric argument spectral (all constitutive use is `sym`).
- dim-2 plane-strain spectral (dim-3 first; dim-2 is a trivial follow-up).
- The MOOSE demo material *project wiring* (separate repo task once #5 lands).

## Risks / watch-items
- **Sort stability at coalescence:** the perm of equal eigenvalues is arbitrary,
  but the *value* tangent is perm-invariant there (this is exactly the
  `LogTangentCoincidentClosedForm` lesson from cas #326 — validate against the
  invariant-projector closed form, **not** FD, at exact degeneracy).
- **Repeated decompositions across outputs:** ensure the cache survives across
  the stress and tangent emission passes (same `CodeGenContext` lifetime) so `A`
  decomposes once per compute function, not once per referencing node.
- **`NUMSIM_DD_REL` drift:** the codegen tolerance and `confluent_dd`'s must be
  a single shared constant, or generated and runtime diverge at near-degeneracy.
  Pin it in a test that reads both.
