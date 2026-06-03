# Critical Review — PR Stack #51 → #55

**Scope:** Pre-Phase-2 refactor PRs addressing the M2–M6 findings from `REVIEW-stack-42-47.md` (tracking: issue #48).
**Lenses (parallel, independent):** architect-reviewer · cpp-pro · code-reviewer.
**Earlier reviews accounted for:** `REVIEW-stack-42-47.md` findings excluded per agent briefs.

---

## Verdict

**No critical merge blockers.** Recommended merge order: **#51 → #52 → #53 → #54 → #55** as-is. Three small **pre-merge fixes** identified (~30 LOC, single commit on the M6 branch); four **Phase-2 follow-up items** worth a new tracking issue but not blocking.

The biggest single residual risk: the const/mutable boundary on `RecipeView` and `PassContext::model`. Phase 2's `TimeIntegrationPass` rewrites expressions — neither the current const-pointer view nor the const-only PassContext admit that cleanly. Worth resolving the design before Phase 2 starts (see PostFollowup item below).

---

## Critical

None.

---

## Major

### M51 — `(void) t2s_emit;` suppresses a future warning that exists for a reason
**File:** `include/numsim_codegen/recipe.h:640-641`
**Convergence:** cpp-pro M-1 + code-reviewer Nit.

```cpp
TensorToScalarCodeEmit &t2s_emit = *t2s_emit_storage;
(void) t2s_emit;
```

The reference is never used; the lambda at line 635-637 captures `t2s_emit_storage` directly. The `(void)` cast hides the `-Wunused-variable` warning that would otherwise signal a future maintainer who adds a direct `t2s_emit.apply(...)` call (bypassing the storage and any future caching/CSE layered on it).

**Fix:** delete both lines. The lambda is what wires the cycle; the named alias adds nothing.

### M52 — Test gap on rank-vs-`expected_rank` branch
**File:** `tests/PassFrameworkTest.cpp:281-318`, branch at `recipe.h:609-616`
**Source:** code-reviewer Major.

`TensorSpaceConsistencyPass::run` has two independent throw branches but only the symmetric-vs-skew one is exercised. A future refactor could silently regress the rank check.

**Fix:** add one test that declares `add_tensor_input("eps", 3, /*rank=*/4, roles::Strain)` (Strain has `expected_rank=2`) and asserts a throw mentioning `rank 4` + `expects rank 2`.

### M53 — TensorSpaceConsistencyPass error messages aren't actionable
**File:** `include/numsim_codegen/recipe.h:601-606`, `:611-616`
**Source:** code-reviewer Major.

The throws diagnose the contradiction but never tell the user *where the bad annotation came from* or *how to fix it*. In practice the `Skew` annotation lands via `cas::assume_skew()` (`numsim-cas/include/numsim_cas/tensor/tensor_assume.h:15`).

**Fix:** rewrite both messages to name the upstream cause. E.g. for symmetric-vs-skew:
> *"...has Skew perm. A symmetric role can't accept a skew-symmetric tensor. Check the cas `assume_skew()` / `assume_symmetric()` annotation on this symbol, or pick a Role whose `is_symmetric=false` (see the `roles::` catalogue in recipe.h)."*

### M54 — RecipeView const-pointer storage blocks the natural Phase-2 extension
**File:** `include/numsim_codegen/passes/recipe_view.h:67`
**Convergence:** architect Major + cpp-pro m-1 (Minor).

`m_model` is `ConstitutiveModel const *`. When `TimeIntegrationPass` lands (Phase 2), the natural moves don't compose:

- A sibling `MutableRecipeView` duplicates the five forwarder bodies.
- Adding `mutate_outputs(...)` to RecipeView casts away const on `m_model` → UB-bait.
- Plus: there's no nullability enforcement; `raw_model()` happily dereferences an out-of-scope model.

**Fix (Phase-2 prep, not strictly pre-merge):** either (a) change storage to `ConstitutiveModel *` (non-const) and gate mutability via a sibling typed view, or (b) store `std::variant<ConstitutiveModel const*, ConstitutiveModel*>`. Either preserves today's const-only surface while making the mutable variant clean.

Also: `raw_model()` and the delegate bodies should `assert(m_model != nullptr)` — minimal lifetime safeguard at zero runtime cost.

### M55 — unique_ptr cycle-break already duplicated; will be 4+ copies after Phase 2/3
**File:** `recipe.h:633-641` and `tests/TensorCodeEmitTest.cpp:668-674`
**Convergence:** code-reviewer Minor + architect Major.

Phase 2 (`TangentEmitPass`) + Phase 3 (`MoosePropertyEmitPass`, etc.) each need the same three-emitter wiring. Hand-coding the `unique_ptr` indirection three more times is one typo from UB.

**Fix (Phase-2 prep):** extract a `CodeEmitPipeline` aggregate that owns scalar/tensor/t2s emitters + closes the cycle internally. Each new emit pass becomes `EmitterPipeline ep{ctx, scalar}; ep.tensor().apply(...)`. Architect's M2 + M6 essentially merge into this recommendation — same fix resolves both "duplicated cycle-break" and "growing list of std::function ctor args".

### M56 — String-tag preconditions need typed constants before count crosses ~10
**File:** `include/numsim_codegen/passes/code_emit_pass.h:33-34`, all `passes/*_pass.h` `preconditions()` / `postconditions()` returns
**Source:** architect Major.

`CodeEmitPass::preconditions()` already lists 3 tags; Phase 2 adds `dt-lowered`, `kt-lowered`, `state-vars-wired`, `tangent-emitted`, probably `simplified` / `cse-applied`. At ~10 string literals, typos become silent (no compile-time check; runtime "unsatisfied precondition" with a string the user has to grep for).

**Fix (Phase-2 prep):** add `namespace pass_tags { inline constexpr std::string_view tensor_space_validated = "tensor-space-validated"; ... }` and require passes to reference the constants. Incidentally fixes cpp-pro n-2 (TensorSpaceConsistencyPass should depend on `identifiers-valid` too — easy to forget today, impossible to forget when referencing a named constant).

Also: `PassManager::run` should log the satisfied-set after each pass in debug builds, so dependency-debug doesn't require by-hand instrumentation.

---

## Minor

### m1 — `const_cast` in M6 test is unnecessary
**File:** `tests/PassFrameworkTest.cpp:306`
**Source:** cpp-pro m-2.

`eps` is a non-const `expression_holder<tensor_expression>`, so `eps.get<cas::tensor_expression>()` already resolves to the non-const overload returning `T&`. The `const_cast` casts `T&` to `T&` — a no-op the reader trips over.

**Fix:** `eps.get<cas::tensor_expression>().set_space(...)` directly. Drop the cast and the explanatory comment.

### m2 — TensorSpaceConsistencyPass O(N·M) symbol-to-decl join
**File:** `include/numsim_codegen/recipe.h:585-617`
**Source:** cpp-pro m-4.

Linear scan over `symbols()` for each entry in `tensor_symbol_map()`. Absolute cost is negligible (symbol counts are tens) but the structural smell is real: Phase 2 will add more passes that need the same join, and each will re-derive it.

**Fix (Phase-2 prep):** `SymbolValidationPass::run` builds an `std::unordered_map<std::string_view, SymbolDecl const*>` once and stows it on `PassContext`. Downstream passes consume it.

### m3 — `detail::` helpers in recipe.h are fine but located awkwardly
**File:** `include/numsim_codegen/recipe.h:382-422`
**Source:** cpp-pro m-3 + architect Minor.

`detail::param_decl` / `detail::output_decl` / `detail::tensor_arg_count` are `inline` free functions in `recipe.h`. ODR-correct today, but they're not reachable without pulling in `ConstitutiveModel` (and the full passes), and Phase 2 will add at least two more `render_*` free functions (state-var wiring, tangent assembly).

**Fix (Phase-2 timeframe):** at the *third* render free function (likely `render_tangent_function`), move the bodies into a `recipe_render_impl.h` included from recipe.h's tail. Not urgent.

Also: add a one-line `// DO NOT include recipe.h here — see comment above` near the top of `recipe_render.h` so a Phase-2 author doesn't "clean up the forward decls" and break the inclusion ordering.

### m4 — PassContext fields will multiply through Phase 2
**File:** `include/numsim_codegen/passes/pass.h:21-25`
**Source:** architect Major (downgraded to Minor — Phase-2 concern, not current-stack issue).

Three fields today; Phase 2 splits `compute_function_source` into `tangent_function_source`, `state_initial_source`, etc.

**Fix (when 5+ output slots accumulate):** replace the individual `std::optional<std::string>` fields with `std::map<std::string, std::string>` (string-keyed output bucket) or a typed `EmitArtifacts` aggregate.

### m5 — Tag name `tensor-space-validated` overpromises
**File:** `include/numsim_codegen/passes/tensor_space_consistency_pass.h:42`
**Source:** architect Minor.

Phase 2 will want a real "spaces inferred end-to-end" guarantee under that name. Today's pass only checks declared-level consistency.

**Fix:** rename to `tensor-space-declarations-checked` (or split into `tensor-rank-checked` + `tensor-space-checked`) so Phase 2 can land a stronger pass under a new tag without redefining the existing one.

### m6 — PR-stack length (debatable)
**Source:** code-reviewer Minor.

Five linear stacked PRs is at the edge of reviewer fatigue. M2/M3 are tightly coupled (both touch `TensorCodeEmit`); M4/M5 are mechanical refactors; M6 is a feature. **Trade-off:** flattening to 3 PRs (M2+M3, M4+M5, M6) trims review surface; the current 5-PR layout keeps each commit independently bisectable. I'd leave as-is at this point but flag for next time.

---

## Nit

- **n1 — `throwing_t2s_apply()` needs one more doc line** at `tests/TensorCodeEmitTest.cpp:32-39` naming the specific cas node type (`tensor_to_scalar_with_tensor_mul`) that triggers the t2s path, so a future test author knows which node type forces the unique_ptr indirection. (cpp-pro n-1)
- **n2 — `TensorSpaceConsistencyPass` should declare `"identifiers-valid"` as a precondition** alongside `"symbols-declared"`. PassManager satisfies it via `SymbolValidationPass` regardless, but the dependency graph is currently under-declared. (cpp-pro n-2)
- **n3 — `validate_role_attributes` (recipe.h:308-331) hardcodes 9 `check()` calls.** Fine today; flag for table-driven cleanup at the 10th role. (architect Nit)
- **n4 — `raw_model()` could grow a `[[deprecated]]`** if more than one site ever calls it. (code-reviewer Nit)

---

## Pre-merge fixup (bundled into one commit on `phase-1.2-m6-tensor-space-pass`)

The Majors and Minors small enough to land before review:

1. **M51:** drop `(void) t2s_emit;` and the named-but-unused reference at `recipe.h:640-641`.
2. **M52:** add `TensorSpaceConsistencyPass.RejectsRankMismatch` test.
3. **M53:** rewrite both `TensorSpaceConsistencyPass::run` throws to name `cas::assume_*()` / `roles::` catalogue.
4. **m1:** drop the unnecessary `const_cast` in `ThrowsOnSymmetricRoleWithSkewSpace`.
5. **n2:** add `"identifiers-valid"` to `TensorSpaceConsistencyPass::preconditions()`.
6. **n1:** one-line doc on `throwing_t2s_apply()`.

Total: ~30 LOC, single commit, runs clean against the existing 109 tests + adds 1 (so 110/110).

---

## Phase-2 follow-up (new tracking issue, NOT pre-merge)

Bundle into a successor to issue #48:

- **M54:** RecipeView storage strategy — pick (a) non-const pointer + sibling view OR (b) variant — before TimeIntegrationPass lands.
- **M55:** `CodeEmitPipeline` aggregate that owns the three emitters and closes the cycle internally. Also subsumes architect's "CrossDomainEmitter" recommendation for Phase 2's growing callback count.
- **M56:** `pass_tags::` constexpr namespace + PassManager debug-log of satisfied set.
- **m2:** `PassContext` carries a name → `SymbolDecl const*` map built once by SymbolValidationPass.
- **m3:** when the third render free function lands, move `recipe_render_*` bodies into a `_impl.h` included from recipe.h's tail.
- **m4:** `PassContext` output-slot bucket when ≥5 emit artifacts accumulate.
- **m5:** rename `tensor-space-validated` → `tensor-space-declarations-checked`.

---

## PR sequencing

**Merge order: `#51 → #52 → #53 → #54 → #55` as-is. No reordering or splits.**

**One bundled fixup commit on `phase-1.2-m6-tensor-space-pass`** for items M51, M52, M53, m1, n1, n2 before merge.

**Forward-look (single biggest residual risk):** `PassContext::model` is `RecipeView`-typed and const-only. Phase 2's `TimeIntegrationPass` mutates expressions. The const-pointer storage in RecipeView means the natural extensions all break or duplicate today's surface. **Resolve M54 before Phase 2 begins** — it's a <50 LOC change today and a several-hundred-LOC change once mutating passes exist.
