# Critical Review — PR #57 (Issue #56 bundle)

**Scope:** P1 (RecipeView variant) + P2 (CodeEmitPipeline) + P3 (pass_tags) + P4 (symbol_lookup) + P6 (tag rename). P5 deferred per issue.
**Lenses (parallel, independent):** architect-reviewer · cpp-pro · code-reviewer.
**Earlier reviews accounted for:** `REVIEW-stack-42-47.md`, `REVIEW-stack-51-55.md`.

---

## Verdict

**One item must be fixed before merge.** The rest are minor or follow-up material.

The blocker: `PassContext::symbol_lookup` stores `SymbolDecl const *` into `m_symbols` (a `std::vector`). The "no mutation of m_symbols after SymbolValidationPass runs" invariant is implicit and unguarded. When Phase 2's first mutating pass calls `add_*` (synthesising e.g. `alpha_old`/`alpha_new` for state variables), every pointer in the map silently dangles. Both architect and cpp-pro flagged this independently. Cheap fix today; potentially silent-corruption bug after Phase 2 starts.

---

## Critical

### C1 — `PassContext::symbol_lookup` invalidation footgun
**File:** `include/numsim_codegen/passes/pass.h:31`, populated at `include/numsim_codegen/recipe.h:511-518`, consumed at `recipe.h:613-622`
**Convergence:** architect (Critical-latent) + cpp-pro (Major Phase-2 watch).

The map values are `SymbolDecl const *` into `model.symbols()`. Phase 1.2 is safe — all mutation happens before `SymbolValidationPass`. But Phase 2 mutating passes that synthesise symbols (state-variable splits, NCP slack variables) will `push_back` the underlying `std::vector<SymbolDecl>`, invalidating every stored pointer. No compile-time signal. No runtime check. The dangling-pointer access is silent UB.

**Fix (~10 LOC):** switch the map to `std::unordered_map<std::string, std::size_t>` storing indices into `m_symbols`. Consumers resolve via `model.symbols()[index]` at lookup time. One indirection per lookup; eliminates the dangling class entirely.

**Alternative (lighter, but doesn't actually fix):** document the lifetime contract on `PassContext::symbol_lookup` + add a debug-only `assert` in `PassManager::run` that `m_symbols.capacity()` is unchanged at end-of-run. Catches the bug but doesn't prevent it. Worse than the index approach.

**Must land before merge.**

---

## Major

### M1 — Stale doc-comment in `code_emit_pass.h`
**File:** `include/numsim_codegen/passes/code_emit_pass.h:23`
**Source:** code-reviewer M1.

The prose doc-comment still says `"tensor-space-validated"`. The `preconditions()` body (line 35) correctly references `pass_tags::tensor_space_declarations_checked`. Exactly the doc-rot pattern that P3's typed tags were meant to prevent — typed tags don't help when the prose still has the old name.

**Fix (1 line):** update the comment to reference the new tag.

### M2 — Variant overload-resolution lets caller silently mis-select const/mutable arm
**File:** `include/numsim_codegen/passes/recipe_view.h:54-57`
**Source:** architect 1.

A caller with a `ConstitutiveModel` lvalue gets the mutable arm whether they intended to or not; const-bound binding silently downgrades. The test fixture at `tests/PassFrameworkTest.cpp:115` already needs `static_cast<ConstitutiveModel const &>(model)` to force a const view — that workaround is the smell.

**Fix (~5 LOC):** add a `make_const_view(ConstitutiveModel &) -> RecipeView` free function (or static factory) that's explicit about const-intent. Today's tests + Phase 2 pipelines can opt in without forcing the cast pattern.

Not a strict merge blocker — the variant approach is correct for the short term — but the explicit factory closes the "how do I force const?" question.

### M3 — Duplicated `find + kind == Tensor` join in `TensorSpaceConsistencyPass::run`
**File:** `include/numsim_codegen/recipe.h:617-622`
**Source:** code-reviewer M3 + architect implicit (compounds with C1's fix).

Phase 2 will add ≥1 more validator that needs the same two-step (name lookup + kind discrimination). Adding the helper now is one method; deferring means N copies later.

**Fix (~8 LOC):** add `PassContext::find_tensor_symbol(std::string_view) -> SymbolDecl const *` (or free function in `passes/pass.h`). Naturally combines with C1's fix — the indexed lookup helper lives in one place.

### M4 — CodeEmitPipeline hard-codes 3 emitters; Phase 2's 4th will need redesign
**File:** `include/numsim_codegen/code_emit/code_emit_pipeline.h:37-80`
**Source:** architect 2.

The pipeline class names three specific emitter members. Adding a 4th (e.g. Phase 2's `DtLoweringEmit`) requires editing the class. Not closed-by-design in spirit, but closed-by-impl.

**Fix:** defer to Phase 2 design issue. Not a merge blocker today; flag in follow-up. When the 4th emitter actually lands, refactor to a small `EmitterRegistry` (the static-typing benefits of the current design should be preserved — e.g. `pipeline.get<TensorCodeEmit>()` rather than runtime polymorphism).

### M5 — `validate()` and `emit_compute_function()` build separate PassContexts; `symbol_lookup` rebuilt twice
**File:** `include/numsim_codegen/recipe.h:213-226` (`validate()`) + `:230-244` (`emit_compute_function()`)
**Source:** architect 8.

The pre-PR comment at line 213-218 documented this isolation as harmless. Post-P4 it's no longer harmless: any caller doing `model.validate(); model.emit_compute_function();` rebuilds the symbol_lookup map twice with different `PassContext` objects, and Phase 2 passes that cache state in `pctx.ctx` will lose it across the boundary.

**Fix:** either (a) `validate()` internally calls `emit_compute_function()` with an early-return flag, or (b) document loudly that `validate()` is fire-and-forget and its `PassContext` state is discarded. (b) is the minimum acceptable; (a) is the cleaner long-term answer but Phase 2 territory.

Flag as a Phase 2 design item, not a merge blocker.

---

## Minor

### m1 — `try_mutable_model()` nullptr-on-failure will duplicate throw-scaffolding across Phase 2 passes
**File:** `include/numsim_codegen/passes/recipe_view.h:89`
**Source:** architect 3.

Phase 2 will land 4-5 mutating passes, each writing the same `if (!p) throw "...";` boilerplate. Adding a `require_mutable_model(char const *pass_name) -> ConstitutiveModel&` helper now (one method, ~5 LOC) prevents the duplication before it starts.

**Fix:** add the helper to RecipeView. Pre-merge if convenient; otherwise file as a Phase 2 follow-up.

### m2 — Non-const ctor requires lvalue (won't bind rvalue) — undocumented
**File:** `include/numsim_codegen/passes/recipe_view.h:56`
**Source:** cpp-pro #2.

A `ConstitutiveModel &&` rvalue temporary can't bind to the non-const overload, so test fixtures that build a temporary will silently get a const view. Almost certainly correct behaviour (you shouldn't be mutating a temporary), but worth a comment so future contributors don't think it's a bug.

**Fix:** one-line doc on the non-const ctor.

### m3 — PassContext aggregate-init consistency
**File:** `include/numsim_codegen/recipe.h:221`, `:235`, several test sites
**Source:** cpp-pro #6 + #7.

Three-argument sites still compile (C++17 trailing value-init) but the four-argument explicit form `{model, ctx, nullopt, {}}` is more reviewer-friendly when PassContext gains a fifth field. Already done in the new `PopulatesPassContextSymbolLookup` test; should be done at all sites for consistency.

**Fix:** add explicit `{}` to all 4-or-fewer-argument `PassContext{...}` constructions.

### m4 — Naming inconsistency on tag suffix
**File:** `include/numsim_codegen/passes/pass_tags.h:42` vs `:48`
**Source:** code-reviewer N2.

`compute_function_emitted` reads as verb-past-tense ("the pass emitted X"); `tensor_space_declarations_checked` reads as state ("X is done"). Both name the postcondition but use different grammatical frames.

**Fix:** rename `tensor_space_declarations_checked` → `tensor_space_declarations_validated`. Propagates through three pass headers' `preconditions/postconditions`. One commit's worth of churn.

### m5 — Comment density audit
**File:** `pass_tags.h:1-29`, `recipe_view.h:1-22`, `recipe.h:611-616`
**Source:** code-reviewer N1.

Several blocks duplicate what's in the PR body, the issue, or sibling files. Specifically:
- `pass_tags.h:24-29` (P6 rationale): duplicates `tensor_space_consistency_pass.h:46-51`. Trim to one line.
- `recipe_view.h:14-22` (Phase 2 prospective examples): speculative; will date. Replace with a generic "see `try_mutable_model()`."
- `recipe.h:611-616` (P4 retrospective): git-blame material. Delete.

**Fix:** ~30 LOC of comment removed. Defer if you don't want commit churn.

### m6 — Recipe.h size projection
**File:** `include/numsim_codegen/recipe.h` (already ~700 LOC; projecting 1500+ by Phase 2)
**Source:** architect 8.

Every pass `run()` body lives in `recipe.h`. By the time Phase 2 lands 4 more passes, the file becomes the dominant compile-time cost.

**Fix (Phase 2 timeframe):** split pass `run()` bodies into `src/passes/*.cpp` with explicit `#include "recipe.h"` per file. Not a merge blocker.

### m7 — pass_tags ordering
**File:** `include/numsim_codegen/passes/pass_tags.h:33-49`
**Source:** architect 5.

Currently sorted alphabetically. Pipeline-phase ordering (validation → consistency → emit) makes the header read as a pipeline map.

**Fix (taste):** reorder. One commit.

---

## Nit

- **n1** — Lambda comment at `code_emit_pipeline.h:44-46` is accurate but easy to misread; sharpen to "lambda is stored, not invoked, here; `m_t2s` is non-null before any caller can reach the lambda." (cpp-pro #1)
- **n2** — Lambda move/copy comment at `code_emit_pipeline.h:58` is slightly imprecise: the `std::function`-stored lambda *can* be moved (it's just an `std::function`); what can't be moved is `CodeEmitPipeline` itself, because the lambda captures `this`. (cpp-pro #7)
- **n3** — Direct test for `CodeEmitPipeline` construction (~3 lines) would document the API entry point. P3's `pass_tags` doesn't need a constant-identity test (would test the language). (code-reviewer N4)
- **n4** — Commit granularity: P1/P2/P3/P4/P6 in one commit. Bisect granularity suffers if a regression lands in P4's hot-path consumer. Splitting into 5 commits in one PR (rebase work: ~5 min) helps future-you. (code-reviewer N6)
- **n5** — PR description's "Phase 2 readiness" framing slightly overstates: P1/P2/P4 are genuinely enabling; P3/P6 are hygiene. Reframe section header. (code-reviewer N7)
- **n6** — `[[nodiscard]]` on `raw_model()` returning `const &` is borderline noise. Keep or drop; doesn't matter. (architect 8.minor)
- **n7** — `CodeEmitPipeline::scalar() / tensor() / t2s()` are non-const member fns returning non-const refs from a logically-mutable pipeline. Consider providing const overloads to avoid bifurcating later. (architect 8.nit)

---

## Pre-merge fixup (bundled into one commit on `phase-1.2-issue-56-bundle`)

The Critical + small Majors:

1. **C1:** switch `symbol_lookup` value to `std::size_t` index; update SVP populator + TSP consumer.
2. **M1:** fix stale doc-comment in `code_emit_pass.h:23`.
3. **M3:** add `PassContext::find_tensor_symbol` helper consuming the new indexed lookup.
4. **m1:** add `RecipeView::require_mutable_model(char const *)` throwing helper.
5. **m3:** explicit `{}` on PassContext aggregate-init sites for forward consistency.

Total: ~50-60 LOC + ~10 LOC of tests (verify `require_mutable_model` throws; verify the indexed lookup survives a hypothetical capacity growth — easy to construct in a test).

## Phase 2 follow-up issue (deferred)

- **M4:** `CodeEmitPipeline` extensibility design (registry vs. visitor-inversion). File as Phase-2 design ticket.
- **M5:** `validate()` / `emit_compute_function()` context unification. Gate on first stateful validator landing.
- **m2:** doc the non-const ctor's lvalue requirement (could go in fixup too).
- **m4:** naming consistency rename (`tensor_space_declarations_validated`). Bundle with other tag additions in Phase 2.
- **m5/m6:** comment audit + `recipe.h` TU split. Phase 2 timeframe.
- **m7:** pass_tags pipeline-order. Bundle with first Phase 2 tag addition.
- **n3:** CodeEmitPipeline direct test. Trivial; bundle anywhere.

---

## Merge recommendation

**Apply the pre-merge fixup (C1 + M1 + M3 + m1 + m3), then merge.** ~70 LOC of code + tests. The remaining items belong in a Phase-2 follow-up issue or in Phase 2 work proper.

**Forward look (single biggest residual risk going into Phase 2):** the implicit `m_symbols` no-mutation invariant under `PassContext::symbol_lookup`. Even after C1 lands (switching to indices), Phase 2's first symbol-synthesising pass needs a deliberate policy on whether the lookup map is rebuilt or grown. File this as the first design question for Phase 2.
