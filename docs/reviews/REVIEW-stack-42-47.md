# Critical Review — PR Stack #42 → #47

**Scope:** Phase 1.1 emit-stub completion (#42, #44, #45, #46) + Phase 1.2 pass framework (#47).
**Lenses (parallel, independent):** architect-reviewer · cpp-pro · code-reviewer.
**Earlier reviews accounted for:** `REVIEW.md` (520f99b multi-lens review) + cpp-pro pass at `2c87949` + 5 nits fixed in `84d149d`. None of those findings re-flagged here.

---

## Verdict

**No critical merge blockers.** Recommended merge order: **#42 → #44 → #45 → #46 → #47** as-is. One Major finding (M1 — keyword reject) should land *inside* #47 before merge OR immediately after as #47a, because once the pass framework is the single validation gate, leaving the gap open is worse than the pre-1.2 state.

The biggest forward-look risk is **M4 + M5 together** — the first mutating pass and the first non-CodeEmit emit pass will both rip the current architecture. Recommend a follow-up refactor PR before Phase 2 work begins.

---

## Critical

None.

---

## Major

### M1 — Identifier validator accepts C++ keywords; locale-sensitive char-class checks
**Files:** `include/numsim_codegen/passes/symbol_validation_pass.h:40-54`
**Convergence:** cpp-pro M1 + code-reviewer M2.

`is_valid_cxx_identifier` green-lights `class`, `if`, `template`, `auto`, `for`, `do`, `return` — correct for *form* (matches `[A-Za-z_][A-Za-z0-9_]*`), wrong for *usability*. The check also uses `std::isalpha` / `std::isalnum` which are locale-sensitive: Turkish locale fails `'I'`; locales that treat accented chars as alpha accept `"épsilon"`.

The pass **replaces** the pre-1.2 `validate()` body, so there is no second guard. A bad name slips through to the compile-check driver / consumer compiler, defeating the point of upstream validation.

**Mitigation:**
- Replace `std::isalpha` / `std::isalnum` with explicit ASCII range comparisons (`'A' <= c && c <= 'Z'`, etc.) — locale-immune.
- Hard-code a `static constexpr std::array` of the ~85 reserved C++ keywords; reject on `std::binary_search` hit after the char-class check.
- Add `RejectsRecipeWithKeywordName` test.

**Should land inside #47 before merge OR as #47a immediately after.**

### M2 — Duplicated projector preset matching across two sites
**Files:** `include/numsim_codegen/code_emit/tensor_code_emit.h:197-217` and `:481-507`
**Source:** code-reviewer M1.

Four `holds_alternative` ladders for (Symmetric/Skew × AnyTrace/Vol/Dev) appear identically in the `tensor_projector` materialise path (#44) and again in `projector_short_circuit_fn` (#45), with parallel return values (formula string vs. unary op name).

Risk: adding `P_harm` support or fixing a tag-name typo upstream needs lockstep edits to both sites. The projector-short-circuit test (`InnerProductWithPSymEmitsTmechSym`) won't catch drift in the materialised path.

**Mitigation:** extract a private helper that classifies `(perm, trace)` → `ProjectorKind` enum/optional, then dispatch both call sites off it. ~30-line follow-up.

### M3 — t2s callback wiring is a runtime trap, not a compile trap
**Files:** `include/numsim_codegen/code_emit/tensor_code_emit.h:382-394`, `include/numsim_codegen/recipe.h:527-528`
**Convergence:** code-reviewer M3 + architect m1.

The t2s callback gets wired only inside `CodeEmitPass::run`. Any future pass / test fixture / backend that constructs `TensorCodeEmit` directly silently lacks the callback until the first `tensor_to_scalar_with_tensor_mul` hits at runtime. The throw exists (`T2sWithTensorMulThrowsWhenCallbackUnset` covers it), but the throw *is* the failure mode being flagged.

**Mitigation:** take `T2sApply` in `TensorCodeEmit`'s constructor — moves the diagnostic from runtime throw to compile error at all call sites. One additional line at the call site (the lambda is already built).

### M4 — `PassContext` const-binds `ConstitutiveModel`; will force Phase 2 breakage
**Files:** `include/numsim_codegen/passes/pass.h:20-24`
**Convergence:** architect M1 + cpp-pro M2.

Phase 2's `TimeIntegrationPass` / `KuhnTuckerLoweringPass` need to rewrite expressions. Today's `ConstitutiveModel const &` leaves three bad options when Phase 2 arrives:
- (a) flip to `ConstitutiveModel &` — every pass that captured `pctx.model` by `auto const &` silently becomes mutable; friend boundaries shift.
- (b) parallel `ConstitutiveModel *mutable_model` slot — bifurcates the API.
- (c) introduce an explicit IR layer — large refactor.

Plus a subtler issue: `validate()` and `emit_compute_function()` each construct independent `PassContext{*this, CodeGenContext{}, ...}`. Currently harmless (SymbolValidationPass doesn't touch ctx), but a future pass that writes to `pctx.ctx` for downstream caching will silently see nothing if a client calls `validate()` separately.

**Mitigation:** insert a thin `RecipeView` handle that exposes the const surface today and gains mutable views later without breaking signatures. <100 LOC today; 500+ after passes proliferate. Also document the validate-vs-emit context isolation, or expose a single `run_pipeline()` entry-point.

### M5 — `friend class CodeEmitPass` is a one-way ratchet
**Files:** `include/numsim_codegen/recipe.h:293`
**Source:** architect M2.

Every future pass needing framing access (`TangentEmitPass`, `MoosePropertyEmitPass`, `StateVarEmitPass`, `AbaqusEmitPass`) must either be friended or route through `CodeEmitPass`. O(passes) friends on the user-facing recipe class.

The friend was justified as *"the rendering ABI is the framework's contract, not the user's surface"* — internally consistent, but Phase 3 (Abaqus / standalone / per-target emit) explicitly contradicts the single-emitter assumption.

**Mitigation:** extract `render_compute_function` + `param_decl` + `output_decl` + `tensor_arg_count` into a free function in `recipe_render.h` taking `ConstitutiveModel const &` through public accessors. Recipe stays a data class; rendering becomes its own seam.

### M6 — Phase 1.2 framework has zero non-trivial consumer
**Files:** conceptual — `include/numsim_codegen/passes/` directory as a whole
**Source:** architect M3.

`SymbolValidationPass` does identifier syntax + leaf-declaration checks. Both could have stayed inside `validate()` as free functions; the pass-framework right now is **zero observable behaviour change**. The design hasn't been stress-tested by a real pass that needs the context. The first non-trivial consumer (Phase 2 `TimeIntegrationPass`) is also the first redesign trigger.

Closely related: the deferred D3' `tensor_space` consistency validation, which this PR was supposed to deliver per the roadmap.

**Mitigation:** land a thin `TensorSpaceConsistencyPass` stub that walks `tensor_space` annotations on declared symbols (no expression-level inference yet). Exercises the postcondition wiring with a *second* validator and exposes any `PassContext` shape weaknesses before Phase 2 rewrite-passes commit you to the current layout.

---

## Minor

### m1 — Chained nested string expressions in `simple_outer_product` / `tensor_mul`
**Files:** `include/numsim_codegen/code_emit/tensor_code_emit.h:262-272`, `:331-338`
**Source:** cpp-pro m2.

For N children, the accumulated `acc` string is O(N)-deep nested function call, never registered as a temp until the final `register_temp`. Child strings are temps, but partial-product intermediates aren't. No CSE across two recipes that share a partial product. N≤4 in practice for elasticity. Readability/scalability note only — no UB, no string-ref invalidation.

### m2 — `tensor_mul` rank arithmetic can underflow `size_t`
**Files:** `include/numsim_codegen/code_emit/tensor_code_emit.h:337`
**Source:** cpp-pro m3.

`acc_rank = acc_rank + rhs_rank - 2` wraps if `rhs_rank < 2`. Guarded by upstream cas today; UBSAN would catch a future regression. Add a defensive `if (acc_rank < 2 || rhs_rank < 2) throw` for explicit-assumption documentation.

### m3 — `PassManager::run` uses `std::set<string_view>` of vector-returned views
**Files:** `include/numsim_codegen/passes/pass_manager.h:36`
**Convergence:** cpp-pro n2 + architect m2.

`postconditions()` returns `vector<string_view>` by value; views point at static literals in concrete passes today. A future pass returning views into temporary `std::string` members would dangle silently.

**Mitigation:** document "postcondition string_views must have static storage duration" in the `Pass` interface, or switch the set to `std::string` (one allocation per condition; negligible in a build step).

### m4 — Numerical correctness coverage gap (acknowledged)
**Files:** `tests/TensorCodeEmitTest.cpp` throughout
**Source:** code-reviewer m1.

String-containment tests verify structure, not math. Acknowledged interim tech debt — the compile-check driver picks it up when a recipe lands. Two specific risk areas worth filing tickets for now: (i) `tensor_mul` 3-tensor accumulator rank arithmetic at `tensor_code_emit.h:337`; (ii) `permute_indices_wrapper` rank-4 patterns (the test asserts string round-trip, not that `{3,4,1,2}` is the elasticity major/minor swap actually intended).

### m5 — `code_emit_pass.h` missing the include-order comment that `symbol_validation_pass.h` has
**Files:** `include/numsim_codegen/passes/code_emit_pass.h`
**Convergence:** cpp-pro m4 + code-reviewer n2.

`symbol_validation_pass.h:5-8` documents "recipe.h must be included elsewhere in the TU before `run()` is instantiated"; `code_emit_pass.h` has the same constraint but no comment. Per code-reviewer n2 the comment is also slightly misleading since `run()` is virtual and instantiated where it's *defined* (recipe.h). Reconcile: rewrite both as "definition lives in recipe.h."

### m6 — Two throw messages end with "open an issue" rather than naming the workaround
**Files:** `include/numsim_codegen/code_emit/tensor_code_emit.h:311-315` (`tensor_mul` coeff), `:218-223` (projector unsupported preset)
**Source:** code-reviewer m2.

Both throws punt to "file an issue" / "is not yet supported". Add the concrete workaround inline: *"rewrite as `tensor_scalar_mul`"* / *"use `P_devi()` instead of `P_harm`"*. The P_harm test already asserts P_dev appears in the message but the throw text doesn't actually suggest that substitution.

---

## Nit

- **n1 — Dead `NUMSIM_CODEGEN_TENSOR_STUB` macro** at `tensor_code_emit.h:140-150`. Zero remaining invocations. (code-reviewer n1)
- **n2 — `m_t2s_apply` default-init signaling** at `tensor_code_emit.h:513`. Add `// intentionally empty; wired by CodeEmitPass` or `{nullptr}`. (cpp-pro n1)
- **n3 — `is_valid_cxx_identifier` cohesion** at `symbol_validation_pass.h:40`. Could be a free function in a utilities header. (cpp-pro n3)
- **n4 — Dim=0 divide-by-zero risk in projector emit** at `tensor_code_emit.h:191`. Cheap pre-check. (code-reviewer n4)
- **n5 — Verbose `const std::string` locals in projector emit** at `tensor_code_emit.h:185-191`. Style only. (cpp-pro n4)

---

## PR sequencing

**Merge order: `#42 → #44 → #45 → #46 → #47` as-is. No splits or reordering.**

**Pre-merge ask:** fold **M1** into #47 before merge, or commit to #47a immediately after.

**Single follow-up refactor PR**, bundled, before any Phase 2 work begins:
- **M2** (DRY projector classification helper)
- **M3** (constructor-inject `T2sApply`)
- **M4** (`RecipeView` to decouple `PassContext` from `ConstitutiveModel` directly)
- **M5** (extract `render_compute_function` → `recipe_render.h`; drop the `friend`)
- **M6** (stub `TensorSpaceConsistencyPass`)

Minor and nit items can be batched separately or rolled into the refactor PR.

---

## Forward look

The biggest architectural risk carried into Phase 2 is **M4 + M5 together**. The first mutating pass (`TimeIntegrationPass`) will either break every existing pass signature or force the recipe to grow a second mutable surface; the first non-CodeEmit emit pass (`AbaqusEmitPass` / `TangentEmitPass`) will trigger the friend ratchet. Both are <100 LOC refactors today and 500+ after passes proliferate. The Phase 1.2 framework, as it stands, has not yet been stress-tested by a real consumer (M6) — landing the `TensorSpaceConsistencyPass` stub the roadmap actually called for would be the cheapest way to validate the design before Phase 2 hardens it.
