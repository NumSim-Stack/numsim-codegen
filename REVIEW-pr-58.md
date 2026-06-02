# Critical Review — PR #58 (Phase 2.1: StateVariable IR)

**Scope:** First Phase 2 PR. New `StateVariable` IR section + `add_*_state_variable` API + `SymbolDecl::Category` extension for current/old symbols.
**Lenses (parallel, independent):** architect-reviewer · cpp-pro · code-reviewer.
**Earlier reviews accounted for:** `REVIEW-stack-42-47.md`, `REVIEW-stack-51-55.md`, `REVIEW-pr-57.md`.

---

## Verdict

**Not a merge blocker, but three Major findings must be fixed before Phase 2.2 lands** — all three lenses converged on them:

1. **No duplicate-name protection.** Calling `add_scalar_state_variable("alpha", ...)` twice silently inserts two `SymbolDecl` entries; the second is shadowed by `unordered_map::emplace` in `symbol_lookup` but lives on as ghost state in `m_symbols` + `m_scalar_symbols`. Uncompilable generated code with no diagnostic.

2. **Hardcoded `_old` suffix collides with user-declared inputs.** `add_scalar_input("alpha_old", ...)` then `add_scalar_state_variable("alpha", ...)` (or reversed) silently creates two `alpha_old` SymbolDecls with different Categories. Same silent-corruption mode.

3. **The `_old` convention is implicit.** Eight `name + "_old"` concatenations across the two `add_*_state_variable` methods. Phase 2.2's `TimeIntegrationPass` will copy-paste the same string concat to map state vars to their paired symbols. The convention exists only in code, not as a named constant.

The fix is small (~30 LOC + ~3 new tests) and prevents Phase 2.2 from inheriting brittle contracts. Pre-merge fixup recommended.

---

## Critical

None.

---

## Major

### M1 — No duplicate-name protection in `add_*_state_variable`
**File:** `include/numsim_codegen/recipe.h:254-308`
**Convergence:** code-reviewer M1 + cpp-pro Major #2 + architect Q1/Q2.

`add_scalar_state_variable("alpha", ...)` called twice pushes two `SymbolDecl` entries with the same name into `m_symbols` + two distinct cas leaves into `m_scalar_symbols`. `SymbolValidationPass::run` uses `unordered_map::emplace` (no overwrite) so `symbol_lookup` retains the FIRST entry; downstream `find_tensor_symbol` resolves to that one, while the duplicate cas leaf is still referenceable from caller code. `render_compute_function` emits a duplicate parameter name → uncompilable C++ with no diagnostic before the user's compiler stage.

`add_scalar_input` / `add_tensor_input` / `add_parameter` have the same gap, but Phase 2.1 widens the blast radius (two symbols per state-var add, plus the `_old` collision below).

**Fix:** add a private `assert_symbol_name_available(std::string_view name)` helper that scans `m_symbols` and throws `std::runtime_error` if the name is taken. Call from all four add methods (state-var calls pass both the bare name and the `_old`-suffixed name to the check).

### M2 — `_old` suffix collides with user-declared inputs
**File:** `include/numsim_codegen/recipe.h:259, 268-273, 287, 296-301`
**Convergence:** code-reviewer M2 + cpp-pro Major #2 + architect Q1/Q2.

`add_scalar_input("alpha_old", ...)` followed by `add_scalar_state_variable("alpha", ...)` produces two `alpha_old` SymbolDecls with different Categories. The collision is non-obvious — the user never typed "alpha_old" against a state variable. Same silent-corruption mode as M1.

**Fix:** the same `assert_symbol_name_available` helper from M1, called against both `name` and `name + state_variable_old_suffix` in `add_*_state_variable`. Also: define a `state_variable_old_suffix` constant somewhere reachable (recipe.h or new `state_variable_naming.h`) so M3's lift can use it.

### M3 — `_old` suffix is a hardcoded literal repeated 8 times
**File:** `include/numsim_codegen/recipe.h:259, 268-273, 287, 296-301` and (post-merge) Phase 2.2 `TimeIntegrationPass`
**Source:** architect Q1/Q2 + cpp-pro Major #2 (compound).

The string `"_old"` appears eight times across `add_*_state_variable`. Phase 2.2's `TimeIntegrationPass` will need to look up the paired symbol by appending `"_old"` to a state variable's name — it'll copy-paste the same convention in another TU. The contract lives in code, not in a named symbol.

**Fix:** lift to `inline constexpr std::string_view state_variable_old_suffix = "_old"` (in recipe.h near the `StateVariable` struct, or a small dedicated header). Reference it from all 8 sites. Phase 2.2 imports the same constant.

**Stronger fix (architect Q1):** store `current_symbol_idx` / `old_symbol_idx` on `StateVariable` populated at registration time. Eliminates the implicit name-based mapping entirely; Phase 2.2 looks up the paired symbol via index, not string suffix. ~10 LOC extra; worth doing if we're touching this code anyway.

### M4 — Test coverage gap on the failure modes M1/M2 create
**File:** `tests/StateVariableTest.cpp`
**Source:** code-reviewer m1 (escalated because the bugs are silent).

No test exercises:
- Duplicate `add_scalar_state_variable` calls (M1)
- `add_scalar_input` then `add_scalar_state_variable` with name collision (M2)
- Reverse order (state-var then conflicting input)
- State-vars-only recipe with no outputs (does the empty-outputs codepath in `CodeEmitPass::run` work?)
- Mixed input + state variable in a single output expression (only parameter+state-var exercised today)

**Fix:** add 5 tests after M1+M2+M3 fixes land. Three for the collision throws, one for the no-outputs case, one for the mixed input+state-var recipe.

### M5 — `render_compute_function` treats `StateVariableCurrent` / `StateVariableOld` as ordinary `const` input parameters
**File:** `include/numsim_codegen/recipe.h` `render_compute_function` body + `detail::param_decl`
**Source:** cpp-pro Major #7.

Today both symbol categories appear in the generated function signature as `const` value parameters. Semantically:
- `StateVariableOld`: correct (framework supplies this from prior step)
- `StateVariableCurrent`: incorrect (this is what we solve for / write)

For Phase 2.1 this is acceptable interim behavior — recipes today pre-supply both at the call site. But Phase 2.2's `TimeIntegrationPass` + Phase 2.6's Newton-lowering will need `StateVariableCurrent` to be an out-param or omitted from the public signature (replaced by `declareProperty<>` writes inside the function body).

**Fix:** not a Phase 2.1 change; document the current behavior in a test (`StateVarUsableInOutputExpression` should assert that BOTH `<name>` and `<name>_old` appear as parameters today, with a `TODO(phase-2.2)` comment for the future shape). Pre-merge add of a single assertion.

---

## Minor

### m1 — `m_state_variables` and `m_symbols` are dual sources of truth
**File:** `include/numsim_codegen/recipe.h:478` (the new vector) + the StateVariable registration paths
**Source:** architect Q5.

Adding a state variable writes to both `m_state_variables` and `m_symbols` atomically today. Phase 2.2 mutating passes that want to replace `initial_value` will touch only the former; passes rewriting cas expressions touch the latter via `m_scalar_symbols`. No invariant statement says they must stay aligned.

**Fix (Phase-2.2 prep):** add a private `validate_state_variable_symbol_alignment()` invariant check; call from `validate()`. ~20 LOC, catches "I forgot to update X" bugs.

### m2 — Missing `state_variables_declared` postcondition tag
**File:** `include/numsim_codegen/passes/pass_tags.h:33-48`
**Source:** architect Q6.

Phase 2.2's `TimeIntegrationPass` materially depends on state variables existing. The pass framework's tag system has no entry for "state variables have been registered." Adding now (while the tag list is short) prevents Phase 2.2 from skipping a missing dependency.

**Fix:** add `inline constexpr std::string_view state_variables_declared = "state-variables-declared"`. `SymbolValidationPass` advertises it unconditionally. Phase 2.2's pass declares it a precondition. Two lines.

### m3 — Missing `find_state_variable_by_name` lookup helper
**File:** `include/numsim_codegen/recipe.h` (next to `find_tensor_symbol`)
**Source:** architect Q9.

Phase 2.2's `TimeIntegrationPass` will need to resolve `Dt(α)` → its `StateVariable` record. Without a helper, it'll do a linear scan over `model.state_variables()` per lookup.

**Fix:** add `find_state_variable_by_name(PassContext const &, std::string_view) -> StateVariable const*` mirroring `find_tensor_symbol`. ~5 LOC.

### m4 — Handle structs nested inside `ConstitutiveModel`
**File:** `include/numsim_codegen/recipe.h:245-252`
**Source:** cpp-pro Minor #4.

`ConstitutiveModel::ScalarStateVariableHandle` is verbose for Phase 2.2 passes that want to pass handles around. `auto` deduction hides the spelling at call sites, but helper-function authors will need the full name.

**Fix (Phase-2.2 timeframe):** lift to free types in `numsim::codegen`. Not blocking now.

### m5 — Redundant doc-string copies + missing `std::move`
**File:** `include/numsim_codegen/recipe.h:263-265, 270-273, 275-276` (and tensor counterparts)
**Source:** cpp-pro Minor #5.

`current_decl.doc = doc` (copy 1), `old_decl.doc = doc` (copy 2), `std::move(doc)` into `StateVariable` (move into copy 3 target). Additionally `m_symbols.push_back(current_decl)` (line 265) doesn't move — gratuitous copy. The pre-PR `add_scalar_input` correctly moves its single `decl`.

**Fix:** `m_symbols.push_back(std::move(current_decl))` for the current_decl push (only the LAST use of `current_decl` should move; here both `current_decl` and `old_decl` are pushed and then discarded). Apply same to tensor counterpart.

### m6 — `using Kind = SymbolDecl::Kind` provides no benefit
**File:** `include/numsim_codegen/recipe.h:149`
**Source:** cpp-pro Minor #6.

The test at `StateVariableTest.cpp:100` uses `SymbolDecl::Kind::Scalar` directly, bypassing the alias. The alias adds a layer that nothing consumes.

**Fix:** delete the `using` and type the field `SymbolDecl::Kind` directly. One-line change.

### m7 — Variant arm not enforced against `Kind` field
**File:** `include/numsim_codegen/recipe.h:148-160`
**Source:** cpp-pro Minor #3 + architect Minor Q7.

`StateVariable` has a `Kind` field AND a `std::variant<scalar_holder, tensor_holder> initial_value`. Today the `add_*_state_variable` method signatures enforce alignment; a future internal codepath that constructs `StateVariable` directly (e.g. Phase 2.2 synthesizing a state var during lowering) could mismatch.

**Fix:** add `assert(std::holds_alternative<scalar_holder>(sv.initial_value))` (etc.) inside both `add_*_state_variable` methods, plus an invariant check in `SymbolValidationPass`. Defer the invariant check to Phase 2.2; the assert is one line.

---

## Nit

- **n1** — Mermaid label `Dt(α.current − α.previous) = ...` at `docs/workflow.md:35` misrepresents what Phase 2.2 lowers (user writes `Dt(α) = rhs`; pass rewrites). Reword. (architect)
- **n2** — Phase-status table at `docs/workflow.md:236-245` will age poorly; consider linking GitHub Milestones / epic #28 instead. (code-reviewer)
- **n3** — `roles::History` deprecation comment is 10 lines; could be 3 + a tracking-issue TODO. (architect + code-reviewer)
- **n4** — Comment density on `add_*_state_variable` and `StateVariable` struct preambles is high (~11-13 line blocks); justified for a freshly-introduced API but flag for cleanup once Phase 2.7 ships. (code-reviewer m2)
- **n5** — `Handle`'s `current` / `previous` field names read more naturally than `old` / `new` in evolution-equation code. Keep. (code-reviewer n2)
- **n6** — `StateVariable` doesn't carry an optional `Role` — Phase 3 might want semantic tagging (e.g. `roles::Hardening`) for backend-specific storage decisions. Not urgent. (architect Q9)
- **n7** — `Handle` has no equality / name accessor; Phase 2.2 lowering may want to ask "is this leaf the same as `α.current`?" — cheap to add. (architect Q9)
- **n8** — `recipe.h` is now ~750 lines, projected 1500+ by Phase 2.7. Already flagged in REVIEW-pr-57.md; not new debt.

---

## Pre-merge fixup (bundled into one commit on `phase-2.1-state-variable`)

The Majors plus the easy Minor:

1. **M3:** lift `state_variable_old_suffix` to a constant; reference from all 8 sites.
2. **M1+M2:** add `assert_symbol_name_available(name)` helper on ConstitutiveModel; call from `add_*_state_variable` against `name` AND `name + state_variable_old_suffix`; call from `add_scalar_input` / `add_tensor_input` / `add_parameter` against just `name`. Three throws should fire.
3. **M4:** add 5 tests for the new throw paths and the no-outputs / mixed-input cases.
4. **M5:** add one assertion to `StateVarUsableInOutputExpression` documenting that both `<name>` and `<name>_old` appear as function parameters today (with `TODO(phase-2.2)`).
5. **m5:** add `std::move` where the copies should be moves.
6. **m6:** delete the `using Kind = SymbolDecl::Kind;` alias.

Total: ~80 LOC code + ~70 LOC tests. Phase 2.2 starts from a clean slate.

## Phase-2-prep follow-up issue (deferred)

Bundle these into a new tracking issue (or extend issue #28's epic):

- **M3 stronger:** store `current_symbol_idx`/`old_symbol_idx` on `StateVariable`.
- **m1:** `validate_state_variable_symbol_alignment` invariant check.
- **m2:** `state_variables_declared` postcondition tag in `pass_tags.h`.
- **m3:** `find_state_variable_by_name` lookup helper.
- **m4:** lift `Handle` structs out of `ConstitutiveModel` to free types.
- **m7:** Variant-vs-Kind invariant check pass.
- **n6/n7:** Handle ergonomics — name accessor, optional Role tagging.
- The cpp-pro Major #7 / M5 above — `render_compute_function` Category-aware branch (touched by Phase 2.2 anyway).

---

## Merge recommendation

**Apply the pre-merge fixup (M1+M2+M3+M4+M5+m5+m6), then merge.** ~150 LOC code + tests. The remaining items belong in a Phase-2-prep follow-up issue.

**Forward look (single biggest residual architectural risk for Phase 2.2-2.7):** the implicit `name + "_old"` convention. Even after M3 lifts it to a named constant, Phase 2.2's `TimeIntegrationPass` needs to map `StateVariable("α")` → its two SymbolDecls (current and old) at lowering time. The architect's stronger fix (storing indices on `StateVariable`) eliminates this entirely; the named-constant fix is half-measure. If you want a clean Phase 2.2 start, do the index fix in the same fixup commit.
