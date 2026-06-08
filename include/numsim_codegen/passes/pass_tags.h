#ifndef NUMSIM_CODEGEN_PASS_TAGS_H
#define NUMSIM_CODEGEN_PASS_TAGS_H

// Phase-1.2 follow-up (issue #56 / P3 + P6): typed pre/postcondition tags
// for the Pass framework.
//
// String literals scattered across pass headers don't catch typos at
// compile time — `"tensor-space-validted"` returns a different
// string_view than `"tensor-space-validated"`, and PassManager only
// catches the mismatch at runtime with a message the user has to grep
// for. Naming these as `inline constexpr std::string_view` constants
// forces every reference through the type system: a typo becomes a
// compile error.
//
// Also incidentally fixes the cpp-pro n-2 finding from
// `REVIEW-stack-51-55.md`: every pass that *requires* a tag references
// the same constant the producer advertises, so under-declared
// preconditions become impossible to write.
//
// Storage: `std::string_view` pointing at the literal's static storage.
// PassManager's `std::set<std::string_view>` lifetime contract is
// satisfied trivially — static-storage backing.
//
// P6 in the same issue: rename `tensor-space-validated` →
// `tensor_space_declarations_checked`. The original tag overpromised
// what TensorSpaceConsistencyPass actually guarantees (declaration-level
// only, not expression-level inference). Phase 2 expression-rewriting
// passes will introduce a separate `tensor_space_inferred` tag for the
// stronger guarantee, without having to redefine the existing one.

#include <array>
#include <string_view>
#include <vector>

namespace numsim::codegen::pass_tags {

// SymbolValidationPass postconditions.
inline constexpr std::string_view symbols_declared = "symbols-declared";
inline constexpr std::string_view identifiers_valid = "identifiers-valid";

// State-variable tags advertised by `SymbolValidationPass` (issue #59 /
// REVIEW-pr-58.md m2, refined by PR #66 review).
//
// Phase 2.2+ passes that touch state variables fall in one of two
// shapes, and the framework can serve them differently with two tags:
//
//   * `state_variables_checked` — ALWAYS advertised. The
//     SymbolValidationPass body has walked `state_variables()` and
//     verified the alignment invariant with `symbols()` (via
//     `verify_state_variable_symbol_alignment`). A pass that runs even
//     on pure-elasticity recipes (e.g. one that emits no-op stubs for
//     consistency) takes this as a precondition.
//
//   * `state_variables_non_empty` — advertised IFF the recipe actually
//     has at least one state variable. A pass whose body is a no-op on
//     empty state vectors (TimeIntegrationPass, KuhnTuckerLoweringPass)
//     takes this as a precondition; on a pure-elasticity recipe
//     PassManager then fails at `run()` with a clear
//     `"pass X requires precondition state-variables-non-empty but no
//     earlier pass advertised that postcondition"` message — surfacing
//     the misconfiguration loudly rather than the pass silently doing
//     nothing. (PassManager checks preconditions at `run()` time, not
//     at registration; see `pass_manager.h:39-46`.)
//
// The split keeps the framework's tag-tracking honest: a passcondition
// has the same name across every consumer, so a typo or rename surfaces
// at compile time, not in the form of a pass that quietly never fires.
inline constexpr std::string_view state_variables_checked =
    "state-variables-checked";
inline constexpr std::string_view state_variables_non_empty =
    "state-variables-non-empty";

// TensorSpaceConsistencyPass postcondition. Renamed from the original
// `tensor-space-validated` per P6 — the original overpromised; Phase 2's
// expression-level inference pass will use a separate tag.
inline constexpr std::string_view tensor_space_declarations_checked =
    "tensor-space-declarations-checked";

// TimeIntegrationPass postcondition (Phase 2.2, issue #68; renamed in
// PR #71 round-1 review #4). Advertises that every evolution equation
// `Dt(α) = rate` has been lowered to the **backward-Euler** discrete
// residual `(α − α_old)/dt − rate`. The previous name `dt_lowered`
// was too coarse — Phase 4's planned Forward-Euler / BDF2 integrators
// would advertise the same tag and silently satisfy consumers
// (`LocalJacobianPass` etc.) that emit backward-Euler-specific shapes.
// Future integration schemes get their own per-scheme tags.
inline constexpr std::string_view backward_euler_residual_emitted =
    "backward-euler-residual-emitted";

// LocalJacobianPass postcondition (Phase 3a-1, issue #70). Advertises
// that every evolution equation now has a `<sv>_jacobian` output
// representing `∂<sv>_residual/∂<sv>` — computed symbolically at
// codegen time via `cas::diff()`. Phase 3a-2's actual Newton-loop
// emission pass will consume this tag as a precondition; external
// Newton drivers can use the residual + Jacobian outputs directly.
inline constexpr std::string_view jacobian_emitted = "jacobian-emitted";

// LocalNewtonLoweringPass postcondition (Phase 3a-2, issue #75).
// Advertises that every scalar evolution equation has been lowered to an
// in-function Newton iteration segment (residual + Jacobian recomputed
// per iteration, state variable solved internally) recorded on the
// PassContext for CodeEmitPass to render. Phase 3b's tangent pass will
// consume this — the converged state is the point the consistent tangent
// is linearised about.
inline constexpr std::string_view newton_lowered = "newton-lowered";

// AlgorithmicTangentPass postcondition (Phase 3b-1, issue #35). Advertises
// that every requested consistent tangent `dσ/dε` has been synthesised as a
// rank-4 tensor output. The explicit term `∂σ/∂ε` is emitted today via
// `cas::diff(tensor, tensor)`; the implicit correction `∂σ/∂x · dx/dε`
// activates once strain-coupled (t2s) residuals + numsim-cas#275
// (`diff(tensor, scalar)`) land — see internal/algorithmic_tangent.h.
inline constexpr std::string_view tangent_emitted = "tangent-emitted";

// CodeEmitPass postcondition.
inline constexpr std::string_view compute_function_emitted =
    "compute-function-emitted";

// ─── Precondition bundles (cross-cutting review, MAJOR/MINOR #6) ──────
//
// Phase 2.2+ state-variable-lowering passes (TimeIntegrationPass,
// LocalJacobianPass, and Phase 3a-2's LocalNewtonLoweringPass / Phase
// 2.5's KuhnTuckerLoweringPass) all require the same baseline: a fully
// validated recipe that actually HAS state variables. Repeating the
// four-tag list in each pass invites drift (one pass forgets a tag and
// silently runs in a weaker context). The bundle centralises the list;
// passes splice it in via `with(...)`, appending any pass-specific tags.
namespace bundles {

// Everything SymbolValidationPass guarantees for a recipe that has at
// least one state variable. The precondition floor for every pass that
// lowers state-variable evolution.
inline constexpr std::array state_variable_lowering_inputs = {
    symbols_declared, identifiers_valid, state_variables_checked,
    state_variables_non_empty};

// Build a precondition vector from a bundle plus zero or more extra
// pass-specific tags. `with(bundle)` returns the bundle as a vector;
// `with(bundle, a, b)` appends a and b.
template <std::size_t N, class... Extra>
[[nodiscard]] inline auto with(std::array<std::string_view, N> const &base,
                               Extra... extra)
    -> std::vector<std::string_view> {
  std::vector<std::string_view> v(base.begin(), base.end());
  (v.push_back(extra), ...);
  return v;
}

} // namespace bundles

} // namespace numsim::codegen::pass_tags

#endif // NUMSIM_CODEGEN_PASS_TAGS_H
