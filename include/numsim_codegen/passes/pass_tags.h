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

#include <string_view>

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
//     takes this as a precondition; PassManager then automatically
//     refuses to register it on a pure-elasticity recipe — surfacing
//     the misconfiguration loudly rather than the pass silently doing
//     nothing.
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

// CodeEmitPass postcondition.
inline constexpr std::string_view compute_function_emitted =
    "compute-function-emitted";

} // namespace numsim::codegen::pass_tags

#endif // NUMSIM_CODEGEN_PASS_TAGS_H
