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
