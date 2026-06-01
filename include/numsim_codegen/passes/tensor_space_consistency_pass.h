#ifndef NUMSIM_CODEGEN_TENSOR_SPACE_CONSISTENCY_PASS_H
#define NUMSIM_CODEGEN_TENSOR_SPACE_CONSISTENCY_PASS_H

#include <numsim_codegen/passes/pass.h>

namespace numsim::codegen {

// Phase 1.2 TensorSpaceConsistencyPass (M6 in issue #48).
//
// Walks declared tensor symbols and validates that any present
// `tensor_space` annotation is consistent with the symbol's Role
// attributes — specifically, a Role with `is_symmetric == true` paired
// with a `Skew` tensor_space is a contradiction (a symmetric Stress
// cannot be skew-symmetric).
//
// **Scope today:** declaration-level cross-check only. Most tensors
// have no `space()` annotation in Phase 1.2, so the pass passes silently
// for most recipes. The pass exists to:
//   1. Validate the PassContext / PassManager shape with a second
//      concrete pass (the framework was previously load-bearing on a
//      single consumer).
//   2. Establish the postcondition tag `tensor-space-validated` that
//      Phase 2 expression-rewriting passes (TimeIntegrationPass,
//      KuhnTuckerLoweringPass) will rely on.
//   3. Provide a hook point for expression-level inference — Phase 2
//      will replace this body with a real walker that infers
//      tensor_space from expression structure (e.g. sym(A) → Symmetric,
//      A:B → propagate from operands) and checks consistency end-to-end.
//
// Per D3' in the prototype plan: tensor_space validation is in scope
// for Phase 1.2 but expression-level inference is deferred to Phase 2.
class TensorSpaceConsistencyPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "TensorSpaceConsistency";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return {"symbols-declared"};
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {"tensor-space-validated"};
  }
  void run(PassContext &pctx) override; // defined in recipe.h.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TENSOR_SPACE_CONSISTENCY_PASS_H
