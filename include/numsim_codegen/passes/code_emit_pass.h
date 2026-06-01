#ifndef NUMSIM_CODEGEN_CODE_EMIT_PASS_H
#define NUMSIM_CODEGEN_CODE_EMIT_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>
// NOTE: `run()` is defined in recipe.h after ConstitutiveModel is complete.
// Any TU instantiating this pass must include recipe.h; in practice
// recipe.h is the only constructor site.

namespace numsim::codegen {

// Phase 1.2 CodeEmitPass.
//
// Drives the existing scalar/tensor/tensor-to-scalar visitor pipeline:
//   1. Register symbol-name → expression mappings with the CodeGenContext.
//   2. Walk each output expression, accumulating statements + the final RHS.
//   3. Render the framing (function signature + body + output writes) via
//      ConstitutiveModel::render_compute_function.
//
// Result lands in `pctx.compute_function_source`.
//
// Preconditions: "symbols-declared" + "identifiers-valid" +
// "tensor-space-validated" (i.e. SymbolValidationPass +
// TensorSpaceConsistencyPass must have run first). If you add a pass
// that transforms expressions (e.g. a future TimeIntegrationPass),
// register it AFTER the validators but BEFORE CodeEmitPass.
class CodeEmitPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "CodeEmit";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::symbols_declared, pass_tags::identifiers_valid,
            pass_tags::tensor_space_declarations_checked};
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::compute_function_emitted};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_CODE_EMIT_PASS_H
