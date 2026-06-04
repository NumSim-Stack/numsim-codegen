#ifndef NUMSIM_CODEGEN_TIME_INTEGRATION_PASS_H
#define NUMSIM_CODEGEN_TIME_INTEGRATION_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>

namespace numsim::codegen {

// Phase 2.2 (issue #68): backward-Euler lowering of evolution equations
// `Dt(α) = rate` into discrete residual outputs `(α − α_old)/dt − rate`.
//
// The pass mutates the recipe via the mutable-RecipeView arm (first
// real consumer of `require_mutable_model` from PR #57) and synthesises
// one new scalar output per evolution equation, named `<state_var>_residual`.
// `CodeEmitPass` then picks the synthesised outputs up unchanged — the
// generated function gains a `double &<state_var>_residual_out`
// parameter and writes the discrete residual into it.
//
// **Pipeline placement.** Registered between SymbolValidationPass /
// TensorSpaceConsistencyPass and CodeEmitPass. Declares
// `state_variables_non_empty` as a precondition — on a pure-elasticity
// recipe (no state vars), PassManager refuses to advance past this
// pass and surfaces a clear "pass requires precondition X" diagnostic.
// `ConstitutiveModel::emit_compute_function()` registers the pass
// only when the recipe actually has evolution equations, so the
// pure-elasticity path doesn't trip the precondition check.
//
// **Newton iteration body is out of scope.** This pass synthesises the
// residual outputs only. Phase 3a-1's `LocalJacobianPass` (issue #70)
// consumes the `backward_euler_residual_emitted` postcondition and adds
// paired `<sv>_jacobian` outputs via `cas::diff`. Phase 3a-2 (issue #34)
// will add `LocalNewtonLoweringPass` on top to emit the actual Newton
// iteration body once the control-flow emit infrastructure lands.
//
// **Scalar-only.** Tensor evolution equations (rate-form plasticity,
// `Dt(ε^p) = ...`) are common in mechanics but out of Phase 2.2 scope.
// A future tensor extension is mechanical given the scalar shape works.
class TimeIntegrationPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "TimeIntegration";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::symbols_declared,
            pass_tags::identifiers_valid,
            pass_tags::state_variables_checked,
            pass_tags::state_variables_non_empty};
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::backward_euler_residual_emitted};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TIME_INTEGRATION_PASS_H
