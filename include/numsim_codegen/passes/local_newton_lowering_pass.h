#ifndef NUMSIM_CODEGEN_LOCAL_NEWTON_LOWERING_PASS_H
#define NUMSIM_CODEGEN_LOCAL_NEWTON_LOWERING_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>

namespace numsim::codegen {

// Phase 3a-2 (issue #75): lower each scalar evolution equation to an
// in-function Newton iteration instead of emitting `<sv>_residual` /
// `<sv>_jacobian` as outputs for an external driver.
//
// This pass is the "solve in-function" alternative to the
// TimeIntegrationPass + LocalJacobianPass pair (which emit the residual
// and Jacobian as outputs). `emit_compute_function` registers ONE or the
// OTHER based on `ConstitutiveModel::local_newton_enabled()`:
//   * default (off)  → TimeIntegrationPass + LocalJacobianPass (R/J outputs)
//   * enable_local_newton() → this pass
//
// The pass does NOT mutate the recipe. For each evolution equation it
// builds the backward-Euler residual + Jacobian via the shared
// `detail::build_backward_euler_residual` / `build_backward_euler_jacobian`
// helpers (the same single-source-of-truth the output passes use) and
// records a `NewtonSegment` on the PassContext. CodeEmitPass renders the
// segments — declaring a local iterate per state variable, emitting the
// loop with loop-local CSE, and writing the converged value to `<sv>_out`.
//
// **Scope (3a-2):** scalar state variables, one independent Newton solve
// per evolution equation (no inter-equation coupling — that needs the
// LocalNewtonSystem IR / a matrix solve, a later slice). No line search,
// no divergence detection beyond the max-iter guard.
class LocalNewtonLoweringPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "LocalNewtonLowering";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    // Builds R + J itself via the shared helpers, so it needs the same
    // baseline as TimeIntegrationPass (a validated recipe with state
    // variables) — NOT `jacobian_emitted`, since LocalJacobianPass does
    // not run in the Newton pipeline.
    return pass_tags::bundles::with(
        pass_tags::bundles::state_variable_lowering_inputs);
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::newton_lowered};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_LOCAL_NEWTON_LOWERING_PASS_H
