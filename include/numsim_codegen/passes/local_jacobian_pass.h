#ifndef NUMSIM_CODEGEN_LOCAL_JACOBIAN_PASS_H
#define NUMSIM_CODEGEN_LOCAL_JACOBIAN_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>

namespace numsim::codegen {

// Phase 3a-1 (issue #70): symbolic-Jacobian emission for each evolution
// equation's discrete residual.
//
// `TimeIntegrationPass` (Phase 2.2, PR #69) emits a backward-Euler
// residual `R_α = (α − α_old)/dt − rate(α)` as a recipe output. This
// pass complements that: for every evolution equation, it computes
// `J_α = ∂R_α/∂α` symbolically via `cas::diff(R, α.current)` and adds
// a second output named `<sv>_jacobian` carrying that expression.
//
// External Newton drivers (or Phase 3a-2's `LocalNewtonLoweringPass`,
// once the control-flow emission infrastructure lands) consume both
// outputs to iterate `α_{k+1} = α_k − R/J`.
//
// **Driver contract (PR #71 round-1 #5):** the emitted `<sv>_jacobian`
// references `dt` as a regular scalar input. A Newton driver MUST hold
// `dt` constant across all iterations of the local solve — otherwise
// the system being solved changes mid-iteration and convergence is
// undefined. MOOSE's `_dt` is per-timestep so this happens naturally;
// hand-written drivers must respect the same invariant.
//
// **Pipeline placement.** Registered AFTER `TimeIntegrationPass` (which
// must have run to populate the synthesised `<sv>_residual` outputs)
// and BEFORE `CodeEmitPass` (so the Jacobian outputs land in the same
// emit sweep). Precondition: `backward_euler_residual_emitted`.
// Postcondition: `jacobian_emitted`.
//
// **Scalar-only.** Tensor evolution equations are out of Phase 3a-1
// scope (and out of Phase 2.2 — they don't exist yet on the recipe
// side). When tensor evolutions land, the rank-4 Jacobian path will
// either fold into this pass or branch into a sibling pass; the
// decision is deferred until the first tensor-evolution consumer
// surfaces.
//
// **Self-referential rate.** The common shape is `rate = f(α)` (e.g.
// `K · α` for linear hardening). `cas::diff` walks the rate via the
// chain rule, producing `J = 1/dt − ∂rate/∂α`. A constant rate
// (`rate = K`) collapses to `J = 1/dt`.
class LocalJacobianPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "LocalJacobian";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::symbols_declared,
            pass_tags::identifiers_valid,
            pass_tags::state_variables_checked,
            pass_tags::state_variables_non_empty,
            pass_tags::backward_euler_residual_emitted};
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::jacobian_emitted};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_LOCAL_JACOBIAN_PASS_H
