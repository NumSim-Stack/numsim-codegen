#ifndef NUMSIM_CODEGEN_ALGORITHMIC_TANGENT_PASS_H
#define NUMSIM_CODEGEN_ALGORITHMIC_TANGENT_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>

namespace numsim::codegen {

// Phase 3b-1 (issue #35): synthesise the consistent (algorithmic) tangent
// `dσ/dε` for each tangent requested via
// `ConstitutiveModel::add_algorithmic_tangent(name, of_output, wrt_input)`,
// adding it as a rank-4 tensor output that the existing CodeEmitPass renders.
//
//     dσ/dε = ∂σ/∂ε  +  ∂σ/∂x · dx/dε ,     dx/dε = −(∂R/∂x)⁻¹ · ∂R/∂ε
//
// **What ships in 3b-1 (no upstream dependency):** the explicit term `∂σ/∂ε`
// via `cas::diff(tensor, tensor)`. With the current scalar-residual Newton
// machinery the local unknown `x` is type-guaranteed independent of the strain
// (a strain-coupled residual would be t2s-typed, which `NewtonSegment.residual`
// cannot hold), so `dx/dε ≡ 0` and `∂σ/∂ε` IS the exact consistent tangent.
// This already covers elastic and strain-independent-internal-variable models.
//
// **What is scaffolded behind a stub:** the implicit correction
// `∂σ/∂x · dx/dε`, needed once strain-coupled (t2s) residuals are supported.
// Its one genuine algebra gap — `∂σ/∂x` = `diff(tensor, scalar)` — is taken
// through `detail::diff_tensor_wrt_scalar` (internal/algorithmic_tangent.h),
// which throws a precise numsim-cas#275 diagnostic until the overload lands.
// The correction assembly lives in `run()` under
// NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR, ready to flip on.
//
// **Pipeline placement.** Registered AFTER any state-variable lowering
// (LocalNewtonLoweringPass / TimeIntegrationPass+LocalJacobianPass) so the
// converged-state seam — `pctx.newton_segments` — is populated, and BEFORE
// CodeEmitPass so the synthesised tangent output lands in the same emit sweep.
// Precondition is the validation floor only (`symbols_declared`,
// `identifiers_valid`) so tangents work on pure-elastic recipes too; ordering
// after newton lowering is guaranteed by registration order, not a tag.
// Postcondition: `tangent_emitted`.
//
// **Scope (3b-1).** Scalar local-Newton unknowns; one independent solve per
// evolution equation (block-diagonal local Jacobian). Coupled tensor-valued
// local systems / multi-surface (rank-4 `(∂R/∂x)⁻¹`, numsim-cas#276 / #43) are
// Phase 3b-2.
class AlgorithmicTangentPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "AlgorithmicTangent";
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::symbols_declared, pass_tags::identifiers_valid};
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {pass_tags::tangent_emitted};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_ALGORITHMIC_TANGENT_PASS_H
