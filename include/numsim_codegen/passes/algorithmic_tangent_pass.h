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
// (LocalNewtonLoweringPass / TimeIntegrationPass+LocalJacobianPass) and BEFORE
// CodeEmitPass so the synthesised tangent output lands in the same emit sweep.
// Precondition is the validation floor only (`symbols_declared`,
// `identifiers_valid`) so tangents work on pure-elastic recipes too.
// Postcondition: `tangent_emitted`.
//
// **Ordering caveat (PR #80 review, finding 1).** Running after Newton lowering
// is ASSUMED from registration order — it is NOT enforced by a precondition
// tag. Safe today: the live path (explicit ∂σ/∂ε) never reads
// `pctx.newton_segments`. When the implicit correction goes live it WILL read
// that seam, and this pass MUST then add `newton_lowered` to `preconditions()`
// (conditional on the recipe having evolution equations) so a misordered
// pipeline fails loudly rather than silently dropping the correction term.
//
// **Scope (3b-1).** Scalar local-Newton unknowns; one independent solve per
// evolution equation (block-diagonal local Jacobian). Coupled tensor-valued
// local systems / multi-surface (rank-4 `(∂R/∂x)⁻¹`, numsim-cas#276 / #43) are
// Phase 3b-2.
//
// **Phase-5 forward-compat note (PR #80 round-2, API finding).** The tangent is
// emitted as an ordinary rank-4 output tagged `roles::ConsistentTangent`. That
// semantic role lives on the `OutputDecl`, but `canonical_arguments` projects
// every output to an `ArgSpec` by KIND only — it drops `OutputDecl::role`, and
// `ArgSpec` has no semantic-role field. So once the tangent reaches a backend
// call-site it is indistinguishable from a user-declared rank-4 output. Phase 5
// (MOOSE `_Jacobian_mult[_qp]` wiring) will need to special-case the tangent and
// must therefore propagate the role (or a `bool is_tangent` / a dedicated
// `ArgSpec::Role`) through `canonical_arguments` into `ArgSpec`. Doing so is
// purely additive (a new field; backends ignore it) and avoids re-touching the
// issue-#77 frozen ArgSpec contract at Phase-5 time. Deferred here (no consumer
// yet) but tracked so the producer-side identity isn't silently lost.
//
// **Minor symmetry (PR #80 review, math finding Q3 — resolved).** A consistent
// stress-strain tangent must satisfy minor symmetry `C_ijkl = C_ijlk`.
// `cas::diff` returns the symmetric rank-4 identity (P_sym = ½(I⊗ᵘI + I⊗ˡI))
// when the strain tensor carries a symmetric `tensor_space`. `add_tensor_input`
// now sets that space for symmetric-role rank-2 inputs (e.g. `roles::Strain`),
// so `∂(2μ ε)/∂ε` emits `2μ·½(otimesu + otimesl)` — minor-symmetric. A plain
// input (`roles::Other`) stays unconstrained, so the behavior is opt-in via the
// role and non-derivative outputs / CSE are unchanged (the space is not part of
// the leaf hash).
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
