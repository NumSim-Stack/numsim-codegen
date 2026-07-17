#ifndef NUMSIM_CODEGEN_PASSES_INTERNAL_ALGORITHMIC_TANGENT_H
#define NUMSIM_CODEGEN_PASSES_INTERNAL_ALGORITHMIC_TANGENT_H

// Phase 3b-1 (issue #35): the ∂(tensor)/∂(scalar) seam.
//
// The consistent tangent of a stress σ(x, ε) where x is a scalar local-Newton
// unknown and ε the strain is
//
//     dσ/dε = ∂σ/∂ε  +  ∂σ/∂x · dx/dε ,     dx/dε = −(∂R/∂x)⁻¹ · ∂R/∂ε
//
// Three of the four derivatives are already supported by numsim-cas:
//   * ∂σ/∂ε   — diff(tensor, tensor)   → rank-4   ✅
//   * ∂R/∂x   — diff(scalar, scalar)   → scalar   ✅ (today's local Jacobian)
//   * ∂R/∂ε   — for a STRAIN-COUPLED residual (t2s-typed) diff(t2s, tensor)
//               → rank-2 ✅; for the current scalar_expression residual it is
//               type-guaranteed zero (a scalar_expression cannot hold a tensor
//               sub-expression — any strain dependence promotes it to t2s).
//
// The one genuine gap is ∂σ/∂x — differentiating a tensor expression w.r.t. a
// SCALAR variable — which numsim-cas does not yet provide
// (`tensor_differentiation` assumes a tensor arg and drops the scalar-
// coefficient product-rule term). Tracked upstream as numsim-cas#275.
//
// `diff_tensor_wrt_scalar` is the single seam through which that derivative is
// taken. It self-detects the cas#275 overload from the pin (see the
// __has_include below) and collapses to a one-line `cas::diff`; absent the
// overload it throws with a precise diagnostic.
//
// NOTE: this term only fires for a strain-coupled (t2s) residual. With the
// current scalar-residual Newton machinery dx/dε ≡ 0, so AlgorithmicTangentPass
// emits the exact tangent (`∂σ/∂ε`) WITHOUT calling this seam. The seam is
// exercised directly by unit tests and by the macro-guarded correction block in
// AlgorithmicTangentPass::run, ready to flip on.

#include <numsim_cas/core/diff.h>
#include <numsim_cas/scalar/scalar_expression.h>
#include <numsim_cas/tensor/tensor_expression.h>

#include <stdexcept>

namespace numsim::codegen::detail {

[[nodiscard]] inline auto diff_tensor_wrt_scalar(
    cas::expression_holder<cas::tensor_expression> const &expr,
    cas::expression_holder<cas::scalar_expression> const &arg)
    -> cas::expression_holder<cas::tensor_expression> {
// Capability detection. The cas#275 overload — diff(tensor, scalar) — ships as
// the `tensor_differentiation_wrt_scalar` visitor header; detect its PRESENCE
// directly rather than relying on an externally-defined compile macro. This
// keeps `diff_tensor_wrt_scalar` a SINGLE inline definition across every TU
// (the body no longer depends on whether a TU links `numsim_codegen_headers`),
// which would otherwise be an ODR hazard. `NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR`
// is kept only as an explicit override/escape hatch.
#if defined(NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR) ||                       \
    __has_include(<numsim_cas/tensor/visitors/tensor_differentiation_wrt_scalar.h>)
  return cas::diff(expr, arg);
#else
  (void)expr;
  (void)arg;
  throw std::runtime_error(
      "AlgorithmicTangentPass: ∂(tensor)/∂(scalar) is not yet "
      "available — blocked on numsim-cas#275 (add "
      "diff(tensor_expression, scalar_expression)). This is the implicit "
      "correction term ∂σ/∂x of the consistent tangent for a "
      "strain-coupled internal variable. Define "
      "NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR once the upstream overload "
      "lands to enable it.");
#endif
}

} // namespace numsim::codegen::detail

#endif // NUMSIM_CODEGEN_PASSES_INTERNAL_ALGORITHMIC_TANGENT_H
