// Direct tests for TensorCodeEmit visitor overrides — complements the
// recipe-level tests in MooseTargetTest / RecipeTest. Lets each tensor-
// leaf and tensor-operator emit case fail in isolation rather than as a
// substring inside the larger generated source.

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>

#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/levi_civita_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/wrappers/tensor_inv.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

TEST(TensorCodeEmit, LeviCivitaEmitsTmechLeviCivita) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  // dim ∈ {2, 3, 4} is supported by numsim-cas.
  auto eps3 = cas::make_expression<cas::levi_civita_tensor>(3);
  emit.apply(eps3);
  auto rendered = ctx.render_statements();

  EXPECT_NE(rendered.find("tmech::levi_civita<double, 3>"),
            std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, LeviCivitaDim2) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto eps2 = cas::make_expression<cas::levi_civita_tensor>(2);
  emit.apply(eps2);
  EXPECT_NE(ctx.render_statements().find("tmech::levi_civita<double, 2>"),
            std::string::npos);
}

TEST(TensorCodeEmit, IdentityTensorRank2EmitsTmechEye) {
  // Regression: identity_tensor is the replacement for the now-removed
  // kronecker_delta node. Ensure the rank-2 emit still produces tmech::eye.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto I = cas::make_expression<cas::identity_tensor>(3, 2);
  emit.apply(I);
  EXPECT_NE(ctx.render_statements().find("tmech::eye<double, 3, 2>"),
            std::string::npos);
}

// ─── tensor_inv (Phase 1.1, first emit-stub implementation) ─────────

TEST(TensorCodeEmit, TensorInvEmitsTmechInv) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  ctx.register_symbol_tensor(A, "A");

  // Construct tensor_inv directly via make_expression to avoid the
  // construction-time space-tracking guards in the `inv()` factory
  // (which require the full tensor_functions.h include cycle and
  // duplicate the simplifier rules already verified upstream).
  emit.apply(cas::make_expression<cas::tensor_inv>(A));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inv(A)"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, TensorInvOnCompoundInputReusesTemp) {
  // The inner expression should resolve through CSE; tmech::inv operates
  // on a temp name, not on a parenthesised inline expression.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  // inv(A + B) — the sum gets its own temp; inv operates on that temp.
  emit.apply(cas::make_expression<cas::tensor_inv>(A + B));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inv(t0)"), std::string::npos)
      << "got: " << rendered;
}

} // namespace numsim::codegen
