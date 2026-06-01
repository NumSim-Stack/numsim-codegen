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

} // namespace numsim::codegen
