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
  // on a temp name, not on a parenthesised inline expression. Substring
  // match on `tmech::inv(t` avoids depending on the exact counter value
  // (which is preserved across `ctx.reset()` calls and could shift if the
  // test fixture ever shares a context).
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  emit.apply(cas::make_expression<cas::tensor_inv>(A + B));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inv(t"), std::string::npos)
      << "got: " << rendered;
  // And the sum itself was emitted as a temp (i.e. the inv argument is a
  // temp name, not an inline parenthesised expression).
  EXPECT_NE(rendered.find(" + "), std::string::npos) << "got: " << rendered;
}

// TODO(numsim-cas#248): once upstream lands rank-4 tensor_inv support,
// remove this test (and the rank-4 throw branch in TensorCodeEmit), and
// add a new test asserting the templated `tmech::inv<sequence<...>,
// sequence<...>>(...)` emit shape. Tracking on the codegen side: #43.
TEST(TensorCodeEmit, TensorInvRank4ThrowsClearly) {
  // The numsim-cas `inv()` factory rejects rank-4 inputs (per
  // numsim-cas#192) so this codepath is unreachable through the normal
  // public API. We bypass the factory via `make_expression<tensor_inv>`
  // to exercise the emitter's independent guard — defensive against
  // any future code path that constructs a rank-4 tensor_inv directly
  // (e.g. serialisation round-trip without factory validation).
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit);

  auto C = cas::make_expression<cas::tensor>("C", 3, 4);
  ctx.register_symbol_tensor(C, "C");

  try {
    emit.apply(cas::make_expression<cas::tensor_inv>(C));
    FAIL() << "expected std::runtime_error for rank-4 tensor_inv";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("rank 4"), std::string::npos) << "msg: " << msg;
    EXPECT_NE(msg.find("numsim-cas#248"), std::string::npos)
        << "msg: " << msg;
  }
}

} // namespace numsim::codegen
