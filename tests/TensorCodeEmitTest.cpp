// Direct tests for TensorCodeEmit visitor overrides — complements the
// recipe-level tests in MooseTargetTest / RecipeTest. Lets each tensor-
// leaf and tensor-operator emit case fail in isolation rather than as a
// substring inside the larger generated source.

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>

#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/levi_civita_tensor.h>
#include <numsim_cas/tensor/projection_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/wrappers/tensor_inv.h>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

namespace numsim::codegen {

namespace {

// M3: TensorCodeEmit constructor now requires a T2sApply. Tests below
// that don't exercise the tensor_to_scalar_with_tensor_mul codepath
// pass this throwing stub — if a future change accidentally routes a
// t2s subterm through a stubbed test, the throw makes it loud rather
// than silent.
//
// If you're authoring a test that constructs a
// `tensor_to_scalar_with_tensor_mul` node (or any future cross-domain
// tensor node carrying a t2s subterm), DO NOT use this stub — construct
// a `CodeEmitPipeline` instead (see `T2sWithTensorMulMultipliesScalarByTensor`
// for the pattern).
inline auto throwing_t2s_apply() -> TensorCodeEmit::T2sApply {
  return [](auto const &) -> std::string {
    throw std::logic_error(
        "throwing_t2s_apply invoked: this test was not wired with a real "
        "t2s callback. Either wire one with the unique_ptr indirection or "
        "move the test off the t2s codepath.");
  };
}

} // namespace

TEST(TensorCodeEmit, LeviCivitaEmitsTmechLeviCivita) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

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
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

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
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto I = cas::make_expression<cas::identity_tensor>(3, 2);
  emit.apply(I);
  EXPECT_NE(ctx.render_statements().find("tmech::eye<double, 3, 2>"),
            std::string::npos);
}

// ─── tensor_inv (Phase 1.1, first emit-stub implementation) ─────────

TEST(TensorCodeEmit, TensorInvEmitsTmechInv) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

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
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

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
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

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

// ─── tensor_projector (Phase 1.1) ────────────────────────────────────
//
// Cover the four presets that ship today (P_sym, P_skew, P_vol, P_dev at
// acts_on_rank=2 → rank-4 projector). For each, assert the tmech building
// blocks appear in the rendered source. Numerical correctness of the
// emitted formulas is exercised end-to-end by the compile-check driver
// when a recipe that uses these projectors lands.

TEST(TensorCodeEmit, ProjectorSymEmitsSymmetricBuildup) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  emit.apply(cas::P_sym(3));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::otimesu(tmech::eye<double, 3, 2>()"),
            std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::otimesl(tmech::eye<double, 3, 2>()"),
            std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find(" + "), std::string::npos) << "got: " << rendered;
  // Sym projector must NOT contain the trace term (otimes(I,I)).
  EXPECT_EQ(rendered.find("tmech::otimes(tmech::eye"), std::string::npos)
      << "P_sym leaked a trace term. got: " << rendered;
}

TEST(TensorCodeEmit, ProjectorSkewEmitsSkewBuildup) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  emit.apply(cas::P_skew(3));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::otimesu"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::otimesl"), std::string::npos)
      << "got: " << rendered;
  // The distinguishing feature is the minus sign between otimesu and otimesl.
  EXPECT_NE(rendered.find(" - "), std::string::npos) << "got: " << rendered;
}

TEST(TensorCodeEmit, ProjectorVolEmitsTraceProjector) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  emit.apply(cas::P_vol(3));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("(1.0 / 3.0)"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::otimes(tmech::eye<double, 3, 2>()"),
            std::string::npos)
      << "got: " << rendered;
  // Pure vol must not carry the sym buildup (otimesu/otimesl).
  EXPECT_EQ(rendered.find("tmech::otimesu"), std::string::npos)
      << "P_vol leaked a sym term. got: " << rendered;
}

TEST(TensorCodeEmit, ProjectorDevEmitsSymMinusVol) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  emit.apply(cas::P_devi(3));
  auto rendered = ctx.render_statements();
  // P_dev = P_sym - P_vol → must include all three of otimesu, otimesl,
  // otimes plus the (1/d) coefficient.
  EXPECT_NE(rendered.find("tmech::otimesu"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::otimesl"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::otimes(tmech::eye"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("(1.0 / 3.0)"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, ProjectorDim2UsesCorrectDim) {
  // Catches the typo of hardcoding dim=3 in the emit.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  emit.apply(cas::P_vol(2));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("(1.0 / 2.0)"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::eye<double, 2, 2>()"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, ProjectorCSEReusesTemp) {
  // P_sym used twice in a single expression should share a single temp.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto Psym = cas::P_sym(3);
  emit.apply(Psym);
  emit.apply(Psym);

  // Count how many times the sym build-up appears in the rendered source.
  auto rendered = ctx.render_statements();
  std::size_t count = 0;
  for (std::size_t pos = 0;
       (pos = rendered.find("tmech::otimesu", pos)) != std::string::npos;
       ++pos) {
    ++count;
  }
  EXPECT_EQ(count, 1u) << "Expected CSE to emit P_sym once. got:\n"
                       << rendered;
}

TEST(TensorCodeEmit, ProjectorHarmonicThrowsClearly) {
  // P_harm is one of the cas presets but not implemented in this phase.
  // The harmonic projector for rank-2 is the trace-free sym projector,
  // which is just P_dev for d=3. Codegen doesn't auto-fold this — the
  // recipe author should write P_devi() explicitly. We assert the error
  // points at the supported alternatives so the failure is actionable.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  try {
    emit.apply(cas::P_harm(3, 2));
    FAIL() << "expected std::runtime_error for P_harm";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("perm, trace"), std::string::npos) << "msg: " << msg;
    EXPECT_NE(msg.find("P_dev"), std::string::npos) << "msg: " << msg;
  }
}

TEST(TensorCodeEmit, ProjectorActsOnRank1ThrowsClearly) {
  // acts_on_rank=1 → rank-2 projector. Not yet emitted; assert the throw
  // is actionable.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  // Build directly via make_projector — the cas P_sym preset hardcodes
  // acts_on_rank=2, so we'd have no way to construct an acts_on_rank=1
  // projector through the high-level API.
  auto proj = cas::make_projector(3, 1, cas::Symmetric{}, cas::AnyTraceTag{});
  try {
    emit.apply(proj);
    FAIL() << "expected std::runtime_error for acts_on_rank=1";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("acts_on_rank=1"), std::string::npos)
        << "msg: " << msg;
  }
}

// ─── outer_product_wrapper / simple_outer_product / inner_product_wrapper ──
// (Phase 1.1, the index-bearing product family.)

TEST(TensorCodeEmit, OuterProductWrapperEmitsTemplateForm) {
  // a ⊗ b with placement (1,2) and (3,4) — i.e. the standard `otimes`.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  emit.apply(cas::make_expression<cas::outer_product_wrapper>(
      A, cas::sequence{1, 2}, B, cas::sequence{3, 4}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(
                "tmech::outer_product<tmech::sequence<1, 2>, "
                "tmech::sequence<3, 4>>(A, B)"),
            std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, OuterProductWrapperPreservesNon1234Indices) {
  // Non-canonical placement — index sequence must round-trip 1-based to
  // tmech, not get accidentally re-sorted or zero-based.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  // a_{ik} b_{jl} — the `otimesu` placement pattern.
  emit.apply(cas::make_expression<cas::outer_product_wrapper>(
      A, cas::sequence{1, 3}, B, cas::sequence{2, 4}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::sequence<1, 3>"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::sequence<2, 4>"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, SimpleOuterProductTwoChildrenChains) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  cas::simple_outer_product sop(3, 4);
  sop.push_back(A);
  sop.push_back(B);
  emit.apply(cas::make_expression<cas::simple_outer_product>(std::move(sop)));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::outer_product<tmech::sequence<1, 2>, "
                          "tmech::sequence<3, 4>>(A, B)"),
            std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, SimpleOuterProductThreeChildrenNests) {
  // Three rank-2 children → rank-6 result. The accumulator picks up indices
  // 1..2, then 1..4, then 1..6 for the third child's placement at <5,6>.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  auto C = cas::make_expression<cas::tensor>("C", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");
  ctx.register_symbol_tensor(C, "C");

  cas::simple_outer_product sop(3, 6);
  sop.push_back(A);
  sop.push_back(B);
  sop.push_back(C);
  emit.apply(cas::make_expression<cas::simple_outer_product>(std::move(sop)));
  auto rendered = ctx.render_statements();
  // Final outer carries the (1..4, 5..6) split, with the inner (A,B) form
  // nested inside.
  EXPECT_NE(rendered.find("tmech::sequence<1, 2, 3, 4>"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find("tmech::sequence<5, 6>"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, InnerProductWrapperEmitsTemplateForm) {
  // Generic double contraction C : ε for C rank-4, ε rank-2.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto C = cas::make_expression<cas::tensor>("C", 3, 4);
  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(C, "C");
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      C, cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(
                "tmech::inner_product<tmech::sequence<3, 4>, "
                "tmech::sequence<1, 2>>(C, eps)"),
            std::string::npos)
      << "got: " << rendered;
}

// Projector short-circuits — confirm `P : A` emits the unary tmech op rather
// than materialising the rank-4 projector.

TEST(TensorCodeEmit, InnerProductWithPSymEmitsTmechSym) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      cas::P_sym(3), cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::sym(eps)"), std::string::npos)
      << "got: " << rendered;
  // And no projector tensor was materialised.
  EXPECT_EQ(rendered.find("tmech::otimesu"), std::string::npos)
      << "P_sym should have been short-circuited, not materialised. got: "
      << rendered;
}

TEST(TensorCodeEmit, InnerProductWithPSkewEmitsTmechSkew) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      cas::P_skew(3), cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  EXPECT_NE(ctx.render_statements().find("tmech::skew(eps)"), std::string::npos);
}

TEST(TensorCodeEmit, InnerProductWithPVolEmitsTmechVol) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      cas::P_vol(3), cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  EXPECT_NE(ctx.render_statements().find("tmech::vol(eps)"), std::string::npos);
}

TEST(TensorCodeEmit, InnerProductWithPDevEmitsTmechDev) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      cas::P_devi(3), cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  EXPECT_NE(ctx.render_statements().find("tmech::dev(eps)"), std::string::npos);
}

TEST(TensorCodeEmit, InnerProductGenericFormUsedWhenLhsIsntKnownProjector) {
  // A rank-4 LHS that ISN'T a projector should fall through to the generic
  // tmech::inner_product form, not be misclassified.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto C = cas::make_expression<cas::tensor>("C", 3, 4);
  auto eps = cas::make_expression<cas::tensor>("eps", 3, 2);
  ctx.register_symbol_tensor(C, "C");
  ctx.register_symbol_tensor(eps, "eps");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      C, cas::sequence{3, 4}, eps, cas::sequence{1, 2}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inner_product<"), std::string::npos)
      << "got: " << rendered;
  // And we did NOT mistakenly emit a unary projector op.
  EXPECT_EQ(rendered.find("tmech::sym("), std::string::npos) << rendered;
  EXPECT_EQ(rendered.find("tmech::dev("), std::string::npos) << rendered;
}

TEST(TensorCodeEmit, InnerProductProjectorOnRank4RhsFallsThrough) {
  // Same projector LHS but rank-4 RHS — short-circuit must NOT trigger
  // (it's only defined for the rank-2 application). Falls back to the
  // generic inner_product emit.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto X = cas::make_expression<cas::tensor>("X", 3, 4);
  ctx.register_symbol_tensor(X, "X");

  emit.apply(cas::make_expression<cas::inner_product_wrapper>(
      cas::P_sym(3), cas::sequence{3, 4}, X, cas::sequence{1, 2}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inner_product<"), std::string::npos)
      << "got: " << rendered;
  EXPECT_EQ(rendered.find("tmech::sym("), std::string::npos) << rendered;
}

// ─── tensor_mul ──────────────────────────────────────────────────────

TEST(TensorCodeEmit, TensorMulTwoChildrenSingleContraction) {
  // A · B for two rank-2 tensors → rank-2 result via single contraction
  // of last(A) with first(B). tmech form is inner_product<seq<2>, seq<1>>.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  cas::tensor_mul m(3, 2);
  m.push_back(A);
  m.push_back(B);
  emit.apply(cas::make_expression<cas::tensor_mul>(std::move(m)));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::inner_product<tmech::sequence<2>, "
                          "tmech::sequence<1>>(A, B)"),
            std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, TensorMulThreeChildrenChains) {
  // A · B · C — accumulator rank stays 2 throughout (2+2-2 = 2).
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  auto C = cas::make_expression<cas::tensor>("C", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");
  ctx.register_symbol_tensor(C, "C");

  cas::tensor_mul m(3, 2);
  m.push_back(A);
  m.push_back(B);
  m.push_back(C);
  emit.apply(cas::make_expression<cas::tensor_mul>(std::move(m)));
  auto rendered = ctx.render_statements();
  // Should see two nested inner_products (since accumulator stays rank-2,
  // both levels use sequence<2>/sequence<1>).
  std::size_t count = 0;
  for (std::size_t pos = 0;
       (pos = rendered.find("tmech::inner_product", pos)) != std::string::npos;
       ++pos) {
    ++count;
  }
  EXPECT_EQ(count, 2u) << "Expected two nested inner_products. got:\n"
                       << rendered;
  EXPECT_NE(rendered.find("C"), std::string::npos) << rendered;
}

// ─── tensor_pow ──────────────────────────────────────────────────────

TEST(TensorCodeEmit, TensorPowRank2EmitsTmechPow) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  auto two = cas::make_expression<cas::scalar_constant>(2.0);

  emit.apply(cas::make_expression<cas::tensor_pow>(A, two));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::pow(A,"), std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, TensorPowRank4ThrowsClearly) {
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto C = cas::make_expression<cas::tensor>("C", 3, 4);
  ctx.register_symbol_tensor(C, "C");
  auto two = cas::make_expression<cas::scalar_constant>(2.0);

  try {
    emit.apply(cas::make_expression<cas::tensor_pow>(C, two));
    FAIL() << "expected std::runtime_error for rank-4 tensor_pow";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("rank 4"), std::string::npos) << "msg: " << msg;
  }
}

// ─── permute_indices_wrapper ─────────────────────────────────────────

TEST(TensorCodeEmit, PermuteIndicesEmitsBasisChange) {
  // The classic transpose pattern: rank-2 with permutation {2, 1}.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  ctx.register_symbol_tensor(A, "A");

  emit.apply(cas::make_expression<cas::permute_indices_wrapper>(
      A, cas::sequence{2, 1}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::basis_change<tmech::sequence<2, 1>>(A)"),
            std::string::npos)
      << "got: " << rendered;
}

TEST(TensorCodeEmit, PermuteIndicesPreservesArbitraryRank4Permutation) {
  // {3,4,1,2} on a rank-4 tensor — minor-major swap pattern from elasticity.
  CodeGenContext ctx;
  ScalarCodeEmit scalar_emit(ctx);
  TensorCodeEmit emit(ctx, scalar_emit, throwing_t2s_apply());

  auto C = cas::make_expression<cas::tensor>("C", 3, 4);
  ctx.register_symbol_tensor(C, "C");

  emit.apply(cas::make_expression<cas::permute_indices_wrapper>(
      C, cas::sequence{3, 4, 1, 2}));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::basis_change<tmech::sequence<3, 4, 1, 2>>(C)"),
            std::string::npos)
      << "got: " << rendered;
}

// ─── tensor_to_scalar_with_tensor_mul ────────────────────────────────

TEST(TensorCodeEmit, T2sWithTensorMulMultipliesScalarByTensor) {
  // trace(A) * B — t2s on the rhs, tensor on the lhs. Expect the t2s
  // callback to fire and the emit to read `<trace_emit> * <tensor>`.
  //
  // P2: cycle-break is now encapsulated in CodeEmitPipeline. Tests that
  // need the full t2s pipeline construct one, skipping the per-test
  // unique_ptr dance.
  CodeGenContext ctx;
  CodeEmitPipeline pipeline(ctx);

  auto A = cas::make_expression<cas::tensor>("A", 3, 2);
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  ctx.register_symbol_tensor(A, "A");
  ctx.register_symbol_tensor(B, "B");

  auto tr_A = cas::make_expression<cas::tensor_trace>(A);
  pipeline.tensor().apply(
      cas::make_expression<cas::tensor_to_scalar_with_tensor_mul>(B, tr_A));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("tmech::trace(A)"), std::string::npos)
      << "got: " << rendered;
  // The final temp should multiply the trace by B.
  EXPECT_NE(rendered.find(" * B"), std::string::npos) << rendered;
}

// M3: `T2sWithTensorMulThrowsWhenCallbackUnset` was deleted with this
// change. The "callback unset" case is no longer reachable — the
// constructor requires the callback at construction time, so the
// "forgot to wire" failure mode is now a compile error rather than a
// runtime throw. See M3 in issue #48 for the rationale.

} // namespace numsim::codegen
