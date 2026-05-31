#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

TEST(ScalarCodeEmit, LeafScalarEmitsName) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto x = cas::make_expression<cas::scalar>("x");
  ctx.register_symbol_scalar(x, "x");
  EXPECT_EQ(emit.apply(x), "x");
  EXPECT_EQ(ctx.temporary_count(), 0u)
      << "Leaf symbols should not allocate temporaries";
}

TEST(ScalarCodeEmit, ScalarZeroEmitsLiteralWithoutTemporary) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  EXPECT_EQ(emit.apply(cas::get_scalar_zero()), "0.0");
  EXPECT_EQ(ctx.temporary_count(), 0u);
}

TEST(ScalarCodeEmit, ScalarOneEmitsLiteralWithoutTemporary) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  EXPECT_EQ(emit.apply(cas::get_scalar_one()), "1.0");
  EXPECT_EQ(ctx.temporary_count(), 0u);
}

TEST(ScalarCodeEmit, AdditionEmitsTemporaryWithBothChildren) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto x = cas::make_expression<cas::scalar>("x");
  auto y = cas::make_expression<cas::scalar>("y");
  ctx.register_symbol_scalar(x, "x");
  ctx.register_symbol_scalar(y, "y");

  auto result = emit.apply(x + y);
  EXPECT_EQ(result, "t0");
  ASSERT_EQ(ctx.temporary_count(), 1u);
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("x"), std::string::npos) << "got: " << rendered;
  EXPECT_NE(rendered.find("y"), std::string::npos) << "got: " << rendered;
  EXPECT_NE(rendered.find(" + "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, NestedExpressionCSEsSharedSubterm) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto x = cas::make_expression<cas::scalar>("x");
  auto y = cas::make_expression<cas::scalar>("y");
  ctx.register_symbol_scalar(x, "x");
  ctx.register_symbol_scalar(y, "y");

  // (x + y) * (x + y) — both factors are the same DAG node (the simplifier
  // canonicalises them to a single shared expression via hash).
  auto sum = x + y;
  auto product = sum * sum;

  auto result = emit.apply(product);
  // The shared sum should appear exactly once as a temporary, and the final
  // multiplication should reference it twice.
  auto rendered = ctx.render_statements();
  std::size_t sum_count = 0;
  for (std::size_t pos = 0; (pos = rendered.find(" + ", pos)) != std::string::npos;
       ++pos) {
    ++sum_count;
  }
  EXPECT_EQ(sum_count, 1u) << "Sum should be emitted once. Got:\n"
                          << rendered;
}

TEST(ScalarCodeEmit, FunctionCallsEmitStdEquivalents) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto x = cas::make_expression<cas::scalar>("x");
  ctx.register_symbol_scalar(x, "x");

  auto sin_result = emit.apply(sin(x));
  EXPECT_NE(ctx.render_statements().find("std::sin"), std::string::npos);

  ctx.reset();
  emit.apply(cas::sqrt(x));
  EXPECT_NE(ctx.render_statements().find("std::sqrt"), std::string::npos);

  ctx.reset();
  emit.apply(cas::exp(x));
  EXPECT_NE(ctx.render_statements().find("std::exp"), std::string::npos);
}

TEST(ScalarCodeEmit, ConstantEmitsAsDoubleLiteral) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto five = cas::make_expression<cas::scalar_constant>(5.0);
  // Constants are not stored as DAG nodes that get temporaries — they're
  // emitted inline.
  auto result = emit.apply(five);
  // Allow either "5" or "5.0" — codegen normalises to a double literal.
  EXPECT_TRUE(result == "5" || result == "5.0") << "got: " << result;
}

// ─── Comparison nodes ─────────────────────────────────────────────────
//
// Each emits `static_cast<double>(lhs OP rhs)` to preserve numsim-cas's
// indicator-double semantics (1.0 / 0.0).

namespace {
auto make_two_scalars(CodeGenContext &ctx)
    -> std::pair<cas::expression_holder<cas::scalar_expression>,
                 cas::expression_holder<cas::scalar_expression>> {
  auto x = cas::make_expression<cas::scalar>("x");
  auto y = cas::make_expression<cas::scalar>("y");
  ctx.register_symbol_scalar(x, "x");
  ctx.register_symbol_scalar(y, "y");
  return {x, y};
}
} // namespace

TEST(ScalarCodeEmit, LessThanEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::lt(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("static_cast<double>"), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find(" < "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, GreaterThanEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::gt(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" > "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, LessEqualEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::le(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" <= "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, GreaterEqualEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::ge(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" >= "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, EqualEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::eq(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" == "), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, NotEqualEmitsStaticCastDouble) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::ne(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" != "), std::string::npos) << "got: " << rendered;
}

// ─── Piecewise nodes ──────────────────────────────────────────────────

TEST(ScalarCodeEmit, MaxEmitsStdMax) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::max(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("std::max"), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, MinEmitsStdMin) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto [x, y] = make_two_scalars(ctx);
  emit.apply(cas::min(x, y));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("std::min"), std::string::npos) << "got: " << rendered;
}

TEST(ScalarCodeEmit, IfThenElseEmitsTernary) {
  CodeGenContext ctx;
  ScalarCodeEmit emit(ctx);
  auto x = cas::make_expression<cas::scalar>("x");
  auto a = cas::make_expression<cas::scalar>("a");
  auto b = cas::make_expression<cas::scalar>("b");
  ctx.register_symbol_scalar(x, "x");
  ctx.register_symbol_scalar(a, "a");
  ctx.register_symbol_scalar(b, "b");

  // if_then_else(x, a, b) — the condition is a scalar that's compared != 0.
  emit.apply(cas::if_then_else(cas::lt(x, cas::get_scalar_zero()), a, b));
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find(" != 0.0 ? "), std::string::npos)
      << "got: " << rendered;
  EXPECT_NE(rendered.find(" : "), std::string::npos) << "got: " << rendered;
}

} // namespace numsim::codegen
