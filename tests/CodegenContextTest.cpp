#include <numsim_codegen/code_emit/codegen_context.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

TEST(CodegenContext, EmptyContextRendersNoStatements) {
  CodeGenContext ctx;
  EXPECT_EQ(ctx.render_statements(), "");
  EXPECT_EQ(ctx.temporary_count(), 0u);
}

TEST(CodegenContext, EmitTemporaryAllocatesFreshName) {
  CodeGenContext ctx;
  int dummy = 0;
  auto name = ctx.emit_temporary(&dummy, "1.0", "double");
  EXPECT_EQ(name, "t0");
  EXPECT_EQ(ctx.temporary_count(), 1u);
}

TEST(CodegenContext, EmitTemporaryAllocatesSequentialNames) {
  CodeGenContext ctx;
  int dummy_a = 0, dummy_b = 0, dummy_c = 0;
  EXPECT_EQ(ctx.emit_temporary(&dummy_a, "1.0", "double"), "t0");
  EXPECT_EQ(ctx.emit_temporary(&dummy_b, "2.0", "double"), "t1");
  EXPECT_EQ(ctx.emit_temporary(&dummy_c, "3.0", "double"), "t2");
}

TEST(CodegenContext, RenderStatementsProducesAutoBindings) {
  CodeGenContext ctx;
  int a = 0, b = 0;
  ctx.emit_temporary(&a, "1.0 + 2.0", "double");
  ctx.emit_temporary(&b, "t0 * t0", "double");
  auto rendered = ctx.render_statements();
  EXPECT_NE(rendered.find("auto t0 = 1.0 + 2.0;"), std::string::npos);
  EXPECT_NE(rendered.find("auto t1 = t0 * t0;"), std::string::npos);
}

TEST(CodegenContext, FindReturnsNullForUnknownPointer) {
  CodeGenContext ctx;
  int dummy = 0;
  EXPECT_EQ(ctx.find(&dummy), nullptr);
}

TEST(CodegenContext, FindReturnsRegisteredName) {
  CodeGenContext ctx;
  int dummy = 0;
  ctx.emit_temporary(&dummy, "1.0", "double");
  auto *found = ctx.find(&dummy);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(*found, "t0");
}

TEST(CodegenContext, ResetClearsStatementsAndCseTable) {
  CodeGenContext ctx;
  int dummy = 0;
  ctx.emit_temporary(&dummy, "1.0", "double");
  EXPECT_EQ(ctx.temporary_count(), 1u);
  ctx.reset();
  EXPECT_EQ(ctx.temporary_count(), 0u);
  EXPECT_EQ(ctx.find(&dummy), nullptr);
}

} // namespace numsim::codegen
