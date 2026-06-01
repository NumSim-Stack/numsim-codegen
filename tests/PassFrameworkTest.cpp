// Tests for the Phase 1.2 pass framework: PassManager ordering / precondition
// enforcement, and the two concrete passes (SymbolValidationPass with the new
// identifier-validity check; CodeEmitPass producing a non-empty compute
// function). End-to-end coverage that the framework integrates with the recipe
// emit path is already exercised by RecipeTest / MooseTargetTest / the
// compile-check driver — this file isolates the framework itself.

#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

namespace {

// Test-only pass that records its name into a shared trace vector. Used to
// verify that PassManager calls passes in registration order.
class TracePass final : public Pass {
public:
  TracePass(std::string n, std::vector<std::string> &trace,
            std::vector<std::string_view> pre,
            std::vector<std::string_view> post)
      : m_name(std::move(n)), m_trace(&trace), m_pre(std::move(pre)),
        m_post(std::move(post)) {}

  [[nodiscard]] auto name() const -> std::string_view override {
    return m_name;
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return m_pre;
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return m_post;
  }
  void run(PassContext & /*pctx*/) override { m_trace->push_back(m_name); }

private:
  std::string m_name;
  std::vector<std::string> *m_trace;
  std::vector<std::string_view> m_pre;
  std::vector<std::string_view> m_post;
};

// Build a recipe shell that satisfies SymbolValidationPass — no outputs,
// trivially valid. Some tests fabricate a PassContext directly; this gives
// them a minimal `ConstitutiveModel const&` to reference.
auto make_empty_model() -> ConstitutiveModel { return ConstitutiveModel("M"); }

} // namespace

// ─── PassManager ─────────────────────────────────────────────────────

TEST(PassManager, RunsPassesInRegistrationOrder) {
  std::vector<std::string> trace;
  PassManager pm;
  pm.emplace<TracePass>("A", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});
  pm.emplace<TracePass>("B", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});
  pm.emplace<TracePass>("C", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{model, CodeGenContext{}, std::nullopt};
  pm.run(pctx);

  ASSERT_EQ(trace.size(), 3u);
  EXPECT_EQ(trace[0], "A");
  EXPECT_EQ(trace[1], "B");
  EXPECT_EQ(trace[2], "C");
}

TEST(PassManager, EnforcesPrecondition) {
  std::vector<std::string> trace;
  PassManager pm;
  // Pass that needs "validated" before running, but no earlier pass advertises
  // it.
  pm.emplace<TracePass>("Bad", trace,
                        std::vector<std::string_view>{"validated"},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{model, CodeGenContext{}, std::nullopt};
  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error for unmet precondition";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("validated"), std::string::npos) << msg;
    EXPECT_NE(msg.find("Bad"), std::string::npos) << msg;
  }
  EXPECT_TRUE(trace.empty()) << "Bad pass should not have run.";
}

TEST(PassManager, SatisfiedPreconditionAllowsRun) {
  std::vector<std::string> trace;
  PassManager pm;
  pm.emplace<TracePass>(
      "A", trace, std::vector<std::string_view>{},
      std::vector<std::string_view>{"feature-x"});
  pm.emplace<TracePass>("B", trace,
                        std::vector<std::string_view>{"feature-x"},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{model, CodeGenContext{}, std::nullopt};
  pm.run(pctx);

  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0], "A");
  EXPECT_EQ(trace[1], "B");
}

TEST(PassManager, RejectsNullPass) {
  PassManager pm;
  EXPECT_THROW(pm.add(nullptr), std::invalid_argument);
}

// ─── SymbolValidationPass — identifier validity ──────────────────────

TEST(SymbolValidationPass, AcceptsValidIdentifiers) {
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("x"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("eps_p"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("_internal"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("a1b2c3"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("CamelCase"));
}

TEST(SymbolValidationPass, RejectsInvalidIdentifiers) {
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier(""));
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("1var"))
      << "digit-leading";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("my-tensor"))
      << "hyphen";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("x.y"))
      << "dot";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("a b"))
      << "space";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("a$b"))
      << "dollar (compiler-extension, not std-portable)";
}

TEST(SymbolValidationPass, RejectsRecipeWithInvalidScalarName) {
  // Use the cas tensor name-only-as-identifier-bearing field directly. The
  // recipe.h add_* methods take a `name` argument that becomes the cas
  // symbol's printable name — passing "my-param" here exercises the
  // identifier check at validate() time.
  ConstitutiveModel model("M");
  model.add_parameter("my-param", 1.0, "bad name on purpose");

  try {
    model.validate();
    FAIL() << "expected runtime_error for invalid scalar identifier";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("my-param"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not a valid C++ identifier"), std::string::npos) << msg;
  }
}

TEST(SymbolValidationPass, RejectsRecipeWithInvalidTensorName) {
  ConstitutiveModel model("M");
  model.add_tensor_input("eps.bad", 3, 2, roles::Strain);

  try {
    model.validate();
    FAIL() << "expected runtime_error for invalid tensor identifier";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("eps.bad"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not a valid C++ identifier"), std::string::npos) << msg;
  }
}

TEST(SymbolValidationPass, PreservesMissingSymbolDiagnostic) {
  // Pre-1.2 validate() threw on outputs referencing undeclared symbols.
  // The pass framework must preserve that behaviour.
  ConstitutiveModel model("M");
  // Output references a symbol that was never declared on the model.
  auto x = cas::make_expression<cas::scalar>("undeclared_x");
  model.add_output("y", x, roles::Other);

  try {
    model.validate();
    FAIL() << "expected runtime_error for undeclared symbol";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("undeclared_x"), std::string::npos) << msg;
  }
}

// ─── CodeEmitPass ────────────────────────────────────────────────────

TEST(CodeEmitPass, ProducesNonEmptyComputeFunctionSource) {
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);

  auto src = model.emit_compute_function();
  EXPECT_NE(src.find("LinearElasticShear_compute"), std::string::npos)
      << "got:\n" << src;
  EXPECT_NE(src.find("stress_out"), std::string::npos) << src;
}

TEST(CodeEmitPass, RefusesToRunWithoutValidationPredecessor) {
  // Direct PassManager invocation without registering SymbolValidationPass
  // should fail loudly — CodeEmitPass advertises preconditions that are
  // unmet.
  ConstitutiveModel model("M");
  PassContext pctx{model, CodeGenContext{}, std::nullopt};
  PassManager pm;
  pm.emplace<CodeEmitPass>();

  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error for missing SymbolValidationPass";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("CodeEmit"), std::string::npos) << msg;
    // Either of CodeEmitPass's two preconditions can show up in the message —
    // both are produced only by SymbolValidationPass.
    bool mentions_pre =
        msg.find("symbols-declared") != std::string::npos ||
        msg.find("identifiers-valid") != std::string::npos;
    EXPECT_TRUE(mentions_pre) << msg;
  }
}

} // namespace numsim::codegen
