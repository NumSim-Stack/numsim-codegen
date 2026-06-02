// Phase 2.1 tests: StateVariable IR additions to ConstitutiveModel.
//
// Covers:
//   - add_scalar_state_variable / add_tensor_state_variable basic shape
//   - Returned handle has distinct current / previous expression names
//   - Both current and previous register as SymbolDecls with the right
//     Category (StateVariableCurrent / StateVariableOld)
//   - state_variables() accessor exposes the declarations
//   - SymbolValidationPass discovers state-var symbols (identifier check,
//     symbol_lookup population)
//   - find_tensor_symbol resolves tensor state-var current and previous
//
// Phase 2.2+ (TimeIntegrationPass, KuhnTuckerLoweringPass, LocalNewton)
// will land separately and test the actual semantics of how state vars
// flow through codegen. This file only exercises the IR layer.

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

// ─── add_scalar_state_variable ──────────────────────────────────────

TEST(StateVariable, AddScalarReturnsCurrentAndPreviousHandles) {
  ConstitutiveModel model("J2Trial");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero,
                                               "equivalent plastic strain");

  // Two distinct expression handles: current and previous. Both are
  // scalar symbols with distinct printable names.
  ASSERT_TRUE(alpha.current.is_valid());
  ASSERT_TRUE(alpha.previous.is_valid());
  EXPECT_NE(&alpha.current.get(), &alpha.previous.get())
      << "current and previous must be distinct cas symbols";
}

TEST(StateVariable, AddScalarRegistersBothSymbolsInModel) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  // symbols() now contains alpha + alpha_old.
  ASSERT_EQ(model.symbols().size(), 2u);

  auto const &cur = model.symbols()[0];
  EXPECT_EQ(cur.name, "alpha");
  EXPECT_EQ(cur.category, SymbolDecl::Category::StateVariableCurrent);
  EXPECT_EQ(cur.kind, SymbolDecl::Kind::Scalar);

  auto const &old = model.symbols()[1];
  EXPECT_EQ(old.name, "alpha_old");
  EXPECT_EQ(old.category, SymbolDecl::Category::StateVariableOld);
  EXPECT_EQ(old.kind, SymbolDecl::Kind::Scalar);
}

TEST(StateVariable, ScalarStateVarAppearsInScalarSymbolMap) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  // Both alpha and alpha_old must be in the scalar symbol map so the
  // CodeGenContext can register their names for emit.
  ASSERT_EQ(model.scalar_symbol_map().size(), 2u);
  EXPECT_EQ(model.scalar_symbol_map()[0].first, "alpha");
  EXPECT_EQ(model.scalar_symbol_map()[1].first, "alpha_old");
}

// ─── add_tensor_state_variable ──────────────────────────────────────

TEST(StateVariable, AddTensorReturnsCurrentAndPreviousHandles) {
  ConstitutiveModel model("Damage");
  auto eps_p_init = cas::make_expression<cas::tensor_zero>(3, 2);
  auto eps_p =
      model.add_tensor_state_variable("eps_p", 3, 2, eps_p_init,
                                      "plastic strain tensor");

  ASSERT_TRUE(eps_p.current.is_valid());
  ASSERT_TRUE(eps_p.previous.is_valid());
  EXPECT_NE(&eps_p.current.get(), &eps_p.previous.get());
}

TEST(StateVariable, AddTensorRegistersBothSymbolsAndPreservesRank) {
  ConstitutiveModel model("M");
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);
  model.add_tensor_state_variable("eps_p", 3, 2, init);

  ASSERT_EQ(model.symbols().size(), 2u);
  EXPECT_EQ(model.symbols()[0].name, "eps_p");
  EXPECT_EQ(model.symbols()[0].dim, 3u);
  EXPECT_EQ(model.symbols()[0].rank, 2u);
  EXPECT_EQ(model.symbols()[1].name, "eps_p_old");
  EXPECT_EQ(model.symbols()[1].dim, 3u);
  EXPECT_EQ(model.symbols()[1].rank, 2u);
}

// ─── state_variables() accessor ─────────────────────────────────────

TEST(StateVariable, StateVariablesAccessorExposesDeclarations) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);

  model.add_scalar_state_variable("alpha", zero, "scalar SV");
  model.add_tensor_state_variable("eps_p", 3, 2, init, "tensor SV");

  ASSERT_EQ(model.state_variables().size(), 2u);
  EXPECT_EQ(model.state_variables()[0].name, "alpha");
  EXPECT_EQ(model.state_variables()[0].kind, SymbolDecl::Kind::Scalar);
  EXPECT_EQ(model.state_variables()[0].doc, "scalar SV");
  EXPECT_EQ(model.state_variables()[1].name, "eps_p");
  EXPECT_EQ(model.state_variables()[1].kind, SymbolDecl::Kind::Tensor);
  EXPECT_EQ(model.state_variables()[1].rank, 2u);
}

TEST(StateVariable, EmptyRecipeHasNoStateVariables) {
  ConstitutiveModel model("ElasticOnly");
  EXPECT_TRUE(model.state_variables().empty());
}

// ─── RecipeView delegate ────────────────────────────────────────────

TEST(StateVariable, RecipeViewDelegatesStateVariables) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  RecipeView view(model);
  ASSERT_EQ(view.state_variables().size(), 1u);
  EXPECT_EQ(view.state_variables()[0].name, "alpha");
}

// ─── Pass framework integration ─────────────────────────────────────

TEST(StateVariable, SymbolValidationPassDiscoversStateVarSymbols) {
  // After SymbolValidationPass runs, pctx.symbol_lookup must contain
  // both current and previous symbols. Phase 2 mutating passes will
  // consume this for expression rewrites (e.g. TimeIntegrationPass
  // lowering Dt(α) → (α − α_old) / dt needs to resolve both names).
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);

  EXPECT_TRUE(pctx.symbol_lookup.contains("alpha"));
  EXPECT_TRUE(pctx.symbol_lookup.contains("alpha_old"));
}

TEST(StateVariable, FindTensorSymbolResolvesTensorStateVarBothHandles) {
  ConstitutiveModel model("M");
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);
  model.add_tensor_state_variable("eps_p", 3, 2, init);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);

  auto const *cur = find_tensor_symbol(pctx, "eps_p");
  ASSERT_NE(cur, nullptr);
  EXPECT_EQ(cur->category, SymbolDecl::Category::StateVariableCurrent);

  auto const *old = find_tensor_symbol(pctx, "eps_p_old");
  ASSERT_NE(old, nullptr);
  EXPECT_EQ(old->category, SymbolDecl::Category::StateVariableOld);
}

TEST(StateVariable, StateVarUsableInOutputExpression) {
  // End-to-end: build a recipe that uses a state variable in an output
  // expression. The compute function must emit successfully (no
  // undeclared-symbol error from SymbolValidationPass).
  ConstitutiveModel model("Hardening");
  auto K = model.add_parameter("K", 1.0, "hardening modulus");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);

  // sigma_y = K * alpha (linear hardening on the current step's alpha)
  auto sigma_y = K * alpha.current;
  model.add_output("sigma_y", sigma_y);

  EXPECT_NO_THROW(model.emit_compute_function());
}

} // namespace numsim::codegen
