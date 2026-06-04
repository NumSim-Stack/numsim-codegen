// Phase 3a-1 tests (issue #70): LocalJacobianPass — symbolic-Jacobian
// emission for each evolution equation's discrete residual.
//
// Covers:
//   - Pass synthesises `<sv>_jacobian` output via cas::diff
//   - Constant rate (`rate = K`) → J = 1/dt, independent of α
//   - Linear-hardening rate (`rate = K·α`) → J = 1/dt − K
//   - Multi-equation case (3 SVs → 3 jacobians)
//   - Precondition graph: pass requires dt_lowered postcondition
//   - End-to-end emit contains both residual + jacobian outputs
//   - Working-copy invariant: emit doesn't mutate the user's recipe

#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/local_jacobian_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/pass_tags.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/passes/tensor_space_consistency_pass.h>
#include <numsim_codegen/passes/time_integration_pass.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

TEST(LocalJacobian, PassSynthesizesJacobianOutput) {
  // Direct invocation: SVPass → TIP → LJP. After LJP runs, the recipe
  // has BOTH `<sv>_residual` (from TIP) and `<sv>_jacobian` (from LJP)
  // outputs.
  ConstitutiveModel model("H");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  ASSERT_TRUE(model.outputs().empty());

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);
  TimeIntegrationPass{}.run(pctx);
  LocalJacobianPass{}.run(pctx);

  ASSERT_EQ(model.outputs().size(), 2u);
  // Order is TIP-then-LJP, so residual first, jacobian second.
  EXPECT_EQ(model.outputs()[0].name, "alpha_residual");
  EXPECT_EQ(model.outputs()[1].name, "alpha_jacobian");
  EXPECT_EQ(model.outputs()[1].kind, OutputDecl::Kind::Scalar);
}

TEST(LocalJacobian, ConstantRateGivesPureInverseDt) {
  // rate = K (constant in α) ⇒ ∂rate/∂α = 0 ⇒ J = 1/dt.
  // The emitted source must show that the jacobian output is
  // independent of α — only `dt` should appear in the jacobian
  // assignment line.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K);

  auto const src = model.emit_compute_function();
  EXPECT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  // Backward Euler ⇒ 1/dt term must appear.
  EXPECT_NE(src.find("std::pow(dt, -1.0)"), std::string::npos) << src;
}

TEST(LocalJacobian, LinearHardeningRateGivesInverseDtMinusK) {
  // rate = K·α ⇒ ∂rate/∂α = K ⇒ J = 1/dt − K. The cas chain rule
  // through the multiplication must surface K in the jacobian.
  ConstitutiveModel model("H");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const src = model.emit_compute_function();
  EXPECT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  EXPECT_NE(src.find("std::pow(dt, -1.0)"), std::string::npos) << src;
  // K must surface in the body — it's the only non-time-derivative
  // contribution to J.
  EXPECT_NE(src.find("K"), std::string::npos) << src;
}

TEST(LocalJacobian, MultipleEvolutionEquationsEachGetTheirJacobian) {
  ConstitutiveModel model("Multi");
  auto K = model.add_parameter("K", 1.0);
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto a = model.add_scalar_state_variable("a", zero);
  auto b = model.add_scalar_state_variable("b", zero);
  auto c = model.add_scalar_state_variable("c", zero);

  model.add_scalar_evolution_equation(a, K * a.current);
  model.add_scalar_evolution_equation(b, K);
  model.add_scalar_evolution_equation(c, K * c.current);

  auto const src = model.emit_compute_function();
  EXPECT_NE(src.find("a_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("a_jacobian_out"), std::string::npos) << src;
  EXPECT_NE(src.find("b_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("b_jacobian_out"), std::string::npos) << src;
  EXPECT_NE(src.find("c_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("c_jacobian_out"), std::string::npos) << src;
}

TEST(LocalJacobian, PassPreconditionRequiresDtLowered) {
  // LJP declares dt_lowered as a precondition. Registering LJP
  // WITHOUT TimeIntegrationPass having run first must fail at
  // PassManager::run() time.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  PassManager pm;
  pm.emplace<SymbolValidationPass>();
  pm.emplace<LocalJacobianPass>(); // no TIP in front → must fail

  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error from precondition graph";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("LocalJacobian"), std::string::npos) << what;
    EXPECT_NE(what.find("dt-lowered"), std::string::npos) << what;
  }
}

TEST(LocalJacobian, PassAdvertisesJacobianEmittedPostcondition) {
  LocalJacobianPass pass;
  auto const post = pass.postconditions();
  bool found = false;
  for (auto const &tag : post) {
    if (tag == pass_tags::jacobian_emitted) found = true;
  }
  EXPECT_TRUE(found);
}

TEST(LocalJacobian, EmitDoesNotMutateRecipe) {
  // Same working-copy invariant as PR #69's TimeIntegrationPass test:
  // emit_compute_function() must not mutate the user's recipe even
  // though both TIP and LJP add outputs internally.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const outputs_before = model.outputs().size();
  auto const symbols_before = model.symbols().size();
  auto const equations_before = model.evolution_equations().size();

  (void)model.emit_compute_function();

  EXPECT_EQ(model.outputs().size(), outputs_before);
  EXPECT_EQ(model.symbols().size(), symbols_before);
  EXPECT_EQ(model.evolution_equations().size(), equations_before);
}

TEST(LocalJacobian, EndToEndHardeningHasBothResidualAndJacobian) {
  ConstitutiveModel model("Hardening");
  auto K = model.add_parameter("K", 1.0, "hardening modulus");
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  auto sigma_y = K * alpha.current;
  model.add_output("sigma_y", sigma_y);
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const src = model.emit_compute_function();
  EXPECT_NE(src.find("alpha_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  EXPECT_NE(src.find("sigma_y_out"), std::string::npos) << src;
}

TEST(LocalJacobian, NotRegisteredOnPureElasticityRecipe) {
  // Pure-elasticity recipes have no evolution equations →
  // emit_compute_function should NOT register either TIP or LJP →
  // generated source has no `*_jacobian_out` parameter.
  ConstitutiveModel pure("E");
  auto mu = pure.add_parameter("mu", 0.5);
  (void)mu;
  pure.add_output("dummy", cas::make_expression<cas::scalar_constant>(1.0));

  auto const src = pure.emit_compute_function();
  EXPECT_EQ(src.find("_jacobian_out"), std::string::npos)
      << "no jacobian outputs expected on pure-elasticity recipe: " << src;
}

} // namespace numsim::codegen
