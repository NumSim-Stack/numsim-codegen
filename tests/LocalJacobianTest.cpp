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

namespace {
// PR #71 round-1 #2: slice helper. The generated source emits all CSE
// temps first, then the output writes (`<sv>_residual_out = tN;`,
// `<sv>_jacobian_out = tM;`). Substring-matching the whole body can't
// distinguish "K appears in the residual" from "K appears in the
// Jacobian". This helper extracts the IMMEDIATE definition of the temp
// referenced by `<output>_out = tN;` so tests can assert on that line.
//
// Doesn't recurse into chained temps — for that, a Jacobian's nested
// `1/dt` factor lives in a separate `auto tK = std::pow(dt, -1.0)` line.
// Recursive expansion is over-engineering for the assertion shapes we
// care about today (presence/absence of K, alpha_old in the Jacobian's
// top-level RHS). If that changes, swap to a full chain-walker.
std::string immediate_def_for_output(std::string const &src,
                                     std::string const &out_name) {
  auto const out_idx = src.find(out_name + " = ");
  if (out_idx == std::string::npos) return {};
  auto const out_eq = src.find('=', out_idx);
  auto const out_semi = src.find(';', out_eq);
  auto const temp_id = src.substr(out_eq + 2, out_semi - out_eq - 2);
  auto const def_idx = src.find("auto " + temp_id + " =");
  if (def_idx == std::string::npos) return {};
  auto const def_semi = src.find(';', def_idx);
  return src.substr(def_idx, def_semi - def_idx);
}
} // namespace

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
  //
  // PR #71 round-1 #2: slice to the Jacobian's immediate definition.
  // The pattern is `alpha_jacobian_out = tN; auto tN = std::pow(dt, -1.0);`
  // — the Jacobian's RHS temp must define `std::pow(dt, -1.0)` and
  // NOT reference K (∂K/∂α = 0) nor alpha_old (independent leaf).
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K);

  auto const src = model.emit_compute_function();
  ASSERT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;

  auto const jac_def = immediate_def_for_output(src, "alpha_jacobian_out");
  ASSERT_FALSE(jac_def.empty())
      << "couldn't extract Jacobian's immediate def from: " << src;
  EXPECT_NE(jac_def.find("std::pow(dt, -1.0)"), std::string::npos)
      << "constant-rate Jacobian must be 1/dt: " << jac_def;
  EXPECT_EQ(jac_def.find("K"), std::string::npos)
      << "constant-rate Jacobian must NOT reference K (∂K/∂α = 0): "
      << jac_def;
  EXPECT_EQ(jac_def.find("alpha_old"), std::string::npos)
      << "Jacobian must be independent of alpha_old: " << jac_def;
}

TEST(LocalJacobian, LinearHardeningRateGivesInverseDtMinusK) {
  // rate = K·α ⇒ ∂rate/∂α = K ⇒ J = 1/dt − K.
  //
  // PR #71 round-1 #2: slice to the Jacobian's immediate def to PIN:
  //   (a) `-K` appears (sign of the chain-rule term)
  //   (b) The Jacobian's `+ <temp>` chains to a `std::pow(dt, -1.0)` temp
  //   (c) The Jacobian's immediate def does NOT contain `alpha_old`
  //       (cas::diff must treat current/old leaves as independent)
  ConstitutiveModel model("H");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const src = model.emit_compute_function();
  ASSERT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  EXPECT_NE(src.find("std::pow(dt, -1.0)"), std::string::npos) << src;

  auto const jac_def = immediate_def_for_output(src, "alpha_jacobian_out");
  ASSERT_FALSE(jac_def.empty())
      << "couldn't extract Jacobian's immediate def from: " << src;

  // (a) Sign: J = −K + 1/dt. The leading `-K` is the signature.
  EXPECT_NE(jac_def.find("-K"), std::string::npos)
      << "linear-hardening Jacobian must contain `-K` (chain-rule sign): "
      << jac_def;
  // Negative pin: a sign-flip regression that emitted `+K` (no minus)
  // would still find `K` in the slice. Pin the absence of the wrong shape:
  EXPECT_EQ(jac_def.find("+K +"), std::string::npos)
      << "Jacobian must NOT emit `+K` (would be sign-flip regression): "
      << jac_def;

  // (b) `alpha_old` MUST NOT appear in the Jacobian's immediate def —
  // cas::diff treats the separate `alpha_old` leaf as independent of α.
  EXPECT_EQ(jac_def.find("alpha_old"), std::string::npos)
      << "Jacobian must be independent of alpha_old "
      << "(cas::diff conflation regression?): " << jac_def;
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

TEST(LocalJacobian, PassPreconditionRequiresBackwardEulerResidualEmitted) {
  // LJP declares backward_euler_residual_emitted as a precondition.
  // Registering LJP WITHOUT TimeIntegrationPass having run first must
  // fail at PassManager::run() time. (PR #71 round-1 #4: per-scheme
  // tag, NOT the coarse legacy `dt_lowered`.)
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
    EXPECT_NE(what.find("backward-euler-residual-emitted"),
              std::string::npos)
        << what;
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
