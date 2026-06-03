// Phase 2.2 tests (issue #68): TimeIntegrationPass + evolution-equation IR.
//
// Covers:
//   - add_scalar_evolution_equation API surface (success + name-mismatch)
//   - dt parameter auto-registration on first evolution equation
//   - TimeIntegrationPass synthesises `<sv>_residual` outputs via the
//     mutable RecipeView (first real consumer of require_mutable_model)
//   - PassManager refuses to advance past TimeIntegrationPass on a
//     pure-elasticity recipe (precondition mismatch)
//   - End-to-end: emit_compute_function on a hardening recipe contains
//     the synthesized residual output

#include <numsim_codegen/passes/code_emit_pass.h>
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

// ─── add_scalar_evolution_equation ──────────────────────────────────

TEST(TimeIntegration, AddScalarEvolutionEquationRegistersDtAndEquation) {
  ConstitutiveModel model("H");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);
  auto K = model.add_parameter("K", 1.0);
  (void)K;

  // Before: no dt symbol, no evolution equations.
  EXPECT_TRUE(model.evolution_equations().empty());
  bool dt_present = false;
  for (auto const &s : model.symbols()) {
    if (s.name == state_time_step_name) dt_present = true;
  }
  EXPECT_FALSE(dt_present);

  auto lambda = cas::make_expression<cas::scalar_constant>(0.5);
  model.add_scalar_evolution_equation(alpha, lambda, "linear hardening");

  // After: one equation + dt auto-registered as a parameter.
  ASSERT_EQ(model.evolution_equations().size(), 1u);
  EXPECT_EQ(model.evolution_equations()[0].state_variable_idx, 0u);
  EXPECT_EQ(model.evolution_equations()[0].doc, "linear hardening");

  // dt should be a registered parameter now.
  for (auto const &s : model.symbols()) {
    if (s.name == state_time_step_name) {
      dt_present = true;
      EXPECT_EQ(s.category, SymbolDecl::Category::Parameter);
      EXPECT_EQ(s.kind, SymbolDecl::Kind::Scalar);
    }
  }
  EXPECT_TRUE(dt_present);
}

TEST(TimeIntegration, AddScalarEvolutionEquationReusesExistingDt) {
  // If the user has already registered "dt" as their own parameter,
  // the evolution-equation auto-register MUST NOT add a duplicate
  // (which would throw via assert_symbol_name_available).
  ConstitutiveModel model("H");
  auto user_dt = model.add_parameter("dt", 0.001, "user-supplied dt");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);
  (void)user_dt;

  auto lambda = cas::make_expression<cas::scalar_constant>(0.5);
  EXPECT_NO_THROW(
      model.add_scalar_evolution_equation(alpha, std::move(lambda)));

  // Exactly one dt symbol — not two.
  int dt_count = 0;
  for (auto const &s : model.symbols()) {
    if (s.name == state_time_step_name) ++dt_count;
  }
  EXPECT_EQ(dt_count, 1);
}

TEST(TimeIntegration, AddEvolutionEquationRejectsUnregisteredHandle) {
  // Construct a handle whose underlying scalar wasn't registered as a
  // state variable on this model. The matching is name-based; an
  // unknown name must throw.
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  (void)model.add_scalar_state_variable("alpha", zero);

  // Build a fake handle that names something else.
  auto beta = cas::make_expression<cas::scalar>("beta_unrelated");
  ConstitutiveModel::ScalarStateVariableHandle foreign{beta, beta};

  auto rate = cas::make_expression<cas::scalar_constant>(0.5);
  try {
    model.add_scalar_evolution_equation(foreign, rate);
    FAIL() << "expected runtime_error for handle naming an unknown SV";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("beta_unrelated"), std::string::npos) << what;
    EXPECT_NE(what.find("no such state variable"), std::string::npos)
        << what;
  }
}

// ─── TimeIntegrationPass lowering ───────────────────────────────────

TEST(TimeIntegration, PassSynthesizesResidualOutput) {
  // Direct pass invocation against a hardening recipe. The pass must
  // mutate the recipe (via require_mutable_model) and add an output
  // named `<sv>_residual`.
  ConstitutiveModel model("H");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);
  auto K = model.add_parameter("K", 1.0);
  auto lambda = K * alpha.current; // rate = K*α
  model.add_scalar_evolution_equation(alpha, lambda);

  ASSERT_TRUE(model.outputs().empty())
      << "no outputs before pass runs";

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  // Run prerequisite SymbolValidationPass first so the alignment
  // check + symbol_lookup are populated (the pass framework's
  // precondition graph would otherwise refuse).
  SymbolValidationPass{}.run(pctx);
  TimeIntegrationPass{}.run(pctx);

  // Exactly one synthesized output: `alpha_residual`.
  ASSERT_EQ(model.outputs().size(), 1u);
  EXPECT_EQ(model.outputs()[0].name, "alpha_residual");
  EXPECT_EQ(model.outputs()[0].kind, OutputDecl::Kind::Scalar);
}

TEST(TimeIntegration, PassPreconditionFailsOnPureElasticity) {
  // No state variables → no `state_variables_non_empty` postcondition
  // from SymbolValidationPass → TimeIntegrationPass's precondition
  // fails at PassManager::run() time.
  ConstitutiveModel pure("E");
  (void)pure.add_parameter("mu", 0.5);

  PassContext pctx{RecipeView{pure}, CodeGenContext{}, std::nullopt, {}};
  PassManager pm;
  pm.emplace<SymbolValidationPass>();
  pm.emplace<TimeIntegrationPass>();

  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error from precondition graph";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("TimeIntegration"), std::string::npos) << what;
    EXPECT_NE(what.find("state-variables-non-empty"), std::string::npos)
        << what;
  }
}

TEST(TimeIntegration, PassAdvertisesDtLoweredPostcondition) {
  TimeIntegrationPass pass;
  auto const post = pass.postconditions();
  bool found = false;
  for (auto const &tag : post) {
    if (tag == pass_tags::dt_lowered) found = true;
  }
  EXPECT_TRUE(found);
}

// ─── End-to-end emit ────────────────────────────────────────────────

TEST(TimeIntegration, EndToEndHardeningRecipeEmitsResidualOutput) {
  // Build a recipe that uses backward-Euler hardening, run
  // emit_compute_function, verify the generated source contains the
  // synthesized residual output as an out-parameter and the discrete
  // residual expression in the body.
  ConstitutiveModel model("Hardening");
  auto K = model.add_parameter("K", 1.0, "hardening modulus");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);
  auto sigma_y = K * alpha.current;
  model.add_output("sigma_y", sigma_y);

  // Trivial rate λ = K so that α evolves linearly.
  auto rate = K;
  model.add_scalar_evolution_equation(alpha, rate);

  auto const src = model.emit_compute_function();

  // The function declares an out-parameter for the synthesized residual.
  EXPECT_NE(src.find("alpha_residual_out"), std::string::npos)
      << "expected `double &alpha_residual_out` in generated signature: "
      << src;
  // The body references alpha, alpha_old, dt — the backward-Euler
  // discretization touches all three.
  EXPECT_NE(src.find("alpha_old"), std::string::npos) << src;
  EXPECT_NE(src.find("dt"), std::string::npos) << src;
}

TEST(TimeIntegration, EmitOnPureElasticityRecipeIsUnchanged) {
  // emit_compute_function on a no-state-var recipe must still produce
  // valid output (the TimeIntegrationPass is NOT registered in this
  // path, so the precondition graph is satisfied).
  ConstitutiveModel pure("E");
  auto mu = pure.add_parameter("mu", 0.5);
  (void)mu;
  pure.add_output("dummy", cas::make_expression<cas::scalar_constant>(1.0));
  EXPECT_NO_THROW({
    auto src = pure.emit_compute_function();
    EXPECT_NE(src.find("E_compute"), std::string::npos) << src;
    EXPECT_EQ(src.find("_residual_out"), std::string::npos)
        << "no residual outputs expected on pure-elasticity recipe";
  });
}

} // namespace numsim::codegen
