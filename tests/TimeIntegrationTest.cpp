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

#include <cmath>

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

TEST(TimeIntegration, AddEvolutionEquationRejectsManuallyConstructedHandle) {
  // PR #69 round-1 #3: a Handle without a `model_token` (manually
  // constructed) is rejected before any name lookup happens. Defends
  // against accidental hand-rolled Handles.
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  (void)model.add_scalar_state_variable("alpha", zero);

  auto beta = cas::make_expression<cas::scalar>("beta_unrelated");
  ConstitutiveModel::ScalarStateVariableHandle foreign{beta, beta};
  // foreign.model_token defaults to nullptr.

  auto rate = cas::make_expression<cas::scalar_constant>(0.5);
  try {
    model.add_scalar_evolution_equation(foreign, rate);
    FAIL() << "expected runtime_error for manually-constructed handle";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("no model token"), std::string::npos) << what;
  }
}

TEST(TimeIntegration, AddEvolutionEquationRejectsHandleFromDifferentModel) {
  // PR #69 round-1 #3 (cross-recipe hijack): two models BOTH have a
  // state variable named "alpha". Without the model_token defense the
  // call would silently bind modelA's handle to modelB's state
  // variable (name collision). With the token, it throws.
  ConstitutiveModel model_a("A");
  ConstitutiveModel model_b("B");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha_a = model_a.add_scalar_state_variable("alpha", zero);
  (void)model_b.add_scalar_state_variable("alpha", zero);

  auto rate = cas::make_expression<cas::scalar_constant>(0.5);
  try {
    model_b.add_scalar_evolution_equation(alpha_a, rate);
    FAIL() << "expected runtime_error for handle from different model";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("different ConstitutiveModel"), std::string::npos)
        << what;
  }
}

TEST(TimeIntegration, AddEvolutionEquationRejectsRateWithUndeclaredLeaves) {
  // PR #69 round-1 #4: rate expression that references an undeclared
  // symbol must throw at add-time, not silently ship to TimeIntegrationPass.
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);

  // Build a rate that references an undeclared parameter "K".
  auto K_unregistered = cas::make_expression<cas::scalar>("K");
  auto bad_rate = K_unregistered * alpha.current;

  try {
    model.add_scalar_evolution_equation(alpha, bad_rate);
    FAIL() << "expected runtime_error for rate with undeclared leaf";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("undeclared"), std::string::npos) << what;
    EXPECT_NE(what.find("'K'"), std::string::npos) << what;
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

TEST(TimeIntegration, PassAdvertisesBackwardEulerResidualEmittedPostcondition) {
  TimeIntegrationPass pass;
  auto const post = pass.postconditions();
  bool found = false;
  for (auto const &tag : post) {
    if (tag == pass_tags::backward_euler_residual_emitted) found = true;
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

// ─── PR #69 round-1 review fixups ────────────────────────────────────

TEST(TimeIntegration, EmitDoesNotMutateRecipe) {
  // PR #69 round-1 #5: working-copy invariant. emit_compute_function
  // claims the user's recipe is unchanged after emit (mutation lives
  // only on a working copy). Pin that as a test contract — without it
  // a future refactor that drops the copy could regress silently.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const outputs_before = model.outputs().size();
  auto const symbols_before = model.symbols().size();
  auto const equations_before = model.evolution_equations().size();

  (void)model.emit_compute_function();

  EXPECT_EQ(model.outputs().size(), outputs_before)
      << "emit must not mutate the user's recipe (outputs grew)";
  EXPECT_EQ(model.symbols().size(), symbols_before)
      << "emit must not mutate the user's recipe (symbols grew)";
  EXPECT_EQ(model.evolution_equations().size(), equations_before)
      << "emit must not mutate the user's recipe (equations grew)";
}

TEST(TimeIntegration, MultipleEvolutionEquationsAllLowered) {
  // PR #69 round-1 #6: N>1 equations. Single-equation tests can't
  // catch loop bugs (off-by-one, stale handles across iterations).
  ConstitutiveModel model("Multi");
  auto K = model.add_parameter("K", 1.0);
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto a = model.add_scalar_state_variable("a", zero);
  auto b = model.add_scalar_state_variable("b", zero);
  auto c = model.add_scalar_state_variable("c", zero);

  model.add_scalar_evolution_equation(a, K * a.current);
  model.add_scalar_evolution_equation(b, K * b.current);
  model.add_scalar_evolution_equation(c, K);

  auto const src = model.emit_compute_function();
  EXPECT_NE(src.find("a_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("b_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("c_residual_out"), std::string::npos) << src;
}

TEST(TimeIntegration, ResidualExpressionStructure) {
  // PR #69 round-1 #7: pin the SEMANTIC shape of the synthesized
  // residual, not just substring presence.
  //
  // The CAS layer canonicalizes operators: `a - b` → `a + (-b)`,
  // `x / y` → `x * pow(y, -1.0)`. So the generated body for the
  // backward-Euler residual `(α − α_old)/dt − rate` lands as:
  //
  //   auto t0 = K * alpha;            // rate
  //   auto t1 = std::pow(dt, -1.0);   // 1/dt
  //   auto t2 = alpha + -alpha_old;   // α − α_old
  //   auto t3 = t1 * t2;              // (α − α_old) · 1/dt
  //   auto t4 = -t0 + t3;             // residual − rate
  //   alpha_residual_out = t4;
  //
  // The KEY invariants we pin are:
  //   (a) `pow(dt, -1.0)` appears — confirms BACKWARD Euler
  //       (forward Euler would multiply by `dt`, not `dt^-1`);
  //   (b) The state-var and its `_old` both appear in the body —
  //       confirms the discrete derivative is structurally present.
  //
  // Sign-flip regressions (`α_old + -α` instead of `α + -α_old`) are
  // a known weakness of this textual check but are unlikely without
  // an explicit CAS-layer breakage; numerical verification belongs
  // in the Phase 4 driver suite.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  // Self-referential rate — the J2 plasticity shape.
  model.add_scalar_evolution_equation(alpha, K * alpha.current);

  auto const src = model.emit_compute_function();

  // (a) Backward Euler ⇒ dt^-1 must appear.
  EXPECT_NE(src.find("std::pow(dt, -1.0)"), std::string::npos)
      << "expected `std::pow(dt, -1.0)` (backward-Euler 1/dt): " << src;

  // (b) Both α and α_old must appear in the body.
  EXPECT_NE(src.find("alpha"), std::string::npos);
  EXPECT_NE(src.find("alpha_old"), std::string::npos);

  // Negative pin: forward Euler would emit `std::pow(dt, 1.0)` or
  // direct `* dt` multiplication on the discrete-derivative path.
  // The latter is harder to assert textually (dt appears in the
  // signature too), but the `std::pow(dt, 1.0)` shape is a clean
  // marker. Assert it's absent.
  EXPECT_EQ(src.find("std::pow(dt, 1.0)"), std::string::npos)
      << "residual must NOT use dt^+1 (would be forward Euler): " << src;
}

TEST(TimeIntegration, DtAutoRegisterDefaultIsNaN) {
  // PR #69 round-1 #2: the auto-registered `dt` parameter has
  // default value NaN — a forgotten dt-wiring then propagates NaN
  // through the residual rather than producing Inf (default-0.0 would).
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);
  model.add_scalar_evolution_equation(alpha, zero);

  bool found_dt = false;
  for (auto const &s : model.symbols()) {
    if (s.name == state_time_step_name) {
      found_dt = true;
      ASSERT_TRUE(s.default_value.has_value());
      EXPECT_TRUE(std::isnan(*s.default_value))
          << "auto-registered dt default must be NaN, not "
          << *s.default_value;
    }
  }
  EXPECT_TRUE(found_dt);
}

} // namespace numsim::codegen
