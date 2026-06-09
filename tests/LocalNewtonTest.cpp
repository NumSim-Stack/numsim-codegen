// Phase 3a-2 tests (issue #75): LocalNewtonLoweringPass — in-function
// scalar Newton iteration.
//
// Structural checks live here; the NUMERICAL end-to-end check (compile +
// call the emitted function, assert convergence to the analytic fixed
// point) lives in tests/generated/compile_check_driver.cpp where a real
// compiler + runtime are available.

#include <numsim_codegen/numsim_codegen.h>
#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/local_newton_lowering_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/pass_tags.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>

#include <numsim_cas/scalar/scalar_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

namespace {
auto build_hardening_newton(NewtonOptions opts = {}) -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("Hardening");
  auto K = m.add_parameter("K", 1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_output("sigma_y", K * alpha.current, roles::Stress);
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  m.enable_local_newton(opts);
  return m;
}
} // namespace

TEST(LocalNewton, OptInFlagDefaultsOff) {
  ConstitutiveModel m("E");
  EXPECT_FALSE(m.local_newton_enabled());
  m.enable_local_newton();
  EXPECT_TRUE(m.local_newton_enabled());
}

TEST(LocalNewton, NewtonOptionsRoundTrip) {
  ConstitutiveModel m("E");
  m.enable_local_newton(NewtonOptions{.tol = 1e-8, .max_iter = 25});
  EXPECT_DOUBLE_EQ(m.newton_options().tol, 1e-8);
  EXPECT_EQ(m.newton_options().max_iter, 25);
}

TEST(LocalNewton, PassAdvertisesNewtonLoweredPostcondition) {
  LocalNewtonLoweringPass pass;
  auto const post = pass.postconditions();
  bool found = false;
  for (auto const &t : post) {
    if (t == pass_tags::newton_lowered) found = true;
  }
  EXPECT_TRUE(found);
}

TEST(LocalNewton, PassRequiresStateVariableLoweringInputs) {
  // LNLP must NOT require jacobian_emitted (LocalJacobianPass does not run
  // in the Newton pipeline) — it builds the Jacobian itself. Registering
  // it after only SymbolValidationPass must succeed (the bundle is
  // satisfied); it must NOT depend on a jacobian tag that nobody emits.
  auto m = build_hardening_newton();
  PassContext pctx{RecipeView{m}, CodeGenContext{}, std::nullopt, {}, {}};
  PassManager pm;
  pm.emplace<SymbolValidationPass>();
  pm.emplace<LocalNewtonLoweringPass>();
  EXPECT_NO_THROW(pm.run(pctx));
  // One segment recorded for the single evolution equation.
  ASSERT_EQ(pctx.newton_segments.size(), 1u);
  EXPECT_EQ(pctx.newton_segments[0].state_var_name, "alpha");
  EXPECT_EQ(pctx.newton_segments[0].max_iter, 50);
}

TEST(LocalNewton, EmitReplacesResidualJacobianOutputsWithLoop) {
  auto const src = StandaloneCxxTarget{}.emit(build_hardening_newton()).at(0).contents;

  // The Newton loop is present.
  EXPECT_NE(src.find("for (int alpha_iter"), std::string::npos) << src;
  EXPECT_NE(src.find("alpha -= alpha_R / alpha_J"), std::string::npos) << src;
  EXPECT_NE(src.find("double alpha = alpha_old"), std::string::npos) << src;

  // The state variable is an OUT-param, not a `double const` input (#60).
  EXPECT_NE(src.find("double &alpha_out"), std::string::npos) << src;
  EXPECT_EQ(src.find("double const alpha,"), std::string::npos) << src;

  // R/J are loop-local, NOT outputs.
  EXPECT_EQ(src.find("alpha_residual_out"), std::string::npos) << src;
  EXPECT_EQ(src.find("alpha_jacobian_out"), std::string::npos) << src;

  // A genuine output still emits, referencing the converged local alpha.
  EXPECT_NE(src.find("sigma_y_out"), std::string::npos) << src;
}

TEST(LocalNewton, MaxIterReflectedInGeneratedLoop) {
  auto const src =
      StandaloneCxxTarget{}
          .emit(build_hardening_newton(NewtonOptions{.tol = 1e-12, .max_iter = 7}))
          .at(0)
          .contents;
  EXPECT_NE(src.find("alpha_iter < 7;"), std::string::npos) << src;
  EXPECT_NE(src.find("< 1e-12)"), std::string::npos) << src;
}

TEST(LocalNewton, DefaultPipelineStillEmitsResidualJacobianOutputs) {
  // Without enable_local_newton, the external-driver shape is unchanged:
  // R/J outputs present, no loop.
  using namespace numsim::cas;
  ConstitutiveModel m("Hardening");
  auto K = m.add_parameter("K", 1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  // NOT enabling local newton.
  auto const src = StandaloneCxxTarget{}.emit(m).at(0).contents;
  EXPECT_NE(src.find("alpha_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  EXPECT_EQ(src.find("for (int alpha_iter"), std::string::npos) << src;
}

TEST(LocalNewton, MultipleEvolutionEquationsGetSeparateLoops) {
  using namespace numsim::cas;
  ConstitutiveModel m("Multi");
  auto K = m.add_parameter("K", 1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  auto b = m.add_scalar_state_variable("b", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  m.add_scalar_evolution_equation(b, K * b.current);
  m.enable_local_newton();
  auto const src = StandaloneCxxTarget{}.emit(m).at(0).contents;
  EXPECT_NE(src.find("for (int a_iter"), std::string::npos) << src;
  EXPECT_NE(src.find("for (int b_iter"), std::string::npos) << src;
  EXPECT_NE(src.find("double &a_out"), std::string::npos) << src;
  EXPECT_NE(src.find("double &b_out"), std::string::npos) << src;
  // Independent equations must NOT be coupled into a dense Eigen solve.
  EXPECT_EQ(src.find("Eigen::Matrix"), std::string::npos) << src;
  EXPECT_FALSE(has_coupled_local_newton(m));
}

// ─── Phase 3b-2b: coupled vector Newton ─────────────────────────────

// Two evolution equations whose rates reference each other's unknown form a
// COUPLED 2×2 system — solved as ONE dense Newton loop via Eigen, not two
// independent scalar reciprocals.
namespace {
auto build_coupled_pair() -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("Coupled");
  auto K = m.add_parameter("K", 1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  auto b = m.add_scalar_state_variable("b", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * b.current); // a' depends on b
  m.add_scalar_evolution_equation(b, K * a.current); // b' depends on a
  m.enable_local_newton();
  return m;
}
} // namespace

TEST(LocalNewton, CoupledSystemDetected) {
  EXPECT_TRUE(has_coupled_local_newton(build_coupled_pair()));
}

TEST(LocalNewton, CoupledSystemEmitsSingleEigenSolve) {
  auto const src = build_coupled_pair().emit_compute_function();
  // ONE Newton loop (the first unknown drives the prefix), not two.
  EXPECT_NE(src.find("for (int a_iter"), std::string::npos) << src;
  EXPECT_EQ(src.find("for (int b_iter"), std::string::npos) << src;
  // Dense residual vector + Jacobian matrix + partial-pivot LU solve.
  EXPECT_NE(src.find("Eigen::Matrix<double, 2, 1> a_r"), std::string::npos)
      << src;
  EXPECT_NE(src.find("Eigen::Matrix<double, 2, 2> a_J"), std::string::npos)
      << src;
  EXPECT_NE(src.find("a_J.partialPivLu().solve(a_r)"), std::string::npos)
      << src;
  // Both unknowns updated from the solution vector and written back.
  EXPECT_NE(src.find("a -= a_dx(0)"), std::string::npos) << src;
  EXPECT_NE(src.find("b -= a_dx(1)"), std::string::npos) << src;
  EXPECT_NE(src.find("a_out = a"), std::string::npos) << src;
  EXPECT_NE(src.find("b_out = b"), std::string::npos) << src;
  // NOT the scalar-reciprocal path.
  EXPECT_EQ(src.find("a -= a_R / a_J"), std::string::npos) << src;
}

TEST(LocalNewton, CoupledSystemJacobianIsCrossCoupled) {
  // R_a = (a-a_old)/dt - K*b ⇒ ∂R_a/∂b = -K (the off-diagonal coupling term).
  auto const src = build_coupled_pair().emit_compute_function();
  ASSERT_NE(src.find("a_J << "), std::string::npos) << src;
  auto const fill = src.substr(src.find("a_J << "));
  EXPECT_NE(fill.find("-K"), std::string::npos) << fill.substr(0, 80);
}

TEST(LocalNewton, CoupledStandaloneEmitsEigenInclude) {
  auto const src =
      StandaloneCxxTarget{}.emit(build_coupled_pair()).at(0).contents;
  EXPECT_NE(src.find("#include <Eigen/Dense>"), std::string::npos) << src;
}

TEST(LocalNewton, UncoupledRecipeEmitsNoEigenInclude) {
  using namespace numsim::cas;
  ConstitutiveModel m("Indep");
  auto K = m.add_parameter("K", 1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  auto b = m.add_scalar_state_variable("b", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  m.add_scalar_evolution_equation(b, K * b.current);
  m.enable_local_newton();
  auto const src = StandaloneCxxTarget{}.emit(m).at(0).contents;
  EXPECT_EQ(src.find("Eigen/Dense"), std::string::npos) << src;
}

} // namespace numsim::codegen
