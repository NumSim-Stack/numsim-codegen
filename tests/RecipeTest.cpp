#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

TEST(Recipe, ScalarRecipeProducesValidFunctionSignature) {
  ConstitutiveModel m("ScaleAdd");
  auto a = m.add_parameter("a", 2.0);
  auto b = m.add_parameter("b", 3.0);
  auto x = m.add_scalar_input("x");
  m.add_output("y", a * x + b);

  auto src = m.emit_compute_function();

  // Function name must match the model name.
  EXPECT_NE(src.find("void ScaleAdd_compute"), std::string::npos)
      << "got: " << src;
  // Parameters and inputs both appear as `const` doubles.
  EXPECT_NE(src.find("double const x"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("double const a"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("double const b"), std::string::npos) << "got: " << src;
  // Output is a non-const double reference.
  EXPECT_NE(src.find("double &y_out"), std::string::npos) << "got: " << src;
  // The final assignment writes to y_out.
  EXPECT_NE(src.find("y_out = "), std::string::npos) << "got: " << src;
}

TEST(Recipe, TensorRecipeProducesTemplatedSignature) {
  ConstitutiveModel m("Scale");
  auto k = m.add_parameter("k", 1.5);
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("y", k * eps);

  auto src = m.emit_compute_function();

  // Tensor arguments are template parameters so the caller can pass a
  // tmech::tensor, a tmech::adaptor, or any other tmech::tensor_base
  // subclass without forcing a materialised copy.
  EXPECT_NE(src.find("template <typename T0, typename T1>"),
            std::string::npos)
      << "got: " << src;
  EXPECT_NE(src.find("T0 const &eps"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("T1 &y_out"), std::string::npos) << "got: " << src;
}

TEST(Recipe, LinearElasticityShearTermEmitsSomething) {
  // Phase-A smoke test: σ = 2μ ε (shear term only). Full LE needs the
  // `t2s * tensor` operator that numsim-cas hasn't exposed yet.
  ConstitutiveModel m("LinearElasticShear");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2);

  auto sigma = 2 * mu * eps;
  m.add_output("stress", sigma);

  auto src = m.emit_compute_function();

  EXPECT_NE(src.find("mu"), std::string::npos) << "got:\n" << src;
  EXPECT_NE(src.find("eps"), std::string::npos) << "got:\n" << src;
  EXPECT_NE(src.find("stress_out = "), std::string::npos) << "got:\n" << src;
  // Templated parameters (T0, T1) replace concrete tmech::tensor.
  EXPECT_NE(src.find("template <"), std::string::npos) << "got:\n" << src;
}

TEST(Recipe, MultipleOutputsShareCseAcrossBody) {
  // If two outputs share a subexpression, it should be emitted once.
  ConstitutiveModel m("Shared");
  auto a = m.add_parameter("a", 1.0);
  auto x = m.add_scalar_input("x");

  auto shared = sin(a * x);
  m.add_output("y1", shared + 1);
  m.add_output("y2", shared * 2);

  auto src = m.emit_compute_function();

  // sin(a*x) should appear exactly once in the body — both outputs reference
  // the same temporary.
  std::size_t sin_count = 0;
  for (std::size_t pos = 0;
       (pos = src.find("std::sin", pos)) != std::string::npos; ++pos) {
    ++sin_count;
  }
  EXPECT_EQ(sin_count, 1u)
      << "Shared sin(a*x) should be emitted exactly once. got:\n"
      << src;
}

// ─── Phase D: implicit residual equations (strain-coupled state) ─────────────

// A scalar state defined by an implicit residual R(z, ε)=0. Unlike a rate, the
// residual MAY reference a tensor strain input — that is the coupling.
TEST(Recipe, AddScalarResidualEquationStoresStrainCoupledResidual) {
  ConstitutiveModel m("ReturnMap");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps)); // R = z - c·tr(ε)
  ASSERT_EQ(m.residual_equations().size(), 1u);
  EXPECT_TRUE(m.residual_equations()[0].residual.is_valid());
  EXPECT_TRUE(m.evolution_equations().empty());
}

// The residual's leaves must all be declared symbols.
TEST(Recipe, ResidualRejectsUndeclaredLeaf) {
  ConstitutiveModel m("Bad");
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  auto bogus = cas::make_expression<cas::scalar>("T_bogus");
  try {
    m.add_scalar_residual_equation(z, z.current - bogus * trace(eps));
    FAIL() << "expected throw on undeclared leaf";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("T_bogus"), std::string::npos)
        << e.what();
  }
}

// The residual must depend on its own state, else ∂R/∂z ≡ 0 (singular Jacobian).
TEST(Recipe, ResidualMustReferenceOwnState) {
  ConstitutiveModel m("NoState");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  try {
    m.add_scalar_residual_equation(z, c * trace(eps)); // no z
    FAIL() << "expected throw on state-independent residual";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("does not reference its own state"),
              std::string::npos)
        << e.what();
  }
}

// A state carries either a rate OR a residual — never both (rate first).
TEST(Recipe, StateCannotHaveRateThenResidual) {
  ConstitutiveModel m("Both");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_evolution_equation(z, c * z.current);
  try {
    m.add_scalar_residual_equation(z, z.current - c * trace(eps));
    FAIL() << "expected throw: state already has a rate";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("at most one rate or residual"),
              std::string::npos)
        << e.what();
  }
}

// ...and residual first, then rate.
TEST(Recipe, StateCannotHaveResidualThenRate) {
  ConstitutiveModel m("Both2");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  try {
    m.add_scalar_evolution_equation(z, c * z.current);
    FAIL() << "expected throw: state already has a residual";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("at most one rate or residual"),
              std::string::npos)
        << e.what();
  }
}

// The state may appear only as a tensor COEFFICIENT inside the t2s (e.g.
// trace(z·ε)) — the state-appears guard must still find it, i.e. collect_t2s
// recurses scalar coefficients. A false rejection here would block valid models.
TEST(Recipe, ResidualStateMayAppearAsTensorCoefficient) {
  ConstitutiveModel m("CoeffState");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, trace(z.current * eps) - c);
  EXPECT_EQ(m.residual_equations().size(), 1u);
}

// A residual may reference a declared SCALAR input (not only tensor inputs).
TEST(Recipe, ResidualMayReferenceScalarInput) {
  ConstitutiveModel m("ScalarInputResid");
  auto temp = m.add_scalar_input("temperature");
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - temp * trace(eps));
  EXPECT_EQ(m.residual_equations().size(), 1u);
}

// A state variable is bound to at most ONE evolution mechanism — adding a second
// rate to the same state is rejected (the XOR guard's evolution-side branch).
TEST(Recipe, StateRejectsSecondRate) {
  ConstitutiveModel m("TwoRates");
  auto c = m.add_parameter("c", 2.0);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_evolution_equation(z, c * z.current);
  try {
    m.add_scalar_evolution_equation(z, c * z.current);
    FAIL() << "expected throw: state already has a rate";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("already has a rate"),
              std::string::npos)
        << e.what();
  }
}

// Finding A (holistic review 2026-06-17): the self-contained code path
// (standalone / MOOSE) has no pass that lowers an implicit residual into a
// Newton solve, so emit_compute_function would otherwise emit a function that
// SILENTLY DROPS the declared state. Reject loudly instead — residual emission
// lives only on the graph-coupled NumSimMaterialTarget (Mode B).
TEST(Recipe, EmitComputeFunctionRejectsResidualRecipe) {
  ConstitutiveModel m("ReturnMapStandalone");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z = m.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  try {
    (void)m.emit_compute_function();
    FAIL() << "expected throw: residuals unsupported on the self-contained path";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("residual"), std::string::npos)
        << e.what();
  }
}

// The shared handle-resolution defends against cross-recipe handle use.
TEST(Recipe, ResidualRejectsForeignHandle) {
  ConstitutiveModel m1("M1");
  ConstitutiveModel m2("M2");
  auto c = m2.add_parameter("c", 2.0);
  auto eps = m2.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z1 = m1.add_scalar_state_variable(
      "z", cas::make_expression<cas::scalar_constant>(0.0));
  try {
    m2.add_scalar_residual_equation(z1, z1.current - c * trace(eps)); // m1 handle
    FAIL() << "expected throw on foreign handle";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("different ConstitutiveModel"),
              std::string::npos)
        << e.what();
  }
}

// ── #108 add_hyperelastic_potential guards ──────────────────────────────────

TEST(Recipe, HyperelasticPotentialDerivesStressAndTangent) {
  using namespace numsim::cas;
  ConstitutiveModel m("Hyper");
  auto mu = m.add_parameter("mu", 0.5);
  auto E = m.add_tensor_input("E", 3, 2, roles::Strain);
  m.add_hyperelastic_potential("S", mu * dot(E), E, "dS_dE");
  // Registers exactly one stress output and one tangent.
  EXPECT_EQ(m.outputs().size(), 1u);
  EXPECT_EQ(m.tangents().size(), 1u);
}

TEST(Recipe, HyperelasticPotentialRejectsNonInputStrain) {
  using namespace numsim::cas;
  ConstitutiveModel m("Hyper");
  auto mu = m.add_parameter("mu", 0.5);
  auto E = m.add_tensor_input("E", 3, 2, roles::Strain);
  // A derived expression, not the registered input leaf, is rejected loudly.
  EXPECT_THROW(m.add_hyperelastic_potential("S", mu * dot(E), 2.0 * E, "dS_dE"),
               std::runtime_error);
}

TEST(Recipe, HyperelasticPotentialRejectsStrainIndependentEnergy) {
  using namespace numsim::cas;
  ConstitutiveModel m("Hyper");
  auto mu = m.add_parameter("mu", 0.5);
  auto E = m.add_tensor_input("E", 3, 2, roles::Strain);
  auto F = m.add_tensor_input("F", 3, 2, roles::Strain);
  // ψ depends on F, not on the strain E we differentiate against → ∂ψ/∂E ≡ 0.
  EXPECT_THROW(m.add_hyperelastic_potential("S", mu * dot(F), E, "dS_dE"),
               std::runtime_error);
}

TEST(Recipe, HyperelasticPotentialRollsBackStressOutputOnBadTangentName) {
  using namespace numsim::cas;
  ConstitutiveModel m("Hyper");
  auto mu = m.add_parameter("mu", 0.5);
  auto E = m.add_tensor_input("E", 3, 2, roles::Strain);
  // An invalid tangent name makes add_algorithmic_tangent throw AFTER the stress
  // output was added — the stress output must be rolled back, leaving the model
  // clean so a corrected retry succeeds.
  EXPECT_THROW(m.add_hyperelastic_potential("S", mu * dot(E), E, "bad name"),
               std::runtime_error);
  EXPECT_EQ(m.outputs().size(), 0u) << "stress output must be rolled back";
  EXPECT_NO_THROW(m.add_hyperelastic_potential("S", mu * dot(E), E, "dS_dE"));
  EXPECT_EQ(m.outputs().size(), 1u);
}

} // namespace numsim::codegen
