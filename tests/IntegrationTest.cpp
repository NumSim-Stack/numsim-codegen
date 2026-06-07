// Cross-phase integration tests (cross-cutting review, MAJOR 1 + 2).
//
// The per-phase suites (StateVariableTest, TimeIntegrationTest,
// LocalJacobianTest) each exercise their phase in isolation. These tests
// run a SINGLE recipe that combines all of: a state variable + an
// evolution equation (→ TimeIntegrationPass synthesises a residual,
// LocalJacobianPass synthesises a Jacobian) + a tensor output + a scalar
// output, through a real backend, and assert the emitted source is
// internally consistent.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

namespace {

// Mixed recipe: elastic tensor stress + scalar hardening force + a
// state variable with linear-hardening evolution. Exercises every
// Phase 2.1/2.2/3a-1 path in one model.
auto build_mixed_hardening() -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("MixedHardening");
  auto K = m.add_parameter("K", 1.0, "hardening modulus");
  auto mu = m.add_parameter("mu", 0.5, "shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto alpha = m.add_scalar_state_variable(
      "alpha", make_expression<scalar_constant>(0.0),
      "equivalent plastic strain");
  // Tensor output (elastic stress) + scalar output (hardening force).
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_output("hardening_force", K * alpha.current);
  // Evolution equation → residual + Jacobian synthesis.
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  return m;
}

} // namespace

// ─── MAJOR 2: full-pipeline end-to-end through the standalone backend ──

TEST(Integration, StandaloneEmitsAllPhaseOutputsConsistently) {
  auto m = build_mixed_hardening();
  StandaloneCxxTarget target;
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 1u);
  auto const &src = files[0].contents;

  // Every phase's contribution must appear in one generated function:
  EXPECT_NE(src.find("MixedHardening_compute"), std::string::npos) << src;
  // Phase A elastic outputs:
  EXPECT_NE(src.find("stress_out"), std::string::npos) << src;
  EXPECT_NE(src.find("hardening_force_out"), std::string::npos) << src;
  // Phase 2.2 residual + Phase 3a-1 Jacobian (synthesised):
  EXPECT_NE(src.find("alpha_residual_out"), std::string::npos) << src;
  EXPECT_NE(src.find("alpha_jacobian_out"), std::string::npos) << src;
  // Phase 2.1 state-variable symbols + auto-registered dt:
  EXPECT_NE(src.find("alpha_old"), std::string::npos) << src;
  EXPECT_NE(src.find("dt"), std::string::npos) << src;

  // Self-consistency: the standalone backend emits the full generic
  // compute function, so every symbol referenced in the body is also a
  // function parameter. validate() already guarantees no undeclared
  // leaves; emitting without throwing confirms the residual + Jacobian
  // synthesis didn't introduce a dangling symbol.
  EXPECT_NO_THROW((void)m.emit_compute_function());
}

TEST(Integration, MixedRecipeOutputOrderingIsStable) {
  // The synthesised residual/jacobian outputs are appended AFTER the
  // user-declared outputs (TimeIntegrationPass + LocalJacobianPass run
  // after the recipe is built). Pin that ordering so a future pass
  // reorder is a deliberate change.
  auto m = build_mixed_hardening();
  StandaloneCxxTarget target;
  auto const src = target.emit(m).at(0).contents;

  auto const hforce = src.find("hardening_force_out");
  auto const resid = src.find("alpha_residual_out");
  auto const jac = src.find("alpha_jacobian_out");
  ASSERT_NE(hforce, std::string::npos);
  ASSERT_NE(resid, std::string::npos);
  ASSERT_NE(jac, std::string::npos);
  // User outputs precede synthesised outputs; residual precedes jacobian.
  EXPECT_LT(hforce, resid) << src;
  EXPECT_LT(resid, jac) << src;
}

// ─── Phase 2.6 (issue #77): MOOSE backend wires state variables ───────

namespace {
// A scalar-Newton hardening recipe — the MOOSE-relevant mode (a MOOSE
// material solves its local system internally and writes the state
// property). enable_local_newton() makes the generated function solve for
// the converged state.
auto build_newton_hardening() -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("Hardening");
  auto K = m.add_parameter("K", 1.0);
  auto alpha = m.add_scalar_state_variable(
      "alpha", make_expression<scalar_constant>(0.0));
  m.add_output("sigma_y", K * alpha.current, roles::Stress);
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  m.enable_local_newton();
  return m;
}
} // namespace

TEST(Integration, MooseDeclaresStateVariableProperties) {
  auto files = MooseMaterialTarget{}.emit(build_newton_hardening());
  ASSERT_EQ(files.size(), 2u);
  std::string const header = files[0].contents;
  std::string const source = files[1].contents;

  // Current state variable: a writable declared property.
  EXPECT_NE(header.find("MaterialProperty<Real> & _alpha;"),
            std::string::npos)
      << header;
  EXPECT_NE(source.find("declareProperty<Real>(\"Hardening_alpha\")"),
            std::string::npos)
      << source;
  // Old value: const property bound via getMaterialPropertyOld of the SAME
  // declared property name (MOOSE versions it automatically).
  EXPECT_NE(header.find("const MaterialProperty<Real> & _alpha_old;"),
            std::string::npos)
      << header;
  EXPECT_NE(
      source.find("getMaterialPropertyOld<Real>(\"Hardening_alpha\")"),
      std::string::npos)
      << source;
}

TEST(Integration, MooseMapsDtToFrameworkTimestep) {
  auto files = MooseMaterialTarget{}.emit(build_newton_hardening());
  std::string const source = files[1].contents;
  // dt must come from MOOSE's framework `_dt`, NOT getParam — and it must
  // NOT appear as an input-file parameter.
  EXPECT_EQ(source.find("getParam<Real>(\"dt\")"), std::string::npos)
      << source;
  EXPECT_EQ(source.find("addParam<Real>(\"dt\""), std::string::npos)
      << source;
  EXPECT_NE(source.find("_dt"), std::string::npos) << source;
}

TEST(Integration, MooseComputeCallArityMatchesGeneratedSignature) {
  auto files = MooseMaterialTarget{}.emit(build_newton_hardening());
  std::string const source = files[1].contents;
  // The call must thread the state-variable arguments in the canonical
  // order — proving the historical arity bug (3-arg call into a 7-param
  // function) is gone. Generated signature is
  //   Hardening_compute(K, alpha_old, dt, sigma_y_out, alpha_out)
  // so the call must pass _alpha_old[_qp], _dt, and _alpha[_qp].
  EXPECT_NE(source.find("_alpha_old[_qp]"), std::string::npos) << source;
  EXPECT_NE(source.find("_dt"), std::string::npos) << source;
  EXPECT_NE(source.find("_alpha[_qp]"), std::string::npos) << source;
}

TEST(Integration, MooseInitialisesStateVariable) {
  auto files = MooseMaterialTarget{}.emit(build_newton_hardening());
  std::string const header = files[0].contents;
  std::string const source = files[1].contents;
  EXPECT_NE(header.find("virtual void initQpStatefulProperties() override;"),
            std::string::npos)
      << header;
  EXPECT_NE(source.find("_alpha[_qp] = 0.0;"), std::string::npos) << source;
}

} // namespace numsim::codegen
