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

// ─── MAJOR 1: MOOSE backend does NOT yet wire state variables ─────────

// KNOWN LIMITATION (Phase 2.6, issues #34/#37): the MooseMaterialTarget
// constructor + validParams iterate only parameters()/inputs()/outputs()
// — they do NOT handle Category::StateVariableCurrent / StateVariableOld.
// For an evolution recipe the generated MOOSE `.C` therefore:
//   * declares no member for `<sv>` / `<sv>_old`,
//   * never emits `getMaterialPropertyOld<>` for `<sv>_old`,
//   * calls the generic compute function with the wrong arity (it omits
//     the state-variable + synthesised residual/jacobian arguments).
// i.e. the emitted MOOSE source does NOT compile for evolution recipes.
//
// This test PINS that broken-but-known state so:
//   (a) the gap is documented + visible in the test suite, and
//   (b) when Phase 2.6 wires state variables, whoever does it MUST flip
//       these assertions — they can't land the wiring and leave the gap
//       silently half-done.
//
// TODO(phase-2.6 / #37): replace the EXPECT_EQ(..., npos) assertions
// below with positive checks that `getMaterialPropertyOld<Real>("alpha")`
// is declared and the compute call passes alpha/alpha_old/residual/jacobian.
TEST(Integration, MooseBackendDoesNotYetWireStateVariables_KNOWN_GAP) {
  auto m = build_mixed_hardening();
  MooseMaterialTarget target;
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 2u);
  std::string const header = files[0].contents;
  std::string const source = files[1].contents;

  // The auto-registered `dt` IS picked up (it's a Parameter, not a
  // state-variable category) — sanity anchor that emission ran.
  EXPECT_NE(header.find("_dt"), std::string::npos) << header;

  // KNOWN GAP: no getMaterialPropertyOld wiring for the `_old` symbol.
  EXPECT_EQ(source.find("getMaterialPropertyOld"), std::string::npos)
      << "Phase 2.6 may have landed — flip this to a positive check: "
      << source;
  // KNOWN GAP: no declared member for the state variable itself.
  EXPECT_EQ(header.find("_alpha "), std::string::npos)
      << "Phase 2.6 may have landed — flip this assertion: " << header;
}

} // namespace numsim::codegen
