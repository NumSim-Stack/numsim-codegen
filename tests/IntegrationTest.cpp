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

#include <algorithm>
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

// ─── PR #78 review fixups ─────────────────────────────────────────────

// #7: direct test of the canonical_arguments contract — the single source
// of truth both the signature and the backend call-sites iterate.
TEST(Integration, CanonicalArgumentsOrderAndRoles) {
  auto m = build_newton_hardening();
  auto const args = canonical_arguments(RecipeView{m});
  using R = ArgSpec::Role;
  ASSERT_EQ(args.size(), 5u);
  EXPECT_EQ(args[0].name, "K");         EXPECT_EQ(args[0].role, R::ScalarParam);
  EXPECT_EQ(args[1].name, "alpha_old"); EXPECT_EQ(args[1].role, R::StateOld);
  EXPECT_EQ(args[2].name, "dt");        EXPECT_EQ(args[2].role, R::TimeStep);
  EXPECT_EQ(args[3].name, "sigma_y");   EXPECT_EQ(args[3].role, R::ScalarOutput);
  EXPECT_EQ(args[4].name, "alpha");     EXPECT_EQ(args[4].role, R::NewtonStateOut);
}

namespace {
// Count comma-separated arguments inside the first `<Model>_compute(` call.
auto count_compute_call_args(std::string const &src, std::string const &model)
    -> std::size_t {
  auto const open = src.find(model + "_compute(");
  if (open == std::string::npos) return 0;
  auto const lp = src.find('(', open);
  auto const rp = src.find(')', lp);
  auto const inner = src.substr(lp + 1, rp - lp - 1);
  if (inner.find_first_not_of(" \t\n") == std::string::npos) return 0;
  return std::size_t{1} +
         static_cast<std::size_t>(std::count(inner.begin(), inner.end(), ','));
}
} // namespace

// #6: real arity check — the MOOSE call must pass exactly as many arguments
// as canonical_arguments (the historical bug was a count mismatch, 3 vs 7).
TEST(Integration, MooseComputeCallArgCountEqualsCanonical) {
  auto m = build_newton_hardening();
  auto const source = MooseMaterialTarget{}.emit(m).at(1).contents;
  EXPECT_EQ(count_compute_call_args(source, "Hardening"),
            canonical_arguments(RecipeView{m}).size());
}

// #7: multi-state-variable MOOSE — the member/ctor/initQp/call loops are the
// generalisation that can break (ordering, threading both vars).
TEST(Integration, MooseWiresMultipleStateVariables) {
  using namespace numsim::cas;
  ConstitutiveModel m("Multi");
  auto K = m.add_parameter("K", 1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  auto b = m.add_scalar_state_variable("b", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  m.add_scalar_evolution_equation(b, K * b.current);
  m.enable_local_newton();
  auto files = MooseMaterialTarget{}.emit(m);
  auto const &h = files[0].contents;
  auto const &c = files[1].contents;
  for (auto const *sv : {"a", "b"}) {
    EXPECT_NE(c.find(std::string("declareProperty<Real>(\"Multi_") + sv + "\")"),
              std::string::npos) << c;
    EXPECT_NE(c.find(std::string("getMaterialPropertyOld<Real>(\"Multi_") + sv + "\")"),
              std::string::npos) << c;
    EXPECT_NE(c.find(std::string("_") + sv + "[_qp] = 0.0;"), std::string::npos)
        << c;
    EXPECT_NE(h.find(std::string("MaterialProperty<Real> & _") + sv + ";"),
              std::string::npos) << h;
  }
  // Both state vars threaded into the call in canonical order.
  EXPECT_EQ(count_compute_call_args(c, "Multi"),
            canonical_arguments(RecipeView{m}).size());
}

// #3: non-constant initial value → MOOSE emit fails loudly (rather than
// emitting an undefined bare symbol name).
TEST(Integration, MooseRejectsNonConstantStateVarInitial) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto K = m.add_parameter("K", 2.0);
  // initial = K * 0.5 references parameter K → not a constant.
  auto c = m.add_scalar_state_variable(
      "c", K * make_expression<scalar_constant>(0.5));
  m.add_scalar_evolution_equation(c, K * c.current);
  m.enable_local_newton();
  EXPECT_THROW((void)MooseMaterialTarget{}.emit(m), std::runtime_error);
}

// #2: output name clashing with a state variable is rejected at build time
// (would otherwise emit duplicate MOOSE properties + a duplicate member).
// Both declaration orders must be guarded (PR #78 round-2 CRITICAL: the
// reverse order initially slipped through assert_symbol_name_available).
TEST(Integration, OutputStateVarNameClashThrowsEitherOrder) {
  using namespace numsim::cas;
  // Forward: state var first, then output.
  {
    ConstitutiveModel m("M");
    (void)m.add_scalar_state_variable("alpha",
                                      make_expression<scalar_constant>(0.0));
    EXPECT_THROW(
        m.add_output("alpha", make_expression<scalar_constant>(1.0)),
        std::runtime_error);
  }
  // Reverse: output first, then state var.
  {
    ConstitutiveModel m("M");
    m.add_output("alpha", make_expression<scalar_constant>(1.0));
    EXPECT_THROW(
        m.add_scalar_state_variable("alpha",
                                    make_expression<scalar_constant>(0.0)),
        std::runtime_error);
  }
  // Also a plain parameter vs output, reverse order.
  {
    ConstitutiveModel m("M");
    m.add_output("beta", make_expression<scalar_constant>(1.0));
    EXPECT_THROW((void)m.add_parameter("beta", 1.0), std::runtime_error);
  }
}

} // namespace numsim::codegen
