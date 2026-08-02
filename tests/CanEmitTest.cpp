// #137: Target::can_emit — the up-front scope guards as a QUERY. Each concrete
// target must (a) accept a recipe its emit() supports, (b) reject an
// out-of-scope recipe with EXACTLY the message its emit() throws (one message,
// two transports — the refactor's whole point), and (c) stay conservative:
// can_emit success does not guarantee emit() success (emit-time validation
// such as non-finite parameter defaults still throws).

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace numsim::codegen {
namespace {

using namespace numsim::cas;

// σ = 2μ ε — in scope for StandaloneCxx and MOOSE; out of scope for
// NumSimMaterial's rate contract (no state variable / evolution equation).
auto build_elastic_shear() -> ConstitutiveModel {
  ConstitutiveModel m("ElasticShear");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  return m;
}

// dα/dt = K·α — the canonical NumSimMaterial rate recipe. Out of scope for
// MOOSE (evolution equations without enable_local_newton()).
auto build_linear_hardening() -> ConstitutiveModel {
  ConstitutiveModel m("LinearHardening");
  auto K = m.add_parameter("K", -1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  return m;
}

// R(z, ε) = z − c·tr(ε), σ = z·ε — the Mode-B residual recipe NumSimMaterial
// supports.
auto build_return_map() -> ConstitutiveModel {
  ConstitutiveModel m("ReturnMap");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  m.add_output("stress", z.current * eps, roles::Stress);
  return m;
}

// Returns the emit() exception message (or a sentinel) so the equality tests
// can compare against the can_emit reason.
auto emit_throw_message(Target const &t, ConstitutiveModel const &m)
    -> std::string {
  try {
    // Expected to throw; the [[nodiscard]] return is never reached.
    [[maybe_unused]] auto const discarded = t.emit(m);
  } catch (std::exception const &e) {
    return e.what();
  }
  return "<did not throw>";
}

// ─── StandaloneCxx: no scope guards — the base default ("try emit") ─────────

TEST(CanEmit, StandaloneAcceptsEverythingViaBaseDefault) {
  StandaloneCxxTarget const t;
  EXPECT_TRUE(t.can_emit(build_elastic_shear()).has_value());
  EXPECT_TRUE(t.can_emit(build_linear_hardening()).has_value());
}

// ─── MooseMaterial ───────────────────────────────────────────────────────────

TEST(CanEmit, MooseAcceptsSupportedRecipe) {
  EXPECT_TRUE(MooseMaterialTarget{}.can_emit(build_elastic_shear()).has_value());
}

TEST(CanEmit, MooseRejectsEvolutionWithoutLocalNewtonWithEmitMessage) {
  MooseMaterialTarget const t;
  auto const m = build_linear_hardening();
  auto const verdict = t.can_emit(m);
  ASSERT_FALSE(verdict.has_value());
  // The documented reason...
  EXPECT_NE(verdict.error().find("local Newton solving is not enabled"),
            std::string::npos)
      << verdict.error();
  // ...and byte-for-byte the message emit() throws.
  EXPECT_EQ(verdict.error(), emit_throw_message(t, m));
}

TEST(CanEmit, MooseRejectsStatefulInputWithEmitMessage) {
  MooseMaterialTarget const t;
  auto m = build_elastic_shear();
  m.add_tensor_input("eps_p", 3, 2,
                     Role{.name = "plastic_strain",
                          .is_stateful = true,
                          .expected_rank = 2});
  auto const verdict = t.can_emit(m);
  ASSERT_FALSE(verdict.has_value());
  EXPECT_NE(verdict.error().find("requires the History machinery"),
            std::string::npos)
      << verdict.error();
  EXPECT_EQ(verdict.error(), emit_throw_message(t, m));
}

// ─── NumSimMaterial: rate path ───────────────────────────────────────────────

TEST(CanEmit, NumSimAcceptsRateRecipe) {
  EXPECT_TRUE(
      NumSimMaterialTarget{}.can_emit(build_linear_hardening()).has_value());
}

TEST(CanEmit, NumSimRejectsStatelessRecipeWithEmitMessage) {
  NumSimMaterialTarget const t;
  auto const m = build_elastic_shear(); // no state variable → rate scope fails
  auto const verdict = t.can_emit(m);
  ASSERT_FALSE(verdict.has_value());
  EXPECT_NE(verdict.error().find("exactly one scalar state variable"),
            std::string::npos)
      << verdict.error();
  EXPECT_EQ(verdict.error(), emit_throw_message(t, m));
}

// ─── NumSimMaterial: residual (Mode-B) path ─────────────────────────────────

TEST(CanEmit, NumSimAcceptsResidualRecipe) {
  EXPECT_TRUE(NumSimMaterialTarget{}.can_emit(build_return_map()).has_value());
}

TEST(CanEmit, NumSimRejectsOutputlessResidualWithEmitMessage) {
  NumSimMaterialTarget const t;
  ConstitutiveModel m("NoOutput");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  auto const verdict = t.can_emit(m);
  ASSERT_FALSE(verdict.has_value());
  EXPECT_NE(verdict.error().find("needs at least one output"),
            std::string::npos)
      << verdict.error();
  EXPECT_EQ(verdict.error(), emit_throw_message(t, m));
}

// ─── Polymorphic use through the factory (the registry generator's path) ────

TEST(CanEmit, QueryableThroughTargetBasePointer) {
  auto const t = make_target("numsim_material");
  EXPECT_TRUE(t->can_emit(build_linear_hardening()).has_value());
  EXPECT_FALSE(t->can_emit(build_elastic_shear()).has_value());
}

// ─── Documented limitation: success does not guarantee emit() success ───────

TEST(CanEmit, SuccessDoesNotGuaranteeEmitSuccess) {
  // A non-finite parameter default passes the up-front SHAPE guards (can_emit
  // only checks those) but is rejected by emit-time validation — the header-
  // documented contract.
  ConstitutiveModel m("NanDefault");
  auto K =
      m.add_parameter("K", std::numeric_limits<double>::quiet_NaN());
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);

  NumSimMaterialTarget const t;
  EXPECT_TRUE(t.can_emit(m).has_value());
  EXPECT_NE(emit_throw_message(t, m).find("non-finite default"),
            std::string::npos);
}

} // namespace
} // namespace numsim::codegen
