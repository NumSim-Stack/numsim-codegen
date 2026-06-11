// Phase B (graph-coupled roadmap): NumSimMaterialTarget emits a numsim-materials
// rate-function material conforming to the rk_integrator contract — `rate` (= f)
// and `rate_derivative` (= ∂f/∂x via cas::diff) as Local-edge properties.
//
// These are STRUCTURAL checks on the emitted text. The end-to-end proof (the
// generated header compiles against real numsim-materials with g++-14 and runs
// through rk_integrator, converging to e^K) was done as a cross-repo spike; it
// can't run in this repo's CI without the numsim-materials toolchain.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>

#include <gtest/gtest.h>

#include <format>
#include <limits>
#include <string>

namespace numsim::codegen {
namespace {

using namespace numsim::cas;

// Linear hardening dα/dt = K·α — the canonical first-increment recipe.
auto build_linear_hardening() -> ConstitutiveModel {
  ConstitutiveModel m("LinearHardening");
  auto K = m.add_parameter("K", -1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  return m;
}

auto header_of(std::vector<EmittedFile> const &files) -> std::string {
  for (auto const &f : files)
    if (f.kind == EmittedFile::Kind::Header) return f.contents;
  return {};
}

// Returns the emit() exception message (or a sentinel) so a rejection test can
// assert WHICH guard fired, not merely that *some* runtime_error was thrown.
auto emit_throw_message(ConstitutiveModel const &m) -> std::string {
  try {
    NumSimMaterialTarget{}.emit(m);
  } catch (std::exception const &e) {
    return e.what();
  }
  return "<did not throw>";
}

TEST(NumSimMaterialTarget, EmitsHeaderAndJsonConfig) {
  auto m = build_linear_hardening();
  auto files = NumSimMaterialTarget{}.emit(m);
  ASSERT_EQ(files.size(), 2u);
  EXPECT_EQ(files[0].filename, "LinearHardening.h");
  EXPECT_EQ(files[0].kind, EmittedFile::Kind::Header);
  EXPECT_EQ(files[1].filename, "LinearHardening.config.json");
}

TEST(NumSimMaterialTarget, MaterialConformsToRkIntegratorContract) {
  auto const src = header_of(NumSimMaterialTarget{}.emit(build_linear_hardening()));
  // CRTP material_base subclass.
  EXPECT_NE(src.find("class LinearHardening final"), std::string::npos) << src;
  EXPECT_NE(src.find("material_base<LinearHardening<Traits>, Traits>"),
            std::string::npos)
      << src;
  // The two contract properties: rate (with compute callback) + rate_derivative.
  EXPECT_NE(src.find("add_output<value_type>(\n            \"rate\", "
                     "&LinearHardening::compute)"),
            std::string::npos)
      << src;
  EXPECT_NE(src.find("add_output<value_type>(\"rate_derivative\")"),
            std::string::npos)
      << src;
  // State read from the integrator over a Local edge.
  EXPECT_NE(src.find("add_input<value_type>(\n            m_integrator_source, "
                     "\"state\",\n            numsim::materials::EdgeKind::Local)"),
            std::string::npos)
      << src;
}

TEST(NumSimMaterialTarget, ComputeEmitsRateAndDerivative) {
  auto const src = header_of(NumSimMaterialTarget{}.emit(build_linear_hardening()));
  // f = K·α  → rate; ∂f/∂α = K → rate_derivative (cas::diff).
  EXPECT_NE(src.find("const auto alpha = m_alpha.get();"), std::string::npos)
      << src;
  EXPECT_NE(src.find("auto t0 = m_K * alpha;"), std::string::npos) << src;
  EXPECT_NE(src.find("m_rate = t0;"), std::string::npos) << src;
  EXPECT_NE(src.find("m_rate_derivative = m_K;"), std::string::npos) << src;
}

TEST(NumSimMaterialTarget, ParameterSchemaUsesRecipeDefault) {
  auto const src = header_of(NumSimMaterialTarget{}.emit(build_linear_hardening()));
  // The recipe default (-1) flows into the material's parameter schema.
  EXPECT_NE(src.find("insert<value_type>(\"K\").template "
                     "add<numsim_core::set_default>(value_type{-1})"),
            std::string::npos)
      << src;
  EXPECT_NE(src.find("insert<std::string>(\"integrator_source\")"),
            std::string::npos)
      << src;
}

TEST(NumSimMaterialTarget, JsonWiresIntegratorToMaterial) {
  auto files = NumSimMaterialTarget{}.emit(build_linear_hardening());
  std::string json;
  for (auto const &f : files)
    if (f.filename == "LinearHardening.config.json") json = f.contents;
  EXPECT_NE(json.find("\"type\": \"rk_integrator\""), std::string::npos) << json;
  EXPECT_NE(json.find("\"function\": \"LinearHardening\""), std::string::npos)
      << json;
  EXPECT_NE(json.find("\"integrator_source\": \"integrator\""),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"K\": -1"), std::string::npos) << json;
}

TEST(NumSimMaterialTarget, RejectsTensorStateMissingEvolution) {
  // Tensor-valued state is out of scope (needs Mandel + the vector solver,
  // numsim-materials#11/#12). Evolution is scalar-only in the recipe API, so a
  // tensor state has no evolution equation and is rejected by the equation-count
  // guard (the tensor-kind branch is defensive/unreachable via the public API).
  ConstitutiveModel m("TensorState");
  m.add_tensor_state_variable("ep", 3, 2, make_expression<tensor_zero>(3, 2));
  EXPECT_THROW(NumSimMaterialTarget{}.emit(m), std::runtime_error);
}

TEST(NumSimMaterialTarget, RejectsMultipleCoupledStates) {
  ConstitutiveModel m("Coupled");
  auto K1 = m.add_parameter("K1", 1.0);
  auto K2 = m.add_parameter("K2", 2.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  auto b = m.add_scalar_state_variable("b", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K1 * b.current);
  m.add_scalar_evolution_equation(b, K2 * a.current);
  // Hits the equation-count guard.
  EXPECT_NE(emit_throw_message(m).find("exactly one scalar state variable"),
            std::string::npos);
}

// Review #88: silently dropping a declared output/tangent/input is worse than a
// hard failure for a code generator — reject loudly instead. Round-2: each test
// asserts WHICH guard fired (message substring), not just that it threw.
TEST(NumSimMaterialTarget, RejectsDeclaredOutput) {
  ConstitutiveModel m("WithOutput");
  auto K = m.add_parameter("K", -1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  m.add_output("extra", K * a.current); // would be silently dropped
  EXPECT_NE(emit_throw_message(m).find("add_output"), std::string::npos);
}

TEST(NumSimMaterialTarget, RejectsDeclaredInput) {
  ConstitutiveModel m("WithInput");
  auto K = m.add_parameter("K", -1.0);
  m.add_scalar_input("temperature"); // scalar inputs are a follow-up
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  EXPECT_NE(emit_throw_message(m).find("declared inputs"), std::string::npos);
}

// Review #88 (cpp-pro): a parameter named like an emitted fixed member would
// produce a duplicate member / ctor initializer in the generated material.
TEST(NumSimMaterialTarget, RejectsReservedParameterName) {
  ConstitutiveModel m("ReservedParam");
  auto rate = m.add_parameter("rate", 1.0); // collides with m_rate
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, rate * a.current);
  EXPECT_NE(emit_throw_message(m).find("collides with an emitted member"),
            std::string::npos);
}

// Round-2 (code-reviewer M2): the STATE-name reserved guard branch — previously
// untested (only the parameter loop was covered).
TEST(NumSimMaterialTarget, RejectsReservedStateName) {
  ConstitutiveModel m("ReservedState");
  auto K = m.add_parameter("K", -1.0);
  // state literally named `rate` → would emit a duplicate m_rate member.
  auto rate =
      m.add_scalar_state_variable("rate", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(rate, K * rate.current);
  EXPECT_NE(emit_throw_message(m).find("collides with an emitted member"),
            std::string::npos);
}

// Review #88 (cpp-pro): a rate function cannot depend on dt / old-state /
// inputs — the integrator owns discretization. Reject at emit time rather than
// emitting an unbound identifier that fails the consumer's build.
TEST(NumSimMaterialTarget, RejectsRateDependingOnOldState) {
  ConstitutiveModel m("UsesOldState");
  auto K = m.add_parameter("K", -1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  // rate references a.previous (the old-step value) — invalid for a rate fn.
  m.add_scalar_evolution_equation(a, K * (a.current - a.previous));
  EXPECT_NE(emit_throw_message(m).find("neither the state"), std::string::npos);
}

// Round-2 (cpp-pro): a non-finite default would emit `value_type{nan}`/`{inf}`
// (no such C++ literal) and an invalid JSON number — reject at emit time.
TEST(NumSimMaterialTarget, RejectsNonFiniteDefault) {
  ConstitutiveModel m("NonFinite");
  auto K = m.add_parameter("K", std::numeric_limits<double>::infinity());
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  EXPECT_NE(emit_throw_message(m).find("non-finite default"), std::string::npos);
}

// Review #88 (cpp-pro / code-reviewer): non-round float defaults must round-trip
// exactly into the emitted C++ and JSON (not truncate to ~6 sig figs).
TEST(NumSimMaterialTarget, FloatDefaultRoundTrips) {
  ConstitutiveModel m("Precise");
  auto K = m.add_parameter("K", 1.0 / 3.0); // 0.3333333333333333
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  auto const src = header_of(NumSimMaterialTarget{}.emit(m));
  // 6-sig-fig truncation would emit "0.333333"; the emitter uses the shortest
  // round-tripping representation (std::format), reproduced here so the
  // assertion can't drift from the formatter.
  EXPECT_EQ(src.find("value_type{0.333333}"), std::string::npos) << src;
  EXPECT_NE(src.find("value_type{" + std::format("{}", 1.0 / 3.0) + "}"),
            std::string::npos)
      << src;
}

} // namespace
} // namespace numsim::codegen
