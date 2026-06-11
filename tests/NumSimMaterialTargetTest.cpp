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

TEST(NumSimMaterialTarget, RejectsTensorStateForNow) {
  // Tensor-valued state is out of scope for the first increment (needs Mandel +
  // the vector solver, numsim-materials#11/#12). Evolution is scalar-only in the
  // recipe API, so a tensor state has no evolution equation — rejected by the
  // scope guard.
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
  EXPECT_THROW(NumSimMaterialTarget{}.emit(m), std::runtime_error);
}

} // namespace
} // namespace numsim::codegen
