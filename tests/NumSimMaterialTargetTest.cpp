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
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

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
    (void)NumSimMaterialTarget{}.emit(m); // expected to throw; [[nodiscard]]
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
  EXPECT_NE(src.find("[[maybe_unused]] const auto alpha = m_alpha.get();"),
            std::string::npos)
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
  EXPECT_THROW((void)NumSimMaterialTarget{}.emit(m), std::runtime_error);
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
// Phase B.1: a scalar output is now emitted as its own `<name>` property with an
// `update_<name>()` callback computed from the state + parameters.
TEST(NumSimMaterialTarget, EmitsScalarOutputProperty) {
  ConstitutiveModel m("WithScalarOutput");
  auto K = m.add_parameter("K", -1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  m.add_output("energy", K * a.current * a.current); // state + param, scalar
  auto const h = header_of(NumSimMaterialTarget{}.emit(m));
  EXPECT_NE(h.find("\"energy\", &WithScalarOutput::update_energy"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("void update_energy() {"), std::string::npos) << h;
  EXPECT_NE(h.find("value_type& m_out_energy;"), std::string::npos) << h;
}

// Phase B (tensor path): a tensor stress output computed from a scalar state and
// a tensor strain input IS emitted — strain wired as a Global-edge input, stress
// as a tmech tensor property with its own callback. (No tensor STATE here, so no
// Mandel/vector solver needed.)
TEST(NumSimMaterialTarget, EmitsTensorStressOutput) {
  ConstitutiveModel m("WithTensorStress");
  auto mu = m.add_parameter("mu", 0.5);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, mu * a.current);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  m.add_output("stress", a.current * eps, roles::Stress);
  auto const h = header_of(NumSimMaterialTarget{}.emit(m));
  EXPECT_NE(h.find("#include <tmech/tmech.h>"), std::string::npos) << h;
  // strain wired as a Global-edge tensor input from a `strain_source` producer
  EXPECT_NE(h.find("add_input<tmech::tensor<value_type, 3, 2>>"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("numsim::materials::EdgeKind::Global"), std::string::npos)
      << h;
  EXPECT_NE(h.find("insert<std::string>(\"strain_source\")"), std::string::npos)
      << h;
  // stress published as a tmech tensor property with its own callback
  EXPECT_NE(h.find("add_output<tmech::tensor<value_type, 3, 2>>(\n"
                   "            \"stress\", &WithTensorStress::update_stress"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("void update_stress() {"), std::string::npos) << h;
}

// A tensor input named like a synthesized member must be rejected (it would
// emit a duplicate `m_rate` member). The recipe permits the name (it isn't a
// recipe-reserved word); the emitter's uniqueness guard catches it.
TEST(NumSimMaterialTarget, RejectsTensorInputCollidingWithSynthesizedMember) {
  ConstitutiveModel m("InputNameClash");
  auto K = m.add_parameter("K", -1.0);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  auto rate = m.add_tensor_input("rate", 3, 2, roles::Strain); // → m_rate clash
  m.add_output("stress", a.current * rate, roles::Stress);
  EXPECT_NE(emit_throw_message(m).find("would be duplicated"), std::string::npos);
}

// An input "strain" alongside a parameter "strain_source" both want
// m_strain_source — the recipe allows the two distinct symbols, the emitter
// uniqueness guard rejects the collision.
TEST(NumSimMaterialTarget, RejectsInputSourceParamCollision) {
  ConstitutiveModel m("SourceNameClash");
  auto K = m.add_parameter("K", -1.0);
  auto ss = m.add_parameter("strain_source", 1.0); // collides with strain's _source
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, (K + ss) * a.current);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  m.add_output("stress", a.current * eps, roles::Stress);
  EXPECT_NE(emit_throw_message(m).find("would be duplicated"), std::string::npos);
}

// Scalar inputs are not yet wired (tensor inputs ARE) — rejected loudly.
TEST(NumSimMaterialTarget, RejectsScalarInput) {
  ConstitutiveModel m("WithScalarInput");
  auto K = m.add_parameter("K", -1.0);
  m.add_scalar_input("temperature"); // scalar inputs are a follow-up
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current);
  EXPECT_NE(emit_throw_message(m).find("scalar input"), std::string::npos);
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
  // Pin to the STATE branch specifically (the param branch shares the "collides"
  // substring).
  EXPECT_NE(emit_throw_message(m).find("state variable name 'rate'"),
            std::string::npos);
}

// Phase D: the consistent tangent dσ/dε is emitted as a rank-4 tensor property
// via cas::diff(stress, strain). With a strain-INDEPENDENT rate (dα/dε=0) this is
// the exact consistent tangent. For σ=2μ·ε the tangent is 2μ·P_sym, so the
// emitted rank-4 RHS is minor-symmetric (otimesu + otimesl).
TEST(NumSimMaterialTarget, EmitsConsistentTangent) {
  ConstitutiveModel m("WithTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, mu * a.current);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  auto const h = header_of(NumSimMaterialTarget{}.emit(m));
  // rank-4 tensor property + its own callback
  EXPECT_NE(h.find("add_output<tmech::tensor<value_type, 3, 4>>(\n"
                   "            \"dstress_dstrain\", "
                   "&WithTangent::update_dstress_dstrain"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("void update_dstress_dstrain() {"), std::string::npos) << h;
  // minor-symmetric: both otimesu AND otimesl present in the tangent RHS
  EXPECT_NE(h.find("tmech::otimesu"), std::string::npos) << h;
  EXPECT_NE(h.find("tmech::otimesl"), std::string::npos) << h;
}

// A tangent whose `of_output` is not a declared tensor output is rejected.
TEST(NumSimMaterialTarget, RejectsTangentOfMissingStress) {
  ConstitutiveModel m("BadTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, mu * a.current);
  m.add_tensor_input("strain", 3, 2, roles::Strain);
  m.add_algorithmic_tangent("dstress_dstrain", "stress", "strain"); // no "stress"
  EXPECT_NE(emit_throw_message(m).find("not a declared tensor output"),
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

// ─── Phase 2a: Mode-B strain-coupled residual emission ───────────────────────

// A return-map recipe R(z, ε) = z − c·tr(ε), σ = z·ε. The state z is solved
// implicitly by backward_euler; the material drives the Newton loop itself.
auto build_return_map() -> ConstitutiveModel {
  ConstitutiveModel m("ReturnMap");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  m.add_output("stress", z.current * eps);
  return m;
}

// The emitted residual material conforms to the Mode-B (backward_euler caller-
// driven) contract: a material_ref<backward_euler>, a solve(eval) call, the
// residual and its jacobian inside the eval lambda, an owned history state, and
// the stress output bound to compute().
TEST(NumSimMaterialTarget, EmitsModeBResidualMaterial) {
  auto const h = header_of(NumSimMaterialTarget{}.emit(build_return_map()));
  // Mode-B structural surface.
  EXPECT_NE(h.find("using solver_type = numsim::materials::backward_euler<Traits>"),
            std::string::npos) << h;
  EXPECT_NE(h.find("add_material_ref<solver_type>"), std::string::npos) << h;
  EXPECT_NE(h.find("m_solver.get().solve(eval)"), std::string::npos) << h;
  EXPECT_NE(h.find("add_history_output<value_type>(\"z\")"), std::string::npos)
      << h;
  // The stress output drives the solve (carries &compute).
  EXPECT_NE(h.find("\"stress\", &ReturnMap::compute"), std::string::npos) << h;
  // The eval lambda returns {residual, jacobian}.
  EXPECT_NE(h.find("return {residual, jacobian};"), std::string::npos) << h;
  // Pin the RESIDUAL RHS, not just the jacobian — the load-bearing output. A
  // dropped coupling term or wrong sign must fail here, at the always-run unit
  // layer (the tensor e2e is gcc-only). R = z − c·tr(ε): the rendered scalar
  // must reference the state z and the strain trace.
  EXPECT_NE(h.find("const value_type residual = "), std::string::npos) << h;
  EXPECT_NE(h.find("tmech::trace(strain)"), std::string::npos) << h;
  // ∂R/∂z = 1 exactly, rendered "1.0" (terminating ';' so "= 1.5" etc can't
  // false-match).
  EXPECT_NE(h.find("const value_type jacobian = 1.0;"), std::string::npos) << h;
  // State applied as old + increment.
  EXPECT_NE(h.find("m_z.new_value() = m_z.old_value() + dz"), std::string::npos)
      << h;
  // The solver_source param is required.
  EXPECT_NE(h.find("insert<std::string>(\"solver_source\")"), std::string::npos)
      << h;
}

// Review (cpp-pro): two outputs share the single compute() body. Each is
// rendered in its own CSE context (temps restart at t0), so without a nested
// scope the second output redeclares `t0` — an uncompilable header the single-
// output recipe never exercised. Each output must be brace-scoped, and EVERY
// output must drive the solve (carry &compute), not just the first.
TEST(NumSimMaterialTarget, MultiOutputResidualScopesCseAndDrivesAll) {
  ConstitutiveModel m("MultiOut");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  m.add_output("stress", sin(z.current) * eps);  // CSE temp
  m.add_output("stress2", cos(z.current) * eps); // CSE temp
  auto const h = header_of(NumSimMaterialTarget{}.emit(m));
  // Both outputs drive the solve.
  EXPECT_NE(h.find("\"stress\", &MultiOut::compute"), std::string::npos) << h;
  EXPECT_NE(h.find("\"stress2\", &MultiOut::compute"), std::string::npos) << h;
  // The two CSE blocks are brace-scoped (each output writes inside a `{ }`),
  // so `t0` is private per block — count the opening braces of output blocks.
  std::size_t braces = 0;
  for (std::size_t pos = 0;
       (pos = h.find("\n    {\n", pos)) != std::string::npos; ++pos)
    ++braces;
  EXPECT_GE(braces, 2u)
      << "each output must be brace-scoped to avoid CSE temp collision:\n"
      << h;
}

// Review (cpp-pro): a residual must not reference the previous-step state
// `<state>_old` — it is a declared symbol (so the recipe accepts it) but has no
// local in the emitted compute(), which would emit an unbound identifier. Guard
// at emit time.
TEST(NumSimMaterialTarget, RejectsResidualReferencingOldState) {
  ConstitutiveModel m("UsesOld");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  // R = z − z_old − c·tr(ε): references the previous-step state.
  m.add_scalar_residual_equation(z, z.current - z.previous - c * trace(eps));
  m.add_output("stress", z.current * eps);
  EXPECT_NE(emit_throw_message(m).find("previous-step state"),
            std::string::npos);
}

// Review (cpp-pro / code-reviewer): the emitter synthesizes the fixed member
// `m_solver` and the compute() locals `eval`/`residual`/`jacobian`. A recipe
// symbol named like one of these must be rejected with a rename message, not
// surface as a downstream redefinition.
TEST(NumSimMaterialTarget, RejectsSolverNameCollisionOnResidualMaterial) {
  ConstitutiveModel m("SolverClash");
  auto solver = m.add_parameter("solver", 2.0); // collides with m_solver
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - solver * trace(eps));
  m.add_output("stress", z.current * eps);
  EXPECT_NE(emit_throw_message(m).find("rename"), std::string::npos);
}

// ...and a state named like a compute() local (`residual`) is rejected too.
TEST(NumSimMaterialTarget, RejectsResidualLocalNameCollision) {
  ConstitutiveModel m("LocalClash");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto residual = m.add_scalar_state_variable(
      "residual", make_expression<scalar_constant>(0.0)); // collides
  m.add_scalar_residual_equation(residual, residual.current - c * trace(eps));
  m.add_output("stress", residual.current * eps);
  EXPECT_NE(emit_throw_message(m).find("rename"), std::string::npos);
}

// The strain-coupled consistent tangent is a follow-up (PR 2b): an algorithmic
// tangent on a residual material must be rejected with a message naming PR 2b,
// not silently dropped.
// Phase 2b: the strain-coupled consistent tangent dσ/dε for a Mode-B residual
// material. σ = z·ε, R = z − c·tr(ε) ⇒
//   dσ/dε = ∂σ/∂ε + ∂σ/∂z ⊗ dz/dε = z·I⁴ˢ + ε ⊗ (c·I),   dz/dε = −∂R/∂ε/∂R/∂z = c·I.
// The correction term ε⊗(c·I) is what a naive ∂σ/∂ε alone would drop.
TEST(NumSimMaterialTarget, EmitsConsistentTangentForResidualMaterial) {
  auto m = build_return_map();
  m.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  auto const h = header_of(NumSimMaterialTarget{}.emit(m));

  // Emitted as a rank-4 output, bound to compute() like every output.
  EXPECT_NE(h.find("add_output<tmech::tensor<value_type, 3, 4>>"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("\"dstress_dstrain\", &ReturnMap::compute"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("tmech::tensor<value_type, 3, 4>& m_out_dstress_dstrain;"),
            std::string::npos)
      << h;

  // The implicit correction ∂σ/∂z ⊗ dz/dε: ∂σ/∂z = ε (the strain), assembled as
  // an outer product. Pinning this is the whole point — a dropped coupling term
  // would still leave a plausible-but-wrong ∂σ/∂ε.
  EXPECT_NE(h.find("tmech::outer_product<tmech::sequence<1, 2>, "
                   "tmech::sequence<3, 4>>(strain,"),
            std::string::npos)
      << h;
  // The explicit base term ∂σ/∂ε = z·I⁴ˢ (minor-symmetric identity).
  EXPECT_NE(h.find("tmech::otimesu(tmech::eye<double, 3, 2>()"),
            std::string::npos)
      << h;
  EXPECT_NE(h.find("m_out_dstress_dstrain ="), std::string::npos) << h;

  // The tangent is evaluated AFTER the solve (uses the converged z).
  auto const solve_at = h.find("m_solver.get().solve(eval)");
  auto const tangent_at = h.find("m_out_dstress_dstrain =");
  ASSERT_NE(solve_at, std::string::npos);
  ASSERT_NE(tangent_at, std::string::npos);
  EXPECT_LT(solve_at, tangent_at) << "tangent must use the converged state";
}

// A tangent can only differentiate a TENSOR (stress) output.
TEST(NumSimMaterialTarget, RejectsTangentOfScalarOutputOnResidualMaterial) {
  auto m = build_return_map();
  m.add_output("scalar_out", 2.0 * make_expression<scalar>("c"));
  m.add_algorithmic_tangent("dsc_dstrain", "scalar_out", "strain");
  EXPECT_NE(emit_throw_message(m).find("is scalar"), std::string::npos);
}

// wrt_input must be a declared tensor input.
TEST(NumSimMaterialTarget, RejectsTangentWrtUnknownInputOnResidualMaterial) {
  auto m = build_return_map();
  m.add_algorithmic_tangent("dstress_dghost", "stress", "ghost");
  EXPECT_NE(emit_throw_message(m).find("not a declared tensor input"),
            std::string::npos);
}

// The tangent output name shares the emitted-identifier collision surface. A
// name like "solver" is not a recipe symbol (so the request-time
// availability check passes) but collides with the emitted m_solver member —
// caught by the emit-time guard.
TEST(NumSimMaterialTarget, RejectsTangentNameCollidingWithEmittedMember) {
  auto m = build_return_map();
  m.add_algorithmic_tangent("solver", "stress", "strain");
  EXPECT_NE(emit_throw_message(m).find("collides"), std::string::npos);
}

// A residual material needs at least one output to anchor compute() (the output
// pull drives the solve) — reject loudly rather than emit an un-driven solve.
TEST(NumSimMaterialTarget, RejectsResidualWithoutOutput) {
  ConstitutiveModel m("NoOutput");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps));
  EXPECT_NE(emit_throw_message(m).find("at least one output"),
            std::string::npos);
}

// Scalar inputs into a residual material are a follow-up — rejected for now.
TEST(NumSimMaterialTarget, RejectsScalarInputOnResidualMaterial) {
  ConstitutiveModel m("ScalarIn");
  auto c = m.add_parameter("c", 2.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto temp = m.add_scalar_input("temperature");
  auto z =
      m.add_scalar_state_variable("z", make_expression<scalar_constant>(0.0));
  m.add_scalar_residual_equation(z, z.current - c * trace(eps) - temp);
  m.add_output("stress", z.current * eps);
  EXPECT_NE(emit_throw_message(m).find("scalar input"), std::string::npos);
}

} // namespace
} // namespace numsim::codegen
