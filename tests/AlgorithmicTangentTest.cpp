// Phase 3b-1 (issue #35): AlgorithmicTangentPass scaffolding.
//
// These cover the consistent-tangent machinery that ships in 3b-1: the IR
// opt-in (`add_algorithmic_tangent`), the pass plumbing/ordering, the EXPLICIT
// tangent `∂σ/∂ε` (computed today via cas::diff(tensor,tensor) — no upstream
// dependency), and the `∂σ/∂x` stub seam that is blocked on numsim-cas#275.
//
// The strain-coupled implicit correction (`∂σ/∂x · dx/dε`) is intentionally
// NOT exercised through the pass: with the current scalar-residual Newton
// machinery `dx/dε ≡ 0`, so `∂σ/∂ε` is the exact tangent and the seam is only
// reached via the macro-guarded block in AlgorithmicTangentPass::run. The seam
// is unit-tested directly below to pin the flip-on point.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace numsim::codegen {

namespace {

// σ = 2μ ε — a representable tensor stress. Its consistent tangent
// dσ/dε = 2μ I⁴ˢ is strain-independent, so it is fully computable today.
auto build_elastic_with_tangent() -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("ElasticTangent");
  auto mu = m.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  return m;
}

TEST(AlgorithmicTangent, OptInReflectsState) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("stress", eps);
  EXPECT_FALSE(m.algorithmic_tangent_enabled());
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_TRUE(m.algorithmic_tangent_enabled());
  ASSERT_EQ(m.tangents().size(), 1u);
  EXPECT_EQ(m.tangents()[0].name, "dstress_deps");
  EXPECT_EQ(m.tangents()[0].of_output, "stress");
  EXPECT_EQ(m.tangents()[0].wrt_input, "eps");
}

// ─── Phase 5: tangent identity through canonical_arguments + MOOSE wiring ───

// Closes PR #80 round-2 MAJOR-3: the tangent's identity must survive the
// canonical_arguments projection (it was previously dropped → indistinguishable
// from a user rank-4 output). Derived STRUCTURALLY from the request, so it's
// present even before AlgorithmicTangentPass runs (the pre-pass model a backend
// sees).
TEST(AlgorithmicTangent, CanonicalArgumentsTagsTangentAsTensorTangentOutput) {
  auto m = build_elastic_with_tangent();
  auto const args = canonical_arguments(RecipeView{m});
  using R = ArgSpec::Role;
  auto const *tan = [&]() -> ArgSpec const * {
    for (auto const &a : args)
      if (a.name == "dstress_deps") return &a;
    return nullptr;
  }();
  ASSERT_NE(tan, nullptr);
  EXPECT_EQ(tan->role, R::TensorTangentOutput);
  EXPECT_EQ(tan->rank, 4u);
  EXPECT_EQ(tan->dim, 3u);
  // The ordinary stress output stays a plain TensorOutput.
  for (auto const &a : args)
    if (a.name == "stress") EXPECT_EQ(a.role, R::TensorOutput);
}

// The MOOSE call-site arg count must equal canonical_arguments — i.e. the
// structurally-derived tangent keeps the call-site and the generated signature
// in lockstep even though the tangent OutputDecl only exists on the pass's
// working copy.
TEST(AlgorithmicTangent, MooseCallSiteArityIncludesTangent) {
  auto m = build_elastic_with_tangent();
  auto const source = MooseMaterialTarget{}.emit(m).at(1).contents;
  auto const open = source.find("ElasticTangent_compute(");
  ASSERT_NE(open, std::string::npos) << source;
  auto const lp = source.find('(', open);
  auto const rp = source.find(')', lp);
  auto const inner = source.substr(lp + 1, rp - lp - 1);
  auto const n = std::size_t{1} +
                 static_cast<std::size_t>(
                     std::count(inner.begin(), inner.end(), ','));
  EXPECT_EQ(n, canonical_arguments(RecipeView{m}).size());
}

// Phase 5: the consistent tangent is wired to MOOSE's framework _Jacobian_mult
// slot — member, declared property (bare conventional name), write adaptor, and
// call-site argument.
TEST(AlgorithmicTangent, MooseWiresTangentToJacobianMult) {
  auto files = MooseMaterialTarget{}.emit(build_elastic_with_tangent());
  std::string const header = files[0].contents;
  std::string const source = files[1].contents;
  EXPECT_NE(header.find("MaterialProperty<RankFourTensor> & _Jacobian_mult;"),
            std::string::npos)
      << header;
  EXPECT_NE(source.find("declareProperty<RankFourTensor>(\"Jacobian_mult\")"),
            std::string::npos)
      << source;
  EXPECT_NE(source.find("Jacobian_mult_ad(_Jacobian_mult[_qp].dataPointer())"),
            std::string::npos)
      << source;
  EXPECT_NE(source.find("      Jacobian_mult_ad"), std::string::npos) << source;
  // The tangent is NOT emitted as a model-scoped property a solver can't find.
  EXPECT_EQ(source.find("ElasticTangent_dstress_deps"), std::string::npos)
      << source;
}

// A recipe with no tangent emits no _Jacobian_mult (regression: the wiring is
// opt-in and doesn't leak into ordinary materials).
TEST(AlgorithmicTangent, MooseWithoutTangentHasNoJacobianMult) {
  using namespace numsim::cas;
  ConstitutiveModel m("Plain");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  auto files = MooseMaterialTarget{}.emit(m);
  EXPECT_EQ(files[0].contents.find("Jacobian_mult"), std::string::npos);
  EXPECT_EQ(files[1].contents.find("Jacobian_mult"), std::string::npos);
}

// Standalone has no _Jacobian_mult concept — the tangent is a plain rank-4
// out-parameter of the generated function.
TEST(AlgorithmicTangent, StandaloneEmitsTangentAsPlainOutParam) {
  auto const src =
      StandaloneCxxTarget{}.emit(build_elastic_with_tangent()).at(0).contents;
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
  EXPECT_EQ(src.find("Jacobian_mult"), std::string::npos) << src;
}

// PR #82 review: a regular output named "Jacobian_mult" collides with the MOOSE
// backend's hardcoded consistent-tangent member when a tangent is present.
TEST(AlgorithmicTangent, MooseRejectsOutputNamedJacobianMultWithTangent) {
  using namespace numsim::cas;
  ConstitutiveModel m("Clash");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_output("Jacobian_mult", 2 * mu * eps); // collides with the member
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_THROW((void)MooseMaterialTarget{}.emit(m), std::runtime_error);
  // Without a tangent the same output name is fine (no _Jacobian_mult member).
  ConstitutiveModel ok("NoTan");
  auto mu2 = ok.add_parameter("mu", 0.5);
  auto eps2 = ok.add_tensor_input("eps", 3, 2, roles::Strain);
  ok.add_output("Jacobian_mult", 2 * mu2 * eps2);
  EXPECT_NO_THROW((void)MooseMaterialTarget{}.emit(ok));
}

// MOOSE has a single consistent-tangent slot; more than one tangent is rejected.
TEST(AlgorithmicTangent, MooseRejectsMultipleTangents) {
  using namespace numsim::cas;
  ConstitutiveModel m("MultiTan");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("t1", "stress", "eps");
  m.add_algorithmic_tangent("t2", "stress", "eps");
  EXPECT_THROW((void)MooseMaterialTarget{}.emit(m), std::runtime_error);
}

// The RHS expression assigned to `<lhs> = ` in the emitted source, trimmed.
auto rhs_of(std::string const &src, std::string const &lhs) -> std::string {
  auto const p = src.find(lhs + " = ");
  if (p == std::string::npos) {
    return {};
  }
  auto const start = p + lhs.size() + 3;
  auto const end = src.find(';', start);
  return src.substr(start, end - start);
}

TEST(AlgorithmicTangent, EmitsRank4TangentAsOutputParameter) {
  auto const src = build_elastic_with_tangent().emit_compute_function();
  // The synthesised tangent becomes a `<name>_out` output parameter.
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
}

// Round-2 review (test-quality MAJOR-2): pin the tangent VALUE to its output,
// not just that `otimesu` appears somewhere. dσ/dε for σ = 2μ ε must be the
// SCALED rank-4 identity — a distinct expression from the stress, not a copy of
// σ or ε.
TEST(AlgorithmicTangent, TangentValueIsScaledRank4IdentityDistinctFromStress) {
  auto const src = build_elastic_with_tangent().emit_compute_function();
  auto const stress_rhs = rhs_of(src, "stress_out");
  auto const tan_rhs = rhs_of(src, "dstress_deps_out");
  ASSERT_FALSE(tan_rhs.empty()) << src;
  ASSERT_FALSE(stress_rhs.empty()) << src;
  // The tangent is NOT a copy of the stress, nor a bare copy of the strain.
  EXPECT_NE(tan_rhs, stress_rhs) << src;
  EXPECT_NE(tan_rhs, "eps") << src;
  // It must be built from the rank-4 identity, scaled. Follow the assigned
  // temp one hop and require it to be a product (`a * b`).
  auto const def = rhs_of(src, "auto " + tan_rhs);
  EXPECT_NE(def.find('*'), std::string::npos)
      << "tangent temp '" << tan_rhs << "' should be a scaled product\n"
      << src;
  EXPECT_NE(src.find("tmech::otimesu(tmech::eye<double, 3, 2>(), "
                     "tmech::eye<double, 3, 2>())"),
            std::string::npos)
      << src;
}

TEST(AlgorithmicTangent, ExplicitTangentEmitsWithoutStub) {
  // σ = 2μ ε is strain-only ⇒ dσ/dε is computed via cas::diff(tensor,tensor).
  // No ∂σ/∂x term is needed, so emit must NOT hit the numsim-cas#275 seam.
  EXPECT_NO_THROW((void)build_elastic_with_tangent().emit_compute_function());
}

TEST(AlgorithmicTangent, RegistersAfterValidationOnlyForElasticRecipe) {
  // A pure-elastic recipe (no evolution equations / no local Newton) still
  // gets a tangent — the pass precondition is the validation floor, not
  // newton_lowered. Exercises that the pipeline runs the tangent pass here.
  EXPECT_NO_THROW(build_elastic_with_tangent().validate());
  auto const src = build_elastic_with_tangent().emit_compute_function();
  EXPECT_NE(src.find("stress_out"), std::string::npos) << src;
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
}

TEST(AlgorithmicTangent, DuplicateTangentNameThrows) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("stress", eps);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_THROW(m.add_algorithmic_tangent("dstress_deps", "stress", "eps"),
               std::runtime_error);
}

TEST(AlgorithmicTangent, TangentNameCollidingWithExistingOutputThrows) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("stress", eps);
  // Name reserved against the output set at request time.
  EXPECT_THROW(m.add_algorithmic_tangent("stress", "stress", "eps"),
               std::runtime_error);
}

TEST(AlgorithmicTangent, UnknownStressOutputThrowsAtEmit) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("stress", eps);
  m.add_algorithmic_tangent("t", "does_not_exist", "eps");
  EXPECT_THROW((void)m.emit_compute_function(), std::runtime_error);
}

TEST(AlgorithmicTangent, ScalarStressOutputRejectedAtEmit) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto k = m.add_parameter("k", 1.0);
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("p", k); // scalar output — dσ/dε needs a tensor σ
  m.add_algorithmic_tangent("t", "p", "eps");
  EXPECT_THROW((void)m.emit_compute_function(), std::runtime_error);
}

TEST(AlgorithmicTangent, UnknownStrainInputThrowsAtEmit) {
  using namespace numsim::cas;
  ConstitutiveModel m("M");
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("stress", eps);
  m.add_algorithmic_tangent("t", "stress", "not_an_input");
  EXPECT_THROW((void)m.emit_compute_function(), std::runtime_error);
}

// Locks the current rank-4 identity emission (PR #80 review, math finding Q3).
// dσ/dε for σ = 2μ ε renders as 2μ · tmech::otimesu(eye,eye) — the
// NON-symmetrized identity δ_ik δ_jl. Correct contracting against symmetric ε,
// but lacking minor symmetry. This test documents the present behavior so the
// planned symmetric-tensor-space fix (→ P_sym / otimesu+otimesl) is a
// deliberate, reviewed change rather than a silent shift.
// A consistent stress-strain tangent must have minor symmetry C_ijkl = C_ijlk.
// Because the strain is declared roles::Strain (symmetric), `cas::diff` returns
// the symmetric rank-4 identity P_sym = ½(I⊗ᵘI + I⊗ˡI), so dσ/dε for σ = 2μ ε
// emits 2μ·½(otimesu + otimesl). (Superseded the earlier non-symmetrized lock.)
TEST(AlgorithmicTangent, ElasticTangentIsMinorSymmetric) {
  auto const src = build_elastic_with_tangent().emit_compute_function();
  // Both halves of the minor-symmetric identity are present, scaled by ½.
  EXPECT_NE(src.find("tmech::otimesu"), std::string::npos) << src;
  EXPECT_NE(src.find("tmech::otimesl"), std::string::npos)
      << "tangent lost minor symmetry — the symmetric tensor space on the "
         "strain input is not reaching cas::diff.\n"
      << src;
  EXPECT_NE(src.find("0.5 *"), std::string::npos) << src;
  // The 2μ factor is shared with the stress via CSE (no recomputation).
  EXPECT_NE(src.find("2.0 * mu"), std::string::npos) << src;
}

// A plain (roles::Other) strain input carries no symmetric space, so its
// tangent stays the non-symmetrized identity — confirms the change is
// opt-in via the symmetric role, not a blanket behavior shift.
TEST(AlgorithmicTangent, PlainInputTangentStaysNonSymmetrized) {
  using namespace numsim::cas;
  ConstitutiveModel m("PlainTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2); // no role → unconstrained
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  auto const src = m.emit_compute_function();
  EXPECT_NE(src.find("tmech::otimesu"), std::string::npos) << src;
  EXPECT_EQ(src.find("tmech::otimesl"), std::string::npos) << src;
}

// Pins the scaffold's flip-on point: the ∂σ/∂x seam throws a precise
// numsim-cas#275 diagnostic until the upstream diff(tensor, scalar) lands.
TEST(AlgorithmicTangent, DiffTensorWrtScalarSeamThrowsUntilCas275) {
  using namespace numsim::cas;
  auto eps = make_expression<tensor>("eps", 3, std::size_t{2});
  auto x = make_expression<scalar>("x");
  try {
    (void)detail::diff_tensor_wrt_scalar(eps, x);
    FAIL() << "expected the diff(tensor,scalar) seam to throw";
  } catch (std::runtime_error const &e) {
    EXPECT_NE(std::string(e.what()).find("numsim-cas#275"), std::string::npos)
        << e.what();
  }
}

// Round-2 review (test-quality MAJOR-5): the pass running ALONGSIDE local Newton
// — i.e. with `pctx.newton_segments` populated, after LocalNewtonLoweringPass —
// had zero coverage. Every other test is pure-elastic (empty segments). This
// exercises the real pipeline interaction the static_assert / ordering contract
// is about. The evolution is strain-independent so dx/dε ≡ 0 and the explicit
// tangent stays exact.
TEST(AlgorithmicTangent, TangentCoexistsWithLocalNewtonStateVariable) {
  using namespace numsim::cas;
  ConstitutiveModel m("HardeningTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto K = m.add_parameter("K", 1.0);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto a = m.add_scalar_state_variable("a", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(a, K * a.current); // strain-independent rate
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.enable_local_newton();
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");

  std::string src;
  ASSERT_NO_THROW(src = m.emit_compute_function());
  // Both the in-function Newton solve (state var → a_out) and the tangent emit.
  EXPECT_NE(src.find("a_out"), std::string::npos) << src;
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
  // The Newton iteration loop is present (the converged-state seam is live).
  EXPECT_NE(src.find("for ("), std::string::npos) << src;
}

// Round-2 review (test-quality MINOR-4): the pass loops over `pending` and
// reserves N outputs, but only N=1 was ever tested. Two tangents off the same
// stress (e.g. dσ/dε and a second alias) exercise the loop + reserve.
TEST(AlgorithmicTangent, MultipleTangentsAllEmit) {
  using namespace numsim::cas;
  ConstitutiveModel m("MultiTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  m.add_algorithmic_tangent("dstress_deps_alias", "stress", "eps");
  ASSERT_EQ(m.tangents().size(), 2u);
  auto const src = m.emit_compute_function();
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
  EXPECT_NE(src.find("dstress_deps_alias_out"), std::string::npos) << src;
}

} // namespace

} // namespace numsim::codegen
