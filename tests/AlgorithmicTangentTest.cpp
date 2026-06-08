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

TEST(AlgorithmicTangent, EmitsRank4TangentAsOutputParameter) {
  auto const src = build_elastic_with_tangent().emit_compute_function();
  // The synthesised tangent becomes a `<name>_out` output parameter.
  EXPECT_NE(src.find("dstress_deps_out"), std::string::npos) << src;
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
TEST(AlgorithmicTangent, ElasticTangentEmitsNonSymmetrizedIdentityForNow) {
  auto const src = build_elastic_with_tangent().emit_compute_function();
  EXPECT_NE(src.find("tmech::otimesu"), std::string::npos) << src;
  // The 2μ factor is shared with the stress via CSE (no recomputation).
  EXPECT_NE(src.find("2.0 * mu"), std::string::npos) << src;
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

} // namespace

} // namespace numsim::codegen
