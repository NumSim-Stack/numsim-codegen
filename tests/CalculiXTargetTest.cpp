#include <numsim_codegen/numsim_codegen.h>
#include <numsim_codegen/targets/calculix_external.h>
#include <numsim_codegen/targets/calculix_umat.h>
#include <numsim_codegen/targets/target_factory.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/operators/tensor_to_scalar/tensor_to_scalar_with_tensor_mul.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_trace.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

namespace {

// Full isotropic linear elasticity σ = λ·tr(ε)·I + 2μ·ε with the consistent
// tangent — the supported (stateless) shape.
auto build_full_elastic(std::string name = "LinearElastic") -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m(std::move(name));
  auto lambda = m.add_parameter("lambda", 1.0);
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto I = make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  auto tr = make_expression<tensor_trace>(eps);
  auto sigma =
      lambda * make_expression<tensor_to_scalar_with_tensor_mul>(I, tr) +
      2 * mu * eps;
  m.add_output("stress", sigma, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  return m;
}

// A recipe whose driving tensor input is a NON-symmetric deformation gradient F
// (roles::DeformationGradient) rather than a symmetric strain — the abq_std
// boundary would silently truncate its antisymmetric part, so both targets must
// reject it.
auto build_deformation_gradient_recipe() -> ConstitutiveModel {
  using namespace numsim::cas;
  ConstitutiveModel m("DefGrad");
  auto mu = m.add_parameter("mu", 0.5);
  auto F = m.add_tensor_input("F", 3, 2, roles::DeformationGradient);
  m.add_output("stress", 2 * mu * F, roles::Stress);
  m.add_algorithmic_tangent("dstress_dF", "stress", "F");
  return m;
}

} // namespace

TEST(CalculiXTarget, EmitsSingleSourceFile) {
  CalculiXUMATTarget target;
  auto m = build_full_elastic();
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].filename, "LinearElastic_umat.cpp");
  EXPECT_EQ(files[0].kind, EmittedFile::Kind::Source);
  EXPECT_TRUE(files[0].install_subdir.empty());
}

TEST(CalculiXTarget, ExposesUmatUserExternC) {
  CalculiXUMATTarget target;
  auto m = build_full_elastic();
  auto const &src = target.emit(m)[0].contents;
  EXPECT_NE(src.find("extern \"C\" void umat_user_("), std::string::npos)
      << src;
}

TEST(CalculiXTarget, EmbedsComputeAndReadsConstantsFromElconloc) {
  CalculiXUMATTarget target;
  auto m = build_full_elastic();
  auto const &src = target.emit(m)[0].contents;
  // The Layer-2 compute is embedded.
  EXPECT_NE(src.find("LinearElastic_compute("), std::string::npos);
  // *USER MATERIAL constants read in declaration order (λ then μ).
  EXPECT_NE(src.find("double const lambda = elconloc[0];"), std::string::npos);
  EXPECT_NE(src.find("double const mu = elconloc[1];"), std::string::npos);
}

TEST(CalculiXTarget, UsesAbqStdBoundaryOrdering) {
  CalculiXUMATTarget target;
  auto m = build_full_elastic();
  auto const &src = target.emit(m)[0].contents;
  // abq_std ordering {11,22,33,12,13,23} for both the strain read and the
  // stress/tangent write — no plain voigt, no engineering-shear scaling.
  EXPECT_NE(src.find("tmech::abq_std<3, false>"), std::string::npos);
  EXPECT_EQ(src.find("tmech::voigt<"), std::string::npos);
}

TEST(CalculiXTarget, PacksStiffColumnMajorUpper) {
  CalculiXUMATTarget target;
  auto m = build_full_elastic();
  auto const &src = target.emit(m)[0].contents;
  // Column-major upper-triangular packing index k = i + j*(j+1)/2, guarded by
  // icmd (stress-only requests skip the tangent).
  EXPECT_NE(src.find("stiff[i + j * (j + 1) / 2]"), std::string::npos);
  EXPECT_NE(src.find("if (*icmd != 3)"), std::string::npos);
}

TEST(CalculiXTarget, FactorySelectorCalculix) {
  auto target = make_target("calculix");
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->target_name(), "CalculiXUMAT");
}

// ── Scope guards (stateless linear-elastic first cut) ────────────────────────

TEST(CalculiXTarget, RejectsMissingConsistentTangent) {
  using namespace numsim::cas;
  CalculiXUMATTarget target;
  ConstitutiveModel m("NoTangent");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  // No add_algorithmic_tangent → CalculiX would have no `stiff`.
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

TEST(CalculiXTarget, RejectsScalarInput) {
  using namespace numsim::cas;
  CalculiXUMATTarget target;
  ConstitutiveModel m("WithScalarInput");
  auto mu = m.add_parameter("mu", 0.5);
  auto T = m.add_scalar_input("T", roles::Temperature);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto sigma = 2 * mu * (1 + T) * eps;
  m.add_output("stress", sigma, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

TEST(CalculiXTarget, RejectsMultipleTensorInputs) {
  using namespace numsim::cas;
  CalculiXUMATTarget target;
  ConstitutiveModel m("TwoTensorInputs");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto eps_p = m.add_tensor_input(
      "eps_p", 3, 2,
      Role{.name = "plastic_strain", .is_symmetric = true, .expected_rank = 2});
  auto sigma = 2 * mu * (eps - eps_p);
  m.add_output("stress", sigma, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

TEST(CalculiXTarget, RejectsStateVariable) {
  using namespace numsim::cas;
  CalculiXUMATTarget target;
  ConstitutiveModel m("WithState");
  auto K = m.add_parameter("K", 1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  m.enable_local_newton();
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

// H3: a non-symmetric tensor input (deformation gradient) must be rejected — the
// abq_std Voigt boundary is symmetric and would silently drop its skew part.
TEST(CalculiXTarget, RejectsNonSymmetricTensorInput) {
  CalculiXUMATTarget target;
  EXPECT_THROW(target.emit(build_deformation_gradient_recipe()),
               std::runtime_error);
}

TEST(CalculiXTarget, RejectsMultipleTensorOutputs) {
  using namespace numsim::cas;
  CalculiXUMATTarget target;
  ConstitutiveModel m("TwoOutputs");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_output("extra", 3 * mu * eps, roles::Stress); // second rank-2 output
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

// ── CalculiXExternalTarget ───────────────────────────────────────────────────

TEST(CalculiXExternalTarget, EmitsSingleSourceFileNamedExt) {
  CalculiXExternalTarget target;
  auto files = target.emit(build_full_elastic());
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].filename, "LinearElastic_ext.cpp");
  EXPECT_EQ(files[0].kind, EmittedFile::Kind::Source);
}

TEST(CalculiXExternalTarget, ExposesNcgUmatAndAbqStdBoundary) {
  CalculiXExternalTarget target;
  auto const &src = target.emit(build_full_elastic())[0].contents;
  EXPECT_NE(src.find("extern \"C\" void NCG_UMAT("), std::string::npos);
  EXPECT_NE(src.find("tmech::abq_std<3, false>"), std::string::npos);
  EXPECT_NE(src.find("stiff[i + j * (j + 1) / 2]"), std::string::npos)
      << "shared column-major packing expected";
}

// The constants must be read from MPROPS INSIDE evaluate() (per call), and the
// thread_local evaluator must be stateless — the regression for the cache bug.
TEST(CalculiXExternalTarget, ReadsConstantsPerCallNotCached) {
  CalculiXExternalTarget target;
  auto const &src = target.emit(build_full_elastic())[0].contents;
  EXPECT_NE(src.find("double const lambda = mprops[0];"), std::string::npos);
  EXPECT_NE(src.find("thread_local ncg_material const"), std::string::npos);
  // The old buggy form cached constants via optional::emplace(mprops).
  EXPECT_EQ(src.find(".emplace(mprops)"), std::string::npos)
      << "constants must not be cached from the first call";
}

// library_name: ccx uppercases the deck name; underscores are dropped so the
// first '_' splits LIB from FUNC. "J2_Plastic" → libJ2PLASTIC.so / @J2PLASTIC_...
TEST(CalculiXExternalTarget, DeckNameIsUppercasedAlnum) {
  CalculiXExternalTarget target;
  auto const &src = target.emit(build_full_elastic("J2_Plastic"))[0].contents;
  EXPECT_NE(src.find("libJ2PLASTIC.so"), std::string::npos) << src;
  EXPECT_NE(src.find("@J2PLASTIC_NCG_UMAT"), std::string::npos) << src;
}

TEST(CalculiXExternalTarget, RejectsNonSymmetricTensorInput) {
  CalculiXExternalTarget target;
  EXPECT_THROW(target.emit(build_deformation_gradient_recipe()),
               std::runtime_error);
}

TEST(CalculiXExternalTarget, RejectsStateVariable) {
  using namespace numsim::cas;
  CalculiXExternalTarget target;
  ConstitutiveModel m("ExtState");
  auto K = m.add_parameter("K", 1.0);
  auto alpha =
      m.add_scalar_state_variable("alpha", make_expression<scalar_constant>(0.0));
  m.add_scalar_evolution_equation(alpha, K * alpha.current);
  m.enable_local_newton();
  EXPECT_THROW(target.emit(m), std::runtime_error);
}

TEST(CalculiXExternalTarget, RejectsEmptyLibraryName) {
  // A model name with no alphanumeric characters ("_" is a valid C++ identifier)
  // maps to an empty library name → must throw at emit, not fail inside ccx.
  CalculiXExternalTarget target;
  EXPECT_THROW(target.emit(build_full_elastic("_")), std::runtime_error);
}

TEST(CalculiXExternalTarget, FactorySelector) {
  auto target = make_target("calculix_external");
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->target_name(), "CalculiXExternal");
}

} // namespace numsim::codegen
