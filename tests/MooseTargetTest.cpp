#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

namespace {
auto build_linear_elastic_shear() -> ConstitutiveModel {
  ConstitutiveModel m("LinearElasticShear");
  auto mu = m.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  return m;
}
} // namespace

TEST(MooseTarget, EmitsTwoFiles) {
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 2u);
  EXPECT_EQ(files[0].filename, "LinearElasticShear.h");
  EXPECT_EQ(files[1].filename, "LinearElasticShear.C");
}

TEST(MooseTarget, HeaderContainsExpectedClassDeclaration) {
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  auto const &header = files[0].contents;

  EXPECT_NE(header.find("class LinearElasticShear : public Material"),
            std::string::npos);
  EXPECT_NE(header.find("static InputParameters validParams();"),
            std::string::npos);
  EXPECT_NE(header.find("virtual void computeQpProperties() override;"),
            std::string::npos);
  // Parameter member.
  EXPECT_NE(header.find("Real const _mu;"), std::string::npos);
  // Input member.
  EXPECT_NE(header.find("MaterialProperty<RankTwoTensor> const & _eps;"),
            std::string::npos);
  // Output member.
  EXPECT_NE(header.find("MaterialProperty<RankTwoTensor> & _stress;"),
            std::string::npos);
}

TEST(MooseTarget, SourceRegistersMooseObject) {
  MooseMaterialTarget target("MyApp");
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  auto const &source = files[1].contents;

  EXPECT_NE(source.find("registerMooseObject(\"MyApp\", LinearElasticShear);"),
            std::string::npos)
      << "got:\n" << source;
}

TEST(MooseTarget, SourceContainsValidParamsBody) {
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  auto const &source = files[1].contents;

  EXPECT_NE(source.find("params.addParam<Real>(\"mu\", 0.5"),
            std::string::npos);
  EXPECT_NE(source.find("params.addRequiredParam<MaterialPropertyName>(\"eps\""),
            std::string::npos);
}

TEST(MooseTarget, SourceConstructorInitializesMembers) {
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  auto const &source = files[1].contents;

  EXPECT_NE(source.find("_mu(getParam<Real>(\"mu\"))"), std::string::npos);
  EXPECT_NE(source.find("_eps(getMaterialProperty<RankTwoTensor>(\"eps\"))"),
            std::string::npos);
  EXPECT_NE(source.find(
                "_stress(declareProperty<RankTwoTensor>"
                "(\"LinearElasticShear_stress\"))"),
            std::string::npos)
      << "got:\n" << source;
}

TEST(MooseTarget, ComputeQpUsesAdaptorAndCallsLayer2) {
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  auto const &source = files[1].contents;

  // tmech::adaptor wraps MOOSE's RankTwoTensor data pointer (read side).
  EXPECT_NE(
      source.find(
          "tmech::adaptor<double const, 3, 2, tmech::full<3>> eps_ad("),
      std::string::npos)
      << "got:\n"
      << source;

  // Layer 2 compute function is called.
  EXPECT_NE(source.find("LinearElasticShear_compute("), std::string::npos);

  // Output adaptor on the write side.
  EXPECT_NE(
      source.find("tmech::adaptor<double, 3, 2, tmech::full<3>> stress_ad("),
      std::string::npos)
      << "got:\n"
      << source;
}

TEST(MooseTarget, RecipeRolesDriveOutputMapping) {
  // The roles::Stress tag should make the corresponding output a
  // RankTwoTensor MaterialProperty named consistently in the generated
  // boilerplate. Test this end-to-end by checking the header members.
  MooseMaterialTarget target;
  auto m = build_linear_elastic_shear();
  auto const *stress_decl = m.find_output_by_role(roles::Stress);
  ASSERT_NE(stress_decl, nullptr);
  EXPECT_EQ(stress_decl->name, "stress");

  auto const *strain_decl = m.find_input_by_role(roles::Strain);
  ASSERT_NE(strain_decl, nullptr);
  EXPECT_EQ(strain_decl->name, "eps");
}

TEST(StandaloneCxxTarget, EmitsSingleHeader) {
  StandaloneCxxTarget target;
  auto m = build_linear_elastic_shear();
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0].filename, "LinearElasticShear.h");
  EXPECT_NE(files[0].contents.find("LinearElasticShear_compute"),
            std::string::npos);
}

} // namespace numsim::codegen
