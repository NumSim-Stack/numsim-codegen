#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

TEST(Recipe, ScalarRecipeProducesValidFunctionSignature) {
  ConstitutiveModel m("ScaleAdd");
  auto a = m.add_parameter("a", 2.0);
  auto b = m.add_parameter("b", 3.0);
  auto x = m.add_scalar_input("x");
  m.add_output("y", a * x + b);

  auto src = m.emit_compute_function();

  // Function name must match the model name.
  EXPECT_NE(src.find("void ScaleAdd_compute"), std::string::npos)
      << "got: " << src;
  // Parameters and inputs both appear as `const` doubles.
  EXPECT_NE(src.find("double const x"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("double const a"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("double const b"), std::string::npos) << "got: " << src;
  // Output is a non-const double reference.
  EXPECT_NE(src.find("double &y_out"), std::string::npos) << "got: " << src;
  // The final assignment writes to y_out.
  EXPECT_NE(src.find("y_out = "), std::string::npos) << "got: " << src;
}

TEST(Recipe, TensorRecipeProducesTemplatedSignature) {
  ConstitutiveModel m("Scale");
  auto k = m.add_parameter("k", 1.5);
  auto eps = m.add_tensor_input("eps", 3, 2);
  m.add_output("y", k * eps);

  auto src = m.emit_compute_function();

  // Tensor arguments are template parameters so the caller can pass a
  // tmech::tensor, a tmech::adaptor, or any other tmech::tensor_base
  // subclass without forcing a materialised copy.
  EXPECT_NE(src.find("template <typename T0, typename T1>"),
            std::string::npos)
      << "got: " << src;
  EXPECT_NE(src.find("T0 const &eps"), std::string::npos) << "got: " << src;
  EXPECT_NE(src.find("T1 &y_out"), std::string::npos) << "got: " << src;
}

TEST(Recipe, LinearElasticityShearTermEmitsSomething) {
  // Phase-A smoke test: σ = 2μ ε (shear term only). Full LE needs the
  // `t2s * tensor` operator that numsim-cas hasn't exposed yet.
  ConstitutiveModel m("LinearElasticShear");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2);

  auto sigma = 2 * mu * eps;
  m.add_output("stress", sigma);

  auto src = m.emit_compute_function();

  EXPECT_NE(src.find("mu"), std::string::npos) << "got:\n" << src;
  EXPECT_NE(src.find("eps"), std::string::npos) << "got:\n" << src;
  EXPECT_NE(src.find("stress_out = "), std::string::npos) << "got:\n" << src;
  // Templated parameters (T0, T1) replace concrete tmech::tensor.
  EXPECT_NE(src.find("template <"), std::string::npos) << "got:\n" << src;
}

TEST(Recipe, MultipleOutputsShareCseAcrossBody) {
  // If two outputs share a subexpression, it should be emitted once.
  ConstitutiveModel m("Shared");
  auto a = m.add_parameter("a", 1.0);
  auto x = m.add_scalar_input("x");

  auto shared = sin(a * x);
  m.add_output("y1", shared + 1);
  m.add_output("y2", shared * 2);

  auto src = m.emit_compute_function();

  // sin(a*x) should appear exactly once in the body — both outputs reference
  // the same temporary.
  std::size_t sin_count = 0;
  for (std::size_t pos = 0;
       (pos = src.find("std::sin", pos)) != std::string::npos; ++pos) {
    ++sin_count;
  }
  EXPECT_EQ(sin_count, 1u)
      << "Shared sin(a*x) should be emitted exactly once. got:\n"
      << src;
}

} // namespace numsim::codegen
