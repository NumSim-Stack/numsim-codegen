// Generator for the CalculiX end-to-end gate. CMake runs this at build time
// with two destination paths:
//   argv[1] = LinearElastic.h        (StandaloneCxx: full dense _compute)
//   argv[2] = LinearElastic_umat.cpp (CalculiXUMAT:  umat_user ABI wrapper)
//
// Both are emitted from the SAME recipe — full isotropic linear elasticity in
// Lamé form, σ = λ·tr(ε)·I + 2μ·ε, with the consistent tangent
// C = λ·I⊗I + 2μ·I_sym emitted via add_algorithmic_tangent. The driver
// (calculix_check_driver.cpp) FD-verifies the tangent through the standalone
// header (Phase 0 milestone) and checks the umat_user boundary against both the
// analytic answer and the _compute path (Phase 1 ABI gate).

#include <numsim_codegen/numsim_codegen.h>
#include <numsim_codegen/targets/calculix_umat.h>
#include <numsim_codegen/targets/standalone_cxx.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/operators/tensor_to_scalar/tensor_to_scalar_with_tensor_mul.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_trace.h>

#include <fstream>
#include <iostream>

namespace {

// Full isotropic linear elasticity, Lamé form:
//   σ = λ·tr(ε)·I + 2μ·ε
// Constants are declared λ then μ, which is the order CalculiX reads them from
// `*USER MATERIAL, CONSTANTS=2` into elconloc[0], elconloc[1].
auto build_linear_elastic() -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("LinearElastic");
  auto lambda = m.add_parameter("lambda", 1.0, "First Lame parameter");
  auto mu = m.add_parameter("mu", 0.5, "Shear modulus (second Lame parameter)");
  // roles::Strain marks eps symmetric, so cas::diff yields the minor-symmetric
  // rank-4 identity — the tangent is minor-symmetric, as a stress-strain
  // tangent must be (and as CalculiX's stiff packing assumes).
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);

  auto I = make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  auto tr = make_expression<tensor_trace>(eps); // t2s
  // λ·(tr(ε)·I): tensor_to_scalar_with_tensor_mul(I, tr) = tr(ε)·I, then ·λ.
  auto vol = lambda * make_expression<tensor_to_scalar_with_tensor_mul>(I, tr);
  auto sigma = vol + 2 * mu * eps;

  m.add_output("stress", sigma, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  return m;
}

auto write_single_file(std::vector<numsim::codegen::EmittedFile> const &files,
                       char const *out_path) -> int {
  if (files.size() != 1) {
    std::cerr << "expected single emitted file, got " << files.size() << "\n";
    return 1;
  }
  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "could not open '" << out_path << "' for writing\n";
    return 1;
  }
  out << files[0].contents;
  return out ? 0 : 1;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <LinearElastic.h> <LinearElastic_umat.cpp>\n";
    return 1;
  }

  auto const model = build_linear_elastic();

  if (int rc = write_single_file(
          numsim::codegen::StandaloneCxxTarget{}.emit(model), argv[1]);
      rc != 0) {
    return rc;
  }
  if (int rc = write_single_file(
          numsim::codegen::CalculiXUMATTarget{}.emit(model), argv[2]);
      rc != 0) {
    return rc;
  }
  return 0;
}
