// linear_elastic — the "generate the material on the fly" program for the
// CalculiX test harness. Emits argv[1] = the CalculiXExternalTarget .so source
// for isotropic linear elasticity σ = λ·tr(ε)·I + 2μ·ε with λ=1.3, μ=0.7, which
// is the exact equivalent of the gold deck's *ELASTIC E=1.855, ν=0.325.
//
// The harness compiles this against numsim::codegen, runs it to produce
// LinearElastic_ext.cpp, builds that into libLINEARELASTIC.so, runs test.inp
// through ccx, and diffs the .dat against gold/.

#include <numsim_codegen/numsim_codegen.h>
#include <numsim_codegen/targets/calculix_external.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/operators/tensor_to_scalar/tensor_to_scalar_with_tensor_mul.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_trace.h>

#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <LinearElastic_ext.cpp>\n";
    return 1;
  }
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("LinearElastic");
  auto lambda = m.add_parameter("lambda", 1.3, "First Lame parameter");
  auto mu = m.add_parameter("mu", 0.7, "Shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto I = make_expression<identity_tensor>(std::size_t{3}, std::size_t{2});
  auto tr = make_expression<tensor_trace>(eps);
  auto sigma =
      lambda * make_expression<tensor_to_scalar_with_tensor_mul>(I, tr) +
      2 * mu * eps;
  m.add_output("stress", sigma, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");

  auto files = CalculiXExternalTarget{}.emit(m);
  if (files.size() != 1) {
    std::cerr << "expected one emitted file, got " << files.size() << "\n";
    return 1;
  }
  std::ofstream out(argv[1]);
  if (!out) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 1;
  }
  out << files[0].contents;
  return out ? 0 : 1;
}
