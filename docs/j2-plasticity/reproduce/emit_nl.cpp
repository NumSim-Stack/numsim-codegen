// Emit J2 return-map materials with NONLINEAR isotropic hardening laws.
#include <numsim_codegen/numsim_codegen.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/tensor_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>
#include <cstdio>
#include <fstream>
using namespace numsim::codegen;
using namespace numsim::cas;

template <class F>
static void emit(const char* name, const char* path, F build_yield) {
  ConstitutiveModel m(name);
  auto G  = m.add_parameter("G", 80000.0);
  auto eps = m.add_tensor_input("strain", 3, 2, roles::Strain);
  auto dg  = m.add_scalar_state_variable("dgamma", make_expression<scalar_constant>(0.0));
  auto s_trial = (2.0 * G) * dev(eps);
  auto q = norm(s_trial);
  auto n = s_trial / q;
  auto sy = build_yield(m, dg);                       // σ_y(Δγ)  (nonlinear)
  m.add_scalar_residual_equation(dg, q - (2.0 * G) * dg.current - sy);
  m.add_output("stress", s_trial - (2.0 * G) * dg.current * n);
  m.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  auto files = NumSimMaterialTarget{}.emit(m);
  for (auto const& f : files)
    if (f.kind == EmittedFile::Kind::Header) {
      std::ofstream(path) << f.contents; std::printf("wrote %s\n", path);
    }
}

int main(int argc, char** argv) {
  // Voce (saturating):  σ_y = σy0 + Q·(1 − exp(−b·Δγ))
  emit("J2Voce", argv[1], [](ConstitutiveModel& m, auto dg) {
    auto sy0 = m.add_parameter("sy0", 250.0);
    auto Q   = m.add_parameter("Q", 300.0);
    auto b   = m.add_parameter("b", 400.0);
    return sy0 + Q * (make_expression<scalar_constant>(1.0) - exp((make_expression<scalar_constant>(0.0) - b) * dg.current));
  });
  // Swift power law:  σ_y = C·(ε0 + Δγ)^k   (finite slope at Δγ=0)
  emit("J2Swift", argv[2], [](ConstitutiveModel& m, auto dg) {
    auto C  = m.add_parameter("C", 900.0);
    auto e0 = m.add_parameter("e0", 0.004);
    auto k  = m.add_parameter("k", 0.35);
    return C * pow(e0 + dg.current, k);
  });
  return 0;
}
