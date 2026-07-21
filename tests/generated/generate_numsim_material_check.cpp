// Phase B end-to-end gate — generator half.
//
// Emits NumSimMaterialTarget rate-function materials and writes their headers to
// argv[1] (linear) and argv[2] (nonlinear). The companion driver compiles these
// generated headers against the REAL numsim-materials / numsim-core headers and
// runs them through the real rk_integrator — proving the emitted code conforms
// to the solver contract, which the string-asserting NumSimMaterialTargetTest
// cannot.
//
// Two recipes on purpose:
//   * LinearHardening   dα/dt = K·α      — constant derivative (K); the plumbing
//                                          floor, α(1) = e^-1 for K = -1.
//   * NonlinearDecay    dα/dt = K·α²     — NON-constant derivative (2·K·α), so
//                                          the value-check exercises real
//                                          cas::diff, not a trivial constant.
//                                          α(1) = 0.5 for K = -1, α(0) = 1.
#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_functions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace numsim::cas;
using namespace numsim::codegen;

namespace {

// Emit the single header for `model` to `path`. Returns false on failure.
bool write_header(ConstitutiveModel const& model, char const* path) {
  for (auto const& f : NumSimMaterialTarget{}.emit(model)) {
    if (f.kind == EmittedFile::Kind::Header) {
      std::ofstream os(path);
      if (!os) {
        std::cerr << "could not open " << path << " for writing\n";
        return false;
      }
      os << f.contents;
      return true;
    }
  }
  std::cerr << "NumSimMaterialTarget emitted no header for " << model.name()
            << "\n";
  return false;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "usage: generate_numsim_material_check <linear.h> "
                 "<nonlinear.h> <viscoelastic.h> <returnmap.h> "
                 "<returnmapcubic.h> <j2pathdep.h>\n";
    return 2;
  }

  // dα/dt = K·α — rate = K·α, rate_derivative = K. Plus a scalar OUTPUT
  // `twice_state` = 2·α (state-only) to exercise B.1 output emission end-to-end.
  ConstitutiveModel linear("LinearHardening");
  {
    auto K = linear.add_parameter("K", -1.0);
    auto alpha = linear.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    linear.add_scalar_evolution_equation(alpha, K * alpha.current);
    linear.add_output("twice_state", alpha.current + alpha.current);
  }

  // dα/dt = K·α² — rate = K·α², rate_derivative = 2·K·α (cas::diff power rule).
  ConstitutiveModel nonlinear("NonlinearDecay");
  {
    auto K = nonlinear.add_parameter("K", -1.0);
    auto alpha = nonlinear.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    nonlinear.add_scalar_evolution_equation(
        alpha, K * alpha.current * alpha.current);
  }

  // Tensor stress from a SCALAR integrated state + a TENSOR strain input:
  // dα/dt = K·α (scalar, for the integrator) and σ = α·ε (tensor stress).
  // Exercises tensor input wiring + tensor output emission — no tensor state,
  // no vector solver needed.
  ConstitutiveModel viscoelastic("Viscoelastic");
  {
    auto K = viscoelastic.add_parameter("K", -1.0);
    auto alpha = viscoelastic.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    viscoelastic.add_scalar_evolution_equation(alpha, K * alpha.current);
    auto eps = viscoelastic.add_tensor_input("strain", 3, 2, roles::Strain);
    viscoelastic.add_output("stress", alpha.current * eps);
    // Phase D: consistent tangent dσ/dε. Since the rate (K·α) is strain-
    // independent, dα/dε=0, so dσ/dε = ∂σ/∂ε = α·P_sym (minor-symmetric).
    viscoelastic.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  }

  // Phase 2a: strain-coupled IMPLICIT residual material (Mode B). The state z is
  // defined by R(z, ε) = z − c·tr(ε) = 0, solved by backward_euler (caller-
  // driven); the stress is σ = z·ε. Unlike the rate materials above, z is solved
  // implicitly inside the material's compute() — no rk_integrator. Exercises the
  // material_ref<backward_euler> + solve(eval) emission end-to-end.
  ConstitutiveModel returnmap("ReturnMap");
  {
    auto c = returnmap.add_parameter("c", 2.0);
    auto eps = returnmap.add_tensor_input("strain", 3, 2, roles::Strain);
    auto z = returnmap.add_scalar_state_variable(
        "z", make_expression<scalar_constant>(0.0));
    returnmap.add_scalar_residual_equation(z, z.current - c * trace(eps));
    returnmap.add_output("stress", z.current * eps);
    // Phase 2b: the strain-coupled consistent tangent dσ/dε. The driver checks
    // the off-block coupling term the naive ∂σ/∂ε alone misses (C_{0011}=c·ε₀₀).
    returnmap.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  }

  // NONLINEAR residual to exercise the t2s-wrt-scalar jacobian (∂R/∂z, cas#285).
  // R(z, ε) = z + z³ − c·tr(ε), so ∂R/∂z = 1 + 3z² is NON-constant. The linear
  // ReturnMap above has ∂R/∂z ≡ 1, so a wrong jacobian would still converge to
  // the right root — it cannot validate the emitted derivative. Here, with a
  // tight Newton budget and c·tr(ε)=1 (root z≈0.682, where ∂R/∂z≈2.4 ≫ 1), an
  // incorrect jacobian (e.g. a constant) oscillates and FAILS to converge — so
  // the e2e value check actually pins the emitted derivative.
  ConstitutiveModel returnmap_cubic("ReturnMapCubic");
  {
    auto c = returnmap_cubic.add_parameter("c", 2.0);
    auto eps = returnmap_cubic.add_tensor_input("strain", 3, 2, roles::Strain);
    auto z = returnmap_cubic.add_scalar_state_variable(
        "z", make_expression<scalar_constant>(0.0));
    returnmap_cubic.add_scalar_residual_equation(
        z, z.current + z.current * z.current * z.current - c * trace(eps));
    returnmap_cubic.add_output("stress", z.current * eps);
    // Phase 2b: a tangent on the NONLINEAR residual — here ∂R/∂z = 1+3z² ≠ 1, so
    // dz/dε = −∂R/∂ε/∂R/∂z genuinely exercises the division by the jacobian
    // (the linear ReturnMap has ∂R/∂z≡1, where dropping the divisor is invisible).
    returnmap_cubic.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  }

  // #92 golden: PATH-DEPENDENT J2 — plastic strain εᵖ carried as tensor history
  // and subtracted in the trial stress, so the response depends on the loading
  // PATH (accumulates under load, retained on unload) — not just the current
  // strain. α (accumulated plastic strain) is the scalar Newton unknown; εᵖ is a
  // tensor internal variable updated post-solve. Requires history_property<tensor>
  // (numsim-core#16 fix) — hence the bumped _ncg_core_pin.
  ConstitutiveModel j2pd("J2PathDep");
  {
    auto G = j2pd.add_parameter("G", 80.0);
    auto sy = j2pd.add_parameter("sy", 1.0);
    auto H = j2pd.add_parameter("H", 10.0);
    auto eps = j2pd.add_tensor_input("strain", 3, 2, roles::Strain);
    auto alpha = j2pd.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    auto epsp = j2pd.add_tensor_state_variable(
        "eps_p", 3, 2, make_expression<tensor_zero>(3, std::size_t{2}));
    auto dg = alpha.current - alpha.previous;             // per-step Δγ
    auto s_trial = (2.0 * G) * dev(eps - epsp.previous);  // subtract accumulated εᵖ
    auto q = norm(s_trial);
    auto n = s_trial / q;
    j2pd.add_scalar_residual_equation(
        alpha, q - (2.0 * G) * dg - sy - H * alpha.current);
    j2pd.add_tensor_update_equation(epsp, epsp.previous + dg * n); // εᵖ += Δγ·n
    j2pd.add_output("stress", s_trial - (2.0 * G) * dg * n);
    j2pd.add_algorithmic_tangent("dstress_dstrain", "stress", "strain");
  }

  if (!write_header(linear, argv[1])) return 1;
  if (!write_header(nonlinear, argv[2])) return 1;
  if (!write_header(viscoelastic, argv[3])) return 1;
  if (!write_header(returnmap, argv[4])) return 1;
  if (!write_header(returnmap_cubic, argv[5])) return 1;
  if (!write_header(j2pd, argv[6])) return 1;
  return 0;
}
