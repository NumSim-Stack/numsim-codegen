// Generator program for the MOOSE-stub compile gate (#132). CMake runs this
// at build time with one destination DIRECTORY (argv[1]); the program emits
// each recipe below through MooseMaterialTarget and writes the resulting
// <Name>.h / <Name>.C pair there. The gate driver then compiles every .C
// against the minimal MOOSE API stubs in tests/moose_stub/ — proving the
// emitted MOOSE boilerplate is valid C++ under -Wall -Wextra (-Werror in
// Debug), which the string-asserting MooseTargetTest cannot.
//
// The recipes (kept in sync with tests/CMakeLists.txt's _ncg_moose_units and
// moose_check_driver.cpp):
//   1. MooseShearCheck    — linear elastic shear: params/inputs/outputs +
//                           RankTwoTensor adaptor boundary (value-checked in
//                           the driver, issue #12 stretch)
//   2. MooseTangentCheck  — nonlinear stress + consistent tangent →
//                           RankFourTensor / _Jacobian_mult wiring
//   3. MooseSpectralCheck — stress = log(C) → the spectral-runtime include
//                           path (numsim_codegen/runtime/spectral.h)
//   4. MooseNewtonCheck   — scalar state variable + in-function local Newton
//                           → stateful properties, _dt, initQpStatefulProperties
//                           (value-checked in the driver)
//   5. MooseCoupledCheck  — coupled 2x2 Newton → Eigen dense solve wrapped in
//                           libmesh/ignore_warnings.h

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/operators/tensor_to_scalar/tensor_to_scalar_with_tensor_mul.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_isotropic_functions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_dot.h>

#include <fstream>
#include <iostream>
#include <string>

namespace {

auto write_material(numsim::codegen::ConstitutiveModel const &model,
                    std::string const &out_dir) -> int {
  numsim::codegen::MooseMaterialTarget target;
  auto const files = target.emit(model);
  if (files.size() != 2) {
    std::cerr << "expected .h/.C pair for '" << model.name() << "', got "
              << files.size() << " files\n";
    return 1;
  }
  for (auto const &f : files) {
    auto const path = out_dir + "/" + f.filename;
    std::ofstream out(path);
    if (!out) {
      std::cerr << "could not open '" << path << "' for writing\n";
      return 1;
    }
    out << f.contents;
  }
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <output-directory>\n";
    return 1;
  }
  std::string const out_dir = argv[1];

  using namespace numsim::cas;
  using namespace numsim::codegen;

  // ── Recipe 1: linear elastic shear ───────────────────────────────────
  //
  // The canonical minimal MOOSE material: one Real parameter, one strain
  // input, one stress output. mu's default is deliberately not 0.5 so the
  // driver's closed-form check (stress = 2·mu·eps) cannot pass by accident.
  {
    ConstitutiveModel model("MooseShearCheck");
    auto mu = model.add_parameter("mu", 0.7, "Shear modulus");
    auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
    model.add_output("stress", 2 * mu * eps, roles::Stress);
    if (int rc = write_material(model, out_dir); rc != 0)
      return rc;
  }

  // ── Recipe 2: consistent tangent (Phase 5 _Jacobian_mult wiring) ─────
  //
  // Nonlinear stress sigma = 2·mu·eps + c·(eps:eps)·eps, so the tangent is a
  // genuine non-constant rank-4 expression. add_algorithmic_tangent routes it
  // to MOOSE's _Jacobian_mult slot — this exercises the RankFourTensor
  // member/declareProperty/adaptor emission.
  {
    ConstitutiveModel model("MooseTangentCheck");
    auto mu = model.add_parameter("mu", 0.7, "Shear modulus");
    auto c = model.add_parameter("c", 1.5, "Cubic coefficient");
    auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
    auto dot = make_expression<tensor_dot>(eps); // eps:eps (t2s)
    auto sigma = 2 * mu * eps +
                 make_expression<tensor_to_scalar_with_tensor_mul>(c * eps, dot);
    model.add_output("stress", sigma, roles::Stress);
    model.add_algorithmic_tangent("dstress_deps", "stress", "eps");
    if (int rc = write_material(model, out_dir); rc != 0)
      return rc;
  }

  // ── Recipe 3: spectral stress (runtime-include path) ─────────────────
  //
  // stress = log(C) lowers through the spectral decomposition, so the emitted
  // .C must pull in <numsim_codegen/runtime/spectral.h> — the include-gating
  // logic this recipe locks down at the compile level.
  {
    ConstitutiveModel model("MooseSpectralCheck");
    auto C = model.add_tensor_input("C", 3, 2, roles::Strain);
    model.add_output("stress", log(C), roles::Stress);
    if (int rc = write_material(model, out_dir); rc != 0)
      return rc;
  }

  // ── Recipe 4: scalar state variable + local Newton ───────────────────
  //
  // Linear hardening rate = K·alpha with enable_local_newton(): the MOOSE
  // target requires evolution equations to be solved in-function (its guard
  // rejects the residual/Jacobian-output mode). Exercises declareProperty /
  // getMaterialPropertyOld, initQpStatefulProperties and the framework _dt.
  // Linear residual → the analytic fixed point alpha* = alpha_old/(1 − K·dt)
  // that the driver value-checks.
  {
    ConstitutiveModel model("MooseNewtonCheck");
    auto K = model.add_parameter("K", 0.5, "Hardening rate coefficient");
    auto alpha = model.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    model.add_output("sigma_y", K * alpha.current);
    model.add_scalar_evolution_equation(alpha, K * alpha.current);
    model.enable_local_newton();
    if (int rc = write_material(model, out_dir); rc != 0)
      return rc;
  }

  // ── Recipe 5: coupled 2x2 Newton (Eigen dense solve) ─────────────────
  //
  // Mutually-referencing evolution equations force the coupled lowering: the
  // emitted .C includes <Eigen/Dense> wrapped in libmesh/ignore_warnings.h /
  // restore_warnings.h — the one MOOSE-specific include idiom the other
  // recipes never reach.
  {
    ConstitutiveModel model("MooseCoupledCheck");
    auto K1 = model.add_parameter("K1", 1.0, "Coupling a <- b");
    auto K2 = model.add_parameter("K2", 2.0, "Coupling b <- a");
    auto a = model.add_scalar_state_variable(
        "a", make_expression<scalar_constant>(0.0));
    auto b = model.add_scalar_state_variable(
        "b", make_expression<scalar_constant>(0.0));
    model.add_output("hardening", K1 * a.current + K2 * b.current);
    model.add_scalar_evolution_equation(a, K1 * b.current);
    model.add_scalar_evolution_equation(b, K2 * a.current);
    model.enable_local_newton();
    if (int rc = write_material(model, out_dir); rc != 0)
      return rc;
  }

  return 0;
}
