// Generator program for the compile-check test. CMake runs this at build
// time with TWO destination header paths as argv[1] and argv[2]. The
// outputs are self-contained headers that the compile-check driver then
// #include's.
//
// Two recipes are emitted:
//   1. **CompileCheck** — scalar/tensor inputs + parameters + outputs.
//      Exercises the full elastic-recipe pipeline + adaptor zero-copy.
//   2. **HardeningCheck** (Phase 3a-1) — scalar state variable +
//      linear-hardening evolution equation. The generated function gains
//      paired `_residual_out` + `_jacobian_out` parameters, and the
//      driver verifies their numerical values match `(α−α_old)/dt − K·α`
//      and `1/dt − K` respectively. End-to-end check that catches
//      `cas::diff` regressions which substring-matching in
//      `tests/LocalJacobianTest.cpp` cannot.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <fstream>
#include <iostream>

namespace {

auto write_single_file(numsim::codegen::ConstitutiveModel const &model,
                       char const *out_path) -> int {
  numsim::codegen::StandaloneCxxTarget target;
  auto files = target.emit(model);
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
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <CompileCheck.h> <HardeningCheck.h>\n";
    return 1;
  }

  using namespace numsim::cas;
  using namespace numsim::codegen;

  // ── Recipe 1: elastic compile-check (existing) ──────────────────────
  {
    ConstitutiveModel model("CompileCheck");
    auto a = model.add_parameter("a", 2.0);
    auto x = model.add_scalar_input("x");
    auto mu = model.add_parameter("mu", 0.5);
    auto eps = model.add_tensor_input("eps", 3, 2);
    model.add_output("y", a * x + sin(x));
    model.add_output("sigma", 2 * mu * eps);
    if (int rc = write_single_file(model, argv[1]); rc != 0) return rc;
  }

  // ── Recipe 2: hardening compile-check (Phase 3a-1) ──────────────────
  //
  // Linear hardening: rate = K·α. Backward-Euler residual then
  // discretises to `R = (α − α_old)/dt − K·α`. cas::diff produces the
  // Jacobian `J = ∂R/∂α = 1/dt − K`. The driver evaluates both at
  // concrete inputs to catch any cas::diff regression.
  {
    ConstitutiveModel model("HardeningCheck");
    auto K = model.add_parameter("K", 1.0);
    auto alpha = model.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    model.add_scalar_evolution_equation(alpha, K * alpha.current);
    if (int rc = write_single_file(model, argv[2]); rc != 0) return rc;
  }

  return 0;
}
