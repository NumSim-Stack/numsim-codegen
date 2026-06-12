// Generator program for the compile-check test. CMake runs this at build
// time with one destination header path per emitted recipe (argv[1..N]). The
// outputs are self-contained headers that the compile-check driver then
// #include's.
//
// The recipes (kept in sync with compile_check_driver.cpp + tests/CMakeLists):
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
#include <numsim_cas/tensor/operators/tensor_to_scalar/tensor_to_scalar_with_tensor_mul.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor/tensor_std.h>
#include <numsim_cas/tensor_to_scalar/tensor_dot.h>
#include <numsim_cas/tensor_to_scalar/tensor_norm.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_std.h>
#include <numsim_cas/tensor_to_scalar/tensor_trace.h>

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
  if (argc < 9) {
    std::cerr << "usage: " << argv[0]
              << " <CompileCheck.h> <HardeningCheck.h> <NewtonCheck.h>"
                 " <AutocatalyticCheck.h> <CoupledCheck.h> <PiecewiseCheck.h>"
                 " <PiecewiseT2sCheck.h> <TangentCheck.h>\n";
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

  // ── Recipe 3: in-function Newton solve (Phase 3a-2, issue #75) ───────
  //
  // Same linear-hardening physics, but `enable_local_newton()` makes the
  // generated function SOLVE for the converged α internally and expose it
  // as `alpha_out`, instead of emitting R/J outputs. The residual is
  // linear in α, so Newton hits the root in one step; the driver verifies
  // convergence to the analytic fixed point α* = α_old / (1 − K·dt) and
  // that the downstream output (sigma_y = K·α) sees the converged value.
  // This is the Phase 1.3 (#32) de-risk spike realised as a CI test.
  {
    ConstitutiveModel model("NewtonCheck");
    auto K = model.add_parameter("K", 1.0);
    auto alpha = model.add_scalar_state_variable(
        "alpha", make_expression<scalar_constant>(0.0));
    model.add_output("sigma_y", K * alpha.current);
    model.add_scalar_evolution_equation(alpha, K * alpha.current);
    model.enable_local_newton();
    if (int rc = write_single_file(model, argv[3]); rc != 0) return rc;
  }

  // ── Recipe 4: AUTOCATALYTIC Newton solve (Phase 3a-2, issue #75) ─────
  //
  // Kamal cure kinetics dα/dt = (K1 + K2·α)·(1−α)^1.5 — NONLINEAR in α, so
  // the generated Newton loop genuinely iterates (the linear NewtonCheck
  // above converges in one step and would not catch a broken iteration).
  // The driver steps backward-Euler through a cure history and verifies
  // the residual is driven to ~0 at every converged point.
  {
    ConstitutiveModel model("AutocatalyticCheck");
    auto K1 = model.add_parameter("K1", 0.2);
    auto K2 = model.add_parameter("K2", 2.0);
    auto c = model.add_scalar_state_variable(
        "c", make_expression<scalar_constant>(0.0));
    auto one = make_expression<scalar_constant>(1.0);
    auto n = make_expression<scalar_constant>(1.5);
    model.add_scalar_evolution_equation(
        c, (K1 + K2 * c.current) * pow(one - c.current, n));
    model.enable_local_newton();
    if (int rc = write_single_file(model, argv[4]); rc != 0) return rc;
  }

  // ── Recipe 5: COUPLED 2×2 Newton solve (Phase 3b-2b, issue #35) ──────
  //
  // Two mutually-referencing evolution equations a' = K1·b, b' = K2·a with
  // DISTINCT coefficients (asymmetric, so a transposed Jacobian would give a
  // different — wrong — answer). LocalNewtonLoweringPass detects the coupling
  // and emits ONE dense Newton loop solved with Eigen. Backward-Euler is
  // linear here, so the analytic fixed point is the solution of
  //   [1, −dt·K1; −dt·K2, 1] [a; b] = [a_old; b_old]
  // The driver compares the generated solve against that closed form — the
  // numerical lock the string-matching structural tests cannot provide
  // (PR #83 review F3). Requires Eigen on the compile path.
  {
    ConstitutiveModel model("CoupledCheck");
    auto K1 = model.add_parameter("K1", 1.0);
    auto K2 = model.add_parameter("K2", 2.0);
    auto a = model.add_scalar_state_variable(
        "a", make_expression<scalar_constant>(0.0));
    auto b = model.add_scalar_state_variable(
        "b", make_expression<scalar_constant>(0.0));
    model.add_scalar_evolution_equation(a, K1 * b.current);
    model.add_scalar_evolution_equation(b, K2 * a.current);
    model.enable_local_newton();
    if (int rc = write_single_file(model, argv[5]); rc != 0) return rc;
  }

  // ── Recipe 6: PIECEWISE tensor output (CAS f3e799e if_then_else) ─────
  //
  // `if_then_else(x, 2*eps, -eps)` builds a `tensor_if_then_else` node —
  // a SCALAR condition selecting between two TENSOR branches. The CAS pin
  // bump (#275/#285, move_to_virtual) made codegen's tensor/t2s visitors
  // grow these as pure-virtuals; this exercises the materialized-ternary
  // emission `(cond != 0.0 ? tmech::tensor<…>(then) : tmech::tensor<…>(else))`
  // end-to-end. The driver compiles it against real tmech and verifies both
  // branches numerically — the lock that proves the emitted construct is a
  // valid tmech expression, which a string test cannot.
  {
    ConstitutiveModel model("PiecewiseCheck");
    auto x = model.add_scalar_input("x");
    auto eps = model.add_tensor_input("eps", 3, 2);
    model.add_output("sigma", if_then_else(x, 2.0 * eps, -eps));
    if (int rc = write_single_file(model, argv[6]); rc != 0) return rc;
  }

  // ── Recipe 7: PIECEWISE tensor-to-scalar (review #87 t2s coverage) ──
  //
  // The SIBLING node `tensor_to_scalar_if_then_else` — cond/then/else ALL
  // t2s — was implemented in the same catch-up but had no end-to-end test
  // (a t2s output is inexpressible: `add_output` takes only scalar/tensor).
  // A t2s if_then_else is only reachable as a SUBTERM, so we lift it into a
  // tensor output via `tensor_to_scalar_with_tensor_mul`:
  //   sigma = (trace(eps) != 0 ? trace(eps) : norm(eps)) * eps
  // The condition is itself t2s (`trace(eps)`), exercising the t2s emitter's
  // condition path (NOT the scalar one) — the one subtlety this node has.
  // The driver picks a trace≠0 strain and a traceless one to hit both
  // branches numerically.
  {
    ConstitutiveModel model("PiecewiseT2sCheck");
    auto eps = model.add_tensor_input("eps", 3, 2);
    auto tr = make_expression<tensor_trace>(eps);  // t2s
    auto nrm = make_expression<tensor_norm>(eps);   // t2s
    auto pick = if_then_else(tr, tr, nrm);          // t2s if_then_else
    auto sigma = make_expression<tensor_to_scalar_with_tensor_mul>(eps, pick);
    model.add_output("sigma", sigma);
    if (int rc = write_single_file(model, argv[7]); rc != 0) return rc;
  }

  // ── Recipe 8: consistent-tangent FD check (verification spine #90) ───
  //
  // Nonlinear hyperelastic stress σ = 2μ·ε + c·(ε:ε)·ε, so the consistent
  // tangent dσ/dε is NON-constant (a linear σ would make FD trivially exact
  // and test nothing). `add_algorithmic_tangent` emits dσ/dε via cas::diff.
  // SCOPE: no state variable / no local Newton, so only the EXPLICIT term
  // ∂σ/∂ε is exercised — the strain-coupled implicit correction (gated on
  // numsim-cas#275) is absent and not verified here. See the driver test.
  {
    ConstitutiveModel model("TangentCheck");
    auto mu = model.add_parameter("mu", 0.7);
    auto c = model.add_parameter("c", 1.5);
    // roles::Strain marks eps SYMMETRIC, so cas::diff returns the minor-
    // symmetric rank-4 identity and the tangent is minor-symmetric (C_ijkl =
    // C_ijlk) — as a stress-strain tangent must be, and as the symmetric FD
    // reference (num_diff_sym_central) expects. (A plain input would give the
    // non-symmetric δ_ik δ_jl identity — the FD harness catches that.)
    auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
    auto dot = make_expression<tensor_dot>(eps); // ε:ε (t2s)
    auto sigma = 2 * mu * eps +
                 make_expression<tensor_to_scalar_with_tensor_mul>(c * eps, dot);
    model.add_output("stress", sigma, roles::Stress);
    model.add_algorithmic_tangent("dstress_deps", "stress", "eps");
    if (int rc = write_single_file(model, argv[8]); rc != 0) return rc;
  }

  return 0;
}
