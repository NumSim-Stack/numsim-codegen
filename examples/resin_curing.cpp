// Phase 3a-2 example: thermoset resin curing with an in-function Newton
// solve. Demonstrates the state-variable + evolution-equation + local-Newton
// path end-to-end (issue #75).
//
// Physics — first-order cure kinetics:
//   The degree of cure c ∈ [0, 1] (0 = uncured, 1 = fully cured) evolves as
//       dc/dt = K · (1 − c)
//   where K is a temperature-dependent rate constant. Backward-Euler
//   discretisation gives the per-step residual
//       R(c) = (c − c_old)/dt − K·(1 − c) = 0
//   which numsim-codegen lowers to an in-function Newton iteration:
//   `enable_local_newton()` makes the generated function SOLVE for the
//   converged c internally and expose it as `c_out`, rather than emitting
//   the residual/Jacobian for an external driver.
//
//   The Jacobian J = ∂R/∂c = 1/dt + K is produced symbolically by cas::diff
//   — no finite differences, no hand-derivation. (The same machinery
//   handles nonlinear autocatalytic kinetics like dc/dt = K·c·(1−c); the
//   only change would be the rate expression below.)
//
//   The developing stiffness E = E0 + (E_inf − E0)·c is a regular output;
//   it references the CONVERGED cure, so the generated code evaluates it
//   after the Newton loop.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>

#include <iostream>

int main() {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("ResinCuring");

  // Parameters (framework-supplied at runtime).
  auto K = model.add_parameter("K", /*default=*/1.0, "Cure rate constant");
  auto E0 = model.add_parameter("E0", 1.0e6, "Uncured modulus [Pa]");
  auto E_inf = model.add_parameter("E_inf", 3.0e9, "Fully-cured modulus [Pa]");

  // State variable: degree of cure, initial value 0 (uncured).
  auto c = model.add_scalar_state_variable(
      "c", make_expression<scalar_constant>(0.0), "Degree of cure [0,1]");

  // First-order cure kinetics: dc/dt = K·(1 − c).
  auto one = make_expression<scalar_constant>(1.0);
  model.add_scalar_evolution_equation(c, K * (one - c.current),
                                      "First-order cure kinetics");

  // Developing stiffness tracks the converged cure.
  model.add_output("E", E0 + (E_inf - E0) * c.current);

  // Solve for the converged cure in-function via Newton.
  model.enable_local_newton();

  std::cout << model.emit_compute_function();
  return 0;
}
