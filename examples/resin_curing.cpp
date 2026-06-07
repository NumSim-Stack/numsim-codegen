// Phase 3a-2 example: thermoset resin curing with an AUTOCATALYTIC cure
// reaction, solved by an in-function Newton iteration (issue #75).
//
// Physics — autocatalytic cure kinetics (Kamal–Sourour model, the
// n-th-order × autocatalysis family; see
// https://kinetics.netzsch.com/en/learn/an-introduction-to-n-th-order-and-autocatalysis-reactions):
//
//     dα/dt = (K1 + K2·α^m) · (1 − α)^n
//
//   α ∈ [0, 1] is the degree of cure. The `K1` term is the n-th-order
//   (non-catalytic) channel that starts the reaction from the uncured
//   state α = 0; the `K2·α^m` term is the AUTOCATALYSIS — the rate
//   accelerates as reaction product (cured material) accumulates, giving
//   the characteristic sigmoidal cure curve. `n` is the reaction order.
//
//   We take the autocatalytic exponent m = 1 so the rate's derivative
//   stays regular at α = 0 (α^(m−1) does not blow up), which lets the
//   in-function Newton converge straight from the uncured state without
//   seeding. General fractional m is fully supported — it is just another
//   `cas::pow` and `cas::diff` differentiates it — but then the caller
//   must seed α_old > 0. The reaction order n is fractional here (1.5), a
//   representative epoxy value, exercising `cas::pow` in the rate.
//
//   Backward-Euler discretisation gives the per-step residual
//       R(α) = (α − α_old)/dt − (K1 + K2·α)·(1 − α)^n = 0
//   which is NONLINEAR in α, so the generated Newton loop genuinely
//   iterates. The Jacobian J = ∂R/∂α is produced symbolically by
//   `cas::diff` — including the d/dα of `(1 − α)^n` — with no finite
//   differences and no hand-derivation.
//
//   `enable_local_newton()` makes the generated function SOLVE for the
//   converged α internally and expose it as `c_out`. The developing
//   stiffness E = E0 + (E_inf − E0)·α is a regular output that references
//   the CONVERGED cure, so it is evaluated after the loop.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/scalar/scalar_std.h>

#include <iostream>

int main() {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel model("ResinCuring");

  // Parameters (framework-supplied at runtime).
  auto K1 = model.add_parameter("K1", /*default=*/0.2,
                                "Non-catalytic rate constant");
  auto K2 = model.add_parameter("K2", 2.0, "Autocatalytic rate constant");
  auto E0 = model.add_parameter("E0", 1.0e6, "Uncured modulus [Pa]");
  auto E_inf = model.add_parameter("E_inf", 3.0e9, "Fully-cured modulus [Pa]");

  // State variable: degree of cure, initial value 0 (uncured).
  auto c = model.add_scalar_state_variable(
      "c", make_expression<scalar_constant>(0.0), "Degree of cure [0,1]");

  // Autocatalytic cure rate: dα/dt = (K1 + K2·α) · (1 − α)^n,  n = 1.5.
  auto one = make_expression<scalar_constant>(1.0);
  auto n_order = make_expression<scalar_constant>(1.5);
  auto rate = (K1 + K2 * c.current) * pow(one - c.current, n_order);
  model.add_scalar_evolution_equation(c, rate, "Kamal autocatalytic kinetics");

  // Developing stiffness tracks the converged cure.
  model.add_output("E", E0 + (E_inf - E0) * c.current);

  // Solve for the converged cure in-function via Newton.
  model.enable_local_newton();

  std::cout << model.emit_compute_function();
  return 0;
}
