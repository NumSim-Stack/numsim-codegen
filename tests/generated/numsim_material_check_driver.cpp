// Phase B end-to-end gate — driver half.
//
// Compiles the NumSimMaterialTarget-GENERATED materials (LinearHardening.h and
// NonlinearDecay.h, produced at build time by generate_numsim_material_check)
// against the REAL numsim-materials + numsim-core headers, wires each to the
// REAL rk_integrator in a property graph, and checks:
//   * convergence to the closed-form solution (e^-1 for dα/dt=K·α with K=-1;
//     0.5 for dα/dt=K·α² with α(0)=1, K=-1), and
//   * the emitted rate / rate_derivative VALUES at a known state — the check
//     that catches a wrong-but-convergent emitted derivative.
//
// This is the proof the string-asserting NumSimMaterialTargetTest cannot give:
// that the emitted code actually compiles and runs against the solver contract.
#include <cmath>
#include <string>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/solvers/butcher_tableau.h"
#include "numsim-materials/solvers/rk_integrator.h"

#include "LinearHardening.h" // generated: dα/dt = K·α
#include "NonlinearDecay.h"  // generated: dα/dt = K·α²

#include <gtest/gtest.h>

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using RK = numsim::materials::rk_integrator<policy>;

using Linear = numsim::materials::generated::LinearHardening<policy>;
using Nonlinear = numsim::materials::generated::NonlinearDecay<policy>;

constexpr T kK = T{-1.0};

// Build a graph of rk_integrator + the generated material `Mat` (registered
// under `name`, matching the integrator's "function"), with parameter K = kK.
// NOTE: the integrator stores the ADDRESS of `tab` (its "tableau" param), so
// `tab` must outlive every subsequent ctx.update(). integrate() satisfies this
// by passing a call-argument temporary (alive for the whole integrate(...)
// expression); emitted_values() uses a named local because build() and the use
// are separate statements.
template <typename Mat>
void build(ctx_type& ctx, const std::string& name,
           const numsim::materials::butcher_tableau& tab, T step_size) {
  param_type p;
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", name);
  p.insert<T>("step_size", step_size);
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &tab);
  ctx.create<RK>(p);

  p.clear();
  p.insert<std::string>("name", name);
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<T>("K", kK);
  ctx.create<Mat>(p);

  ctx.finalize();
}

// Integrate over t ∈ [0,1] in N steps from α(0)=1 through the real rk_integrator
// wired to the generated material `Mat`. Returns α(1).
template <typename Mat>
T integrate(const std::string& name, int N,
            const numsim::materials::butcher_tableau& tab) {
  ctx_type ctx;
  build<Mat>(ctx, name, tab, T{1.0} / T(N));

  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(
      ctx.find_property("integrator", "state"));
  EXPECT_NE(hist, nullptr) << "integrator must publish a `state` history";
  hist->old_value() = T{1};
  hist->new_value() = T{1};

  for (int i = 0; i < N; ++i) {
    ctx.update();
    ctx.commit();
  }
  return ctx.get<T>("integrator", "state");
}

// Fire the generated material's compute() at a known α and return its emitted
// (rate, rate_derivative). White-box: we trigger compute() via the "rate"
// property's update callback — the same callback the integrator invokes through
// update_source() — so this reads exactly what the solver would read.
template <typename Mat>
std::pair<T, T> emitted_values(const std::string& name, T alpha) {
  ctx_type ctx;
  const auto tab = numsim::materials::forward_euler(); // must outlive build()
  build<Mat>(ctx, name, tab, T{0.1});

  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(
      ctx.find_property("integrator", "state"));
  EXPECT_NE(hist, nullptr);
  hist->new_value() = alpha; // input_property::get() reads the trial new_value

  auto* rate_prop = ctx.find_property(name, "rate");
  EXPECT_NE(rate_prop, nullptr) << "generated material must publish a `rate`";
  if (!rate_prop) return {std::nan(""), std::nan("")};
  rate_prop->traits().update(); // fire compute()
  return {ctx.get<T>(name, "rate"), ctx.get<T>(name, "rate_derivative")};
}

// ── Linear: dα/dt = K·α (constant derivative) ────────────────────────────────

const T kExactLinear = std::exp(-1.0); // α(1) for dα/dt = -α, α(0) = 1

TEST(NumSimMaterialEndToEnd, LinearForwardEulerConverges) {
  EXPECT_NEAR(integrate<Linear>("LinearHardening", 1000,
                                numsim::materials::forward_euler()),
              kExactLinear, 1e-3);
}

// High-order explicit near machine precision confirms the rate is evaluated at
// the integrator's *trial* state (input_property::get() returns new_value).
TEST(NumSimMaterialEndToEnd, LinearRK4HitsHighAccuracy) {
  EXPECT_NEAR(
      integrate<Linear>("LinearHardening", 100, numsim::materials::rk4()),
      kExactLinear, 1e-8);
}

// Implicit path wires+reads the generated `rate_derivative` edge (a MISSING one
// fails here). Convergence alone does NOT verify its value — see the value test.
TEST(NumSimMaterialEndToEnd, LinearImplicitEulerWiresRateDerivative) {
  EXPECT_NEAR(integrate<Linear>("LinearHardening", 1000,
                                numsim::materials::implicit_euler()),
              kExactLinear, 1e-3);
}

TEST(NumSimMaterialEndToEnd, LinearEmittedValuesAreCorrect) {
  auto [rate, drate] = emitted_values<Linear>("LinearHardening", T{2});
  EXPECT_NEAR(rate, kK * T{2}, 1e-12);  // K·α
  EXPECT_NEAR(drate, kK, 1e-12);        // K
}

// ── Nonlinear: dα/dt = K·α² (non-constant derivative 2·K·α) ───────────────────

TEST(NumSimMaterialEndToEnd, NonlinearRK4ConvergesToClosedForm) {
  // dα/dt = -α², α(0)=1 ⇒ α(t) = 1/(1+t), α(1) = 0.5.
  EXPECT_NEAR(
      integrate<Nonlinear>("NonlinearDecay", 200, numsim::materials::rk4()),
      T{0.5}, 1e-6);
}

// Verifies cas::diff produced a NON-constant derivative: at α=3, rate = K·9,
// rate_derivative = 2·K·3 — distinct values, so a constant/swapped derivative
// is caught.
TEST(NumSimMaterialEndToEnd, NonlinearEmittedValuesAreCorrect) {
  auto [rate, drate] = emitted_values<Nonlinear>("NonlinearDecay", T{3});
  EXPECT_NEAR(rate, kK * T{9}, 1e-12);       // K·α²
  EXPECT_NEAR(drate, T{2} * kK * T{3}, 1e-12); // 2·K·α
}

} // namespace
