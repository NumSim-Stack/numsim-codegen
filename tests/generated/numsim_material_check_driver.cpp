// Phase B end-to-end gate — driver half.
//
// Compiles the NumSimMaterialTarget-GENERATED material (LinearHardening.h,
// produced at build time by generate_numsim_material_check) against the REAL
// numsim-materials + numsim-core headers, wires it to the REAL rk_integrator
// in a property graph, and integrates dα/dt = K·α with K = -1, asserting
// convergence to y(1) = e^-1 across explicit, RK4, and implicit tableaux.
//
// This is the proof the string-asserting NumSimMaterialTargetTest cannot give:
// that the emitted code actually compiles and runs against the solver contract.
// implicit_euler additionally exercises the `rate_derivative` Local edge.
#include <cmath>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/solvers/butcher_tableau.h"
#include "numsim-materials/solvers/rk_integrator.h"

#include "LinearHardening.h" // generated

#include <gtest/gtest.h>

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using RK = numsim::materials::rk_integrator<policy>;
using Gen = numsim::materials::generated::LinearHardening<policy>;

// Integrate dα/dt = K·α (K = -1), α(0) = 1, over t ∈ [0,1] in N steps, through
// the real rk_integrator wired to the generated material. Returns α(1).
T run(int N, const numsim::materials::butcher_tableau& tab) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "LinearHardening");
  p.insert<T>("step_size", T{1.0} / T(N));
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &tab);
  ctx.create<RK>(p);

  p.clear();
  p.insert<std::string>("name", "LinearHardening");
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<T>("K", T{-1.0});
  ctx.create<Gen>(p);

  ctx.finalize();

  auto* prop = ctx.find_property("integrator", "state");
  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(prop);
  EXPECT_NE(hist, nullptr) << "integrator must publish a `state` history property";
  hist->old_value() = T{1};
  hist->new_value() = T{1};

  for (int i = 0; i < N; ++i) {
    ctx.update();
    ctx.commit();
  }
  return ctx.get<T>("integrator", "state");
}

const T kExact = std::exp(-1.0); // α(1) for dα/dt = -α, α(0) = 1

// Explicit path: the generated `rate` Local edge only.
TEST(NumSimMaterialEndToEnd, ForwardEulerConvergesToExact) {
  EXPECT_NEAR(run(1000, numsim::materials::forward_euler()), kExact, 1e-3);
}

// High-order explicit: near machine precision confirms the rate is evaluated at
// the trial state (i.e. input_property::get() returns the integrator's
// new_value, not old_value).
TEST(NumSimMaterialEndToEnd, RK4HitsHighAccuracy) {
  EXPECT_NEAR(run(100, numsim::materials::rk4()), kExact, 1e-8);
}

// Implicit path: wires and reads the generated `rate_derivative` Local edge
// inside the integrator's per-stage Newton solve (so a MISSING/unwired
// rate_derivative fails here). NOTE: convergence alone does NOT verify the
// derivative VALUE — at this conditioning the per-stage Newton converges as a
// contraction regardless of the Jacobian. The value is verified separately by
// EmittedRateAndDerivativeValuesAreCorrect below.
TEST(NumSimMaterialEndToEnd, ImplicitEulerWiresRateDerivative) {
  EXPECT_NEAR(run(1000, numsim::materials::implicit_euler()), kExact, 1e-3);
}

// Directly verify the emitted VALUES (not just convergence): fire the generated
// material's compute() at a known α and read its rate / rate_derivative
// properties. For dα/dt = K·α this must give rate = K·α and rate_derivative = K.
// This is the check that catches a wrong-but-convergent emitted derivative.
TEST(NumSimMaterialEndToEnd, EmittedRateAndDerivativeValuesAreCorrect) {
  ctx_type ctx;
  param_type p;
  auto fe = numsim::materials::forward_euler();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "LinearHardening");
  p.insert<T>("step_size", T{0.1});
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &fe);
  ctx.create<RK>(p);
  p.clear();
  p.insert<std::string>("name", "LinearHardening");
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<T>("K", T{-1.0});
  ctx.create<Gen>(p);
  ctx.finalize();

  auto* sp = ctx.find_property("integrator", "state");
  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(sp);
  ASSERT_NE(hist, nullptr);
  const T alpha = T{2.0};
  hist->new_value() = alpha; // input_property::get() reads the trial new_value

  // Fire the generated material's compute() via the rate property's callback.
  ctx.find_property("LinearHardening", "rate")->traits().update();

  const T K = T{-1.0};
  EXPECT_NEAR(ctx.get<T>("LinearHardening", "rate"), K * alpha, 1e-12);
  EXPECT_NEAR(ctx.get<T>("LinearHardening", "rate_derivative"), K, 1e-12);
}

} // namespace
