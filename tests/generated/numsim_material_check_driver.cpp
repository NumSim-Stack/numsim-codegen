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

#include "LinearHardening.h" // generated: dα/dt = K·α (+ scalar output)
#include "NonlinearDecay.h"  // generated: dα/dt = K·α²

// The tensor-stress end-to-end case is GCC-only. Wiring a tensor input
// (input_property<tmech::tensor>::wire) instantiates history_property<tensor>'s
// `static_assert(is_trivially_copyable_v<T>)`, which FAILS under clang-19 because
// tmech::tensor is trivially copyable under gcc-14 but not clang-19. This is an
// UPSTREAM numsim-core/tmech incompatibility (it breaks numsim-materials' own
// linear_elasticity under clang too), not a defect in the generated code — which
// gcc compiles and runs correctly. Tracked: numsim-core#16. The emit shape itself
// is verified compiler-independently by NumSimMaterialTargetTest.EmitsTensorStressOutput.
#if defined(__GNUC__) && !defined(__clang__)
#define NCG_TENSOR_E2E 1
#include <tmech/tmech.h>
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/postprocessing/numerical_diff_checker.h"
#include "Viscoelastic.h" // generated: σ = α·ε (tensor stress from scalar state)
#include "ReturnMap.h"     // generated: implicit residual R(z,ε)=z−c·tr(ε), σ=z·ε
#include "ReturnMapCubic.h" // generated: NONLINEAR residual R=z+z³−c·tr(ε)
#include "J2PathDep.h"   // generated: PATH-DEPENDENT J2 (tensor εᵖ history, #92)
#endif

#include <gtest/gtest.h>

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using RK = numsim::materials::rk_integrator<policy>;

using Linear = numsim::materials::generated::LinearHardening<policy>;
using Nonlinear = numsim::materials::generated::NonlinearDecay<policy>;
#ifdef NCG_TENSOR_E2E
using Visco = numsim::materials::generated::Viscoelastic<policy>;
using ReturnMap = numsim::materials::generated::ReturnMap<policy>;
using ReturnMapCubic = numsim::materials::generated::ReturnMapCubic<policy>;
using J2PathDep = numsim::materials::generated::J2PathDep<policy>;
using tensor2 = tmech::tensor<T, 3, 2>;
#endif

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

// Fire a generated scalar OUTPUT's update callback at a known α and return its
// emitted value — verifies B.1 output emission through the real material graph.
template <typename Mat>
T emitted_output(const std::string& name, const std::string& prop, T alpha) {
  ctx_type ctx;
  const auto tab = numsim::materials::forward_euler(); // must outlive build()
  build<Mat>(ctx, name, tab, T{0.1});

  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(
      ctx.find_property("integrator", "state"));
  EXPECT_NE(hist, nullptr);
  hist->new_value() = alpha;

  auto* p = ctx.find_property(name, prop);
  EXPECT_NE(p, nullptr) << "generated material must publish output `" << prop << "`";
  if (!p) return std::nan("");
  p->traits().update();
  return ctx.get<T>(name, prop);
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

// B.1: a generated scalar output property computes correctly through the graph.
TEST(NumSimMaterialEndToEnd, LinearEmittedScalarOutputIsCorrect) {
  EXPECT_NEAR(emitted_output<Linear>("LinearHardening", "twice_state", T{2.5}),
              T{5.0}, 1e-12); // 2·α
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

// ── Tensor stress (GCC-only — see the NCG_TENSOR_E2E note at the top) ─────────
#ifdef NCG_TENSOR_E2E

// Read a plain tensor property (stress/strain are add_output properties).
tensor2 read_tensor(ctx_type& ctx, const char* mat, const char* prop) {
  auto* tp = dynamic_cast<
      numsim_core::property<tensor2, numsim::materials::property_traits>*>(
      ctx.find_property(mat, prop));
  EXPECT_NE(tp, nullptr) << "tensor property " << mat << "::" << prop;
  return tp ? tp->get() : tensor2{};
}

// Drive a strain producer + scalar integrator + the generated Viscoelastic
// material through the real graph; check the tensor stress output σ = α·ε.
TEST(NumSimMaterialEndToEnd, TensorStressFromScalarStateAndStrain) {
  ctx_type ctx;
  param_type p;

  // strain producer: accumulates `increment` into component (0,0) per update.
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.01});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  const auto tab = numsim::materials::forward_euler();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "Viscoelastic");
  p.insert<T>("step_size", T{0.1});
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &tab);
  ctx.create<RK>(p);

  p.clear();
  p.insert<std::string>("name", "Viscoelastic");
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<T>("K", kK);
  ctx.create<Visco>(p);

  ctx.finalize();
  auto* hist = dynamic_cast<
      numsim_core::history_property<T, numsim::materials::property_traits>*>(
      ctx.find_property("integrator", "state"));
  ASSERT_NE(hist, nullptr);
  hist->old_value() = T{1};
  hist->new_value() = T{1};
  for (int i = 0; i < 5; ++i) {
    ctx.update();
    ctx.commit();
  }

  const T alpha = ctx.get<T>("integrator", "state"); // scalar history — fine
  const auto eps = read_tensor(ctx, "stepper", "strain");
  const auto sig = read_tensor(ctx, "Viscoelastic", "stress");
  EXPECT_NEAR(sig(0, 0), alpha * eps(0, 0), 1e-12); // σ = α·ε on the driven comp
  EXPECT_NEAR(sig(0, 1), T{0}, 1e-12);              // off-diagonal strain is 0

  // Phase D: consistent tangent dσ/dε = α·P_sym (rank-4, minor-symmetric).
  using tensor4 = tmech::tensor<T, 3, 4>;
  auto* cp = dynamic_cast<
      numsim_core::property<tensor4, numsim::materials::property_traits>*>(
      ctx.find_property("Viscoelastic", "dstress_dstrain"));
  ASSERT_NE(cp, nullptr);
  const auto C = cp->get();
  EXPECT_NEAR(C(0, 0, 0, 0), alpha, 1e-12);       // P_sym(0,0,0,0)=1
  EXPECT_NEAR(C(0, 1, 0, 1), alpha * 0.5, 1e-12); // minor-symmetric ½
  EXPECT_NEAR(C(0, 0, 1, 1), T{0}, 1e-12);        // off-block zero
}

// ── Strain-coupled implicit residual (Mode B) ────────────────────────────────
// Proof the residual emit (material_ref<backward_euler> + solve(eval)) compiles
// and runs against the REAL backward_euler in caller-driven mode: drive a strain
// producer + the generated ReturnMap; the material solves R(z,ε)=z−c·tr(ε)=0
// internally, so z = c·tr(ε) and σ = z·ε.
TEST(NumSimMaterialEndToEnd, ResidualReturnMapSolvesAgainstBackwardEuler) {
  ctx_type ctx;
  param_type p;

  // strain producer: accumulates `increment` into component (0,0) per update.
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.02});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  // caller-driven backward_euler (no "function" → the material drives solve()).
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-12});
  p.insert<int>("max_iter", 50);
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  const T c = T{2.0};
  p.insert<std::string>("name", "ReturnMap");
  p.insert<T>("c", c);
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<ReturnMap>(p);

  ctx.finalize();
  ctx.update();
  ctx.commit();

  const auto eps = read_tensor(ctx, "stepper", "strain");
  const T tr = eps(0, 0) + eps(1, 1) + eps(2, 2);
  const T z = ctx.get<T>("ReturnMap", "z");        // solved scalar state
  const auto sig = read_tensor(ctx, "ReturnMap", "stress");

  EXPECT_NEAR(z, c * tr, 1e-10);                   // R=0 ⇒ z = c·tr(ε)
  EXPECT_NEAR(sig(0, 0), z * eps(0, 0), 1e-10);    // σ = z·ε
  EXPECT_NEAR(sig(0, 1), T{0}, 1e-12);             // off-diagonal strain is 0
}

// Phase 2b: the strain-coupled CONSISTENT TANGENT of the generated ReturnMap,
// verified numerically through the real backward_euler solve. For σ = z·ε with
// z solving R = z − c·tr(ε) = 0:
//   dσ/dε = ∂σ/∂ε + ∂σ/∂z ⊗ dz/dε = z·I⁴ˢ + ε ⊗ (c·I).
// The second term is the implicit coupling a naive explicit ∂σ/∂ε drops. The
// load-bearing assertion is the off-block C_{0011} = c·ε₀₀ ≠ 0 — exactly the
// component the rate-path (strain-only) Viscoelastic tangent above has as ZERO.
TEST(NumSimMaterialEndToEnd, ResidualReturnMapConsistentTangentHasCouplingTerm) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.02});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-12});
  p.insert<int>("max_iter", 50);
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  const T c = T{2.0};
  p.insert<std::string>("name", "ReturnMap");
  p.insert<T>("c", c);
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<ReturnMap>(p);

  ctx.finalize();
  ctx.update();
  ctx.commit();

  const auto eps = read_tensor(ctx, "stepper", "strain");
  const T e00 = eps(0, 0);
  const T z = ctx.get<T>("ReturnMap", "z"); // = c·tr(ε) = c·e00

  using tensor4 = tmech::tensor<T, 3, 4>;
  auto* cp = dynamic_cast<
      numsim_core::property<tensor4, numsim::materials::property_traits>*>(
      ctx.find_property("ReturnMap", "dstress_dstrain"));
  ASSERT_NE(cp, nullptr);
  const auto C = cp->get();

  // Coupling term (the whole point): C_{0011} = z·I⁴ˢ_{0011} + ε₀₀·(c·I)_{11}
  //                                           = 0 + c·e00.  Naive ∂σ/∂ε ⇒ 0.
  EXPECT_NEAR(C(0, 0, 1, 1), c * e00, 1e-10);
  EXPECT_GT(std::abs(C(0, 0, 1, 1)), 1e-6) << "coupling term must be nonzero";
  // Explicit base term (correction vanishes here): C_{0101} = z·I⁴ˢ_{0101} = z·½.
  EXPECT_NEAR(C(0, 1, 0, 1), z * T{0.5}, 1e-10);
  // Diagonal: C_{0000} = z·I⁴ˢ_{0000} + c·e00 = z + c·e00.
  EXPECT_NEAR(C(0, 0, 0, 0), z + c * e00, 1e-10);
}

// Phase C (#90, roadmap D18): INDEPENDENT verification of the generated
// consistent tangent. The prior tests pin specific components against a
// hand-derived closed form; this instead points numsim-materials'
// `numerical_diff_checker` at the generated ReturnMap material and finite-
// differences the WHOLE dσ/dε through the real property graph — perturbing the
// strain, re-solving the Newton (reverting the z history each sample so every
// perturbation restarts from the same z_old), and comparing every rank-4
// component against our emitted `dstress_dstrain`. This catches any component
// the closed-form assertions did not enumerate, and validates the coupling term
// via the re-solve rather than by construction.
TEST(NumSimMaterialEndToEnd, ResidualReturnMapTangentMatchesNumericalDiff) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.02});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13});
  p.insert<int>("max_iter", 50);
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "ReturnMap");
  p.insert<T>("c", T{2.0});
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<ReturnMap>(p);

  // The FD checker: FD-differentiate ReturnMap::stress w.r.t. stepper::strain and
  // compare to the analytical ReturnMap::dstress_dstrain. Revert the z history
  // between perturbations so each re-solve starts from the same previous state.
  p.clear();
  p.insert<std::string>("name", "checker");
  p.insert<ctx_type*>("context", &ctx);
  p.insert<std::string>("output_source", "ReturnMap::stress");
  p.insert<std::string>("input_source", "stepper::strain");
  p.insert<std::string>("analytical_source", "ReturnMap::dstress_dstrain");
  p.insert<std::vector<std::string>>("history_sources", {"ReturnMap::z"});
  p.insert<T>("epsilon", T{1e-7});
  ctx.create<numsim::materials::tangent_checker<policy>>(p);

  ctx.finalize();

  // Step through several increments; the tangent must FD-match at every state.
  T max_rel = 0;
  for (int i = 0; i < 8; ++i) {
    ctx.update();
    max_rel = std::max(max_rel, ctx.get<T>("checker", "rel_error"));
    ctx.commit();
  }
  // Smooth response (z = c·tr(ε), no transitions), so central FD is tight.
  EXPECT_LT(max_rel, 1e-6) << "generated dσ/dε disagrees with numerical diff";
}

// NONLINEAR residual R(z,ε) = z + z³ − c·tr(ε), so ∂R/∂z = 1 + 3z². This is the
// test that VALIDATES THE EMITTED JACOBIAN: with a tight Newton budget and a
// root z≈0.682 (where ∂R/∂z≈2.4 ≫ 1), a wrong derivative oscillates and the
// converged residual is NOT ~0 — unlike the linear case where any jacobian
// reaches the root. We drive c·tr(ε)=1 (strain increment 0.5 on one component).
TEST(NumSimMaterialEndToEnd, NonlinearResidualValidatesEmittedJacobian) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.5}); // tr(ε) = 0.5 after one update
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  // Tight iteration budget: the correct jacobian (1+3z²) converges in ~6 Newton
  // steps; a wrong constant jacobian would not reach tol here.
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13});
  p.insert<int>("max_iter", 12);
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  const T c = T{2.0};
  p.insert<std::string>("name", "ReturnMapCubic");
  p.insert<T>("c", c);
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<ReturnMapCubic>(p);

  ctx.finalize();
  ctx.update();
  ctx.commit();

  const auto eps = read_tensor(ctx, "stepper", "strain");
  const T tr = eps(0, 0) + eps(1, 1) + eps(2, 2);
  const T z = ctx.get<T>("ReturnMapCubic", "z");
  const auto sig = read_tensor(ctx, "ReturnMapCubic", "stress");

  // The converged state must satisfy R(z,ε)=0: z + z³ = c·tr(ε). A wrong
  // emitted ∂R/∂z fails to converge within the budget, so this residual ≠ 0.
  const T residual = z + z * z * z - c * tr;
  EXPECT_NEAR(residual, T{0}, 1e-9) << "z=" << z << " (∂R/∂z must be 1+3z²)";
  // Sanity: the root of z+z³=1 is ≈ 0.6823278.
  EXPECT_NEAR(z, T{0.6823278038280193}, 1e-6);
  EXPECT_NEAR(sig(0, 0), z * eps(0, 0), 1e-10); // σ = z·ε

  // Phase 2b: the tangent on the NONLINEAR residual VALIDATES THE DIVISION by
  // ∂R/∂z. Here ∂R/∂z = 1+3z² ≠ 1, so dz/dε = c·I/(1+3z²) and the coupling term
  // C_{0011} = ε₀₀·c/(1+3z²). If the emitter dropped the /∂R/∂z factor (which the
  // LINEAR ReturnMap's ∂R/∂z≡1 cannot detect), C_{0011} would be ε₀₀·c instead —
  // off by the ~2.4× jacobian here. This is the load-bearing division check.
  using tensor4 = tmech::tensor<T, 3, 4>;
  auto* cp = dynamic_cast<
      numsim_core::property<tensor4, numsim::materials::property_traits>*>(
      ctx.find_property("ReturnMapCubic", "dstress_dstrain"));
  ASSERT_NE(cp, nullptr);
  const auto C = cp->get();
  const T dRdz = T{1} + T{3} * z * z; // = ∂R/∂z at the converged z
  EXPECT_GT(dRdz, T{2}) << "must be ≫1 so the division is observable";
  EXPECT_NEAR(C(0, 0, 1, 1), c * eps(0, 0) / dRdz, 1e-9);
  // Contrast: without the division it would be c·ε₀₀ — assert we are NOT that.
  EXPECT_GT(std::abs(C(0, 0, 1, 1) - c * eps(0, 0)), 1e-3)
      << "tangent must divide by ∂R/∂z, not omit it";
}

// ── #92 golden: PATH-DEPENDENT J2 (tensor εᵖ history) ────────────────────────
// The property that single-increment plasticity CANNOT show: plastic strain
// accumulates under load and is RETAINED on unload (residual plastic strain),
// and unloading is elastic. εᵖ is carried across steps as tensor history.
namespace {
void set_strain00(ctx_type& ctx, tensor2& eps,
                  const std::unordered_set<const numsim::materials::property_base*>& excl,
                  T e) {
  tensor2 E{}; E(0, 0) = e; eps = E;
  ctx.update_property("J2", "stress", excl);
}
} // namespace

TEST(NumSimMaterialEndToEnd, J2PathDependentRetainsPlasticStrainOnUnload) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13});
  p.insert<int>("max_iter", 50);
  ctx.create<numsim::materials::backward_euler<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "J2");
  p.insert<T>("G", T{80}); p.insert<T>("sy", T{1.0}); p.insert<T>("H", T{10.0});
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<J2PathDep>(p);
  ctx.finalize();

  auto& eps = ctx.get_mutable<tensor2>("stepper", "strain");
  auto* ep = ctx.find_property("stepper", "strain");
  const std::unordered_set<const numsim::materials::property_base*> excl{ep};

  // Load well into the plastic regime, committing each step so εᵖ accumulates.
  T peak_epsp = 0;
  for (T e : {T{0.006}, T{0.010}, T{0.015}, T{0.020}}) {
    set_strain00(ctx, eps, excl, e);
    ctx.commit();
    peak_epsp = ctx.get<tensor2>("J2", "eps_p")(0, 0);
  }
  ASSERT_GT(peak_epsp, T{1e-3}) << "should have accumulated plastic strain";
  const T peak_sig = read_tensor(ctx, "J2", "stress")(0, 0);

  // Unload elastically to a lower strain (still above the reverse-yield point).
  set_strain00(ctx, eps, excl, T{0.010});
  ctx.commit();
  const T unl_epsp = ctx.get<tensor2>("J2", "eps_p")(0, 0);
  const T unl_sig = read_tensor(ctx, "J2", "stress")(0, 0);

  // εᵖ is RETAINED (elastic unloading — no plastic flow), stress drops.
  EXPECT_NEAR(unl_epsp, peak_epsp, 1e-12)
      << "plastic strain must be retained on elastic unloading";
  EXPECT_LT(unl_sig, peak_sig) << "stress must decrease on unload";
  // Elastic unload slope: Δσ = 2G·Δε on the driven component's deviatoric part.
  // (σ00 vs ε00 slope ~ 2G·2/3.) Just assert it moved a lot (elastic, steep).
  EXPECT_GT(peak_sig - unl_sig, T{0.5}) << "unload must be elastic (steep)";
}

// FD-verify the consistent tangent of the path-dependent material — the FD
// checker must revert BOTH histories (α AND εᵖ) so each perturbation is taken at
// fixed history, matching the analytical tangent (which holds εᵖ_old fixed).
// NOTE: the emitted tangent is the PLASTIC-branch (algorithmic) tangent; on the
// elastic branch (backward_euler's max(x,0) clamp gives Δγ=0) it does NOT match
// the elastic FD tangent — that switch is the Kuhn-Tucker work (#35). So we load
// solidly into yield (increment ≫ ε_yield) and only check plastic states.
TEST(NumSimMaterialEndToEnd, J2PathDependentTangentMatchesNumericalDiff) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.01}); // ε_yield ≈ 0.0077 → every step is plastic
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13});
  p.insert<int>("max_iter", 60);
  ctx.create<numsim::materials::backward_euler<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "J2");
  p.insert<T>("G", T{80}); p.insert<T>("sy", T{1.0}); p.insert<T>("H", T{10.0});
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<J2PathDep>(p);
  p.clear();
  p.insert<std::string>("name", "checker");
  p.insert<ctx_type*>("context", &ctx);
  p.insert<std::string>("output_source", "J2::stress");
  p.insert<std::string>("input_source", "stepper::strain");
  p.insert<std::string>("analytical_source", "J2::dstress_dstrain");
  // BOTH histories reverted per FD sample — the load-bearing part for a
  // path-dependent material.
  p.insert<std::vector<std::string>>("history_sources", {"J2::alpha", "J2::eps_p"});
  p.insert<T>("epsilon", T{1e-7});
  ctx.create<numsim::materials::tangent_checker<policy>>(p);
  ctx.finalize();

  T max_rel = 0;
  for (int i = 0; i < 6; ++i) {
    ctx.update();
    ASSERT_GT(ctx.get<T>("J2", "alpha"), T{0}) << "step " << i << ": expected plastic";
    max_rel = std::max(max_rel, ctx.get<T>("checker", "rel_error"));
    ctx.commit();
  }
  EXPECT_LT(max_rel, 1e-6) << "path-dependent J2 tangent disagrees with FD";
}

#endif // NCG_TENSOR_E2E

} // namespace
