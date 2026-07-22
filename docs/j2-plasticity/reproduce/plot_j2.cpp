// Sweep the codegen-GENERATED J2 return-map material over several loading modes
// and print (mode, strain-measure, stress components) as CSV. The strain tensor
// is set directly on the producer and Δγ is reverted per point, so each row is
// the monotonic σ(ε) of the generated material (elastic branch via the clamp,
// then plastic radial return).
#include <cstdio>
#include <unordered_set>
#include <cmath>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "J2Return.h"

using policy   = numsim::materials::material_policy_default;
using T        = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param    = policy::ParameterHandler;
using tensor2  = tmech::tensor<T, 3, 2>;
using J2Return = numsim::materials::generated::J2Return<policy>;

// Material parameters (steel-like, MPa).
static constexpr T G  = T{80000};  // shear modulus
static constexpr T SY = T{250};    // yield stress
static constexpr T H  = T{5000};   // linear isotropic hardening modulus

static tensor2 vonmises_dev(const tensor2& s) {
  const T tr = s(0, 0) + s(1, 1) + s(2, 2);
  tensor2 d = s;
  d(0, 0) -= tr / T{3}; d(1, 1) -= tr / T{3}; d(2, 2) -= tr / T{3};
  return d;
}

int main() {
  ctx_type ctx;
  param p;
  p.insert<std::string>("name", "stepper");                 // producer of `strain`
  p.insert<T>("increment", T{0});                           // we set strain directly
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13});
  p.insert<int>("max_iter", 50);
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "J2");
  p.insert<T>("G", G); p.insert<T>("sy", SY); p.insert<T>("H", H);
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<J2Return>(p);
  ctx.finalize();

  auto& eps   = ctx.get_mutable<tensor2>("stepper", "strain");
  auto* eps_p = ctx.find_property("stepper", "strain");
  auto* dg    = ctx.find_property("J2", "dgamma");
  const std::unordered_set<const numsim::materials::property_base*> excl{eps_p};

  auto sample = [&](tensor2 e) -> tensor2 {
    eps = e;
    dg->revert();                                   // Δγ_old = 0 → monotonic σ(ε)
    ctx.update_property("J2", "stress", excl);
    return ctx.get<tensor2>("J2", "stress");
  };

  std::printf("mode,drive,sig,vm\n");
  const int N = 200;
  const T emax = T{0.02};
  for (int k = 1; k <= N; ++k) {                    // skip t=0 (flow dir undefined at dev=0)
    const T t = emax * T(k) / T(N);
    struct Mode { const char* name; tensor2 e; T drive; bool shear; };
    tensor2 uni{};  uni(0,0) = t;                                   // uniaxial strain
    tensor2 sh{};   sh(0,1) = t; sh(1,0) = t;                       // simple shear (ε01=ε10=t)
    tensor2 bi{};   bi(0,0) = t; bi(1,1) = t;                       // equibiaxial (2 equal eig)
    Mode modes[] = {
      {"uniaxial",    uni, t,   false},
      {"shear",       sh,  2*t, true},   // engineering shear γ = 2 ε01
      {"equibiaxial", bi,  t,   false},
    };
    for (auto& m : modes) {
      try {
        tensor2 s = sample(m.e);
        tensor2 d = vonmises_dev(s);
        const T vm = std::sqrt(T{1.5}) * std::sqrt(tmech::dcontract(d, d));
        const T sig = m.shear ? s(0,1) : s(0,0);
        std::printf("%s,%.6g,%.6g,%.6g\n", m.name, m.drive, sig, vm);
      } catch (const std::exception&) { /* singular point — skip */ }
    }
  }
  return 0;
}
