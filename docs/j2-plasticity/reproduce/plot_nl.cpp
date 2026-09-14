// Uniaxial stress-strain of the GENERATED J2 material for three isotropic
// hardening laws: linear, Voce (saturating), Swift (power). Prints CSV.
#include <cstdio>
#include <unordered_set>
#include <cmath>
#include <string>
#include <functional>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "J2Return.h"
#include "J2Voce.h"
#include "J2Swift.h"

using policy   = numsim::materials::material_policy_default;
using T        = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param    = policy::ParameterHandler;
using tensor2  = tmech::tensor<T, 3, 2>;
namespace gen  = numsim::materials::generated;

static constexpr T G = T{80000};

template <class Mat>
void sweep(const char* tag, std::function<void(param&)> set_params) {
  ctx_type ctx; param p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-13}); p.insert<int>("max_iter", 60);
  ctx.create<numsim::materials::backward_euler<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "M");
  p.insert<T>("G", G);
  set_params(p);
  p.insert<std::string>("solver_source", "solver");
  p.insert<std::string>("strain_source", "stepper");
  ctx.create<Mat>(p);
  ctx.finalize();

  auto& eps = ctx.get_mutable<tensor2>("stepper", "strain");
  auto* ep  = ctx.find_property("stepper", "strain");
  auto* dg  = ctx.find_property("M", "dgamma");
  const std::unordered_set<const numsim::materials::property_base*> excl{ep};

  const int N = 200; const T emax = T{0.02};
  for (int k = 1; k <= N; ++k) {
    const T t = emax * T(k) / T(N);
    tensor2 e{}; e(0,0) = t;
    try {
      eps = e; dg->revert();
      ctx.update_property("M", "stress", excl);
      tensor2 s = ctx.get<tensor2>("M", "stress");
      const T tr = s(0,0)+s(1,1)+s(2,2);
      tensor2 d = s; d(0,0)-=tr/3; d(1,1)-=tr/3; d(2,2)-=tr/3;
      const T vm = std::sqrt(T{1.5}) * std::sqrt(tmech::dcontract(d, d));
      std::printf("%s,%.6g,%.6g,%.6g\n", tag, t, s(0,0), vm);
    } catch (const std::exception&) {}
  }
}

int main() {
  std::printf("law,eps,sig,vm\n");
  sweep<gen::J2Return<policy>>("linear", [](param& p){
    p.insert<T>("sy", T{250}); p.insert<T>("H", T{5000}); });
  sweep<gen::J2Voce<policy>>("voce", [](param& p){
    p.insert<T>("sy0", T{250}); p.insert<T>("Q", T{300}); p.insert<T>("b", T{400}); });
  sweep<gen::J2Swift<policy>>("swift", [](param& p){
    p.insert<T>("C", T{1700}); p.insert<T>("e0", T{0.004}); p.insert<T>("k", T{0.35}); });
  return 0;
}
