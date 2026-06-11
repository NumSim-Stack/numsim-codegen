#ifndef NUMSIM_CODEGEN_TARGETS_NUMSIM_MATERIAL_H
#define NUMSIM_CODEGEN_TARGETS_NUMSIM_MATERIAL_H

#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// numsim-materials target (graph-coupled / property-graph runtime).
//
// Under the graph-coupled architecture (roadmap D14/D17/D19), the solver,
// time integration, and linear algebra live in numsim-materials. codegen's job
// is to emit a constitutive *material* exposing the constitutive RATE and its
// derivative as properties, plus the JSON wiring; the existing
// `rk_integrator` (or `backward_euler`) drives the time stepping.
//
// First increment scope: a recipe with exactly ONE scalar state variable and
// ONE scalar evolution equation `dx/dt = f(x, params)`. Emits a rate-function
// material conforming to the `rk_integrator` contract — `rate` (= f) and
// `rate_derivative` (= ∂f/∂x, via cas::diff) as Local-edge properties, reading
// `x` from the integrator's `state`. The compute() body is filled by the
// existing scalar code emitter; this target adds only the material boilerplate
// + property/parameter mapping + a JSON config. Tensor state, multiple coupled
// states (needs numsim-materials#12), and scalar inputs are out of scope here
// and throw with a clear message.
class NumSimMaterialTarget : public Target {
public:
  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override;
  [[nodiscard]] auto target_name() const -> std::string override;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_NUMSIM_MATERIAL_H
