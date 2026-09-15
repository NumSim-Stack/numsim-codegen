#ifndef NUMSIM_CODEGEN_TARGETS_CALCULIX_EXTERNAL_H
#define NUMSIM_CODEGEN_TARGETS_CALCULIX_EXTERNAL_H

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>
#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// CalculiX external-behaviour target: emits a `.cpp` compiling to `lib<MODEL>.so`,
// which ccx dlopens at RUNTIME — build ccx once with
// `-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT -ldl`, then select per material with
// `*MATERIAL, NAME=@<MODEL>_NCG_UMAT` (ccx uppercases, splits `@<LIB>_<FUNC>`,
// dlopens `lib<LIB>.so`, dlsyms `<FUNC>`). No recompile per material.
//
// The hook passes NATIVE quantities under Abaqus-flavoured names — STRAN1=emec
// (tensorial), STRESS=stre, DDSDDE=stiff(21) — so the boundary is a tmech
// `abq_std` adaptor plus the stiff(21) packing in calculix_boundary.h. The
// evaluator is thread_local: ccx runs the element loop multi-threaded.
//
// SCOPE: stateless recipes only — one rank-2 strain in, one rank-2 stress out,
// one rank-4 tangent, scalar parameters (the *USER MATERIAL constants). The
// STATEV round-trip is the numsim-materials-backed follow-up (see #160).
class CalculiXExternalTarget : public Target {
public:
  explicit CalculiXExternalTarget(
      LinearAlgebraEmitter const &la = default_linear_algebra_emitter())
      : m_la(la) {}
  CalculiXExternalTarget(LinearAlgebraEmitter const &&) = delete;
  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override;
  [[nodiscard]] auto target_name() const -> std::string override;

private:
  LinearAlgebraEmitter const &m_la;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_CALCULIX_EXTERNAL_H
