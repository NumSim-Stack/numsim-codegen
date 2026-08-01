#ifndef NUMSIM_CODEGEN_TARGETS_CALCULIX_EXTERNAL_H
#define NUMSIM_CODEGEN_TARGETS_CALCULIX_EXTERNAL_H

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>
#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// CalculiX *external behaviour* target — emits a `.cpp` that compiles to a
// shared library (`lib<MODEL>.so`) which `ccx` loads at RUNTIME via dlopen. No
// per-material ccx recompile: build ccx ONCE with external-behaviour support
// (`-DCALCULIX_EXTERNAL_BEHAVIOURS_SUPPORT -ldl`), then select the material in
// the input deck with `*MATERIAL, NAME=@<MODEL>_NCG_UMAT` — ccx uppercases the
// name, parses `@<LIB>_<FUNC>`, `dlopen`s `lib<LIB>.so` and `dlsym`s `<FUNC>`.
//
// vs. CalculiXUMATTarget (which links the material INTO ccx as umat_user_): same
// tmech `abq_std` Voigt boundary (the external STANDARD interface passes NATIVE
// quantities — STRAN1=emec tensorial, STRESS=stre, DDSDDE=stiff(21) packed), but
// wrapped in CalculiX's external ABI (the `calculixptr` signature) and organised
// as a plugin lifecycle:
//   * ② a `thread_local` material instance is built once per thread (ccx runs
//     the element loop multi-threaded, so per-thread state is required for a
//     stateful material and cheap for a stateless one);
//   * ③ each call unpacks STRAN1 → evaluates → packs STRESS/DDSDDE.
// (① a load-time registry is only needed for a multi-model library and for the
// numsim-materials-backed stateful path; documented in the emitted source.)
//
// SCOPE (first cut, linear elasticity): stateless recipes only — one rank-2
// strain input, one rank-2 stress output, one rank-4 consistent tangent, scalar
// parameters (→ the *USER MATERIAL constants, `MPROPS`). State variables (the
// STATEV round-trip) are the numsim-materials-backed follow-up.
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
