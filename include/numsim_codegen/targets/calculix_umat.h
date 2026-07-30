#ifndef NUMSIM_CODEGEN_TARGETS_CALCULIX_UMAT_H
#define NUMSIM_CODEGEN_TARGETS_CALCULIX_UMAT_H

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>
#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// CalculiX user-material target — emits a single `.cpp` exposing an
// `extern "C" void umat_user_(...)` with CalculiX's native umat_user ABI, so
// the generated material can be linked into `ccx` built from source (its
// `umat_main.f` dispatches to `umat_user` for any *MATERIAL whose name begins
// with `USER`).
//
// The file embeds the target-agnostic Layer-2 `<Model>_compute` (full dense
// tmech tensors, no Voigt) and wraps it in a Voigt boundary built from tmech's
// `abq_std` adaptor, whose ordering `{11,22,33,12,13,23}` is exactly
// CalculiX's `emec`/`stre`/tangent order:
//   * `emec` (tensorial Lagrange strain) → full strain tensor via
//     `abq_std<3,false>` (no engineering-shear scaling — CalculiX passes the
//     tensorial components);
//   * stress tensor → `stre(6)` via the same adaptor;
//   * the minor-symmetric rank-4 tangent → a 6×6 via `abq_std<3,false>` (which,
//     for a minor-symmetric C, equals CalculiX's engineering `stiff` D-matrix
//     directly — no ×2 factors), then packed into `stiff(21)` column-major
//     upper-triangular with symmetrization.
//
// SCOPE (first cut, linear elasticity): stateless recipes only — exactly one
// rank-2 tensor input (strain), one rank-2 tensor output (stress), one rank-4
// consistent tangent, and any number of scalar parameters (mapped to the
// *USER MATERIAL constants via `elconloc`). Scalar inputs, state variables,
// evolution equations, time-step, and Newton state-outputs are rejected at
// emit time — the stateful (xstate) path is a tracked follow-up.
class CalculiXUMATTarget : public Target {
public:
  explicit CalculiXUMATTarget(
      LinearAlgebraEmitter const &la = default_linear_algebra_emitter())
      : m_la(la) {}
  // `m_la` borrows — reject a temporary emitter at compile time so it can't
  // dangle (mirrors StandaloneCxxTarget). Pass an accessor singleton
  // (default_/eigen_/armadillo_linear_algebra_emitter()), not `Emitter{}`.
  CalculiXUMATTarget(LinearAlgebraEmitter const &&) = delete;
  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override;
  [[nodiscard]] auto target_name() const -> std::string override;

private:
  LinearAlgebraEmitter const &m_la;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_CALCULIX_UMAT_H
