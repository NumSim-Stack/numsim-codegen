#ifndef NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H
#define NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>
#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// Plain C++ target — emits a single inline header containing the generic
// compute function with the user's declared signature. No framework
// boundary; the user calls the function directly with tmech::tensor
// arguments. This is the simplest target and the one used by tests.
//
// Phase 3b-2b: takes the dense-solve backend for coupled local-Newton systems
// (default Eigen). The SAME instance drives both emission and the include, so
// they cannot disagree, and a single build can target different backends per
// output (e.g. Armadillo here, Eigen for MOOSE).
class StandaloneCxxTarget : public Target {
public:
  explicit StandaloneCxxTarget(
      LinearAlgebraEmitter const &la = default_linear_algebra_emitter())
      : m_la(la) {}
  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override;
  [[nodiscard]] auto target_name() const -> std::string override;

private:
  LinearAlgebraEmitter const &m_la;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H
