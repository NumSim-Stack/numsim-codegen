#ifndef NUMSIM_CODEGEN_TARGETS_MOOSE_MATERIAL_H
#define NUMSIM_CODEGEN_TARGETS_MOOSE_MATERIAL_H

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>
#include <numsim_codegen/targets/target.h>

#include <string>

namespace numsim::codegen {

// MOOSE Material backend. Generates a header + source pair:
//
//   <ModelName>.h  — Material class declaration with validParams,
//                    constructor signature, MaterialProperty members.
//   <ModelName>.C  — Material class implementation: validParams body,
//                    constructor body, computeQpProperties body that
//                    delegates to the Layer 2 compute function.
//
// The Layer 2 compute function is embedded as an inline header at the
// top of the .C file. The boundary conversion uses tmech::adaptor over
// MOOSE's RankTwoTensor / RankFourTensor raw data pointers — assumes the
// underlying storage is contiguous row-major doubles, which matches
// MOOSE's internal layout for RankTwoTensorTempl<Real> / RankFourTensorTempl.
//
// Inputs tagged with roles::Strain become MaterialProperty<RankTwoTensor>
// reads. Outputs tagged with roles::Stress become MaterialProperty<RankTwoTensor>
// writes. Role attributes (is_stateful, is_symmetric, ...) drive backend
// decisions — user-defined roles flow through transparently.
class MooseMaterialTarget : public Target {
public:
  explicit MooseMaterialTarget(
      std::string app_name = "MyApp",
      LinearAlgebraEmitter const &la = default_linear_algebra_emitter())
      : m_app_name(std::move(app_name)), m_la(la) {}

  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override;
  [[nodiscard]] auto target_name() const -> std::string override;

private:
  std::string m_app_name;
  LinearAlgebraEmitter const &m_la;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_MOOSE_MATERIAL_H
