#ifndef NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H
#define NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H

#include <numsim_codegen/targets/target.h>

namespace numsim::codegen {

// Plain C++ target — emits a single inline header containing the generic
// compute function with the user's declared signature. No framework
// boundary; the user calls the function directly with tmech::tensor
// arguments. This is the simplest target and the one used by tests.
class StandaloneCxxTarget : public Target {
public:
  [[nodiscard]] auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> override {
    return {EmittedFile{model.name() + ".h", model.emit_compute_function()}};
  }

  [[nodiscard]] auto target_name() const -> std::string override {
    return "StandaloneCxx";
  }
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_STANDALONE_CXX_H
