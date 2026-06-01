#ifndef NUMSIM_CODEGEN_PASS_H
#define NUMSIM_CODEGEN_PASS_H

#include <numsim_codegen/code_emit/codegen_context.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace numsim::codegen {

class ConstitutiveModel;

// Shared state for a single PassManager invocation. Passes read the
// (read-only) recipe, mutate the codegen context, and deposit their final
// products into the output slots below. Phase 1.2 only needs one output
// slot (the rendered compute function); Phase 2/3 will add more (tangent
// source, state-variable wiring, etc.).
struct PassContext {
  ConstitutiveModel const &model;
  CodeGenContext ctx;
  std::optional<std::string> compute_function_source;
};

// Abstract base for a single codegen pass.
//
// Passes advertise their pre/postconditions as string tags. PassManager
// verifies, before running each pass, that every declared precondition has
// been advertised as a postcondition by some earlier pass — catching
// ill-ordered pipelines at run time rather than producing silently-wrong
// output.
//
// Phase 1.2 ships two concrete passes (symbol validation + code emit).
// Future phases register additional passes (time integration, tangent,
// Kuhn-Tucker lowering, …) into the same framework.
class Pass {
public:
  virtual ~Pass() = default;
  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
  [[nodiscard]] virtual auto preconditions() const
      -> std::vector<std::string_view> {
    return {};
  }
  [[nodiscard]] virtual auto postconditions() const
      -> std::vector<std::string_view> {
    return {};
  }
  virtual void run(PassContext &ctx) = 0;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_PASS_H
