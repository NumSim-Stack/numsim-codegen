#ifndef NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H
#define NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H

#include <numsim_codegen/passes/pass.h>
// NOTE: include order — this header expects `recipe.h` to be included
// elsewhere in the TU before `run()` is instantiated. Since recipe.h is
// the only place that constructs this pass (in its emit pipeline), that
// ordering is automatic.

#include <cctype>
#include <stdexcept>
#include <string>

namespace numsim::codegen {

// Phase 1.2 SymbolValidationPass.
//
// Two checks:
//   1. Every symbol referenced by an output expression is declared as an
//      input, parameter, or other symbol. (Pre-1.2 this was the body of
//      ConstitutiveModel::validate(); the pass now owns it.)
//   2. Every declared symbol name is a valid C++ identifier — catches the
//      case where a user passes "my-tensor" or "x.y" and the generated
//      code wouldn't compile. Previously deferred (see the cross-ref in
//      moose_material.cpp:63 to #14 → CORR-B4); landed here.
//
// Throws std::runtime_error on the first violation found.
class SymbolValidationPass final : public Pass {
public:
  [[nodiscard]] auto name() const -> std::string_view override {
    return "SymbolValidation";
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return {"symbols-declared", "identifiers-valid"};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.

  // Static check exposed so other passes / direct callers can reuse it.
  static auto is_valid_cxx_identifier(std::string const &s) -> bool {
    if (s.empty()) {
      return false;
    }
    if (!(std::isalpha(static_cast<unsigned char>(s.front())) ||
          s.front() == '_')) {
      return false;
    }
    for (char c : s) {
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
        return false;
      }
    }
    return true;
  }
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H
