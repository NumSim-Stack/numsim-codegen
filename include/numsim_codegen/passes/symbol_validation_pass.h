#ifndef NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H
#define NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>
// NOTE: include order — `run()` is defined in recipe.h after
// ConstitutiveModel is complete. Any TU instantiating this pass must
// include recipe.h; in practice recipe.h is the only constructor site.

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

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
    return {pass_tags::symbols_declared, pass_tags::identifiers_valid};
  }
  void run(PassContext &pctx) override; // defined in recipe.h after class.

  // Static checks exposed so other passes / direct callers can reuse them.
  //
  // The char-class checks deliberately use explicit ASCII ranges rather
  // than `std::isalpha` / `std::isalnum` — the latter are locale-sensitive
  // (e.g. Turkish locale fails `'I'`; locales that treat accented chars as
  // alpha would accept `"épsilon"`). Generated C++ identifiers must be
  // portable across compilation hosts.
  static constexpr auto is_ascii_alpha(char c) noexcept -> bool {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
  }
  static constexpr auto is_ascii_alnum(char c) noexcept -> bool {
    return is_ascii_alpha(c) || (c >= '0' && c <= '9');
  }

  // Reject reserved C++ keywords. The literal regex
  // `[A-Za-z_][A-Za-z0-9_]*` accepts `class`, `if`, `auto`, … which the
  // compiler would then refuse to parse as identifiers — far worse
  // diagnostic than catching it at validate() time.
  static auto is_cxx_keyword(std::string_view s) noexcept -> bool {
    // C++23 keywords + TM TS atomic_* + withdrawn reflexpr (cheap
    // defense). MUST stay sorted for std::binary_search. C array form so
    // the compiler counts entries — no off-by-one on edits.
    static constexpr std::string_view kKeywords[] = {
        "alignas",        "alignof",         "and",
        "and_eq",         "asm",             "atomic_cancel",
        "atomic_commit",  "atomic_noexcept", "auto",
        "bitand",         "bitor",           "bool",
        "break",          "case",            "catch",
        "char",           "char16_t",        "char32_t",
        "char8_t",        "class",           "co_await",
        "co_return",      "co_yield",        "compl",
        "concept",        "const",           "const_cast",
        "consteval",      "constexpr",       "constinit",
        "continue",       "decltype",        "default",
        "delete",         "do",              "double",
        "dynamic_cast",   "else",            "enum",
        "explicit",       "export",          "extern",
        "false",          "float",           "for",
        "friend",         "goto",            "if",
        "inline",         "int",             "long",
        "mutable",        "namespace",       "new",
        "noexcept",       "not",             "not_eq",
        "nullptr",        "operator",        "or",
        "or_eq",          "private",         "protected",
        "public",         "reflexpr",        "register",
        "reinterpret_cast","requires",       "return",
        "short",          "signed",          "sizeof",
        "static",         "static_assert",   "static_cast",
        "struct",         "switch",          "synchronized",
        "template",       "this",            "thread_local",
        "throw",          "true",            "try",
        "typedef",        "typeid",          "typename",
        "union",          "unsigned",        "using",
        "virtual",        "void",            "volatile",
        "wchar_t",        "while",           "xor",
        "xor_eq"};
    return std::binary_search(std::begin(kKeywords), std::end(kKeywords), s);
  }

  static auto is_valid_cxx_identifier(std::string const &s) noexcept -> bool {
    if (s.empty()) {
      return false;
    }
    if (!is_ascii_alpha(s.front()) && s.front() != '_') {
      return false;
    }
    for (char c : s) {
      if (!is_ascii_alnum(c) && c != '_') {
        return false;
      }
    }
    return !is_cxx_keyword(s);
  }
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_SYMBOL_VALIDATION_PASS_H
