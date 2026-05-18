#ifndef NUMSIM_CODEGEN_CONTEXT_H
#define NUMSIM_CODEGEN_CONTEXT_H

#include <numsim_cas/scalar/scalar_expression.h>
#include <numsim_cas/tensor/tensor_expression.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_expression.h>
#include <numsim_cas/core/expression_holder.h>

#include <cassert>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace numsim::codegen {

// Shared state for the three per-domain code emitters. Tracks the CSE table
// (pointer-identity-based, since numsim-cas's simplifier already dedupes
// structurally-equal subexpressions via hash so they share the same node),
// the list of generated statements in topological order, and the symbol
// table mapping declared inputs / parameters to their C++ names.
class CodeGenContext {
public:
  struct Statement {
    std::string lhs;       // e.g. "t5"
    std::string rhs;       // e.g. "lam * t3"
    std::string type;      // e.g. "double" or "tmech::tensor<double, 3, 2>"
  };

  // Register a named external symbol (input, parameter, history variable).
  // The codegen will use this name verbatim when it encounters the symbol's
  // expression node, rather than allocating a temporary.
  void register_symbol_scalar(
      cas::expression_holder<cas::scalar_expression> const &expr,
      std::string const &name) {
    assert(expr.is_valid());
    m_named_symbols[&expr.get()] = name;
  }

  void register_symbol_tensor(
      cas::expression_holder<cas::tensor_expression> const &expr,
      std::string const &name) {
    assert(expr.is_valid());
    m_named_symbols[&expr.get()] = name;
  }

  // CSE table operations.
  [[nodiscard]] auto find(void const *ptr) const -> std::string const * {
    if (auto it = m_cse_table.find(ptr); it != m_cse_table.end()) {
      return &it->second;
    }
    return nullptr;
  }

  [[nodiscard]] auto find_named(void const *ptr) const -> std::string const * {
    if (auto it = m_named_symbols.find(ptr); it != m_named_symbols.end()) {
      return &it->second;
    }
    return nullptr;
  }

  // Allocate a fresh temporary name and emit an `auto tN = rhs;` statement.
  // The void* is the pointer-identity of the expression node, used for
  // future CSE lookups.
  auto emit_temporary(void const *ptr, std::string rhs, std::string type)
      -> std::string {
    auto name = fresh_name();
    m_statements.push_back({name, std::move(rhs), std::move(type)});
    m_cse_table[ptr] = name;
    return name;
  }

  // Render the accumulated statements as a sequence of `auto tN = ...;` lines.
  [[nodiscard]] auto render_statements(std::string const &indent = "  ") const
      -> std::string {
    std::ostringstream os;
    for (auto const &s : m_statements) {
      os << indent << "auto " << s.lhs << " = " << s.rhs << ";\n";
    }
    return os.str();
  }

  [[nodiscard]] auto statements() const -> std::vector<Statement> const & {
    return m_statements;
  }

  [[nodiscard]] auto temporary_count() const -> std::size_t {
    return m_statements.size();
  }

  void reset() {
    m_statements.clear();
    m_cse_table.clear();
    m_counter = 0;
    // Note: deliberately keep m_named_symbols across resets so the user can
    // emit multiple functions sharing the same input declarations.
  }

private:
  auto fresh_name() -> std::string {
    return "t" + std::to_string(m_counter++);
  }

  std::vector<Statement> m_statements;
  std::unordered_map<void const *, std::string> m_cse_table;
  std::unordered_map<void const *, std::string> m_named_symbols;
  int m_counter = 0;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_CONTEXT_H
