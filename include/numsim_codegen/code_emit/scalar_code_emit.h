#ifndef NUMSIM_CODEGEN_SCALAR_CODE_EMIT_H
#define NUMSIM_CODEGEN_SCALAR_CODE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_visitor_typedef.h>

#include <algorithm>
#include <complex>
#include <ranges>
#include <sstream>
#include <string>
#include <variant>

namespace numsim::codegen {

// Codegen visitor for the scalar domain. Walks an expression tree and
// returns a C++ expression string (variable name or literal) for the
// visited node. Side-effect: registers new `auto tN = ...;` statements in
// the shared CodeGenContext for non-trivial subexpressions, with pointer-
// based CSE.
//
// Uses the void-returning visitor pattern (write to m_result, read after
// accept) because numsim-cas's visitor_return template is fixed to return
// expression_holder<Base>, not arbitrary types.
class ScalarCodeEmit final : public cas::scalar_visitor_const_t {
public:
  explicit ScalarCodeEmit(CodeGenContext &ctx) : m_ctx(ctx) {}

  // Entry point. Returns the C++ expression that evaluates to the value of
  // the input expression. Reuses temporaries from the CSE table when the
  // same DAG node has been emitted already.
  auto apply(cas::expression_holder<cas::scalar_expression> const &expr)
      -> std::string {
    if (!expr.is_valid()) {
      return "0.0";
    }
    void const *ptr = &expr.get();
    if (auto *name = m_ctx.find_named(ptr)) {
      return *name;
    }
    if (auto *name = m_ctx.find(ptr)) {
      return *name;
    }
    expr.template get<cas::scalar_visitable_t>().accept(*this);
    return m_result;
  }

  // ─── Leaf nodes ──────────────────────────────────────────────────

  void operator()(cas::scalar const &v) override {
    if (auto *name = m_ctx.find_named(&v)) {
      m_result = *name;
    } else {
      m_result = v.name();
    }
  }

  void operator()(cas::scalar_zero const &) override { m_result = "0.0"; }
  void operator()(cas::scalar_one const &) override { m_result = "1.0"; }

  void operator()(cas::scalar_constant const &v) override {
    auto [literal, is_compound] = std::visit(
        [](auto const &val) -> std::pair<std::string, bool> {
          using V = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<V, std::complex<double>>) {
            std::ostringstream os;
            os << "std::complex<double>(" << val.real() << ", "
               << val.imag() << ")";
            return {os.str(), true};
          } else if constexpr (std::is_same_v<V, cas::rational_t>) {
            std::ostringstream os;
            os << "(" << val.num << ".0 / " << val.den << ".0)";
            return {os.str(), true};
          } else {
            std::ostringstream os;
            os << static_cast<double>(val);
            auto s = os.str();
            if (s.find('.') == std::string::npos &&
                s.find('e') == std::string::npos &&
                s.find('E') == std::string::npos) {
              s += ".0";
            }
            return {s, false};
          }
        },
        v.value().raw());

    // Trivial literals (plain doubles) stay inline — they read naturally
    // and the compiler folds duplicates. Compound forms (rational, complex)
    // benefit from a temporary: if the same constant appears twice, we
    // emit the literal once and reference the temp.
    if (is_compound) {
      m_result = register_temp(&v, literal);
    } else {
      m_result = literal;
    }
  }

  // ─── Unary / function nodes ──────────────────────────────────────

  void operator()(cas::scalar_negative const &v) override {
    auto inner = apply(v.expr());
    // If the inner result is a single token (named leaf, temp, or literal)
    // emit `-x` inline without allocating a temporary. If it has structure,
    // wrap and register as a temp.
    if (is_single_token(inner)) {
      m_result = "-" + inner;
    } else {
      m_result = register_temp(&v, "-(" + inner + ")");
    }
  }

  void operator()(cas::scalar_sign const &v) override {
    m_result = register_temp(&v, "std::copysign(1.0, " + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_abs const &v) override {
    m_result = register_temp(&v, "std::abs(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_sqrt const &v) override {
    m_result = register_temp(&v, "std::sqrt(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_log const &v) override {
    m_result = register_temp(&v, "std::log(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_exp const &v) override {
    m_result = register_temp(&v, "std::exp(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_sin const &v) override {
    m_result = register_temp(&v, "std::sin(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_cos const &v) override {
    m_result = register_temp(&v, "std::cos(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_tan const &v) override {
    m_result = register_temp(&v, "std::tan(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_asin const &v) override {
    m_result = register_temp(&v, "std::asin(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_acos const &v) override {
    m_result = register_temp(&v, "std::acos(" + apply(v.expr()) + ")");
  }

  void operator()(cas::scalar_atan const &v) override {
    m_result = register_temp(&v, "std::atan(" + apply(v.expr()) + ")");
  }

  // ─── Binary nodes ────────────────────────────────────────────────

  void operator()(cas::scalar_pow const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    m_result = register_temp(&v, "std::pow(" + lhs + ", " + rhs + ")");
  }

  // ─── N-ary nodes ─────────────────────────────────────────────────

  void operator()(cas::scalar_add const &v) override {
    std::ostringstream os;
    bool first = true;
    if (v.coeff().is_valid()) {
      os << wrap_if_compound(apply(v.coeff()));
      first = false;
    }
    for (auto const &child : v.symbol_map() | std::views::values) {
      auto term = apply(child);
      if (!first) {
        os << " + ";
      }
      os << wrap_if_compound(term);
      first = false;
    }
    if (first) {
      m_result = "0.0";
    } else {
      m_result = register_temp(&v, os.str());
    }
  }

  void operator()(cas::scalar_mul const &v) override {
    std::ostringstream os;
    bool first = true;
    if (v.coeff().is_valid()) {
      os << wrap_if_compound(apply(v.coeff()));
      first = false;
    }
    for (auto const &child : v.symbol_map() | std::views::values) {
      auto term = apply(child);
      if (!first) {
        os << " * ";
      }
      os << wrap_if_compound(term);
      first = false;
    }
    if (first) {
      m_result = "1.0";
    } else {
      m_result = register_temp(&v, os.str());
    }
  }

  void operator()(cas::scalar_named_expression const &v) override {
    // Inline the body — named-expression is a documentation wrapper.
    m_result = apply(v.expr());
  }

  // ─── Comparisons — emit C++ operator, cast to indicator double ───
  //
  // numsim-cas comparison nodes evaluate to 1.0 / 0.0 (per scalar_ne.h);
  // emit `static_cast<double>(lhs OP rhs)` to preserve that semantics.

#define NUMSIM_CODEGEN_SCALAR_CMP(T, OP)                                       \
  void operator()(cas::T const &v) override {                                  \
    auto lhs = apply(v.expr_lhs());                                            \
    auto rhs = apply(v.expr_rhs());                                            \
    m_result = register_temp(&v, "static_cast<double>(" +                      \
                                     wrap_if_compound(lhs) + " " #OP " " +     \
                                     wrap_if_compound(rhs) + ")");             \
  }

  NUMSIM_CODEGEN_SCALAR_CMP(scalar_lt, <)
  NUMSIM_CODEGEN_SCALAR_CMP(scalar_gt, >)
  NUMSIM_CODEGEN_SCALAR_CMP(scalar_le, <=)
  NUMSIM_CODEGEN_SCALAR_CMP(scalar_ge, >=)
  NUMSIM_CODEGEN_SCALAR_CMP(scalar_eq, ==)
  NUMSIM_CODEGEN_SCALAR_CMP(scalar_ne, !=)

#undef NUMSIM_CODEGEN_SCALAR_CMP

  // ─── Piecewise nodes ─────────────────────────────────────────────

  void operator()(cas::scalar_max const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    m_result = register_temp(&v, "std::max(" + lhs + ", " + rhs + ")");
  }

  void operator()(cas::scalar_min const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    m_result = register_temp(&v, "std::min(" + lhs + ", " + rhs + ")");
  }

  void operator()(cas::scalar_if_then_else const &v) override {
    auto cond = apply(v.expr_cond());
    auto then_branch = apply(v.expr_then());
    auto else_branch = apply(v.expr_else());
    m_result = register_temp(
        &v, "(" + wrap_if_compound(cond) + " != 0.0 ? " +
                wrap_if_compound(then_branch) + " : " +
                wrap_if_compound(else_branch) + ")");
  }

private:
  auto register_temp(void const *ptr, std::string rhs) -> std::string {
    if (auto *existing = m_ctx.find(ptr)) {
      return *existing;
    }
    return m_ctx.emit_temporary(ptr, std::move(rhs));
  }

  // A "single token" is a named leaf (no spaces, no parens) — safe to use
  // unparenthesised as a sub-expression. Anything else has internal
  // structure and needs parens to maintain precedence.
  static auto is_single_token(std::string const &s) -> bool {
    return s.find(' ') == std::string::npos &&
           s.find('(') == std::string::npos;
  }

  // Wrap a sub-expression in parens iff it has internal operators. Keeps
  // generated code readable: `t1 + t2 + t3` rather than `(t1) + (t2) + (t3)`.
  static auto wrap_if_compound(std::string const &s) -> std::string {
    return is_single_token(s) ? s : "(" + s + ")";
  }

  CodeGenContext &m_ctx;
  std::string m_result;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_SCALAR_CODE_EMIT_H
