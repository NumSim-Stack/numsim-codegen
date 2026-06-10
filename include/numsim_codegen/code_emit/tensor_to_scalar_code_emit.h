#ifndef NUMSIM_CODEGEN_T2S_CODE_EMIT_H
#define NUMSIM_CODEGEN_T2S_CODE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>

#include <numsim_cas/tensor_to_scalar/operators/tensor_to_scalar_add.h>
#include <numsim_cas/tensor_to_scalar/operators/tensor_to_scalar_mul.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_definitions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_visitor_typedef.h>

#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

namespace numsim::codegen {

// Codegen visitor for the tensor-to-scalar domain. Emits Real-typed C++
// expressions backed by tmech reductions.
//
// Phase A MVP scope: trace, det, dot, norm, the t2s arithmetic combinators
// (add, mul, negative, pow), exp/log/sqrt, scalar_wrapper.
class TensorToScalarCodeEmit final
    : public cas::tensor_to_scalar_visitor_const_t {
public:
  TensorToScalarCodeEmit(CodeGenContext &ctx, ScalarCodeEmit &scalar,
                         TensorCodeEmit &tensor)
      : m_ctx(ctx), m_scalar(scalar), m_tensor(tensor) {}

  auto apply(
      cas::expression_holder<cas::tensor_to_scalar_expression> const &expr)
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
    expr.template get<cas::tensor_to_scalar_visitable_t>().accept(*this);
    return m_result;
  }

  // ─── Leaf nodes ──────────────────────────────────────────────────

  void operator()(cas::tensor_to_scalar_zero const &) override {
    m_result = "0.0";
  }

  void operator()(cas::tensor_to_scalar_one const &) override {
    m_result = "1.0";
  }

  void operator()(cas::tensor_to_scalar_scalar_wrapper const &v) override {
    m_result = m_scalar.apply(v.expr());
  }

  // Piecewise selection. For this node cond/then/else are ALL
  // tensor-to-scalar (scalar-VALUED), so it is an ordinary scalar ternary —
  // the condition is also t2s (compared `!= 0.0`), not a separate scalar.
  void operator()(cas::tensor_to_scalar_if_then_else const &v) override {
    auto cond = apply(v.expr_cond());
    auto then_branch = apply(v.expr_then());
    auto else_branch = apply(v.expr_else());
    m_result = register_temp(
        &v, "(" + wrap_if_compound(cond) + " != 0.0 ? " +
                wrap_if_compound(then_branch) + " : " +
                wrap_if_compound(else_branch) + ")");
  }

  // ─── Tensor reductions ───────────────────────────────────────────

  void operator()(cas::tensor_trace const &v) override {
    auto inner = m_tensor.apply(v.expr());
    m_result = register_temp(&v, "tmech::trace(" + inner + ")");
  }

  void operator()(cas::tensor_det const &v) override {
    auto inner = m_tensor.apply(v.expr());
    m_result = register_temp(&v, "tmech::det(" + inner + ")");
  }

  void operator()(cas::tensor_dot const &v) override {
    auto inner = m_tensor.apply(v.expr());
    m_result = register_temp(
        &v, "tmech::dcontract(" + inner + ", " + inner + ")");
  }

  void operator()(cas::tensor_norm const &v) override {
    auto inner = m_tensor.apply(v.expr());
    m_result = register_temp(
        &v,
        "std::sqrt(tmech::dcontract(" + inner + ", " + inner + "))");
  }

  // ─── Algebraic combinators ───────────────────────────────────────

  void operator()(cas::tensor_to_scalar_negative const &v) override {
    auto inner = apply(v.expr());
    if (is_single_token(inner)) {
      m_result = "-" + inner;
    } else {
      m_result = register_temp(&v, "-(" + inner + ")");
    }
  }

  void operator()(cas::tensor_to_scalar_add const &v) override {
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

  void operator()(cas::tensor_to_scalar_mul const &v) override {
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

  void operator()(cas::tensor_to_scalar_pow const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    m_result = register_temp(&v, "std::pow(" + lhs + ", " + rhs + ")");
  }

  void operator()(cas::tensor_to_scalar_log const &v) override {
    m_result = register_temp(&v, "std::log(" + apply(v.expr()) + ")");
  }

  void operator()(cas::tensor_to_scalar_exp const &v) override {
    m_result = register_temp(&v, "std::exp(" + apply(v.expr()) + ")");
  }

  void operator()(cas::tensor_to_scalar_sqrt const &v) override {
    m_result = register_temp(&v, "std::sqrt(" + apply(v.expr()) + ")");
  }

  // ─── Phase A stub ───────────────────────────────────────────────

  void operator()(cas::tensor_inner_product_to_scalar const &) override {
    throw std::runtime_error(
        "TensorToScalarCodeEmit: codegen for tensor_inner_product_to_scalar "
        "not yet implemented. See numsim-codegen Phase A roadmap.");
  }

private:
  auto register_temp(void const *ptr, std::string rhs) -> std::string {
    if (auto *existing = m_ctx.find(ptr)) {
      return *existing;
    }
    return m_ctx.emit_temporary(ptr, std::move(rhs));
  }

  static auto is_single_token(std::string const &s) -> bool {
    return s.find(' ') == std::string::npos &&
           s.find('(') == std::string::npos;
  }

  static auto wrap_if_compound(std::string const &s) -> std::string {
    return is_single_token(s) ? s : "(" + s + ")";
  }

  CodeGenContext &m_ctx;
  ScalarCodeEmit &m_scalar;
  TensorCodeEmit &m_tensor;
  std::string m_result;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_T2S_CODE_EMIT_H
