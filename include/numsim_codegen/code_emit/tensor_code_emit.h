#ifndef NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
#define NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>

#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/levi_civita_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_visitor_typedef.h>
#include <numsim_cas/tensor/wrappers/tensor_inv.h>

#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

namespace numsim::codegen {

// Codegen visitor for the tensor domain. Emits tmech-typed C++ expressions.
//
// Scope (Phase A MVP): covers the nodes required for linear elasticity.
// All other node types throw at codegen time. Subsequent issues fill in
// inner_product, basis_change, outer_product, tensor_inv, etc.
class TensorCodeEmit final : public cas::tensor_visitor_const_t {
public:
  TensorCodeEmit(CodeGenContext &ctx, ScalarCodeEmit &scalar)
      : m_ctx(ctx), m_scalar(scalar) {}

  auto apply(cas::expression_holder<cas::tensor_expression> const &expr)
      -> std::string {
    if (!expr.is_valid()) {
      throw std::runtime_error(
          "TensorCodeEmit::apply called on invalid expression");
    }
    void const *ptr = &expr.get();
    if (auto *name = m_ctx.find_named(ptr)) {
      return *name;
    }
    if (auto *name = m_ctx.find(ptr)) {
      return *name;
    }
    expr.template get<cas::tensor_visitable_t>().accept(*this);
    return m_result;
  }

  // ─── Leaf nodes ──────────────────────────────────────────────────

  void operator()(cas::tensor const &v) override {
    if (auto *name = m_ctx.find_named(&v)) {
      m_result = *name;
    } else {
      m_result = v.name();
    }
  }

  void operator()(cas::tensor_zero const &v) override {
    std::ostringstream os;
    os << "tmech::tensor<double, " << v.dim() << ", " << v.rank() << ">{}";
    m_result = register_temp(&v, os.str());
  }

  void operator()(cas::identity_tensor const &v) override {
    if (v.rank() == 2) {
      std::ostringstream os;
      os << "tmech::eye<double, " << v.dim() << ", 2>()";
      m_result = register_temp(&v, os.str());
      return;
    }
    if (v.rank() == 4) {
      std::ostringstream os;
      os << "tmech::otimesu(tmech::eye<double, " << v.dim()
         << ", 2>(), tmech::eye<double, " << v.dim() << ", 2>())";
      m_result = register_temp(&v, os.str());
      return;
    }
    throw std::runtime_error(
        "TensorCodeEmit: identity_tensor of rank " +
        std::to_string(v.rank()) + " not supported in Phase A MVP");
  }

  void operator()(cas::levi_civita_tensor const &v) override {
    // tmech::levi_civita<T, Dim> is a leaf; rank == dim. Supported for
    // dim ∈ {2, 3, 4}, matching numsim-cas's construction-time validation.
    std::ostringstream os;
    os << "tmech::levi_civita<double, " << v.dim() << ">{}";
    m_result = register_temp(&v, os.str());
  }

  // ─── Algebraic nodes ─────────────────────────────────────────────

  void operator()(cas::tensor_negative const &v) override {
    auto inner = apply(v.expr());
    if (is_single_token(inner)) {
      m_result = "-" + inner;
    } else {
      m_result = register_temp(&v, "-(" + inner + ")");
    }
  }

  void operator()(cas::tensor_add const &v) override {
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
      // Empty tensor_add — emit a zero tensor of matching dim/rank.
      // The simplifier normally collapses this, but a defensive fallback
      // is cheap and avoids the `auto t0 = ;` trap if a future rewrite
      // produces an empty add.
      std::ostringstream zero;
      zero << "tmech::tensor<double, " << v.dim() << ", " << v.rank() << ">{}";
      m_result = register_temp(&v, zero.str());
    } else {
      m_result = register_temp(&v, os.str());
    }
  }

  void operator()(cas::tensor_scalar_mul const &v) override {
    auto lhs = m_scalar.apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    m_result = register_temp(&v, wrap_if_compound(lhs) + " * " +
                                     wrap_if_compound(rhs));
  }

  // ─── Phase A stubs — node types not yet implemented ─────────────

#define NUMSIM_CODEGEN_TENSOR_STUB(T)                                          \
  void operator()(cas::T const &) override {                                   \
    throw std::runtime_error(                                                  \
        "TensorCodeEmit: codegen for " #T                                      \
        " not yet implemented. See numsim-codegen Phase A roadmap.");          \
  }

  NUMSIM_CODEGEN_TENSOR_STUB(tensor_mul)
  NUMSIM_CODEGEN_TENSOR_STUB(tensor_pow)
  NUMSIM_CODEGEN_TENSOR_STUB(inner_product_wrapper)
  NUMSIM_CODEGEN_TENSOR_STUB(permute_indices_wrapper)
  NUMSIM_CODEGEN_TENSOR_STUB(outer_product_wrapper)
  NUMSIM_CODEGEN_TENSOR_STUB(simple_outer_product)
  NUMSIM_CODEGEN_TENSOR_STUB(tensor_projector)
  NUMSIM_CODEGEN_TENSOR_STUB(tensor_to_scalar_with_tensor_mul)

#undef NUMSIM_CODEGEN_TENSOR_STUB

  // ─── Implemented unary nodes ─────────────────────────────────────

  // tensor_inv → tmech::inv(...). Rank-2 only in this phase.
  //
  // For rank ≠ 2 the inverse is well-defined algebraically but tmech's
  // `inv` needs explicit contraction-index sequences (e.g.
  // `tmech::inv<tmech::sequence<1,2>, tmech::sequence<3,4>>(A)` for the
  // natural rank-4 inverse). That index pair is a property of the
  // tensor's algebraic structure and is being added to `cas::tensor_inv`
  // upstream — see NumSim-Stack/numsim-cas#248 (decision: Option A,
  // landing in a future numsim-cas release). Once the accessor is
  // available, this emit case reads it and produces the templated
  // `tmech::inv<sequence<...>, sequence<...>>(...)` form; the rank-2
  // path below is unchanged. Local follow-up: #43.
  void operator()(cas::tensor_inv const &v) override {
    if (v.rank() != 2) {
      throw std::runtime_error(
          "TensorCodeEmit: tensor_inv of rank " + std::to_string(v.rank()) +
          " is not yet supported. Rank-2 ships now; rank-4 emit lands once "
          "numsim-cas exposes the inv-contraction pair on the tensor_inv "
          "node (NumSim-Stack/numsim-cas#248).");
    }
    auto inner = apply(v.expr());
    m_result = register_temp(&v, "tmech::inv(" + inner + ")");
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
  std::string m_result;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
