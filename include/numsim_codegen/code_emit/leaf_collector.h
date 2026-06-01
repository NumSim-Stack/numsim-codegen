#ifndef NUMSIM_CODEGEN_LEAF_COLLECTOR_H
#define NUMSIM_CODEGEN_LEAF_COLLECTOR_H

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_visitor_typedef.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_visitor_typedef.h>
#include <numsim_cas/tensor_to_scalar/operators/tensor_to_scalar_add.h>
#include <numsim_cas/tensor_to_scalar/operators/tensor_to_scalar_mul.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_definitions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_visitor_typedef.h>

#include <ranges>
#include <set>
#include <string>

namespace numsim::codegen {

// Walks an expression DAG and collects the names of all leaf symbol nodes
// (`scalar(name)` and `tensor(name)`). Used by ConstitutiveModel::validate()
// to detect outputs that reference symbols the recipe never declared.
//
// Three sibling visitors share a single output set; cross-domain expressions
// (e.g. tensor containing `trace(eps)` t2s subterm) route through all three
// as needed.
class LeafCollector {
public:
  void collect_scalar(
      cas::expression_holder<cas::scalar_expression> const &expr) {
    if (!expr.is_valid() || !m_visited_scalar.insert(&expr.get()).second) {
      return;
    }
    expr.template get<cas::scalar_visitable_t>().accept(m_scalar_v);
  }

  void collect_tensor(
      cas::expression_holder<cas::tensor_expression> const &expr) {
    if (!expr.is_valid() || !m_visited_tensor.insert(&expr.get()).second) {
      return;
    }
    expr.template get<cas::tensor_visitable_t>().accept(m_tensor_v);
  }

  void collect_t2s(
      cas::expression_holder<cas::tensor_to_scalar_expression> const &expr) {
    if (!expr.is_valid() || !m_visited_t2s.insert(&expr.get()).second) {
      return;
    }
    expr.template get<cas::tensor_to_scalar_visitable_t>().accept(m_t2s_v);
  }

  [[nodiscard]] auto scalar_names() const -> std::set<std::string> const & {
    return m_scalar_names;
  }

  [[nodiscard]] auto tensor_names() const -> std::set<std::string> const & {
    return m_tensor_names;
  }

private:
  class ScalarV final : public cas::scalar_visitor_const_t {
  public:
    explicit ScalarV(LeafCollector &p) : m_p(p) {}
    void operator()(cas::scalar const &v) override {
      m_p.m_scalar_names.insert(v.name());
    }
    void operator()(cas::scalar_zero const &) override {}
    void operator()(cas::scalar_one const &) override {}
    void operator()(cas::scalar_constant const &) override {}
    void operator()(cas::scalar_add const &v) override {
      if (v.coeff().is_valid()) m_p.collect_scalar(v.coeff());
      for (auto const &c : v.symbol_map() | std::views::values)
        m_p.collect_scalar(c);
    }
    void operator()(cas::scalar_mul const &v) override {
      if (v.coeff().is_valid()) m_p.collect_scalar(v.coeff());
      for (auto const &c : v.symbol_map() | std::views::values)
        m_p.collect_scalar(c);
    }
    void operator()(cas::scalar_pow const &v) override {
      m_p.collect_scalar(v.expr_lhs());
      m_p.collect_scalar(v.expr_rhs());
    }

#define NCG_BINARY_S(T)                                                        \
  void operator()(cas::T const &v) override {                                  \
    m_p.collect_scalar(v.expr_lhs());                                          \
    m_p.collect_scalar(v.expr_rhs());                                          \
  }
    NCG_BINARY_S(scalar_lt)
    NCG_BINARY_S(scalar_gt)
    NCG_BINARY_S(scalar_le)
    NCG_BINARY_S(scalar_ge)
    NCG_BINARY_S(scalar_eq)
    NCG_BINARY_S(scalar_ne)
    NCG_BINARY_S(scalar_max)
    NCG_BINARY_S(scalar_min)
#undef NCG_BINARY_S

    void operator()(cas::scalar_if_then_else const &v) override {
      m_p.collect_scalar(v.expr_cond());
      m_p.collect_scalar(v.expr_then());
      m_p.collect_scalar(v.expr_else());
    }

#define NCG_UNARY_S(T)                                                         \
  void operator()(cas::T const &v) override { m_p.collect_scalar(v.expr()); }
    NCG_UNARY_S(scalar_negative)
    NCG_UNARY_S(scalar_sin)
    NCG_UNARY_S(scalar_cos)
    NCG_UNARY_S(scalar_tan)
    NCG_UNARY_S(scalar_asin)
    NCG_UNARY_S(scalar_acos)
    NCG_UNARY_S(scalar_atan)
    NCG_UNARY_S(scalar_sqrt)
    NCG_UNARY_S(scalar_log)
    NCG_UNARY_S(scalar_exp)
    NCG_UNARY_S(scalar_sign)
    NCG_UNARY_S(scalar_abs)
    NCG_UNARY_S(scalar_named_expression)
#undef NCG_UNARY_S

  private:
    LeafCollector &m_p;
  };

  class TensorV final : public cas::tensor_visitor_const_t {
  public:
    explicit TensorV(LeafCollector &p) : m_p(p) {}
    void operator()(cas::tensor const &v) override {
      m_p.m_tensor_names.insert(v.name());
    }
    void operator()(cas::tensor_zero const &) override {}
    void operator()(cas::identity_tensor const &) override {}
    void operator()(cas::levi_civita_tensor const &) override {}
    void operator()(cas::tensor_projector const &) override {}
    void operator()(cas::tensor_negative const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_add const &v) override {
      if (v.coeff().is_valid()) m_p.collect_tensor(v.coeff());
      for (auto const &c : v.symbol_map() | std::views::values)
        m_p.collect_tensor(c);
    }
    void operator()(cas::tensor_scalar_mul const &v) override {
      m_p.collect_scalar(v.expr_lhs());
      m_p.collect_tensor(v.expr_rhs());
    }
    void operator()(cas::tensor_mul const &) override {}
    void operator()(cas::tensor_pow const &v) override {
      m_p.collect_tensor(v.expr_lhs());
      m_p.collect_scalar(v.expr_rhs());
    }
    void operator()(cas::inner_product_wrapper const &v) override {
      m_p.collect_tensor(v.expr_lhs());
      m_p.collect_tensor(v.expr_rhs());
    }
    void operator()(cas::permute_indices_wrapper const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::outer_product_wrapper const &v) override {
      m_p.collect_tensor(v.expr_lhs());
      m_p.collect_tensor(v.expr_rhs());
    }
    void operator()(cas::simple_outer_product const &) override {}
    void operator()(cas::tensor_inv const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_to_scalar_with_tensor_mul const &v) override {
      m_p.collect_tensor(v.expr_lhs());
      m_p.collect_t2s(v.expr_rhs());
    }

  private:
    LeafCollector &m_p;
  };

  class T2sV final : public cas::tensor_to_scalar_visitor_const_t {
  public:
    explicit T2sV(LeafCollector &p) : m_p(p) {}
    void operator()(cas::tensor_to_scalar_zero const &) override {}
    void operator()(cas::tensor_to_scalar_one const &) override {}
    void operator()(cas::tensor_to_scalar_scalar_wrapper const &v) override {
      m_p.collect_scalar(v.expr());
    }
    void operator()(cas::tensor_trace const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_det const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_dot const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_norm const &v) override {
      m_p.collect_tensor(v.expr());
    }
    void operator()(cas::tensor_to_scalar_negative const &v) override {
      m_p.collect_t2s(v.expr());
    }
    void operator()(cas::tensor_to_scalar_add const &v) override {
      if (v.coeff().is_valid()) m_p.collect_t2s(v.coeff());
      for (auto const &c : v.symbol_map() | std::views::values)
        m_p.collect_t2s(c);
    }
    void operator()(cas::tensor_to_scalar_mul const &v) override {
      if (v.coeff().is_valid()) m_p.collect_t2s(v.coeff());
      for (auto const &c : v.symbol_map() | std::views::values)
        m_p.collect_t2s(c);
    }
    void operator()(cas::tensor_to_scalar_pow const &v) override {
      m_p.collect_t2s(v.expr_lhs());
      m_p.collect_t2s(v.expr_rhs());
    }
    void operator()(cas::tensor_to_scalar_log const &v) override {
      m_p.collect_t2s(v.expr());
    }
    void operator()(cas::tensor_to_scalar_exp const &v) override {
      m_p.collect_t2s(v.expr());
    }
    void operator()(cas::tensor_to_scalar_sqrt const &v) override {
      m_p.collect_t2s(v.expr());
    }
    void operator()(cas::tensor_inner_product_to_scalar const &v) override {
      m_p.collect_tensor(v.expr_lhs());
      m_p.collect_tensor(v.expr_rhs());
    }

  private:
    LeafCollector &m_p;
  };

  std::set<std::string> m_scalar_names;
  std::set<std::string> m_tensor_names;
  std::set<void const *> m_visited_scalar;
  std::set<void const *> m_visited_tensor;
  std::set<void const *> m_visited_t2s;
  ScalarV m_scalar_v{*this};
  TensorV m_tensor_v{*this};
  T2sV m_t2s_v{*this};
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_LEAF_COLLECTOR_H
