#ifndef NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
#define NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>

#include <numsim_cas/basic_functions.h>
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
  NUMSIM_CODEGEN_TENSOR_STUB(permute_indices_wrapper)
  NUMSIM_CODEGEN_TENSOR_STUB(tensor_to_scalar_with_tensor_mul)

#undef NUMSIM_CODEGEN_TENSOR_STUB

  // ─── Implemented unary nodes ─────────────────────────────────────

  // tensor_projector → rank-4 constant projector built from tmech primitives.
  //
  // Scope: `acts_on_rank == 2` (rank-4 projectors on rank-2 tensors). This is
  // the case used by every Phase-A elasticity/plasticity recipe and matches
  // the only presets currently exposed by numsim-cas (P_sym/P_skew/P_vol/
  // P_dev/P_harm).
  //
  // We materialise the projector as a constant tensor rather than emitting
  // `tmech::sym(A)` etc. Reason: the cas IR can hand us a bare
  // `tensor_projector` not adjacent to an inner_product (e.g. as the LHS of
  // an outer product, or as the argument to a differentiation). The
  // applied-form short-circuit (`P_sym : A → sym(A)`) belongs in the
  // `inner_product_wrapper` emit when that lands — same place numsim-cas's
  // tensor_evaluator does it (tensor_evaluator.h:110).
  //
  // Component formulas (δ is the Kronecker delta, d = dim):
  //   P_sym_{ijkl}  = ½(δ_ik δ_jl + δ_il δ_jk)  = ½(otimesu(I,I) + otimesl(I,I))
  //   P_skew_{ijkl} = ½(δ_ik δ_jl − δ_il δ_jk)  = ½(otimesu(I,I) − otimesl(I,I))
  //   P_vol_{ijkl}  = (1/d) δ_ij δ_kl           = (1/d) otimes(I,I)
  //   P_dev_{ijkl}  = P_sym − P_vol
  void operator()(cas::tensor_projector const &v) override {
    const auto r = v.acts_on_rank();
    if (r != 2) {
      throw std::runtime_error(
          "TensorCodeEmit: tensor_projector with acts_on_rank=" +
          std::to_string(r) +
          " is not yet supported. Only acts_on_rank=2 (rank-4 projectors on "
          "rank-2 tensors) ships now; higher-rank projector emit lands once "
          "we have a recipe that needs it.");
    }
    const auto d_str = std::to_string(v.dim());
    const std::string eye = "tmech::eye<double, " + d_str + ", 2>()";
    const std::string ou = "tmech::otimesu(" + eye + ", " + eye + ")";
    const std::string ol = "tmech::otimesl(" + eye + ", " + eye + ")";
    const std::string oo = "tmech::otimes(" + eye + ", " + eye + ")";
    // (1.0 / d) keeps the literal a double; integer d in the denominator
    // would silently floor for non-trivial d.
    const std::string inv_d = "(1.0 / " + d_str + ".0)";

    auto const &sp = v.space();
    auto const &perm = sp.perm;
    auto const &trace = sp.trace;

    if (std::holds_alternative<cas::Symmetric>(perm) &&
        std::holds_alternative<cas::AnyTraceTag>(trace)) {
      m_result = register_temp(&v, "0.5 * (" + ou + " + " + ol + ")");
      return;
    }
    if (std::holds_alternative<cas::Skew>(perm) &&
        std::holds_alternative<cas::AnyTraceTag>(trace)) {
      m_result = register_temp(&v, "0.5 * (" + ou + " - " + ol + ")");
      return;
    }
    if (std::holds_alternative<cas::Symmetric>(perm) &&
        std::holds_alternative<cas::VolumetricTag>(trace)) {
      m_result = register_temp(&v, inv_d + " * " + oo);
      return;
    }
    if (std::holds_alternative<cas::Symmetric>(perm) &&
        std::holds_alternative<cas::DeviatoricTag>(trace)) {
      m_result = register_temp(
          &v, "0.5 * (" + ou + " + " + ol + ") - " + inv_d + " * " + oo);
      return;
    }
    throw std::runtime_error(
        "TensorCodeEmit: tensor_projector with this (perm, trace) "
        "combination is not yet supported. Supported presets at "
        "acts_on_rank=2: P_sym {Symmetric, AnyTrace}, "
        "P_skew {Skew, AnyTrace}, P_vol {Symmetric, Volumetric}, "
        "P_dev {Symmetric, Deviatoric}.");
  }

  // outer_product_wrapper → tmech::outer_product<sequence<...>, sequence<...>>.
  //
  // numsim-cas stores contraction/placement sequences 0-based; tmech's
  // template sequence is 1-based, hence the +1 in `tmech_sequence_str`.
  void operator()(cas::outer_product_wrapper const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    std::ostringstream os;
    os << "tmech::outer_product<" << tmech_sequence_str(v.indices_lhs()) << ", "
       << tmech_sequence_str(v.indices_rhs()) << ">(" << lhs << ", " << rhs
       << ")";
    m_result = register_temp(&v, os.str());
  }

  // simple_outer_product → chained tmech::outer_product calls.
  //
  // The cas node is n-ary with no per-child index control: indices of the
  // result are children's indices concatenated left-to-right. We accumulate
  // left-to-right with rank-based 1-based sequences `<1..r_acc>` × `<r_acc+1
  // ..r_acc+r_child>`. Inlining the nested calls keeps a single `auto tN =
  // …;` line — the compiler's expression-template machinery sees through it.
  void operator()(cas::simple_outer_product const &v) override {
    auto const &children = v.data();
    if (children.empty()) {
      // Defensive fallback — simplifier normally collapses an empty product.
      std::ostringstream zero;
      zero << "tmech::tensor<double, " << v.dim() << ", " << v.rank() << ">{}";
      m_result = register_temp(&v, zero.str());
      return;
    }
    if (children.size() == 1) {
      m_result = apply(children.front());
      return;
    }
    std::string acc = apply(children.front());
    std::size_t acc_rank = children.front().get().rank();
    for (std::size_t i = 1; i < children.size(); ++i) {
      auto rhs_name = apply(children[i]);
      const std::size_t rhs_rank = children[i].get().rank();
      std::ostringstream os;
      os << "tmech::outer_product<" << contiguous_sequence_str(1, acc_rank)
         << ", " << contiguous_sequence_str(acc_rank + 1, acc_rank + rhs_rank)
         << ">(" << acc << ", " << rhs_name << ")";
      acc = os.str();
      acc_rank += rhs_rank;
    }
    m_result = register_temp(&v, std::move(acc));
  }

  // inner_product_wrapper → tmech::inner_product<sequence<...>, sequence<...>>.
  //
  // Short-circuit: if LHS is a rank-4 P_{sym|skew|vol|dev} projector and the
  // contraction matches the rank-2 application pattern `{3,4} × {1,2}`, emit
  // the equivalent unary tmech op directly. This is the same optimisation
  // numsim-cas's tensor_evaluator applies (tensor_evaluator.h:110) — keeps
  // generated code readable and avoids materialising the projector tensor.
  void operator()(cas::inner_product_wrapper const &v) override {
    if (auto fn = projector_short_circuit_fn(v)) {
      auto rhs = apply(v.expr_rhs());
      m_result = register_temp(&v, std::string{"tmech::"} + fn + "(" + rhs +
                                       ")");
      return;
    }
    auto lhs = apply(v.expr_lhs());
    auto rhs = apply(v.expr_rhs());
    std::ostringstream os;
    os << "tmech::inner_product<" << tmech_sequence_str(v.indices_lhs()) << ", "
       << tmech_sequence_str(v.indices_rhs()) << ">(" << lhs << ", " << rhs
       << ")";
    m_result = register_temp(&v, os.str());
  }

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
          " is not yet supported. Use rank-2 tensors, or track "
          "NumSim-Stack/numsim-cas#248 for rank-4 support landing upstream.");
    }
    auto inner = apply(v.expr());
    // wrap_if_compound: today `inner` is always a single token (named symbol
    // or temp from a prior register_temp), so this is a no-op. The wrap
    // future-proofs against any node that starts returning a compound
    // `m_result` (cf. the same pattern in tensor_scalar_mul).
    m_result = register_temp(&v, "tmech::inv(" + wrap_if_compound(inner) + ")");
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

  // Render a cas::sequence (0-based) as a 1-based tmech::sequence<...>.
  static auto tmech_sequence_str(cas::sequence const &s) -> std::string {
    std::ostringstream os;
    os << "tmech::sequence<";
    bool first = true;
    for (auto i : s.indices()) {
      if (!first)
        os << ", ";
      os << (i + 1); // cas stores 0-based; tmech uses 1-based.
      first = false;
    }
    os << ">";
    return os.str();
  }

  // Render `tmech::sequence<lo, lo+1, ..., hi>` (1-based, inclusive).
  static auto contiguous_sequence_str(std::size_t lo, std::size_t hi)
      -> std::string {
    std::ostringstream os;
    os << "tmech::sequence<";
    for (std::size_t k = lo; k <= hi; ++k) {
      if (k != lo)
        os << ", ";
      os << k;
    }
    os << ">";
    return os.str();
  }

  // If `v` is `P : A` for a known rank-4 projector preset applied to a
  // rank-2 tensor (`{3,4} × {1,2}` contraction), return the tmech unary
  // function name. Otherwise return nullptr.
  static auto projector_short_circuit_fn(cas::inner_product_wrapper const &v)
      -> const char * {
    if (v.indices_lhs() != cas::sequence{3, 4} ||
        v.indices_rhs() != cas::sequence{1, 2}) {
      return nullptr;
    }
    if (v.expr_rhs().get().rank() != 2) {
      return nullptr;
    }
    if (!cas::is_same<cas::tensor_projector>(v.expr_lhs())) {
      return nullptr;
    }
    auto const &proj = v.expr_lhs().template get<cas::tensor_projector>();
    if (proj.acts_on_rank() != 2) {
      return nullptr;
    }
    auto const &sp = proj.space();
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::AnyTraceTag>(sp.trace))
      return "sym";
    if (std::holds_alternative<cas::Skew>(sp.perm) &&
        std::holds_alternative<cas::AnyTraceTag>(sp.trace))
      return "skew";
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::VolumetricTag>(sp.trace))
      return "vol";
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::DeviatoricTag>(sp.trace))
      return "dev";
    return nullptr;
  }

  CodeGenContext &m_ctx;
  ScalarCodeEmit &m_scalar;
  std::string m_result;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
