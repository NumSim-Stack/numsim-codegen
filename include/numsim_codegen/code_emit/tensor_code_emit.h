#ifndef NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
#define NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/spectral_decompose_emit.h>

#include <numsim_cas/basic_functions.h>
#include <numsim_cas/tensor/identity_tensor.h>
#include <numsim_cas/tensor/levi_civita_tensor.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_if_then_else_scalar.h>
#include <numsim_cas/tensor/tensor_if_then_else_t2s.h>
#include <numsim_cas/tensor/tensor_visitor_typedef.h>
#include <numsim_cas/tensor/wrappers/tensor_eigenprojection.h>
#include <numsim_cas/tensor/wrappers/tensor_eigenvector.h>
#include <numsim_cas/tensor/wrappers/tensor_inv.h>
#include <numsim_cas/tensor/wrappers/tensor_isotropic_function.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_expression.h>

#include <format>
#include <functional>
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
  // Callback bridging into the tensor_to_scalar emitter. Required because
  // the tensor and t2s visitors form a cycle (t2s emit holds a TensorCodeEmit&
  // for sub-tensor expressions; some tensor nodes — namely
  // `tensor_to_scalar_with_tensor_mul` — embed a t2s subterm). Taking the
  // callback at construction time means a TensorCodeEmit cannot be built
  // without it, so the "callback unset" case is a compile error rather
  // than a runtime throw (M3 in issue #48).
  using T2sApply = std::function<std::string(
      cas::expression_holder<cas::tensor_to_scalar_expression> const &)>;

  TensorCodeEmit(CodeGenContext &ctx, ScalarCodeEmit &scalar, T2sApply t2s)
      : m_ctx(ctx), m_scalar(scalar), m_t2s_apply(std::move(t2s)) {}

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
    throw std::runtime_error(std::format("TensorCodeEmit: identity_tensor of "
                                         "rank {} not supported in Phase A MVP",
                                         v.rank()));
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

  // Piecewise tensor selection with a SCALAR condition. The two branches are
  // tmech expression templates of (generally) different types, so a C++
  // ternary needs a common type — materialise both to a concrete
  // `tmech::tensor<double, dim, rank>` (same dim/rank, asserted equal in the
  // node). The condition goes through the scalar emitter.
  //
  // EAGER-BRANCH limitation (shared with scalar_if_then_else): both branches'
  // sub-expression temporaries are emitted into the flat preamble, so BOTH are
  // always evaluated — the ternary only selects the result. A partial function
  // on the untaken branch (e.g. inv() of a tensor singular only there) still
  // computes and may produce NaN/Inf. Total functions are unaffected. A real
  // fix needs branch bodies lowered into a C++ if/else, not a ternary.
  void operator()(cas::tensor_if_then_else_scalar const &v) override {
    auto cond = m_scalar.apply(v.expr_cond());
    emit_if_then_else(&v, cond, v.expr_then(), v.expr_else(), v.dim(),
                      v.rank());
  }

  // Same ternary lowering as the scalar-condition sibling, but the condition
  // is a tensor_to_scalar expression (#241) — emitted via the t2s callback.
  void operator()(cas::tensor_if_then_else_t2s const &v) override {
    auto cond = m_t2s_apply(v.expr_cond());
    emit_if_then_else(&v, cond, v.expr_then(), v.expr_else(), v.dim(),
                      v.rank());
  }

  // ─── Phase A stubs — node types not yet implemented ─────────────

#define NUMSIM_CODEGEN_TENSOR_STUB(T)                                          \
  void operator()(cas::T const &) override {                                   \
    throw std::runtime_error(                                                  \
        "TensorCodeEmit: codegen for " #T                                      \
        " not yet implemented. See numsim-codegen Phase A roadmap.");          \
  }

  // All Phase 1.1 stubs landed — keep the macro for the next batch.

#undef NUMSIM_CODEGEN_TENSOR_STUB

  // ─── Implemented unary nodes ─────────────────────────────────────

  // Recognised rank-4 projector presets at acts_on_rank=2. The
  // (perm, trace) → kind mapping is the single source of truth for both
  // the standalone materialisation (`tensor_projector` emit) and the
  // applied-form short-circuit in `inner_product_wrapper`. M2 (#48).
  enum class ProjectorKind { Unknown, Sym, Skew, Vol, Dev };

  // Classify the (perm, trace) pair of a cas tensor_space against the
  // four currently-supported projector presets.
  static auto
  classify_projector(cas::tensor_space const &sp) noexcept -> ProjectorKind {
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::AnyTraceTag>(sp.trace))
      return ProjectorKind::Sym;
    if (std::holds_alternative<cas::Skew>(sp.perm) &&
        std::holds_alternative<cas::AnyTraceTag>(sp.trace))
      return ProjectorKind::Skew;
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::VolumetricTag>(sp.trace))
      return ProjectorKind::Vol;
    if (std::holds_alternative<cas::Symmetric>(sp.perm) &&
        std::holds_alternative<cas::DeviatoricTag>(sp.trace))
      return ProjectorKind::Dev;
    return ProjectorKind::Unknown;
  }

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
  //   P_sym_{ijkl}  = ½(δ_ik δ_jl + δ_il δ_jk)  = ½(otimesu(I,I) +
  //   otimesl(I,I)) P_skew_{ijkl} = ½(δ_ik δ_jl − δ_il δ_jk)  = ½(otimesu(I,I)
  //   − otimesl(I,I)) P_vol_{ijkl}  = (1/d) δ_ij δ_kl           = (1/d)
  //   otimes(I,I) P_dev_{ijkl}  = P_sym − P_vol
  void operator()(cas::tensor_projector const &v) override {
    const auto r = v.acts_on_rank();
    if (r != 2) {
      throw std::runtime_error(std::format(
          "TensorCodeEmit: tensor_projector with acts_on_rank={} is not yet "
          "supported. Only acts_on_rank=2 (rank-4 projectors on rank-2 "
          "tensors) ships now; higher-rank projector emit lands once we "
          "have a recipe that needs it.",
          r));
    }
    const auto d_str = std::to_string(v.dim());
    const std::string eye = "tmech::eye<double, " + d_str + ", 2>()";
    const std::string ou = "tmech::otimesu(" + eye + ", " + eye + ")";
    const std::string ol = "tmech::otimesl(" + eye + ", " + eye + ")";
    const std::string oo = "tmech::otimes(" + eye + ", " + eye + ")";
    // (1.0 / d) keeps the literal a double; integer d in the denominator
    // would silently floor for non-trivial d.
    const std::string inv_d = "(1.0 / " + d_str + ".0)";

    switch (classify_projector(v.space())) {
    case ProjectorKind::Sym:
      m_result = register_temp(&v, "0.5 * (" + ou + " + " + ol + ")");
      return;
    case ProjectorKind::Skew:
      m_result = register_temp(&v, "0.5 * (" + ou + " - " + ol + ")");
      return;
    case ProjectorKind::Vol:
      m_result = register_temp(&v, inv_d + " * " + oo);
      return;
    case ProjectorKind::Dev:
      m_result = register_temp(&v, "0.5 * (" + ou + " + " + ol + ") - " +
                                       inv_d + " * " + oo);
      return;
    case ProjectorKind::Unknown:
      break;
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
      m_result =
          register_temp(&v, std::string{"tmech::"} + fn + "(" + rhs + ")");
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

  // tensor_mul → chained single-contraction (matrix-product-style).
  //
  // Cas semantics (per tensor_evaluator.h:196): n-ary, accumulates
  // left-to-right by contracting the last index of the accumulator with the
  // first index of the next child. Result rank = Σranks − 2·(nchildren−1).
  // We mirror that with `tmech::inner_product<sequence<r_acc>, sequence<1>>`.
  //
  // The cas node also carries an optional `coeff()` that the evaluator
  // applies as an element-wise product on the flattened buffer. That's an
  // unusual semantic for `tensor_mul` and no shipping recipe uses it; we
  // throw with a clear pointer rather than guess at the emit shape.
  void operator()(cas::tensor_mul const &v) override {
    if (v.coeff().is_valid()) {
      throw std::runtime_error(
          "TensorCodeEmit: tensor_mul with a coeff() (element-wise product on "
          "the flattened buffer) has no clean tmech analogue. If you hit this, "
          "open an issue with the recipe so we can pin down the intended "
          "semantics.");
    }
    auto const &children = v.data();
    if (children.empty()) {
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
      os << "tmech::inner_product<tmech::sequence<" << acc_rank
         << ">, tmech::sequence<1>>(" << acc << ", " << rhs_name << ")";
      acc = os.str();
      acc_rank = acc_rank + rhs_rank - 2;
    }
    m_result = register_temp(&v, std::move(acc));
  }

  // tensor_pow → tmech::pow for rank-2 bases. tmech's pow is documented
  // only for second-order tensors; higher-rank bases would need explicit
  // unrolling of `inner_product<sequence<r>, sequence<1>>` repeated N times,
  // which requires a compile-time-known integer exponent. We defer that
  // until a recipe needs it.
  void operator()(cas::tensor_pow const &v) override {
    if (v.rank() != 2) {
      throw std::runtime_error(std::format(
          "TensorCodeEmit: tensor_pow of rank {} is not yet supported. "
          "tmech::pow ships rank-2 only; higher-rank bases require "
          "static-exponent unrolling — not implemented in this phase.",
          v.rank()));
    }
    auto base = apply(v.expr_lhs());
    auto exp = m_scalar.apply(v.expr_rhs());
    m_result = register_temp(&v, "tmech::pow(" + base + ", " + exp + ")");
  }

  // permute_indices_wrapper → tmech::basis_change<sequence<...>>(A).
  //
  // The classic special case is transpose (`indices == {2, 1}` for rank-2),
  // but arbitrary permutations of any rank are supported by both sides.
  // cas stores indices 0-based; tmech::basis_change expects 1-based.
  void operator()(cas::permute_indices_wrapper const &v) override {
    auto inner = apply(v.expr());
    m_result = register_temp(&v, "tmech::basis_change<" +
                                     tmech_sequence_str(v.indices()) + ">(" +
                                     inner + ")");
  }

  // tensor_to_scalar_with_tensor_mul → scalar · tensor.
  //
  // LHS is a tensor; RHS is a tensor_to_scalar (a scalar-valued tensor
  // reduction like trace/det/norm or a composite thereof). The result is
  // a tensor of LHS's shape with every entry multiplied by the scalar.
  //
  // The t2s emitter lives in a separate visitor that already includes this
  // header. We accept the t2s callback at construction time to break the
  // include cycle without leaving room for a runtime "callback unset"
  // case — the constructor signature guarantees the bridge is wired.
  void operator()(cas::tensor_to_scalar_with_tensor_mul const &v) override {
    auto lhs = apply(v.expr_lhs());
    auto rhs = m_t2s_apply(v.expr_rhs());
    m_result = register_temp(&v, wrap_if_compound(rhs) + " * " +
                                     wrap_if_compound(lhs));
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
      throw std::runtime_error(std::format(
          "TensorCodeEmit: tensor_inv of rank {} is not yet supported. "
          "Use rank-2 tensors, or track NumSim-Stack/numsim-cas#248 for "
          "rank-4 support landing upstream.",
          v.rank()));
    }
    auto inner = apply(v.expr());
    // wrap_if_compound: today `inner` is always a single token (named symbol
    // or temp from a prior register_temp), so this is a no-op. The wrap
    // future-proofs against any node that starts returning a compound
    // `m_result` (cf. the same pattern in tensor_scalar_mul).
    m_result = register_temp(&v, "tmech::inv(" + wrap_if_compound(inner) + ")");
  }

  // ─── Spectral nodes (#325/#326) ──────────────────────────────────
  // Each reads the shared ascending eigendecomposition of its tensor argument
  // (one `spectral_decompose(A)` per distinct A, interned by argument name).

  // E_i(A) = n_i ⊗ n_i, the i-th eigenprojection (i in ascending eigenvalue
  // order).
  void operator()(cas::tensor_eigenprojection const &v) override {
    auto const vec = eigenvector_ref(v.expr(), v.index());
    m_result = register_temp(&v, "tmech::otimes(" + vec + ", " + vec + ")");
  }

  // n_i(A), the i-th eigenvector (rank-1).
  void operator()(cas::tensor_eigenvector const &v) override {
    m_result = register_temp(&v, eigenvector_ref(v.expr(), v.index()));
  }

  // f(A) = Σ_i f(λ_i) E_i — the primal isotropic tensor function value.
  // Deferred to slice 3 of the spectral-lowering roadmap (#105).
  void operator()(cas::tensor_isotropic_function const &) override {
    throw std::runtime_error(
        "TensorCodeEmit: codegen for tensor_isotropic_function (the primal "
        "f(A) value) is not yet implemented. Tracked as slice 3 of the "
        "spectral-lowering roadmap (numsim-codegen #105).");
  }

private:
  // Emit (once per argument) the shared decomposition of `arg` and return the
  // C++ expression for its i-th ascending eigenvector.
  auto
  eigenvector_ref(cas::expression_holder<cas::tensor_expression> const &arg,
                  std::size_t index) -> std::string {
    auto const a = apply(arg);
    auto const decomp = emit_shared_decomposition(m_ctx, a, arg.get().dim());
    return decomp + ".eigenvectors[" + std::to_string(index) + "]";
  }

  // Ternary lowering shared by the scalar- and t2s-condition if_then_else.
  void emit_if_then_else(
      void const *ptr, std::string const &cond,
      cas::expression_holder<cas::tensor_expression> const &then_expr,
      cas::expression_holder<cas::tensor_expression> const &else_expr,
      std::size_t dim, std::size_t rank) {
    auto then_branch = apply(then_expr);
    auto else_branch = apply(else_expr);
    std::string const tt = "tmech::tensor<double, " + std::to_string(dim) +
                           ", " + std::to_string(rank) + ">";
    m_result = register_temp(ptr, "(" + wrap_if_compound(cond) + " != 0.0 ? " +
                                      tt + "(" + then_branch + ") : " + tt +
                                      "(" + else_branch + "))");
  }

  auto register_temp(void const *ptr, std::string rhs) -> std::string {
    if (auto *existing = m_ctx.find(ptr)) {
      return *existing;
    }
    return m_ctx.emit_temporary(ptr, std::move(rhs));
  }

  static auto is_single_token(std::string const &s) -> bool {
    return s.find(' ') == std::string::npos && s.find('(') == std::string::npos;
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
  static auto contiguous_sequence_str(std::size_t lo,
                                      std::size_t hi) -> std::string {
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
  //
  // M2 (#48): the (perm, trace) → kind mapping lives in
  // `classify_projector`; we just map the kind to a unary op name here.
  static auto
  projector_short_circuit_fn(cas::inner_product_wrapper const &v) -> const
      char * {
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
    switch (classify_projector(proj.space())) {
    case ProjectorKind::Sym:
      return "sym";
    case ProjectorKind::Skew:
      return "skew";
    case ProjectorKind::Vol:
      return "vol";
    case ProjectorKind::Dev:
      return "dev";
    case ProjectorKind::Unknown:
      return nullptr;
    }
    return nullptr; // unreachable but quiets non-exhaustive-switch warnings.
  }

  CodeGenContext &m_ctx;
  ScalarCodeEmit &m_scalar;
  T2sApply m_t2s_apply; // required at construction (M3) — never empty.
  std::string m_result;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TENSOR_CODE_EMIT_H
