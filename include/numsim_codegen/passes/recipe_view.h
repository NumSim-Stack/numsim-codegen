#ifndef NUMSIM_CODEGEN_RECIPE_VIEW_H
#define NUMSIM_CODEGEN_RECIPE_VIEW_H

// Phase-1.2 follow-up (issue #48 / M4): thin handle indirecting between
// PassContext and ConstitutiveModel.
//
// Today RecipeView is a const-only delegate — its accessors mirror the
// subset of ConstitutiveModel that current passes (SymbolValidationPass,
// CodeEmitPass) actually use.
//
// Phase 2 introduces passes that mutate expressions
// (TimeIntegrationPass lowers `Dt(α) → (α_new − α_old)/dt` via cas
// substitute; KuhnTuckerLoweringPass rewrites `λ ≥ 0, λ·f = 0` to a
// Fischer-Burmeister NCP residual). Adding `replace_output_expr()` /
// `set_state_initial(...)` etc. to RecipeView THEN will not break the
// existing `Pass::run(PassContext&)` signature — only the bodies of the
// passes that need mutation.
//
// The alternative (PassContext holds `ConstitutiveModel &` directly)
// would mean every pass that captured `pctx.model` by `auto const &`
// silently becomes mutable, and any friend boundary set up against
// ConstitutiveModel shifts. RecipeView keeps that decoupling clean.

#include <numsim_cas/core/expression_holder.h>
#include <numsim_cas/scalar/scalar_expression.h>
#include <numsim_cas/tensor/tensor_expression.h>

#include <cassert>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace numsim::codegen {

class ConstitutiveModel;
struct SymbolDecl;
struct OutputDecl;
struct StateVariable;

class RecipeView {
public:
  using ScalarSymbolMap =
      std::vector<std::pair<std::string,
                            cas::expression_holder<cas::scalar_expression>>>;
  using TensorSymbolMap =
      std::vector<std::pair<std::string,
                            cas::expression_holder<cas::tensor_expression>>>;

  // P1: dual constructors so both const and non-const callers can wrap
  // a recipe. `ConstitutiveModel const &` produces a const-only view;
  // `ConstitutiveModel &` retains write access for Phase 2 passes via
  // `try_mutable_model()`. Validation/emit today exercise the const
  // path; Phase 2's TimeIntegrationPass will use the non-const ctor.
  explicit RecipeView(ConstitutiveModel const &model) noexcept
      : m_model(&model) {}
  explicit RecipeView(ConstitutiveModel &model) noexcept
      : m_model(&model) {}

  // Read-only delegates to ConstitutiveModel's public accessors. Bodies
  // live at the bottom of recipe.h (where ConstitutiveModel is complete).
  [[nodiscard]] auto name() const -> std::string const &;
  [[nodiscard]] auto symbols() const -> std::vector<SymbolDecl> const &;
  [[nodiscard]] auto outputs() const -> std::vector<OutputDecl> const &;
  [[nodiscard]] auto state_variables() const
      -> std::vector<StateVariable> const &;
  [[nodiscard]] auto scalar_symbol_map() const -> ScalarSymbolMap const &;
  [[nodiscard]] auto tensor_symbol_map() const -> TensorSymbolMap const &;

  // Escape hatch for callers that need the full ConstitutiveModel reference
  // (e.g. find_*_by_role, parameters(), inputs()). Treat usage as a signal
  // that RecipeView's surface should widen rather than as a permanent
  // solution.
  [[nodiscard]] auto raw_model() const noexcept -> ConstitutiveModel const & {
    // P1 (cpp-pro m-1 follow-up): assert non-null. The variant always
    // holds a pointer set at construction; nullptr would mean the view
    // was constructed from a default-init ctor (we don't expose one),
    // memory was clobbered, or a future bug forgot to initialise.
    return *std::visit(
        [](auto const *p) -> ConstitutiveModel const * {
          assert(p != nullptr);
          return p;
        },
        m_model);
  }

  // P1: non-null `ConstitutiveModel *` if the view was constructed from
  // a non-const ref; `nullptr` if from a const ref. Phase 2 mutating
  // passes (TimeIntegrationPass, KuhnTuckerLoweringPass) call this and
  // check the nullptr — that's a pipeline-misconfiguration error, not
  // silent corruption. For the common case (every mutating pass would
  // otherwise write the same if/throw boilerplate), prefer
  // `require_mutable_model()` below.
  //
  // The non-const constructor accepts `ConstitutiveModel &` only —
  // rvalues bind to the const overload, so a temporary cannot produce a
  // mutable view. Mutation requires a persistent lvalue.
  [[nodiscard]] auto try_mutable_model() noexcept -> ConstitutiveModel * {
    if (auto **p = std::get_if<ConstitutiveModel *>(&m_model)) {
      assert(*p != nullptr);
      return *p;
    }
    return nullptr;
  }

  // Convenience for Phase 2 mutating passes: returns a non-const reference
  // or throws with a canonical pipeline-misconfiguration message.
  // `pass_name` is interpolated into the message so the user sees which
  // pass failed. m1 in REVIEW-pr-57.md.
  [[nodiscard]] auto require_mutable_model(char const *pass_name)
      -> ConstitutiveModel & {
    if (auto *p = try_mutable_model()) {
      return *p;
    }
    throw std::runtime_error(std::format(
        "Pass '{}' requires a mutable RecipeView (constructed from "
        "ConstitutiveModel&), but the PassContext holds a const view. "
        "This is a pipeline-construction error — the pass shouldn't have "
        "been registered in a const-only path (e.g. validate()).",
        pass_name));
  }

private:
  // Variant rather than always-const-pointer: lets the view advertise
  // "I came from a non-const recipe" without bifurcating the public
  // API into ConstRecipeView/MutableRecipeView types. The const surface
  // (every other accessor) works identically for either variant arm.
  std::variant<ConstitutiveModel const *, ConstitutiveModel *> m_model;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_VIEW_H
