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

#include <string>
#include <utility>
#include <vector>

namespace numsim::codegen {

class ConstitutiveModel;
struct SymbolDecl;
struct OutputDecl;

class RecipeView {
public:
  using ScalarSymbolMap =
      std::vector<std::pair<std::string,
                            cas::expression_holder<cas::scalar_expression>>>;
  using TensorSymbolMap =
      std::vector<std::pair<std::string,
                            cas::expression_holder<cas::tensor_expression>>>;

  explicit RecipeView(ConstitutiveModel const &model) noexcept
      : m_model(&model) {}

  // Read-only delegates to ConstitutiveModel's public accessors. Bodies
  // live at the bottom of recipe.h (where ConstitutiveModel is complete).
  [[nodiscard]] auto name() const -> std::string const &;
  [[nodiscard]] auto symbols() const -> std::vector<SymbolDecl> const &;
  [[nodiscard]] auto outputs() const -> std::vector<OutputDecl> const &;
  [[nodiscard]] auto scalar_symbol_map() const -> ScalarSymbolMap const &;
  [[nodiscard]] auto tensor_symbol_map() const -> TensorSymbolMap const &;

  // Escape hatch for callers that need the full ConstitutiveModel reference
  // (e.g. find_*_by_role, parameters(), inputs()). Treat usage as a signal
  // that RecipeView's surface should widen rather than as a permanent
  // solution.
  [[nodiscard]] auto raw_model() const noexcept -> ConstitutiveModel const & {
    return *m_model;
  }

private:
  ConstitutiveModel const *m_model;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_VIEW_H
