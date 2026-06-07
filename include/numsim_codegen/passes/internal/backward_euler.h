#ifndef NUMSIM_CODEGEN_PASSES_INTERNAL_BACKWARD_EULER_H
#define NUMSIM_CODEGEN_PASSES_INTERNAL_BACKWARD_EULER_H

// Internal implementation header — include <numsim_codegen/recipe.h>, not
// this directly. It is included from the bottom of recipe.h AFTER
// ConstitutiveModel is complete (the helper calls its accessors).
#ifndef NUMSIM_CODEGEN_RECIPE_H
#error "Do not include passes/internal/backward_euler.h directly; include <numsim_codegen/recipe.h>."
#endif

// Shared backward-Euler residual-construction helper. Single source of
// truth across `TimeIntegrationPass::run` and `LocalJacobianPass::run`
// for the discrete residual shape `(α − α_old)/dt − rate`. If the shape
// ever changes (sign convention, scaling factor, BDF2 numerator), both
// passes pick up the change for free.
//
// Carries `cur_expr` alongside the synthesized residual so LJP can call
// `cas::diff` on the same DAG without re-resolving the state-variable
// handle.
namespace numsim::codegen::detail {

// Phase 3a-2 will need to extend or pair with this helper to access
// the Jacobian alongside the residual. See #73 for the decision.
struct BackwardEulerResidual {
  cas::expression_holder<cas::scalar_expression> residual;
  cas::expression_holder<cas::scalar_expression> cur_expr;
};

// Takes the recipe by `const &` (helper only reads); mutation happens
// at the `add_output` call in the calling pass. `calling_pass_name` is
// `std::string_view` so callers can pass `Pass::name()` directly.
inline auto build_backward_euler_residual(
    ConstitutiveModel const &model, EvolutionEquation const &eq,
    std::string_view calling_pass_name) -> BackwardEulerResidual {
  auto const &svs = model.state_variables();
  auto const &sv = svs[eq.state_variable_idx];

  // Resolve `dt`, `<sv>`, `<sv>_old` scalar handles by walking the
  // scalar symbol map. Single pass over the map captures all three.
  // The `dt` symbol is auto-registered by
  // `add_scalar_evolution_equation` → `ensure_dt_symbol_registered_`
  // so the lookup must succeed if the recipe was built through the
  // public API.
  cas::expression_holder<cas::scalar_expression> dt_expr;
  cas::expression_holder<cas::scalar_expression> cur_expr;
  cas::expression_holder<cas::scalar_expression> old_expr;
  auto const &cur_name = model.symbols()[sv.current_symbol_idx].name;
  auto const &old_name = model.symbols()[sv.old_symbol_idx].name;
  for (auto const &[name, h] : model.scalar_symbol_map()) {
    if (name == state_time_step_name) dt_expr = h;
    else if (name == cur_name) cur_expr = h;
    else if (name == old_name) old_expr = h;
  }
  if (!dt_expr.is_valid()) {
    throw std::runtime_error(std::format(
        "ConstitutiveModel '{}': {} cannot find the '{}' symbol — "
        "`ensure_dt_symbol_registered_()` should have auto-registered "
        "it on the first add_scalar_evolution_equation call. State "
        "variable / symbol vectors are out of sync.",
        model.name(), calling_pass_name, state_time_step_name));
  }
  if (!cur_expr.is_valid() || !old_expr.is_valid()) {
    throw std::runtime_error(std::format(
        "ConstitutiveModel '{}': {} cannot resolve current/old scalar "
        "handles for state variable '{}'. State variable / symbol "
        "vectors are out of sync.",
        model.name(), calling_pass_name, sv.name));
  }

  auto residual = (cur_expr - old_expr) / dt_expr - eq.rate;
  return {std::move(residual), std::move(cur_expr)};
}

} // namespace numsim::codegen::detail

#endif // NUMSIM_CODEGEN_PASSES_INTERNAL_BACKWARD_EULER_H
