#ifndef NUMSIM_CODEGEN_RECIPE_RENDER_H
#define NUMSIM_CODEGEN_RECIPE_RENDER_H

// Phase-1.2 follow-up (issue #48 / M5): the function-frame rendering that
// turns a populated CodeGenContext into the final `_compute` function
// source lives here as free functions rather than as private methods on
// `ConstitutiveModel`. Reasons:
//
// 1. Phase 3+ adds new emit passes (TangentEmitPass, MoosePropertyEmitPass,
//    StateVarEmitPass, AbaqusEmitPass). Each would have to be friended on
//    ConstitutiveModel to call a private render helper — O(passes) friends
//    on a user-facing class.
// 2. The render functions only need ConstitutiveModel's read-only public
//    accessors (`name()`, `symbols()`, `outputs()`). They never touched
//    private state to begin with — the previous private-member layout was
//    leakage from the original single-emitter monolith.
//
// Note: this header intentionally does NOT include recipe.h. It uses
// forward declarations and trailing inline definitions so the TU that
// includes both gets ConstitutiveModel's full definition (recipe.h),
// followed by the inline render bodies that depend on it. Recipe.h
// includes this header at its bottom; that's the standard inclusion
// pattern for callers.

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/passes/recipe_view.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace numsim::codegen {

// Phase 2.6 (issue #77): the single source of truth for the generated
// `_compute` function's parameter order. `render_compute_function` builds
// the SIGNATURE by iterating these, and every backend's call-site builds
// the ACTUAL arguments by iterating the same list — so the two cannot
// drift (the MOOSE call-site and the signature historically rebuilt the
// order independently, which is exactly how the state-variable arity bug
// arose).
//
// One ArgSpec per generated parameter, in declaration order:
//   symbols() in registration order (minus in-function-solved state
//   variables) → outputs() → in-function-solved state variables.
struct ArgSpec {
  enum class Role {
    ScalarParam,      // double const <name>   ← MOOSE: _<name>
    ScalarInput,      // double const <name>   ← MOOSE: _<name>[_qp]
    TensorInput,      // T# const & <name>     ← MOOSE: <name>_ad
    StateOld,         // double const <name>   ← MOOSE: _<name>[_qp] (getMaterialPropertyOld)
    StateCurrentRead, // double const <name>   ← MOOSE: _<name>[_qp] (declared property, read)
    TimeStep,         // double const dt       ← MOOSE: _dt (framework timestep)
    ScalarOutput,       // double & <name>_out   ← MOOSE: _<name>[_qp]
    TensorOutput,       // T# & <name>_out       ← MOOSE: <name>_ad
    TensorTangentOutput,// T# & <name>_out       ← MOOSE: _Jacobian_mult[_qp]
                        //   (Phase 5: a roles::ConsistentTangent rank-4 output;
                        //    identical to TensorOutput in the Layer-2 signature,
                        //    but a backend routes it to the framework's
                        //    consistent-tangent slot rather than a fresh output.)
    NewtonStateOut,     // double & <name>_out   ← MOOSE: _<name>[_qp] (solved, written)
  };
  std::string name;     // base symbol/output name (no `_out` suffix)
  Role role;
  std::size_t dim = 0;  // tensor only
  std::size_t rank = 0; // tensor only
};

// Compute the canonical argument list for a recipe. Defined inline at the
// bottom of recipe.h (needs the complete ConstitutiveModel + IR types).
// `inline` on the declaration makes the linkage explicit — the definition
// is inline and only reachable via recipe.h's bottom include.
[[nodiscard]] inline auto canonical_arguments(RecipeView model)
    -> std::vector<ArgSpec>;

// A Newton segment after CodeEmitPass has rendered its loop-local CSE temps
// + the residual/Jacobian right-hand sides to text (Phase 3a-2, issue #75).
// `render_compute_function` assembles these into the iteration body.
struct RenderedNewtonSegment {
  std::string state_var_name;   // the current symbol, e.g. "alpha"
  std::string loop_local_decls; // `    auto tN = ...;\n` lines (4-indented)
  std::string residual_rhs;     // expression for the residual R
  std::string jacobian_rhs;     // expression for the Jacobian J = dR/dsv
  double tol;
  int max_iter;
};

// Render the function-frame source (signature + body + output writes)
// from a populated CodeGenContext and the model view. Bodies are inline
// at the bottom of recipe.h (where ConstitutiveModel is complete and
// RecipeView's delegates are defined).
//
// Takes a RecipeView rather than ConstitutiveModel const & so any emit
// pass can call it from a PassContext directly (M4: passes hold views,
// not raw recipe refs).
[[nodiscard]] auto render_compute_function(
    RecipeView model, CodeGenContext const &ctx,
    std::vector<std::string> const &output_rhs,
    std::vector<RenderedNewtonSegment> const &newton = {}) -> std::string;

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_RENDER_H
