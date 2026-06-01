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
    std::vector<std::string> const &output_rhs) -> std::string;

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_RENDER_H
