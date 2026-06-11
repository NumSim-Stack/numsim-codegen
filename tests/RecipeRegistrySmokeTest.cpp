// Smoke test: every recipe in examples/recipe_registry.h emits non-empty
// files through both shipped targets. Catches regressions in the example
// itself (e.g. a recipe that builds but fails at emit due to a missing
// operator).

#include "recipe_registry.h"

#include <gtest/gtest.h>

namespace numsim::codegen {

class RecipeRegistrySmoke
    : public ::testing::TestWithParam<numsim::examples::RecipeEntry> {};

TEST_P(RecipeRegistrySmoke, EmitsViaStandaloneTarget) {
  StandaloneCxxTarget target;
  auto model = GetParam().build();
  auto files = target.emit(model);
  ASSERT_FALSE(files.empty()) << "no files emitted for " << GetParam().name;
  for (auto const &f : files) {
    EXPECT_FALSE(f.contents.empty())
        << "empty content for " << f.filename << " in " << GetParam().name;
  }
}

TEST_P(RecipeRegistrySmoke, EmitsViaMooseTarget) {
  MooseMaterialTarget target("SmokeTest");
  auto model = GetParam().build();
  auto files = target.emit(model);
  ASSERT_FALSE(files.empty()) << "no files emitted for " << GetParam().name;
  for (auto const &f : files) {
    EXPECT_FALSE(f.contents.empty())
        << "empty content for " << f.filename << " in " << GetParam().name;
  }
}

// Use ValuesIn over the registry — when a new recipe is appended in
// recipe_registry.h, it's automatically picked up here without test edits.
INSTANTIATE_TEST_SUITE_P(
    AllRegistryRecipes, RecipeRegistrySmoke,
    ::testing::ValuesIn(numsim::examples::registry()),
    [](::testing::TestParamInfo<numsim::examples::RecipeEntry> const &param_info) {
      // `info` would shadow the gtest macro's own parameter (-Wshadow).
      return std::string(param_info.param.name);
    });

} // namespace numsim::codegen
