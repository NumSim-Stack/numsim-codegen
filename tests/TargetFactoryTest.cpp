// #93 (descoped): the name-keyed Target backend factory promoted into the
// library. Flat function, not a registry — these pin the three first-party
// targets, the default, and the unknown-name diagnostic.

#include <numsim_codegen/numsim_codegen.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace numsim::codegen {
namespace {

TEST(TargetFactory, ConstructsEachKnownTarget) {
  EXPECT_EQ(make_target("numsim_material")->target_name(), "NumSimMaterial");
  EXPECT_EQ(make_target("standalone")->target_name(), "StandaloneCxx");
  EXPECT_EQ(make_target("moose")->target_name(), "MooseMaterial");
}

TEST(TargetFactory, DefaultIsNumSimMaterial) {
  EXPECT_EQ(default_target_name, "numsim_material");
  EXPECT_EQ(make_target(default_target_name)->target_name(), "NumSimMaterial");
}

TEST(TargetFactory, NamesListDefaultFirstAndAreAllConstructible) {
  auto const &names = target_names();
  ASSERT_FALSE(names.empty());
  EXPECT_EQ(names.front(), default_target_name);
  // Every advertised name must actually construct.
  for (auto const n : names) {
    EXPECT_NE(make_target(n), nullptr) << "name '" << n << "' did not construct";
  }
}

TEST(TargetFactory, UnknownNameThrowsWithDiagnostic) {
  try {
    (void)make_target("abaqus");
    FAIL() << "expected throw on unknown target";
  } catch (std::runtime_error const &e) {
    std::string const msg(e.what());
    EXPECT_NE(msg.find("abaqus"), std::string::npos) << msg;       // the bad name
    EXPECT_NE(msg.find("numsim_material"), std::string::npos) << msg; // the valid list
  }
}

} // namespace
} // namespace numsim::codegen
