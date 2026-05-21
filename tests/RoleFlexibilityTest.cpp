// Tests for the open-set Role API: user-defined roles, attribute-based
// backend dispatch, name-uniqueness enforcement, and string lifetime.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

namespace numsim::codegen {

// ─── User-defined custom Role round-trip ─────────────────────────────────

TEST(RoleFlexibility, CustomRoleRoundTripsThroughMooseBackend) {
  ConstitutiveModel m("PhaseFieldShear");
  auto mu = m.add_parameter("mu", 0.5, "Shear modulus");
  // Custom role not in the roles:: catalogue.
  Role phase_field{.name = "phase_field", .is_driving = true,
                   .expected_rank = 0};
  auto phi = m.add_scalar_input("phi", phase_field);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * phi * eps, roles::Stress);

  MooseMaterialTarget target;
  auto files = target.emit(m);
  ASSERT_EQ(files.size(), 2u);

  // The custom role name flows through to the validParams docstring.
  EXPECT_NE(files[1].contents.find("Coupled phase_field"), std::string::npos)
      << "got:\n" << files[1].contents;
}

// ─── Stateful guard fires on any is_stateful=true, not just History ──────

TEST(RoleFlexibility, MooseRejectsAnyStatefulRoleOnInput) {
  ConstitutiveModel m("PlasticityTest");
  Role plastic_strain{.name = "plastic_strain", .is_stateful = true,
                      .is_symmetric = true, .expected_rank = 2};
  auto eps_p = m.add_tensor_input("eps_p", 3, 2, plastic_strain);
  m.add_output("stress", eps_p, roles::Stress);

  MooseMaterialTarget target;
  try {
    (void)target.emit(m);
    FAIL() << "expected std::runtime_error for stateful input role";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("stateful role 'plastic_strain'"), std::string::npos)
        << "message: " << msg;
    EXPECT_NE(msg.find("on input 'eps_p'"), std::string::npos)
        << "message: " << msg;
  }
}

// ─── find_*_by_role honours name-based equality for custom roles ─────────

TEST(RoleFlexibility, FindByRoleMatchesCustomRoleByName) {
  ConstitutiveModel m("CustomLookup");
  Role damage{.name = "damage", .is_driving = true, .expected_rank = 0};
  auto d = m.add_scalar_input("d", damage);
  m.add_output("d_out", d);

  auto const *found = m.find_input_by_role(Role{.name = "damage"});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name, "d");

  // Lookup with a different custom name returns nullptr.
  EXPECT_EQ(m.find_input_by_role(Role{.name = "not_damage"}), nullptr);
}

// ─── Name collision with catalogue + attribute mismatch throws ───────────

TEST(RoleFlexibility, RejectsNameCollisionWithCatalogueButMismatchedAttrs) {
  ConstitutiveModel m("BadRoleTest");

  // Same name as roles::Strain but is_driving=false. Silent equality would
  // otherwise let backends mis-route this.
  Role bad{.name = "strain", .is_stateful = false, .is_driving = false,
           .is_symmetric = false, .expected_rank = 2};

  try {
    (void)m.add_tensor_input("eps", 3, 2, bad);
    FAIL() << "expected throw on attribute mismatch with catalogue role";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("roles::Strain"), std::string::npos) << "message: " << msg;
    EXPECT_NE(msg.find("Role 'strain'"), std::string::npos) << "message: " << msg;
  }
}

TEST(RoleFlexibility, AcceptsExactCatalogueMatchConstructedFromScratch) {
  ConstitutiveModel m("ExactMatchTest");
  // A user explicitly rebuilding roles::Strain with the same attributes
  // should not throw — the contract is "same name => same attributes."
  Role rebuilt{.name = "strain", .is_stateful = false, .is_driving = true,
               .is_symmetric = true, .expected_rank = 2};
  EXPECT_NO_THROW({
    m.add_tensor_input("eps", 3, 2, rebuilt);
  });
}

// ─── Role::name owns the string — no lifetime hazard on temporaries ──────

// Role names with embedded special characters must be escaped before
// being concatenated into the generated C++ string literal. Without the
// escape, a literal `"` or `\n` produces ill-formed C++.

TEST(RoleFlexibility, RoleNameWithSpecialCharsIsEscapedInGeneratedSource) {
  ConstitutiveModel m("EscapeTest");
  // Pathological-but-not-rejected role names — none of the catalogue
  // entries would match, and there's no identifier validator yet.
  Role bad_name{.name = "with\"quote\nand\\bs",
                .is_driving = true, .expected_rank = 0};
  auto x = m.add_scalar_input("x", bad_name);
  m.add_output("y", x);

  MooseMaterialTarget target;
  auto files = target.emit(m);
  auto const &source = files[1].contents;

  // The four special chars must appear in their escaped form, never as
  // raw characters that would break the validParams docstring literal.
  EXPECT_NE(source.find("Coupled with\\\"quote\\nand\\\\bs"),
            std::string::npos)
      << "expected escaped form 'Coupled with\\\"quote\\nand\\\\bs' in:\n"
      << source;
  // Confirm none of the raw forms slipped through.
  EXPECT_EQ(source.find("Coupled with\""), std::string::npos);
  EXPECT_EQ(source.find("Coupled with\\\"quote\n"), std::string::npos);
}

TEST(RoleFlexibility, RoleNameSurvivesTemporaryStringSource) {
  ConstitutiveModel m("LifetimeTest");
  {
    // Role constructed from a local string that goes out of scope after
    // add_*. The Role's std::string member must own the data.
    std::string local_name = "temporary_role";
    Role transient{.name = local_name, .is_driving = true,
                   .expected_rank = 0};
    (void)m.add_scalar_input("x", transient);
    local_name = "GARBAGE";  // mutate to detect any aliasing
  }
  auto const *found = m.find_input_by_role(Role{.name = "temporary_role"});
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name, "x");
  EXPECT_EQ(found->role.name, "temporary_role");
}

} // namespace numsim::codegen
