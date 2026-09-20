// Negative-path coverage (issue #11): every intentional error path in the
// codegen must fail loudly, and stay failing. Each guard below can silently
// regress without a test pinning it. This file exercises the ones reachable
// through the public recipe / target API.
//
// Coverage map (issue #11 table, reconciled with the current tree):
//   * SymbolValidationPass: output references an undeclared symbol   ✔ here
//   * SymbolValidationPass: declared name is not a C++ identifier     ✔ here
//   * MooseMaterialTarget: stateful INPUT (roles::History)            ✔ here
//   * MooseMaterialTarget: stateful OUTPUT (roles::History)           MooseTargetTest (#15)
//   * MooseMaterialTarget: evolution eqs without enable_local_newton  ✔ here
//   * MooseMaterialTarget: more than one consistent tangent           ✔ here
//   * MooseMaterialTarget: output named "Jacobian_mult" + a tangent   ✔ here
//   * tensor_storage_type: tensor rank ∉ {2, 4}                        ✔ here
//   * duplicate add (symbol / output)                                 ✔ here
//
// Retired / unreachable rows from the original #11 table:
//   * NUMSIM_CODEGEN_TENSOR_STUB throws — all Phase-A tensor stubs have since
//     landed; the macro in tensor_code_emit.h has no remaining uses, so there
//     is no runtime throw to exercise.
//   * NUMSIM_CODEGEN_T2S_STUB / tensor_inner_product_to_scalar — the only
//     surviving code-emit stub. It is NOT producible through the recipe API:
//     the `:` / dot double-contraction maps to the implemented `tensor_dot`
//     handler, not to this general partial-index node. Reaching it needs a
//     hand-built cas node with explicit index sequences, out of scope for a
//     recipe-level negative test. Left documented rather than silently dropped.

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {
namespace {

// ─── SymbolValidationPass guards ─────────────────────────────────────────

TEST(NegativePath, ValidateRejectsUndeclaredSymbol) {
  ConstitutiveModel m("Ghost");
  // `ghost` is never registered via add_*; referencing it in an output must
  // be caught by SymbolValidationPass.
  auto ghost = cas::make_expression<cas::scalar>("ghost");
  m.add_output("y", ghost);
  try {
    m.validate();
    FAIL() << "expected validate() to reject the undeclared symbol";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("undeclared"), std::string::npos) << msg;
    EXPECT_NE(msg.find("ghost"), std::string::npos) << msg;
  }
}

TEST(NegativePath, ValidateRejectsNonIdentifierSymbolName) {
  ConstitutiveModel m("BadName");
  // The cas layer accepts any string for a symbol; the bad character only
  // manifests when codegen splices it into source — SymbolValidationPass
  // catches it first.
  m.add_parameter("bad-name", 1.0);
  auto x = m.add_scalar_input("x");
  m.add_output("y", x);
  try {
    m.validate();
    FAIL() << "expected validate() to reject the non-identifier name";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("bad-name"), std::string::npos) << msg;
    EXPECT_NE(msg.find("identifier"), std::string::npos) << msg;
  }
}

// ─── MooseMaterialTarget stateful-role guards ────────────────────────────

TEST(NegativePath, MooseRejectsStatefulInput) {
  ConstitutiveModel m("StatefulIn");
  auto h = m.add_tensor_input("h", 3, 2, roles::History); // is_stateful
  m.add_output("stress", h, roles::Stress);
  MooseMaterialTarget target;
  try {
    [[maybe_unused]] auto const discarded = target.emit(m);
    FAIL() << "expected emit() to reject the stateful input";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("stateful"), std::string::npos) << msg;
    EXPECT_NE(msg.find("input 'h'"), std::string::npos) << msg;
  }
}

// (The symmetric stateful-OUTPUT guard, issue #15, is unit-tested alongside the
// guard itself in MooseTargetTest.cpp.)

// ─── MooseMaterialTarget structural guards ───────────────────────────────

TEST(NegativePath, MooseRejectsEvolutionWithoutLocalNewton) {
  ConstitutiveModel m("EvoNoNewton");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", eps, roles::Stress);
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = m.add_scalar_state_variable("alpha", zero);
  auto rate = cas::make_expression<cas::scalar_constant>(0.5);
  m.add_scalar_evolution_equation(alpha, rate);
  // No enable_local_newton() — the MOOSE material has no external Newton loop,
  // so the synthesised residual/Jacobian outputs would never be solved.
  MooseMaterialTarget target;
  try {
    [[maybe_unused]] auto const discarded = target.emit(m);
    FAIL() << "expected emit() to reject evolution without local Newton";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("local Newton"), std::string::npos) << msg;
  }
}

TEST(NegativePath, MooseRejectsMultipleTangents) {
  ConstitutiveModel m("TwoTangents");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  m.add_algorithmic_tangent("dstress_deps2", "stress", "eps");
  MooseMaterialTarget target;
  try {
    [[maybe_unused]] auto const discarded = target.emit(m);
    FAIL() << "expected emit() to reject more than one consistent tangent";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("one consistent tangent"), std::string::npos) << msg;
  }
}

TEST(NegativePath, MooseRejectsJacobianMultOutputCollision) {
  ConstitutiveModel m("JacCollision");
  auto mu = m.add_parameter("mu", 0.5);
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  // A regular output named exactly like the framework consistent-tangent
  // member — legal on its own, illegal once a tangent is also requested.
  m.add_output("Jacobian_mult", eps);
  m.add_algorithmic_tangent("dstress_deps", "stress", "eps");
  MooseMaterialTarget target;
  try {
    [[maybe_unused]] auto const discarded = target.emit(m);
    FAIL() << "expected emit() to reject the Jacobian_mult output collision";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("Jacobian_mult"), std::string::npos) << msg;
  }
}

TEST(NegativePath, MooseRejectsUnsupportedTensorRank) {
  ConstitutiveModel m("Rank3");
  // rank-3 tensor: valid in cas / roles::Other passes validation, but MOOSE
  // has storage types only for rank 2 and rank 4.
  auto x = m.add_tensor_input("x", 3, 3, roles::Other);
  m.add_output("y", x, roles::Other);
  MooseMaterialTarget target;
  try {
    [[maybe_unused]] auto const discarded = target.emit(m);
    FAIL() << "expected emit() to reject the rank-3 tensor";
  } catch (std::runtime_error const &e) {
    std::string const msg = e.what();
    EXPECT_NE(msg.find("rank 3"), std::string::npos) << msg;
    EXPECT_NE(msg.find("no MOOSE storage type"), std::string::npos) << msg;
  }
}

// ─── Duplicate-declaration guards (add-time) ─────────────────────────────

TEST(NegativePath, DuplicateOutputRejected) {
  ConstitutiveModel m("DupOut");
  auto x = m.add_scalar_input("x");
  m.add_output("y", x);
  EXPECT_THROW(m.add_output("y", 2 * x), std::runtime_error);
}

TEST(NegativePath, DuplicateSymbolRejected) {
  ConstitutiveModel m("DupSym");
  [[maybe_unused]] auto const x = m.add_scalar_input("x");
  EXPECT_THROW(m.add_scalar_input("x"), std::runtime_error);
  EXPECT_THROW(m.add_parameter("x", 1.0), std::runtime_error);
}

// A symbol and an output cannot share a raw name (either declaration order) —
// backends derive both a C++ member and a MOOSE property name from each.
TEST(NegativePath, SymbolOutputNameClashRejected) {
  ConstitutiveModel m("Clash");
  auto x = m.add_scalar_input("x");
  m.add_output("shared", x);
  EXPECT_THROW(m.add_scalar_input("shared"), std::runtime_error);

  ConstitutiveModel m2("Clash2");
  [[maybe_unused]] auto const shared = m2.add_scalar_input("shared");
  auto y = m2.add_scalar_input("y");
  EXPECT_THROW(m2.add_output("shared", y), std::runtime_error);
}

} // namespace
} // namespace numsim::codegen
