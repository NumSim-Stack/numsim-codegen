// Phase 2.1 tests: StateVariable IR additions to ConstitutiveModel.
//
// Covers:
//   - add_scalar_state_variable / add_tensor_state_variable basic shape
//   - Returned handle has distinct current / previous expression names
//   - Both current and previous register as SymbolDecls with the right
//     Category (StateVariableCurrent / StateVariableOld)
//   - state_variables() accessor exposes the declarations
//   - SymbolValidationPass discovers state-var symbols (identifier check,
//     symbol_lookup population)
//   - find_tensor_symbol resolves tensor state-var current and previous
//
// Phase 2.2+ (TimeIntegrationPass, KuhnTuckerLoweringPass, LocalNewton)
// will land separately and test the actual semantics of how state vars
// flow through codegen. This file only exercises the IR layer.

#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_tags.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/recipe.h>

// Shared friend-access header for negative-path tests — see header
// comment for the ODR rationale (PR #66 round-3 review #7).
#include "internal/state_variable_alignment_access.h"

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

// ─── add_scalar_state_variable ──────────────────────────────────────

TEST(StateVariable, AddScalarReturnsCurrentAndPreviousHandles) {
  ConstitutiveModel model("J2Trial");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero,
                                               "equivalent plastic strain");

  // Two distinct expression handles: current and previous. Both are
  // scalar symbols with distinct printable names.
  ASSERT_TRUE(alpha.current.is_valid());
  ASSERT_TRUE(alpha.previous.is_valid());
  EXPECT_NE(&alpha.current.get(), &alpha.previous.get())
      << "current and previous must be distinct cas symbols";
}

TEST(StateVariable, AddScalarRegistersBothSymbolsInModel) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  // symbols() now contains alpha + alpha_old.
  ASSERT_EQ(model.symbols().size(), 2u);

  auto const &cur = model.symbols()[0];
  EXPECT_EQ(cur.name, "alpha");
  EXPECT_EQ(cur.category, SymbolDecl::Category::StateVariableCurrent);
  EXPECT_EQ(cur.kind, SymbolDecl::Kind::Scalar);

  auto const &old = model.symbols()[1];
  EXPECT_EQ(old.name, "alpha_old");
  EXPECT_EQ(old.category, SymbolDecl::Category::StateVariableOld);
  EXPECT_EQ(old.kind, SymbolDecl::Kind::Scalar);
}

TEST(StateVariable, ScalarStateVarAppearsInScalarSymbolMap) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  // Both alpha and alpha_old must be in the scalar symbol map so the
  // CodeGenContext can register their names for emit.
  ASSERT_EQ(model.scalar_symbol_map().size(), 2u);
  EXPECT_EQ(model.scalar_symbol_map()[0].first, "alpha");
  EXPECT_EQ(model.scalar_symbol_map()[1].first, "alpha_old");
}

// ─── add_tensor_state_variable ──────────────────────────────────────

TEST(StateVariable, AddTensorReturnsCurrentAndPreviousHandles) {
  ConstitutiveModel model("Damage");
  auto eps_p_init = cas::make_expression<cas::tensor_zero>(3, 2);
  auto eps_p =
      model.add_tensor_state_variable("eps_p", 3, 2, eps_p_init,
                                      "plastic strain tensor");

  ASSERT_TRUE(eps_p.current.is_valid());
  ASSERT_TRUE(eps_p.previous.is_valid());
  EXPECT_NE(&eps_p.current.get(), &eps_p.previous.get());
}

TEST(StateVariable, AddTensorRegistersBothSymbolsAndPreservesRank) {
  ConstitutiveModel model("M");
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);
  model.add_tensor_state_variable("eps_p", 3, 2, init);

  ASSERT_EQ(model.symbols().size(), 2u);
  EXPECT_EQ(model.symbols()[0].name, "eps_p");
  EXPECT_EQ(model.symbols()[0].dim, 3u);
  EXPECT_EQ(model.symbols()[0].rank, 2u);
  EXPECT_EQ(model.symbols()[1].name, "eps_p_old");
  EXPECT_EQ(model.symbols()[1].dim, 3u);
  EXPECT_EQ(model.symbols()[1].rank, 2u);
}

// ─── state_variables() accessor ─────────────────────────────────────

TEST(StateVariable, StateVariablesAccessorExposesDeclarations) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);

  model.add_scalar_state_variable("alpha", zero, "scalar SV");
  model.add_tensor_state_variable("eps_p", 3, 2, init, "tensor SV");

  ASSERT_EQ(model.state_variables().size(), 2u);
  EXPECT_EQ(model.state_variables()[0].name, "alpha");
  EXPECT_EQ(model.state_variables()[0].kind, SymbolDecl::Kind::Scalar);
  EXPECT_EQ(model.state_variables()[0].doc, "scalar SV");
  EXPECT_EQ(model.state_variables()[1].name, "eps_p");
  EXPECT_EQ(model.state_variables()[1].kind, SymbolDecl::Kind::Tensor);
  EXPECT_EQ(model.state_variables()[1].rank, 2u);
}

TEST(StateVariable, EmptyRecipeHasNoStateVariables) {
  ConstitutiveModel model("ElasticOnly");
  EXPECT_TRUE(model.state_variables().empty());
}

// ─── RecipeView delegate ────────────────────────────────────────────

TEST(StateVariable, RecipeViewDelegatesStateVariables) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  RecipeView view(model);
  ASSERT_EQ(view.state_variables().size(), 1u);
  EXPECT_EQ(view.state_variables()[0].name, "alpha");
}

// ─── Pass framework integration ─────────────────────────────────────

TEST(StateVariable, SymbolValidationPassDiscoversStateVarSymbols) {
  // After SymbolValidationPass runs, pctx.symbol_lookup must contain
  // both current and previous symbols. Phase 2 mutating passes will
  // consume this for expression rewrites (e.g. TimeIntegrationPass
  // lowering Dt(α) → (α − α_old) / dt needs to resolve both names).
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);

  EXPECT_TRUE(pctx.symbol_lookup.contains("alpha"));
  EXPECT_TRUE(pctx.symbol_lookup.contains("alpha_old"));
}

TEST(StateVariable, FindTensorSymbolResolvesTensorStateVarBothHandles) {
  ConstitutiveModel model("M");
  auto init = cas::make_expression<cas::tensor_zero>(3, 2);
  model.add_tensor_state_variable("eps_p", 3, 2, init);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);

  auto cur = find_tensor_symbol(pctx, "eps_p");
  ASSERT_TRUE(cur.has_value());
  EXPECT_EQ((*cur)->category, SymbolDecl::Category::StateVariableCurrent);

  auto old = find_tensor_symbol(pctx, "eps_p_old");
  ASSERT_TRUE(old.has_value());
  EXPECT_EQ((*old)->category, SymbolDecl::Category::StateVariableOld);
}

// ─── Name-collision protection (M1+M2 in REVIEW-pr-58.md) ───────────

TEST(StateVariable, DuplicateScalarStateVarNameThrows) {
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  try {
    model.add_scalar_state_variable("alpha", zero);
    FAIL() << "expected runtime_error for duplicate state-var name";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("alpha"), std::string::npos) << msg;
    EXPECT_NE(msg.find("already declared"), std::string::npos) << msg;
  }
}

TEST(StateVariable, InputThenStateVarWithOldCollisionThrows) {
  // M2: user declared `alpha_old` as an input; then add_*_state_variable
  // for `alpha` would auto-generate the same `alpha_old` symbol.
  ConstitutiveModel model("M");
  model.add_scalar_input("alpha_old");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);

  try {
    model.add_scalar_state_variable("alpha", zero);
    FAIL() << "expected runtime_error for _old name collision";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("alpha_old"), std::string::npos) << msg;
    EXPECT_NE(msg.find("already declared"), std::string::npos) << msg;
  }
}

TEST(StateVariable, StateVarThenInputWithOldCollisionThrows) {
  // Reverse order of the previous test: declare state var first,
  // then try to add an input with the reserved `_old` suffix.
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  try {
    model.add_scalar_input("alpha_old");
    FAIL() << "expected runtime_error for input clashing with state-var _old";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("alpha_old"), std::string::npos) << msg;
  }
}

TEST(StateVariable, DuplicateInputNameThrows) {
  // Sibling to the state-var case — input-vs-input collision was also
  // unprotected pre-fix.
  ConstitutiveModel model("M");
  model.add_scalar_input("x");
  EXPECT_THROW(model.add_scalar_input("x"), std::runtime_error);
}

// ─── Edge cases (M4) ────────────────────────────────────────────────

TEST(StateVariable, StateVarsOnlyNoOutputsRecipeBuilds) {
  // A recipe with state variables but no outputs should still validate
  // and emit (the compute function will be a no-op body with state-var
  // arguments — fine until Phase 2.2's lowering populates the body).
  ConstitutiveModel model("StateOnly");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);
  EXPECT_NO_THROW(model.validate());
  EXPECT_NO_THROW(model.emit_compute_function());
}

TEST(StateVariable, MixedInputAndStateVarInOutput) {
  // The pre-fix coverage exercised parameter+state-var; this covers
  // the input+state-var combination, which goes through a different
  // SymbolDecl::Category and exercises both the StateVariableCurrent
  // and Input paths through SymbolValidationPass.
  ConstitutiveModel model("HardeningWithStrain");
  auto eps = model.add_scalar_input("eps");
  auto K = model.add_parameter("K", 1.0);
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);

  // sigma = K * (eps + alpha.current) — uses input + parameter +
  // state-var current in one expression.
  model.add_output("sigma", K * (eps + alpha.current));
  EXPECT_NO_THROW(model.emit_compute_function());
}

// ─── StateVariable carries paired symbol indices (architect Q1) ─────

TEST(StateVariable, StoresSymbolIndicesForLowering) {
  // Architect Q1 stronger fix: StateVariable records `current_symbol_idx`
  // and `old_symbol_idx`. Phase 2.2's TimeIntegrationPass uses these
  // instead of re-deriving via `name + state_variable_old_suffix`.
  ConstitutiveModel model("M");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  model.add_scalar_state_variable("alpha", zero);

  ASSERT_EQ(model.state_variables().size(), 1u);
  auto const &sv = model.state_variables()[0];
  ASSERT_LT(sv.current_symbol_idx, model.symbols().size());
  ASSERT_LT(sv.old_symbol_idx, model.symbols().size());

  EXPECT_EQ(model.symbols()[sv.current_symbol_idx].name, "alpha");
  EXPECT_EQ(model.symbols()[sv.current_symbol_idx].category,
            SymbolDecl::Category::StateVariableCurrent);
  EXPECT_EQ(model.symbols()[sv.old_symbol_idx].name, "alpha_old");
  EXPECT_EQ(model.symbols()[sv.old_symbol_idx].category,
            SymbolDecl::Category::StateVariableOld);
}

TEST(StateVariable, StateVarUsableInOutputExpression) {
  // End-to-end: build a recipe that uses a state variable in an output
  // expression. The compute function must emit successfully (no
  // undeclared-symbol error from SymbolValidationPass).
  ConstitutiveModel model("Hardening");
  auto K = model.add_parameter("K", 1.0, "hardening modulus");
  auto zero = cas::make_expression<cas::scalar_constant>(0.0);
  auto alpha = model.add_scalar_state_variable("alpha", zero);

  // sigma_y = K * alpha (linear hardening on the current step's alpha)
  auto sigma_y = K * alpha.current;
  model.add_output("sigma_y", sigma_y);

  auto src = model.emit_compute_function();
  EXPECT_NE(src.find("Hardening_compute"), std::string::npos) << src;
  // M5 (REVIEW-pr-58.md): Phase 2.1 deliberately renders BOTH the
  // current and the _old symbols as `const` value parameters. This is
  // semantically wrong for `alpha` (it's what we solve for, should be
  // an out-param) — Phase 2.2's TimeIntegrationPass + Phase 2.6's
  // Newton lowering will rewire `StateVariableCurrent` to a
  // declareProperty / out-arg form. Pin the current behaviour so the
  // Phase-2.2 transition is a deliberate test rewrite, not a silent
  // regression.
  // TODO(phase-2.2): replace these assertions when render_compute_function
  // gains a Category-aware branch for StateVariableCurrent.
  EXPECT_NE(src.find("double const alpha"), std::string::npos)
      << "current value still appears as const input param in Phase 2.1: " << src;
  EXPECT_NE(src.find("double const alpha_old"), std::string::npos)
      << "_old value appears as const input param: " << src;
}

// ─── Phase 2.2 prep (issue #59) ──────────────────────────────────────

TEST(StateVariablePhase22Prep, AlignmentInvariantValidatesScalarAndTensor) {
  // Item 1 / REVIEW-pr-58.md m1: validate() runs the
  // validate_state_variable_symbol_alignment() invariant check before
  // the pass pipeline. A well-formed recipe with both scalar and tensor
  // state variables (interleaved with inputs + parameters so indices
  // land at non-trivial offsets) must pass.
  ConstitutiveModel model("M");
  auto K = model.add_parameter("K", 1.0);
  (void)model.add_scalar_input("eps_v");
  auto alpha = model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  (void)K;
  (void)alpha;
  auto eps_p_init = cas::make_expression<cas::tensor_zero>(3, 2);
  (void)model.add_tensor_state_variable("eps_p", 3, 2, eps_p_init);

  EXPECT_NO_THROW(model.validate());

  // Indices land where add_*_state_variable claims they do:
  for (auto const &sv : model.state_variables()) {
    auto const &cur = model.symbols()[sv.current_symbol_idx];
    auto const &old = model.symbols()[sv.old_symbol_idx];
    EXPECT_EQ(cur.name, sv.name);
    EXPECT_EQ(old.name, sv.name + "_old");
    EXPECT_EQ(cur.kind, sv.kind);
    EXPECT_EQ(old.kind, sv.kind);
    EXPECT_EQ(cur.category, SymbolDecl::Category::StateVariableCurrent);
    EXPECT_EQ(old.category, SymbolDecl::Category::StateVariableOld);
    if (sv.kind == SymbolDecl::Kind::Tensor) {
      EXPECT_EQ(cur.dim, sv.dim);
      EXPECT_EQ(cur.rank, sv.rank);
      EXPECT_EQ(old.dim, sv.dim);
      EXPECT_EQ(old.rank, sv.rank);
    }
  }
}

TEST(StateVariablePhase22Prep, SymbolValidationPassAdvertisesStateVarTags) {
  // PR #66 review: tag split. SymbolValidationPass advertises
  // `state_variables_checked` unconditionally and
  // `state_variables_non_empty` only when the recipe actually has
  // state variables. PassManager queries postconditions() AFTER run(),
  // so the conditional advertisement is observable through it.
  auto has = [](auto const &v, std::string_view t) {
    for (auto const &x : v) {
      if (x == t) return true;
    }
    return false;
  };

  // PR #66 round-2 #6: pin the pre-`run()` contract. A pristine pass
  // instance reports the safe (no non_empty) shape. Without this test
  // a future refactor that lazy-inits in `run()` could silently regress
  // the safe default.
  {
    SymbolValidationPass pristine;
    auto const pre_run = pristine.postconditions();
    EXPECT_TRUE(has(pre_run, pass_tags::state_variables_checked));
    EXPECT_FALSE(has(pre_run, pass_tags::state_variables_non_empty))
        << "non_empty tag must NOT fire before run()";
  }

  // Pure-elasticity recipe: only the always-on tag advertised.
  {
    ConstitutiveModel pure_elastic("E");
    (void)pure_elastic.add_parameter("mu", 0.5);
    PassContext pctx{RecipeView{pure_elastic}, CodeGenContext{},
                     std::nullopt, {}};
    SymbolValidationPass pass;
    pass.run(pctx);
    auto const post = pass.postconditions();
    EXPECT_TRUE(has(post, pass_tags::state_variables_checked));
    EXPECT_FALSE(has(post, pass_tags::state_variables_non_empty))
        << "non_empty tag must NOT fire for state-var-free recipes";
    // validate() itself must still succeed:
    EXPECT_NO_THROW(pure_elastic.validate());
  }

  // Recipe with at least one state variable: both tags advertised.
  {
    ConstitutiveModel hardening("H");
    (void)hardening.add_scalar_state_variable(
        "alpha", cas::make_expression<cas::scalar_constant>(0.0));
    PassContext pctx{RecipeView{hardening}, CodeGenContext{},
                     std::nullopt, {}};
    SymbolValidationPass pass;
    pass.run(pctx);
    auto const post = pass.postconditions();
    EXPECT_TRUE(has(post, pass_tags::state_variables_checked));
    EXPECT_TRUE(has(post, pass_tags::state_variables_non_empty));
  }

  // Same-instance reuse: PR #66 round-3 #5. Mode (b) of the reset
  // comment in `SymbolValidationPass::run` ("same instance reused
  // across recipes with different state-var counts") is exercised
  // here. Without the reset-at-entry the second `run()` would leave
  // the non-empty tag advertised from the first run.
  {
    SymbolValidationPass reused;

    ConstitutiveModel with_sv("S");
    (void)with_sv.add_scalar_state_variable(
        "alpha", cas::make_expression<cas::scalar_constant>(0.0));
    ConstitutiveModel pure("P");
    (void)pure.add_parameter("k", 1.0);

    // First run: state-var recipe → both tags advertised.
    {
      PassContext pctx1{RecipeView{with_sv}, CodeGenContext{},
                        std::nullopt, {}};
      reused.run(pctx1);
      auto const post1 = reused.postconditions();
      EXPECT_TRUE(has(post1, pass_tags::state_variables_checked));
      EXPECT_TRUE(has(post1, pass_tags::state_variables_non_empty));
    }

    // Second run on the SAME instance: pure-elasticity recipe →
    // non-empty tag must be cleared.
    {
      PassContext pctx2{RecipeView{pure}, CodeGenContext{},
                        std::nullopt, {}};
      reused.run(pctx2);
      auto const post2 = reused.postconditions();
      EXPECT_TRUE(has(post2, pass_tags::state_variables_checked));
      EXPECT_FALSE(has(post2, pass_tags::state_variables_non_empty))
          << "reset-at-entry must clear stale non-empty advert "
             "when the same pass instance is reused across recipes "
             "with different state-var counts (mode b)";
    }

    // Reverse order on the SAME instance for symmetry — pure first,
    // then state-var. Catches a hypothetical "lazy-init that only
    // initialises once" regression.
    {
      PassContext pctx3{RecipeView{pure}, CodeGenContext{},
                        std::nullopt, {}};
      reused.run(pctx3);
      EXPECT_FALSE(has(reused.postconditions(),
                       pass_tags::state_variables_non_empty));
      PassContext pctx4{RecipeView{with_sv}, CodeGenContext{},
                        std::nullopt, {}};
      reused.run(pctx4);
      EXPECT_TRUE(has(reused.postconditions(),
                      pass_tags::state_variables_non_empty));
    }
  }
}

TEST(StateVariablePhase22Prep, FindStateVariableByName) {
  // Item 3 / REVIEW-pr-58.md m3: find_state_variable_by_name resolves
  // a name to its StateVariable record. Returns nullptr on miss.
  ConstitutiveModel model("M");
  (void)model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  auto eps_p_init = cas::make_expression<cas::tensor_zero>(3, 2);
  (void)model.add_tensor_state_variable("eps_p", 3, 2, eps_p_init);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};

  auto const *alpha_sv = find_state_variable_by_name(pctx, "alpha");
  ASSERT_NE(alpha_sv, nullptr);
  EXPECT_EQ(alpha_sv->name, "alpha");
  EXPECT_EQ(alpha_sv->kind, SymbolDecl::Kind::Scalar);

  auto const *eps_p_sv = find_state_variable_by_name(pctx, "eps_p");
  ASSERT_NE(eps_p_sv, nullptr);
  EXPECT_EQ(eps_p_sv->name, "eps_p");
  EXPECT_EQ(eps_p_sv->kind, SymbolDecl::Kind::Tensor);

  // The paired `_old` symbol is NOT itself a StateVariable record — it
  // is a SymbolDecl whose owning StateVariable is named without the
  // suffix. Looking up by the suffixed name must miss.
  EXPECT_EQ(find_state_variable_by_name(pctx, "alpha_old"), nullptr);

  // Unrelated names miss.
  EXPECT_EQ(find_state_variable_by_name(pctx, "nope"), nullptr);
  EXPECT_EQ(find_state_variable_by_name(pctx, ""), nullptr);
}

// ─── Negative-path tests for the alignment invariant ────────────────
//
// `verify_state_variable_symbol_alignment` has 5 distinct throw branches.
// The public ConstitutiveModel API cannot violate the invariant today,
// so the only way to verify each branch fires correctly is to mutate
// the internals directly via the `testing_detail::state_variable_alignment_access`
// friend struct. Its definition lives in `tests/internal/` (PR #66
// round-3 #7) so future test TUs that need the same access include the
// shared header rather than redefining the struct — making the ODR
// contract mechanical, not social.

namespace {
// Anonymous-namespace alias (PR #66 round-3 #8) so a future test TU
// using `using poison = ...` for a different access struct doesn't
// quietly mean different things across TUs.
using sv_align_poison =
    testing_detail::state_variable_alignment_access;
} // namespace

TEST(StateVariablePhase22Prep, AlignmentDetectsCorruptionOutOfBoundsIdx) {
  // PR #66 round-2 #7: single try/catch + FAIL() — silent test-pass on
  // non-throw is impossible. Negative-substring asserts pin which arm
  // fires (round-2 #8).
  ConstitutiveModel model("M");
  (void)model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  sv_align_poison::poison_state_var_current_idx(model, 0, 999);
  try {
    model.validate();
    FAIL() << "expected std::runtime_error from alignment OOB arm";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("out of bounds"), std::string::npos) << what;
    EXPECT_NE(what.find("alpha"), std::string::npos) << what;
    // Pin: no other arm should have fired.
    EXPECT_EQ(what.find("wrong Category"), std::string::npos) << what;
    EXPECT_EQ(what.find("Kind mismatched"), std::string::npos) << what;
    EXPECT_EQ(what.find("dim/rank"), std::string::npos) << what;
  }
}

TEST(StateVariablePhase22Prep, AlignmentDetectsCorruptionRenamedSymbol) {
  ConstitutiveModel model("M");
  (void)model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  auto const cur_idx = model.state_variables()[0].current_symbol_idx;
  sv_align_poison::poison_symbol_name(model, cur_idx, "beta");
  try {
    model.validate();
    FAIL() << "expected std::runtime_error from alignment name arm";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("named 'beta'"), std::string::npos) << what;
    EXPECT_NE(what.find("expected 'alpha'"), std::string::npos) << what;
    EXPECT_EQ(what.find("wrong Category"), std::string::npos) << what;
    EXPECT_EQ(what.find("Kind mismatched"), std::string::npos) << what;
    EXPECT_EQ(what.find("dim/rank"), std::string::npos) << what;
    EXPECT_EQ(what.find("out of bounds"), std::string::npos) << what;
  }
}

TEST(StateVariablePhase22Prep, AlignmentDetectsCorruptionWrongCategory) {
  ConstitutiveModel model("M");
  (void)model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  auto const cur_idx = model.state_variables()[0].current_symbol_idx;
  sv_align_poison::poison_symbol_category(model, cur_idx,
                                 SymbolDecl::Category::Input);
  try {
    model.validate();
    FAIL() << "expected std::runtime_error from alignment category arm";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    // The message must print both observed and expected enum values
    // (PR #66 round-1 #4).
    EXPECT_NE(what.find("wrong Category"), std::string::npos) << what;
    EXPECT_NE(what.find("expected="), std::string::npos) << what;
    EXPECT_NE(what.find("got="), std::string::npos) << what;
    EXPECT_EQ(what.find("Kind mismatched"), std::string::npos) << what;
    EXPECT_EQ(what.find("dim/rank"), std::string::npos) << what;
    EXPECT_EQ(what.find("out of bounds"), std::string::npos) << what;
  }
}

TEST(StateVariablePhase22Prep, AlignmentDetectsCorruptionWrongKind) {
  ConstitutiveModel model("M");
  (void)model.add_scalar_state_variable(
      "alpha", cas::make_expression<cas::scalar_constant>(0.0));
  auto const cur_idx = model.state_variables()[0].current_symbol_idx;
  sv_align_poison::poison_symbol_kind(model, cur_idx, SymbolDecl::Kind::Tensor);
  try {
    model.validate();
    FAIL() << "expected std::runtime_error from alignment kind arm";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("Kind mismatched"), std::string::npos) << what;
    EXPECT_NE(what.find("expected="), std::string::npos) << what;
    EXPECT_NE(what.find("got="), std::string::npos) << what;
    // Pin: must be the kind arm, not the dim arm — the dim arm is
    // gated on `sv.kind == Tensor` (record side, still Scalar here);
    // a future reorder that swapped gating could silently still pass.
    EXPECT_EQ(what.find("dim/rank"), std::string::npos) << what;
    EXPECT_EQ(what.find("wrong Category"), std::string::npos) << what;
    EXPECT_EQ(what.find("out of bounds"), std::string::npos) << what;
  }
}

TEST(StateVariablePhase22Prep, AlignmentDetectsCorruptionWrongDim) {
  ConstitutiveModel model("M");
  auto eps_p_init = cas::make_expression<cas::tensor_zero>(3, 2);
  (void)model.add_tensor_state_variable("eps_p", 3, 2, eps_p_init);
  auto const cur_idx = model.state_variables()[0].current_symbol_idx;
  sv_align_poison::poison_symbol_dim(model, cur_idx, 2);
  try {
    model.validate();
    FAIL() << "expected std::runtime_error from alignment dim arm";
  } catch (std::runtime_error const &e) {
    std::string const what = e.what();
    EXPECT_NE(what.find("dim/rank"), std::string::npos) << what;
    EXPECT_NE(what.find("eps_p"), std::string::npos) << what;
    EXPECT_EQ(what.find("wrong Category"), std::string::npos) << what;
    EXPECT_EQ(what.find("Kind mismatched"), std::string::npos) << what;
    EXPECT_EQ(what.find("out of bounds"), std::string::npos) << what;
  }
}

} // namespace numsim::codegen
