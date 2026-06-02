// Tests for the Phase 1.2 pass framework: PassManager ordering / precondition
// enforcement, and the two concrete passes (SymbolValidationPass with the new
// identifier-validity check; CodeEmitPass producing a non-empty compute
// function). End-to-end coverage that the framework integrates with the recipe
// emit path is already exercised by RecipeTest / MooseTargetTest / the
// compile-check driver — this file isolates the framework itself.

#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/passes/tensor_space_consistency_pass.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <gtest/gtest.h>

#include <string>

namespace numsim::codegen {

namespace {

// Test-only pass that records its name into a shared trace vector. Used to
// verify that PassManager calls passes in registration order.
class TracePass final : public Pass {
public:
  TracePass(std::string n, std::vector<std::string> &trace,
            std::vector<std::string_view> pre,
            std::vector<std::string_view> post)
      : m_name(std::move(n)), m_trace(&trace), m_pre(std::move(pre)),
        m_post(std::move(post)) {}

  [[nodiscard]] auto name() const -> std::string_view override {
    return m_name;
  }
  [[nodiscard]] auto preconditions() const
      -> std::vector<std::string_view> override {
    return m_pre;
  }
  [[nodiscard]] auto postconditions() const
      -> std::vector<std::string_view> override {
    return m_post;
  }
  void run(PassContext & /*pctx*/) override { m_trace->push_back(m_name); }

private:
  std::string m_name;
  std::vector<std::string> *m_trace;
  std::vector<std::string_view> m_pre;
  std::vector<std::string_view> m_post;
};

// Build a recipe shell that satisfies SymbolValidationPass — no outputs,
// trivially valid. Some tests fabricate a PassContext directly; this gives
// them a minimal `ConstitutiveModel const&` to reference.
auto make_empty_model() -> ConstitutiveModel { return ConstitutiveModel("M"); }

} // namespace

// ─── RecipeView (M4) ─────────────────────────────────────────────────

TEST(RecipeView, DelegatesNameSymbolsOutputs) {
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);

  RecipeView view(model);
  EXPECT_EQ(view.name(), "LinearElasticShear");
  EXPECT_EQ(view.symbols().size(), 2u); // mu (parameter) + eps (input)
  EXPECT_EQ(view.outputs().size(), 1u); // stress
  EXPECT_EQ(view.scalar_symbol_map().size(), 1u); // mu
  EXPECT_EQ(view.tensor_symbol_map().size(), 1u); // eps
}

TEST(RecipeView, RawModelEscapeHatchExposesFullRecipe) {
  // Until RecipeView's surface widens, callers needing parameters()/
  // inputs()/find_*_by_role go through raw_model(). Verify the hatch
  // returns the same ConstitutiveModel we constructed it with.
  ConstitutiveModel model("M");
  RecipeView view(model);
  EXPECT_EQ(&view.raw_model(), &model);
}

TEST(RecipeView, ConstConstructorGivesNoMutableAccess) {
  // P1: a view constructed from `ConstitutiveModel const &` reports no
  // mutable model. Phase 2 passes that need mutation throw on this
  // rather than silently proceeding through a const path.
  ConstitutiveModel model("M");
  ConstitutiveModel const &cref = model;
  RecipeView view(cref);
  EXPECT_EQ(view.try_mutable_model(), nullptr);
  // const surface still works.
  EXPECT_EQ(view.name(), "M");
}

TEST(RecipeView, MutableConstructorGivesMutableAccess) {
  // P1: a view constructed from `ConstitutiveModel &` exposes the
  // original mutable pointer for Phase 2 mutating passes.
  ConstitutiveModel model("M");
  RecipeView view(model);
  EXPECT_EQ(view.try_mutable_model(), &model);
  // const surface still works.
  EXPECT_EQ(view.name(), "M");
}

TEST(RecipeView, RequireMutableModelThrowsOnConstView) {
  // m1: the canonical helper that Phase 2 mutating passes will call.
  // Throws with the pass name + a pipeline-misconfiguration message.
  ConstitutiveModel model("M");
  RecipeView view(static_cast<ConstitutiveModel const &>(model));
  try {
    view.require_mutable_model("TestMutatingPass");
    FAIL() << "expected runtime_error when require_mutable on a const view";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("TestMutatingPass"), std::string::npos) << msg;
    EXPECT_NE(msg.find("mutable RecipeView"), std::string::npos) << msg;
  }
}

TEST(RecipeView, RequireMutableModelReturnsRefOnMutableView) {
  ConstitutiveModel model("M");
  RecipeView view(model);
  EXPECT_EQ(&view.require_mutable_model("TestPass"), &model);
}

TEST(RecipeView, BothCtorPathsAgreeOnRawModel) {
  // Sanity: regardless of how the view was constructed, raw_model()
  // returns the same underlying recipe.
  ConstitutiveModel model("M");
  RecipeView mut_view(model);
  RecipeView const_view(static_cast<ConstitutiveModel const &>(model));
  EXPECT_EQ(&mut_view.raw_model(), &const_view.raw_model());
}

// ─── PassManager ─────────────────────────────────────────────────────

TEST(PassManager, RunsPassesInRegistrationOrder) {
  std::vector<std::string> trace;
  PassManager pm;
  pm.emplace<TracePass>("A", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});
  pm.emplace<TracePass>("B", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});
  pm.emplace<TracePass>("C", trace, std::vector<std::string_view>{},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  pm.run(pctx);

  ASSERT_EQ(trace.size(), 3u);
  EXPECT_EQ(trace[0], "A");
  EXPECT_EQ(trace[1], "B");
  EXPECT_EQ(trace[2], "C");
}

TEST(PassManager, EnforcesPrecondition) {
  std::vector<std::string> trace;
  PassManager pm;
  // Pass that needs "validated" before running, but no earlier pass advertises
  // it.
  pm.emplace<TracePass>("Bad", trace,
                        std::vector<std::string_view>{"validated"},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error for unmet precondition";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("validated"), std::string::npos) << msg;
    EXPECT_NE(msg.find("Bad"), std::string::npos) << msg;
  }
  EXPECT_TRUE(trace.empty()) << "Bad pass should not have run.";
}

TEST(PassManager, SatisfiedPreconditionAllowsRun) {
  std::vector<std::string> trace;
  PassManager pm;
  pm.emplace<TracePass>(
      "A", trace, std::vector<std::string_view>{},
      std::vector<std::string_view>{"feature-x"});
  pm.emplace<TracePass>("B", trace,
                        std::vector<std::string_view>{"feature-x"},
                        std::vector<std::string_view>{});

  auto model = make_empty_model();
  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  pm.run(pctx);

  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0], "A");
  EXPECT_EQ(trace[1], "B");
}

TEST(PassManager, RejectsNullPass) {
  PassManager pm;
  EXPECT_THROW(pm.add(nullptr), std::invalid_argument);
}

// ─── SymbolValidationPass — identifier validity ──────────────────────

TEST(SymbolValidationPass, AcceptsValidIdentifiers) {
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("x"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("eps_p"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("_internal"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("a1b2c3"));
  EXPECT_TRUE(SymbolValidationPass::is_valid_cxx_identifier("CamelCase"));
}

TEST(SymbolValidationPass, RejectsCxxKeywords) {
  // The pre-fix check accepted these — they look like valid identifiers
  // per the [A-Za-z_][A-Za-z0-9_]* regex but the compiler refuses them
  // as variable names. We catch them at validate() time instead.
  for (auto const *kw : {"class", "if", "for", "auto", "return", "template",
                         "void", "int", "do", "co_await"}) {
    EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier(kw))
        << "should reject keyword '" << kw << "'";
    EXPECT_TRUE(SymbolValidationPass::is_cxx_keyword(kw))
        << "should classify '" << kw << "' as keyword";
  }
}

TEST(SymbolValidationPass, NonKeywordPrefixesAreNotKeywords) {
  // Spot-check that the binary-search bounds are right — `classroom`
  // shares a prefix with `class` but isn't itself a keyword.
  EXPECT_FALSE(SymbolValidationPass::is_cxx_keyword("classroom"));
  EXPECT_FALSE(SymbolValidationPass::is_cxx_keyword("ifelse"));
  EXPECT_FALSE(SymbolValidationPass::is_cxx_keyword(""));
  EXPECT_FALSE(SymbolValidationPass::is_cxx_keyword("eps_p"));
}

TEST(SymbolValidationPass, IdentifierCheckIsLocaleImmune) {
  // The pre-fix check used std::isalpha which is locale-sensitive — a
  // Turkish locale would reject 'I', and locales that classify accented
  // chars as alpha would accept non-ASCII inputs. The post-fix uses
  // explicit ASCII ranges. Verify non-ASCII strings get rejected.
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("\xc3\xa9psilon"))
      << "é-prefixed identifier (UTF-8)";
  // Split hex escape from following alpha chars — `\xb1bc` would
  // otherwise be parsed as one 4-digit hex value > 0xFF.
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("\xce\xb1" "bc"))
      << "α-prefixed identifier (UTF-8)";
}

TEST(SymbolValidationPass, RejectsRecipeWithKeywordName) {
  // End-to-end through validate() — the keyword check must fire from
  // the pass pipeline, not just the static helper.
  ConstitutiveModel model("M");
  model.add_parameter("class", 1.0, "keyword on purpose");

  try {
    model.validate();
    FAIL() << "expected runtime_error for keyword parameter name";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("class"), std::string::npos) << msg;
    EXPECT_NE(msg.find("keyword"), std::string::npos)
        << "throw message should mention keyword: " << msg;
  }
}

TEST(SymbolValidationPass, RejectsInvalidIdentifiers) {
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier(""));
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("1var"))
      << "digit-leading";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("my-tensor"))
      << "hyphen";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("x.y"))
      << "dot";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("a b"))
      << "space";
  EXPECT_FALSE(SymbolValidationPass::is_valid_cxx_identifier("a$b"))
      << "dollar (compiler-extension, not std-portable)";
}

TEST(SymbolValidationPass, RejectsRecipeWithInvalidScalarName) {
  // Use the cas tensor name-only-as-identifier-bearing field directly. The
  // recipe.h add_* methods take a `name` argument that becomes the cas
  // symbol's printable name — passing "my-param" here exercises the
  // identifier check at validate() time.
  ConstitutiveModel model("M");
  model.add_parameter("my-param", 1.0, "bad name on purpose");

  try {
    model.validate();
    FAIL() << "expected runtime_error for invalid scalar identifier";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("my-param"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not a usable C++ identifier"), std::string::npos) << msg;
  }
}

TEST(SymbolValidationPass, RejectsRecipeWithInvalidTensorName) {
  ConstitutiveModel model("M");
  model.add_tensor_input("eps.bad", 3, 2, roles::Strain);

  try {
    model.validate();
    FAIL() << "expected runtime_error for invalid tensor identifier";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("eps.bad"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not a usable C++ identifier"), std::string::npos) << msg;
  }
}

TEST(SymbolValidationPass, PopulatesPassContextSymbolLookup) {
  // P4: SymbolValidationPass builds pctx.symbol_lookup so downstream
  // passes can resolve name → SymbolDecl in O(1) instead of an O(N)
  // scan. Verify the map contains every declared symbol after the pass
  // runs.
  //
  // C1 fixup: the value type is `std::size_t` index into `model.symbols()`,
  // not a pointer, so vector growth (Phase 2 synthetic symbols) won't
  // dangle. Direct map access verifies the indices; `find_tensor_symbol`
  // (next test) verifies the canonical resolver.
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass pass;
  pass.run(pctx);

  ASSERT_EQ(pctx.symbol_lookup.size(), 2u); // mu + eps
  auto mu_it = pctx.symbol_lookup.find("mu");
  ASSERT_NE(mu_it, pctx.symbol_lookup.end());
  EXPECT_LT(mu_it->second, model.symbols().size());
  auto const &mu_decl = model.symbols()[mu_it->second];
  EXPECT_EQ(mu_decl.kind, SymbolDecl::Kind::Scalar);
  EXPECT_EQ(mu_decl.category, SymbolDecl::Category::Parameter);

  auto eps_it = pctx.symbol_lookup.find("eps");
  ASSERT_NE(eps_it, pctx.symbol_lookup.end());
  EXPECT_LT(eps_it->second, model.symbols().size());
  auto const &eps_decl = model.symbols()[eps_it->second];
  EXPECT_EQ(eps_decl.kind, SymbolDecl::Kind::Tensor);
  EXPECT_EQ(eps_decl.category, SymbolDecl::Category::Input);
  EXPECT_EQ(eps_decl.role.name, "strain");
}

TEST(SymbolValidationPass, FindTensorSymbolResolvesByName) {
  // M3 fixup: the canonical `find + kind == Tensor` helper. Returns a
  // valid SymbolDecl* for tensors, nullptr for scalars and unknowns.
  ConstitutiveModel model("M");
  model.add_parameter("mu", 0.5, "");
  model.add_tensor_input("eps", 3, 2, roles::Strain);

  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  SymbolValidationPass{}.run(pctx);

  auto eps = find_tensor_symbol(pctx, "eps");
  ASSERT_TRUE(eps.has_value());
  EXPECT_EQ((*eps)->name, "eps");
  EXPECT_EQ((*eps)->kind, SymbolDecl::Kind::Tensor);

  // Issue #62 item 1: the two failure modes are now distinguishable.
  // Scalar parameter — exists but wrong kind.
  auto mu = find_tensor_symbol(pctx, "mu");
  ASSERT_FALSE(mu.has_value());
  EXPECT_EQ(mu.error(), LookupError::WrongKind);

  // Unknown name — not in the lookup at all.
  auto missing = find_tensor_symbol(pctx, "nonexistent");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error(), LookupError::NotFound);
}

TEST(SymbolValidationPass, PreservesMissingSymbolDiagnostic) {
  // Pre-1.2 validate() threw on outputs referencing undeclared symbols.
  // The pass framework must preserve that behaviour.
  ConstitutiveModel model("M");
  // Output references a symbol that was never declared on the model.
  auto x = cas::make_expression<cas::scalar>("undeclared_x");
  model.add_output("y", x, roles::Other);

  try {
    model.validate();
    FAIL() << "expected runtime_error for undeclared symbol";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("undeclared_x"), std::string::npos) << msg;
  }
}

// ─── TensorSpaceConsistencyPass (M6) ─────────────────────────────────

TEST(TensorSpaceConsistencyPass, PassesWithoutSpaceAnnotations) {
  // The default Phase 1.2 recipe declares tensors without any tensor_space
  // annotation. The consistency pass must silently accept them — there's
  // nothing to contradict.
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);
  EXPECT_NO_THROW(model.validate());
}

TEST(TensorSpaceConsistencyPass, ThrowsOnSymmetricRoleWithSkewSpace) {
  // Construct a tensor symbol whose Role declares symmetric (roles::Strain)
  // but whose cas tensor_space annotation declares Skew perm. The pass
  // must catch the contradiction.
  ConstitutiveModel model("M");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);

  // Mutate the underlying tensor's space to introduce the contradiction.
  // `eps` is a non-const expression_holder, so get<T>() resolves to the
  // non-const overload and returns T&. In production code this annotation
  // flows from a cas `assume_*()` helper or a constructive operator.
  eps.get<cas::tensor_expression>().set_space(
      cas::tensor_space{cas::Skew{}, cas::AnyTraceTag{}});

  try {
    model.validate();
    FAIL() << "expected runtime_error for skew tensor with symmetric role";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("eps"), std::string::npos) << msg;
    EXPECT_NE(msg.find("Skew"), std::string::npos) << msg;
    EXPECT_NE(msg.find("symmetric"), std::string::npos) << msg;
  }
}

TEST(TensorSpaceConsistencyPass, ThrowsOnRankMismatchVsExpectedRank) {
  // The second throw branch: a tensor symbol whose actual rank doesn't
  // match its Role's `expected_rank`. Strain has expected_rank=2, so
  // declaring it as rank-4 must throw with both rank values mentioned.
  ConstitutiveModel model("M");
  model.add_tensor_input("eps", 3, /*rank=*/4, roles::Strain);

  try {
    model.validate();
    FAIL() << "expected runtime_error for rank mismatch";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("eps"), std::string::npos) << msg;
    EXPECT_NE(msg.find("rank 4"), std::string::npos) << msg;
    EXPECT_NE(msg.find("expects rank 2"), std::string::npos) << msg;
    // Actionable: should name the add_tensor_input call site or the
    // roles:: catalogue.
    EXPECT_NE(msg.find("roles::"), std::string::npos)
        << "expected message to point at the roles:: catalogue: " << msg;
  }
}

TEST(TensorSpaceConsistencyPass, RegistersAsSecondValidator) {
  // Smoke test: emit_compute_function runs SymbolValidationPass +
  // TensorSpaceConsistencyPass + CodeEmitPass. Verifies the postcondition
  // wiring works with two validators in series.
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);
  EXPECT_NO_THROW(model.emit_compute_function());
}

// ─── CodeEmitPass ────────────────────────────────────────────────────

TEST(CodeEmitPass, ProducesNonEmptyComputeFunctionSource) {
  ConstitutiveModel model("LinearElasticShear");
  auto mu = model.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = model.add_tensor_input("eps", 3, 2, roles::Strain);
  model.add_output("stress", 2 * mu * eps, roles::Stress);

  auto src = model.emit_compute_function();
  EXPECT_NE(src.find("LinearElasticShear_compute"), std::string::npos)
      << "got:\n" << src;
  EXPECT_NE(src.find("stress_out"), std::string::npos) << src;
}

TEST(CodeEmitPass, RefusesToRunWithoutValidationPredecessor) {
  // Direct PassManager invocation without registering SymbolValidationPass
  // should fail loudly — CodeEmitPass advertises preconditions that are
  // unmet.
  ConstitutiveModel model("M");
  PassContext pctx{RecipeView{model}, CodeGenContext{}, std::nullopt, {}};
  PassManager pm;
  pm.emplace<CodeEmitPass>();

  try {
    pm.run(pctx);
    FAIL() << "expected runtime_error for missing SymbolValidationPass";
  } catch (std::runtime_error const &e) {
    std::string msg(e.what());
    EXPECT_NE(msg.find("CodeEmit"), std::string::npos) << msg;
    // Any of CodeEmitPass's three preconditions can show up first.
    // P3: tags are now typed string_view constants in pass_tags::; the
    // PassManager error message embeds them verbatim, so we look for the
    // same strings the constants resolve to.
    bool mentions_pre =
        msg.find(pass_tags::symbols_declared) != std::string::npos ||
        msg.find(pass_tags::identifiers_valid) != std::string::npos ||
        msg.find(pass_tags::tensor_space_declarations_checked) !=
            std::string::npos;
    EXPECT_TRUE(mentions_pre) << msg;
  }
}

} // namespace numsim::codegen
