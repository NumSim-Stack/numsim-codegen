#ifndef NUMSIM_CODEGEN_RECIPE_H
#define NUMSIM_CODEGEN_RECIPE_H

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/leaf_collector.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>
#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/passes/tensor_space_consistency_pass.h>
#include <numsim_codegen/recipe_render.h>

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace numsim::codegen {

// Semantic role describing what an input or output represents in a
// constitutive model. Carries a name (used for documentation and identity
// comparison) plus structured attributes that backends switch on to decide
// how to wire the symbol to the framework — MOOSE's stateful-pair handling
// triggers on `is_stateful`, Abaqus's symmetric-storage encoding triggers
// on `is_symmetric` etc.
//
// The `roles::` namespace below ships a catalogue of common roles. Users
// can construct custom Role values directly (e.g.
//   `Role{.name="phase_field", .is_driving=true, .expected_rank=0}`)
// without library changes — backends route them by attribute, not identity.
//
// Equality is name-based: two Role values with the same name compare equal
// regardless of attribute differences. Constructing a Role that shares its
// name with a `roles::` catalogue entry but carries different attributes
// throws at `add_*` time (see `ConstitutiveModel::validate_role_attributes`)
// — silent attribute mismatch would otherwise mis-route through
// `find_*_by_role`.
//
// `name` is held by value (`std::string`) so users may construct a Role
// from a temporary or a local string without lifetime hazards.
struct Role {
  std::string name;
  bool is_stateful  = false;   // requires old/new pair (state variable)
  bool is_driving   = false;   // primary kinematic/kinetic input
  bool is_symmetric = false;   // symmetric rank-2 tensor
  std::optional<std::size_t> expected_rank = std::nullopt;

  friend auto operator==(Role const &a, Role const &b) -> bool {
    return a.name == b.name;
  }
};

// Catalogue of common roles. Backends recognise these by attribute, not
// identity, so user-defined roles flow through correctly as long as their
// attributes are set appropriately. `inline const` (not `constexpr`)
// because `Role` contains `std::string`, which is not a portable literal
// type — `inline` still gives single-definition-across-TUs.
//
// NOTE: these globals are dynamically initialised (std::string member).
// Calling `ConstitutiveModel::add_*` from another TU's static-init phase
// is unsafe — the catalogue may not yet be constructed. Build recipes
// at runtime (inside `main`, in a function), not at namespace scope as
// `static ConstitutiveModel g_model("...");`. Tracked for hardening if a
// real consumer ever hits the dynamic-init-order trap.
namespace roles {
  inline const Role Strain              {"strain",               false, true,  true,  2};
  inline const Role StrainIncrement     {"strain_increment",     false, true,  true,  2};
  inline const Role DeformationGradient {"deformation_gradient", false, true,  false, 2};
  inline const Role Stress              {"stress",               false, false, true,  2};
  inline const Role ConsistentTangent   {"consistent_tangent",   false, false, false, 4};
  inline const Role Temperature         {"temperature",          false, true,  false, 0};
  inline const Role History             {"history",              true,  false, false};
  inline const Role Dissipation         {"dissipation",          false, false, false, 0};
  inline const Role Other               {"other"};
} // namespace roles

// Declaration of an external symbol (input, parameter). The codegen treats
// inputs and parameters the same on the scalar/tensor level (both are
// read-only) but backends distinguish them — parameters come from MOOSE
// input file via getParam, inputs come from coupled variables / state.
struct SymbolDecl {
  enum class Category { Input, Parameter };
  enum class Kind { Scalar, Tensor };

  std::string name;
  Category category;
  Kind kind;
  std::size_t dim = 0;   // tensor only
  std::size_t rank = 0;  // tensor only
  std::optional<double> default_value; // parameter only
  std::string doc;
  Role role = roles::Other;  // semantic tag; Other for parameters
};

// Declaration of a computed output that the generated function emits.
struct OutputDecl {
  enum class Kind { Scalar, Tensor };

  std::string name;
  Kind kind;
  std::variant<
      cas::expression_holder<cas::scalar_expression>,
      cas::expression_holder<cas::tensor_expression>>
      expr;
  std::size_t dim = 0;
  std::size_t rank = 0;
  Role role = roles::Other;
};

// The constitutive-model registry. Holds declared inputs, parameters,
// outputs, and their semantic role tags. Target-agnostic — the same
// recipe can be emitted as standalone C++, MOOSE Material, Abaqus UMAT,
// etc., by passing it through the appropriate Target backend.
class ConstitutiveModel {
public:
  explicit ConstitutiveModel(std::string name) : m_name(std::move(name)) {}

  [[nodiscard]] auto name() const -> std::string const & { return m_name; }

  // ─── Input declarations ─────────────────────────────────────────

  auto add_scalar_input(std::string name, Role role = roles::Other)
      -> cas::expression_holder<cas::scalar_expression> {
    validate_role_attributes(role);
    auto var = cas::make_expression<cas::scalar>(name);
    SymbolDecl decl{name, SymbolDecl::Category::Input,
                    SymbolDecl::Kind::Scalar};
    decl.role = std::move(role);
    m_inputs_cache.push_back(decl);            // copy
    m_symbols.push_back(std::move(decl));      // move (no copy)
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  auto add_tensor_input(std::string name, std::size_t dim, std::size_t rank,
                        Role role = roles::Other)
      -> cas::expression_holder<cas::tensor_expression> {
    validate_role_attributes(role);
    auto var = cas::make_expression<cas::tensor>(name, dim, rank);
    SymbolDecl decl{name, SymbolDecl::Category::Input,
                    SymbolDecl::Kind::Tensor, dim, rank};
    decl.role = std::move(role);
    m_inputs_cache.push_back(decl);
    m_symbols.push_back(std::move(decl));
    m_tensor_symbols.emplace_back(name, var);
    return var;
  }

  // ─── Parameter declarations ─────────────────────────────────────

  auto add_parameter(std::string name, double default_value,
                     std::string doc = "")
      -> cas::expression_holder<cas::scalar_expression> {
    auto var = cas::make_expression<cas::scalar>(name);
    SymbolDecl decl{name, SymbolDecl::Category::Parameter,
                    SymbolDecl::Kind::Scalar};
    decl.default_value = default_value;
    decl.doc = std::move(doc);
    m_parameters_cache.push_back(decl);
    m_symbols.push_back(std::move(decl));
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  // ─── Output declarations ────────────────────────────────────────

  void add_output(std::string name,
                  cas::expression_holder<cas::scalar_expression> expr,
                  Role role = roles::Other) {
    validate_role_attributes(role);
    OutputDecl decl{name, OutputDecl::Kind::Scalar, expr, 0, 0,
                    std::move(role)};
    m_outputs.push_back(std::move(decl));
  }

  void add_output(std::string name,
                  cas::expression_holder<cas::tensor_expression> expr,
                  Role role = roles::Other) {
    validate_role_attributes(role);
    OutputDecl decl{name, OutputDecl::Kind::Tensor, expr, expr.get().dim(),
                    expr.get().rank(), std::move(role)};
    m_outputs.push_back(std::move(decl));
  }

  // ─── Layer 2: generic compute function emission ────────────────
  //
  // Emit a target-agnostic C++ function `<name>_compute` taking the declared
  // inputs/parameters and writing outputs to out-parameters. All tensors are
  // tmech-typed. Backends (MOOSE, Abaqus, ...) wrap this function with their
  // framework-specific boundary conversions.

  // Verify that every leaf symbol referenced by any output expression has
  // been declared (via add_*_input or add_parameter). Throws
  // std::runtime_error listing the missing symbols or invalid identifiers.
  // Called automatically from emit_compute_function() but exposed for
  // explicit pre-flight checks. Backed by SymbolValidationPass under the
  // Phase 1.2 pass framework.
  //
  // NOTE on context isolation (issue #48 / M4): this constructs its own
  // independent PassContext. State written to pctx.ctx here is NOT shared
  // with a subsequent emit_compute_function() call (which builds another
  // fresh context). Today this is harmless because SymbolValidationPass
  // does not touch pctx.ctx. A future pass that wants to cache validation
  // artifacts for the emit pipeline must run inside the same
  // emit_compute_function() invocation, not a separate validate() call.
  void validate() const {
    PassContext pctx{RecipeView{*this}, CodeGenContext{}, std::nullopt};
    PassManager pm;
    pm.emplace<SymbolValidationPass>();
    pm.emplace<TensorSpaceConsistencyPass>();
    pm.run(pctx);
  }

  // Emit the generic compute function. Does not emit #include directives;
  // backends own the file-level framing (includes, namespacing, registration).
  //
  // Runs the Phase 1.2 pass pipeline: SymbolValidationPass → CodeEmitPass.
  // Future phases (TimeIntegrationPass, AlgorithmicTangentPass, …) plug
  // additional passes into this same pipeline.
  [[nodiscard]] auto emit_compute_function() const -> std::string {
    PassContext pctx{RecipeView{*this}, CodeGenContext{}, std::nullopt};
    PassManager pm;
    pm.emplace<SymbolValidationPass>();
    pm.emplace<TensorSpaceConsistencyPass>();
    pm.emplace<CodeEmitPass>();
    pm.run(pctx);
    if (!pctx.compute_function_source) {
      throw std::runtime_error(
          "ConstitutiveModel::emit_compute_function: pass pipeline finished "
          "without populating compute_function_source. Did CodeEmitPass run?");
    }
    return std::move(*pctx.compute_function_source);
  }

  // ─── Read-only accessors for backends ───────────────────────────

  [[nodiscard]] auto symbols() const -> std::vector<SymbolDecl> const & {
    return m_symbols;
  }

  [[nodiscard]] auto outputs() const -> std::vector<OutputDecl> const & {
    return m_outputs;
  }

  // Helpers for backends to find specific roles.
  [[nodiscard]] auto find_input_by_role(Role const &role) const
      -> SymbolDecl const * {
    for (auto const &s : m_symbols) {
      if (s.category == SymbolDecl::Category::Input && s.role == role) {
        return &s;
      }
    }
    return nullptr;
  }

  [[nodiscard]] auto find_output_by_role(Role const &role) const
      -> OutputDecl const * {
    for (auto const &o : m_outputs) {
      if (o.role == role) {
        return &o;
      }
    }
    return nullptr;
  }

  // Cached views maintained incrementally by the add_* methods — avoids
  // an O(N) filter on every call. Backends typically call these multiple
  // times per emit.
  [[nodiscard]] auto parameters() const
      -> std::vector<SymbolDecl> const & {
    return m_parameters_cache;
  }

  [[nodiscard]] auto inputs() const -> std::vector<SymbolDecl> const & {
    return m_inputs_cache;
  }

  // Symbol → expression maps. Exposed for the Phase 1.2 pass framework so
  // a CodeEmitPass can register them with a CodeGenContext without having
  // to re-derive them from `symbols()` (which loses the expression handle).
  [[nodiscard]] auto scalar_symbol_map() const noexcept -> auto const & {
    return m_scalar_symbols;
  }

  [[nodiscard]] auto tensor_symbol_map() const noexcept -> auto const & {
    return m_tensor_symbols;
  }

private:
  // Reject a user-defined Role that shares a name with a `roles::` catalogue
  // entry but carries different attribute values. Name-only equality means
  // such a mismatch would otherwise silently mis-route through
  // find_*_by_role and confuse backends. Called from every `add_*` so the
  // error surfaces at the offending call, not at emit time.
  static void validate_role_attributes(Role const &r) {
    auto check = [&](Role const &canon, char const *constant_name) {
      if (r.name != canon.name) return;
      if (r.is_stateful   != canon.is_stateful
       || r.is_driving    != canon.is_driving
       || r.is_symmetric  != canon.is_symmetric
       || r.expected_rank != canon.expected_rank) {
        throw std::runtime_error(
            "Role '" + r.name + "' shares its name with predefined " +
            constant_name + " but carries different attribute values. " +
            "Either use the predefined constant directly, or pick a different "
            "name for the custom role.");
      }
    };
    check(roles::Strain,              "roles::Strain");
    check(roles::StrainIncrement,     "roles::StrainIncrement");
    check(roles::DeformationGradient, "roles::DeformationGradient");
    check(roles::Stress,              "roles::Stress");
    check(roles::ConsistentTangent,   "roles::ConsistentTangent");
    check(roles::Temperature,         "roles::Temperature");
    check(roles::History,             "roles::History");
    check(roles::Dissipation,         "roles::Dissipation");
    check(roles::Other,               "roles::Other");
  }

  std::string m_name;
  std::vector<SymbolDecl> m_symbols;
  std::vector<SymbolDecl> m_parameters_cache;
  std::vector<SymbolDecl> m_inputs_cache;
  std::vector<OutputDecl> m_outputs;

  std::vector<
      std::pair<std::string, cas::expression_holder<cas::scalar_expression>>>
      m_scalar_symbols;
  std::vector<
      std::pair<std::string, cas::expression_holder<cas::tensor_expression>>>
      m_tensor_symbols;
};

// ─── RecipeView delegate bodies ──────────────────────────────────────
//
// Declared in passes/recipe_view.h; defined here because ConstitutiveModel
// must be complete to instantiate the delegations.
//
// P1: each delegate visits through the variant to get a `const
// ConstitutiveModel *` (the non-const arm decays to const for read
// access). The lambda is `inline` and the visit body is one line —
// optimisers should fold it to the same code as the pre-P1 direct
// pointer dereference.

namespace detail {
inline auto recipe_view_const_ptr(
    std::variant<ConstitutiveModel const *, ConstitutiveModel *> const &v)
    noexcept -> ConstitutiveModel const * {
  return std::visit(
      [](auto const *p) -> ConstitutiveModel const * { return p; }, v);
}
} // namespace detail

inline auto RecipeView::name() const -> std::string const & {
  return detail::recipe_view_const_ptr(m_model)->name();
}

inline auto RecipeView::symbols() const -> std::vector<SymbolDecl> const & {
  return detail::recipe_view_const_ptr(m_model)->symbols();
}

inline auto RecipeView::outputs() const -> std::vector<OutputDecl> const & {
  return detail::recipe_view_const_ptr(m_model)->outputs();
}

inline auto RecipeView::scalar_symbol_map() const -> ScalarSymbolMap const & {
  return detail::recipe_view_const_ptr(m_model)->scalar_symbol_map();
}

inline auto RecipeView::tensor_symbol_map() const -> TensorSymbolMap const & {
  return detail::recipe_view_const_ptr(m_model)->tensor_symbol_map();
}

// ─── Function-frame rendering (free functions) ───────────────────────
//
// These render the surrounding function frame after the body statements
// have been emitted into the CodeGenContext. They were ConstitutiveModel
// methods pre-1.2-M5; promoted to free functions so any future emit pass
// (TangentEmitPass, MoosePropertyEmitPass, StateVarEmitPass,
// AbaqusEmitPass, …) can call them without needing to be friended on
// ConstitutiveModel. Declaration lives in recipe_render.h for the public
// API surface; definitions are inline here to stay header-only.

namespace detail {

inline auto param_decl(SymbolDecl const &s, int &tmpl_counter) -> std::string {
  std::ostringstream os;
  if (s.kind == SymbolDecl::Kind::Scalar) {
    os << "double const " << s.name;
  } else {
    // Tensor parameters are templates over a tensor-like type so the
    // caller can pass a `tmech::tensor<...>`, a `tmech::adaptor<...>`,
    // or any other `tmech::tensor_base<Derived>` without forcing a
    // copy. This is what makes the MOOSE boundary zero-copy.
    os << "T" << tmpl_counter++ << " const &" << s.name;
  }
  return os.str();
}

inline auto output_decl(OutputDecl const &o, int &tmpl_counter) -> std::string {
  std::ostringstream os;
  if (o.kind == OutputDecl::Kind::Scalar) {
    os << "double &" << o.name << "_out";
  } else {
    os << "T" << tmpl_counter++ << " &" << o.name << "_out";
  }
  return os.str();
}

// Count how many tensor parameters / outputs need template params.
inline auto tensor_arg_count(RecipeView model) -> int {
  int n = 0;
  for (auto const &s : model.symbols()) {
    if (s.kind == SymbolDecl::Kind::Tensor)
      ++n;
  }
  for (auto const &o : model.outputs()) {
    if (o.kind == OutputDecl::Kind::Tensor)
      ++n;
  }
  return n;
}

} // namespace detail

inline auto render_compute_function(
    RecipeView model, CodeGenContext const &ctx,
    std::vector<std::string> const &output_rhs) -> std::string {
  std::ostringstream os;

  os << "// Auto-generated by numsim-codegen. Do not edit.\n";
  os << "// Model: " << model.name() << "\n\n";

  // Template parameter list: one T<n> per tensor argument so the caller
  // can pass any tmech::tensor_base subclass (tensor, adaptor, expression
  // template) without forcing a materialised copy.
  int const n_tmpl = detail::tensor_arg_count(model);
  if (n_tmpl > 0) {
    os << "template <";
    for (int i = 0; i < n_tmpl; ++i) {
      if (i > 0)
        os << ", ";
      os << "typename T" << i;
    }
    os << ">\n";
  }
  os << "inline void " << model.name() << "_compute(\n";

  int tmpl_counter = 0;
  bool first = true;
  for (auto const &s : model.symbols()) {
    if (!first) {
      os << ",\n";
    }
    first = false;
    os << "    " << detail::param_decl(s, tmpl_counter);
  }

  for (auto const &out : model.outputs()) {
    if (!first) {
      os << ",\n";
    }
    first = false;
    os << "    " << detail::output_decl(out, tmpl_counter);
  }
  os << ") {\n";

  os << ctx.render_statements();

  for (std::size_t i = 0; i < model.outputs().size(); ++i) {
    os << "  " << model.outputs()[i].name << "_out = " << output_rhs[i]
       << ";\n";
  }

  os << "}\n";
  return os.str();
}

// ─── Pass run() definitions ──────────────────────────────────────────
//
// These live after the ConstitutiveModel class so that the pass bodies can
// see its full definition (they call methods like `outputs()`,
// `scalar_symbol_map()`). They're inline so the existing header-mostly
// layout stays intact; the static library only houses the per-target
// wrappers in src/targets/.

inline void SymbolValidationPass::run(PassContext &pctx) {
  auto const &model = pctx.model;

  // P4: build the name → SymbolDecl* lookup once. Downstream passes
  // (TensorSpaceConsistencyPass today; Phase 2 rewriters tomorrow)
  // consume it instead of re-deriving the O(N·M) join. Capturing
  // SymbolDecl const* keeps the lookup tied to the symbols() vector's
  // lifetime — safe for the duration of one PassManager::run, which is
  // the only scope where PassContext is alive.
  for (auto const &s : model.symbols()) {
    pctx.symbol_lookup.emplace(s.name, &s);
  }

  // (1) C++ identifier validity on declared symbol names. We check
  // *declared* names (not user-typed reference strings) — the cas
  // expression layer accepts anything for tensor("foo-bar", ...), so the
  // bad-character codepath only manifests once codegen tries to splice
  // the name into generated source.
  for (auto const &[name, _] : model.scalar_symbol_map()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(name)) {
      throw std::runtime_error(
          "ConstitutiveModel '" + model.name() + "': scalar symbol '" + name +
          "' is not a usable C++ identifier (bad syntax or reserved keyword). "
          "Generated code would not compile. Use a [A-Za-z_][A-Za-z0-9_]* "
          "name that is not a C++ keyword.");
    }
  }
  for (auto const &[name, _] : model.tensor_symbol_map()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(name)) {
      throw std::runtime_error(
          "ConstitutiveModel '" + model.name() + "': tensor symbol '" + name +
          "' is not a usable C++ identifier (bad syntax or reserved keyword). "
          "Generated code would not compile. Use a [A-Za-z_][A-Za-z0-9_]* "
          "name that is not a C++ keyword.");
    }
  }
  for (auto const &out : model.outputs()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(out.name)) {
      throw std::runtime_error(
          "ConstitutiveModel '" + model.name() + "': output '" + out.name +
          "' is not a usable C++ identifier (bad syntax or reserved keyword). "
          "Generated code would not compile. Use a [A-Za-z_][A-Za-z0-9_]* "
          "name that is not a C++ keyword.");
    }
  }

  // (2) Output-expression-references-declared-symbols (pre-1.2 validate()
  // body — preserved verbatim).
  LeafCollector lc;
  for (auto const &o : model.outputs()) {
    std::visit(
        [&](auto const &e) {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<
                            T,
                            cas::expression_holder<cas::scalar_expression>>) {
            lc.collect_scalar(e);
          } else {
            lc.collect_tensor(e);
          }
        },
        o.expr);
  }

  std::set<std::string> declared_scalar;
  std::set<std::string> declared_tensor;
  for (auto const &[name, _] : model.scalar_symbol_map())
    declared_scalar.insert(name);
  for (auto const &[name, _] : model.tensor_symbol_map())
    declared_tensor.insert(name);

  std::vector<std::string> missing;
  for (auto const &name : lc.scalar_names()) {
    if (!declared_scalar.contains(name)) {
      missing.push_back("scalar input/parameter '" + name + "'");
    }
  }
  for (auto const &name : lc.tensor_names()) {
    if (!declared_tensor.contains(name)) {
      missing.push_back("tensor input '" + name + "'");
    }
  }

  if (!missing.empty()) {
    std::string msg = "ConstitutiveModel '" + model.name() +
                      "': outputs reference undeclared symbol(s):";
    for (auto const &m : missing)
      msg += "\n  - " + m;
    msg += "\nCall add_scalar_input / add_tensor_input / add_parameter "
           "before referencing a symbol in an output expression.";
    throw std::runtime_error(msg);
  }
}

inline void TensorSpaceConsistencyPass::run(PassContext &pctx) {
  auto const &model = pctx.model;

  // For each declared tensor symbol (input or parameter, though
  // parameters are scalars in this catalogue), cross-check Role
  // attributes against the cas tensor's space() annotation.
  //
  // Two conflict patterns checked today:
  //   role.is_symmetric == true + space().perm is Skew → contradiction
  //   role.expected_rank set    + tensor.rank() != expected_rank → mismatch
  //
  // Phase 2 will replace the body with expression-level inference
  // (walk output expressions, propagate space() through operators,
  // validate end-to-end). The current shallow check exists primarily
  // to validate the framework with a second non-trivial pass before
  // Phase 2 wires expression rewriters into the same PassContext.
  // P4: O(N) walk; lookup is constant-time per symbol. The pre-P4 body
  // did a linear scan over model.symbols() *inside* this loop, giving
  // O(N·M) — fine for the dozen-symbol recipes we have today, but the
  // structural smell scaled badly across Phase-2 passes that all needed
  // the same join. Now SymbolValidationPass populates the lookup once
  // and every downstream pass consumes it.
  for (auto const &[name, expr] : model.tensor_symbol_map()) {
    auto it = pctx.symbol_lookup.find(name);
    if (it == pctx.symbol_lookup.end() ||
        it->second->kind != SymbolDecl::Kind::Tensor) {
      continue; // Should not happen — SymbolValidationPass guards it.
    }
    Role const *role_ptr = &it->second->role;

    auto const &t = expr.get();
    auto const &space_opt = t.space();
    if (space_opt.has_value() && role_ptr->is_symmetric) {
      if (std::holds_alternative<cas::Skew>(space_opt->perm)) {
        throw std::runtime_error(
            "ConstitutiveModel '" + model.name() + "': tensor symbol '" +
            name + "' is declared with role '" + role_ptr->name +
            "' (is_symmetric=true) but its tensor_space annotation has Skew "
            "perm — a skew-symmetric tensor cannot satisfy a symmetric role. "
            "Fix one of: (a) remove the cas-side `assume_skew()` annotation "
            "on this symbol; (b) re-declare with a Role whose "
            "`is_symmetric=false` (construct a custom `Role{...}`, or pick "
            "from the `roles::` catalogue in recipe.h).");
      }
    }

    if (role_ptr->expected_rank.has_value() &&
        t.rank() != *role_ptr->expected_rank) {
      throw std::runtime_error(
          "ConstitutiveModel '" + model.name() + "': tensor symbol '" + name +
          "' has rank " + std::to_string(t.rank()) + " but role '" +
          role_ptr->name + "' expects rank " +
          std::to_string(*role_ptr->expected_rank) +
          ". Fix one of: (a) adjust the `rank` argument in your "
          "`add_tensor_input(name, dim, rank, ...)` call to match the role; "
          "(b) re-declare with a different Role (see the `roles::` catalogue "
          "in recipe.h for their `expected_rank` values).");
    }
  }
}

inline void CodeEmitPass::run(PassContext &pctx) {
  auto const &model = pctx.model;
  auto &ctx = pctx.ctx;

  // P2: cycle-break is now encapsulated in CodeEmitPipeline. The class
  // owns the three emitters + closes the TensorCodeEmit ↔
  // TensorToScalarCodeEmit cycle internally. See its header for the
  // construction-order reasoning.
  CodeEmitPipeline pipeline(ctx);

  for (auto const &[name, expr] : model.scalar_symbol_map()) {
    ctx.register_symbol_scalar(expr, name);
  }
  for (auto const &[name, expr] : model.tensor_symbol_map()) {
    ctx.register_symbol_tensor(expr, name);
  }

  std::vector<std::string> output_rhs;
  output_rhs.reserve(model.outputs().size());
  for (auto const &out : model.outputs()) {
    output_rhs.push_back(std::visit(
        [&](auto const &e) -> std::string {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<
                            T,
                            cas::expression_holder<cas::scalar_expression>>) {
            return pipeline.scalar().apply(e);
          } else {
            return pipeline.tensor().apply(e);
          }
        },
        out.expr));
  }

  pctx.compute_function_source = render_compute_function(model, ctx, output_rhs);
}

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_H
