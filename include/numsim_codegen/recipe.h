#ifndef NUMSIM_CODEGEN_RECIPE_H
#define NUMSIM_CODEGEN_RECIPE_H

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/leaf_collector.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>
#include <numsim_codegen/passes/algorithmic_tangent_pass.h>
#include <numsim_codegen/passes/code_emit_pass.h>
#include <numsim_codegen/passes/local_jacobian_pass.h>
#include <numsim_codegen/passes/local_newton_lowering_pass.h>
#include <numsim_codegen/passes/pass.h>
#include <numsim_codegen/passes/pass_manager.h>
#include <numsim_codegen/passes/symbol_validation_pass.h>
#include <numsim_codegen/passes/tensor_space_consistency_pass.h>
#include <numsim_codegen/passes/time_integration_pass.h>
#include <numsim_codegen/recipe_render.h>

#include <numsim_cas/core/diff.h>
#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_diff.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_diff.h> // Phase 3b-1: diff(tensor, tensor)

#include <cstddef>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
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
  // SUPERSEDED in Phase 2.1 by ConstitutiveModel::add_*_state_variable().
  // The new API replaces "marker-on-input" with first-class StateVariable
  // declarations that carry an initial-value expression, register both
  // `<name>` (current) and `<name>_old` symbols, and let backends route
  // through `Category::StateVariable*` instead of inspecting Role.
  // `roles::History` stays in the catalogue for one release as an alias
  // for existing recipes; new code should use `add_*_state_variable`.
  // `[[deprecated]]` attribute is deferred until Phase 2.6 when backends
  // actually rewire away from this marker.
  inline const Role History             {"history",              true,  false, false};
  inline const Role Dissipation         {"dissipation",          false, false, false, 0};
  inline const Role Other               {"other"};
} // namespace roles

// Declaration of an external symbol (input, parameter). The codegen treats
// inputs and parameters the same on the scalar/tensor level (both are
// read-only) but backends distinguish them — parameters come from MOOSE
// input file via getParam, inputs come from coupled variables / state.
struct SymbolDecl {
  // Phase 2.1: state variables register two distinct symbols per declared
  // variable — a `*Current` for the value the recipe writes (i.e. what
  // Newton iterates) and a `*Old` for the framework-supplied previous-
  // step value (e.g. MOOSE's `getMaterialPropertyOld<>`). Backends
  // switch on this category to wire each correctly:
  //   - StateVariableCurrent → declareProperty (MOOSE) / function out-arg
  //   - StateVariableOld     → getMaterialPropertyOld (MOOSE) / function in-arg
  // Inputs/Parameters are unchanged.
  enum class Category {
    Input,
    Parameter,
    StateVariableCurrent,
    StateVariableOld
  };
  enum class Kind { Scalar, Tensor };

  std::string name;
  Category category;
  Kind kind;
  std::size_t dim = 0;   // tensor only
  std::size_t rank = 0;  // tensor only
  std::optional<double> default_value; // parameter only
  std::string doc;
  Role role = roles::Other;  // semantic tag; Other for parameters
  // Phase 2.6 (issue #77 / PR #78 review #4): set only on the `dt` symbol
  // auto-registered by `ensure_dt_symbol_registered_`. Backends route a
  // time-step symbol specially (MOOSE → framework `_dt`); matching on this
  // flag rather than the name avoids hijacking a user-declared `dt`
  // parameter that isn't the framework timestep.
  bool is_time_step = false;
};

// Phase 2.1: suffix appended to a state variable's name to form the
// paired "old"-value symbol. Lifted to a named constant (M3 in
// REVIEW-pr-58.md) so Phase 2.2's TimeIntegrationPass and Phase 2.6's
// MOOSE wiring reference the same string. Prior to this lift, eight
// hardcoded `"_old"` literals lived across the two add methods —
// brittle contract.
inline constexpr std::string_view state_variable_old_suffix = "_old";

// Phase 2.1: declaration of a state variable — a quantity whose value at
// step n+1 the recipe writes (during Newton iteration on the local
// constitutive system) and whose value at step n the framework supplies
// from storage. The classic plasticity example is the equivalent plastic
// strain α: its evolution `Dt(α) = λ` is solved alongside the stress
// equations, and the value at the start of each step (α_old) comes from
// the previous step's storage.
//
// The codegen wires two distinct symbols per state variable:
//   - `<name>` (the "current" value — what we're solving for)
//   - `<name>_old` (the previous step's value — framework-supplied)
// Both appear in `symbols()` with respective `Category::StateVariableCurrent`
// / `StateVariableOld` so backends can route each to the correct framework
// API (MOOSE `declareProperty` / `getMaterialPropertyOld`).
//
// `initial_value` is the expression evaluated at simulation start (or
// when state is reset). Backends emit a one-shot initialisation routine.
struct StateVariable {
  std::string name;
  SymbolDecl::Kind kind;
  std::size_t dim = 0;   // tensor only
  std::size_t rank = 0;  // tensor only
  std::variant<
      cas::expression_holder<cas::scalar_expression>,
      cas::expression_holder<cas::tensor_expression>>
      initial_value;
  std::string doc;

  // Phase 2.1 stronger-fix (architect Q1, REVIEW-pr-58.md): explicit
  // indices into `ConstitutiveModel::symbols()` for the paired
  // SymbolDecl entries. Eliminates Phase 2.2+'s reliance on
  // `name + state_variable_old_suffix` string concat to find the
  // partner symbol — lowering passes just read these.
  std::size_t current_symbol_idx = 0;
  std::size_t old_symbol_idx = 0;
};

// Phase 2.2 (issue #68): the framework-supplied symbol used as the time
// step in the backward-Euler discretization. Auto-registered as a
// scalar parameter when the first evolution equation is added to a
// recipe. Lifted to a named constant so TimeIntegrationPass and any
// future backend wiring (MOOSE `_dt`, standalone driver argument)
// reference the same string.
inline constexpr std::string_view state_time_step_name = "dt";

// Phase 2.2 (issue #68): declaration of a state variable's evolution.
// "α has rate λ" → backward-Euler discretization is the residual
// `(α − α_old)/dt − λ = 0`, which TimeIntegrationPass synthesises as
// a new output expression. Phase 3a (issue #34) will consume that
// residual via a local-Newton lowering; for Phase 2.2 it surfaces as
// a regular output and the user supplies the Newton driver.
//
// Scalar-only in this Phase 2.2 scope — tensor evolutions are common
// in plasticity (rate-form `Dt(ε^p) = ...`) but the Phase 2.2 scope
// caps at scalar to keep the residual-synthesis path narrow. Tensor
// evolutions land in a follow-up once a first consumer surfaces.
//
// `state_variable_idx` is an index into `m_state_variables` (not a
// pointer or reference) — mirrors the index-based PR #58 design that
// survives vector growth from future mutating passes.
struct EvolutionEquation {
  std::size_t state_variable_idx;
  cas::expression_holder<cas::scalar_expression> rate;
  std::string doc;
};

// Phase 3a-2 (issue #75): tuning for the in-function Newton solve emitted
// by LocalNewtonLoweringPass. `tol` is the absolute residual threshold for
// convergence; `max_iter` caps the iteration count (no line search /
// divergence detection in 3a-2 — a plain max-iter guard). Recipe-level for
// now (uniform across all evolution equations); per-equation tuning can be
// added when a real consumer needs it.
struct NewtonOptions {
  double tol = 1e-10;
  int max_iter = 50;
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

// Phase 3b-1 (issue #35): a requested algorithmic (consistent) tangent
// `dσ/dε`. `name` is the rank-4 output the generated function emits;
// `of_output` names the (tensor) stress output to differentiate; `wrt_input`
// names the (tensor) strain input to differentiate against.
// AlgorithmicTangentPass resolves these against the final output/symbol sets
// and synthesises one rank-4 tensor output per spec.
struct TangentSpec {
  std::string name;
  std::string of_output;
  std::string wrt_input;
};

// Forward-declared in a sub-namespace so `ConstitutiveModel`'s
// `friend struct testing_detail::state_variable_alignment_access;`
// resolves. Definition lives only in the test TU (see
// `tests/StateVariableTest.cpp`). PR #66 round-2 review #5.
namespace testing_detail {
struct state_variable_alignment_access;
} // namespace testing_detail

// The constitutive-model registry. Holds declared inputs, parameters,
// outputs, and their semantic role tags. Target-agnostic — the same
// recipe can be emitted as standalone C++, MOOSE Material, Abaqus UMAT,
// etc., by passing it through the appropriate Target backend.
class ConstitutiveModel {
public:
  explicit ConstitutiveModel(std::string name) : m_name(std::move(name)) {
    // Security (PR #78 round-5): the model name flows UNESCAPED into
    // generated C++ — the `<name>_compute` function name, the MOOSE
    // `class <name> : public Material`, `registerMooseObject(...)`, and
    // `declareProperty("<name>_...")` string literals. An invalid name
    // (spaces, `"`, `;`, `)`) is a malformed-output / code-injection
    // vector. SymbolValidationPass already validates *symbol* identifiers;
    // the model name had no such guard. Reject it at construction.
    if (!SymbolValidationPass::is_valid_cxx_identifier(m_name)) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel: model name '{}' is not a valid C++ identifier "
          "(it becomes the generated function/class name). Use a "
          "[A-Za-z_][A-Za-z0-9_]* name that is not a C++ keyword.",
          m_name));
    }
  }

  [[nodiscard]] auto name() const -> std::string const & { return m_name; }

  // ─── Input declarations ─────────────────────────────────────────

  auto add_scalar_input(std::string name, Role role = roles::Other)
      -> cas::expression_holder<cas::scalar_expression> {
    validate_role_attributes(role);
    assert_symbol_name_available(name); // M1
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
    assert_symbol_name_available(name); // M1
    auto var = cas::make_expression<cas::tensor>(name, dim, rank);
    // Phase 3b-1 follow-up (#80 review, math Q3): a symmetric-role rank-2
    // input (e.g. roles::Strain) carries a Symmetric tensor space, so
    // `cas::diff` returns the minor-symmetric rank-4 identity (P_sym =
    // ½(I⊗ᵘI + I⊗ˡI)) rather than the bare `δ_ik δ_jl`. This makes consistent
    // tangents `∂σ/∂ε` minor-symmetric (C_ijkl = C_ijlk) — and, for σ = 2μ ε,
    // major-symmetric too — as a stress-strain tangent must be. Plain inputs
    // (roles::Other) stay unconstrained — backward compatible. The space is
    // metadata only (not part of the leaf hash), so non-derivative outputs and
    // CSE are byte-identical.
    //
    // `AnyTraceTag` (no trace constraint) is required: a strain has a non-zero
    // volumetric part, so `DeviatoricTag`/`HarmonicTag` would wrongly project
    // it out of the tangent. The guard is keyed on `is_symmetric && rank==2`,
    // so any symmetric rank-2 role qualifies (roles::Stress too, if ever used
    // as an input) — which is correct, since the space only affects
    // differentiation and those tensors genuinely are symmetric.
    //
    // Propagation: `data()` mutates the shared leaf node, and `cas::diff` reads
    // the space from its ARGUMENT handle (the `m_tensor_symbols` copy, shared
    // via the holder's shared_ptr) — NOT from the leaf inside the forward
    // expression. So the "2-arg ctors drop the space" CAS caveat does not
    // affect this path.
    if (role.is_symmetric && rank == 2) {
      var.data()->set_space(
          cas::tensor_space{cas::Symmetric{}, cas::AnyTraceTag{}});
    }
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
    assert_symbol_name_available(name); // M1
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

  // ─── State-variable declarations (Phase 2.1) ────────────────────
  //
  // Each `add_*_state_variable` returns a `Handle{current, previous}`.
  // - `current`: the value the recipe writes (Newton iterates on this).
  // - `previous`: the framework-supplied previous-step value, named
  //   `<name>_old`. Use it in evolution equations like
  //   `Dt(α) = (α.current − α.previous) / dt`, lowered by Phase 2.2's
  //   TimeIntegrationPass.
  //
  // Both symbols register in `m_symbols` with Category
  // StateVariableCurrent / StateVariableOld, so they flow through the
  // existing validation pipeline (identifier check, symbol_lookup).

  // PR #69 round-1 #3: `model_token` defends against cross-recipe hijack
  // — a handle whose underlying scalar name happens to collide with a
  // state variable on a different model. Without the token, name-based
  // matching in `add_scalar_evolution_equation` would silently bind to
  // the wrong recipe. The token is set by `add_*_state_variable` to
  // `this` and verified at every Handle-consuming API entry. Manual
  // construction of Handles (legitimate test-only use) sets it to
  // nullptr; consumers reject nullptr with a clear diagnostic.
  struct ScalarStateVariableHandle {
    cas::expression_holder<cas::scalar_expression> current;
    cas::expression_holder<cas::scalar_expression> previous;
    ConstitutiveModel const *model_token = nullptr;
  };
  struct TensorStateVariableHandle {
    cas::expression_holder<cas::tensor_expression> current;
    cas::expression_holder<cas::tensor_expression> previous;
    ConstitutiveModel const *model_token = nullptr;
  };

  auto add_scalar_state_variable(
      std::string name,
      cas::expression_holder<cas::scalar_expression> initial_value,
      std::string doc = "") -> ScalarStateVariableHandle {
    auto old_name = name + std::string{state_variable_old_suffix};
    assert_symbol_name_available(name);     // M1
    assert_symbol_name_available(old_name); // M2

    auto current = cas::make_expression<cas::scalar>(name);
    auto previous = cas::make_expression<cas::scalar>(old_name);

    SymbolDecl current_decl{name,
                            SymbolDecl::Category::StateVariableCurrent,
                            SymbolDecl::Kind::Scalar};
    current_decl.doc = doc;
    std::size_t const current_idx = m_symbols.size();
    m_symbols.push_back(std::move(current_decl));
    m_scalar_symbols.emplace_back(name, current);

    SymbolDecl old_decl{old_name,
                        SymbolDecl::Category::StateVariableOld,
                        SymbolDecl::Kind::Scalar};
    old_decl.doc = doc;
    std::size_t const old_idx = m_symbols.size();
    m_symbols.push_back(std::move(old_decl));
    m_scalar_symbols.emplace_back(std::move(old_name), previous);

    StateVariable sv{std::move(name), SymbolDecl::Kind::Scalar, 0, 0,
                     std::move(initial_value), std::move(doc),
                     current_idx, old_idx};
    m_state_variables.push_back(std::move(sv));

    return {current, previous, this};
  }

  auto add_tensor_state_variable(
      std::string name, std::size_t dim, std::size_t rank,
      cas::expression_holder<cas::tensor_expression> initial_value,
      std::string doc = "") -> TensorStateVariableHandle {
    auto old_name = name + std::string{state_variable_old_suffix};
    assert_symbol_name_available(name);     // M1
    assert_symbol_name_available(old_name); // M2

    auto current = cas::make_expression<cas::tensor>(name, dim, rank);
    auto previous = cas::make_expression<cas::tensor>(old_name, dim, rank);

    SymbolDecl current_decl{name,
                            SymbolDecl::Category::StateVariableCurrent,
                            SymbolDecl::Kind::Tensor, dim, rank};
    current_decl.doc = doc;
    std::size_t const current_idx = m_symbols.size();
    m_symbols.push_back(std::move(current_decl));
    m_tensor_symbols.emplace_back(name, current);

    SymbolDecl old_decl{old_name,
                        SymbolDecl::Category::StateVariableOld,
                        SymbolDecl::Kind::Tensor, dim, rank};
    old_decl.doc = doc;
    std::size_t const old_idx = m_symbols.size();
    m_symbols.push_back(std::move(old_decl));
    m_tensor_symbols.emplace_back(std::move(old_name), previous);

    StateVariable sv{std::move(name), SymbolDecl::Kind::Tensor, dim, rank,
                     std::move(initial_value), std::move(doc),
                     current_idx, old_idx};
    m_state_variables.push_back(std::move(sv));

    return {current, previous, this};
  }

  // ─── Evolution equations (Phase 2.2, issue #68) ─────────────────
  //
  // Declares that a state variable evolves at a given rate. Backward-
  // Euler lowering is performed by `TimeIntegrationPass`, which
  // synthesises `(<sv> − <sv>_old)/dt − rate` as a new scalar output.
  //
  // The first call to this method auto-registers a `dt` scalar
  // parameter (default value 0.0; the framework supplies the actual
  // time step at runtime). If the user has already declared a symbol
  // named `dt`, that one is reused and no new symbol is registered.

  auto add_scalar_evolution_equation(
      ScalarStateVariableHandle const &state_var,
      cas::expression_holder<cas::scalar_expression> rate,
      std::string doc = "") -> void {
    // PR #69 round-1 #3: cross-recipe hijack defense. The handle's
    // `model_token` MUST equal `this` — otherwise the user passed a
    // handle that came from a different ConstitutiveModel (or
    // manually-constructed). Pure name-based matching would silently
    // bind to a coincidentally-named state variable on this model.
    if (state_var.model_token == nullptr) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': add_scalar_evolution_equation handle "
          "has no model token. Handles must come from "
          "add_scalar_state_variable on this model — manually-constructed "
          "handles are not accepted.",
          m_name));
    }
    if (state_var.model_token != this) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': add_scalar_evolution_equation handle "
          "came from a different ConstitutiveModel (handle.model_token = "
          "{}, this = {}). Each handle is bound to the recipe that "
          "created it; cross-recipe use would silently bind by name and "
          "discretise the wrong state variable.",
          m_name, static_cast<void const *>(state_var.model_token),
          static_cast<void const *>(this)));
    }

    // PR #69 round-1 review (CRITICAL): the handle's `current` MUST be a
    // bare scalar leaf (i.e. `cas::scalar`), not a compound expression.
    // `expression_holder::get<cas::scalar>()` would use an
    // `assert(dynamic_cast != nullptr)` path that's stripped under
    // `NDEBUG`, then UB on the unchecked `static_cast`. Use an explicit
    // runtime dynamic_cast that throws a clear diagnostic instead.
    auto const *typed = dynamic_cast<cas::scalar const *>(
        state_var.current.data().get());
    if (!typed) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': add_scalar_evolution_equation handle's "
          "`current` is not a bare scalar leaf symbol. Handles must come "
          "from add_scalar_state_variable — you cannot synthesise one "
          "from a compound expression like `K * alpha`.",
          m_name));
    }
    auto const &sv_name = typed->name();

    std::size_t found_idx = m_state_variables.size();
    for (std::size_t i = 0; i < m_state_variables.size(); ++i) {
      if (m_state_variables[i].kind == SymbolDecl::Kind::Scalar &&
          m_state_variables[i].name == sv_name) {
        found_idx = i;
        break;
      }
    }
    if (found_idx == m_state_variables.size()) {
      // List registered scalar state variables to help debug.
      std::string registered;
      for (auto const &sv : m_state_variables) {
        if (sv.kind == SymbolDecl::Kind::Scalar) {
          if (!registered.empty()) registered += ", ";
          registered += sv.name;
        }
      }
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': add_scalar_evolution_equation handle "
          "names scalar state variable '{}' but no such state variable "
          "is registered on this model. Did the handle come from a "
          "different ConstitutiveModel, or was the state variable added "
          "with add_tensor_state_variable instead? Registered scalar "
          "state variables: [{}].",
          m_name, sv_name, registered));
    }

    // Validate that the rate expression's leaves are all declared
    // symbols on this model (PR #69 round-1 #4). Fail fast in the
    // user's stack frame; otherwise CodeEmitPass surfaces a less-clear
    // leaf-not-in-symbol_lookup error after lowering.
    validate_rate_expression_leaves_(rate, sv_name);

    ensure_dt_symbol_registered_();

    EvolutionEquation eq{found_idx, std::move(rate), std::move(doc)};
    m_evolution_equations.push_back(std::move(eq));
  }

  // ─── Output declarations ────────────────────────────────────────

  void add_output(std::string name,
                  cas::expression_holder<cas::scalar_expression> expr,
                  Role role = roles::Other) {
    validate_role_attributes(role);
    assert_output_name_available(name); // PR #78 review #2
    OutputDecl decl{name, OutputDecl::Kind::Scalar, expr, 0, 0,
                    std::move(role)};
    m_outputs.push_back(std::move(decl));
  }

  void add_output(std::string name,
                  cas::expression_holder<cas::tensor_expression> expr,
                  Role role = roles::Other) {
    validate_role_attributes(role);
    assert_output_name_available(name); // PR #78 review #2
    OutputDecl decl{name, OutputDecl::Kind::Tensor, expr, expr.get().dim(),
                    expr.get().rank(), std::move(role)};
    m_outputs.push_back(std::move(decl));
  }

  // Capacity hint ahead of a batch of `add_output` calls (cross-cutting
  // review MAJOR 4). A mutating pass that synthesises one output per
  // evolution equation calls this once before its loop so `m_outputs`
  // doesn't reallocate incrementally. Purely a `reserve` — no semantic
  // effect — but it also guarantees any output span/reference held
  // across the batch stays valid (the hazard the span-invalidation note
  // below warns about). Reserves `additional` slots beyond the current
  // size.
  void reserve_outputs(std::size_t additional) {
    m_outputs.reserve(m_outputs.size() + additional);
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
  //
  // **Phase 2.2 note (PR #69 round-1 #8):** `validate()` does NOT run
  // `TimeIntegrationPass`. Evolution-equation correctness — i.e. rate
  // expressions referencing only declared symbols — is enforced at
  // `add_scalar_evolution_equation()` time via
  // `validate_rate_expression_leaves_`, so `validate()` doesn't need to
  // re-walk those expressions. The user-side contract is: build the
  // recipe (add fails fast on bad rates), then call `validate()` for
  // belt-and-suspenders, then `emit_compute_function()`.
  void validate() const {
    PassContext pctx{RecipeView{*this}, CodeGenContext{}, std::nullopt, {}};
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
    // Phase 2.2 (issue #68): TimeIntegrationPass synthesises new
    // outputs via the mutable-RecipeView arm. Working on a copy of
    // `*this` lets us keep `emit_compute_function()` const — the
    // public contract stays "emit produces source, no recipe state
    // change" while the pass's mutation lives only inside this call.
    // ConstitutiveModel is value-typed (vectors + shared_ptr handles)
    // so the copy is cheap relative to the codegen itself.
    ConstitutiveModel working_copy = *this;
    PassContext pctx{RecipeView{working_copy}, CodeGenContext{},
                     std::nullopt, {}};
    PassManager pm;
    pm.emplace<SymbolValidationPass>();
    pm.emplace<TensorSpaceConsistencyPass>();
    // State-variable lowering is registered only when the recipe has
    // evolution equations (the `state_variables_non_empty` precondition
    // would otherwise fail at run() on pure-elasticity recipes). Two
    // mutually-exclusive shapes:
    //   * default — TimeIntegrationPass + LocalJacobianPass emit
    //     `<sv>_residual` / `<sv>_jacobian` as outputs for an external
    //     Newton driver.
    //   * local-newton (issue #75, opt-in via `enable_local_newton`) —
    //     LocalNewtonLoweringPass records an in-function Newton solve;
    //     CodeEmitPass renders the loop and the state variable becomes
    //     a `<sv>_out` parameter instead of R/J outputs.
    if (!m_evolution_equations.empty()) {
      if (m_local_newton) {
        pm.emplace<LocalNewtonLoweringPass>();
      } else {
        pm.emplace<TimeIntegrationPass>();
        pm.emplace<LocalJacobianPass>();
      }
    }
    // Phase 3b-1 (issue #35): consistent-tangent synthesis. Registered AFTER
    // state-variable lowering (so `pctx.newton_segments` — the converged-state
    // seam — is populated) and BEFORE CodeEmitPass (so the synthesised rank-4
    // tangent outputs render in the same sweep). Opt-in via
    // `add_algorithmic_tangent`.
    if (!m_tangents.empty()) {
      pm.emplace<AlgorithmicTangentPass>();
    }
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

  [[nodiscard]] auto symbols() const noexcept
      -> std::span<SymbolDecl const> {
    return m_symbols;
  }

  [[nodiscard]] auto outputs() const noexcept
      -> std::span<OutputDecl const> {
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
  //
  // **Span invalidation warning:** these accessors return a non-owning
  // span over `m_*_cache`, which is a `std::vector` that grows via
  // `push_back` inside `add_*`. Any caller that stores a span across a
  // subsequent `add_*` call risks the underlying vector reallocating —
  // the span's pointer-plus-size capture is then stale. The hazard
  // doesn't exist in Phase 1.2 passes (model construction is complete
  // before passes run), but Phase 2 mutating passes that synthesise
  // symbols mid-pipeline must re-acquire the span after every mutation.
  // Don't store a long-lived span across mutations.
  [[nodiscard]] auto parameters() const noexcept
      -> std::span<SymbolDecl const> {
    return m_parameters_cache;
  }

  [[nodiscard]] auto inputs() const noexcept
      -> std::span<SymbolDecl const> {
    return m_inputs_cache;
  }

  // Phase 2.1: state-variable declarations. Each entry is a single
  // StateVariable; the corresponding "current" and "_old" SymbolDecl
  // entries live in `m_symbols` (categories StateVariableCurrent /
  // StateVariableOld). Empty for recipes without internal state (e.g.
  // pure elasticity).
  [[nodiscard]] auto state_variables() const noexcept
      -> std::span<StateVariable const> {
    return m_state_variables;
  }

  // Phase 2.2 (issue #68): evolution-equation declarations. Empty for
  // recipes without rate-form constitutive equations. Consumed by
  // TimeIntegrationPass to synthesise the discrete residual outputs.
  [[nodiscard]] auto evolution_equations() const noexcept
      -> std::span<EvolutionEquation const> {
    return m_evolution_equations;
  }

  // Phase 3a-2 (issue #75): opt into in-function local Newton solving.
  // When enabled (and the recipe has evolution equations),
  // `emit_compute_function` registers `LocalNewtonLoweringPass` instead
  // of the residual/Jacobian-output passes (TimeIntegration + LocalJacobian):
  // the generated function solves each scalar state variable internally
  // and exposes it as a `<sv>_out` parameter, rather than emitting
  // `<sv>_residual` / `<sv>_jacobian` outputs for an external driver.
  // Default OFF — recipes keep the external-driver (R/J-output) shape
  // unless they ask for the solve.
  void enable_local_newton(NewtonOptions opts = {}) {
    m_local_newton = true;
    m_newton_options = opts;
  }
  [[nodiscard]] auto local_newton_enabled() const noexcept -> bool {
    return m_local_newton;
  }
  [[nodiscard]] auto newton_options() const noexcept -> NewtonOptions {
    return m_newton_options;
  }

  // ─── Algorithmic tangent (Phase 3b-1, issue #35) ────────────────
  //
  // Request a consistent tangent `dσ/dε` emitted as a rank-4 tensor output
  // `name`. `of_output` must name a tensor output (added via add_output);
  // `wrt_input` a tensor input (added via add_tensor_input). Existence and
  // rank are checked at emit time by AlgorithmicTangentPass, where the full
  // output set is final. Opt-in: a recipe with no tangent specs runs no
  // tangent pass (mirrors enable_local_newton). The emitted output is tagged
  // `roles::ConsistentTangent`.
  void add_algorithmic_tangent(std::string name, std::string of_output,
                               std::string wrt_input) {
    assert_output_name_available(name); // reserve vs outputs + symbols
    for (auto const &t : m_tangents) {
      if (t.name == name) {
        throw std::runtime_error(
            "ConstitutiveModel::add_algorithmic_tangent: a tangent named '" +
            name + "' was already requested.");
      }
    }
    m_tangents.push_back(
        {std::move(name), std::move(of_output), std::move(wrt_input)});
  }
  [[nodiscard]] auto algorithmic_tangent_enabled() const noexcept -> bool {
    return !m_tangents.empty();
  }
  [[nodiscard]] auto tangents() const noexcept
      -> std::span<TangentSpec const> {
    return m_tangents;
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
  // ─── Test-only friends ──────────────────────────────────────────
  //
  // PR #66 round-2 review #5: grants negative-path tests in `tests/`
  // controlled write access to `m_symbols` and `m_state_variables` so
  // they can simulate the corruption shapes Phase 2.2+ mutating passes
  // might produce. Placed under the `private:` label (rather than the
  // public block) so a reader scanning the class layout doesn't mistake
  // the line for a public-API affordance.
  //
  // Lives in the `testing_detail::` sub-namespace — both as a marker
  // ("if you're writing production code and naming this, you've taken
  // a wrong turn") and to keep the friend's reachable surface narrow
  // from production-namespace TUs.
  //
  // The struct is declared here but never defined inside
  // `numsim::codegen`; tests provide the only definition (see
  // `tests/StateVariableTest.cpp`). Keeping the surface to a single
  // friend struct means a single ODR slot in the test binary; if a
  // future test TU needs the same access, factor the definition to a
  // shared `tests/internal/` header rather than redefining it.
  friend struct testing_detail::state_variable_alignment_access;
  // Phase 2.1 (M1+M2 in REVIEW-pr-58.md): reject a duplicate symbol
  // name at add time, with a clear pipeline-misconfiguration message.
  // Two collision modes this catches:
  //   1. Calling `add_*("foo", ...)` twice (direct shadow).
  //   2. Adding an input named `<name>_old` followed by adding a state
  //      variable named `<name>` (the auto-generated `_old` symbol
  //      collides with the prior input). Or the reverse order.
  // Both produced silent corruption pre-fix: duplicate SymbolDecl
  // entries that `unordered_map::emplace` masked in the lookup but
  // survived in `m_symbols` and the symbol maps, leading to
  // uncompilable generated code or wrong-decl resolution.
  void assert_symbol_name_available(std::string_view name) const {
    for (auto const &existing : m_symbols) {
      if (existing.name == name) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': symbol name '{}' is already declared. "
            "Pick a unique name. (If you tried to add a state variable, "
            "remember that the auto-generated `<name>{}` paired symbol is "
            "also reserved — do not separately declare an input or "
            "parameter with that suffix.)",
            m_name, name, state_variable_old_suffix));
      }
    }
    // Symmetric with assert_output_name_available (PR #78 round-2): a
    // symbol must also not clash with an already-declared output, else a
    // state var / output declared in EITHER order collides on the MOOSE
    // property name `<Model>_<name>` + the `_<name>` member.
    for (auto const &o : m_outputs) {
      if (o.name == name) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': name '{}' is already declared as an "
            "output; a symbol (parameter / input / state variable) cannot "
            "reuse it. Backends derive member + MOOSE property names from "
            "both.",
            m_name, name));
      }
    }
  }

  // PR #78 review #2: an output's name must not clash with a symbol (param,
  // input, or state variable) or another output — backends derive both
  // C++ member names and MOOSE property names (`<Model>_<name>`) from these,
  // so a clash produces a duplicate-property / redefinition in generated
  // code. e.g. a state variable `alpha` + an output `alpha` would both emit
  // `declareProperty<Real>("<Model>_alpha")`.
  void assert_output_name_available(std::string_view name) const {
    for (auto const &s : m_symbols) {
      if (s.name == name) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': output name '{}' clashes with an "
            "existing symbol (parameter / input / state variable). Backends "
            "derive member + MOOSE property names from both; pick a unique "
            "output name.",
            m_name, name));
      }
    }
    for (auto const &o : m_outputs) {
      if (o.name == name) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': output '{}' is already declared.",
            m_name, name));
      }
    }
  }

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
        throw std::runtime_error(std::format(
            "Role '{}' shares its name with predefined {} but carries "
            "different attribute values. Either use the predefined constant "
            "directly, or pick a different name for the custom role.",
            r.name, constant_name));
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

  // Phase 2.2 (issue #68): auto-register a `dt` scalar parameter on
  // the first add_*_evolution_equation call. If the user has already
  // declared a symbol named `dt`, reuse it — don't throw on the name
  // collision the way `assert_symbol_name_available` would.
  //
  // Default value is NaN (PR #69 round-1 #2): the framework MUST
  // supply the actual time step at runtime. A forgotten dt wiring then
  // propagates NaN through `(α − α_old)/dt − rate`, which is loudly
  // visible in any Newton residual check or numeric inspection. A
  // default of 0.0 would silently produce Inf/-Inf, harder to triage.
  void ensure_dt_symbol_registered_() {
    for (auto const &s : m_symbols) {
      if (s.name == state_time_step_name) return;
    }
    add_parameter(std::string{state_time_step_name},
                  std::numeric_limits<double>::quiet_NaN(),
                  "time step (framework-supplied; auto-registered "
                  "by the first add_*_evolution_equation call). "
                  "Default is NaN — the framework must overwrite it.");
    // Mark the auto-registered symbol as the time step (PR #78 review #4):
    // backends route it specially (MOOSE → `_dt`). add_parameter pushed
    // the decl to both m_symbols and m_parameters_cache.
    m_symbols.back().is_time_step = true;
    m_parameters_cache.back().is_time_step = true;
  }

  // PR #69 round-1 #4: validate that the leaves of a rate expression
  // are all registered symbols on this model. Called from
  // `add_scalar_evolution_equation` to fail fast in the user's stack
  // frame, rather than letting the bug surface from CodeEmitPass after
  // TimeIntegrationPass has synthesised an output referencing the
  // undeclared leaf. Mirrors the leaf-validity check in
  // SymbolValidationPass::run but anchored at the recipe-build call.
  void validate_rate_expression_leaves_(
      cas::expression_holder<cas::scalar_expression> const &rate,
      std::string_view sv_name) const {
    LeafCollector lc;
    lc.collect_scalar(rate);
    std::vector<std::string> missing;
    for (auto const &leaf_name : lc.scalar_names()) {
      bool found = false;
      for (auto const &[name, _] : m_scalar_symbols) {
        if (name == leaf_name) { found = true; break; }
      }
      if (!found) missing.push_back("scalar '" + leaf_name + "'");
    }
    for (auto const &leaf_name : lc.tensor_names()) {
      bool found = false;
      for (auto const &[name, _] : m_tensor_symbols) {
        if (name == leaf_name) { found = true; break; }
      }
      if (!found) missing.push_back("tensor '" + leaf_name + "'");
    }
    if (!missing.empty()) {
      std::string msg = std::format(
          "ConstitutiveModel '{}': add_scalar_evolution_equation rate "
          "expression for state variable '{}' references undeclared "
          "symbol(s):",
          m_name, sv_name);
      for (auto const &m : missing) {
        msg += std::format("\n  - {}", m);
      }
      msg += "\nCall add_scalar_input / add_tensor_input / add_parameter "
             "before referencing a symbol in a rate expression.";
      throw std::runtime_error(msg);
    }
  }

  std::string m_name;
  std::vector<SymbolDecl> m_symbols;
  std::vector<SymbolDecl> m_parameters_cache;
  std::vector<SymbolDecl> m_inputs_cache;
  std::vector<OutputDecl> m_outputs;
  std::vector<StateVariable> m_state_variables; // Phase 2.1
  std::vector<EvolutionEquation> m_evolution_equations; // Phase 2.2
  bool m_local_newton = false;          // Phase 3a-2 (issue #75)
  NewtonOptions m_newton_options{};     // Phase 3a-2 (issue #75)
  std::vector<TangentSpec> m_tangents;  // Phase 3b-1 (issue #35)

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

// **`detail::` namespace policy.** Implementation
// helpers that aren't part of the public recipe-builder API live under
// `numsim::codegen::detail::`. This file currently has multiple
// `namespace detail { ... }` blocks (here, in the rendering helpers
// section, and in the pass-internal block near TIP/LJP) — they merge
// at the language level. **Promote** a `detail::` block to
// `include/numsim_codegen/passes/internal/<topic>.h` when it exceeds
// ~75 LOC OR introduces a non-trivial type. Without the threshold,
// `recipe.h` drifts unboundedly as Phase 3a-2 / Phase 4 helpers land.
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

inline auto RecipeView::symbols() const noexcept
    -> std::span<SymbolDecl const> {
  return detail::recipe_view_const_ptr(m_model)->symbols();
}

inline auto RecipeView::outputs() const noexcept
    -> std::span<OutputDecl const> {
  return detail::recipe_view_const_ptr(m_model)->outputs();
}

inline auto RecipeView::state_variables() const noexcept
    -> std::span<StateVariable const> {
  return detail::recipe_view_const_ptr(m_model)->state_variables();
}

inline auto RecipeView::evolution_equations() const noexcept
    -> std::span<EvolutionEquation const> {
  return detail::recipe_view_const_ptr(m_model)->evolution_equations();
}

inline auto RecipeView::scalar_symbol_map() const -> ScalarSymbolMap const & {
  return detail::recipe_view_const_ptr(m_model)->scalar_symbol_map();
}

inline auto RecipeView::tensor_symbol_map() const -> TensorSymbolMap const & {
  return detail::recipe_view_const_ptr(m_model)->tensor_symbol_map();
}

// ─── PassContext symbol-lookup helpers ───────────────────────────────
//
// Declared in passes/pass.h; defined here because SymbolDecl::Kind is
// only complete after the ConstitutiveModel class body. The lookup
// values are `std::size_t` indices into `model.symbols()` (post-C1
// fixup — see REVIEW-pr-57.md); these helpers resolve the index
// through the model and apply the kind discriminator in one place.

[[nodiscard]] inline auto find_tensor_symbol(PassContext const &pctx,
                                             std::string const &name) noexcept
    -> std::expected<SymbolDecl const *, LookupError> {
  auto const it = pctx.symbol_lookup.find(name);
  if (it == pctx.symbol_lookup.end()) {
    return std::unexpected(LookupError::NotFound);
  }
  auto const &symbols = pctx.model.symbols();
  if (it->second >= symbols.size()) {
    // Defensive: shouldn't happen if SymbolValidationPass populated
    // the lookup correctly. Treat as NotFound — a stale index is
    // semantically equivalent to "no longer findable."
    return std::unexpected(LookupError::NotFound);
  }
  auto const &decl = symbols[it->second];
  if (decl.kind != SymbolDecl::Kind::Tensor) {
    return std::unexpected(LookupError::WrongKind);
  }
  return &decl;
}

[[nodiscard]] inline auto find_state_variable_by_name(
    PassContext const &pctx, std::string_view name) noexcept
    -> StateVariable const * {
  for (auto const &sv : pctx.model.state_variables()) {
    if (sv.name == name) {
      return &sv;
    }
  }
  return nullptr;
}

// Phase 2.2 prep (issue #59 / REVIEW-pr-58.md m1 + PR #66 review):
// verify the structural alignment between `model.state_variables()` and
// `model.symbols()`. Throws `std::runtime_error` naming the specific
// mismatch — silent corruption is worse than a loud throw.
//
// Today no public `ConstitutiveModel` API can produce a violation
// because `add_*_state_variable` writes both vectors atomically. The
// check earns its keep when Phase 2.2+ mutating passes start touching
// `m_state_variables` / `m_symbols` independently — at that point this
// function lives inside `SymbolValidationPass`, so any pipeline that
// registers `SymbolValidationPass` re-verifies the invariant on every
// run. Phase 2.2 mutating passes can re-run the pass (or a derived
// `StructuralIntegrityPass` that calls this) to re-check between
// mutations.
//
// **Bypass caveat:** a caller that constructs a `PassManager` and
// omits `SymbolValidationPass` skips the check entirely. The framework
// has no enforcement against that today; the convention is that any
// pipeline that emits code or mutates state variables registers
// `SymbolValidationPass` first. `ConstitutiveModel::validate()` and
// `::emit_compute_function()` both do so.
//
// TODO(phase-2.2): this scaffolding fault-injects the assertion arms
// (see the friend-poison tests in `tests/StateVariableTest.cpp`). When
// the first mutating pass lands, mutator-driven integration tests
// (apply mutator → assert throws) will be added alongside this corpus,
// not replace it — the fault-injection harness covers the assertion
// arms; the integration tests cover the mutator-to-arm wiring.
inline void verify_state_variable_symbol_alignment(RecipeView model) {
  // Hoisted out of the lambda body (PR #66 round-3 #6): the span is
  // identical across all `check()` invocations within a single call
  // — fetching once makes that explicit and removes a redundant copy
  // per state variable.
  auto const symbols = model.symbols();
  auto check = [&](std::size_t idx, std::string_view expected_name,
                   SymbolDecl::Category expected_cat,
                   StateVariable const &sv, char const *which) {
    if (idx >= symbols.size()) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': StateVariable '{}' carries {} index "
          "{} which is out of bounds for symbols() (size {}). State "
          "variable / symbol vectors are out of sync.",
          model.name(), sv.name, which, idx, symbols.size()));
    }
    auto const &s = symbols[idx];
    if (s.name != expected_name) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': StateVariable '{}' {} index points "
          "to SymbolDecl named '{}'; expected '{}' (derived from "
          "StateVariable.name). The symbols() entry has been renamed or "
          "the indices have shifted.",
          model.name(), sv.name, which, s.name, expected_name));
    }
    if (s.category != expected_cat) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': StateVariable '{}' {} symbol has "
          "wrong Category (expected={}, got={}). State variable / symbol "
          "vectors are out of sync.",
          model.name(), sv.name, which,
          static_cast<int>(expected_cat),
          static_cast<int>(s.category)));
    }
    if (s.kind != sv.kind) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': StateVariable '{}' {} symbol has Kind "
          "mismatched against the state-variable record (expected={}, "
          "got={}).",
          model.name(), sv.name, which,
          static_cast<int>(sv.kind),
          static_cast<int>(s.kind)));
    }
    if (sv.kind == SymbolDecl::Kind::Tensor &&
        (s.dim != sv.dim || s.rank != sv.rank)) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': StateVariable '{}' {} symbol has "
          "dim/rank ({}, {}) mismatched against the state-variable "
          "record ({}, {}).",
          model.name(), sv.name, which, s.dim, s.rank, sv.dim, sv.rank));
    }
  };
  for (auto const &sv : model.state_variables()) {
    check(sv.current_symbol_idx, sv.name,
          SymbolDecl::Category::StateVariableCurrent, sv, "current");
    auto const old_name =
        std::string{sv.name} + std::string{state_variable_old_suffix};
    check(sv.old_symbol_idx, old_name,
          SymbolDecl::Category::StateVariableOld, sv, "old");
  }
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

// Phase 2.6 (issue #77): the single source of truth for the generated
// function's parameter order. Both `render_compute_function` (signature)
// and every backend call-site iterate this so they cannot drift. See
// `ArgSpec` in recipe_render.h for the role → declaration / MOOSE-argument
// mapping.
inline auto canonical_arguments(RecipeView model) -> std::vector<ArgSpec> {
  auto const &m = model.raw_model();

  // State variables solved in-function (local-Newton) become out-params,
  // not inputs (issue #60): the function returns the converged value.
  std::set<std::string> newton_owned;
  if (m.local_newton_enabled()) {
    for (auto const &eq : m.evolution_equations()) {
      newton_owned.insert(
          m.symbols()[m.state_variables()[eq.state_variable_idx]
                          .current_symbol_idx]
              .name);
    }
  }

  std::vector<ArgSpec> args;
  for (auto const &s : m.symbols()) {
    if (newton_owned.contains(s.name)) {
      continue; // emitted as NewtonStateOut below
    }
    // Classify by CATEGORY first, then kind (PR #78 review #1) — so a
    // tensor that is a state variable / time step isn't mis-read as a
    // tensor input. Tensor state variables are out of Phase 2.x scope;
    // fail loudly rather than silently emit scalar wiring for them.
    ArgSpec::Role role;
    if (s.is_time_step) {
      role = ArgSpec::Role::TimeStep;
    } else if (s.category == SymbolDecl::Category::StateVariableOld) {
      if (s.kind == SymbolDecl::Kind::Tensor) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': tensor state variable '{}' is not "
            "supported yet (scalar state variables only).",
            m.name(), s.name));
      }
      role = ArgSpec::Role::StateOld;
    } else if (s.category == SymbolDecl::Category::StateVariableCurrent) {
      if (s.kind == SymbolDecl::Kind::Tensor) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': tensor state variable '{}' is not "
            "supported yet (scalar state variables only).",
            m.name(), s.name));
      }
      role = ArgSpec::Role::StateCurrentRead; // non-Newton current
    } else if (s.category == SymbolDecl::Category::Parameter) {
      if (s.kind == SymbolDecl::Kind::Tensor) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': tensor parameter '{}' is not "
            "supported (parameters are scalar).",
            m.name(), s.name));
      }
      role = ArgSpec::Role::ScalarParam;
    } else { // Category::Input
      role = s.kind == SymbolDecl::Kind::Tensor ? ArgSpec::Role::TensorInput
                                                : ArgSpec::Role::ScalarInput;
    }
    args.push_back({s.name, role, s.dim, s.rank});
  }
  for (auto const &o : m.outputs()) {
    args.push_back({o.name,
                    o.kind == OutputDecl::Kind::Tensor
                        ? ArgSpec::Role::TensorOutput
                        : ArgSpec::Role::ScalarOutput,
                    o.dim, o.rank});
  }
  // In-function-solved state variables, in evolution-equation order (the
  // same order render_compute_function emits the loops + writebacks).
  if (m.local_newton_enabled()) {
    for (auto const &eq : m.evolution_equations()) {
      auto const &name =
          m.symbols()[m.state_variables()[eq.state_variable_idx]
                          .current_symbol_idx]
              .name;
      args.push_back({name, ArgSpec::Role::NewtonStateOut, 0, 0});
    }
  }
  return args;
}

inline auto render_compute_function(
    RecipeView model, CodeGenContext const &ctx,
    std::vector<std::string> const &output_rhs,
    std::vector<RenderedNewtonSegment> const &newton) -> std::string {
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

  // Signature built from the canonical argument list (issue #77) — the
  // exact order backends must reproduce in their call-sites.
  int tmpl_counter = 0;
  bool first = true;
  for (auto const &a : canonical_arguments(model)) {
    if (!first) {
      os << ",\n";
    }
    first = false;
    os << "    ";
    // Every role enumerated explicitly (PR #78 review #5): no `default:`,
    // so adding an ArgSpec::Role is a -Wswitch compile warning here.
    switch (a.role) {
    case ArgSpec::Role::TensorInput:
      os << "T" << tmpl_counter++ << " const &" << a.name;
      break;
    case ArgSpec::Role::TensorOutput:
      os << "T" << tmpl_counter++ << " &" << a.name << "_out";
      break;
    case ArgSpec::Role::ScalarOutput:
    case ArgSpec::Role::NewtonStateOut:
      os << "double &" << a.name << "_out";
      break;
    case ArgSpec::Role::ScalarParam:
    case ArgSpec::Role::ScalarInput:
    case ArgSpec::Role::StateOld:
    case ArgSpec::Role::StateCurrentRead:
    case ArgSpec::Role::TimeStep:
      os << "double const " << a.name;
      break;
    }
  }
  os << ") {\n";

  // Newton iteration segments first — they declare + solve the local
  // state-variable iterates that downstream output expressions reference.
  for (auto const &seg : newton) {
    os << "  double " << seg.state_var_name << " = " << seg.state_var_name
       << "_old;\n";
    os << "  for (int " << seg.state_var_name << "_iter = 0; "
       << seg.state_var_name << "_iter < " << seg.max_iter << "; ++"
       << seg.state_var_name << "_iter) {\n";
    os << seg.loop_local_decls; // already 4-indented, trailing newline
    os << "    double " << seg.state_var_name << "_R = " << seg.residual_rhs
       << ";\n";
    os << "    double " << seg.state_var_name << "_J = " << seg.jacobian_rhs
       << ";\n";
    os << "    if (std::abs(" << seg.state_var_name << "_R) < " << seg.tol
       << ") {\n      break;\n    }\n";
    os << "    " << seg.state_var_name << " -= " << seg.state_var_name
       << "_R / " << seg.state_var_name << "_J;\n";
    os << "  }\n";
  }

  os << ctx.render_statements();

  for (std::size_t i = 0; i < model.outputs().size(); ++i) {
    os << "  " << model.outputs()[i].name << "_out = " << output_rhs[i]
       << ";\n";
  }

  // Write converged state-variable values back to their out-params.
  for (auto const &seg : newton) {
    os << "  " << seg.state_var_name << "_out = " << seg.state_var_name
       << ";\n";
  }

  os << "}\n";
  return os.str();
}

} // namespace numsim::codegen

// Inline pass run() definitions + the backward-Euler residual helper live
// in internal impl headers, included here AFTER ConstitutiveModel and the
// RecipeView delegates are complete (the bodies call their accessors). This
// keeps recipe.h focused on the IR + builder surface while preserving the
// "include recipe.h, get the whole framework" contract.
#include <numsim_codegen/passes/internal/pass_bodies.h>

#endif // NUMSIM_CODEGEN_RECIPE_H
