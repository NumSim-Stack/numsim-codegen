#ifndef NUMSIM_CODEGEN_RECIPE_H
#define NUMSIM_CODEGEN_RECIPE_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/leaf_collector.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>

#include <cstddef>
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
  // std::runtime_error listing the missing symbols. Called automatically
  // from emit_compute_function() but exposed for explicit pre-flight checks.
  void validate() const {
    LeafCollector lc;
    for (auto const &o : m_outputs) {
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
    for (auto const &[name, _] : m_scalar_symbols) declared_scalar.insert(name);
    for (auto const &[name, _] : m_tensor_symbols) declared_tensor.insert(name);

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
      std::string msg = "ConstitutiveModel '" + m_name +
                        "': outputs reference undeclared symbol(s):";
      for (auto const &m : missing) msg += "\n  - " + m;
      msg += "\nCall add_scalar_input / add_tensor_input / add_parameter "
             "before referencing a symbol in an output expression.";
      throw std::runtime_error(msg);
    }
  }

  // Emit the generic compute function. Does not emit #include directives;
  // backends own the file-level framing (includes, namespacing, registration).
  [[nodiscard]] auto emit_compute_function() const -> std::string {
    validate();
    CodeGenContext ctx;
    ScalarCodeEmit scalar_emit(ctx);
    TensorCodeEmit tensor_emit(ctx, scalar_emit);
    TensorToScalarCodeEmit t2s_emit(ctx, scalar_emit, tensor_emit);

    for (auto const &[name, expr] : m_scalar_symbols) {
      ctx.register_symbol_scalar(expr, name);
    }
    for (auto const &[name, expr] : m_tensor_symbols) {
      ctx.register_symbol_tensor(expr, name);
    }

    std::vector<std::string> output_rhs;
    output_rhs.reserve(m_outputs.size());
    for (auto const &out : m_outputs) {
      output_rhs.push_back(std::visit(
          [&](auto const &e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<
                              T,
                              cas::expression_holder<cas::scalar_expression>>) {
              return scalar_emit.apply(e);
            } else {
              return tensor_emit.apply(e);
            }
          },
          out.expr));
    }

    return render_compute_function(ctx, output_rhs);
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

  [[nodiscard]] auto render_compute_function(
      CodeGenContext const &ctx,
      std::vector<std::string> const &output_rhs) const -> std::string {
    std::ostringstream os;

    os << "// Auto-generated by numsim-codegen. Do not edit.\n";
    os << "// Model: " << m_name << "\n\n";

    // Template parameter list: one T<n> per tensor argument so the caller
    // can pass any tmech::tensor_base subclass (tensor, adaptor, expression
    // template) without forcing a materialised copy.
    int const n_tmpl = tensor_arg_count();
    if (n_tmpl > 0) {
      os << "template <";
      for (int i = 0; i < n_tmpl; ++i) {
        if (i > 0) os << ", ";
        os << "typename T" << i;
      }
      os << ">\n";
    }
    os << "inline void " << m_name << "_compute(\n";

    int tmpl_counter = 0;
    bool first = true;
    for (auto const &s : m_symbols) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << param_decl(s, tmpl_counter);
    }

    for (auto const &out : m_outputs) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << output_decl(out, tmpl_counter);
    }
    os << ") {\n";

    os << ctx.render_statements();

    for (std::size_t i = 0; i < m_outputs.size(); ++i) {
      os << "  " << m_outputs[i].name << "_out = " << output_rhs[i] << ";\n";
    }

    os << "}\n";
    return os.str();
  }

  static auto param_decl(SymbolDecl const &s, int &tmpl_counter)
      -> std::string {
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

  static auto output_decl(OutputDecl const &o, int &tmpl_counter)
      -> std::string {
    std::ostringstream os;
    if (o.kind == OutputDecl::Kind::Scalar) {
      os << "double &" << o.name << "_out";
    } else {
      os << "T" << tmpl_counter++ << " &" << o.name << "_out";
    }
    return os.str();
  }

  // Count how many tensor parameters / outputs need template params.
  [[nodiscard]] auto tensor_arg_count() const -> int {
    int n = 0;
    for (auto const &s : m_symbols) {
      if (s.kind == SymbolDecl::Kind::Tensor) ++n;
    }
    for (auto const &o : m_outputs) {
      if (o.kind == OutputDecl::Kind::Tensor) ++n;
    }
    return n;
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

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_H
