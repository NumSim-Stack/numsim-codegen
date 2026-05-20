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
#include <variant>
#include <vector>

namespace numsim::codegen {

// Semantic tag indicating what role an input plays in the constitutive
// model. Backends interpret these to wire the recipe to framework-specific
// inputs — Strain becomes MaterialProperty<RankTwoTensor>("strain") in
// MOOSE, the STRAN array in Abaqus UMAT, etc.
enum class InputRole {
  Strain,             // ε — primary driving variable
  StrainIncrement,    // Δε — Abaqus DSTRAN, viscoplastic increment
  DeformationGradient,// F — finite-strain models
  Stress,             // σ — for stress-driven formulations
  Temperature,        // T
  History,            // a state variable carried from the previous step
  Other,              // generic input the backend treats as a coupled var
};

// Semantic tag for outputs. Backends route these to the right framework
// sink — Stress to _stress[_qp] in MOOSE / STRESS in Abaqus UMAT, etc.
enum class OutputRole {
  Stress,             // σ
  ConsistentTangent,  // dσ/dε
  HistoryNew,         // updated state variable
  Dissipation,        // for dissipation-checking
  Other,
};

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
  InputRole role = InputRole::Other;  // semantic tag; Other for parameters
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
  OutputRole role = OutputRole::Other;
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

  auto add_scalar_input(std::string name, InputRole role = InputRole::Other)
      -> cas::expression_holder<cas::scalar_expression> {
    auto var = cas::make_expression<cas::scalar>(name);
    SymbolDecl decl{name, SymbolDecl::Category::Input,
                    SymbolDecl::Kind::Scalar};
    decl.role = role;
    m_symbols.push_back(decl);
    m_inputs_cache.push_back(decl);
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  auto add_tensor_input(std::string name, std::size_t dim, std::size_t rank,
                        InputRole role = InputRole::Other)
      -> cas::expression_holder<cas::tensor_expression> {
    auto var = cas::make_expression<cas::tensor>(name, dim, rank);
    SymbolDecl decl{name, SymbolDecl::Category::Input,
                    SymbolDecl::Kind::Tensor, dim, rank};
    decl.role = role;
    m_symbols.push_back(decl);
    m_inputs_cache.push_back(decl);
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
    m_symbols.push_back(decl);
    m_parameters_cache.push_back(decl);
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  // ─── Output declarations ────────────────────────────────────────

  void add_output(std::string name,
                  cas::expression_holder<cas::scalar_expression> expr,
                  OutputRole role = OutputRole::Other) {
    OutputDecl decl{name, OutputDecl::Kind::Scalar, expr, 0, 0, role};
    m_outputs.push_back(std::move(decl));
  }

  void add_output(std::string name,
                  cas::expression_holder<cas::tensor_expression> expr,
                  OutputRole role = OutputRole::Other) {
    OutputDecl decl{name, OutputDecl::Kind::Tensor, expr, expr.get().dim(),
                    expr.get().rank(), role};
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
  [[nodiscard]] auto find_input_by_role(InputRole role) const
      -> SymbolDecl const * {
    for (auto const &s : m_symbols) {
      if (s.category == SymbolDecl::Category::Input && s.role == role) {
        return &s;
      }
    }
    return nullptr;
  }

  [[nodiscard]] auto find_output_by_role(OutputRole role) const
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
