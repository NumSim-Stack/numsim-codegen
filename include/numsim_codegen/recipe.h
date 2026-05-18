#ifndef NUMSIM_CODEGEN_RECIPE_H
#define NUMSIM_CODEGEN_RECIPE_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>

#include <numsim_cas/scalar/scalar_all.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>

#include <cstddef>
#include <optional>
#include <sstream>
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
    m_symbols.push_back(std::move(decl));
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
    m_symbols.push_back(std::move(decl));
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

  [[nodiscard]] auto emit_compute_function(bool emit_includes = true) const
      -> std::string {
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

    return render_compute_function(ctx, output_rhs, emit_includes);
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

  // Parameters as a separate filtered view (useful for MOOSE's validParams).
  [[nodiscard]] auto parameters() const -> std::vector<SymbolDecl> {
    std::vector<SymbolDecl> out;
    for (auto const &s : m_symbols) {
      if (s.category == SymbolDecl::Category::Parameter) {
        out.push_back(s);
      }
    }
    return out;
  }

  [[nodiscard]] auto inputs() const -> std::vector<SymbolDecl> {
    std::vector<SymbolDecl> out;
    for (auto const &s : m_symbols) {
      if (s.category == SymbolDecl::Category::Input) {
        out.push_back(s);
      }
    }
    return out;
  }

private:
  [[nodiscard]] auto render_compute_function(
      CodeGenContext const &ctx,
      std::vector<std::string> const &output_rhs,
      bool emit_includes) const -> std::string {
    std::ostringstream os;

    os << "// Auto-generated by numsim-codegen. Do not edit.\n";
    os << "// Model: " << m_name << "\n\n";
    if (emit_includes) {
      os << "#include <tmech/tmech.h>\n";
      os << "#include <cmath>\n\n";
    }

    os << "inline void " << m_name << "_compute(\n";

    bool first = true;
    for (auto const &s : m_symbols) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << param_decl(s);
    }

    for (auto const &out : m_outputs) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << output_decl(out);
    }
    os << ") {\n";

    os << ctx.render_statements();

    for (std::size_t i = 0; i < m_outputs.size(); ++i) {
      os << "  " << m_outputs[i].name << "_out = " << output_rhs[i] << ";\n";
    }

    os << "}\n";
    return os.str();
  }

  static auto param_decl(SymbolDecl const &s) -> std::string {
    std::ostringstream os;
    if (s.kind == SymbolDecl::Kind::Scalar) {
      os << "double const " << s.name;
    } else {
      os << "tmech::tensor<double, " << s.dim << ", " << s.rank << "> const &"
         << s.name;
    }
    return os.str();
  }

  static auto output_decl(OutputDecl const &o) -> std::string {
    std::ostringstream os;
    if (o.kind == OutputDecl::Kind::Scalar) {
      os << "double &" << o.name << "_out";
    } else {
      os << "tmech::tensor<double, " << o.dim << ", " << o.rank << "> &"
         << o.name << "_out";
    }
    return os.str();
  }

  std::string m_name;
  std::vector<SymbolDecl> m_symbols;
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
