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

// Declaration of an external symbol (input, parameter) that the generated
// function takes as a parameter.
struct SymbolDecl {
  enum class Role { Input, Parameter };
  enum class Kind { Scalar, Tensor };

  std::string name;
  Role role;
  Kind kind;
  std::size_t dim = 0;   // tensor only
  std::size_t rank = 0;  // tensor only
  std::optional<double> default_value; // parameter only
  std::string doc;
};

// Declaration of a computed output that the generated function returns.
struct OutputDecl {
  enum class Kind { Scalar, Tensor };

  std::string name;
  Kind kind;
  std::variant<
      cas::expression_holder<cas::scalar_expression>,
      cas::expression_holder<cas::tensor_expression>>
      expr;
  // For tensor outputs, store the realised dim/rank so the codegen knows the
  // return type without re-querying the expression (the expression might be
  // moved / consumed elsewhere by the time emit() runs).
  std::size_t dim = 0;
  std::size_t rank = 0;
};

// The constitutive-model registry. Holds declared inputs, parameters, and
// outputs; emits a self-contained C++ function that computes the outputs
// from the inputs. Phase A scope — no history, no internal Newton solves,
// no MOOSE Material wrapping yet.
class ConstitutiveModel {
public:
  explicit ConstitutiveModel(std::string name) : m_name(std::move(name)) {}

  [[nodiscard]] auto name() const -> std::string const & { return m_name; }

  // ─── Input declarations ─────────────────────────────────────────

  auto add_scalar_input(std::string name)
      -> cas::expression_holder<cas::scalar_expression> {
    auto var = cas::make_expression<cas::scalar>(name);
    m_symbols.push_back(
        {name, SymbolDecl::Role::Input, SymbolDecl::Kind::Scalar});
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  auto add_tensor_input(std::string name, std::size_t dim, std::size_t rank)
      -> cas::expression_holder<cas::tensor_expression> {
    auto var = cas::make_expression<cas::tensor>(name, dim, rank);
    m_symbols.push_back({name, SymbolDecl::Role::Input,
                         SymbolDecl::Kind::Tensor, dim, rank});
    m_tensor_symbols.emplace_back(name, var);
    return var;
  }

  // ─── Parameter declarations ─────────────────────────────────────

  auto add_parameter(std::string name, double default_value,
                     std::string doc = "")
      -> cas::expression_holder<cas::scalar_expression> {
    auto var = cas::make_expression<cas::scalar>(name);
    SymbolDecl decl{name, SymbolDecl::Role::Parameter,
                    SymbolDecl::Kind::Scalar};
    decl.default_value = default_value;
    decl.doc = std::move(doc);
    m_symbols.push_back(std::move(decl));
    m_scalar_symbols.emplace_back(name, var);
    return var;
  }

  // ─── Output declarations ────────────────────────────────────────

  void add_output(std::string name,
                  cas::expression_holder<cas::scalar_expression> expr) {
    OutputDecl decl{name, OutputDecl::Kind::Scalar, expr, 0, 0};
    m_outputs.push_back(std::move(decl));
  }

  void add_output(std::string name,
                  cas::expression_holder<cas::tensor_expression> expr) {
    OutputDecl decl{name, OutputDecl::Kind::Tensor, expr, expr.get().dim(),
                    expr.get().rank()};
    m_outputs.push_back(std::move(decl));
  }

  // ─── Emit ───────────────────────────────────────────────────────

  // Emit a self-contained C++ function `<name>_compute` that takes the
  // declared inputs/parameters and writes the outputs to out-parameters.
  [[nodiscard]] auto emit() const -> std::string {
    CodeGenContext ctx;
    ScalarCodeEmit scalar_emit(ctx);
    TensorCodeEmit tensor_emit(ctx, scalar_emit);
    TensorToScalarCodeEmit t2s_emit(ctx, scalar_emit, tensor_emit);

    // Register every declared symbol so the visitor uses the user-facing
    // name (e.g. "lam", "eps") instead of allocating a temporary.
    for (auto const &[name, expr] : m_scalar_symbols) {
      ctx.register_symbol_scalar(expr, name);
    }
    for (auto const &[name, expr] : m_tensor_symbols) {
      ctx.register_symbol_tensor(expr, name);
    }

    // Walk every output. Each call appends its dependency chain (in topo
    // order) to the shared statement list.
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

    return render_function(ctx, output_rhs);
  }

  // ─── Read-only accessors (for downstream MOOSE-wrapper generation) ──

  [[nodiscard]] auto symbols() const -> std::vector<SymbolDecl> const & {
    return m_symbols;
  }

  [[nodiscard]] auto outputs() const -> std::vector<OutputDecl> const & {
    return m_outputs;
  }

private:
  [[nodiscard]] auto render_function(
      CodeGenContext const &ctx,
      std::vector<std::string> const &output_rhs) const -> std::string {
    std::ostringstream os;

    os << "// Auto-generated by numsim-codegen. Do not edit.\n";
    os << "// Model: " << m_name << "\n\n";
    os << "#include <tmech/tmech.h>\n";
    os << "#include <cmath>\n\n";

    os << "inline void " << m_name << "_compute(\n";

    // Parameters (read-only)
    bool first = true;
    for (auto const &s : m_symbols) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << param_decl(s, /*as_input=*/true);
    }

    // Outputs (write)
    for (auto const &out : m_outputs) {
      if (!first) {
        os << ",\n";
      }
      first = false;
      os << "    " << output_decl(out);
    }
    os << ") {\n";

    // Body — temporaries in topological order.
    os << ctx.render_statements();

    // Assignments to out-parameters.
    for (std::size_t i = 0; i < m_outputs.size(); ++i) {
      os << "  " << m_outputs[i].name << "_out = " << output_rhs[i] << ";\n";
    }

    os << "}\n";
    return os.str();
  }

  static auto param_decl(SymbolDecl const &s, bool as_input) -> std::string {
    std::ostringstream os;
    if (s.kind == SymbolDecl::Kind::Scalar) {
      os << "double " << (as_input ? "const " : "") << s.name;
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

  // Storage for the symbol expression_holders. Needed because the codegen
  // context uses pointer identity for lookup — the holders must outlive the
  // codegen walk.
  std::vector<
      std::pair<std::string, cas::expression_holder<cas::scalar_expression>>>
      m_scalar_symbols;
  std::vector<
      std::pair<std::string, cas::expression_holder<cas::tensor_expression>>>
      m_tensor_symbols;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_RECIPE_H
