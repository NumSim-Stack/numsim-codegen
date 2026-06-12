#include <numsim_codegen/targets/numsim_material.h>

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/leaf_collector.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/core/diff.h>

#include <cmath>
#include <format>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::codegen {

namespace {

// ── numsim-materials contract surface ──────────────────────────────────────
// The property/parameter names and include this target emits against. They
// live in the numsim-materials repo with no compile-time link, so a drift only
// surfaces at the downstream compile. Hoisted here so that drift is a one-line
// change and the magic strings have a single home. Validated against
// numsim-materials `materials/curing_rate.h` + `solvers/rk_integrator.h`.
namespace contract {
constexpr char const *rate_property = "rate";
constexpr char const *rate_derivative_property = "rate_derivative";
constexpr char const *state_property = "state";
constexpr char const *integrator_source_param = "integrator_source";
constexpr char const *material_base_include =
    "numsim-materials/core/material_base.h";
} // namespace contract

// A recipe symbol whose name equals one of the emitted fixed members would
// produce a duplicate member / duplicate ctor initializer in the generated
// material (uncompilable). Reject at emit time. Note `state_property` ("state")
// is intentionally NOT reserved: it is only an edge-property KEY string, never
// an emitted C++ member name (the state member is `m_<state>`), so a recipe
// whose state/parameter is literally named `state` is legitimate.
bool is_reserved_name(std::string const &n) {
  return n == contract::rate_property ||
         n == contract::rate_derivative_property ||
         n == contract::integrator_source_param;
}

// Round-trip-exact double formatting for emitted C++ / JSON. `std::format("{}")`
// uses the shortest representation that round-trips (charconv), unlike a bare
// `ostream <<` which truncates to ~6 significant figures.
std::string fmt(double v) { return std::format("{}", v); }

// First-increment scope: exactly one scalar state variable + one scalar
// evolution equation `dx/dt = f(x, params)`, and NOTHING this increment can't
// emit. Rejecting (rather than silently emitting a partial material) is the
// whole contract — a code generator that quietly drops a declared output is a
// correctness hazard.
void check_scope(ConstitutiveModel const &model) {
  auto const svs = model.state_variables();
  auto const eqs = model.evolution_equations();
  if (svs.size() != 1 || eqs.size() != 1) {
    throw std::runtime_error(
        "NumSimMaterialTarget: first increment supports exactly one scalar "
        "state variable with one scalar evolution equation (the rk_integrator "
        "rate contract). Coupled / multi-state systems need the "
        "numsim-materials vector solver (numsim-materials#12).");
  }
  if (svs[0].kind != SymbolDecl::Kind::Scalar) {
    // Defensive: unreachable through the public API today (evolution equations
    // are scalar-only — add_scalar_evolution_equation binds only scalar state),
    // so a tensor state has no equation and is caught above. Kept for the day a
    // tensor-evolution API lands; until then this branch cannot fire.
    throw std::runtime_error(
        "NumSimMaterialTarget: tensor-valued state is not yet supported "
        "(needs Mandel + the vector solver, numsim-materials#11/#12).");
  }
  // Features this increment does not emit — reject loudly instead of dropping.
  if (!model.outputs().empty()) {
    throw std::runtime_error(
        "NumSimMaterialTarget: add_output(...) outputs (stress, etc.) are not "
        "yet emitted as material properties — Phase B follow-up. Got '" +
        model.outputs().front().name + "'.");
  }
  if (!model.tangents().empty()) {
    // Defensive: a tangent's `of_output` names an existing output, so any
    // tangent-bearing recipe is already rejected by the outputs() guard above.
    // Kept so the rejection is explicit if outputs ever become supported first.
    throw std::runtime_error(
        "NumSimMaterialTarget: algorithmic tangents are out of scope for the "
        "rate material (Phase D emits tangent-block properties).");
  }
  if (!model.inputs().empty()) {
    throw std::runtime_error(
        "NumSimMaterialTarget: declared inputs ('" + model.inputs().front().name +
        "') are not yet wired into the rate material — scalar inputs are a "
        "Phase B follow-up.");
  }
}

} // namespace

auto NumSimMaterialTarget::emit(ConstitutiveModel const &model) const
    -> std::vector<EmittedFile> {
  check_scope(model);

  auto const &sv = model.state_variables()[0];
  auto const &eq = model.evolution_equations()[0];
  auto const &cur_name = model.symbols()[sv.current_symbol_idx].name;

  // Scalar parameters the rate may reference (skip the framework time-step:
  // the integrator owns discretization, so a rate material never sees `dt`).
  std::vector<SymbolDecl> params;
  std::set<std::string> param_names;
  for (auto const &p : model.parameters()) {
    if (p.is_time_step || p.kind != SymbolDecl::Kind::Scalar) continue;
    params.push_back(p);
    param_names.insert(p.name);
  }

  // Reserved-name guard: a state/parameter named like an emitted fixed member
  // (`rate`, `rate_derivative`, `integrator_source`) would collide.
  if (is_reserved_name(cur_name)) {
    throw std::runtime_error(
        "NumSimMaterialTarget: state variable name '" + cur_name +
        "' collides with an emitted member; rename it.");
  }
  for (auto const &p : params) {
    if (is_reserved_name(p.name)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: parameter name '" + p.name +
          "' collides with an emitted member (rate/rate_derivative/"
          "integrator_source); rename it.");
    }
    // A non-finite default would emit `value_type{nan}`/`{inf}` (no such C++
    // literal) and an invalid JSON number — reject rather than emit broken code.
    if (p.default_value.has_value() && !std::isfinite(*p.default_value)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: parameter '" + p.name +
          "' has a non-finite default (" + fmt(*p.default_value) +
          "); cannot emit as a C++ literal or JSON number.");
    }
  }

  // Rate-leaf guard: every scalar symbol the rate references must be the state
  // or a parameter, else compute() would emit an unbound identifier. This
  // catches a rate that (wrongly for the rate contract) uses `dt`, the `_old`
  // state, or an input — converting a confusing downstream compile error into a
  // clear emit-time message.
  {
    LeafCollector lc;
    lc.collect_scalar(eq.rate);
    for (auto const &n : lc.scalar_names()) {
      if (n != cur_name && !param_names.contains(n)) {
        throw std::runtime_error(
            "NumSimMaterialTarget: the rate expression references '" + n +
            "', which is neither the state '" + cur_name +
            "' nor a parameter. A rate function f(x, params) cannot depend on "
            "dt, the previous-step state, or inputs (the integrator owns "
            "discretization).");
      }
    }
  }

  // Resolve the current-state scalar holder (the diff variable), mirroring
  // detail::build_backward_euler_residual's single symbol-map walk.
  cas::expression_holder<cas::scalar_expression> cur_expr;
  for (auto const &[name, h] : model.scalar_symbol_map()) {
    if (name == cur_name) cur_expr = h;
  }
  if (!cur_expr.is_valid()) {
    throw std::runtime_error(
        "NumSimMaterialTarget: cannot resolve the current-state scalar handle "
        "for '" + sv.name + "' — state/symbol vectors out of sync.");
  }

  // Render `rate = f` and `rate_derivative = ∂f/∂x` with one shared CSE block.
  // Emit-name mapping: the state current → a local (bare `cur_name`, bound from
  // m_<state>.get()); parameters → their member `m_<name>`; the rate-leaf guard
  // above guarantees nothing else can appear.
  CodeGenContext ctx;
  CodeEmitPipeline pipeline(ctx);
  for (auto const &[name, expr] : model.scalar_symbol_map()) {
    if (param_names.contains(name)) {
      ctx.register_symbol_scalar(expr, "m_" + name);
    } else {
      ctx.register_symbol_scalar(expr, name); // cur_name (a local) + the rest
    }
  }
  ctx.reset(); // clear CSE + statements; keep symbol registrations + counter
  auto const rate_rhs = pipeline.scalar().apply(eq.rate);
  auto const drate_rhs = pipeline.scalar().apply(cas::diff(eq.rate, cur_expr));
  auto const decls = ctx.render_statements("    ");

  auto const &cls = model.name();

  // ── Material header ────────────────────────────────────────────────
  std::ostringstream h;
  h << "// Auto-generated by numsim-codegen (NumSimMaterialTarget). Do not "
       "edit.\n";
  h << "// Rate-function material for recipe \"" << cls << "\": d" << cur_name
    << "/dt = f(" << cur_name << ").\n";
  h << "// Conforms to the numsim-materials rk_integrator contract: exposes\n";
  h << "// `" << contract::rate_property << "` (= f) and `"
    << contract::rate_derivative_property << "` (= df/d" << cur_name
    << ") as Local-edge\n";
  h << "// properties; reads " << cur_name << " from the integrator's `"
    << contract::state_property << "`.\n";
  h << "#pragma once\n";
  h << "#include \"" << contract::material_base_include << "\"\n";
  h << "#include <utility>\n\n";
  h << "namespace numsim::materials::generated {\n\n";
  h << "template <typename Traits>\n";
  h << "class " << cls << " final\n";
  h << "    : public numsim::materials::material_base<" << cls
    << "<Traits>, Traits> {\n";
  h << "public:\n";
  h << "  using base = numsim::materials::material_base<" << cls
    << "<Traits>, Traits>;\n";
  h << "  using value_type = typename base::value_type;\n";
  h << "  using input_parameter_controller =\n"
       "      typename base::input_parameter_controller;\n\n";

  // Constructor. Member-declaration order (below) matches the initialization
  // order; m_<state>'s add_input depends on m_integrator_source, which is
  // declared/initialized first — keep that ordering invariant.
  h << "  template <typename... Args>\n";
  h << "  explicit " << cls << "(Args&&... args)\n";
  h << "      : base(std::forward<Args>(args)...),\n";
  h << "        m_rate(base::template add_output<value_type>(\n";
  h << "            \"" << contract::rate_property << "\", &" << cls
    << "::compute)),\n";
  h << "        m_rate_derivative(\n";
  h << "            base::template add_output<value_type>(\""
    << contract::rate_derivative_property << "\")),\n";
  for (auto const &p : params) {
    h << "        m_" << p.name
      << "(base::template get_parameter<value_type>(\"" << p.name << "\")),\n";
  }
  h << "        m_integrator_source(\n";
  h << "            base::template get_parameter<std::string>(\""
    << contract::integrator_source_param << "\")),\n";
  h << "        m_" << cur_name << "(base::template add_input<value_type>(\n";
  h << "            m_integrator_source, \"" << contract::state_property
    << "\",\n";
  h << "            numsim::materials::EdgeKind::Local)) {}\n\n";

  // parameters() schema
  h << "  static input_parameter_controller parameters() {\n";
  h << "    input_parameter_controller para{base::parameters()};\n";
  for (auto const &p : params) {
    if (p.default_value.has_value()) {
      h << "    para.template insert<value_type>(\"" << p.name
        << "\").template add<numsim_core::set_default>(value_type{"
        << fmt(*p.default_value) << "});\n";
    } else {
      // Defensive: `add_parameter` always sets a default, so this is currently
      // unreachable via the public API. Kept (with the JSON-omission sibling
      // below) for the day a no-default/required-parameter API lands.
      h << "    para.template insert<value_type>(\"" << p.name
        << "\").template add<numsim_core::is_required>();\n";
    }
  }
  h << "    para.template insert<std::string>(\""
    << contract::integrator_source_param << "\")\n";
  h << "        .template add<numsim_core::is_required>();\n";
  h << "    return para;\n";
  h << "  }\n\n";

  // compute()
  h << "  // " << contract::rate_property << " = f(" << cur_name << "); "
    << contract::rate_derivative_property << " = df/d" << cur_name
    << " (cas::diff).\n";
  h << "  void compute() {\n";
  // [[maybe_unused]]: a state-independent rate (e.g. dx/dt = const) never reads
  // the state local — keep the binding uniform but suppress -Wunused-variable in
  // a -Werror consumer build.
  h << "    [[maybe_unused]] const auto " << cur_name << " = m_" << cur_name
    << ".get();\n";
  if (!decls.empty()) h << decls;
  h << "    m_rate = " << rate_rhs << ";\n";
  h << "    m_rate_derivative = " << drate_rhs << ";\n";
  h << "  }\n\n";

  // members
  h << "private:\n";
  h << "  value_type& m_rate;\n";
  h << "  value_type& m_rate_derivative;\n";
  for (auto const &p : params) {
    // [[maybe_unused]]: a parameter declared but not referenced by the rate is
    // still stored (it stays in the schema/JSON) — suppress clang's
    // -Wunused-private-field in a -Werror consumer build.
    h << "  [[maybe_unused]] const value_type& m_" << p.name << ";\n";
  }
  h << "  const std::string& m_integrator_source;\n";
  h << "  const numsim::materials::input_property<value_type,\n";
  h << "      numsim::materials::property_traits>& m_" << cur_name << ";\n";
  h << "};\n\n";
  h << "} // namespace numsim::materials::generated\n";

  // ── JSON config (a SCAFFOLD — placeholder integration policy) ──────
  std::ostringstream j;
  j << "{\n";
  j << "  \"_comment\": \"SCAFFOLD generated by numsim-codegen. step_size and "
       "tableau are PLACEHOLDERS — set them for your problem. The 'type' names "
       "must be registered in the material factory before create_from_json "
       "can instantiate them.\",\n";
  j << "  \"materials\": [\n";
  j << "    { \"name\": \"integrator\", \"type\": \"rk_integrator\",\n";
  j << "      \"function\": \"" << cls << "\",\n";
  j << "      \"step_size\": 0.01, \"tableau\": \"implicit_euler\",\n";
  j << "      \"tolerance\": 1e-12, \"max_iter\": 50 },\n";
  j << "    { \"name\": \"" << cls << "\", \"type\": \"" << cls << "\",\n";
  j << "      \"" << contract::integrator_source_param << "\": \"integrator\"";
  for (auto const &p : params) {
    // Only emit a value for parameters that HAVE a default; a required
    // parameter must be supplied by the user, not faked with 0.
    if (p.default_value.has_value()) {
      j << ",\n      \"" << p.name << "\": " << fmt(*p.default_value);
    }
  }
  j << " }\n";
  j << "  ]\n";
  j << "}\n";

  return {
      EmittedFile{cls + ".h", h.str(), "include/materials",
                  EmittedFile::Kind::Header},
      EmittedFile{cls + ".config.json", j.str(), "", EmittedFile::Kind::Other},
  };
}

auto NumSimMaterialTarget::target_name() const -> std::string {
  return "NumSimMaterial";
}

} // namespace numsim::codegen
