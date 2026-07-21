#include <numsim_codegen/targets/numsim_material.h>

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/leaf_collector.h>
#include <numsim_codegen/passes/internal/algorithmic_tangent.h>
#include <numsim_codegen/recipe.h>

#include <numsim_cas/core/diff.h>
#include <numsim_cas/scalar/scalar_constant.h>
#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_diff.h>
#include <numsim_cas/tensor/tensor_functions.h>
#include <numsim_cas/tensor/tensor_operators.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_diff.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_functions.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_operators.h>

#include <cmath>
#include <format>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
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
// Mode-B (strain-coupled implicit residual) contract surface. The material
// holds a material_ref<backward_euler> and drives the Newton loop itself via
// solve(eval). Validated against numsim-materials `materials/
// small_strain_plasticity.h` + `solvers/backward_euler.h`.
constexpr char const *solver_source_param = "solver_source";
constexpr char const *backward_euler_include =
    "numsim-materials/solvers/backward_euler.h";
constexpr char const *material_ref_include =
    "numsim-materials/core/material_ref.h";
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

// The tmech tensor type a rank-`rank`, `dim`-D property uses — matching the
// numsim-materials graph's `tmech::tensor<value_type, Dim, Rank>`. The recipe's
// `dim` must equal the consuming Traits::Dim (property types are matched by
// dynamic_cast at wire time); recipes are 3-D today, as is the default policy.
std::string tensor_cxx_type(std::size_t dim, std::size_t rank) {
  return "tmech::tensor<value_type, " + std::to_string(dim) + ", " +
         std::to_string(rank) + ">";
}

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
  // Outputs (scalar AND tensor, e.g. stress = f(state, strain)) are emitted as
  // properties with their own update callbacks. Only tensor INTERNAL STATE is
  // still blocked (Mandel + vector solver) — a tensor stress from a scalar state
  // needs neither, so it is supported here.
  // Algorithmic/consistent tangents dσ/dε ARE emitted (Phase D) as rank-4 tensor
  // properties via cas::diff. In THIS target's scope the rate cannot depend on
  // strain (rate-leaf guard), so the integrated state is independent of the
  // current strain (dx/dε ≡ 0) and dσ/dε = ∂σ/∂ε exactly — no J⁻¹ solve / no
  // rank-4 inverse, just diff(stress, strain). Validate each spec resolves.
  for (auto const &t : model.tangents()) {
    bool of_ok = false;
    for (auto const &o : model.outputs())
      if (o.name == t.of_output && o.kind == OutputDecl::Kind::Tensor) of_ok = true;
    if (!of_ok) {
      throw std::runtime_error(
          "NumSimMaterialTarget: tangent '" + t.name + "' differentiates '" +
          t.of_output + "', which is not a declared tensor output (stress).");
    }
    bool wrt_ok = false;
    for (auto const &in : model.inputs())
      if (in.name == t.wrt_input && in.kind == SymbolDecl::Kind::Tensor) wrt_ok = true;
    if (!wrt_ok) {
      throw std::runtime_error(
          "NumSimMaterialTarget: tangent '" + t.name + "' differentiates w.r.t. '" +
          t.wrt_input + "', which is not a declared tensor input (strain).");
    }
  }
  // Tensor inputs (e.g. strain) ARE wired (Global-edge input_property); scalar
  // inputs are a separate small follow-up — reject those loudly for now.
  for (auto const &in : model.inputs()) {
    if (in.kind != SymbolDecl::Kind::Tensor) {
      throw std::runtime_error(
          "NumSimMaterialTarget: scalar input '" + in.name +
          "' is not yet wired into the material — a Phase B follow-up (tensor "
          "inputs like strain ARE supported).");
    }
  }
}

// ── Mode-B strain-coupled residual emission ────────────────────────────────
// A recipe defined by an implicit residual R(x, ε)=0 (add_scalar_residual_
// equation) is lowered to a numsim-materials material that holds a
// material_ref<backward_euler> and drives the Newton loop itself in ONE
// compute() (bound to the stress output — the always-pulled driver):
//
//   compute() {
//     auto eval = [&](value_type dz) {                  // trial x = x_old + dz
//       value_type x = m_x.old_value() + dz;
//       return std::pair{ R(x, ε), dR/dx(x, ε) };       // both t2s → scalar
//     };
//     value_type dz = m_solver.get().solve(eval);
//     m_x.new_value() = m_x.old_value() + dz;
//     m_out_stress = σ(x, ε); ...                        // every output, post-solve
//   }
//
// This is the robust Mode B (small_strain_plasticity's pattern): one compute()
// owns solve + state-apply + every output, so there is NO Local-edge ordering
// fragility (contrast a Mode-A material exposing residual/jacobian PROPERTIES,
// where the solve-vs-apply order rides on hash order). The consistent tangent
// dσ/dε is a follow-up (PR 2b) — tangents are rejected here.
//
// ⚠ backward_euler::solve() CLAMPS its result with std::max(x, 0) — a
// plasticity convention (a plastic-multiplier increment is physically ≥0). A
// generated residual material therefore correctly models only states whose
// solved INCREMENT is non-negative. This is surfaced in the emitted header;
// lifting it needs an unclamped solver mode upstream (numsim-materials).
std::vector<EmittedFile> emit_residual_material(ConstitutiveModel const &model) {
  // ── Scope validation (residual contract) ──
  auto const reqs = model.residual_equations();
  auto const svs = model.state_variables();
  auto const ueqs = model.update_equations();
  // Exactly ONE scalar Newton unknown (the residual's state). Any OTHER state
  // variable must be an INTERNAL variable — set by a post-solve update equation,
  // not solved. This is the #92 path: a scalar solve (no vector solver / Mandel)
  // with tensor/scalar HISTORY (e.g. J2 plastic strain εᵖ) carried across steps.
  if (reqs.size() != 1) {
    throw std::runtime_error(
        "NumSimMaterialTarget: a residual material needs exactly one residual "
        "equation (one scalar Newton unknown). Coupled multi-unknown return maps "
        "need the numsim-materials vector solver (numsim-materials#12).");
  }
  {
    std::size_t const newton_idx = reqs[0].state_variable_idx;
    std::set<std::size_t> updated;
    for (auto const &ue : ueqs) {
      if (ue.state_variable_idx == newton_idx) {
        throw std::runtime_error(
            "NumSimMaterialTarget: the Newton state '" +
            svs[newton_idx].name +
            "' has both a residual and an update equation — a state is either "
            "solved (residual) or updated post-solve (internal variable), not "
            "both.");
      }
      if (!updated.insert(ue.state_variable_idx).second) {
        throw std::runtime_error(
            "NumSimMaterialTarget: internal variable '" +
            svs[ue.state_variable_idx].name +
            "' has more than one update equation.");
      }
    }
    for (std::size_t i = 0; i < svs.size(); ++i) {
      if (i == newton_idx || updated.contains(i)) continue;
      throw std::runtime_error(
          "NumSimMaterialTarget: state variable '" + svs[i].name +
          "' is neither the Newton unknown (a residual) nor an internal "
          "variable (an update equation). Add one via "
          "add_scalar/tensor_update_equation, or remove it.");
    }
  }
  if (!model.evolution_equations().empty()) {
    // Unreachable through the public API (a state carries a rate XOR a residual,
    // enforced by the recipe), but a residual recipe must never also carry a
    // rate — guard against a future API that could mix them.
    throw std::runtime_error(
        "NumSimMaterialTarget: a residual recipe must not also declare rate "
        "(evolution) equations — the state is defined by the residual alone.");
  }
  for (auto const &in : model.inputs()) {
    if (in.kind != SymbolDecl::Kind::Tensor) {
      throw std::runtime_error(
          "NumSimMaterialTarget: scalar input '" + in.name +
          "' is not yet wired into a residual material — a follow-up (tensor "
          "inputs like strain ARE supported).");
    }
  }
  if (model.outputs().empty()) {
    throw std::runtime_error(
        "NumSimMaterialTarget: a residual material needs at least one output "
        "(e.g. stress) to anchor compute() — the output's pull drives the "
        "Newton solve. A state-only solve has no consumer.");
  }

  auto const &req = reqs[0];
  auto const &sv = svs[req.state_variable_idx];
  auto const &cur_name = model.symbols()[sv.current_symbol_idx].name;

  // Scalar parameters the residual / outputs may reference (skip the framework
  // time step — a residual material is rate-independent in this increment).
  std::vector<SymbolDecl> params;
  std::set<std::string> param_names;
  for (auto const &p : model.parameters()) {
    if (p.is_time_step || p.kind != SymbolDecl::Kind::Scalar) continue;
    params.push_back(p);
    param_names.insert(p.name);
  }

  std::vector<SymbolDecl> tensor_inputs;
  std::set<std::string> tensor_input_names;
  for (auto const &in : model.inputs()) {
    tensor_inputs.push_back(in);
    tensor_input_names.insert(in.name);
  }

  // #92: internal variables (state vars set by a post-solve update equation).
  // Each carries its own history (scalar or tensor); their `_old` value is bound
  // as a local and referenced by the residual/outputs/updates; their `_new` is
  // set post-solve. `<name>` (current) is NOT bound — only `<name>_old`.
  struct InternalVar {
    std::string name;      // current bare name (add_history_output key + member)
    std::string old_name;  // "<name>_old" — the bound local
    bool is_tensor = false;
    std::size_t dim = 0, rank = 0;
    UpdateEquation const *eq = nullptr;
  };
  std::vector<InternalVar> internals;
  std::set<std::string> internal_names;                       // current names
  std::set<std::string> internal_scalar_old, internal_tensor_old; // bound _old
  for (auto const &ue : ueqs) {
    auto const &isv = svs[ue.state_variable_idx];
    bool const is_tensor = isv.kind == SymbolDecl::Kind::Tensor;
    std::string const old_name =
        isv.name + std::string{state_variable_old_suffix};
    internals.push_back(
        {isv.name, old_name, is_tensor, isv.dim, isv.rank, &ue});
    internal_names.insert(isv.name);
    (is_tensor ? internal_tensor_old : internal_scalar_old).insert(old_name);
  }
  std::string const cur_old_name =
      cur_name + std::string{state_variable_old_suffix};

  // Reserved-name guard. A residual material emits the fixed member `m_solver`
  // (the material_ref), the schema key `solver_source`, and the compute()-local
  // identifiers `eval`, `residual`, `jacobian`. A recipe symbol that renders to
  // one of these bare names would produce a duplicate member / redeclared local
  // (uncompilable). Applied to the state, parameters AND tensor inputs — all
  // three render to bare identifiers in compute() (params are m_-prefixed in
  // expressions, but `solver_source`/`solver` would still collide with the
  // member/schema, so params are checked too).
  auto is_reserved_residual = [](std::string const &n) {
    return n == contract::solver_source_param || n == "solver" || n == "eval" ||
           n == "residual" || n == "jacobian";
  };
  auto reject_reserved = [&](std::string const &n, char const *what) {
    if (is_reserved_residual(n)) {
      throw std::runtime_error(
          std::string("NumSimMaterialTarget: ") + what + " '" + n +
          "' collides with an emitted member / local "
          "(solver/solver_source/eval/residual/jacobian); rename it.");
    }
  };
  reject_reserved(cur_name, "state variable name");
  for (auto const &p : params) {
    reject_reserved(p.name, "parameter name");
    if (p.default_value.has_value() && !std::isfinite(*p.default_value)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: parameter '" + p.name +
          "' has a non-finite default (" + fmt(*p.default_value) +
          "); cannot emit as a C++ literal or JSON number.");
    }
  }
  // The Newton increment is a compute()-local named `d<state>`; a tensor input
  // (also a bare compute() local) named the same would redeclare it.
  std::string const increment_local = "d" + cur_name;
  for (auto const &in : tensor_inputs) {
    reject_reserved(in.name, "tensor input name");
    if (in.name == increment_local) {
      throw std::runtime_error(
          "NumSimMaterialTarget: tensor input '" + in.name +
          "' collides with the synthesized Newton-increment local '" +
          increment_local + "'; rename the input or the state.");
    }
  }

  // Resolve the current-state scalar holder — the diff variable for ∂R/∂x.
  cas::expression_holder<cas::scalar_expression> cur_expr;
  for (auto const &[name, h] : model.scalar_symbol_map()) {
    if (name == cur_name) cur_expr = h;
  }
  if (!cur_expr.is_valid()) {
    throw std::runtime_error(
        "NumSimMaterialTarget: cannot resolve the current-state scalar handle "
        "for '" + sv.name + "' — state/symbol vectors out of sync.");
  }

  // Register scalar symbols (state→local bare name, params→m_<name>) and tensor
  // symbols (input→local bare name) into an emit context. Shared by the residual
  // /jacobian render and reused (fresh contexts) per output.
  auto register_all = [&](CodeGenContext &c) {
    for (auto const &[name, expr] : model.scalar_symbol_map()) {
      if (param_names.contains(name)) {
        c.register_symbol_scalar(expr, "m_" + name);
      } else {
        c.register_symbol_scalar(expr, name);
      }
    }
    for (auto const &[name, expr] : model.tensor_symbol_map()) {
      c.register_symbol_tensor(expr, name);
    }
  };

  // Leaf-guard predicates: a scalar leaf must be the Newton state (current), its
  // `_old`, a parameter, or an internal variable's scalar `_old`; a tensor leaf
  // must be a declared tensor input or an internal variable's tensor `_old`. All
  // of these are bound as locals in compute() (#92 binds the `_old` values); a
  // reference to anything else would emit an unbound identifier. Reused for the
  // residual, the outputs, and the update equations.
  auto bound_scalar = [&](std::string const &n) {
    return n == cur_name || n == cur_old_name || param_names.contains(n) ||
           internal_scalar_old.contains(n);
  };
  auto bound_tensor = [&](std::string const &n) {
    return tensor_input_names.contains(n) || internal_tensor_old.contains(n);
  };
  {
    LeafCollector rlc;
    rlc.collect_t2s(req.residual);
    for (auto const &n : rlc.scalar_names()) {
      if (!bound_scalar(n)) {
        throw std::runtime_error(
            "NumSimMaterialTarget: the residual references scalar '" + n +
            "', which is not the state '" + cur_name +
            "', its previous value, a parameter, or an internal variable's "
            "previous value.");
      }
    }
    for (auto const &n : rlc.tensor_names()) {
      if (!bound_tensor(n)) {
        throw std::runtime_error(
            "NumSimMaterialTarget: the residual references tensor '" + n +
            "', which is not a declared tensor input or an internal variable's "
            "previous value.");
      }
    }
  }

  // Residual R and jacobian ∂R/∂x rendered with one shared CSE block (both live
  // inside the eval lambda, so the state local and any tensor-input local they
  // reference are in scope there).
  CodeGenContext rc;
  CodeEmitPipeline rp(rc);
  register_all(rc);
  rc.reset();
  auto const res_rhs = rp.t2s().apply(req.residual);
  auto const jac_rhs = rp.t2s().apply(cas::diff(req.residual, cur_expr));
  auto const eval_decls = rc.render_statements("      "); // inside the lambda

  // Outputs (stress + any others), each in a fresh context so the CSE temp
  // counter restarts. All are computed AFTER the solve, inside compute().
  struct EmittedOutput {
    std::string name, decls, rhs;
    bool is_tensor = false;
    std::size_t dim = 0, rank = 0;
  };
  std::vector<EmittedOutput> outputs;
  for (auto const &o : model.outputs()) {
    if (is_reserved_residual(o.name) || o.name == cur_name ||
        param_names.contains(o.name) || tensor_input_names.contains(o.name) ||
        internal_names.contains(o.name)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: output name '" + o.name +
          "' collides with an emitted member, the state, a parameter, a "
          "tensor input, or an internal variable; rename it.");
    }
    CodeGenContext oc;
    CodeEmitPipeline op(oc);
    register_all(oc);
    if (o.kind == OutputDecl::Kind::Scalar) {
      auto const &oexpr =
          std::get<cas::expression_holder<cas::scalar_expression>>(o.expr);
      LeafCollector olc;
      olc.collect_scalar(oexpr);
      for (auto const &n : olc.scalar_names()) {
        if (!bound_scalar(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: scalar output '" + o.name +
              "' references '" + n + "', which is neither the state '" +
              cur_name + "' nor a parameter.");
        }
      }
      oc.reset();
      auto const orhs = op.scalar().apply(oexpr);
      outputs.push_back({o.name, oc.render_statements("    "), orhs, false, 0, 0});
    } else {
      auto const &oexpr =
          std::get<cas::expression_holder<cas::tensor_expression>>(o.expr);
      LeafCollector olc;
      olc.collect_tensor(oexpr);
      for (auto const &n : olc.scalar_names()) {
        if (!bound_scalar(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: tensor output '" + o.name +
              "' references scalar '" + n + "', which is neither the state '" +
              cur_name + "' nor a parameter.");
        }
      }
      for (auto const &n : olc.tensor_names()) {
        if (!bound_tensor(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: tensor output '" + o.name +
              "' references tensor '" + n +
              "', which is not a declared tensor input.");
        }
      }
      oc.reset();
      auto const orhs = op.tensor().apply(oexpr);
      outputs.push_back(
          {o.name, oc.render_statements("    "), orhs, true, o.dim, o.rank});
    }
  }

  // ── Internal-variable post-solve updates (#92) ──
  // Each internal variable's `_new` value is set from its update expression
  // (referencing the converged Newton state, any `_old` values, params, inputs).
  struct EmittedUpdate {
    std::string name, decls, rhs;
    bool is_tensor = false;
  };
  std::vector<EmittedUpdate> internal_updates;
  for (auto const &iv : internals) {
    CodeGenContext uc;
    CodeEmitPipeline up(uc);
    register_all(uc);
    std::string rhs;
    if (iv.is_tensor) {
      auto const &uexpr =
          std::get<cas::expression_holder<cas::tensor_expression>>(
              iv.eq->update);
      LeafCollector ulc;
      ulc.collect_tensor(uexpr);
      for (auto const &n : ulc.scalar_names())
        if (!bound_scalar(n))
          throw std::runtime_error("NumSimMaterialTarget: update of '" +
                                   iv.name + "' references unbound scalar '" +
                                   n + "'.");
      for (auto const &n : ulc.tensor_names())
        if (!bound_tensor(n))
          throw std::runtime_error("NumSimMaterialTarget: update of '" +
                                   iv.name + "' references unbound tensor '" +
                                   n + "'.");
      uc.reset();
      rhs = up.tensor().apply(uexpr);
    } else {
      auto const &uexpr =
          std::get<cas::expression_holder<cas::scalar_expression>>(
              iv.eq->update);
      LeafCollector ulc;
      ulc.collect_scalar(uexpr);
      for (auto const &n : ulc.scalar_names())
        if (!bound_scalar(n))
          throw std::runtime_error("NumSimMaterialTarget: update of '" +
                                   iv.name + "' references unbound scalar '" +
                                   n + "'.");
      uc.reset();
      rhs = up.scalar().apply(uexpr);
    }
    internal_updates.push_back(
        {iv.name, uc.render_statements("    "), rhs, iv.is_tensor});
  }

  // ── Consistent tangent(s) dσ/dε (strain-coupled, Mode B) ──
  // For a requested tangent of a stress output σ w.r.t. a strain input ε:
  //   dσ/dε = ∂σ/∂ε + ∂σ/∂z ⊗ dz/dε ,   dz/dε = −∂R/∂ε / ∂R/∂z
  // where z is the scalar Newton state. ∂R/∂z is the scalar jacobian (a scalar —
  // NO matrix inverse, since the state is scalar), so the whole tangent is a
  // rank-4 tensor EXPRESSION in (z, ε, params). It is emitted as an extra
  // post-solve output, evaluated at the converged z exactly like the stress
  // (and, like every output, bound to &compute). ∂σ/∂z is taken through the
  // detail::diff_tensor_wrt_scalar seam (cas#275).
  for (auto const &t : model.tangents()) {
    // of_output must be an emitted TENSOR output (the stress).
    OutputDecl const *src = nullptr;
    for (auto const &o : model.outputs()) {
      if (o.name == t.of_output) {
        src = &o;
        break;
      }
    }
    if (src == nullptr) {
      throw std::runtime_error(
          "NumSimMaterialTarget: consistent tangent '" + t.name +
          "' references output '" + t.of_output +
          "', which is not a declared output.");
    }
    if (src->kind != OutputDecl::Kind::Tensor) {
      throw std::runtime_error(
          "NumSimMaterialTarget: consistent tangent '" + t.name +
          "' differentiates output '" + t.of_output +
          "', which is scalar — dσ/dε requires a tensor (stress) output.");
    }
    // wrt_input must be a declared tensor input (the strain), and share σ's dim.
    SymbolDecl const *arg = nullptr;
    for (auto const &ti : tensor_inputs) {
      if (ti.name == t.wrt_input) {
        arg = &ti;
        break;
      }
    }
    if (arg == nullptr) {
      throw std::runtime_error(
          "NumSimMaterialTarget: consistent tangent '" + t.name +
          "' differentiates w.r.t. '" + t.wrt_input +
          "', which is not a declared tensor input.");
    }
    if (arg->dim != src->dim) {
      throw std::runtime_error(
          "NumSimMaterialTarget: consistent tangent '" + t.name + "': output '" +
          t.of_output + "' (dim " + std::to_string(src->dim) + ") and input '" +
          t.wrt_input + "' (dim " + std::to_string(arg->dim) +
          ") have different tensor dimensions.");
    }
    // The tangent output name shares the same collision surface as any output.
    if (is_reserved_residual(t.name) || t.name == cur_name ||
        param_names.contains(t.name) || tensor_input_names.contains(t.name)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: consistent-tangent name '" + t.name +
          "' collides with an emitted member, the state, a parameter, or a "
          "tensor input; rename it.");
    }
    for (auto const &o : outputs) {
      if (o.name == t.name) {
        throw std::runtime_error(
            "NumSimMaterialTarget: consistent-tangent name '" + t.name +
            "' collides with output '" + o.name + "'; rename it.");
      }
    }

    // Resolve σ and ε expression handles.
    auto const &sigma =
        std::get<cas::expression_holder<cas::tensor_expression>>(src->expr);
    cas::expression_holder<cas::tensor_expression> eps;
    for (auto const &[name, h] : model.tensor_symbol_map()) {
      if (name == t.wrt_input) eps = h;
    }
    if (!eps.is_valid()) {
      throw std::runtime_error(
          "NumSimMaterialTarget: cannot resolve the tensor-input handle for '" +
          t.wrt_input + "' — input/symbol maps out of sync.");
    }

    // dσ/dε = ∂σ/∂ε + otimes(∂σ/∂z, (−1/∂R/∂z)·∂R/∂ε).
    auto const dsig_deps = cas::diff(sigma, eps);                     // rank-4
    auto const dsig_dz = detail::diff_tensor_wrt_scalar(sigma, cur_expr); // rank-2
    auto const dR_deps = cas::diff(req.residual, eps);               // rank-2
    auto const dR_dz = cas::diff(req.residual, cur_expr);            // scalar
    auto const neg1 = cas::make_expression<cas::scalar_constant>(-1.0);
    auto const dz_deps = (neg1 / dR_dz) * dR_deps;                   // rank-2
    auto const tangent = dsig_deps + cas::otimes(dsig_dz, dz_deps);  // rank-4

    CodeGenContext tc;
    CodeEmitPipeline tp(tc);
    register_all(tc);
    tc.reset();
    auto const trhs = tp.tensor().apply(tangent);
    // dim from the operands (checked equal); rank = rank(σ) + rank(ε).
    outputs.push_back({t.name, tc.render_statements("    "), trhs, true,
                       src->dim, src->rank + arg->rank});
  }

  // Emitted-member uniqueness guard (same hazard as the rate path: synthesized
  // member names can collide with recipe symbols).
  {
    std::set<std::string> member_bases;
    auto claim = [&member_bases](std::string const &base) {
      if (!member_bases.insert(base).second) {
        throw std::runtime_error(
            "NumSimMaterialTarget: emitted member 'm_" + base +
            "' would be duplicated — a recipe symbol collides with a synthesized "
            "member name; rename the offending state/parameter/input/output.");
      }
    };
    claim("solver"); // the fixed material_ref member m_solver
    claim(cur_name);
    for (auto const &iv : internals) claim(iv.name); // internal-var histories
    for (auto const &p : params) claim(p.name);
    for (auto const &o : outputs) claim("out_" + o.name);
    for (auto const &ti : tensor_inputs) {
      claim(ti.name);
      claim(ti.name + "_source");
    }
  }

  auto const &cls = model.name();
  auto const &driver = outputs.front().name; // compute() is bound to this output

  // ── Material header ──
  std::ostringstream h;
  h << "// Auto-generated by numsim-codegen (NumSimMaterialTarget). Do not "
       "edit.\n";
  h << "// Strain-coupled residual material for recipe \"" << cls
    << "\": state " << cur_name << " solves R(" << cur_name
    << ", inputs) = 0\n";
  h << "// (implicit, Mode B). Holds a material_ref<backward_euler> and drives "
       "the\n";
  h << "// Newton loop in compute() (bound to the '" << driver
    << "' output); the\n";
  h << "// converged state then feeds every output.\n";
  h << "//\n";
  h << "// CONTRACT: pull the '" << driver
    << "' output to drive the solve. NOTE backward_euler::solve()\n";
  h << "// clamps its increment to >= 0 (a plasticity convention), so this "
       "material\n";
  h << "// models only states whose solved increment is non-negative. This "
       "caveat\n";
  h << "// applies to the CONSISTENT TANGENT dσ/dε too (if emitted): it is\n";
  h << "// evaluated at the solved state via the implicit-function theorem, so "
       "on a\n";
  h << "// clamped (elastic) increment where the raw root is negative the "
       "tangent is\n";
  h << "// outside its domain of validity — the correct value there is ∂σ/∂ε "
       "alone.\n";
  h << "#pragma once\n";
  h << "#include \"" << contract::material_base_include << "\"\n";
  h << "#include \"" << contract::backward_euler_include << "\"\n";
  h << "#include \"" << contract::material_ref_include << "\"\n";
  h << "#include <tmech/tmech.h>\n";
  h << "#include <stdexcept>\n";
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
       "      typename base::input_parameter_controller;\n";
  h << "  using solver_type = numsim::materials::backward_euler<Traits>;\n\n";

  // Constructor.
  h << "  template <typename... Args>\n";
  h << "  explicit " << cls << "(Args&&... args)\n";
  h << "      : base(std::forward<Args>(args)...),\n";
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto const &o = outputs[i];
    std::string const ty =
        o.is_tensor ? tensor_cxx_type(o.dim, o.rank) : "value_type";
    h << "        m_out_" << o.name << "(base::template add_output<" << ty
      << ">(\n";
    // EVERY output carries &compute: pulling ANY output drives the one compute()
    // that solves the state and sets all outputs. (compute() is idempotent
    // within a graph evaluation — it re-solves to the same root — so a consumer
    // pulling several outputs is correct, if mildly redundant.) Binding only the
    // first output would leave the others stale when pulled in isolation.
    h << "            \"" << o.name << "\", &" << cls << "::compute)),\n";
    (void)i;
  }
  h << "        m_" << cur_name
    << "(base::template add_history_output<value_type>(\"" << cur_name
    << "\")),\n";
  for (auto const &iv : internals) { // #92 internal-variable histories
    std::string const ty =
        iv.is_tensor ? tensor_cxx_type(iv.dim, iv.rank) : "value_type";
    h << "        m_" << iv.name << "(base::template add_history_output<" << ty
      << ">(\"" << iv.name << "\")),\n";
  }
  for (auto const &p : params) {
    h << "        m_" << p.name
      << "(base::template get_parameter<value_type>(\"" << p.name << "\")),\n";
  }
  h << "        m_solver(base::template add_material_ref<solver_type>(\n";
  h << "            base::template get_parameter<std::string>(\""
    << contract::solver_source_param << "\")))";
  if (tensor_inputs.empty()) {
    h << " {}\n\n";
  } else {
    h << ",\n";
    for (std::size_t i = 0; i < tensor_inputs.size(); ++i) {
      auto const &ti = tensor_inputs[i];
      h << "        m_" << ti.name << "(base::template add_input<"
        << tensor_cxx_type(ti.dim, ti.rank) << ">(\n";
      h << "            base::template get_parameter<std::string>(\"" << ti.name
        << "_source\"),\n";
      h << "            \"" << ti.name
        << "\", numsim::materials::EdgeKind::Global))"
        << (i + 1 < tensor_inputs.size() ? ",\n" : " {}\n\n");
    }
  }

  // parameters() schema
  h << "  static input_parameter_controller parameters() {\n";
  h << "    input_parameter_controller para{base::parameters()};\n";
  for (auto const &p : params) {
    if (p.default_value.has_value()) {
      h << "    para.template insert<value_type>(\"" << p.name
        << "\").template add<numsim_core::set_default>(value_type{"
        << fmt(*p.default_value) << "});\n";
    } else {
      h << "    para.template insert<value_type>(\"" << p.name
        << "\").template add<numsim_core::is_required>();\n";
    }
  }
  h << "    para.template insert<std::string>(\""
    << contract::solver_source_param << "\")\n";
  h << "        .template add<numsim_core::is_required>();\n";
  for (auto const &ti : tensor_inputs) {
    h << "    para.template insert<std::string>(\"" << ti.name << "_source\")\n";
    h << "        .template add<numsim_core::is_required>();\n";
  }
  h << "    return para;\n";
  h << "  }\n\n";

  // compute()
  h << "  // Solves R(" << cur_name << ", inputs)=0 for the increment via\n";
  h << "  // backward_euler::solve(), applies the state, then computes every "
       "output.\n";
  h << "  void compute() {\n";
  for (auto const &ti : tensor_inputs) {
    h << "    [[maybe_unused]] const auto& " << ti.name << " = m_" << ti.name
      << ".get();\n";
  }
  // #92: bind the previous-step values. `<state>_old` and every internal
  // variable's `_old` are captured by the eval lambda (residual may reference
  // them) AND used post-solve (outputs + updates).
  h << "    [[maybe_unused]] const value_type " << cur_old_name << " = m_"
    << cur_name << ".old_value();\n";
  for (auto const &iv : internals) {
    if (iv.is_tensor)
      h << "    [[maybe_unused]] const auto& " << iv.old_name << " = m_"
        << iv.name << ".old_value();\n";
    else
      h << "    [[maybe_unused]] const value_type " << iv.old_name << " = m_"
        << iv.name << ".old_value();\n";
  }
  h << "    auto eval = [&](value_type d" << cur_name
    << ") -> std::pair<value_type, value_type> {\n";
  h << "      const value_type " << cur_name << " = m_" << cur_name
    << ".old_value() + d" << cur_name << ";\n";
  if (!eval_decls.empty()) h << eval_decls;
  h << "      const value_type residual = " << res_rhs << ";\n";
  h << "      const value_type jacobian = " << jac_rhs << ";\n";
  h << "      return {residual, jacobian};\n";
  h << "    };\n";
  h << "    const value_type d" << cur_name
    << " = m_solver.get().solve(eval);\n";
  // Reject loudly if the Newton solve did not converge: every output (stress AND
  // the consistent tangent) is evaluated at this state, so a non-converged root
  // would silently feed a wrong stress / wrong stiffness into the graph. (Note:
  // this does NOT catch backward_euler's std::max(x,0) clamp returning a
  // non-root on the elastic branch — see the header caveat; that needs an
  // upstream was_clamped() accessor, numsim-materials follow-up.)
  h << "    if (!m_solver.get().converged())\n";
  h << "      throw std::runtime_error(\"" << cls
    << "::compute: backward_euler did not converge solving R(" << cur_name
    << ", inputs)=0; stress/tangent would be evaluated at a non-root state.\");\n";
  h << "    m_" << cur_name << ".new_value() = m_" << cur_name
    << ".old_value() + d" << cur_name << ";\n";
  h << "    [[maybe_unused]] const value_type " << cur_name << " = m_"
    << cur_name << ".new_value();\n";
  // #92: internal-variable updates (post-solve, before outputs — outputs may
  // read the updated state). Brace-scoped for CSE-temp isolation like outputs.
  for (auto const &u : internal_updates) {
    h << "    {\n";
    if (!u.decls.empty()) h << u.decls;
    h << "      m_" << u.name << ".new_value() = " << u.rhs << ";\n";
    h << "    }\n";
  }
  for (auto const &o : outputs) {
    // Each output is rendered in its OWN CodeGenContext, so its CSE temporaries
    // restart at t0 — and ALL outputs share this single compute() body. Without
    // a nested scope, a second output with CSE temps would redeclare `t0` (a
    // compile error the single-output test recipe never hit). Brace-scope every
    // output so each block's temps are private. (A decl-free output gets a
    // harmless empty-ish block.)
    h << "    {\n";
    if (!o.decls.empty()) h << o.decls; // rendered at 4-space; nested is fine
    h << "      m_out_" << o.name << " = " << o.rhs << ";\n";
    h << "    }\n";
  }
  h << "  }\n\n";

  // members
  h << "private:\n";
  for (auto const &o : outputs) {
    std::string const ty =
        o.is_tensor ? tensor_cxx_type(o.dim, o.rank) : "value_type";
    h << "  " << ty << "& m_out_" << o.name << ";\n";
  }
  h << "  numsim_core::history_property<value_type>& m_" << cur_name << ";\n";
  for (auto const &iv : internals) { // #92 internal-variable histories
    std::string const ty =
        iv.is_tensor ? tensor_cxx_type(iv.dim, iv.rank) : "value_type";
    h << "  numsim_core::history_property<" << ty << ">& m_" << iv.name
      << ";\n";
  }
  for (auto const &p : params) {
    h << "  [[maybe_unused]] const value_type& m_" << p.name << ";\n";
  }
  h << "  numsim::materials::material_ref<solver_type, Traits>& m_solver;\n";
  for (auto const &ti : tensor_inputs) {
    h << "  const numsim::materials::input_property<"
      << tensor_cxx_type(ti.dim, ti.rank) << ",\n";
    h << "      numsim::materials::property_traits>& m_" << ti.name << ";\n";
  }
  h << "};\n\n";
  h << "} // namespace numsim::materials::generated\n";

  // ── JSON config scaffold ──
  std::ostringstream j;
  j << "{\n";
  j << "  \"_comment\": \"SCAFFOLD generated by numsim-codegen. The "
       "backward_euler solver is created WITHOUT a 'function' (caller-driven "
       "mode — the material drives the Newton loop). The '*_source' values are "
       "PLACEHOLDER producer names; a material publishing a property of the "
       "matching name/type must exist in the graph. The 'type' names must be "
       "registered in the material factory.\",\n";
  j << "  \"materials\": [\n";
  j << "    { \"name\": \"solver\", \"type\": \"backward_euler\",\n";
  j << "      \"tolerance\": 1e-12, \"max_iter\": 50 },\n";
  j << "    { \"name\": \"" << cls << "\", \"type\": \"" << cls << "\",\n";
  j << "      \"" << contract::solver_source_param << "\": \"solver\"";
  for (auto const &ti : tensor_inputs) {
    j << ",\n      \"" << ti.name << "_source\": \"" << ti.name << "_producer\"";
  }
  for (auto const &p : params) {
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

} // namespace

auto NumSimMaterialTarget::emit(ConstitutiveModel const &model) const
    -> std::vector<EmittedFile> {
  // Strain-coupled implicit-residual recipes take the Mode-B path; the rate
  // (rk_integrator) path below handles the rest. Routing here (before
  // check_scope) keeps each path's scope validation self-contained.
  if (!model.residual_equations().empty()) {
    return emit_residual_material(model);
  }
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

  // Tensor inputs (e.g. strain): each is wired from a producer material via a
  // Global-edge input_property, read by the property name `<input-name>` (the
  // producer must publish a property of that name). Each gets a `<name>_source`
  // string parameter naming the producer. check_scope already rejected scalars.
  std::vector<SymbolDecl> tensor_inputs;
  std::set<std::string> tensor_input_names;
  for (auto const &in : model.inputs()) {
    tensor_inputs.push_back(in);
    tensor_input_names.insert(in.name);
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

  // Register the scalar symbols (state→local, params→m_<name>) into a fresh
  // emit context — shared by every per-output render below.
  auto register_scalars = [&](CodeGenContext &c) {
    for (auto const &[name, expr] : model.scalar_symbol_map()) {
      if (param_names.contains(name)) {
        c.register_symbol_scalar(expr, "m_" + name);
      } else {
        c.register_symbol_scalar(expr, name);
      }
    }
  };

  // Outputs become their own `<name>` property + `update_<name>()` callback,
  // rendered in a FRESH context each so the CSE temp-counter restarts per
  // callback. Scalar outputs read state+params; tensor outputs additionally read
  // tensor inputs (e.g. stress = g(state, strain)).
  struct EmittedOutput {
    std::string name, decls, rhs;
    bool is_tensor = false;
    std::size_t dim = 0, rank = 0;
  };
  std::vector<EmittedOutput> outputs;
  for (auto const &o : model.outputs()) {
    // Name-collision guard (common): the output member `m_out_<name>` / property
    // key must not clash with a fixed member, the state, a parameter, or a
    // tensor input.
    if (is_reserved_name(o.name) || o.name == cur_name ||
        param_names.contains(o.name) || tensor_input_names.contains(o.name)) {
      throw std::runtime_error(
          "NumSimMaterialTarget: output name '" + o.name +
          "' collides with an emitted member, the state, a parameter, or a "
          "tensor input; rename it.");
    }

    CodeGenContext oc;
    CodeEmitPipeline op(oc);
    register_scalars(oc);

    if (o.kind == OutputDecl::Kind::Scalar) {
      auto const &oexpr =
          std::get<cas::expression_holder<cas::scalar_expression>>(o.expr);
      LeafCollector olc;
      olc.collect_scalar(oexpr);
      for (auto const &n : olc.scalar_names()) {
        if (n != cur_name && !param_names.contains(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: scalar output '" + o.name +
              "' references '" + n + "', which is neither the state '" +
              cur_name + "' nor a parameter.");
        }
      }
      oc.reset();
      auto const orhs = op.scalar().apply(oexpr);
      outputs.push_back({o.name, oc.render_statements("    "), orhs, false, 0, 0});
    } else {
      auto const &oexpr =
          std::get<cas::expression_holder<cas::tensor_expression>>(o.expr);
      // Tensor-output leaf guard: scalar leaves (coefficients like α) must be
      // the state or a parameter; tensor leaves must be declared tensor inputs.
      LeafCollector olc;
      olc.collect_tensor(oexpr);
      for (auto const &n : olc.scalar_names()) {
        if (n != cur_name && !param_names.contains(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: tensor output '" + o.name +
              "' references scalar '" + n + "', which is neither the state '" +
              cur_name + "' nor a parameter.");
        }
      }
      for (auto const &n : olc.tensor_names()) {
        if (!tensor_input_names.contains(n)) {
          throw std::runtime_error(
              "NumSimMaterialTarget: tensor output '" + o.name +
              "' references tensor '" + n +
              "', which is not a declared tensor input.");
        }
      }
      // Register tensor inputs by their bare local name (bound in the callback
      // from m_<name>.get()).
      for (auto const &[name, expr] : model.tensor_symbol_map()) {
        oc.register_symbol_tensor(expr, name);
      }
      oc.reset();
      auto const orhs = op.tensor().apply(oexpr);
      outputs.push_back(
          {o.name, oc.render_statements("    "), orhs, true, o.dim, o.rank});
    }
  }

  // Consistent tangents dσ/dε: a rank-4 tensor property = cas::diff(stress, strain).
  // Folded into `outputs` as rank-4 tensor outputs, so the ctor/callback/member
  // emission below handles them uniformly. The strain's roles::Strain symmetric
  // space makes diff return the minor-symmetric rank-4 identity (P_sym), so the
  // tangent is minor-symmetric as a stress-strain tangent must be.
  for (auto const &t : model.tangents()) {
    cas::expression_holder<cas::tensor_expression> sigma;
    std::size_t tdim = 0;
    for (auto const &o : model.outputs()) {
      if (o.name == t.of_output && o.kind == OutputDecl::Kind::Tensor) {
        sigma = std::get<cas::expression_holder<cas::tensor_expression>>(o.expr);
        tdim = o.dim;
      }
    }
    cas::expression_holder<cas::tensor_expression> eps;
    for (auto const &[name, h] : model.tensor_symbol_map()) {
      if (name == t.wrt_input) eps = h;
    }
    // check_scope already verified both resolve; guard belt-and-braces so a
    // future check_scope/emit drift surfaces as a clear error, not cas::diff UB.
    if (!sigma.is_valid() || !eps.is_valid()) {
      throw std::runtime_error(
          "NumSimMaterialTarget: tangent '" + t.name +
          "' could not resolve its stress output / strain input handle.");
    }
    auto const tangent_expr = cas::diff(sigma, eps); // rank-4 ∂σ/∂ε

    CodeGenContext tc;
    CodeEmitPipeline tp(tc);
    register_scalars(tc);
    for (auto const &[name, expr] : model.tensor_symbol_map()) {
      tc.register_symbol_tensor(expr, name);
    }
    tc.reset();
    auto const trhs = tp.tensor().apply(tangent_expr);
    outputs.push_back({t.name, tc.render_statements("    "), trhs, true, tdim, 4});
  }

  bool has_tensor = !tensor_inputs.empty();
  for (auto const &o : outputs)
    if (o.is_tensor) has_tensor = true;

  // Comprehensive emitted-member uniqueness guard. Every generated `m_<base>`
  // member basename must be distinct, else the class gets duplicate members /
  // initializers (uncompilable). The recipe enforces symbol-name uniqueness, but
  // the emitter SYNTHESIZES extra names — `rate`, `rate_derivative`,
  // `integrator_source`, `out_<output>`, `<input>_source` — that a recipe symbol
  // can still collide with (a tensor input literally named "rate"; an input
  // "strain" alongside a parameter "strain_source"; an input named "integrator"
  // whose "_source" param clashes with the fixed integrator_source). Reject
  // loudly rather than emit broken C++. (The is_reserved_name checks above still
  // fire first for state/parameter cases, giving a more specific message.)
  {
    std::set<std::string> member_bases;
    auto claim = [&member_bases](std::string const &base) {
      if (!member_bases.insert(base).second) {
        throw std::runtime_error(
            "NumSimMaterialTarget: emitted member 'm_" + base +
            "' would be duplicated — a recipe symbol collides with a synthesized "
            "member name; rename the offending state/parameter/input/output.");
      }
    };
    claim(contract::rate_property);
    claim(contract::rate_derivative_property);
    claim(contract::integrator_source_param);
    claim(cur_name);
    for (auto const &p : params) claim(p.name);
    for (auto const &o : outputs) claim("out_" + o.name);
    for (auto const &ti : tensor_inputs) {
      claim(ti.name);
      claim(ti.name + "_source");
    }
  }

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
  if (has_tensor) h << "#include <tmech/tmech.h>\n";
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
  for (auto const &o : outputs) {
    std::string const ty =
        o.is_tensor ? tensor_cxx_type(o.dim, o.rank) : "value_type";
    h << "        m_out_" << o.name << "(base::template add_output<" << ty
      << ">(\n";
    h << "            \"" << o.name << "\", &" << cls << "::update_" << o.name
      << ")),\n";
  }
  for (auto const &p : params) {
    h << "        m_" << p.name
      << "(base::template get_parameter<value_type>(\"" << p.name << "\")),\n";
  }
  h << "        m_integrator_source(\n";
  h << "            base::template get_parameter<std::string>(\""
    << contract::integrator_source_param << "\")),\n";
  // Tensor-input source-name params (declared/initialized before the input
  // members that wire against them).
  for (auto const &ti : tensor_inputs) {
    h << "        m_" << ti.name << "_source(\n";
    h << "            base::template get_parameter<std::string>(\"" << ti.name
      << "_source\")),\n";
  }
  // The integrator state input (Local edge).
  h << "        m_" << cur_name << "(base::template add_input<value_type>(\n";
  h << "            m_integrator_source, \"" << contract::state_property
    << "\",\n";
  h << "            numsim::materials::EdgeKind::Local))";
  if (tensor_inputs.empty()) {
    h << " {}\n\n";
  } else {
    h << ",\n";
    // Tensor inputs (Global edge), read by property name = input name.
    for (std::size_t i = 0; i < tensor_inputs.size(); ++i) {
      auto const &ti = tensor_inputs[i];
      h << "        m_" << ti.name << "(base::template add_input<"
        << tensor_cxx_type(ti.dim, ti.rank) << ">(\n";
      h << "            m_" << ti.name << "_source, \"" << ti.name << "\",\n";
      h << "            numsim::materials::EdgeKind::Global))"
        << (i + 1 < tensor_inputs.size() ? ",\n" : " {}\n\n");
    }
  }

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
  for (auto const &ti : tensor_inputs) {
    h << "    para.template insert<std::string>(\"" << ti.name << "_source\")\n";
    h << "        .template add<numsim_core::is_required>();\n";
  }
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

  // One update callback per output (pulled independently by consumers). A tensor
  // output additionally binds the tensor-input locals it may reference.
  for (auto const &o : outputs) {
    h << "  // " << o.name << " = g(" << cur_name
      << (o.is_tensor ? ", inputs, params) [tensor].\n" : ", params).\n");
    h << "  void update_" << o.name << "() {\n";
    h << "    [[maybe_unused]] const auto " << cur_name << " = m_" << cur_name
      << ".get();\n";
    if (o.is_tensor) {
      for (auto const &ti : tensor_inputs) {
        h << "    [[maybe_unused]] const auto& " << ti.name << " = m_" << ti.name
          << ".get();\n";
      }
    }
    if (!o.decls.empty()) h << o.decls;
    h << "    m_out_" << o.name << " = " << o.rhs << ";\n";
    h << "  }\n\n";
  }

  // members
  h << "private:\n";
  h << "  value_type& m_rate;\n";
  h << "  value_type& m_rate_derivative;\n";
  for (auto const &o : outputs) {
    std::string const ty =
        o.is_tensor ? tensor_cxx_type(o.dim, o.rank) : "value_type";
    h << "  " << ty << "& m_out_" << o.name << ";\n";
  }
  for (auto const &p : params) {
    // [[maybe_unused]]: a parameter declared but not referenced by the rate is
    // still stored (it stays in the schema/JSON) — suppress clang's
    // -Wunused-private-field in a -Werror consumer build.
    h << "  [[maybe_unused]] const value_type& m_" << p.name << ";\n";
  }
  h << "  const std::string& m_integrator_source;\n";
  for (auto const &ti : tensor_inputs) {
    h << "  const std::string& m_" << ti.name << "_source;\n";
  }
  h << "  const numsim::materials::input_property<value_type,\n";
  h << "      numsim::materials::property_traits>& m_" << cur_name << ";\n";
  for (auto const &ti : tensor_inputs) {
    h << "  const numsim::materials::input_property<"
      << tensor_cxx_type(ti.dim, ti.rank) << ",\n";
    h << "      numsim::materials::property_traits>& m_" << ti.name << ";\n";
  }
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
  for (auto const &ti : tensor_inputs) {
    // PLACEHOLDER producer name — a material publishing a `<name>` property of
    // the matching tmech type must exist in the graph.
    j << ",\n      \"" << ti.name << "_source\": \"" << ti.name << "_producer\"";
  }
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
