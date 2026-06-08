#ifndef NUMSIM_CODEGEN_PASSES_INTERNAL_PASS_BODIES_H
#define NUMSIM_CODEGEN_PASSES_INTERNAL_PASS_BODIES_H

// Internal implementation header — include <numsim_codegen/recipe.h>, not
// this directly. Holds the inline pass run() definitions, included from
// the bottom of recipe.h AFTER ConstitutiveModel + the RecipeView
// delegates + render_compute_function are complete (the bodies call them).
#ifndef NUMSIM_CODEGEN_RECIPE_H
#error "Do not include passes/internal/pass_bodies.h directly; include <numsim_codegen/recipe.h>."
#endif

#include <numsim_codegen/passes/internal/algorithmic_tangent.h>
#include <numsim_codegen/passes/internal/backward_euler.h>

namespace numsim::codegen {

// ─── Pass run() definitions ──────────────────────────────────────────
//
// These live after the ConstitutiveModel class so that the pass bodies can
// see its full definition (they call methods like `outputs()`,
// `scalar_symbol_map()`). They're inline so the existing header-mostly
// layout stays intact; the static library only houses the per-target
// wrappers in src/targets/.

inline void SymbolValidationPass::run(PassContext &pctx) {
  auto const &model = pctx.model;

  // Reset the conditional-postcondition flag on entry — PR #66 round-2
  // review #2. Two failure modes leak prior state without this:
  //   (a) `verify_state_variable_symbol_alignment` below throws → the
  //       end-of-run assignment never executes → the flag retains its
  //       previous value;
  //   (b) the same pass instance is reused across two recipes with
  //       different state-variable counts → state advertised between
  //       calls reflects the prior recipe.
  // (A pre-`run()` query on a pristine pass instance is already covered
  // by the default-initialiser at `symbol_validation_pass.h:130` — the
  // reset-at-entry here does not need to address that case.)
  // Resetting up-front guarantees `m_state_variables_non_empty` always
  // reflects the *current* run; the real value is computed after
  // validation succeeds at the bottom of this function.
  m_state_variables_non_empty = false;

  // P4 (post-fixup C1): build the name → index lookup once. Indices
  // survive any future `m_symbols.push_back()` reallocation, whereas
  // the original SymbolDecl pointers would silently dangle the first
  // time a Phase 2 mutating pass synthesised a new symbol. Consumers
  // resolve through `find_tensor_symbol` (one indirection per lookup;
  // a fair trade for eliminating the dangling class).
  auto const &symbols = model.symbols();
  for (std::size_t i = 0; i < symbols.size(); ++i) {
    pctx.symbol_lookup.emplace(symbols[i].name, i);
  }

  // (1) C++ identifier validity on declared symbol names. We check
  // *declared* names (not user-typed reference strings) — the cas
  // expression layer accepts anything for tensor("foo-bar", ...), so the
  // bad-character codepath only manifests once codegen tries to splice
  // the name into generated source.
  for (auto const &[name, _] : model.scalar_symbol_map()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(name)) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': scalar symbol '{}' is not a usable C++ "
          "identifier (bad syntax or reserved keyword). Generated code "
          "would not compile. Use a [A-Za-z_][A-Za-z0-9_]* name that is "
          "not a C++ keyword.",
          model.name(), name));
    }
  }
  for (auto const &[name, _] : model.tensor_symbol_map()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(name)) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': tensor symbol '{}' is not a usable C++ "
          "identifier (bad syntax or reserved keyword). Generated code "
          "would not compile. Use a [A-Za-z_][A-Za-z0-9_]* name that is "
          "not a C++ keyword.",
          model.name(), name));
    }
  }
  for (auto const &out : model.outputs()) {
    if (!SymbolValidationPass::is_valid_cxx_identifier(out.name)) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': output '{}' is not a usable C++ "
          "identifier (bad syntax or reserved keyword). Generated code "
          "would not compile. Use a [A-Za-z_][A-Za-z0-9_]* name that is "
          "not a C++ keyword.",
          model.name(), out.name));
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
    std::string msg = std::format(
        "ConstitutiveModel '{}': outputs reference undeclared symbol(s):",
        model.name());
    for (auto const &m : missing) {
      msg += std::format("\n  - {}", m);
    }
    msg += "\nCall add_scalar_input / add_tensor_input / add_parameter "
           "before referencing a symbol in an output expression.";
    throw std::runtime_error(msg);
  }

  // (3) State-variable alignment invariant (issue #59 / PR #66 review).
  // Throws if any StateVariable's `current_symbol_idx` / `old_symbol_idx`
  // points at a mismatched SymbolDecl. Today's public API can't produce
  // a violation, but running the check here means Phase 2.2+ mutating
  // passes that touch state variables can re-run SymbolValidationPass
  // to re-verify between stages — the invariant isn't bypassable by
  // running passes outside `ConstitutiveModel::validate()`.
  verify_state_variable_symbol_alignment(model);

  // Drive the conditional `state_variables_non_empty` postcondition.
  m_state_variables_non_empty = !model.state_variables().empty();
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
  // P4 (post-fixup M3): one-shot lookup via the canonical helper. The
  // pre-P4 body did a linear scan over model.symbols() inside this
  // loop, giving O(N·M); the post-P4-pre-fixup body open-coded
  // `find + kind == Tensor`, which would duplicate across every
  // Phase 2 validator. `find_tensor_symbol` is the single canonical
  // join.
  for (auto const &[name, expr] : model.tensor_symbol_map()) {
    auto const result = find_tensor_symbol(pctx, name);
    if (!result) {
      // NotFound: SymbolValidationPass should have caught this earlier.
      // WrongKind: shouldn't happen for a name we just read from
      // tensor_symbol_map(). Either way, no work to do.
      continue;
    }
    Role const *role_ptr = &(*result)->role;

    auto const &t = expr.get();
    auto const &space_opt = t.space();
    if (space_opt.has_value() && role_ptr->is_symmetric) {
      if (std::holds_alternative<cas::Skew>(space_opt->perm)) {
        throw std::runtime_error(std::format(
            "ConstitutiveModel '{}': tensor symbol '{}' is declared with "
            "role '{}' (is_symmetric=true) but its tensor_space annotation "
            "has Skew perm — a skew-symmetric tensor cannot satisfy a "
            "symmetric role. Fix one of: (a) remove the cas-side "
            "`assume_skew()` annotation on this symbol; (b) re-declare "
            "with a Role whose `is_symmetric=false` (construct a custom "
            "Role aggregate, or pick from the `roles::` catalogue in "
            "recipe.h).",
            model.name(), name, role_ptr->name));
      }
    }

    if (role_ptr->expected_rank.has_value() &&
        t.rank() != *role_ptr->expected_rank) {
      throw std::runtime_error(std::format(
          "ConstitutiveModel '{}': tensor symbol '{}' has rank {} but "
          "role '{}' expects rank {}. Fix one of: (a) adjust the `rank` "
          "argument in your `add_tensor_input(name, dim, rank, ...)` "
          "call to match the role; (b) re-declare with a different Role "
          "(see the `roles::` catalogue in recipe.h for their "
          "`expected_rank` values).",
          model.name(), name, t.rank(), role_ptr->name,
          *role_ptr->expected_rank));
    }
  }
}

// Phase 2.2 (issue #68): backward-Euler lowering. For each evolution
// equation `(state_var, rate)` declared on the recipe, synthesise a
// new scalar output `<state_var>_residual` whose expression is the
// discrete residual `(state_var − state_var_old)/dt − rate`.
inline void TimeIntegrationPass::run(PassContext &pctx) {
  auto &model_mut = pctx.model.require_mutable_model(
      "TimeIntegrationPass");
  auto const &equations = model_mut.evolution_equations();
  auto const &svs = model_mut.state_variables();

  // Reserve ahead of the add_output batch (cross-cutting review MAJOR 4):
  // one synthesised residual per evolution equation. Prevents incremental
  // m_outputs reallocation mid-loop — both a perf hint at Phase-4 scale
  // (~50 eqs) and a hardening guarantee that any output span/reference a
  // future pass holds across this loop stays valid.
  model_mut.reserve_outputs(equations.size());

  for (auto const &eq : equations) {
    auto const &sv = svs[eq.state_variable_idx];
    auto built = detail::build_backward_euler_residual(
        model_mut, eq, "TimeIntegrationPass");
    model_mut.add_output(sv.name + "_residual",
                         std::move(built.residual));
  }
}

// Phase 3a-1 (issue #70): symbolic-Jacobian emission. For each
// evolution equation, computes `J = ∂R/∂α` via `cas::diff()` and adds
// it as a `<sv>_jacobian` output. Runs AFTER TimeIntegrationPass (the
// residual outputs must already be synthesised) and BEFORE CodeEmitPass.
//
// Re-derives the residual through the shared
// `build_backward_euler_residual` helper — the single source of truth
// for the discrete residual shape. If TimeIntegrationPass ever changes
// its residual form (sign convention, BDF2 numerator, scaling factor),
// LJP automatically picks up the same change here.
inline void LocalJacobianPass::run(PassContext &pctx) {
  auto &model_mut = pctx.model.require_mutable_model("LocalJacobianPass");
  auto const &equations = model_mut.evolution_equations();
  auto const &svs = model_mut.state_variables();

  // One synthesised Jacobian output per evolution equation (MAJOR 4).
  model_mut.reserve_outputs(equations.size());

  for (auto const &eq : equations) {
    auto const &sv = svs[eq.state_variable_idx];
    auto built = detail::build_backward_euler_residual(
        model_mut, eq, "LocalJacobianPass");

    // Issue #73 (option 2): the Jacobian comes from the sibling helper,
    // which diffs the residual `built` carries — same DAG, same resolved
    // `cur_expr`, no second symbol-map walk.
    //
    // TODO(numsim-codegen#72): this residual is pointer-distinct from the
    // one TimeIntegrationPass built, so the CodeGenContext's pointer-keyed
    // CSE re-emits temporaries the residual pass already emitted. Fine at
    // single-SV scale; Phase 4's multi-SV / rank-4 path needs value-based
    // CSE or residual+Jacobian co-emission. See #72.
    auto jacobian = detail::build_backward_euler_jacobian(built);
    model_mut.add_output(sv.name + "_jacobian", std::move(jacobian));
  }
}

// Phase 3b-1 (issue #35): synthesise the consistent tangent dσ/dε for each
// requested tangent and add it as a rank-4 tensor output. Runs AFTER any
// state-variable lowering (the converged-state seam `pctx.newton_segments`
// is populated) and BEFORE CodeEmitPass.
//
//     dσ/dε = ∂σ/∂ε  +  Σ_i (∂σ/∂x_i)·(dx_i/dε) ,   dx_i/dε = −(1/J_i)·∂R_i/∂ε
//
// The explicit term ∂σ/∂ε is emitted via cas::diff(tensor, tensor). The
// implicit correction is provably zero with the current scalar-residual Newton
// machinery (∂R_i/∂ε ≡ 0 because a scalar_expression residual cannot depend on
// a tensor), so the exact tangent is ∂σ/∂ε. The correction assembly is staged
// behind NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR for when strain-coupled
// (t2s) residuals + numsim-cas#275 land — see internal/algorithmic_tangent.h.
inline void AlgorithmicTangentPass::run(PassContext &pctx) {
  auto &model_mut = pctx.model.require_mutable_model("AlgorithmicTangentPass");
  auto const &specs = model_mut.tangents();
  if (specs.empty()) {
    return;
  }

  // Resolve a tensor input variable (the strain ε) by name.
  auto find_tensor_input =
      [&](std::string const &nm)
      -> cas::expression_holder<cas::tensor_expression> const * {
    for (auto const &[name, expr] : model_mut.tensor_symbol_map()) {
      if (name == nm) {
        return &expr;
      }
    }
    return nullptr;
  };

  // Snapshot the specs before mutating m_outputs (add_output below does not
  // touch m_tangents, so the span stays valid — but resolve eagerly anyway).
  std::vector<TangentSpec> pending(specs.begin(), specs.end());
  model_mut.reserve_outputs(pending.size());

  for (auto const &spec : pending) {
    // Resolve the stress output σ — must be an existing tensor output.
    cas::expression_holder<cas::tensor_expression> const *sigma = nullptr;
    for (auto const &o : model_mut.outputs()) {
      if (o.name == spec.of_output) {
        if (o.kind != OutputDecl::Kind::Tensor) {
          throw std::runtime_error(std::format(
              "AlgorithmicTangentPass: tangent '{}' references output '{}', "
              "which is not a tensor output — dσ/dε needs a tensor σ.",
              spec.name, spec.of_output));
        }
        sigma = &std::get<cas::expression_holder<cas::tensor_expression>>(
            o.expr);
        break;
      }
    }
    if (sigma == nullptr) {
      throw std::runtime_error(std::format(
          "AlgorithmicTangentPass: tangent '{}' references output '{}', which "
          "was not declared via add_output.",
          spec.name, spec.of_output));
    }

    // Resolve the strain input ε — must be a declared tensor input/symbol.
    auto const *eps = find_tensor_input(spec.wrt_input);
    if (eps == nullptr) {
      throw std::runtime_error(std::format(
          "AlgorithmicTangentPass: tangent '{}' differentiates w.r.t. '{}', "
          "which is not a declared tensor input.",
          spec.name, spec.wrt_input));
    }

    // Explicit term: ∂σ/∂ε (rank-4). Real cas::diff — no upstream dependency.
    auto tangent = cas::diff(*sigma, *eps);

#ifdef NUMSIM_CODEGEN_HAVE_DIFF_TENSOR_WRT_SCALAR
    // Implicit correction Σ_i (−1/J_i)·(∂σ/∂x_i)·(∂R_i/∂ε). Only meaningful
    // once NewtonSegment carries a strain-coupled (t2s) residual so that
    // ∂R_i/∂ε ≠ 0; with the scalar-residual machinery this loop is a no-op.
    // ∂σ/∂x_i is taken through the diff(tensor, scalar) seam (numsim-cas#275).
    for (auto const &seg : pctx.newton_segments) {
      auto const *x = /* scalar var named seg.state_var_name */ nullptr;
      (void)seg;
      (void)x;
      // TODO(3b-1 strain-coupled): build dR_i/dε (diff of a t2s residual
      // w.r.t. ε), dσ/dx_i = detail::diff_tensor_wrt_scalar(*sigma, x), and
      // accumulate (-1/J_i) * outer(dσ/dx_i, dR_i/dε) into `tangent`.
    }
#endif

    model_mut.add_output(spec.name, std::move(tangent),
                         roles::ConsistentTangent);
  }
}

// Phase 3a-2 (issue #75): record an in-function Newton solve per scalar
// evolution equation. Does NOT mutate the recipe and does NOT emit R/J as
// outputs — it builds the residual + Jacobian via the shared helpers (the
// same single source of truth the output passes use) and hands them to
// CodeEmitPass as `NewtonSegment`s. Registered instead of TIP+LJP when the
// recipe opted into `enable_local_newton`.
inline void LocalNewtonLoweringPass::run(PassContext &pctx) {
  auto const &model = pctx.model.raw_model();
  auto const opts = model.newton_options();
  auto const &equations = model.evolution_equations();
  auto const &svs = model.state_variables();

  pctx.newton_segments.reserve(equations.size());
  for (auto const &eq : equations) {
    auto const &sv = svs[eq.state_variable_idx];
    auto built = detail::build_backward_euler_residual(
        model, eq, "LocalNewtonLoweringPass");
    auto jacobian = detail::build_backward_euler_jacobian(built);
    pctx.newton_segments.push_back(NewtonSegment{
        sv.name, std::move(built.residual), std::move(jacobian),
        opts.tol, opts.max_iter});
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

  // Phase 3a-2: render each Newton segment's residual + Jacobian with
  // LOOP-LOCAL CSE. `ctx.reset()` clears statements + the CSE table but
  // keeps the symbol registrations and the temp counter, so each block
  // gets fresh temps (recomputed every iteration) that can't collide
  // with the others or with the function-level output temps below.
  std::vector<RenderedNewtonSegment> newton;
  newton.reserve(pctx.newton_segments.size());
  for (auto const &seg : pctx.newton_segments) {
    ctx.reset();
    auto r_rhs = pipeline.scalar().apply(seg.residual);
    auto j_rhs = pipeline.scalar().apply(seg.jacobian);
    newton.push_back(RenderedNewtonSegment{
        seg.state_var_name, ctx.render_statements("    "),
        std::move(r_rhs), std::move(j_rhs), seg.tol, seg.max_iter});
  }
  // Clean slate so the output temps below don't include any segment's
  // loop-local statements.
  if (!pctx.newton_segments.empty()) {
    ctx.reset();
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

  pctx.compute_function_source =
      render_compute_function(model, ctx, output_rhs, newton);
}

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_PASSES_INTERNAL_PASS_BODIES_H
