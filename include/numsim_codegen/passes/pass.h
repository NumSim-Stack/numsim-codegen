#ifndef NUMSIM_CODEGEN_PASS_H
#define NUMSIM_CODEGEN_PASS_H

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/passes/recipe_view.h>

#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace numsim::codegen {

struct SymbolDecl;
struct StateVariable;
class LinearAlgebraEmitter;

// Phase 3a-2 (issue #75): one in-function Newton solve recorded by
// `LocalNewtonLoweringPass` and rendered by `CodeEmitPass`. Each segment
// owns a scalar state variable: the generated function declares a local
// `<state_var_name>` initialised from `<state_var_name>_old`, iterates
// `residual` / `jacobian` (recomputed per iteration with loop-local CSE)
// until `|residual| < tol` or `max_iter` is hit, then writes the
// converged value to `<state_var_name>_out`. The residual + Jacobian are
// the same expressions `TimeIntegrationPass` / `LocalJacobianPass` would
// emit as outputs — here they become loop internals instead.
struct NewtonSegment {
  std::string state_var_name;
  cas::expression_holder<cas::scalar_expression> residual;
  cas::expression_holder<cas::scalar_expression> jacobian;
  double tol;
  int max_iter;
};

// Phase 3b-2b (issue #35): a COUPLED local Newton system of N>1 scalar unknowns.
// Produced by LocalNewtonLoweringPass when a group of evolution equations
// reference each other's current state variable (multi-surface plasticity etc.)
// — those equations cannot be solved independently. The generated function
// declares one local iterate per unknown, then runs a single Newton loop that
// each iteration assembles the residual vector R (size N) and the dense Jacobian
// J (N×N, `jacobian[i][j] = ∂R_i/∂x_j`), solves `J·Δx = −R`, and updates
// `x -= Δx` until `max|R_i| < tol` or `max_iter`. Uncoupled equations stay as
// 1×1 `NewtonSegment`s (the existing scalar-reciprocal path, byte-identical).
//
// SCALAR-UNKNOWNS ONLY (PR #83 round-2): each `unknowns[i]` is one `double` and
// the emit hard-codes `Eigen::Matrix<double,N,1>` / `x -= dx(i)`. Phase 3b-2a
// (Fischer-Burmeister) and #285 t2s residuals fit unchanged (an FB multiplier is
// just another scalar unknown + residual row; a t2s residual still produces a
// scalar). Phase 3b-2d (tensor unknowns — e.g. a back-stress tensor) does NOT
// fit and will need a sibling type / block generalization, not a mangled
// `unknowns: vector<string>`.
struct NewtonSystem {
  std::vector<std::string> unknowns;                            // size N
  std::vector<cas::expression_holder<cas::scalar_expression>> residuals; // N
  std::vector<std::vector<cas::expression_holder<cas::scalar_expression>>>
      jacobian; // N×N, row-major: jacobian[i][j] = ∂R_i/∂x_j
  double tol;
  int max_iter;
};

// Shared state for a single PassManager invocation. Passes read the
// recipe via `model` (a RecipeView — const-only today, will gain a
// mutable surface in Phase 2 without breaking pass signatures; see M4
// in issue #48 for the rationale), mutate the codegen context, and
// deposit their final products into the output slots below. Phase 1.2
// only needs one output slot (the rendered compute function); Phase
// 2/3 will add more (tangent source, state-variable wiring, etc.).
//
// `symbol_lookup` (P4 in issue #56) is populated by SymbolValidationPass
// during validation and consumed by downstream passes (e.g.
// TensorSpaceConsistencyPass) that need to resolve a symbol name to its
// SymbolDecl. Empty until SymbolValidationPass runs; populating it is
// part of that pass's contract.
//
// **Value is an INDEX into `model.symbols()`, not a pointer.** Pre-fixup
// the value type was `SymbolDecl const *`, which dangled the moment any
// Phase 2 mutating pass `push_back`-ed onto `m_symbols` and reallocated
// the vector. Indices survive vector growth at the cost of one
// indirection per lookup. The free function `find_tensor_symbol` below
// hides the indirection and provides the canonical "look up a symbol
// known to be a tensor" entry point.
struct PassContext {
  RecipeView model;
  CodeGenContext ctx;
  std::optional<std::string> compute_function_source;
  std::unordered_map<std::string, std::size_t> symbol_lookup;
  // Phase 3a-2 (issue #75): populated by LocalNewtonLoweringPass, consumed
  // by CodeEmitPass. Empty unless the recipe opted into local Newton
  // solving (`ConstitutiveModel::enable_local_newton`). Holds the 1×1
  // (uncoupled) solves; coupled groups go in `newton_systems` below.
  std::vector<NewtonSegment> newton_segments;
  // Phase 3b-2b (issue #35): coupled N>1 systems (mutually-referencing
  // evolution equations). Rendered as a single dense Newton loop.
  std::vector<NewtonSystem> newton_systems;
  // Phase 3b-2b: the dense-solve backend for coupled systems, injected by the
  // caller (per-target). Null → CodeEmitPass uses the Eigen default.
  LinearAlgebraEmitter const *linear_algebra = nullptr;
};

// Why a symbol lookup failed. The pre-modernization API returned a
// nullable `SymbolDecl const *`, which collapsed two genuinely distinct
// failure modes into one — callers that wanted to distinguish "no
// symbol with this name" from "symbol exists but is scalar, not tensor"
// had to inspect the model separately. Issue #62 item 1.
//
// **Reuse policy for Phase 2.2+ helpers** (per `docs/workflow.md`'s
// fallible-API convention): future symbol-lookup helpers
// (`find_state_variable_by_name` from issue #59, etc.) that share
// these two failure modes should reuse this enum, NOT introduce
// per-helper error types. If a future helper grows a genuinely third
// failure mode (e.g. `Ambiguous` for multi-match), introduce a sibling
// enum at that point rather than expanding `LookupError` globally.
enum class LookupError {
  NotFound, // name absent from symbol_lookup
  WrongKind // name present but the resolved SymbolDecl is not a tensor
};

// Convenience: resolve a name to its SymbolDecl, requiring it to be a
// tensor. Returns std::expected — on success, a non-null SymbolDecl *;
// on failure, a LookupError that distinguishes "not found" from "found
// but wrong kind". Centralises the "find + kind == Tensor" check that
// would otherwise duplicate across every validator (M3 in
// REVIEW-pr-57.md). Definition lives in recipe.h where
// `ConstitutiveModel::symbols()` and `SymbolDecl::Kind` are complete.
[[nodiscard]] inline auto find_tensor_symbol(PassContext const &pctx,
                                             std::string const &name) noexcept
    -> std::expected<SymbolDecl const *, LookupError>;

// Convenience: resolve a name to its StateVariable record. Returns
// nullable pointer rather than `std::expected` per `docs/workflow.md`
// §6.1 — only one failure mode ("no state variable with this name")
// that the caller could distinguish. Linear scan over
// `pctx.model.state_variables()`; state-variable counts are small
// (single-digit typical, tens worst-case for multi-surface plasticity)
// so the scan is fine. Could be promoted to a populated map in
// PassContext if that ever changes. Issue #59 / REVIEW-pr-58.md m3.
//
// Definition lives in recipe.h where `StateVariable` is complete.
[[nodiscard]] inline auto find_state_variable_by_name(
    PassContext const &pctx, std::string_view name) noexcept
    -> StateVariable const *;

// Abstract base for a single codegen pass.
//
// Passes advertise their pre/postconditions as string tags. PassManager
// verifies, before running each pass, that every declared precondition has
// been advertised as a postcondition by some earlier pass — catching
// ill-ordered pipelines at run time rather than producing silently-wrong
// output.
//
// Phase 1.2 ships two concrete passes (symbol validation + code emit).
// Future phases register additional passes (time integration, tangent,
// Kuhn-Tucker lowering, …) into the same framework.
class Pass {
public:
  virtual ~Pass() = default;
  [[nodiscard]] virtual auto name() const -> std::string_view = 0;

  // Tags this pass requires to be satisfied before `run()` is called.
  // Stable across the pass's lifetime — `preconditions()` is queried by
  // `PassManager::run()` *before* `run()`, so the value must not depend
  // on per-call state.
  [[nodiscard]] virtual auto preconditions() const
      -> std::vector<std::string_view> {
    return {};
  }

  // Tags this pass advertises as satisfied after `run()` returns
  // successfully. **Lifecycle (PR #66 round-2 review #11, round-3 #2):**
  // `PassManager::run()` queries `postconditions()` AFTER each call to
  // `run()`, never before. Implementations are therefore free to return
  // values that depend on per-call state set during `run()`.
  //
  // Most passes (`TensorSpaceConsistencyPass`, `CodeEmitPass`) return
  // literal initialiser lists from `postconditions()` and don't need to
  // think about lifecycle. `SymbolValidationPass` uses the pattern to
  // conditionally advertise `state_variables_non_empty` based on the
  // recipe — see its `run()` body for the canonical shape.
  //
  // **Guidance for passes that DO store per-call postcondition state:**
  // reset that state at the *start* of `run()`, not the end, so a
  // throwing `run()` leaves the pass in a coherent observable state and
  // a pre-`run()` query reports the safe-default shape. This is
  // convention, not a framework-enforced contract — PassManager has no
  // way to check it.
  [[nodiscard]] virtual auto postconditions() const
      -> std::vector<std::string_view> {
    return {};
  }
  virtual void run(PassContext &ctx) = 0;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_PASS_H
