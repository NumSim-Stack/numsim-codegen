#ifndef NUMSIM_CODEGEN_LINEAR_ALGEBRA_EMITTER_H
#define NUMSIM_CODEGEN_LINEAR_ALGEBRA_EMITTER_H

// Phase 3b-2b (issue #35): the dense linear-algebra backend used by GENERATED
// code for a coupled local-Newton system's `J·Δx = R` solve — behind a small
// emit interface so the library (Eigen today; Armadillo / LAPACK / a hand-rolled
// fallback later) is a SINGLE swap point: `default_linear_algebra_emitter()`.
//
// Division of labour: the Newton-loop *frame* is library-agnostic and stays in
// `render_compute_function` — it declares one `double` iterate per unknown, the
// `for (… < max_iter; …)` loop, the loop-local CSE temps, and the writeback.
// This interface owns only the per-iteration *body*: assemble the residual
// vector R and the dense Jacobian J (row-major, `jacobian_rhs[i][j] = ∂R_i/∂x_j`),
// `break` when the residual ∞-norm < tol, solve `J·Δx = R`, and update each
// unknown `x_i -= Δx_i`.

#include <ostream>
#include <string>
#include <vector>

namespace numsim::codegen {

class LinearAlgebraEmitter {
public:
  virtual ~LinearAlgebraEmitter() = default;

  // Header(s) the generated code needs, raw/unwrapped (e.g. {"<Eigen/Dense>"}).
  // A backend wraps them per its own convention — MOOSE, for instance, wraps
  // third-party linalg headers in libmesh/ignore_warnings.h to survive -Werror.
  [[nodiscard]] virtual auto includes() const -> std::vector<std::string> = 0;

  // A token that appears in emitted code IFF this backend was used. A backend
  // gates its include on `emitted_source.contains(usage_marker())`, so the
  // include can never drift from what was actually emitted.
  [[nodiscard]] virtual auto usage_marker() const -> std::string = 0;

  // The per-system local-variable name suffixes this backend introduces (e.g.
  // {"r","J","dx"} → `<prefix>_r`, `<prefix>_J`, `<prefix>_dx`). The caller uses
  // these to pick a `<prefix>` that collides with no user identifier, so the
  // collision-avoidance stays correct when the backend changes its locals.
  [[nodiscard]] virtual auto local_suffixes() const
      -> std::vector<std::string> = 0;

  // Emit the per-iteration body (4-space indented, inside the Newton for-loop).
  // `prefix` is a collision-free local-name stem chosen by the caller. Emits a
  // `break;` on convergence and `<unknown> -= …;` updates for each unknown.
  virtual void emit_newton_step(
      std::ostream &os, std::string const &prefix,
      std::vector<std::string> const &unknowns,
      std::vector<std::string> const &residual_rhs,
      std::vector<std::vector<std::string>> const &jacobian_rhs,
      double tol) const = 0;
};

// Eigen implementation — fixed-size `Eigen::Matrix<double,N,·>` + partial-pivot
// LU (no heap). The whole Eigen dependency of generated code lives here.
class EigenLinearAlgebraEmitter final : public LinearAlgebraEmitter {
public:
  [[nodiscard]] auto includes() const -> std::vector<std::string> override {
    return {"<Eigen/Dense>"};
  }
  [[nodiscard]] auto usage_marker() const -> std::string override {
    return "Eigen::";
  }
  [[nodiscard]] auto local_suffixes() const
      -> std::vector<std::string> override {
    return {"r", "J", "dx"};
  }
  void emit_newton_step(
      std::ostream &os, std::string const &prefix,
      std::vector<std::string> const &unknowns,
      std::vector<std::string> const &residual_rhs,
      std::vector<std::vector<std::string>> const &jacobian_rhs,
      double tol) const override {
    auto const n = unknowns.size();
    auto L = [&](char const *s) { return prefix + "_" + s; };
    // Residual vector R (size N).
    os << "    Eigen::Matrix<double, " << n << ", 1> " << L("r") << ";\n    "
       << L("r") << " << ";
    for (std::size_t i = 0; i < n; ++i) {
      os << (i ? ", " : "") << residual_rhs[i];
    }
    os << ";\n";
    // Dense Jacobian J (N×N), `<<` fills row-major: J(i,j) = ∂R_i/∂x_j.
    os << "    Eigen::Matrix<double, " << n << ", " << n << "> " << L("J")
       << ";\n    " << L("J") << " << ";
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        os << ((i || j) ? ", " : "") << jacobian_rhs[i][j];
      }
    }
    os << ";\n";
    // Convergence on the residual ∞-norm.
    os << "    if (" << L("r") << ".cwiseAbs().maxCoeff() < " << tol
       << ") {\n      break;\n    }\n";
    // Solve J·Δx = R and update x -= Δx.
    os << "    Eigen::Matrix<double, " << n << ", 1> " << L("dx") << " = "
       << L("J") << ".partialPivLu().solve(" << L("r") << ");\n";
    for (std::size_t i = 0; i < n; ++i) {
      os << "    " << unknowns[i] << " -= " << L("dx") << "(" << i << ");\n";
    }
  }
};

// Armadillo implementation — fixed-size `arma::mat::fixed` / `arma::vec::fixed`
// + `arma::solve` (LAPACK). A worked second backend, proving the interface is
// library-agnostic: Armadillo's API differs from Eigen's in every emitted line
// (element assignment vs comma-init, `arma::norm(·,"inf")` vs
// `cwiseAbs().maxCoeff()`, `arma::solve` vs `partialPivLu().solve`), yet it
// slots behind the same four methods. NOTE: `arma::solve` links LAPACK/BLAS, a
// heavier downstream dependency than header-only Eigen.
class ArmadilloLinearAlgebraEmitter final : public LinearAlgebraEmitter {
public:
  [[nodiscard]] auto includes() const -> std::vector<std::string> override {
    return {"<armadillo>"};
  }
  [[nodiscard]] auto usage_marker() const -> std::string override {
    return "arma::";
  }
  [[nodiscard]] auto local_suffixes() const
      -> std::vector<std::string> override {
    return {"r", "J", "dx"};
  }
  void emit_newton_step(
      std::ostream &os, std::string const &prefix,
      std::vector<std::string> const &unknowns,
      std::vector<std::string> const &residual_rhs,
      std::vector<std::vector<std::string>> const &jacobian_rhs,
      double tol) const override {
    auto const n = unknowns.size();
    auto L = [&](char const *s) { return prefix + "_" + s; };
    // Residual vector R (size N) — element assignment (Armadillo has no
    // version-stable fixed-size comma-init).
    os << "    arma::vec::fixed<" << n << "> " << L("r") << ";\n";
    for (std::size_t i = 0; i < n; ++i) {
      os << "    " << L("r") << "(" << i << ") = " << residual_rhs[i] << ";\n";
    }
    // Dense Jacobian J (N×N): J(i,j) = ∂R_i/∂x_j.
    os << "    arma::mat::fixed<" << n << ", " << n << "> " << L("J") << ";\n";
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        os << "    " << L("J") << "(" << i << ", " << j
           << ") = " << jacobian_rhs[i][j] << ";\n";
      }
    }
    // Convergence on the residual ∞-norm.
    os << "    if (arma::norm(" << L("r") << ", \"inf\") < " << tol
       << ") {\n      break;\n    }\n";
    // Solve J·Δx = R and update x -= Δx.
    os << "    arma::vec::fixed<" << n << "> " << L("dx") << " = arma::solve("
       << L("J") << ", " << L("r") << ");\n";
    for (std::size_t i = 0; i < n; ++i) {
      os << "    " << unknowns[i] << " -= " << L("dx") << "(" << i << ");\n";
    }
  }
};

// THE linear-algebra backend selection point. Swap the returned implementation
// to change the library across emission, includes, and the usage gate at once.
// Compile-time selectable via NUMSIM_CODEGEN_USE_ARMADILLO (the codebase's
// macro-seam idiom); Eigen is the default (header-only, MOOSE-bundled).
//
// The macro is a BUILD-WIDE setting: both the inline emit path (render, in this
// header) and the compiled backends (standalone/moose .cpp, which gate the
// include on `usage_marker()`) must see the same selection, or the emitted body
// and its include would disagree. Define it for the whole project build.
[[nodiscard]] inline auto default_linear_algebra_emitter()
    -> LinearAlgebraEmitter const & {
#ifdef NUMSIM_CODEGEN_USE_ARMADILLO
  static ArmadilloLinearAlgebraEmitter const emitter;
#else
  static EigenLinearAlgebraEmitter const emitter;
#endif
  return emitter;
}

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_LINEAR_ALGEBRA_EMITTER_H
