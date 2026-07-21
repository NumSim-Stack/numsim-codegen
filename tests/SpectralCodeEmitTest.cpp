// Slice-1 tests for the spectral emit handlers (#105): eigenvalue,
// eigenprojection, eigenvector — plus the shared-decomposition interning and
// the shipped runtime helper. The primal isotropic value (slice 3) and the
// divided-difference ternary (slice 4) are still stubbed and are asserted to
// throw here so the stub boundary is explicit.

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>
#include <numsim_codegen/runtime/spectral.h>

#include <numsim_cas/tensor/isotropic_kind.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/wrappers/tensor_eigenprojection.h>
#include <numsim_cas/tensor/wrappers/tensor_eigenvector.h>
#include <numsim_cas/tensor/wrappers/tensor_isotropic_function.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_divided_difference.h>
#include <numsim_cas/tensor_to_scalar/tensor_to_scalar_eigenvalue.h>

#include <tmech/tmech.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::codegen {
namespace {

auto throwing_t2s_apply() -> TensorCodeEmit::T2sApply {
  return [](auto const &) -> std::string {
    throw std::logic_error("throwing_t2s_apply invoked unexpectedly");
  };
}

// Count non-overlapping occurrences of `needle` in `hay`.
auto count(std::string const &hay, std::string const &needle) -> std::size_t {
  std::size_t n = 0, pos = 0;
  while ((pos = hay.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += needle.size();
  }
  return n;
}

struct Fixture {
  CodeGenContext ctx;
  ScalarCodeEmit scalar{ctx};
  TensorCodeEmit tensor{ctx, scalar, throwing_t2s_apply()};
  TensorToScalarCodeEmit t2s{ctx, scalar, tensor};
  cas::expression_holder<cas::tensor_expression> A =
      cas::make_expression<cas::tensor>("A", 3, 2);
  Fixture() { ctx.register_symbol_tensor(A, "A"); }
};

// ─── Eigenvalue: reads .eigenvalues[i] off a shared decomposition ──────
TEST(SpectralCodeEmit, EigenvalueReadsSharedDecomposition) {
  Fixture f;
  auto ev = cas::make_expression<cas::tensor_to_scalar_eigenvalue>(f.A, 1);
  f.t2s.apply(ev);
  auto r = f.ctx.render_statements();

  EXPECT_NE(r.find("numsim::codegen::rt::spectral_decompose("
                   "tmech::tensor<double, 3, 2>(A))"),
            std::string::npos)
      << r;
  EXPECT_NE(r.find(".eigenvalues[1]"), std::string::npos) << r;
}

// ─── Eigenprojection: E_i = n_i ⊗ n_i ──────────────────────────────────
TEST(SpectralCodeEmit, EigenprojectionEmitsOtimesOfEigenvector) {
  Fixture f;
  auto E = cas::make_expression<cas::tensor_eigenprojection>(f.A, 0);
  f.tensor.apply(E);
  auto r = f.ctx.render_statements();

  EXPECT_NE(r.find("tmech::otimes("), std::string::npos) << r;
  EXPECT_EQ(count(r, ".eigenvectors[0]"), 2u)
      << "E_i is n_i ⊗ n_i — the eigenvector appears twice\n"
      << r;
}

// ─── Eigenvector: n_i ───────────────────────────────────────────────────
TEST(SpectralCodeEmit, EigenvectorReadsEigenvectorsSlot) {
  Fixture f;
  auto n = cas::make_expression<cas::tensor_eigenvector>(f.A, 2);
  f.tensor.apply(n);
  auto r = f.ctx.render_statements();
  EXPECT_NE(r.find(".eigenvectors[2]"), std::string::npos) << r;
}

// ─── The interning payoff: eig_0(A) and E_0(A) share ONE decompose ──────
TEST(SpectralCodeEmit, SharedDecompositionEmittedOncePerArgument) {
  Fixture f;
  // Same tensor A, one t2s spectral quantity and two tensor spectral
  // quantities — all must reference a single spectral_decompose(A).
  f.t2s.apply(cas::make_expression<cas::tensor_to_scalar_eigenvalue>(f.A, 0));
  f.tensor.apply(cas::make_expression<cas::tensor_eigenprojection>(f.A, 1));
  f.tensor.apply(cas::make_expression<cas::tensor_eigenvector>(f.A, 2));
  auto r = f.ctx.render_statements();

  EXPECT_EQ(count(r, "spectral_decompose("), 1u)
      << "all spectral quantities over the same A must share one "
         "decomposition\n"
      << r;
}

// ─── Distinct arguments get distinct decompositions ────────────────────
TEST(SpectralCodeEmit, DistinctArgumentsGetDistinctDecompositions) {
  Fixture f;
  auto B = cas::make_expression<cas::tensor>("B", 3, 2);
  f.ctx.register_symbol_tensor(B, "B");
  // Hold both eigenvalue nodes alive: the pointer-keyed CSE would otherwise
  // alias if the first (temporary) node were freed and the second reallocated
  // at the same address. In real codegen the whole tree stays live during
  // emission, so this only matters for the test.
  auto eA = cas::make_expression<cas::tensor_to_scalar_eigenvalue>(f.A, 0);
  auto eB = cas::make_expression<cas::tensor_to_scalar_eigenvalue>(B, 0);
  f.t2s.apply(eA);
  f.t2s.apply(eB);
  auto r = f.ctx.render_statements();
  EXPECT_EQ(count(r, "spectral_decompose("), 2u) << r;
}

// ─── Stub boundary: divided difference still throws (slice 4) ───────────
TEST(SpectralCodeEmit, DeferredNodesThrowClearly) {
  Fixture f;
  auto dd = cas::make_expression<cas::tensor_to_scalar_divided_difference>(
      f.A, cas::isotropic_kind::log, std::vector<std::size_t>{0, 1});
  EXPECT_THROW(f.t2s.apply(dd), std::runtime_error);
}

// ─── Isotropic value f(A) = Σ f(λ_i) E_i (slice 3) ─────────────────────
TEST(SpectralCodeEmit, IsotropicValueEmitsSpectralSum) {
  Fixture f;
  auto L = cas::make_expression<cas::tensor_isotropic_function>(
      f.A, cas::isotropic_kind::log);
  f.tensor.apply(L);
  auto r = f.ctx.render_statements();

  // dim 3 ⇒ three spectral terms, one shared decomposition, std::log per λ_i.
  EXPECT_EQ(count(r, "spectral_decompose("), 1u) << r;
  EXPECT_EQ(count(r, "std::log("), 3u) << r;
  EXPECT_EQ(count(r, "tmech::otimes("), 3u) << r;
  EXPECT_EQ(count(r, ".eigenvalues["), 3u) << r;
  // E_i = n_i ⊗ n_i ⇒ eigenvectors referenced twice per term (6 total).
  EXPECT_EQ(count(r, ".eigenvectors["), 6u) << r;
}

TEST(SpectralCodeEmit, IsotropicValueUsesRequestedKind) {
  Fixture f;
  auto S = cas::make_expression<cas::tensor_isotropic_function>(
      f.A, cas::isotropic_kind::sqrt);
  f.tensor.apply(S);
  auto r = f.ctx.render_statements();
  EXPECT_NE(r.find("std::sqrt("), std::string::npos) << r;
  EXPECT_EQ(r.find("std::log("), std::string::npos) << r;
}

// The emitted formula's MATH, exercised directly through the runtime helper:
// L = Σ log(λ_i) E_i then exp of that reconstructs A (exp(log A) = A).
TEST(SpectralRuntime, IsotropicLogExpRoundTrips) {
  tmech::tensor<double, 3, 2> A;
  A = {4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0}; // SPD, distinct eigs
  auto isofn = [](auto const &X, auto f) {
    auto d = rt::spectral_decompose(X);
    tmech::tensor<double, 3, 2> out;
    for (std::size_t i = 0; i < 3; ++i)
      out += f(d.eigenvalues[i]) *
             tmech::otimes(d.eigenvectors[i], d.eigenvectors[i]);
    return out;
  };
  auto L = isofn(A, [](double x) { return std::log(x); });
  auto back = isofn(L, [](double x) { return std::exp(x); });
  EXPECT_TRUE(tmech::almost_equal(back, A, 1e-12));
}

// ─── Runtime helper: ascending eigenvalues, correct projectors ─────────
TEST(SpectralRuntime, SpectralDecomposeSortsAscending) {
  tmech::tensor<double, 3, 2> A;
  A = {9.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0}; // eigs {9,2,5}
  auto d = rt::spectral_decompose(A);
  EXPECT_NEAR(d.eigenvalues[0], 2.0, 1e-12);
  EXPECT_NEAR(d.eigenvalues[1], 5.0, 1e-12);
  EXPECT_NEAR(d.eigenvalues[2], 9.0, 1e-12);
  // Reconstruct A = Σ λ_i E_i from the returned eigenpairs.
  tmech::tensor<double, 3, 2> recon;
  for (std::size_t i = 0; i < 3; ++i)
    recon +=
        d.eigenvalues[i] * tmech::otimes(d.eigenvectors[i], d.eigenvectors[i]);
  EXPECT_TRUE(tmech::almost_equal(recon, A, 1e-12));
}

} // namespace
} // namespace numsim::codegen
