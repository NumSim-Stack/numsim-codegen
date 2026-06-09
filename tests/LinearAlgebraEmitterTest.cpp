// Phase 3b-2b (issue #35): the LinearAlgebraEmitter abstraction — the single
// swap point for the dense-solve backend used by generated coupled-Newton code.
// These pin the Eigen implementation's contract and that it's the default, so a
// future backend swap (Armadillo / LAPACK / …) is a deliberate, visible change.

#include <numsim_codegen/code_emit/linear_algebra_emitter.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace numsim::codegen {

namespace {

TEST(LinearAlgebraEmitter, DefaultBackendIsEigen) {
  auto const &la = default_linear_algebra_emitter();
  EXPECT_EQ(la.usage_marker(), "Eigen::");
  ASSERT_EQ(la.includes().size(), 1u);
  EXPECT_EQ(la.includes()[0], "<Eigen/Dense>");
}

TEST(LinearAlgebraEmitter, EigenAdvertisesItsLocalSuffixes) {
  // The caller uses these to keep the solve-local names collision-free; they
  // must cover every local the emitter introduces.
  EigenLinearAlgebraEmitter const e;
  auto const s = e.local_suffixes();
  for (char const *expected : {"r", "J", "dx"}) {
    EXPECT_NE(std::find(s.begin(), s.end(), expected), s.end())
        << "missing suffix '" << expected << "'";
  }
}

TEST(LinearAlgebraEmitter, EigenEmitsDenseSolveBody) {
  EigenLinearAlgebraEmitter const e;
  std::ostringstream os;
  e.emit_newton_step(os, "sys", {"a", "b"}, {"Ra", "Rb"},
                     {{"J00", "J01"}, {"J10", "J11"}}, 1e-10);
  auto const src = os.str();
  // Fixed-size matrices, row-major fill, ∞-norm convergence, partial-pivot LU.
  EXPECT_NE(src.find("Eigen::Matrix<double, 2, 1> sys_r"), std::string::npos)
      << src;
  EXPECT_NE(src.find("sys_r << Ra, Rb"), std::string::npos) << src;
  EXPECT_NE(src.find("Eigen::Matrix<double, 2, 2> sys_J"), std::string::npos)
      << src;
  EXPECT_NE(src.find("sys_J << J00, J01, J10, J11"), std::string::npos) << src;
  EXPECT_NE(src.find("sys_r.cwiseAbs().maxCoeff() < 1e-10"), std::string::npos)
      << src;
  EXPECT_NE(src.find("sys_J.partialPivLu().solve(sys_r)"), std::string::npos)
      << src;
  // Each unknown updated from the solution vector.
  EXPECT_NE(src.find("a -= sys_dx(0)"), std::string::npos) << src;
  EXPECT_NE(src.find("b -= sys_dx(1)"), std::string::npos) << src;
}

// The Armadillo backend exercises a genuinely different API behind the same
// interface — proof the seam is library-agnostic.
TEST(LinearAlgebraEmitter, ArmadilloAdvertisesItsContract) {
  ArmadilloLinearAlgebraEmitter const a;
  EXPECT_EQ(a.usage_marker(), "arma::");
  ASSERT_EQ(a.includes().size(), 1u);
  EXPECT_EQ(a.includes()[0], "<armadillo>");
  auto const s = a.local_suffixes();
  for (char const *expected : {"r", "J", "dx"}) {
    EXPECT_NE(std::find(s.begin(), s.end(), expected), s.end());
  }
}

TEST(LinearAlgebraEmitter, ArmadilloEmitsDenseSolveBody) {
  ArmadilloLinearAlgebraEmitter const a;
  std::ostringstream os;
  a.emit_newton_step(os, "sys", {"a", "b"}, {"Ra", "Rb"},
                     {{"J00", "J01"}, {"J10", "J11"}}, 1e-10);
  auto const src = os.str();
  // Fixed-size types, element assignment, ∞-norm via arma::norm, arma::solve.
  EXPECT_NE(src.find("arma::vec::fixed<2> sys_r"), std::string::npos) << src;
  EXPECT_NE(src.find("sys_r(0) = Ra;"), std::string::npos) << src;
  EXPECT_NE(src.find("arma::mat::fixed<2, 2> sys_J"), std::string::npos) << src;
  EXPECT_NE(src.find("sys_J(0, 1) = J01;"), std::string::npos) << src; // ∂R_0/∂x_1
  EXPECT_NE(src.find("sys_J(1, 0) = J10;"), std::string::npos) << src; // ∂R_1/∂x_0
  EXPECT_NE(src.find("arma::norm(sys_r, \"inf\") < 1e-10"), std::string::npos)
      << src;
  EXPECT_NE(src.find("arma::solve(sys_J, sys_r)"), std::string::npos) << src;
  EXPECT_NE(src.find("a -= sys_dx(0)"), std::string::npos) << src;
  EXPECT_NE(src.find("b -= sys_dx(1)"), std::string::npos) << src;
  // No Eigen leakage.
  EXPECT_EQ(src.find("Eigen::"), std::string::npos) << src;
}

} // namespace

} // namespace numsim::codegen
