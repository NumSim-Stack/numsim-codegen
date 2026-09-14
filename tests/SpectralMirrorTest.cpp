// Mirror-sync guard (#105 review finding #1): numsim_codegen/runtime/spectral.h
// hand-transcribes numsim-cas's confluent divided difference so generated code
// carries no numsim-cas dependency. That duplication is only safe if the two
// stay bit-identical. This test links BOTH and asserts they agree across kinds
// and point multisets — so a future drift in either side fails loudly here
// rather than silently producing wrong generated code.

#include <numsim_codegen/runtime/spectral.h>

#include <numsim_cas/tensor/data/tensor_data_isotropic.h>
#include <numsim_cas/tensor/isotropic_kind.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace numsim::codegen {
namespace {

struct KindPair {
  rt::scalar_fn rt_kind;
  cas::isotropic_kind cas_kind;
  const char *name;
};

const KindPair kinds[] = {
    {rt::scalar_fn::log, cas::isotropic_kind::log, "log"},
    {rt::scalar_fn::exp, cas::isotropic_kind::exp, "exp"},
    {rt::scalar_fn::sqrt, cas::isotropic_kind::sqrt, "sqrt"},
};

// Compare rt::divided_difference (codegen mirror) against cas confluent_dd
// (evaluator source of truth) for one point set, across all three kinds.
template <std::size_t N>
void expect_agree(std::array<double, N> pts, const char *label) {
  for (auto const &k : kinds) {
    const double got = rt::divided_difference(k.rt_kind, pts);
    std::vector<double> v(pts.begin(), pts.end());
    const double ref = cas::iso_detail::confluent_dd(k.cas_kind, v);
    const double scale = std::max(1.0, std::abs(ref));
    EXPECT_NEAR(got, ref, 1e-12 * scale)
        << "kind=" << k.name << " case=" << label << " N=" << N;
  }
}

TEST(SpectralMirror, DividedDifferenceMatchesCas) {
  // Distinct.
  expect_agree(std::array{2.0, 5.0}, "distinct_pair");
  expect_agree(std::array{1.1, 8.3}, "distinct_pair_2");
  expect_agree(std::array{1.0, 2.0, 4.0}, "distinct_triple");
  expect_agree(std::array{0.5, 1.5, 3.5, 7.0}, "distinct_quad");

  // Fully coincident (exercises the derivative-form branch on both sides).
  expect_agree(std::array{3.0, 3.0}, "coincident_pair");
  expect_agree(std::array{3.0, 3.0, 3.0}, "coincident_triple");
  expect_agree(std::array{2.0, 2.0, 2.0, 2.0}, "coincident_quad");

  // Mixed coincidence.
  expect_agree(std::array{2.0, 2.0, 5.0}, "two_then_one");
  expect_agree(std::array{2.0, 5.0, 5.0}, "one_then_two");
  expect_agree(std::array{1.0, 1.0, 4.0, 4.0}, "pair_pair");

  // Near-coincident, straddling the sqrt(eps) tolerance boundary — the case
  // where a mismatched guard/constant would surface first.
  expect_agree(std::array{2.0, 2.0 + 1e-9}, "within_tol");
  expect_agree(std::array{2.0, 2.0 + 1e-6}, "just_outside_tol");

  // Small and large magnitudes (relative-tolerance scaling). Kept below exp's
  // overflow threshold so all three kinds stay finite — the point is rt≡cas,
  // not domain limits.
  expect_agree(std::array{1e-6, 2e-6}, "tiny_distinct");
  expect_agree(std::array{30.0, 30.0 + 1e-3}, "large_near");
}

// The diagonal fast path confluent_derivative must equal the general
// divided_difference on an all-coincident set (that identity is what lets the
// emitter substitute it for the i==j tangent terms).
TEST(SpectralMirror, ConfluentDerivativeMatchesDividedDifference) {
  for (auto const &k : kinds) {
    for (double x : {0.7, 2.0, 5.5}) {
      EXPECT_NEAR(rt::confluent_derivative(k.rt_kind, x, 1),
                  rt::divided_difference(k.rt_kind, std::array{x, x}),
                  1e-12 * std::max(1.0, std::abs(x)))
          << "order 1, kind=" << k.name;
      EXPECT_NEAR(rt::confluent_derivative(k.rt_kind, x, 2),
                  rt::divided_difference(k.rt_kind, std::array{x, x, x}),
                  1e-12 * std::max(1.0, std::abs(x)))
          << "order 2, kind=" << k.name;
    }
  }
}

} // namespace
} // namespace numsim::codegen
