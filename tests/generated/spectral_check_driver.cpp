// Slice-5 end-to-end driver (#105): compile the generated spectral header and
// assert the generated log(A) value and d log(A)/dA tangent match the CAS
// evaluator across distinct, coalesced, and b=I states. The header is emitted
// by generate_spectral_check at build time.

#include "SpectralCheck.h"

#include <gtest/gtest.h>

TEST(SpectralE2E, GeneratedTangentMatchesEvaluator) {
  // 0 mismatches across all baked states (distinct_spd, two_coincident,
  // identity_bI). The b=I case is the payoff: fully coalesced eigenvalues,
  // where the divided-difference guard must give the finite analytic limit
  // rather than NaN.
  EXPECT_EQ(spectral_check::run_spectral_check(), 0);
}
