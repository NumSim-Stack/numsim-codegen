#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_TWO_TENSOR_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_TWO_TENSOR_H

// MOOSE-API stub (#132): RankTwoTensor with the storage contract the emitted
// boundary code assumes — contiguous row-major `Real` (3x3) exposed through
// `dataPointer()` (const and non-const, matching the read/write adaptors).

#include "MooseTypes.h"

class RankTwoTensor {
public:
  static constexpr unsigned int dimension = 3;

  [[nodiscard]] auto dataPointer() -> Real * { return m_vals; }
  [[nodiscard]] auto dataPointer() const -> Real const * { return m_vals; }

  // Row-major element access — driver-side convenience, not emitted-code API.
  [[nodiscard]] auto operator()(unsigned int i, unsigned int j) -> Real & {
    return m_vals[i * dimension + j];
  }
  [[nodiscard]] auto operator()(unsigned int i,
                                unsigned int j) const -> Real const & {
    return m_vals[i * dimension + j];
  }

private:
  Real m_vals[dimension * dimension]{};
};

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_TWO_TENSOR_H
