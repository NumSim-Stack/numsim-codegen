#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_FOUR_TENSOR_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_FOUR_TENSOR_H

// MOOSE-API stub (#132): RankFourTensor with the storage contract the emitted
// boundary code assumes — contiguous row-major `Real` (3^4) exposed through
// `dataPointer()` (const and non-const).

#include "MooseTypes.h"

class RankFourTensor {
public:
  static constexpr unsigned int dimension = 3;

  [[nodiscard]] auto dataPointer() -> Real * { return m_vals; }
  [[nodiscard]] auto dataPointer() const -> Real const * { return m_vals; }

private:
  Real m_vals[dimension * dimension * dimension * dimension]{};
};

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_RANK_FOUR_TENSOR_H
