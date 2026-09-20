#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MOOSE_TYPES_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MOOSE_TYPES_H

// Minimal MOOSE-API stub (#132). The real MOOSE framework is far too heavy to
// build in this repo's CI, but MooseMaterialTarget's emitted .h/.C must be
// COMPILED somewhere or regressions ship silently. These headers reproduce
// exactly the API surface the emitter uses (see src/targets/moose_material.cpp)
// — nothing more. They are a compile gate first; just functional enough for
// the driver to run one quadrature-point evaluation (issue #12 stretch).

#include <string>

// MOOSE's floating-point scalar. Must be `double`: the emitted boundary wraps
// tensor storage in `tmech::adaptor<double, ...>` over `dataPointer()`.
using Real = double;

// In real MOOSE this is a distinct string-derived type used for parameter
// classification; the emitter only ever passes it as a template argument to
// `addRequiredParam`, so a plain alias suffices.
using MaterialPropertyName = std::string;

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MOOSE_TYPES_H
