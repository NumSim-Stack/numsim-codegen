#ifndef NUMSIM_CODEGEN_SPECTRAL_DECOMPOSE_EMIT_H
#define NUMSIM_CODEGEN_SPECTRAL_DECOMPOSE_EMIT_H

#include <numsim_codegen/code_emit/codegen_context.h>

#include <cstddef>
#include <string>

namespace numsim::codegen {

// Emit (once per distinct tensor argument) the shared spectral decomposition
// that the eigenvalue / eigenprojection / eigenvector emitters read from, and
// return the temporary's name. `arg` is the already-emitted C++ name of the
// tensor argument; `dim` is its dimension. The argument is materialised into a
// concrete `tmech::tensor<double,dim,2>` so `spectral_decompose` deduces its
// dim from the type (the incoming `arg` may be an adaptor or an expression
// template). Keyed on `arg`, so every spectral quantity over the same tensor
// shares one `numsim::codegen::rt::spectral_decompose(...)` call.
inline auto emit_shared_decomposition(CodeGenContext &ctx,
                                      std::string const &arg,
                                      std::size_t dim) -> std::string {
  const std::string d = std::to_string(dim);
  const std::string rhs = "numsim::codegen::rt::spectral_decompose("
                          "tmech::tensor<double, " +
                          d + ", 2>(" + arg + "))";
  return ctx.emit_shared("spectral_decompose:" + arg, rhs);
}

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_SPECTRAL_DECOMPOSE_EMIT_H
