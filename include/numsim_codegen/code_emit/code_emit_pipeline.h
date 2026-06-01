#ifndef NUMSIM_CODEGEN_CODE_EMIT_PIPELINE_H
#define NUMSIM_CODEGEN_CODE_EMIT_PIPELINE_H

// Phase-1.2 follow-up (issue #56 / P2): bundles the three visitor
// emitters (Scalar / Tensor / TensorToScalar) and closes the cycle
// between TensorCodeEmit and TensorToScalarCodeEmit internally.
//
// Before this header: every consumer that wanted the full t2s-capable
// pipeline open-coded the unique_ptr indirection — see the pre-P2
// patterns in `CodeEmitPass::run` and the t2s test. The pattern is
// already two copies; Phase 2/3 (`TangentEmitPass`,
// `MoosePropertyEmitPass`, future cross-domain emit passes) would each
// need the same wiring. Each hand-coded copy is one typo away from
// invoking the callback before its storage is populated — UB.
//
// CodeEmitPipeline encapsulates the cycle-break once. Consumers write:
//
//   CodeEmitPipeline pipeline(ctx);
//   auto rhs = pipeline.tensor().apply(some_tensor_expr);
//   auto sc  = pipeline.scalar().apply(some_scalar_expr);
//
// The three emitters live for the pipeline's lifetime. The lambda that
// bridges TensorCodeEmit → TensorToScalarCodeEmit captures `this`; it
// only dereferences `m_t2s` after `make_unique` populates it. No UB
// even if the bridge gets invoked during arbitrary points of pipeline
// use, because by then construction is complete.

#include <numsim_codegen/code_emit/codegen_context.h>
#include <numsim_codegen/code_emit/scalar_code_emit.h>
#include <numsim_codegen/code_emit/tensor_code_emit.h>
#include <numsim_codegen/code_emit/tensor_to_scalar_code_emit.h>

#include <memory>

namespace numsim::codegen {

class CodeEmitPipeline {
public:
  explicit CodeEmitPipeline(CodeGenContext &ctx)
      : m_scalar(std::make_unique<ScalarCodeEmit>(ctx)) {
    // Order matters: tensor_emit needs a T2sApply callback at
    // construction (M3 invariant); t2s_emit needs a TensorCodeEmit&
    // (also at construction). We break the cycle by:
    //   1. Constructing tensor_emit with a lambda that defers through
    //      `this->m_t2s` (still nullptr at this point).
    //   2. Constructing t2s_emit with a stable reference to *m_tensor.
    //   3. Populating m_t2s — the lambda's deferred lookup now resolves.
    // The lambda is only ever invoked during emit_*().apply() calls
    // below, well after step (3).
    m_tensor = std::make_unique<TensorCodeEmit>(
        ctx, *m_scalar,
        [this](auto const &e) { return m_t2s->apply(e); });
    m_t2s = std::make_unique<TensorToScalarCodeEmit>(ctx, *m_scalar,
                                                     *m_tensor);
  }

  // Pipeline is non-copyable, non-movable — moving the lambdas'
  // captures invalidates them silently. Construct in-place.
  CodeEmitPipeline(CodeEmitPipeline const &) = delete;
  CodeEmitPipeline(CodeEmitPipeline &&) = delete;
  auto operator=(CodeEmitPipeline const &) -> CodeEmitPipeline & = delete;
  auto operator=(CodeEmitPipeline &&) -> CodeEmitPipeline & = delete;

  [[nodiscard]] auto scalar() noexcept -> ScalarCodeEmit & {
    return *m_scalar;
  }
  [[nodiscard]] auto tensor() noexcept -> TensorCodeEmit & {
    return *m_tensor;
  }
  [[nodiscard]] auto t2s() noexcept -> TensorToScalarCodeEmit & {
    return *m_t2s;
  }

private:
  // unique_ptr storage so the lambda's `this->m_t2s` lookup has stable
  // address even after construction. (References can't be late-bound.)
  std::unique_ptr<ScalarCodeEmit> m_scalar;
  std::unique_ptr<TensorCodeEmit> m_tensor;
  std::unique_ptr<TensorToScalarCodeEmit> m_t2s;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_CODE_EMIT_PIPELINE_H
