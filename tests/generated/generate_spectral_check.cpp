// Slice-5 end-to-end generator (#105): lower the isotropic value f(A) and its
// spectral tangent d f(A)/dA to tmech-only C++ via the codegen emitters, and
// bake CAS-evaluator reference values. Emits a header exposing
// run_spectral_check(), which the driver asserts returns 0.
//
// This is the compile-and-run gate the string-asserting unit tests can't be: it
// proves the emitted value AND tangent are syntactically valid, tmech-only C++
// that matches the evaluator numerically — for log/exp/sqrt, in dim 2 and 3,
// and including at eigenvalue coalescence (b=I), where the divided-difference
// guard must take the analytic-limit branch instead of producing NaN.

#include <numsim_codegen/code_emit/code_emit_pipeline.h>
#include <numsim_codegen/code_emit/codegen_context.h>

#include <numsim_cas/core/diff.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_diff.h>
#include <numsim_cas/tensor/tensor_isotropic_functions.h>
#include <numsim_cas/tensor/visitors/tensor_evaluator.h>
#include <numsim_cas/tensor_to_scalar/visitors/tensor_to_scalar_evaluator.h>

#include <tmech/tmech.h>

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace numsim::cas;

template <std::size_t Dim>
static std::shared_ptr<tensor_data<double, Dim, 2>>
data_from(tmech::tensor<double, Dim, 2> const &t) {
  auto p = std::make_shared<tensor_data<double, Dim, 2>>();
  for (std::size_t i = 0; i < Dim * Dim; ++i)
    p->raw_data()[i] = t.raw_data()[i];
  return p;
}

static expression_holder<tensor_expression>
make_iso(expression_holder<tensor_expression> const &A, isotropic_kind kind) {
  switch (kind) {
  case isotropic_kind::log:
    return log(A);
  case isotropic_kind::exp:
    return exp(A);
  case isotropic_kind::sqrt:
    return sqrt(A);
  }
  return A;
}

// Emit one `fn(A, L, dL)` function for f=kind in dimension Dim, plus a check
// block per sample A comparing the generated value and tangent to the CAS
// evaluator. Function definitions go to `defs`, check blocks to `checks`.
template <std::size_t Dim>
static void emit_kind(
    std::ostream &defs, std::ostream &checks, tensor_evaluator<double> &ev,
    std::string const &fn, isotropic_kind kind,
    std::vector<std::pair<std::string, tmech::tensor<double, Dim, 2>>> const
        &cases) {
  constexpr std::size_t N2 = Dim * Dim;
  constexpr std::size_t N4 = Dim * Dim * Dim * Dim;
  const std::string TT =
      "tmech::tensor<double, " + std::to_string(Dim) + ", 2>";
  const std::string TT4 =
      "tmech::tensor<double, " + std::to_string(Dim) + ", 4>";

  auto A = make_expression<tensor>("A", Dim, 2);
  auto L = make_iso(A, kind);
  auto dL = diff(L, A);

  numsim::codegen::CodeGenContext ctx;
  numsim::codegen::CodeEmitPipeline pipe(ctx);
  ctx.register_symbol_tensor(A, "A");
  const std::string lname = pipe.tensor().apply(L);
  const std::string dname = pipe.tensor().apply(dL);

  defs << "inline void " << fn << "(" << TT << " const &A, " << TT
       << " &L_out, " << TT4 << " &dL_out) {\n"
       << ctx.render_statements("  ") << "  L_out = " << lname
       << ";\n  dL_out = " << dname << ";\n}\n\n";

  for (auto const &[nm, Aval] : cases) {
    ev.set(A, data_from<Dim>(Aval));
    // Hold the result holders alive while reading — a reference into a
    // temporary *ev.apply(...) would dangle before the next apply().
    auto Ld = ev.apply(L);
    auto dLd = ev.apply(dL);
    auto const &Lref =
        static_cast<tensor_data<double, Dim, 2> const &>(*Ld).data();
    auto const &dRef =
        static_cast<tensor_data<double, Dim, 4> const &>(*dLd).data();

    checks << "  { // " << fn << " / " << nm << "\n    " << TT << " A; A = {";
    for (std::size_t i = 0; i < N2; ++i)
      checks << (i ? ", " : "") << Aval.raw_data()[i];
    checks << "};\n    " << TT << " L; " << TT4 << " dL; " << fn
           << "(A, L, dL);\n    const double Lref[" << N2 << "] = {";
    for (std::size_t i = 0; i < N2; ++i)
      checks << (i ? ", " : "") << Lref.raw_data()[i];
    checks << "};\n    const double dRef[" << N4 << "] = {";
    for (std::size_t i = 0; i < N4; ++i)
      checks << (i ? ", " : "") << dRef.raw_data()[i];
    checks << "};\n    for (int i = 0; i < " << N2
           << "; ++i)\n      if (std::abs(L.raw_data()[i] - Lref[i]) > 1e-10) "
              "++fails;\n    for (int i = 0; i < "
           << N4
           << "; ++i)\n      if (std::abs(dL.raw_data()[i] - dRef[i]) > 1e-8) "
              "++fails;\n  }\n";
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: generate_spectral_check <out.h>\n";
    return 2;
  }

  // dim-3 samples: distinct, a coalesced pair {2,2,5}, and b=I.
  const std::vector<std::pair<std::string, tmech::tensor<double, 3, 2>>> d3 = {
      {"distinct_spd", {4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0}},
      {"two_coincident", {3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0}},
      {"identity_bI", {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}},
  };
  // dim-2 samples: distinct and identity — covers the dim-2 emit path.
  const std::vector<std::pair<std::string, tmech::tensor<double, 2, 2>>> d2 = {
      {"distinct_2d", {4.0, 1.0, 1.0, 3.0}},
      {"identity_2d", {1.0, 0.0, 0.0, 1.0}},
  };

  std::ostringstream defs, checks;
  defs << std::setprecision(17);
  checks << std::setprecision(17);

  tensor_evaluator<double> ev;
  emit_kind<3>(defs, checks, ev, "hencky_log_3", isotropic_kind::log, d3);
  emit_kind<3>(defs, checks, ev, "hencky_exp_3", isotropic_kind::exp, d3);
  emit_kind<3>(defs, checks, ev, "hencky_sqrt_3", isotropic_kind::sqrt, d3);
  emit_kind<2>(defs, checks, ev, "hencky_log_2", isotropic_kind::log, d2);
  emit_kind<2>(defs, checks, ev, "hencky_sqrt_2", isotropic_kind::sqrt, d2);

  std::ostringstream out;
  out << "// Auto-generated by numsim-codegen (slice-5 spectral e2e). Do not "
         "edit.\n#pragma once\n";
  out << "#include <tmech/tmech.h>\n#include <cmath>\n";
  out << "#include <numsim_codegen/runtime/spectral.h>\n\n";
  out << "namespace spectral_check {\n\n";
  out << defs.str();
  out << "// Number of components that disagree with the CAS evaluator.\n";
  out << "inline int run_spectral_check() {\n  int fails = 0;\n";
  out << checks.str();
  out << "  return fails;\n}\n\n} // namespace spectral_check\n";

  const std::string text = out.str();
  // Self-guard (review finding #7): the shipped/generated code must be
  // tmech-only. If a future emitter change ever leaks a numsim-cas symbol or
  // include into the generated output, fail the build here rather than ship it.
  if (text.find("numsim_cas") != std::string::npos ||
      text.find("numsim::cas") != std::string::npos) {
    std::cerr
        << "generate_spectral_check: generated code references numsim-cas "
           "— it must be tmech-only.\n";
    return 1;
  }

  std::ofstream f(argv[1]);
  f << text;
  return f ? 0 : 1;
}
