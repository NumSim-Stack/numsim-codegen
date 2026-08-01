#ifndef NUMSIM_CODEGEN_SRC_TARGETS_CALCULIX_BOUNDARY_H
#define NUMSIM_CODEGEN_SRC_TARGETS_CALCULIX_BOUNDARY_H

// Shared boundary logic for the two CalculiX targets (calculix_umat.cpp linked
// into ccx, calculix_external.cpp loaded via dlopen). Both encode ONE contract —
// the stateless-elastic scope rules and the CalculiX `stiff(21)` packing — so it
// lives here rather than being copy-pasted (review: arch #2). The lifecycle
// (extern "C" umat_user_ vs the external plugin) stays per-target.

#include <numsim_codegen/recipe.h>

#include <cstddef>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::codegen::detail {

struct CalculiXTensorArg {
  std::string name;
  std::size_t dim = 0;
  std::size_t rank = 0;
};

// The collected, validated boundary variables of a stateless-elastic recipe.
struct CalculiXScope {
  std::vector<std::string> params; // → *USER MATERIAL constants, in order
  CalculiXTensorArg strain;        // the one rank-2 (symmetric) tensor input
  CalculiXTensorArg stress;        // the one rank-2 (symmetric) tensor output
  CalculiXTensorArg tangent;       // the one rank-4 consistent tangent
};

// Scan the canonical argument list (the same post-emit order the generated
// signature uses — issue #77) and enforce the stateless-elastic scope. `label`
// prefixes diagnostics (the calling target's name). Throws std::runtime_error
// on any violation; returns the collected variables otherwise.
[[nodiscard]] inline auto
scan_calculix_scope(ConstitutiveModel const &model, char const *label)
    -> CalculiXScope {
  CalculiXScope scope;
  std::optional<CalculiXTensorArg> strain, stress, tangent;

  auto const reject = [&](std::string const &what) {
    throw std::runtime_error(
        std::string(label) + ": recipe '" + model.name() + "' " + what +
        ". This target's first cut supports stateless materials only (one "
        "symmetric strain input, one stress output, one consistent tangent, "
        "plus scalar parameters → the *USER MATERIAL constants); state "
        "variables, scalar inputs and rate/implicit forms are a follow-up.");
  };

  // is_symmetric of a declared tensor input / output (roles::Strain, Stress are
  // symmetric; roles::DeformationGradient is NOT — abq_std is a symmetric
  // 6-component adaptor, so a non-symmetric leaf would be silently truncated).
  auto input_is_symmetric = [&](std::string const &name) {
    for (auto const &s : model.inputs())
      if (s.name == name) return s.role.is_symmetric;
    return false;
  };
  auto output_is_symmetric = [&](std::string const &name) {
    for (auto const &o : model.outputs())
      if (o.name == name) return o.role.is_symmetric;
    return false;
  };

  for (auto const &a : canonical_arguments(RecipeView{model})) {
    switch (a.role) {
    case ArgSpec::Role::ScalarParam:
      scope.params.push_back(a.name);
      break;
    case ArgSpec::Role::TensorInput:
      if (strain) reject("has more than one tensor input");
      if (a.dim != 3 || a.rank != 2)
        reject("has a strain input '" + a.name +
               "' that is not a 3D rank-2 tensor");
      if (!input_is_symmetric(a.name))
        reject("has a non-symmetric tensor input '" + a.name +
               "' (e.g. a deformation gradient); CalculiX's abq_std Voigt "
               "boundary is symmetric — use a symmetric strain measure "
               "(roles::Strain)");
      strain = CalculiXTensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::TensorOutput:
      if (stress) reject("has more than one tensor output");
      if (a.dim != 3 || a.rank != 2)
        reject("has a stress output '" + a.name +
               "' that is not a 3D rank-2 tensor");
      if (!output_is_symmetric(a.name))
        reject("has a non-symmetric tensor output '" + a.name +
               "' (CalculiX stress storage is symmetric)");
      stress = CalculiXTensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::TensorTangentOutput:
      if (tangent) reject("has more than one consistent tangent");
      if (a.dim != 3) reject("has a consistent tangent that is not 3D");
      tangent = CalculiXTensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::ScalarInput:
      reject("has a scalar input '" + a.name + "'");
      break;
    case ArgSpec::Role::TimeStep:
      reject("uses the time step (rate/implicit form)");
      break;
    case ArgSpec::Role::ScalarOutput:
      reject("has a scalar output '" + a.name + "'");
      break;
    case ArgSpec::Role::StateOld:
    case ArgSpec::Role::StateCurrentRead:
    case ArgSpec::Role::NewtonStateOut:
      reject("has an internal state variable '" + a.name + "'");
      break;
    }
  }

  if (!strain) reject("has no tensor (strain) input");
  if (!stress) reject("has no tensor (stress) output");
  if (!tangent)
    reject("has no consistent tangent (add_algorithmic_tangent is required so "
           "CalculiX gets the material stiffness)");

  scope.strain = *strain;
  scope.stress = *stress;
  scope.tangent = *tangent;
  return scope;
}

// Emit the pack of a row-major 6x6 (`d6_name`) into CalculiX's symmetric
// stiff(21): column-major upper-triangular, k = i + j*(j+1)/2 (0-based i<=j),
// symmetrized. The consistent tangent is assumed MAJOR-symmetric (D_IJ==D_JI);
// for a material whose tangent is not (e.g. non-associative plasticity) the
// antisymmetric part is silently averaged away — revisit when that lands.
// `stiff_ptr` is the destination array expression, `icmd_expr` the (int) icmd
// value expression (ccx passes icmd==3 to request stress only).
inline void emit_stiff21_packing(std::ostream &os, std::string const &d6_name,
                                 std::string const &stiff_ptr,
                                 std::string const &icmd_expr,
                                 std::string const &indent) {
  os << indent << "// Pack the 6x6 into CalculiX's symmetric stiff(21):\n";
  os << indent << "// column-major upper, k = i + j*(j+1)/2 (0-based i<=j),\n";
  os << indent << "// major-symmetrized. icmd==3 → CalculiX wants stress only.\n";
  os << indent << "if (" << icmd_expr << " != 3) {\n";
  os << indent << "  for (int j = 0; j < 6; ++j)\n";
  os << indent << "    for (int i = 0; i <= j; ++i)\n";
  os << indent << "      " << stiff_ptr << "[i + j * (j + 1) / 2] =\n";
  os << indent << "          0.5 * (" << d6_name << "[i * 6 + j] + " << d6_name
     << "[j * 6 + i]);\n";
  os << indent << "}\n";
}

} // namespace numsim::codegen::detail

#endif // NUMSIM_CODEGEN_SRC_TARGETS_CALCULIX_BOUNDARY_H
