#include <numsim_codegen/targets/calculix_umat.h>

#include <numsim_codegen/recipe.h>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::codegen {

namespace {

// A rank-2 tensor argument we route across the CalculiX boundary.
struct TensorArg {
  std::string name;
  std::size_t dim = 0;
  std::size_t rank = 0;
};

// CalculiX's native `umat_user` argument list (see umat_main.f's CALL). All
// Fortran arguments are passed by reference; default `integer` maps to `int`,
// `real*8` to `double`. The leading `character*80` name adds a hidden trailing
// length argument under gfortran's calling convention. We only *use* a handful
// (elconloc, emec, icmd, stre, stiff); the rest are named in comments so the
// ABI (argument count + order + widths) matches exactly and the used ones land
// in the right slots. This exact prototype is also what the ABI-driver test
// declares, and must match the `umat_main.f` CALL in the `ccx` we link against.
constexpr char const *kUmatUserSignature = R"(extern "C" void umat_user_(
    char *      /*amat*/,      int *    /*iel*/,     int *    /*iint*/,
    int *       /*kode*/,      double * elconloc,    double * emec,
    double *    /*emec0*/,     double * /*beta*/,    double * /*xokl*/,
    double *    /*voj*/,       double * /*xkl*/,     double * /*vj*/,
    int *       /*ithermal*/,  double * /*t1l*/,     double * /*dtime*/,
    double *    /*time*/,      double * /*ttime*/,   int *    icmd,
    int *       /*ielas*/,     int *    /*mint_*/,   int *    /*nstate_*/,
    double *    /*xstateini*/, double * /*xstate*/,  double * stre,
    double *    stiff,         int *    /*iorien*/,  double * /*pgauss*/,
    double *    /*orab*/,      double * /*pnewdt*/,  int *    /*ipkon*/,
    long        /*amat_len*/))";

} // namespace

auto CalculiXUMATTarget::emit(ConstitutiveModel const &model) const
    -> std::vector<EmittedFile> {
  // Emit the Layer-2 body FIRST (also lets History/unsupported roles throw
  // their own diagnostics), then read the canonical argument list — the same
  // post-emit order MOOSE builds its call from, so the wrapper cannot drift
  // from the generated signature (issue #77). Gating the linalg include on the
  // emitter's usage marker (not a re-derived predicate) cannot drift either.
  std::string const body = model.emit_compute_function(m_la);
  bool const needs_la = body.find(m_la.usage_marker()) != std::string::npos;

  // ── Validate scope + collect the boundary variables ──────────────────────
  // Walk the SAME canonical argument order the Layer-2 signature uses, so the
  // wrapper's call cannot drift from the generated `_compute` (issue #77).
  std::vector<std::string> param_names; // → *USER MATERIAL constants (elconloc)
  std::optional<TensorArg> strain_in;   // → emec
  std::optional<TensorArg> stress_out;  // → stre
  std::optional<TensorArg> tangent_out; // → stiff

  auto const reject = [&](std::string const &what) {
    throw std::runtime_error(
        "CalculiXUMATTarget: recipe '" + model.name() + "' " + what +
        ". This target's first cut supports stateless materials only "
        "(one strain input, one stress output, one consistent tangent, plus "
        "scalar parameters); state variables / scalar inputs / time-stepping "
        "are a tracked follow-up.");
  };

  for (auto const &a : canonical_arguments(RecipeView{model})) {
    switch (a.role) {
    case ArgSpec::Role::ScalarParam:
      param_names.push_back(a.name);
      break;
    case ArgSpec::Role::TensorInput:
      if (strain_in) reject("has more than one tensor input");
      if (a.dim != 3 || a.rank != 2)
        reject("has a strain input that is not a 3D rank-2 tensor");
      strain_in = TensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::TensorOutput:
      if (stress_out) reject("has more than one tensor output");
      if (a.dim != 3 || a.rank != 2)
        reject("has a stress output that is not a 3D rank-2 tensor");
      stress_out = TensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::TensorTangentOutput:
      if (tangent_out) reject("has more than one consistent tangent");
      if (a.dim != 3)
        reject("has a consistent tangent that is not 3D");
      tangent_out = TensorArg{a.name, a.dim, a.rank};
      break;
    case ArgSpec::Role::ScalarInput:
      reject("has a scalar input '" + a.name + "'");
      break;
    case ArgSpec::Role::TimeStep:
      reject("uses the time step (rate/implicit form)");
      break;
    case ArgSpec::Role::ScalarOutput:
      reject("has a scalar output '" + a.name +
             "' (residual/Jacobian or state output)");
      break;
    case ArgSpec::Role::StateOld:
    case ArgSpec::Role::StateCurrentRead:
    case ArgSpec::Role::NewtonStateOut:
      reject("has an internal state variable '" + a.name + "'");
      break;
    }
  }

  if (!strain_in) reject("has no tensor (strain) input");
  if (!stress_out) reject("has no tensor (stress) output");
  if (!tangent_out)
    reject("has no consistent tangent (add_algorithmic_tangent is required "
           "so CalculiX gets the material stiffness `stiff`)");

  std::ostringstream os;
  os << "// Auto-generated by numsim-codegen. Do not edit.\n";
  os << "//\n";
  os << "// CalculiX user material (umat_user ABI). Link this object into `ccx`\n";
  os << "// built from source in place of the stock umat_user.f; drive it from an\n";
  os << "// input deck with `*MATERIAL, NAME=USER" << model.name() << "` +\n";
  os << "// `*USER MATERIAL, CONSTANTS=" << param_names.size() << "`.\n\n";
  os << "#include <tmech/tmech.h>\n";
  os << "#include <cmath>\n";
  if (needs_la) {
    for (auto const &inc : m_la.includes()) {
      os << "#include " << inc << "\n";
    }
  }
  os << "\n";
  os << body;
  os << "\n";

  // ── The umat_user boundary wrapper ───────────────────────────────────────
  os << kUmatUserSignature << " {\n";

  os << "  // Material constants, in *USER MATERIAL declaration order.\n";
  for (std::size_t k = 0; k < param_names.size(); ++k) {
    os << "  double const " << param_names[k] << " = elconloc[" << k << "];\n";
  }

  os << "  // Strain in: CalculiX passes the tensorial Lagrange strain in\n";
  os << "  // abq_std order {11,22,33,12,13,23}; abq_std<3,false> reads it as a\n";
  os << "  // full symmetric tensor with no engineering-shear scaling.\n";
  os << "  tmech::adaptor<double const, 3, 2, tmech::abq_std<3, false>> "
     << strain_in->name << "_ad(emec);\n";

  os << "  // Stress out: written straight into stre(6) in the same order.\n";
  os << "  tmech::adaptor<double, 3, 2, tmech::abq_std<3, false>> "
     << stress_out->name << "_ad(stre);\n";

  os << "  // Consistent tangent: fill a 6x6 (row-major) via abq_std<3,false>.\n";
  os << "  // For the minor-symmetric C emitted from a symmetric-strain recipe\n";
  os << "  // this 6x6 IS CalculiX's engineering stiffness D-matrix (no factors).\n";
  os << "  double " << tangent_out->name << "_D6[36];\n";
  os << "  tmech::adaptor<double, 3, 4, tmech::abq_std<3, false>> "
     << tangent_out->name << "_ad(" << tangent_out->name << "_D6);\n";

  // Build the call in canonical argument order (issue #77): map each role to
  // the local wrapper actual.
  os << "\n  " << model.name() << "_compute(\n";
  bool first = true;
  for (auto const &a : canonical_arguments(RecipeView{model})) {
    if (!first) os << ",\n";
    first = false;
    switch (a.role) {
    case ArgSpec::Role::ScalarParam:
      os << "      " << a.name; // local const from elconloc
      break;
    case ArgSpec::Role::TensorInput:
    case ArgSpec::Role::TensorOutput:
    case ArgSpec::Role::TensorTangentOutput:
      os << "      " << a.name << "_ad"; // boundary adaptor
      break;
    // Unreachable: every other role was rejected above. Enumerated (no
    // default:) so a new ArgSpec::Role is a -Wswitch warning here too.
    case ArgSpec::Role::ScalarInput:
    case ArgSpec::Role::StateOld:
    case ArgSpec::Role::StateCurrentRead:
    case ArgSpec::Role::TimeStep:
    case ArgSpec::Role::ScalarOutput:
    case ArgSpec::Role::NewtonStateOut:
      throw std::runtime_error(
          "CalculiXUMATTarget: internal error — unrejected role reached the "
          "call site for '" + a.name + "'");
    }
  }
  os << ");\n";

  os << "\n";
  os << "  // Pack the 6x6 into CalculiX's symmetric stiff(21): column-major\n";
  os << "  // upper-triangular, k = i + j*(j+1)/2 (0-based i<=j), symmetrized.\n";
  os << "  // icmd==3 means CalculiX wants stress only, so skip the tangent.\n";
  os << "  if (*icmd != 3) {\n";
  os << "    for (int j = 0; j < 6; ++j) {\n";
  os << "      for (int i = 0; i <= j; ++i) {\n";
  os << "        stiff[i + j * (j + 1) / 2] =\n";
  os << "            0.5 * (" << tangent_out->name << "_D6[i * 6 + j] + "
     << tangent_out->name << "_D6[j * 6 + i]);\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";

  return {EmittedFile{model.name() + "_umat.cpp", os.str(), "",
                      EmittedFile::Kind::Source}};
}

auto CalculiXUMATTarget::target_name() const -> std::string {
  return "CalculiXUMAT";
}

} // namespace numsim::codegen
