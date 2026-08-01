#include <numsim_codegen/targets/calculix_external.h>

#include <numsim_codegen/recipe.h>

#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::codegen {

namespace {

struct TensorArg {
  std::string name;
  std::size_t dim = 0;
  std::size_t rank = 0;
};

// CalculiX external-behaviour ABI (the `calculixptr` typedef in
// call_external_umat_user.c). All CalculiX quantities are passed by pointer;
// under the STANDARD interface they are NATIVE: STRAN1=emec (tensorial, abq_std
// order {11,22,33,12,13,23}), STRESS=stre, DDSDDE=stiff(21) packed column-major
// upper, STATEV0/1=history old/new, MPROPS=the *USER MATERIAL constants. The
// trailing `int size` is the amat length, passed BY VALUE. We name only the
// arguments we use.
constexpr char const *kExternalSignature = R"(extern "C" void NCG_UMAT(
    char const *   /*amat*/,   int const *    /*iel*/,    int const *    /*iint*/,
    int const *    /*NPROPS*/, double const * MPROPS,     double const * STRAN1,
    double const * /*STRAN0*/, double const * /*beta*/,   double const * /*F0*/,
    double const * /*voj*/,    double const * /*F1*/,     double const * /*vj*/,
    int const *    /*ithermal*/, double const * /*TEMP1*/, double const * /*DTIME*/,
    double const * /*time*/,   double const * /*ttime*/,  int const *    icmd,
    int const *    /*ielas*/,  int const *    /*mi*/,     int const *    /*NSTATV*/,
    double const * STATEV0,    double *       STATEV1,    double *       STRESS,
    double *       DDSDDE,      int const *    /*iorien*/, double const * /*pgauss*/,
    double const * /*orab*/,   double *       /*PNEWDT*/, int const *    /*ipkon*/,
    int            /*size*/))";

// Uppercase, alphanumeric-only: the library-name form CalculiX parses out of the
// (uppercased) `@<LIB>_NCG_UMAT` material name. Underscores are dropped so the
// first `_` in the deck name reliably splits LIB from the fixed FUNC `NCG_UMAT`.
auto library_name(std::string const &model) -> std::string {
  std::string out;
  for (char c : model)
    if (std::isalnum(static_cast<unsigned char>(c)))
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  return out;
}

} // namespace

auto CalculiXExternalTarget::emit(ConstitutiveModel const &model) const
    -> std::vector<EmittedFile> {
  // Emit the Layer-2 body first (also lets unsupported roles throw their own
  // diagnostics), then read the canonical arg order (same post-emit order the
  // signature uses, so the wrapper cannot drift — issue #77).
  std::string const body = model.emit_compute_function(m_la);
  bool const needs_la = body.find(m_la.usage_marker()) != std::string::npos;

  std::vector<std::string> param_names; // → MPROPS (the *USER MATERIAL constants)
  std::optional<TensorArg> strain_in;   // → STRAN1
  std::optional<TensorArg> stress_out;  // → STRESS
  std::optional<TensorArg> tangent_out; // → DDSDDE (stiff(21))

  auto const reject = [&](std::string const &what) {
    throw std::runtime_error(
        "CalculiXExternalTarget: recipe '" + model.name() + "' " + what +
        ". This target's first cut supports stateless materials only (one strain "
        "input, one stress output, one consistent tangent, plus scalar "
        "parameters); state variables (the STATEV round-trip) are the "
        "numsim-materials-backed follow-up.");
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
      if (a.dim != 3) reject("has a consistent tangent that is not 3D");
      tangent_out = TensorArg{a.name, a.dim, a.rank};
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

  if (!strain_in) reject("has no tensor (strain) input");
  if (!stress_out) reject("has no tensor (stress) output");
  if (!tangent_out)
    reject("has no consistent tangent (add_algorithmic_tangent is required so "
           "CalculiX gets the material stiffness)");

  auto const lib = library_name(model.name());

  std::ostringstream os;
  os << "// Auto-generated by numsim-codegen. Do not edit.\n";
  os << "//\n";
  os << "// CalculiX EXTERNAL behaviour (loaded at runtime via dlopen — no ccx\n";
  os << "// recompile). Build:  g++ -std=c++23 -O2 -fPIC -shared this.cpp \\\n";
  os << "//                         -I<tmech/include> -o lib" << lib << ".so\n";
  os << "// Deck:  *MATERIAL, NAME=@" << lib << "_NCG_UMAT\n";
  os << "//        *USER MATERIAL, CONSTANTS=" << param_names.size() << "\n";
  os << "// (ccx uppercases the name → dlopen lib" << lib
     << ".so, dlsym NCG_UMAT.)\n";
  os << "//\n";
  os << "// ① A load-time registry is unnecessary for a single-model library and\n";
  os << "//    is where the numsim-materials material would be registered for the\n";
  os << "//    stateful path; ② the material is built once per thread; ③ every\n";
  os << "//    call unpacks the strain, evaluates, and packs stress + tangent.\n\n";
  os << "#include <tmech/tmech.h>\n";
  os << "#include <cmath>\n";
  os << "#include <optional>\n";
  if (needs_la) {
    for (auto const &inc : m_la.includes()) os << "#include " << inc << "\n";
  }
  os << "\n";
  os << body;
  os << "\n";

  // ── ② per-thread material: captures the constants, evaluates per call ──────
  os << "namespace {\n\n";
  os << "// The material instance: constructed once per thread (see thread_state_for)\n";
  os << "// from the *USER MATERIAL constants, then reused for every call.\n";
  os << "struct ncg_material {\n";
  for (std::size_t k = 0; k < param_names.size(); ++k) {
    os << "  double const " << param_names[k] << ";\n";
  }
  os << "  explicit ncg_material(double const *mprops)";
  if (!param_names.empty()) {
    os << "\n      : ";
    for (std::size_t k = 0; k < param_names.size(); ++k) {
      if (k) os << ", ";
      os << param_names[k] << "(mprops[" << k << "])";
    }
  }
  os << " {}\n\n";

  os << "  void evaluate(double const *strain_in, double *stress_out,\n";
  os << "                double *tangent_stiff, int icmd) const {\n";
  os << "    // Strain in: tensorial, abq_std order {11,22,33,12,13,23}.\n";
  os << "    tmech::adaptor<double const, 3, 2, tmech::abq_std<3, false>> "
     << strain_in->name << "_ad(strain_in);\n";
  os << "    tmech::adaptor<double, 3, 2, tmech::abq_std<3, false>> "
     << stress_out->name << "_ad(stress_out);\n";
  os << "    double " << tangent_out->name << "_D6[36];\n";
  os << "    tmech::adaptor<double, 3, 4, tmech::abq_std<3, false>> "
     << tangent_out->name << "_ad(" << tangent_out->name << "_D6);\n";
  os << "\n    " << model.name() << "_compute(\n";
  bool first = true;
  for (auto const &a : canonical_arguments(RecipeView{model})) {
    if (!first) os << ",\n";
    first = false;
    switch (a.role) {
    case ArgSpec::Role::ScalarParam:
      os << "        " << a.name;
      break;
    case ArgSpec::Role::TensorInput:
    case ArgSpec::Role::TensorOutput:
    case ArgSpec::Role::TensorTangentOutput:
      os << "        " << a.name << "_ad";
      break;
    case ArgSpec::Role::ScalarInput:
    case ArgSpec::Role::StateOld:
    case ArgSpec::Role::StateCurrentRead:
    case ArgSpec::Role::TimeStep:
    case ArgSpec::Role::ScalarOutput:
    case ArgSpec::Role::NewtonStateOut:
      throw std::runtime_error(
          "CalculiXExternalTarget: internal error — unrejected role reached the "
          "call site for '" + a.name + "'");
    }
  }
  os << ");\n";
  os << "    // Pack the 6x6 into CalculiX's stiff(21): column-major upper,\n";
  os << "    // k = i + j*(j+1)/2 (0-based i<=j), symmetrized. icmd==3 → stress only.\n";
  os << "    if (icmd != 3) {\n";
  os << "      for (int j = 0; j < 6; ++j)\n";
  os << "        for (int i = 0; i <= j; ++i)\n";
  os << "          tangent_stiff[i + j * (j + 1) / 2] =\n";
  os << "              0.5 * (" << tangent_out->name << "_D6[i * 6 + j] + "
     << tangent_out->name << "_D6[j * 6 + i]);\n";
  os << "    }\n";
  os << "  }\n";
  os << "};\n\n";

  os << "// ② One material per thread, built on first call (ccx runs the element\n";
  os << "// loop multi-threaded; a per-thread instance avoids data races and\n";
  os << "// amortises construction).\n";
  os << "ncg_material &thread_state_for(double const *mprops) {\n";
  os << "  thread_local std::optional<ncg_material> cache;\n";
  os << "  if (!cache) cache.emplace(mprops);\n";
  os << "  return *cache;\n";
  os << "}\n\n";
  os << "} // namespace\n\n";

  // ── ③ the exported CalculiX external entry point ──────────────────────────
  os << kExternalSignature << " {\n";
  os << "  ncg_material &m = thread_state_for(MPROPS);\n";
  os << "  (void)STATEV0;\n";
  os << "  (void)STATEV1; // stateless: no history round-trip\n";
  os << "  m.evaluate(STRAN1, STRESS, DDSDDE, *icmd);\n";
  os << "}\n";

  return {EmittedFile{model.name() + "_ext.cpp", os.str(), "",
                      EmittedFile::Kind::Source}};
}

auto CalculiXExternalTarget::target_name() const -> std::string {
  return "CalculiXExternal";
}

} // namespace numsim::codegen
