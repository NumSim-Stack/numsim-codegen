#ifndef NUMSIM_CODEGEN_TARGETS_TARGET_H
#define NUMSIM_CODEGEN_TARGETS_TARGET_H

#include <numsim_codegen/recipe.h>

#include <expected>
#include <string>
#include <vector>

namespace numsim::codegen {

// Multi-file output bundle: a target may emit one or more source files
// (e.g. MOOSE produces a .h + .C pair; standalone C++ produces a single
// inline header). Backends construct EmittedFile entries and the user
// writes them to disk in the target's conventional install layout.
struct EmittedFile {
  enum class Kind {
    Header,        // C++ header — typically lives in include/...
    Source,        // C++ source — typically lives in src/...
    Other,         // anything else (build snippet, CMake fragment, etc.)
  };

  std::string filename;        // basename, no directory
  std::string contents;        // full source text
  std::string install_subdir;  // suggested relative install directory,
                               // e.g. "include/materials" or "src/materials".
                               // Empty if the target has no convention.
  Kind kind = Kind::Other;
};

// Abstract backend interface. Each target framework (MOOSE, Abaqus UMAT,
// ANSYS USERMAT, ...) provides one concrete Target subclass. The recipe
// is target-agnostic; the Target interprets it.
class Target {
public:
  virtual ~Target() = default;

  // Generate the source files for this constitutive model in the target's
  // framework conventions.
  [[nodiscard]] virtual auto emit(ConstitutiveModel const &model) const
      -> std::vector<EmittedFile> = 0;

  // Capability query (#137): would this target's UP-FRONT scope guards accept
  // the recipe's shape? On rejection the error carries the exact reason string
  // the matching emit() throw uses — one message, two transports — so a
  // generic driver can skip-and-report without catching exceptions.
  //
  // Success does NOT guarantee emit() succeeds: can_emit checks only the
  // up-front recipe-shape guards; emit-time validation (name collisions with
  // synthesized members, unbound expression leaves, non-finite parameter
  // defaults, pass-level checks) may still throw. The default implementation
  // is conservatively permissive — it reports success ("try emit"), so a
  // target without shape guards needs no override and one with them still
  // rejects loudly inside emit().
  [[nodiscard]] virtual auto can_emit(ConstitutiveModel const & /*model*/) const
      -> std::expected<void, std::string> {
    return {};
  }

  // Human-readable name of the target framework — used in error messages
  // and diagnostics.
  [[nodiscard]] virtual auto target_name() const -> std::string = 0;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_TARGET_H
