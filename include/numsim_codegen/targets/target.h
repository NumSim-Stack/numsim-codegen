#ifndef NUMSIM_CODEGEN_TARGETS_TARGET_H
#define NUMSIM_CODEGEN_TARGETS_TARGET_H

#include <numsim_codegen/recipe.h>

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

  // Human-readable name of the target framework — used in error messages
  // and diagnostics.
  [[nodiscard]] virtual auto target_name() const -> std::string = 0;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_TARGET_H
