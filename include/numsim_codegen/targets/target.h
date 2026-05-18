#ifndef NUMSIM_CODEGEN_TARGETS_TARGET_H
#define NUMSIM_CODEGEN_TARGETS_TARGET_H

#include <numsim_codegen/recipe.h>

#include <string>
#include <vector>

namespace numsim::codegen {

// Multi-file output bundle: a target may emit one or more source files
// (e.g. MOOSE produces a .h + .C pair; standalone C++ produces a single
// inline header). Backends construct EmittedFiles entries and the user
// writes them to disk.
struct EmittedFile {
  std::string filename;   // suggested filename (no directory)
  std::string contents;   // full source text
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
