// Multi-recipe, multi-target generator driven by the registry in
// `examples/recipe_registry.h`. Iterates every registered recipe and
// emits its source via the target selected on argv.
//
// Usage:
//   ./recipe_registry_gen <out-dir> <target: standalone|moose>
//
// Files land under <out-dir>/<install_subdir>/<filename> following the
// target's install convention (e.g. include/materials/<Name>.h for MOOSE).

#include "recipe_registry.h"

#include <numsim_codegen/targets/target_factory.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Write an emitted file to disk. Returns true on success; on failure
// (directory-create error, file-open error, or write error) prints the
// path + error to stderr and returns false. The caller should propagate
// a non-zero exit code so a missing recipe doesn't go silently to /dev/null.
[[nodiscard]] bool write_file(fs::path const &out_dir,
                              numsim::codegen::EmittedFile const &f) {
  auto path = out_dir;
  if (!f.install_subdir.empty()) path /= f.install_subdir;
  path /= f.filename;

  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  if (ec) {
    std::cerr << "create_directories failed for " << path.parent_path()
              << ": " << ec.message() << "\n";
    return false;
  }

  std::ofstream os(path);
  if (!os) {
    std::cerr << "open failed: " << path << "\n";
    return false;
  }
  os << f.contents;
  if (!os) {
    std::cerr << "write failed: " << path << "\n";
    return false;
  }

  std::cout << "  wrote " << path << "\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <out-dir> <target: numsim_material|standalone|moose>\n";
    return 1;
  }
  fs::path const out_dir = argv[1];
  std::string const target_kind = argv[2];

  // The library's name-keyed factory (throws on an unknown name).
  std::unique_ptr<numsim::codegen::Target> target;
  try {
    target = numsim::codegen::make_target(target_kind);
  } catch (std::exception const &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  // issue #135: a target may legitimately reject a recipe outside its
  // supported scope (e.g. NumSimMaterialTarget on the J2 trial recipe).
  // Skip it with the target's own explanation and keep going — one
  // unsupported catalogue entry must not abort the whole generation run.
  // Exit non-zero only when nothing was generated (or a write failed).
  int exit_code = 0;
  std::size_t emitted = 0;
  for (auto const &entry : numsim::examples::registry()) {
    std::cout << entry.name << " -> " << target->target_name() << "\n";
    auto model = entry.build();
    std::vector<numsim::codegen::EmittedFile> files;
    try {
      files = target->emit(model);
    } catch (std::exception const &e) {
      std::cout << "  SKIPPED (" << e.what() << ")\n";
      continue;
    }
    for (auto const &file : files) {
      if (!write_file(out_dir, file)) {
        exit_code = 2;
      }
    }
    ++emitted;
  }
  if (emitted == 0) {
    std::cerr << "no recipe in the catalogue could be emitted via '"
              << target_kind << "'\n";
    return 3;
  }
  return exit_code;
}
