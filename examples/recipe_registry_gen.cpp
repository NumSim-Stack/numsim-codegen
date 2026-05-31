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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {

void write_file(fs::path const &out_dir,
                numsim::codegen::EmittedFile const &f) {
  auto path = out_dir;
  if (!f.install_subdir.empty()) path /= f.install_subdir;
  path /= f.filename;
  fs::create_directories(path.parent_path());
  std::ofstream(path) << f.contents;
  std::cout << "  wrote " << path << "\n";
}

auto make_target(std::string const &kind)
    -> std::unique_ptr<numsim::codegen::Target> {
  using namespace numsim::codegen;
  if (kind == "standalone") return std::make_unique<StandaloneCxxTarget>();
  if (kind == "moose")      return std::make_unique<MooseMaterialTarget>("ExampleApp");
  return nullptr;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <out-dir> <target: standalone|moose>\n";
    return 1;
  }
  fs::path const out_dir = argv[1];
  std::string const target_kind = argv[2];

  auto target = make_target(target_kind);
  if (!target) {
    std::cerr << "unknown target: " << target_kind
              << " (expected 'standalone' or 'moose')\n";
    return 1;
  }

  for (auto const &entry : numsim::examples::registry()) {
    std::cout << entry.name << " -> " << target->target_name() << "\n";
    auto model = entry.build();
    for (auto const &file : target->emit(model)) {
      write_file(out_dir, file);
    }
  }
  return 0;
}
