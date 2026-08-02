#include <numsim_codegen/targets/target_factory.h>

#include <numsim_codegen/targets/calculix_external.h>
#include <numsim_codegen/targets/moose_material.h>
#include <numsim_codegen/targets/numsim_material.h>
#include <numsim_codegen/targets/standalone_cxx.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace numsim::codegen {

namespace {

// Single source of truth for the target registry — both `target_names()` and
// `make_target()` iterate this, so a target can't appear in one list but not
// the other (review: arch #5). Default first (matches `default_target_name`).
// The LA-backed targets default-construct their `LinearAlgebraEmitter const&`
// from the static `default_linear_algebra_emitter()` accessor (safe lifetime,
// not the `=delete`'d rvalue ctor).
struct Entry {
  std::string_view name;
  std::unique_ptr<Target> (*make)();
};

auto registry() -> std::vector<Entry> const & {
  static std::vector<Entry> const entries{
      {"numsim_material",
       [] { return std::unique_ptr<Target>(std::make_unique<NumSimMaterialTarget>()); }},
      {"standalone",
       [] { return std::unique_ptr<Target>(std::make_unique<StandaloneCxxTarget>()); }},
      {"moose",
       [] { return std::unique_ptr<Target>(std::make_unique<MooseMaterialTarget>()); }},
      {"calculix",
       [] { return std::unique_ptr<Target>(std::make_unique<CalculiXExternalTarget>()); }},
  };
  return entries;
}

} // namespace

auto target_names() -> std::vector<std::string_view> const & {
  static std::vector<std::string_view> const names = [] {
    std::vector<std::string_view> n;
    for (auto const &e : registry()) n.push_back(e.name);
    return n;
  }();
  return names;
}

auto make_target(std::string_view name) -> std::unique_ptr<Target> {
  for (auto const &e : registry())
    if (e.name == name) return e.make();

  std::string msg = "make_target: unknown target '";
  msg.append(name);
  msg += "'. Known:";
  for (auto const &e : registry()) {
    msg += ' ';
    msg.append(e.name);
  }
  throw std::runtime_error(msg);
}

} // namespace numsim::codegen
