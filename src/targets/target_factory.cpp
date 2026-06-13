#include <numsim_codegen/targets/target_factory.h>

#include <numsim_codegen/targets/moose_material.h>
#include <numsim_codegen/targets/numsim_material.h>
#include <numsim_codegen/targets/standalone_cxx.h>

#include <stdexcept>
#include <string>

namespace numsim::codegen {

auto target_names() -> std::vector<std::string_view> const & {
  // Default first (kept in sync with make_target + default_target_name).
  static std::vector<std::string_view> const names{"numsim_material",
                                                   "standalone", "moose"};
  return names;
}

auto make_target(std::string_view name) -> std::unique_ptr<Target> {
  // The LA-backed targets default-construct their `LinearAlgebraEmitter const&`
  // from the static `default_linear_algebra_emitter()` accessor — safe lifetime,
  // and not the `=delete`'d rvalue ctor.
  if (name == "numsim_material") return std::make_unique<NumSimMaterialTarget>();
  if (name == "standalone") return std::make_unique<StandaloneCxxTarget>();
  if (name == "moose") return std::make_unique<MooseMaterialTarget>();

  std::string msg = "make_target: unknown target '";
  msg.append(name);
  msg += "'. Known:";
  for (auto const known : target_names()) {
    msg += ' ';
    msg.append(known);
  }
  throw std::runtime_error(msg);
}

} // namespace numsim::codegen
