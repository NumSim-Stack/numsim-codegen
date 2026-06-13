#ifndef NUMSIM_CODEGEN_TARGETS_TARGET_FACTORY_H
#define NUMSIM_CODEGEN_TARGETS_TARGET_FACTORY_H

#include <numsim_codegen/targets/target.h>

#include <memory>
#include <string_view>
#include <vector>

namespace numsim::codegen {

// Name-keyed construction of a Target backend — the single source of truth for
// "which backend", promoted out of the hand-written if/else in the example
// driver (numsim-codegen#93). The keys are the stable SELECTOR (a future
// `target.type` in JSON config) and are intentionally independent of each
// `Target::target_name()` (a human-readable display label, e.g. "NumSimMaterial")
// — do not assume they match or couple one to the other.
//
// `make_target` constructs targets with their DEFAULT app-name / linear-algebra
// backend; passing those (e.g. a non-default MOOSE app name or Armadillo) is the
// deferred config-driven path (#93), so for now construct the target directly if
// you need non-defaults.
//
// Deliberately a flat function, NOT an extensible registry: there are three
// first-party targets and no third-party-registration consumer yet. The
// registry / per-strategy-schema framework is the deferred parameterization
// backbone (#93) — gated on the verification spine + a real consumer.

inline constexpr std::string_view default_target_name = "numsim_material";

// The valid target selectors, default first.
[[nodiscard]] auto target_names() -> std::vector<std::string_view> const &;

// Construct a Target by name. The linear-algebra-backed targets bind the default
// `LinearAlgebraEmitter` (a static singleton — safe lifetime). Throws
// std::runtime_error on an unknown name, listing the valid ones.
[[nodiscard]] auto make_target(std::string_view name) -> std::unique_ptr<Target>;

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_TARGETS_TARGET_FACTORY_H
