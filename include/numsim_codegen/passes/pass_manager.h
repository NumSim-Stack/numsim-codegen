#ifndef NUMSIM_CODEGEN_PASS_MANAGER_H
#define NUMSIM_CODEGEN_PASS_MANAGER_H

#include <numsim_codegen/passes/pass.h>

#include <format>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace numsim::codegen {

// Runs a sequence of passes against a shared PassContext.
//
// Ordering is registration order — the PassManager does NOT topologically
// reorder. The pre/postcondition check exists only to fail loudly on
// misordered pipelines, not to silently fix them. Forcing the caller to
// register in the right order keeps debugging predictable: a pipeline that
// looks wrong on the page is wrong; one that looks right is right.
class PassManager {
public:
  void add(std::unique_ptr<Pass> p) {
    if (!p) {
      throw std::invalid_argument("PassManager::add: null pass");
    }
    m_passes.push_back(std::move(p));
  }

  template <class P, class... Args> void emplace(Args &&...args) {
    add(std::make_unique<P>(std::forward<Args>(args)...));
  }

  void run(PassContext &ctx) const {
    std::set<std::string_view> satisfied;
    for (auto const &p : m_passes) {
      for (auto pre : p->preconditions()) {
        if (!satisfied.contains(pre)) {
          throw std::runtime_error(std::format(
              "PassManager: pass '{}' requires precondition '{}' but no "
              "earlier pass advertised that postcondition.",
              p->name(), pre));
        }
      }
      p->run(ctx);
      for (auto post : p->postconditions()) {
        satisfied.insert(post);
      }
    }
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return m_passes.size();
  }

  // Read-only access for inspection (debugging, test harnesses, future
  // recipe-IR explorers).
  [[nodiscard]] auto passes() const noexcept
      -> std::vector<std::unique_ptr<Pass>> const & {
    return m_passes;
  }

private:
  std::vector<std::unique_ptr<Pass>> m_passes;
};

} // namespace numsim::codegen

#endif // NUMSIM_CODEGEN_PASS_MANAGER_H
