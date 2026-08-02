#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_PROPERTY_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_PROPERTY_H

// MOOSE-API stub (#132): MaterialProperty<T> — the per-quadrature-point value
// array. The emitted code only ever indexes it with `_qp`; the stub carries a
// single quadrature point.
//
// `moose_stub::PropertyStore` stands in for MOOSE's MaterialData: a process-
// global, name-keyed registry so `declareProperty` / `getMaterialProperty`
// resolve the SAME storage regardless of declaration order (in real MOOSE the
// producer of a coupled input is another Material object), and so the test
// driver can reach input/output properties that are only protected members on
// the Material.

#include "MooseTypes.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T> class MaterialProperty {
public:
  MaterialProperty() : m_data(1) {}
  [[nodiscard]] auto operator[](unsigned int qp) -> T & { return m_data[qp]; }
  [[nodiscard]] auto operator[](unsigned int qp) const -> T const & {
    return m_data[qp];
  }

private:
  std::vector<T> m_data;
};

namespace moose_stub {

// Key under which a property's previous-step ("old") values live. Real MOOSE
// versions stateful properties internally; the stub keeps old state as a
// separate registry entry the driver can seed directly.
[[nodiscard]] inline auto old_property_key(std::string const &name)
    -> std::string {
  return name + "@old";
}

class PropertyStore {
public:
  [[nodiscard]] static auto global() -> PropertyStore & {
    static PropertyStore store;
    return store;
  }

  template <typename T>
  [[nodiscard]] auto get_or_create(std::string const &name)
      -> MaterialProperty<T> & {
    auto it = m_entries.find(name);
    if (it == m_entries.end()) {
      it = m_entries.emplace(name, std::make_shared<Holder<T>>()).first;
    }
    auto holder = std::dynamic_pointer_cast<Holder<T>>(it->second);
    if (!holder) {
      throw std::runtime_error("PropertyStore: property '" + name +
                               "' requested with a mismatching type");
    }
    return holder->value;
  }

private:
  struct HolderBase {
    HolderBase() = default;
    HolderBase(HolderBase const &) = delete;
    auto operator=(HolderBase const &) -> HolderBase & = delete;
    virtual ~HolderBase() = default;
  };
  template <typename T> struct Holder final : HolderBase {
    MaterialProperty<T> value;
  };

  std::map<std::string, std::shared_ptr<HolderBase>> m_entries;
};

} // namespace moose_stub

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_PROPERTY_H
