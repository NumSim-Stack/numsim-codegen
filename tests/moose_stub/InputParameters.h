#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_INPUT_PARAMETERS_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_INPUT_PARAMETERS_H

// MOOSE-API stub (#132): InputParameters. The emitter uses exactly
// `addClassDescription`, `addParam<Real>` (with a default) and
// `addRequiredParam<MaterialPropertyName>`; the Material constructor reads
// back through `getParam<Real>`. Only Real-valued parameters are stored —
// MaterialPropertyName parameters are name-plumbing MOOSE resolves at coupling
// time, which the stub's PropertyStore replaces.

#include "MooseTypes.h"

#include <map>
#include <string>
#include <type_traits>

class InputParameters {
public:
  void addClassDescription(std::string const & /*description*/) {}

  template <typename T>
  void addParam(std::string const &name, T default_value,
                std::string const & /*doc*/) {
    static_assert(std::is_same_v<T, Real>,
                  "moose_stub: only Real parameters are supported");
    m_reals[name] = default_value;
  }

  template <typename T>
  void addRequiredParam(std::string const & /*name*/,
                        std::string const & /*doc*/) {
    static_assert(std::is_same_v<T, MaterialPropertyName>,
                  "moose_stub: only MaterialPropertyName required parameters "
                  "are supported");
  }

  template <typename T>
  [[nodiscard]] auto get(std::string const &name) const -> T const & {
    static_assert(std::is_same_v<T, Real>,
                  "moose_stub: only Real parameters are supported");
    return m_reals.at(name);
  }

  // Driver-side convenience: override a parameter before construction.
  void set(std::string const &name, Real value) { m_reals[name] = value; }

private:
  std::map<std::string, Real> m_reals;
};

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_INPUT_PARAMETERS_H
