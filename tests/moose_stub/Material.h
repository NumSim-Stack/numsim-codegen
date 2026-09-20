#ifndef NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_H
#define NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_H

// MOOSE-API stub (#132): the Material base class surface MooseMaterialTarget's
// emitted classes derive from:
//   * `static InputParameters validParams()` + ctor from `InputParameters`,
//   * protected virtual `computeQpProperties` / `initQpStatefulProperties`
//     hooks (the emitted class overrides them),
//   * `getParam<Real>`, `declareProperty<T>`, `getMaterialProperty<T>`,
//     `getMaterialPropertyOld<Real>`,
//   * the `_qp` index and the framework timestep `_dt`,
//   * the `registerMooseObject` registration macro.
// Property resolution goes through the process-global PropertyStore (see
// MaterialProperty.h) so the driver can seed inputs and read outputs.

#include "InputParameters.h"
#include "MaterialProperty.h"
#include "MooseTypes.h"

#include <string>

class Material {
public:
  [[nodiscard]] static auto validParams() -> InputParameters { return {}; }

  explicit Material(InputParameters parameters)
      : m_parameters(std::move(parameters)) {}
  Material(Material const &) = delete;
  auto operator=(Material const &) -> Material & = delete;
  virtual ~Material() = default;

  // Stub-only driver hook (real MOOSE invokes the protected hooks itself
  // during a timestep): initialise stateful properties, then evaluate one
  // quadrature point.
  void stubEvaluateQp(unsigned int qp = 0) {
    _qp = qp;
    initQpStatefulProperties();
    computeQpProperties();
  }

protected:
  virtual void computeQpProperties() {}
  virtual void initQpStatefulProperties() {}

  template <typename T>
  [[nodiscard]] auto getParam(std::string const &name) const -> T const & {
    return m_parameters.get<T>(name);
  }

  template <typename T>
  [[nodiscard]] auto declareProperty(std::string const &name)
      -> MaterialProperty<T> & {
    return moose_stub::PropertyStore::global().get_or_create<T>(name);
  }

  template <typename T>
  [[nodiscard]] auto getMaterialProperty(std::string const &name)
      -> MaterialProperty<T> const & {
    return moose_stub::PropertyStore::global().get_or_create<T>(name);
  }

  template <typename T>
  [[nodiscard]] auto getMaterialPropertyOld(std::string const &name)
      -> MaterialProperty<T> const & {
    return moose_stub::PropertyStore::global().get_or_create<T>(
        moose_stub::old_property_key(name));
  }

  unsigned int _qp = 0;
  Real _dt = 1.0;

private:
  InputParameters m_parameters;
};

// Real MOOSE registers the class with its factory; the stub only needs the
// statement to compile (and to keep referencing the class so a typo still
// fails). `sizeof` on the app-name string literal keeps that argument legal.
#define registerMooseObject(app, classname)                                   \
  static_assert(sizeof(app) > 0 && sizeof(classname) > 0,                     \
                "moose_stub registerMooseObject")

#endif // NUMSIM_CODEGEN_TESTS_MOOSE_STUB_MATERIAL_H
