// Example registry of constitutive recipes for a multi-target generator.
//
// Pattern: each recipe is a free function returning a ConstitutiveModel.
// The registry maps a human-readable name to its factory. A single
// generator binary can iterate the registry and emit code for every
// recipe, optionally selecting a target (StandaloneCxxTarget,
// MooseMaterialTarget, ...) from argv. See examples/recipe_registry_gen.cpp
// for the matching main.

#ifndef NUMSIM_CODEGEN_EXAMPLES_RECIPE_REGISTRY_H
#define NUMSIM_CODEGEN_EXAMPLES_RECIPE_REGISTRY_H

#include <numsim_codegen/numsim_codegen.h>

#include <numsim_cas/scalar/scalar_operators.h>
#include <numsim_cas/tensor/tensor_definitions.h>
#include <numsim_cas/tensor/tensor_operators.h>

#include <functional>
#include <string_view>
#include <vector>

namespace numsim::examples {

struct RecipeEntry {
  std::string_view name;
  std::function<numsim::codegen::ConstitutiveModel()> build;
};

// σ = 2μ ε — shear-only linear elasticity. The simplest non-trivial
// recipe; serves as the smoke test for any new target.
inline auto build_linear_elastic_shear()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("LinearElasticShear");
  auto mu  = m.add_parameter("mu", 0.5, "Shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  return m;
}

// σ = 2 μ(T) ε with μ(T) = μ₀ (1 − α T). Exercises a scalar input
// (temperature) coupled multiplicatively with a tensor input.
inline auto build_thermo_elastic_shear()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("ThermoElasticShear");
  auto mu0   = m.add_parameter("mu0", 1.0, "Reference shear modulus");
  auto alpha = m.add_parameter("alpha", 1.0e-3, "Thermal softening rate");
  auto T     = m.add_scalar_input("T", roles::Temperature);
  auto eps   = m.add_tensor_input("eps", 3, 2, roles::Strain);

  auto mu = mu0 * (1 - alpha * T);
  m.add_output("stress", 2 * mu * eps, roles::Stress);
  return m;
}

// σ = 2 (μ + γ φ) ε — phase-coupled shear with a user-defined Role
// for the phase field, demonstrating the open-set Role API: backends
// route this on its attributes (is_driving), not the catalogue identity.
inline auto build_phase_coupled_shear()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("PhaseCoupledShear");
  auto mu    = m.add_parameter("mu", 0.5, "Base shear modulus");
  auto gamma = m.add_parameter("gamma", 0.1, "Phase coupling coefficient");

  auto phi = m.add_scalar_input("phi",
      Role{.name = "phase_field",
           .is_driving = true,
           .expected_rank = 0});
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);

  m.add_output("stress", 2 * (mu + gamma * phi) * eps, roles::Stress);
  return m;
}

// Linear elasticity in K/G (bulk + shear) form.
//
// PHASE A LIMITATION: the full Hookean form σ = K · tr(ε) · I + 2G · dev(ε)
// requires a `tensor_to_scalar × tensor` operator producing the volumetric
// stress K · tr(ε) · I from the identity tensor. That operator is not yet
// exposed by numsim-cas (see the limitation note at the top of
// examples/linear_elasticity.cpp). This recipe emits only the shear-only
// approximation σ = 2G · ε; K is declared in the API surface so callers
// already pass both moduli — when the missing operator lands, just swap
// in the full expression and the signature stays stable.
inline auto build_linear_elasticity_KG()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("LinearElasticityKG");
  [[maybe_unused]] auto K = m.add_parameter("K", 1.5, "Bulk modulus");
  auto G   = m.add_parameter("G", 0.5, "Shear modulus");
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);

  // TODO(numsim-cas): when t2s × tensor lands, replace with
  //   auto sigma = K * trace(eps) * Identity{3,2} + 2 * G * eps;
  auto sigma = 2 * G * eps;
  m.add_output("stress", sigma, roles::Stress);
  return m;
}

// Strain-coupled isotropic damage: σ = (1 − D) · 2μ · ε.
//
// The damage variable D is supplied as a scalar input — the consuming
// framework computes it from a damage-evolution model running outside
// this recipe. A self-contained `D = f(||ε||)` form would need the
// tensor norm + clamp operators (deferred). The scalar-input approach
// is the clean Phase A pattern: it decouples the damage evolution from
// the constitutive emission and lets users plug in any D-update law.
inline auto build_strain_based_damage()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("StrainBasedDamage");
  auto mu = m.add_parameter("mu", 0.5, "Undamaged shear modulus");

  auto D = m.add_scalar_input("D",
      Role{.name = "damage",
           .is_driving = true,
           .expected_rank = 0});
  auto eps = m.add_tensor_input("eps", 3, 2, roles::Strain);

  m.add_output("stress", (1 - D) * 2 * mu * eps, roles::Stress);
  return m;
}

// J2 plasticity — trial stress only.
//
//   σ_trial = 2G · (ε − ε_p_old)
//
// PHASE A LIMITATION: full J2 plasticity needs (a) stateful old/new
// plastic-strain handling (tracked in #15) and (b) a return-mapping
// algorithm that iteratively solves for the plastic multiplier Δλ.
// Neither is emittable from the symbolic recipe — the return mapping
// is an imperative algorithm, not a closed-form symbolic expression.
//
// This recipe emits the trial deviatoric stress (the *input* to the
// return mapping). The consuming framework runs the return mapping
// outside the generated function: takes σ_trial, performs the radial
// return, updates ε_p, and stores the new state.
//
// `eps_p_old` carries a custom Role rather than `roles::History` so
// that the MOOSE Phase A guard does not reject this recipe today.
// Once Phase B lands and history wiring works, switch this to
// `roles::History` and the backend will allocate the stateful pair.
inline auto build_j2_plasticity_trial()
    -> numsim::codegen::ConstitutiveModel {
  using namespace numsim::cas;
  using namespace numsim::codegen;

  ConstitutiveModel m("J2PlasticityTrial");
  auto G = m.add_parameter("G", 0.5, "Shear modulus");

  auto eps       = m.add_tensor_input("eps", 3, 2, roles::Strain);
  auto eps_p_old = m.add_tensor_input("eps_p_old", 3, 2,
      Role{.name = "plastic_strain_old",
           .expected_rank = 2});

  // Trial deviatoric-shear stress; the volumetric part needs the same
  // missing t2s × I operator that LinearElasticityKG flags.
  auto sigma_trial = 2 * G * (eps - eps_p_old);
  m.add_output("stress_trial", sigma_trial, roles::Stress);
  return m;
}

// The catalogue. Append new entries here as you add recipes; the
// generator main iterates this vector and needs no changes per recipe.
inline auto registry() -> std::vector<RecipeEntry> const & {
  static std::vector<RecipeEntry> const entries = {
      {"LinearElasticShear", &build_linear_elastic_shear},
      {"ThermoElasticShear", &build_thermo_elastic_shear},
      {"PhaseCoupledShear",  &build_phase_coupled_shear},
      {"LinearElasticityKG", &build_linear_elasticity_KG},
      {"StrainBasedDamage",  &build_strain_based_damage},
      {"J2PlasticityTrial",  &build_j2_plasticity_trial},
  };
  return entries;
}

} // namespace numsim::examples

#endif // NUMSIM_CODEGEN_EXAMPLES_RECIPE_REGISTRY_H
