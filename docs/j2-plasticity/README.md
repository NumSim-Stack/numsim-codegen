# J2 plasticity — stress–strain plots

Response of the **codegen-generated** J2 return-map materials (`J2Return`, `J2Voce`,
`J2Swift` — see `tests/generated/generate_numsim_material_check.cpp`), driven through
the real `numsim-materials` property graph and `backward_euler` solver. These are the
same materials the e2e goldens (#90) FD-verify; the plots are a visual companion.

## Figures

### `j2_stress_strain.png` — loading modes
Uniaxial, simple shear, and equibiaxial loading of the linear-hardening `J2Return`
(`G=80 GPa, σ_y=250 MPa, H=5 GPa`).
- **Left:** physical stress component vs drive strain — elastic branch → yield knee →
  linear hardening; elastic slope and yield strain differ by mode geometry.
- **Right:** von Mises equivalent stress — all modes yield at the same level
  `√(3/2)·σ_y ≈ 306 MPa` (the J2 / pressure-insensitivity signature).

### `j2_nonlinear_hardening.png` — hardening laws
Uniaxial response for three isotropic hardening laws, sharing the elastic branch and
initial yield:
- **Linear** `σ_y = σ_y0 + H·Δγ` (straight),
- **Voce** `σ_y = σ_y0 + Q(1 − e^{−bΔγ})` (concave, saturating),
- **Swift** `σ_y = C(ε0 + Δγ)^k` (power law).

`cas::diff` differentiates `exp`/`pow` in the hardening law, so the consistent tangent
is correct for each (validated by the e2e FD checks).

## Scope / caveats (intrinsic to these goldens)
- **Deviatoric radial return, single-increment plastic branch.** No plastic-strain
  tensor history (εᵖ) and no elastic⇄plastic switch — those are epic #102 / Phase E.
  Curves are monotonic loading, not cyclic hysteresis.
- **Zero-strain / hydrostatic states are singular** (flow direction `n = s/‖s‖` is
  `0/0` at `dev=0`); they are excluded, and the emitted `converged()` guard throws
  there rather than emitting a NaN.
- Nonlinear-hardening figures use gentle, well-conditioned params: `backward_euler`'s
  `max(x,0)` clamp is fragile near the elastic-plastic boundary for stiff hardening.

## Reproducing
The `reproduce/` drivers instantiate the generated headers and sweep strain, writing
CSV that the `*.py` scripts plot. They are illustrative — they need the e2e include
paths (generated `J2*.h` from the build tree, plus `numsim-materials` / `numsim-core`
/ `tmech` headers), e.g.:

```
g++-14 -std=c++23 -O2 -I <build>/tests/generated \
  -I <numsim-materials>/include -I <numsim-core>/include -I <tmech>/include \
  reproduce/plot_nl.cpp -o plot_nl && ./plot_nl > nl_data.csv
python3 reproduce/plot_nl.py     # -> j2_nonlinear_hardening.png
```
`emit_nl.cpp` shows how the Voce/Swift materials are emitted standalone via
`NumSimMaterialTarget` (the e2e generator does the same at build time).
