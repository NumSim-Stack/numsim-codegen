# #92 increment: tensor internal-variable history → path-dependent J2

Turn the single-increment Mode-B residual material into a real path-dependent one:
the state carried across steps includes a **tensor** internal variable (plastic
strain εᵖ) updated by a post-solve formula. Scalar solve unchanged — this is
storage, not a coupled solve (no Mandel, no vector solver).

## State model (the new distinction)

Two kinds of state variable in a Mode-B residual material:
- **Newton unknown** — has a `residual` (solved by backward_euler). Already carried
  as history (`m_<x>` = `add_history_output`). For J2 this is **α** (accumulated
  equivalent plastic strain); the per-step increment is `Δγ = α.current − α.previous`.
- **Internal variable** — has an **update equation** (set post-solve by a formula,
  NOT solved). For J2 this is **εᵖ** (tensor), `εᵖ.new = εᵖ.old + Δγ·n`.

Path-dependent J2 (deviatoric, linear hardening), with `εᵖ_old` history:
```
s_trial = 2G·dev(ε − εᵖ.previous)          n = s_trial/‖s_trial‖
R(α)    = ‖s_trial‖ − 2G·(α.current − α.previous) − σ_y − H·α.current
εᵖ.new  = εᵖ.previous + (α.current − α.previous)·n         (update equation)
σ       = s_trial − 2G·(α.current − α.previous)·n
```

## New recipe API
- `add_tensor_state_variable` — **exists** (returns `{current, previous}`).
- **NEW** `add_update_equation(handle, expr)` (scalar + tensor overloads): records
  that `handle` is an internal variable set to `expr` post-solve. Stored in
  `m_update_equations` (parallel to `m_residual_equations`).

## Emitter changes (emit_residual_material)
1. **Bind `_old` for every state/internal variable** as a local in the eval lambda
   AND post-solve, register its symbol → local. (Relaxes the current residual-leaf
   guard, which rejected `_old` because it was unbound — now it's bound.)
2. **Emit `add_history_output<tensor>`** for tensor internal variables (εᵖ) — the
   capability numsim-core#16's `history_property<tensor>` fix unlocked. Scalar
   internal vars (if any) similarly.
3. **Post-solve updates**: after the solve + converged guard, emit
   `m_<iv>.new_value() = <rendered update_expr>` for each internal variable, in a
   brace-scoped block (CSE isolation).
4. Members/ctor for the internal-variable histories; uniqueness/reserved-name
   guards extended to internal-variable names.

## Golden (path-dependent, multi-step)
`J2PathDependent` recipe. Drive several load steps (accumulating εᵖ), then verify:
- **physics**: `‖dev σ‖ = σ_y + H·α` on the yield surface each step;
- **path dependence**: εᵖ accumulates (α strictly increasing under load); unload a
  step and check εᵖ is retained (residual plastic strain) — the thing the
  single-increment golden can't show;
- **tangent**: FD-match dσ/dε through the real re-solve at each step (now with εᵖ
  history reverted per FD sample, like the checker already does).

## Guards / caveats
- Elastic branch still rides the `max(x,0)` clamp (KKT switch = #35). Keep loading
  monotonic-plastic for the FD sweep; the unload check is a separate value read.
- Leaf guard: `_old` now allowed for declared state/internal vars (bound); still
  reject undeclared symbols and non-input tensor leaves.
