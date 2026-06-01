# numsim-codegen Workflow

Three views of the same system: the high-level user workflow (activity), the pass-pipeline runtime flow (sequence), and the static class structure. Reflects the codebase after PRs #51–#55 land (i.e. with the Phase-1.2 pass framework + the M2–M6 follow-ups).

---

## 1. High-level workflow (activity)

User-facing flow from recipe construction to generated source on disk.

```mermaid
flowchart TD
    A[User builds ConstitutiveModel] --> B[add_scalar_input<br/>add_tensor_input<br/>add_parameter]
    B --> C[add_output with semantic Role]
    C --> D[Pick a Target<br/>StandaloneCxx / MooseMaterial / ...]
    D --> E[Target.emit&#40;model&#41;]
    E --> F[Target calls<br/>model.emit_compute_function&#40;&#41;]
    F --> G[Pass pipeline runs<br/>see Sequence below]
    G --> H[Target wraps result<br/>with framework framing<br/>&#40;headers, class decl, ...&#41;]
    H --> I[User writes generated<br/>files to disk]

    style F fill:#e1f5e1
    style G fill:#fff4d6
```

The CAS expression DAG (numsim-cas) lives "behind" the recipe — `add_output(name, expr, role)` accepts an `expression_holder<scalar|tensor>` built via cas operators. The DAG is immutable through this entire flow; passes consume it and emit code referencing its leaves.

---

## 2. Pass pipeline (sequence)

What happens when `model.emit_compute_function()` is called. This is the core of the system.

```mermaid
sequenceDiagram
    actor User
    participant Model as ConstitutiveModel
    participant PM as PassManager
    participant SVP as SymbolValidationPass
    participant TSP as TensorSpaceConsistencyPass
    participant CEP as CodeEmitPass
    participant SE as ScalarCodeEmit
    participant TE as TensorCodeEmit
    participant T2S as TensorToScalarCodeEmit
    participant Render as render_compute_function

    User->>Model: emit_compute_function()
    Note over Model: Construct PassContext<br/>{RecipeView{*this},<br/> CodeGenContext{},<br/> nullopt}
    Model->>PM: emplace<SymbolValidationPass>()
    Model->>PM: emplace<TensorSpaceConsistencyPass>()
    Model->>PM: emplace<CodeEmitPass>()
    Model->>PM: run(pctx)

    PM->>SVP: run(pctx)
    SVP->>SVP: walk symbols, check identifier validity<br/>(ASCII range + C++ keyword reject)
    SVP->>SVP: walk outputs, check no undeclared symbols
    SVP-->>PM: postconditions:<br/>{symbols-declared, identifiers-valid}

    PM->>TSP: run(pctx)
    TSP->>TSP: cross-check Role.is_symmetric vs<br/>tensor_space.perm (Skew = conflict)
    TSP->>TSP: cross-check Role.expected_rank vs<br/>tensor.rank()
    TSP-->>PM: postconditions:<br/>{tensor-space-validated}

    PM->>CEP: run(pctx)
    Note over CEP: Cycle-break via<br/>unique_ptr<T2sCodeEmit> indirection
    CEP->>SE: construct(ctx)
    CEP->>TE: construct(ctx, scalar_emit, t2s_callback)
    CEP->>T2S: construct(ctx, scalar_emit, tensor_emit)
    CEP->>CEP: register symbol names with CodeGenContext
    loop For each output expr
        CEP->>SE: apply(expr) — scalar path
        CEP->>TE: apply(expr) — tensor path
        Note over TE,T2S: tensor↔t2s nodes use the<br/>std::function callback bridge
    end
    CEP->>Render: render_compute_function(view, ctx, output_rhs)
    Render-->>CEP: returns full function source
    CEP-->>PM: postcondition:<br/>{compute-function-emitted}
    Note over PM: precondition check throws<br/>if any pass declares an unmet<br/>dependency before its turn

    PM-->>Model: pctx.compute_function_source populated
    Model-->>User: returns std::string
```

**Key invariants:**
- Each pass advertises **preconditions** (tags it needs satisfied first) and **postconditions** (tags it satisfies on success). `PassManager::run` enforces these in registration order — registering a pass with unmet preconditions throws *before* the pass runs.
- The **`CodeEmitPass` cycle-break** uses `std::unique_ptr<TensorToScalarCodeEmit>` indirection because `TensorCodeEmit` requires a `T2sApply` callback at construction (M3) but `TensorToScalarCodeEmit` holds a `TensorCodeEmit&` — a real cycle. The lambda captures the storage pointer; it's only invoked AFTER `make_unique` populates it.
- The **CodeGenContext** accumulates `auto tN = ...;` lines via pointer-keyed CSE. Two references to the same `expression_holder` DAG node emit one temp, not two.

---

## 3. Static structure (class)

The abstractions and their relationships. Trimmed to the spine — accessors and field details omitted.

```mermaid
classDiagram
    direction LR

    class ConstitutiveModel {
        +name() string
        +symbols() vector~SymbolDecl~
        +outputs() vector~OutputDecl~
        +scalar_symbol_map()
        +tensor_symbol_map()
        +add_*_input(...)
        +add_parameter(...)
        +add_output(...)
        +validate()
        +emit_compute_function() string
        -m_symbols
        -m_outputs
    }

    class RecipeView {
        +name() string
        +symbols()
        +outputs()
        +scalar_symbol_map()
        +tensor_symbol_map()
        +raw_model() ConstitutiveModel
        -m_model : ConstitutiveModel*
    }

    class PassContext {
        +model : RecipeView
        +ctx : CodeGenContext
        +compute_function_source : optional~string~
    }

    class Pass {
        <<abstract>>
        +name() string_view
        +preconditions() vector~string_view~
        +postconditions() vector~string_view~
        +run(PassContext&)
    }

    class PassManager {
        +add(unique_ptr~Pass~)
        +emplace~P~(...)
        +run(PassContext&) const
        -m_passes : vector~unique_ptr~Pass~~
    }

    class SymbolValidationPass {
        +run(PassContext&)
        +is_valid_cxx_identifier(string)$
        +is_cxx_keyword(string_view)$
    }
    class TensorSpaceConsistencyPass {
        +run(PassContext&)
    }
    class CodeEmitPass {
        +run(PassContext&)
    }

    class CodeGenContext {
        +emit_temporary(void*, string)
        +find(void*)
        +render_statements() string
        -m_statements
        -m_cse_table
    }

    class ScalarCodeEmit {
        +apply(scalar_expression) string
    }
    class TensorCodeEmit {
        +apply(tensor_expression) string
        -m_t2s_apply : T2sApply
    }
    class TensorToScalarCodeEmit {
        +apply(t2s_expression) string
        -m_tensor : TensorCodeEmit&
    }

    class Target {
        <<abstract>>
        +emit(model) vector~GeneratedFile~
    }
    class StandaloneCxxTarget
    class MooseMaterialTarget

    ConstitutiveModel ..> PassContext : constructs
    ConstitutiveModel ..> PassManager : constructs
    PassContext *-- RecipeView
    PassContext *-- CodeGenContext
    RecipeView --> ConstitutiveModel : views (const ptr)
    PassManager o-- Pass : owns N

    Pass <|-- SymbolValidationPass
    Pass <|-- TensorSpaceConsistencyPass
    Pass <|-- CodeEmitPass

    CodeEmitPass ..> ScalarCodeEmit : constructs
    CodeEmitPass ..> TensorCodeEmit : constructs
    CodeEmitPass ..> TensorToScalarCodeEmit : constructs
    TensorCodeEmit ..> TensorToScalarCodeEmit : callback<br/>(unique_ptr indirection)
    TensorToScalarCodeEmit --> TensorCodeEmit : ref

    Target <|-- StandaloneCxxTarget
    Target <|-- MooseMaterialTarget
    Target ..> ConstitutiveModel : reads<br/>+ emits via
```

---

## 4. What's outside the framework

Three layers wrap or feed the pipeline above:

- **numsim-cas** (external dependency, pinned via CPM): the symbolic expression DAG. Tensor / scalar / tensor-to-scalar expression types, the `expression_holder<T>` shared_ptr handle, the simplifier, the substitute/diff machinery. The recipe holds *handles*; the DAG itself is immutable through codegen.
- **tmech** (system include): the generated source's tensor type. Expression-template-backed; provides `tmech::adaptor<...>` for zero-copy bridging at the FEM-framework boundary (MOOSE `RealTensorValue`, deal.II `Tensor<rank, dim>`).
- **Backends** (`include/numsim_codegen/targets/`): wrap `emit_compute_function()` output with framework-specific framing. `StandaloneCxxTarget` outputs a single inline header. `MooseMaterialTarget` outputs a `.h` + `.C` pair implementing a MOOSE `Material` subclass. Future targets (`AbaqusUMATTarget`, `AnsysUSERMATTarget`) follow the same pattern.

---

## 5. Phase 2 / 3 extension points

Where future work plugs in (per epic #28 and follow-up issue #56):

| Phase | Addition | Insertion point |
|---|---|---|
| **2** | `TimeIntegrationPass` lowering `Dt(α) → (α_new − α_old)/dt` via cas substitute | Between SymbolValidationPass and CodeEmitPass; advertises `dt-lowered` postcondition |
| **2** | `KuhnTuckerLoweringPass` rewriting NCP constraints to Fischer-Burmeister | Between SymbolValidationPass and CodeEmitPass; advertises `kt-lowered` |
| **2** | `LocalNewtonLoweringPass` emitting the Newton iteration body | Between lowering passes and CodeEmitPass |
| **2** | Mutable `RecipeView` surface for the above rewrites | See #56 / P1 — sibling typed view or `variant` storage |
| **3** | `AlgorithmicTangentPass` (consistent tangent via implicit diff) | After Newton lowering, before CodeEmitPass |
| **3** | `TangentEmitPass` / `MoosePropertyEmitPass` (additional emit targets) | After CodeEmitPass or replacing it depending on target |
| **3** | `CodeEmitPipeline` aggregate replacing the unique_ptr cycle break | See #56 / P2 |

The pass framework's value proposition is exactly this: new behaviour ships as a new `Pass` subclass + a one-line `pm.emplace<...>()` registration in the pipeline. The existing passes don't need to change.
