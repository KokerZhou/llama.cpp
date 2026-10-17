# CANNGE backend internals

The layout follows the sibling CPU/CUDA backends: a backend entry point, shared infrastructure, a private context header, and operation source/header pairs. CANNGE additionally needs a compiled-graph execution layer because GE owns scheduling inside each graph.

## Responsibilities

| File | Responsibility |
| --- | --- |
| `ggml-cannge.cpp` | Backend registration, device discovery, buffers, transfers and events |
| `graph.cpp` | Context and plan lifetime, graph compilation, binding, execution and output completion |
| `supports.cpp` | Scheduling policy and per-operation capability predicates |
| `cpy.cpp` | Layout conversion, staging allocation and row scatter |
| `plan.cpp` | Per-call IO analysis, alias resolution and cache signatures |
| `ge.cpp` | GE initialization, sessions and graph lifecycle |
| `common.cpp` / `common.h` | Shared ACL diagnostics, constants and timing |
| `ops.cpp` / `ops.h` | Operation dispatch and general ES mappings |
| `graph-build.cpp` | Assemble graph inputs, nodes and output ports |
| `mul-mat.cpp` | Matrix multiplication, GEMV and associated cache-row splicing |
| `fattn.cpp` | Flash-attention mapping |
| `graph-build-io.cpp` | Graph inputs, views and axis relabeling |

Shared primitives belong in `common.h`; execution context ownership belongs in `ggml-cannge-impl.h`. Consumers include the relevant module interface (`cpy.h`, `supports.h`, `graph.h`, `mul-mat.h`, or `fattn.h`) instead of a catch-all private header. ES builder internals remain in `graph-build-priv.h`.

The dependency flow is `ggml-cannge.cpp -> graph.cpp -> graph-build.cpp -> ops.cpp -> operation modules`. `plan.cpp` supplies per-call analysis and `ge.cpp` owns the GE session. When adding an operation, declare its interface in its own header, include that header in `ops.cpp`, and register the files in this directory's CMake target.

## Support decisions and configuration

`ggml_backend_cannge_device_supports_op` applies these checks in order:

1. The graph path must be enabled by `GGML_CANNGE_GRAPH` (first character `1`, read once).
2. Leaf and metadata-only operations are accepted.
3. `ggml_cannge_disabled_by` applies compute-disable overrides and the flash-attention opt-in.
4. Alias restrictions are checked.
5. A named operation predicate checks the supported types, shapes and parameters.

The existing environment-variable behavior is preserved. Most disable flags test presence: setting one to `0` still disables the operation. `GGML_CANNGE_FA` also tests presence. `GGML_CANNGE_GQA` accepts `kq`, `v` or `all` in the capability check. Shape-specific workarounds remain beside the relevant predicate.

Use `GGML_CANNGE_DEBUG_COMPILE` to see policy rejections and compilation diagnostics, `GGML_CANNGE_DEBUG_BUILD` for graph mapping, `GGML_CANNGE_DEBUG_IO` for transfer checksums, and `GGML_CANNGE_DEBUG_TIMING` for timing totals. These diagnostics can synchronize or copy device data and are unsuitable for performance measurements.

When adding a graph-affecting option, update the plan signature as well as the builder. Do not change configuration concurrently with execution.

## Execution stages

`ggml_backend_cannge_graph_compute` coordinates capability checks and IO analysis, then calls separate stages for plan preparation, input binding, output binding, index snapshots, GE execution and output completion. Verbose IO diagnostics are separate from this orchestration.

Plan preparation includes cache lookup, compilation and loading. Timing totals therefore include loading in `lookup`, rather than `bind`. Input-binding timestamps retain the distinction between staging and its synchronization.

Mask and index snapshots are implemented in `cpy.cpp`. They check synchronization and copy results before consuming device data. Index snapshots preserve tensor strides and widen I32 values to I64 only after a successful read; attention-mask reads use the first row's element stride. Failed snapshots stop the graph before GE execution.

## Ownership and failure handling

Backend initialization rejects failed device selection, failed stream creation and null streams. Execution also checks device selection before preparing a graph.

The execution context owns the GE session and cached plans. Plans own staging allocations; per-call IO analysis holds current tensor pointers. Reusable staging assumes that weight contents remain immutable at a given source address.

`ggml_cannge_check(error, operation, function)` logs the failed operation and returns a boolean. Callers must propagate that result when their interface supports failure; the function does not throw or return from the caller. Cleanup and void backend callbacks still have separate error-handling limitations.

`ggml_cannge_finish_execution` owns temporary output snapshots through copy-back and scatter completion. Transfer failure paths invalidate reusable staging and attempt to drain queued copies before returning. A failed stream synchronization is not proof that queued work completed; device-error recovery needs real-runtime validation.

## Validation boundaries

SDK-free tests can check support decisions, allocation failure handling, layout copies and deferred-copy lifetimes. Stub compilation checks C++ structure only. A real CANN SDK build, device correctness tests and performance measurements are required before treating changes as hardware-validated.
