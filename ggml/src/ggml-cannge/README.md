# CANNGE backend internals

## Responsibilities

| File | Responsibility |
| --- | --- |
| `ggml-cannge.cpp` | Backend registration, device discovery, buffers, transfers and events |
| `ggml-cannge-runtime.cpp` | Context and plan lifetime, graph compilation, binding, execution and output completion |
| `ggml-cannge-gates.cpp` | Scheduling policy and per-operation capability predicates |
| `ggml-cannge-io.cpp` | Layout conversion, staging allocation and row scatter |
| `plan.cpp` | Per-call IO analysis, alias resolution and cache signatures |
| `ge.cpp` | GE initialization, sessions and graph lifecycle |
| `graph-build.cpp` | Operation dispatch and general ES mappings |
| `graph-build-matmul.cpp` | Matrix multiplication, GEMV and associated cache-row splicing |
| `graph-build-attention.cpp` | Flash-attention mapping |
| `graph-build-io.cpp` | Graph inputs, views and axis relabeling |

Backend-private declarations belong in `ggml-cannge-priv.h`. Execution context ownership belongs in `ggml-cannge-runtime.h`; ES builder internals belong in `graph-build-priv.h`.

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

## Ownership and failure handling

The execution context owns the GE session and cached plans. Plans own staging allocations; per-call IO analysis holds current tensor pointers. Reusable staging assumes that weight contents remain immutable at a given source address.

`ggml_cannge_check(error, operation, function)` logs the failed operation and returns a boolean. Callers must propagate that result when their interface supports failure; the function does not throw or return from the caller. Cleanup and void backend callbacks still have separate error-handling limitations.

`ggml_cannge_finish_execution` owns temporary output snapshots through copy-back and scatter completion. Transfer failure paths invalidate reusable staging and attempt to drain queued copies before returning. A failed stream synchronization is not proof that queued work completed; device-error recovery needs real-runtime validation.

## Validation boundaries

SDK-free tests can check support decisions, allocation failure handling, layout copies and deferred-copy lifetimes. Stub compilation checks C++ structure only. A real CANN SDK build, device correctness tests and performance measurements are required before treating changes as hardware-validated.
