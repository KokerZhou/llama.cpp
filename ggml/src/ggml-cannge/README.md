# CANNGE backend internals

The layout follows the sibling CPU/CUDA backends: a backend entry point, shared infrastructure, a private context header, and operation source/header pairs. CANNGE additionally needs a compiled-graph execution layer because GE owns scheduling inside each graph.

## Responsibilities

| File | Responsibility |
| --- | --- |
| `ggml-cannge.cpp` | Backend registration, device discovery, buffers, transfers and events |
| `graph.cpp` | Graph compilation, binding, execution and output completion |
| `context.cpp` | Context and plan lifetime, cache eviction |
| `supports.cpp` | Scheduling policy and per-operation capability predicates |
| `options.cpp` / `options.h` | Parsed graph configuration and its cache signature |
| `cpy.cpp` | Layout conversion, staging allocation and row scatter |
| `plan.cpp` | Per-call IO analysis |
| `plan-key.cpp` | Address-independent graph and binding signatures |
| `view.cpp` / `view.h` | Shared view metadata, cycle detection and registered-base selection |
| `ge.cpp` | GE initialization, sessions and graph lifecycle |
| `common.cpp` / `common.h` | Shared ACL diagnostics, constants and timing |
| `ops.cpp` / `ops.h` | Operation dispatch |
| `elementwise.cpp` | Binary arithmetic, scaling, unary activations and GLU |
| `norm.cpp` | Softmax, RMS normalization and layer normalization |
| `tensor.cpp` | Views, copies, row access and concatenation |
| `rope.cpp` | Rotary position embedding |
| `graph-build.cpp` | Assemble graph inputs, nodes and output ports |
| `mul-mat.cpp` | Matrix multiplication, GEMV and associated cache-row splicing |
| `fattn.cpp` | Flash-attention mapping |
| `graph-build-io.cpp` | Graph inputs and fresh-row splicing |
| `graph-build-view.cpp` | Tensor lookup, views, axis relabeling and checked prefix slices |

Shared primitives belong in `common.h`; execution context ownership belongs in `ggml-cannge-impl.h`. Consumers include the relevant module interface (`cpy.h`, `supports.h`, `graph.h`, `mul-mat.h`, or `fattn.h`) instead of a catch-all private header. ES builder internals remain in `graph-build-priv.h`.

The dependency flow is `ggml-cannge.cpp -> graph.cpp -> graph-build.cpp -> ops.cpp -> operation modules`. `plan.cpp` supplies per-call analysis and `ge.cpp` owns the GE session. When adding an operation, declare its interface in its own header, include that header in `ops.cpp`, and register the files in this directory's CMake target.

## Support decisions and configuration

`ggml_backend_cannge_device_supports_op` applies these checks in order:

1. The graph path must be enabled by `GGML_CANNGE_GRAPH` (first character `1`, read once).
2. Leaf and metadata-only operations are accepted.
3. `ggml_cannge_disabled_by` applies compute-disable overrides and the flash-attention opt-in.
4. Alias restrictions are checked.
5. A named operation predicate checks the supported types, shapes and parameters.

Most existing environment-variable behavior is preserved. Most disable flags test presence: setting one to `0` still disables the operation. `GGML_CANNGE_FA` also tests presence. `GGML_CANNGE_GQA` accepts `kq`, `v` or `all` in the capability check. Shape-specific workarounds remain beside the relevant predicate. The optional reason output of `ggml_cannge_supports_op` identifies the disabling option or failed constraint category; it is null on success. Scheduler diagnostics log rejections under `GGML_CANNGE_DEBUG_COMPILE`, while execution pre-check failures always report the tensor name, operation and reason.

For non-contiguous CONT, `GGML_CANNGE_NO_CONT_NC` overrides `GGML_CANNGE_CONT_NC`. Scheduling, merged-cache planning and graph construction share this effective policy. Dense CONT remains available with either flag, but all CONT operations must preserve element count. The merged-attention opt-in still requires device validation; consistent policy does not resolve its historical model-scale numerical issues.

Use `GGML_CANNGE_DEBUG_COMPILE` to see policy rejections and compilation diagnostics, `GGML_CANNGE_DEBUG_BUILD` for graph mapping, `GGML_CANNGE_DEBUG_IO` for transfer checksums, and `GGML_CANNGE_DEBUG_TIMING` for timing totals. These diagnostics can synchronize or copy device data and are unsuitable for performance measurements.

Graph execution captures `ggml_cannge_options` once and passes it to capability checks and IO analysis. The analyzed IO owns that snapshot; builders and the plan signature read it without rereading the environment. Scheduler capability callbacks capture their own current options. Parsed values determine cache identity, so equivalent spellings of a presence flag or integer share a key.

When adding a graph-affecting option, add its field, environment parsing and signature serialization in `options.h` / `options.cpp`, then consume the field in the predicate or builder. Presence and value remain separate for GQA and dump filters. Matrix-multiplication scheduling and construction share `ggml_cannge_use_gemv`, including head-ratio checks and GQA mode validation. The existing KQ/V classification by contraction width remains a hardware-policy assumption requiring numerical validation. Diagnostic logging and `GGML_CANNGE_NO_WONCE` remain live controls; `GGML_CANNGE_GRAPH` remains a startup switch. Do not change configuration concurrently with execution.

## Plan analysis

The plan analyzer separates graph indexing, external inputs, boundary outputs, side effects, output-port ordering, layout resolution and memory estimation. The coordinator keeps this order explicit; producer and consumer maps are local to one analysis. The separate debug stage does not change planning decisions. Analysis replaces the IO result only after all stages succeed; failure preserves the previous result and identifies the invalid input, output or side-effect view. Reusing an IO object does not accumulate ports or memory estimates. Cache signatures include external view/CONT dependencies, their operation parameters, and resolved input-base bindings. Tensor IDs follow traversal order, so relocating the same graph preserves its key.

## Execution stages

`ggml_backend_cannge_graph_compute` coordinates capability checks and IO analysis, then calls separate stages for plan preparation, input binding, output binding, index snapshots, GE execution and output completion. Verbose IO diagnostics are separate from this orchestration. Operation builders that return failure report the node index, tensor name and operation, even when debug logging is disabled. Missing source tensors return a build error. Tensor lookup and CONT fallback validate view chains before walking them, and share a prefix-slice helper with input planning. The metadata rule checks positive element counts, overflow, alignment and that the prefix fits within its base. Exact-size views keep their holder; oversized views fail construction.

Plan preparation includes cache lookup, compilation and loading. Timing totals therefore include loading in `lookup`, rather than `bind`. Input-binding timestamps retain the distinction between staging and its synchronization.

Mask and index snapshots are implemented in `cpy.cpp`. They check synchronization and copy results before consuming device data. Index snapshots preserve tensor strides and widen I32 values to I64 only after a successful read; attention-mask reads use the first row's element stride. Failed snapshots stop the graph before GE execution.

CONT-chain construction distinguishes success, an inapplicable optimization, and failure. Callers may use a plain alias only for the inapplicable case; invalid relabels and splice failures stop construction. The helper publishes its output holder only on success and preserves the caller's diagnostic when an optional lookup does not apply. This does not roll back ES nodes already created during an attempt.

## Ownership and failure handling

Backend initialization rejects failed device selection, failed stream creation and null streams. Execution also checks device selection before preparing a graph.

The execution context owns the GE session and cached plans. Plans own staging allocations; per-call IO analysis holds current tensor pointers. Reusable staging assumes that weight contents remain immutable at a given source address.

Plans track unconfirmed input-arena uploads. After a failed drain, even same-capacity arena reuse must synchronize successfully before host contents can be overwritten. Normal reuse after a completed upload does not add a drain.

Eviction retains a plan if stream synchronization or graph removal fails, so a later attempt can retry. A cache miss makes room before building its replacement; failed eviction stops preparation without adding another GE graph. Removing a graph already released by compile rollback succeeds without calling GE again. Allocation-release failures retain the plan for retry after graph removal; successful releases clear their pointers, and partially released plans are marked failed so they cannot execute. Shutdown also retains failed plan allocations before destroying the stream.

If GEFinalize fails, its status and diagnostic are logged and retained. Later CANNGE session creation fails without reinitializing GE; restart the process to use CANNGE again. This contains partially completed global teardown and does not claim recovery. Graph removal can also fail after internal mutation; see the [source audit](tests/GE-SOURCE-AUDIT.md) for the remaining recovery limitations.

Backend release selects the context's device and drains its stream before destruction. Failed contexts remain on a mutex-protected retired list, retaining their session and staging buffers. Later backend releases retry them. After a successful drain, session and buffer cleanup precedes stream destruction; a failed stream destruction retains the context and stream handle for another release attempt. If no later drain succeeds, these resources remain allocated until process exit; this path does not reset the device or claim recovery from a permanent device failure. Context destruction is private so callers cannot bypass this release path.

`ggml_cannge_check(error, operation, function)` logs the failed operation and returns a boolean. Callers must propagate that result when their interface supports failure; the function does not throw or return from the caller. Cleanup and void backend callbacks still have separate error-handling limitations.

Temporary output snapshots and converted SET_ROWS rows use blocking transfers. Device snapshots drain the producer stream before reading; device scatter drains it before uploading rows. No asynchronous transfer retains these temporary host buffers after a failure. This changes host-scatter scheduling and needs device performance measurements. Transfers between persistent tensor/staging buffers remain asynchronous; failure paths invalidate reusable staging and attempt to drain queued copies before returning. A failed stream synchronization is not proof that queued work completed; device-error recovery and deferred context release still need real-runtime validation.

## Validation boundaries

The pending real-device procedure is in [tests/HARDWARE.md](tests/HARDWARE.md).

SDK-free tests can check support decisions, allocation failure handling, layout copies and deferred-copy lifetimes. Stub compilation checks C++ structure only. A real CANN SDK build, device correctness tests and performance measurements are required before treating changes as hardware-validated.


The persistent regression targets in `tests/` build without CANN or GE. Metadata tests compile the production view, plan and options modules with small ggml metadata stubs. A separate capability target checks RoPE, matmul, GEMV selection and attention constraints; plan tests retain checks that compare scheduling policy with plan analysis. Lifecycle tests compile production context and GE wrapper code against fault-injecting ACL/GE stubs, checking eviction retention, removal retries, compile rollback and normal teardown ordering. Copy tests compile production copy code and check snapshot/scatter ordering, failure propagation and F32-to-F16 row conversion without queuing temporary buffers. Builder lookup tests compile the production view, IO and tensor builders against ES holder stubs. They distinguish optional absence from reconstruction failure and check that CONT, fresh-row splicing and scatter materialization propagate errors. These tests do not validate SDK compatibility or device execution. From the repository root:

```sh
cmake -S ggml/src/ggml-cannge/tests -B build-cannge-metadata
cmake --build build-cannge-metadata --config Release
ctest --test-dir build-cannge-metadata -C Release --output-on-failure
```
