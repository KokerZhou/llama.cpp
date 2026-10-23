# CANNGE official API contract audit

The subsequent [official GE source audit](GE-SOURCE-AUDIT.md) identifies partial-removal and finalization failure behavior that the API documentation alone did not establish.

Reviewed 2026-10-18 against source at dc6d218, with the device-selection comment corrected afterward. This is a focused lifecycle and transfer contract review, not an SDK build, full operator audit, numerical test or performance result. Documentation versions below differ; recheck against the actual installed SDK and device before deployment.

| Area | Source assessment | Official evidence |
| --- | --- | --- |
| Graph ownership | `ge.cpp` uses AddGraphWithCopy; GE keeps its own graph copy. | [CANN 9.1 AddGraphWithCopy](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendgraphapi/atlasgeapi_07_0092.html) |
| Execution sequence | `add_and_compile`, `load`, and `execute` follow the required compile/load/execute sequence. `graph.cpp` passes the context stream to both load and execution. | [CANN 8.5 LoadGraph](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0094.html) |
| Inputs and outputs | `graph.cpp` binds preallocated device buffers. Execution success means asynchronous submission; CPU access must follow successful synchronization. Snapshot helpers do this. This does not certify every diagnostic path. | [CANN 8.3.RC1 ExecuteGraphWithStreamAsync](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendgraphapi/atlasgeapi_07_0099.html) |
| Device selection | `ge.session_device_id` is documented for per-session device selection. The old comment claiming the global default pins all sessions to the first device was misleading. Multi-device execution is still untested. | [CANN 8.1.RC1 multi-device guide](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/graph/graphdevg/atlasag_25_0075.html) |
| Buffer release | `context.cpp` drains the stream before eviction or shutdown releases plan buffers. This is necessary because aclrtFree does not synchronize. | [CANN 8.1.RC1.alpha002 aclrtFree](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0100.html) |
| Stream destruction | Shutdown selects the device and requires successful stream synchronization before destruction. The API also requires the correct current context and no concurrent users; device selection assumes the default-context model. | [CANN 9.1 aclrtDestroyStream](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0070.html) |
| Temporary copy buffers | Snapshot and conversion helpers use synchronous aclrtMemcpy before destroying or reusing temporary host storage. The guide distinguishes synchronous copies from asynchronous copies requiring synchronization. It permits ordinary host allocations. | [CANN 8.2.RC1.alpha002 data transfers](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha002/appdevg/acldevg/aclcppdevg_000010.html) |

## Unresolved issues and limits

- Product support is API-specific. The cited 8.3.RC1 ExecuteGraphWithStreamAsync table excludes Atlas 200I/500 A2 inference products. An available Ascend device alone does not establish compatibility; match the exact product and SDK support tables.
- Failed-release retry semantics remain unverified. The cited free/destroy pages describe success and failure, but do not guarantee that every failure leaves a live, reusable handle or allocation. Stub tests prove our bookkeeping under their injected semantics, not recovery from arbitrary runtime faults. Validate supported error classes before claiming shutdown recovery complete.
- GEFinalize has a Status signature, but its [8.3.RC1 reference](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendgraphapi/atlasgeapi_07_0087.html) supplies no return-value or recovery contract. Following the source audit, the wrapper now logs and preserves finalization failure and rejects subsequent session creation until process restart. This contains failure; it does not recover partially torn-down runtime state.
- Debug output reads in `graph.cpp` still ignore some copy errors and a synchronization result. Diagnostic output after those failures cannot be trusted. This audit does not mark that path corrected.
- This review does not establish ES builder signature compatibility, numerical correctness, all host-buffer asynchronous-copy constraints, or correctness with custom ACL contexts.
- Blocking-copy latency and throughput require paired device measurements. Documentation provides no performance result for this implementation. Follow [HARDWARE.md](HARDWARE.md); no local SDK downloads or hardware-free SDK builds are needed for this audit.

## Tiling and matmul follow-up, 2026-10-24

The local official GE clone at `3f1fd9f7ea4208e3e88255d0c5c55fed8097d845` documents `ge.tiling_schedule_optimize` as disabled by default in `docs/en/design/features/tiling_sink.md`; its feature table lists A2/A3 products. `api/atc/atc_flags.cc` also defines the default as `0`. This source revision is separate from the installed CANN 9.1.0 binaries, so the table alone does not prove installed-runtime behavior. The backend currently enables the option unconditionally. Disabling it experimentally on 310P3 did not fix the large-matmul failure; no claim that it causes that failure is justified.

The installed ES headers expose both BatchMatMulV3 and MatMulV3. Their presence does not establish executable operator support: the direct GE MatMulV3 probe failed on a missing registered datatype-inference function. The BatchMatMulV3 probe compiled but failed at execution for the large projection, while a small control passed. See [310P3 results](310P3-RESULTS.md) for the independent reproducer, exact shapes, and limits of these conclusions.
