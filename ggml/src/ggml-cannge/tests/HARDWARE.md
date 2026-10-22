# CANNGE hardware validation

Status: partial 910B3 validation on CANN 9.1.0; see [910B3 results](910B3-RESULTS.md). SDK-free tests are not substitutes for these checks.
The [official API contract audit](API-AUDIT.md) records documentation evidence and unresolved assumptions separately from runtime validation.
Run on an Ascend host with its matching driver and CANN SDK. Do not download SDK images or build on the local Windows workstation. Preserve branch `cannge` and use separate build directories or source archives for comparisons; do not rewrite history or switch branches.

## Record the environment

Record the source commit, device model, driver and SDK versions, compiler, model file hash, command lines and all `GGML_CANNGE_*` settings. Keep logs outside the checkout. Source the installed SDK environment script before configuring. Set `CANN_INSTALL_DIR` to that installation and `SOC_TYPE` to the actual device model, not an assumed value.

## Build the real backend

From the repository root, with an external build directory selected in `build_dir`:

```sh
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANNGE=ON -DCANN_INSTALL_DIR="$CANN_INSTALL_DIR" -DSOC_TYPE="$SOC_TYPE" \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build "$build_dir" --parallel 4 --target test-backend-ops llama-bench
```

Require successful compilation and linking of every CANNGE translation unit. The standalone tests use fake ACL/GE/ES headers and cannot establish SDK compatibility.

## Correctness and support

Enable `GGML_CANNGE_GRAPH=1`. Select the exact CANNGE device name reported by the backend as `backend_name`; the test's `-b` filter is an exact match.

This revision's shared comparison helper ignores graph execution return codes. Build this directory's standalone test project on Linux, then set `LD_PRELOAD` to its `libcannge-status-guard.so` when running numerical tests. The guard exits with status 90 on a failed execution and prints the backend and graph nodes. A zero test exit without this guard is insufficient when stderr reports execution failures. The guard is for validation only; do not preload it for performance measurements.

```sh
export GGML_CANNGE_GRAPH=1
"$build_dir/bin/test-backend-ops" support -b "$backend_name" --output csv
"$build_dir/bin/test-backend-ops" test -b "$backend_name" \
  -o ADD,MUL,MUL_MAT,CPY,CONT,SET_ROWS,ROPE,SOFT_MAX --output csv
```

Check the actual executed and skipped case counts. A successful command with no matched device or no executed operations is not validation. Inspect rejected configurations separately from numerical failures. Add focused cases where the existing operator suite lacks coverage:

- F16/F32 dense and strided copies, transposed views, offset views and aliased outputs.
- Single-row and batched SET_ROWS, both conversion directions, partial cache views and repeated updates.
- Matrix multiplication, GEMV and GQA modes; verify the accepted policy against actual execution.
- CONT reconstruction, fresh-row splicing and repeated graph-cache hits across relocated tensors.
- Decode attention with supported head ratios and masks, with its opt-in enabled explicitly.

Run model-level prompt processing and repeated decode with CPU comparisons, including cache reuse and mixed CPU/CANNGE splits. Record the model and numerical comparison criteria before evaluating results. Operator tests alone do not prove model correctness.

## Recovery

Verify normal repeated backend creation/release and cache eviction on the real runtime. Use supported fault injection, if available, for stream drain, graph removal, allocation release and stream destruction failures. Confirm that retries preserve required resources, reclaim them after success, and do not execute partially released plans. Record unavailable fault injection as untested, not passed.

## Blocking-copy performance

Use the same device, SDK, model, thread count, offload settings and backend options for both versions. Compare the refactored version with the source immediately before `1aedf6d` using an isolated source archive, without moving the checkout. A baseline with incorrect outputs or incomplete asynchronous work is not a valid performance reference.

After warm-up, collect repeated prompt and decode samples, for example:

```sh
"$build_dir/bin/llama-bench" -m "$model_path" -ngl 99 -p 128,512 -n 32,128 -r 5 -o json
```

Confirm actual CANNGE execution and host-scatter use; CPU fallback or in-graph scatter can hide the blocking-copy cost. Record per-run latency/throughput, variation, tensor shapes, copied bytes and transfer counts. Use a device profiler to separate stream-drain and blocking-copy costs. Include F32-to-F16 SET_ROWS and temporary output snapshots. Disable debug dumps/timing instrumentation for end-to-end throughput measurements, and record profiling overhead separately.

## Completion evidence

Retain build logs, executed-case counts, numerical results, recovery observations and paired benchmark samples. Resolve failures before removing experimental gates or claiming the series is hardware-validated. The existence of this checklist does not satisfy any of these checks.
