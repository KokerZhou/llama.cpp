# Ascend 310P3 validation, 2026-10-23

Status: SDK build succeeds, but model execution fails. These results do not establish model correctness or performance.

## Environment and source

- Revision: `49e9873fb10001ea7c073b6f0a0ffaea0dc06a4e`, exported to `/workspace/cannge-validation-49e9873-310/source`.
- Build and logs: `/workspace/cannge-validation-49e9873-310`; the existing dirty checkout was not modified.
- Ascend310P3 with two devices exposed; tests select `CANNGE0`. Driver/npu-smi: `26.0.rc1`; CANN 9.1.0 in `/usr/local/Ascend/cann-9.1.0`.
- Build: Release, `GGML_CANNGE=ON`, `SOC_TYPE=Ascend310P3`. Runtime: `GGML_CANNGE_GRAPH=1`, disk cache disabled, optional paths unset.
- Model: Qwen3-4B-Instruct-2507-F16, 8,051,285,344 bytes, SHA-256 `336228a2810f17724ecd6e9e2e40e8381eee81e886316fee311ade2899c31030`.

## Confirmed results

- Real SDK compilation and linking of `test-backend-ops` and `llama-bench` succeeded.
- Standalone regressions passed 6/6.
- Strict MUL_MAT selection `m=16,n=.,k=256,` passed 36/36 supported cases. Unsupported configurations were skipped; this is not exhaustive matmul coverage.
- Strict CPY/CONT/SET_ROWS passed 132/132 supported cases.
- A fresh CPU model reference completed both prompt/decode rounds.
- The device model failed on the first prompt with llama decode status `-3`. CANN reported ACL error `507011` and an AI-core MTE on-chip write out of bounds, attributed to `BatchMatMulV3_13`.

## Reduced reproducer

`matmul-projection.txt` uses the existing `test-backend-ops --test-file` format. It contains F16 weights with ggml shape `[2560,4096,1,1]`, five F32 input rows, an all-F16 variant, and an all-F16 variant with 16 rows. The latter requires no M padding. All three failed on this environment with the same class of runtime error; the comparison reported infinite error. The standalone mixed-type case attributed the failure to `BatchMatMulV3_5`.

Run each case separately because the failed runtime can stall during shutdown:

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export GGML_CANNGE_GRAPH=1 GGML_CANNGE_GE_CACHE_DIR=
LD_PRELOAD="$metadata_build/libcannge-status-guard.so" \
  timeout -k 10 90 "$build_dir/bin/test-backend-ops" test -b CANNGE0 \
  --test-file "$source_dir/ggml/src/ggml-cannge/tests/matmul-projection.txt" \
  -p 'name=qwen-q-projection,' --output console
```

Select the other cases using `-p qwen-q-projection-f16` and `-p qwen-q-projection-aligned`. The numeric operation ID in this export format follows the tested revision's `ggml_op` enum; regenerate the fixture if that enum changes.

An isolated experiment reshaped rank-2 BMM operands to rank three with a batch dimension of one, then restored the output rank. It also failed. That patch was reverted and is not a proposed fix. The all-F16 and aligned-M failures show that mixed-type conversion and the new padding are not individually sufficient explanations. These observations do not yet establish whether the root cause is lowering, runtime configuration, or the installed kernel/tiling implementation.

## Failure handling and remaining work

The model propagated the execution error, but the failed process remained alive after SIGTERM. Failed operator tests also stalled while draining retired contexts. Only the identified test processes were terminated; no device reset or global configuration change was performed. This is a remaining real-runtime recovery concern despite passing stub lifecycle tests.

The validation guard was extended after these runs to catch `aclrtSynchronizeStream` failures, not only graph-compute return codes. Built separately as `strict-status-sync.so`, it allowed two small matmul cases to pass and terminated the failing mixed-type projection with status 90, without waiting for the stalled cleanup path. This is a diagnostic fix, not a recovery fix. Immediate exit can produce SDK subprocess/semaphore cleanup warnings; do not use this guard for recovery tests or benchmarks.

Relevant logs: `device.stderr`, `matmul-repro.log`, `matmul-f16.log`, `matmul-aligned.log`, and `matmul-rank3.log`. The original model process also produced `/root/ascend/log/debug/plog/plog-33892_20261023114550933.log`.

Resolve the reduced failure and shutdown behavior before model comparisons, optional-path validation, or paired blocking-copy benchmarks. Do not treat successful small-matrix tests as evidence that the large projection is supported safely.

## Direct GE isolation, 2026-10-24

The reusable 310 connection was re-established. The same SDK, source snapshot, and idle 310P3 devices were used. Disabling the backend's `ge.tiling_schedule_optimize` option did not resolve the projection failure. The experimental configuration patch was reverted.

`ge-matmul-probe.cpp` removes ggml entirely. It creates an ES graph containing two F16 inputs, BatchMatMulV3 with the right operand transposed, and an F32 output cast. It uses host `ge::Tensor` inputs with `Session::RunGraph`; there is no ggml planning, device-buffer binding, M padding, compile-cache key, or zero-copy reuse option. Both inputs contain ones, so every output must equal the contraction dimension K.

| Probe | Result |
| --- | --- |
| M=16, N=16, K=256 | All 256 outputs equal 256; removal and GE finalization succeed |
| M=16, N=4096, K=2560 | GE execution fails with the same MTE on-chip write-out-of-bounds error |
| Same large shape using MatMulV3 | Compilation/inference fails because MatMulV3 has no registered `infer_datatype` function in this installation |

The large BatchMatMulV3 probe reports the same kernel name as the backend's aligned-M failure: `te_batchmatmulv3_df42adbf9b253c82a5cbd6ddd09bee38ca5d73b4104b030e803c7c9af463ae1d__kernel0`. This isolates a failing GE/operator execution path independently of ggml. It does not distinguish a kernel implementation defect from a driver, runtime, or installation incompatibility. Do not disable GE inference validation to bypass the MatMulV3 error.

Reproduce with the installed CANN 9.1.0 headers and libraries (ABI setting matches this installation):

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sdk=/usr/local/Ascend/ascend-toolkit/latest
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  "$source_dir/ggml/src/ggml-cannge/tests/ge-matmul-probe.cpp" \
  -I "$sdk/include" -I "$sdk/include/es" -I "$sdk/include/ge" \
  -L "$sdk/lib64" -lascendcl -lge_runner -lgraph -lgraph_base \
  -les_math -les_nn -leager_style_graph_builder_base -lc_sec -o ge-matmul-probe
timeout -k 10 90 ./ge-matmul-probe bmm small
timeout -k 10 90 ./ge-matmul-probe
timeout -k 10 90 ./ge-matmul-probe matmul
```

The probe prints the failing phase and GE error, then exits 90 without cleanup on GE failure. It is a diagnostic reproducer, not production recovery code or a benchmark. Logs are `ge-probe-small.log`, `ge-probe-bmm-final.log`, `ge-probe-matmul.log`, and `no-tiling-repro.log` in the validation directory.
