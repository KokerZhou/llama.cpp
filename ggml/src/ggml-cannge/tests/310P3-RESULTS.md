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

### Format investigation

On 2026-10-24 the unchanged large probe was run with `DUMP_GE_GRAPH=2`, `DUMP_GRAPH_LEVEL=3`, and `DUMP_GRAPH_PATH` pointing to a separate validation artifact directory, under the same 90-second timeout. It again exited 90. The trace lists TransData (task 2), the same failing BatchMatMulV3 kernel (task 3), and Cast (task 4), each with eight blocks. The initial inspection incorrectly reported that only a task trace was emitted: it listed only the top-level directory and missed the graph files in `pid_18752_deviceid_0`. Recursive inspection recovered `ge_proto_00000000_graph_1_Build.txt` and its ONNX counterpart.

Artifacts: `ge-probe-format-dump.log` and the recursive `ge-probe-format-dump` directory in the remote validation directory. Local copies are in `cannge-validation-310-49e9873`, with an additional JSON copy named `ge-probe-final-graph.json` (despite that filename, its content is a task trace). No probe process remained after this run.

The unchanged small probe was also rerun with level 1, `DUMP_GRAPH_FORMAT=ge_proto`, and info logging. It passed all 256 outputs and finalized successfully. Its final graph is `ge-probe-small-dump/pid_22470_deviceid_0/ge_proto_00000087_graph_1_Build.txt`, copied locally as `ge-probe-small-build.txt`.

| BatchMatMulV3 descriptor | Passing control | Failing projection |
| --- | --- | --- |
| Input A, FP16 ND | `[16,256]` | `[16,2560]` |
| Input B, FP16 FRACTAL_NZ | `[16,1,16,16]` | `[160,256,16,16]` |
| B original ND shape | `[16,256]` | `[4096,2560]` |
| Output, FP16 ND | `[16,16]` | `[16,4096]` |
| Transpose attributes | `adj_x1=false, adj_x2=true` | Same |

Both graphs insert TransData on B and match the installed 310P ND/NZ -> ND registration. This rules out a missing format conversion in the compiled graph, but does not validate converted values or tiling. See [API-AUDIT.md](API-AUDIT.md) for the version-pinned source findings.

### Kernel and exception evidence

Before SSH became unavailable, both unchanged probes were rerun in separate `ge-probe-kernel-evidence` and `ge-probe-small-kernel-evidence` directories with `NPU_COLLECT_PATH` set to the corresponding directory, `ASCEND_GLOBAL_LOG_LEVEL=1`, and `ASCEND_SLOG_PRINT_TO_STDOUT=1`. This retains compilation and exception artifacts; it is not a benchmark configuration. The small probe passed and finalized. The large probe exited 90 with the same MTE on-chip write-out-of-bounds error. Collection changed the generated kernel hash to `te_batchmatmulv3_239f1345519b31ab6f1b63e5e3c236b1cbeb0b10314bf773646b2ffeec814ad5`; do not describe this as the identical binary from the earlier run.

The final `SetL2CacheFlag` log, after intermediate tiling calculations, reports:

| Tiling field | Passing control | Failing projection |
| --- | --- | --- |
| Operator tiling key | `0x10000` | `0x10000` |
| baseM, baseN, baseK | `16,16,256` | `16,256,64` |
| singleCoreM, singleCoreN, singleCoreK | `16,16,256` | `16,256,2560` |
| stepM, stepN | `1,1` | `1,1` |
| stepKa, stepKb | `3,3` | `3,3` |
| depthA1, depthB1 | `6,6` | `6,6` |

The failing kernel metadata reports eight blocks and 20,971,520 bytes of workspace. The runtime's separate `tilingKey=0` field must not be confused with the operator compiler's `0x10000` selection.

The installed `tools/operator_cmp/compare/msaccucmp.py convert` parsed the exception dump into NumPy arrays. An elementwise check of both input arrays found:

- A: FP16 shape `[16,2560]`, 40,960 elements, all finite and exactly one.
- B: FP16 shape `[160,256,16,16]`, 10,485,760 elements, all finite and exactly one.

This confirms the expected input values are present in the exception snapshot for this uniform-input reproducer. Uniform values cannot validate general ND/NZ permutation ordering, and the failed output is not a correctness result. The captured instruction addresses have an offset of `0xb2c` from the reported kernel start; the installed llvm-objdump printed `<not available>` there, so no instruction-level cause has been established.

Local evidence includes `ge-probe-kernel-evidence.log`, `ge-probe-small-kernel-evidence.log`, and `bmm-kernel-evidence.json` in `cannge-validation-310-49e9873`. The binary, original exception dump, converted arrays, and disassembly remain in the remote evidence directory; they were not all downloaded. The server log clock reads October 23 while the client date is October 24; preserve the original filenames when correlating artifacts.

Next device experiments should vary N and K independently (M fixed at 16), starting with `(N,K)=(16,2560)` and `(4096,256)`. They distinguish the wide-output tile from the long contraction; they are not yet run. Do not encode either dimension as a backend rejection threshold from the two existing cases.

## Dimension batch and diagnostic refactor, 2026-10-24

The later run completed those independent experiments using exact source `a790f595e27bd3e102dd24155c98ee6777aabfd8` in `/workspace/cannge-validation-a790f59/source`. The transferred source archive SHA-256 was `087b0c5298cdf9f843d7e81d70aee9c8d879f3be73cf2c4c942e5ac409378f2e`, verified at both ends. It contains the checked diagnostic reader from `f8492a8` and the configurable probe/batch runner from `a790f59`. The collaborator checkout remained untouched and dirty at `d3a3ef81f` on `cannge`. The host container name changed to `ad2f550545f6`; device and SDK remained Ascend310P3/CANN 9.1.0, driver 26.0.rc1.

The full SDK build of `test-backend-ops` succeeded. All six standalone tests passed both locally with MSVC and on the host with GCC. The direct probe's build also succeeded. The runner's Git provenance commands report an archive without `.git`; the archive revision and hash above supply that provenance.

| M | N | K | Result | Base tile M,N,K |
| --- | --- | --- | --- | --- |
| 16 | 16 | 256 | Pass, exit 0 | 16,16,256 |
| 16 | 16 | 2560 | Pass, exit 0 | 16,16,1024 |
| 16 | 4096 | 256 | Pass, exit 0 | 16,256,64 |
| 16 | 4096 | 768 | Pass, exit 0 | 16,256,64 |
| 16 | 4096 | 2048 | MTE failure, exit 90 | 16,256,64 |
| 16 | 4096 | 2304 | MTE failure, exit 90 | See retained compiler log |
| 16 | 4096 | 2560 | MTE failure, exit 90 | 16,256,64 |

The four runner cases are in `matmul-batch/results.json`; the additional cases are separate bounded runs in `k768`, `k2048`, and `k2304`. No case timed out. All passing cases checked every output against K. Both independent dimension cases pass, so neither large N nor large K alone explains the observed failure. The wide-N passing case shares the failing projection's base tile, buffering depths and K steps. K=2304 is divisible by the observed 192-element K step, so the failure does not require a final partial step. These observations narrow the reproducer; they do not establish a safe rejection threshold or justify modifying SDK tiling values.

The installed SDK's `asc/impl/adv_api/detail/matmul/stage/copy_cube_in` headers were copied locally for read-only inspection. `copy_tile_to_cube_using_ub.h` selects `CopyND2NZWithVecOp` for ND input when `enVecND2NZ` is enabled; `data_copy_wrapper_using_ub_nd.h` uses local workspace, vector conversion and a copy into the cube input buffer. The NZ operand follows a separate copy wrapper. This is a source trace, not proof that either specific copy caused the reported fault; instruction disassembly remains unavailable.

For the diagnostic refactor, a single exported F16/F16 MUL_MAT fixture with ggml source shapes `[256,16,1,1]` and output `[16,16,1,1]` was run using the newly built backend, `GGML_CANNGE_GRAPH=1`, `GGML_CANNGE_DEBUG_IO=1`, the new status guard, and a 120-second timeout. The case passed CPU comparison (1/1 executed), exited 0, and emitted five `[DBG-IO]` lines. This validates the real diagnostic path; injected copy/drain failures remain covered by the standalone tests rather than real-runtime fault injection.

Logs and the fixture were archived to `evidence.tar.gz` in the validation directory and downloaded as `cannge-validation-310-49e9873/a790f59-evidence.tar.gz`. The final check found healthy, idle devices and no remaining probe or backend-test processes. No SDK changes, device resets, history rewrites or pushes were performed. Large-matmul correctness, shutdown recovery after runtime faults, model accuracy and blocking-copy benchmarks remain outstanding.

## Rank-two MatMulV2 correction

A subsequent direct GE probe built the same F16/F16 matrices through the public MatMulV2 graph prototype. Both the small control and M=16, N=4096, K=2560 passed every output check, graph removal, and finalization. The final graph retained MatMulV2 and used `te_matmulv2_0adf6b3e1f31eb2db886284e685d202c60adb070be03563035f13b864ad7fe7b__kernel0`. It did not select the failing BatchMatMulV3 kernel. In the probe, `matmul-v2` constructs the public graph directly because no corresponding generated ES wrapper is installed.

The backend candidate uses MatMulV2 only for two rank-two operands; it preserves batched lowering and the existing padding/casts. Its checked ES adapter uses the public prototype's input/output/attribute definitions and leaves inference validation enabled. Cache emission version is 4. An initial factory-based adapter could not find the prototype before GE initialization, so it was replaced before validation with explicit ES node construction, following the SDK's own ES pattern.

The isolated SDK build used the `a790f59` snapshot plus the recorded candidate patches and `mul-mat-ge.cpp`. Patch preimages were verified against local Git blob hashes after accounting for CRLF line endings in the source archive. The final adapter compiled and linked successfully. The six standalone regressions passed locally and on the host. The three committed projection fixtures now all pass CPU comparison (3/3), including mixed F16/F32 and aligned M=16 cases. All 36 supported selected small matmuls (`m=16,n=.,k=256,`) also pass. Numerical runs use the synchronization-aware status guard and external timeouts.

The same Qwen3-4B harness then completed both prompt/decode rounds and normal shutdown with the candidate libraries (paths verified with `ldd`). This removes the earlier projection crash but exposes a separate decode correctness failure:

- All logits are finite; prompt NMSE is `8.488455827696573e-6`, below the predeclared `1e-5` limit.
- Decode NMSE reaches `6.794395952671169`; only 6/20 top-one comparisons match CPU, with maximum absolute error `20.067551612854004`.
- Repeated device rounds differ (repeat NMSE up to about `0.0514`); the CPU reference repeats exactly.

Thus model correctness is not established. The large, non-repeatable decode errors require separate state/lifetime or operation isolation; they cannot be dismissed as the small precision drift seen earlier on 910. The CPU reference and harness are the previously recorded 310 validation artifacts; the harness ran against the candidate `build/bin` libraries through `LD_LIBRARY_PATH`.

Artifacts in `/workspace/cannge-validation-a790f59`: `matmul-v2/small.log`, `matmul-v2/projection.log`, its recursive final graph dump, `matmul-v2-backend-build.log`, `matmul-v2-projections.log`, `matmul-v2-small.log`, `matmul-v2-model.stderr`, and `matmul-v2-model-comparison.json`. The clean synchronization checkout is `/workspace/llama.cpp-cannge`; the dirty collaborator checkout at `/root/cannge-ggml` remains preserved. The 910 endpoint is currently unavailable, so its synchronization and validation remain pending.
