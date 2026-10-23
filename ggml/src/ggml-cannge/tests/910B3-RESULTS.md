# Ascend 910B3 validation

Validation began on 2026-10-20 and continued on 2026-10-22. Base revision: `9224022f5fdfe4b517785c5bfdd00958add1ddca`, plus the metadata-only execution and empty-update fixes accompanying this report. This is partial validation, not approval of all backend paths.

## Environment

- Ascend 910B3, exposed as CANNGE0; npu-smi/driver reports 25.5.0.
- CANN 9.1.0 at `/home/developer/Ascend/cann-9.1.0`.
- aarch64, GCC 9.4.0, CMake 3.20.5.
- Source: `/home/developer/cannge-kit/cannge-ggml`, branch `cannge`.
- Build and logs: `/mnt/workspace/cannge-validation-9224022`.
- SDK environment sourced; `GGML_CANNGE_GRAPH=1`; other backend switches unset.
- SDK Python emitted a SciPy/NumPy compatibility warning (NumPy 2.4.6, SciPy requires below 2.3.0). No package changes were made.

## Confirmed results

- Real SDK compilation and linking of CANNGE, test-backend-ops and llama-bench succeeded.
- Support discovery: 1,003 accepted configurations out of 20,117. This does not establish correctness.
- Initial ADD/MUL smoke: eight F16/F32 cases passed CPU comparison. This run preceded strict status instrumentation.
- Strict SET_ROWS: 86 supported cases passed CPU comparison, including conversion, batching, views and empty updates. Unsupported configurations were skipped.
- Strict CPY/CONT: all 46 supported cases passed CPU comparison.
- Strict selected MUL_MAT/ROPE/SOFT_MAX: 33 supported cases passed CPU comparison. Matrix selection used M=16, N=1 or 8, K=256; RoPE selection used shape [128,32,2,1]; softmax selection used shape [16,16,1,1]. These counts exclude rejected configurations.
- Standalone regressions: 6/6 passed with GCC, including the empty-update planning regression.

## Failures found and corrected

The shared `ggml_backend_compare_graph_backend` helper ignores `ggml_backend_graph_compute` return codes. The initial CPY/CONT/SET_ROWS run exited zero despite backend errors. Its exit code is not accepted as a successful validation result. The Linux `libcannge-status-guard.so` test tool now exits 90 on any execution failure; it does not change the shared helper or relax numerical thresholds.

Metadata-only graph fragments attempted GE construction and failed on valid views. Execution now returns success without compiling graphs containing only leaves, empty outputs and metadata operations. Mixed graphs still build their view dependencies normally.

SET_ROWS with zero update rows attempted index snapshots and host scatter. Planning now omits that nonexistent write, and optional scatter materialization ignores the empty update tensor. Destination contents remain unchanged; the strict device suite covers the zero-row cases.

The standalone test project lacked `ggml_abort` when GCC emitted unused inline helpers without optimization. A terminating assertion stub now supports that build. Its byte-size stub also returns zero for empty tensors.

## Reproduction

Build the standalone project as described in README.md, then run against the real backend:

```sh
export GGML_CANNGE_GRAPH=1
LD_PRELOAD="$metadata_build/libcannge-status-guard.so" \
  "$build_dir/bin/test-backend-ops" test -b CANNGE0 -o SET_ROWS --output console
LD_PRELOAD="$metadata_build/libcannge-status-guard.so" \
  "$build_dir/bin/test-backend-ops" test -b CANNGE0 -o CPY,CONT --output console
LD_PRELOAD="$metadata_build/libcannge-status-guard.so" \
  "$build_dir/bin/test-backend-ops" test -b CANNGE0 -o MUL_MAT,ROPE,SOFT_MAX \
  -p 'm=16,n=(1|8),k=256,|ne_a=\[128,32,2,1\]|ne=\[16,16,1,1\]' --output console
```

## Remaining validation

Full operator coverage, opt-in attention/GQA/scatter modes, merged model graphs, repeated decode/cache reuse, runtime recovery and paired blocking-copy performance measurements remain outstanding. No performance claim follows from these correctness checks.

An existing model is available for subsequent validation: `/home/developer/models/Qwen3-4B-Instruct-2507-F16.gguf`, 8,051,285,344 bytes, SHA-256 `336228a2810f17724ecd6e9e2e40e8381eee81e886316fee311ade2899c31030`. Model inference has not yet been validated in this run.

## Isolated revision check, 2026-10-23

Revision `49e9873fb10001ea7c073b6f0a0ffaea0dc06a4e` was exported to `/mnt/workspace/cannge-validation-49e9873/source` and built separately from the shared checkout. The real SDK build of `test-backend-ops` and `llama-bench` completed successfully. Standalone regressions passed 6/6. Disk caching was disabled; only `GGML_CANNGE_GRAPH=1` was enabled for the default-path checks.

While the tools were building, the previous test driver was run against the new revision's libraries, with resolution recorded in `test-library-paths.txt`. The driver used the new strict-status guard. CPY/CONT/SET_ROWS passed 132/132 supported cases. MUL_MAT passed 40/40 supported cases selected by `m=16,n=.,k=256,`, including small-M padding. These are library-validation results, not a claim that the newly built driver was used for those runs.

The Qwen3-4B model check used a harness compiled against the isolated build, a fresh CPU reference, and strict status checking on the device run. It processed the prompt `The capital of France is` followed by the fixed continuation ` Paris. The capital of Japan is Tokyo.` one token at a time. It repeated the sequence after clearing the KV cache, producing ten logits vectors in each of two rounds.

All logits were finite, all 20 top-token selections matched CPU, and the two device rounds were identical. However, eight of ten steps in each round exceeded the predetermined normalized mean squared error limit of `1e-5`. Prompt NMSE was `7.3049e-6`, worst decode NMSE was `3.5859e-5`, and maximum absolute logits error was `0.1076384`. The model check therefore failed; matching sampled tokens does not override the numerical criterion. Logs and comparison data are in `device.stderr` and `model-comparison.json` under the isolated validation directory.

An earlier GEMV32 experiment also exceeded the limit, but its build came from the previously shared checkout and is diagnostic evidence only. The isolated no-matmul experiment lost its SSH session before a result was collected. Work moved to the owner's 310P3 environment because the 910 lab became busy. No blocking-copy performance claim follows from this run.
