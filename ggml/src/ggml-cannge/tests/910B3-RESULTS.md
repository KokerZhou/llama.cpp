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
