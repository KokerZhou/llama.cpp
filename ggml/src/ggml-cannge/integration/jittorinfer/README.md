# Retained JittorInfer changes

This directory holds one small combined patch, not a JittorInfer checkout or source snapshot.

Upstream: https://github.com/Jittor/JittorInfer
Base: `55d1c1e1c2603ce9d15b6a4066ed17cd6bf3aa4d`

Apply from a checkout of that exact upstream revision:

```sh
git apply --check /path/to/310-diagnostics.patch
git apply /path/to/310-diagnostics.patch
```

The patch preserves the optional custom GE RoPE build, portable host FP16 conversions, eager Qwen3 dispatch, targeted GET_ROWS tests, explicit thread linking, and corrected batched GET_ROWS weights. It does not provide general 310P GE compatibility.

Recovery used the surviving `ebb4594` source archive and the recorded edits for `837a28c` and `3688267`. The later GET_ROWS preimage and postimage match the Git blob prefixes in the original session diff. Thread linking was reconstructed from the recorded change. This is a combined source delta, not a reconstruction of the original six Git commits. Packaging aids and a dereferenced AGENTS.md symlink are excluded.

`provenance.json` records the input hashes and before/after blobs. `git apply --check` passed against the upstream preimages, and all nine applied files matched the expected blobs after line-ending normalization. No SDK build or device test was run for this recovery. These historical changes retain the upstream project's licensing requirements.

The deleted KokerZhou/JittorInfer fork is not restored. Standalone ggml remains a separate TODO; no checkout or source copy is maintained here.
