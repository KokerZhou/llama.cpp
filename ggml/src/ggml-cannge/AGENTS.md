# CANNGE collaboration rules

These instructions apply to this directory and its descendants. Read the repository-root `AGENTS.md` as well. Direct user instructions take precedence. This file records the owner's constraints; it does not grant permission to push, change remote systems, or contact other collaborators.

## Branch and ownership

- Work only on `cannge`. Do not switch to or create another branch, including `cann-ge`.
- Preserve history. Never force-push, rebase published work, amend existing commits, or reset away commits. Use fast-forward-only pulls.
- If histories diverge, inspect both sides and report the differences before integrating. A merge commit requires explicit user authorization. The authorized merge `3e4dfe4` is not blanket permission for future merges.
- Use the reviewed local implementation as the integration baseline. Review collaborator-only changes individually; do not replace our implementation with a remote tree or blindly cherry-pick equivalent fixes.
- Keep repository edits inside `ggml/src/ggml-cannge/` unless the user explicitly expands scope. Do not stage unrelated files or unfinished work from another contributor.
- Before editing or synchronizing, inspect branch, HEAD, working-tree status, and recent commits on each checkout involved. Recheck before writing if another contributor may have changed it.
- Never copy whole source files over a collaborator's checkout. Use a reviewed patch against a verified revision and stop on unexpected differences. Preserve uncommitted work; do not automatically stash or discard someone else's changes.
- Commit only within the user's authorization. Use the configured author and the `Assisted-by: Codex` trailer for Codex-assisted commits. Do not push merely because a commit was created.

## Implementation and review

- Follow the CPU/CUDA-style module boundaries already established here. Keep operation builders, support policy, planning, execution, and resource ownership separate.
- Fix causes instead of adding overlapping switches, shape-specific exceptions, or silent fallbacks. Keep support predicates consistent with the graphs builders actually emit.
- Preserve numerical behavior and failure propagation when optimizing. Do not label a workaround necessary without a reproducible case and official API/source evidence.
- Keep experimental paths opt-in until correctness and performance are demonstrated. In particular, do not silently enable scatter, disk caching, or BMM GEMV by default.
- Preserve canonical ggml shapes at builder boundaries, including singleton dimensions. Test affected ranks and dtypes when changing shape or cast handling.
- Treat asynchronous resource lifetime as a correctness requirement. Failed synchronization does not permit releasing resources; failed removal must retain a retry path. Remove bookkeeping only after the underlying operation succeeds.
- Update cache identity when semantics change. Bump `GGML_CANNGE_EMIT_VERSION` when lowering changes for the same graph, and include relevant options in the plan signature.
- Consult the local official GE source clone and official API documentation for API assumptions. Record version-specific evidence in `tests/API-AUDIT.md`; distinguish documented contracts from observed behavior.

## Validation and reporting

- Follow `tests/HARDWARE.md`. Run relevant standalone regressions, but do not equate stub tests with an SDK build or numerical device validation.
- Use an isolated source snapshot and build directory for hardware experiments. Record the exact commit and any patch, SDK, device, model hash, commands, and options. Results from a checkout changed during a run cannot certify a commit.
- Check execution return codes, stderr, and actual executed/skipped counts. Use the documented status guard for numerical tests; a zero process exit alone is insufficient.
- Validate model prompt processing and repeated decode against CPU, including cache reuse. Establish numerical tolerances before inspecting results.
- Compare cold and warm disk-cache runs in separate processes. Validate optional paths separately before changing defaults.
- Measure performance only after correctness passes, using matching settings and repeated samples. Blocking-copy changes require the paired baseline described in `tests/HARDWARE.md`.
- Report what changed, what was tested, failures, untested paths, and whether commits were pushed. Do not call the backend finished while SDK/device correctness, recovery, or required performance checks remain unresolved.

## Remote access and storage

- Use the owner's configured npussh connection and its local forwarded port with ordinary SSH. Keep one long-lived tunnel; do not start overlapping npussh commands that can tear down shared forwarding state.
- Ask the owner for the direct SSH command/setup before using a new 910/310 environment. Do not infer authorization for another machine from an alias or credential being available.
- `buaax` is permitted only as `ssh -L 22022:127.0.0.1:22022 buaax` when that route is needed. Direct inspection, cloning, Docker operations, or other use requires explicit approval. Prefer the established direct npussh route when available.
- On the Ascend side, stop on network failures or hangs and let the owner resolve them. Do not retry, invent proxy routes, or change network configuration. This restriction is specific to the Ascend side.
- Do not download Ascend Docker images or perform device-less SDK builds without approval. Local source clones and standalone stub tests are allowed; real backend validation belongs on the configured NPU host.
- Do not prune Docker, delete checkouts, or perform broad cleanup without authorization for the concrete targets. Keep logs and build outputs outside the source checkout.
- Never print or commit credentials, tokens, private keys, or OAuth callback query strings. Do not copy credentials between hosts without explicit authorization.
