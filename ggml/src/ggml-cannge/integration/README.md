# Fork working set

The owner's active inference implementation is KokerZhou/llama.cpp, branch cannge. Keep one checkout and use its embedded ggml. Standalone ggml is deferred; do not create another copy or attempt unification now.

JittorInfer is patch-only: integration/jittorinfer/310-diagnostics.patch preserves the recovered small changes against a pinned upstream revision. Do not retain source archives, Git bundles or an extracted JittorInfer tree as a second working set.

The separate cannge-handoff-notes branch on the same GitHub repository stores collaborator notes. On 2026-10-24 the retained local notes tip cace21fecc309c7c6b9a5a2e4434ea869334950e was confirmed to be an ancestor of remote 7afabecbd2e69dc032ed7cc3f1072260856e0a5e. Notes and recovery metadata need not be duplicated locally. Review individual findings before integrating code into cannge.

Before removing a source package, verify its exact commit is reachable from the published branch or preserve and validate its unique changes as a reviewed patch. Existing llama.cpp packages identified 9224022f5fdfe4b517785c5bfdd00958add1ddca and a790f595e27bd3e102dd24155c98ee6777aabfd8; both are ancestors of published efee827d41f7b777c43878f2fc899992ea64d439. The transfer bundle also pointed to efee827. Deleting these packages does not delete their Git history.

This policy applies to our fork-related source material. It does not remove the separate npussh tooling or official GE/operator audit clones. Private session exports and validation evidence are not source snapshots and stay outside Git. Temporary transport files belong in the workspace SSH exchange area and should be removed after verified promotion, rather than retained as parallel source copies.
