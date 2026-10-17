#include "ggml-cannge-impl.h"
#include "ggml-impl.h"
#include <ge/ge_api.h>

#include <cstdio>
#include <cstdlib>

static bool fail_sync = false;
static int free_calls = 0;

static void check(bool result, const char * message) {
    if (!result) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

aclError aclrtSetDevice(int32_t) { return ACL_SUCCESS; }
aclError aclrtCreateStream(aclrtStream * stream) { *stream = &fail_sync; return ACL_SUCCESS; }
aclError aclrtSynchronizeStream(aclrtStream) {
    cannge_test::events.push_back("sync");
    return fail_sync ? 1 : ACL_SUCCESS;
}
aclError aclrtDestroyStream(aclrtStream) { return ACL_SUCCESS; }
aclError aclrtFree(void * ptr) {
    if (ptr != nullptr) { free_calls++; cannge_test::events.push_back("free"); }
    return ACL_SUCCESS;
}
aclError aclrtFreeHost(void * ptr) { return aclrtFree(ptr); }
void ggml_backend_cannge_set_device(int32_t) {}
bool ggml_cannge_check(aclError error, const char *, const char *) { return error == ACL_SUCCESS; }
void ggml_log_internal(enum ggml_log_level, const char *, ...) {}

static uint32_t add_plan(ggml_backend_cannge_context & ctx, const char * key, uint64_t seq, bool compiled) {
    std::unique_ptr<ggml_cannge_plan> plan(new ggml_cannge_plan);
    ge::Graph graph;
    std::string err;
    check(ctx.ge->add_and_compile(&graph, plan->graph_id, err) == compiled, "compile outcome");
    plan->state = compiled ? ggml_cannge_plan::COMPILED : ggml_cannge_plan::FAILED;
    plan->seq = seq;
    plan->staging.push_back(&free_calls);
    const uint32_t id = plan->graph_id;
    ctx.plans.emplace(key, std::move(plan));
    return id;
}

int main() {
    using namespace cannge_test;
    {
        ggml_backend_cannge_context ctx(0);
        std::string err;
        ctx.ge.reset(ggml_cannge_ge_context::create(0, err));
        check(ctx.initialized && ctx.ge != nullptr, "context initialization");
        // Lexical map order differs from insertion order.
        const auto oldest = add_plan(ctx, "z-old", 0, true);
        const auto newest = add_plan(ctx, "a-new", 1, true);
        events.clear();
        fail_sync = true;
        check(!ctx.evict_oldest_plan(), "failed synchronization stops eviction");
        check(ctx.plans.size() == 2 && graphs.size() == 2 && free_calls == 0, "sync failure retains ownership");
        check(events == std::vector<std::string>{"sync"}, "sync failure skips graph removal");
        fail_sync = false;
        fail_remove = true;
        events.clear();
        check(!ctx.evict_oldest_plan(), "failed removal stops eviction");
        check(ctx.plans.size() == 2 && graphs.count(oldest) == 1 && free_calls == 0, "removal failure retains oldest");
        check(events == std::vector<std::string>({"sync", "remove"}), "removal follows drain");
        fail_remove = false;
        events.clear();
        check(ctx.evict_oldest_plan(), "retry succeeds");
        check(ctx.plans.count("z-old") == 0 && ctx.plans.count("a-new") == 1, "oldest insertion evicted");
        check(graphs.count(oldest) == 0 && graphs.count(newest) == 1 && free_calls == 1, "retry releases only oldest");
        check(events == std::vector<std::string>({"sync", "remove", "free"}), "buffers released after removal");
        check(ctx.evict_oldest_plan(), "remaining plan evicted");

        fail_compile = true;
        fail_remove = true;
        const auto rollback = add_plan(ctx, "rollback", 2, false);
        check(graphs.count(rollback) == 1, "failed rollback retains graph");
        const int before_retry = remove_calls;
        check(!ctx.evict_oldest_plan() && ctx.plans.count("rollback") == 1, "failed rollback eviction retains plan");
        fail_remove = false;
        check(ctx.evict_oldest_plan(), "rollback retry succeeds");
        check(remove_calls == before_retry + 2 && graphs.empty(), "rollback removal retried");

        const auto removed = add_plan(ctx, "already-removed", 3, false);
        check(graphs.count(removed) == 0, "successful compile rollback removes graph");
        int before_remove = remove_calls;
        check(ctx.evict_oldest_plan(), "already removed graph does not block eviction");
        check(remove_calls == before_remove, "removed graph is not removed twice");

        fail_compile = false;
        fail_add = true;
        add_plan(ctx, "add-failed", 4, false);
        before_remove = remove_calls;
        check(ctx.evict_oldest_plan() && remove_calls == before_remove, "failed add needs no GE removal");
        fail_add = false;
        check(ctx.evict_oldest_plan(), "empty cache eviction succeeds");
        check(free_calls == 5 && graphs.empty(), "all evicted allocations released");
        add_plan(ctx, "teardown", 5, true);
        events.clear();
    }
    check(events == std::vector<std::string>({"sync", "session", "finalize", "free"}), "normal teardown ordering");
    std::puts("PASS: CANNGE eviction and graph rollback failure recovery");
}
