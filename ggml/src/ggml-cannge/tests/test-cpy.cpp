#include "cpy.h"
#include "plan.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool fail_sync = false;
static int fail_copy = 0;
static int copy_calls = 0;
static int async_calls = 0;
static std::vector<char> events;

static void check(bool result, const char * message) {
    if (!result) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

bool ggml_cannge_check(aclError error, const char *, const char *) { return error == ACL_SUCCESS; }
void ggml_log_internal(enum ggml_log_level, const char *, ...) {}
float ggml_fp16_to_fp32(ggml_fp16_t value) { return GGML_COMPUTE_FP16_TO_FP32(value); }
ggml_fp16_t ggml_fp32_to_fp16(float value) { return GGML_COMPUTE_FP32_TO_FP16(value); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) {
    return GGML_BACKEND_BUFFER_USAGE_ANY;
}
aclError aclrtSynchronizeStream(aclrtStream) {
    events.push_back('S');
    return fail_sync ? 1 : ACL_SUCCESS;
}
aclError aclrtMemcpy(void * dst, size_t capacity, const void * src, size_t size, aclrtMemcpyKind) {
    events.push_back('C');
    copy_calls++;
    if (copy_calls == fail_copy) { return 1; }
    check(size <= capacity, "copy capacity");
    std::memcpy(dst, src, size);
    return ACL_SUCCESS;
}
aclError aclrtMemcpyAsync(void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream) {
    async_calls++;
    return ACL_SUCCESS;
}
aclError aclrtMemcpy2dAsync(void *, size_t, const void *, size_t, size_t, size_t, aclrtMemcpyKind, aclrtStream) {
    async_calls++;
    return ACL_SUCCESS;
}
aclError aclrtMalloc(void ** ptr, size_t size, int) { *ptr = std::malloc(size); return *ptr ? ACL_SUCCESS : 1; }
aclError aclrtMallocHost(void ** ptr, size_t size) { return aclrtMalloc(ptr, size, 0); }
aclError aclrtFree(void * ptr) { std::free(ptr); return ACL_SUCCESS; }
aclError aclrtFreeHost(void * ptr) { return aclrtFree(ptr); }

static ggml_tensor tensor(ggml_type type, int64_t width, int64_t rows, void * data) {
    ggml_tensor t{};
    t.type = type; t.ne[0] = width; t.ne[1] = rows; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = type == GGML_TYPE_F16 ? 2 : 4;
    t.nb[1] = width * t.nb[0]; t.nb[2] = t.nb[3] = rows * t.nb[1];
    t.data = data;
    return t;
}
static void reset() {
    fail_sync = false; fail_copy = 0; copy_calls = 0; async_calls = 0; events.clear();
}
int main() {
    float values[] = {1, 2, 3, 4};
    float destination[6]{};
    auto src = tensor(GGML_TYPE_F32, 2, 2, values);
    auto dst = tensor(GGML_TYPE_F32, 2, 3, destination);
    auto idx = tensor(GGML_TYPE_I32, 2, 1, nullptr);
    ggml_cannge_set_rows_op op{&dst, &src, &idx};
    std::vector<char> data(sizeof(values));
    std::memcpy(data.data(), values, sizeof(values));
    std::vector<int64_t> slots{2, 0};

    fail_sync = true;
    check(!ggml_cannge_set_rows(op, slots, data, nullptr), "failed drain stops scatter");
    check(copy_calls == 0 && async_calls == 0, "failed drain queues no host copies");

    reset();
    fail_copy = 2;
    check(!ggml_cannge_set_rows(op, slots, data, nullptr), "upload failure propagates");
    check(events == std::vector<char>({'S', 'C', 'C'}) && async_calls == 0, "failed scatter leaves no queued temporary source");
    check(destination[4] == 1 && destination[0] == 0, "only completed rows are written");

    reset();
    check(ggml_cannge_set_rows(op, slots, data, nullptr), "F32 scatter");
    check(destination[0] == 3 && destination[1] == 4 && destination[4] == 1 && destination[5] == 2, "row mapping");
    check(events == std::vector<char>({'S', 'C', 'C'}) && async_calls == 0, "scatter drains before blocking copies");

    ggml_fp16_t halves[6]{};
    dst = tensor(GGML_TYPE_F16, 2, 3, halves);
    reset();
    check(ggml_cannge_set_rows(op, slots, data, nullptr), "converted scatter");
    check(ggml_fp16_to_fp32(halves[0]) == 3 && ggml_fp16_to_fp32(halves[4]) == 1, "converted row values");
    check(async_calls == 0, "converted storage never queued");

    reset();
    std::vector<char> snapshot;
    check(ggml_cannge_read_snapshot(&src, sizeof(values), nullptr, snapshot), "device snapshot");
    check(events == std::vector<char>({'S', 'C'}) && async_calls == 0, "snapshot drains then copies synchronously");
    check(snapshot == data, "snapshot bytes");
    reset(); fail_sync = true;
    check(!ggml_cannge_read_snapshot(&src, sizeof(values), nullptr, snapshot), "snapshot drain failure");
    check(copy_calls == 0 && async_calls == 0, "snapshot failure queues nothing");
    reset(); fail_copy = 1;
    check(!ggml_cannge_read_snapshot(&src, sizeof(values), nullptr, snapshot), "snapshot copy failure");
    check(async_calls == 0, "failed snapshot leaves no queued destination");
    std::puts("PASS: CANNGE temporary host copy lifetime and failure propagation");
}
