#include "cpy.h"
#include "debug.h"
#include "plan.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

static bool fail_sync = false;
static int fail_copy = 0;
static int copy_calls = 0;
static int async_calls = 0;
static std::vector<char> events;
static int fail_async = 0;
static int fail_alloc = 0;
static int alloc_calls = 0;
static std::map<void *, size_t> allocations;
struct pending_copy {
    char * dst;
    const char * src;
    size_t dpitch, spitch, width, height;
};
static std::vector<pending_copy> pending;


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
    if (fail_sync) { return 1; }
    for (const auto & copy : pending) {
        for (size_t row = 0; row < copy.height; row++) {
            std::memcpy(copy.dst + row * copy.dpitch, copy.src + row * copy.spitch, copy.width);
        }
    }
    pending.clear();
    return ACL_SUCCESS;
}
aclError aclrtMemcpy(void * dst, size_t capacity, const void * src, size_t size, aclrtMemcpyKind) {
    events.push_back('C');
    copy_calls++;
    if (copy_calls == fail_copy) { return 1; }
    check(size <= capacity, "copy capacity");
    std::memcpy(dst, src, size);
    return ACL_SUCCESS;
}
aclError aclrtMemcpy2dAsync(void * dst, size_t dpitch, const void * src, size_t spitch,
                             size_t width, size_t height, aclrtMemcpyKind, aclrtStream) {
    async_calls++;
    if (async_calls == fail_async) { return 1; }
    check(width <= dpitch && width <= spitch, "2D pitch covers row");
    pending.push_back({(char *) dst, (const char *) src, dpitch, spitch, width, height});
    return ACL_SUCCESS;
}
aclError aclrtMemcpyAsync(void * dst, size_t capacity, const void * src, size_t size, aclrtMemcpyKind kind, aclrtStream stream) {
    check(size <= capacity, "async copy capacity");
    return aclrtMemcpy2dAsync(dst, size, src, size, size, 1, kind, stream);
}
aclError aclrtMalloc(void ** ptr, size_t size, int) {
    alloc_calls++;
    if (alloc_calls == fail_alloc) { return 1; }
    *ptr = std::malloc(size);
    if (*ptr == nullptr) { return 1; }
    allocations[*ptr] = size;
    return ACL_SUCCESS;
}
aclError aclrtMallocHost(void ** ptr, size_t size) { return aclrtMalloc(ptr, size, 0); }
aclError aclrtFree(void * ptr) {
    if (ptr == nullptr) { return ACL_SUCCESS; }
    const auto it = allocations.find(ptr);
    check(it != allocations.end(), "free owned allocation");
    const uintptr_t begin = (uintptr_t) ptr;
    auto contains = [&](const void * address) {
        return (uintptr_t) address >= begin && (uintptr_t) address - begin < it->second;
    };
    for (const auto & copy : pending) {
        check(!contains(copy.src) && !contains(copy.dst), "cannot free allocation referenced by queued copy");
    }
    allocations.erase(it);
    std::free(ptr);
    return ACL_SUCCESS;
}
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
    check(pending.empty(), "test must drain pending copies");
    fail_sync = false; fail_copy = 0; fail_async = 0; fail_alloc = 0;
    copy_calls = 0; async_calls = 0; alloc_calls = 0; events.clear();
}
static void test_batched_row_conversion() {
    for (ggml_type source_type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (bool host : {false, true}) {
            reset();
            const ggml_type destination_type = source_type == GGML_TYPE_F16 ? GGML_TYPE_F32 : GGML_TYPE_F16;
            auto src = tensor(source_type, 2, 2, nullptr);
            auto dst = tensor(destination_type, 2, 3, nullptr);
            auto idx = tensor(GGML_TYPE_I32, 2, 1, nullptr);
            src.ne[2] = dst.ne[2] = 2;
            src.nb[1] = 4 * src.nb[0]; // Two padding elements between rows.
            src.nb[2] = 2 * src.nb[1]; src.nb[3] = 2 * src.nb[2];
            dst.nb[3] = 2 * dst.nb[2];
            std::vector<char> data(ggml_nbytes(&src), 0);
            std::vector<char> result(ggml_nbytes(&dst), 0);
            std::vector<char> expected(result.size(), 0);
            dst.data = result.data();
            if (host) { dst.buffer = reinterpret_cast<ggml_backend_buffer_t>(&dst); }
            const std::vector<int64_t> slots{2, 0};
            auto store = [](char * address, ggml_type type, float value) {
                if (type == GGML_TYPE_F32) {
                    std::memcpy(address, &value, sizeof(value));
                } else {
                    const auto half = ggml_fp32_to_fp16(value);
                    std::memcpy(address, &half, sizeof(half));
                }
            };
            for (int batch = 0; batch < 2; batch++) {
                for (int row = 0; row < 2; row++) {
                    for (int column = 0; column < 2; column++) {
                        const float value = float(1 + batch * 4 + row * 2 + column);
                        store(data.data() + batch * src.nb[2] + row * src.nb[1] + column * src.nb[0], source_type, value);
                        store(expected.data() + batch * dst.nb[2] + slots[row] * dst.nb[1] + column * dst.nb[0], destination_type, value);
                    }
                }
            }
            ggml_cannge_set_rows_op op{&dst, &src, &idx};
            check(ggml_cannge_set_rows(op, slots, data, nullptr), "batched padded row conversion");
            check(result == expected, "converted rows preserve batches, slot mapping and untouched rows");
            check(async_calls == 0 && pending.empty(), "row conversion storage is not retained");
            check(copy_calls == (host ? 0 : 4), "each device row completes its blocking upload");
        }
    }
    reset();
}

int main() {
    for (bool host : {false, true}) {
        reset();
        float source[] = {7, 8};
        float storage[] = {0, 0};
        auto t = tensor(GGML_TYPE_F32, 2, 1, storage);
        if (host) { t.buffer = reinterpret_cast<ggml_backend_buffer_t>(&t); }
        FILE * output = std::tmpfile();
        check(output != nullptr, "diagnostic capture file");
        check(aclrtMemcpyAsync(storage, sizeof(storage), source, sizeof(source), ACL_MEMCPY_DEVICE_TO_HOST, nullptr) == 0,
              "queued diagnostic producer");
        fail_sync = true;
        check(!ggml_cannge_debug_tensor("test", &t, nullptr, output), "diagnostic rejects failed drain");
        check(std::ftell(output) == 0 && copy_calls == 0, "failed drain prints no values or checksums");
        fail_sync = false;
        if (!host) {
            fail_copy = 1;
            check(!ggml_cannge_debug_tensor("test", &t, nullptr, output), "diagnostic rejects failed copy");
            check(std::ftell(output) == 0, "failed copy prints no values or checksums");
            fail_copy = 0;
        }
        check(ggml_cannge_debug_tensor("test", &t, nullptr, output), "diagnostic snapshot succeeds");
        check(storage[0] == 7 && pending.empty(), "diagnostic observes completed host and device producers");
        std::rewind(output);
        char line[512]{};
        check(std::fgets(line, sizeof(line), output) != nullptr && std::strstr(line, "i0=7") != nullptr,
              "diagnostic prints completed values");
        std::fclose(output);
    }
    reset();
    test_batched_row_conversion();
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
    reset();
    float transposed[] = {1, 2, 3, 4, 5, 6};
    float dense[6]{};
    auto view = tensor(GGML_TYPE_F32, 2, 3, transposed);
    view.nb[0] = 12; view.nb[1] = 4;
    check(ggml_cannge_strided_copy(&view, dense, false, nullptr), "enqueue strided gather");
    check(dense[0] == 0 && pending.size() == 3, "strided copies execute only on drain");
    check(aclrtSynchronizeStream(nullptr) == ACL_SUCCESS, "gather drain");
    const float gathered[] = {1, 4, 2, 5, 3, 6};
    check(std::memcmp(dense, gathered, sizeof(dense)) == 0, "gather uses source pitch");
    for (int i = 0; i < 6; i++) { dense[i] = float((i + 1) * 10); }
    check(ggml_cannge_strided_copy(&view, dense, true, nullptr), "enqueue strided scatter");
    check(aclrtSynchronizeStream(nullptr) == ACL_SUCCESS, "scatter drain");
    const float scattered[] = {10, 30, 50, 20, 40, 60};
    check(std::memcmp(transposed, scattered, sizeof(transposed)) == 0, "scatter uses destination pitch");

    reset(); fail_async = 2;
    std::memset(transposed, 0, sizeof(transposed));
    check(!ggml_cannge_strided_copy(&view, dense, true, nullptr), "partial enqueue failure propagates");
    check(pending.size() == 1 && transposed[0] == 0, "earlier copy remains pending after enqueue failure");
    check(aclrtSynchronizeStream(nullptr) == ACL_SUCCESS, "partial scatter drain");
    check(transposed[0] == 10 && transposed[3] == 20 && transposed[1] == 0, "only accepted copy executes");

    reset();
    ggml_fp16_t sparse[] = {0, 0, 0x3555, 0, 0};
    ggml_fp16_t compact[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    auto half_view = tensor(GGML_TYPE_F16, 2, 2, sparse);
    half_view.nb[1] = 6; half_view.nb[2] = half_view.nb[3] = 12;
    check(ggml_cannge_strided_copy(&half_view, compact, true, nullptr), "F16 bounce scatter");
    check(sparse[0] == compact[0] && sparse[1] == compact[1] &&
          sparse[3] == compact[2] && sparse[4] == compact[3] && sparse[2] == 0x3555, "bounce preserves holes");
    reset(); fail_sync = true;
    compact[0] = 0x4800;
    check(!ggml_cannge_strided_copy(&half_view, compact, true, nullptr), "bounce drain failure propagates");
    check(copy_calls == 0 && sparse[0] == 0x3c00, "failed bounce drain leaves destination unchanged");

    reset();
    struct layout_case { int64_t ne[4]; size_t nb[4]; };
    const layout_case layouts[] = {
        {{3, 2, 1, 1}, {2, 6, 12, 12}},
        {{3, 5, 1, 1}, {10, 2, 30, 30}},
        {{19, 23, 1, 1}, {46, 2, 874, 874}},
        {{3, 2, 2, 2}, {16, 2, 4, 8}},
        {{3, 2, 2, 2}, {4, 16, 40, 96}},
    };
    for (const auto & layout : layouts) {
        for (bool host_view : {false, true}) {
            auto t = tensor(GGML_TYPE_F16, 1, 1, nullptr);
            std::memcpy(t.ne, layout.ne, sizeof(t.ne));
            std::memcpy(t.nb, layout.nb, sizeof(t.nb));
            if (host_view) { t.buffer = reinterpret_cast<ggml_backend_buffer_t>(&t); }
            std::vector<ggml_fp16_t> storage(ggml_nbytes(&t) / 2, 0x3555);
            std::vector<ggml_fp16_t> expected(ggml_nelements(&t));
            std::vector<size_t> offsets;
            for (int64_t i3 = 0; i3 < t.ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < t.ne[2]; i2++) {
                    for (int64_t i1 = 0; i1 < t.ne[1]; i1++) {
                        for (int64_t i0 = 0; i0 < t.ne[0]; i0++) {
                            offsets.push_back((i0 * t.nb[0] + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3]) / 2);
                        }
                    }
                }
            }
            for (size_t i = 0; i < expected.size(); i++) {
                expected[i] = ggml_fp32_to_fp16(float(i + 1));
                storage[offsets[i]] = expected[i];
            }
            t.data = storage.data();
            std::vector<ggml_fp16_t> gathered_half(expected.size(), 0);
            reset();
            check(ggml_cannge_strided_copy(&t, gathered_half.data(), false, nullptr), "host layout gather");
            check(gathered_half == expected, "F16 gather matches independent element offsets");
            const auto original = storage;
            std::fill(storage.begin(), storage.end(), 0x3555);
            check(ggml_cannge_strided_copy(&t, gathered_half.data(), true, nullptr), "host layout scatter");
            check(storage == original, "F16 scatter restores elements and preserves gaps");
            check(pending.empty(), "bounce copies leave no queued temporary storage");
        }
    }
    reset();
    void * device = nullptr;
    void * host = nullptr;
    size_t capacity = 0;
    check(ggml_cannge_reserve_input_arena(device, host, capacity, 16, nullptr, false), "initial arena");
    void * old_device = device;
    void * old_host = host;
    std::memset(host, 7, 16);
    check(aclrtMemcpyAsync(device, 16, host, 16, ACL_MEMCPY_HOST_TO_DEVICE, nullptr) == ACL_SUCCESS, "pending arena transfer");
    fail_sync = true;
    check(!ggml_cannge_reserve_input_arena(device, host, capacity, 16, nullptr, true), "same-capacity reuse must drain pending upload");
    check(device == old_device && host == old_host && pending.size() == 1, "failed reuse retains queued source");
    check(!ggml_cannge_reserve_input_arena(device, host, capacity, 32, nullptr, false), "arena drain failure");
    check(device == old_device && host == old_host && capacity == 16 && pending.size() == 1, "failed drain preserves pending arena");
    fail_sync = false;
    const int allocations_before_reuse = alloc_calls;
    check(ggml_cannge_reserve_input_arena(device, host, capacity, 16, nullptr, true), "same-capacity retry drains successfully");
    check(pending.empty() && ((char *) device)[0] == 7 && alloc_calls == allocations_before_reuse, "retry completes upload without allocation");
    const auto events_after_reuse = events.size();
    check(ggml_cannge_reserve_input_arena(device, host, capacity, 16, nullptr, false), "completed arena reuses without drain");
    check(events.size() == events_after_reuse, "normal reuse adds no synchronization");
    fail_alloc = alloc_calls + 1;
    check(!ggml_cannge_reserve_input_arena(device, host, capacity, 32, nullptr, false), "device allocation failure");
    check(pending.empty() && ((char *) device)[0] == 7, "old transfer completed before growth");
    check(device == old_device && host == old_host && capacity == 16 && allocations.size() == 2, "device failure preserves arena");
    fail_alloc = alloc_calls + 2;
    check(!ggml_cannge_reserve_input_arena(device, host, capacity, 32, nullptr, false), "host allocation failure");
    check(device == old_device && host == old_host && capacity == 16 && allocations.size() == 2, "host failure frees candidate and preserves arena");
    fail_alloc = 0;
    check(ggml_cannge_reserve_input_arena(device, host, capacity, 32, nullptr, false), "arena growth succeeds");
    check(capacity == 32 && allocations.size() == 2 && device != old_device && host != old_host, "arena replacement owns both allocations");
    aclrtFree(device); aclrtFreeHost(host);
    check(allocations.empty() && pending.empty(), "no test allocations or transfers left");
    std::puts("PASS: CANNGE copy lifetime, strided transfer and arena failure regressions");
}
