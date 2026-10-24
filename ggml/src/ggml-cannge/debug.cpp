#include "debug.h"

#include <algorithm>
#include <cstring>

static uint32_t checksum(const char * data, size_t size) {
    uint32_t a = 0, b = 0;
    for (size_t i = 0; i < size; ++i) {
        a = (a + (uint8_t) data[i]) % 65535;
        b = (b + a) % 65535;
    }
    return (b << 16) | a;
}

bool ggml_cannge_debug_tensor(const char * tag, const ggml_tensor * t, aclrtStream stream, FILE * output) {
    if (t == nullptr || t->data == nullptr || output == nullptr) {
        return false;
    }
    if (!ggml_cannge_check(aclrtSynchronizeStream(stream), "debug snapshot synchronization", __func__)) {
        return false;
    }
    constexpr size_t chunk = 4096;
    const size_t size = std::min(ggml_nbytes(t), 9 * chunk);
    std::vector<char> bytes(size);
    if (size != 0) {
        if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(bytes.data(), t->data, size);
        } else if (!ggml_cannge_check(aclrtMemcpy(bytes.data(), size, t->data, size, ACL_MEMCPY_DEVICE_TO_HOST),
                                      "debug snapshot copy", __func__)) {
            return false;
        }
    }
    std::fprintf(output, "[DBG-IO] %s %s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu sum=%08x", tag, t->name,
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                 t->nb[0], t->nb[1], t->nb[2], t->nb[3], checksum(bytes.data(), std::min(size, chunk)));
    for (size_t offset = chunk; offset + chunk <= size; offset += chunk) {
        std::fprintf(output, " +%zuk=%08x", offset / 1024, checksum(bytes.data() + offset, chunk));
    }
    // These are storage samples, not logical indices of a strided tensor.
    if (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32) {
        const size_t element_size = t->type == GGML_TYPE_F16 ? 2 : 4;
        for (int index : {0, 1, 2, 127, 128, 129, 255}) {
            const size_t offset = (size_t) index * element_size;
            if (offset + element_size > size) {
                break;
            }
            float value;
            if (t->type == GGML_TYPE_F16) {
                ggml_fp16_t half;
                std::memcpy(&half, bytes.data() + offset, sizeof(half));
                value = ggml_fp16_to_fp32(half);
            } else {
                std::memcpy(&value, bytes.data() + offset, sizeof(value));
            }
            std::fprintf(output, " i%d=%.4g", index, value);
        }
    }
    std::fprintf(output, "\n");
    return true;
}
