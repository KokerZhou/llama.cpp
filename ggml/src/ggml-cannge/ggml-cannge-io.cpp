/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// CANNGE backend IO copies: strided/dense staging transfers between ggml
// buffers and GE input/output buffers (split out of ggml-cannge.cpp).

#include "ggml-cannge-priv.h"
#include "plan.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>

ggml_cannge_dbg_timing g_dbg_time;

bool ggml_cannge_check(aclError error, const char * function, const char * file, int line) {
    if (error != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: ACL error %d (%s) at %s:%d\n", function, error, aclGetRecentErrMsg(), file, line);
        return false;
    }
    return true;
}

const void * ggml_cannge_staging_source(const ggml_tensor * tensor) {
    if (getenv("GGML_CANNGE_NO_WONCE") != nullptr || tensor->op != GGML_OP_NONE ||
        (tensor->flags & GGML_TENSOR_FLAG_INPUT) || tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_host(tensor->buffer) ||
        ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        ggml_nbytes(tensor) <= 64 * 1024) {
        return nullptr;
    }
    return tensor->data;
}

bool ggml_cannge_reserve_input_arena(void *& device, void *& host, size_t & capacity, size_t size, aclrtStream stream) {
    if (size <= capacity) {
        return true;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    void * new_device = nullptr;
    void * new_host = nullptr;
    if (aclrtMalloc(&new_device, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtMallocHost(&new_host, size) != ACL_SUCCESS) {
        ACL_CHECK(aclrtFree(new_device));
        return false;
    }
    if (device != nullptr) {
        ACL_CHECK(aclrtFree(device));
    }
    if (host != nullptr) {
        ACL_CHECK(aclrtFreeHost(host));
    }
    device = new_device;
    host = new_host;
    capacity = size;
    return true;
}

bool ggml_cannge_set_rows(const ggml_cannge_set_rows_op & op, const std::vector<int64_t> & slots,
                          const std::vector<char> & data, aclrtStream stream) {
    const ggml_tensor * src = op.data;
    const ggml_tensor * dst = op.dst;
    const ggml_tensor * idx = op.idx;
    if (src == nullptr || dst == nullptr || idx == nullptr) {
        return false;
    }
    const size_t es_src = ggml_element_size(src);
    const size_t es_dst = ggml_element_size(dst);
    const bool host = dst->buffer != nullptr && ggml_backend_buffer_is_host(dst->buffer);
    if ((src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_F32) ||
        (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) ||
        src->ne[0] != dst->ne[0] || src->ne[1] != idx->ne[0] ||
        src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3] ||
        src->ne[0] <= 0 || src->ne[2] <= 0 || src->ne[3] <= 0 || dst->data == nullptr ||
        idx->ne[0] <= 0 || idx->ne[1] <= 0 || idx->ne[2] <= 0 ||
        slots.size() / (size_t) idx->ne[0] / (size_t) idx->ne[1] < (size_t) idx->ne[2] ||
        data.size() < ggml_nbytes(src) || src->nb[0] != es_src || dst->nb[0] != es_dst) {
        return false;
    }
    for (int64_t slot : slots) {
        if (slot < 0 || slot >= dst->ne[1]) {
            return false;
        }
    }

    std::vector<char> converted;
    if (es_src != es_dst) {
        if (data.size() / es_src > std::numeric_limits<size_t>::max() / es_dst) {
            return false;
        }
        converted.resize(data.size() / es_src * es_dst);
    }
    const size_t row_size = (size_t) src->ne[0] * es_dst;
    bool copied = true;
    for (int64_t i3 = 0; i3 < src->ne[3] && copied; i3++) {
        for (int64_t i2 = 0; i2 < src->ne[2] && copied; i2++) {
            for (int64_t i = 0; i < idx->ne[0]; i++) {
                const size_t slot_id = (size_t) i + (size_t) (i2 % idx->ne[1]) * idx->ne[0] +
                                      (size_t) (i3 % idx->ne[2]) * idx->ne[0] * idx->ne[1];
                const size_t offset = i * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3];
                const char * row = data.data() + offset;
                char * dst_row = (char *) dst->data + slots[slot_id] * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
                if (es_src != es_dst) {
                    char * cast_row = converted.data() + offset / es_src * es_dst;
                    for (int64_t e = 0; e < src->ne[0]; e++) {
                        if (src->type == GGML_TYPE_F32) {
                            ((ggml_fp16_t *) cast_row)[e] = ggml_fp32_to_fp16(((const float *) row)[e]);
                        } else {
                            ((float *) cast_row)[e] = ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[e]);
                        }
                    }
                    row = cast_row;
                }
                if (host) {
                    memcpy(dst_row, row, row_size);
                } else if (aclrtMemcpyAsync(dst_row, row_size, row, row_size, ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
                    copied = false;
                    break;
                }
            }
        }
    }
    // Both snapshots must stay alive until every queued host copy completes.
    if (!host && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    return copied;
}

ggml_cannge_dbg_timing::~ggml_cannge_dbg_timing() {
    dump();
}

void ggml_cannge_dbg_timing::dump() {
    if (!on || calls == 0) return;
    fprintf(stderr, "[DBG-TIME] calls=%lld analyze=%.1f sig=%.1f find=%.1f bind=%.1f(stage=%.1f+sync=%.1f+obind=%.1f) exec=%.1f post=%.1f total=%.1f | cpy n=%lld sync=%.1f ms | staged=%.0f MB\n",
            calls, analyze, sig, find, bind, stage, stage_sync, obind, exec, post, total, cpy_calls, cpy_sync,
            stage_bytes / 1e6);
    std::vector<std::pair<double, std::string>> top;
    top.reserve(stage_names.size());
    for (const auto & kv : stage_names) {
        top.emplace_back(kv.second, kv.first);
    }
    std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    for (size_t i = 0; i < top.size() && i < 8; i++) {
        fprintf(stderr, "[DBG-TIME]   staged %-28s %.1f MB\n", top[i].second.c_str(), top[i].first / 1e6);
    }
    fprintf(stderr, "[DBG-TIME] evicts=%lld\n", evicts);
}

// element matrix; 16x16 tiles keep both sides cache-resident (the naive
// element loop jumps columns*es per element and is the hot path of staged
// transposed/permuted inputs)
static void ggml_cannge_repack_2d(char * out, size_t o_r, size_t o_c, const char * in, size_t i_r, size_t i_c,
                                  int64_t R, int64_t C, size_t es) {
    const int64_t T = 16;
    for (int64_t rr = 0; rr < R; rr += T) {
        const int64_t rmax = std::min(R, rr + T);
        for (int64_t cc = 0; cc < C; cc += T) {
            const int64_t cmax = std::min(C, cc + T);
            for (int64_t r = rr; r < rmax; r++) {
                char * orow = out + (size_t) r * o_r;
                const char * irow = in + (size_t) r * i_r;
                for (int64_t c = cc; c < cmax; c++) {
                    memcpy(orow + (size_t) c * o_c, irow + (size_t) c * i_c, es);
                }
            }
        }
    }
}

// copy between a strided ggml view layout and a compact dense buffer, via
// ne1*ne2*ne3 element-granularity 2D copies (width = element size, height =
// ne0, pitch = nb[0]); nb[0] can exceed the element size on permuted views
bool ggml_cannge_strided_copy(const ggml_tensor * t, void * dense, bool dense_to_strided, aclrtStream stream) {
    const size_t es   = ggml_element_size(t);
    const size_t slab = (size_t) t->ne[0] * es; // one dense (i1,i2,i3) slab
    const int64_t n   = t->ne[1] * t->ne[2] * t->ne[3];
    // a staged input may live in host memory (CPU fallback output): ACL has no
    // auto-direction memcpy
    const bool    host = t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
    // SDMA strided copies need >=4B rows and 4B-aligned pitches (f16
    // transposed views have 2B strides): bounce the whole footprint through
    // host memory for those, one aligned H2D/D2H per direction
    bool aligned_ok = es >= 4;
    for (int i = 0; i < GGML_MAX_DIMS && aligned_ok; i++) {
        aligned_ok = t->nb[i] % 4 == 0;
    }
    if (!aligned_ok) {
        // clean 2D-repack family: dims 1..3 form a contiguous inner extent
        // C (standard strides) and dim0 carries stride C*es, so the whole
        // footprint is one packed [ne0 x C] matrix to repack tiled; anything
        // else falls back to the element gather
        const int64_t C_elems   = t->ne[1] * t->ne[2] * t->ne[3];
        const bool   repack2d   = t->nb[1] == (size_t) es && t->nb[0] == (size_t) C_elems * es &&
                                (t->ne[2] <= 1 || t->nb[2] == (size_t) t->ne[1] * es) &&
                                (t->ne[3] <= 1 || t->nb[3] == (size_t) t->ne[1] * t->ne[2] * es);
        if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
            fprintf(stderr, "[DBG-COPY] bounce %s ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld repack2d=%d\n",
                    t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
                    (int) repack2d);
        }
        int64_t off_last = 0;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            off_last += (t->ne[i] - 1) * t->nb[i];
        }
        const size_t span = (size_t) off_last + es;
        const size_t slab = (size_t) t->ne[0] * es;
        // the blocking bounce copies below run on the default stream and do
        // NOT wait for pending work on this stream: a dense_to_strided
        // copy-back would read the staging buffer while the async GE execute
        // is still writing it (observed as a previous layer's transpose bytes
        // scattered into the next layer's cache view, GQA=all garbage).
        // Drain first whenever the source side lives on device.
        if (dense_to_strided || !host) {
            if (!ACL_CHECK(aclrtSynchronizeStream(stream))) {
                return false;
            }
        }
        std::vector<char> bounce(span);
        if (dense_to_strided) {
            // Preserve bytes between view elements when copying back the full span.
            if (span > (size_t) n * slab) {
                if (host) {
                    memcpy(bounce.data(), t->data, span);
                } else {
                    if (!ACL_CHECK(aclrtMemcpy(bounce.data(), span, t->data, span, ACL_MEMCPY_DEVICE_TO_HOST))) {
                        return false;
                    }
                }
            }
            std::vector<char> dense_host((size_t) n * slab);
            if (!ACL_CHECK(aclrtMemcpy(dense_host.data(), (size_t) n * slab, dense, (size_t) n * slab,
                                      ACL_MEMCPY_DEVICE_TO_HOST))) {
                return false;
            }
            if (repack2d && t->nb[0] != (size_t) es) {
                // dense [ne0 x C] -> view layout (dim0 carries the C stride)
                ggml_cannge_repack_2d(bounce.data(), (size_t) C_elems * es, es, dense_host.data(), es,
                                      (size_t) t->ne[0] * es, t->ne[0], C_elems, es);
            } else {
                for (int64_t l = 0; l < n; l++) {
                    const int64_t i1 = l % t->ne[1];
                    const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                    const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                    char * dst = bounce.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    const char * src = dense_host.data() + (size_t) l * slab;
                    // a view whose dim0 is not the memory-fastest dim (nb0 > es,
                    // e.g. a transposed V cache view) needs element-wise scatter
                    if (t->nb[0] == (size_t) es) {
                        memcpy(dst, src, slab);
                    } else {
                        for (int64_t e = 0; e < t->ne[0]; e++) {
                            memcpy(dst + (size_t) e * t->nb[0], src + (size_t) e * es, es);
                        }
                    }
                }
            }
            if (host) {
                memcpy(t->data, bounce.data(), span);
            } else {
                if (!ACL_CHECK(aclrtMemcpy(t->data, span, bounce.data(), span, ACL_MEMCPY_HOST_TO_DEVICE))) {
                    return false;
                }
            }
        } else {
            if (host) {
                if (repack2d && t->nb[0] != (size_t) es) {
                    ggml_cannge_repack_2d(bounce.data(), es, (size_t) t->ne[0] * es, (const char *) t->data,
                                          (size_t) C_elems * es, es, t->ne[0], C_elems, es);
                } else {
                    for (int64_t l = 0; l < n; l++) {
                        const int64_t i1 = l % t->ne[1];
                        const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                        const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                        const char * src = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                        char * dst = bounce.data() + (size_t) l * slab;
                        if (t->nb[0] == (size_t) es) {
                            memcpy(dst, src, slab);
                        } else {
                            for (int64_t e = 0; e < t->ne[0]; e++) {
                                memcpy(dst + (size_t) e * es, src + (size_t) e * t->nb[0], es);
                            }
                        }
                    }
                }
            } else {
                // device-resident strided source: D2H the strided span into a
                // host mirror, gather it into dense layout in bounce; the
                // shared H2D below lands it in staging
                std::vector<char> mirror(span);
                if (!ACL_CHECK(aclrtMemcpy(mirror.data(), span, t->data, span, ACL_MEMCPY_DEVICE_TO_HOST))) {
                    return false;
                }
                if (repack2d && t->nb[0] != (size_t) es) {
                    ggml_cannge_repack_2d(bounce.data(), es, (size_t) t->ne[0] * es, mirror.data(),
                                          (size_t) C_elems * es, es, t->ne[0], C_elems, es);
                } else {
                    for (int64_t l = 0; l < n; l++) {
                        const int64_t i1 = l % t->ne[1];
                        const int64_t i2 = (l / t->ne[1]) % t->ne[2];
                        const int64_t i3 = l / (t->ne[1] * t->ne[2]);
                        const char * src = mirror.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                        char * dst = bounce.data() + (size_t) l * slab;
                        if (t->nb[0] == (size_t) es) {
                            memcpy(dst, src, slab);
                        } else {
                            for (int64_t e = 0; e < t->ne[0]; e++) {
                                memcpy(dst + (size_t) e * es, src + (size_t) e * t->nb[0], es);
                            }
                        }
                    }
                }
            }
            if (!ACL_CHECK(aclrtMemcpy(dense, (size_t) n * slab, bounce.data(), (size_t) n * slab,
                                      ACL_MEMCPY_HOST_TO_DEVICE))) {
                return false;
            }
        }
        return true;
    }
    const aclrtMemcpyKind kind =
        dense_to_strided ? (host ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE)
                         : (host ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (getenv("GGML_CANNGE_DEBUG_COMPILE")) {
        fprintf(stderr, "[DBG-COPY] %s t->data=%p host=%d dir=%s ne=%lld,%lld,%lld,%lld nb0=%lld\n",
                dense_to_strided ? "out" : "in", t->data, (int) host, dense_to_strided ? "D2H" : (host ? "H2D" : "D2D"),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                (long long) t->nb[0]);
    }
    for (int64_t l = 0; l < n; l++) {
        const int64_t i1 = l % t->ne[1];
        const int64_t i2 = (l / t->ne[1]) % t->ne[2];
        const int64_t i3 = l / (t->ne[1] * t->ne[2]);
        char *       view_row = (char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        char *       dense_row = (char *) dense + (size_t) l * slab;
        const void * src    = dense_to_strided ? (const void *) dense_row : (const void *) view_row;
        void *       dst    = dense_to_strided ? (void *) view_row : (void *) dense_row;
        const size_t spitch = dense_to_strided ? es : (size_t) t->nb[0];
        const size_t dpitch = dense_to_strided ? (size_t) t->nb[0] : es;
        if (!ACL_CHECK(aclrtMemcpy2dAsync(dst, dpitch, src, spitch, es, (size_t) t->ne[0], kind, stream))) {
            return false;
        }
    }
    return true;
}
